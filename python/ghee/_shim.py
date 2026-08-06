"""The ghee procedural-scene shim.

The shim builds a scene, answers the probe, and feeds the data.
A procedural scene is an executable: ghee starts the process when the scene
becomes active, the process writes its scene as one JSON line, then binary
messages, one batch for each tick. See docs/procedural.md for the wire
contract.

    #!/usr/bin/env python3
    import ghee

    args = ghee.args([ghee.Arg("bins", "int", 64, "how many bars")])
    scene = ghee.Scene(passes=[ghee.Pass("main", CODE,
                               arrays=[ghee.ArrayInput("level", args.bins)])])

    @ghee.feed("level")
    def level(tick):
        return [...]                  # None sends nothing, and the array stands

    ghee.run(scene)

A feed that returns bytes, a bytearray or a numpy array of dtype uint8 sends
the u8 wire form, where the value is byte / 255.

The same scene file runs two ways. When ghee starts it, run() talks the
wire. When you run the file yourself, run() opens the scene on the ghee
bindings and drives the same feeds there. $GHEE_WIRE, which ghee sets in
the processes it starts, tells the two apart.
"""

from __future__ import annotations

import argparse
import array
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any

_VEC = {"vec2": 2, "vec3": 3, "vec4": 4}
_PARAM_TYPES = ("float", "int", "bool", "color", "string", *_VEC)

# ghee sets this in the scene process it starts. Without it, run() self-hosts.
WIRE = "GHEE_WIRE"


# ---------------------------------------------------------------- scene model


@dataclass
class Param:
    """A live value.

    The scene, the console and the scene process name it the same way.
    A color is [r, g, b] in 0..1.
    A vecK is K numbers.
    A string stays host-side, and reaches the scene process in the tick line.
    """

    name: str
    type: str
    value: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[str] | None = None

    def __post_init__(self) -> None:
        if self.type not in _PARAM_TYPES:
            raise ValueError(
                f"param '{self.name}' has unknown type {self.type!r}; "
                f"known types: {', '.join(_PARAM_TYPES)}"
            )
        n = 3 if self.type == "color" else _VEC.get(self.type)
        if n is not None:
            v = list(self.value)
            if len(v) != n:
                raise ValueError(f"param '{self.name}' needs {n} numbers, got {v!r}")
            self.value = [float(x) for x in v]
        elif self.type == "float":
            self.value = float(self.value)
        elif self.type == "int":
            self.value = int(self.value)
        elif self.type == "bool":
            self.value = bool(self.value)
        elif not isinstance(self.value, str):
            raise ValueError(
                f"string param '{self.name}' needs a string, got {self.value!r}"
            )

    def to_dict(self) -> dict:
        d: dict = {"type": self.type, "value": self.value}
        for k in ("min", "max", "step", "options"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        return d


@dataclass
class ArrayInput:
    """A named float input the scene fills.

    Size N is a 1xN array. The shader reads it with <name>_at(i).
    Size [w, h] is a 2D mask. The shader reads it with <name>_sample2(uv).
    """

    name: str
    size: Any

    def __post_init__(self) -> None:
        if isinstance(self.size, (list, tuple)):
            if len(self.size) != 2:
                raise ValueError(
                    f"array '{self.name}' size needs an int or [w, h], "
                    f"got {self.size!r}"
                )
            self.size = (int(self.size[0]), int(self.size[1]))
        else:
            self.size = int(self.size)

    @property
    def is_2d(self) -> bool:
        return not isinstance(self.size, int)

    @property
    def count(self) -> int:
        """The values one message carries. A 2D array sends w*h values."""
        return self.size[0] * self.size[1] if self.is_2d else self.size

    def to_dict(self) -> dict:
        return {"name": self.name, "size": list(self.size) if self.is_2d else self.size}


@dataclass
class Channel:
    """A numbered sampler. iChannel<index> reads the output of one pass."""

    buffer: str | None = None

    def to_dict(self) -> dict:
        return {"buffer": self.buffer}


@dataclass
class Pass:
    """One fragment shader.

    `size` renders the pass at its own [w, h].
    The last pass leaves `size` None, because it feeds the display.
    """

    name: str
    code: str
    arrays: list[ArrayInput] = field(default_factory=list)
    channels: list[Channel] = field(default_factory=list)
    params: dict[str, Param] = field(default_factory=dict)
    filter: str = ""
    size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.size is not None:
            self.size = (int(self.size[0]), int(self.size[1]))

    def to_dict(self) -> dict:
        out: dict = {"name": self.name, "code": self.code}
        if self.size is not None:
            out["size"] = list(self.size)
        if self.arrays:
            out["arrays"] = [a.to_dict() for a in self.arrays]
        if self.channels:
            out["channels"] = [c.to_dict() for c in self.channels]
        if self.params:
            out["params"] = {n: p.to_dict() for n, p in self.params.items()}
        if self.filter:
            out["filter"] = self.filter
        return out


@dataclass
class Scene:
    """The passes in render order, plus what they share.

    A scene param reaches every pass.
    `common` is GLSL that ghee puts in front of each pass.
    `filters` are named post-process passes ghee appends after the scene's own.
    """

    passes: list[Pass] = field(default_factory=list)
    version: int = 1
    filters: list[str] = field(default_factory=list)
    params: dict[str, Param] = field(default_factory=dict)
    common: str | None = None

    def to_dict(self) -> dict:
        out: dict = {
            "version": self.version,
            "passes": [p.to_dict() for p in self.passes],
        }
        if self.common is not None:
            out["common"] = {"code": self.common}
        if self.filters:
            out["filters"] = self.filters
        if self.params:
            out["params"] = {n: p.to_dict() for n, p in self.params.items()}
        return out


# ---------------------------------------------------------------------- args


SIZE = "GHEE_SIZE"

# A buffer is usually asked for as a share of the frame, so the words say that
# directly. Powers of two: an odd divisor lands the buffer off the pixel grid
# and the magnify combs.
_FRACTIONS = {"full": 1, "half": 2, "quarter": 4, "eighth": 8, "sixteenth": 16}


def _render_size() -> tuple[int, int] | None:
    """The size ghee is about to render at.

    ghee puts it in the environment before it starts the scene. A direct run
    has no ghee, so the host --size stands in, and the default matches the one
    `ghee render` uses with no display.
    """
    given = os.environ.get(SIZE)
    if not given:
        argv = sys.argv[1:]
        if "--size" in argv[:-1]:
            given = argv[argv.index("--size") + 1]
    if not given:
        return (320, 240)
    try:
        w, h = given.lower().split("x")
        return (int(w), int(h))
    except ValueError:
        return None


def _size(s: str) -> tuple[int, int]:
    share = _FRACTIONS.get(s.strip().lower())
    if share is not None:
        size = _render_size()
        if size is None:
            raise argparse.ArgumentTypeError(
                f"{s!r} is a share of the render size, and this run has none: pass WxH"
            )
        return (max(1, size[0] // share), max(1, size[1] // share))
    try:
        w, h = s.lower().split("x")
        wh = (int(w), int(h))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"want WxH or one of {', '.join(_FRACTIONS)}, got {s!r}"
        ) from None
    if min(wh) < 1:
        raise argparse.ArgumentTypeError(f"want a positive WxH, got {s!r}")
    return wh


_ARG_TYPES = {
    "string": {"type": str},
    "int": {"type": int},
    "float": {"type": float},
    "size": {"type": _size, "metavar": "WxH|SHARE"},
    "bool": {"action": argparse.BooleanOptionalAction},
}


@dataclass
class Arg:
    """A build-time argument the scene takes.

    The ghee command line gives it after `--`.
    A `size` arg parses WxH to (w, h), and a share of the render size by
    name: full, half, quarter, eighth, sixteenth.
    A `bool` arg takes --name or --no-name.
    """

    name: str
    type: str = "string"
    default: Any = None
    help: str = ""

    def __post_init__(self) -> None:
        if self.type not in _ARG_TYPES:
            raise ValueError(
                f"arg '{self.name}' has unknown type {self.type!r}; "
                f"known types: {', '.join(_ARG_TYPES)}"
            )

    def declaration(self) -> dict:
        """One entry of the probe answer.

        A size default is the WxH the command line takes.
        """
        d = self.default
        if self.type == "size" and isinstance(d, tuple):
            d = f"{d[0]}x{d[1]}"
        return {"name": self.name, "type": self.type, "default": d, "help": self.help}


def args(specs: list[Arg] | tuple[Arg, ...] = ()):
    """Parse the command line against the declared Args.

    `--ghee-args` writes the declaration line and stops: the ghee probe.
    `--help` lists the args.
    A dash in a name becomes an underscore on the result.
    A direct run takes the host flags as well, and run() reads them.
    """
    global _HOST
    argv = sys.argv[1:]
    if "--ghee-args" in argv:
        line = json.dumps({"args": [s.declaration() for s in specs]})
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        raise SystemExit(0)
    p = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), allow_abbrev=False)
    for s in specs:
        d = s.default
        shown = f"{d[0]}x{d[1]}" if isinstance(d, tuple) else d
        help_ = s.help if d is None else f"{s.help} (default: {shown})".lstrip()
        p.add_argument(f"--{s.name}", default=d, help=help_, **_ARG_TYPES[s.type])
    if os.environ.get(WIRE):
        return p.parse_args(argv)
    _host_flags(p)
    _HOST = p.parse_args(argv)
    return _HOST


def _host_flags(p: argparse.ArgumentParser) -> None:
    """What a direct run takes on top of the scene's own args.

    Each flag is the flag `ghee render` spells the same way.
    A scene that declares one of these names collides here, and argparse says
    so.
    """
    p.add_argument(
        "--output",
        default="kms",
        metavar="NAME",
        help="a display driver name, 'kms', or 'none' for a headless render "
        "(default: kms)",
    )
    p.add_argument(
        "--size",
        type=_size,
        metavar="WxH",
        help="WxH to render at; a display or a panel states its own",
    )
    p.add_argument("--fps", type=float, metavar="N", help="frame rate cap")
    p.add_argument(
        "--stats", type=int, metavar="PORT", help="serve the console on PORT"
    )
    p.add_argument(
        "--frames",
        type=int,
        default=0,
        metavar="N",
        help="frames to render; 0 runs until SIGINT or SIGTERM (default: 0)",
    )
    p.add_argument(
        "--timestep",
        type=float,
        metavar="S",
        help="pin time to frame*S seconds instead of the wall clock",
    )


# ------------------------------------------------------------------ the loop


@dataclass
class Tick:
    """One render tick.

    `params` holds the string params that changed.
    The first tick holds every string param.
    """

    frame: int
    time: float
    dt: float
    params: dict


_FEEDS: dict = {}
_HOOKS: list = []
_OUT: list = []
_CLEAN = False
# a direct run: the host flags args() read, the handle it drives, the signals
_HOST = None
_HANDLE = None
_STOPPED: list = []


def feed(name: str):
    """Decorate the source of a named array.

    The function takes the tick and returns the values.
    It returns None to leave the array as it stands.
    A uint8 buffer goes out as bytes, and anything else as floats.
    """

    def hold(fn):
        _FEEDS[name] = fn
        return fn

    return hold


def tick(fn):
    """Decorate a per-tick hook. It runs before the feeds."""
    _HOOKS.append(fn)
    return fn


def set_param(name: str, value, pass_: str | None = None) -> None:
    """Set a live param for this tick.

    A float takes one number, a color takes three, and a vecN takes N.
    `pass_` addresses the param of one pass, such as a filter param.
    A bare name is the scene param, else the first pass that declares it.
    """
    if _HANDLE is not None:
        _HANDLE.set_param(name, value, pass_=pass_)
        return
    wire = f"{pass_}.{name}" if pass_ else name
    count, body = _f32((value,) if isinstance(value, (int, float)) else value)
    _OUT.append(_message(1, wire, count, body))


def skip() -> None:
    """This tick needs no render. The display keeps the last frame."""
    global _CLEAN
    _CLEAN = True


def _message(kind: int, name: str, count: int, body) -> bytes:
    nb = name.encode("utf-8")
    if len(nb) > 255:
        raise ValueError(f"name {name!r} is over the 255 bytes the wire holds")
    return bytes((kind, len(nb))) + nb + count.to_bytes(4, "little") + body


def _view(values):
    """The buffer of `values`, or None for a source that holds no buffer."""
    try:
        return memoryview(values)
    except TypeError:
        return None


def _f32(values) -> tuple[int, bytes]:
    """(count, little-endian f32 bytes).

    A float32 buffer, such as a numpy array, copies straight out.
    Any other iterable of numbers converts first.
    """
    if sys.byteorder == "little":
        mv = _view(values)
        if mv is not None and mv.format == "f" and mv.c_contiguous:
            return mv.nbytes // 4, mv.tobytes()
    a = array.array("f", values)
    if sys.byteorder != "little":
        a.byteswap()
    return len(a), a.tobytes()


def _array_message(name: str, values) -> tuple[int, bytes]:
    """(count, one array message) for what a feed returned.

    A buffer of bytes goes out as kind 4: one byte for each value, which the
    engine reads as byte / 255. `bytes`, a `bytearray` and a numpy array of
    dtype uint8 are such buffers. A large mask then costs a quarter of the
    bytes, and the engine uploads it with no conversion.
    Anything else goes out as kind 0 floats.
    """
    mv = _view(values)
    if mv is not None and mv.format == "B":
        # the message copies the body once. A view spares a mask a second copy.
        body = mv if mv.c_contiguous else mv.tobytes()
        return mv.nbytes, _message(4, name, mv.nbytes, body)
    count, body = _f32(values)
    return count, _message(0, name, count, body)


def _tick(line: str) -> Tick:
    d = json.loads(line)
    return Tick(
        int(d.get("frame", 0)),
        float(d.get("time", 0.0)),
        float(d.get("dt", 0.0)),
        d.get("params") or {},
    )


def _counts(scene: Scene) -> dict:
    """The values each declared array takes, keyed by name.

    A feed of a name the scene never declared stops the run here.
    """
    counts = {a.name: a.count for p in scene.passes for a in p.arrays}
    for name in _FEEDS:
        if name not in counts:
            raise ValueError(
                f"feed '{name}' is not an array this scene declares "
                f"({', '.join(counts) or 'none'})"
            )
    return counts


def run(scene: Scene) -> None:
    """Render the scene.

    A run under ghee answers the wire.
    A run of its own opens the scene on the bindings and drives it.

    Ctrl-C reaches every process in the terminal's group, so ghee takes it
    too and is already stopping. The scene ends where it stands, quietly: a
    traceback here would say a scene crashed, and it did what it was told.
    """
    try:
        if os.environ.get(WIRE):
            _wire(scene)
        else:
            _direct(scene)
    except KeyboardInterrupt:
        pass


def _wire(scene: Scene) -> None:
    """Write the handshake, then answer ticks until ghee closes stdin."""
    global _CLEAN
    out = sys.stdout.buffer
    out.write(json.dumps(scene.to_dict()).encode("utf-8") + b"\n")
    out.flush()
    counts = _counts(scene)
    for line in sys.stdin:
        if not line.strip():
            continue
        t = _tick(line)
        del _OUT[:]
        _CLEAN = False
        for hook in _HOOKS:
            hook(t)
        for name, fn in _FEEDS.items():
            values = fn(t)
            if values is None:
                continue
            count, msg = _array_message(name, values)
            if count != counts[name]:
                raise ValueError(
                    f"array '{name}' takes {counts[name]} values, got {count}"
                )
            _OUT.append(msg)
        if _CLEAN:
            _OUT.append(b"\x02")
        if _OUT:
            out.write(b"".join(_OUT))
            out.flush()


# --------------------------------------------------------------- direct mode


def _host_args():
    """The host flags of a run that declared no args of its own."""
    p = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), allow_abbrev=False)
    _host_flags(p)
    return p.parse_args(sys.argv[1:])


def _open(scene: Scene, a):
    """Open the scene on the bindings. `--output none` renders headless."""
    from ghee import open as open_scene

    w, h = a.size or (None, None)
    return open_scene(
        scene.to_dict(),
        output=None if a.output == "none" else a.output,
        width=w,
        height=h,
        fps=a.fps,
        stats=a.stats,
        timestep=a.timestep,
    )


def _stop(sig, _frame) -> None:
    _STOPPED.append(sig)


def _direct(scene: Scene) -> None:
    """Drive the scene from the same hooks and feeds the wire drives.

    The tick keeps its wire shape. `params` stays empty, because no ghee holds
    the string params here.
    A frame budget of 0 runs until SIGINT or SIGTERM.
    """
    global _CLEAN, _HANDLE
    a = _HOST if _HOST is not None else _host_args()
    _counts(scene)
    handle = _open(scene, a)
    del _STOPPED[:]
    caught = [(s, signal.signal(s, _stop)) for s in (signal.SIGINT, signal.SIGTERM)]
    _HANDLE = handle
    frame, started, last = 0, time.monotonic(), 0.0
    try:
        while not _STOPPED and (a.frames == 0 or frame < a.frames):
            now = (
                frame * a.timestep
                if a.timestep is not None
                else time.monotonic() - started
            )
            t = Tick(frame, now, now - last, {})
            last = now
            _CLEAN = False
            for hook in _HOOKS:
                hook(t)
            for name, fn in _FEEDS.items():
                values = fn(t)
                if values is not None:
                    handle.feed(name, values)
            if _CLEAN:
                handle.skip()
            else:
                handle.frame()
            frame += 1
    finally:
        _HANDLE = None
        for s, held in caught:
            signal.signal(s, held)
        handle.close()
