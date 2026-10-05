"""The ghee procedural-scene shim.

The shim builds a scene, answers the probe, and feeds the data.
A procedural scene is an executable: ghee starts the process when the scene
becomes active, the process writes its scene as one JSON line, then binary
messages, one batch for each tick. See docs/procedural.md for the wire
contract.

    #!/usr/bin/env -S uv run --script
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
import numbers
import os
import signal
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import SupportsFloat

    from typing_extensions import Self

_VEC = {"vec2": 2, "vec3": 3, "vec4": 4}


ParamType = Literal["float", "int", "bool", "color", "string", "vec2", "vec3", "vec4"]
Align = Literal["left", "center", "right"]
Fill = Literal["mtsdf", "slug"]
Blend = Literal["normal", "add", "multiply", "screen"]
ArgType = Literal["string", "int", "float", "size", "bool"]


class Atlas(TypedDict):
    """The MTSDF atlas of a Text: the texels per em of a glyph."""

    font_size: int


# ghee sets this in the scene process it starts. Without it, run() self-hosts.
WIRE = "GHEE_WIRE"
# ghee sets this when it reads kind 5 and the handshake's "lockstep". Without
# it, the wire goes on with neither.
LOCKSTEP = "GHEE_LOCKSTEP"


# ---------------------------------------------------------------- scene model


def _plain(v: Any) -> Any:
    """`v` in the types that json and the bindings both take: a tuple as a
    list, a path as its str, a numpy number or array as its Python value.
    Anything else goes out as given, for ghee to judge."""
    if isinstance(v, os.PathLike):
        return os.fspath(v)
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (str, bytes)) or not hasattr(v, "tolist"):
        return v
    return v.tolist()


def _floats(v: Any) -> Any:
    """`v` as `_plain` leaves it, with each real number in it a float."""
    v = _plain(v)
    if isinstance(v, list):
        return [_floats(x) for x in v]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return v


@dataclass
class Param:
    """A live value.

    The scene, the console and the scene process name it the same way.
    A color is "#rrggbb" or [r, g, b] in 0..1.
    A vecK is K numbers.
    A string never reaches a shader: the scene process reads it in the tick
    line. ghee declares the string param of each Text's name itself, and a
    Param under that name fails the scene.
    """

    name: str
    type: ParamType
    value: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[str] | None = None

    def to_dict(self) -> dict:
        floats = self.type in ("float", "color", *_VEC)
        value = _floats(self.value) if floats else _plain(self.value)
        d: dict = {"type": self.type, "value": value}
        for k in ("min", "max", "step", "options"):
            if getattr(self, k) is not None:
                d[k] = _plain(getattr(self, k))
        return d


def _keyed(params: dict[str, Param] | list[Param], what) -> dict[str, Param]:
    """`params` keyed by name, as the document keys them. `what(name)` says
    whose param that name is."""
    if isinstance(params, dict):
        for k, p in params.items():
            if k != p.name:
                raise ValueError(f"{what(p.name)} sits under the key '{k}'")
        return params
    keyed: dict[str, Param] = {}
    for p in params:
        if p.name in keyed:
            raise ValueError(f"{what(p.name)} is declared twice")
        keyed[p.name] = p
    return keyed


@dataclass
class ArrayInput:
    """A named float input for the scene to fill.

    Size N is a 1xN array. The shader reads it with <name>_at(i).
    Size [w, h] is a 2D mask. The shader reads it with <name>_sample2(uv).
    """

    name: str
    size: int | tuple[int, int]

    @property
    def is_2d(self) -> bool:
        return isinstance(_plain(self.size), list)

    @property
    def count(self) -> int:
        """How many values one message carries. A 2D array sends w*h values."""
        size = _plain(self.size)
        return size[0] * size[1] if isinstance(size, list) else size

    def to_dict(self) -> dict:
        return {"name": self.name, "size": _plain(self.size)}


@dataclass
class Channel:
    """A numbered sampler. iChannel<index> reads the output of one pass."""

    buffer: str | None = None

    def to_dict(self) -> dict:
        return {"buffer": self.buffer}


@dataclass
class Layer:
    """A text drawn into the buffer of its pass, one quad over each glyph.

    `text` names a Text of the scene.
    `fragment` holds a void main() that colours a glyph's pixels,
    premultiplied by alpha. None draws the letters white.
    `vertex` defines vec2 glyph_place(vec2 em), which takes a point of the
    block, in em from its top left with y down, to the pass pixel where it
    lands.
    `font_size` draws the block at that many pixels per em, and `at` puts its
    top left at that pixel of the pass, [x, y] with y down. Without `at` the
    block sits centred in the pass, and without `font_size` it takes the
    largest size that fits. `vertex` takes the place of both.
    `pad` is how far past the letters the layer reaches, in em, for a glow or
    an outline. An MTSDF layer fades to zero at the edge of its pad, over the
    last `fade` em: a quarter of `pad` by default, and 0 turns the fade off.
    float glyph_fade holds the factor in the fragment GLSL.
    `blend` is normal, add, multiply or screen.
    `emoji` sets whether the layer paints the text's colour glyphs. None
    leaves them to the last layer of the text in its pass.
    `name` names the layer in errors and in the console: ASCII letters,
    digits, '_' and '-', unique among the layers of its pass. A layer with no
    name goes by its text, and the next layers of the text with no name by
    "t 2", "t 3".
    """

    text: str
    fragment: str | None = None
    vertex: str | None = None
    font_size: float | None = None
    at: Sequence[float] | None = None
    fade: float | None = None
    blend: Blend = "normal"
    pad: float | None = None
    name: str | None = None
    emoji: bool | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        # The keywords that the scene JSON renamed, which the dataclass would
        # refuse with no word of the new name.
        for old, new in (("code", "fragment"), ("place", "vertex")):
            if old in kwargs:
                raise TypeError(
                    f"ghee.Layer takes its {new} GLSL in {new}=, got {old}="
                )
        if "alpha" in kwargs:
            raise TypeError(
                "ghee.Layer takes no alpha=: its alpha blends by blend=, as its "
                "colour does"
            )
        return super().__new__(cls)

    def to_dict(self) -> dict:
        d: dict = {} if self.name is None else {"name": self.name}
        d["text"] = self.text
        if self.vertex is not None:
            d["vertex"] = self.vertex
        if self.font_size is not None:
            d["font_size"] = _floats(self.font_size)
        if self.at is not None:
            d["at"] = _floats(self.at)
        if self.fragment is not None:
            d["fragment"] = self.fragment
        if self.blend != "normal":
            d["blend"] = self.blend
        if self.pad is not None:
            d["pad"] = _floats(self.pad)
        if self.fade is not None:
            d["fade"] = _floats(self.fade)
        if self.emoji is not None:
            d["emoji"] = _plain(self.emoji)
        return d


@dataclass
class Pass:
    """One fragment shader.

    `size` renders the pass at its own [w, h].
    The last pass leaves `size` None, because it feeds the display.
    `mipmaps` builds the mip chain of the pass's buffer after each frame, so
    a pass that reads it far smaller than its size samples it smooth.
    `layers` draw texts into the pass's buffer after its code, in order.
    A pass with layers and no code starts each frame transparent.
    `params` holds the Params of this pass alone, in a list or in a dict keyed
    by their names.
    `visible` False hides the pass: it draws a copy of the pass before it, or
    black as the first pass, its layers draw nothing, and the frame is the
    last visible pass.
    """

    name: str
    code: str = ""
    arrays: list[ArrayInput] = field(default_factory=list)
    channels: list[Channel] = field(default_factory=list)
    params: dict[str, Param] | list[Param] = field(default_factory=dict)
    filter: str = ""
    size: tuple[int, int] | None = None
    layers: list[Layer] = field(default_factory=list)
    mipmaps: bool = False
    visible: bool = True

    def __post_init__(self) -> None:
        self.params = _keyed(
            self.params, lambda n: f"param '{n}' in pass '{self.name}'"
        )

    def to_dict(self) -> dict:
        out: dict = {"name": self.name}
        if self.visible is not True:
            out["visible"] = _plain(self.visible)
        code = self.code
        if not self.layers or not isinstance(code, str) or code.strip():
            out["code"] = code
        if self.size is not None:
            out["size"] = _plain(self.size)
        if self.mipmaps is not False:
            out["mipmaps"] = _plain(self.mipmaps)
        if self.arrays:
            out["arrays"] = [a.to_dict() for a in self.arrays]
        if self.channels:
            out["channels"] = [c.to_dict() for c in self.channels]
        if self.params:
            out["params"] = {n: p.to_dict() for n, p in self.params.items()}
        if self.filter:
            out["filter"] = self.filter
        if self.layers:
            out["layers"] = [d.to_dict() for d in self.layers]
        return out


@dataclass
class Asset:
    """A named file a scene declares once.

    An image (.png, .jpg) is `uniform sampler2D <name>_tex` in every pass,
    and `vec2 <name>_size` is its size in pixels. A font (.ttf, .otf) is for a
    Text to name. `file` resolves beside the scene's script, then through
    $GHEE_ASSETS.
    """

    name: str
    file: str | os.PathLike[str]

    def to_dict(self) -> dict:
        return {"file": _plain(self.file)}


@dataclass
class Text:
    """Words that the GPU draws from the outlines of a font.

    `value` holds the words. ghee declares the scene string param of the
    text's name, and set_param on it changes the words, live.
    Every scene pass, filter passes excepted, reads them with
    float <name>_cover(vec2 p, float font_size): p in em from the top left of
    the block, y down like v_texCoord, and font_size the pixels per em.
    vec2 <name>_size is the block, in em.
    `font` is a font file (.ttf or .otf), resolved as an Asset's file is, or
    the name of a font Asset of the scene, or a list that mixes both: each
    character draws from the first font whose cmap holds it. None draws from
    the DejaVu Sans Latin that ghee carries.
    `align` lines up the lines of the block: left, center or right.
    `fill` is where a Layer of the text takes its coverage from: mtsdf reads
    an atlas of the glyphs' distance fields, fast and with glyph_dist and
    glyph_round_dist, and slug computes Slug's exact coverage from the curves,
    exact edges at any size.
    `atlas` sets the texels per em of a glyph in the MTSDF atlas,
    {"font_size": 32} by default. The atlas's field reaches the largest pad
    of the text's layers.
    """

    name: str
    font: str | os.PathLike[str] | list[str | os.PathLike[str]] | None = None
    value: str = ""
    align: Align = "left"
    fill: Fill = "mtsdf"
    atlas: Atlas | None = None

    def to_dict(self) -> dict:
        d: dict = {"value": self.value}
        if self.font is not None:
            d["font"] = _plain(self.font)
        if self.align != "left":
            d["align"] = self.align
        if self.fill != "mtsdf":
            d["fill"] = self.fill
        if self.atlas is not None:
            d["atlas"] = _plain(self.atlas)
        return d


@dataclass
class Scene:
    """The passes in render order, plus what they share.

    `params` holds the Params that reach every pass, in a list or in a dict
    keyed by their names.
    `common` is GLSL that ghee puts in front of each pass.
    `filters` are named post-process passes that ghee appends after the
    scene's own.
    `assets` are the image and font files the scene names.
    `texts` are words drawn from a font, and a Layer in a pass names one of
    them. ghee declares the string param of each text's name, which holds its
    words.
    Two assets, or two texts, of one name raise ValueError.
    """

    passes: list[Pass] = field(default_factory=list)
    version: int = 1
    filters: list[str] = field(default_factory=list)
    params: dict[str, Param] | list[Param] = field(default_factory=dict)
    common: str | None = None
    assets: list[Asset] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.params = _keyed(self.params, lambda n: f"scene param '{n}'")
        for kind, named in (("asset", self.assets), ("text", self.texts)):
            names = [x.name for x in named]
            for n in names:
                if names.count(n) > 1:
                    raise ValueError(f"{kind} '{n}' is declared twice")

    def to_dict(self) -> dict:
        out: dict = {
            "version": self.version,
            "passes": [p.to_dict() for p in self.passes],
        }
        if self.common is not None:
            out["common"] = {"code": self.common}
        if self.filters:
            out["filters"] = list(self.filters)
        if self.params:
            out["params"] = {n: p.to_dict() for n, p in self.params.items()}
        if self.assets:
            out["assets"] = {a.name: a.to_dict() for a in self.assets}
        if self.texts:
            out["texts"] = {t.name: t.to_dict() for t in self.texts}
        return out


# ---------------------------------------------------------------------- args


SIZE = "GHEE_SIZE"

# A buffer is usually asked for as a share of the frame, so the words say that
# directly. Powers of two: an odd divisor lands the buffer off the pixel grid
# and the magnify combs.
_FRACTIONS = {"full": 1, "half": 2, "quarter": 4, "eighth": 8, "sixteenth": 16}


def _render_size() -> tuple[int, int] | None:
    """ghee's next render size.

    ghee puts it in the environment before it starts the scene. A direct run
    has no ghee, so the host --size stands in, and the default matches the one
    that `ghee render` uses with no display.
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
    """A build-time argument of the scene.

    The ghee command line gives it after `--`.
    A `size` arg parses WxH to (w, h), and a share of the render size by
    name: full, half, quarter, eighth, sixteenth.
    A `bool` arg takes --name or --no-name.
    """

    name: str
    type: ArgType = "string"
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

        A size default is the WxH that the command line takes.
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


def _display_opts(s: str) -> dict:
    """K=V,... as `ghee render --display-opts` takes it: each value a Python
    literal, else its text."""
    import ast

    opts = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, eq, v = pair.partition("=")
        if not eq:
            raise argparse.ArgumentTypeError(f"wants k=v pairs, got '{pair}'")
        try:
            opts[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            opts[k.strip()] = v.strip()
    return opts


def _image_pattern(s: str) -> str:
    """An -o value: a .png or .jpg name or pattern, or '-'."""
    if s == "-" or s.lower().endswith((".png", ".jpg", ".jpeg")):
        return s
    raise argparse.ArgumentTypeError(
        f"wants an image pattern such as frame.####.png, or '-', got {s!r}; "
        "a display takes --display"
    )


def _host_flags(p: argparse.ArgumentParser) -> None:
    """What a direct run takes on top of the scene's own args.

    Each flag is the flag that `ghee render` spells the same way. With no -o
    and no --serve, the frames go to the connected display.
    A scene that declares one of these names collides here, and argparse says
    so.
    """
    p.add_argument(
        "-o",
        "--output",
        type=_image_pattern,
        metavar="PATTERN",
        help="image pattern (frame.####.png), or '-' for raw RGBA frames to stdout",
    )
    p.add_argument(
        "--display",
        nargs="?",
        const="kms",
        metavar="NAME",
        help="a display driver name, or bare or 'kms' for the connected KMS "
        "display (the default with no -o and no --serve)",
    )
    p.add_argument(
        "--display-opts",
        type=_display_opts,
        default={},
        metavar="K=V,...",
        help="options for the --display driver",
    )
    p.add_argument(
        "--serve", action="store_true", help="also run the console on this render"
    )
    p.add_argument(
        "--port",
        type=int,
        default=8090,
        metavar="PORT",
        help="the console's port (default: 8090)",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        metavar="HOST",
        help="the console's bind address (default: 127.0.0.1)",
    )
    p.add_argument(
        "--size",
        type=_size,
        metavar="WxH",
        help="WxH to render at; a display or a panel states its own",
    )
    p.add_argument("--fps", type=float, metavar="N", help="frame rate cap")
    p.add_argument(
        "--stats",
        action="store_true",
        help="print a timing summary when the render ends",
    )
    p.add_argument(
        "--frames",
        type=int,
        metavar="N",
        help="frame count (default 1 for images, loop for a display or the console)",
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

    `frame` counts the ticks, and `time` and `dt` are in seconds.
    `params` holds the string params that changed.
    The first tick holds every string param.
    """

    frame: int
    time: float
    dt: float
    params: dict[str, str]


_FEEDS: dict = {}
_HOOKS: list = []
_OUT: list = []
_CLEAN = False
# a direct run: the host flags from args(), the handle it drives, the signals
_HOST = None
_HANDLE = None
_STOPPED: list = []
# the scene of the run: the type of each param that set_param names, and the
# filters that ghee appends
_PARAMS: dict | None = None
_FILTERS: list = []


def feed(name: str):
    """Decorate the source of a named array.

    The function takes the tick and returns the values.
    It returns None to leave the array as it stands.
    A uint8 buffer goes out as bytes, and any other buffer or iterable of
    numbers as float32.
    """

    def hold(fn):
        _FEEDS[name] = fn
        return fn

    return hold


def tick(fn):
    """Decorate a per-tick hook. The hook takes the Tick, and it runs before
    the feeds."""
    _HOOKS.append(fn)
    return fn


def set_param(
    name: str,
    value: SupportsFloat | str | Iterable[SupportsFloat],
    pass_: str | None = None,
) -> None:
    """Set a live param for this tick.

    A float takes one number, a color takes "#rrggbb" or three numbers, and a
    vecN takes N.
    A string takes words, and the Text of its name lays them out again.
    `pass_` addresses the param of one pass, such as a filter param.
    A bare name is the scene param, else the first pass that declares it.

    Raises ValueError for a param that the scene does not declare, or for a
    value that does not fit the param's type, under ghee and in direct mode
    alike. The params of a filter are ghee's to check: a name that reaches
    a filter goes out unchecked.
    """
    wire = f"{pass_}.{name}" if pass_ else name
    _check(wire, value)
    if _HANDLE is not None:
        _HANDLE.set_param(name, value, pass_=pass_)
        return
    # The wire carries a color as three floats.
    rgb = None if _PARAMS and _PARAMS.get(wire) == "string" else _hex(value)
    if rgb is not None:
        value = rgb
    if isinstance(value, str):
        text = value.encode("utf-8")
        _OUT.append(_message(3, wire, len(text), text))
        return
    values = (value,) if isinstance(value, numbers.Real) else value
    count, body = _f32(values, f"param '{wire}'")
    _OUT.append(_message(1, wire, count, body))


def _param_types(scene: Scene) -> dict:
    """The type of each param that set_param names in `scene`.

    A pass param takes `pass.name`, and its bare name too when no param
    before it holds that name.
    """
    types = {n: p.type for n, p in scene.params.items()}
    for t in scene.texts:
        types[t.name] = "string"
    for ps in scene.passes:
        for n, p in ps.params.items():
            types[f"{ps.name}.{n}"] = p.type
            types.setdefault(n, p.type)
    return types


def _number(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def _hex(v) -> list[float] | None:
    """The [r, g, b] of a color written "#rrggbb", each channel 0..1, else
    None."""
    digits = "0123456789abcdefABCDEF"
    if not isinstance(v, str) or len(v) != 7 or v[0] != "#":
        return None
    if not all(c in digits for c in v[1:]):
        return None
    return [int(v[i : i + 2], 16) / 255 for i in (1, 3, 5)]


def _numbers(v, n: int) -> bool:
    if isinstance(v, (str, bytes)):
        return False
    try:
        items = list(v)
    except TypeError:
        return False
    return len(items) == n and all(_number(x) for x in items)


def _check(wire: str, value) -> None:
    """Raise ValueError unless the scene of the run declares param `wire`
    and `value` fits its type. Outside run() nothing is checked.
    """
    if _PARAMS is None:
        return
    ty = _PARAMS.get(wire)
    if ty is None:
        owner = wire.rpartition(".")[0]
        if owner in _FILTERS or (not owner and _FILTERS):
            return  # a filter pass may hold it
        raise ValueError(
            f"param '{wire}' is not a param this scene declares "
            f"({', '.join(_PARAMS) or 'none'})"
        )
    n = 3 if ty == "color" else _VEC.get(ty)
    if ty == "string":
        if not isinstance(value, str):
            raise ValueError(f"string param '{wire}' needs a string, got {value!r}")
    elif ty == "color":
        if _hex(value) is None and not _numbers(value, n):
            raise ValueError(
                f"color param '{wire}' needs '#rrggbb' or three numbers [r, g, b], "
                f"got {value!r}"
            )
    elif n is not None:
        if not _numbers(value, n):
            raise ValueError(f"{ty} param '{wire}' needs {n} numbers, got {value!r}")
    elif ty == "bool":
        if not isinstance(value, numbers.Real):
            raise ValueError(f"bool param '{wire}' needs true or false, got {value!r}")
    elif not _number(value):
        raise ValueError(f"{ty} param '{wire}' needs a number, got {value!r}")


def skip() -> None:
    """This tick needs no render. The display keeps the last frame."""
    global _CLEAN
    _CLEAN = True


def _message(kind: int, name: str, count: int, body) -> bytes:
    nb = name.encode("utf-8")
    if len(nb) > 255:
        raise ValueError(f"name {name!r} is over the wire's 255 bytes")
    return bytes((kind, len(nb))) + nb + count.to_bytes(4, "little") + body


def _view(values):
    """The buffer of `values`, or None for a source that holds no buffer."""
    try:
        return memoryview(values)
    except TypeError:
        return None


# the struct codes of one number each
_NUMBER_CODES = "?bBhHiIlLqQnNefd"


def _unpacked(mv: memoryview, what: str) -> tuple:
    """The numbers in a buffer, in C order whatever its shape."""
    fmt = mv.format
    code = fmt.lstrip("@=<>!")
    order = fmt[: len(fmt) - len(code)]
    if len(code) == 1 and code in _NUMBER_CODES:
        try:
            return struct.unpack(
                f"{order}{mv.nbytes // mv.itemsize}{code}", mv.tobytes()
            )
        except struct.error:
            pass
    raise ValueError(f"{what} takes numbers, got a buffer of format {fmt!r}")


def _f32(values, what: str) -> tuple[int, bytes]:
    """(count, little-endian f32 bytes) of a buffer or an iterable of numbers.

    A float32 buffer copies straight out. Any other buffer converts value by
    value, in C order whatever its shape and its type, and so does any other
    iterable. `what` names the values in an error.
    """
    mv = _view(values)
    if mv is not None:
        if mv.format == "f" and sys.byteorder == "little":
            return mv.nbytes // 4, mv.tobytes()
        values = _unpacked(mv, what)
    try:
        items = values if isinstance(values, (list, tuple)) else list(values)
    except TypeError:
        items = None
    if items is None or isinstance(values, str):
        raise ValueError(
            f"{what} takes a buffer or an iterable of numbers, "
            f"got {type(values).__name__}"
        )
    try:
        a = array.array("f", items)
    except TypeError:
        bad = next(x for x in items if not _converts(x))
        raise ValueError(
            f"{what} takes numbers, got {type(values).__name__} holding "
            f"{type(bad).__name__}"
        ) from None
    if sys.byteorder != "little":
        a.byteswap()
    return len(a), a.tobytes()


def _converts(x) -> bool:
    try:
        array.array("f", [x])
    except TypeError:
        return False
    return True


def _array_message(name: str, values) -> tuple[int, bytes]:
    """(count, one array message) for what a feed returned.

    A buffer of bytes goes out as kind 4: one byte for each value, which the
    engine reads as byte / 255. `bytes`, a `bytearray` and a numpy array of
    dtype uint8 are such buffers. A large mask then costs a quarter of the
    bytes, and the engine uploads it with no conversion.
    Any other buffer or iterable of numbers goes out as kind 0 floats.
    """
    mv = _view(values)
    if mv is not None and mv.format == "B":
        # the message copies the body once. A view spares a mask a second copy.
        body = mv if mv.c_contiguous else mv.tobytes()
        return mv.nbytes, _message(4, name, mv.nbytes, body)
    count, body = _f32(values, f"feed '{name}'")
    return count, _message(0, name, count, body)


def _tick(line: str) -> Tick:
    d = json.loads(line)
    return Tick(
        int(d.get("frame", 0)),
        float(d.get("time", 0.0)),
        float(d.get("dt", 0.0)),
        d.get("params") or {},
    )


def _arrays(scene: Scene) -> dict:
    """The declared arrays, keyed by name.

    A feed of an undeclared name stops the run here.
    """
    arrays = {a.name: a for p in scene.passes for a in p.arrays}
    for name in _FEEDS:
        if name not in arrays:
            raise ValueError(
                f"feed '{name}' is not an array this scene declares "
                f"({', '.join(arrays) or 'none'})"
            )
    return arrays


def run(scene: Scene) -> None:
    """Render the scene.

    A run under ghee answers the wire.
    A run of its own opens the scene on the bindings and drives it.

    Ctrl-C reaches every process in the terminal's group, so ghee takes it
    too and is already stopping. The scene ends where it stands, quietly: a
    traceback here would say a scene crashed, and it did what it was told.
    """
    global _PARAMS, _FILTERS
    _PARAMS, _FILTERS = _param_types(scene), list(scene.filters)
    try:
        if os.environ.get(WIRE):
            _wire(scene)
        else:
            _direct(scene)
    except KeyboardInterrupt:
        pass
    finally:
        _PARAMS, _FILTERS = None, []


def _wire(scene: Scene) -> None:
    """Write the handshake, then answer ticks until ghee closes stdin.

    Under a ghee that sets $GHEE_LOCKSTEP, every answer ends with kind 5, as
    the handshake's `"lockstep": true` promises.
    """
    global _CLEAN
    out = sys.stdout.buffer
    lockstep = bool(os.environ.get(LOCKSTEP))
    hello = scene.to_dict()
    if lockstep:
        hello["lockstep"] = True
    out.write(json.dumps(hello).encode("utf-8") + b"\n")
    out.flush()
    arrays = _arrays(scene)
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
            # ghee sends ticks only for a scene that it took, so the size is valid
            want = arrays[name].count
            if count != want:
                raise ValueError(f"array '{name}' takes {want} values, got {count}")
            _OUT.append(msg)
        if _CLEAN:
            _OUT.append(b"\x02")
        if lockstep:
            _OUT.append(b"\x05")
        if _OUT:
            out.write(b"".join(_OUT))
            out.flush()


# --------------------------------------------------------------- direct mode


def _host_args():
    """The host flags of a run that declared no args of its own."""
    p = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), allow_abbrev=False)
    _host_flags(p)
    return p.parse_args(sys.argv[1:])


def _script_dir() -> str | None:
    """The directory of the scene's script, where its files resolve, or None
    when no script runs, as in a REPL."""
    path = getattr(sys.modules.get("__main__"), "__file__", None)
    return os.path.dirname(os.path.abspath(path)) if path else None


def _display(a) -> str | None:
    """The display of a direct run: --display, else the connected display
    when the run names no -o and no --serve, else none."""
    if a.display is not None or a.output is not None or a.serve:
        return a.display
    return "kms"


def _open(scene: Scene, a):
    """Open the scene on the bindings."""
    from ghee import open as open_scene

    w, h = a.size or (None, None)
    return open_scene(
        scene.to_dict(),
        output=_display(a),
        opts=a.display_opts or None,
        width=w,
        height=h,
        fps=a.fps,
        stats=(a.host, a.port) if a.serve else None,
        timestep=a.timestep,
        base_dir=_script_dir(),
    )


def _frames(a) -> int:
    """--frames as given, else as `ghee render` counts them: one frame for an
    image or a pipe, and 0, a run until SIGINT or SIGTERM, for a display or the
    console."""
    if a.frames is not None:
        return a.frames
    return int(_display(a) is None and not a.serve)


def _destinations(a) -> str:
    """Where a direct run's frames go, as `ghee render --stats` names them."""
    out = []
    display = _display(a)
    if display is not None:
        out.append("kms" if display == "kms" else "panel")
    if a.output is not None:
        out.append("pipe" if a.output == "-" else "file")
    if not out:
        out.append("console")
    return "+".join(out)


def _summary(times: list, wall: float, a) -> str:
    """The line of `ghee render --stats`."""
    n = len(times)
    return (
        f"{'done':>9}  stats: {n} frame(s) in {wall:.2f}s, "
        f"{1000 * sum(times) / n:.2f} ms/frame (min {1000 * min(times):.2f}, "
        f"max {1000 * max(times):.2f}), {n / wall if wall > 0 else 0:.1f} fps -> "
        f"{_destinations(a)}"
    )


def _stop(sig, _frame) -> None:
    _STOPPED.append(sig)


def _direct(scene: Scene) -> None:
    """Drive the scene from the wire's own hooks and feeds.

    The tick keeps its wire shape: `params` holds the string params that the
    console changed, and the first tick holds every string param.
    """
    global _CLEAN, _HANDLE
    a = _HOST if _HOST is not None else _host_args()
    _arrays(scene)
    handle = _open(scene, a)
    del _STOPPED[:]
    caught = [(s, signal.signal(s, _stop)) for s in (signal.SIGINT, signal.SIGTERM)]
    _HANDLE = handle
    frames = _frames(a)
    frame, started, last = 0, time.monotonic(), 0.0
    times: list = []
    try:
        while not _STOPPED and (frames == 0 or frame < frames):
            now = (
                frame * a.timestep
                if a.timestep is not None
                else time.monotonic() - started
            )
            t = Tick(frame, now, now - last, handle.changed_params())
            last = now
            _CLEAN = False
            for hook in _HOOKS:
                hook(t)
            for name, fn in _FEEDS.items():
                values = fn(t)
                if values is not None:
                    handle.feed(name, values)
            # A file or a pipe takes every frame, as under `ghee render`.
            if _CLEAN and a.output is None:
                handle.skip()
            else:
                drawn = time.monotonic()
                handle.frame()
                if a.output == "-":
                    sys.stdout.buffer.write(handle.pixels())
                    sys.stdout.buffer.flush()
                elif a.output is not None:
                    handle.save(a.output)
                times.append(time.monotonic() - drawn)
            frame += 1
    finally:
        _HANDLE = None
        for s, held in caught:
            signal.signal(s, held)
        handle.close()
    if a.stats and times:
        print(_summary(times, time.monotonic() - started, a), file=sys.stderr)
