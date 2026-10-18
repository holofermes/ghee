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
wire. When you run the file yourself, run() renders the scene as `ghee
render` renders it, and the same feeds answer each tick in this process.
$GHEE_WIRE, which ghee sets in the processes it starts, tells the two apart.
"""

from __future__ import annotations

import argparse
import array
import functools
import json
import numbers
import os
import struct
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ghee._look import _call_look, _circle, _floats, _look, _plain  # noqa: F401
from ghee._tick import Tick

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import SupportsFloat

    from ._native import Drawing


_VEC = {"vec2": 2, "vec3": 3, "vec4": 4}


ParamType = Literal["float", "int", "bool", "color", "string", "vec2", "vec3", "vec4"]
TextAlign = Literal["left", "center", "right"]
Render = Literal["auto", "hinted", "mono", "mtsdf", "slug"]
Shape = Literal["rect", "line", "circle"]
Blend = Literal["normal", "add", "multiply", "screen"]
ArgType = Literal["string", "int", "float", "size", "bool"]


# ghee sets this in the scene process it starts. Without it, run() self-hosts.
WIRE = "GHEE_WIRE"


# ---------------------------------------------------------------- scene model


@dataclass
class Param:
    """A live value.

    The scene, the console and the scene process name it the same way.
    A color is "#rrggbb", a CSS colour name or [r, g, b] in 0..1.
    A vecK is K numbers.
    A string never reaches a shader: the scene process reads it in the tick
    line. A scene Param that shares its name with a layer fails the scene.
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
    """Words, a picture or a shape that ghee paints over the buffer of its pass.

    `name` is required. A key left at None takes ghee's default.
    `visible` False hides the layer. `x` and `y` put the block's top left at
    that pass pixel, y down, and a missing one centres the block on its axis.
    `color` is "#rrggbb", a CSS colour name such as "navy" or "transparent",
    or [r, g, b] in 0..1, `opacity` is 0 to 1, and
    `blend` paints the layer over the pass as "normal", "add", "multiply" or
    "screen".

    A text layer draws `text`, which ghee.layers sets live. Every pass reads
    the words with
    float <name>_cover(vec2 p, float font_size) and vec2 <name>_size.
    `font_size` is the pixels per em, and without it the block takes the
    largest size that fits its room. Its `color` takes a gradient too,
    {"from": ..., "to": ..., "angle": degrees}, with CSS's angle.
    `font_family` is a font file (.ttf or .otf), the name of a font Asset,
    or "DejaVu Sans" or "Pixel Operator 8", the two fonts that ghee carries,
    or a list of them, where each character draws from the first font that
    holds it, and None draws from DejaVu Sans.
    `text_align` is "left", "center" or "right", and `render` is "auto",
    "hinted", "mono", "mtsdf" or "slug".

    Each effect of a text layer is a dict of its fields, {} for every
    default, and None leaves it off. Its sizes are in em, and `visible`
    False bypasses it and keeps its values. `text_shadow` takes x, y, blur,
    color and strength, `glow` takes blur, color and strength, `text_stroke`
    takes width and color, `karaoke` takes sung and color, `wave` takes
    height and speed, `scroll` takes speed, and `blink` takes rate. Under
    `render` "slug", ghee skips text_shadow, glow and text_stroke.

    An image layer draws `image`, the name of an image Asset or a .png or
    .jpg file, over `width` by `height` pass pixels: one of them left out
    keeps the image's aspect, and both left out take its own size. Its
    `color` is a tint that multiplies the picture.

    A shape layer draws `shape`, "rect", "line" or "circle", in its `color`.
    A rect fills `width` by `height` pass pixels, the pass's own on a size
    left out, with `x`, `y` at its top left. `border` draws inside its edge,
    a dict of `width` in pixels, `color` and `visible`, and `border_radius`
    rounds its corners by that many pixels. A line runs through `points`,
    each [x, y] a pixel, `stroke_width` pixels thick, and `x`, `y` shift
    every point. A circle of `radius` pixels, the largest that fits the pass
    on a radius left out, has the top left of its box at `x`, `y`, and takes
    a `border` as a rect does.

    `ghee.layers` and a handle's `layers` set a layer's live fields while
    the scene runs.
    """

    name: str
    text: str | None = None
    visible: bool = True
    x: float | None = None
    y: float | None = None
    color: str | Sequence[float] | dict | None = None
    opacity: float | None = None
    blend: Blend = "normal"
    font_size: float | None = None
    font_family: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None
    text_align: TextAlign = "left"
    render: Render | None = None
    text_shadow: dict | None = None
    glow: dict | None = None
    text_stroke: dict | None = None
    karaoke: dict | None = None
    wave: dict | None = None
    scroll: dict | None = None
    blink: dict | None = None
    image: str | os.PathLike[str] | None = None
    width: float | None = None
    height: float | None = None
    shape: Shape | None = None
    border: dict | None = None
    border_radius: float | None = None
    points: Sequence[Sequence[float]] | None = None
    stroke_width: float | None = None
    radius: float | None = None

    def to_dict(self) -> dict:
        return _look(vars(self), self.name)


@dataclass
class Pass:
    """One fragment shader, and the layers that paint over its buffer.

    `size` renders the pass at its own [w, h], and `scale` at a share of the
    frame, one number for both sides or (x, y). A pass takes one of the two.
    The last pass leaves both None, because it feeds the display.
    `sampling` reads the pass's buffer "linear", by default, "nearest", or
    "trilinear", which builds its mip chain after each frame, so a pass that
    reads it far smaller than its size samples it smooth.
    `fragment` is the pass's GLSL. `layers` paint over the pass's buffer
    after it, in list order. A pass with layers and no fragment starts each
    frame transparent.
    `params` holds the Params of this pass alone, in a list or in a dict keyed
    by their names.
    `visible` False hides the pass: it draws a copy of the pass before it, or
    black as the first pass, its layers draw nothing, and the frame is the
    last visible pass.
    """

    name: str
    fragment: str = ""
    arrays: list[ArrayInput] = field(default_factory=list)
    channels: list[Channel] = field(default_factory=list)
    params: dict[str, Param] | list[Param] = field(default_factory=dict)
    filter: str = ""
    size: tuple[int, int] | None = None
    layers: list[Layer] = field(default_factory=list)
    sampling: Literal["nearest", "linear", "trilinear"] | None = None
    visible: bool = True
    scale: float | tuple[float, float] | None = None

    def __post_init__(self) -> None:
        self.params = _keyed(
            self.params, lambda n: f"param '{n}' in pass '{self.name}'"
        )

    def to_dict(self) -> dict:
        out: dict = {"name": self.name}
        if self.visible is not True:
            out["visible"] = _plain(self.visible)
        fragment = self.fragment
        if not self.layers or not isinstance(fragment, str) or fragment.strip():
            out["fragment"] = fragment
        if self.size is not None:
            out["size"] = _plain(self.size)
        if self.scale is not None:
            out["scale"] = _plain(self.scale)
        if self.sampling is not None:
            out["sampling"] = _plain(self.sampling)
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

    An image (.png, .jpg, or .raw bare pixels) is `uniform sampler2D
    <name>_tex` in every pass, and `vec2 <name>_size` is its size in pixels.
    A font (.ttf, .otf) is for a Layer's font_family to name. `file` resolves
    beside the scene's script, then through $GHEE_ASSETS.
    `sampling` reads an image "nearest", by default, or "linear".
    A .raw file declares its `width` and `height` in pixels, its `format`,
    "u8" or "f32", and its `components`, 1, 3 or 4, 1 by default.
    """

    name: str
    file: str | os.PathLike[str]
    sampling: Literal["nearest", "linear"] | None = None
    width: int | None = None
    height: int | None = None
    format: Literal["u8", "f32"] | None = None
    components: int | None = None

    def to_dict(self) -> dict:
        d: dict = {"file": _plain(self.file)}
        for k in ("width", "height", "format", "components", "sampling"):
            if getattr(self, k) is not None:
                d[k] = _plain(getattr(self, k))
        return d


@dataclass
class Scene:
    """The passes in render order, plus what they share.

    `params` holds the Params that reach every pass, in a list or in a dict
    keyed by their names.
    `common` is GLSL that ghee puts in front of each pass.
    `filters` are named post-process passes that ghee appends after the
    scene's own: each a filter's name, or a dict of its `name` and values
    for its params, such as `{"name": "frc", "calm": 8}`.
    `assets` are the image and font files the scene names.
    Two assets of one name raise ValueError.
    """

    passes: list[Pass] = field(default_factory=list)
    version: int = 1
    filters: list[str | dict] = field(default_factory=list)
    params: dict[str, Param] | list[Param] = field(default_factory=dict)
    common: str | None = None
    assets: list[Asset] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.params = _keyed(self.params, lambda n: f"scene param '{n}'")
        names = [a.name for a in self.assets]
        for n in names:
            if names.count(n) > 1:
                raise ValueError(f"asset '{n}' is declared twice")

    def to_dict(self) -> dict:
        out: dict = {
            "version": self.version,
            "passes": [p.to_dict() for p in self.passes],
        }
        if self.common is not None:
            out["common"] = self.common
        if self.filters:
            out["filters"] = list(self.filters)
        if self.params:
            out["params"] = {n: p.to_dict() for n, p in self.params.items()}
        if self.assets:
            out["assets"] = {a.name: a.to_dict() for a in self.assets}
        return out


# --------------------------------------------------------------- live layers


class Layers:
    """The live fields of a scene's layers, set by attribute or by key.

    A set takes the fields and the values that Handle.set_layer takes, and
    raises ValueError where set_layer raises. A handle's `layers` sets its
    scene at once. `ghee.layers` sets the scene that run() runs, from a tick
    or a feed, and under ghee each set goes out with the tick's answer.

    Move the title, set its words, sing its first twelve characters by
    attribute and again by key, and switch the icon to the image asset
    wifi_off:

        layers.title.x = 10
        layers.title.text = "hi"
        layers.title.karaoke.sung = 12
        layers["title"]["karaoke"]["sung"] = 12
        layers.icon.image = "wifi_off"
    """

    __slots__ = ("_set",)

    def __init__(self, set_layer) -> None:
        object.__setattr__(self, "_set", set_layer)

    def __getattr__(self, name: str) -> _LiveLayer:
        if name.startswith("_"):
            raise AttributeError(name)
        return _LiveLayer(self._set, name)

    def __getitem__(self, name: str) -> _LiveLayer:
        return _LiveLayer(self._set, name)


class _LiveLayer:
    """One layer of Layers: a set of a field sets it, and an effect's name
    gives the fields of that effect."""

    __slots__ = ("_name", "_set")

    def __init__(self, set_layer, name: str) -> None:
        object.__setattr__(self, "_set", set_layer)
        object.__setattr__(self, "_name", name)

    def __getattr__(self, effect: str) -> _LiveEffect:
        if effect.startswith("_"):
            raise AttributeError(effect)
        return _LiveEffect(self._set, self._name, effect)

    def __getitem__(self, effect: str) -> _LiveEffect:
        return _LiveEffect(self._set, self._name, effect)

    def __setattr__(self, field: str, value: Any) -> None:
        self._set(self._name, field, value)

    def __setitem__(self, field: str, value: Any) -> None:
        self._set(self._name, field, value)

    def __repr__(self) -> str:
        return f"<layer '{self._name}'>"


class _LiveEffect:
    """One effect of a layer of Layers: a set of a field sets it."""

    __slots__ = ("_layer", "_name", "_set")

    def __init__(self, set_layer, layer: str, name: str) -> None:
        object.__setattr__(self, "_set", set_layer)
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "_name", name)

    def __setattr__(self, field: str, value: Any) -> None:
        self._set(self._layer, field, value, self._name)

    def __setitem__(self, field: str, value: Any) -> None:
        self._set(self._layer, field, value, self._name)

    def __repr__(self) -> str:
        return f"<layer '{self._layer}' {self._name}>"


# ------------------------------------------------------------------- drawing


def gradient(start, end, angle: float = 180) -> dict:
    """A text's colour that runs from `start` to `end` along `angle` degrees,
    as CSS's angle: the gradient object of the scene format."""
    return {"from": start, "to": end, "angle": angle}


def __getattr__(name: str):
    # A drawing is the native half's: it loads on the first one.
    if name in ("Drawing", "_Screen"):
        from ghee import _ext

        return getattr(_ext(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def measure(
    words: str,
    font_size: float | None = None,
    font_family: str
    | os.PathLike[str]
    | Sequence[str | os.PathLike[str]]
    | None = None,
) -> tuple[float, float]:
    """The width and height in pixels of the block that `words` lay out in,
    as a text of the scene that run() runs lays them out on its last pass:
    at `font_size`, or at the size that fits the pass when None, in
    the fonts of `font_family`, DejaVu Sans when None.

    Raises RuntimeError outside run(), and ValueError for a font that the
    scene cannot give a text.
    """
    if _SCENE is None:
        raise RuntimeError(
            "ghee.measure lays words out in the scene that run() runs: call it in "
            "a tick or a feed"
        )
    size = render_size()
    base = _script_dir() or os.getcwd()
    return _live().measure(words, size, base, font_size, _plain(font_family))


def draw() -> Drawing:
    """A drawing of the tick's frame of the scene that run() runs, for a
    `with` block in a tick or a feed: `with ghee.draw() as d:`. Its calls
    are a screen's, and the frame of the tick draws their layers over the
    scene's own on its last pass before its filters. One block in a tick
    states the whole drawing of that frame.

    The layers go out with the tick's answer, and the native half of the
    package checks each against the scene first, so a call that ghee would
    refuse raises ValueError. Raises RuntimeError outside run().
    """
    global _DRAWN
    if _SCENE is None:
        raise RuntimeError(
            "ghee.draw draws on the scene that run() runs: call it in a tick or a feed"
        )
    from ghee import _ext

    native = _ext()
    if _DRAWN is None:
        _DRAWN = native._Screen()
    return native.Drawing(_Wire(), _DRAWN, ends_frame=False)


class _Wire:
    """The scene that ghee runs, as a drawing reaches it: each layer that the
    drawing adds goes out with the tick's answer as kind 6, the new looks
    that differ in live fields alone as one kind 8, any other new look as
    kind 7, a layer taken out as kind 9, a layer moved as kind 10, and a
    hidden layer as a live set."""

    def _tick(self) -> Tick | None:
        return _TICK

    def _drawing(self) -> dict:
        return _live().drawing()

    def _live(self):
        return _live()

    def _send(self, kind: int, wire: str, text: str) -> None:
        _send(kind, wire, text)

    def _add_layer(self, pass_: str, index: int, layer: dict) -> None:
        _send(*_live().add(pass_, index, layer))

    def _update_layer(self, layer: dict) -> None:
        message = _live().update(layer)
        if message is not None:
            _send(*message)

    def _remove_layer(self, name: str) -> None:
        _send(*_live().remove(name))

    def _move_layer(self, name: str, index: int) -> None:
        _send(*_live().move(name, index))

    def set_layer(self, name, field, value, effect=None) -> None:
        _set_layer(name, field, value, effect)


# ---------------------------------------------------------------------- args


SIZE = "GHEE_SIZE"

# A buffer is usually asked for as a share of the frame, so the words say that
# directly. Powers of two: an odd divisor lands the buffer off the pixel grid
# and the magnify combs.
_FRACTIONS = {"full": 1, "half": 2, "quarter": 4, "eighth": 8, "sixteenth": 16}


def render_size() -> tuple[int, int]:
    """The size that the scene renders at, as (w, h).

    Under ghee it is ghee's render size. A direct run takes its --size, else
    the size of the display that it drives, else 320x240, the size of
    `ghee render` with no display, and a --rotate of 90 or 270 turns it. A
    pass with a size of its own reads that size in iResolution, and every
    pass reads this size in iOutputResolution.

    Raises RuntimeError when ghee gives no WxH size.
    """
    size = _render_size()
    if size is None:
        raise RuntimeError(f"${SIZE} names no WxH render size")
    return size


def _render_size() -> tuple[int, int] | None:
    """ghee's next render size, None when $GHEE_SIZE names none.

    ghee puts it in the environment before it starts the scene. A direct run
    has no ghee, so the host flags stand in.
    """
    given = os.environ.get(SIZE)
    if given:
        return _wxh(given)
    return _direct_size(tuple(sys.argv[1:]))


@functools.cache
def _direct_size(argv: tuple[str, ...]) -> tuple[int, int] | None:
    """The render size of a direct run of `argv`, read before the parse: its
    --size, else the size of its display as ghee.open() works it out, else
    the size of `ghee render` with no display."""
    size = _flag(argv, "--size")
    if size is not None:
        size = _wxh(size)
    else:
        # the display that _display() picks
        display = _display_flag(argv)
        if display is None and _flag(argv, "-o", "--output") is None:
            display = None if "--serve" in argv else "kms"
        found = None
        if display == "kms":
            from ghee import _ext

            found = _ext()._display_size(_flag(argv, "--dri"))
        size = found or (320, 240)
    # A run turned a quarter renders its scene at the size turned.
    turn = _flag(argv, "--rotate") or ""
    if size and turn.strip() in ("90", "270"):
        size = (size[1], size[0])
    return size


def _flag(argv: tuple[str, ...], *names: str) -> str | None:
    """The value of the last of the flags `names` in `argv`, given as
    `--name value` or `--name=value`, or a one-dash flag's `-nvalue`, as
    argparse reads them, or None."""
    got = None
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                got = argv[i + 1]
            elif a.startswith(n + "="):
                got = a[len(n) + 1 :]
            elif not n.startswith("--") and a.startswith(n) and a != n:
                got = a[len(n) :]
    return got


def _display_flag(argv: tuple[str, ...]) -> str | None:
    """--display in `argv` as the host flags parse it: its name, "kms" when
    bare, or None."""
    got = None
    for i, a in enumerate(argv):
        if a == "--display":
            name = argv[i + 1] if i + 1 < len(argv) else "-"
            got = "kms" if name.startswith("-") else name
        elif a.startswith("--display="):
            got = a[len("--display=") :]
    return got


def _wxh(s: str) -> tuple[int, int] | None:
    """WxH as two ints, None for any other text."""
    try:
        w, h = s.lower().split("x")
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
    A direct run takes the host flags as well, and run() hands them to ghee.
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
        return _known(p, argv)
    _HOST = _host_parse(p, argv)
    return _HOST


def _known(p: argparse.ArgumentParser, argv: list[str]):
    """`argv` parsed by `p`, with every flag that `p` lacks left out and named
    on stderr, one line each with the values that follow it. `ghee serve`
    hands every scene the same arguments, and each scene takes its own."""
    got, rest = p.parse_known_args(argv)
    flags: list[list[str]] = []
    for token in rest:
        if token.startswith("-") or not flags:
            flags.append([token])
        else:
            flags[-1].append(token)
    for flag in flags:
        print(f"{p.prog}: ignoring {' '.join(flag)}", file=sys.stderr)
    return got


# The value of a flag given bare that takes a value or none, as --display.
_BARE = "\0bare"


def _host_flags(p: argparse.ArgumentParser) -> None:
    """The flags of `ghee render` that a direct run takes on top of the
    scene's own args, with ghee render's help. A scene that declares one of
    these names collides here, and argparse says so.
    """
    from ghee import _ext

    for f in _ext()._render_flags():
        names = [f"-{f['short']}"] if f["short"] else []
        names.append(f"--{f['long']}")
        kw: dict = {"help": f["help"]}
        if f["default"] is not None:
            kw["help"] = f"{f['help']} [default: {f['default']}]".lstrip()
        if f["takes"] == 0:
            kw["action"] = "store_true"
        elif f["metavar"]:
            kw["metavar"] = f["metavar"]
        if f["takes"] == 2:
            kw.update(nargs="?", const=_BARE)
        p.add_argument(*names, **kw)
    p.epilog = "With no -o and no --serve, the frames go to the connected display."


def _host_parse(p: argparse.ArgumentParser, argv: list[str]):
    """`argv` parsed by `p` and the host flags, which ghee checks as `ghee
    render` checks them: a value that it refuses stops the run here."""
    from ghee import _ext

    _host_flags(p)
    got = _known(p, argv)
    why = _ext()._render_check(_flags(got))
    if why:
        p.error(why)
    return got


def _flags(a) -> list[str]:
    """The host flags that `a` holds, as `ghee render` takes them."""
    from ghee import _ext

    out = []
    for f in _ext()._render_flags():
        value = getattr(a, f["long"].replace("-", "_"), None)
        if value is None or value is False:
            continue
        flag = f"--{f['long']}"
        out.append(flag if value is True or value == _BARE else f"{flag}={value}")
    return out


# ------------------------------------------------------------------ the loop


_FEEDS: dict = {}
_HOOKS: list = []
_OUT: list = []
_CLEAN = False
# a direct run: the host flags from args()
_HOST = None
# the scene of the run: the type of each param that set_param names, made on
# the first set, and the filters that ghee appends
_PARAMS: dict | None = None
_FILTERS: list = []
# the scene of the run, under ghee the native check of its layer sets, and
# what ghee.draw() keeps from one tick to the next
_SCENE: Scene | None = None
_LIVE = None
_DRAWN = None
# the tick that the wire handed this iteration
_TICK: Tick | None = None


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
    in_pass: str | None = None,
) -> None:
    """Set a live param for this tick.

    A float takes one number, a color takes "#rrggbb", a CSS colour name or
    three numbers, a vecN takes N, and a string takes text.
    `in_pass` addresses the param of one pass, such as a filter param.
    A bare name is the scene param, else the first pass that declares it.

    Raises ValueError for a param that the scene does not declare, or for a
    value that does not fit the param's type, under ghee and in direct mode
    alike. The params of a filter are ghee's to check: a name that reaches
    a filter goes out unchecked.
    """
    wire = f"{in_pass}.{name}" if in_pass else name
    _check(wire, value)
    # The wire carries a color written "#rrggbb" as three floats, and a
    # colour name as text.
    rgb = None if _SCENE and _types().get(wire) == "string" else _hex(value)
    if rgb is not None:
        value = rgb
    if isinstance(value, str):
        text = value.encode("utf-8")
        _OUT.append(_message(3, wire, len(text), text))
        return
    values = (value,) if isinstance(value, numbers.Real) else value
    count, body = _f32(values, f"param '{wire}'")
    _OUT.append(_message(1, wire, count, body))


def _set_layer(name: str, field: str, value: Any, effect: str | None = None) -> None:
    """Set a live field of a layer for this tick, the set of `layers`.

    The native half of the wheel checks the set against the scene, as
    Handle.set_layer does, and gives the message.
    """
    if _SCENE is None:
        raise RuntimeError(
            "ghee.layers sets a layer of the scene that run() runs: set it in a "
            "tick or a feed"
        )
    kind, wire, body = _live().message(name, field, value, effect)
    _send(kind, wire, body, f"layer '{name}'")


def _live():
    """The native check of the scene that run() runs, made on the first
    use."""
    global _LIVE
    if _LIVE is None:
        from ghee import _ext

        _LIVE = _ext()._LayerWire(_SCENE.to_dict())
    return _LIVE


def _send(kind: int, wire: str, body, what: str = "") -> None:
    """Queue one message for the tick's answer: `body` the text of kinds 3
    and 6 to 10, or the numbers of kinds 0 and 1, which `what` names in an
    error."""
    if isinstance(body, str):
        text = body.encode("utf-8")
        _OUT.append(_message(kind, wire, len(text), text))
        return
    count, data = _f32(body, what)
    _OUT.append(_message(kind, wire, count, data))


# The live fields of the layers of the scene that run() runs, set from a tick
# or a feed: ghee.layers.title.x = 10.
layers = Layers(_set_layer)


def _types() -> dict:
    """The type of each param that set_param names in the scene that run()
    runs, by its wire name, from the native check."""
    global _PARAMS
    if _PARAMS is None:
        _PARAMS = dict(_live().params())
    return _PARAMS


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


def _colour_name(v) -> bool:
    """True for a CSS colour name, in any case. The native half holds the
    names."""
    if not isinstance(v, str) or not v.isascii() or not v.isalpha():
        return False
    from ghee import _ext

    return _ext()._colour_name(v)


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
    if _SCENE is None:
        return
    ty = _types().get(wire)
    if ty is None:
        owner = wire.rpartition(".")[0]
        if owner in _FILTERS or (not owner and _FILTERS):
            return  # a filter pass may hold it
        raise ValueError(
            f"param '{wire}' is not a param this scene declares "
            f"({', '.join(_types()) or 'none'})"
        )
    n = 3 if ty == "color" else _VEC.get(ty)
    if ty == "string":
        if not isinstance(value, str):
            raise ValueError(f"string param '{wire}' needs a string, got {value!r}")
    elif ty == "color":
        if _hex(value) is None and not _numbers(value, n) and not _colour_name(value):
            raise ValueError(
                f"color param '{wire}' needs '#rrggbb', a CSS colour name or three "
                f"numbers [r, g, b], got {value!r}"
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
    return _tick_of(json.loads(line))


def _tick_of(d: dict) -> Tick:
    """The Tick of a tick as the wire and a handle write it."""
    return Tick(
        int(d.get("frame", 0)),
        float(d.get("time", 0.0)),
        float(d.get("dt", 0.0)),
        dict(d.get("params") or {}),
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


# The GLSL of the scene of a run that names none: one pass, black.
_BLACK = "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }"


def run(scene: Scene | None = None) -> None:
    """Render the scene. With no scene, the scene is one pass, `main`, that
    clears to black, for ghee.draw() to paint on.

    A run under ghee answers the wire.
    A run of its own renders the scene as `ghee render` renders it, and
    answers each tick in this process.

    Ctrl-C reaches every process in the terminal's group, so ghee takes it
    too and is already stopping. The scene ends where it stands, quietly: a
    traceback here would say a scene crashed, and it did what it was told.
    """
    global _PARAMS, _FILTERS, _SCENE, _LIVE, _DRAWN, _TICK
    if scene is None:
        scene = Scene(passes=[Pass("main", _BLACK)])
    _PARAMS = None
    _FILTERS = [f["name"] if isinstance(f, dict) else f for f in scene.filters]
    _SCENE = scene
    _DRAWN = None
    try:
        if os.environ.get(WIRE):
            _wire(scene)
        else:
            _direct(scene)
    except KeyboardInterrupt:
        pass
    finally:
        _PARAMS, _FILTERS, _SCENE, _LIVE, _TICK = None, [], None, None, None


def _wire(scene: Scene) -> None:
    """Write the handshake, then answer ticks until ghee closes stdin.

    Every answer ends with kind 5, as the handshake's `"lockstep": true`
    promises.
    """
    out = sys.stdout.buffer
    out.write(_hello(scene).encode("utf-8") + b"\n")
    out.flush()
    arrays = _arrays(scene)
    for line in sys.stdin:
        if not line.strip():
            continue
        out.write(_answer(arrays, line))
        out.flush()


def _hello(scene: Scene) -> str:
    """The handshake: the scene, which answers every tick in lockstep."""
    hello = scene.to_dict()
    hello["lockstep"] = True
    return json.dumps(hello)


def _answer(arrays: dict, line: str) -> bytes:
    """The answer to the tick of `line`: what the hooks and the feeds send,
    kind 5 last."""
    global _CLEAN, _TICK
    t = _TICK = _tick(line)
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
    _OUT.append(b"\x05")
    return b"".join(_OUT)


# --------------------------------------------------------------- direct mode


def _host_args():
    """The host flags of a run that declared no args of its own."""
    p = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), allow_abbrev=False)
    return _host_parse(p, sys.argv[1:])


def _script_dir() -> str | None:
    """The directory of the scene's script, where its files resolve, or None
    when no script runs, as in a REPL."""
    path = getattr(sys.modules.get("__main__"), "__file__", None)
    return os.path.dirname(os.path.abspath(path)) if path else None


def _direct(scene: Scene) -> None:
    """Render the scene as `ghee render` renders a scene file that is a
    program, with the host flags. ghee asks this process for the scene, and
    for the answer to each tick, as it asks a process on the wire.

    Raises what a hook or a feed raises, and SystemExit with the status of
    `ghee render` when that is not 0.
    """
    from ghee import _ext

    a = _HOST if _HOST is not None else _host_args()
    arrays = _arrays(scene)

    def start(_w: int, _h: int) -> str:
        global _DRAWN
        _DRAWN = None
        return _hello(scene)

    main = getattr(sys.modules.get("__main__"), "__file__", None)
    path = os.path.abspath(main) if main else os.path.join(os.getcwd(), "scene")
    status = _ext()._direct(
        os.path.basename(sys.argv[0]),
        path,
        _flags(a),
        start,
        lambda line: _answer(arrays, line),
        a.dri,
    )
    if status:
        raise SystemExit(status)
