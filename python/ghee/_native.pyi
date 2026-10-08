"""The renderer that ghee.open opens, the native half of ghee-render."""

import os
from collections.abc import Iterable, Sequence
from typing import Any, Literal, SupportsFloat, TypeAlias, TypedDict

from typing_extensions import Buffer, Self

from ._shim import Blend, Layers, Render, Scene, TextAlign

__version__: str

# The connected KMS display.
_Display: TypeAlias = Literal["kms"]

class _Tick(TypedDict):
    """One tick: the frame's number, its time, the iTimeDelta that it draws
    with, and the params that changed since the last tick."""

    frame: int
    time: float
    dt: float
    params: dict[str, object]

class Handle:
    """A rendering scene, and the output it draws to."""

    @property
    def size(self) -> tuple[int, int]: ...
    @property
    def components(self) -> Literal[3, 4]: ...
    @property
    def layers(self) -> Layers: ...
    @property
    def brightness(self) -> float | None: ...
    @brightness.setter
    def brightness(self, level: float) -> None: ...
    def draw(self) -> Drawing: ...
    def measure(
        self,
        words: str,
        font_size: float | None = None,
        font_family: str | os.PathLike[str] | None = None,
    ) -> tuple[float, float]: ...
    def set_param(
        self,
        name: str,
        value: SupportsFloat | str | Iterable[SupportsFloat],
        pass_: str | None = None,
    ) -> None: ...
    def set_layer(
        self,
        name: str,
        field: str,
        value: SupportsFloat | str | Iterable[SupportsFloat] | dict[str, Any] | None,
        effect: str | None = None,
    ) -> None: ...
    def feed(self, name: str, data: Buffer | Iterable[SupportsFloat]) -> None: ...
    def frame(self) -> _Tick: ...
    def skip(self) -> _Tick: ...
    def pixels(self) -> bytes: ...
    def save(self, path: str | os.PathLike[str]) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *_args: object) -> bool: ...

# A colour as a layer takes it: "#rrggbb", a CSS colour name, or [r, g, b]
# in 0..1.
_Colour: TypeAlias = str | Sequence[float]

class _Screen:
    """What a screen's drawing keeps from one frame to the next: the layer of
    each call, and the values that each layer shows."""

class Drawing:
    """One frame of a screen, the block of `with screen.draw() as d:`, or of
    the scene that run() runs, the block of `with ghee.draw() as d:` in a
    tick or a feed.

    Each call writes a layer. Leaving a screen's block draws the frame, and
    leaving the block of ghee.draw() hands its layers to the tick's frame.
    A call takes the keys of its layer as keywords, the effects included, as
    ghee.Layer takes them, and `name`. Positions and box sizes are pixels
    from the top left, y down, on the screen's last pass before its filters,
    and the drawing paints over the scene's own layers there, in the order
    of the calls.

    Each frame states the whole look of each call: a keyword left out takes
    its default again. A call keeps its layer from frame to frame by the line
    of code that makes it and its count from that line in the frame, or by
    its `name`, which names the layer too. A call with no name names its
    layer after its kind: text, text2, image, rect, line. Two calls that draw
    one layer in a frame raise ValueError.

    Leaving the block sends ghee what changed: a live field changes live,
    any other change builds that layer alone again, a call that runs for the
    first time adds its layer after the layer of the call before it, a layer
    whose call did not run hides, and an effect that a call stops naming is
    bypassed. A frame whose calls changed nothing sends nothing, and draws.
    """

    def __init__(
        self, handle: Any, screen: _Screen, ends_frame: bool = True
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *_args: object) -> bool: ...
    def text(
        self,
        words: str,
        *,
        name: str | None = None,
        x: float | None = None,
        y: float | None = None,
        color: _Colour | dict[str, Any] | None = None,
        opacity: float | None = None,
        blend: Blend | None = None,
        visible: bool | None = None,
        font_size: float | None = None,
        font_family: str | os.PathLike[str] | None = None,
        text_align: TextAlign | None = None,
        render: Render | None = None,
        text_shadow: dict[str, Any] | None = None,
        glow: dict[str, Any] | None = None,
        text_stroke: dict[str, Any] | None = None,
        karaoke: dict[str, Any] | None = None,
        wave: dict[str, Any] | None = None,
        scroll: dict[str, Any] | None = None,
        blink: dict[str, Any] | None = None,
    ) -> None:
        """Draw `words`, a text layer. With no `font_size` the words fit the
        screen, and an axis with no position centres them."""
    def image(
        self,
        image: str | os.PathLike[str],
        *,
        name: str | None = None,
        x: float | None = None,
        y: float | None = None,
        width: float | None = None,
        height: float | None = None,
        color: _Colour | None = None,
        opacity: float | None = None,
        blend: Blend | None = None,
        visible: bool | None = None,
    ) -> None:
        """Draw `image`, an image layer of a .png or .jpg file or an image
        asset of the scene. A file loads once and stays with the screen."""
    def rect(
        self,
        *,
        name: str | None = None,
        x: float | None = None,
        y: float | None = None,
        width: float | None = None,
        height: float | None = None,
        color: _Colour | None = None,
        opacity: float | None = None,
        blend: Blend | None = None,
        visible: bool | None = None,
        border: dict[str, Any] | None = None,
        border_radius: float | None = None,
    ) -> None:
        """Draw a box, a rect shape layer, the whole screen on a size left
        out."""
    def circle(
        self,
        centre: Sequence[float],
        radius: float,
        *,
        name: str | None = None,
        color: _Colour | None = None,
        opacity: float | None = None,
        blend: Blend | None = None,
        visible: bool | None = None,
        border: dict[str, Any] | None = None,
    ) -> None:
        """Draw a circle of `radius` pixels around `centre`, (x, y): a rect
        of `2 * radius` with `border_radius` at `radius`."""
    def line(
        self,
        a: Sequence[float],
        b: Sequence[float],
        *,
        name: str | None = None,
        x: float | None = None,
        y: float | None = None,
        color: _Colour | None = None,
        opacity: float | None = None,
        blend: Blend | None = None,
        visible: bool | None = None,
        stroke_width: float | None = None,
    ) -> None:
        """Draw a line from `a` to `b`, each (x, y): `lines([a, b])`."""
    def lines(
        self,
        points: Sequence[Sequence[float]],
        *,
        name: str | None = None,
        x: float | None = None,
        y: float | None = None,
        color: _Colour | None = None,
        opacity: float | None = None,
        blend: Blend | None = None,
        visible: bool | None = None,
        stroke_width: float | None = None,
    ) -> None:
        """Draw a line through `points`, each (x, y), two or more. `x` and
        `y` shift every point."""

def _colour_name(text: str) -> bool:
    """True for a CSS colour name, in any case, transparent too."""

class _LayerWire:
    """A scene that a scene process runs under ghee, to check each live layer
    set and each drawn layer against and to give its message."""

    def __init__(self, scene: dict[str, Any]) -> None: ...
    def drawing(self) -> dict[str, Any]: ...
    def add(
        self, pass_: str, index: int, layer: dict[str, Any], /
    ) -> tuple[int, str, str]: ...
    def update(self, layer: dict[str, Any]) -> tuple[int, str, str] | None: ...
    def message(
        self,
        name: str,
        field: str,
        value: SupportsFloat | str | Iterable[SupportsFloat] | dict[str, Any] | None,
        effect: str | None = None,
    ) -> tuple[int, str, list[float] | str]: ...
    def measure(
        self,
        words: str,
        size: tuple[float, float],
        base_dir: str,
        font_size: float | None = None,
        font_family: str | None = None,
    ) -> tuple[float, float]: ...

def _numbered(pattern: str) -> bool:
    """True when the image file name `pattern` numbers its frames."""

def _display_size(
    dri: str | os.PathLike[str] | None = None,
) -> tuple[int, int] | None:
    """The mode of the connected display on card `dri`, else on the first
    card with one, worked out before anything opens it. None for no
    connected display."""

def open(
    scene: str | os.PathLike[str] | dict[str, Any] | Scene | None = None,
    display: _Display | None = None,
    opts: dict[str, Any] | None = None,
    dri: str | os.PathLike[str] | None = None,
    rotate: Literal[0, 90, 180, 270] = 0,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    serve: int | tuple[str, int] | None = None,
    stream_max: float | None = None,
    verbose: bool = False,
    timestep: float | None = None,
    alpha: bool = False,
    base_dir: str | os.PathLike[str] | None = None,
) -> Handle: ...
