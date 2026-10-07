"""The renderer that ghee.open opens, the native half of ghee-render."""

import os
from collections.abc import Iterable
from typing import Any, Literal, SupportsFloat, TypeAlias, TypedDict

from typing_extensions import Buffer, Self

from ._shim import Drawing, Layers, Scene

__version__: str

# A display driver of this build, or "kms" for the connected KMS display.
_Display: TypeAlias = Literal["kms", "st7789", "ssd1306"]

class _Tick(TypedDict):
    """One tick: the frame's number, its time, the DT that it draws with,
    and the string params that changed since the last tick."""

    frame: int
    time: float
    dt: float
    params: dict[str, str]

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
