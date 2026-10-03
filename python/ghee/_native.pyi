"""The renderer that ghee.open opens, the native half of ghee-render."""

import os
from collections.abc import Iterable
from typing import Any, Literal, SupportsFloat, TypeAlias, TypedDict

from typing_extensions import Buffer, Self

from ._shim import Scene

__version__: str

# A display driver of this build, or "kms" for the connected KMS display.
_Output: TypeAlias = Literal["kms", "st7789", "ssd1306"]

class _Tick(TypedDict):
    """One tick: the frame's number, its time and the time since the last."""

    frame: int
    time: float
    dt: float

class Handle:
    """A rendering scene, and the output it draws to."""

    @property
    def size(self) -> tuple[int, int]: ...
    @property
    def components(self) -> Literal[3, 4]: ...
    def set_param(
        self,
        name: str,
        value: SupportsFloat | str | Iterable[SupportsFloat],
        pass_: str | None = None,
    ) -> None: ...
    def feed(self, name: str, data: Buffer | Iterable[float]) -> None: ...
    def frame(self) -> _Tick: ...
    def skip(self) -> _Tick: ...
    def changed_params(self) -> dict[str, str]: ...
    def pixels(self) -> bytes: ...
    def save(self, path: str | os.PathLike[str]) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *_args: object) -> bool: ...

def open(
    scene: str | os.PathLike[str] | dict[str, Any] | Scene,
    output: _Output | None = None,
    opts: dict[str, Any] | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    stats: int | tuple[str, int] | None = None,
    stream_max: float | None = None,
    verbose: bool = False,
    timestep: float | None = None,
    alpha: bool = False,
    base_dir: str | os.PathLike[str] | None = None,
) -> Handle: ...
