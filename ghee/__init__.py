from importlib.metadata import PackageNotFoundError, version

from ghee.engine import UNCHANGED
from ghee.filters import apply_filters, load_filters
from ghee.procedural import SceneController, load_scene
from ghee.scene import ArrayInput, Channel, Param, Pass, Scene

try:
    __version__ = version("ghee-render")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "UNCHANGED",
    "Scene",
    "Pass",
    "Param",
    "Channel",
    "ArrayInput",
    "SceneController",
    "load_scene",
    "load_filters",
    "apply_filters",
    "__version__",
]
