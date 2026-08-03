"""ghee: render a GLSL scene from Python, or write one.

The package holds two surfaces. `ghee.open` drives the renderer, and it needs
the native half of the wheel. Everything else is the procedural-scene shim,
which is Python alone. python/README.md and docs/procedural.md hold them.
"""

from __future__ import annotations

import importlib
import sysconfig
from importlib.metadata import version

from ._shim import (
    Arg,
    ArrayInput,
    Channel,
    Param,
    Pass,
    Scene,
    Tick,
    args,
    feed,
    run,
    set_param,
    skip,
    tick,
)

__version__ = version("ghee-render")

__all__ = [
    "Arg",
    "ArrayInput",
    "Channel",
    "Handle",
    "Param",
    "Pass",
    "Scene",
    "Tick",
    "__version__",
    "args",
    "feed",
    "open",
    "run",
    "set_param",
    "skip",
    "tick",
]


def _ext():
    """The native module, imported on the first use.

    A scene that only builds and feeds runs without it.
    """
    try:
        return importlib.import_module("ghee._native")
    except ImportError as e:
        raise ImportError(
            "ghee._native is missing. It is the native half of ghee-render, "
            f"and this install carries none for {sysconfig.get_platform()}."
        ) from e


def __getattr__(name):
    if name in ("open", "Handle"):
        return getattr(_ext(), name)
    raise AttributeError(f"module 'ghee' has no attribute '{name}'")
