"""ghee: render a GLSL scene from Python, or write one.

The package holds two surfaces. `ghee.open` drives the renderer, and it needs
the native half of the wheel. Everything else is the procedural-scene shim,
which is Python alone until a scene that ghee runs sets a layer through
`ghee.layers`: the native half then checks each set against the scene.
python/README.md and docs/procedural.md hold them.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from ._shim import (
    Arg,
    ArrayInput,
    Asset,
    Channel,
    Layer,
    Param,
    Pass,
    Scene,
    Tick,
    args,
    feed,
    layers,
    run,
    set_param,
    skip,
    tick,
)

if TYPE_CHECKING:
    from ._native import Handle, open

__all__ = [
    "Arg",
    "ArrayInput",
    "Asset",
    "Channel",
    "Handle",
    "Layer",
    "Param",
    "Pass",
    "Scene",
    "Tick",
    "__version__",
    "args",
    "feed",
    "layers",
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
    import sysconfig

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
    if name == "__version__":
        # importlib.metadata reads the installed metadata, and reading it costs
        # about a second on a Pi Zero: it pulls in email, inspect and zipfile
        # to do it. A scene asks for the version about never, and every scene
        # pays this at import, so the answer waits until someone asks.
        from importlib.metadata import version

        return version("ghee-render")
    raise AttributeError(f"module 'ghee' has no attribute '{name}'")
