"""ghee: render a GLSL scene from Python, or write one.

The package holds two surfaces. `ghee.open` drives the renderer, and it needs
the native half of the wheel. Everything else is the procedural-scene shim,
which is Python alone. python/README.md and docs/procedural.md hold them.
"""

from __future__ import annotations

import importlib

from ._shim import (
    Arg,
    ArrayInput,
    Asset,
    Channel,
    Param,
    Pass,
    Scene,
    Text,
    Tick,
    args,
    feed,
    run,
    set_param,
    skip,
    tick,
)

__all__ = [
    "Arg",
    "ArrayInput",
    "Asset",
    "Channel",
    "Handle",
    "Param",
    "Pass",
    "Scene",
    "Text",
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
