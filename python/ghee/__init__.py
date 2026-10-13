"""ghee: render a GLSL scene from Python, or write one.

The package holds two surfaces. `ghee.open` drives the renderer, and it needs
the native half of the wheel. Everything else is the procedural-scene shim,
which is Python alone until a scene that ghee runs sets a layer through
`ghee.layers`, draws with `ghee.draw`, sets a color param to a CSS colour
name, or measures words with `ghee.measure`: the native half then checks
each set and each drawn layer against the scene, holds the names, and lays
the words out.
docs/python.md and docs/procedural.md hold them.
"""

TYPE_CHECKING = False
if TYPE_CHECKING:
    from ._native import Handle, open
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
        draw,
        feed,
        gradient,
        layers,
        measure,
        render_size,
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
    "Layer",
    "Param",
    "Pass",
    "Scene",
    "Tick",
    "__version__",
    "args",
    "draw",
    "feed",
    "gradient",
    "layers",
    "measure",
    "open",
    "render_size",
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
        from . import _native
    except ImportError as e:
        import sysconfig

        raise ImportError(
            "ghee._native is missing. It is the native half of ghee-render, "
            f"and this install carries none for {sysconfig.get_platform()}."
        ) from e
    return _native


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
    if name in __all__:
        # The ghee command imports this package, and runs without the shim.
        from . import _shim

        globals().update({n: getattr(_shim, n) for n in __all__ if n in vars(_shim)})
        return globals()[name]
    raise AttributeError(f"module 'ghee' has no attribute '{name}'")


def __dir__():
    return sorted({*globals(), *__all__})
