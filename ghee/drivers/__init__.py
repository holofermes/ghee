"""Display driver discovery.

Bundled drivers ship in this package and need the ghee[drivers] extra to import (the
panel libraries are hardware-only). Third-party drivers register through a "ghee.drivers"
entry point. available_drivers() reports which ones import on this host.
"""

import importlib
import importlib.metadata as _md

from ghee.drivers.base import DisplayDriver, register

__all__ = ["DisplayDriver", "register", "available_drivers", "get_driver"]

_BUNDLED = ["st7789", "ssd1306"]


def _entry_points():
    try:
        return {ep.name: ep for ep in _md.entry_points(group="ghee.drivers")}
    except Exception:
        return {}


def available_drivers():
    """name -> "ok" or the import error, across bundled and entry-point drivers."""
    out = {}
    for name in _BUNDLED:
        try:
            importlib.import_module(f"ghee.drivers.{name}")
            out[name] = "ok"
        except Exception as e:
            out[name] = str(e)
    for name, ep in _entry_points().items():
        if name in out:
            continue
        try:
            ep.load()
            out[name] = "ok"
        except Exception as e:
            out[name] = str(e)
    return out


def get_driver(name):
    """Return the driver class registered under `name`, importing it first."""
    from ghee.drivers.base import _REGISTRY

    if name in _BUNDLED:
        try:
            importlib.import_module(f"ghee.drivers.{name}")
        except ImportError as e:
            raise ImportError(
                f"display driver '{name}' needs the panel libraries: "
                f"pip install 'ghee-render[drivers]' ({e})"
            ) from e
    elif name not in _REGISTRY:
        ep = _entry_points().get(name)
        if ep is not None:
            ep.load()
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"no display driver '{name}'")
    return cls
