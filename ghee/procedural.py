"""Procedural scenes: a scene can ship a Python controller alongside (or instead of)
its JSON. The controller builds the scene and/or feeds it data each frame, so a demo
can showcase custom arrays and per-frame logic with no host changes.

A scene package is a directory `<name>/`, a `<name>.scene.json`, or a `<name>.py`,
holding any of:
  <name>.scene.json   the declarative shape (passes, arrays, params)
  <name>.py           a SceneController subclass, or a load() factory

load_scene() returns (scene, controller). Drive it: setup(r) once, then each frame
update(r, frame, t) and render with provide() as the array source. A pure-JSON scene
(no controller) still works.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys

from ghee.scene import Scene

_SUFFIX = ".scene.json"


class SceneController:
    """Hooks a scene into the render loop. Every method is optional."""

    def build(self, base: Scene | None) -> Scene | None:
        """Return the scene to render. `base` is the loaded JSON scene, or None.
        Override to generate passes in code, or to augment the JSON."""
        return base

    def setup(self, renderer) -> None:
        """Called once after the scene compiles (init state, future: load textures)."""

    def provide(self, name: str, size: int, frame: int):
        """Floats for a declared array this frame, or None to fall through to the
        host's source / a default."""
        return None

    def update(self, renderer, frame: int, t: float) -> None:
        """Per-frame hook before the draw (e.g. set live params)."""


def _import_file(path):
    mod_name = "ghee_scene_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    base_dir = os.path.dirname(os.path.abspath(path))
    sys.path.insert(0, base_dir)  # let the scene import sibling modules
    try:
        spec.loader.exec_module(mod)
    finally:
        if sys.path and sys.path[0] == base_dir:
            sys.path.pop(0)
    return mod


def _controller_in(mod):
    if callable(getattr(mod, "load", None)):
        return mod.load()
    for _name, obj in inspect.getmembers(mod, inspect.isclass):
        if (
            obj is not SceneController
            and issubclass(obj, SceneController)
            and obj.__module__ == mod.__name__
        ):
            return obj()
    return None


def _resolve(path):
    """(json_path | None, py_path | None) for a directory, a .scene.json, or a .py."""
    if os.path.isdir(path):
        stem = os.path.join(path, os.path.basename(os.path.normpath(path)))
    elif path.endswith(".py"):
        stem = path[:-3]
    elif path.endswith(_SUFFIX):
        stem = path[: -len(_SUFFIX)]
    else:
        stem = os.path.splitext(path)[0]
    j, p = stem + _SUFFIX, stem + ".py"
    return (j if os.path.isfile(j) else None, p if os.path.isfile(p) else None)


def load_scene(path):
    """Load a scene package -> (Scene, SceneController | None). Accepts a directory, a
    .scene.json, or a .py."""
    json_path, py_path = _resolve(path)
    base = Scene.load(json_path) if json_path else None
    controller = _controller_in(_import_file(py_path)) if py_path else None
    scene = controller.build(base) if controller is not None else base
    if scene is None:
        raise ValueError(
            f"no scene at {path}: need a {_SUFFIX} or a controller with build()"
        )
    return scene, controller
