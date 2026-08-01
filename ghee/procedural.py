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

A controller can declare build-time arguments (ARGS, a list of Arg): the values
come off the command line after `--` and land on it as `self.args` before
build() runs, which is where array sizes and shader constants get frozen.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ghee.scene import Scene
from ghee.sources import record

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

_SUFFIX = ".scene.json"


def _size(s: str) -> tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        wh = (int(w), int(h))
    except ValueError:
        raise argparse.ArgumentTypeError(f"want WxH, got {s!r}") from None
    if min(wh) < 1:
        raise argparse.ArgumentTypeError(f"want a positive WxH, got {s!r}")
    return wh


_ARG_TYPES = {
    "string": {"type": str},
    "int": {"type": int},
    "float": {"type": float},
    "size": {"type": _size, "metavar": "WxH"},
    "bool": {"action": argparse.BooleanOptionalAction},
}


@dataclass
class Arg:
    """A build-time argument a scene accepts, declared on its controller as
    ARGS and given after `--` on the command line:

        ghee render --demo bluesun -- --emit 480x300

    A `size` arg parses WxH to (w, h); a `bool` arg takes --name / --no-name.
    The parsed values reach the controller as `self.args` before build(), so an
    arg can set an array size, a pass size or a shader constant, none of which a
    live param can move. __init__ runs before the parse and sees the declared
    defaults, so read them from build() on."""

    name: str
    type: str = "string"
    default: Any = None
    help: str = ""

    def __post_init__(self) -> None:
        if self.type not in _ARG_TYPES:
            raise ValueError(
                f"arg '{self.name}' has unknown type {self.type!r}; "
                f"known types: {', '.join(_ARG_TYPES)}"
            )
        if self.name.startswith("-") or not self.name.replace("-", "_").isidentifier():
            raise ValueError(
                f"arg name {self.name!r} has to be a word, it becomes an option"
            )
        if self.name in ("h", "help"):
            raise ValueError(f"arg name {self.name!r} is taken by the help option")


def declare_args(parser, specs) -> None:
    """Add declared Args to an argparse parser or one of its argument groups."""
    for spec in specs:
        d = spec.default
        shown = f"{d[0]}x{d[1]}" if spec.type == "size" and d is not None else d
        help_ = spec.help if d is None else f"{spec.help} (default: {shown})".lstrip()
        try:
            parser.add_argument(
                f"--{spec.name}", default=d, help=help_, **_ARG_TYPES[spec.type]
            )
        except argparse.ArgumentError as e:
            raise ValueError(f"scene arg: {e}") from None


def parse_scene_args(specs, tokens, prog="scene --", strict=True):
    """The tokens after `--` against a scene's declared Args -> a namespace of
    values. strict: the parser owns --help and rejects anything undeclared.
    Lenient does neither, for a host handing one set of args to whichever scene
    it loads."""
    p = argparse.ArgumentParser(prog=prog, add_help=strict, allow_abbrev=False)
    declare_args(p, specs)
    return p.parse_args(tokens) if strict else p.parse_known_args(tokens)[0]


def check_scene_args(paths, tokens, prog):
    """One set of scene args against several scenes, for a host that loads them
    all: each scene's declarations form a group, so --help says what takes what,
    and a token no scene declares is a typo rather than silence. `paths` is
    {scene name -> path}; a name repeating an earlier scene's arg is skipped."""
    p = argparse.ArgumentParser(prog=prog, allow_abbrev=False)
    seen: set[str] = set()
    for name, path in paths.items():
        try:
            specs = [s for s in declared_args(path) if s.name not in seen]
        except Exception as e:
            # a scene that cannot even import reports itself when it is loaded
            print(f"{name}: {e}", file=sys.stderr)
            continue
        if specs:
            seen.update(s.name for s in specs)
            declare_args(p.add_argument_group(f"{name} arguments"), specs)
    p.parse_args(tokens)


class _ArgDefaults:
    """`self.args` on a controller no loader has parsed args into: the declared
    defaults. A plain __get__ descriptor, so assigning args shadows it."""

    def __get__(self, obj, cls):
        return parse_scene_args(cls.ARGS, [])


class SceneController:
    """Hooks a scene into the render loop. Every method is optional."""

    # Build-time arguments this scene accepts, as Arg declarations. The parsed
    # values arrive as self.args before build().
    ARGS: tuple[Arg, ...] = ()
    args = _ArgDefaults()

    def build(self, base: Scene | None) -> Scene | None:
        """Return the scene to render. `base` is the loaded JSON scene, or None.
        Override to generate passes in code, or to augment the JSON."""
        return base

    def setup(self, renderer) -> None:
        """Called once after the scene compiles (init state)."""

    def on_param(self, renderer, name: str, value) -> None:
        """Called when a host-side (string) param changes, and once per string
        param after setup() so saved values reach the controller. Rebuild
        whatever the string drives (rasterized text, file choices) and push it
        via provide()/set_param."""

    def provide(self, name: str, size: int, frame: int) -> ArrayLike | None:
        """Array-like floats for a declared array this frame, None to fall through
        to the host's source / a default, or ghee.UNCHANGED to declare the
        engine's texture already holds this array (skips the upload; push real
        data again after setup(), which is when engine textures reset)."""
        return None

    def update(self, renderer, frame: int, t: float) -> None:
        """Per-frame hook before the draw (e.g. set live params)."""

    def dirty(self, frame: int, t: float) -> bool:
        """False = this frame would look identical to the last one, so the loop
        may skip it entirely (the display keeps scanning out the last flipped
        buffer). Default True: render every frame, as any scene animating on
        `time` needs. `frame` advances on skipped frames too."""
        return True


def _import_file(path):
    mod_name = "ghee_scene_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load scene module: {path}")
    mod = importlib.util.module_from_spec(spec)
    base_dir = os.path.dirname(os.path.abspath(path))
    sys.path.insert(0, base_dir)  # let the scene import sibling modules
    try:
        spec.loader.exec_module(mod)
    finally:
        if sys.path and sys.path[0] == base_dir:
            sys.path.pop(0)
    record(path)
    for lib in _scene_libs(mod, base_dir):
        record(lib)
    return mod


def _scene_libs(mod, base_dir):
    """The modules a scene imported from its own directory, reachable from its
    namespace. A scene's own libraries are part of what it was built from: they
    hold scene code, and the scene reloads them with itself."""
    root = os.path.abspath(base_dir) + os.sep
    seen = {id(mod)}
    out = []
    todo = [mod]
    while todo:
        for v in list(vars(todo.pop()).values()):
            if not inspect.ismodule(v):
                # not a module: the module the imported name came from, if any
                v = sys.modules.get(getattr(v, "__module__", None))
            if v is None or id(v) in seen:
                continue
            seen.add(id(v))
            f = getattr(v, "__file__", None)
            if f and os.path.abspath(f).startswith(root):
                out.append(os.path.abspath(f))
                todo.append(v)
    return out


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


def _prog(path):
    name = os.path.basename(os.path.normpath(path))
    for suffix in (_SUFFIX, ".py"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return f"ghee render {name} --"


def declared_args(path):
    """The Args a scene package declares, without building its scene."""
    _, py_path = _resolve(path)
    if py_path is None:
        return ()
    controller = _controller_in(_import_file(py_path))
    return () if controller is None else tuple(controller.ARGS)


def load_scene(path, args=None, strict=True):
    """Load a scene package -> (Scene, SceneController | None). Accepts a directory, a
    .scene.json, or a .py.

    `args` are the command-line tokens after `--`: parsed against the
    controller's declared ARGS and set as its `args` before build() runs. A
    scene declaring nothing takes no args. strict=False leaves undeclared tokens
    alone, for a host handing one set of args to every scene it loads."""
    json_path, py_path = _resolve(path)
    base = Scene.load(json_path) if json_path else None
    controller = _controller_in(_import_file(py_path)) if py_path else None
    specs = () if controller is None else tuple(controller.ARGS)
    if args or specs:
        parsed = parse_scene_args(specs, list(args or ()), _prog(path), strict)
        if controller is not None:
            controller.args = parsed
    scene = controller.build(base) if controller is not None else base
    if scene is None:
        raise ValueError(
            f"no scene at {path}: need a {_SUFFIX} or a controller with build()"
        )
    return scene, controller
