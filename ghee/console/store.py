"""The scenes a console shows, edits and saves.

A directory of .scene.json and .py scenes, plus the modified/ copies a save
writes beside them. The original file is never written: a JSON scene saves a
modified copy that shadows it, a procedural scene saves a .values.json sidecar
(param values + filter list) applied over the code's defaults at load.

A host whose scenes live somewhere else (in memory, in a database, in one
controller it built itself) implements names/resolve/save and hands that to its
ConsoleHost instead.
"""

import glob
import json
import os
import sys

from ghee import Param, apply_filters, load_scene
from ghee.console.core import NotFound


def _apply_values(scene, name, vals):
    """Overlay a saved values sidecar's param values onto the resolved scene
    (filter passes included: their params are tweakable). The code's current
    shape wins: unknown names are reported and skipped, saved values only
    fill what still exists."""
    targets = [(scene.params, "scene", vals.get("scene") or {})]
    byname = {ps.name: ps for ps in scene.passes}
    for pn, params in (vals.get("passes") or {}).items():
        if pn not in byname:
            print(f"values {name}: no pass '{pn}'", file=sys.stderr)
            continue
        targets.append((byname[pn].params, pn, params))
    for existing, label, saved in targets:
        for n, v in saved.items():
            p = existing.get(n)
            if p is None:
                print(f"values {name}: no param '{label}.{n}'", file=sys.stderr)
                continue
            if p.type in ("float", "int") and (
                isinstance(v, bool) or not isinstance(v, (int, float))
            ):
                print(f"values {name}: bad '{label}.{n}': {v!r}", file=sys.stderr)
                continue
            if p.type == "bool" and not isinstance(v, bool):
                print(f"values {name}: bad '{label}.{n}': {v!r}", file=sys.stderr)
                continue
            try:
                existing[n] = Param(p.name, p.type, v, p.min, p.max, p.step, p.options)
            except (ValueError, TypeError) as e:
                print(f"values {name}: bad '{label}.{n}': {e}", file=sys.stderr)


class SceneStore:
    def __init__(self, scenes_dir, read_only=False):
        self.dir = os.path.abspath(scenes_dir)
        self.read_only = read_only
        self.scenes = self._scan()

    def _scan(self):
        out = {}
        for p in sorted(glob.glob(os.path.join(self.dir, "*.scene.json"))):
            out[os.path.basename(p)[: -len(".scene.json")]] = p
        for p in sorted(glob.glob(os.path.join(self.dir, "*.py"))):
            name = os.path.basename(p)[:-3]
            out.setdefault(
                name, p
            )  # a .py beside its .scene.json: load_scene finds both
        return out

    @property
    def names(self):
        return list(self.scenes)

    def __contains__(self, name):
        return name in self.scenes

    def _modified_path(self, name):
        return os.path.join(self.dir, "modified", f"{name}.scene.json")

    def _values_path(self, name):
        return os.path.join(self.dir, "modified", f"{name}.values.json")

    def _read_values(self, name):
        """The saved values sidecar for a procedural scene, or None."""
        path = self._values_path(name)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"values {name}: unreadable ({e})", file=sys.stderr)
            return None

    def path(self, name):
        """A saved modified copy shadows the original; the original is never written."""
        if name not in self.scenes:
            raise NotFound("no scene")
        p = self.scenes[name]
        if p.endswith(".scene.json") and os.path.isfile(self._modified_path(name)):
            return self._modified_path(name)
        return p

    def resolve(self, name):
        """Scene ready to render/edit: shader files inlined, declared filters
        expanded, saved values overlaid. Returns (scene, controller)."""
        scene, controller = load_scene(self.path(name))
        vals = (
            None
            if self.scenes[name].endswith(".scene.json")
            else self._read_values(name)
        )
        if vals is not None and isinstance(vals.get("filters"), list):
            scene.filters = [f for f in vals["filters"] if isinstance(f, str)]
        apply_filters(scene, scene.filters)
        if vals is not None:
            _apply_values(scene, name, vals)
        return scene, controller

    def save(self, name, scene):
        if name not in self.scenes:
            raise NotFound("no scene")
        # the working copy carries applied filters as expanded passes; recover
        # the declared list from them so a save never drops a filter the
        # client forgot to record
        applied = [p.filter for p in scene.passes if p.filter]
        if applied:
            scene.filters = applied
        if not self.scenes[name].endswith(".scene.json"):
            # a procedural scene's shape lives in code; save its tweakable
            # surface (param values + filter list) as a sidecar the loader
            # applies over the controller's defaults. Filter passes' params
            # are included: they are the filter's tweakable knobs.
            vals = {
                "filters": scene.filters,
                "scene": {n: p.value for n, p in scene.params.items()},
                "passes": {
                    ps.name: {n: p.value for n, p in ps.params.items()}
                    for ps in scene.passes
                },
            }
            return self._write(self._values_path(name), vals)
        # store unexpanded: filter passes come back at load time from the filters list
        scene.passes = [p for p in scene.passes if not p.filter]
        self._write(self._modified_path(name), scene.to_dict())

    @staticmethod
    def _write(path, obj):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
