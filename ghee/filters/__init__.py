"""Filters: named post-process passes shipped with ghee (dither, ...). A filter
file (`<name>.filter.json`) is a single pass -- name, code, params -- that reads
iChannel0 (the frame so far) and writes the processed result. Applying a filter
appends it as a pass that reads the previous pass's output, so it becomes the
on-screen result. Built-ins live here; extra dirs can be scanned for more."""

import glob
import json
import os
import sys


def builtin_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_filters(extra_dirs=None) -> dict:
    """name -> filter def (a pass dict). Later dirs win on a name clash.
    $GHEE_FILTERS adds colon-separated directories after the builtin."""
    env = [d for d in os.environ.get("GHEE_FILTERS", "").split(":") if d]
    out = {}
    for d in [builtin_dir(), *env, *(extra_dirs or [])]:
        for p in sorted(glob.glob(os.path.join(d, "*.filter.json"))):
            try:
                with open(p, encoding="utf-8") as f:
                    fd = json.load(f)
                if not isinstance(fd.get("name"), str) or not isinstance(
                    fd.get("code"), str
                ):
                    raise ValueError("needs a string 'name' and 'code'")
                out[fd["name"]] = fd
            except Exception as e:
                print(f"filter {p}: skipped ({e})", file=sys.stderr)
    return out


def apply_filters(scene, names, registry=None):
    """Append each named filter as a pass reading the current last pass, in order,
    so the last filter is what reaches the display. Mutates and returns the scene."""
    if not names or not scene.passes:
        return scene
    from ghee.scene import Pass  # local import: scene.py imports nothing from here

    reg = registry if registry is not None else load_filters()
    for name in names:
        fd = reg.get(name)
        if fd is None:
            print(
                f"filter '{name}': unknown, skipped (known: {', '.join(reg)})",
                file=sys.stderr,
            )
            continue
        prev = scene.passes[-1].name
        p = Pass.from_dict(
            {
                "name": _unique(scene, fd["name"]),
                "code": fd["code"],
                "params": fd.get("params", {}),
                "channels": [{"buffer": prev}],
            }
        )
        p.filter = name  # mark as filter-sourced (the console renders it read-only)
        scene.passes.append(p)
    return scene


def _unique(scene, base):
    taken = {p.name for p in scene.passes}
    name, k = base, 1
    while name in taken:
        name, k = f"{base}{k}", k + 1
    return name
