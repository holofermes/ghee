"""Render each bundled demo scene: it must compile and produce a frame, and fuzzily
match its committed golden (see assert_golden for the cross-GPU tolerance). First run
records the baseline; regenerate with GHEE_REGEN=1."""

import glob
import importlib.resources
import os

import pytest

from ghee import Scene

from .helpers import assert_golden, ramp

_SUFFIX = ".scene.json"
_DIR = str(importlib.resources.files("ghee") / "examples")
_SCENES = sorted(glob.glob(os.path.join(_DIR, "*" + _SUFFIX)))
_IDS = [os.path.basename(p)[: -len(_SUFFIX)] for p in _SCENES]


@pytest.mark.parametrize("path", _SCENES, ids=_IDS)
def test_demo_scene(renderer, path):
    name = os.path.basename(path)[: -len(_SUFFIX)]
    err = renderer.setup(Scene.load(path))
    assert err == "", f"{name} failed to compile: {err}"
    renderer.set_time(0.5)
    frame = None
    for _ in range(2):  # a second frame lets feedback scenes (trail, echo) evolve
        frame = renderer.render(ramp).copy()
    assert frame.shape[2] == 4
    assert_golden(name, frame)
