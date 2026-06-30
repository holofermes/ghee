"""Procedural scene loading: build()-only, JSON-only, and JSON + controller. The
loader tests are pure Python; the render test needs a GL device (skips otherwise)."""

import importlib.resources
import os

import numpy as np

from ghee import SceneController, load_scene

CONTROLLER_PY = """
from ghee import ArrayInput, Pass, Scene, SceneController

class Demo(SceneController):
    def build(self, base):
        return Scene(passes=[Pass(name="main", code="void main(){}",
                                  arrays=[ArrayInput("vals", 4)])])

    def provide(self, name, size, frame):
        return [float(frame)] * size
"""

PROVIDE_ONLY_PY = """
from ghee import SceneController

class Feed(SceneController):
    def provide(self, name, size, frame):
        return [1.0] * size
"""


def test_base_controller_defaults():
    c = SceneController()
    assert c.build("x") == "x"  # default returns base untouched
    assert c.provide("a", 4, 0) is None
    assert c.setup(None) is None
    assert c.update(None, 0, 0.0) is None


def test_build_only(tmp_path):
    (tmp_path / "demo.py").write_text(CONTROLLER_PY)
    scene, ctrl = load_scene(str(tmp_path / "demo.py"))
    assert ctrl is not None
    assert [p.name for p in scene.passes] == ["main"]
    assert scene.passes[0].arrays[0].name == "vals"
    assert ctrl.provide("vals", 4, 7) == [7.0, 7.0, 7.0, 7.0]


def test_json_standalone():
    bars = str(importlib.resources.files("ghee") / "examples" / "bars.scene.json")
    scene, ctrl = load_scene(bars)
    assert ctrl is None
    assert len(scene.passes) >= 1


def test_json_plus_controller(tmp_path):
    d = tmp_path / "wob"  # a scene dir holds <name>.scene.json + <name>.py
    d.mkdir()
    (d / "wob.scene.json").write_text(
        '{"version":1,"passes":[{"name":"main","code":"void main(){}",'
        '"arrays":[{"name":"level","size":2}]}]}'
    )
    (d / "wob.py").write_text(PROVIDE_ONLY_PY)
    scene, ctrl = load_scene(str(d))  # directory form
    assert ctrl is not None  # controller feeds data
    assert scene.passes[0].arrays[0].name == "level"  # shape from the JSON
    assert ctrl.provide("level", 2, 0) == [1.0, 1.0]


def test_render_example_spectrum(renderer):
    path = str(importlib.resources.files("ghee") / "examples" / "spectrum.py")
    if not os.path.isfile(path):
        import pytest

        pytest.skip("spectrum example not installed")
    scene, ctrl = load_scene(path)
    assert renderer.setup(scene) == ""
    ctrl.setup(renderer)
    renderer.set_time(0.5)
    frame = None
    for i in range(2):
        ctrl.update(renderer, i, i / 60.0)
        frame = renderer.render(lambda n, s, i=i: ctrl.provide(n, s, i)).copy()
    assert frame.std() > 2  # the custom array drove a non-blank spectrum
    assert not np.array_equal(frame[..., 0], frame[..., 1])
