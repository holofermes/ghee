"""Procedural scene loading: build()-only, JSON-only, and JSON + controller. The
loader tests are pure Python; the render test needs a GL device (skips otherwise)."""

import importlib.resources
import importlib.util
import os

import numpy as np
import pytest

from ghee import SceneController, load_scene

TEXT_PY = str(importlib.resources.files("ghee") / "examples" / "text.py")

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
    assert c.dirty(0, 0.0) is True  # default: render every frame


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


def test_spectrum_gain_in_shader_and_python(renderer):
    path = str(importlib.resources.files("ghee") / "examples" / "spectrum.py")
    if not os.path.isfile(path):
        pytest.skip("spectrum example not installed")
    scene, ctrl = load_scene(path)
    assert renderer.setup(scene) == ""
    ctrl.setup(renderer)

    def frame_at(g):
        scene.passes[0].params["gain"].value = g  # python side: provide() reads this
        renderer.set_param("main", "gain", "float", g)  # shader side: the uniform
        return renderer.render(lambda n, s: ctrl.provide(n, s, 5)).copy()

    lo, hi = frame_at(0.05), frame_at(0.5)
    assert not np.array_equal(lo, hi)  # the param changes the render
    assert hi.mean() > lo.mean()  # higher gain = taller + brighter bars


def test_render_example_text(renderer):
    scene, ctrl = load_scene(TEXT_PY)
    assert renderer.setup(scene) == ""
    renderer.set_time(0.0)
    frame = renderer.render(lambda name, size: ctrl.provide(name, size, 0))
    lum = frame[:, :, 1].astype(int)  # the tint is green
    bright = (lum > 120).mean()
    assert lum.max() > 150  # glyphs lit
    assert 0.005 < bright < 0.4  # text-like coverage, not a flood or a blank


def _unsized_font(monkeypatch):
    """Pillow < 10.1: load_default() takes no size."""
    from PIL import ImageFont

    plain = ImageFont.load_default

    def load_default(size=None):
        if size is not None:
            raise TypeError("load_default() got an unexpected keyword argument 'size'")
        return plain()

    monkeypatch.setattr(ImageFont, "load_default", load_default)


def _text_module():
    """The example's own module: load_scene never registers it in sys.modules."""
    spec = importlib.util.spec_from_file_location("ghee_text_example", TEXT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_text_bakes_at_the_font_size():
    from PIL import Image, ImageDraw, ImageFont

    scene, _ = load_scene(TEXT_PY)
    try:
        font = ImageFont.load_default(size=_text_module().FONT_SIZE)
    except TypeError:
        pytest.skip("Pillow < 10.1: no sized load_default()")
    ink = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), "GHEE", font=font)
    w, h = scene.passes[0].arrays[0].size
    # the mask is the sized font's own ink box, padded equally on both axes
    assert w - (ink[2] - ink[0]) == h - (ink[3] - ink[1]) > 0


def test_text_falls_back_to_the_bitmap_font(renderer, monkeypatch):
    _unsized_font(monkeypatch)
    scene, ctrl = load_scene(TEXT_PY)
    w, h = scene.passes[0].arrays[0].size
    assert (w % 4, h % 4) == (0, 0)  # the old bake, zoomed 4x
    assert renderer.setup(scene) == ""
    ctrl.setup(renderer)
    renderer.set_time(0.0)
    lum = renderer.render(lambda name, size: ctrl.provide(name, size, 0))[:, :, 1]
    assert lum.max() > 150


def test_text_mask_uploads_once():
    import ghee

    scene, ctrl = load_scene(TEXT_PY)
    size = scene.passes[0].arrays[0].size
    ctrl.setup(None)
    assert len(ctrl.provide("text", size, 0)) == size[0] * size[1]
    assert ctrl.provide("text", size, 1) is ghee.UNCHANGED
    assert ctrl.provide("text", size, 2) is ghee.UNCHANGED
    ctrl.setup(None)  # a fresh setup empties the engine's textures
    assert ctrl.provide("text", size, 3) is not ghee.UNCHANGED
    assert ctrl.provide("other", size, 4) is None


def test_text_shader_fits_the_mask_aspect():
    scene, _ = load_scene(TEXT_PY)
    w, h = scene.passes[0].arrays[0].size
    code = scene.passes[0].code
    assert f"const float mask_aspect = {w / h:.6f};" in code
    assert "resolution.x / resolution.y" in code  # fit against the output, no stretch


def test_text_letterboxes_a_tall_output():
    from ghee.engine import Renderer

    scene, ctrl = load_scene(TEXT_PY)
    w, h = scene.passes[0].arrays[0].size
    try:
        r = Renderer(64, 256, components=4)
    except Exception as e:  # one GPU node, one renderer at a time
        pytest.skip(f"no second renderer: {e}")
    try:
        assert r.setup(scene) == ""
        ctrl.setup(r)
        r.set_time(0.0)
        frame = r.render(lambda name, size: ctrl.provide(name, size, 0))
        rows = np.flatnonzero((frame[:, :, 1] > 120).any(axis=1))
    finally:
        r.close()
    band = 256.0 * (64.0 / 256.0) / (w / h)  # the mask at the output's own scale
    assert 0.4 * band <= rows[-1] - rows[0] + 1 <= band
    assert abs((rows[0] + rows[-1]) / 2 - 128) <= 4  # centered, bars top and bottom
