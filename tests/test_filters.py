"""Filters: shipped post-process passes appended after a scene's own passes."""

import numpy as np

from ghee import Pass, Scene, apply_filters, load_filters

FLAT = "void main(){ gl_FragColor = vec4(vec3(0.5), 1.0); }"


def test_registry_ships_dither():
    assert "dither" in load_filters()


def test_apply_appends_reading_prev_output():
    sc = Scene(passes=[Pass(name="main", code=FLAT)])
    apply_filters(sc, ["dither"])
    assert [p.name for p in sc.passes] == ["main", "dither"]
    d = sc.passes[-1]
    assert d.filter == "dither"  # marked filter-sourced
    assert [c.buffer for c in d.channels] == ["main"]  # iChannel0 = the frame so far
    assert set(d.params) == {"amount", "levels", "color", "scale"}


def test_unknown_filter_skipped():
    sc = Scene(passes=[Pass(name="main", code=FLAT)])
    apply_filters(sc, ["does-not-exist"])
    assert [p.name for p in sc.passes] == ["main"]


def test_dither_flat_gray_is_1bit(renderer):
    sc = Scene(passes=[Pass(name="main", code=FLAT)])
    apply_filters(sc, ["dither"])  # defaults: levels 2, color 0 -> 1-bit B/W
    assert renderer.setup(sc) == ""
    f = renderer.render().copy()
    vals = set(np.unique(f[..., 0]).tolist())
    assert vals <= {0, 255}  # only black or white
    assert 0 in vals and 255 in vals  # a real bayer mix, not flat
