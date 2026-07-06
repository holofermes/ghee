"""Pure-Python tests: scene model, preamble, output patterns, driver registry.
No GL context needed, so these run anywhere."""

import pytest

import ghee
from ghee import Pass, Scene
from ghee.output import expand_pattern


def test_no_display_profile():
    assert not hasattr(ghee, "DisplayProfile")


def test_preamble_has_resolution_no_orientation():
    g = Pass(name="main", code="void main(){}").glsl()
    assert "uniform vec2 resolution;" in g
    assert "uniform float time;" in g
    assert "VERTICAL" not in g and "HORIZONTAL" not in g


def test_preamble_declares_array_and_accessor():
    p = Pass.from_dict(
        {"name": "m", "code": "x", "arrays": [{"name": "amps", "size": 8}]}
    )
    g = p.glsl()
    assert "uniform sampler2D amps_tex;" in g  # arrays are 1xN data textures
    assert "const int amps_size = 8;" in g
    assert "amps_at" in g
    assert "amps_sample" in g


def test_scene_roundtrip():
    d = {
        "version": 1,
        "passes": [
            {
                "name": "m",
                "code": "void main(){}",
                "arrays": [{"name": "a", "size": 4}],
                "params": {"k": {"type": "float", "value": 0.5}},
            }
        ],
    }
    assert Scene.from_dict(d).to_dict() == d


def test_expand_pattern():
    assert expand_pattern("frame.####.jpg", 7) == "frame.0007.jpg"
    assert expand_pattern("f.%04d.png", 42) == "f.0042.png"
    with pytest.raises(ValueError):
        expand_pattern("no-index.jpg", 1)


def test_driver_registry_unknown():
    from ghee.drivers import available_drivers, get_driver

    assert set(available_drivers()) >= {"st7789", "ssd1306"}
    with pytest.raises(KeyError):
        get_driver("does-not-exist")
