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


def test_pass_size_roundtrip():
    d = {
        "version": 1,
        "passes": [
            {"name": "pre", "code": "void main(){}", "size": [32, 18]},
            {"name": "main", "code": "void main(){}"},
        ],
    }
    assert Scene.from_dict(d).to_dict() == d


def test_pass_size_validates():
    Pass(name="ok", code="x", size=[8, 4])  # constructs and normalizes to a tuple
    assert Pass(name="ok", code="x", size=[8, 4]).size == (8, 4)
    for bad in ([0, 4], [4, 0], [-1, 4], [8], [8, 4, 2], "8x4", [8.5, 4], [8, True]):
        with pytest.raises(ValueError):
            Pass(name="bad", code="x", size=bad)


def test_color_param_roundtrip():
    d = {
        "version": 1,
        "passes": [
            {
                "name": "m",
                "code": "void main(){}",
                "params": {"tint": {"type": "color", "value": [1.0, 0.0, 0.5]}},
            }
        ],
    }
    assert Scene.from_dict(d).to_dict() == d


def test_color_param_validates_length_and_type():
    from ghee import Param

    Param("ok", "color", [0.1, 0.2, 0.3])  # constructs fine
    for bad in ([0.1, 0.2], [0.1, 0.2, 0.3, 0.4], "red", 0.5, [0.1, 0.2, "x"]):
        with pytest.raises(ValueError):
            Param("bad", "color", bad)


def test_color_param_preamble_decomposes():
    from ghee import Param

    g = Pass(
        name="m",
        code="void main(){ gl_FragColor = vec4(tint, 1.0); }",
        params={"tint": Param("tint", "color", [1.0, 0.0, 0.0])},
    ).glsl()
    assert "uniform float tint_r;" in g
    assert "uniform float tint_g;" in g
    assert "uniform float tint_b;" in g
    assert "#define tint vec3(tint_r, tint_g, tint_b)" in g
    assert "uniform color" not in g  # a color never leaks a bogus GLSL type
    assert "uniform vec3 tint;" not in g


def test_color_param_glsl_type_raises():
    from ghee import Param

    p = Param("tint", "color", [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        _ = p.glsl_type


def test_vec_param_roundtrip():
    d = {
        "version": 1,
        "passes": [
            {
                "name": "m",
                "code": "void main(){}",
                "params": {
                    "a": {"type": "vec2", "value": [0.1, 0.2]},
                    "b": {"type": "vec3", "value": [0.1, 0.2, 0.3]},
                    "c": {"type": "vec4", "value": [0.1, 0.2, 0.3, 0.4]},
                },
            }
        ],
    }
    assert Scene.from_dict(d).to_dict() == d


def test_vec_param_validates_length_and_type():
    from ghee import Param

    Param("ok2", "vec2", [0.1, 0.2])  # exact arity constructs fine
    Param("ok3", "vec3", [0.1, 0.2, 0.3])
    Param("ok4", "vec4", [0.1, 0.2, 0.3, 0.4])
    bad_by_type = {
        "vec2": ([0.1], [0.1, 0.2, 0.3], "xy", 0.5, [0.1, "x"]),
        "vec3": ([0.1, 0.2], [0.1, 0.2, 0.3, 0.4], "rgb", 1, [0.1, 0.2, "x"]),
        "vec4": ([0.1, 0.2, 0.3], [0.1] * 5, "rgba", 2, [0.1, 0.2, 0.3, "x"]),
    }
    for t, bads in bad_by_type.items():
        for bad in bads:
            with pytest.raises(ValueError):
                Param("bad", t, bad)


def test_vec_param_preamble_decomposes():
    from ghee import Param

    cases = {
        "vec2": ("off", ["_x", "_y"]),
        "vec3": ("dir", ["_x", "_y", "_z"]),
        "vec4": ("q", ["_x", "_y", "_z", "_w"]),
    }
    for t, (name, sfx) in cases.items():
        g = Pass(
            name="m",
            code="void main(){}",
            params={name: Param(name, t, [0.0] * len(sfx))},
        ).glsl()
        for s in sfx:
            assert f"uniform float {name}{s};" in g
        args = ", ".join(f"{name}{s}" for s in sfx)
        assert f"#define {name} {t}({args})" in g
        assert f"uniform {t} {name};" not in g  # the vec type never leaks as a uniform


def test_vec_param_glsl_type_raises():
    from ghee import Param

    for t, n in (("vec2", 2), ("vec3", 3), ("vec4", 4)):
        p = Param("v", t, [0.0] * n)
        with pytest.raises(ValueError):
            _ = p.glsl_type


def test_scene_params_roundtrip():
    d = {
        "version": 1,
        "passes": [
            {
                "name": "m",
                "code": "void main(){}",
                "params": {"k": {"type": "float", "value": 0.5}},
            }
        ],
        "params": {
            "speed": {
                "type": "float",
                "value": 0.06,
                "min": -0.5,
                "max": 0.5,
                "step": 0.005,
            },
            "tint": {"type": "color", "value": [1.0, 0.0, 0.5]},
            "off": {"type": "vec2", "value": [0.1, 0.2]},
        },
    }
    assert Scene.from_dict(d).to_dict() == d


def test_scene_pass_param_name_collision_raises():
    from ghee import Param

    with pytest.raises(ValueError) as e:
        Scene(
            passes=[
                Pass(
                    name="main",
                    code="void main(){}",
                    params={"speed": Param("speed", "float", 0.1)},
                )
            ],
            params={"speed": Param("speed", "float", 0.2)},
        )
    assert "speed" in str(e.value) and "main" in str(e.value)


def test_scene_pass_param_collision_via_from_dict_raises():
    d = {
        "version": 1,
        "passes": [
            {
                "name": "p0",
                "code": "x",
                "params": {"g": {"type": "float", "value": 0.1}},
            }
        ],
        "params": {"g": {"type": "float", "value": 0.2}},
    }
    with pytest.raises(ValueError):
        Scene.from_dict(d)


def test_scene_params_in_every_pass_preamble():
    from ghee import Channel, Param

    sc = Scene(
        passes=[
            Pass(name="a", code="void main(){}"),
            Pass(name="b", code="void main(){}", channels=[Channel("a")]),
        ],
        params={
            "gain": Param("gain", "float", 0.5),
            "tint": Param("tint", "color", [1.0, 0.0, 0.0]),
            "off": Param("off", "vec2", [0.1, 0.2]),
        },
    )
    for p in sc.passes:
        g = p.glsl(sc.params)
        assert "uniform float gain;" in g  # scalar, one uniform
        # a color scene param decomposes to three floats + a vec3 alias, per pass
        assert "uniform float tint_r;" in g
        assert "uniform float tint_g;" in g
        assert "uniform float tint_b;" in g
        assert "#define tint vec3(tint_r, tint_g, tint_b)" in g
        # a vec2 scene param decomposes to two floats + a vec2 alias, per pass
        assert "uniform float off_x;" in g
        assert "uniform float off_y;" in g
        assert "#define off vec2(off_x, off_y)" in g


def test_no_scene_params_emits_nothing_extra():
    from ghee import Param

    sc = Scene(
        passes=[
            Pass(
                name="m",
                code="void main(){}",
                params={"k": Param("k", "float", 0.3)},
            )
        ]
    )
    # an empty scene-params dict threads nothing into the preamble
    for p in sc.passes:
        assert p.glsl(sc.params) == p.glsl()


def test_expand_pattern():
    assert expand_pattern("frame.####.jpg", 7) == "frame.0007.jpg"
    assert expand_pattern("f.%04d.png", 42) == "f.0042.png"
    assert expand_pattern("single.png", 1) == "single.png"  # plain name, as is


def test_driver_registry_unknown():
    from ghee.drivers import available_drivers, get_driver

    assert set(available_drivers()) >= {"st7789", "ssd1306"}
    with pytest.raises(KeyError):
        get_driver("does-not-exist")


def test_adaptive_pacer():
    from ghee.output import AdaptivePacer

    p = AdaptivePacer(factor=2.0, max_fps=60)
    # slow sink: 20ms writes in a 25ms frame -> sleep tops the period up to 40ms
    s = p.sleep_for(0.020, 0.025)
    assert abs(s - 0.015) < 1e-9
    # EMA converges toward a changed write time
    for _ in range(200):
        s = p.sleep_for(0.040, 0.045)
    assert 0.030 < s < 0.036
    # fast sink: sub-ms writes are floored at max_fps, never a busy spin
    q = AdaptivePacer(factor=2.0, max_fps=60)
    s = q.sleep_for(0.0005, 0.002)
    assert abs(s - (1 / 60 - 0.002)) < 1e-6
    # a frame already longer than the period sleeps zero
    r = AdaptivePacer(factor=2.0, max_fps=60)
    assert r.sleep_for(0.020, 0.050) == 0.0


def test_fixed_pacer():
    from ghee.output import FixedPacer

    p = FixedPacer(20)
    assert abs(p.sleep_for(0.001, 0.010) - 0.040) < 1e-9
    assert p.sleep_for(0.001, 0.060) == 0.0


def test_string_param_roundtrip_with_options():
    d = {
        "version": 1,
        "passes": [{"name": "m", "code": "void main(){}"}],
        "params": {
            "text": {"type": "string", "value": "GHEE"},
            "font": {
                "type": "string",
                "value": "Verdana",
                "options": ["Verdana", "Impact"],
            },
        },
    }
    assert Scene.from_dict(d).to_dict() == d


def test_string_param_validates():
    from ghee import Param

    Param("ok", "string", "hello")
    Param("ok", "string", "a", options=["a", "b"])
    for bad in (1.0, ["x"], None, True):
        with pytest.raises(ValueError):
            Param("bad", "string", bad)
    with pytest.raises(ValueError):
        Param("bad", "string", "a", options=[1, 2])
    with pytest.raises(ValueError):
        Param("bad", "float", 0.5, options=["a"])  # options are string-only


def test_string_param_emits_no_uniform():
    from ghee import Param

    g = Pass(
        name="m",
        code="void main(){ gl_FragColor = vec4(1.0); }",
        params={"text": Param("text", "string", "GHEE")},
    ).glsl()
    assert "text" not in g.split("void main")[0]
    s = Scene(
        passes=[Pass(name="m", code="void main(){}")],
        params={"text": Param("text", "string", "GHEE")},
    )
    g2 = s.passes[0].glsl(s.params)
    assert "uniform" not in g2.split("void main")[0].split("uniform float time;")[1]


def test_string_param_glsl_type_raises():
    from ghee import Param

    p = Param("text", "string", "GHEE")
    with pytest.raises(ValueError):
        _ = p.glsl_type
