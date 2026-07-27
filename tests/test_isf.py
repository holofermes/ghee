"""The ISF importer: header parsing, input mapping, the two body rewrites the
GLES2 compiler forces, pass/size handling, filter mode, and every refusal.

The ISF sources here are written for the test, not taken from any library. The
one GPU test compiles and renders an import and skips cleanly without a device.
"""

import json

import pytest

from ghee.isf import (
    ISFError,
    eval_size,
    fetch_estimate,
    hoist_globals,
    slug,
    split_header,
    to_filter,
    to_scene,
)

FILTER = """/*{
    "DESCRIPTION": "half tint",
    "CREDIT": "tests",
    "INPUTS": [
        { "NAME": "inputImage", "TYPE": "image" },
        { "NAME": "amount", "TYPE": "float", "DEFAULT": 0.5, "MIN": 0.0, "MAX": 1.0 },
        { "NAME": "invert", "TYPE": "bool", "DEFAULT": true },
        { "NAME": "mode", "TYPE": "long", "VALUES": [0, 1, 2], "DEFAULT": 1 },
        { "NAME": "tint", "TYPE": "color", "DEFAULT": [1.0, 0.2, 0.0, 1.0] },
        { "NAME": "centre", "TYPE": "point2D", "DEFAULT": [0.5, 0.5] }
    ]
}*/

void main() {
    vec4 src = IMG_THIS_PIXEL(inputImage);
    vec4 far = IMG_PIXEL(inputImage, isf_FragNormCoord * RENDERSIZE);
    float d = distance(isf_FragNormCoord, centre) * float(mode) + TIME * 0.0;
    vec3 c = mix(src.rgb, tint.rgb, amount * d);
    gl_FragColor = vec4(invert ? 1.0 - c : c, far.a);
}
"""

TWO_PASS = """/*{
    "INPUTS": [ { "NAME": "gain", "TYPE": "float", "DEFAULT": 2.0 } ],
    "PASSES": [
        { "TARGET": "acc", "WIDTH": "floor($WIDTH/2.0)", "HEIGHT": "max($HEIGHT/4.0, 1.0)", "PERSISTENT": true },
        { }
    ]
}*/

void main() {
    if (PASSINDEX == 0) {
        vec4 prev = IMG_THIS_PIXEL(acc);
        gl_FragColor = vec4(prev.rgb * 0.9 + isf_FragNormCoord.x * 0.1, 1.0);
        return;
    }
    gl_FragColor = IMG_NORM_PIXEL(acc, isf_FragNormCoord) * gain;
}
"""

AUDIO = """/*{
    "INPUTS": [ { "NAME": "fft", "TYPE": "audioFFT", "MAX": 32 } ]
}*/

void main() {
    vec4 v = IMG_NORM_PIXEL(fft, vec2(isf_FragNormCoord.x, 0.5));
    gl_FragColor = vec4(vec3((v.r + v.g + v.b) / 3.0), 1.0);
}
"""


def isf(meta, body="void main(){ gl_FragColor = vec4(1.0); }"):
    return "/*" + json.dumps(meta) + "*/\n" + body


def test_header_split():
    meta, body = split_header(FILTER)
    assert meta["DESCRIPTION"] == "half tint"
    assert body.lstrip().startswith("void main()")


def test_header_missing_is_refused():
    with pytest.raises(ISFError, match="not ISF"):
        split_header("void main(){}")


def test_header_bad_json_is_refused():
    with pytest.raises(ISFError, match="does not parse"):
        split_header("/*{ nope }*/\nvoid main(){}")


# --- inputs ----------------------------------------------------------------


def test_input_types_map_to_params():
    scene, _ = to_scene(FILTER, "half tint.fs", (64, 48))
    p = scene.params
    assert (p["amount"].type, p["amount"].value) == ("float", 0.5)
    assert (p["amount"].min, p["amount"].max) == (0.0, 1.0)
    assert (p["invert"].type, p["invert"].value) == ("bool", True)
    # a long is an int over the range its VALUES span (ghee has no labelled int)
    assert (p["mode"].type, p["mode"].value) == ("int", 1)
    assert (p["mode"].min, p["mode"].max, p["mode"].step) == (0, 2, 1)
    # an ISF color is RGBA, so vec4 rather than ghee's three-component color
    assert (p["tint"].type, p["tint"].value) == ("vec4", [1.0, 0.2, 0.0, 1.0])
    assert (p["centre"].type, p["centre"].value) == ("vec2", [0.5, 0.5])
    assert "inputImage" not in p  # an image is a channel


def test_event_is_a_bool():
    scene, _ = to_scene(
        isf({"INPUTS": [{"NAME": "go", "TYPE": "event"}]}), "e.fs", (8, 8)
    )
    assert (scene.params["go"].type, scene.params["go"].value) == ("bool", False)


def test_defaults_when_the_input_names_none():
    scene, _ = to_scene(
        isf(
            {
                "INPUTS": [
                    {"NAME": "f", "TYPE": "float", "MIN": 3.0},
                    {"NAME": "c", "TYPE": "color"},
                    {"NAME": "p", "TYPE": "point2D"},
                ]
            }
        ),
        "d.fs",
        (8, 8),
    )
    assert scene.params["f"].value == 3.0  # falls back to MIN, then 0.0
    assert scene.params["c"].value == [0.0, 0.0, 0.0, 1.0]
    assert scene.params["p"].value == [0.0, 0.0]


def test_unknown_input_type_is_refused():
    with pytest.raises(ISFError, match="unsupported TYPE 'audioFFTBins'"):
        to_scene(isf({"INPUTS": [{"NAME": "x", "TYPE": "audioFFTBins"}]}), "u.fs")


def test_audio_input_becomes_an_array():
    scene, _ = to_scene(AUDIO, "audio.fs", (64, 48))
    (a,) = scene.passes[0].arrays
    assert (a.name, a.size) == ("fft", 32)
    # ghee binds the array as sampler2D <name>_tex in GL_LUMINANCE, so the ISF
    # (r+g+b)/3 reads back the array value
    assert "#define fft fft_tex" in scene.passes[0].code
    assert scene.passes[0].channels == []


def test_audio_input_defaults_to_256_bins():
    scene, _ = to_scene(
        isf({"INPUTS": [{"NAME": "w", "TYPE": "audio"}]}), "a.fs", (8, 8)
    )
    assert scene.passes[0].arrays[0].size == 256


# --- passes, channels, sizes ------------------------------------------------


def test_image_input_gets_a_generated_source_pass():
    scene, warn = to_scene(FILTER, "half tint.fs", (64, 48))
    assert [p.name for p in scene.passes] == ["isf_source", "out"]
    assert [c.buffer for c in scene.passes[1].channels] == ["isf_source"]
    assert "#define inputImage iChannel0" in scene.passes[1].code
    assert any("no producer" in w for w in warn)


def test_multipass_shares_one_body_with_its_own_passindex():
    scene, _ = to_scene(TWO_PASS, "two.fs", (320, 240))
    assert [p.name for p in scene.passes] == ["acc", "out"]
    assert "const int PASSINDEX = 0;" in scene.passes[0].code
    assert "const int PASSINDEX = 1;" in scene.passes[1].code
    # ISF puts one body in every pass; only the constant differs
    body = "void main() {"
    assert scene.passes[0].code.split(body)[1] == scene.passes[1].code.split(body)[1]


def test_persistent_target_is_a_self_channel():
    scene, _ = to_scene(TWO_PASS, "two.fs", (320, 240))
    assert [c.buffer for c in scene.passes[0].channels] == ["acc"]
    assert [c.buffer for c in scene.passes[1].channels] == ["acc"]
    assert "#define acc iChannel0" in scene.passes[0].code


def test_pass_sizes_come_from_the_equations():
    scene, _ = to_scene(TWO_PASS, "two.fs", (320, 240))
    assert scene.passes[0].size == (160, 60)
    assert scene.passes[1].size is None  # the output pass takes the render size
    assert "const vec2 acc_isf_size = vec2(160.0, 60.0);" in scene.passes[0].code


def test_a_sized_last_pass_is_dropped_with_a_warning():
    src = isf(
        {"PASSES": [{"TARGET": "a"}, {"WIDTH": "$WIDTH/2.0", "HEIGHT": "$HEIGHT"}]}
    )
    scene, warn = to_scene(src, "s.fs", (64, 48))
    assert scene.passes[-1].size is None
    assert any("output pass renders at the render size" in w for w in warn)


def test_reserved_name_is_renamed_in_the_body():
    src = isf(
        {"PASSES": [{"TARGET": "time"}, {}]},
        "void main(){ gl_FragColor = IMG_THIS_PIXEL(time) * TIME; }",
    )
    scene, warn = to_scene(src, "r.fs", (8, 8))
    assert [p.name for p in scene.passes] == ["isf_time", "out"]
    assert "IMG_THIS_PIXEL(isf_time)" in scene.passes[0].code
    assert any("renamed to 'isf_time'" in w for w in warn)


def test_img_size_becomes_a_constant():
    src = isf(
        {"INPUTS": [{"NAME": "inputImage", "TYPE": "image"}]},
        "void main(){ gl_FragColor = vec4(IMG_SIZE(inputImage), 0.0, 1.0); }",
    )
    scene, _ = to_scene(src, "z.fs", (64, 48))
    code = scene.passes[-1].code
    assert "inputImage_isf_size" in code and "IMG_SIZE(" not in code
    assert "const vec2 inputImage_isf_size = vec2(64.0, 48.0);" in code


def test_img_pixel_is_handed_the_sampler_size():
    scene, _ = to_scene(FILTER, "f.fs", (64, 48))
    assert "IMG_PIXEL(inputImage_isf_size, inputImage," in scene.passes[-1].code


def test_commented_out_img_call_is_left_alone():
    src = isf(
        {},
        "void main(){\n// gl_FragColor = IMG_PIXEL(nosuchimage, vec2(0.0));\n"
        "gl_FragColor = vec4(1.0); }",
    )
    scene, _ = to_scene(src, "c.fs", (8, 8))
    assert "IMG_PIXEL(nosuchimage, vec2(0.0))" in scene.passes[-1].code


def test_img_pixel_on_an_unknown_sampler_is_refused():
    src = isf({}, "void main(){ gl_FragColor = IMG_PIXEL(mystery, vec2(0.0)); }")
    with pytest.raises(ISFError, match="mystery"):
        to_scene(src, "m.fs", (8, 8))


# --- refusals ---------------------------------------------------------------


def test_float_buffer_is_refused():
    src = isf({"PASSES": [{"TARGET": "acc", "FLOAT": True}, {}]})
    with pytest.raises(ISFError, match="FLOAT"):
        to_scene(src, "acc.fs")


def test_float_refusal_names_the_file_and_the_buffer():
    src = isf({"PASSES": [{"TARGET": "acc", "FLOAT": True}, {}]})
    with pytest.raises(ISFError) as e:
        to_scene(src, "acc.fs")
    assert "acc.fs" in str(e.value) and "acc" in str(e.value)
    assert "8-bit" in str(e.value)


def test_sibling_vertex_shader_is_refused():
    with pytest.raises(ISFError, match="vertex shader"):
        to_scene(FILTER, "f.fs", (8, 8), vertex_shader=True)


def test_custom_varying_is_refused():
    src = isf({}, "varying vec2 texOffsets;\nvoid main(){ gl_FragColor = vec4(1.0); }")
    with pytest.raises(ISFError, match="texOffsets"):
        to_scene(src, "v.fs")


def test_imported_image_is_refused():
    src = isf({"IMPORTED": {"noiseTex": {"PATH": "noise.png"}}})
    with pytest.raises(ISFError, match="noiseTex"):
        to_scene(src, "i.fs")


def test_no_main_to_hoist_into_is_refused():
    src = isf({}, "float ratio = RENDERSIZE.x;\n")
    with pytest.raises(ISFError, match="no void main"):
        to_scene(src, "n.fs")


# --- the two body rewrites --------------------------------------------------


def test_hoist_moves_a_uniform_initializer_into_main():
    body = "float ratio = RENDERSIZE.x / RENDERSIZE.y;\nvoid main(){ f(ratio); }"
    out, names = hoist_globals(body)
    assert names == ["ratio"]
    assert out.startswith("float ratio;")
    assert "void main(){\n    ratio = RENDERSIZE.x / RENDERSIZE.y;" in out


def test_hoist_handles_several_declarators():
    body = "vec2 a = vec2(1.0), b, c = a * 2.0;\nvoid main(){}"
    out, names = hoist_globals(body)
    assert names == ["a", "c"]
    assert out.startswith("vec2 a, b, c;")
    assert out.index("a = vec2(1.0);") < out.index("c = a * 2.0;")


def test_hoist_leaves_const_and_declarations_alone():
    body = (
        "const float PI = 3.14;\nuniform float u;\nvec4 f(vec2 p);\n"
        "struct S { float a; };\nvoid main(){}"
    )
    out, names = hoist_globals(body)
    assert names == []
    assert out == body


def test_hoist_keeps_a_leading_comment():
    body = "// keep me\nfloat r = RENDERSIZE.x;\nvoid main(){}"
    out, _ = hoist_globals(body)
    assert out.startswith("// keep me\nfloat r;")


def test_hoist_ignores_locals():
    body = "void main(){ float x = RENDERSIZE.x; }"
    out, names = hoist_globals(body)
    assert (out, names) == (body, [])


# --- size equations ---------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "want"),
    [
        (512, 512),
        (1.0, 1),
        ("$WIDTH", 320),
        ("$HEIGHT", 240),
        ("floor($WIDTH/4.0)", 80),
        ("$WIDTH / 100.0", 3),
        ("max(floor($WIDTH*0.02),1.0)", 6),
        ("max($HEIGHT / 243.0, 1.0)", 1),
        ("floor($WIDTH*min((0.3),1.0))", 96),
        ("max(8.0,floor($HEIGHT*$quality))", 24),
        ("$WIDTH - $HEIGHT", 80),
        ("-$WIDTH", 1),  # a target is at least one pixel
    ],
)
def test_size_equations(expr, want):
    values = {"WIDTH": 320.0, "HEIGHT": 240.0, "quality": 0.1}
    assert eval_size(expr, values) == want


def test_size_equation_unknown_name_is_refused():
    with pytest.raises(ISFError, match="\\$nope"):
        eval_size("$nope * 2.0", {"WIDTH": 320.0, "HEIGHT": 240.0})


def test_size_equation_garbage_is_refused():
    with pytest.raises(ISFError):
        eval_size("$WIDTH >> 2", {"WIDTH": 320.0, "HEIGHT": 240.0})


def test_size_equation_reads_a_numeric_input():
    src = isf(
        {
            "INPUTS": [{"NAME": "trailCount", "TYPE": "float", "DEFAULT": 8}],
            "PASSES": [{"TARGET": "a", "WIDTH": "$trailCount", "HEIGHT": "1"}, {}],
        }
    )
    scene, _ = to_scene(src, "t.fs", (64, 48))
    assert scene.passes[0].size == (8, 1)


# --- filter mode ------------------------------------------------------------


def test_filter_mode_writes_a_filter_definition():
    fd, _ = to_filter(FILTER, "half tint.fs", (64, 48))
    assert fd["name"] == "half_tint"
    assert fd["description"] == "half tint tests"
    assert "#define inputImage iChannel0" in fd["code"]
    assert set(fd["params"]) == {"amount", "invert", "mode", "tint", "centre"}
    assert fd["params"]["invert"] == {"type": "bool", "value": True}


def test_filter_mode_loads_through_the_filter_registry(tmp_path, monkeypatch):
    from ghee import Pass, Scene, apply_filters, load_filters

    fd, _ = to_filter(FILTER, "half tint.fs", (64, 48))
    (tmp_path / "half_tint.filter.json").write_text(json.dumps(fd))
    monkeypatch.setenv("GHEE_FILTERS", str(tmp_path))
    sc = Scene(
        passes=[Pass(name="main", code="void main(){ gl_FragColor=vec4(1.0); }")]
    )
    apply_filters(sc, ["half_tint"], registry=load_filters())
    assert [p.name for p in sc.passes] == ["main", "half_tint"]
    assert [c.buffer for c in sc.passes[-1].channels] == ["main"]


@pytest.mark.parametrize(
    ("meta", "match"),
    [
        ({"PASSES": [{"TARGET": "a"}, {}]}, "2 passes"),
        (
            {
                "INPUTS": [
                    {"NAME": "a", "TYPE": "image"},
                    {"NAME": "b", "TYPE": "image"},
                ]
            },
            "2 image inputs",
        ),
        ({}, "no image input"),
        (
            {
                "INPUTS": [
                    {"NAME": "a", "TYPE": "image"},
                    {"NAME": "f", "TYPE": "audio"},
                ]
            },
            "reads audio",
        ),
    ],
)
def test_filter_mode_refusals(meta, match):
    with pytest.raises(ISFError, match=match):
        to_filter(isf(meta), "x.fs", (8, 8))


def test_filter_mode_refuses_feedback():
    src = isf(
        {
            "INPUTS": [{"NAME": "inputImage", "TYPE": "image"}],
            "PASSES": [{"TARGET": "last", "PERSISTENT": True}],
        },
        "void main(){ gl_FragColor = IMG_THIS_PIXEL(last); }",
    )
    with pytest.raises(ISFError, match="reads its own buffer"):
        to_filter(src, "fb.fs", (8, 8))


# --- warnings ---------------------------------------------------------------


def test_fetch_estimate_multiplies_by_literal_loop_bounds():
    assert fetch_estimate("void main(){ vec4 a = IMG_THIS_PIXEL(x); }") == 1
    body = "void main(){ for (int i = 0; i < 16; i++) { c += IMG_THIS_PIXEL(x); } }"
    assert fetch_estimate(body) == 16
    nested = (
        "void main(){ for (int i = 0; i < 4; i++) { "
        "for (int j = 0; j < 8; j++) { c += IMG_THIS_PIXEL(x); } } "
        "c += IMG_THIS_PIXEL(x); }"
    )
    assert fetch_estimate(nested) == 33


def test_heavy_shader_warns_but_still_imports():
    src = isf(
        {"INPUTS": [{"NAME": "inputImage", "TYPE": "image"}]},
        "void main(){ vec4 c = vec4(0.0);\n"
        "for (int i = 0; i < 300; i++) { c += IMG_THIS_PIXEL(inputImage); }\n"
        "gl_FragColor = c; }",
    )
    scene, warn = to_scene(src, "heavy.fs", (8, 8))
    assert scene.passes
    assert any("texture fetches per fragment" in w for w in warn)


def test_slug():
    assert slug("Radial Spectrogram.fs") == "radial_spectrogram"
    assert slug("/a/b/RGB Trails 3.0.fs") == "rgb_trails_3_0"


# --- the bool param type ----------------------------------------------------


def test_bool_param_declares_a_bool_uniform():
    from ghee import Param, Pass

    p = Pass(
        name="main", code="void main(){}", params={"on": Param("on", "bool", True)}
    )
    assert "uniform bool on;" in p.glsl()


def test_bool_param_round_trips_through_json():
    from ghee import Scene

    d = {
        "version": 1,
        "passes": [
            {
                "name": "main",
                "code": "void main(){}",
                "params": {"on": {"type": "bool", "value": False}},
            }
        ],
    }
    sc = Scene.from_dict(d)
    assert sc.passes[0].params["on"].value is False
    assert sc.to_dict() == d


def test_bool_param_rejects_a_number():
    from ghee import Param

    with pytest.raises(ValueError, match="needs true or false"):
        Param("on", "bool", 1)


def test_bool_reaches_the_c_abi_as_an_int():
    from ghee.cparam import CParameterType, create_c_parameter

    cp = create_c_parameter({"type": "bool", "value": True})
    assert cp.type == CParameterType.INT
    assert cp.value.int_param.contents.value == 1
    assert (
        create_c_parameter(
            {"type": "bool", "value": False}
        ).value.int_param.contents.value
        == 0
    )


# --- the CLI ----------------------------------------------------------------


def test_cli_import_writes_a_scene(tmp_path, monkeypatch, capsys):
    from ghee.cli import main

    src = tmp_path / "Two Pass.fs"
    src.write_text(TWO_PASS)
    monkeypatch.chdir(tmp_path)
    main(["isf", "import", str(src), "--size", "64x48"])
    out = json.loads((tmp_path / "two_pass.scene.json").read_text())
    assert [p["name"] for p in out["passes"]] == ["acc", "out"]
    assert out["passes"][0]["size"] == [32, 12]
    assert "wrote scene" in capsys.readouterr().out


def test_cli_import_from_a_url(tmp_path, monkeypatch):
    import functools
    import http.server
    import threading

    from ghee.cli import main

    (tmp_path / "Two Pass.fs").write_text(TWO_PASS)
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(tmp_path)
    )
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.chdir(tmp_path)
    try:
        main(["isf", "import", f"http://127.0.0.1:{srv.server_port}/Two%20Pass.fs"])
    finally:
        srv.shutdown()
    # the percent-encoding is not part of the name the output is called after
    assert (tmp_path / "two_pass.scene.json").exists()


def test_cli_import_as_filter(tmp_path, monkeypatch):
    from ghee.cli import main

    src = tmp_path / "Tint.fs"
    src.write_text(FILTER)
    monkeypatch.chdir(tmp_path)
    main(["isf", "import", str(src), "--as-filter", "-o", "t.filter.json"])
    fd = json.loads((tmp_path / "t.filter.json").read_text())
    assert fd["name"] == "tint" and "code" in fd


def test_cli_import_refusal_exits(tmp_path, monkeypatch, capsys):
    from ghee.cli import main

    src = tmp_path / "Acc.fs"
    src.write_text(isf({"PASSES": [{"TARGET": "a", "FLOAT": True}, {}]}))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        main(["isf", "import", str(src)])
    assert "FLOAT" in str(e.value) and "Acc.fs" in str(e.value)


def test_cli_import_refuses_a_sibling_vertex_shader(tmp_path, monkeypatch):
    from ghee.cli import main

    (tmp_path / "Vs.fs").write_text(FILTER)
    (tmp_path / "Vs.vs").write_text("void main(){}")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="vertex shader"):
        main(["isf", "import", str(tmp_path / "Vs.fs")])


# --- on a GPU ---------------------------------------------------------------


def test_imported_scenes_compile_and_render(renderer):
    import numpy as np

    for src, name in ((FILTER, "f.fs"), (TWO_PASS, "t.fs"), (AUDIO, "a.fs")):
        scene, _ = to_scene(src, name, (renderer.width, renderer.height))
        assert renderer.setup(scene) == "", name
        frame = renderer.render(lambda n, s: np.linspace(0, 1, s, dtype=np.float32))
        assert frame.shape == (renderer.height, renderer.width, renderer.components)
        assert frame[..., :3].std() > 0.5, f"{name} rendered flat"
