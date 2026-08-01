"""The scene's common block: one chunk of GLSL every scene pass compiles with.

Carried inline or in a file next to the scene, prepended after the preamble,
never handed to a filter's pass, and truthful about the line a compile error
sits on. The GL parts skip when no device is available.
"""

import json
import os
import re

from ghee import Pass, Scene, apply_filters, load_scene
from ghee.console import web_dir
from ghee.engine import COMMON_PASS

COMMON = "float sat(float x){ return clamp(x, 0.0, 1.0); }"
CODE = "void main(){ gl_FragColor = vec4(sat(v_texCoord.x), 0.0, 0.0, 1.0); }"
FLAT = "void main(){ gl_FragColor = vec4(vec3(0.5), 1.0); }"


def _scene(common=None, passes=None):
    d = {"version": 1, "passes": passes or [{"name": "main", "code": CODE}]}
    if common is not None:
        d["common"] = common
    return d


# the document


def test_inline_common_roundtrips():
    d = _scene({"code": COMMON})
    assert Scene.from_dict(d).to_dict() == d


def test_a_common_file_is_inlined_like_a_pass_shader(tmp_path):
    (tmp_path / "shared.glsl").write_text(COMMON, encoding="utf-8")
    sc = Scene.from_dict(_scene({"shader": "shared.glsl"}), base_dir=str(tmp_path))
    assert sc.common == COMMON
    # what the console posts back and what a save writes: code, inline
    assert sc.to_dict()["common"] == {"code": COMMON}


def test_a_scene_file_loads_its_common_file(tmp_path):
    (tmp_path / "shared.glsl").write_text(COMMON, encoding="utf-8")
    p = tmp_path / "x.scene.json"
    p.write_text(json.dumps(_scene({"shader": "shared.glsl"})), encoding="utf-8")
    scene, _ = load_scene(str(p))
    assert scene.common == COMMON


def test_an_empty_common_is_a_block_that_exists():
    # the console's "+ common" creates one before anything is typed into it
    sc = Scene.from_dict(_scene({"code": ""}))
    assert sc.common == ""
    assert sc.to_dict()["common"] == {"code": ""}


def test_a_scene_without_one_carries_no_common_key():
    assert "common" not in Scene.from_dict(_scene()).to_dict()
    assert Scene.from_dict(_scene()).common is None


# assembly


def test_the_common_reaches_every_scene_pass():
    sc = Scene.from_dict(
        _scene(
            {"code": COMMON},
            [{"name": "a", "code": CODE}, {"name": "b", "code": FLAT}],
        )
    )
    for p in sc.passes:
        g = p.glsl(sc.params, sc.common)
        assert COMMON in g
        # after the generated preamble, before the pass's own code
        assert g.index("uniform float time;") < g.index(COMMON) < g.index(p.code)


def test_no_common_assembles_exactly_as_before():
    p = Pass(name="m", code=CODE)
    assert p.glsl(None, None) == p.glsl()
    assert p.glsl(None, "") == p.glsl()


def test_a_filter_pass_never_sees_the_common():
    sc = Scene.from_dict(_scene({"code": COMMON}))
    apply_filters(sc, ["dither"])
    filt = sc.passes[-1]
    assert filt.filter == "dither"
    assert COMMON not in filt.glsl(sc.params, sc.common)
    assert COMMON in sc.passes[0].glsl(sc.params, sc.common)


# the compiler


def test_a_scene_with_a_common_renders(renderer):
    sc = Scene.from_dict(
        _scene(
            {"code": "vec3 tint(float x){ return vec3(x, 0.25, 0.5); }"},
            [
                {
                    "name": "main",
                    "code": "void main(){ gl_FragColor = "
                    "vec4(tint(v_texCoord.x), 1.0); }",
                }
            ],
        )
    )
    assert renderer.setup(sc) == ""
    f = renderer.render().copy()
    assert abs(int(f[..., 1].mean()) - 64) <= 1
    assert abs(int(f[..., 2].mean()) - 128) <= 1
    assert int(f[..., 0].max()) > int(f[..., 0].min())  # the x ramp the function made


def test_an_edited_common_recompiles_into_the_passes(renderer):
    sc = Scene.from_dict(
        _scene(
            {"code": "float level(){ return 0.25; }"},
            [{"name": "main", "code": "void main(){ gl_FragColor = vec4(level()); }"}],
        )
    )
    assert renderer.setup(sc) == ""
    dim = renderer.render().copy()
    sc.common = "float level(){ return 0.75; }"
    assert renderer.setup(sc) == ""
    assert int(renderer.render()[..., 0].mean()) > int(dim[..., 0].mean())


def test_an_error_in_the_common_reports_the_common_line(renderer):
    bad = (
        "float sat(float x){ return clamp(x, 0.0, 1.0); }\nfloat b(){ return nope(); }"
    )
    sc = Scene.from_dict(_scene({"code": bad}))
    assert renderer.setup(sc)
    assert renderer.error_pass == COMMON_PASS
    assert renderer.error_line == 2  # the line the author wrote nope() on


def test_an_error_in_a_pass_still_reports_that_pass(renderer):
    sc = Scene.from_dict(
        _scene(
            {"code": COMMON},
            [
                {"name": "a", "code": FLAT},
                {"name": "b", "code": "void main(){\n  float v = 1.0;\n  nope(v);\n}"},
            ],
        )
    )
    assert renderer.setup(sc)
    assert renderer.error_pass == 1
    assert renderer.error_line == 3


def test_a_good_compile_clears_the_attribution(renderer):
    sc = Scene.from_dict(_scene({"code": COMMON}))
    assert renderer.setup(sc) == ""
    assert renderer.error_pass is None and renderer.error_line is None


# the page


def _index():
    with open(os.path.join(web_dir(), "index.html"), encoding="utf-8") as f:
        return f.read()


def _fn(src, head):
    i = src.index(head)
    return src[i : src.index("\n}", i)]


def test_the_common_tab_is_built_before_every_pass_tab():
    tabs = _fn(_index(), "function buildTabs()")
    assert tabs.index("hasCommon()") < tabs.index("work.passes.forEach")


def test_the_common_tab_carries_no_mute():
    tabs = _fn(_index(), "function buildTabs()")
    head = tabs[: tabs.index("work.passes.forEach")]
    assert "delCommon" in head  # removed like a pass
    assert "toggleMute" not in head and "EYE" not in head


def test_the_picker_offers_common_only_when_there_is_none():
    tabs = _fn(_index(), "function buildTabs()")
    assert re.search(r'if \(!hasCommon\(\)\) item\("common"', tabs)


def test_the_page_routes_a_common_error_to_the_common_tab():
    src = _index()
    assert f'const COMMON = "{COMMON_PASS}"' in src  # the token /stats sends
    assert "s.error_pass === curPass" in src


def test_a_fetched_common_surfaces_in_the_working_copy():
    merge = _fn(_index(), "function mergeNewParams(")
    assert "work.common = clone(pristine.common)" in merge
    assert "commonEdited" in merge  # an edited working copy keeps its own


def test_the_posted_scene_carries_the_common():
    body = _fn(_index(), "function compiledScene()")
    assert "common: hasCommon() ? {code: commonCode()} : undefined" in body


def test_the_common_tab_styles_both_themes():
    src = _index()
    css = src[src.index("<style>") : src.index("</style>")]
    for rule in re.findall(r"\.tab\.cm[^{]*\{[^}]*\}", css):
        assert "#" not in rule, rule  # themed through the variables, not literals
