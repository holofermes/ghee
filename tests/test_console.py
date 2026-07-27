"""Console request helpers, tested without a running server or GL device."""

import pytest

from ghee.console.core import parse_param_value as _parse_param_value
from ghee.console.core import parse_scope
from ghee.console.server import _set_item


def test_parse_float():
    assert _parse_param_value("float", "0.5") == 0.5
    assert _parse_param_value("float", "3") == 3.0


def test_parse_int_rides_float_path():
    # int params parse as float here; the render thread rounds using the
    # scene's own param type, so the wire value stays a plain number.
    assert _parse_param_value("int", "5") == 5.0


def test_parse_color():
    assert _parse_param_value("color", "1,0,0.5") == [1.0, 0.0, 0.5]


def test_parse_color_wrong_length():
    with pytest.raises(ValueError):
        _parse_param_value("color", "1,0")
    with pytest.raises(ValueError):
        _parse_param_value("color", "1,0,0,0")


def test_parse_rejects_non_numbers():
    with pytest.raises(ValueError):
        _parse_param_value("float", "notanumber")
    with pytest.raises(ValueError):
        _parse_param_value("color", "1,x,0")


def test_parse_vec():
    assert _parse_param_value("vec2", "0.1,0.2") == [0.1, 0.2]
    assert _parse_param_value("vec3", "1,0,0.5") == [1.0, 0.0, 0.5]
    assert _parse_param_value("vec4", "0,0.25,0.5,1") == [0.0, 0.25, 0.5, 1.0]


def test_parse_vec_wrong_length():
    with pytest.raises(ValueError):
        _parse_param_value("vec2", "0.1")
    with pytest.raises(ValueError):
        _parse_param_value("vec2", "0.1,0.2,0.3")
    with pytest.raises(ValueError):
        _parse_param_value("vec3", "0.1,0.2")
    with pytest.raises(ValueError):
        _parse_param_value("vec4", "0.1,0.2,0.3")


def test_parse_vec_rejects_non_numbers():
    with pytest.raises(ValueError):
        _parse_param_value("vec3", "0.1,x,0.3")


def test_set_item_scene_scope():
    # pass="scene" (literal) routes to a distinct scene-scoped queue item
    assert _set_item("scene", "speed", 0.5) == ("scene_param", "speed", 0.5)


def test_set_item_pass_scope():
    assert _set_item("2", "k", 0.5) == ("param", 2, "k", 0.5)
    assert _set_item("0", "tint", [1.0, 0.0, 0.0]) == (
        "param",
        0,
        "tint",
        [1.0, 0.0, 0.0],
    )


def test_set_item_bad_pass_index():
    with pytest.raises(ValueError):
        _set_item("notanint", "k", 0.5)
    with pytest.raises(ValueError):
        parse_scope("notanint")


def test_parse_string():
    assert _parse_param_value("string", "GHEE") == "GHEE"
    assert _parse_param_value("string", "hello world") == "hello world"


def test_apply_values_overlays_and_keeps_shape():
    from ghee import Param, Pass, Scene
    from ghee.console.store import _apply_values

    scene = Scene(
        passes=[
            Pass(
                name="m",
                code="void main(){}",
                params={"amt": Param("amt", "float", 0.5, 0.0, 1.0, 0.01)},
            )
        ],
        params={
            "text": Param("text", "string", "GHEE"),
            "font": Param("font", "string", "Verdana", options=["Verdana", "Impact"]),
        },
    )
    _apply_values(
        scene,
        "t",
        {
            "scene": {"text": "WOW", "font": "Impact", "gone": 1.0},
            "passes": {"m": {"amt": 0.9, "bad": "x"}, "nopass": {"a": 1}},
        },
    )
    assert scene.params["text"].value == "WOW"
    assert scene.params["font"].value == "Impact"
    assert scene.params["font"].options == ["Verdana", "Impact"]  # survives overlay
    assert scene.passes[0].params["amt"].value == 0.9


def test_apply_values_rejects_wrong_types():
    from ghee import Param, Pass, Scene
    from ghee.console.store import _apply_values

    scene = Scene(
        passes=[
            Pass(
                name="m",
                code="void main(){}",
                params={"amt": Param("amt", "float", 0.5)},
            )
        ],
        params={"text": Param("text", "string", "GHEE")},
    )
    _apply_values(
        scene,
        "t",
        {"scene": {"text": 5.0}, "passes": {"m": {"amt": "not-a-number"}}},
    )
    assert scene.params["text"].value == "GHEE"  # bad value skipped
    assert scene.passes[0].params["amt"].value == 0.5
