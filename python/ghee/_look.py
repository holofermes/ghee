"""The look of a layer as the scene format writes it, from a Layer's fields
or from the values of a drawing's call.

A drawing takes each new call's look from here, and this module imports os
alone, so a drawing loads none of the scene shim.
"""

from __future__ import annotations

import os

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Any


def _plain(v: Any) -> Any:
    """`v` in the types that json and the bindings both take: a tuple as a
    list, a path as its str, a numpy number or array as its Python value.
    Anything else goes out as given, for ghee to judge."""
    if isinstance(v, os.PathLike):
        return os.fspath(v)
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (str, bytes)) or not hasattr(v, "tolist"):
        return v
    return v.tolist()


def _floats(v: Any) -> Any:
    """`v` as `_plain` leaves it, with each real number in it a float."""
    v = _plain(v)
    if isinstance(v, list):
        return [_floats(x) for x in v]
    if isinstance(v, dict):
        return {k: _floats(x) for k, x in v.items()}
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return v


# The effects of a text layer, in the order that the document writes them.
_EFFECTS = ("text_shadow", "glow", "text_stroke", "karaoke", "wave", "scroll", "blink")

# Every field of a Layer but its name, at its default, in the order of
# the Layer's fields.
_FIELDS = {
    "text": None,
    "visible": True,
    "x": None,
    "y": None,
    "color": None,
    "opacity": None,
    "blend": "normal",
    "font_size": None,
    "font_family": None,
    "text_align": "left",
    "render": None,
    "text_shadow": None,
    "glow": None,
    "text_stroke": None,
    "karaoke": None,
    "wave": None,
    "scroll": None,
    "blink": None,
    "image": None,
    "width": None,
    "height": None,
    "shape": None,
    "border": None,
    "border_radius": None,
    "points": None,
    "stroke_width": None,
    "radius": None,
}


def _look(f: dict, name: str | None) -> dict:
    """The layer of the fields `f`, every field of a Layer, as the scene
    format writes it, under `name`, or with no name for None."""
    d: dict = {} if name is None else {"name": name}
    if f["visible"] is not True:
        d["visible"] = _plain(f["visible"])
    for k in ("x", "y", "color", "opacity"):
        if f[k] is not None:
            d[k] = _floats(f[k])
    if f["blend"] != "normal":
        d["blend"] = f["blend"]
    if f["text"] is not None:
        d["text"] = f["text"]
    if f["image"] is not None:
        d["image"] = _plain(f["image"])
    if f["shape"] is not None:
        d["shape"] = f["shape"]
    for k in (
        "width",
        "height",
        "radius",
        "border",
        "border_radius",
        "points",
        "stroke_width",
    ):
        if f[k] is not None:
            d[k] = _floats(f[k])
    if f["font_size"] is not None:
        d["font_size"] = _floats(f["font_size"])
    if f["font_family"] is not None:
        d["font_family"] = _plain(f["font_family"])
    if f["text_align"] != "left":
        d["text_align"] = f["text_align"]
    if f["render"] is not None:
        d["render"] = f["render"]
    for k in _EFFECTS:
        if f[k] is not None:
            d[k] = _floats(f[k])
    return d


# The keywords of each kind of call, in the order of a call's values, as
# the native half numbers the kinds: a text, a rect, an image, a line and a
# circle, and the shape that each kind's look names.
_CALL_KEYS = (
    (
        "text",
        "x",
        "y",
        "color",
        "opacity",
        "blend",
        "visible",
        "font_size",
        "font_family",
        "text_align",
        "render",
        "text_shadow",
        "glow",
        "text_stroke",
        "karaoke",
        "wave",
        "scroll",
        "blink",
    ),
    (
        "x",
        "y",
        "width",
        "height",
        "color",
        "opacity",
        "blend",
        "visible",
        "border",
        "border_radius",
    ),
    ("image", "x", "y", "width", "height", "color", "opacity", "blend", "visible"),
    ("points", "x", "y", "color", "opacity", "blend", "visible", "stroke_width"),
    ("x", "y", "radius", "color", "opacity", "blend", "visible", "border"),
)
_CALL_SHAPES = (None, "rect", None, "line", "circle")


def _call_look(kind: int, vals: tuple, name: str | None) -> dict:
    """The look of a call of `kind` with `vals`, as the scene format writes
    it, under `name`, or with no name for None."""
    given = {k: v for k, v in zip(_CALL_KEYS[kind], vals) if v is not None}
    shape = _CALL_SHAPES[kind]
    if shape is not None:
        given["shape"] = shape
    return _look({**_FIELDS, **given}, name)


def _circle(centre, radius) -> tuple:
    """The x and y of the top left of the box of a circle of `radius`
    around `centre`, and the radius."""
    cx, cy = _plain(centre)
    return cx - radius, cy - radius, radius
