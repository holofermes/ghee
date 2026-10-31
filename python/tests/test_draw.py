"""A screen's drawing: which layer each call keeps, and what a frame sends.

A fake handle records what a drawing hands it, and a scene script's drawing
writes the wire, bar the tests at the end, which draw on the renderer.
"""

import dataclasses
import inspect
import os
import pathlib
import re
import runpy
import subprocess
import sys

import pytest
from conftest import REPO, numpy
from test_api import png_pixels
from test_errors import CLI
from test_shim import END, drive, load, messages

import ghee
from ghee import _look, _shim


class Fake:
    """A handle as a drawing reaches it. The scene holds `layers` in its
    last pass, and `taken` beside them."""

    def __init__(self, layers=(), taken=()):
        self.sent = []
        self.scene = {
            "pass": "main",
            "layers": list(layers),
            "taken": ["main", *layers, *taken],
            "all": list(layers),
        }

    def _tick(self):
        return ghee.Tick(0, 0.0, 1 / 60, {})

    def _drawing(self):
        return self.scene

    def _add_layer(self, pass_, index, layer):
        self.sent.append(("add", pass_, index, layer))

    def _update_layer(self, layer):
        self.sent.append(("update", layer))
        return True

    def set_layer(self, name, field, value, effect=None):
        self.sent.append(("set", name, field, value, effect))

    def _remove_layer(self, name):
        self.sent.append(("remove", name))

    def _move_layer(self, name, index):
        self.sent.append(("move", name, index))

    def frame(self):
        self.sent.append(("frame",))


class Typed(Fake):
    """A handle that takes a drawing's changed calls with no look written,
    as the native half does, and leaves the calls of the layers named in
    `left` to _update_layer()."""

    def __init__(self, layers=(), taken=(), left=()):
        super().__init__(layers, taken)
        self.left = set(left)

    def _draw(self, calls):
        self.sent.append(
            ("draw", [(kind, layer, list(vals)) for kind, layer, vals in calls])
        )
        return [k for k, (_, layer, _) in enumerate(calls) if layer in self.left]


# The keywords of a text call and of a rect call, in the order of the values
# that _draw() takes.
TEXT_KEYS = [
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
]
LINE_KEYS = ["points", "x", "y", "color", "opacity", "blend", "visible", "stroke_width"]
RECT_KEYS = [
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
]


def values(keys, **given):
    """The values of a call that states `given`, in the order of `keys`."""
    return [given.get(k) for k in keys]


def frames(fake, draw, n):
    """Draw `n` frames on one screen, each through `draw(d, i)`, and give
    what each frame sent."""
    screen = _shim._Screen()
    sent = []
    for i in range(n):
        with _shim.Drawing(fake, screen) as d:
            draw(d, i)
        sent.append(fake.sent)
        fake.sent = []
    return sent


FRAME = ("frame",)


def test_a_call_keeps_its_layer_by_its_line_whatever_runs_above_it():
    """The alarm comes and goes above the scrolling title, which keeps its
    layer, and each bar of a loop takes its own."""

    def draw(d, i):
        if i == 1:
            d.text("ALARM", blink={"rate": 2})
        d.text("news", y=16, scroll={"speed": 2})
        for k in range(i):
            d.rect(x=k * 10, width=8)

    title = {"name": "text", "text": "news", "y": 16.0, "scroll": {"speed": 2.0}}
    alarm = {"name": "text2", "text": "ALARM", "blink": {"rate": 2.0}}
    bar = {"name": "rect", "x": 0.0, "shape": "rect", "width": 8.0}
    bar2 = {"name": "rect2", "x": 10.0, "shape": "rect", "width": 8.0}
    assert frames(Fake(), draw, 3) == [
        [("add", "main", 0, title), FRAME],
        [("add", "main", 0, alarm), ("add", "main", 2, bar), FRAME],
        [("remove", "text2"), ("add", "main", 2, bar2), FRAME],
    ]


def test_the_calls_on_one_line_keep_one_layer_whichever_of_them_runs():
    def draw(d, i):
        d.rect(x=1) if i == 0 else d.rect(x=2)

    assert frames(Fake(), draw, 2)[1] == [
        ("update", {"name": "rect", "x": 2.0, "shape": "rect"}),
        FRAME,
    ]


def test_a_frame_whose_calls_changed_nothing_sends_nothing():
    def draw(d, i):
        d.text("12:00", x=0, y=0, font_size=16)
        d.rect(x=64, y=20, width=60, height=4)
        if i == 0:
            d.circle((110, 14), 10)

    sent = frames(Fake(), draw, 4)
    assert len(sent[0]) == 4
    assert sent[1] == [("remove", "circle"), FRAME]
    assert sent[2] == sent[3] == [FRAME]


def test_an_unchanged_call_writes_its_look_once(monkeypatch):
    """A frame compares a call's keywords with its last frame's, and only a
    call whose keywords changed writes its look again."""
    looks = []
    look = _look._look
    monkeypatch.setattr(
        _look, "_look", lambda f, name: looks.append(name) or look(f, name)
    )

    def draw(d, i):
        d.text("12:00", x=0, y=0, font_size=16, color=(1.0, 0.5, 0.0))
        d.rect(x=64, y=20, width=60 if i < 2 else 30, height=4)

    frames(Fake(), draw, 3)
    assert looks == [None, None, "rect"]


@pytest.mark.parametrize("plain", [True, False], ids=["list", "numpy"])
def test_a_keyword_changed_in_place_sends_its_new_look(plain):
    """A list or an array that a keyword holds may change between frames
    and stay the same object, and the frame sends the look that it then
    gives."""
    make = list if plain else numpy().array
    points = make([[0.0, 0.0], [10.0, 10.0]])
    color = make([1.0, 0.0, 0.0])

    def draw(d, i):
        if i:
            points[1][0] = 20.0
            color[1] = 1.0
        d.lines(points, color=color)

    sent = frames(Fake(), draw, 2)
    look = {
        "name": "line",
        "shape": "line",
        "points": [[0.0, 0.0], [20.0, 10.0]],
        "color": [1.0, 1.0, 0.0],
    }
    assert sent[1] == [("update", look), FRAME]


def test_a_keyword_that_turns_from_a_name_to_an_array_sends_its_new_look():
    def draw(d, i):
        d.rect(width=8, color="red" if i == 0 else numpy().array([0.0, 1.0, 0.0]))

    sent = frames(Fake(), draw, 2)
    look = {"name": "rect", "shape": "rect", "width": 8.0, "color": [0.0, 1.0, 0.0]}
    assert sent[1] == [("update", look), FRAME]


def test_each_frame_states_the_whole_look_of_a_call():
    """A keyword left out takes its default again, so the look that goes out
    holds only what the frame's call states."""

    looks = [{"x": 4, "color": "red", "font_size": 12, "glow": {}}, {"font_size": 16}]

    def draw(d, i):
        d.text("a", **looks[i])

    sent = frames(Fake(), draw, 2)
    assert sent[1] == [
        ("update", {"name": "text", "text": "a", "font_size": 16.0}),
        FRAME,
    ]


def test_the_layer_of_a_call_that_stops_leaves_and_comes_back_with_the_call():
    def draw(d, i):
        if i != 1:
            d.image("wifi.png", x=110, y=0)

    icon = {"name": "image", "x": 110.0, "y": 0.0, "image": "wifi.png"}
    assert frames(Fake(), draw, 4) == [
        [("add", "main", 0, icon), FRAME],
        [("remove", "image"), FRAME],
        [("add", "main", 0, icon), FRAME],
        [FRAME],
    ]


def test_a_layer_of_the_scenes_own_whose_call_stops_hides():
    """A call that names a layer of the scene draws it, and the layer stays
    with the scene after the call stops."""

    def draw(d, i):
        if i == 0:
            d.text("12:00", name="clock")

    assert frames(Fake(layers=["clock"]), draw, 2)[1] == [
        ("set", "clock", "visible", False, None),
        FRAME,
    ]


def test_layers_paint_in_the_order_of_the_frames_calls():
    """The fewest layers move: the one call that comes first, then last
    again."""

    def draw(d, i):
        for name in ("c", "a", "b") if i == 1 else ("a", "b", "c"):
            d.rect(name=name)

    assert frames(Fake(), draw, 4)[1:] == [
        [("move", "c", 0), FRAME],
        [("move", "c", 2), FRAME],
        [FRAME],
    ]


def test_a_new_call_goes_in_after_the_layer_of_the_call_before_it():
    """The scene's own layer stays under the drawing."""

    def draw(d, i):
        d.text("a")
        if i == 1:
            d.text("b")
        d.text("c")

    sent = frames(Fake(layers=["backdrop"]), draw, 2)
    assert [s[:3] for s in sent[0][:-1]] == [("add", "main", 1), ("add", "main", 2)]
    assert sent[1] == [
        ("add", "main", 2, {"name": "text3", "text": "b"}),
        FRAME,
    ]


def test_a_name_names_the_layer_and_a_scene_layer_of_that_name_takes_the_call():
    def draw(d, i):
        d.text("12:00", name="clock")
        d.text("hi", name="title")

    fake = Fake(layers=["clock"])
    sent = frames(fake, draw, 1)
    assert sent[0] == [
        ("update", {"name": "clock", "text": "12:00"}),
        ("add", "main", 1, {"name": "title", "text": "hi"}),
        FRAME,
    ]


def test_a_call_that_takes_another_name_draws_the_layer_of_that_name():
    def draw(d, i):
        d.text("hi", name="title" if i == 0 else "banner")

    sent = frames(Fake(), draw, 2)
    assert sent[1] == [
        ("remove", "title"),
        ("add", "main", 0, {"name": "banner", "text": "hi"}),
        FRAME,
    ]


def test_an_automatic_name_skips_every_name_that_the_scene_holds():
    def draw(d, i):
        d.text("a")
        d.lines([(0, 0), (8, 8)])

    sent = frames(Fake(taken=["text", "text2", "line"]), draw, 1)
    assert [s[3]["name"] for s in sent[0][:-1]] == ["text3", "line2"]


@pytest.mark.parametrize(
    "draw",
    [
        lambda d: (d.text("a", name="title"), d.text("b", name="title")),
        lambda d: (d.text("a", name="text"), d.image("x.png", name="text")),
    ],
)
def test_two_calls_that_draw_one_layer_in_a_frame_raise(draw):
    fake = Fake()
    with pytest.raises(ValueError) as raised, _shim.Drawing(fake, _shim._Screen()) as d:
        draw(d)
    assert str(raised.value).startswith("two calls in one frame draw layer '")
    assert fake.sent == []


def test_a_name_given_to_a_call_clashes_with_the_name_that_another_call_took():
    def draw(d, named):
        d.text("a")
        if named:
            d.text("b", name="text")

    screen = _shim._Screen()
    fake = Fake()
    with _shim.Drawing(fake, screen) as d:
        draw(d, False)
    fake.sent = []
    with pytest.raises(ValueError) as raised, _shim.Drawing(fake, screen) as d:
        draw(d, True)
    assert str(raised.value) == "two calls in one frame draw layer 'text'"
    assert fake.sent == []


# (a call, and the look that it writes)
SHAPES = [
    (
        lambda d: d.circle((110, 14), 10, color="white"),
        {
            "name": "circle",
            "x": 110.0,
            "y": 14.0,
            "color": "white",
            "shape": "circle",
            "radius": 10.0,
        },
    ),
    (
        lambda d: d.line((0, 31), (127, 31), color="white", stroke_width=1),
        {
            "name": "line",
            "color": "white",
            "shape": "line",
            "points": [[0.0, 31.0], [127.0, 31.0]],
            "stroke_width": 1.0,
        },
    ),
    (
        lambda d: d.rect(color="navy"),
        {"name": "rect", "color": "navy", "shape": "rect"},
    ),
    (
        lambda d: d.text("hi", color=ghee.gradient("red", "yellow", angle=90)),
        {
            "name": "text",
            "color": {"from": "red", "to": "yellow", "angle": 90.0},
            "text": "hi",
        },
    ),
    (
        lambda d: d.text(
            "hi", font_family=("Pixel Operator 8", pathlib.Path("noto.ttf"))
        ),
        {"name": "text", "text": "hi", "font_family": ["Pixel Operator 8", "noto.ttf"]},
    ),
]


@pytest.mark.parametrize("call,look", SHAPES)
def test_a_call_writes_the_layer_of_the_scene_format(call, look):
    def draw(d, i):
        call(d)

    assert frames(Fake(), draw, 1) == [[("add", "main", 0, look), FRAME]]


def test_a_frame_hands_draw_the_calls_that_changed_alone_with_their_values():
    """A handle with _draw takes each frame's changed calls of layers that
    the screen holds in one call, each as its kind, its layer and its
    values in the order of its keywords."""

    def draw(d, i):
        d.text(f"{min(i, 1)}", x=0, y=0, font_size=8, font_family="Pixel Operator 8")
        d.rect(x=4, width=8, height=2 + (i >= 2))

    text = values(
        TEXT_KEYS, text="1", x=0, y=0, font_size=8, font_family="Pixel Operator 8"
    )
    rect = values(RECT_KEYS, x=4, width=8, height=3)
    assert frames(Typed(), draw, 4)[1:] == [
        [("draw", [(0, "text", text)]), FRAME],
        [("draw", [(1, "rect", rect)]), FRAME],
        [FRAME],
    ]


def test_a_call_whose_lists_and_dicts_hold_the_same_values_sends_nothing():
    def draw(d, i):
        d.text("a", color=[1.0, 0.5, 0.0], glow={"strength": 0.5, "color": (1, 0, 0)})
        d.lines([(0, 0), (4, 4 if i < 2 else 5)])

    assert frames(Typed(), draw, 3)[1:] == [
        [FRAME],
        [("draw", [(3, "line", values(LINE_KEYS, points=[(0, 0), (4, 5)]))]), FRAME],
    ]


def test_calls_on_one_line_of_two_files_draw_two_layers(tmp_path):
    """The file of a call is part of its line: a call of one file that stops
    and a call of another file on the same line that starts draw apart."""
    made = {}
    for x, name in enumerate(("one.py", "two.py"), 1):
        (tmp_path / name).write_text(f"def draw(d):\n    d.rect(x={x})\n")
        made[name] = runpy.run_path(str(tmp_path / name))["draw"]

    def draw(d, i):
        made["one.py" if i == 0 else "two.py"](d)

    assert frames(Fake(), draw, 2)[1] == [
        ("remove", "rect"),
        ("add", "main", 0, {"name": "rect", "x": 2.0, "shape": "rect"}),
        FRAME,
    ]


def test_a_call_that_draw_leaves_goes_out_as_its_look():
    def draw(d, i):
        d.rect(x=i)
        d.text("a", x=i)

    sent = frames(Typed(left=["text"]), draw, 2)
    assert sent[1] == [
        (
            "draw",
            [
                (1, "rect", values(RECT_KEYS, x=1)),
                (0, "text", values(TEXT_KEYS, text="a", x=1)),
            ],
        ),
        ("update", {"name": "text", "x": 1.0, "text": "a"}),
        FRAME,
    ]


def test_a_call_that_draw_takes_writes_no_look(monkeypatch):
    looks = []
    look = _look._look
    monkeypatch.setattr(
        _look, "_look", lambda f, name: looks.append(name) or look(f, name)
    )

    def draw(d, i):
        d.text(f"{i}", x=0, y=0, glow={"strength": 0.5})
        d.rect(x=i, border={"width": 1})

    frames(Typed(), draw, 3)
    assert looks == [None, None]


def test_the_calls_of_a_draw_that_raised_go_to_draw_again():
    """A call that _draw may have taken before it raised goes to _draw
    again, though its values are those of the frame before."""

    class Breaks(Typed):
        def _draw(self, calls):
            super()._draw(calls)
            if len(self.sent) == 1:
                raise RuntimeError("the GPU went away")
            return []

    def draw(d, i):
        d.rect(x=1 if i == 1 else 0)

    screen = _shim._Screen()
    fake = Breaks()
    with _shim.Drawing(fake, screen) as d:
        draw(d, 0)
    fake.sent = []
    with pytest.raises(RuntimeError), _shim.Drawing(fake, screen) as d:
        draw(d, 1)
    with _shim.Drawing(fake, screen) as d:
        draw(d, 2)
    assert fake.sent == [
        ("draw", [(1, "rect", values(RECT_KEYS, x=1))]),
        ("draw", [(1, "rect", values(RECT_KEYS, x=0))]),
        FRAME,
    ]


@pytest.mark.parametrize(
    "call,error",
    [
        (lambda d: d.text("a", name=3), "name needs a string, got int"),
        (lambda d: d.circle((1, 1), 1, name=b"c"), "name needs a string, got bytes"),
    ],
    ids=["text", "circle"],
)
def test_a_name_that_is_no_string_raises_type_error(call, error):
    fake = Fake()
    with pytest.raises(TypeError) as raised, _shim.Drawing(fake, _shim._Screen()) as d:
        call(d)
    assert str(raised.value) == error
    assert fake.sent == []


# --------------------------------------------------------- a scene script


def ticks(n):
    return [{"frame": i, "time": 0.0, "dt": 0.0, "params": {}} for i in range(n)]


def answers(buf):
    """The messages of each tick's answer, kind 5 left out."""
    out = [[]]
    for m in messages(buf):
        if m == END:
            out.append([])
        else:
            out[-1].append(m)
    return out[:-1]


def test_a_drawing_in_a_tick_goes_out_with_the_ticks_answer(monkeypatch):
    """Each change goes out as a kind 6 JSON Patch: a call that runs for the
    first time adds its layer, the layer of a call that did not run leaves,
    the calls whose looks change in live fields alone replace their layers
    in one patch, a call that builds its layer again replaces its layer
    alone, and a tick whose calls changed nothing sends nothing."""
    shim = load()

    @shim.tick
    def show(tick):
        with shim.draw() as d:
            d.text(f"t{min(tick.frame, 1)}", x=0, y=0)
            d.text("big", y=8, font_size=8 if tick.frame == 0 else 9)
            if tick.frame == 0:
                d.rect(x=2, width=4, height=4)

    hello, out = drive(shim, None, ticks(3), monkeypatch)
    assert [p["name"] for p in hello["passes"]] == ["main"]
    text = {"name": "text", "x": 0.0, "y": 0.0}
    big = {"name": "text2", "text": "big", "y": 8.0}
    rect = {"name": "rect", "x": 2.0, "shape": "rect", "width": 4.0, "height": 4.0}

    def patch(op, path, value=None):
        o = {"op": op, "path": path}
        if value is not None:
            o["value"] = value
        return (6, "", [o])

    assert answers(out) == [
        [
            patch("add", "/main/0", {**text, "text": "t0"}),
            patch("add", "/main/1", {**big, "font_size": 8.0}),
            patch("add", "/main/2", rect),
        ],
        [
            patch("remove", "/main/2"),
            patch("replace", "/main/0", {**text, "text": "t1"}),
            patch("replace", "/main/1", {**big, "font_size": 9.0}),
        ],
        [],
    ]


def test_a_drawing_in_a_tick_moves_a_layer_into_the_order_of_the_calls(monkeypatch):
    shim = load()

    @shim.tick
    def show(tick):
        with shim.draw() as d:
            for name in ("b", "a") if tick.frame == 1 else ("a", "b"):
                d.rect(name=name)

    _, out = drive(shim, None, ticks(2), monkeypatch)
    assert answers(out)[1] == [
        (6, "", [{"op": "move", "from": "/main/1", "path": "/main/0"}])
    ]


def test_numpy_values_and_paths_go_out_as_the_values_that_they_hold(monkeypatch):
    """A numpy array, a numpy number and a path draw as the list, the number
    and the str that they hold: a change in live fields alone goes out in
    one patch, and the same values again send nothing."""
    np = numpy()
    shim = load()

    @shim.tick
    def show(tick):
        i = min(tick.frame, 1)
        with shim.draw() as d:
            d.lines(np.array([[0.0, 0.0], [4.0, 4.0 + i]]), x=np.float32(1))
            d.image(pathlib.Path("wifi.png"), x=np.int64(i))

    _, out = drive(shim, None, ticks(3), monkeypatch)
    sent = answers(out)
    line = {
        "name": "line",
        "shape": "line",
        "points": [[0.0, 0.0], [4.0, 5.0]],
        "x": 1.0,
    }
    image = {"name": "image", "image": "wifi.png", "x": 1.0}
    assert [m[0] for m in sent[0]] == [6, 6]
    replace = [
        {"op": "replace", "path": "/main/0", "value": line},
        {"op": "replace", "path": "/main/1", "value": image},
    ]
    assert sent[1:] == [[(6, "", replace)], []]


@pytest.mark.parametrize("bad", [0, 1])
def test_a_drawn_layer_that_ghee_would_refuse_raises_in_the_script(bad, monkeypatch):
    """The call that adds the layer, and the call that gives it a new look."""
    shim = load()

    @shim.tick
    def show(tick):
        with shim.draw() as d:
            d.rect(color="navyy" if tick.frame == bad else "navy")

    with pytest.raises(ValueError) as raised:
        drive(shim, None, ticks(2), monkeypatch)
    assert str(raised.value).startswith("layer 'rect' color needs")


def test_a_drawn_file_leaves_measure_alone(monkeypatch):
    """ghee loads a drawn picture, so the script's measure never opens it."""
    shim = load()
    got = []

    @shim.tick
    def show(tick):
        with shim.draw() as d:
            d.image("nowhere.png", x=0, y=0)
        got.append(shim.measure("hi", font_size=12))

    drive(shim, None, ticks(1), monkeypatch)
    assert got[0][0] > 0


def test_draw_outside_run_says_where_it_belongs():
    shim = load()
    with pytest.raises(RuntimeError) as raised:
        shim.draw()
    assert str(raised.value) == (
        "ghee.draw draws on the scene that run() runs: call it in a tick or a feed"
    )


# ---------------------------------------------------------------- the engine


def red_columns(h):
    """The columns of the first row whose red is lit."""
    px = h.pixels()
    w = h.size[0]
    return [x for x in range(w) if px[4 * x] > 127]


def test_a_drawing_reaches_the_screen_and_a_call_that_stops_takes_its_layer_out():
    with ghee.open(width=16, height=4, timestep=0) as screen:
        for i in range(3):
            with screen.draw() as d:
                if i != 1:
                    d.rect(x=0, y=0, width=8 if i == 0 else 4, height=4, color="red")
            assert (
                red_columns(screen) == [[0, 1, 2, 3, 4, 5, 6, 7], [], [0, 1, 2, 3]][i]
            )
            assert screen._drawing()["layers"] == [["rect"], [], ["rect"]][i]


def test_the_first_frame_of_a_drawing_holds_its_words_with_a_glow(
    tmp_path, monkeypatch
):
    """On a cold glyph atlas, as a frame under a pinned time draws them."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    frames = []
    for timestep in (None, 0):
        with ghee.open(width=64, height=32, timestep=timestep) as screen:
            with screen.draw() as d:
                d.text("hi", glow={})
            frames.append(screen.pixels())
    assert max(frames[1][0::4]) == 255
    assert frames[0] == frames[1]


def test_the_last_call_paints_on_top_whichever_came_last_before():
    with ghee.open(width=4, height=4, timestep=0) as screen:
        for top in ("red", "lime", "red"):
            with screen.draw() as d:
                for color in ("lime", "red") if top == "red" else ("red", "lime"):
                    d.rect(name=color, color=color)
            want = (255, 0, 0) if top == "red" else (0, 255, 0)
            assert tuple(screen.pixels()[:3]) == want, top


class Looks:
    """A handle whose drawing writes the look of every changed call."""

    def __init__(self, h):
        self._h = h

    def __getattr__(self, name):
        return getattr(self._h, name)


COLOURS = ["red", "#20c040", (0.2, 0.4, 1.0), [1.0, 1.0, 0.0], "Navy", None]


def busy(d, i):
    """One frame of a drawing whose calls change their keywords from frame to
    frame, a call that comes and goes, and a size that builds a text
    again at frame 4."""
    c = COLOURS[i % len(COLOURS)]
    d.rect(color="navy")
    d.text(f"{i % 10}", x=1, y=1, font_size=8, font_family="Pixel Operator 8", color=c)
    d.text(
        "AV",
        font_size=12 if i < 4 else 14,
        opacity=0.5 if i % 2 else None,
        color=ghee.gradient("red", "yellow", angle=30 * i) if i % 3 == 0 else None,
        glow={"strength": 0.5, "color": c or "white"} if i % 4 != 2 else None,
    )
    if i % 5 != 3:
        d.text("sing", x=30, y=1, font_size=8, karaoke={"sung": i % 4})
    d.rect(
        x=20 + i,
        y=20,
        width=8,
        height=4,
        border={"width": 1 + i % 2, "color": "red"} if i % 3 else None,
        border_radius=i % 3,
    )
    d.circle((50, 24), 3 + i % 2, color=c or "white")
    d.line((0, 31), (i * 6, 24), stroke_width=1 + i % 2, color="lime")
    d.lines([(10, 30), (14, 22 + i % 3), (18, 30)], x=i % 2)


def test_a_drawing_draws_the_frames_that_its_looks_written_out_draw(monkeypatch):
    """A screen's drawing draws each frame as a drawing that writes the look
    of each changed call out, and writes a look only for a call that builds
    its layer again."""
    names = []
    look = _look._call_look
    monkeypatch.setattr(
        _look,
        "_call_look",
        lambda kind, vals, name: names.append(name) or look(kind, vals, name),
    )
    with (
        ghee.open(width=64, height=32, timestep=0) as screen,
        ghee.open(width=64, height=32, timestep=0) as written,
    ):
        state = _shim._Screen()
        updates = 0
        for i in range(12):
            names.clear()
            with screen.draw() as d:
                busy(d, i)
            updates += sum(name is not None for name in names)
            with _shim.Drawing(Looks(written), state) as d:
                busy(d, i)
            assert screen.pixels() == written.pixels(), f"frame {i}"
        assert updates == 1


def test_a_drawing_draws_its_frames_without_the_scene_shim():
    """A module that a drawing imports loads again at each start: the scene
    shim's imports took 230 ms on a Pi Zero 2 W."""
    probe = (
        "import sys, ghee\n"
        "with ghee.open(width=16, height=4, timestep=0) as screen:\n"
        "    with screen.draw() as d:\n"
        "        d.rect(color='navy')\n"
        "        d.text('hi', name='t', x=0, y=0, font_size=4, visible=False)\n"
        "        d.circle((2, 2), 1)\n"
        "print('ghee._shim' in sys.modules)\n"
    )
    ran = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    assert ran.stdout.split() == ["False"]


def test_a_call_starts_from_the_defaults_of_a_layer():
    """A drawing's call and a Layer write the same look for the same fields."""
    fields = {f.name: f.default for f in dataclasses.fields(ghee.Layer)}
    del fields["name"]
    assert _look._FIELDS == fields


def test_an_error_of_a_call_reaches_python_and_draws_no_frame():
    with ghee.open(width=16, height=4) as screen:
        with pytest.raises(ValueError) as raised, screen.draw() as d:
            d.rect(color="navyy")
        assert str(raised.value).startswith("layer 'rect' color needs")
        with pytest.raises(RuntimeError, match="nothing is drawn yet"):
            screen.pixels()


# A pass that shows the array `level`, fed once, in red.
LEVEL = {
    "version": 1,
    "passes": [
        {
            "name": "main",
            "arrays": [{"name": "level", "size": 1}],
            "fragment": "void main(){ gl_FragColor = vec4(level_at(0), 0.0, 0.0, 1.0); }",
        }
    ],
}


def test_a_call_built_again_leaves_the_scene_and_the_other_layers_running():
    """The clock takes a new size at frame 2 and builds again, alone: the
    array fed at frame 0 and the scrolling news above the clock draw every
    frame as on a screen whose clock keeps its size."""

    def run(grows):
        rows = []
        with ghee.open(LEVEL, width=64, height=48, timestep=0.1) as screen:
            screen.feed("level", [0.5])
            for i in range(5):
                with screen.draw() as d:
                    d.text("news of the day", y=0, font_size=12, scroll={"speed": 4})
                    d.text("12:00", y=24, font_size=16 if grows and i >= 2 else 12)
                rows.append(screen.pixels()[: 64 * 4 * 16])
        return rows

    rows = run(True)
    # the last pixel of row 15, under the news, shows the array
    assert abs(rows[-1][(15 * 64 + 63) * 4] - 127.5) <= 1
    assert rows == run(False)


# A scene script with no scene of its own that draws a white box on every
# tick, and a red box beside it on the first alone, on a 12x4 frame. It
# writes the number of each tick on stderr.
DRAWN = """#!{python}
import sys

import ghee


@ghee.tick
def show(tick):
    print("tick", tick.frame, file=sys.stderr)
    with ghee.draw() as d:
        d.rect(x=0, y=0, width=4, height=4)
        if tick.frame == 0:
            d.rect(x=6, y=0, width=4, height=4, color="red")


ghee.run()
"""


def test_a_scene_script_with_no_scene_draws_on_black_in_its_own_frames(tmp_path):
    """A direct run: each tick's drawing lands in the tick's frame, and draws
    no frame of its own."""
    path = tmp_path / "drawn.py"
    path.write_text(DRAWN.format(python=sys.executable))
    ran = subprocess.run(
        [sys.executable, str(path), "-o", "-", "--size", "12x4"]
        + ["--frames", "2", "--timestep", "0"],
        capture_output=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    seen = [
        line for line in ran.stderr.decode().splitlines() if line.startswith("tick")
    ]
    assert seen == ["tick 0", "tick 1"]
    n = len(ran.stdout) // (2 * 12 * 4)
    first, second = ran.stdout[: 12 * 4 * n], ran.stdout[12 * 4 * n :]

    def rgb(frame, x, y):
        return list(frame[(y * 12 + x) * n : (y * 12 + x) * n + 3])

    for frame in (first, second):
        assert rgb(frame, 2, 2) == [255, 255, 255]
        assert rgb(frame, 11, 2) == [0, 0, 0]
    assert rgb(first, 8, 2) == [255, 0, 0]
    assert rgb(second, 8, 2) == [0, 0, 0]


# A scene script with no scene of its own whose tick draws busy() above.
BUSY = """#!{python}
import ghee

COLOURS = {colours!r}


{busy}

@ghee.tick
def show(tick):
    with ghee.draw() as d:
        busy(d, tick.frame)


ghee.run()
"""


def test_a_scene_script_under_ghee_draws_the_frames_that_a_screen_draws(tmp_path):
    """The script's drawing reaches ghee as kind 6 patches and live sets, and
    ghee draws each frame as a screen's drawing does."""
    if not CLI.exists():
        pytest.skip(f"no {CLI} to run the scene")
    path = tmp_path / "busy.py"
    path.write_text(
        BUSY.format(
            python=sys.executable, colours=COLOURS, busy=inspect.getsource(busy)
        )
    )
    path.chmod(0o755)
    frames = 12
    # ghee takes the screen's glyph atlas cache: a glyph that sits elsewhere
    # in the atlas can draw a level apart
    ran = subprocess.run(
        [str(CLI), "render", str(path), "-o", "-", "--size", "64x32"]
        + ["--frames", str(frames), "--timestep", "0"],
        capture_output=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    assert b"changes nothing" not in ran.stderr, ran.stderr
    size = 64 * 32 * 4
    assert len(ran.stdout) == frames * size
    with ghee.open(width=64, height=32, timestep=0) as screen:
        for i in range(frames):
            with screen.draw() as d:
                busy(d, i)
            got = ran.stdout[i * size : (i + 1) * size]
            assert got == screen.pixels(), f"frame {i}"


def test_ghee_render_demo_stats_draws_the_chart_and_the_readings(tmp_path):
    """`ghee render --demo stats` runs the bundled example under ghee, which
    draws its whole screen from its tick."""
    if not CLI.exists():
        pytest.skip(f"no {CLI} to run the scene")
    demos = tmp_path / "demos"
    (demos / "fonts").mkdir(parents=True)
    font = "fonts/DepartureMono-Regular.otf"
    (demos / font).write_bytes((REPO / "ghee/examples" / font).read_bytes())
    # the Python of this test, which holds this wheel, in place of uv's
    body = (REPO / "ghee/examples/stats.py").read_text().partition("\n")[2]
    stats = demos / "stats.py"
    stats.write_text(f"#!{sys.executable}\n{body}")
    stats.chmod(0o755)
    ran = subprocess.run(
        [str(CLI), "render", "--demo", "stats", "-o", str(tmp_path / "f.png")]
        + ["--size", "128x32", "--timestep", "0.1"],
        env={
            **os.environ,
            "GHEE_EXAMPLES": str(demos),
            "XDG_CACHE_HOME": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    assert "changes nothing" not in ran.stderr, ran.stderr
    w, h, px = png_pixels((tmp_path / "f.png").read_bytes())
    assert (w, h) == (128, 32)

    def lit(x0, x1, y0, y1):
        return any(
            max(px[(y * w + x) * 4 : (y * w + x) * 4 + 3]) > 127
            for y in range(y0, y1)
            for x in range(x0, x1)
        )

    # the chart's frame at the left, the load at the top right, and a
    # reading at the bottom right
    assert lit(0, 1, 0, 32)
    assert lit(64, 128, 0, 9)
    assert lit(64, 128, 22, 31)


def pictures(d, i):
    """A numpy array whose colour turns each frame on the left half, and a
    Pillow image on the right half."""
    import numpy as np
    from PIL import Image

    rgb = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i % 3]
    d.image(np.full((2, 2, 3), rgb, np.uint8), x=0, y=0, width=4, height=4)
    d.image(Image.new("RGBA", (1, 1), (255, 255, 0, 255)), x=4, y=0, width=4, height=4)


PICTURES = """#!{python}
import ghee


{pictures}

@ghee.tick
def show(tick):
    with ghee.draw() as d:
        pictures(d, tick.frame)


ghee.run()
"""


def test_a_picture_in_memory_draws_on_a_screen_and_under_ghee_alike(tmp_path):
    numpy()
    pytest.importorskip("PIL.Image")
    frames = 3
    size = 8 * 4 * 4
    with ghee.open(width=8, height=4, timestep=0) as screen:
        drawn = []
        for i in range(frames):
            with screen.draw() as d:
                pictures(d, i)
            drawn.append(screen.pixels())
    for i, rgb in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        assert tuple(drawn[i][0:3]) == rgb
        assert tuple(drawn[i][16:19]) == (255, 255, 0)
    if not CLI.exists():
        pytest.skip(f"no {CLI} to run the scene")
    path = tmp_path / "pictures.py"
    path.write_text(
        PICTURES.format(python=sys.executable, pictures=inspect.getsource(pictures))
    )
    path.chmod(0o755)
    ran = subprocess.run(
        [str(CLI), "render", str(path), "-o", "-", "--size", "8x4"]
        + ["--frames", str(frames), "--timestep", "0"],
        capture_output=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    assert [ran.stdout[i * size : (i + 1) * size] for i in range(frames)] == drawn


@pytest.mark.parametrize(
    "image,error",
    [
        (lambda np: np.zeros((2, 2, 3), np.float32), "float32 of shape (2, 2, 3)"),
        (lambda np: np.zeros((2, 2), np.uint8), "uint8 of shape (2, 2)"),
        (lambda np: 3, "got int"),
    ],
)
def test_a_picture_that_is_no_image_raises_type_error(image, error):
    np = numpy()
    with (
        ghee.open(width=8, height=4) as screen,
        pytest.raises(TypeError, match=re.escape(error)),
        screen.draw() as d,
    ):
        d.image(image(np))


def test_a_scene_layer_takes_the_pictures_that_d_image_takes():
    np = numpy()
    Image = pytest.importorskip("PIL.Image")
    pictures = [
        np.full((2, 2, 3), (255, 0, 0), np.uint8),
        Image.new("RGBA", (1, 1), (255, 255, 0, 255)),
    ]
    black = "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }"
    with ghee.open(width=8, height=4, timestep=0) as screen:
        with screen.draw() as d:
            for i, picture in enumerate(pictures):
                d.image(picture, x=4 * i, y=0, width=4, height=4)
        drawn = screen.pixels()
    layers = [
        ghee.Layer(f"p{i}", image=picture, x=4 * i, y=0, width=4, height=4)
        for i, picture in enumerate(pictures)
    ]
    scene = ghee.Scene(passes=[ghee.Pass("main", black, layers=layers)])
    with ghee.open(scene, width=8, height=4, timestep=0) as screen:
        screen.frame()
        assert screen.pixels() == drawn
    assert tuple(drawn[0:3]) == (255, 0, 0)
    assert tuple(drawn[16:19]) == (255, 255, 0)


# A scene whose text layer `title` declares its place, its size and its
# colour, red.
OWN = {
    "version": 1,
    "passes": [
        {
            "name": "main",
            "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
            "layers": [
                {
                    "name": "title",
                    "text": "I",
                    "x": 2,
                    "y": 2,
                    "font_size": 12,
                    "color": "#ff0000",
                }
            ],
        }
    ],
}


def own(draw, layers, i):
    """Frame 0 draws the scene's title by its name, frame 1 green, frame 2
    leaves the colour out again, and frames 3 and 4 set its font_size
    instead, the second larger."""
    if i < 3:
        with draw() as d:
            d.text("I", name="title", color="#00ff00" if i == 1 else None)
    else:
        layers.title.font_size = 12 if i == 3 else 20


OWNED = """#!{python}
import ghee

{own}

@ghee.tick
def show(tick):
    own(ghee.draw, ghee.layers, tick.frame)


ghee.run({scene!r})
"""


def test_a_scene_layer_keeps_its_own_values_on_a_screen_and_under_ghee_alike(tmp_path):
    def peaks(px):
        return [max(px[c::4]) for c in range(3)]

    def lit(px):
        return sum(1 for i in range(0, len(px), 4) if max(px[i : i + 3]) > 127)

    frames = 5
    size = 24 * 24 * 4
    with ghee.open(OWN, width=24, height=24, timestep=0) as screen:
        drawn = []
        for i in range(frames):
            own(screen.draw, screen.layers, i)
            if i >= 3:
                screen.frame()
            drawn.append(screen.pixels())
    lit_by = [[c > 127 for c in peaks(px)] for px in drawn[:3]]
    assert lit_by == [[True, False, False], [False, True, False], [True, False, False]]
    assert lit(drawn[4]) > lit(drawn[3])
    if not CLI.exists():
        pytest.skip(f"no {CLI} to run the scene")
    path = tmp_path / "own.py"
    path.write_text(
        OWNED.format(python=sys.executable, own=inspect.getsource(own), scene=OWN)
    )
    path.chmod(0o755)
    ran = subprocess.run(
        [str(CLI), "render", str(path), "-o", "-", "--size", "24x24"]
        + ["--frames", str(frames), "--timestep", "0"],
        capture_output=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    assert [ran.stdout[i * size : (i + 1) * size] for i in range(frames)] == drawn
