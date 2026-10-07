"""A screen's drawing: which layer each call keeps, and what a frame sends.

A fake handle records what a drawing hands it, and a scene script's drawing
writes the wire, bar the tests at the end, which draw on the renderer.
"""

import os
import subprocess
import sys

import pytest
from conftest import REPO, numpy
from test_api import png_pixels
from test_errors import CLI
from test_shim import END, drive, load, messages

import ghee
from ghee import _shim


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

    def _drawing(self):
        return self.scene

    def _add_layer(self, pass_, index, layer):
        self.sent.append(("add", pass_, index, layer))

    def _update_layer(self, layer):
        self.sent.append(("update", layer))
        return True

    def set_layer(self, name, field, value, effect=None):
        self.sent.append(("set", name, field, value, effect))

    def frame(self):
        self.sent.append(("frame",))


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
        [
            ("add", "main", 3, bar2),
            ("set", "text2", "visible", False, None),
            FRAME,
        ],
    ]


def test_a_frame_whose_calls_changed_nothing_sends_nothing():
    def draw(d, i):
        d.text("12:00", x=0, y=0, font_size=16)
        d.rect(x=64, y=20, width=60, height=4)
        if i == 0:
            d.circle((110, 14), 10)

    sent = frames(Fake(), draw, 4)
    assert len(sent[0]) == 4
    assert sent[1] == [("set", "rect2", "visible", False, None), FRAME]
    assert sent[2] == sent[3] == [FRAME]


def test_an_unchanged_call_writes_its_look_once(monkeypatch):
    """A frame compares a call's keywords with its last frame's, and only a
    call whose keywords changed writes its look again."""
    looks = []
    look = _shim._look
    monkeypatch.setattr(
        _shim, "_look", lambda f, name: looks.append(name) or look(f, name)
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


def test_a_layer_whose_call_comes_back_takes_its_look_again():
    def draw(d, i):
        if i != 1:
            d.image("wifi.png", x=110, y=0)

    icon = {"name": "image", "x": 110.0, "y": 0.0, "image": "wifi.png"}
    assert frames(Fake(), draw, 4) == [
        [("add", "main", 0, icon), FRAME],
        [("set", "image", "visible", False, None), FRAME],
        [("update", icon), FRAME],
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
        ("add", "main", 0, {"name": "banner", "text": "hi"}),
        ("set", "title", "visible", False, None),
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
            "name": "rect",
            "x": 100.0,
            "y": 4.0,
            "color": "white",
            "shape": "rect",
            "width": 20.0,
            "height": 20.0,
            "border_radius": 10.0,
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
]


@pytest.mark.parametrize("call,look", SHAPES)
def test_a_call_writes_the_layer_of_the_scene_format(call, look):
    def draw(d, i):
        call(d)

    assert frames(Fake(), draw, 1) == [[("add", "main", 0, look), FRAME]]


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
    """A call that runs for the first time adds its layer, kind 6, a call
    whose look changes gives its layer the new look, kind 7, a layer whose
    call did not run hides through its live eye, and a tick whose calls
    changed nothing sends nothing."""
    shim = load()

    @shim.tick
    def show(tick):
        with shim.draw() as d:
            d.text(f"t{min(tick.frame, 1)}", x=0, y=0)
            if tick.frame == 0:
                d.rect(x=2, width=4, height=4)

    hello, out = drive(shim, None, ticks(3), monkeypatch)
    assert [p["name"] for p in hello["passes"]] == ["main"]
    text = {"name": "text", "x": 0.0, "y": 0.0}
    rect = {"name": "rect", "x": 2.0, "shape": "rect", "width": 4.0, "height": 4.0}
    assert answers(out) == [
        [
            (6, "main", {"index": 0, "layer": {**text, "text": "t0"}}),
            (6, "main", {"index": 1, "layer": rect}),
        ],
        [(7, "text", {**text, "text": "t1"}), (1, "rect.visible", [0.0])],
        [],
    ]


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


def test_a_drawing_reaches_the_screen_and_a_call_that_stops_hides():
    with ghee.open(width=16, height=4, timestep=0) as screen:
        for i in range(3):
            with screen.draw() as d:
                if i != 1:
                    d.rect(x=0, y=0, width=8 if i == 0 else 4, height=4, color="red")
            assert (
                red_columns(screen) == [[0, 1, 2, 3, 4, 5, 6, 7], [], [0, 1, 2, 3]][i]
            )


def test_an_error_of_a_call_reaches_python_and_draws_no_frame():
    with ghee.open(width=16, height=4) as screen:
        with pytest.raises(ValueError) as raised, screen.draw() as d:
            d.rect(color="navyy")
        assert str(raised.value) == (
            "layer 'rect' color needs '#rrggbb', a CSS colour name or three numbers "
            "[r, g, b], got 'navyy'"
        )
        with pytest.raises(RuntimeError, match="nothing is drawn yet"):
            screen.pixels()


# A pass that shows the array `level`, fed once, in red.
LEVEL = {
    "version": 1,
    "passes": [
        {
            "name": "main",
            "arrays": [{"name": "level", "size": 1}],
            "code": "void main(){ gl_FragColor = vec4(level_at(0), 0.0, 0.0, 1.0); }",
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
        timeout=120,
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


def test_ghee_render_demo_stats_draws_the_time_and_the_load(tmp_path):
    """`ghee render --demo stats` runs the bundled example under ghee, which
    draws its whole screen from its tick."""
    if not CLI.exists():
        pytest.skip(f"no {CLI} to run the scene")
    demos = tmp_path / "demos"
    demos.mkdir()
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
        timeout=120,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    assert "changes nothing" not in ran.stderr, ran.stderr
    w, h, px = png_pixels((tmp_path / "f.png").read_bytes())
    assert (w, h) == (128, 32)

    def lit(x0, x1, y0, y1):
        return any(
            px[(y * w + x) * 4] > 127 for y in range(y0, y1) for x in range(x0, x1)
        )

    # the time at the top left, and the frame of the load bar at the right
    assert lit(0, 64, 0, 16)
    assert lit(64, 128, 20, 28)
