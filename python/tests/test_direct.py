"""Direct mode: a scene with no ghee parent hosts itself on the bindings.

$GHEE_WIRE marks the process ghee started. Without it, run() opens the scene,
drives the same hooks and feeds, and honours the same skip. A fake `ghee.open`
in sys.modules stands in for the renderer, bar the last test.
"""

import io
import re
import shutil
import signal
import subprocess
import sys
import types

import pytest
from conftest import REPO
from test_shim import REFUSED, load, scene_of


class Fake:
    """The handle open() hands back. It records what the loop asks of it, and
    its ticks run half a second apart."""

    def __init__(self, on_frame=None):
        self.calls = []
        self.on_frame = on_frame
        self.drawn = 0
        self.ticks = 0
        self.opened = None
        # the params of each tick, then {}
        self.changes = [{"words": "hi"}, {}, {"words": "ho"}]

    def _tick(self):
        if self.opened is None:
            params = self.changes.pop(0) if self.changes else {}
            n = self.ticks
            self.opened = {"frame": n, "time": 0.5 * n, "dt": 0.5, "params": params}
        return self.opened

    def _taken(self):
        tick = self._tick()
        self.opened = None
        self.ticks += 1
        return tick

    def feed(self, name, values):
        self.calls.append(("feed", name, list(values)))

    def set_param(self, name, value, pass_=None):
        self.calls.append(("set_param", name, value, pass_))

    def set_layer(self, name, field, value, effect=None):
        self.calls.append(("set_layer", name, field, value, effect))

    def frame(self):
        self.calls.append(("frame",))
        self.drawn += 1
        if self.on_frame:
            self.on_frame(self.drawn)
        return self._taken()

    def skip(self):
        self.calls.append(("skip",))
        return self._taken()

    def save(self, path):
        self.calls.append(("save", path))

    def pixels(self):
        return bytes([self.drawn]) * 4

    def close(self):
        self.calls.append(("close",))

    @property
    def brightness(self):
        return None

    @brightness.setter
    def brightness(self, level):
        self.calls.append(("brightness", level))


def host(monkeypatch, argv, on_frame=None):
    """A run of `argv` with a fake renderer -> (shim, handle, open kwargs)."""
    shim = load()
    fake = Fake(on_frame)
    opened = {}

    def open_scene(scene, **kw):
        opened.update(kw, scene=scene)
        return fake

    from ghee import _native

    module = types.ModuleType("ghee")
    module.open = open_scene
    # the native half holds the colour names
    module._ext = lambda: _native
    monkeypatch.delenv(shim.WIRE, raising=False)
    monkeypatch.setitem(sys.modules, "ghee", module)
    monkeypatch.setattr(sys, "argv", ["feeder.py", *argv])
    return shim, fake, opened


def test_the_loop_runs_the_hooks_then_the_feeds_then_the_frame(monkeypatch):
    shim, fake, _ = host(monkeypatch, ["--display", "kms", "--frames", "3"])
    seen = []

    @shim.tick
    def hook(t):
        seen.append((t.frame, t.time, t.dt, t.params))
        shim.set_param("gain", 0.5)
        if t.frame == 1:
            shim.skip()

    @shim.feed("v")
    def v(t):
        return None if t.frame == 2 else [float(t.frame)] * 4

    shim.run(scene_of(shim))
    # the tick is the handle's, in its wire shape, with the string params
    # that changed
    assert seen == [
        (0, 0.0, 0.5, {"words": "hi"}),
        (1, 0.5, 0.5, {}),
        (2, 1.0, 0.5, {"words": "ho"}),
    ]
    gain = ("set_param", "gain", 0.5, None)
    assert fake.calls == [
        gain,
        ("feed", "v", [0.0] * 4),
        ("frame",),
        gain,
        ("feed", "v", [1.0] * 4),
        ("skip",),
        gain,
        ("frame",),
        ("close",),
    ]


@pytest.mark.parametrize("call,error", REFUSED)
def test_a_param_the_scene_cannot_take_raises_as_on_the_wire(call, error, monkeypatch):
    shim, fake, _ = host(monkeypatch, ["--display", "kms", "--frames", "1"])

    @shim.tick
    def hook(t):
        shim.set_param(*call)

    with pytest.raises(ValueError, match=re.escape(error)):
        shim.run(scene_of(shim))
    assert not [c for c in fake.calls if c[0] == "set_param"]


def test_a_layer_set_reaches_the_handle(monkeypatch):
    shim, fake, _ = host(monkeypatch, ["--display", "kms", "--frames", "1"])

    @shim.tick
    def hook(t):
        shim.layers.title.x = 10
        shim.layers["title"]["glow"]["color"] = "#ffffff"

    shim.run(scene_of(shim))
    assert fake.calls[:2] == [
        ("set_layer", "title", "x", 10, None),
        ("set_layer", "title", "color", "#ffffff", "glow"),
    ]


def test_the_host_flags_reach_the_bindings(monkeypatch):
    shim, _, opened = host(
        monkeypatch,
        [
            "--display",
            "kms",
            "--size",
            "64x48",
            "--fps",
            "30",
            "--serve",
            "--port",
            "8099",
            "--host",
            "0.0.0.0",
            "--frames",
            "1",
            "--timestep",
            "0.25",
            "--rotate",
            "270",
            "--dri",
            "/dev/dri/card1",
        ],
    )
    shim.run(scene_of(shim))
    assert opened["scene"] == scene_of(shim).to_dict()
    assert opened["display"] == "kms"
    assert (opened["width"], opened["height"]) == (64, 48)
    assert (opened["fps"], opened["serve"], opened["timestep"]) == (
        30.0,
        ("0.0.0.0", 8099),
        0.25,
    )
    assert (opened["rotate"], opened["dri"]) == (270, "/dev/dri/card1")


def test_brightness_sets_the_backlight_of_the_kms_display(monkeypatch):
    shim, fake, opened = host(
        monkeypatch, ["--display", "kms", "--brightness", "0.25", "--frames", "1"]
    )
    shim.run(scene_of(shim))
    assert opened["display"] == "kms"
    assert fake.calls[0] == ("brightness", 0.25)


def test_brightness_with_no_kms_display_stops_the_run_as_ghee_render_stops(
    monkeypatch, capsys
):
    shim, fake, opened = host(monkeypatch, ["-o", "f.png", "--brightness", "0.5"])
    with pytest.raises(SystemExit) as stopped:
        shim.run(scene_of(shim))
    assert stopped.value.code == 1
    assert capsys.readouterr().err == "feeder.py: --brightness needs --display kms\n"
    assert not opened and not fake.calls


@pytest.mark.parametrize(
    "argv,opts",
    [
        (["--display", "kms", "--rgb565"], {"rgb565": True}),
        (["--rgb565"], {"rgb565": True}),
        (["--display", "kms"], None),
        (["-o", "f.png", "--rgb565"], None),
    ],
)
def test_rgb565_scans_the_display_out_in_16_bit_color_as_ghee_render_does(
    argv, opts, monkeypatch
):
    shim, _, opened = host(monkeypatch, [*argv, "--frames", "1"])
    shim.run(scene_of(shim))
    assert opened["opts"] == opts


@pytest.mark.parametrize(
    "argv,error",
    [
        (["--rotate", "45"], "argument --rotate: '45' is not 0, 90, 180 or 270"),
        (
            ["--brightness", "2"],
            "argument --brightness: '2' is not a level from 0 to 1",
        ),
    ],
)
def test_rotate_and_brightness_take_the_values_of_ghee_render(
    argv, error, monkeypatch, capsys
):
    shim, _, _ = host(monkeypatch, argv)
    with pytest.raises(SystemExit):
        shim.args([])
    assert error in capsys.readouterr().err


def test_a_share_of_the_render_size_turns_with_a_quarter_rotate(monkeypatch):
    shim, _, _ = host(monkeypatch, ["--size", "128x32", "--rotate", "90"])
    got = shim.args([shim.Arg("bar", "size", default="half")])
    assert got.bar == (16, 64)


def connected(monkeypatch, mode):
    """A connected display of `mode` on every card, as ghee's probe finds it,
    -> the cards that the probe was asked about."""
    from ghee import _native

    asked = []

    def probe(dri=None):
        asked.append(dri)
        return mode

    monkeypatch.setattr(_native, "_display_size", probe)
    return asked


def test_a_share_of_the_render_size_is_a_share_of_the_display(monkeypatch):
    asked = connected(monkeypatch, (128, 32))
    shim, _, _ = host(monkeypatch, ["--display", "--dri", "/dev/dri/card1"])
    got = shim.args([shim.Arg("bar", "size", default="half")])
    assert got.bar == (64, 16)
    assert asked == ["/dev/dri/card1"]


@pytest.mark.parametrize(
    "argv,size",
    [
        (["-o", "f.png"], (320, 240)),
        (["--serve"], (320, 240)),
        (["--size", "128x32"], (128, 32)),
        (["--size", "128x32", "--rotate", "270"], (32, 128)),
        ([], (128, 32)),
        (["--display"], (128, 32)),
        (["--display=kms", "--rotate", "90"], (32, 128)),
    ],
)
def test_a_direct_run_renders_at_its_size_turned_by_its_rotate(argv, size, monkeypatch):
    connected(monkeypatch, (128, 32))
    shim, _, _ = host(monkeypatch, argv)
    monkeypatch.delenv(shim.SIZE, raising=False)
    shim.args([])
    assert shim.render_size() == size


@pytest.mark.parametrize(
    "argv,display,serve",
    [
        ([], "kms", None),
        (["--display"], "kms", None),
        (["-o", "f.png"], None, None),
        (["--serve"], None, ("127.0.0.1", 8090)),
        (["-o", "-", "--display", "kms"], "kms", None),
    ],
)
def test_a_run_with_no_output_goes_to_the_display(argv, display, serve, monkeypatch):
    shim, _, opened = host(monkeypatch, [*argv, "--frames", "1"])
    monkeypatch.setattr(sys, "stdout", Out())
    shim.run(scene_of(shim))
    assert (opened["display"], opened["serve"]) == (display, serve)
    assert (opened["width"], opened["height"]) == (None, None)


class Out:
    """sys.stdout as run() writes raw frames to it."""

    def __init__(self):
        self.buffer = io.BytesIO()


@pytest.mark.parametrize(
    "target,saved,piped,dest",
    [
        ("f.####.png", [("save", "f.####.png")] * 3, b"", "file"),
        ("-", [], bytes([1] * 4 + [2] * 4 + [3] * 4), "pipe"),
    ],
)
def test_o_writes_every_frame_as_ghee_render_does(
    target, saved, piped, dest, monkeypatch, capsys
):
    """A file or a pipe takes the skipped ticks too, and --stats sums them up."""
    shim, fake, _ = host(monkeypatch, ["-o", target, "--frames", "3", "--stats"])
    out = Out()
    monkeypatch.setattr(sys, "stdout", out)

    @shim.tick
    def hook(t):
        shim.skip()

    shim.run(scene_of(shim))
    assert [c for c in fake.calls if c[0] in ("save", "skip")] == saved
    assert out.buffer.getvalue() == piped
    err = capsys.readouterr().err
    assert re.fullmatch(
        r"     done  stats: 3 frame\(s\) in \d+\.\d\ds, \d+\.\d\d ms/frame "
        rf"\(min \d+\.\d\d, max \d+\.\d\d\), \d+\.\d fps -> {dest}\n",
        err,
    ), err


def test_o_takes_an_image_pattern_or_a_pipe(monkeypatch, capsys):
    shim, _, _ = host(monkeypatch, ["-o", "kms"])
    with pytest.raises(SystemExit):
        shim.args([])
    assert (
        "argument -o/--output: wants an image pattern such as frame.####.png, or "
        "'-', got 'kms'; a display takes --display" in capsys.readouterr().err
    )


def test_a_scene_takes_its_own_args_beside_the_host_flags(monkeypatch):
    shim, _, opened = host(
        monkeypatch, ["--bins", "8", "--display", "kms", "--frames", "1"]
    )
    got = shim.args([shim.Arg("bins", "int", default=4)])
    assert got.bins == 8
    shim.run(scene_of(shim))
    assert opened["display"] == "kms"


def test_a_wire_run_refuses_the_host_flags(monkeypatch):
    """ghee owns the render under the wire, so the scene takes its args alone."""
    shim, _, _ = host(monkeypatch, ["--display", "kms"])
    monkeypatch.setenv(shim.WIRE, "1")
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    with pytest.raises(SystemExit):
        shim.args([shim.Arg("bins", "int", default=4)])


def test_a_feed_the_scene_never_declared_stops_before_the_render(monkeypatch):
    shim, fake, opened = host(monkeypatch, ["--display", "kms", "--frames", "1"])

    @shim.feed("nope")
    def nope(t):
        return [0.0]

    with pytest.raises(ValueError, match="feed 'nope' is not an array"):
        shim.run(scene_of(shim))
    assert not opened and not fake.calls


def test_many_frames_into_one_file_stop_the_run_as_ghee_render_stops(
    monkeypatch, capsys
):
    shim, fake, opened = host(monkeypatch, ["-o", "one.png", "--frames", "4"])
    with pytest.raises(SystemExit) as stopped:
        shim.run(scene_of(shim))
    assert stopped.value.code == 1
    assert capsys.readouterr().err == (
        "feeder.py: multiple frames into one filename; number them with a # run "
        "(frame.####.png)\n"
    )
    assert not opened and not fake.calls


@pytest.mark.parametrize("error", [ValueError, RuntimeError])
def test_a_run_that_ghee_refuses_stops_with_its_one_line(error, monkeypatch, capsys):
    """A scene or a display that ghee refuses ends the run with ghee's line,
    and an exception of the script's own keeps its traceback."""
    shim, _, _ = host(monkeypatch, ["--frames", "1"])

    def refuse(scene, **kw):
        raise error("no connected display to size from; pass width and height")

    sys.modules["ghee"].open = refuse
    with pytest.raises(SystemExit) as stopped:
        shim.run(scene_of(shim))
    assert stopped.value.code == 1
    assert capsys.readouterr().err == (
        "feeder.py: no connected display to size from; pass width and height\n"
    )


def test_an_exception_of_the_script_keeps_its_traceback(monkeypatch):
    shim, _, _ = host(monkeypatch, ["--display", "kms", "--frames", "1"])

    @shim.tick
    def hook(t):
        raise KeyError("own")

    with pytest.raises(KeyError, match="own"):
        shim.run(scene_of(shim))


@pytest.mark.parametrize(
    "argv,drawn",
    [
        (["-o", "f.####.png"], 1),
        (["-o", "-"], 1),
        (["-o", "f.####.png", "--frames", "2"], 2),
        (["-o", "f.####.png", "--frames", "-1"], 0),
        (["-o", "f.####.png", "--frames", "0"], 4),
        (["-o", "f.####.png", "--display", "kms"], 4),
        (["-o", "f.####.png", "--serve"], 4),
        ([], 4),
    ],
)
def test_a_run_takes_the_frame_count_of_ghee_render(argv, drawn, monkeypatch):
    """One frame for an image or a pipe, and a display or the console runs
    until a signal, here a SIGTERM at the fourth frame."""

    def on_frame(n):
        if n == 4:
            signal.raise_signal(signal.SIGTERM)

    shim, fake, _ = host(monkeypatch, argv, on_frame)
    monkeypatch.setattr(sys, "stdout", Out())
    shim.run(scene_of(shim))
    assert fake.drawn == drawn


def test_an_endless_run_ends_on_a_sigterm(monkeypatch):
    """--frames 0 runs until a signal. The handlers it replaced come back."""

    def on_frame(drawn):
        if drawn == 2:
            signal.raise_signal(signal.SIGTERM)

    shim, fake, _ = host(monkeypatch, ["--display", "kms"], on_frame)
    held = signal.getsignal(signal.SIGTERM)
    shim.run(scene_of(shim))
    assert [c[0] for c in fake.calls] == ["frame", "frame", "close"]
    assert signal.getsignal(signal.SIGTERM) is held


SCENE = '''#!/usr/bin/env python3
"""A scene that renders itself: no ghee, no wire."""

import ghee

CODE = "void main(){ gl_FragColor = vec4(level_at(0), 0.0, 0.0, 1.0); }"


@ghee.tick
def hook(t):
    print("tick", t.frame, t.time, t.dt, t.params)


@ghee.feed("level")
def level(t):
    return [0.5] * 4


ghee.run(
    ghee.Scene(
        passes=[ghee.Pass("main", CODE, arrays=[ghee.ArrayInput("level", 4)])]
    )
)
'''


def test_a_scene_file_renders_itself(tmp_path):
    """The real render: a scene process, a GL device, three frames."""
    path = tmp_path / "feed.py"
    path.write_text(SCENE)
    ran = subprocess.run(
        [
            sys.executable,
            str(path),
            "-o",
            str(tmp_path / "f.####.png"),
            "--size",
            "16x12",
            "--frames",
            "3",
            "--timestep",
            "0.5",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert ran.returncode == 0, ran.stderr
    # frame 0 draws with the pinned step too
    assert ran.stdout.splitlines() == [
        "tick 0 0.0 0.5 {}",
        "tick 1 0.5 0.5 {}",
        "tick 2 1.0 0.5 {}",
    ]


TEXT_SCENE = '''#!/usr/bin/env python3
"""A scene that sets new words each tick, in a font beside it."""

import hashlib

import ghee
from ghee import _shim

CODE = (
    "void main(){ vec2 px = floor(uv * iResolution.xy) + 0.5;"
    " gl_FragColor = vec4(vec3(words_cover(px / 12.0, 12.0)), 1.0); }"
)


@ghee.tick
def hook(t):
    if t.frame:
        print(hashlib.sha256(_shim._HANDLE.pixels()).hexdigest())
    ghee.layers.words.text = ("ab", "cd")[t.frame % 2]


LAYER = ghee.Layer("words", "ab", visible=False, font_family="sans")
ghee.run(
    ghee.Scene(
        passes=[ghee.Pass("main", CODE, layers=[LAYER])],
        assets=[ghee.Asset("sans", "fonts/DejaVuSans-Latin.ttf")],
    )
)
'''


def test_a_direct_run_finds_its_font_beside_its_script(tmp_path):
    """Run from another directory, the font beside the script loads, and
    each frame draws the words that the tick before it set."""
    scene_dir = tmp_path / "scene"
    (scene_dir / "fonts").mkdir(parents=True)
    shutil.copy(REPO / "ghee/examples/fonts/DejaVuSans-Latin.ttf", scene_dir / "fonts")
    path = scene_dir / "words.py"
    path.write_text(TEXT_SCENE)
    ran = subprocess.run(
        [
            sys.executable,
            str(path),
            "-o",
            str(tmp_path / "f.####.png"),
            "--size",
            "48x16",
            "--frames",
            "4",
            "--timestep",
            "0",
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert ran.returncode == 0, ran.stderr
    ab, cd, ab_again = ran.stdout.split()
    assert ab != cd and ab == ab_again


LAYER_SCENE = '''#!/usr/bin/env python3
"""A scene whose layer puts its words at 4, 4."""

import ghee
from ghee import _shim


@ghee.tick
def hook(t):
    if t.frame:
        print(_shim._HANDLE.pixels().hex())


ghee.run(
    ghee.Scene(
        passes=[
            ghee.Pass(
                "main",
                "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                layers=[
                    ghee.Layer("words", "ab", x=4, y=4, font_size=16, font_family="sans")
                ],
            )
        ],
        assets=[ghee.Asset("sans", "fonts/DejaVuSans-Latin.ttf")],
    )
)
'''


def ink(frame, width):
    """{(x, y): red} of every pixel that the words cover, from a hex frame."""
    px = bytes.fromhex(frame)
    return {
        (i // 4 % width, i // 4 // width): px[i] for i in range(0, len(px), 4) if px[i]
    }


def test_a_scene_file_draws_its_words_where_its_layer_places_them(tmp_path):
    """The words land right of and below the layer's x and y."""
    scene_dir = tmp_path / "scene"
    (scene_dir / "fonts").mkdir(parents=True)
    shutil.copy(REPO / "ghee/examples/fonts/DejaVuSans-Latin.ttf", scene_dir / "fonts")
    path = scene_dir / "layer.py"
    path.write_text(LAYER_SCENE)
    ran = subprocess.run(
        [
            sys.executable,
            str(path),
            "-o",
            str(tmp_path / "f.####.png"),
            "--size",
            "64x32",
            "--frames",
            "2",
            "--timestep",
            "0",
        ],
        cwd=scene_dir,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    (first,) = (ink(f, 64) for f in ran.stdout.split())
    assert max(first.values()) == 255
    assert min(x for x, _ in first) >= 4 and min(y for _, y in first) >= 4
