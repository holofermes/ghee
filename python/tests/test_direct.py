"""Direct mode: a scene with no ghee parent renders itself as `ghee render`
renders a program, and its hooks and feeds answer each tick in this process.

$GHEE_WIRE marks the process ghee started. Without it, run() hands the host
flags to ghee's render, which asks the shim for the scene and for each
tick's answer.
"""

import re
import shutil
import signal
import subprocess
import sys

import pytest
from conftest import REPO, pillow
from test_console import free_port
from test_shim import REFUSED, load, scene_of


def host(monkeypatch, argv):
    """A fresh shim, run as `feeder.py` with `argv`."""
    shim = load()
    monkeypatch.delenv(shim.WIRE, raising=False)
    monkeypatch.setattr(sys, "argv", ["feeder.py", *argv])
    return shim


def frames(tmp_path):
    """The frames that a run wrote into `tmp_path`, by name."""
    return sorted(p.name for p in tmp_path.glob("f*.png"))


def pixels(path):
    """The RGBA bytes of the image at `path`."""
    return pillow().open(path).convert("RGBA").tobytes()


def test_the_loop_runs_the_hooks_then_the_feeds_then_the_frame(tmp_path, monkeypatch):
    shim = host(
        monkeypatch,
        ["-o", str(tmp_path / "f.####.png"), "--frames", "3", "--timestep", "0.5"],
    )
    seen = []

    @shim.tick
    def hook(t):
        seen.append(("tick", t.frame, t.time, t.dt))
        shim.set_param("gain", 0.5)
        if t.frame == 1:
            shim.skip()

    @shim.feed("v")
    def v(t):
        seen.append(("feed", t.frame))
        return None if t.frame == 2 else [float(t.frame)] * 4

    shim.run(scene_of(shim))
    assert seen == [
        ("tick", 0, 0.0, 0.5),
        ("feed", 0),
        ("tick", 1, 0.5, 0.5),
        ("feed", 1),
        ("tick", 2, 1.0, 0.5),
        ("feed", 2),
    ]
    # a file takes the skipped tick too, as under `ghee render`
    assert frames(tmp_path) == ["f.0000.png", "f.0001.png", "f.0002.png"]


@pytest.mark.parametrize("call,error", REFUSED)
def test_a_param_the_scene_cannot_take_raises_as_on_the_wire(
    call, error, tmp_path, monkeypatch
):
    shim = host(monkeypatch, ["-o", str(tmp_path / "f.png")])

    @shim.tick
    def hook(t):
        shim.set_param(*call)

    with pytest.raises(ValueError, match=re.escape(error)):
        shim.run(scene_of(shim))


def test_the_host_flags_reach_the_render(tmp_path, monkeypatch):
    shim = host(
        monkeypatch,
        [
            "-o",
            str(tmp_path / "f.png"),
            "--size",
            "64x48",
            "--rotate",
            "270",
            "--timestep",
            "0.25",
        ],
    )
    seen = []

    @shim.tick
    def hook(t):
        seen.append((t.dt, shim.render_size()))

    shim.run(scene_of(shim))
    assert seen == [(0.25, (48, 64))]
    assert pillow().open(tmp_path / "f.png").size == (64, 48)


@pytest.mark.parametrize(
    "flag,error",
    [
        (["--brightness", "0.5"], "--brightness needs --display kms"),
        (["--drop"], "--drop needs --display kms"),
    ],
)
def test_a_display_flag_with_no_kms_display_stops_the_run_as_ghee_render_stops(
    flag, error, tmp_path, monkeypatch, capfd
):
    shim = host(monkeypatch, ["-o", str(tmp_path / "f.png"), *flag])
    with pytest.raises(SystemExit) as stopped:
        shim.run(scene_of(shim))
    assert stopped.value.code == 1
    assert capfd.readouterr().err == f"feeder.py: {error}\n"
    assert not frames(tmp_path)


@pytest.mark.parametrize(
    "argv,error",
    [
        (["--rotate", "45"], "argument --rotate: wants 0, 90, 180 or 270, got '45'"),
        (
            ["--brightness", "2"],
            "argument --brightness: wants a level from 0 to 1, got '2'",
        ),
    ],
)
def test_rotate_and_brightness_take_the_values_of_ghee_render(
    argv, error, monkeypatch, capsys
):
    shim = host(monkeypatch, argv)
    with pytest.raises(SystemExit) as stopped:
        shim.args([])
    assert stopped.value.code == 2
    assert capsys.readouterr().err.splitlines()[-1] == f"feeder.py: error: {error}"


def test_the_help_lists_the_flags_of_ghee_render_with_their_help(monkeypatch, capsys):
    shim = host(monkeypatch, ["--help"])
    with pytest.raises(SystemExit):
        shim.args([shim.Arg("bins", "int", 4, "how many bars")])
    out = capsys.readouterr().out
    assert "how many bars (default: 4)" in out
    from ghee import _native

    for f in _native._render_flags():
        assert f"--{f['long']}" in out
    assert "the frames go to the connected display" in out


def test_a_share_of_the_render_size_turns_with_a_quarter_rotate(monkeypatch):
    shim = host(monkeypatch, ["--size", "128x32", "--rotate", "90"])
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
    shim = host(monkeypatch, ["--display", "--dri", "/dev/dri/card1"])
    got = shim.args([shim.Arg("bar", "size", default="half")])
    assert got.bar == (64, 16)
    assert asked == ["/dev/dri/card1"]


@pytest.mark.parametrize(
    "argv,size",
    [
        (["-o", "f.png"], (320, 240)),
        (["-of.png"], (320, 240)),
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
    shim = host(monkeypatch, argv)
    monkeypatch.delenv(shim.SIZE, raising=False)
    shim.args([])
    assert shim.render_size() == size


@pytest.mark.parametrize("target", ["f.####.png", "-"])
def test_o_writes_every_frame_as_ghee_render_does(
    target, tmp_path, monkeypatch, capfdbinary
):
    """A file or a pipe takes the skipped ticks too, and --stats sums them up."""
    out = target if target == "-" else str(tmp_path / target)
    shim = host(monkeypatch, ["-o", out, "--frames", "3", "--size", "4x2", "--stats"])

    @shim.tick
    def hook(t):
        shim.skip()

    shim.run(scene_of(shim))
    out, err = capfdbinary.readouterr()
    if target == "-":
        assert len(out) == 3 * 4 * 2 * 4
    else:
        assert len(frames(tmp_path)) == 3
    dest = "pipe" if target == "-" else "file"
    assert re.search(
        r"     done  stats: 3 frame\(s\) in \d+\.\d\ds, \d+\.\d\d ms/frame "
        rf"\(min \d+\.\d\d, max \d+\.\d\d\), \d+\.\d fps -> {dest}\n",
        err.decode(),
    ), err


def test_o_takes_an_image_pattern_or_a_pipe(monkeypatch, capfd):
    shim = host(monkeypatch, ["-o", "kms"])
    with pytest.raises(SystemExit) as stopped:
        shim.run(scene_of(shim))
    assert stopped.value.code == 1
    assert capfd.readouterr().err == (
        "feeder.py: -o wants an image pattern such as frame.####.png, or '-', "
        "got 'kms'; a display takes --display\n"
    )


def test_a_scene_takes_its_own_args_beside_the_host_flags(tmp_path, monkeypatch):
    shim = host(monkeypatch, ["--bins", "8", "-o", str(tmp_path / "f.png")])
    got = shim.args([shim.Arg("bins", "int", default=4)])
    assert got.bins == 8
    shim.run(scene_of(shim))
    assert frames(tmp_path) == ["f.png"]


def test_a_wire_run_leaves_the_host_flags_alone(monkeypatch, capsys):
    """ghee owns the render under the wire, so the scene takes its args alone,
    and names each flag that it leaves."""
    shim = host(monkeypatch, ["--display", "kms", "--bins", "8"])
    monkeypatch.setenv(shim.WIRE, "1")
    got = shim.args([shim.Arg("bins", "int", default=4)])
    assert got.bins == 8
    assert capsys.readouterr().err == "feeder.py: ignoring --display kms\n"


def test_a_flag_that_no_scene_declares_is_named_and_the_run_goes_on(
    tmp_path, monkeypatch, capsys
):
    shim = host(monkeypatch, ["--bisn", "32", "-o", str(tmp_path / "f.png")])
    got = shim.args([shim.Arg("bins", "int", default=4)])
    assert got.bins == 4
    assert capsys.readouterr().err == "feeder.py: ignoring --bisn 32\n"
    shim.run(scene_of(shim))
    assert frames(tmp_path) == ["f.png"]


def test_a_feed_the_scene_never_declared_stops_before_the_render(tmp_path, monkeypatch):
    shim = host(monkeypatch, ["-o", str(tmp_path / "f.png")])

    @shim.feed("nope")
    def nope(t):
        return [0.0]

    with pytest.raises(ValueError, match="feed 'nope' is not an array"):
        shim.run(scene_of(shim))
    assert not frames(tmp_path)


def test_many_frames_into_one_file_stop_the_run_as_ghee_render_stops(
    tmp_path, monkeypatch, capfd
):
    shim = host(monkeypatch, ["-o", str(tmp_path / "f.png"), "--frames", "4"])
    with pytest.raises(SystemExit) as stopped:
        shim.run(scene_of(shim))
    assert stopped.value.code == 1
    assert capfd.readouterr().err == (
        "feeder.py: multiple frames into one filename; number them with a # run "
        "(frame.####.png)\n"
    )
    assert not frames(tmp_path)


def test_a_scene_that_ghee_refuses_stops_the_run_with_its_one_line(
    tmp_path, monkeypatch, capfd
):
    shim = host(monkeypatch, ["-o", str(tmp_path / "f.png")])
    scene = shim.Scene(passes=[shim.Pass("main", "void main(){ nope; }")])
    with pytest.raises(SystemExit) as stopped:
        shim.run(scene)
    assert stopped.value.code == 1
    err = capfd.readouterr().err
    assert err.startswith("feeder.py: compile error: ") and "nope" in err, err
    assert not frames(tmp_path)


def test_an_exception_of_the_script_keeps_its_traceback(tmp_path, monkeypatch, capfd):
    shim = host(monkeypatch, ["-o", str(tmp_path / "f.png")])

    @shim.tick
    def hook(t):
        raise KeyError("own")

    with pytest.raises(KeyError, match="own"):
        shim.run(scene_of(shim))
    assert capfd.readouterr().err == ""


@pytest.mark.parametrize(
    "argv,drawn",
    [
        ([], 1),
        (["--frames", "2"], 2),
        (["--frames", "-1"], 0),
        (["--frames", "0"], 4),
        (["--serve"], None),
    ],
)
def test_a_run_takes_the_frame_count_of_ghee_render(argv, drawn, tmp_path, monkeypatch):
    """One frame for an image, and an endless run, the console's too, runs
    until a signal, here a SIGTERM in the tick of the fourth frame. The
    console takes the signal on a thread of its own, so it may draw a frame
    more. The handlers that the run replaced come back, and the SIGTERM goes
    to the one that held it before the run."""
    if argv == ["--serve"]:
        argv = [*argv, "--port", str(free_port())]
    shim = host(monkeypatch, ["-o", str(tmp_path / "f.####.png"), *argv])

    @shim.tick
    def hook(t):
        if t.frame == 3:
            signal.raise_signal(signal.SIGTERM)

    got = []

    def held(sig, _frame):
        got.append(sig)

    before = signal.signal(signal.SIGTERM, held)
    try:
        shim.run(scene_of(shim))
        assert signal.getsignal(signal.SIGTERM) is held
    finally:
        signal.signal(signal.SIGTERM, before)
    if drawn is None:
        assert len(frames(tmp_path)) in (4, 5)
    else:
        assert len(frames(tmp_path)) == drawn
    assert got == ([signal.SIGTERM] if drawn in (4, None) else [])


SIGNALLED = """#!/usr/bin/env python3
import os
import signal

import ghee


@ghee.tick
def hook(t):
    if t.frame == 2:
        os.kill(os.getpid(), signal.{name})


ghee.run(ghee.Scene(passes=[ghee.Pass("main", "void main(){{ gl_FragColor = vec4(1.0); }}")]))
"""


@pytest.mark.parametrize("name", ["SIGTERM", "SIGINT"])
def test_a_signal_ends_a_scene_that_runs_itself_by_that_signal(name, tmp_path):
    """systemd counts a death by SIGTERM or SIGINT as a clean stop. The
    frames drawn before the signal are written first."""
    path = tmp_path / "signalled.py"
    path.write_text(SIGNALLED.format(name=name))
    ran = subprocess.run(
        [sys.executable, str(path), "-o", str(tmp_path / "f.####.png")]
        + ["--size", "8x8", "--frames", "0"],
        # pytest-xdist's workers ignore SIGINT, and a child keeps that
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
        capture_output=True,
        check=False,
        text=True,
    )
    assert ran.returncode == -getattr(signal, name), ran.stderr
    assert len(frames(tmp_path)) in (3, 4)


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

import ghee

CODE = (
    "void main(){ vec2 px = vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y);"
    " gl_FragColor = vec4(vec3(words_cover(px / 12.0, 12.0)), 1.0); }"
)


@ghee.tick
def hook(t):
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
    each frame draws the words that its tick set."""
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
    )
    assert ran.returncode == 0, ran.stderr
    ab, cd, ab_again = (pixels(tmp_path / f"f.{n:04d}.png") for n in range(3))
    assert ab != cd and ab == ab_again


LAYER_SCENE = '''#!/usr/bin/env python3
"""A scene whose layer puts its words at 4, 4."""

import ghee

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


def ink(px, width):
    """{(x, y): red} of every pixel that the words cover, from RGBA bytes."""
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
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    first = ink(pixels(tmp_path / "f.0000.png"), 64)
    assert max(first.values()) == 255
    assert min(x for x, _ in first) >= 4 and min(y for _, y in first) >= 4


def test_a_scene_file_run_by_path_finds_its_files_beside_it(tmp_path, monkeypatch):
    from test_shim import sampled

    sampled(tmp_path)
    out = tmp_path / "f.png"
    monkeypatch.chdir(tmp_path)
    host(monkeypatch, ["-o", str(out), "--size", "4x4", "--frames", "1"]).run(
        "sub/a.scene.json"
    )
    assert pixels(out)[:4] == bytes([255, 0, 0, 255])
