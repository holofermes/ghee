"""Direct mode: a scene with no ghee parent hosts itself on the bindings.

$GHEE_WIRE marks the process ghee started. Without it, run() opens the scene,
drives the same hooks and feeds, and honours the same skip. A fake `ghee.open`
in sys.modules stands in for the renderer, bar the last test.
"""

import io
import signal
import subprocess
import sys
import types

import pytest
from test_shim import load, scene_of


class Fake:
    """The handle open() hands back. It records what the loop asks of it."""

    def __init__(self, on_frame=None):
        self.calls = []
        self.on_frame = on_frame
        self.drawn = 0

    def feed(self, name, values):
        self.calls.append(("feed", name, list(values)))

    def set_param(self, name, value, pass_=None):
        self.calls.append(("set_param", name, value, pass_))

    def frame(self):
        self.calls.append(("frame",))
        self.drawn += 1
        if self.on_frame:
            self.on_frame(self.drawn)
        return {"frame": self.drawn - 1, "time": 0.0, "dt": 0.0}

    def skip(self):
        self.calls.append(("skip",))
        return {"frame": self.drawn, "time": 0.0, "dt": 0.0}

    def close(self):
        self.calls.append(("close",))


def host(monkeypatch, argv, on_frame=None):
    """A run of `argv` with a fake renderer -> (shim, handle, open kwargs)."""
    shim = load()
    fake = Fake(on_frame)
    opened = {}

    def open_scene(scene, **kw):
        opened.update(kw, scene=scene)
        return fake

    module = types.ModuleType("ghee")
    module.open = open_scene
    monkeypatch.delenv(shim.WIRE, raising=False)
    monkeypatch.setitem(sys.modules, "ghee", module)
    monkeypatch.setattr(sys, "argv", ["feeder.py", *argv])
    return shim, fake, opened


def test_the_loop_runs_the_hooks_then_the_feeds_then_the_frame(monkeypatch):
    shim, fake, _ = host(
        monkeypatch, ["--output", "none", "--frames", "3", "--timestep", "0.5"]
    )
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
    # the tick keeps its wire shape, and params stays empty here
    assert seen == [(0, 0.0, 0.0, {}), (1, 0.5, 0.5, {}), (2, 1.0, 0.5, {})]
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


def test_the_host_flags_reach_the_bindings(monkeypatch):
    shim, _, opened = host(
        monkeypatch,
        [
            "--output",
            "st7789",
            "--size",
            "64x48",
            "--fps",
            "30",
            "--stats",
            "8090",
            "--frames",
            "1",
            "--timestep",
            "0.25",
        ],
    )
    shim.run(scene_of(shim))
    assert opened["scene"] == scene_of(shim).to_dict()
    assert opened["output"] == "st7789"
    assert (opened["width"], opened["height"]) == (64, 48)
    assert (opened["fps"], opened["stats"], opened["timestep"]) == (30.0, 8090, 0.25)


def test_the_defaults_are_the_display_at_its_own_size(monkeypatch):
    shim, _, opened = host(monkeypatch, ["--frames", "1"])
    shim.run(scene_of(shim))
    assert opened["output"] == "kms"
    assert (opened["width"], opened["height"]) == (None, None)
    assert (opened["fps"], opened["stats"], opened["timestep"]) == (None, None, None)
    # 'none' is the headless render, as open() spells it
    shim, _, opened = host(monkeypatch, ["--output", "none", "--frames", "1"])
    shim.run(scene_of(shim))
    assert opened["output"] is None


def test_a_scene_takes_its_own_args_beside_the_host_flags(monkeypatch):
    shim, _, opened = host(
        monkeypatch, ["--bins", "8", "--output", "none", "--frames", "1"]
    )
    got = shim.args([shim.Arg("bins", "int", default=4)])
    assert got.bins == 8
    shim.run(scene_of(shim))
    assert opened["output"] is None


def test_a_wire_run_refuses_the_host_flags(monkeypatch):
    """ghee owns the render under the wire, so the scene takes its args alone."""
    shim, _, _ = host(monkeypatch, ["--output", "none"])
    monkeypatch.setenv(shim.WIRE, "1")
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    with pytest.raises(SystemExit):
        shim.args([shim.Arg("bins", "int", default=4)])


def test_a_feed_the_scene_never_declared_stops_before_the_render(monkeypatch):
    shim, fake, opened = host(monkeypatch, ["--output", "none", "--frames", "1"])

    @shim.feed("nope")
    def nope(t):
        return [0.0]

    with pytest.raises(ValueError, match="feed 'nope' is not an array"):
        shim.run(scene_of(shim))
    assert not opened and not fake.calls


def test_an_endless_run_ends_on_a_sigterm(monkeypatch):
    """--frames 0 runs until a signal. The handlers it replaced come back."""

    def on_frame(drawn):
        if drawn == 2:
            signal.raise_signal(signal.SIGTERM)

    shim, fake, _ = host(monkeypatch, ["--output", "none"], on_frame)
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
            "--output",
            "none",
            "--size",
            "16x12",
            "--frames",
            "3",
            "--timestep",
            "0.5",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.splitlines() == [
        "tick 0 0.0 0.0 {}",
        "tick 1 0.5 0.5 {}",
        "tick 2 1.0 0.5 {}",
    ]
