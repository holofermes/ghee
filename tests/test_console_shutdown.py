"""What the console leaves behind when it goes down.

A console driving a panel or writing files owns those sinks: on the way out the
surface is blanked and powered off, a queued writer is drained, and every sink is
closed before run() returns. Signals go through a child process, the rest runs
run() on a thread against a stub renderer, so none of it needs a GPU.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from ghee.console import server
from ghee.output import Output
from ghee.stats import DEST_PANEL

W, H = 64, 48
SCENE = {
    "version": 1,
    "passes": [{"name": "main", "code": "void main(){ gl_FragColor = vec4(1.0); }"}],
}
HOOKS = ["turn_on", "clear", "turn_off", "close"]
# stop() ends serve_forever(), which polls at 0.5s, so a sink that cleans up
# faster than that cannot tell a run() that waits from one that does not
SLOW = 0.4
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# a console in its own process, so a real signal can end it
CHILD = (
    "import sys; sys.path.insert(0, %r); "
    "from tests.test_console_shutdown import child; child(*sys.argv[1:])"
)

_GLOBALS = (
    "STORE",
    "WEB",
    "W",
    "H",
    "API",
    "_lib_path",
    "_provider",
    "_hwinfo",
    "_display",
    "_rgb565",
    "_preview_max",
    "_preview_fps",
    "_stream_max",
    "_h264_bitrate",
    "_sinks",
    "_components",
    "_frames",
    "_timestep",
    "_dri",
    "_httpd",
    "_fatal",
)


class StubRenderer:
    """Enough renderer for the loop to run frames through: no GL."""

    width, height, components = W, H, 4
    error_line = error_pass = None

    def setup(self, scene):
        return ""

    def display_vrefresh(self):
        return 0

    def draw(self, provider=None):
        pass

    def render(self, provider=None, out=None, out_size=None):
        return out

    def set_param(self, *a):
        pass

    def set_scene_param(self, *a):
        pass

    def set_dt(self, value):
        pass


class Panel(Output):
    """A sink that records the lifecycle hooks it is given, in order. The delay
    stands in for a bus: a run() that does not wait cannot have seen the hooks
    that follow it."""

    dest = DEST_PANEL
    persists = True

    def __init__(self, delay=0.0, wedge=None):
        self.hooks = []
        self.frames = 0
        self.delay = delay
        self.wedge = wedge

    def write(self, frame):
        self.frames += 1

    def turn_on(self):
        self.hooks.append("turn_on")

    def clear(self):
        time.sleep(self.delay)
        self.hooks.append("clear")

    def turn_off(self):
        self.hooks.append("turn_off")

    def close(self):
        if self.wedge is not None:
            self.wedge.wait()
        time.sleep(self.delay)
        self.hooks.append("close")


class LogPanel(Output):
    """Panel for the child process: hooks land in a file the test reads."""

    dest = DEST_PANEL
    persists = True

    def __init__(self, path):
        self.log = open(path, "w", buffering=1, encoding="utf-8")

    def _note(self, hook):
        self.log.write(hook + "\n")

    def write(self, frame):
        pass

    def turn_on(self):
        self._note("turn_on")

    def clear(self):
        self._note("clear")

    def turn_off(self):
        self._note("turn_off")

    def close(self):
        self._note("close")


class SlowSave:
    """PIL through a delay, so ImageSequence's writer falls behind the render and
    its queue has to be drained at close()."""

    def __init__(self, delay):
        from PIL import Image

        self.Image = Image
        self.delay = delay

    def fromarray(self, buf, mode):
        img = self.Image.fromarray(buf, mode)
        return _SlowImage(img, self.delay)


class _SlowImage:
    def __init__(self, img, delay):
        self.img = img
        self.delay = delay

    def save(self, path, **opts):
        time.sleep(self.delay)
        self.img.save(path, **opts)


def child(log, scenes, port):
    """A console with a hook-logging sink, ended by a signal the way the CLI ends
    one (ghee.cli.main installs this SIGTERM handler; _serve swallows the exit)."""
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    server._make_renderer = StubRenderer
    try:
        server.run(
            scenes_dir=scenes, size=(W, H), port=int(port), sinks=[LogPanel(log)]
        )
    except (KeyboardInterrupt, SystemExit):
        pass


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _answering(port):
    """The console is inside serve_forever, not still starting up."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=1) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _wait(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def _scenes(tmp_path):
    (tmp_path / "one.scene.json").write_text(json.dumps(SCENE), encoding="utf-8")
    return str(tmp_path)


class Console:
    """server.run() on a thread, on an ephemeral port, against a stub renderer.

    snapshot is read the instant run() comes back, which is what the sinks being
    cleaned up by then means: a render thread still going on could finish the job
    a moment later and no assertion after the fact would know the difference."""

    def __init__(self, scenes, snapshot=None, **kw):
        self.error = None
        self.returned = threading.Event()
        self.snapshot = snapshot
        self.at_return = None
        self.kw = dict(scenes_dir=scenes, size=(W, H), port=0, **kw)
        self.thread = threading.Thread(target=self._main, daemon=True)

    def _main(self):
        try:
            server.run(**self.kw)
        except BaseException as e:  # noqa: BLE001  reported on the test's thread
            self.error = e
        finally:
            if self.snapshot is not None:
                self.at_return = self.snapshot()
            self.returned.set()

    def start(self):
        self.thread.start()
        # the socket is bound after run() clears _stopping, so a stop() past this
        # point is one run() sees
        assert _wait(
            lambda: server._httpd is not None and server._running["cur"] is not None
        ), "the console did not come up"
        return self

    def wait(self, timeout=20.0):
        assert self.returned.wait(timeout), "run() never returned"
        self.thread.join(timeout)
        assert self.error is None, self.error

    def stop(self, timeout=20.0):
        """Seconds run() took to come back after stop()."""
        t0 = time.monotonic()
        server.stop()
        self.wait(timeout)
        return time.monotonic() - t0


@pytest.fixture(autouse=True)
def module_state(monkeypatch):
    """run() writes this module's globals and starts its own threads, both of
    which the rest of the suite reads."""
    monkeypatch.setattr(server, "_make_renderer", StubRenderer)
    monkeypatch.setattr(server, "_nclients", 0)
    monkeypatch.setattr(server, "_fanout", server.H264Fanout())
    monkeypatch.setattr(server, "_httpd", None)
    monkeypatch.setitem(server._running, "cur", None)
    saved = {n: getattr(server, n) for n in _GLOBALS}
    before = set(threading.enumerate())
    yield
    server._stopping.set()
    end = time.monotonic() + 10.0
    while time.monotonic() < end:
        with server._raw_lock:
            server._raw_lock.notify_all()
        if not [t for t in threading.enumerate() if t not in before]:
            break
        time.sleep(0.02)
    server._stopping.clear()
    for name, value in saved.items():
        setattr(server, name, value)
    while not server._q.empty():
        server._q.get_nowait()


def test_a_served_sink_is_blanked_powered_down_and_closed(tmp_path):
    panel = Panel(delay=SLOW)
    console = Console(
        _scenes(tmp_path), sinks=[panel], snapshot=lambda: list(panel.hooks)
    ).start()
    assert _wait(lambda: panel.frames > 0), "no frame reached the sink"
    assert panel.hooks == ["turn_on"]
    console.stop()
    assert console.at_return == HOOKS


def test_a_console_with_no_sinks_comes_straight_back(tmp_path):
    console = Console(_scenes(tmp_path)).start()
    assert console.stop() < 2.0


def test_a_finite_render_still_cleans_up_and_ends_the_console(tmp_path):
    panel = Panel()
    console = Console(_scenes(tmp_path), sinks=[panel], frames=25).start()
    console.wait()
    assert panel.frames == 25
    assert panel.hooks == HOOKS


def test_a_queued_image_writer_is_drained_before_run_returns(tmp_path):
    pytest.importorskip("PIL.Image")
    from ghee.output import ImageSequence

    out = tmp_path / "out"
    out.mkdir()
    seq = ImageSequence(str(out / "f.####.png"), depth=16)
    seq._Image = SlowSave(0.05)  # a writer the render outruns, so a queue builds

    def written():
        return seq.index, len(list(out.glob("f.*.png")))

    console = Console(_scenes(tmp_path), sinks=[seq], snapshot=written).start()
    assert _wait(lambda: seq.index >= 8), "the writer took no frames"
    console.stop()
    submitted, on_disk = console.at_return
    assert submitted >= 8
    assert on_disk == submitted


def test_a_wedged_render_thread_does_not_hold_the_process(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(server, "RENDER_JOIN_TIMEOUT", 0.3)
    release = threading.Event()
    panel = Panel(wedge=release)
    try:
        console = Console(_scenes(tmp_path), sinks=[panel]).start()
        assert _wait(lambda: panel.frames > 0), "no frame reached the sink"
        took = console.stop()
    finally:
        release.set()
    assert 0.3 <= took < 3.0
    assert "render thread still going" in capsys.readouterr().err
    assert _wait(lambda: panel.hooks == HOOKS)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_signal_ends_the_console_with_its_sinks_cleaned(tmp_path, sig):
    log = tmp_path / "hooks.log"
    log.write_text("", encoding="utf-8")
    port = _free_port()
    p = subprocess.Popen(
        [sys.executable, "-c", CHILD % ROOT, str(log), _scenes(tmp_path), str(port)],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _wait(
            lambda: "turn_on" in log.read_text(encoding="utf-8") and _answering(port),
            timeout=30.0,
        ), "the console never came up"
        t0 = time.monotonic()
        p.send_signal(sig)
        err = p.communicate(timeout=30)[1]
        took = time.monotonic() - t0
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 0, err
    assert "Traceback" not in err, err
    assert log.read_text(encoding="utf-8").split() == HOOKS
    assert took < 5.0
