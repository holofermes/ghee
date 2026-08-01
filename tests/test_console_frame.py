"""GET /frame.png: one lossless PNG of the next frame.

The render loop runs for real against a stub renderer that paints what it was
last set to, so what comes back over HTTP is what the running scene holds. No
GPU: the stub fills the readback buffer itself.
"""

import http.client
import io
import json
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from ghee.console import server, web_dir
from ghee.console.store import SceneStore

W, H = 64, 48
SCENE = {
    "version": 1,
    "passes": [
        {
            "name": "main",
            "code": "void main(){ gl_FragColor = vec4(1.0); }",
            "params": {"level": {"type": "float", "value": 0.0, "min": 0, "max": 1}},
        }
    ],
}
STILL = """
from ghee import Pass, Scene, SceneController


class C(SceneController):
    def build(self, base):
        return Scene(
            passes=[Pass(name="main", code="void main(){ gl_FragColor = vec4(1.0); }")]
        )

    def dirty(self, frame, t):
        return False
"""


class StubRenderer:
    """Paints the frame red with whatever /set last landed and green with the
    frame number, so a fetched PNG says which frame it is and what the scene
    holds. reads records the size of every readback the loop asked for."""

    width, height, components = W, H, 4
    error_line = error_pass = None

    def __init__(self):
        self.level = 0
        self.frames = 0
        self.reads = []
        self.lock = threading.Lock()

    def setup(self, scene):
        return ""

    def display_vrefresh(self):
        return 0

    def draw(self, provider=None):
        self.frames += 1

    def render(self, provider=None, out=None, out_size=None):
        self.frames += 1
        with self.lock:
            self.reads.append(out_size)
        out[:, :, 0] = self.level
        out[:, :, 1] = self.frames % 256
        out[:, :, 2] = 0
        out[:, :, 3] = 255
        return out

    def set_param(self, pass_name, name, ptype, value):
        if name == "level":
            self.level = int(round(value * 255))

    def set_scene_param(self, *a):
        pass

    def set_dt(self, value):
        pass


class Client:
    """One request per call, so a refused (connection-closing) response cannot
    strand the next one."""

    def __init__(self, port):
        self.port = port

    def __call__(self, method, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request(method, path)
            r = conn.getresponse()
            return r.status, dict(r.getheaders()), r.read()
        finally:
            conn.close()


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def _serve(monkeypatch, scenes_dir):
    monkeypatch.setattr(server, "WEB", web_dir())
    monkeypatch.setattr(server, "STORE", SceneStore(str(scenes_dir), read_only=True))
    monkeypatch.setattr(server, "API", server._build_api())
    monkeypatch.setattr(server, "W", W)
    monkeypatch.setattr(server, "H", H)
    monkeypatch.setattr(server, "_nclients", 0)
    monkeypatch.setattr(server, "_display", False)
    monkeypatch.setattr(server, "_sinks", ())
    monkeypatch.setattr(server, "_fanout", server.H264Fanout())
    monkeypatch.setattr(server, "_shot", server.FrameShot())
    monkeypatch.setitem(server._stat, "error", "")
    monkeypatch.setitem(server._running, "cur", None)
    server._stopping.clear()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _shutdown(httpd, render=None):
    server._stopping.set()
    server._q.put(("stop",))
    if render is not None:
        render.join(timeout=5)
    httpd.shutdown()
    httpd.server_close()
    server._stopping.clear()
    while not server._q.empty():
        server._q.get_nowait()


@pytest.fixture
def console(monkeypatch, tmp_path):
    (tmp_path / "one.scene.json").write_text(json.dumps(SCENE), encoding="utf-8")
    stub = StubRenderer()
    httpd = _serve(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "_make_renderer", lambda: stub)
    render = threading.Thread(target=server.render_loop, daemon=True)
    render.start()
    try:
        assert _wait(lambda: server._running["cur"] is not None), "nothing loaded"
        yield Client(httpd.server_address[1]), stub
    finally:
        _shutdown(httpd, render)


@pytest.fixture
def still(monkeypatch, tmp_path):
    """A console on a scene whose controller reports clean, so the loop parks on
    the command queue instead of drawing."""
    (tmp_path / "still.py").write_text(STILL, encoding="utf-8")
    stub = StubRenderer()
    httpd = _serve(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "_make_renderer", lambda: stub)
    render = threading.Thread(target=server.render_loop, daemon=True)
    render.start()
    try:
        assert _wait(lambda: server._running["cur"] is not None), "nothing loaded"
        yield Client(httpd.server_address[1]), stub
    finally:
        _shutdown(httpd, render)


@pytest.fixture
def no_loop(monkeypatch, tmp_path):
    """The server with no render thread under it: nothing ever answers a shot."""
    (tmp_path / "one.scene.json").write_text(json.dumps(SCENE), encoding="utf-8")
    monkeypatch.setattr(server, "FRAME_TIMEOUT", 0.3)
    httpd = _serve(monkeypatch, tmp_path)
    try:
        yield Client(httpd.server_address[1])
    finally:
        _shutdown(httpd)


def _png(client, path="/frame.png"):
    status, headers, body = client("GET", path)
    assert status == 200, body
    assert body[:8] == b"\x89PNG\r\n\x1a\n"
    return headers, np.asarray(Image.open(io.BytesIO(body))), body


def _set(client, name, value):
    status, _, body = client("POST", f"/set?pass=0&name={name}&value={value}")
    assert (status, body) == (200, b"ok")


# the frame


def test_a_request_answers_with_a_png_at_the_render_size(console):
    client, _ = console
    headers, pixels, _ = _png(client)
    assert headers["Content-Type"] == "image/png"
    assert headers["Cache-Control"] == "no-store"
    assert pixels.shape == (H, W, 4)


def test_the_readback_is_the_whole_frame_not_the_preview_size(monkeypatch, console):
    client, stub = console
    monkeypatch.setattr(server, "_preview_max", 16)
    _png(client)
    assert stub.reads == [None]  # None = the render size, read back directly


def test_an_idle_scene_still_answers(still):
    client, stub = still
    assert _wait(lambda: stub.frames > 0), "nothing drew"
    parked = stub.frames
    time.sleep(0.2)
    assert stub.frames == parked, "the loop is not idle"
    headers, pixels, _ = _png(client)
    assert pixels.shape == (H, W, 4)
    assert stub.frames > parked


# what the frame shows


def test_a_set_made_before_the_request_is_in_the_frame(console):
    client, _ = console
    assert _png(client)[1][0, 0, 0] == 0
    _set(client, "level", 0.6)
    assert _png(client)[1][0, 0, 0] == round(0.6 * 255)


def test_every_request_gets_a_frame_drawn_after_it(console):
    client, _ = console
    _set(client, "level", 0.25)
    first = _png(client)[1][0, 0, 1]  # the frame number, in green
    _set(client, "level", 0.75)
    headers, pixels, _ = _png(client)
    assert pixels[0, 0, 0] == round(0.75 * 255)
    assert pixels[0, 0, 1] != first


# coalescing


def test_two_requests_in_flight_share_one_frame(console):
    client, stub = console
    ready = threading.Barrier(3)
    got = {}

    def fetch(i):
        ready.wait()
        got[i] = client("GET", "/frame.png")

    threads = [threading.Thread(target=fetch, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    ready.wait()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()
    assert [got[i][0] for i in range(2)] == [200, 200]
    assert got[0][2] == got[1][2]  # the same frame, byte for byte
    assert stub.reads == [None]  # one draw answered both


def test_the_generations_answer_one_draw_each():
    shot = server.FrameShot()
    frame = np.zeros((2, 2, 4), dtype=np.uint8)
    a, b = shot.request(), shot.request()
    shot.arm()
    assert shot.wants(0)
    shot.write(frame)
    assert not shot.wants(0)
    assert shot.wait(a, 0.0) is frame
    assert shot.wait(b, 0.0) is frame
    c = shot.request()  # asked after the frame was published: not answered by it
    assert shot.wait(c, 0.0) is None


# a loop that never answers


def test_a_wedged_loop_is_a_503(no_loop):
    status, headers, body = no_loop("GET", "/frame.png")
    assert status == 503
    assert headers["Content-Type"] == "text/plain"
    assert body.startswith(b"no frame:")
    assert len(body.splitlines()) == 1


def test_a_broken_scene_is_a_409_not_a_frame(monkeypatch, no_loop):
    monkeypatch.setitem(server._stat, "error", "render: boom\nsecond line")
    status, headers, body = no_loop("GET", "/frame.png")
    assert status == 409
    assert headers["Content-Type"] == "text/plain"
    assert body == b"scene error: render: boom"


# stats


def test_a_frame_request_is_not_a_destination(console):
    client, _ = console
    before = json.loads(client("GET", "/stats")[2])
    _png(client)
    after = json.loads(client("GET", "/stats")[2])
    assert before["out"] == after["out"] == []
    assert before["stream_fps"] == after["stream_fps"] == 0.0
