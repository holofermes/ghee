"""The console over HTTP: what it refuses, what it only takes as POST, and the
render loop's failure modes. The handler is driven against a stub renderer, so
none of this needs a GPU."""

import http.client
import json
import os
import re
import socket
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from ghee.console import core, server, web_dir
from ghee.console.store import SceneStore

SCENE = {
    "version": 1,
    "passes": [{"name": "main", "code": "void main(){ gl_FragColor = vec4(1.0); }"}],
}


def _drain():
    while not server._q.empty():
        server._q.get_nowait()


@pytest.fixture(autouse=True)
def clean_module_state():
    yield
    server._stopping.clear()
    server._fatal = ""
    _drain()


@pytest.fixture
def scene_file(tmp_path):
    p = tmp_path / "demo.scene.json"
    p.write_text(json.dumps(SCENE), encoding="utf-8")
    return str(p)


class Client:
    """One request per call, so a refused (connection-closing) response cannot
    strand the next one."""

    def __init__(self, port):
        self.port = port
        self.origin = f"http://127.0.0.1:{port}"

    def __call__(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request(method, path, body=body, headers=dict(headers or {}))
            r = conn.getresponse()
            return r.status, r.read().decode("utf-8", "replace")
        finally:
            conn.close()

    def json(self, method, path, obj, headers=None):
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        return self(method, path, json.dumps(obj), h)


def _store(scene_file, read_only=True, names=("demo", "other")):
    st = SceneStore(os.path.dirname(scene_file), read_only=read_only)
    st.scenes = dict.fromkeys(names, scene_file)
    return st


@pytest.fixture
def console(monkeypatch, scene_file):
    monkeypatch.setattr(server, "WEB", web_dir())
    monkeypatch.setattr(server, "STORE", _store(scene_file))
    monkeypatch.setattr(server, "API", server._build_api())
    monkeypatch.setitem(server._stat, "active", "demo")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield Client(httpd.server_address[1])
    httpd.shutdown()
    httpd.server_close()


# Origin (CSRF)


def test_origin_helper():
    assert core.origin_allowed("", "127.0.0.1:8090")
    assert core.origin_allowed("http://127.0.0.1:8090", "127.0.0.1:8090")
    assert not core.origin_allowed("http://evil.example", "127.0.0.1:8090")
    assert not core.origin_allowed("null", "127.0.0.1:8090")


def test_cross_origin_post_is_refused(console):
    code, body = console.json(
        "POST", "/scene", SCENE, {"Origin": "http://evil.example"}
    )
    assert code == 403
    assert "cross-origin" in body
    assert server._q.empty()


def test_same_origin_post_is_accepted(console):
    code, _ = console.json("POST", "/scene", SCENE, {"Origin": console.origin})
    assert code == 200


def test_post_without_origin_is_accepted(console):
    code, _ = console.json("POST", "/scene", SCENE)
    assert code == 200


def test_cross_origin_set_is_refused(console):
    code, _ = console(
        "POST",
        "/set?pass=0&name=gain&value=0.5",
        headers={"Origin": "http://evil.example"},
    )
    assert code == 403
    assert server._q.empty()


# state-changing endpoints are POST only


@pytest.mark.parametrize(
    "path",
    ["/set?pass=0&name=gain&value=0.5", "/switch?name=other", "/next", "/prev"],
)
def test_state_changing_get_is_405(console, path):
    code, body = console("GET", path)
    assert code == 405
    assert "POST" in body
    assert server._q.empty()


def test_set_post_queues_the_param(console):
    code, body = console("POST", "/set?pass=0&name=gain&value=0.5")
    assert (code, body) == (200, "ok")
    assert server._q.get_nowait() == ("param", 0, "gain", 0.5)


def test_set_post_queues_a_scene_scoped_param(console):
    code, _ = console("POST", "/set?pass=scene&name=tint&type=color&value=1,0,0.5")
    assert code == 200
    assert server._q.get_nowait() == ("scene_param", "tint", [1.0, 0.0, 0.5])


def test_set_post_rejects_a_malformed_value(console):
    code, body = console("POST", "/set?pass=0&name=gain&type=color&value=1,0")
    assert code == 400
    assert "bad /set" in body


def test_next_post_switches(console):
    code, body = console("POST", "/next")
    assert code == 200
    assert json.loads(body)["active"] == "other"
    assert server._q.get_nowait() == ("switch", "other")


def test_switch_post_switches(console):
    code, body = console("POST", "/switch?name=other")
    assert code == 200
    assert json.loads(body)["active"] == "other"
    assert server._q.get_nowait() == ("switch", "other")


def test_switch_post_unknown_scene_is_404(console):
    code, _ = console("POST", "/switch?name=nope")
    assert code == 404
    assert server._q.empty()


# bodies


def test_oversized_body_is_413(console):
    code, body = console(
        "POST",
        "/scene",
        b"{}",
        {
            "Content-Type": "application/json",
            "Content-Length": str(core.MAX_BODY + 1),
        },
    )
    assert code == 413
    assert str(core.MAX_BODY) in body


def test_bad_content_length_is_400(console):
    code, _ = console(
        "POST",
        "/scene",
        b"{}",
        {"Content-Type": "application/json", "Content-Length": "not-a-number"},
    )
    assert code == 400


def test_scene_needs_a_json_content_type(console):
    code, body = console("POST", "/scene", json.dumps(SCENE))
    assert code == 415
    assert "application/json" in body


def test_save_needs_a_json_content_type(console):
    code, _ = console("POST", "/save", json.dumps({"name": "demo", "scene": SCENE}))
    assert code == 415


def test_posted_scene_is_validated_before_it_reaches_the_render_thread(console):
    code, body = console.json("POST", "/scene", {"passes": "not-a-list"})
    assert code == 400
    assert "passes" in body
    assert server._q.empty()


def test_posted_scene_arrives_as_a_scene_object(console):
    from ghee import Scene

    code, _ = console.json("POST", "/scene", SCENE)
    assert code == 200
    kind, obj = server._q.get_nowait()
    assert kind == "scene_obj"
    assert isinstance(obj, Scene)
    assert obj.passes[0].name == "main"


# static files


def test_serves_the_console_page(console):
    code, body = console("GET", "/")
    assert code == 200
    assert "<title>ghee</title>" in body


def test_traversal_out_of_the_web_dir_is_404(console):
    code, _ = console("GET", "/../server.py")
    assert code == 404


def test_a_sibling_directory_sharing_the_prefix_is_404(monkeypatch, console, tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("ok", encoding="utf-8")
    sibling = tmp_path / "webXYZ"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("secret", encoding="utf-8")
    monkeypatch.setattr(server.API, "web", str(web))
    code, _ = console("GET", "/../webXYZ/secret.txt")
    assert code == 404


# render loop failure modes


class StubRenderer:
    width, height, components = 64, 48, 4
    error_line = None
    error_pass = None

    def __init__(self, boom=False):
        self.boom = boom
        self.frames = 0
        self.reads = []

    def setup(self, scene):
        return ""

    def display_vrefresh(self):
        return 0

    def draw(self, provider=None):
        self.frames += 1
        self.reads.append(None)
        if self.boom:
            raise RuntimeError("boom")

    def render(self, provider=None, out=None, out_size=None):
        self.frames += 1
        self.reads.append(out_size)
        if self.boom:
            raise RuntimeError("boom")
        return out

    def set_param(self, *a):
        pass

    def set_scene_param(self, *a):
        pass

    def set_dt(self, value):
        pass


@pytest.fixture
def loop_state(monkeypatch, scene_file):
    monkeypatch.setattr(server, "STORE", _store(scene_file, names=("demo",)))
    monkeypatch.setattr(server, "W", 64)
    monkeypatch.setattr(server, "H", 48)
    monkeypatch.setitem(server._stat, "error", "")
    server._stopping.clear()
    yield
    server._stopping.set()


def _run_loop():
    t = threading.Thread(target=server.render_loop, daemon=True)
    t.start()
    return t


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_a_failing_frame_keeps_the_render_thread_alive(monkeypatch, loop_state):
    stub = StubRenderer(boom=True)
    monkeypatch.setattr(server, "_make_renderer", lambda: stub)
    t = _run_loop()
    assert _wait(lambda: "boom" in server._stat["error"])
    assert t.is_alive()
    server._stopping.set()
    server._q.put(("stop",))
    t.join(timeout=5)
    assert not t.is_alive()


def test_a_hardware_encoder_failing_mid_stream_hands_over_to_software(
    monkeypatch, loop_state
):
    closed = []

    class DyingEncoder:
        def __init__(self, renderer, w, h, fps=30, bitrate=0):
            pass

        def force_keyframe(self):
            pass

        def encode_last(self):
            raise RuntimeError("encoder stalled (no free input)")

        def close(self):
            closed.append(True)

    stub = StubRenderer()
    monkeypatch.setattr(server, "_make_renderer", lambda: stub)
    monkeypatch.setattr(server, "HardwareH264", DyingEncoder)
    monkeypatch.setattr(server, "_h264_hw_dead", False)
    monkeypatch.setattr(server, "_fanout", server.H264Fanout())
    server._fanout.add()
    t = _run_loop()
    assert _wait(lambda: server._h264_hw_dead)  # h264_sw_loop owns the stream now
    assert t.is_alive()
    _stop_loop(t)
    assert closed


def test_the_software_handover_lifts_when_the_last_viewer_leaves(
    monkeypatch, loop_state
):
    """A hardware failure belongs to the stream it happened in. Left standing it
    costs every later stream the readback the software encoder needs, for the
    life of the process."""
    monkeypatch.setattr(server, "_make_renderer", StubRenderer)
    monkeypatch.setattr(server, "_h264_hw_dead", True)
    monkeypatch.setattr(server, "_fanout", server.H264Fanout())
    c = server._fanout.add()
    t = _run_loop()
    assert _wait(lambda: server._h264_hw_dead)
    server._fanout.remove(c)
    assert _wait(lambda: not server._h264_hw_dead)
    _stop_loop(t)


def test_a_renderer_that_cannot_be_created_is_fatal(monkeypatch, loop_state):
    def boom():
        raise RuntimeError("no GL device for the console")

    monkeypatch.setattr(server, "_make_renderer", boom)
    server.render_loop()
    assert "no GL device" in server._fatal
    assert "no GL device" in server._stat["error"]
    assert server._stopping.is_set()


def _stop_loop(t):
    server._stopping.set()
    server._q.put(("stop",))
    t.join(timeout=5)


# the readback is sized by whoever consumes it: nobody, the MJPEG preview, or
# the software H264 encoder (which cannot recover detail from a preview frame)


def test_an_unwatched_loop_never_reads_a_frame_back(monkeypatch, loop_state):
    stub = StubRenderer()
    monkeypatch.setattr(server, "_make_renderer", lambda: stub)
    monkeypatch.setattr(server, "_nclients", 0)
    t = _run_loop()
    assert _wait(lambda: stub.frames > 3)
    _stop_loop(t)
    assert stub.reads and all(r is None for r in stub.reads)


def test_a_watched_loop_reads_back_at_the_preview_size(monkeypatch, loop_state):
    stub = StubRenderer()
    monkeypatch.setattr(server, "_make_renderer", lambda: stub)
    monkeypatch.setattr(server, "_nclients", 1)
    monkeypatch.setattr(server, "_preview_max", 32)
    monkeypatch.setattr(server, "_preview_fps", 0)
    t = _run_loop()
    assert _wait(lambda: stub.frames > 3)
    _stop_loop(t)
    assert stub.reads[0] == (32, 24)
    assert server._raw["rgb"].shape == (24, 32, 4)


def test_a_software_h264_viewer_reads_back_at_the_stream_size(monkeypatch, loop_state):
    stub = StubRenderer()
    monkeypatch.setattr(server, "_make_renderer", lambda: stub)
    monkeypatch.setattr(server, "_nclients", 0)
    monkeypatch.setattr(server, "_preview_max", 32)
    monkeypatch.setattr(server, "_h264_hw_dead", True)  # no /dev/video11: software
    monkeypatch.setattr(server, "_fanout", server.H264Fanout())
    server._fanout.add()
    t = _run_loop()
    assert _wait(lambda: any(r is not None for r in stub.reads))
    _stop_loop(t)
    reads = [r for r in stub.reads if r is not None]
    assert reads[0] == server._stream_size(server.W, server.H, server._stream_max)
    assert server._raw["rgb"].shape == (server.H, server.W, 4)


def test_the_readback_follows_a_software_h264_viewer_joining_mid_run(
    monkeypatch, loop_state
):
    stub = StubRenderer()
    monkeypatch.setattr(server, "_make_renderer", lambda: stub)
    monkeypatch.setattr(server, "_nclients", 1)
    monkeypatch.setattr(server, "_preview_max", 32)
    monkeypatch.setattr(server, "_preview_fps", 0)
    monkeypatch.setattr(server, "_h264_hw_dead", True)
    monkeypatch.setattr(server, "_fanout", server.H264Fanout())
    t = _run_loop()
    assert _wait(lambda: len([r for r in stub.reads if r == (32, 24)]) > 2)
    c = server._fanout.add()
    assert _wait(lambda: stub.reads[-1] == (64, 48))  # grown for the stream
    server._fanout.remove(c)
    assert _wait(lambda: stub.reads[-1] == (32, 24))  # and back down when it leaves
    _stop_loop(t)


def test_a_h264_viewer_that_left_is_dropped_with_nothing_to_send(monkeypatch, console):
    """A stream handler learns its viewer left from a write that fails, and no
    write comes while the encoder is handing out nothing. A viewer nobody
    notices leaving holds the console in the mode its stream asked for."""
    monkeypatch.setattr(server, "_fanout", server.H264Fanout())
    s = socket.create_connection(("127.0.0.1", console.port), timeout=5)
    try:
        s.sendall(b"GET /h264 HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        assert _wait(lambda: server._fanout.nclients == 1)
    finally:
        s.close()
    assert _wait(lambda: server._fanout.nclients == 0, timeout=10)


# smaller fixes


def test_client_count_is_atomic():
    start = server._nclients
    threads = [
        threading.Thread(target=lambda: [server._count_clients(1) for _ in range(200)])
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert server._nclients == start + 8 * 200
    for _ in range(8 * 200):
        server._count_clients(-1)
    assert server._nclients == start


def test_fanout_wires_each_frame_once():
    from ghee.h264 import wire_frame

    fanout = server.H264Fanout()
    a, b = fanout.add(), fanout.add()
    sc = b"\x00\x00\x00\x01"
    sps, pps, idr = b"\x67\x42\x00\x1e", b"\x68\xce\x06\xe2", b"\x65\x88\x80"
    fanout.push(sc + sps + sc + pps + sc + idr)
    a_cfg, a_key = a.q.get_nowait(), a.q.get_nowait()
    b_cfg, b_key = b.q.get_nowait(), b.q.get_nowait()
    assert a_cfg is b_cfg and a_key is b_key
    assert a_cfg == wire_frame(0, fanout.cfg)


# the web UI never builds markup out of scene data


def _index_src():
    with open(os.path.join(web_dir(), "index.html"), encoding="utf-8") as f:
        return f.read()


def test_no_innerhtml_is_built_from_interpolated_values():
    for m in re.finditer(r"innerHTML\s*=(.*?);", _index_src(), re.S):
        assert "${" not in m.group(1), m.group(1)


def test_the_ui_renders_every_param_type_the_model_accepts():
    from ghee.scene import _PARAM_TYPES

    src = _index_src()
    listed = re.search(r"paramNames = obj =>.*?\[(.*?)\]", src, re.S).group(1)
    shown = set(re.findall(r'"(\w+)"', listed))
    assert shown == set(_PARAM_TYPES), set(_PARAM_TYPES) ^ shown


def test_the_bool_row_is_a_checkbox_posting_true_or_false():
    src = _index_src()
    row = src[src.index('if (pm.type === "bool")') :]
    row = row[: row.index('if (pm.type === "color")')]
    assert 'inp.type = "checkbox"' in row
    assert 'send("bool", inp.checked ? "true" : "false")' in row
    assert "innerHTML" not in row  # built with createElement, like every other row


def test_the_ui_posts_every_state_changing_endpoint():
    src = _index_src()
    for path in ("/set?", "/next", "/prev", "/switch?"):
        for m in re.finditer(re.escape(path), src):
            line = src[src.rfind("\n", 0, m.start()) + 1 : src.find("\n", m.end())]
            assert "POST" in line or "jpost" in line, line


def test_a_restored_working_copy_takes_pass_sizes_from_the_running_scene():
    """A sized pass renders at the size the scene's build arguments gave it,
    and the editor has no control for it. The copy the browser restores on load
    is re-applied to whatever session is running, so carrying its own sizes
    would silently re-render a session at another one's geometry."""
    src = _index_src()
    merge = re.split(
        r"\n(?:async )?function ",
        src[src.index("function mergeNewParams(") :],
        maxsplit=1,
    )[0]
    assert "wp.size = clone(p.size)" in merge
    assert "delete wp.size" in merge
