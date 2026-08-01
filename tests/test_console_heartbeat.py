"""Stream viewers that have to say they are still there.

A viewer costs the render loop a readback, an encoder and a frame rate, so the
console has to know when one leaves. A closed socket is caught the instant it
happens; a page frozen in a browser's back/forward cache never closes one and
keeps draining it, so only the heartbeat separates it from someone watching.

The handler runs over a real socket with no render loop under it: what is under
test is who the server counts as a viewer.
"""

import http.client
import os
import re
import socket
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from ghee.console import server, web_dir
from ghee.console.server import Heartbeat

SC = b"\x00\x00\x00\x01"
KEY_AU = SC + b"\x67\x42\x00\x1e" + SC + b"\x68\xce\x06\xe2" + SC + b"\x65\x88\x80"


def _wait(pred, timeout=8.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture(autouse=True)
def clean_module_state():
    yield
    server._stopping.clear()
    while not server._q.empty():
        server._q.get_nowait()


@pytest.fixture
def port():
    """The console's HTTP handler, alone: streams and /ping need no scene."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    server._stopping.set()  # release the stream handlers still in their loops
    with server._lock:
        server._lock.notify_all()
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def deadline(monkeypatch):
    """A deadline the tests can outlast, in the units the server measures."""
    monkeypatch.setattr(server._viewers, "deadline", 0.4)
    return 0.4


class Viewer:
    """A stream connection that never speaks again after its request, which is
    what both a watching page and a frozen one look like on the wire."""

    def __init__(self, port, path, drain=False):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.sock.sendall(
            f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode()
        )
        self.port = port
        self.read = 0
        self._stop = threading.Event()
        self.thread = None
        if drain:
            self.thread = threading.Thread(target=self._drain, daemon=True)
            self.thread.start()

    def _drain(self):
        while not self._stop.is_set():
            try:
                b = self.sock.recv(65536)
            except OSError:
                return
            if not b:
                return
            self.read += len(b)

    def ping(self, vid):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("POST", "/ping?vid=" + vid)
            r = conn.getresponse()
            r.read()
            return r.status
        finally:
            conn.close()

    def close(self):
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass
        if self.thread is not None:
            self.thread.join(timeout=2)


@pytest.fixture
def viewers():
    open_ones = []
    yield open_ones.append
    for v in open_ones:
        v.close()


# the registry


def test_a_connection_that_named_no_vid_never_expires():
    hb = Heartbeat(deadline=0.0)
    v = hb.join(None)
    time.sleep(0.01)
    assert hb.gone(v) is False


def test_a_named_connection_expires_once_its_pings_stop():
    hb = Heartbeat(deadline=0.05)
    v = hb.join("abc")
    assert hb.gone(v) is False
    time.sleep(0.08)
    assert hb.gone(v) is True


def test_a_ping_puts_a_named_connection_back_inside_the_deadline():
    hb = Heartbeat(deadline=0.05)
    v = hb.join("abc")
    time.sleep(0.08)
    assert hb.ping("abc") == 1
    assert hb.gone(v) is False


def test_every_connection_under_one_vid_is_kept_alive_together():
    # a page that falls back from h264 to MJPEG opens a second stream under the
    # id it already pings with
    hb = Heartbeat(deadline=0.05)
    a, b = hb.join("abc"), hb.join("abc")
    time.sleep(0.08)
    assert hb.ping("abc") == 2
    assert (hb.gone(a), hb.gone(b)) == (False, False)


def test_a_connection_that_left_is_not_pinged_for_any_more():
    hb = Heartbeat(deadline=0.05)
    a, b = hb.join("abc"), hb.join("abc")
    hb.leave(a)
    assert hb.ping("abc") == 1
    hb.leave(b)
    assert hb.ping("abc") == 0
    assert hb._by_vid == {}


def test_a_ping_for_an_id_with_no_stream_matches_nothing():
    hb = Heartbeat(deadline=0.05)
    assert hb.ping("nobody") == 0
    assert hb.ping(None) == 0


def test_the_deadline_leaves_room_for_lost_pings():
    assert server.VIEWER_DEADLINE >= 3 * server.PING_INTERVAL


# over HTTP


def test_an_h264_viewer_is_dropped_once_its_pings_stop(port, deadline, viewers):
    viewers(Viewer(port, "/h264?vid=frozen"))
    assert _wait(lambda: server._fanout.nclients == 1), "the viewer never counted"
    assert _wait(lambda: server._fanout.nclients == 0), "the viewer was never dropped"


def test_an_mjpeg_viewer_is_dropped_once_its_pings_stop(port, deadline, viewers):
    viewers(Viewer(port, "/stream?vid=frozen"))
    assert _wait(lambda: server._nclients == 1), "the viewer never counted"
    assert _wait(lambda: server._nclients == 0), "the viewer was never dropped"


def test_a_viewer_that_keeps_pinging_keeps_its_stream(port, deadline, viewers):
    v = Viewer(port, "/h264?vid=watching")
    viewers(v)
    assert _wait(lambda: server._fanout.nclients == 1)
    end = time.monotonic() + deadline * 4
    while time.monotonic() < end:
        assert v.ping("watching") == 200
        time.sleep(deadline / 4)
    assert server._fanout.nclients == 1


def test_a_stream_opened_without_a_vid_is_never_dropped(port, deadline, viewers):
    # curl, and any page older than the heartbeat
    viewers(Viewer(port, "/h264"))
    assert _wait(lambda: server._fanout.nclients == 1)
    time.sleep(deadline * 4)
    assert server._fanout.nclients == 1


def test_a_viewer_still_draining_frames_is_dropped_all_the_same(
    port, deadline, viewers
):
    """The frozen page: it consumes every byte the server sends and is gone all
    the same, so nothing about throughput may enter the decision."""
    v = Viewer(port, "/h264?vid=frozen", drain=True)
    viewers(v)
    assert _wait(lambda: server._fanout.nclients == 1)
    feeding = threading.Event()

    def feed():
        while not feeding.is_set():
            server._fanout.push(KEY_AU)
            time.sleep(0.02)

    t = threading.Thread(target=feed, daemon=True)
    t.start()
    try:
        assert _wait(lambda: server._fanout.nclients == 0), (
            "a viewer reading every frame was never dropped"
        )
    finally:
        feeding.set()
        t.join(timeout=2)
    assert v.read > 0, "the test viewer never actually read the stream"


def test_a_ping_for_an_unknown_viewer_is_still_answered(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("POST", "/ping?vid=nobody")
        r = conn.getresponse()
        assert (r.status, r.read()) == (200, b"ok")
    finally:
        conn.close()


def test_a_cross_origin_ping_is_refused_like_any_other_post(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("POST", "/ping?vid=x", headers={"Origin": "http://evil.example"})
        r = conn.getresponse()
        assert r.status == 403
        r.read()
    finally:
        conn.close()


# the page


def _index():
    with open(os.path.join(web_dir(), "index.html"), encoding="utf-8") as f:
        return f.read()


def _fn(src, head):
    i = src.index(head)
    return src[i : src.index("\n}", i)]


def test_the_page_opens_both_streams_under_its_viewer_id():
    src = _index()
    assert '"/h264?vid=" + VID' in src
    assert '"/stream?vid=" + VID' in src


def test_the_page_pings_faster_than_the_server_gives_up():
    src = _index()
    ms = int(re.search(r"const PING_MS = (\d+);", src).group(1))
    assert ms / 1000.0 <= server.PING_INTERVAL


def test_the_stream_preference_is_its_own_key():
    """A UI preference stored in the scene working copy comes back as scene
    state and is posted to the renderer, so the two never share a key."""
    src = _index()
    assert 'STREAM_KEY = "ghee:ui:stream"' in src
    assert 'KEY = n => "ghee:scene:" + n' in src
    written = re.findall(r"localStorage\.setItem\(KEY\([^)]*\),\s*([^)]+\))", src)
    assert written and set(written) == {"JSON.stringify(work)"}
    assert re.search(r"localStorage\.setItem\(STREAM_KEY,\s*streamMuted \?", src)


def test_a_page_going_into_the_back_forward_cache_stops_its_stream():
    src = _index()
    assert (
        'addEventListener("pagehide", e => { if (e.persisted) stopStream(); })' in src
    )
    assert (
        'addEventListener("pageshow", e => { if (e.persisted) startStream(); })' in src
    )
    assert 'document.addEventListener("freeze", stopStream)' in src


def test_a_backgrounded_tab_is_a_viewer_like_any_other():
    # a hidden but live tab keeps streaming: only a frozen or closed one stops
    src = _index()
    assert 'addEventListener("visibilitychange"' not in src
    assert "document.hidden" not in src


def test_the_stream_flag_is_not_the_muted_pass_count():
    # refreshInfo() has a `muted` of its own (how many passes are muted), and a
    # stream flag by that name is shadowed inside it
    info = _fn(_index(), "function refreshInfo()")
    assert "const muted = " in info
    assert 'streamMuted ? "muted"' in info


def test_the_mute_control_stops_the_stream_rather_than_hiding_it():
    stop = _fn(_index(), "function stopStream()")
    assert "h264.stop()" in stop
    assert 'IMG.removeAttribute("src")' in stop
    assert "clearInterval(pingTimer)" in stop


def test_the_mute_control_is_reachable_while_it_is_muted():
    src = _index()
    assert 'id="muteBtn"' in src
    css = src[src.index("<style>") : src.index("</style>")]
    assert ".screen.muted .ovtools{opacity:1; visibility:visible}" in css
