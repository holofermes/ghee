"""The console the host embeds: what it serves, and what a /set does."""

import json
import os
import shutil
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest
from conftest import RED, REPO, VECS, near, quarters

import ghee

LEVEL = [1.0] * 8

# without /dev/video11, /h264 runs on ffmpeg/libx264
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="software H264 takes ffmpeg on PATH"
)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(port, path, timeout=5):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
        return r.status, r.read()


def post(port, path, timeout=5):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=b"", method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def json_of(port, path):
    status, body = get(port, path)
    assert status == 200
    return json.loads(body)


def read_exact(r, n):
    buf = b""
    while len(buf) < n:
        chunk = r.read(n - len(buf))
        if not chunk:
            raise EOFError("the stream ended")
        buf += chunk
    return buf


def messages(r, count):
    """The wire messages of /h264: [u8 type][u32 len][payload]."""
    out = []
    while len(out) < count:
        head = read_exact(r, 5)
        out.append((head[0], read_exact(r, int.from_bytes(head[1:5], "big"))))
    return out


def watching(h, read, limit=30.0):
    """Run `read` against the console while the host renders. The stream
    advances with the host's frames, so something has to draw."""
    box = {}
    done = threading.Event()

    def watcher():
        try:
            box["got"] = read()
        except Exception as e:  # the reader's failure belongs to the test
            box["error"] = e
        finally:
            done.set()

    reader = threading.Thread(target=watcher)
    reader.start()
    end = time.monotonic() + limit
    while not done.is_set() and time.monotonic() < end:
        h.feed("level", LEVEL)
        h.frame()
    reader.join(5)
    if "error" in box:
        raise box["error"]
    assert "got" in box, f"the stream gave nothing in {limit}s"
    return box["got"]


@pytest.fixture
def served():
    port = free_port()
    with ghee.open(RED, width=32, height=32, stats=port) as h:
        h.feed("level", LEVEL)
        h.frame()
        yield h, port


def test_the_console_publishes_the_render(served):
    h, port = served
    stats = json_of(port, "/stats")
    assert stats["active"] == "scene"
    assert stats["error"] == ""
    assert stats["display"] == {"w": 32, "h": 32, "format": "rgba", "rotation": 0}
    assert stats["out"] == []
    assert stats["frames"] == 1
    assert stats["frame_ms"] > 0.0
    h.skip()
    assert json_of(port, "/stats")["skipped"] == 1


def test_the_console_serves_the_scene_it_renders(served):
    _, port = served
    assert json_of(port, "/scenes") == {"scenes": ["scene"], "active": "scene"}
    scene = json_of(port, "/scene")
    assert [p["name"] for p in scene["passes"]] == ["main"]
    assert scene["params"]["tint"]["type"] == "color"
    assert "dither" in json_of(port, "/filters")


INK = "void main(){ gl_FragColor = vec4(glyph_cover()); }"
PLACE = "vec2 glyph_place(vec2 em) { return em * 24.0; }"


def test_the_console_serves_a_shim_scene_as_the_shim_wrote_it(monkeypatch):
    """The renderer reads the shim's draws and texts, and writes them back the
    same: every key under its own name, every default left out on both sides."""
    monkeypatch.chdir(REPO / "ghee" / "examples")
    scene = ghee.Scene(
        params={
            "words": ghee.Param("words", "string", "hi"),
            "title": ghee.Param("title", "string", "ho"),
        },
        assets=[ghee.Asset("sans", "fonts/DejaVuSans-Latin.ttf")],
        texts=[
            ghee.Text(
                "words", "sans", align="right", atlas={"px_per_em": 24, "range": 4}
            ),
            ghee.Text("title", "sans", fill="slug"),
        ],
        passes=[
            ghee.Pass(
                "ink",
                size=(64, 32),
                mipmaps=True,
                draws=[
                    ghee.Draw(
                        "words", INK, PLACE, blend="add", pad=0.0625, alpha="keep"
                    ),
                    ghee.Draw("title", INK, PLACE, blend="replace"),
                ],
            ),
            ghee.Pass(
                "main",
                "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }",
                channels=[ghee.Channel("ink")],
                draws=[ghee.Draw("words", INK, PLACE)],
            ),
        ],
    ).to_dict()
    port = free_port()
    with ghee.open(scene, width=32, height=32, stats=port) as h:
        h.frame()
        served = json_of(port, "/scene")
    assert served == scene
    assert json.dumps(served) == json.dumps(scene)


def test_the_console_serves_its_own_page(served):
    _, port = served
    status, body = get(port, "/")
    assert status == 200
    assert b"<html" in body.lower()


def test_a_set_lands_on_the_next_frame(served):
    h, port = served
    assert post(port, "/set?pass=0&name=gain&value=0.25")[0] == 200
    h.feed("level", LEVEL)
    h.frame()
    assert abs(h.pixels()[0] - 64) <= 2
    # a scene param takes the scene scope
    assert post(port, "/set?pass=scene&name=tint&value=0,1,0")[0] == 200
    h.feed("level", LEVEL)
    h.frame()
    assert h.pixels()[1] > 0 and h.pixels()[0] == 0


def test_a_set_reaches_a_vec_or_color_uniform():
    port = free_port()
    with ghee.open(VECS, width=8, height=1, stats=port) as h:
        h.frame()
        for q in [
            "pass=scene&name=a&value=1,0",
            "pass=0&name=b&value=0,0.25,1",
            "pass=0&name=c&value=0.5,1,0,1",
            "pass=scene&name=tint&value=0,1,0.5",
        ]:
            assert post(port, "/set?" + q)[0] == 200
        h.frame()
        got = quarters(h)
        assert near(got, [[1, 0, 0], [0, 0.25, 1], [0.5, 1, 0], [0, 1, 0.5]]), got


def test_a_set_the_scene_refuses(served):
    _, port = served
    with pytest.raises(urllib.error.HTTPError) as raised:
        post(port, "/set?pass=scene&name=nope&value=1")
    assert raised.value.code == 400


def test_this_console_switches_no_scene_and_compiles_none(served):
    _, port = served
    for path in ["/next", "/switch?name=scene"]:
        with pytest.raises(urllib.error.HTTPError) as raised:
            post(port, path)
        assert raised.value.code == 403


def test_a_still_is_the_next_frame_the_host_draws(served):
    h, port = served
    answer = {}

    def ask():
        answer["got"] = get(port, "/frame.png", timeout=10)

    asking = threading.Thread(target=ask)
    asking.start()
    while asking.is_alive():
        h.feed("level", LEVEL)
        h.frame()
    asking.join()
    status, body = answer["got"]
    assert status == 200
    assert body[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_still_reaches_a_settled_host(served):
    h, port = served
    answer = {}

    def ask():
        answer["got"] = get(port, "/frame.png", timeout=10)

    asking = threading.Thread(target=ask)
    asking.start()
    while asking.is_alive():
        h.skip()
    asking.join()
    status, body = answer["got"]
    assert status == 200
    assert body[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_port_closes_with_the_handle():
    port = free_port()
    h = ghee.open(RED, width=8, height=8, stats=port)
    assert json_of(port, "/stats")["active"] == "scene"
    h.close()
    with pytest.raises(OSError):
        get(port, "/stats", timeout=2)


def test_stats_binds_where_it_is_told():
    port = free_port()
    with ghee.open(RED, width=8, height=8, stats=("127.0.0.1", port)) as h:
        h.feed("level", LEVEL)
        h.frame()
        assert json_of(port, "/stats")["active"] == "scene"


def test_stats_takes_a_port_or_a_pair():
    with pytest.raises(ValueError):
        ghee.open(RED, width=8, height=8, stats="8090")


@needs_ffmpeg
def test_h264_serves_what_webcodecs_decodes(served):
    h, port = served

    def read():
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/h264", timeout=20) as r:
            return messages(r, 3)

    msgs = watching(h, read)
    kinds = [t for t, _ in msgs]
    assert kinds[0] == 0  # the decoder config opens the stream
    cfg = json.loads(msgs[0][1])
    assert cfg["codec"].startswith("avc1.")
    assert (cfg["codedWidth"], cfg["codedHeight"]) == (32, 32)
    assert cfg["colorSpace"]["matrix"] == "smpte170m"
    assert kinds[1] == 1  # a key access unit, then whatever follows
    assert kinds[2] in (1, 2)
    au = msgs[1][1]
    nal = int.from_bytes(au[:4], "big")  # AVCC: a length in front of each NAL
    assert 0 < nal <= len(au) - 4


@pytest.mark.skipif(
    os.path.exists("/dev/video11"), reason="this board encodes H264 in hardware"
)
def test_a_stream_with_no_encoder_answers_503_with_the_reason(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")  # no ffmpeg, so no software lane
    port = free_port()
    with ghee.open(RED, width=32, height=32, stats=port) as h:
        h.feed("level", LEVEL)
        h.frame()

        def read():
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/h264", timeout=20)
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode()
            return 200, ""

        status, why = watching(h, read)
    assert status == 503
    assert "no H264 encoder" in why and "ffmpeg" in why, why


@needs_ffmpeg
def test_h264_reaches_a_settled_host(served):
    h, port = served
    box = {}
    done = threading.Event()

    def watcher():
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/h264", timeout=20
            ) as r:
                box["got"] = messages(r, 2)
        except Exception as e:
            box["error"] = e
        finally:
            done.set()

    reader = threading.Thread(target=watcher)
    reader.start()
    end = time.monotonic() + 30.0
    while not done.is_set() and time.monotonic() < end:
        h.skip()
    reader.join(5)
    if "error" in box:
        raise box["error"]
    assert [t for t, _ in box["got"]] == [0, 1]


@needs_ffmpeg
def test_stream_max_caps_the_preview():
    port = free_port()
    with ghee.open(RED, width=64, height=64, stats=port, stream_max=0.5) as h:
        h.feed("level", LEVEL)
        h.frame()

        def read():
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/h264", timeout=20
            ) as r:
                return messages(r, 1)

        msgs = watching(h, read)
        cfg = json.loads(msgs[0][1])
        assert (cfg["codedWidth"], cfg["codedHeight"]) == (32, 32)


def test_stream_max_rejects_what_it_cannot_read():
    for bad in (1.5, 0, -3, "half"):
        with pytest.raises(ValueError):
            ghee.open(RED, width=8, height=8, stream_max=bad)


@needs_ffmpeg
def test_a_second_viewer_gets_the_stream(served):
    h, port = served

    def read():
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/h264", timeout=20) as r:
            return messages(r, 2)

    for _ in range(2):
        assert [t for t, _ in watching(h, read)] == [0, 1]


@needs_ffmpeg
def test_the_stream_runs_only_while_a_client_watches(served):
    h, port = served
    assert "h264" not in json_of(port, "/stats")["out"]

    def read():
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/h264", timeout=20) as r:
            messages(r, 2)
            return json_of(port, "/stats")

    assert "h264" in watching(h, read)["out"]
    # the console drops the client on the write that follows the close
    end = time.monotonic() + 5
    while "h264" in json_of(port, "/stats")["out"] and time.monotonic() < end:
        h.feed("level", LEVEL)
        h.frame()
    assert "h264" not in json_of(port, "/stats")["out"]
