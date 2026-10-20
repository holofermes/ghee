"""The console the host embeds: what it serves, and what a /set does."""

import concurrent.futures
import contextlib
import copy
import http.client
import inspect
import io
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest
from conftest import RED, REPO, VECS, near, pillow, quarters

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


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
        return r.status, r.read()


def post(port, path):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=b"", method="POST"
    )
    with urllib.request.urlopen(req) as r:
        return r.status, r.read()


def post_json(port, path, doc):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(doc).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return r.status, r.read()


def refused(port, doc, at, value, what):
    """POST /scene with `doc`, its key at the path `at` set to `value`, and
    assert the 403 that names `what`, a key that the program sets."""
    changed = copy.deepcopy(doc)
    node = changed
    for k in at[:-1]:
        node = node[k]
    node[at[-1]] = value
    with pytest.raises(urllib.error.HTTPError) as raised:
        post_json(port, "/scene", changed)
    assert (raised.value.code, raised.value.read().decode()) == (
        403,
        f"the scene's program sets {what}",
    )


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


def watching(h, read, step=None):
    """Run `read` against the console while the host renders, and raise what
    `read` raises. The stream advances with the host's frames, so something has
    to draw: `step`, else a fed frame."""

    def frame():
        h.feed("level", LEVEL)
        h.frame()

    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        got = pool.submit(read)
        while not got.done():
            (step or frame)()
        return got.result()


@pytest.fixture
def served():
    port = free_port()
    with ghee.open(RED, width=32, height=32, serve=port) as h:
        h.feed("level", LEVEL)
        h.frame()
        yield h, port


def test_the_console_lists_a_text_problem_apart_from_the_error(tmp_path):
    port = free_port()
    doc = {
        "version": 1,
        "passes": [
            {
                "name": "main",
                "fragment": "void main(){ gl_FragColor = vec4(1.0); }",
                "layers": [
                    {"name": "words", "text": "hi", "font_family": "Missing.ttf"}
                ],
            }
        ],
    }
    with ghee.open(doc, width=16, height=16, serve=port, base_dir=tmp_path) as h:
        h.frame()
        stats = json_of(port, "/stats")
    assert stats["error"] == ""
    assert [e.split(" (")[0] for e in stats["text_errors"]] == [
        "layer 'words' font_family 'Missing.ttf'"
    ]


def test_a_tick_hands_the_host_the_params_the_console_set():
    port = free_port()
    doc = {
        "version": 1,
        "params": {
            "mood": {"type": "string", "value": "calm"},
            "gain": {"type": "float", "value": 0.0},
        },
        "passes": [
            {
                "name": "main",
                "fragment": "void main(){ gl_FragColor = vec4(1.0); }",
                "params": {"tone": {"type": "float", "value": 0.0}},
                "layers": [{"name": "title", "text": "hi"}],
            },
            {
                "name": "glow",
                "fragment": "void main(){ gl_FragColor = vec4(1.0); }",
                "params": {"tone": {"type": "float", "value": 0.0}},
            },
        ],
    }
    with ghee.open(doc, width=8, height=8, serve=port) as h:
        first = {
            "mood": "calm",
            "gain": 0.0,
            "tone": 0.0,
            "glow.tone": 0.0,
            "title.text": "hi",
        }
        assert h.frame().params == first
        assert h.skip().params == {}
        post(port, "/set?name=mood&value=wild")
        post(port, "/set?name=gain&value=0.5")
        post(port, "/set?pass=glow&name=tone&value=0.75")
        post(port, "/set?layer=title&field=text&value=ho")
        assert h.frame().params == {
            "mood": "wild",
            "gain": 0.5,
            "glow.tone": 0.75,
            "title.text": "ho",
        }
        h.set_param("mood", "own")
        h.set_param("gain", 0.25)
        h.set_param("tone", 0.1, in_pass="glow")
        h.layers.title.text = "own"
        assert h.frame().params == {}


def test_a_drawing_reads_the_console_s_change_before_it_draws():
    port = free_port()
    doc = {
        "version": 1,
        "params": {"gain": {"type": "float", "value": 0.0}},
        "passes": [
            {"name": "main", "fragment": "void main(){ gl_FragColor = vec4(gain); }"}
        ],
    }
    with ghee.open(doc, width=8, height=8, serve=port) as h:
        h.frame()
        post(port, "/set?name=gain&value=0.5")
        with h.draw() as d:
            seen = d.tick.params
            d.text(f"{seen.get('gain')}", name="seen")
        assert seen == {"gain": 0.5}
        tick = h.frame()
        assert (tick.frame, tick.params) == (2, {})
        assert isinstance(tick, ghee.Tick)


def test_the_console_publishes_the_render(served):
    h, port = served
    stats = json_of(port, "/stats")
    assert stats["active"] == "scene"
    assert stats["error"] == ""
    assert stats["display"] == {"w": 32, "h": 32, "format": "rgba", "rotation": 0}
    assert isinstance(stats["gpu"]["trilinear_npot"], bool)
    assert stats["out"] == []
    assert stats["frames"] == 1
    assert stats["frame_ms"] > 0.0
    h.skip()
    assert json_of(port, "/stats")["skipped"] == 1


def test_the_console_gives_the_turn_of_the_frame():
    port = free_port()
    with ghee.open(RED, width=32, height=24, rotate=90, serve=port) as h:
        h.frame()
        assert json_of(port, "/stats")["display"] == {
            "w": 32,
            "h": 24,
            "format": "rgba",
            "rotation": 90,
        }


def test_the_console_serves_the_scene_it_renders(served):
    _, port = served
    assert json_of(port, "/scenes") == {
        "scenes": ["scene"],
        "active": "scene",
        "procedural": ["scene"],
        "mine": [],
        "my_scenes": None,
    }
    scene = json_of(port, "/scene")
    assert [p["name"] for p in scene["passes"]] == ["main"]
    assert scene["params"]["tint"]["type"] == "color"
    assert "dither" in json_of(port, "/filters")


def test_the_console_serves_a_shim_scene_as_the_shim_wrote_it(monkeypatch):
    """The renderer reads the shim's scene, which it would refuse for a key
    that it does not take, and writes it back the same: every key under its
    own name, every default left out on both sides."""
    monkeypatch.chdir(REPO / "ghee" / "examples")
    scene = ghee.Scene(
        common="float halved(float x){ return x * 0.5; }",
        params={
            "gain": ghee.Param("gain", "float", 0.5, min=0, max=1, step=0.25),
            "mood": ghee.Param("mood", "string", "calm", options=["calm", "wild"]),
        },
        assets=[ghee.Asset("sans", "fonts/DejaVuSans-Latin.ttf")],
        passes=[
            ghee.Pass(
                "ink",
                size=(64, 32),
                sampling="trilinear",
                arrays=[ghee.ArrayInput("level", 4), ghee.ArrayInput("mask", (4, 2))],
                layers=[
                    ghee.Layer(
                        "words",
                        "hi",
                        x=2,
                        color=[1.0, 0.5, 0.0],
                        opacity=0.5,
                        blend="add",
                        font_family="sans",
                        text_align="right",
                    ),
                    ghee.Layer(
                        "title",
                        "ho",
                        font_size=24,
                        blend="screen",
                        font_family="fonts/DejaVuSans-Latin.ttf",
                        render="slug",
                    ),
                    ghee.Layer("plain", "ok", visible=False, y=3),
                ],
            ),
            ghee.Pass(
                "main",
                "void main(){ gl_FragColor = texture2D(iChannel0, uv); }",
                channels=[ghee.Channel("ink")],
                layers=[
                    ghee.Layer("glow", "hi", color={"from": "#ff0000", "to": "#0000ff"})
                ],
            ),
        ],
    ).to_dict()
    port = free_port()
    with ghee.open(scene, width=32, height=32, serve=port) as h:
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
    assert post(port, "/set?name=main.gain&value=0.25")[0] == 200
    h.feed("level", LEVEL)
    h.frame()
    assert abs(h.pixels()[0] - 64) <= 2
    # a scene param takes the scene scope
    assert post(port, "/set?name=tint&value=0,1,0")[0] == 200
    h.feed("level", LEVEL)
    h.frame()
    assert h.pixels()[1] > 0 and h.pixels()[0] == 0


def test_a_set_reaches_a_vec_or_color_uniform():
    port = free_port()
    with ghee.open(VECS, width=8, height=1, serve=port) as h:
        h.frame()
        for q in [
            "name=a&value=1,0",
            "pass=main&name=b&value=0,0.25,1",
            "pass=main&name=c&value=0.5,1,0,1",
            "name=tint&value=0,1,0.5",
        ]:
            assert post(port, "/set?" + q)[0] == 200
        h.frame()
        got = quarters(h)
        assert near(got, [[1, 0, 0], [0, 0.25, 1], [0.5, 1, 0], [0, 1, 0.5]]), got


def test_a_layer_set_lands_on_the_next_frame():
    # DejaVu Sans draws the layer's words, red over black.
    words = {
        "version": 1,
        "passes": [
            {
                "name": "main",
                "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                "layers": [
                    {"name": "t", "text": "I", "font_size": 24, "color": "#ff0000"}
                ],
            }
        ],
    }
    port = free_port()
    with ghee.open(words, width=32, height=32, timestep=0, serve=port) as h:
        h.frame()
        assert max(h.pixels()[0::4]) == 255
        assert post(port, "/set?layer=t&field=visible&value=false")[0] == 200
        h.frame()
        assert max(h.pixels()[0::4]) == 0
        with pytest.raises(urllib.error.HTTPError) as raised:
            post(port, "/set?layer=t&field=name&value=u")
        assert raised.value.code == 400


def first_width(doc):
    """The width of the first layer of the first pass of `doc`, or None."""
    return doc["passes"][0].get("layers", [{}])[0].get("width")


def test_the_console_shows_the_layers_that_a_drawing_adds_and_changes():
    port = free_port()
    with ghee.open(width=32, height=16, serve=port) as h:
        counts = [json_of(port, "/stats")["layer_changes"]]
        drawn = []
        for width in (4, 9):
            with h.draw() as d:
                d.rect(x=1, y=2, width=width, height=3)
                d.text("hi", name="label", font_size=8)
            until(lambda w=width: first_width(json_of(port, "/scene")) == w, h.frame)
            layers = json_of(port, "/scene")["passes"][0]["layers"]
            drawn.append([(d["name"], d.get("width")) for d in layers])
            counts.append(json_of(port, "/stats")["layer_changes"])
        assert drawn == [[("rect", 4), ("label", None)], [("rect", 9), ("label", None)]]
        assert counts[0] < counts[1] < counts[2]


def test_a_drawing_reaches_the_scene_with_its_frame_and_the_next_one_soon_after():
    port = free_port()
    with ghee.open(width=32, height=16, serve=port) as h:
        with h.draw() as d:
            d.rect(x=1, y=2, width=4, height=3)
        assert first_width(json_of(port, "/scene")) == 4
        with h.draw() as d:
            d.rect(x=1, y=2, width=9, height=3)
        until(lambda: first_width(json_of(port, "/scene")) == 9, h.frame)


def test_a_param_set_by_the_console_or_the_host_reaches_the_scene(served):
    h, port = served
    gain = lambda: json_of(port, "/scene")["passes"][0]["params"]["gain"]["value"]
    assert post(port, "/set?pass=main&name=gain&value=0.5")[0] == 200
    h.feed("level", LEVEL)
    h.frame()
    assert gain() == 0.5
    h.set_param("gain", 0.25)
    h.feed("level", LEVEL)
    h.frame()
    assert gain() == 0.25


def test_the_console_locks_what_the_host_sets_and_the_rest_still_edits():
    doc = copy.deepcopy(RED)
    doc["passes"][0]["layers"] = [{"name": "title", "text": "hi", "font_size": 8}]
    port = free_port()
    with ghee.open(doc, width=32, height=16, serve=port) as h:
        h.set_param("gain", 0.5)
        h.set_layer("title", "x", 3)
        with h.draw() as d:
            d.text("12:00", font_size=8)
        until(lambda: len(json_of(port, "/scene")["passes"][0]["layers"]) == 2, h.frame)
        assert json_of(port, "/stats")["locked"] == {
            "params": ["main.gain"],
            "fields": ["text.font_size", "text.text", "title.x"],
            "scene": True,
        }
        for path, key in [
            ("/set?pass=main&name=gain&value=1", "param 'main.gain'"),
            ("/set?name=gain&value=1", "param 'main.gain'"),
            ("/set?layer=title&field=x&value=1", "layer field 'title.x'"),
            ("/set?layer=text&field=text&value=ho", "layer field 'text.text'"),
        ]:
            with pytest.raises(urllib.error.HTTPError) as raised:
                post(port, path)
            assert raised.value.code == 403, path
            assert raised.value.read().decode().endswith(key), path
        assert post(port, "/set?name=tint&value=0,1,0")[0] == 200
        assert post(port, "/set?layer=title&field=y&value=2")[0] == 200
        h.frame()
        looked = json_of(port, "/scene")
        title, text = looked["passes"][0]["layers"]
        assert (looked["params"]["tint"]["value"], title["y"]) == ([0, 1, 0], 2)
        refused(
            port, looked, ["passes", 0, "layers", 0, "x"], 9, "layer field 'title.x'"
        )
        refused(
            port,
            looked,
            ["passes", 0, "layers", 1, "text"],
            "ho",
            "layer field 'text.text'",
        )
        title["opacity"] = 0.5
        assert post_json(port, "/scene", looked)[0] == 200
        h.frame()
        title, text = json_of(port, "/scene")["passes"][0]["layers"]
        assert (title["x"], title["opacity"], text["text"]) == (3, 0.5, "12:00")


def test_a_drawing_locks_the_keys_it_passes_and_the_console_keeps_the_rest():
    port = free_port()
    with ghee.open(copy.deepcopy(RED), width=32, height=16, serve=port) as h:

        def draw(words):
            with h.draw() as d:
                d.text(words, x=1, font_size=8, glow={})

        def text():
            return json_of(port, "/scene")["passes"][0]["layers"][0]

        draw("a")
        assert json_of(port, "/stats")["locked"]["fields"] == [
            "text.font_size",
            "text.glow",
            "text.text",
            "text.x",
        ]
        for path, key in [
            ("/set?layer=text&field=x&value=5", "text.x"),
            ("/set?layer=text&effect=glow&field=strength&value=0.5", "text.glow"),
        ]:
            with pytest.raises(urllib.error.HTTPError) as raised:
                post(port, path)
            assert raised.value.code == 403, path
            assert raised.value.read().decode().endswith(f"'{key}'"), path
        assert post(port, "/set?layer=text&field=opacity&value=0.5")[0] == 200
        draw("b")
        looked = json_of(port, "/scene")
        refused(
            port, looked, ["passes", 0, "layers", 0, "x"], 7, "layer field 'text.x'"
        )
        looked["passes"][0]["layers"][0]["text_align"] = "center"
        assert post_json(port, "/scene", looked)[0] == 200
        for words in "cde":
            draw(words)
        until(lambda: text()["text"] == "e", h.frame)
        got = text()
        assert (got["text"], got["x"], got["opacity"], got["text_align"]) == (
            "e",
            1,
            0.5,
            "center",
        )


def call(d, n):
    """The call of the program of drawn() at frame n. It passes x, font_size
    and glow, and font_size changes each frame, so each frame gives the
    layer a new look."""
    d.text(f"w{n}", x=1, font_size=8 + n % 2, glow={})


SERVED = f"""#!{sys.executable}
import ghee

{inspect.getsource(call)}

@ghee.tick
def draw(t):
    with ghee.draw() as d:
        call(d, t.frame)


ghee.run()
"""


@contextlib.contextmanager
def drawn(host, tmp_path, size=(32, 16)):
    """A program that makes call() each frame, run by `host`: ghee.open with
    serve=, or `ghee serve` as a scene of its own, with what it prints on
    stderr in tmp_path/"stderr". Gives the console's port, and a step that lets the
    program draw on."""
    port = free_port()
    width, height = size
    if host == "open":
        with ghee.open(copy.deepcopy(RED), width=width, height=height, serve=port) as h:
            frames = iter(range(10**6))

            def step():
                with h.draw() as d:
                    call(d, next(frames))

            step()
            yield port, step
        return
    scene = tmp_path / "drawn.py"
    scene.write_text(SERVED)
    scene.chmod(0o755)
    cli = shutil.which("ghee", path=os.path.dirname(sys.executable))
    assert cli, f"no ghee command beside {sys.executable}"
    env = {
        **os.environ,
        "XDG_STATE_HOME": str(tmp_path),
        "XDG_DATA_HOME": str(tmp_path),
    }
    served = subprocess.Popen(
        [cli, "serve", str(scene), "--port", str(port), "--size", f"{width}x{height}"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=(tmp_path / "stderr").open("w"),
    )

    def alive():
        if served.poll() is not None:
            pytest.fail((tmp_path / "stderr").read_text())
        time.sleep(0.01)

    # ghee serve draws frames while someone watches
    def watch():
        url = f"http://127.0.0.1:{port}/stream"
        with contextlib.suppress(OSError), urllib.request.urlopen(url) as r:
            while r.read(4096):
                pass

    try:
        until(lambda: get(port, "/stats")[0] == 200, alive)
        threading.Thread(target=watch, daemon=True).start()
        yield port, alive
    finally:
        served.terminate()
        served.wait()


def until(done, step):
    """Run `step` until `done` holds."""
    while True:
        try:
            if done():
                return
        except (OSError, urllib.error.URLError):
            pass
        step()


@pytest.mark.parametrize("host", ["open", "serve"])
def test_a_drawing_locks_the_keys_it_passes_through_either_host(host, tmp_path):
    with drawn(host, tmp_path) as (port, step):
        locked = ["text.font_size", "text.glow", "text.text", "text.x"]

        def drawn_and_locked():
            layers = json_of(port, "/scene")["passes"][0].get("layers", [])
            fields = json_of(port, "/stats")["locked"]["fields"]
            return [d["name"] for d in layers] == ["text"] and fields == locked

        until(drawn_and_locked, step)
        for path, key in [
            ("/set?layer=text&field=x&value=5", "text.x"),
            ("/set?layer=text&effect=glow&field=strength&value=0.5", "text.glow"),
        ]:
            with pytest.raises(urllib.error.HTTPError) as raised:
                post(port, path)
            why = raised.value.read().decode()
            assert (raised.value.code, why.endswith(f"'{key}'")) == (403, True), why
        assert post(port, "/set?layer=text&field=opacity&value=0.5")[0] == 200
        sizes = []

        def text():
            layer = json_of(port, "/scene")["passes"][0].get("layers", [{}])[0]
            if not sizes or sizes[-1] != layer.get("font_size"):
                sizes.append(layer.get("font_size"))
            return layer

        until(lambda: len(sizes) > 3, lambda: (step(), text()))
        got = text()
        assert (got["x"], got.get("opacity")) == (1, 0.5)
        # font_family takes a rebuild under ghee serve
        looked = json_of(port, "/scene")
        refused(
            port, looked, ["passes", 0, "layers", 0, "x"], 6, "layer field 'text.x'"
        )

        def font():
            # the program gives font_size a new value each frame, so a post
            # that read an older one is refused, and goes again
            now = json_of(port, "/scene")
            now["passes"][0]["layers"][0]["font_family"] = ["Pixel Operator 8"]
            try:
                return post_json(port, "/scene", now)[0] == 200
            except urllib.error.HTTPError as e:
                assert e.code == 403, e.read()
                return False

        until(font, step)
        del sizes[:]
        until(lambda: len(sizes) > 3, lambda: (step(), text()))
        got = text()
        assert (got["x"], got.get("opacity"), got.get("font_family")) == (
            1,
            0.5,
            "Pixel Operator 8",
        )


def jpeg_size(jpg):
    """(width, height) from a JPEG's start of frame."""
    at = 2
    while jpg[at + 1] != 0xC0:
        at += 2 + int.from_bytes(jpg[at + 2 : at + 4], "big")
    return int.from_bytes(jpg[at + 7 : at + 9], "big"), int.from_bytes(
        jpg[at + 5 : at + 7], "big"
    )


def first_jpeg(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/stream") as r:
        head = b""
        while not head.endswith(b"\r\n\r\n"):
            head += read_exact(r, 1)
        jpg = b""
        while not jpg.endswith(b"\r\n--frame"):
            jpg += read_exact(r, 1)
        return jpg[: -len(b"\r\n--frame")]


@pytest.mark.parametrize("host", ["open", "serve"])
def test_either_host_gives_its_address_and_serves_the_same_stats_and_preview(
    host, tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger="ghee")
    with drawn(host, tmp_path, size=(640, 320)) as (port, step):
        until(lambda: json_of(port, "/stats")["frames"] > 0, step)
        assert sorted(json_of(port, "/stats")) == sorted(
            ["active", "display", "error", "error_line", "error_pass", "fps"]
            + ["frame_ms", "frames", "gpu", "layer_changes", "locked", "out"]
            + ["rev", "skipped", "stale", "stream_fps", "text_errors"]
        )
        assert jpeg_size(watching(None, lambda: first_jpeg(port), step=step)) == (
            480,
            240,
        )
    printed = caplog.text
    if host == "serve":
        printed = (tmp_path / "stderr").read_text()
    line = next(x for x in printed.splitlines() if "ghee console:" in x)
    line = line[line.index("ghee console:") :]
    assert line.startswith(f"ghee console: http://127.0.0.1:{port}/ 1 scene ['"), line
    assert line.endswith("preview 640x320 rgba)"), line


def answer(port, method, path):
    """The status, headers and body of one request."""
    conn = http.client.HTTPConnection("127.0.0.1", port)
    try:
        conn.request(method, path)
        r = conn.getresponse()
        return r.status, dict(r.getheaders()), r.read()
    finally:
        conn.close()


@pytest.mark.parametrize("host", ["open", "serve"])
def test_head_answers_as_get_and_a_405_names_the_methods(host, tmp_path):
    with drawn(host, tmp_path) as (port, _step):
        for path in ["/stats", "/scene", "/nope"]:
            status, headers, _ = answer(port, "GET", path)
            want = (status, headers.get("Content-Type"), headers.get("ETag"), b"")
            status, headers, body = answer(port, "HEAD", path)
            got = (status, headers.get("Content-Type"), headers.get("ETag"), body)
            assert got == want, path
        status, headers, body = answer(port, "HEAD", "/stream")
        assert (status, body) == (200, b"")
        assert headers["Content-Type"].startswith("multipart/x-mixed-replace")
        for method, path, want in [
            ("DELETE", "/stats", (405, "GET, HEAD")),
            ("GET", "/set", (405, "POST")),
            ("PUT", "/scene", (405, "GET, HEAD, POST")),
            ("BREW", "/stats", (501, None)),
        ]:
            status, headers, _ = answer(port, method, path)
            assert (status, headers.get("Allow")) == want, (method, path)


@pytest.mark.parametrize("host", ["open", "serve"])
def test_a_get_answers_its_if_match_and_its_if_none_match(host, tmp_path):
    def read(**headers):
        conn = http.client.HTTPConnection("127.0.0.1", port)
        try:
            conn.request("GET", "/scene", headers=headers)
            r = conn.getresponse()
            return r.status, r.getheader("ETag"), r.read()
        finally:
            conn.close()

    with drawn(host, tmp_path) as (port, step):
        assert read(**{"If-Match": '"0"'})[0] == 412
        assert read(**{"If-None-Match": "*"})[0] == 304
        if host == "serve":
            # the program's changes give the scene a new ETag each frame
            def unchanged():
                tag = read()[1]
                return read(**{"If-None-Match": tag}) == (304, tag, b"")

            until(unchanged, step)


def test_an_apply_gives_layers_a_new_look_and_changes_nothing_else():
    doc = {
        "version": 1,
        "passes": [
            {
                "name": "main",
                "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                "layers": [
                    {
                        "name": "box",
                        "shape": "rect",
                        "x": 0,
                        "y": 0,
                        "width": 6,
                        "height": 6,
                    }
                ],
            }
        ],
    }
    centre = slice((3 * 8 + 3) * 4, (3 * 8 + 3) * 4 + 3)
    port = free_port()
    with ghee.open(doc, width=8, height=8, timestep=0, serve=port) as h:
        h.frame()
        assert list(h.pixels()[centre]) == [255, 255, 255]
        served = json_of(port, "/scene")
        looked = copy.deepcopy(served)
        # blend changes with a rebuild of the layer
        looked["passes"][0]["layers"][0].update(blend="add", color="red")
        assert post_json(port, "/scene", looked)[0] == 200
        h.frame()
        assert list(h.pixels()[centre]) == [255, 0, 0]
        assert json_of(port, "/scene") == looked
        # a post that leaves a layer out leaves the layer as it is
        bare = copy.deepcopy(served)
        del bare["passes"][0]["layers"]
        assert post_json(port, "/scene", bare)[0] == 200
        h.frame()
        assert json_of(port, "/scene") == looked
        coded = copy.deepcopy(looked)
        coded["passes"][0]["fragment"] = "void main(){ gl_FragColor = vec4(1.0); }"
        with pytest.raises(urllib.error.HTTPError) as raised:
            post_json(port, "/scene", coded)
        assert raised.value.code == 403
        assert raised.value.read().decode() == (
            "this console changes layers alone: its program owns the passes, the "
            "params and the assets"
        )


def test_an_apply_reports_the_error_of_every_look_that_fails(tmp_path):
    rect = {"shape": "rect", "width": 4, "height": 4}
    doc = {
        "version": 1,
        "passes": [
            {
                "name": "main",
                "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                "layers": [{"name": "a", **rect}, {"name": "b", **rect}],
            }
        ],
    }
    port = free_port()
    with ghee.open(doc, width=8, height=8, serve=port, base_dir=tmp_path) as h:
        h.frame()
        looked = json_of(port, "/scene")
        looked["passes"][0]["layers"] = [
            {"name": "a", "image": "one.png"},
            {"name": "b", "image": "two.png"},
        ]
        assert post_json(port, "/scene", looked)[0] == 200
        h.frame()
        error = json_of(port, "/stats")["error"]
    a, b = error.split("\n")
    assert a.startswith("layer 'a' image 'one.png'"), error
    assert b.startswith("layer 'b' image 'two.png'"), error


def test_a_set_the_scene_refuses_reads_as_set_param_reads(served):
    h, port = served
    with pytest.raises(ValueError) as python:
        h.set_param("nope", 1.0, in_pass="main")
    with pytest.raises(urllib.error.HTTPError) as raised:
        post(port, "/set?pass=main&name=nope&value=1")
    assert raised.value.code == 400
    assert raised.value.read().decode() == f"bad /set: {python.value}"


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
        answer["got"] = get(port, "/frame.png")

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
        answer["got"] = get(port, "/frame.png")

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
    h = ghee.open(RED, width=8, height=8, serve=port)
    assert json_of(port, "/stats")["active"] == "scene"
    h.close()
    with pytest.raises(OSError):
        get(port, "/stats")


def test_serve_binds_where_it_is_told():
    port = free_port()
    with ghee.open(RED, width=8, height=8, serve=("127.0.0.1", port)) as h:
        h.feed("level", LEVEL)
        h.frame()
        assert json_of(port, "/stats")["active"] == "scene"


def test_serve_takes_a_port_or_a_pair():
    with pytest.raises(ValueError):
        ghee.open(RED, width=8, height=8, serve="8090")


def test_the_mjpeg_stream_shows_a_frame_that_holds_alpha(served):
    h, port = served

    jpg = watching(h, lambda: first_jpeg(port))
    assert jpg[:2] == b"\xff\xd8"
    # the frame's size, from the JPEG's start of frame: its height, its width
    at = 2
    while jpg[at + 1] != 0xC0:
        at += 2 + int.from_bytes(jpg[at + 2 : at + 4], "big")
    assert jpg[at + 5 : at + 9] == (32).to_bytes(2, "big") * 2


@needs_ffmpeg
def test_h264_serves_what_webcodecs_decodes(served):
    h, port = served

    def read():
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/h264") as r:
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
    with ghee.open(RED, width=32, height=32, serve=port) as h:
        h.feed("level", LEVEL)
        h.frame()

        def read():
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/h264")
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode()
            return 200, ""

        status, why = watching(h, read)
    assert status == 503
    assert "no H264 encoder" in why and "ffmpeg" in why, why


@needs_ffmpeg
def test_h264_reaches_a_settled_host(served):
    h, port = served

    def read():
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/h264") as r:
            return messages(r, 2)

    assert [t for t, _ in watching(h, read, step=h.skip)] == [0, 1]


@needs_ffmpeg
def test_stream_max_caps_the_preview():
    port = free_port()
    with ghee.open(RED, width=64, height=64, serve=port, stream_max=0.5) as h:
        h.feed("level", LEVEL)
        h.frame()

        def read():
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/h264") as r:
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
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/h264") as r:
            return messages(r, 2)

    for _ in range(2):
        assert [t for t, _ in watching(h, read)] == [0, 1]


@needs_ffmpeg
def test_the_stream_runs_only_while_a_client_watches(served):
    h, port = served
    assert "h264" not in json_of(port, "/stats")["out"]

    def read():
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/h264") as r:
            messages(r, 2)
            return json_of(port, "/stats")

    assert "h264" in watching(h, read)["out"]
    # the console drops the client on the write that follows the close
    while "h264" in json_of(port, "/stats")["out"]:
        h.feed("level", LEVEL)
        h.frame()


SIZED = {
    "version": 1,
    "passes": [
        {
            "name": "main",
            "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
            "layers": [{"name": "title", "text": "Hi", "font_size": 8}],
        }
    ],
}

GROWN = """#!{python}
import ghee


@ghee.tick
def grow(tick):
    ghee.layers.title.font_size = 16


ghee.run({scene!r})
"""


def test_a_key_that_rebuilds_its_layer_sets_alike_through_set_python_and_the_wire(
    tmp_path,
):
    image = pillow()
    size = 64 * 32 * 4
    with ghee.open(copy.deepcopy(SIZED), width=64, height=32, timestep=0) as h:
        h.frame()
        small = h.pixels()
        h.layers.title.font_size = 16
        h.frame()
        want = h.pixels()
    assert want != small

    port = free_port()
    with ghee.open(copy.deepcopy(SIZED), width=64, height=32, serve=port) as h:
        h.frame()
        assert post(port, "/set?layer=title&field=font_size&value=16")[0] == 200
        h.frame()
        assert h.pixels() == want, "/set of ghee.open"

    cli = shutil.which("ghee", path=os.path.dirname(sys.executable))
    assert cli, f"no ghee command beside {sys.executable}"
    script = tmp_path / "grown.py"
    script.write_text(GROWN.format(python=sys.executable, scene=SIZED))
    script.chmod(0o755)
    ran = subprocess.run(
        [cli, "render", str(script), "-o", "-", "--size", "64x32"]
        + ["--frames", "3", "--timestep", "0"],
        capture_output=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout[-size:] == want, "the wire"

    scene = tmp_path / "sized.scene.json"
    scene.write_text(json.dumps(SIZED))
    port = free_port()
    env = {**os.environ, "XDG_STATE_HOME": str(tmp_path)}
    served = subprocess.Popen(
        [cli, "serve", str(scene), "--port", str(port), "--size", "64x32"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        # a /set answers 503 while the scene loads
        until(
            lambda: post(port, "/set?layer=title&field=font_size&value=16")[0] == 200,
            lambda: time.sleep(0.05),
        )
        _, png = get(port, "/frame.png")
        got = image.open(io.BytesIO(png)).convert("RGBA").tobytes()
        assert got == want, "/set of ghee serve"
        layer = json_of(port, "/scene")["passes"][0]["layers"][0]
        assert layer["font_size"] == 16
    finally:
        served.terminate()
        served.communicate()
