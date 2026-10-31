"""The handle: what open, frame, skip, set_param and feed do."""

import array
import json
import pathlib
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import zlib

import pytest
from conftest import RED, REPO, VECS, near, numpy, pillow, quarters, scene_file

import ghee

LEVEL = array.array("f", [1.0] * 8)

# iTimeDelta, scaled into red: a quarter second reads 255.
SHOWS_DT = {
    "version": 1,
    "passes": [
        {
            "name": "main",
            "fragment": "void main(){ gl_FragColor = vec4(iTimeDelta * 4.0, 0, 0, 1); }",
        }
    ],
}


def pixel(h, i=0):
    """Pixel `i` of the last frame, as a tuple of components."""
    px = h.pixels()
    n = h.components
    return tuple(px[i * n : i * n + n])


def test_a_scene_renders_the_frame_it_declares():
    with ghee.open(RED) as h:
        h.feed("level", LEVEL)
        tick = h.frame()
        assert h.size == (320, 240)
        assert h.components == 4
        assert pixel(h) == (255, 0, 0, 255)
        assert tick.frame == 0
        assert len(h.pixels()) == 320 * 240 * 4


def test_the_size_a_headless_render_takes():
    with ghee.open(RED, width=64, height=48) as h:
        assert h.size == (64, 48)
        h.feed("level", LEVEL)
        h.frame()
        assert len(h.pixels()) == 64 * 48 * 4


def test_a_scene_loads_from_a_path_a_string_or_a_dict(tmp_path):
    for source in [scene_file(tmp_path, RED), json.dumps(RED), RED]:
        with ghee.open(source, width=8, height=8) as h:
            h.feed("level", LEVEL)
            h.frame()
            assert pixel(h) == (255, 0, 0, 255)


def test_a_pass_draws_the_same_in_glsl_es_1_and_3():
    es1 = "void main(){ gl_FragColor = vec4(uv, fwidth(uv.x) * 4.0, 1.0); }"
    es3 = (
        "#version 300 es\nout vec4 c;\n"
        "void main(){ c = vec4(uv, fwidth(uv.x) * 4.0, 1.0); }"
    )
    frames = []
    for fragment in (es1, es3):
        doc = {"version": 1, "passes": [{"name": "main", "fragment": fragment}]}
        with ghee.open(doc, width=8, height=4) as h:
            h.frame()
            frames.append(h.pixels())
    assert frames[0] == frames[1]
    # 0.5 lies halfway between two levels, and GL may write either
    assert frames[0][2] in (127, 128)


def test_a_param_reaches_the_shader_live():
    with ghee.open(RED, width=8, height=8) as h:
        h.feed("level", LEVEL)
        h.set_param("gain", 0.5)
        h.frame()
        assert abs(pixel(h)[0] - 128) <= 2
        h.set_param("gain", 1.0, in_pass="main")
        h.feed("level", LEVEL)
        h.frame()
        assert pixel(h)[0] == 255
        h.set_param("main.gain", 0.0)
        h.feed("level", LEVEL)
        h.frame()
        assert pixel(h)[0] == 0
        with pytest.raises(TypeError):
            h.set_param("gain", 1.0, pass_="main")
        with pytest.raises(ValueError, match="^nan is not a number$"):
            h.set_param("gain", float("nan"))


def test_a_filter_entry_starts_the_filter_where_set_param_puts_it():
    ramp = "void main(){ gl_FragColor = vec4(vec3(uv.x), 1.0); }"

    def frame(filters, levels=None):
        doc = {"version": 1, "passes": [{"fragment": ramp}], "filters": filters}
        with ghee.open(doc, width=16, height=2) as h:
            if levels is not None:
                h.set_param("levels", levels, in_pass="dither")
            h.frame()
            return h.pixels()

    entry = frame([{"name": "dither", "levels": 8}])
    assert entry == frame(["dither"], 8)
    assert entry != frame(["dither"])


def test_vec_and_color_values_reach_the_shader():
    built = ghee.Scene(
        passes=[
            ghee.Pass(
                "main",
                VECS["passes"][0]["fragment"],
                params={
                    "b": ghee.Param("b", "vec3", [0.75, 0.5, 0.25]),
                    "c": ghee.Param("c", "vec4", [1.0, 0.5, 0.25, 0.5]),
                },
            )
        ],
        params={
            "a": ghee.Param("a", "vec2", [0.25, 0.5]),
            "tint": ghee.Param("tint", "color", [0.5, 0.25, 1.0]),
        },
    ).to_dict()
    for doc in [VECS, built]:
        with ghee.open(doc, width=8, height=1) as h:
            h.frame()
            got = quarters(h)
            assert near(
                got,
                [[0.25, 0.5, 0], [0.75, 0.5, 0.25], [0.5, 0.25, 0.125], [0.5, 0.25, 1]],
            ), got
            h.set_param("a", [1.0, 0.0])
            h.set_param("b", (0.0, 0.25, 1.0), in_pass="main")
            h.set_param("main.c", [0.5, 1.0, 0.0, 1.0])
            h.set_param("tint", [0.0, 1.0, 0.5])
            h.frame()
            got = quarters(h)
            assert near(got, [[1, 0, 0], [0, 0.25, 1], [0.5, 1, 0], [0, 1, 0.5]]), got


# (what a feed takes, and the red that it draws). A uint8 buffer
# reaches the texture as byte / 255, and any other numbers as they are.
FEEDS = {
    "float32 array": (lambda: LEVEL, 255),
    "list": (lambda: [1.0] * 8, 255),
    "memoryview": (lambda: memoryview(bytes([255] * 8)), 255),
    "bytes": (lambda: bytes([128] * 8), 128),
    "uint8 slice": (lambda: numpy().full(16, 128, "uint8")[::2], 128),
    "float64": (lambda: numpy().full(8, 0.5), 128),
    "int64 2D": (lambda: numpy().ones((2, 4), "int64"), 255),
    "float16": (lambda: numpy().full(8, 0.5, "float16"), 128),
    "bool": (lambda: numpy().ones(8, bool), 255),
    "array d": (lambda: array.array("d", [0.5] * 8), 128),
    "generator": (lambda: (0.5 for _ in range(8)), 128),
}


@pytest.mark.parametrize("kind", FEEDS)
def test_a_feed_takes_any_buffer_or_iterable_of_numbers(kind):
    make, red = FEEDS[kind]
    with ghee.open(RED, width=8, height=8) as h:
        h.feed("level", make())
        h.frame()
        assert abs(pixel(h)[0] - red) <= 1


def test_the_next_frame_consumes_the_feed_once():
    # the texture holds what it was fed, so a frame with no feed renders the same
    with ghee.open(RED, width=8, height=8) as h:
        h.feed("level", LEVEL)
        h.frame()
        first = h.pixels()
        h.frame()
        assert h.pixels() == first


def test_skip_takes_the_tick_and_draws_nothing():
    with ghee.open(RED, width=8, height=8) as h:
        h.feed("level", LEVEL)
        h.frame()
        drawn = h.pixels()
        h.set_param("gain", 0.0)
        tick = h.skip()
        assert tick.frame == 1
        assert h.pixels() == drawn
        assert h.frame().frame == 2


def test_a_tick_carries_the_frame_the_time_and_the_step():
    with ghee.open(RED, width=8, height=8, timestep=0.1) as h:
        ticks = [h.frame(), h.skip(), h.frame()]
    assert [t.frame for t in ticks] == [0, 1, 2]
    assert [round(t.time, 6) for t in ticks] == [0.0, 0.1, 0.2]
    assert [t.dt for t in ticks] == [0.1, 0.1, 0.1]


def test_a_frame_after_a_long_wait_steps_a_quarter_second_at_most(clock):
    # one frame must never step a simulation by seconds
    clock(0.0)
    with ghee.open(SHOWS_DT, width=4, height=4) as h:
        h.frame()
        clock(0.4)
        tick = h.frame()
        assert (tick.dt, pixel(h)[0]) == (0.25, 255)


def test_an_unpinned_tick_reads_the_clock_and_a_frame_waits_out_its_period(clock):
    clock(10.0)
    with ghee.open(RED, width=8, height=8, fps=50) as h:
        first = h.frame()
        second = h.frame()
        clock(10.5)
        third = h.frame()
    assert first.time == 0.0
    assert first.dt == 1.0 / 60.0  # nothing is measured before the second frame
    assert (second.time, second.dt) == pytest.approx((0.02, 0.02))
    assert (third.time, third.dt) == pytest.approx((0.5, 0.25))


def test_the_ticks_dt_add_up_to_the_time_passed_over_skips(clock):
    clock(0.0)
    with ghee.open(RED, width=8, height=8) as h:
        ticks = []
        for at, call in [(0.0, h.frame), (0.1, h.skip), (0.15, h.skip), (0.3, h.frame)]:
            clock(at)
            ticks.append(call())
    passed = ticks[-1].time - ticks[0].time
    assert passed == pytest.approx(0.3)
    assert sum(t.dt for t in ticks[1:]) == pytest.approx(passed, abs=1e-9)


def test_a_pinned_timestep_repeats_a_render_bit_for_bit():
    def run():
        with ghee.open(RED, width=32, height=32, timestep=1 / 30) as h:
            for _ in range(4):
                h.feed("level", LEVEL)
                h.frame()
            return h.pixels()

    assert run() == run()


def test_a_handle_closes_once_and_says_what_it_is():
    h = ghee.open(RED, width=8, height=8)
    assert repr(h) == "<ghee 8x8>"
    h.close()
    h.close()
    assert repr(h) == "<ghee 8x8 closed>"
    assert h.size == (8, 8)


def test_a_sigterm_closes_a_held_handle_and_then_ends_the_program_by_it(monkeypatch):
    """open() holds SIGTERM so for a display. A program's own handler stays,
    and close() puts the default back."""
    before = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        h = ghee.open(RED, width=8, height=8)
        h._hold_sigterm()
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        raised = []
        monkeypatch.setattr(signal, "raise_signal", raised.append)
        handler(signal.SIGTERM, None)
        assert repr(h) == "<ghee 8x8 closed>"
        assert raised == [signal.SIGTERM]
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

        with ghee.open(RED, width=8, height=8) as h:
            h._hold_sigterm()
            assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

        def own(_sig, _frame):
            pass

        signal.signal(signal.SIGTERM, own)
        with ghee.open(RED, width=8, height=8) as h:
            h._hold_sigterm()
            assert signal.getsignal(signal.SIGTERM) is own
        assert signal.getsignal(signal.SIGTERM) is own
    finally:
        signal.signal(signal.SIGTERM, before)


def test_a_sigterm_ends_a_program_with_a_held_handle_by_sigterm(tmp_path):
    """systemd counts a death by SIGTERM as a clean stop. The signal lands
    in frame() too, which runs Python code with the handle in use."""
    path = tmp_path / "held.py"
    path.write_text(
        "import ghee\n"
        f"h = ghee.open({RED!r}, width=8, height=8)\n"
        "h._hold_sigterm()\n"
        "print('ready', flush=True)\n"
        "while True:\n"
        "    h.frame()\n"
    )
    held = subprocess.Popen(
        [sys.executable, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert held.stdout.readline() == "ready\n"
    time.sleep(0.2)
    held.terminate()
    _, err = held.communicate(timeout=10)
    assert held.returncode == -signal.SIGTERM, err


def test_the_module_names_its_version():
    """One release, two spellings: PEP 440 for the wheel, semver for the crate.

    The package surface is test_shim.py's.
    """
    from ghee import _native

    assert ghee.__version__.startswith(_native.__version__.split("-")[0])
    assert set(_native.__all__) == {"Handle", "open", "__version__"}


def test_a_frame_lets_the_other_threads_run():
    # the host's own threads keep working while a paced frame renders and waits
    inside = threading.Event()
    stop = threading.Event()
    counted = [0]

    def count():
        while not stop.is_set():
            if inside.is_set():
                counted[0] += 1
            time.sleep(0.001)

    worker = threading.Thread(target=count)
    worker.start()
    try:
        with ghee.open(RED, width=64, height=64, fps=20) as h:
            h.feed("level", LEVEL)
            # A frame that held the GIL would let the counter in between two
            # bytecodes at most, so a count of five inside one frame never comes.
            while counted[0] < 5:
                counted[0] = 0
                inside.set()
                h.frame()
                inside.clear()
    finally:
        stop.set()
        worker.join()


# the shape a procedural host renders: two passes that ping-pong, a mask fed as
# bytes, a float32 array, and per-pass params
HOST_SCENE = {
    "version": 1,
    "passes": [
        {
            "name": "hold",
            "channels": [{"buffer": "main"}, {"buffer": "hold"}],
            "params": {"snap": {"type": "float", "value": 0.0}},
            "fragment": (
                "void main(){ gl_FragColor = mix(texture2D(iChannel1, uv),"
                " texture2D(iChannel0, uv), snap); }"
            ),
        },
        {
            "name": "main",
            "channels": [{"buffer": "hold"}],
            "arrays": [
                {"name": "text", "size": [8, 4]},
                {"name": "state", "size": 2},
            ],
            "params": {
                "tint": {"type": "vec3", "value": [1.0, 0.0, 0.0]},
                "lines": {"type": "float", "value": 1.0},
            },
            "fragment": (
                "void main(){ gl_FragColor = vec4(tint * (text_sample2(uv)"
                " * lines + state_at(0)), 1.0); }"
            ),
        },
    ],
}


def test_a_host_feeds_a_mask_a_state_and_per_pass_params():
    mask = bytes([255] * 32)
    with ghee.open(HOST_SCENE, width=16, height=16) as h:
        h.feed("text", mask)
        h.feed("state", array.array("f", [0.0, 0.0]))
        h.set_param("snap", 1.0, in_pass="hold")
        h.set_param("tint", [1.0, 0.0, 0.0], in_pass="main")
        h.frame()
        assert pixel(h) == (255, 0, 0, 255)
        h.set_param("lines", 0.0, in_pass="main")
        h.feed("state", array.array("f", [0.5, 0.0]))
        h.frame()
        assert abs(pixel(h)[0] - 128) <= 2
        # a clean tick feeds nothing and draws nothing
        held = h.pixels()
        h.skip()
        assert h.pixels() == held


FONT = "fonts/DejaVuSans-Latin.ttf"

# the words at 24 pixels per em, from the top left of the frame
WORDS = {
    "version": 1,
    "assets": {"sans": {"file": FONT}},
    "passes": [
        {
            "name": "main",
            "fragment": (
                "void main(){ vec2 px = vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y);"
                " gl_FragColor = vec4(vec3(words_cover(px / 24.0, 24.0)), 1.0); }"
            ),
            "layers": [
                {
                    "name": "words",
                    "text": "ghee",
                    "font_family": "sans",
                    "visible": False,
                }
            ],
        }
    ],
}


def test_new_words_lay_out_on_the_next_frame(monkeypatch):
    monkeypatch.chdir(REPO / "ghee" / "examples")
    with ghee.open(WORDS, width=96, height=32, timestep=0) as h:
        h.frame()
        ghee_drawn = h.pixels()
        assert max(ghee_drawn[0::4]) == 255  # full cover inside a stroke
        h.layers.words.text = "lettering"
        h.frame()
        assert h.pixels() != ghee_drawn
        h.layers.words.text = "ghee"
        h.frame()
        assert h.pixels() == ghee_drawn


def test_a_dict_scene_finds_its_font_in_base_dir_else_the_working_directory(
    tmp_path, monkeypatch
):
    (tmp_path / "fonts").mkdir()
    shutil.copy(REPO / "ghee" / "examples" / FONT, tmp_path / "fonts")
    with ghee.open(scene_file(tmp_path, WORDS), width=96, height=32, timestep=0) as h:
        h.frame()
        want = h.pixels()
    for cwd, base_dir in ((REPO, tmp_path), (tmp_path, None)):
        monkeypatch.chdir(cwd)
        with ghee.open(WORDS, width=96, height=32, timestep=0, base_dir=base_dir) as h:
            h.frame()
            assert h.pixels() == want


def png_pixels(data):
    """(width, height, rows of RGBA bytes) of a PNG of filter type 0, read with
    zlib alone."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    at, idat, head = 8, b"", None
    while at < len(data):
        (n,) = struct.unpack(">I", data[at : at + 4])
        tag, body = data[at + 4 : at + 8], data[at + 8 : at + 8 + n]
        if tag == b"IHDR":
            head = struct.unpack(">IIBB", body[:10])
        elif tag == b"IDAT":
            idat += body
        at += 12 + n
    w, h, depth, color = head
    assert (depth, color) == (8, 6)
    raw = zlib.decompress(idat)
    rows = [raw[y * (w * 4 + 1) : (y + 1) * (w * 4 + 1)] for y in range(h)]
    assert all(r[0] == 0 for r in rows)
    return w, h, b"".join(r[1:] for r in rows)


def test_save_writes_the_last_frame_numbered_by_its_tick(tmp_path):
    with ghee.open(VECS, width=8, height=1, timestep=0) as h:
        with pytest.raises(RuntimeError, match="nothing is drawn yet"):
            h.save(tmp_path / "early.png")
        h.frame()
        h.frame()
        h.save(tmp_path / "frame.####.png")
        h.save(str(tmp_path / "frame.jpg"))
        with pytest.raises(ValueError) as raised:
            h.save(tmp_path / "frame.bmp")
        assert str(raised.value) == (
            f"save writes a .png or a .jpg file, got {tmp_path}/frame.bmp"
        )
        assert png_pixels((tmp_path / "frame.0001.png").read_bytes()) == (
            8,
            1,
            h.pixels(),
        )
    jpg = (tmp_path / "frame.jpg").read_bytes()
    assert jpg[:2] == b"\xff\xd8"
    sof = jpg.index(b"\xff\xc0")
    assert struct.unpack(">HH", jpg[sof + 5 : sof + 9]) == (1, 8)


def test_open_takes_a_ghee_scene_and_a_path_like(tmp_path):
    """A ghee.Scene renders as its dict does, and a pathlib path as its str."""
    (tmp_path / "fonts").mkdir()
    shutil.copy(REPO / "ghee" / "examples" / FONT, tmp_path / "fonts")
    layer = ghee.Layer("words", "ghee", visible=False, font_family="sans")
    scene = ghee.Scene(
        assets=[ghee.Asset("sans", pathlib.Path(FONT))],
        passes=[ghee.Pass("main", WORDS["passes"][0]["fragment"], layers=[layer])],
    )
    assert scene.to_dict()["assets"] == {"sans": {"file": FONT}}
    frames = []
    for source in (scene, scene.to_dict()):
        with ghee.open(source, width=96, height=32, timestep=0, base_dir=tmp_path) as h:
            h.frame()
            frames.append(h.pixels())
    with ghee.open(
        pathlib.Path(scene_file(tmp_path, WORDS)), width=96, height=32, timestep=0
    ) as h:
        h.frame()
        frames.append(h.pixels())
    assert frames[0] == frames[1] == frames[2]
    assert max(frames[0][0::4]) == 255


def test_a_scene_file_resolves_its_font_beside_itself(tmp_path):
    (tmp_path / "fonts").mkdir()
    shutil.copy(REPO / "ghee" / "examples" / FONT, tmp_path / "fonts")
    with ghee.open(scene_file(tmp_path, WORDS), width=96, height=32) as h:
        h.frame()
        assert max(h.pixels()[0::4]) == 255


HALF = {
    "version": 1,
    "passes": [
        {
            "name": "main",
            "fragment": "void main(){ gl_FragColor = vec4(0.25, 0.5, 0.0, 0.5); }",
        }
    ],
}


def reads_back(px, want):
    """Each component of `px` is its value in `want` times 255, rounded either
    way."""
    return len(px) == len(want) and all(abs(p - v * 255) < 1 for p, v in zip(px, want))


def test_a_frame_is_opaque_unless_open_keeps_the_alpha():
    with ghee.open(HALF, width=8, height=8) as h:
        h.frame()
        assert reads_back(pixel(h), (0.25, 0.5, 0.0, 1.0))
    with ghee.open(HALF, width=8, height=8, alpha=True) as h:
        h.frame()
        assert reads_back(pixel(h), (0.25, 0.5, 0.0, 0.5))


def test_image_is_the_last_frame_as_a_pillow_image_with_unpremultiplied_colour():
    image = pillow()
    with ghee.open(HALF, width=8, height=4) as h:
        h.frame()
        got = h.image()
        assert isinstance(got, image.Image)
        assert (got.mode, got.size) == ("RGBA", (8, 4))
        assert got.tobytes() == h.pixels()
    # The kept alpha holds the colour premultiplied, (0.25, 0.5, 0) at 0.5,
    # and Pillow holds it unpremultiplied.
    with ghee.open(HALF, width=8, height=4, alpha=True) as h:
        h.frame()
        want = (0.5, 1.0, 0.0, 0.5)
        px = h.image().getpixel((0, 0))
        assert all(abs(p - v * 255) <= 2 for p, v in zip(px, want)), px


def test_alpha_names_one_form_for_every_output(tmp_path):
    """HALF is orange at half alpha: (0.25, 0.5, 0) premultiplied, (0.5, 1, 0)
    unpremultiplied. True keeps each output's own form, and a form's name
    holds for raw frames, Pillow images and `ghee render -o -`. A PNG holds
    alpha unpremultiplied, so a save under "premultiplied" is refused, and
    so is `ghee render --alpha premultiplied` into a PNG."""
    image = pillow()
    pre, unpre = (0.25, 0.5, 0.0, 0.5), (0.5, 1.0, 0.0, 0.5)
    forms = {
        True: ((), pre, unpre),
        "premultiplied": (("premultiplied",), pre, pre),
        "unpremultiplied": (("unpremultiplied",), unpre, unpre),
    }
    scene = scene_file(tmp_path, HALF)

    def close(px, want):
        return all(abs(p - v * 255) <= 2 for p, v in zip(px, want))

    for alpha, (flag, raw, picture) in forms.items():
        png = tmp_path / "f.png"
        with ghee.open(HALF, width=8, height=4, alpha=alpha) as h:
            h.frame()
            assert close(pixel(h), raw), (alpha, pixel(h))
            got = h.image().getpixel((0, 0))
            if raw == picture:
                assert got == pixel(h), (alpha, got, pixel(h))
            if alpha == "premultiplied":
                with pytest.raises(ValueError, match="unpremultiplied"):
                    h.save(png)
                continue
            h.save(png)
        saved = image.open(png).getpixel((0, 0))
        assert close(saved, picture), alpha
        assert got == saved, (alpha, got, saved)
        ran = subprocess.run(
            [
                str(pathlib.Path(sys.executable).parent / "ghee"),
                "render",
                scene,
                "-o",
                "-",
                "--size",
                "8x4",
                "--alpha",
                *flag,
            ],
            capture_output=True,
            check=True,
        )
        assert close(ran.stdout[:4], raw), (alpha, tuple(ran.stdout[:4]))
    ran = subprocess.run(
        [str(pathlib.Path(sys.executable).parent / "ghee"), "render", scene]
        + ["-o", str(tmp_path / "f.png"), "--alpha", "premultiplied"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert ran.returncode == 1, ran.stderr
    assert "holds alpha unpremultiplied" in ran.stderr
    with pytest.raises(ValueError):
        ghee.open(HALF, width=8, height=4, alpha="linear")


def test_image_without_pillow_says_that_it_needs_pillow(monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL.Image", None)
    with ghee.open(HALF, width=8, height=4) as h:
        h.frame()
        with pytest.raises(ImportError) as raised:
            h.image()
    assert str(raised.value) == (
        "image() needs Pillow, an optional dependency of ghee-render"
    )


def test_a_layer_draws_its_words_in_its_color(monkeypatch):
    monkeypatch.chdir(REPO / "tools" / "fixtures" / "fonts")
    scene = ghee.Scene(
        assets=[ghee.Asset("sans", "DejaVuSans-ASCII.ttf")],
        passes=[
            ghee.Pass(
                "main",
                "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                layers=[
                    ghee.Layer(
                        "words",
                        "I",
                        x=4,
                        y=0,
                        font_size=24,
                        color="#ff0000",
                        font_family="sans",
                    )
                ],
            )
        ],
    )
    with ghee.open(scene, width=48, height=32, timestep=0) as h:
        h.frame()
        px = h.pixels()
    assert (255, 0, 0) in {tuple(px[i : i + 3]) for i in range(0, len(px), 4)}


def test_a_layer_field_reaches_the_engine_live():
    # DejaVu Sans draws the layer's words, red over black.
    words = {
        "version": 1,
        "passes": [
            {
                "name": "main",
                "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                "layers": [
                    {
                        "name": "t",
                        "text": "I",
                        "font_size": 24,
                        "color": "#ff0000",
                        "glow": {},
                    }
                ],
            }
        ],
    }
    with ghee.open(words, width=32, height=32, timestep=0) as h:
        h.frame()
        assert max(h.pixels()[0::4]) == 255
        h.set_layer("t", "color", (0, 1, 0))
        h.set_layer("t", "visible", False, effect="glow")
        h.frame()
        assert max(h.pixels()[0::4]) == 0 and max(h.pixels()[1::4]) == 255
        with pytest.raises(ValueError, match="layer 't' keeps its name"):
            h.set_layer("t", "name", "u")


def test_a_shape_field_reaches_the_engine_live():
    # A red rect fills the frame, and its border lies inside its edge.
    shapes = {
        "version": 1,
        "passes": [
            {
                "name": "main",
                "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                "layers": [{"name": "box", "shape": "rect", "color": "red"}],
            }
        ],
    }
    with ghee.open(shapes, width=8, height=8, timestep=0) as h:
        h.frame()
        assert tuple(h.pixels()[0:3]) == (255, 0, 0)
        h.layers.box.border.color = "lime"
        h.frame()
        assert tuple(h.pixels()[0:3]) == (0, 255, 0)
        with pytest.raises(ValueError, match="layer 'box' has no field 'points'"):
            h.layers.box.points = [(0, 0), (1, 1)]


def test_a_handle_sets_a_layer_field_by_attribute_or_by_key():
    # DejaVu Sans draws the layer's words, red over black, with an
    # orange glow.
    words = {
        "version": 1,
        "passes": [
            {
                "name": "main",
                "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                "layers": [
                    {
                        "name": "t",
                        "text": "I",
                        "font_size": 24,
                        "color": "#ff0000",
                        "glow": {},
                    }
                ],
            }
        ],
    }

    def peaks(h):
        h.frame()
        px = h.pixels()
        return [max(px[c::4]) for c in range(3)]

    with ghee.open(words, width=32, height=32, timestep=0) as h:
        h.layers.t.color = "#00ff00"
        h.layers["t"]["glow"]["visible"] = False
        assert peaks(h) == [0, 255, 0]
        h.layers["t"]["color"] = "#0000ff"
        h.layers.t.glow.visible = True
        red, _, blue = peaks(h)
        assert red > 0 and blue == 255
        h.layers.t.text = ""
        assert peaks(h) == [0, 0, 0]


def test_a_handle_reads_a_layer_field_as_its_value():
    """Python's data model: an attribute that takes a value gives it back,
    an int as an int, and a missing layer raises KeyError by key,
    AttributeError by name."""
    scene = {
        "version": 1,
        "passes": [
            {"name": "main", "layers": [{"name": "t", "text": "I", "x": 3, "glow": {}}]}
        ],
    }
    with ghee.open(scene, width=16, height=16) as h:
        assert (h.layers.t.text, h.layers.t.x, h.layers.t.y) == ("I", 3, None)
        assert type(h.layers.t.x) is int
        assert h.layers.t.glow.strength is None
        h.layers.t.x = 5
        h.layers.t.glow.strength = 0.5
        assert (h.layers["t"]["x"], h.layers.t.glow.strength) == (5, 0.5)
        assert type(h.layers.t.x) is int
        h.layers.t.x = None
        assert h.layers.t.x is None
        with pytest.raises(AttributeError):
            _ = h.layers.nope
        with pytest.raises(KeyError):
            h.layers["nope"]


def test_open_with_no_scene_clears_to_black():
    with ghee.open(width=8, height=4) as h:
        h.frame()
        assert set(h.pixels()) == {0, 255}
        assert all(pixel(h, i) == (0, 0, 0, 255) for i in range(32))


def test_a_tick_carries_the_dt_that_its_frame_draws_with(clock):
    """Frame 0 draws with 1/60, and each later frame with the time since the
    frame before, which the tick opened by _tick() pins."""
    now = 0.0
    clock(now)
    with ghee.open(SHOWS_DT, width=4, height=4) as h:
        ticks = []
        for wait in (0.0, 0.03, 0.12, 0.06):
            now += wait
            clock(now)
            opened = h._tick()
            now += 0.02
            clock(now)
            tick = h.frame()
            assert tick == opened
            ticks.append((tick.dt, pixel(h)[0]))
    assert [dt for dt, _ in ticks] == pytest.approx([1.0 / 60.0, 0.05, 0.14, 0.08])
    for dt, red in ticks:
        assert abs(dt * 4 * 255 - red) <= 1.5, ticks


def test_rotate_turns_the_frame_clockwise_and_the_scene_renders_turned():
    # The scene, 2 wide and 4 tall, is red on its left half and blue on its
    # right, so the frame, 4 wide and 2 tall, is red on top.
    halves = {
        "version": 1,
        "passes": [
            {
                "name": "main",
                "fragment": "void main(){ gl_FragColor = uv.x < 0.5 ?"
                " vec4(1, 0, 0, 1) : vec4(0, 0, 1, 1); }",
            }
        ],
    }
    with ghee.open(halves, width=4, height=2, rotate=90) as h:
        assert h.size == (4, 2)
        h.frame()
        assert [pixel(h, i) for i in range(8)] == [(255, 0, 0, 255)] * 4 + [
            (0, 0, 255, 255)
        ] * 4
    with pytest.raises(ValueError) as raised:
        ghee.open(halves, width=4, height=2, rotate=45)
    assert str(raised.value).startswith("rotate takes 0, 90, 180 or 270")


def test_a_headless_screen_reads_no_brightness_and_refuses_one():
    with ghee.open(width=8, height=8) as h:
        assert h.brightness is None
        with pytest.raises(ValueError) as raised:
            h.brightness = 0.5
    assert str(raised.value).startswith("brightness needs a KMS display")


def test_dri_names_the_device_that_renders():
    with pytest.raises(OSError, match="/dev/dri/no-such-node"):
        ghee.open(width=8, height=8, dri="/dev/dri/no-such-node")


def test_a_display_node_that_wont_open_raises_the_oserror_of_its_errno():
    """PEP 3151: OSError picks the subclass of the open's errno."""
    with pytest.raises(FileNotFoundError, match="/dev/dri/no-such-card"):
        ghee.open(display="kms", dri="/dev/dri/no-such-card")


def test_measure_gives_the_size_of_the_words_and_raises_for_an_unknown_font():
    with ghee.open(width=128, height=32) as h:
        w, ht = h.measure("Hi", font_size=20)
        assert h.measure("Hi Hi", font_size=20)[0] > w > 0
        assert h.measure("Hi", font_size=40) == pytest.approx((2 * w, 2 * ht))
        # with no size the block fits the screen
        fit = h.measure("a wide line of words")
        assert fit[0] <= 128 and fit[1] <= 32
        # a list draws each character from the first font that holds it
        fonts = ["Pixel Operator 8", "DejaVu Sans"]
        pixel = h.measure("Hi", font_size=16, font_family=fonts[0])
        dejavu = h.measure("\u00a7", font_size=16, font_family=fonts[1])
        whole = h.measure("Hi\u00a7", font_size=16, font_family=fonts)
        assert whole == pytest.approx((pixel[0] + dejavu[0], pixel[1]))
        with pytest.raises(ValueError) as raised:
            h.measure("Hi", font_family="serif")
    assert str(raised.value).startswith("font_family 'serif' is no asset of the scene")


# The words with which the scene format refuses a font_family, before the
# value.
NO_FAMILY = (
    "font_family needs the name of a font asset or a .ttf or .otf file, or a "
    "list of them, got "
)


def test_an_empty_list_of_fonts_is_refused_as_the_scene_format_refuses_it():
    """A measure, a drawing's call and a Layer refuse [] in the same words,
    and a measure refuses a list with an empty name too."""
    with ghee.open(width=16, height=8) as h:
        for family, shown in (
            ([], "[]"),
            (("Pixel Operator 8", ""), "['Pixel Operator 8', '']"),
        ):
            with pytest.raises(ValueError) as raised:
                h.measure("Hi", font_family=family)
            assert str(raised.value) == NO_FAMILY + shown
        with pytest.raises(ValueError) as raised, h.draw() as d:
            d.text("hi", font_family=[])
        assert str(raised.value) == "layer 'text' " + NO_FAMILY + "[]"
    layer = ghee.Layer("words", "hi", font_family=[])
    scene = ghee.Scene(passes=[ghee.Pass("main", "void main(){}", layers=[layer])])
    with pytest.raises(ValueError) as raised:
        ghee.open(scene, width=8, height=8)
    assert str(raised.value) == "scene: layer 'words' " + NO_FAMILY + "[]"


def test_a_handles_layers_list_their_names():
    scene = {
        "version": 1,
        "passes": [
            {
                "name": "main",
                "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                "layers": [
                    {"name": "title", "text": "hi"},
                    {"name": "bar", "shape": "rect"},
                ],
            }
        ],
    }
    with ghee.open(scene, width=16, height=16) as h:
        with h.draw() as d:
            d.circle((8, 8), 4, name="dot")
        assert list(h.layers) == ["title", "bar", "dot"]
        assert len(h.layers) == 3
        assert "bar" in h.layers and "nope" not in h.layers
        assert repr(h.layers) == "<layers title, bar, dot>"
        with pytest.raises(TypeError, match="a layer's name is a str, got int"):
            h.layers[0]


def test_a_screen_with_no_display_takes_blank_as_a_no_op():
    with ghee.open(RED, width=4, height=4) as h:
        h.blank = True
        assert h.blank is False
        h.feed("level", LEVEL)
        h.frame()
        assert pixel(h) == (255, 0, 0, 255)
