"""The handle: what open, frame, skip, set_param and feed do."""

import array
import json
import pathlib
import shutil
import struct
import threading
import time
import zlib

import pytest
from conftest import RED, REPO, VECS, near, quarters, scene_file

import ghee

LEVEL = array.array("f", [1.0] * 8)


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
        assert tick["frame"] == 0
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


def test_a_param_reaches_the_shader_live():
    with ghee.open(RED, width=8, height=8) as h:
        h.feed("level", LEVEL)
        h.set_param("gain", 0.5)
        h.frame()
        assert abs(pixel(h)[0] - 128) <= 2
        h.set_param("gain", 1.0, pass_="main")
        h.feed("level", LEVEL)
        h.frame()
        assert pixel(h)[0] == 255
        h.set_param("main.gain", 0.0)
        h.feed("level", LEVEL)
        h.frame()
        assert pixel(h)[0] == 0


def test_vec_and_color_values_reach_the_shader():
    built = ghee.Scene(
        passes=[
            ghee.Pass(
                "main",
                VECS["passes"][0]["code"],
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
            h.set_param("b", (0.0, 0.25, 1.0), pass_="main")
            h.set_param("main.c", [0.5, 1.0, 0.0, 1.0])
            h.set_param("tint", [0.0, 1.0, 0.5])
            h.frame()
            got = quarters(h)
            assert near(got, [[1, 0, 0], [0, 0.25, 1], [0.5, 1, 0], [0, 1, 0.5]]), got


def test_a_feed_takes_any_buffer_of_values():
    half = bytes([128] * 8)
    for values in [LEVEL, [1.0] * 8, memoryview(bytes([255] * 8))]:
        with ghee.open(RED, width=8, height=8) as h:
            h.feed("level", values)
            h.frame()
            assert pixel(h)[0] == 255
    # a uint8 buffer reaches the texture as byte / 255
    with ghee.open(RED, width=8, height=8) as h:
        h.feed("level", half)
        h.frame()
        assert abs(pixel(h)[0] - 128) <= 2


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
        assert tick["frame"] == 1
        assert h.pixels() == drawn
        assert h.frame()["frame"] == 2


def test_a_tick_carries_the_frame_the_time_and_the_step():
    with ghee.open(RED, width=8, height=8, timestep=0.1) as h:
        ticks = [h.frame(), h.skip(), h.frame()]
    assert [t["frame"] for t in ticks] == [0, 1, 2]
    assert [round(t["time"], 6) for t in ticks] == [0.0, 0.1, 0.2]
    assert [t["dt"] for t in ticks] == [0.1, 0.1, 0.1]


def test_a_step_no_frame_may_hand_a_scene():
    # one frame must never step a simulation by seconds
    with ghee.open(RED, width=8, height=8, timestep=1.0) as h:
        assert h.frame()["dt"] == 0.25


def test_an_unpinned_tick_reads_the_wall_clock():
    with ghee.open(RED, width=8, height=8, fps=60) as h:
        first = h.frame()
        second = h.frame()
        third = h.frame()
    assert first["time"] == 0.0
    assert 0.0 < second["time"] < third["time"] < 5.0
    assert first["dt"] == 1.0 / 60.0  # nothing is measured before the second frame
    assert 0.0 < third["dt"] <= 0.25


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


def test_the_module_names_its_version():
    """One release, two spellings: PEP 440 for the wheel, semver for the crate.

    The package surface is test_shim.py's.
    """
    from ghee import _native

    assert ghee.__version__.startswith(_native.__version__.split("-")[0])
    assert set(_native.__all__) == {"Handle", "open", "__version__"}


def test_a_frame_lets_the_other_threads_run():
    # the host's own threads keep working while a paced frame renders and waits
    counted = []
    stop = threading.Event()

    def count():
        while not stop.is_set():
            counted.append(time.monotonic())
            time.sleep(0.001)

    worker = threading.Thread(target=count)
    worker.start()
    try:
        with ghee.open(RED, width=64, height=64, fps=20) as h:
            h.feed("level", LEVEL)
            for _ in range(4):
                h.frame()
    finally:
        stop.set()
        worker.join()
    # four frames at 20 fps hold the loop about 0.2s, and the counter ran through it
    assert len(counted) > 20


# the shape a procedural host renders: two passes that ping-pong, a mask fed as
# bytes, a float32 array, and per-pass params
HOST_SCENE = {
    "version": 1,
    "passes": [
        {
            "name": "hold",
            "channels": [{"buffer": "main"}, {"buffer": "hold"}],
            "params": {"snap": {"type": "float", "value": 0.0}},
            "code": (
                "void main(){ gl_FragColor = mix(texture2D(iChannel1, v_texCoord),"
                " texture2D(iChannel0, v_texCoord), snap); }"
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
                "hotColor": {"type": "vec3", "value": [1.0, 0.0, 0.0]},
                "visLines": {"type": "float", "value": 1.0},
            },
            "code": (
                "void main(){ gl_FragColor = vec4(hotColor * (text_sample2(v_texCoord)"
                " * visLines + state_at(0)), 1.0); }"
            ),
        },
    ],
}


def test_a_host_feeds_a_mask_a_state_and_per_pass_params():
    mask = bytes([255] * 32)
    with ghee.open(HOST_SCENE, width=16, height=16) as h:
        h.feed("text", mask)
        h.feed("state", array.array("f", [0.0, 0.0]))
        h.set_param("snap", 1.0, pass_="hold")
        h.set_param("hotColor", [1.0, 0.0, 0.0], pass_="main")
        h.frame()
        assert pixel(h) == (255, 0, 0, 255)
        h.set_param("visLines", 0.0, pass_="main")
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
    "texts": {"words": {"value": "ghee", "font": "sans"}},
    "passes": [
        {
            "name": "main",
            "code": (
                "void main(){ vec2 px = floor(v_texCoord * resolution) + 0.5;"
                " gl_FragColor = vec4(vec3(words_cover(px / 24.0, 24.0)), 1.0); }"
            ),
        }
    ],
}


def test_new_words_lay_out_on_the_next_frame(monkeypatch):
    monkeypatch.chdir(REPO / "ghee" / "examples")
    with ghee.open(WORDS, width=96, height=32, timestep=0) as h:
        h.frame()
        ghee_drawn = h.pixels()
        assert max(ghee_drawn[0::4]) == 255  # full cover inside a stroke
        h.set_param("words", "lettering")
        h.frame()
        assert h.pixels() != ghee_drawn
        h.set_param("words", "ghee")
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
    scene = ghee.Scene(
        assets=[ghee.Asset("sans", pathlib.Path(FONT))],
        texts=[ghee.Text("words", "sans", "ghee")],
        passes=[ghee.Pass("main", WORDS["passes"][0]["code"])],
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
            "code": "void main(){ gl_FragColor = vec4(0.25, 0.5, 0.0, 0.5); }",
        }
    ],
}


def test_a_frame_is_opaque_unless_open_keeps_the_alpha():
    with ghee.open(HALF, width=8, height=8) as h:
        h.frame()
        assert pixel(h) == (64, 128, 0, 255)
    with ghee.open(HALF, width=8, height=8, alpha=True) as h:
        h.frame()
        r, g, b, a = pixel(h)
        assert (r, g, b) == (64, 128, 0)
        assert abs(a - 128) <= 1


def test_a_text_draws_a_letter_and_an_emoji_from_its_list_of_fonts(monkeypatch):
    """The I comes from sans, white, and the emoji from the emoji font, in its
    own colours to the right of the I."""
    monkeypatch.chdir(REPO / "tools" / "fixtures" / "fonts")
    scene = ghee.Scene(
        assets=[
            ghee.Asset("sans", "DejaVuSans-ASCII.ttf"),
            ghee.Asset("emoji", "NotoColorEmoji-Smiles.ttf"),
        ],
        texts=[ghee.Text("words", ["sans", "emoji"], "I\U0001f600")],
        passes=[
            ghee.Pass(
                "main",
                "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                layers=[
                    ghee.Layer(
                        "words",
                        "void main(){ gl_FragColor = vec4(glyph_cover()); }",
                        "vec2 glyph_place(vec2 em) { return em * 24.0; }",
                    )
                ],
            )
        ],
    )
    with ghee.open(scene.to_dict(), width=48, height=32, timestep=0) as h:
        h.frame()
        px = h.pixels()
    at = [(i // 4 % 48, px[i : i + 3]) for i in range(0, len(px), 4)]
    letter = [x for x, (r, g, b) in at if r == g == b and r > 200]
    emoji = [x for x, (r, g, b) in at if r - b > 100]
    assert emoji
    assert any(x < min(emoji) for x in letter)
