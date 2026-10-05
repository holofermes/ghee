"""What a caller gets wrong, and what it reads.

A scene, a GLSL or a param error is a ValueError. A driver, a system or a
thread error is a RuntimeError.
"""

import concurrent.futures
import json
import shutil
import subprocess
import sys

import pytest
from conftest import BROKEN, RED, REPO, scene_file

import ghee

CLI = REPO / "target" / "release" / "ghee"


def test_bad_glsl_reads_the_same_as_the_command_line(tmp_path):
    path = scene_file(tmp_path, BROKEN)
    with pytest.raises(ValueError) as raised:
        ghee.open(path, width=8, height=8)
    message = str(raised.value)
    assert message.startswith("compile error: ")
    assert "no_such_symbol" in message
    if not CLI.exists():
        pytest.skip(f"no {CLI} to compare against")
    ran = subprocess.run(
        [str(CLI), "render", path, "-o", str(tmp_path / "out.png"), "--size", "8x8"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert ran.returncode == 1
    # the engine writes its own lines first, and the failure closes the output
    assert ran.stderr.strip().endswith(message)


def test_a_scene_that_is_no_scene(tmp_path):
    for source, why in [
        ("{not json", "invalid JSON"),
        (str(tmp_path / "gone.scene.json"), "no scene at"),
        (17, "scene takes a path"),
    ]:
        with pytest.raises(ValueError) as raised:
            ghee.open(source)
        assert why in str(raised.value)


def test_a_procedural_scene_runs_itself(tmp_path):
    path = tmp_path / "proc.py"
    path.write_text("#!/usr/bin/env python3\n")
    path.chmod(0o755)
    with pytest.raises(ValueError) as raised:
        ghee.open(str(path))
    assert "runs itself" in str(raised.value)


def test_a_param_the_scene_does_not_declare():
    with ghee.open(RED, width=8, height=8) as h:
        with pytest.raises(ValueError) as raised:
            h.set_param("nope", 1.0)
        assert "param 'nope' is not a param this scene declares" in str(raised.value)
        assert "gain" in str(raised.value)
        with pytest.raises(ValueError):
            h.set_param("gain", 1.0, pass_="nosuchpass")


def test_a_font_that_is_nowhere_draws_error_boxes(tmp_path, monkeypatch, capfd):
    monkeypatch.chdir(tmp_path)
    doc = {
        "version": 1,
        "assets": {"sans": {"file": "fonts/Missing.ttf"}},
        "passes": [
            {
                "name": "main",
                "code": (
                    "void main(){ vec2 px = floor(v_texCoord * resolution) + 0.5;"
                    " gl_FragColor = vec4(vec3(words_cover(px / 16.0, 16.0)), 1.0); }"
                ),
                "layers": [
                    {
                        "name": "words",
                        "text": "hi",
                        "font_family": "sans",
                        "visible": False,
                    }
                ],
            }
        ],
    }
    with ghee.open(doc, width=32, height=24, timestep=0) as h:
        h.frame()
        assert max(h.pixels()[0::4]) == 255  # a stripe of a box
        h.set_param("words", "ho")
        h.frame()
    # the first path tried is the working directory's
    assert (
        f"asset 'sans' ({tmp_path}/fonts/Missing.ttf): No such file or directory"
        in capfd.readouterr().err
    )


WIRE_WORDS = """#!{python}
import ghee

CODE = (
    "void main(){{ vec2 px = floor(v_texCoord * resolution) + 0.5;"
    " gl_FragColor = vec4(vec3(words_cover(px / 12.0, 12.0)), 1.0); }}"
)


@ghee.tick
def hook(t):
    if t.frame:
        ghee.set_param("words", "I" * 400_000)


LAYER = ghee.Layer("words", "ab", visible=False, font_family="sans")
ghee.run(
    ghee.Scene(
        passes=[ghee.Pass("main", CODE, layers=[LAYER])],
        assets=[ghee.Asset("sans", "{font}")],
    )
)
"""


def test_words_past_the_texture_from_a_procedural_scene_end_no_run(tmp_path):
    if not CLI.exists():
        pytest.skip(f"no {CLI} to run the scene")
    (tmp_path / "fonts").mkdir()
    shutil.copy(REPO / "ghee/examples/fonts/DejaVuSans-Latin.ttf", tmp_path / "fonts")
    path = tmp_path / "words.py"
    path.write_text(
        WIRE_WORDS.format(python=sys.executable, font="fonts/DejaVuSans-Latin.ttf")
    )
    path.chmod(0o755)
    ran = subprocess.run(
        [str(CLI), "render", str(path), "-o", str(tmp_path / "f.####.png")]
        + ["--size", "48x16", "--frames", "20", "--fps", "20", "--timestep", "0"],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stderr.count("layer 'words': the words need a ") == 1, ran.stderr
    # words that do not fit leave the last words
    frames = {(tmp_path / f"f.{i:04d}.png").read_bytes() for i in range(20)}
    assert len(frames) == 1


# a layer whose blend ghee does not know
MISTAKE = {
    "version": 1,
    "passes": [
        {
            "name": "main",
            "code": "void main(){ gl_FragColor = vec4(1.0); }",
            "layers": [
                {
                    "name": "words",
                    "text": "hi",
                    "font_family": "sans.ttf",
                    "blend": "mix",
                }
            ],
        }
    ],
}

MISTAKE_SCENE = """#!{python}
import ghee

ghee.run(
    ghee.Scene(
        passes=[
            ghee.Pass(
                "main",
                "void main(){{ gl_FragColor = vec4(1.0); }}",
                layers=[ghee.Layer("words", "hi", font_family="sans.ttf", blend="mix")],
            )
        ],
    )
)
"""


def test_a_scene_mistake_reads_as_ghee_wrote_it_in_open_a_direct_run_and_ghee_render(
    tmp_path,
):
    with pytest.raises(ValueError) as raised:
        ghee.open(MISTAKE, width=8, height=8)
    message = str(raised.value)
    assert "blend" in message
    path = tmp_path / "mistake.py"
    path.write_text(MISTAKE_SCENE.format(python=sys.executable))
    path.chmod(0o755)
    out = ["-o", str(tmp_path / "f.png"), "--size", "8x8"]
    direct = subprocess.run(
        [str(path), *out], capture_output=True, check=False, text=True, timeout=120
    )
    assert direct.returncode == 1
    assert direct.stderr.splitlines()[-1] == f"ValueError: {message}"
    if not CLI.exists():
        pytest.skip(f"no {CLI} to run the scene")
    ran = subprocess.run(
        [str(CLI), "render", str(path), *out],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert ran.returncode == 1
    assert ran.stderr.count(message.removeprefix("scene: ")) == 1, ran.stderr


def test_a_feed_the_scene_does_not_declare():
    with ghee.open(RED, width=8, height=8) as h:
        with pytest.raises(ValueError) as raised:
            h.feed("nope", [1.0])
        assert str(raised.value) == (
            "feed 'nope' is not an array this scene declares (level)"
        )


def test_a_feed_of_the_wrong_count():
    with ghee.open(RED, width=8, height=8) as h:
        with pytest.raises(ValueError) as raised:
            h.feed("level", [1.0, 1.0])
        assert str(raised.value) == "array 'level' takes 8 values, got 2"


def test_a_feed_that_carries_no_numbers():
    with ghee.open(RED, width=8, height=8) as h, pytest.raises(ValueError) as raised:
        h.feed("level", "eight")
    assert str(raised.value).startswith("feed 'level' takes")


def test_an_output_no_driver_answers_to():
    with pytest.raises(ValueError) as raised:
        ghee.open(RED, output="nosuchpanel")
    assert "no display driver 'nosuchpanel'" in str(raised.value)
    assert "st7789" in str(raised.value)


def test_an_option_the_driver_does_not_take():
    with pytest.raises(ValueError) as raised:
        ghee.open(RED, output="st7789", opts={"widht": 240})
    assert str(raised.value).startswith("opts: st7789 has no option widht")
    with pytest.raises(ValueError) as raised:
        ghee.open(RED, output="kms", opts={"width": 240})
    assert str(raised.value) == "opts: kms has no option width (takes: rgb565)"


def test_a_size_takes_both_sides():
    with pytest.raises(ValueError) as raised:
        ghee.open(RED, width=64)
    assert "width and height go together" in str(raised.value)


def test_a_call_from_another_thread():
    with ghee.open(RED, width=8, height=8) as h:
        calls = [h.frame, h.skip, h.pixels, h.close, lambda: h.feed("level", [1.0] * 8)]
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            raised = [pool.submit(call).exception() for call in calls]
        for e in raised:
            assert isinstance(e, RuntimeError), e
            assert "thread-affine" in str(e)
        # the handle still works on its own thread
        h.feed("level", [1.0] * 8)
        h.frame()


def test_a_closed_handle_says_so():
    h = ghee.open(RED, width=8, height=8)
    h.close()
    for call in [h.frame, h.skip, h.pixels]:
        with pytest.raises(RuntimeError) as raised:
            call()
        assert str(raised.value) == "this handle is closed"


def test_a_scene_with_a_filter_it_has_no_definition_for(tmp_path):
    doc = dict(RED)
    doc["filters"] = ["nosuchfilter"]
    with pytest.raises(ValueError) as raised:
        ghee.open(json.dumps(doc), width=8, height=8)
    assert str(raised.value).startswith("unknown filter: nosuchfilter (known: ")


def test_pixels_before_the_first_frame():
    with ghee.open(RED, width=8, height=8) as h:
        with pytest.raises(RuntimeError) as raised:
            h.pixels()
        assert str(raised.value) == "nothing is drawn yet"
