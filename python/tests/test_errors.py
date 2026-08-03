"""What a caller gets wrong, and what it reads.

A scene, a GLSL or a param error is a ValueError. A driver, a system or a
thread error is a RuntimeError.
"""

import json
import queue
import subprocess
import threading

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
    with ghee.open(RED, width=8, height=8) as h:
        with pytest.raises(ValueError):
            h.feed("level", "eight")


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
    answers = queue.Queue()
    with ghee.open(RED, width=8, height=8) as h:

        def elsewhere():
            for call in [
                h.frame,
                h.skip,
                h.pixels,
                h.close,
                lambda: h.feed("level", [1.0] * 8),
            ]:
                try:
                    call()
                    answers.put(None)
                except BaseException as e:
                    answers.put(e)

        worker = threading.Thread(target=elsewhere)
        worker.start()
        worker.join()
        while not answers.empty():
            raised = answers.get()
            assert isinstance(raised, RuntimeError), raised
            assert "thread-affine" in str(raised)
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
