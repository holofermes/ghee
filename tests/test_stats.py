"""What /stats publishes and how the numbers are arrived at.

The rate is driven on a synthetic clock, so a bimodal frame sequence is exact
rather than approximately timed. The loop and the page are checked against the
real thing.
"""

import json
import os
import re
import threading
import time

import pytest

from ghee.console import core, server, web_dir
from ghee.console.store import SceneStore
from ghee.drivers.base import DisplayDriver
from ghee.output import ImageSequence, RawStream
from ghee.stats import (
    DEST_FILE,
    DEST_H264,
    DEST_KMS,
    DEST_MJPEG,
    DEST_PANEL,
    DEST_PIPE,
    FrameRate,
    rate,
)

# the rate


def _feed(fr, periods, seconds, t0=1000.0):
    """Tick `fr` at the given periods, cycling, for `seconds` of clock."""
    t = t0
    fr.reset(t)
    end = t + seconds
    i = 0
    while t < end:
        t += periods[i % len(periods)]
        fr.tick(t)
        i += 1
    return t, i


def test_a_bimodal_sequence_publishes_its_true_rate():
    # alternating 4 ms and 32 ms frames: 2 frames per 36 ms
    fr = FrameRate()
    t, n = _feed(fr, [0.004, 0.032], 4.0)
    true = 2.0 / 0.036
    assert fr.value(t) == pytest.approx(true, rel=0.02)
    # what a mean of reciprocals would have said instead
    assert fr.value(t) < 0.5 * ((1 / 0.004) + (1 / 0.032)) / 2


def test_a_constant_sequence_publishes_that_rate():
    fr = FrameRate()
    t, _ = _feed(fr, [0.02], 4.0)
    assert fr.value(t) == pytest.approx(50.0, rel=0.02)


def test_the_rate_is_completed_work_over_elapsed_time():
    assert rate(60, 6.0) == 10.0
    assert rate(1, 0.0) == 0.0
    assert rate(0, 5.0) == 0.0


def test_a_skipped_frame_invents_nothing():
    fr = FrameRate()
    fr.reset(0.0)
    for i in range(1, 51):
        fr.tick(i * 0.02)
    published = fr.value(1.0)
    assert published == pytest.approx(50.0)
    # a skipped frame is simply not a tick: the number it would have counted
    # into is left alone, not scaled by a factor no clock produced
    assert fr.value(1.02) == published
    assert fr.value(1.5) == published


def test_an_idle_stretch_falls_with_the_clock():
    fr = FrameRate()
    fr.reset(0.0)
    for i in range(1, 51):
        fr.tick(i * 0.02)  # the window closes at 1.0 s carrying nothing over
    assert fr.value(2.0) == 0.0
    assert fr.value(11.0) == 0.0
    fr.tick(11.1)  # one lone frame closes a ten-second window
    assert fr.value(11.1) == pytest.approx(1 / 10.1)
    assert fr.value(21.1) == 0.0


def test_a_counter_that_stops_mid_window_reads_zero():
    # the viewer that went away: ticks stop with a part-filled window, and
    # count/elapsed would decay towards zero without ever printing it
    fr = FrameRate()
    fr.reset(0.0)
    for i in range(1, 31):
        fr.tick(i * 0.01)  # 30 frames, window still open at 0.3 s
    assert fr.value(0.4) > 0
    assert fr.value(1.3) == 0.0
    assert fr.value(60.0) == 0.0


def test_a_fresh_counter_reports_the_window_it_has():
    fr = FrameRate()
    fr.reset(0.0)
    assert fr.value(0.5) == 0.0
    for i in range(1, 11):
        fr.tick(i * 0.02)
    assert fr.value(0.2) == pytest.approx(50.0)


def test_reset_drops_the_window_it_was_measuring():
    fr = FrameRate()
    t, _ = _feed(fr, [0.02], 4.0)
    assert fr.value(t) > 0
    fr.reset(t)
    assert fr.value(t + 2.0) == 0.0


def test_the_cli_summary_reports_the_same_rate_the_console_would(capsys):
    from argparse import Namespace

    from ghee import cli

    times = [0.004, 0.032] * 25  # the bimodal workload again, over 1.8 s of wall
    cli._print_stats(Namespace(stats=True), times, 1.8, [DEST_KMS])
    line = capsys.readouterr().err
    assert f"{rate(len(times), 1.8):.1f} fps" in line
    assert f"{50 / 1.8:.1f} fps" in line
    assert "in 1.80s" in line
    assert f"-> {DEST_KMS}" in line


# destination names


def test_every_sink_carries_its_own_destination():
    assert ImageSequence.dest == DEST_FILE
    assert RawStream.dest == DEST_PIPE
    assert DisplayDriver.dest == DEST_PANEL


def test_a_driver_inherits_the_panel_destination():
    import ghee.drivers  # noqa: F401  the registry is populated on driver import
    from ghee.drivers.base import _REGISTRY

    for cls in _REGISTRY.values():
        assert cls.dest == DEST_PANEL


@pytest.mark.parametrize(
    ("display", "h264", "mjpeg", "want"),
    [
        (False, 0, 0, []),
        (True, 0, 0, [DEST_KMS]),
        (True, 1, 0, [DEST_KMS, DEST_H264]),
        (False, 1, 1, [DEST_H264, DEST_MJPEG]),
        (False, 0, 1, [DEST_MJPEG]),
        (True, 1, 1, [DEST_KMS, DEST_H264, DEST_MJPEG]),
    ],
)
def test_out_names_every_destination_the_frames_reach(
    monkeypatch, display, h264, mjpeg, want
):
    fanout = server.H264Fanout()
    for _ in range(h264):
        fanout.add()
    monkeypatch.setattr(server, "_fanout", fanout)
    monkeypatch.setattr(server, "_display", display)
    monkeypatch.setattr(server, "_nclients", mjpeg)
    assert server.stats()["out"] == want


# the published field set


DOCUMENTED = {
    "fps",
    "frame_ms",
    "stream_fps",
    "out",
    "error",
    "error_line",
    "error_pass",
    "active",
    "display",
    "stale",
}


def test_stats_publishes_exactly_what_the_contract_documents():
    documented = set(re.findall(r"\w+", core.ConsoleHost.stats.__doc__.split("}")[0]))
    assert documented == DOCUMENTED
    assert set(server.stats()) == DOCUMENTED


def test_the_replaced_fields_are_gone_with_no_alias():
    s = server.stats()
    assert "render_ms" not in s
    assert "encode_ms" not in s
    src = open(server.__file__, encoding="utf-8").read()
    assert "render_ms" not in src and "encode_ms" not in src


def test_the_display_object_no_longer_claims_a_name():
    assert "name" not in server._stat["display"]


# the console loop


SCENE = {
    "version": 1,
    "passes": [{"name": "main", "code": "void main(){ gl_FragColor = vec4(1.0); }"}],
}


class StubRenderer:
    width, height, components = 64, 48, 4
    error_line = None
    error_pass = None

    def __init__(self):
        self.frames = 0

    def setup(self, scene):
        return ""

    def display_vrefresh(self):
        return 0

    def draw(self, provider=None):
        self.frames += 1

    def render(self, provider=None, out=None, out_size=None):
        self.frames += 1
        return out

    def set_param(self, *a):
        pass

    def set_scene_param(self, *a):
        pass

    def set_dt(self, value):
        pass


@pytest.fixture
def loop(monkeypatch, tmp_path):
    p = tmp_path / "demo.scene.json"
    p.write_text(json.dumps(SCENE), encoding="utf-8")
    store = SceneStore(str(tmp_path), read_only=True)
    store.scenes = dict.fromkeys(("demo", "other"), str(p))
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "W", 64)
    monkeypatch.setattr(server, "H", 48)
    monkeypatch.setattr(server, "_make_renderer", lambda: StubRenderer())
    monkeypatch.setattr(server, "_nclients", 0)
    monkeypatch.setattr(server, "_fanout", server.H264Fanout())
    server._stopping.clear()
    t = threading.Thread(target=server.render_loop, daemon=True)
    t.start()
    yield t
    server._stopping.set()
    server._q.put(("stop",))
    t.join(timeout=5)
    server._stopping.clear()
    while not server._q.empty():
        server._q.get_nowait()


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_a_running_loop_publishes_a_rate_and_a_frame_cost(loop):
    assert _wait(lambda: server.stats()["frame_ms"] > 0)
    assert _wait(lambda: server.stats()["fps"] > 0, timeout=6.0)


def test_a_scene_switch_resets_the_stats(monkeypatch, loop):
    resets = []
    real = server._reset_stats
    monkeypatch.setattr(server, "_reset_stats", lambda: (resets.append(1), real())[1])
    assert _wait(lambda: server._stat["frame_ms"] > 0)
    server._q.put(("switch", "other"))
    assert _wait(lambda: server._stat["active"] == "other")
    assert resets


def test_the_stats_a_new_run_starts_from_are_not_the_last_run_s():
    server._stat["frame_ms"] = 123.0
    server._frame_rate.tick()
    server._reset_stats()
    assert server._stat["frame_ms"] == 0.0
    assert server._stat["error"] == ""
    assert server.stats()["fps"] == 0.0
    assert "_reset_stats()" in open(server.__file__, encoding="utf-8").read()


# stream_fps


def _au(key=True):
    sc = b"\x00\x00\x00\x01"
    sps, pps = b"\x67\x42\x00\x1e", b"\x68\xce\x06\xe2"
    return sc + sps + sc + pps + sc + b"\x65\x88\x80" if key else sc + b"\x41\x88\x80"


def test_the_fanout_counts_what_it_hands_to_viewers():
    fanout = server.H264Fanout()
    fanout.add()
    for _ in range(4):
        fanout.push(_au())
    assert fanout.stream_fps() > 0


def test_nothing_is_streamed_to_nobody():
    fanout = server.H264Fanout()
    fanout.push(_au())
    assert fanout.stream_fps() == 0.0


def test_one_counter_covers_both_codecs():
    shared = FrameRate()
    fanout = server.H264Fanout(rate=shared)
    fanout.add()
    fanout.push(_au())
    assert fanout.rate is shared
    # the JPEG path ticks the same counter the console publishes
    assert server._fanout.rate is server._stream_rate


# the page


def _index():
    with open(os.path.join(web_dir(), "index.html"), encoding="utf-8") as f:
        return f.read()


def test_the_page_reads_the_fields_the_server_sends():
    src = _index()
    for field in ("s.fps", "s.frame_ms", "s.stream_fps", "s.out"):
        assert field in src
    assert "render_ms" not in src and "encode_ms" not in src


def test_a_missing_field_degrades_instead_of_throwing():
    src = _index()
    # the poll's catch also holds the error banner, so a throw here blinds the
    # one readout that would explain it
    assert re.search(r"const statNum = .*typeof v === \"number\"", src)
    assert not re.search(r"s\.\w+\.toFixed", src)
