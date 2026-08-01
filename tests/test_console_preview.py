"""What the console hands a browser for the preview, and how the page shows it.

The stream side runs for real (the encoder thread against a fed frame); the
page side is asserted against the served source, like the other web tests.
"""

import io
import os
import re
import threading
import time

import numpy as np
import pytest
from PIL import Image

from ghee.console import server, web_dir

# stream bitrate


def test_a_panel_sized_stream_keeps_the_configured_rate():
    assert server.stream_bitrate(128, 32, 30, 10_000_000) == 10_000_000
    assert server.stream_bitrate(320, 240, 30, 10_000_000) == 10_000_000


def test_a_2k_stream_is_lifted_off_its_pixel_rate():
    bps = server.stream_bitrate(2048, 2048, 30, 10_000_000)
    assert bps / (2048 * 2048 * 30) == pytest.approx(server.H264_BPP)
    assert bps > 4_000_000  # the rate serve used to hand every size


def test_the_derived_rate_stops_at_the_ceiling():
    assert (
        server.stream_bitrate(4096, 4096, 60, 10_000_000) == server.H264_BITRATE_CEILING
    )


def test_a_configured_rate_above_the_ceiling_still_wins():
    assert server.stream_bitrate(2048, 2048, 30, 100_000_000) == 100_000_000


# MJPEG fallback


@pytest.fixture
def encoder(monkeypatch):
    """The encoder thread, fed by hand. It only works while a viewer is
    counted, so the fixture claims one."""
    monkeypatch.setattr(server, "_nclients", 1)
    server._stopping.clear()
    server._latest["jpg"] = None
    server._latest["seq"] = 0
    server._raw["rgb"] = None
    server._raw["seq"] = 0
    t = threading.Thread(target=server.encoder_loop, daemon=True)
    t.start()
    yield
    server._stopping.set()
    with server._raw_lock:
        server._raw_lock.notify_all()
    t.join(timeout=5)
    server._stopping.clear()


def _feed(rgb):
    with server._raw_lock:
        server._raw["rgb"] = rgb
        server._raw["seq"] += 1
        server._raw_lock.notify_all()
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        with server._lock:
            if server._latest["jpg"]:
                return server._latest["jpg"]
        time.sleep(0.02)
    raise AssertionError("the encoder produced no preview frame")


def _frame(w, h):
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.stack([x % 256, y % 256, (x * y) % 256, np.full_like(x, 255)], axis=-1)
    return rgb.astype(np.uint8)


def test_the_preview_jpeg_carries_the_configured_quality(encoder):
    frame = _frame(96, 96)
    jpg = _feed(frame)
    want = io.BytesIO()
    Image.fromarray(frame, "RGBA").convert("RGB").save(
        want, "JPEG", quality=server.PREVIEW_QUALITY
    )
    assert jpg == want.getvalue()


def test_the_preview_quality_is_not_the_old_mush():
    # q50 on a 480px frame is what the fallback used to look like enlarged
    assert server.PREVIEW_QUALITY > 50


def test_a_frame_bigger_than_the_preview_cap_is_downscaled(monkeypatch, encoder):
    monkeypatch.setattr(server, "_preview_max", 120)
    im = Image.open(io.BytesIO(_feed(_frame(480, 240))))
    assert max(im.size) == 120
    assert im.size == (120, 60)


# the page


def _index():
    with open(os.path.join(web_dir(), "index.html"), encoding="utf-8") as f:
        return f.read()


def _css(src):
    return src[src.index("<style>") : src.index("</style>")]


def _fn(src, head):
    """The body of a top-level function, up to the line that closes it."""
    i = src.index(head)
    return src[i : src.index("\n}", i)]


def test_the_preview_is_not_capped_at_a_hardcoded_box():
    src = _index()
    assert "MAX = 360" not in src
    assert "clientWidth" in _fn(src, "function previewBox()")


def test_only_a_popped_out_preview_may_upscale():
    src = _index()
    box = _fn(src, "function previewBox()")
    assert box.count("fill: true") == 2  # the pip window, and fullscreen
    assert "fill: false" in _fn(src, "function inlineBox()")  # in the stage card
    assert "Math.min(1, s)" in _fn(src, "function fitPreview(")


def test_the_stage_keeps_its_footprint_while_the_preview_is_away():
    src = _index()
    assert 'id="popped"' in src
    hold = _fn(src, "function holdStagePlace(")
    assert "ph.hidden = !on" in hold
    assert "inlineBox()" in hold
    assert "holdStagePlace(!!pipWin)" in _fn(src, "function applyRotation()")


def test_nearest_neighbour_is_conditional_never_a_blanket_rule():
    src = _index()
    for m in re.finditer(r"image-rendering\s*:\s*pixelated", src):
        line = src[src.rfind("\n", 0, m.start()) + 1 : src.find("\n", m.end())]
        assert ".crisp" in line, line


def test_a_downscaled_or_proxied_preview_is_resampled_not_aliased():
    fn = _fn(_index(), "function crispPreview(")
    assert "srcW >= renderW" in fn  # a smaller proxy is never called native
    assert "cssW * dpr >= srcW" in fn  # and neither is a shrink


def test_the_preview_nodes_are_held_not_re_queried():
    # popping out moves them into another document, where a document-scoped
    # query stops finding them
    found = re.findall(r'\$\("(#stream|#streamcv|\.frame)"\)', _index())
    assert sorted(found) == sorted(["#stream", "#streamcv", ".frame"])


def test_the_pop_out_moves_the_live_canvas_and_never_re_encodes():
    src = _index()
    assert "captureStream" not in src
    assert "MediaRecorder" not in src
    assert "documentPictureInPicture.requestWindow" in src
    assert "win.document.body.appendChild(FRAME)" in src


def test_fullscreen_is_offered_wherever_document_pip_is_not():
    src = _index()
    assert 'HAS_PIP = "documentPictureInPicture" in window' in src
    assert "HAS_FS = !!SCREEN.requestFullscreen" in src
    sync = _fn(src, "function syncPopOut()")
    assert '$("#pipBtn").hidden = !HAS_PIP' in sync
    assert '$("#fsBtn").hidden = !HAS_FS' in sync


def test_a_closed_pop_out_puts_the_preview_back():
    src = _index()
    assert '"pagehide", () => { pipWin = null; homeFrame(); }' in src
    assert "SCREEN.insertBefore(FRAME" in _fn(src, "function homeFrame()")


def test_the_two_pop_outs_cannot_both_hold_the_preview():
    sync = _fn(_index(), "function syncPopOut()")
    assert '$("#pipBtn").disabled = fs' in sync
    assert '$("#fsBtn").disabled = !!pipWin' in sync


def test_the_info_row_names_the_preview_transport():
    src = _index()
    assert 'id="degraded"' not in src
    fn = _fn(src, "function refreshInfo()")
    assert 'row(out, "preview"' in fn
    assert "pv.kind" in fn


def test_the_mjpeg_fallback_survives_for_browsers_that_need_it():
    fallback = _fn(_index(), "function startPreview()")
    assert '"VideoDecoder" in window && window.isSecureContext' in fallback
    assert "fallbackToMjpeg(" in fallback


def test_every_css_variable_the_console_reads_is_defined():
    css = _css(_index())
    root = css[css.index(":root{") : css.index("}", css.index(":root{"))]
    defined = set(re.findall(r"(--[\w-]+)\s*:", root))
    used = set(re.findall(r"var\((--[\w-]+)\)", css))  # no fallback to save it
    assert used <= defined, used - defined


def test_no_css_variable_falls_back_to_a_literal_colour():
    # a themed value hiding behind a fallback ships one theme's colour to both
    for m in re.finditer(r"var\(--[\w-]+\s*,([^)]*)\)", _css(_index())):
        assert not re.search(r"#[0-9a-fA-F]{3,8}|rgba?\(", m.group(1)), m.group(0)


def test_the_overlay_controls_are_touch_sized_on_a_coarse_pointer():
    css = _css(_index())
    coarse = re.search(r"@media \(pointer:coarse\)\{ \.ovbtn\{([^}]*)\}", css)
    assert coarse, "the pop-out buttons have no coarse-pointer size"
    assert "width:40px" in coarse.group(1)
