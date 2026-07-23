"""H264 output: wire helpers (pure), the software encoder round-trip (needs
ffmpeg), and encoder selection. The hardware path needs /dev/video11 (a Pi);
here it must fail cleanly and fall back."""

import shutil
import struct
import subprocess

import numpy as np
import pytest

from ghee.h264 import (
    HW_COLOR,
    NAL_AUD,
    NAL_IDR,
    NAL_SPS,
    SoftwareH264,
    avcc,
    build_config,
    split_annexb,
    wire_frame,
)

SC = b"\x00\x00\x00\x01"


def test_split_annexb_basic():
    sps, pps, idr = b"\x67\x42\x00\x1e", b"\x68\xce\x06\xe2", b"\x65\x88\x80"
    au = SC + sps + b"\x00\x00\x01" + pps + SC + idr
    nals = split_annexb(au)
    assert nals == [sps, pps, idr]
    assert [x[0] & 0x1F for x in nals] == [NAL_SPS, 8, NAL_IDR]


def test_split_annexb_garbage_prefix_and_padding():
    idr = b"\x65\x01\x02"
    assert split_annexb(b"\xff\xfe" + SC + idr + b"\x00\x00") == [idr]
    assert split_annexb(b"no start code at all") == []


def test_avcc_length_prefixes():
    out = avcc([b"\x65\x01", b"\x41\x02\x03"])
    assert out == b"\x00\x00\x00\x02\x65\x01\x00\x00\x00\x03\x41\x02\x03"


def test_build_config_codec_string():
    sps = bytes([0x67, 0x42, 0xC0, 0x1E, 0xAA])
    pps = bytes([0x68, 0xCE, 0x06, 0xE2])
    import json

    cfg = json.loads(build_config(sps, pps, 320, 240))
    assert cfg["codec"] == "avc1.42c01e"
    assert cfg["codedWidth"] == 320 and cfg["codedHeight"] == 240
    import base64

    box = base64.b64decode(cfg["description"])
    assert box[0] == 1  # avcC version
    assert sps in box and pps in box


def test_build_config_colorspace():
    import json

    sps = bytes([0x67, 0x42, 0xC0, 0x1E, 0xAA])
    pps = bytes([0x68, 0xCE, 0x06, 0xE2])
    plain = json.loads(build_config(sps, pps, 320, 240))
    assert "colorSpace" not in plain
    hinted = json.loads(build_config(sps, pps, 320, 240, color=HW_COLOR))
    assert hinted["colorSpace"]["matrix"] == "smpte170m"
    assert hinted["colorSpace"]["fullRange"] is True


def test_wire_frame_framing():
    f = wire_frame(1, b"abc")
    assert f[0] == 1
    assert struct.unpack(">I", f[1:5])[0] == 3
    assert f[5:] == b"abc"


needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="no ffmpeg")


def _gradient_frames(w, h, n):
    for i in range(n):
        f = np.zeros((h, w, 4), dtype=np.uint8)
        f[:, :, 1] = int(255 * i / max(1, n - 1))  # green ramps over time
        f[:, :, 3] = 255
        yield f


@needs_ffmpeg
def test_software_roundtrip_decodes():
    w, h, n = 160, 120, 30
    enc = SoftwareH264(w, h, fps=30, bitrate=1_000_000)
    aus = []
    for f in _gradient_frames(w, h, n):
        enc.feed(f)
        au, _ = enc.get_au(timeout=2.0)
        if au:
            aus.append(au)
    aus.extend(enc.drain())
    enc.close()
    stream = b"".join(aus)
    assert len(aus) == n  # zerolatency: one AU per frame once flushed
    # first AU carries config + a keyframe
    types0 = [x[0] & 0x1F for x in split_annexb(aus[0])]
    assert NAL_SPS in types0 and NAL_IDR in types0
    assert all(split_annexb(au)[0][0] & 0x1F == NAL_AUD for au in aus)
    # the whole stream decodes back to n frames at the right size
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "csv=p=0",
            "-",
        ],
        input=stream,
        capture_output=True,
    )
    assert r.stdout.decode().strip() == f"{w},{h},{n}"


@needs_ffmpeg
def test_software_stream_is_tagged():
    enc = SoftwareH264(160, 120, fps=30, bitrate=1_000_000)
    f = np.zeros((120, 160, 4), dtype=np.uint8)
    f[:, :, 3] = 255
    for _ in range(5):
        enc.feed(f)
    stream = b"".join(enc.drain())
    enc.close()
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=color_space,color_range",
            "-of",
            "csv=p=0",
            "-",
        ],
        input=stream,
        capture_output=True,
    )
    assert r.stdout.decode().strip() == "tv,smpte170m"


@needs_ffmpeg
def test_software_red_stays_red():
    # solid red, decoded honoring the stream's own tags. Red catches channel
    # swaps and matrix errors that green frames are blind to.
    w, h = 160, 120
    enc = SoftwareH264(w, h, fps=30, bitrate=2_000_000)
    f = np.zeros((h, w, 4), dtype=np.uint8)
    f[:, :, 0] = 255
    f[:, :, 3] = 255
    for _ in range(10):
        enc.feed(f)
    stream = b"".join(enc.drain())
    enc.close()
    r = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            "-",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        input=stream,
        capture_output=True,
    )
    frames = np.frombuffer(r.stdout, np.uint8).reshape(-1, h, w, 3)
    mean = frames[-1].reshape(-1, 3).mean(axis=0)
    assert mean[0] > 200 and mean[1] < 50 and mean[2] < 50, mean


@needs_ffmpeg
def test_software_content_survives():
    # brightness ramp in, brightness ramp out: proves pixels reach the codec
    w, h, n = 160, 120, 20
    enc = SoftwareH264(w, h, fps=30, bitrate=2_000_000)
    for f in _gradient_frames(w, h, n):
        enc.feed(f)
    aus = list(enc.drain())
    enc.close()
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "-", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        input=b"".join(aus),
        capture_output=True,
    )
    frames = np.frombuffer(r.stdout, np.uint8).reshape(-1, h, w)
    assert frames.shape[0] == n
    assert frames[0].mean() < 60 and frames[-1].mean() > 120  # dark -> bright


def test_hardware_unavailable_falls_back(renderer):
    import os

    from ghee.h264 import HardwareH264, make_encoder

    if os.path.exists("/dev/video11"):
        pytest.skip("board has a hardware encoder; fallback not exercised")
    with pytest.raises(RuntimeError):
        HardwareH264(renderer, 64, 64)
    if shutil.which("ffmpeg"):
        enc = make_encoder(renderer, 64, 64)
        assert isinstance(enc, SoftwareH264)
        enc.close()
