"""CLI output features: --timestep reproducibility, -o - raw stdout (with EPIPE
handling), --save-opts parsing, and the async pooled ImageSequence writer.

The render-path tests spawn the CLI and compare against an in-process render on
the same GPU, so they need a GL device and skip cleanly without one. The parser
and writer tests are pure Python (the writer needs Pillow)."""

import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

from .helpers import scene

W, H, C = 64, 48, 4
SCENE = (
    '{"version":1,"passes":[{"name":"main","code":'
    '"void main(){ vec2 uv=gl_FragCoord.xy/resolution; '
    'gl_FragColor=vec4(uv, abs(sin(time)), 1.0);}"}]}'
)


def _cli(*args):
    return [sys.executable, "-m", "ghee.cli", "render", *args]


def test_parse_save_opts_types():
    from ghee.cli import _parse_save_opts

    d = _parse_save_opts("quality=90,lossless=True,method=fast,note=hi there")
    assert d == {
        "quality": 90,
        "lossless": True,
        "method": "fast",
        "note": "hi there",
    }
    assert isinstance(d["quality"], int)
    assert d["lossless"] is True


def test_parse_save_opts_malformed_exits():
    from ghee.cli import _parse_save_opts

    with pytest.raises(SystemExit):
        _parse_save_opts("quality=90,nope")


def test_timestep_reproducible(renderer):
    # frame i renders identically across two independent runs and matches a
    # fresh render pinned to the same time.
    sc = scene(
        [
            {
                "name": "main",
                "code": "void main(){ gl_FragColor=vec4(abs(sin(time)),0.2,0.4,1.0);}",
            }
        ]
    )
    assert renderer.setup(sc) == ""
    step = 0.1

    def run():
        out = []
        for i in range(5):
            renderer.set_time(i * step)
            out.append(renderer.render(None).copy())
        return out

    a, b = run(), run()
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
    renderer.set_time(3 * step)  # frame 3 alone equals frame 3 of the run
    assert np.array_equal(renderer.render(None), a[3])


def test_raw_stdout_matches_render(renderer, tmp_path):
    from ghee import load_scene

    path = tmp_path / "s.scene.json"
    path.write_text(SCENE)
    sc, _ = load_scene(str(path))
    assert renderer.setup(sc) == ""
    frames, step = 3, 0.25
    ref = b""
    for i in range(frames):
        renderer.set_time(i * step)
        ref += renderer.render(None).tobytes()

    res = subprocess.run(
        _cli(
            str(path),
            "-o",
            "-",
            "--frames",
            str(frames),
            "--timestep",
            str(step),
            "--size",
            f"{W}x{H}",
            "--components",
            str(C),
        ),
        capture_output=True,
    )
    assert res.returncode == 0, res.stderr.decode()
    assert len(res.stdout) == frames * W * H * C  # no header, no "wrote" leak
    assert res.stdout == ref


@pytest.mark.skipif(not shutil.which("head"), reason="no head")
def test_raw_stdout_epipe_clean(renderer, tmp_path):
    # a dead consumer ends the render like Ctrl-C: no traceback, clean exit.
    path = tmp_path / "s.scene.json"
    path.write_text(SCENE)
    p1 = subprocess.Popen(
        _cli(
            str(path),
            "-o",
            "-",
            "--frames",
            "100000",
            "--timestep",
            "0.01",
            "--size",
            f"{W}x{H}",
            "--components",
            str(C),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    p2 = subprocess.Popen(
        ["head", "-c", str(2 * W * H * C)],
        stdin=p1.stdout,
        stdout=subprocess.DEVNULL,
    )
    p1.stdout.close()  # only head holds the read end
    p2.wait()
    err = p1.stderr.read().decode()
    p1.stderr.close()
    assert p1.wait(timeout=30) == 0
    assert "Traceback" not in err
    assert "BrokenPipeError" not in err


def test_h264_without_any_encoder_exits_clean(renderer, tmp_path):
    # no board encoder and no ffmpeg: one line naming the fix, no stack.
    if os.path.exists("/dev/video11"):
        pytest.skip("board has a hardware encoder; the no-encoder path is unreachable")
    path = tmp_path / "s.scene.json"
    path.write_text(SCENE)
    p = subprocess.run(
        _cli(str(path), "--h264", "-", "--frames", "1", "--size", f"{W}x{H}"),
        capture_output=True,
        text=True,
        env=dict(os.environ, PATH=str(tmp_path / "no-ffmpeg-here")),
        timeout=60,
    )
    assert p.returncode == 1
    assert "Traceback" not in p.stderr
    assert "no H264 encoder" in p.stderr
    assert "apt install ffmpeg" in p.stderr


def test_render_library_error_exits_clean(tmp_path):
    # the clean exit is general, not an h264 special case: a scene the loader
    # rejects lands the same way, before any GL device is touched.
    bad = tmp_path / "bad.scene.json"
    bad.write_text('{"version":1,"passes":[{"name":"main"}]}')
    p = subprocess.run(
        _cli(str(bad), "-o", str(tmp_path / "x.png")),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 1
    assert "Traceback" not in p.stderr
    assert p.stderr.strip().startswith("render: ")
    assert "neither 'code'" in p.stderr


def test_image_sequence_write_byte_identical(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    from ghee.output import ImageSequence

    frames = [np.full((8, 6, 4), i * 25, dtype=np.uint8) for i in range(4)]
    for f in frames:
        f[..., 3] = 255
    seq = ImageSequence(str(tmp_path / "w.####.png"))
    for f in frames:
        seq.write(f)
    seq.close()
    for i, f in enumerate(frames):
        ref = tmp_path / f"ref.{i:04d}.png"
        Image.fromarray(f, "RGBA").save(str(ref))
        assert (tmp_path / f"w.{i:04d}.png").read_bytes() == ref.read_bytes()


def test_image_sequence_pool_recycles(tmp_path):
    # distinct content per frame through the acquire/submit pool, with a shallow
    # depth so buffers recycle every couple of frames; each file must still be
    # byte-identical to a synchronous save of that frame.
    Image = pytest.importorskip("PIL.Image")
    from ghee.output import ImageSequence

    n = 10
    frames = [np.full((8, 6, 4), i * 20, dtype=np.uint8) for i in range(n)]
    for f in frames:
        f[..., 3] = 255
    seq = ImageSequence(str(tmp_path / "p.####.png"), depth=2)
    for f in frames:
        buf = seq.acquire(f.shape)
        np.copyto(buf, f)  # stands in for r.render(out=buf)
        seq.submit(buf)
    seq.close()
    for i, f in enumerate(frames):
        got = np.asarray(Image.open(tmp_path / f"p.{i:04d}.png"))
        assert np.array_equal(got, f)
        ref = tmp_path / f"ref.{i:04d}.png"
        Image.fromarray(f, "RGBA").save(str(ref))
        assert (tmp_path / f"p.{i:04d}.png").read_bytes() == ref.read_bytes()


def _boom(self, *a, **k):
    raise OSError("disk full")


def test_worker_error_surfaces_on_write(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    from ghee.output import ImageSequence

    monkeypatch.setattr(Image.Image, "save", _boom)
    seq = ImageSequence(str(tmp_path / "e.####.png"))
    with pytest.raises(OSError, match="disk full"):
        for _ in range(100):
            seq.write(np.zeros((8, 6, 4), dtype=np.uint8))
            time.sleep(0.002)


def test_worker_error_surfaces_on_close(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    from ghee.output import ImageSequence

    monkeypatch.setattr(Image.Image, "save", _boom)
    seq = ImageSequence(str(tmp_path / "e.####.png"))
    seq.write(np.zeros((8, 6, 4), dtype=np.uint8))
    with pytest.raises(OSError, match="disk full"):
        seq.close()
