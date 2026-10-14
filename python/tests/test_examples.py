"""The bundled examples, run on the renderer as a user runs them directly."""

import os
import subprocess
import sys

import pytest
from conftest import REPO
from test_api import png_pixels


def run_example(name, out, *flags):
    """(width, height, RGBA bytes) of the first frame of a direct run of the
    example `name` with `flags`."""
    env = {k: v for k, v in os.environ.items() if k not in ("GHEE_SIZE", "GHEE_WIRE")}
    ran = subprocess.run(
        [sys.executable, str(REPO / "ghee/examples" / f"{name}.py"), "-o", str(out)]
        + ["--frames", "1", "--timestep", "0.1", *flags],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    return png_pixels(out.read_bytes())


@pytest.mark.parametrize(
    ("size", "radius", "hub"), [("256x64", 10, 4), ("128x32", 5, 2)]
)
def test_nowplaying_runs_the_tape_between_two_reels(tmp_path, size, radius, hub):
    """The reels take the bottom corners inside the padding, 2 px, and the
    bar between them fills from the left with the share of the track
    played, its rest dark. The padding stays dark."""
    w, h, px = run_example("nowplaying", tmp_path / "f.png", "--size", size)
    assert f"{w}x{h}" == size
    pad = 2

    def lit(x, y):
        return max(px[(y * w + x) * 4 : (y * w + x) * 4 + 3]) > 127

    edges = [(x, y) for x in range(w) for y in (*range(pad), *range(h - pad, h))]
    edges += [(x, y) for y in range(h) for x in (*range(pad), *range(w - pad, w))]
    assert not any(lit(x, y) for x, y in edges), "the padding"
    y = h - 1 - pad - radius
    assert lit(pad + radius + hub, y), "the left reel's hub"
    assert lit(w - 1 - pad - radius - hub, y), "the right reel's hub"
    left, right = pad + 2 * radius + 5, w - pad - 2 * radius - 6
    assert lit(left, y), "the bar's left end"
    assert lit(right, y), "the bar's right end"
    assert lit(left + 2, y), "the bar fills from the left"
    assert not lit(right - 2, y), "the bar's rest stays dark"
