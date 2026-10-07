"""The bundled examples, run on the renderer as a user runs them directly."""

import os
import subprocess
import sys

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


def test_nowplaying_draws_the_artist_at_its_panel_size_on_a_wider_frame(tmp_path):
    """The artist's pass, as wide as the panel, lands in the frame pixel for
    pixel: the top left of a 256x64 frame for a 128x32 panel holds what a
    128x32 frame holds, and the rest of those rows stays dark."""
    w, h, panel = run_example("nowplaying", tmp_path / "panel.png", "--size", "128x32")
    W, H, wide = run_example(
        "nowplaying", tmp_path / "wide.png", "--size", "256x64", "--panel", "128x32"
    )
    assert (w, h, W, H) == (128, 32, 256, 64)
    rows = round(0.4 * h)

    def row(px, width, y, x0, x1):
        return px[(y * width + x0) * 4 : (y * width + x1) * 4]

    artist = [row(panel, w, y, 0, w) for y in range(rows)]
    assert any(max(r[0::4]) > 127 for r in artist), "the artist draws nothing"
    assert [row(wide, W, y, 0, w) for y in range(rows)] == artist
    assert all(max(row(wide, W, y, w, W)[0::4]) == 0 for y in range(rows))
