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


def test_nowplaying_centres_the_same_artist_on_a_wider_frame(tmp_path):
    """The artist's pass takes the frame's width from ghee.render_size(), and
    main copies it pixel for pixel: the artist's rows of a 256x32 frame hold
    the 128x32 frame's artist, moved right by half the extra width, and stay
    dark on either side."""
    w, h, narrow = run_example(
        "nowplaying", tmp_path / "narrow.png", "--size", "128x32"
    )
    W, H, wide = run_example("nowplaying", tmp_path / "wide.png", "--size", "256x32")
    assert (w, h, W, H) == (128, 32, 256, 32)
    rows = round(0.4 * h)
    left = (W - w) // 2

    def row(px, width, y, x0, x1):
        return px[(y * width + x0) * 4 : (y * width + x1) * 4]

    artist = [row(narrow, w, y, 0, w) for y in range(rows)]
    assert any(max(r[0::4]) > 127 for r in artist), "the artist draws nothing"
    assert [row(wide, W, y, left, left + w) for y in range(rows)] == artist
    for y in range(rows):
        assert max(row(wide, W, y, 0, left)[0::4]) == 0
        assert max(row(wide, W, y, left + w, W)[0::4]) == 0
