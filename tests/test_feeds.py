"""The placeholder a declared array falls back to when nothing feeds it."""

import numpy as np

from ghee import feeds
from ghee.console import feeds as console_feeds


def test_demo_feed_2d_matches_its_formula_and_caches_the_grid():
    feeds._grids.clear()
    got = feeds.demo_feed("mask", (7, 3), 4)
    xi, yi = np.meshgrid(np.arange(7, dtype=np.float32), np.arange(3, dtype=np.float32))
    x, y, t = xi / 6.0, yi / 2.0, 4 * 0.05
    want = np.clip(
        0.42
        + 0.24 * np.sin(x * 4.4 + t)
        + 0.10 * np.sin(y * 9.7 - t * 0.6)
        + 0.08 * np.sin((x + y) * 6.1 + t * 0.35),
        0.0,
        1.0,
    )
    assert got.shape == (21,)
    assert np.allclose(got, want.reshape(-1).astype(np.float32), atol=1e-6)
    grid = feeds._grids[(7, 3)]
    feeds.demo_feed("mask", (7, 3), 5)
    assert feeds._grids[(7, 3)] is grid


def test_demo_feed_1d_is_two_travelling_sines_and_stays_in_range():
    a = feeds.demo_feed("amps", 24, 0)
    b = feeds.demo_feed("amps", 24, 40)
    assert a.shape == (24,) and b.shape == (24,)
    assert a.min() >= 0.0 and a.max() <= 1.0
    # it travels, and the second sine keeps the crest off the beat, so no
    # frame is a shift of another
    assert not np.allclose(a, b, atol=1e-3)
    assert not np.allclose(a, np.roll(b, 1), atol=1e-2)


def test_a_headless_render_does_not_reach_into_the_console_for_it():
    src = open(f"{feeds.__file__[: -len('feeds.py')]}cli.py", encoding="utf-8").read()
    assert "ghee.console.feeds" not in src
    assert "from ghee.feeds import demo_feed" in src


def test_the_old_console_path_still_answers():
    """The console package re-exports it: an embedding host may import either."""
    assert console_feeds.demo_feed is feeds.demo_feed
