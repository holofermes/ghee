"""Placeholder data for a declared array nothing feeds.

A scene declares its arrays (ghee.scene.ArrayInput) and a controller or a host
fills them each frame. One that nothing fills still has to render, so it gets
this animated stand-in instead of a black texture.
"""

import numpy as np

_grids = {}


def _grid(size):
    """Normalised coordinates for a size, built once and kept: only the size
    changes them, not the frame."""
    g = _grids.get(size)
    if g is None:
        if isinstance(size, int):
            g = (np.arange(size, dtype=np.float32) / max(size - 1, 1),)
        else:
            w, h = size
            y, x = np.mgrid[0:h, 0:w].astype(np.float32)
            g = (x / max(w - 1, 1), y / max(h - 1, 1))
        _grids[size] = g
    return g


def demo_feed(name, size, frame):
    """Placeholder values for any named array, so a scene still animates if
    nothing is attached.
    """
    t = frame * 0.05
    if isinstance(size, int):
        (x,) = _grid(size)
        v = 0.45 + 0.28 * np.sin(x * 4.4 + t) + 0.10 * np.sin(x * 9.7 - t * 0.6)
        return np.clip(v, 0.0, 1.0).astype(np.float32)
    x, y = _grid(size)
    v = (
        0.42
        + 0.24 * np.sin(x * 4.4 + t)
        + 0.10 * np.sin(y * 9.7 - t * 0.6)
        + 0.08 * np.sin((x + y) * 6.1 + t * 0.35)
    )
    return np.clip(v, 0.0, 1.0).astype(np.float32).reshape(-1)
