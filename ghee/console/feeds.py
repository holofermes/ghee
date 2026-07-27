import math

import numpy as np

_bands = {}


def _band_phase(w, h):
    """(x + y) scaled to the band period. Only the size changes it, so it is
    built once per size instead of every frame."""
    p = _bands.get((w, h))
    if p is None:
        y, x = np.mgrid[0:h, 0:w]
        p = ((x + y) * (2 * math.pi / max(w, 1))).astype(np.float32)
        _bands[(w, h)] = p
    return p


def demo_feed(name, size, frame):
    """Placeholder values for any named array (a traveling wave; a 2D input gets
    drifting diagonal bands), so the standalone console still animates with
    nothing attached."""
    if not isinstance(size, int):
        w, h = size
        m = 0.5 + 0.5 * np.sin(_band_phase(w, h) + frame * 0.05)
        return (m * 0.6).astype(np.float32).reshape(-1)
    i = np.arange(size)
    return (0.5 + 0.5 * np.sin(frame * 0.05 + i * (2 * math.pi / max(size, 1)))).astype(
        np.float32
    ) * 0.6


def default_hwinfo():
    """Generic host info when no real provider is injected."""
    import platform

    return {"model": platform.platform(), "gpu": "n/a"}
