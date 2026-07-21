import math

import numpy as np


def demo_feed(name, size, frame):
    """Placeholder values for any named array (a traveling wave; a 2D input gets
    drifting diagonal bands), so the standalone console still animates with
    nothing attached."""
    if not isinstance(size, int):
        w, h = size
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        m = 0.5 + 0.5 * np.sin((x + y) * (2 * math.pi / max(w, 1)) + frame * 0.05)
        return (m * 0.6).astype(np.float32).reshape(-1)
    i = np.arange(size)
    return (0.5 + 0.5 * np.sin(frame * 0.05 + i * (2 * math.pi / max(size, 1)))).astype(
        np.float32
    ) * 0.6


def default_hwinfo():
    """Generic host info when no real provider is injected."""
    import platform

    return {"model": platform.platform(), "gpu": "n/a"}
