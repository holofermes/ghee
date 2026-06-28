"""Host-free defaults for the console.

A scene declares named float arrays ({name, size}); the engine fills each one
every frame. When no host supplies a real source, these defaults keep the
standalone preview animated.
A host overrides them via server.run(arrays={...}, hwinfo=...).
"""
import math

import numpy as np


def demo_feed(name, size, frame):
    """Placeholder values for any named array (a traveling wave), so the standalone
    console still animates with nothing attached."""
    i = np.arange(size)
    return (0.5 + 0.5 * np.sin(frame * 0.05 + i * (2 * math.pi / max(size, 1)))).astype(np.float32) * 0.6


def default_hwinfo():
    """Generic host info when no real provider is injected."""
    import platform
    return {"model": platform.platform(), "gpu": "n/a"}
