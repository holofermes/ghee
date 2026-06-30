import os

import numpy as np
import pytest

from ghee import Scene

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "goldens")


def scene(passes):
    return Scene.from_dict({"version": 1, "passes": passes})


def ramp(name, size):
    """A stable per-array signal, independent of frame and time, so renders that read
    it are reproducible."""
    return (0.5 + 0.5 * np.sin(np.arange(size) * 0.7)).astype(np.float32)


def const(values):
    """A provider that always feeds the same array, whatever the name/size asked."""
    arr = np.array(values, dtype=np.float32)
    return lambda name, size: arr


def assert_golden(name, frame, tol=None, max_bad_frac=None):
    """Snapshot test with a fuzzy compare, so one committed baseline survives small
    cross-GPU differences (mediump precision, transcendental approximations) between the
    machine that recorded it and the one running the test. Fails when more than
    max_bad_frac of channel samples differ from the golden by more than tol (0-255).

    Defaults: tol=16, max_bad_frac=0.05. Override per call or via GHEE_GOLDEN_TOL /
    GHEE_GOLDEN_MAXBAD. tol=0 makes it an exact match. First run (or GHEE_REGEN=1)
    records the baseline."""
    from PIL import Image

    if tol is None:
        tol = int(os.environ.get("GHEE_GOLDEN_TOL", 16))
    if max_bad_frac is None:
        max_bad_frac = float(os.environ.get("GHEE_GOLDEN_MAXBAD", 0.05))

    os.makedirs(GOLDEN_DIR, exist_ok=True)
    path = os.path.join(GOLDEN_DIR, name + ".png")
    mode = "RGBA" if frame.shape[2] == 4 else "RGB"
    if os.environ.get("GHEE_REGEN") or not os.path.exists(path):
        Image.fromarray(frame, mode).save(path)
        pytest.skip(f"golden '{name}' recorded")

    golden = np.asarray(Image.open(path).convert(mode)).astype(int)
    diff = np.abs(golden - frame.astype(int))
    bad_frac = float((diff > tol).mean())
    if bad_frac > max_bad_frac:
        Image.fromarray(frame, mode).save(
            os.path.join(GOLDEN_DIR, name + ".actual.png")
        )
        raise AssertionError(
            f"'{name}': {bad_frac:.1%} of samples differ by > {tol} "
            f"(allowed {max_bad_frac:.1%}); mean={diff.mean():.2f} max={diff.max()} "
            f"(wrote {name}.actual.png)"
        )
