"""DRM master and primary-node file descriptors.

Only a render that scans out may hold DRM master. An offscreen render has
nothing to put on a connector, and the hardware encoder opens a primary node
solely to allocate contiguous buffers on it. Either one holding master makes
coexistence depend on launch order: whoever starts second is refused.

The fd checks read /proc/self/fd, so they only run on Linux; the encoder ones
need a board with /dev/video11.
"""

import fcntl
import glob
import os

import pytest

from ghee.engine import Renderer
from ghee.h264 import HardwareH264

# drm.h: _IO('d', 0x1e) and _IO('d', 0x1f)
DRM_IOCTL_SET_MASTER = 0x641E
DRM_IOCTL_DROP_MASTER = 0x641F

# above HW_MIN_PIXELS: the codec refuses an input buffer under 64KiB
ENC_W, ENC_H = 160, 128

needs_proc = pytest.mark.skipif(
    not os.path.isdir("/proc/self/fd"), reason="needs /proc/self/fd"
)
needs_hw_encoder = pytest.mark.skipif(
    not os.path.exists("/dev/video11"), reason="no hardware H264 encoder"
)


def dri_fds():
    """The /dev/dri nodes this process holds open, by node name."""
    names = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:  # the listdir entry closed under us
            continue
        if target.startswith("/dev/dri/"):
            names.append(os.path.basename(target))
    return sorted(names)


def master_free(card):
    """Whether a fresh fd on this primary node can still take DRM master.
    False once any other fd holds it. Opening a node nobody has mastered grants
    master implicitly, so the drop is what keeps this a read-only question."""
    try:
        fd = os.open(card, os.O_RDWR | os.O_CLOEXEC)
    except OSError:
        return False
    try:
        fcntl.ioctl(fd, DRM_IOCTL_SET_MASTER, 0)
    except OSError:
        return False
    else:
        fcntl.ioctl(fd, DRM_IOCTL_DROP_MASTER, 0)
        return True
    finally:
        os.close(fd)


@needs_proc
def test_offscreen_render_holds_no_primary_node(renderer):
    if not glob.glob("/dev/dri/renderD*"):
        pytest.skip("no render node for an offscreen render to prefer")
    held = [n for n in dri_fds() if n.startswith("card")]
    assert held == [], f"offscreen render holds primary nodes: {held}"


@needs_hw_encoder
def test_encoder_returns_every_fd_it_opened():
    r = Renderer(ENC_W, ENC_H, components=4)
    try:
        try:
            HardwareH264(r, ENC_W, ENC_H, fps=30, bitrate=1_000_000).close()
        except RuntimeError as e:
            pytest.skip(f"hardware encoder unavailable: {e}")
        base = dri_fds()
        for _ in range(3):
            enc = HardwareH264(r, ENC_W, ENC_H, fps=30, bitrate=1_000_000)
            enc.close()
            assert dri_fds() == base
    finally:
        r.close()


@needs_hw_encoder
def test_encoder_leaves_drm_master_free():
    free = [c for c in sorted(glob.glob("/dev/dri/card*")) if master_free(c)]
    if not free:
        pytest.skip("something already holds DRM master on every primary node")
    r = Renderer(ENC_W, ENC_H, components=4)
    try:
        enc = HardwareH264(r, ENC_W, ENC_H, fps=30, bitrate=1_000_000)
    except RuntimeError as e:
        r.close()
        pytest.skip(f"hardware encoder unavailable: {e}")
    try:
        taken = [c for c in free if not master_free(c)]
        assert taken == [], f"encoder holds DRM master on {taken}"
    finally:
        enc.close()
        r.close()
