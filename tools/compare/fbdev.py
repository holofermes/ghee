import fcntl
import os
import struct
import time
from pathlib import Path

FBIOGET_VSCREENINFO = 0x4600
FBIOBLANK = 0x4611


class Framebuffer:
    """A Linux framebuffer in XRGB8888, written whole with one pwrite a frame."""

    def __init__(self, path, width, height):
        self.fd = os.open(path, os.O_WRONLY)
        var = struct.unpack_from(
            "17I", fcntl.ioctl(self.fd, FBIOGET_VSCREENINFO, bytes(160))
        )
        stride = int(
            (Path("/sys/class/graphics") / Path(path).name / "stride").read_text()
        )
        fmt = (var[0], var[1], var[6], var[8], var[11], var[14], stride)
        if fmt != (width, height, 32, 16, 8, 0, width * 4):
            raise ValueError(f"{path}: {fmt}, want {width}x{height} XRGB8888")
        fcntl.ioctl(self.fd, FBIOBLANK, 0)

    def write(self, data):
        self.sent = time.monotonic()
        os.pwrite(self.fd, data, 0)
