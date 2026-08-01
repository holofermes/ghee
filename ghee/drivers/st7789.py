"""ST7789 SPI panel.

Packs each RGB frame to RGB565 and blits it in one writebytes2 on a private
spidev handle, with DC, reset and backlight driven through libgpiod. Chip
select is the hardware CE line: the kernel SPI controller owns that GPIO and
libgpiod cannot claim it.

Pins are BCM numbers. The defaults are the 135x240 breakout this was written on.
"""

import struct
import sys
import time

import numpy as np
import spidev

from ghee.drivers.base import DisplayDriver, register
from ghee.drivers.gpio import Pins
from ghee.engine import pack_frame

SWRESET = 0x01
SLPOUT = 0x11
NORON = 0x13
INVON = 0x21
DISPON = 0x29
CASET = 0x2A
RASET = 0x2B
RAMWR = 0x2C
MADCTL = 0x36
COLMOD = 0x3A


def _backlight_pins(offset, consumer):
    """The backlight is requested on its own, and a refusal is survivable: it is
    the one line another kernel driver may already hold (GPIO22 doubles as the
    DigiAMP+ mute), and losing it must not cost the panel its DC line."""
    if offset is None:
        return None
    try:
        return Pins([offset], consumer=consumer)
    except OSError as e:
        print(
            f"st7789: GPIO{offset} is not available ({e}); no backlight control",
            file=sys.stderr,
        )
        return None


class ST7789(DisplayDriver):
    native_format = "rgb565"

    def __init__(
        self,
        width=135,
        height=240,
        x_offset=53,
        y_offset=40,
        dc=25,
        rst=None,
        backlight=22,
        baudrate=99000000,
        spi_bus=0,
        spi_device=0,
        # MADCTL: row/column order and RGB/BGR, so the panel's rotation and
        # mirroring. Per board, and a wrong value shows as a flipped image.
        madctl=0xC0,
    ):
        self.width = width
        self.height = height
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.dc = dc
        self.rst = rst
        self.bl = backlight
        self.madctl = madctl
        self.pins = Pins([dc, rst], consumer="ghee-st7789")
        self.bl_pins = _backlight_pins(backlight, "ghee-st7789-backlight")

        self.spi = spidev.SpiDev()
        self.spi.open(spi_bus, spi_device)
        self.spi.mode = 0
        self.spi.max_speed_hz = baudrate

        if rst is not None:
            self.pins.set(rst, False)
            time.sleep(0.050)
            self.pins.set(rst, True)
            time.sleep(0.050)

        self._init_panel()
        self.turn_on()

        self._buf = np.zeros(width * height * 2, dtype=np.uint8)
        self._window_ready = False

    def _cmd(self, command, data=None):
        self.pins.set(self.dc, False)
        self.spi.writebytes2(bytes([command]))
        if data:
            self.pins.set(self.dc, True)
            self.spi.writebytes2(data)

    def _init_panel(self):
        cols = struct.pack(">HH", self.x_offset, self.width + self.x_offset)
        rows = struct.pack(">HH", self.y_offset, self.height + self.y_offset)
        for command, data in (
            (SWRESET, None),
            (SLPOUT, None),
            (COLMOD, b"\x55"),  # 16 bit colour
            (MADCTL, b"\x08"),
            (CASET, cols),
            (RASET, rows),
            (INVON, None),
            (NORON, None),
            (DISPON, None),
            (MADCTL, bytes([self.madctl])),
        ):
            self._cmd(command, data)

    def _window(self, x0, y0, x1, y1):
        self._cmd(CASET, struct.pack(">HH", x0 + self.x_offset, x1 + self.x_offset))
        self._cmd(RASET, struct.pack(">HH", y0 + self.y_offset, y1 + self.y_offset))

    def _blit(self):
        if not self._window_ready:
            self._window(0, 0, self.width - 1, self.height - 1)
            self._window_ready = True
        self.pins.set(self.dc, False)
        self.spi.writebytes2(bytes([RAMWR]))
        self.pins.set(self.dc, True)
        self.spi.writebytes2(self._buf)  # buffer protocol: no per-frame copy

    def write(self, frame):
        pack_frame(frame, self.width, self.height, 3, bw=False, out=self._buf)
        self._blit()

    def clear(self):
        self._buf[:] = 0
        self._blit()

    def turn_on(self):
        if self.bl_pins is not None:
            self.bl_pins.set(self.bl, True)

    def turn_off(self):
        if self.bl_pins is not None:
            self.bl_pins.set(self.bl, False)

    def close(self):
        self.spi.close()
        self.pins.close()
        if self.bl_pins is not None:
            self.bl_pins.close()


register("st7789", ST7789)
