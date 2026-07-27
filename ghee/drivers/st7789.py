"""ST7789 SPI panel.

Packs each RGB frame to RGB565 and blits it in one writebytes2 on a private spidev
handle. Falls back to the adafruit block path when spidev is unavailable.

The defaults are the 135x240 breakout this was written on.
"""

import board
import digitalio
import numpy as np
from adafruit_rgb_display import st7789

from ghee.drivers.base import DisplayDriver, register
from ghee.engine import pack_frame


def _pin(name):
    """A board pin: "D25" or an already-resolved pin object."""
    return getattr(board, name) if isinstance(name, str) else name


class ST7789(DisplayDriver):
    native_format = "rgb565"

    def __init__(
        self,
        width=135,
        height=240,
        x_offset=53,
        y_offset=40,
        cs="CE0",
        dc="D25",
        rst=None,
        backlight="D22",
        baudrate=99000000,
        spi_bus=0,
        spi_device=0,
    ):
        self.width = width
        self.height = height
        self.baudrate = baudrate
        self.display = st7789.ST7789(
            board.SPI(),
            cs=digitalio.DigitalInOut(_pin(cs)),
            dc=digitalio.DigitalInOut(_pin(dc)),
            rst=digitalio.DigitalInOut(_pin(rst)) if rst is not None else None,
            baudrate=baudrate,
            width=width,
            height=height,
            x_offset=x_offset,
            y_offset=y_offset,
        )
        self.backlight = None
        if backlight is not None:
            self.backlight = digitalio.DigitalInOut(_pin(backlight))
            self.backlight.switch_to_output()
            self.backlight.value = True
        self._buf = np.zeros(width * height * 2, dtype=np.uint8)
        self._window_ready = False
        self._fast = self._setup_fast_blit(spi_bus, spi_device)

    def _setup_fast_blit(self, bus, device):
        try:
            import spidev
        except ImportError:
            return None
        cs = getattr(self.display.spi_device, "chip_select", None)
        dc = getattr(self.display, "dc_pin", None)
        if cs is None or dc is None:
            return None
        spi = spidev.SpiDev()
        spi.open(bus, device)
        spi.mode = 0
        spi.max_speed_hz = self.baudrate
        try:
            spi.no_cs = (
                True  # adafruit drives CS via the digitalio pin, not hardware CE
            )
        except Exception:
            pass
        if not hasattr(spi, "writebytes2"):
            spi.close()
            return None
        return (spi, cs, dc)

    def write(self, frame):
        pack_frame(frame, self.width, self.height, 3, bw=False, out=self._buf)
        if self._fast is None or not self._window_ready:
            self.display._block(
                0, 0, self.width - 1, self.height - 1, self._buf.tobytes()
            )
            self._window_ready = True
            return
        spi, cs, dc = self._fast
        cs.value = False
        dc.value = False
        spi.writebytes2(bytes([self.display._RAM_WRITE]))
        dc.value = True
        spi.writebytes2(self._buf)  # buffer protocol: no per-frame copy
        cs.value = True

    def turn_on(self):
        if self.backlight is not None:
            self.backlight.value = True

    def turn_off(self):
        if self.backlight is not None:
            self.backlight.value = False


register("st7789", ST7789)
