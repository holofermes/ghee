"""ST7789 135x240 SPI panel.

Packs each RGB frame to RGB565 and blits it in one writebytes2 on a private spidev
handle. Falls back to the adafruit block path when spidev is unavailable.
"""

import board
import digitalio
import numpy as np
from adafruit_rgb_display import st7789

from ghee.drivers.base import DisplayDriver, register
from ghee.engine import pack_frame


class ST7789(DisplayDriver):
    width = 135
    height = 240
    native_format = "rgb565"

    def __init__(self):
        cs = digitalio.DigitalInOut(board.CE0)
        dc = digitalio.DigitalInOut(board.D25)
        self.display = st7789.ST7789(
            board.SPI(),
            cs=cs,
            dc=dc,
            rst=None,
            baudrate=99000000,
            width=self.width,
            height=self.height,
            x_offset=53,
            y_offset=40,
        )
        self.backlight = digitalio.DigitalInOut(board.D22)
        self.backlight.switch_to_output()
        self.backlight.value = True
        self._buf = np.zeros(self.width * self.height * 2, dtype=np.uint8)
        self._window_ready = False
        self._fast = self._setup_fast_blit()

    def _setup_fast_blit(self):
        try:
            import spidev
        except ImportError:
            return None
        cs = getattr(self.display.spi_device, "chip_select", None)
        dc = getattr(self.display, "dc_pin", None)
        if cs is None or dc is None:
            return None
        spi = spidev.SpiDev()
        spi.open(0, 0)
        spi.mode = 0
        spi.max_speed_hz = 99000000
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
        data = self._buf.tobytes()
        if self._fast is None or not self._window_ready:
            self.display._block(0, 0, self.width - 1, self.height - 1, data)
            self._window_ready = True
            return
        spi, cs, dc = self._fast
        cs.value = False
        dc.value = False
        spi.writebytes2(bytes([self.display._RAM_WRITE]))
        dc.value = True
        spi.writebytes2(data)
        cs.value = True

    def turn_on(self):
        self.backlight.value = True

    def turn_off(self):
        self.backlight.value = False


register("st7789", ST7789)
