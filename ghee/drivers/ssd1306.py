"""SSD1306 128x32 monochrome I2C panel.

Packs each RGB frame to 1bpp straight into the adafruit framebuffer (skipping its
leading command byte), then shows it.
"""

import adafruit_ssd1306
import busio
import numpy as np
from board import SCL, SDA

from ghee.drivers.base import DisplayDriver, register
from ghee.engine import pack_frame


class SSD1306(DisplayDriver):
    width = 128
    height = 32
    native_format = "mono1"

    def __init__(self):
        i2c = busio.I2C(SCL, SDA)
        self.display = adafruit_ssd1306.SSD1306_I2C(self.width, self.height, i2c)
        self.display.fill(0)
        self.display.show()
        self._buf = np.frombuffer(self.display.buffer, dtype=np.uint8)[1:]

    def write(self, frame):
        pack_frame(frame, self.width, self.height, 3, bw=True, out=self._buf)
        self.display.show()

    def turn_on(self):
        self.display.poweron()

    def turn_off(self):
        self.display.poweroff()


register("ssd1306", SSD1306)
