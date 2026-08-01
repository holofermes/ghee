"""SSD1306 128x32 monochrome I2C panel.

Packs each RGB frame to 1bpp into the display buffer, behind the 0x40 control
byte the panel wants in front of display RAM, and pushes the lot in one
i2c_rdwr transfer.
"""

import numpy as np
from smbus2 import SMBus, i2c_msg

from ghee.drivers.base import DisplayDriver, register
from ghee.engine import pack_frame

CMD = 0x80  # control byte: one command follows
DATA = 0x40  # control byte: the rest of the transfer is display RAM

DISP_OFF = 0xAE
DISP_ON = 0xAF
COL_ADDR = 0x21
PAGE_ADDR = 0x22

INIT = (
    (DISP_ON,),  # on, then straight off again on the next command
    (DISP_OFF,),
    (0x20, 0x00),  # horizontal addressing
    (0x40,),  # start line 0
    (0xA1,),  # column 127 maps to SEG0
    (0xA8, 0x1F),  # mux ratio, height - 1
    (0xC8,),  # scan COM[N] to COM[0]
    (0xD3, 0x00),  # no display offset
    (0xDA, 0x02),  # COM pin config, 128x32 layout
    (0xD5, 0x80),  # clock divide
    (0xD9, 0xF1),  # precharge
    (0xDB, 0x30),  # VCOMH deselect
    (0x81, 0xFF),  # contrast
    (0xA4,),  # follow RAM, not all-on
    (0xA6,),  # not inverted
    (0xAD, 0x30),  # internal IREF
    (0x8D, 0x14),  # charge pump on
    (DISP_ON,),
)


class SSD1306(DisplayDriver):
    width = 128
    height = 32
    native_format = "mono1"

    def __init__(self, bus=1, addr=0x3C):
        self.addr = addr
        self.bus = SMBus(bus)
        for entry in INIT:
            for cmd in entry:
                self._cmd(cmd)
        self._buf = bytearray(1 + self.width * self.height // 8)
        self._buf[0] = DATA
        self._pixels = np.frombuffer(self._buf, dtype=np.uint8)[1:]
        self._show()

    def _cmd(self, cmd):
        self.bus.write_byte_data(self.addr, CMD, cmd)

    def _show(self):
        for cmd in (COL_ADDR, 0, self.width - 1, PAGE_ADDR, 0, self.height // 8 - 1):
            self._cmd(cmd)
        self.bus.i2c_rdwr(i2c_msg.write(self.addr, self._buf))

    def write(self, frame):
        pack_frame(frame, self.width, self.height, 3, bw=True, out=self._pixels)
        self._show()

    def clear(self):
        self._pixels[:] = 0
        self._show()

    def turn_on(self):
        self._cmd(DISP_ON)

    def turn_off(self):
        self._cmd(DISP_OFF)

    def close(self):
        self.bus.close()


register("ssd1306", SSD1306)
