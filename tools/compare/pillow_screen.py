from PIL import Image, ImageDraw, ImageFont

from fbdev import Framebuffer


class Screen:
    def __init__(self, lay, panel=None):
        self.lay = lay
        self.fb = panel and Framebuffer(panel.fb, lay.w, lay.h)
        self.image = Image.new("RGB", (lay.w, lay.h))
        self.d = ImageDraw.Draw(self.image)
        self.clock = ImageFont.truetype(lay.sans, lay.clock[2])
        self.load = ImageFont.truetype(lay.pixel, lay.load[2])
        self.hot = ImageFont.truetype(lay.sans, lay.hot[2])

    def draw(self, clock, load, blink):
        lay, d = self.lay, self.d
        fx, fy, fw, fh = lay.frame
        bx, by, bw, bh = lay.bar
        width = round(bw * min(load, 1))
        d.rectangle((0, 0, lay.w - 1, lay.h - 1), fill=lay.screen)
        d.text(lay.clock[:2], clock, font=self.clock, fill=lay.phosphor)
        d.text(lay.load[:2], f"load {load:.0%}", font=self.load, fill=lay.phosphor)
        d.rectangle(
            (fx, fy, fx + fw - 1, fy + fh - 1), outline=lay.phosphor, width=lay.border
        )
        if width > 0:
            d.rectangle((bx, by, bx + width - 1, by + bh - 1), fill=lay.mint)
        if load > 0.9 and blink:
            d.text(lay.hot[:2], "HOT", font=self.hot, fill=lay.mint)
        if self.fb:
            self.fb.write(self.image.tobytes("raw", "BGRX"))
        return self.image

    def save(self, path):
        self.image.save(path)
