import os
import struct
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")

import pyray as rl


def colour(hex_):
    return rl.Color(int(hex_[1:3], 16), int(hex_[3:5], 16), int(hex_[5:7], 16), 255)


def line_height(path):
    """A font's ascent plus descent in em, the height that raylib sizes it by."""
    data = Path(path).read_bytes()
    (count,) = struct.unpack_from(">H", data, 4)
    tables = {}
    for i in range(count):
        tag, _, offset, _ = struct.unpack_from(">4sIII", data, 12 + 16 * i)
        tables[tag] = offset
    (upem,) = struct.unpack_from(">H", data, tables[b"head"] + 18)
    ascent, descent = struct.unpack_from(">hh", data, tables[b"hhea"] + 4)
    return (ascent - descent) / upem


def load_font(path, em):
    size = round(em * line_height(path))
    return rl.load_font_ex(path, size, None, 0), size


class Screen:
    def __init__(self, lay):
        self.lay = lay
        rl.set_trace_log_level(rl.LOG_WARNING)
        rl.set_config_flags(rl.FLAG_WINDOW_HIDDEN)
        rl.init_window(lay.w, lay.h, "")
        self.target = rl.load_render_texture(lay.w, lay.h)
        self.clock = load_font(lay.sans, lay.clock[2])
        self.load = load_font(lay.pixel, lay.load[2])
        self.hot = load_font(lay.sans, lay.hot[2])
        self.screen, self.phosphor, self.mint = map(
            colour, (lay.screen, lay.phosphor, lay.mint)
        )
        self.image = None

    def draw(self, clock, load, blink):
        lay = self.lay
        bx, by, bw, bh = lay.bar
        width = round(bw * min(load, 1))
        rl.begin_texture_mode(self.target)
        rl.clear_background(self.screen)
        font, size = self.clock
        rl.draw_text_ex(font, clock, lay.clock[:2], size, 0, self.phosphor)
        font, size = self.load
        rl.draw_text_ex(font, f"load {load:.0%}", lay.load[:2], size, 0, self.phosphor)
        rl.draw_rectangle_lines_ex(rl.Rectangle(*lay.frame), lay.border, self.phosphor)
        if width > 0:
            rl.draw_rectangle(bx, by, width, bh, self.mint)
        if load > 0.9 and blink:
            font, size = self.hot
            rl.draw_text_ex(font, "HOT", lay.hot[:2], size, 0, self.mint)
        rl.end_texture_mode()
        if self.image is not None:
            rl.unload_image(self.image)
        self.image = rl.load_image_from_texture(self.target.texture)
        return self.image

    def save(self, path):
        rl.image_flip_vertical(self.image)
        rl.export_image(self.image, path)
