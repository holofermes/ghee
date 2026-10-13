import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame

from fbdev import Framebuffer


class Screen:
    def __init__(self, lay, panel=None):
        self.lay = lay
        pygame.display.init()
        pygame.font.init()
        self.surface = pygame.display.set_mode((lay.w, lay.h), depth=32)
        self.fb = panel and Framebuffer(panel.fb, lay.w, lay.h)
        self.clock = pygame.font.Font(lay.sans, lay.clock[2])
        self.load = pygame.font.Font(lay.pixel, lay.load[2])
        self.hot = pygame.font.Font(lay.sans, lay.hot[2])
        self.screen, self.phosphor, self.mint = (
            pygame.Color(c) for c in (lay.screen, lay.phosphor, lay.mint)
        )

    def draw(self, clock, load, blink):
        lay, s = self.lay, self.surface
        bx, by, bw, bh = lay.bar
        width = round(bw * min(load, 1))
        s.fill(self.screen)
        s.blit(self.clock.render(clock, True, self.phosphor), lay.clock[:2])
        s.blit(self.load.render(f"load {load:.0%}", True, self.phosphor), lay.load[:2])
        pygame.draw.rect(s, self.phosphor, lay.frame, width=lay.border)
        if width > 0:
            s.fill(self.mint, (bx, by, width, bh))
        if load > 0.9 and blink:
            s.blit(self.hot.render("HOT", True, self.mint), lay.hot[:2])
        if self.fb:
            self.fb.write(s.get_buffer())
        return s

    def save(self, path):
        pygame.image.save(self.surface, path)
