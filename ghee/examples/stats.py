#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render==0.4.1.dev0"]
# ///
"""The time and the load of the machine, drawn from Python on a 128x32 panel.

Each tick draws the screen's colour, the time, the load average as a share
of the cores with a bar beside it, and a blinking HOT past 90%.
`ghee.draw()` writes each call as a layer, so the scene holds no GLSL:
`ghee.run()` with no scene starts from one black pass, and the first tick's
drawing grows it.

    ghee render --demo stats --display kms
    ghee render --demo stats -o stats.png --size 128x32
"""

import os
import time

import ghee

ghee.args()  # the probe, this scene takes no arguments

SCREEN = "#0d201a"
PHOSPHOR = "#3ce94d"
MINT = "#a1fee3"


@ghee.tick
def show(tick):
    load = os.getloadavg()[0] / (os.cpu_count() or 1)
    with ghee.draw() as d:
        d.rect(color=SCREEN)
        d.text(time.strftime("%H:%M"), x=0, y=0, font_size=16, color=PHOSPHOR)
        d.text(
            f"load {load:.0%}",
            x=0,
            y=20,
            font_size=8,
            font_family="Pixel Operator 8",
            color=PHOSPHOR,
        )
        d.rect(
            x=64,
            y=20,
            width=64,
            height=8,
            color="transparent",
            border={"width": 1, "color": PHOSPHOR},
        )
        d.rect(x=66, y=22, width=60 * min(load, 1), height=4, color=MINT)
        if load > 0.9:
            d.text("HOT", x=100, y=0, font_size=12, color=MINT, blink={"rate": 2})


ghee.run()
