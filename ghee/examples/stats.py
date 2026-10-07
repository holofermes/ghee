#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["ghee-render"]
# ///
"""The time and the load of the machine, drawn from Python on a 128x32 panel.

Each tick draws the time, the load average as a share of the cores with a
bar beside it, and a blinking HOT past 90%. `ghee.draw()` writes each call
as a layer, so the scene holds no GLSL: `ghee.run()` with no scene starts
from one black pass, and the first tick's drawing grows it.

    ghee render --demo stats --display kms
    ghee render --demo stats -o stats.png --size 128x32
"""

import os
import time

import ghee

ghee.args()  # the probe, this scene takes no arguments


@ghee.tick
def show(tick):
    load = os.getloadavg()[0] / (os.cpu_count() or 1)
    with ghee.draw() as d:
        d.text(time.strftime("%H:%M"), x=0, y=0, font_size=16)
        d.text(f"load {load:.0%}", x=0, y=20, font_size=8, font_family="pixel")
        d.rect(
            x=64,
            y=20,
            width=64,
            height=8,
            color="transparent",
            border={"width": 1, "color": "white"},
        )
        d.rect(x=66, y=22, width=60 * min(load, 1), height=4)
        if load > 0.9:
            d.text("HOT", x=100, y=0, font_size=12, blink={"rate": 2})


ghee.run()
