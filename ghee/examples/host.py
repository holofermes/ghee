# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render"]
# ///
"""A Python program that opens a screen and draws on it in its own loop.

ghee.open() loads bars.scene.json, from beside this file, onto the connected
display, or headless at --size. Each turn of the loop feeds the scene's
array `amplitudes` sixteen levels, sets its param `glow`, and draws the time
over the bars with screen.draw(), which renders the frame. The console on
http://127.0.0.1:8090 shows the same frames and sets the other params.

The program runs ghee, so ghee never starts this file: it is no scene, and
`--demo` does not list it.

    uv run host.py
    uv run host.py --size 320x240 -o frame.####.png --frames 60
"""

import argparse
import math
import time
from array import array
from pathlib import Path

import ghee

parser = argparse.ArgumentParser(description="Draw on a screen from a loop.")
parser.add_argument("--size", help="render headless at WxH, with no display")
parser.add_argument("-o", dest="out", help="save each frame, as frame.####.png")
parser.add_argument("--frames", type=int, default=0, help="stop after N frames")
parser.add_argument("--port", type=int, default=8090, help="the console's port")
args = parser.parse_args()

if args.size:
    width, height = (int(n) for n in args.size.split("x"))
    where = {"width": width, "height": height}
else:
    where = {"display": "kms"}

scene = Path(__file__).with_name("bars.scene.json")
with ghee.open(str(scene), fps=30, serve=args.port, **where) as screen:
    levels = array("f", bytes(4 * 16))
    frame = 0
    try:
        while not args.frames or frame < args.frames:
            t = time.monotonic()
            for i in range(len(levels)):
                levels[i] = 0.45 + 0.3 * math.sin(i * 0.5 - t * 3.0) * math.sin(t * 0.7)
            screen.feed("amplitudes", levels)
            screen.set_param("glow", 0.6 + 0.4 * math.sin(t * 2.0))
            with screen.draw() as d:
                d.text(time.strftime("%H:%M:%S"), x=8, y=8, font_size=24)
            if args.out:
                screen.save(args.out)
            frame += 1
    except KeyboardInterrupt:
        pass
