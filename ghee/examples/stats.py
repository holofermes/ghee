#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render==0.4.1.dev0"]
# ///
"""The machine's load, memory, temperature and uptime as a ship's readout,
drawn from Python on a 128x32 or a 256x64 panel.

Each tick draws the whole screen with `ghee.draw()`: a framed chart of the
cores' load that scrolls left, and beside it the readings in Departure Mono,
values right-aligned, with the load and the memory as bars of blocks. The
layout follows the screen's size, and the scene holds no GLSL.

    ghee render --demo stats --display kms
    ./stats.py -o stats.png --size 256x64
"""

import functools
import math
import time
from collections import deque

import ghee

ghee.args()  # the probe, this scene takes no arguments

FONT = "fonts/DepartureMono-Regular.otf"
SIZE = 11  # the font's grid, 11 px to the em: capitals 8 px tall, 7 px apart
CAP = 3  # the rows between a text block's top and its capitals
INK = "#a6e8cc"
FRAME = "#86c9ae"
DIM = "#1f3d34"
SAMPLE = 0.25  # seconds between two points of the chart
SPACING = 3  # pixels between two points of the chart

W, H = ghee.render_size()
BIG = H >= 64

_cpu = {"was": None, "busy": 0.0}


def cpu():
    """The cores' busy share since the last call, from /proc/stat."""
    with open("/proc/stat") as f:
        v = [int(n) for n in f.readline().split()[1:9]]
    now = (v[3] + v[4], sum(v))
    was, _cpu["was"] = _cpu["was"], now
    if was and now[1] > was[1]:
        _cpu["busy"] = 1 - (now[0] - was[0]) / (now[1] - was[1])
    return _cpu["busy"]


def memory():
    """The share of memory in use, from /proc/meminfo."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":")
            info[key] = int(value.split()[0])
    return 1 - info["MemAvailable"] / info["MemTotal"]


def temperature():
    """The first thermal zone in °C, or None on a machine without one."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000
    except OSError:
        return None


def uptime():
    """The time since boot as days and hours, or hours and minutes."""
    with open("/proc/uptime") as f:
        s = int(float(f.read().split()[0]))
    days, hours, minutes = s // 86400, s // 3600 % 24, s // 60 % 60
    return f"{days}D {hours:02}H" if days else f"{hours:02}:{minutes:02}"


@functools.cache
def width(words):
    return ghee.measure(words, font_size=SIZE, font_family=FONT)[0]


def text(d, words, x, y, color=INK, right=False, **keys):
    """Words with their capitals' top left at x, y, or their right end at x."""
    if right:
        x -= width(words)
    d.text(
        words,
        x=x,
        y=y - CAP,
        font_size=SIZE,
        font_family=FONT,
        color=color,
        render="mono",
        **keys,
    )


def blocks(d, x, y, w, h, share):
    """A bar of blocks 2 px wide, lit up to `share` of its length."""
    n = (w + 1) // 3
    lit = round(share * n)
    for i in range(n):
        d.rect(x=x + 3 * i, y=y, width=2, height=h, color=INK if i < lit else DIM)


BOX = round(W * 0.55) if BIG else round(W * 0.44)
EDGE = 2 if BIG else 1
LEFT, RIGHT = EDGE + 2, BOX - EDGE - 3
TOP, BOTTOM = EDGE + (14 if BIG else 2), H - EDGE - 3
COLUMN = BOX + (6 if BIG else 4)
RULE = H // 2 if BIG else 18
POINTS = (RIGHT - LEFT) // SPACING + 2
history = deque([0.0] * POINTS, maxlen=POINTS)
START = time.time()
_state = {
    "next": 0.0,
    "samples": 0,
    "shown": 0.0,
    "load": 0.0,
    "mem": 0.0,
    "temp": None,
    "up": "",
}


def chart(d, tick, s):
    """The load of the last samples in steps, scaled to the highest tenth
    among them, moved left by whole pixels between two samples."""
    scale = max(0.1, math.ceil(max(history) * 10) / 10)
    shift = round((s["next"] - tick.time) / SAMPLE * SPACING) - SPACING
    span = BOTTOM - TOP
    points = []
    for i, v in enumerate(history):
        x = RIGHT - (POINTS - 1 - i) * SPACING + shift
        if x >= LEFT - SPACING:
            y = BOTTOM - round(min(v / scale, 1) * span)
            if points:
                points.append((max(LEFT, x), points[-1][1]))
            points.append((max(LEFT, x), y))
    d.line((LEFT, TOP), (RIGHT, TOP), color=DIM)
    d.lines(points, color=INK, stroke_width=1)
    hx, hy = points[-1]
    d.rect(
        x=hx - 1,
        y=hy - 1,
        width=3,
        height=3,
        color=INK,
        visible=tick.time % 0.5 < 0.25,
    )
    return scale


@ghee.tick
def show(tick):
    s = _state
    if tick.time >= s["next"]:
        history.append(cpu())
        s["next"] += SAMPLE
        s["samples"] += 1
        if s["samples"] % 4 == 1:
            s["load"] = sum(list(history)[-4:]) / 4
            s["mem"], s["temp"], s["up"] = memory(), temperature(), uptime()
    s["shown"] += (s["load"] - s["shown"]) * min(1.0, tick.dt * 6)
    temp = "--" if s["temp"] is None else f"{s['temp']:.0f}°C"
    right = W - 1
    with ghee.draw() as d:
        d.rect(
            x=0,
            y=0,
            width=BOX,
            height=H,
            color="transparent",
            border={"width": EDGE, "color": FRAME},
        )
        d.line((round(BOX * 0.45), RULE), (RIGHT, RULE), color=DIM)
        d.line((BOX - EDGE, RULE), (right, RULE), color=FRAME)
        scale = chart(d, tick, s)
        text(d, "CPU", COLUMN, 4 if BIG else 1)
        text(d, f"{s['load']:.0%}", right, 4 if BIG else 1, right=True)
        if BIG:
            clock = time.strftime("%H:%M:%S", time.localtime(START + tick.time))
            text(d, clock, LEFT + 1, EDGE + 3)
            text(d, f"{scale:.0%}", RIGHT, EDGE + 3, right=True)
            text(d, "MEM", COLUMN, 17)
            text(d, f"{s['mem']:.0%}", right, 17, right=True)
            bar, end = COLUMN + 4 * 7 + 3, right - 4 * 7 - 2
            blocks(d, bar, 4, end - bar, 8, s["shown"])
            blocks(d, bar, 17, end - bar, 8, s["mem"])
            text(d, "TEMP", COLUMN, 38)
            text(d, temp, right, 38, right=True)
            text(d, "UP", COLUMN, 51)
            text(d, s["up"], right, 51, right=True)
        else:
            blocks(d, COLUMN, 11, right - COLUMN + 1, 4, s["shown"])
            rows = [("MEM", f"{s['mem']:.0%}"), ("TEMP", temp), ("UP", s["up"])]
            label, value = rows[int(tick.time // 3) % len(rows)]
            text(d, label, COLUMN, 23)
            text(d, value, right, 23, right=True)


ghee.run()
