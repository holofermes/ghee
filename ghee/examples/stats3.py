#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render==0.4.1.dev0"]
# ///
"""The time and the machine's load, memory, temperature and uptime as an
alarm clock's display, drawn from Python on a 128x32 or a 256x64 panel.

Each tick draws the whole screen with `ghee.draw()`: seven-segment digits
over the faint segments that stay unlit, each reading named in Silkscreen,
the load as a bar of segments that grow along it, and HOT, lit and blinking
past 90% of the cores. The layout follows the screen's size, and the scene
holds no GLSL.

    ghee render --demo stats3 --display kms
    ./stats3.py -o stats3.png --size 256x64
"""

import functools
import time

import ghee

ghee.args()  # the probe, this scene takes no arguments

SEVEN = "fonts/LCDSeven-Bold.ttf"
TAGS = "fonts/Silkscreen-Regular.ttf"
LIT = "#62f4d6"
UNLIT = "#123630"
SAMPLE = 0.5  # seconds between two readings

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
    """The time since boot in whole days, or in hours below a day."""
    with open("/proc/uptime") as f:
        s = int(float(f.read().split()[0]))
    return (s // 86400, "D") if s >= 86400 else (s // 3600, "H")


@functools.cache
def seven(words, size):
    return ghee.measure(words, font_size=size, font_family=SEVEN)[0]


def segments(d, value, x, y, size, digits):
    """`value` right-aligned in `digits` seven-segment digits, over the unlit
    segments of every digit."""
    ghost = "8" * digits
    lit = f"{value}"[-digits:]
    shift = seven(ghost, size) - seven(lit, size)
    d.text(ghost, x=x, y=y, font_size=size, font_family=SEVEN, color=UNLIT)
    d.text(lit, x=x + shift, y=y, font_size=size, font_family=SEVEN, color=LIT)


def tag(d, words, x, y, on=True, right=False, **keys):
    """Small printed letters with their capitals' bottom row at y, lit or
    unlit, from x or ending at x."""
    if right:
        x -= ghee.measure(words, font_size=8, font_family=TAGS)[0]
    d.text(
        words,
        x=x,
        y=y - 7,
        font_size=8,
        font_family=TAGS,
        color=LIT if on else UNLIT,
        render="mono",
        **keys,
    )


def wedge(d, x, y, w, h, share):
    """A bar of segments 3 px wide that grow from a third of `h` to `h`,
    lit up to `share` of its length."""
    n = (w + 2) // 5
    lit = round(share * n)
    for i in range(n):
        tall = round(h / 3 + (h - h / 3) * i / max(n - 1, 1))
        d.rect(
            x=x + 5 * i,
            y=y + h - tall,
            width=3,
            height=tall,
            color=LIT if i < lit else UNLIT,
        )


CLOCK = 40 if BIG else 22
SECONDS = 16
READ = 13 if BIG else 15
START = time.time()
_state = {"next": 0.0, "load": 0.0, "mem": 0.0, "temp": None, "up": (0, "H")}


@ghee.tick
def show(tick):
    s = _state
    if tick.time >= s["next"]:
        s["next"] += SAMPLE
        s["load"], s["mem"] = cpu(), memory()
        s["temp"], s["up"] = temperature(), uptime()
    now = time.localtime(START + tick.time)
    colon = ":" if (START + tick.time) % 1 < 0.5 else " "
    hot = s["load"] > 0.9
    temp = "--" if s["temp"] is None else round(s["temp"])
    readings = [
        ("CPU", round(s["load"] * 100), "%"),
        ("MEM", round(s["mem"] * 100), "%"),
        ("TEMP", temp, "°C"),
        ("UP", s["up"][0], s["up"][1]),
    ]
    right = W - 1
    top = 2 if BIG else 1
    clock = seven("88:88", CLOCK)
    with ghee.draw() as d:
        words = time.strftime("%H{}%M", now).format(colon)
        for text, color in (("88:88", UNLIT), (words, LIT)):
            d.text(text, x=1, y=top, font_size=CLOCK, font_family=SEVEN, color=color)
        under = top + CLOCK + (6 if BIG else 4)
        if BIG:
            x = round(1 + clock + 7)
            segments(d, f"{now.tm_sec:02}", x, top + CLOCK - SECONDS, SECONDS, 2)
            tag(d, "HOT", x, top + 6, on=hot, blink={"rate": 2, "visible": hot})
            wedge(
                d, 1, under, x + round(seven("88", SECONDS)), H - under - 2, s["load"]
            )
            column = x + round(seven("88", SECONDS)) + 10
            for i, (name, value, unit) in enumerate(readings):
                y = 1 + i * 16
                tag(d, name, column, y + READ - 1)
                segments(d, value, right - 11 - round(seven("888", READ)), y, READ, 3)
                tag(d, unit, right, y + READ - 1, right=True)
        else:
            wedge(d, 1, under, round(clock), H - under, s["load"])
            column = round(1 + clock + 6)
            name, value, unit = readings[int(tick.time // 3) % len(readings)]
            tag(d, name, column, 6)
            tag(d, unit, right, 6, right=True)
            segments(d, value, right - round(seven("888", READ)), 10, READ, 3)


ghee.run()
