#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render==0.4.1.dev0"]
# ///
"""The machine's load, memory, temperature and uptime as a teletext page,
drawn from Python on a 128x32 or a 256x64 panel.

Each tick draws the whole screen with `ghee.draw()` on teletext's grid of
6x10 cells in Bedstead: the header row with the clock, a double-height title
on a band of colour, the readings with bars of blocks, and the load's history
in block graphics, two columns of samples to a cell. The 128x32 page turns
through the readings one at a time. The scene holds no GLSL.

    ghee render --demo stats4 --display kms
    ./stats4.py -o stats4.png --size 256x64
"""

import math
import os
import platform
import time
from collections import deque

import ghee

ghee.args()  # the probe, this scene takes no arguments

FONT = "fonts/Bedstead.otf"
TALL = "fonts/Bedstead-UltraCondensed.otf"  # half as wide: at 20 px, double height
WHITE = "#ffffff"
YELLOW = "#ffff00"
CYAN = "#00ffff"
GREEN = "#00ff00"
BLUE = "#0000ff"
SAMPLE = 0.5  # seconds between two samples of the history

W, H = ghee.render_size()
COLS, ROWS = W // 6, H // 10
X0, Y0 = (W - COLS * 6) // 2, (H - ROWS * 10) // 2
HOST = platform.node().split(".")[0].upper()

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


def sextant(cells):
    """The block graphic that lights `cells`, a 2x3 cell's six bits, top
    left first, row by row."""
    if cells == 0:
        return " "
    special = {21: "▌", 42: "▐", 63: "█"}
    if cells in special:
        return special[cells]
    return chr(0x1FB00 + cells - 1 - (cells > 21) - (cells > 42))


def graph(samples, rows):
    """Block graphics `rows` cells tall that fill each sample's column from
    the bottom, two samples to a cell, as one line of words per row. The top
    stands for the highest tenth among the samples."""
    levels = rows * 3
    top = max(0.1, math.ceil(max(samples) * 10) / 10)
    samples = [v / top for v in samples]
    lines = []
    for row in range(rows):
        line = ""
        for i in range(0, len(samples) - 1, 2):
            cells = 0
            for side, v in enumerate(samples[i : i + 2]):
                lit = round(min(v, 1) * levels)
                for k in range(3):
                    if lit >= (rows - 1 - row) * 3 + 3 - k:
                        cells |= 1 << (k * 2 + side)
            line += sextant(cells)
        lines.append(line)
    return "\n".join(lines)


def put(d, words, col, row, color=WHITE, tall=False):
    """Words from a cell of the grid, in a double-height row when `tall`."""
    d.text(
        words,
        x=X0 + col * 6,
        y=Y0 + row * 10,
        font_size=20 if tall else 10,
        font_family=TALL if tall else FONT,
        color=color,
        render="mono",
    )


def band(d, col, row, cols, rows, color):
    """A background colour over a run of cells."""
    d.rect(
        x=X0 + col * 6, y=Y0 + row * 10, width=cols * 6, height=rows * 10, color=color
    )


def bar(d, share, col, row, cells):
    """A bar of green blocks lit to `share` of `cells` in half cells, and blue
    dots leading on to the end."""
    halves = round(min(share, 1) * cells * 2)
    blocks = "█" * (halves // 2) + ("▌" if halves % 2 else "")
    put(d, blocks, col, row, GREEN)
    put(d, "." * (cells - len(blocks)), col + len(blocks), row, BLUE)


history = deque([0.0] * 84, maxlen=84)
START = time.time()
_state = {"next": 0.0, "load": 0.0, "mem": 0.0, "temp": None, "up": ""}


@ghee.tick
def show(tick):
    s = _state
    if tick.time >= s["next"]:
        s["next"] += SAMPLE
        s["load"], s["mem"] = cpu(), memory()
        history.append(s["load"])
        s["temp"], s["up"] = temperature(), uptime()
    now = time.localtime(START + tick.time)
    clock = time.strftime("%H:%M/%S", now)
    temp = "--" if s["temp"] is None else f"{s['temp']:.0f}°C"
    with ghee.draw() as d:
        if ROWS >= 6:
            band(d, 7, 1, COLS - 7, 2, BLUE)
            for i in range(3):
                d.rect(x=X0 + i * 14, y=Y0 + 10, width=12, height=20, color=WHITE)
            band(d, 0, 5, COLS, 1, BLUE)
            put(d, "P100", 0, 0)
            put(d, HOST[: COLS - 25], 5, 0)
            put(d, time.strftime("%a %d %b", now), COLS - 19, 0)
            put(d, clock, COLS - 8, 0, YELLOW)
            for i, letter in enumerate("SYS"):
                d.text(
                    letter,
                    x=X0 + i * 14 + 3,
                    y=Y0 + 10,
                    font_size=20,
                    font_family=TALL,
                    color="black",
                    render="mono",
                )
            put(d, "STATUS", 9, 1, YELLOW, tall=True)
            if s["load"] > 0.9:
                put(d, "HOT", COLS - 5, 1, WHITE, tall=True)
            half = COLS // 2 - 1
            for row, (name, share) in enumerate(
                (("CPU", s["load"]), ("MEM", s["mem"]))
            ):
                put(d, name, 0, 3 + row, CYAN)
                bar(d, share, 4, 3 + row, half - 9)
                put(d, f"{share:>4.0%}", half - 4, 3 + row)
            put(
                d,
                graph(list(history)[-(COLS - half - 1) * 2 :], 2),
                half + 1,
                3,
                GREEN,
            )
            put(d, "TEMP", 1, 5, YELLOW)
            put(d, temp, 6, 5)
            put(d, "UP", 14, 5, YELLOW)
            put(d, s["up"], 17, 5)
            put(d, "LOAD", 27, 5, YELLOW)
            put(d, f"{os.getloadavg()[0]:.2f}", 32, 5)
        else:
            pages = [
                ("CPU", f"{s['load']:.0%}"),
                ("MEM", f"{s['mem']:.0%}"),
                ("TEMP", temp),
                ("UP", s["up"]),
            ]
            page = int(tick.time // 4) % len(pages)
            name, value = pages[page]
            put(d, f"P10{page + 1}", 0, 0)
            put(d, time.strftime("%a %d", now), 5, 0)
            put(d, clock, COLS - 8, 0, YELLOW)
            put(d, name, 0, 1, CYAN, tall=True)
            put(d, value, len(name) + 1, 1, YELLOW, tall=True)
            start = len(name) + len(value) + 2
            if page == 0:
                samples = list(history)[-(COLS - start) * 2 :]
                put(d, graph(samples, 2), start, 1, GREEN)
            elif page == 1:
                bar(d, s["mem"], start, 2, COLS - start)


ghee.run()
