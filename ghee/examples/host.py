# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render==0.4.1.dev0"]
# ///
"""A Python program that opens a screen and draws on it in its own loop.

ghee.open() takes a scene built in code and shows it on the connected
display, or headless at --size. The scene draws ROWS pen lines on paper, one
for each row of its 2D array `walk`. Each turn of the loop moves the pen one
step along a random walk, feeds the rows to the scene, sets the param `pen`
to where the pen got, and draws the time over the paper with screen.draw(),
which renders the frame. At the right edge the rows move up, and the pen
starts a new one. The console on http://127.0.0.1:8090 shows the same frames
and sets the other params.

The program runs ghee, so ghee never starts this file: it is no scene, and
`--demo` does not list it.

    uv run host.py
    uv run host.py --size 320x240 -o frame.####.png --frames 120
"""

import argparse
import random
import time
from array import array

import ghee

STEPS = 64  # the points along a row
ROWS = 8

SHADER = (
    f"const int ROWS = {ROWS};\n"
    + """
void main() {
    float gap = 1.0 / float(ROWS + 1);
    vec3 col = paper;
    for (int r = 0; r < ROWS; r++) {
        float row = float(r);
        float v = walk_sample2(vec2(uv.x, 1.0 - (row + 0.5) / float(ROWS)));
        float y = (row + 1.0) * gap + 0.45 * gap * v;
        float d = abs(uv.y - y) * iResolution.y;
        float ink = 1.0 - smoothstep(0.5 * weight, 0.5 * weight + 1.0, d);
        float newest = max(1.0 - row, 0.0);
        ink *= 1.0 - newest * smoothstep(pen, pen + 0.004, uv.x);
        col = mix(col, mix(pencil, graphite, newest), ink);
    }
    gl_FragColor = vec4(col, 1.0);
}
"""
)

SCENE = ghee.Scene(
    passes=[
        ghee.Pass(
            "main",
            SHADER,
            arrays=[ghee.ArrayInput("walk", (STEPS, ROWS))],
            params=[
                ghee.Param("pen", "float", 0.0, min=0.0, max=1.0, step=0.001),
                ghee.Param("weight", "float", 2.5, min=0.5, max=6.0, step=0.1),
                ghee.Param("paper", "color", "#dcdfe0"),
                ghee.Param("pencil", "color", "#6e7275"),
                ghee.Param("graphite", "color", "#121416"),
            ],
        )
    ]
)

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

walk = random.Random(7)
rows = [[0.0] * STEPS for _ in range(ROWS)]
step = 0
drift = 0.0
with ghee.open(SCENE, fps=30, serve=args.port, **where) as screen:
    frame = 0
    try:
        while not args.frames or frame < args.frames:
            if step == STEPS:
                rows = [[rows[0][-1]] * STEPS] + rows[:-1]
                step = 0
            here = rows[0][step - 1] if step else rows[0][0]
            drift = 0.85 * drift + walk.uniform(-0.06, 0.06)
            here = max(-1.0, min(1.0, here + drift))
            rows[0][step:] = [here] * (STEPS - step)
            step += 1
            screen.feed("walk", array("f", [v for row in rows for v in row]))
            screen.set_param("pen", (step - 0.5) / STEPS)
            with screen.draw() as d:
                d.text(
                    time.strftime("%H:%M:%S"), x=6, y=4, font_size=12, color="#121416"
                )
            if args.out:
                screen.save(args.out)
            frame += 1
    except KeyboardInterrupt:
        pass
