#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render"]
# ///
"""A spectrum and its history, from a scene with no JSON that feeds a 1D
and a 2D array each frame.

The scene is built in code. The tick hook stands in for an analyser and
makes one level per band. The feed `bars` sends those levels, and the shader
stands a bar on each. The feed `history` sends the last ROWS frames of
levels as a BANDS x ROWS grid, the newest row first, and the shader reads it
with `history_sample2(uv)` as a spectrogram that scrolls down under the
bars. The feeds send raw levels, and `gain` scales them in GLSL.

    ghee render --demo spectrum -o frame.####.png --frames 60
    ghee render --demo spectrum --serve
    ./spectrum.py -o frame.####.png --size 320x240 --frames 60
"""

import ghee

ghee.args()  # the probe, this scene takes no arguments

# Slow imports come after the parse: the probe must answer inside its window.
import math  # noqa: E402
from array import array  # noqa: E402

from lib import look  # noqa: E402

BANDS = 48
ROWS = 40

SHADER = """
const float FLOOR = 0.55;  // the bars stand on this line, the history hangs under it

void main() {
    float fi = uv.x * float(bars_count);
    int i = int(min(fi, float(bars_count - 1)));
    float h = gain * bars_at(i);
    float y = (FLOOR - uv.y) / FLOOR;  // 0 on the floor, 1 at the top
    float column = smoothstep(0.46, 0.30, abs(fract(fi) - 0.5));
    float body = column * (1.0 - smoothstep(h - 0.004, h + 0.004, y));
    vec3 above = palette(0.16 + 0.30 * uv.x + 0.06 * h) * body * (0.3 + y);

    float v = (uv.y - FLOOR) / (1.0 - FLOOR);  // 0 under the floor, 1 at the bottom
    float level = gain * history_sample2(vec2((floor(fi) + 0.5) / float(bars_count), v));
    vec3 below = palette(0.16 + 0.30 * uv.x + 0.06 * level) * level * level * (1.0 - v);

    vec3 col = mix(above, below, smoothstep(FLOOR - 0.002, FLOOR + 0.002, uv.y));
    gl_FragColor = vec4(tone(col * (0.4 + gain), 0.02, 1.5), 1.0);
}
"""

_levels = array("f", bytes(4 * BANDS))
_history = array("f", bytes(4 * BANDS * ROWS))


@ghee.tick
def listen(tick):
    """The levels of this frame, pushed onto the history."""
    t = tick.time
    beat = t * 2.0 % 1.0  # two beats a second
    for i in range(BANDS):
        x = i / (BANDS - 1)
        # two sines that are not quite harmonics, running opposite ways, so
        # the crest travels and never lands twice on the same band
        w = 0.28 * math.sin(x * 4.4 + t) + 0.10 * math.sin(x * 9.7 - t * 0.6)
        kick = math.exp(-8.0 * beat - 4.0 * x)
        hat = math.exp(-12.0 * ((beat + 0.5) % 1.0)) * x
        _levels[i] = 0.5 * (0.65 + w) + 0.9 * kick + 0.6 * hat
    _history[BANDS:] = _history[:-BANDS]
    _history[:BANDS] = _levels


@ghee.feed("bars")
def bars(tick):
    return _levels


@ghee.feed("history")
def history(tick):
    return _history


def build():
    return ghee.Scene(
        common=look.PALETTE + look.TONE,
        passes=[
            ghee.Pass(
                name="main",
                fragment=SHADER,
                arrays=[
                    ghee.ArrayInput("bars", BANDS),
                    ghee.ArrayInput("history", (BANDS, ROWS)),
                ],
                params={
                    "gain": ghee.Param(
                        "gain", "float", 0.6, min=0.0, max=1.0, step=0.01
                    )
                },
            )
        ],
    )


ghee.run(build())
