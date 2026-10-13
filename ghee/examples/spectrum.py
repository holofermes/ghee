#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render"]
# ///
"""A spectrum and its history, from a scene with no JSON that feeds a 1D
and a 2D array each frame.

The scene is built in code. The tick hook stands in for an analyser and
makes one level per band. The feed `bars` sends those levels, and the
shader stands them as blocks in a strip at the right edge. The feed
`history` sends the last ROWS frames of levels as a BANDS x ROWS grid, the
newest row first, and the shader reads it with `history_sample2(uv)` as a
heat map that scrolls from the strip to the left, the low bands at the
bottom. The feeds send raw levels, and `gain` scales them in GLSL.

    ghee render --demo spectrum -o frame.####.png --frames 60
    ghee render --demo spectrum --serve
    ./spectrum.py -o frame.####.png --size 320x240 --frames 60
"""

import ghee

ghee.args()  # the probe, this scene takes no arguments

# Slow imports come after the parse: the probe must answer inside its window.
import math  # noqa: E402
from array import array  # noqa: E402

BANDS = 32
ROWS = 64
NOTES = [5, 8, 6, 10]  # the band of each beat's note, and its harmonics

SHADER = (
    f"const float ROWS = {ROWS}.0;\n"
    + """
const float STRIP = 0.12;  // the live strip's share of the width, at the right

// step(edge, v) with no compare, which VC4 can misdraw after a texture read
float over(float v, float edge) {
    return clamp((v - edge) * 400.0, 0.0, 1.0);
}

vec3 heat(float v) {
    vec3 col = mix(soot, ember, over(v, 0.25));
    col = mix(col, flame, over(v, 0.5));
    return mix(col, ash, over(v, 0.75));
}

void main() {
    float b = floor((1.0 - uv.y) * float(bars_count));
    float band = (b + 0.5) / float(bars_count);
    float edge = 1.0 - STRIP;
    float age = floor(max(edge - uv.x, 0.0) / edge * ROWS);
    vec3 col = heat(gain * history_sample2(vec2(band, (age + 0.5) / ROWS)));

    float x = (uv.x - edge) / STRIP;
    float live = gain * bars_at(int(b));
    float block = over(x, 0.1) * (1.0 - over(x, 0.1 + 0.9 * min(live, 1.0)));
    vec3 strip = mix(soot, ash, block);
    gl_FragColor = vec4(mix(col, strip, over(uv.x, edge)), 1.0);
}
"""
)

_levels = array("f", bytes(4 * BANDS))
_history = array("f", bytes(4 * BANDS * ROWS))


@ghee.tick
def listen(tick):
    """The levels of this frame, pushed onto the history: a kick and a
    note on each beat, two beats a second, and a hi-hat between them."""
    beats = tick.time * 2.0
    beat = beats % 1.0
    note = NOTES[int(beats) % len(NOTES)]
    for i in range(BANDS):
        x = i / (BANDS - 1)
        level = 0.15 + 0.05 * math.sin(i * 1.7 + tick.time * 3.0)
        level += 0.9 * math.exp(-8.0 * beat - 6.0 * x)
        level += 0.5 * math.exp(-14.0 * ((beat + 0.5) % 1.0)) * x
        if i % note == 0 and i // note in (1, 2, 3):
            level += (1.0 - 0.6 * beat) * 0.8 / (i // note)
        _levels[i] = level
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
        passes=[
            ghee.Pass(
                name="main",
                fragment=SHADER,
                arrays=[
                    ghee.ArrayInput("bars", BANDS),
                    ghee.ArrayInput("history", (BANDS, ROWS)),
                ],
                params=[
                    ghee.Param("gain", "float", 1.0, min=0.0, max=2.0, step=0.01),
                    ghee.Param("soot", "color", "#131313"),
                    ghee.Param("ember", "color", "#bd393d"),
                    ghee.Param("flame", "color", "#f5855a"),
                    ghee.Param("ash", "color", "#f7c787"),
                ],
            )
        ],
    )


ghee.run(build())
