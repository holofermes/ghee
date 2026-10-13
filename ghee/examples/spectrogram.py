#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render==0.4.1.dev0"]
# ///
"""A spectrogram that a playhead paints from left to right, from a scene
with no JSON that feeds a 1D and a 2D array.

The scene is built in code. The tick hook stands in for an analyser and
makes one level per band each frame. The feed `history` sends BANDS x
COLUMNS levels, one row of levels for each column of the picture, and the
hook moves `head` across the columns at `speed` columns a second. Each
column that the playhead leaves takes the loudest level of each band since
the column before, so the picture holds the last COLUMNS / speed seconds,
the low bands at the bottom. The feed `bars` sends the levels of each
frame, and the shader paints them between the last column and the
playhead. The feeds send raw levels, and `gain` scales them in GLSL.

    ghee render --demo spectrogram -o frame.####.png --frames 60
    ghee render --demo spectrogram --serve
    ./spectrogram.py -o frame.####.png --size 320x240 --frames 60
"""

import ghee

ghee.args()  # the probe, this scene takes no arguments

# Slow imports come after the parse: the probe must answer inside its window.
import math  # noqa: E402
from array import array  # noqa: E402

BANDS = 48
COLUMNS = 128
SPEED = 12.0
CHORDS = [17.0, 19.5, 21.0, 20.5]  # the band of each bar's root note
TUNE = [0.0, 1.5, 2.5, 3.5, 5.0, 3.5, 2.5, 1.5]  # bands above the root
ACCENTS = [1.0, 0.6, 0.8, 0.5]

SHADER = (
    f"const float BANDS = {BANDS}.0;\nconst float COLUMNS = {COLUMNS}.0;\n"
    + """
vec3 ramp(float t) {
    t = clamp(t, 0.0, 1.0) * 4.0;
    vec3 c = mix(ramp1, ramp2, clamp(t, 0.0, 1.0));
    c = mix(c, ramp3, clamp(t - 1.0, 0.0, 1.0));
    c = mix(c, ramp4, clamp(t - 2.0, 0.0, 1.0));
    return mix(c, ramp5, clamp(t - 3.0, 0.0, 1.0));
}

void main() {
    float u = ((1.0 - uv.y) * (BANDS - 1.0) + 0.5) / BANDS;
    float column = uv.x * COLUMNS;
    float done = floor(head);
    float behind = step(column, done);
    float v = mix(max(column, done + 0.5), min(column, done - 0.5), behind) / COLUMNS;

    float past = 0.0;
    float now = 0.0;
    if (gl_FragCoord.x >= 0.0) {
        past = history_sample2(vec2(u, v));
        now = bars_sample(u);
    }
    float painting = step(done, column) * step(column, head);
    vec3 col = ramp(1.25 * sqrt(gain * mix(past, now, painting)) - 0.25);

    float x = (uv.x * COLUMNS - head) * iResolution.x / COLUMNS;
    float line = clamp(1.0 - abs(x), 0.0, 1.0);
    gl_FragColor = vec4(mix(col, playhead, line), 1.0);
}
"""
)

_levels = array("f", bytes(4 * BANDS))
_history = array("f", bytes(4 * BANDS * COLUMNS))
_held = array("f", bytes(4 * BANDS))
_clock = {"speed": SPEED, "head": 0.0}


def peak(i, centre, width):
    return math.exp(-(((i - centre) / width) ** 2))


@ghee.tick
def listen(tick):
    """The levels of this frame, six bands to an octave, in a song at two
    beats a second: a kick on each beat, a hi-hat between them, a bass and
    a chord that change every bar, and a tune of eight notes to the bar.
    Each note stands as a peak, with its harmonics above it. A column that
    the playhead leaves takes the loudest level of each band since the
    column before."""
    beats = tick.time * 2.0
    beat = beats % 1.0
    bar = beats / 4.0 % 1.0
    root = CHORDS[int(beats / 4.0) % len(CHORDS)]
    step = int(beats * 2.0)
    tune = root + 6.0 + TUNE[step % len(TUNE)]
    sung = (beats * 2.0) % 1.0
    notes = [
        (root - 6.0, 0.45 * (0.6 + 0.4 * math.exp(-3.0 * beat)), 1.6, 2),
        (root, 0.35 * (0.5 + 0.5 * math.exp(-2.0 * bar)), 1.3, 2),
        (root + 2.0, 0.25 * (0.5 + 0.5 * math.exp(-2.0 * bar)), 1.3, 2),
        (root + 3.5, 0.25 * (0.5 + 0.5 * math.exp(-2.0 * bar)), 1.3, 2),
        (tune, 0.5 * math.exp(-3.0 * sung), 1.1, 2),
    ]
    kick = ACCENTS[int(beats) % len(ACCENTS)] * math.exp(-12.0 * beat)
    hat = math.exp(-18.0 * ((beat + 0.5) % 1.0))
    t = tick.time
    for i in range(BANDS):
        level = 0.05 + 0.025 * math.sin(i * 2.7 + t * 4.1) * math.sin(i * 0.9 - t * 2.3)
        level += 0.08 * peak(i, 22.0 + 4.0 * math.sin(t * 0.35), 8.0)
        level += 0.6 * kick * peak(i, 7.0, 2.0)
        level += 0.2 * hat * peak(i, 42.0, 4.0)
        for band, loud, width, harmonics in notes:
            for h in range(1, harmonics + 1):
                level += loud / h * peak(i, band + 6.0 * math.log2(h), width)
        _levels[i] = level
        _held[i] = max(_held[i], level)

    _clock["speed"] = tick.params.get("speed", _clock["speed"])
    head = _clock["head"] + tick.dt * _clock["speed"]
    for column in range(int(_clock["head"]), int(head)):
        at = column % COLUMNS * BANDS
        _history[at : at + BANDS] = _held
        _held[:] = _levels
    _clock["head"] = head % COLUMNS
    ghee.set_param("head", _clock["head"])


@ghee.feed("bars")
def bars(tick):
    return _levels


@ghee.feed("history")
def history(tick):
    return _history


def build():
    return ghee.Scene(
        params=[
            ghee.Param("speed", "float", SPEED, min=0.0, max=30.0, step=0.5),
        ],
        passes=[
            ghee.Pass(
                name="main",
                fragment=SHADER,
                arrays=[
                    ghee.ArrayInput("bars", BANDS),
                    ghee.ArrayInput("history", (BANDS, COLUMNS)),
                ],
                params=[
                    ghee.Param("gain", "float", 1.0, min=0.0, max=2.0, step=0.01),
                    ghee.Param("head", "float", 0.0, min=0.0, max=COLUMNS, step=0.01),
                    ghee.Param("ramp1", "color", "#000000"),
                    ghee.Param("ramp2", "color", "#3b0040"),
                    ghee.Param("ramp3", "color", "#c4001a"),
                    ghee.Param("ramp4", "color", "#ff5a10"),
                    ghee.Param("ramp5", "color", "#ffd23a"),
                    ghee.Param("playhead", "color", "#d8d8d8"),
                ],
            )
        ],
    )


ghee.run(build())
