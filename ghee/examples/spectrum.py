#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render==0.4.1.dev0"]
# ///
"""A spectrum and its history as stacked ridge lines, white on black, from
a scene with no JSON that feeds a 1D and a 2D array.

The scene is built in code. The tick hook stands in for an analyser and
makes one level per band each frame. The feed `bars` sends those levels,
and the shader draws them as the front line. Every 1/`speed` seconds the
hook copies the levels to the top of the history, BANDS x ROWS levels with
the newest row first, which the feed `history` sends. The shader draws each
row as a line behind the one before, and a line hides the lines behind it.
The hook also sets `scroll`, the share of a row since the last copy, so the
lines glide back between copies. The feeds send raw levels, and `gain`
scales them in GLSL.

    ghee render --demo spectrum -o frame.####.png --frames 60
    ghee render --demo spectrum --serve
    ./spectrum.py -o frame.####.png --size 320x240 --frames 60
"""

import ghee

ghee.args()  # the probe, this scene takes no arguments

# Slow imports come after the parse: the probe must answer inside its window.
import math  # noqa: E402
from array import array  # noqa: E402

BANDS = 48
ROWS = 64
SPEED = 5.0
CHORDS = [17.0, 19.5, 21.0, 20.5]  # the band of each bar's root note
TUNE = [0.0, 1.5, 2.5, 3.5, 5.0, 3.5, 2.5, 1.5]  # bands above the root
ACCENTS = [1.0, 0.6, 0.8, 0.5]

SHADER = (
    f"const float BANDS = {BANDS}.0;\nconst float ROWS = {ROWS}.0;\n"
    + """
float bare = 1.0;
float shine = 0.0;

void ridge(float left, float right, float along, float base, float y, float env, float A, float span, float shown) {
    float level = gain * mix(left, right, along);
    float slope = A * env * gain * (right - left) * (BANDS - 1.0) / span;
    float d = (y - base - A * env * level) / sqrt(1.0 + slope * slope);
    float stroke = clamp(0.5 * weight + 0.5 - abs(d), 0.0, 1.0);
    float under = clamp(0.5 - d, 0.0, 1.0);
    shine += bare * stroke * shown;
    bare *= 1.0 - max(stroke, under) * shown;
}

void main() {
    vec2 px = uv * iResolution.xy;
    float span = 0.56 * iResolution.x;
    float x = (px.x - 0.5 * (iResolution.x - span)) / span;
    float inside = step(0.0, x) * step(x, 1.0);
    x = clamp(x, 0.0, 1.0);
    float y = 0.94 * iResolution.y - px.y;
    float count = min(lines, floor(0.9 * iResolution.y / 4.0 - 0.5 * height));
    float gap = 0.9 * iResolution.y / (count + 0.5 * height);
    float A = height * gap;
    float env = mix(0.12, 1.0, smoothstep(0.02, 0.3, x) * smoothstep(0.98, 0.7, x));

    float t = x * (BANDS - 1.0);
    float i = min(floor(t), BANDS - 2.0);
    float f = t - i;
    vec2 at = vec2((i + 0.5) / BANDS, (i + 1.5) / BANDS);

    float a = 0.0;
    float b = 0.0;
    if (gl_FragCoord.x >= 0.0) {
        a = bars_at(int(i));
        b = bars_at(int(i) + 1);
    }
    ridge(a, b, f, 0.0, y, env, A, span, 1.0);

    float first = floor((y - A - 2.0) / gap - scroll);
    for (int k = 0; k < 12; k++) {
        float row = first + float(k);
        float base = (row + scroll) * gap;
        float there = step(0.0, row) * step(row, count - 1.0);
        float v = (row + 0.5) / ROWS;
        if (gl_FragCoord.x >= 0.0) {
            a = history_sample2(vec2(at.x, v));
            b = history_sample2(vec2(at.y, v));
        }
        float fade = mix(1.0, 1.0 - scroll, step(count - 1.5, row));
        ridge(a, b, f, base, y, env, A, span, there * fade);
    }

    gl_FragColor = vec4(mix(background, line_color, clamp(shine, 0.0, 1.0) * inside), 1.0);
}
"""
)

_levels = array("f", bytes(4 * BANDS))
_history = array("f", bytes(4 * BANDS * ROWS))
_clock = {"speed": SPEED, "phase": 0.0}


def peak(i, centre, width):
    return math.exp(-(((i - centre) / width) ** 2))


@ghee.tick
def listen(tick):
    """The levels of this frame, six bands to an octave, in a song at two
    beats a second: a kick on each beat, a hi-hat between them, a bass and
    a chord that change every bar, and a tune of eight notes to the bar.
    Each note stands as a peak, with its harmonics above it. Every
    1/speed seconds the levels go on top of the history."""
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

    _clock["speed"] = tick.params.get("speed", _clock["speed"])
    _clock["phase"] += tick.dt * _clock["speed"]
    while _clock["phase"] >= 1.0:
        _history[BANDS:] = _history[:-BANDS]
        _history[:BANDS] = _levels
        _clock["phase"] -= 1.0
    ghee.set_param("scroll", _clock["phase"])


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
                    ghee.ArrayInput("history", (BANDS, ROWS)),
                ],
                params=[
                    ghee.Param("gain", "float", 1.0, min=0.0, max=2.0, step=0.01),
                    ghee.Param("scroll", "float", 0.0, min=0.0, max=1.0, step=0.01),
                    ghee.Param("lines", "float", 40.0, min=4.0, max=63.0, step=1.0),
                    ghee.Param("height", "float", 8.0, min=0.0, max=10.0, step=0.1),
                    ghee.Param("weight", "float", 1.2, min=0.5, max=3.0, step=0.1),
                    ghee.Param("background", "color", "#000000"),
                    ghee.Param("line_color", "color", "#ffffff"),
                ],
            )
        ],
    )


ghee.run(build())
