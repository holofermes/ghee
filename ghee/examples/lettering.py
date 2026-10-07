#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render"]
# ///
"""Words that the GPU draws from the outlines of a font, sharp at every size.

The rows stand on a ladder of sizes, each `grow` times the one above it, and
the ladder slides down: a row grows as it falls and scrolls sideways, and a
new one enters at the top. The text layer `words` is hidden, so the pass
shader alone shows its words: it reads them with `words_cover` at the font
size of each row, and an edge stays one pixel wide at any size.

The tick hook steps the words through WORDS: it sets the layer's text
through `ghee.layers`, and ghee lays out each new entry with no recompile.
Words set in the console end the steps.

    ghee render --demo lettering -o out.####.png --frames 60 --timestep 0.1
    ghee render --demo lettering --display st7789
    ghee render --demo lettering --serve            # type the words live
"""

import ghee

ghee.args()  # the probe, this scene takes no arguments

WORDS = ("ghee", "lettering", "sharp at any size", "set live")
HOLD = 4.0  # seconds on each entry of WORDS

SHADER = """
// A row under this font size, in pixels per em, reads as noise, so it draws
// nothing.
const float MIN_SIZE = 4.0;
// The space between two repeats of the words along a row, in em.
const float GAP = 1.0;

void main() {
    // VC4 interpolates v_texCoord up to 0.04 px off the pixel centre.
    vec2 px = floor(v_texCoord * resolution) + 0.5;

    // Row k spans grow^(k + time * zoom) to grow^(k + 1 + time * zoom) of
    // the frame height. rise is this pixel's height over its row's top.
    float lg = log(grow);
    float u = log(px.y / resolution.y) / lg - time * zoom;
    float k = floor(u);
    float rise = exp((u - k) * lg);
    // The words fill the row, so the row's height sets their font size.
    float font_size = px.y / rise * (grow - 1.0) / words_size.y;

    // Rows run in turn left and right, each 0.37 of a repeat on from the
    // row above, so no column of words lines up.
    float period = words_size.x + GAP;
    float side = mod(k, 2.0) * 2.0 - 1.0;
    float x = (px.x - 0.5 * resolution.x) / font_size + 0.5 * words_size.x
            + side * time * scroll + k * 0.37 * period;
    vec2 p = vec2(mod(x, period), (rise - 1.0) / (grow - 1.0) * words_size.y);
    p = font_size < MIN_SIZE ? vec2(-1.0e4) : p;

    float c = words_cover(p, font_size);
    float far = 1.0 - v_texCoord.y;
    gl_FragColor = vec4(mix(paper, ink, c * (1.0 - 0.5 * far * far)), 1.0);
}
"""


def build():
    return ghee.Scene(
        passes=[
            ghee.Pass(
                name="main",
                code=SHADER,
                layers=[
                    ghee.Layer("words", WORDS[0], visible=False, text_align="center")
                ],
            )
        ],
        params={
            "zoom": ghee.Param("zoom", "float", 0.25, min=-1.0, max=1.0, step=0.01),
            "scroll": ghee.Param("scroll", "float", 0.4, min=-2.0, max=2.0, step=0.01),
            "grow": ghee.Param("grow", "float", 1.6, min=1.1, max=3.0, step=0.01),
            "ink": ghee.Param("ink", "color", [0.96, 0.9, 0.76]),
            "paper": ghee.Param("paper", "color", [0.04, 0.05, 0.09]),
        },
    )


_steps = {"words": WORDS[0], "on": True}


@ghee.tick
def step(tick):
    """Step `words` through WORDS until the console sets words of its own.

    The first tick carries the words as the scene declares them.
    """
    given = tick.params.get("words")
    if given is not None and given != _steps["words"]:
        _steps["on"] = False
    if not _steps["on"]:
        return
    word = WORDS[int(tick.time / HOLD) % len(WORDS)]
    if word != _steps["words"]:
        _steps["words"] = word
        ghee.layers.words.text = word


ghee.run(build())
