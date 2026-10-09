#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render"]
# ///
"""A word from a hidden text layer, scrolled by the shader over a wave.

The text layer `word` is hidden, so the pass shader alone shows its words: it
reads them with `word_cover`, scrolls them, then thresholds them and tints
them over a wave.

The words, motion, crispness and color stay live. Line art needs no dither.
The wave behind the text stays under the threshold of a 1-bit panel while
`glow` stays under it.

    ghee render --demo marquee -o out.####.png --frames 60
    ghee render --demo marquee --display kms           # to a panel, no dither
    ghee render --demo marquee -- --text "hello there"
"""

import ghee

args = ghee.args(
    [ghee.Arg("text", "string", default="GHEE", help="the words that scroll")]
)

# Slow imports come after the parse: the probe must answer inside its window.
from lib import look  # noqa: E402

SHADER = """
// The room left and right of the words, in em. The block's line height
// holds the room over and under them.
const float PAD = 0.2;
// Each edge ramps over 1/RAMP em at the least.
const float RAMP = 28.0;

void main() {
    vec2 box = word_size + vec2(2.0 * PAD, 0.0);
    float mask_aspect = box.x / box.y;
    float aspect = iResolution.x / iResolution.y;
    float across = max(1.0, aspect / mask_aspect);
    float down = max(1.0, mask_aspect / aspect);
    float v = (uv.y - 0.5) * down + 0.5;
    vec2 p = vec2(fract(uv.x * across - iTime * speed), v) * box - vec2(PAD, 0.0);
    float m = word_cover(p, min(RAMP, iResolution.y / (down * box.y)));
    m *= step(0.0, v) * step(v, 1.0);
    float g = smoothstep(threshold - 0.08, threshold + 0.08, m);

    float band = exp(-abs(uv.y - 0.5 - wave(uv.x, iTime * 0.5)) * 9.0);
    vec3 col = palette(0.14 + 0.16 * uv.x) * glow * band + ink * g * 2.0;
    gl_FragColor = vec4(tone(col, 0.02, 1.4), 1.0);
}
"""


def build():
    return ghee.Scene(
        common=look.PALETTE + look.WAVE + look.TONE,
        passes=[
            ghee.Pass(
                name="main",
                fragment=SHADER,
                layers=[ghee.Layer("word", args.text, visible=False)],
                params={
                    "speed": ghee.Param(
                        "speed", "float", 0.1, min=-0.6, max=0.6, step=0.01
                    ),
                    "threshold": ghee.Param(
                        "threshold", "float", 0.4, min=0.05, max=0.95, step=0.01
                    ),
                    "glow": ghee.Param(
                        "glow", "float", 0.3, min=0.0, max=1.0, step=0.01
                    ),
                    "ink": ghee.Param("ink", "color", [0.72, 0.92, 1.0]),
                },
            )
        ],
    )


ghee.run(build())
