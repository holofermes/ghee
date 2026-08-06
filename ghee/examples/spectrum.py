#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["ghee-render"]
# ///
"""A scene with no JSON. It builds its pass in code, and feeds one array each
frame.

The shader reads `gain` twice: for the bar height and for the brightness. The
feed sends raw levels; every scaling stays in GLSL.

    ghee render --demo spectrum -o out.####.png --frames 60
    ghee serve --demo spectrum                      # drive it in the console
"""

import ghee

ghee.args()  # the probe, this scene takes no arguments

# Slow imports come after the parse: the probe must answer inside its window.
import math  # noqa: E402
from array import array  # noqa: E402

from lib import look  # noqa: E402

BARS = 48

SHADER = """
void main() {
    vec2 uv = vec2(v_texCoord.x, 1.0 - v_texCoord.y);  // y climbs off the floor
    float fi = uv.x * float(bars_size);
    int i = int(min(fi, float(bars_size - 1)));
    float h = gain * bars_at(i);

    // bars stand on the baseline, their reflection hangs under it
    float base = 0.34;
    float under = step(uv.y, base);
    float y = mix(uv.y - base, (base - uv.y) * 2.4, under) / (1.0 - base);

    float column = smoothstep(0.46, 0.30, abs(fract(fi) - 0.5));
    float body = column * (1.0 - smoothstep(h - 0.004, h + 0.004, y));
    float cap = body * smoothstep(h - 0.12, h, y);

    vec3 col = palette(0.16 + 0.30 * uv.x + 0.06 * h);
    col *= (body * (0.15 + 0.6 * y) + cap * 1.3) * mix(1.0, 0.30, under) * (0.4 + gain);
    gl_FragColor = vec4(tone(col, 0.02, 1.5), 1.0);
}
"""


def build():
    return ghee.Scene(
        common=look.PALETTE + look.TONE,
        passes=[
            ghee.Pass(
                name="main",
                code=SHADER,
                arrays=[ghee.ArrayInput("bars", BARS)],
                params={
                    "gain": ghee.Param(
                        "gain", "float", 0.6, min=0.0, max=1.0, step=0.01
                    )
                },
            )
        ],
    )


@ghee.feed("bars")
def bars(tick):
    t = tick.frame * 0.05
    out = array("f", bytes(4 * BARS))
    for i in range(BARS):
        x = i / (BARS - 1)
        # look.WAVE: two sines that are not quite harmonics running opposite
        # ways, so the crest travels and never lands twice on the same bar
        w = 0.28 * math.sin(x * 4.4 + t) + 0.10 * math.sin(x * 9.7 - t * 0.6)
        tilt = math.exp(-1.7 * x)  # a spectrum leans on its low end
        out[i] = (0.55 + tilt) * (0.65 + w)
    return out


ghee.run(build())
