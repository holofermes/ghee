#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render"]
# ///
"""A karaoke line that fills as it is sung, mirrored in water by a later pass.

The pass `lyrics` has a layer and no code, so its buffer starts each frame
transparent and holds the words alone. Its text layer `line` fits the words
to the pass and draws them with a dark glow and the karaoke effect. The tick
hook sets the effect's `sung`, how many characters are sung, through
`ghee.layers`: it sweeps the line in BEAT seconds, then holds the whole line
sung for a second. The karaoke effect fills the sung letters coral and lifts
the letter being sung. The pass `main` paints the dusk and a moon low over
the sea, and reads `lyrics` through iChannel0 twice: as it is, and flipped
and rippled below the horizon, where the moon's reflection ripples too.

The emoji come from Noto Color Emoji, which ghee finds beside the scene,
through $GHEE_ASSETS, or in /usr/share/fonts/truetype/noto/. Without it,
each emoji draws as the missing-glyph box.

    ghee render --demo karaoke -o out.####.png --frames 40 --timestep 0.1
    ghee render --demo karaoke --display kms
"""

import ghee

ghee.args()  # the probe, this scene takes no arguments

LINE = "sing it with me 🎤✨"
BEAT = 4.0  # seconds to sing LINE

SKY = """
// The moon's cover at p, a disc low over the horizon.
float disc(vec2 p) {
    vec2 aspect = vec2(iResolution.x / iResolution.y, 1.0);
    vec2 d = (p - vec2(0.5, horizon - 0.035)) * aspect;
    return 1.0 - smoothstep(0.07, 0.075, length(d));
}

void main() {
    // The water mirrors the lyrics about the horizon, and ripples sideways.
    float depth = max(uv.y - horizon, 0.0);
    vec2 mirror = vec2(uv.x + 0.004 * sin(depth * 90.0 - iTime * 3.0),
                       2.0 * horizon - uv.y);
    float fade = 1.0 - depth * 2.0;
    vec4 words = texture2D(iChannel0, uv);
    // On VC4 a short pass that reads a texture, then selects on a compare,
    // can draw wrong 4x4 blocks when it runs on two threads. Its last read
    // inside an `if` keeps it on one.
    vec4 shine = vec4(0.0);
    if (gl_FragCoord.x >= 0.0) shine = texture2D(iChannel0, mirror) * 0.45 * fade;

    vec3 air = mix(mix(night, sky, uv.y / horizon), moon, disc(uv));
    vec3 water = mix(mix(night, sky, 0.2), moon, disc(mirror) * 0.4 * fade);
    // Both reads hold colour premultiplied by alpha, so each composites over
    // its background as c + background * (1 - alpha).
    vec3 above = words.rgb + air * (1.0 - words.a);
    vec3 below = shine.rgb + water * (1.0 - shine.a);
    gl_FragColor = vec4(mix(above, below, step(horizon, uv.y)), 1.0);
}
"""


@ghee.tick
def sing(tick):
    """Set the sung point: LINE in BEAT seconds, then a second's hold."""
    sung = min(tick.time % (BEAT + 1.0) / BEAT, 1.0) * len(LINE)
    ghee.layers.line.karaoke.sung = sung


def build():
    return ghee.Scene(
        params={
            "horizon": ghee.Param(
                "horizon", "float", 0.66, min=0.3, max=0.9, step=0.01
            ),
        },
        passes=[
            ghee.Pass(
                "lyrics",
                layers=[
                    ghee.Layer(
                        "line",
                        LINE,
                        color="#ddd9cb",
                        glow={"color": "#191523"},
                        karaoke={"color": "#d57b70"},
                    )
                ],
            ),
            ghee.Pass(
                "main",
                SKY,
                channels=[ghee.Channel("lyrics")],
                params=[
                    ghee.Param("night", "color", "#191523"),
                    ghee.Param("sky", "color", "#48625f"),
                    ghee.Param("moon", "color", "#ddd9cb"),
                ],
            ),
        ],
    )


ghee.run(build())
