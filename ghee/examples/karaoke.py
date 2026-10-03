#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["ghee-render"]
# ///
"""A karaoke line drawn by layers, and mirrored in water by a later pass.

The pass `lyrics` has layers and no code, so its buffer starts each frame
transparent and holds the words alone. Two layers of the text `line` paint
into it in order. `halo` reaches PAD em past the letters and turns
`glyph_round_dist()`, the distance to the outline, into a glow. `fill` fills
the letters amber up to the sung point and white past it, and as the last
layer of the text it also paints the emoji from their bitmaps. Both layers
place the glyphs with one `glyph_place`, which lifts the glyph under the sung
point. The pass `main` paints the night and reads `lyrics` through
iChannel0 twice: as it is, and flipped and rippled below the horizon.

The letters come from DejaVu Sans in fonts/, the emoji from Noto Color
Emoji, which ghee finds through $GHEE_ASSETS. Without it, ghee logs the
missing font and draws each emoji as the missing-glyph box of DejaVu Sans.

    ghee render --demo karaoke -o out.####.png --frames 40 --timestep 0.1
    GHEE_ASSETS=/usr/share/fonts/truetype/noto ghee render --demo karaoke --display kms
"""

import ghee

ghee.args()  # the probe, this scene takes no arguments

PAD = 0.35  # how far the halo reaches past the letters, in em

COMMON = """
// The sung point, in em from the left of the line. It sweeps the line in
// `beat` seconds, holds the whole line sung for a second, and starts again.
float sung_at() {
    return min(mod(time, beat + 1.0) / beat, 1.0) * line_size.x;
}
"""

# One placement for both layers, so the halo stays under its letters.
PLACE = """
vec2 glyph_place(vec2 em) {
    // The line fills 0.8 of the frame's width, centred on 0.4 of its height.
    float ppem = 0.8 * resolution.x / line_size.x;
    vec2 corner = vec2(0.5, 0.4) * resolution - 0.5 * line_size * ppem;
    // A glyph rises by up to 0.15 em as the sung point passes its middle.
    float mid = 0.5 * (glyph_span.x + glyph_span.y);
    float d = (mid - sung_at()) * 2.0;
    float hop = 0.15 * exp(-d * d);
    return corner + (em - vec2(0.0, hop)) * ppem;
}
"""

# 1 left of the sung point and 0 right of it, ramped over one pixel.
SUNG = "clamp((sung_at() - glyph_em.x) * glyph_font_size + 0.5, 0.0, 1.0)"

HALO = f"""
void main() {{
    // glyph_round_dist() is the distance to the outline in pixels, > 0
    // inside the letters, and the layer's pad lets it reach PAD em out. The
    // glow fades with that distance in em.
    float far = max(-glyph_round_dist(), 0.0) / glyph_font_size;
    float a = 0.7 * exp(-far * 12.0);
    vec3 c = mix(unsung, sung, {SUNG}) * a;
    gl_FragColor = vec4(c, a);
}}
"""

FILL = f"""
void main() {{
    float c = glyph_cover();
    gl_FragColor = vec4(mix(vec3(1.0), sung, {SUNG}) * c, c);
}}
"""

SKY = """
void main() {
    vec2 uv = v_texCoord;
    // The water mirrors the lyrics about the horizon, and ripples sideways.
    float depth = max(uv.y - horizon, 0.0);
    vec2 mirror = vec2(uv.x + 0.004 * sin(depth * 90.0 - time * 3.0),
                       2.0 * horizon - uv.y);
    vec4 words = texture2D(iChannel0, uv);
    // On VC4 a short pass that reads a texture, then selects on a compare,
    // can draw wrong 4x4 blocks when it runs on two threads. Its last read
    // inside an `if` keeps it on one.
    vec4 shine = vec4(0.0);
    if (gl_FragCoord.x >= 0.0) shine = texture2D(iChannel0, mirror) * 0.45 * (1.0 - depth * 2.0);

    vec3 night = mix(vec3(0.02, 0.02, 0.07), vec3(0.12, 0.06, 0.20), uv.y / horizon);
    vec3 water = vec3(0.01, 0.02, 0.06);
    // Both reads hold colour premultiplied by alpha, so each composites over
    // its background as c + background * (1 - alpha).
    vec3 above = words.rgb + night * (1.0 - words.a);
    vec3 below = shine.rgb + water * (1.0 - shine.a);
    gl_FragColor = vec4(mix(above, below, step(horizon, uv.y)), 1.0);
}
"""


def build():
    return ghee.Scene(
        common=COMMON,
        params={
            "beat": ghee.Param("beat", "float", 4.0, min=1.0, max=10.0, step=0.1),
            "sung": ghee.Param("sung", "color", [1.0, 0.68, 0.18]),
            "unsung": ghee.Param("unsung", "color", [0.35, 0.45, 1.0]),
            "horizon": ghee.Param(
                "horizon", "float", 0.58, min=0.3, max=0.9, step=0.01
            ),
        },
        assets=[ghee.Asset("sans", "fonts/DejaVuSans-Latin.ttf")],
        texts=[
            ghee.Text("line", ["sans", "NotoColorEmoji.ttf"], "sing it with me 🎤✨")
        ],
        passes=[
            ghee.Pass(
                "lyrics",
                layers=[
                    ghee.Layer("line", HALO, PLACE, pad=PAD, name="halo"),
                    ghee.Layer("line", FILL, PLACE, name="fill"),
                ],
            ),
            ghee.Pass("main", SKY, channels=[ghee.Channel("lyrics")]),
        ],
    )


ghee.run(build())
