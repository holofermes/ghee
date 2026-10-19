#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render==0.4.1.dev0"]
# ///
"""A karaoke line over a pixel-art sea at dusk, sung word by word.

The pass `words` holds the line alone. Its text layer `line` draws the words
in the pixel font with a glow, and its emoji inline from the font asset
`noto`, fonts/NotoColorEmoji-karaoke.ttf, the two emoji of the line cut from
Noto Color Emoji. The tick hook fills the sung letters through the karaoke
effect's `sung`, word by word, LETTER seconds a letter. The tick measures the
line with `ghee.measure` and centres it on the plate, a dialog box at the
height that `line_y` sets. The words that the console gives `line` reach the
tick in its params as `line.text`, and the tick measures and times them
again.

The pass `sea` paints the scene and the plate ART pixels wide, and `main`
reads it nearest, so its pixels show whole. Below the horizon, `sea` reads
`words` upside down, and the plate and the line shimmer in the water with the
moon. `main` lays `words` over `sea`.

    ghee render --demo karaoke -o karaoke.####.png --frames 100 --timestep 0.05
    ghee render --demo karaoke --display kms
"""

import math
import re

import ghee

ghee.args()  # the probe, this scene takes no arguments

LINE = "sing it with me 🎤✨"
FONTS = ["Pixel Operator 8", "noto"]
LINE_Y = 0.22  # the plate's place, from the frame's top (0) to its bottom (1)
LETTER = 0.2  # seconds to sing a letter
BREATH = 0.2  # seconds between two words
HOLD = 1.0  # seconds the last word is held
REST = 1.0  # seconds from the end of the line to its next start
ART = 160  # the sea's width in pixels
EM = 8  # the line's size, in the sea's pixels per em
PAD = (5, 3)  # the sea's pixels between the line and the plate's border
BORDER = 4  # the plate's border, in the sea's pixels
MARGIN = 2  # the sea's pixels between the plate and the frame's edge

NIGHT = "#141833"

SEA = """
float fbm(vec2 p) {
    return 0.55 * ghee_noise(p) + 0.3 * ghee_noise(p * 2.03 + 7.1) + 0.15 * ghee_noise(p * 4.07 + 3.7);
}

float bayer2(vec2 a) {
    a = floor(a);
    return fract(a.x * 0.5 + a.y * a.y * 0.75);
}

float bayer(vec2 p) {
    return bayer2(0.5 * p) * 0.25 + bayer2(p);
}

// Quantises s to n bands, dithering only near each band's edge.
float bands(float s, float n, float threshold) {
    float f = fract(s * n);
    float up = step(threshold, clamp((f - 0.5) * 3.0 + 0.5, 0.0, 1.0));
    return (floor(s * n) + up) / n;
}

vec3 sky_ramp(float s) {
    vec3 c = mix(zenith, dusk, smoothstep(0.0, 0.55, s));
    c = mix(c, rose, smoothstep(0.5, 0.82, s));
    return mix(c, ember, smoothstep(0.8, 1.0, s));
}

float far_cloud(vec2 p, float hz) {
    float s = p.y / hz;
    float deck = smoothstep(0.12, 0.3, s) * (1.0 - smoothstep(0.55, 0.72, s));
    float n = fbm(vec2((p.x + iTime * 2.0) * 0.02, p.y * 0.17));
    return step(0.62, n * (0.6 + 0.5 * deck));
}

float bank(vec2 p, float hz) {
    float x = p.x + iTime * 5.0;
    float h = max(fbm(vec2(x * 0.035, 3.0)) * 1.5 - 0.5, 0.0) * hz * 0.3;
    float puff = fbm(vec2(x * 0.1, p.y * 0.15) + 5.0) * 5.0;
    return step(hz + 2.0 - h - puff, p.y);
}

float frond(vec2 p, vec2 c, float side, float e, float len, float droop) {
    float run = len * cos(e);
    float t = (p.x - c.x) * side / run;
    float v = p.y - c.y + len * sin(e) * t - len * droop * t * t;
    float w = 0.9 * (1.0 - t) + 0.4;
    float comb = step(0.5, fract((t * run - v * 0.6) / 2.5));
    float leaf = (1.0 + 3.0 * t) * (1.0 - t * t * t) * comb;
    return step(0.0, t) * step(t, 1.0) * step(-0.8 * w, v) * step(v, w + leaf);
}

float palm(vec2 p, vec2 base, vec2 crown, float len, float seed) {
    float s = (base.y - p.y) / (base.y - crown.y);
    float x = mix(base.x, crown.x, 1.0 - (1.0 - s) * (1.0 - s));
    float ring = 0.4 * step(0.5, fract(p.y * 0.34));
    float m = step(abs(p.x - x), mix(2.4, 1.3, s) + ring) * step(0.0, s) * step(s, 1.0);
    if (length(p - crown) < len * 1.4) {
        for (int k = 0; k < 10; k++) {
            float j = mod(float(k), 5.0);
            float h = ghee_hash(vec2(seed, float(k)));
            float e = 1.35 - 0.51 * j + 0.2 * (h - 0.5) + 0.07 * sin(iTime * 1.4 + seed + 1.9 * float(k));
            float l = len * (0.5 + 0.75 * sin(j * 0.667)) * (0.9 + 0.2 * h);
            m = max(m, frond(p, crown, sign(float(k) - 4.5), e, l, 1.4 - 0.25 * j));
        }
        m = max(m, step(length(p - crown - vec2(0.0, 1.0)), 2.0));
    }
    return m;
}

// A crater at c.xy of radius c.z on the moon's face, lit from the right: 1 in
// x on its floor, and 1 in y on the wall in its rim's shade.
vec2 crater(vec2 m, vec3 c) {
    float shade = step(c.z, length(m - c.xy + vec2(0.3, -0.15) * c.z));
    return step(length(m - c.xy), c.z) * vec2(1.0, shade);
}

// The moon's face at m, in the moon's own space, its rim at 1.
vec3 moon_face(vec2 m, float th) {
    float mare = step(0.64, fbm(m * 1.6 + 2.7) + (th - 0.5) * 0.08);
    float limb = step(1.0, length(m - vec2(0.16, -0.06)) + (th - 0.5) * 0.1);
    vec2 pit = max(max(crater(m, vec3(-0.36, -0.34, 0.22)), crater(m, vec3(0.3, 0.4, 0.18))),
                   max(crater(m, vec3(0.5, -0.32, 0.13)), crater(m, vec3(-0.18, 0.56, 0.12))));
    vec3 face = mix(moon, moon_shade, max(mare, limb));
    face = mix(face, mix(moon_shade, rose, 0.15), pit.x);
    return mix(face, mix(moon_shade, rose, 0.45), pit.y);
}

// The line's block in the sea's pixels: left, top, size.
vec4 block(vec2 res) {
    vec2 size = floor(line_size * EM + 0.5);
    float room = max(res.y - size.y - 2.0 * (EDGE.y + MARGIN), 0.0);
    return vec4(floor((res.x - size.x) * 0.5), MARGIN + EDGE.y + floor(room * line_y + 0.5), size);
}

// The dialog box around block b at q, and its shadow: colour, and coverage in alpha.
vec4 dialog(vec2 q, vec4 b, float th) {
    vec2 lo = b.xy - EDGE;
    vec2 hi = b.xy + b.zw + EDGE - 1.0;
    vec2 e = min(q - lo, hi - q);
    float k = min(min(e.x, e.y), e.x + e.y - 2.0);
    vec2 s = min(q - lo - 1.0, hi + 1.0 - q);
    float shadow = step(0.0, min(min(s.x, s.y), s.x + s.y - 2.0));
    float v = (q.y - lo.y - BORDER) / max(hi.y - lo.y - 2.0 * BORDER, 1.0);
    vec3 c = mix(plate, plate * 0.55, bands(clamp(v, 0.0, 1.0), 3.0, th));
    c = mix(c, mix(plate, frame, 0.3), step(q.y - lo.y, BORDER + 0.5));
    c = mix(c, palms, step(k, BORDER - 0.5));
    c = mix(c, mix(frame, rose, 0.6), step(k, BORDER - 1.5));
    c = mix(c, frame, step(k, 1.5));
    c = mix(c, palms, step(k, 0.5));
    float on = step(0.0, k);
    return vec4(mix(palms, c, on), max(on, shadow * 0.6));
}

void main() {
    vec2 p = floor(vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y));
    vec2 res = iResolution.xy;
    float hz = floor(res.y * horizon);
    float th = bayer(p);
    float below = step(hz, p.y);
    float wobble = floor(sin(p.y * 0.9 + iTime * 3.0) * 1.5 + 0.5);
    vec2 at = mix(p, vec2(p.x + wobble, 2.0 * hz - p.y - 1.0), below);
    vec4 b = block(res);
    vec2 lo = b.xy - EDGE;
    vec2 hi = b.xy + b.zw + EDGE;
    float real = step(lo.x, p.x) * step(p.x, hi.x) * step(lo.y, p.y) * step(p.y, hi.y);
    vec4 box = dialog(mix(at, p, real), b, th);
    float R = floor(res.y * moon_size);
    vec2 mc = vec2(floor(res.x * moon_x), hz - floor(R * 0.15));
    vec3 col;
    float fade = 1.0;

    if (p.y < hz) {
        float s = p.y / hz;
        s += (ghee_noise(vec2(p.x * 0.035 + iTime * 0.1, p.y * 0.6)) - 0.5) * 0.05;
        col = sky_ramp(bands(clamp(s, 0.0, 1.0), 9.0, th));

        float star = ghee_hash(p);
        float twinkle = 0.5 + 0.5 * sin(iTime * (1.5 + 4.0 * star) + star * 60.0);
        float shine = step(0.988, star) * step(s, 0.6) * step(0.35, twinkle);
        col = mix(col, moon, shine * (0.5 + 0.5 * twinkle) * (1.0 - s));

        float n = floor(iTime / 7.0);
        float age = iTime - n * 7.0 - 1.0;
        vec2 dir = vec2(-0.92, 0.39);
        vec2 head = res * vec2(0.45 + 0.4 * ghee_hash(vec2(n, 1.0)), 0.04 + 0.1 * ghee_hash(vec2(n, 2.0))) + dir * age * 110.0;
        vec2 q = p + 0.5 - head;
        float behind = -dot(q, dir);
        float meteor = step(abs(dot(q, vec2(-dir.y, dir.x))), 0.75) * step(0.0, behind) * step(behind, 12.0);
        col = mix(col, moon, meteor * step(0.0, age) * step(age, 0.6) * (1.0 - behind / 12.0));

        float d = length(p + 0.5 - mc);
        float halo = clamp(1.0 - (d - R) / (R * 0.9), 0.0, 1.0);
        col = mix(col, mix(ember, moon, 0.4), floor(halo * 3.0 + th) / 3.0 * 0.45);
        if (d < R) col = moon_face((p + 0.5 - mc) / R, th);

        float far = far_cloud(p, hz);
        float far_lit = far * (1.0 - far_cloud(p + vec2(0.0, 1.0), hz));
        col = mix(col, mix(dusk, rose, 0.45), far);
        col = mix(col, mix(rose, ember, 0.5), far_lit);

        float near = bank(p, hz);
        float rim = near * (1.0 - bank(p - vec2(0.0, 1.0), hz));
        float lit = clamp(1.6 - d / R, 0.0, 1.0);
        col = mix(col, mix(cloud, dusk, 0.25 * s), near);
        col = mix(col, mix(mix(rose, ember, 0.6), moon, lit), rim);
    } else {
        float d = (p.y - hz) / max(res.y - hz, 1.0);
        vec3 deep = water * 0.45;
        vec3 shallow = mix(water, dusk, 0.55);
        col = mix(shallow, deep, bands(d, 4.0, th));

        float row = ghee_hash(vec2(p.y, 7.0));
        float drift = iTime * (1.0 + 3.0 * row) * sign(row - 0.5);
        float ripple = ghee_noise(vec2((p.x + drift) * 0.3 / (0.4 + 1.6 * d), p.y * 3.1));
        col = mix(col, mix(water, dusk, 0.9), step(0.74, ripple));
        float glint = step(0.8, ghee_noise(vec2((p.x - drift) * 0.5, p.y * 5.3))) * step(d, 0.12);
        col = mix(col, mix(rose, ember, 0.4), glint * (1.0 - d / 0.12));
        col = mix(col, mix(rose, ember, 0.7), step(p.y, hz));

        float sway = floor(sin(p.y * 0.5 - iTime * 2.2) * 1.5 + 0.5);
        float x = abs(p.x - mc.x + sway);
        float reach = R * (0.85 - 0.35 * d) * (0.35 + 0.65 * abs(sin(p.y * 0.9 + iTime * 1.6 + row * 6.0)));
        float gap = step(1.0, abs(x - reach * (0.3 + 0.4 * row)));
        float on = step(-0.3, sin(p.y * 2.1 - iTime * 5.0 + row * 3.0));
        float shine = step(x, reach) * gap * on;
        vec3 glare = mix(mix(moon, ember, 0.15 + 0.7 * d), mix(ember, rose, d), step(reach * 0.45, x));
        col = mix(col, glare, shine * (1.0 - 0.4 * d));

        vec2 cell = iOutputResolution.xy / res;
        float keep = step(-0.5, sin(p.y * 2.3 - iTime * 4.0 + row * 3.0)) * step(0.2, ghee_hash(vec2(floor(at.x / 3.0), p.y)));
        fade = keep * (0.8 - 0.5 * d);
        col = mix(col, box.rgb * vec3(0.85, 0.85, 1.0), box.a * fade * (1.0 - real));
        vec2 w = (at + 0.5) * cell / iOutputResolution.xy;
        vec4 words = texture2D(iChannel0, vec2(w.x, 1.0 - w.y));
        col = col * (1.0 - words.a * fade) + words.rgb * vec3(0.85, 0.85, 1.0) * fade;
    }

    float fg = palm(p, vec2(res.x * 0.03, res.y + 4.0), vec2(res.x * 0.1, res.y * 0.1), res.x * 0.1, 0.0);
    fg = max(fg, palm(p, vec2(res.x * 0.99, res.y + 4.0), vec2(res.x * 0.9, res.y * 0.08), res.x * 0.11, 2.0));
    fg = max(fg, palm(p, vec2(res.x * 0.84, res.y + 4.0), vec2(res.x * 0.99, res.y * 0.42), res.x * 0.075, 4.0));
    vec2 shore = vec2(p.x / (res.x * 0.3), (res.x - p.x) / (res.x * 0.32));
    fg = max(fg, step(res.y * (0.88 + 0.12 * shore.x * shore.x), p.y));
    fg = max(fg, step(res.y * (0.86 + 0.14 * shore.y * shore.y), p.y));

    col = mix(col, palms, fg);
    col = mix(col, box.rgb, box.a * real);
    gl_FragColor = vec4(col, 1.0);
}
"""

MAIN = """
void main() {
    vec2 px = vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y);
    vec2 art = (floor(px / CELL) + 0.5) / iChannelResolution[0].xy;
    vec3 sea = texture2D(iChannel0, vec2(art.x, 1.0 - art.y)).rgb;
    vec4 words = vec4(0.0);
    if (gl_FragCoord.x >= 0.0) words = texture2D(iChannel1, uv);
    gl_FragColor = vec4(words.rgb + sea * (1.0 - words.a), 1.0);
}
"""


def cell(width):
    """The frame's pixels to one pixel of the sea, on a frame `width` wide."""
    return max(1, round(width / ART))


def song(text):
    """Each word of `text`: where its characters start and end, and when it
    is sung, in seconds from the line's start."""
    words, t = [], 0.0
    for m in re.finditer(r"\S*\w\S*", text):
        words.append([m.start(), m.end(), t, t + LETTER * len(m[0])])
        t = words[-1][3] + BREATH
    if words:
        words[-1][3] += HOLD
    return words


def sung(words, t):
    """How many characters of the line are sung `t` seconds into its bar."""
    count = 0.0
    for first, last, start, end in words:
        if t >= start:
            count = first + min((t - start) / (end - start), 1.0) * (last - first)
    return count


_line = {
    "line.text": LINE,
    "line_y": LINE_Y,
    "song": song(LINE),
    "start": 0.0,
}


def whole(n):
    """`n` rounded to a whole number, halves up, as the sea's `block` rounds."""
    return math.floor(n + 0.5)


def lay_out():
    """Centre the line on the plate, as `block` in SEA does."""
    width, height = ghee.render_size()
    px = cell(width)
    art_width, art_height = math.ceil(width / px), math.ceil(height / px)
    size = ghee.measure(_line["line.text"], font_size=EM * px, font_family=FONTS)
    wide, tall = (whole(n / px) for n in size)
    room = max(art_height - tall - 2 * (PAD[1] + BORDER + MARGIN), 0)
    ghee.layers.line.x = (art_width - wide) // 2 * px
    ghee.layers.line.y = (MARGIN + PAD[1] + BORDER + whole(room * _line["line_y"])) * px


@ghee.tick
def sing(tick):
    """Lay the line out again when its words or its height change, and sing
    new words from their start, then fill the sung letters."""
    changed = [k for k in ("line.text", "line_y") if k in tick.params]
    _line.update((k, tick.params[k]) for k in changed)
    if "line.text" in changed:
        _line["song"] = song(_line["line.text"])
        _line["start"] = tick.time
    if changed or tick.frame == 0:
        lay_out()
    words = _line["song"]
    bar = (words[-1][3] if words else 0.0) + REST
    ghee.layers.line.karaoke.sung = sung(words, (tick.time - _line["start"]) % bar)


def build():
    width, height = ghee.render_size()
    px = cell(width)
    art = (math.ceil(width / px), math.ceil(height / px))
    return ghee.Scene(
        params={
            "horizon": ghee.Param(
                "horizon", "float", 0.6, min=0.4, max=0.85, step=0.01
            ),
            "line_y": ghee.Param(
                "line_y", "float", LINE_Y, min=0.0, max=1.0, step=0.01
            ),
        },
        assets=[ghee.Asset("noto", "fonts/NotoColorEmoji-karaoke.ttf")],
        passes=[
            ghee.Pass(
                "words",
                layers=[
                    ghee.Layer(
                        "line",
                        LINE,
                        font_size=EM * px,
                        font_family=FONTS,
                        render="hinted",
                        color="#c8b6dc",
                        text_shadow={
                            "x": 0.125,
                            "y": 0.125,
                            "blur": 0,
                            "color": NIGHT,
                            "strength": 1,
                        },
                        glow={"color": "#ff6f6a", "blur": 0.45, "strength": 0.5},
                        karaoke={"color": "#ffcf8a"},
                    )
                ],
            ),
            ghee.Pass(
                "sea",
                f"const float EM = {EM:.1f};\n"
                f"const vec2 EDGE = vec2({PAD[0] + BORDER:.1f}, {PAD[1] + BORDER:.1f});\n"
                f"const float BORDER = {BORDER:.1f};\n"
                f"const float MARGIN = {MARGIN:.1f};\n{SEA}",
                size=art,
                sampling="nearest",
                channels=[ghee.Channel("words")],
                params=[
                    ghee.Param("zenith", "color", NIGHT),
                    ghee.Param("dusk", "color", "#3d5470"),
                    ghee.Param("rose", "color", "#b05a78"),
                    ghee.Param("ember", "color", "#f29a6c"),
                    ghee.Param("moon", "color", "#fbefd5"),
                    ghee.Param("moon_shade", "color", "#e2cba6"),
                    ghee.Param("cloud", "color", "#2a2244"),
                    ghee.Param("water", "color", "#1d2a52"),
                    ghee.Param("palms", "color", "#0b0a16"),
                    ghee.Param("plate", "color", "#252c5e"),
                    ghee.Param("frame", "color", "#f6e7cf"),
                    ghee.Param(
                        "moon_size", "float", 0.17, min=0.05, max=0.4, step=0.01
                    ),
                    ghee.Param("moon_x", "float", 0.5, min=0.0, max=1.0, step=0.01),
                ],
            ),
            ghee.Pass(
                "main",
                f"const float CELL = {px:.1f};\n{MAIN}",
                channels=[ghee.Channel("sea"), ghee.Channel("words")],
            ),
        ],
    )


ghee.run(build())
