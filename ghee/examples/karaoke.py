#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render==0.4.1.dev0"]
# ///
"""A karaoke line over a pixel-art sea at dusk, sung word by word.

The pass `words` holds the line alone. Its text layer `line` draws the words
in the pixel font with a glow, and the tick hook fills the sung letters
through the karaoke effect's `sung`, word by word on the times of SONG. The
pass `cue` holds the emoji, which the font asset `noto` draws from
fonts/NotoColorEmoji-karaoke.ttf, the two emoji of the line cut from Noto
Color Emoji. The first tick measures the line with `ghee.measure` and puts the
emoji after it.

The pass `sea` paints the scene ART pixels wide, and `main` reads it nearest,
so its pixels show whole. Below the horizon, `sea` reads `words` and `cue`
upside down, and the line shimmers in the water with the moon. `main` lays
`words` over `sea`.

    ghee render --demo karaoke -o karaoke.####.png --frames 100 --timestep 0.05
    ghee render --demo karaoke --display kms
"""

import math

import ghee

ghee.args()  # the probe, this scene takes no arguments

LINE = "sing it with me"
CUE = "🎤✨"
# Each word of LINE: its first character and when it is sung, in seconds.
SONG = ((0, 0.0, 0.8), (5, 1.0, 1.4), (8, 1.5, 2.3), (13, 2.5, 4.0))
BAR = 5.0  # seconds from one start of the line to the next
ART = 160  # the sea's width in pixels
CUE_SIZE = 13  # the emoji's size, in the sea's pixels per em
GAP = 4  # the sea's pixels between the line and its emoji
TOP = 0.28  # the line's top, as a share of the frame's height

NIGHT = "#141833"

SEA = """
float hash(vec2 p) {
    vec3 q = fract(p.xyx * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

float fbm(vec2 p) {
    return 0.55 * noise(p) + 0.3 * noise(p * 2.03 + 7.1) + 0.15 * noise(p * 4.07 + 3.7);
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
            float h = hash(vec2(seed, float(k)));
            float e = 1.35 - 0.51 * j + 0.2 * (h - 0.5) + 0.07 * sin(iTime * 1.4 + seed + 1.9 * float(k));
            float l = len * (0.5 + 0.75 * sin(j * 0.667)) * (0.9 + 0.2 * h);
            m = max(m, frond(p, crown, sign(float(k) - 4.5), e, l, 1.4 - 0.25 * j));
        }
        m = max(m, step(length(p - crown - vec2(0.0, 1.0)), 2.0));
    }
    return m;
}

void main() {
    vec2 p = floor(gl_FragCoord.xy);
    vec2 res = iResolution.xy;
    float hz = floor(res.y * horizon);
    float th = bayer(p);
    float R = floor(res.y * moon_size);
    vec2 mc = vec2(floor(res.x * 0.5), hz - floor(R * 0.15));
    vec3 col;
    vec2 at = p;
    float fade = 1.0;

    if (p.y < hz) {
        float s = p.y / hz;
        s += (noise(vec2(p.x * 0.035 + iTime * 0.1, p.y * 0.6)) - 0.5) * 0.05;
        col = sky_ramp(bands(clamp(s, 0.0, 1.0), 9.0, th));

        float star = hash(p);
        float twinkle = 0.5 + 0.5 * sin(iTime * (1.5 + 4.0 * star) + star * 60.0);
        float shine = step(0.988, star) * step(s, 0.6) * step(0.35, twinkle);
        col = mix(col, moon, shine * (0.5 + 0.5 * twinkle) * (1.0 - s));

        float n = floor(iTime / 7.0);
        float age = iTime - n * 7.0 - 1.0;
        vec2 dir = vec2(-0.92, 0.39);
        vec2 head = res * vec2(0.45 + 0.4 * hash(vec2(n, 1.0)), 0.04 + 0.1 * hash(vec2(n, 2.0))) + dir * age * 110.0;
        vec2 q = p + 0.5 - head;
        float behind = -dot(q, dir);
        float meteor = step(abs(dot(q, vec2(-dir.y, dir.x))), 0.75) * step(0.0, behind) * step(behind, 12.0);
        col = mix(col, moon, meteor * step(0.0, age) * step(age, 0.6) * (1.0 - behind / 12.0));

        float d = length(p + 0.5 - mc);
        float halo = clamp(1.0 - (d - R) / (R * 0.9), 0.0, 1.0);
        col = mix(col, mix(ember, moon, 0.4), floor(halo * 3.0 + th) / 3.0 * 0.45);
        float disc = step(d, R);
        float maria = step(0.64, fbm(p * 0.11 + 3.0));
        vec3 face = mix(moon, moon_shade, max(maria, step(R - 1.5, d) * step(p.x, mc.x - R * 0.2)));
        col = mix(col, face, disc);

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

        float row = hash(vec2(p.y, 7.0));
        float drift = iTime * (1.0 + 3.0 * row) * sign(row - 0.5);
        float ripple = noise(vec2((p.x + drift) * 0.3 / (0.4 + 1.6 * d), p.y * 3.1));
        col = mix(col, mix(water, dusk, 0.9), step(0.74, ripple));
        float glint = step(0.8, noise(vec2((p.x - drift) * 0.5, p.y * 5.3))) * step(d, 0.12);
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

        float wobble = floor(sin(p.y * 0.9 + iTime * 3.0) * 1.5 + 0.5);
        at = vec2(p.x + wobble, 2.0 * hz - p.y - 1.0);
        vec2 cell = iOutputResolution.xy / res;
        float keep = step(-0.5, sin(p.y * 2.3 - iTime * 4.0 + row * 3.0)) * step(0.2, hash(vec2(floor(at.x / 3.0), p.y)));
        fade = keep * (0.8 - 0.5 * d);
        vec4 words = texture2D(iChannel0, (at + 0.5) * cell / iOutputResolution.xy);
        col = col * (1.0 - words.a * fade) + words.rgb * vec3(0.85, 0.85, 1.0) * fade;
    }

    float fg = palm(p, vec2(res.x * 0.03, res.y + 4.0), vec2(res.x * 0.1, res.y * 0.1), res.x * 0.1, 0.0);
    fg = max(fg, palm(p, vec2(res.x * 0.99, res.y + 4.0), vec2(res.x * 0.9, res.y * 0.08), res.x * 0.11, 2.0));
    fg = max(fg, palm(p, vec2(res.x * 0.84, res.y + 4.0), vec2(res.x * 0.99, res.y * 0.42), res.x * 0.075, 4.0));
    vec2 shore = vec2(p.x / (res.x * 0.3), (res.x - p.x) / (res.x * 0.32));
    fg = max(fg, step(res.y * (0.88 + 0.12 * shore.x * shore.x), p.y));
    fg = max(fg, step(res.y * (0.86 + 0.14 * shore.y * shore.y), p.y));
    float below = step(hz, p.y);

    vec4 cue = vec4(0.0);
    if (gl_FragCoord.x >= 0.0) cue = texture2D(iChannel1, (at + 0.5) / res);
    vec3 sprite = floor(cue.rgb / max(cue.a, 0.01) * 4.0 + 0.5) / 4.0;
    col = mix(col, palms, fg * (1.0 - below));
    col = mix(col, sprite, step(0.5, cue.a) * fade);
    col = mix(col, palms, fg * below);
    gl_FragColor = vec4(col, 1.0);
}
"""

MAIN = """
void main() {
    vec2 px = floor(uv * iResolution.xy) + 0.5;
    vec2 art = (floor(px / CELL) + 0.5) / iChannelResolution[0].xy;
    vec3 sea = texture2D(iChannel0, art).rgb;
    vec4 words = vec4(0.0);
    if (gl_FragCoord.x >= 0.0) words = texture2D(iChannel1, px / iResolution.xy);
    gl_FragColor = vec4(words.rgb + sea * (1.0 - words.a), 1.0);
}
"""

_laid = {"out": False}


def cell(width):
    """The frame's pixels to one pixel of the sea, on a frame `width` wide."""
    return max(1, round(width / ART))


def sung(t):
    """How many characters of LINE are sung `t` seconds into the bar."""
    count = 0.0
    for k, (first, start, end) in enumerate(SONG):
        last = SONG[k + 1][0] - 1 if k + 1 < len(SONG) else len(LINE)
        if t >= start:
            count = first + min((t - start) / (end - start), 1.0) * (last - first)
    return count


@ghee.tick
def sing(tick):
    """Lay the line and its emoji out once, then fill the sung letters."""
    if not _laid["out"]:
        width, height = ghee.render_size()
        px = cell(width)
        art_width = math.ceil(width / px)
        words, rows = ghee.measure(LINE, font_size=8 * px, font_family="Pixel Operator 8")
        cue, cue_rows = ghee.measure(CUE, font_size=CUE_SIZE, font_family="noto")
        left = (art_width - (words / px + GAP + cue)) // 2
        top = round(math.ceil(height / px) * TOP)
        ghee.layers.line.x = left * px
        ghee.layers.line.y = top * px
        ghee.layers.emoji.x = left + words / px + GAP
        ghee.layers.emoji.y = top + round((rows / px - cue_rows) / 2) - 1
        _laid["out"] = True
    ghee.layers.line.karaoke.sung = sung(tick.time % BAR)


def build():
    width, height = ghee.render_size()
    px = cell(width)
    art = (math.ceil(width / px), math.ceil(height / px))
    return ghee.Scene(
        params={
            "horizon": ghee.Param(
                "horizon", "float", 0.6, min=0.4, max=0.85, step=0.01
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
                        font_size=8 * px,
                        font_family="Pixel Operator 8",
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
                "cue",
                size=art,
                layers=[
                    ghee.Layer(
                        "emoji",
                        CUE,
                        font_size=CUE_SIZE,
                        font_family="noto",
                        wave={"height": 0.12, "speed": 0.8},
                    )
                ],
            ),
            ghee.Pass(
                "sea",
                SEA,
                size=art,
                sampling="nearest",
                channels=[ghee.Channel("words"), ghee.Channel("cue")],
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
                    ghee.Param(
                        "moon_size", "float", 0.17, min=0.05, max=0.4, step=0.01
                    ),
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
