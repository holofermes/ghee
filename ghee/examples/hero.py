#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render==0.4.1.dev0"]
# ///
"""A word slides over water, and particles stream under the surface.

A feedback buffer holds the particle positions. Each frame takes one advection
step along a curl-noise flow. The stream function of that flow ramps to zero at
the wave. The surface is then a streamline, and the flow slides along it.

The two sizes below are compile-time. A loop that a uniform breaks out of early
still costs its maximum, so a live particle-count slider buys nothing.

    ghee render --demo hero -o hero.####.png --frames 300
    ghee render --demo hero -- --particle-grid 12x8 --no-dither
    ghee render --demo hero --display kms -- --particle-buffer 160x100
"""

import ghee

ARGS = [
    ghee.Arg(
        "particle-buffer",
        "size",
        help="the particle field's own buffer size, magnified onto the frame, "
        "an eighth of the frame by default",
    ),
    # A count, not a share of the frame: the shader unrolls the grid, so the
    # cost is the whole grid every frame however small the panel is. A share
    # of a 1280x800 frame is 16000 particles and the render stops moving.
    ghee.Arg(
        "particle-grid",
        "size",
        default=(20, 12),
        help="the particle grid, columns x rows",
    ),
    ghee.Arg("dither", "bool", default=True, help="the dither filter pass"),
]

args = ghee.args(ARGS)

# Slow imports come after the parse: the probe must answer inside its window.
from lib import look  # noqa: E402

# The dither filter defaults to 1-bit black and white, these give it partial
# colour on a coarse grid. It costs one full-res pass, which --no-dither drops.
DITHER_PARAMS = {"amount": 1.0, "levels": 2.0, "color": 0.65, "scale": 4.0}

# The buffer of the pass `canvas`, which the word fits in.
CANVAS = (480, 192)
GAP = 0.6  # trailing gap between repeats, as a fraction of the word width
STROKE = 1 / 22  # the stroke's width in em: a display face's weight

# A motion delta packs into a byte per axis over this span.
DSPAN = 0.25
# The longest step still read as motion. `attr` drops a particle that moved
# further and `glow` takes it as the reach of a stroke, so it has to be one
# number: raise it in one place only and the stroke loop stops rejecting.
JUMP = 0.1

SCENE_COMMON = (
    look.WAVE
    + """
// Everything downstream works in [0, 1] uv, so the flow and the particles look
// the same at every render size.
float waterline(float x) {
    return 0.55 - wave(x, iTime * wave_speed * 6.28318);
}
float depth(vec2 p) {
    return p.y - waterline(p.x);
}

float hash(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}

// A pass buffer is RGB8, so a position is three bytes: twelve bits per axis,
// the high byte of each in R and G, both low nibbles sharing B. The domain is
// [-0.25, 1.25] so a particle runs off-screen before it wraps.
vec3 packPos(vec2 p) {
    vec2 v = floor(clamp((p + 0.25) / 1.5, 0.0, 1.0) * 4095.0);
    vec2 hi = floor(v / 16.0);
    vec2 lo = v - hi * 16.0;
    return vec3(hi / 255.0, (lo.x * 16.0 + lo.y) / 255.0);
}
vec2 unpackPos(vec3 e) {
    float lo = e.z * (255.0 / 16.0);
    float lx = floor(lo);
    return (e.xy * 4080.0 + vec2(lx, (lo - lx) * 16.0)) * (1.5 / 4095.0) - 0.25;
}
// The high bytes read as a position on their own: what unpackPos returns, short
// by up to one low nibble, for two multiplies against unpackPos's dozen.
vec2 coarsePos(vec3 e) {
    return e.xy * (4080.0 * 1.5 / 4095.0) - 0.25;
}
"""
)

# `state` is a later pass, so this reads it a frame back. A channel reaches one
# frame and a stroke needs two.
SHADER_PREV = """
void main() {
    gl_FragColor = vec4(texture2D(iChannel0, uv).rgb, 1.0);
}
"""

SHADER_STATE = """
// Stream function: drift plus curl noise, faded out over surface_band so the
// wave is a solid border. The noise amplitude divides by its own frequency, so
// swirl_scale moves the size of the swirls and not their strength.
float psi(vec2 p) {
    float f = 24.0 * pow(0.5, swirl_scale * 4.0);
    vec2 q = p * vec2(ASPECT, 1.0) * f;
    float tf = iTime * 0.1;
    float n = sin(q.x + tf) * cos(q.y * 1.1 - tf * 0.7)
            + 0.5 * sin(q.x * 2.3 - tf * 1.3) * cos(q.y * 1.9 + tf * 0.9);
    float free = drift.x * p.y - drift.y * p.x + n * swirl_strength * 0.5 / f;
    return free * smoothstep(0.0, surface_band, depth(p));
}

vec2 vel(vec2 p) {
    float e = 0.004;
    float p0 = psi(p);
    return vec2(psi(p + vec2(0.0, e)) - p0, p0 - psi(p + vec2(e, 0.0))) / e;
}

void main() {
    vec2 p = unpackPos(texture2D(iChannel0, uv).rgb);
    bool born = iTime < 0.1 || texture2D(iChannel1, uv).r <= 1.5 / 255.0;
    if (!born) {
        // particle_speed_spread is ratio-symmetric around particle_speed:
        // 0 = every particle alike, 1 = 0.5x to 2x
        float sv = pow(2.0, (hash(uv * 9.3) - 0.5) * 2.0 * particle_speed_spread);
        p += vel(p) * (iTimeDelta * particle_speed * sv);
        born = p.x < -0.25 || p.x > 1.25 || p.y > 1.25
            // the surface moves and can sweep over a particle riding it; past
            // the surface the field is zero, so it would strand there
            || depth(p) < -0.02;
    }
    if (born) {
        // Every spawn lands off-frame on the upstream edge and under the
        // surface, so a stroke can only enter the frame already moving.
        float r1 = hash(uv * 3.1), r2 = hash(uv * 7.7), r3 = hash(uv * 5.3);
        float ax = abs(drift.x), ay = abs(drift.y);
        if (r3 * (ax + ay + 1e-4) < ax) {
            p.x = drift.x < 0.0 ? 1.02 + r1 * 0.15 : -0.02 - r1 * 0.15;
            p.y = mix(waterline(p.x) + 0.05, 1.2, r2);
        } else {
            p.y = 1.02 + r1 * 0.15;   // the cross-flow edge is the floor
            p.x = r2 * 1.4 - 0.2;
        }
    }
    gl_FragColor = vec4(packPos(p), 1.0);
}
"""

# One countdown per particle, ordered after `state` so both read the same value
# this frame.
SHADER_LIFE = """
void main() {
    float life = texture2D(iChannel0, uv).r;
    if (iTime < 0.1) life = hash(uv * 7.3);
    else if (life <= 1.5 / 255.0) life = mix(0.35, 1.0, hash(uv + iTime));
    else life -= (1.0 / 255.0) * (0.35 / max(particle_life, 0.02)) * iTimeDelta * 60.0;
    gl_FragColor = vec4(life, 0.0, 0.0, 1.0);
}
"""

# Per-particle precompute: the motion delta and the brightness, one texel each
# instead of recomputed under every pixel of the stroke loop.
SHADER_ATTR = (
    f"""
const float DSPAN = {DSPAN};
const float JUMP = {JUMP};
"""
    + """
void main() {
    vec2 p1 = unpackPos(texture2D(iChannel0, uv).rgb);
    vec2 d = p1 - unpackPos(texture2D(iChannel1, uv).rgb);
    // surface_fade tapers a particle out as it nears the surface; a step longer
    // than JUMP is a spawn. Both land in the brightness, so a stroke that
    // should not show multiplies out to zero and the loop tests neither.
    float b = mix(particle_dim, particle_bright, hash(uv * 9.3))
            * smoothstep(-0.02, max(surface_fade, 0.02) * 0.4 - 0.02, depth(p1))
            * step(dot(d, d), JUMP * JUMP);
    gl_FragColor = vec4(clamp(d / DSPAN + 0.5, 0.0, 1.0), b, 1.0);
}
"""
)


def _grid_consts(col, row):
    """The grid bounds, as shader constants.

    The stroke loop unrolls only over constant bounds.
    """
    return f"""
const int COL = {col};
const int ROW = {row};
const vec2 GRID = vec2({col}.0, {row}.0);
const float DSPAN = {DSPAN};
const float JUMP = {JUMP};
"""


# The only loop in the scene, and the reason this pass has its own size: it
# costs particles x pixels, and it is the pixels that are cheap to give up.
SHADER_GLOW = """
float sdSeg(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / (dot(ba, ba) + 1e-9), 0.0, 1.0);
    return length(pa - ba * h);
}

// One particle, rejected on the coarse position the caller already holds.
// Everything the stroke needs past that (the full position, the delta, the
// brightness, the second fetch) sits behind the test, and a VC4 does skip a
// branch that its whole block of pixels fails, which is what makes 240 of these
// per pixel affordable. Written as a macro rather than a function because a
// function's early return gets flattened back into a select, work and all.
#define STROKE(cell, texel) \\
    away = uva - coarsePos(texel) * asp; \\
    if (dot(away, away) < reach2) { \\
        vec2 p1 = unpackPos(texel); \\
        vec3 a = texture2D(iChannel1, cell).rgb; \\
        vec2 p0 = p1 - (a.rg - 0.5) * DSPAN; \\
        float d = sdSeg(uva, p0 * asp, p1 * asp); \\
        ink = max(ink, a.b * smoothstep(w, w * 0.35, d)); \\
    }

void main() {
    vec2 asp = vec2(ASPECT, 1.0);
    vec2 uva = uv * asp;

    // Last frame's ink, decayed. The sub-LSB dither is what keeps an 8-bit
    // buffer fading smoothly instead of banding into contour rings.
    float dj = hash(uv + fract(iTime) * 113.1);
    float steps = iTimeDelta * 60.0;
    float ink = max(texture2D(iChannel2, uv).r * pow(trail_length, steps)
                    - trail_cutoff * 0.01 * steps + (dj - 0.5) / 255.0, 0.0);

    // The width is in uv, not in texels, so this buffer's size changes how
    // sharp a stroke is and never how big it is.
    float w = 0.000625 + particle_size * particle_size * 0.035;
    // How far a particle can sit from this pixel and still mark it: the stroke
    // width, the longest segment it can be strung along, and a tenth of that
    // again for the truncated position and the byte-packed delta. Past it the
    // stroke is exactly zero, so rejecting on it costs nothing but time.
    float reach = w + (JUMP * 1.1) * max(ASPECT, 1.0);
    float reach2 = reach * reach;

    vec2 away;
    for (int ix = 0; ix < COL; ix++) {
        // Four rows at a time, so four fetches issue together instead of one
        // stalling per step. A row index past the last clamps back onto it, so
        // an odd ROW strokes its final particle twice and max() does not care.
        for (int iy = 0; iy < ROW; iy += 4) {
            vec2 c0 = (vec2(float(ix), float(iy)) + 0.5) / GRID;
            vec2 c1 = (vec2(float(ix), float(iy + 1)) + 0.5) / GRID;
            vec2 c2 = (vec2(float(ix), float(iy + 2)) + 0.5) / GRID;
            vec2 c3 = (vec2(float(ix), float(iy + 3)) + 0.5) / GRID;
            vec3 s0 = texture2D(iChannel0, c0).rgb;
            vec3 s1 = texture2D(iChannel0, c1).rgb;
            vec3 s2 = texture2D(iChannel0, c2).rgb;
            vec3 s3 = texture2D(iChannel0, c3).rgb;
            STROKE(c0, s0);
            STROKE(c1, s1);
            STROKE(c2, s2);
            STROKE(c3, s3);
        }
    }
    gl_FragColor = vec4(ink, 0.0, 0.0, 1.0);
}
"""

# The only pass at the output resolution: the wave edge and the letters are
# hard edges, and the particles arrive magnified.
SHADER_MAIN = (
    f"""
const float TEXT_ASPECT = {CANVAS[0] * (1 + GAP) / CANVAS[1]:.4f};
const float GAP = {GAP};
"""
    + """
void main() {

    // The texture reads come before the surface: with `wet` computed ahead of
    // them, VC4 reads it back wrong in 4x4 blocks and the glyph body goes dry
    // in patches under the wave.
    float v = (uv.y - text_y + text_height * 0.5) / text_height;
    float u = fract(uv.x * ASPECT / (text_height * TEXT_ASPECT) - iTime * text_speed)
            * (1.0 + GAP);
    float glow = texture2D(iChannel0, uv).r;
    float m = texture2D(iChannel1, vec2(u, v)).a * step(0.0, v) * step(v, 1.0) * step(u, 1.0);
    m = smoothstep(0.35, 0.55, m);

    float wet = step(waterline(uv.x), uv.y);
    float ink = glow * wet;
    vec3 col = mix(mix(background, water, wet), particle_color, ink);
    gl_FragColor = vec4(mix(col, mix(ink_background, ink_water, wet), m), 1.0);
}
"""
)


def knob(name, val, lo, hi, step):
    return ghee.Param(name, "float", val, min=lo, max=hi, step=step)


def build(a):
    grid = a.particle_grid
    col, row = grid
    # The frame's shape, which the passes with a size of their own cannot
    # read in iResolution.
    w, h = ghee.render_size()
    return ghee.Scene(
        common=f"const float ASPECT = {w / h!r};\n" + SCENE_COMMON,
        filters=["dither"] if a.dither else [],
        passes=[
            ghee.Pass(
                name="prev",
                fragment=SHADER_PREV,
                size=grid,
                channels=[ghee.Channel(buffer="state")],
            ),
            ghee.Pass(
                name="state",
                fragment=SHADER_STATE,
                size=grid,
                channels=[ghee.Channel(buffer="state"), ghee.Channel(buffer="life")],
                params={
                    "drift": ghee.Param(
                        "drift",
                        "vec2",
                        [-0.556, -0.088],
                        min=-1.0,
                        max=1.0,
                        step=0.005,
                    ),
                    "particle_speed": knob("particle_speed", 0.465, 0.0, 1.0, 0.005),
                    "particle_speed_spread": knob(
                        "particle_speed_spread", 1.0, 0.0, 1.0, 0.01
                    ),
                    "swirl_strength": knob("swirl_strength", 0.91, 0.0, 1.0, 0.01),
                    "swirl_scale": knob("swirl_scale", 0.88, 0.0, 1.0, 0.01),
                    "surface_band": knob("surface_band", 0.45, 0.02, 0.6, 0.01),
                },
            ),
            ghee.Pass(
                name="life",
                fragment=SHADER_LIFE,
                size=grid,
                channels=[ghee.Channel(buffer="life")],
                params={"particle_life": knob("particle_life", 0.83, 0.0, 1.0, 0.01)},
            ),
            ghee.Pass(
                name="attr",
                fragment=SHADER_ATTR,
                size=grid,
                channels=[ghee.Channel(buffer="state"), ghee.Channel(buffer="prev")],
                params={
                    "surface_fade": knob("surface_fade", 0.61, 0.0, 1.0, 0.01),
                    "particle_dim": knob("particle_dim", 0.09, 0.0, 1.0, 0.01),
                    "particle_bright": knob("particle_bright", 0.9, 0.0, 1.0, 0.01),
                },
            ),
            ghee.Pass(
                name="glow",
                fragment=_grid_consts(col, row) + SHADER_GLOW,
                size=a.particle_buffer,
                scale=None if a.particle_buffer else 1 / 8,
                channels=[
                    ghee.Channel(buffer="state"),
                    ghee.Channel(buffer="attr"),
                    ghee.Channel(buffer="glow"),
                ],
                params={
                    "particle_size": knob("particle_size", 0.53, 0.0, 1.0, 0.01),
                    "trail_length": knob("trail_length", 0.623, 0.0, 0.999, 0.001),
                    "trail_cutoff": knob("trail_cutoff", 0.22, 0.0, 1.0, 0.01),
                },
            ),
            ghee.Pass(
                name="canvas",
                size=CANVAS,
                layers=[
                    ghee.Layer(
                        "word", "GHEE", text_stroke={"width": STROKE, "color": "white"}
                    )
                ],
            ),
            ghee.Pass(
                name="main",
                fragment=SHADER_MAIN,
                channels=[ghee.Channel(buffer="glow"), ghee.Channel(buffer="canvas")],
                params={
                    "text_speed": knob("text_speed", 0.101, -0.5, 0.5, 0.005),
                    "text_y": knob("text_y", 0.47, 0.0, 1.0, 0.01),
                    "text_height": knob("text_height", 0.57, 0.05, 1.5, 0.01),
                    "background": ghee.Param(
                        "background", "color", [0.855, 0.859, 0.502]
                    ),
                    "water": ghee.Param("water", "color", [0.133, 1.0, 0.980]),
                    "ink_background": ghee.Param(
                        "ink_background", "color", [0.0, 0.639, 0.675]
                    ),
                    "ink_water": ghee.Param("ink_water", "color", [0.753, 0.757, 0.0]),
                    "particle_color": ghee.Param(
                        "particle_color", "color", [0.0, 0.027, 0.890]
                    ),
                },
            ),
        ],
        params={"wave_speed": knob("wave_speed", -0.315, -1.0, 1.0, 0.005)},
    )


_state = {"first": True}


@ghee.tick
def start(tick):
    """ghee appends a filter after the scene's own passes.

    The filter params exist only after ghee compiles the filter, so the first
    tick sets them.
    """
    if not _state["first"]:
        return
    _state["first"] = False
    if args.dither:
        for name, value in DITHER_PARAMS.items():
            ghee.set_param(name, value, pass_="dither")


ghee.run(build(args))
