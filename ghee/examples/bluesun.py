#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render", "numpy", "pillow"]
# ///
"""Blue Sun: a wall of light pipes, and the wall that the pipes light.

`coneForm` is the closed form of the irradiance a Lambertian cone delivers to a
surface near it. It costs two atan for each pixel. The integral gives a pool
that is sharp under a tube and soft far away. The scene needs no blur radius and
no falloff curve, and `standoff` moves the pool.

Lengths are in wall units, where the frame is 1.0 tall.

    ghee render --demo bluesun -o out.####.png --frames 60
    ghee render --demo bluesun --display kms -- --emit 480x300
"""

import os
import sys

import ghee

ARGS = [
    ghee.Arg(
        "emit",
        "size",
        default="half",
        help="draw the tubes at this size and blit, instead of the full frame",
    ),
    ghee.Arg("wash", "size", default="quarter", help="the wall buffer"),
    ghee.Arg(
        "near-seg",
        "int",
        default=3,
        help="segments either side of a wall point that reach it",
    ),
    ghee.Arg(
        "near-ray",
        "int",
        default=1,
        help="rays either side, ditto; cost is (2*seg+1)*(2*ray+1) per pixel",
    ),
    ghee.Arg(
        "max-leds",
        "int",
        default=8,
        help="LEDs per segment; each one costs two form factors",
    ),
]

args = ghee.args(ARGS)

# Slow imports come after the parse: the probe must answer inside its window.
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

SEG = (72, 128)  # segment table: columns are segments, rows are rays, twice over

WALL = (256, 256)  # the wall texture, one mono float per texel


def _wash_size(emit):
    """The wall buffer. It follows the aspect of the draw size when there is one.

    A square buffer magnified across a 16:10 panel combs the falloff diagonally.
    The wall then reads as low resolution, however sharp the tubes are.
    """
    if emit is None:
        return (384, 384)
    w, h = emit
    short = 288
    return (
        (max(1, round(short * w / h)), short)
        if w >= h
        else (short, max(1, round(short * h / w)))
    )


# an RGBA8 buffer clips at 1.0, so the wall stores the form factor E/(pi L),
# which is bounded by 1 for any Lambertian emitter, and main scales it back
PI = 3.14159265

SCENE_COMMON = (
    f"""
uniform float aspect;
#define PI {PI}
#define GAIN_MAX 4.0
#define SEG_W {SEG[0]}.0
#define SEG_H {SEG[1]}.0
#define SEG_HALF {SEG[1] // 2}.0
"""
    + """
const float TAU = 6.2831853;

"""
    + """
float hash1(float n) {
    return fract(sin(n * 127.1) * 43758.5453);
}

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0;
    return v * mix(vec3(1.0), clamp(abs(k) - 1.0, 0.0, 1.0), s);
}
"""
    + """
// A chunk's own brightness gradient, head end to tail end. ht_jit pulls each
// end toward its own hash independently: a jitter that only swapped the pair
// would do nothing at all when head == tail.
void segGrad(float k, float row, out float hd, out float tl) {
    hd = mix(head, hash1(k * 5.1 + row * 2.9), ht_jit);
    tl = mix(tail, hash1(k * 3.7 + row * 8.1), ht_jit);
}

// Where a ray ends. At square=0 the tips lie on a circle, at 1 on a square, so
// the joints of a given index land on a scaled copy of that figure.
float rayReach(float ang) {
    float t = ang - corner * TAU;
    float sq = 1.0 / max(max(abs(cos(t)), abs(sin(t))), 1e-3);
    return reach * mix(1.0, sq, square);
}

// Segment k of a ray reaching `len`: where it starts and ends along the ray,
// and the tube's radius at each end. The taper runs the whole ray, so a
// segment is a frustum of it.
void segSpan(float k, float len, out float g0, out float g1,
             out float r0, out float r1) {
    float run = len - hub;
    float w = run / float(segs);
    float gap = seam * w;
    // half the gap at each end: a chunk sits centred in its cell, so its pool
    // profile is symmetric instead of flush at the head and cut at the tail
    g0 = hub + k * w + 0.5 * gap;
    g1 = g0 + w - gap;
    r0 = mix(tube_r0, tube_r1, (g0 - hub) / max(run, 1e-6));
    r1 = mix(tube_r0, tube_r1, (g1 - hub) / max(run, 1e-6));
}
"""
)

# only the passes that bind the table can read it, so this is not in COMMON
READ_SEG = """
// a ray leaves the hub at `standoff` and climbs with `tilt` as it runs out,
// so its height is a function of how far along it you are
float rayZ(float g) {
    return standoff + (g - hub) * tan(clamp(tilt, -1.2, 1.2));
}

vec4 segA(float k, float ray) {
    return texture2D(iChannel0, vec2((k + 0.5) / SEG_W, (ray + 0.5) / SEG_H));
}
"""

SCENE_COMMON += (
    """
float coneForm(vec3 P, vec3 A, vec3 B, float r0, float r1, float l0, float l1) {
    vec3 seg = B - A;
    float L = max(length(seg), 1e-6);
    vec3 d = seg / L;
    vec3 m = P - 0.5 * (A + B);
    float mdotd = dot(m, d);
    vec3 mperp = m - mdotd * d;
    float rho2 = dot(mperp, mperp);
    float rho = max(sqrt(rho2), 1e-6);
    float a = -0.5 * L - mdotd, b = 0.5 * L - mdotd;
    float base = mperp.z, ndotd = d.z;      // the wall faces +z
    if (abs(ndotd) < 1e-7) {
        if (-base <= 0.0) return 0.0;
    } else if (ndotd > 0.0) {
        a = max(a, base / ndotd);
    } else {
        b = min(b, base / ndotd);
    }
    if (b <= a) return 0.0;

    float sl = (r1 - r0) / L;
    float Ra = r0 + sl * (mdotd + 0.5 * L);
    float gl = (l1 - l0) / L;
    float La = l0 + gl * (mdotd + 0.5 * L);
    float qa = a * a + rho2, qb = b * b + rho2;
    float ata = atan(a / rho) / rho, atb = atan(b / rho) / rho;
    float t1 = (b / qb + atb) - (a / qa + ata);
    float t2 = 1.0 / qb - 1.0 / qa;
    float t3 = (-b / qb + atb) - (-a / qa + ata);
    float t4 = log(qb / qa) + rho2 * t2;
    float c0 = -base * Ra * La;
    float c1 = -base * (Ra * gl + sl * La) + ndotd * Ra * La;
    float c2 = -base * sl * gl + ndotd * (Ra * gl + sl * La);
    float c3 = ndotd * sl * gl;
    float e = c0 * t1 / rho + rho * (-c1 * t2 + c2 * t3 + c3 * t4);
    return max(e, 0.0) / PI;
}

float coneForm(vec3 P, vec3 A, vec3 B, float r0, float r1) {
    return coneForm(P, A, B, r0, r1, 1.0, 1.0);
}
"""
    + """
float ledForm(vec3 P, vec3 A, vec3 B, vec3 lat, float r0, float r1,
              float l0, float l1) {
    float sum = 0.0;
    float n = float(leds);
    for (int l = 0; l < MAX_LED; l++) {
        if (l >= leds) break;
        float t = (float(l) + 0.5) / n;
        vec3 S = mix(A, B, t);
        vec3 off = lat * (led_spread * mix(r0, r1, t));
        // split between the pair, carrying the chunk's own gradient
        float I = mix(r0, r1, t) * mix(l0, l1, t) * length(B - A) / n;
        for (int q = 0; q < 2; q++) {
            vec3 Sq = q == 0 ? S + off : S - off;
            vec3 dP = P - Sq;
            float r2 = max(dot(dP, dP), 1e-8);
            sum += I * max(Sq.z, 0.0) / (r2 * sqrt(r2));
        }
    }
    return sum / PI;
}
"""
)

# One texel per segment per ray. Only the segment index and the clock decide any
# of it, so no pixel has any business recomputing it.
SHADER_SEG = """
float level(float k) {
    return mix(1.0 - level_jit, 1.0, hash1(k * 1.7 + 3.1));
}

void main() {
    vec2 fc = floor(gl_FragCoord.xy);
    float k = fc.x;
    float ray = fc.y;
    float t = (k + 0.5) / float(segs);          // 0 at the hub, 1 at the tip

    // brightness travels, colour does not: each LED reads a slot of a pattern
    // scrolling along the index, crossfaded into its neighbour
    float ph = k - time * speed + ray_jit * (hash1(ray * 5.1) - 0.5) * float(segs);
    float f = smoothstep(0.0, 1.0, fract(ph));
    float b = mix(level(floor(ph)), level(floor(ph) + 1.0), f);
    b = mix(b, smoothstep(0.3, 0.8, b), snap);
    b *= mix(1.0, smoothstep(1.0, 0.1, t), taper);

    // Alpha carries the tint's luminance: a white LED runs all three emitters
    // and a saturated one only some, so a purple chunk throws less light than
    // a white one. Luma weights, because green is most of what reads as bright.
    float h = hue + hue_span * t + (hash1(k * 5.3 + 11.7) - 0.5) * hue_jit;
    vec3 tint = hsv(fract(h), sat, 1.0);
    float luma = dot(tint, vec3(0.2126, 0.7152, 0.0722));
    gl_FragColor = vec4(tint * b, luma);
}
"""

SHADER_WASH = (
    READ_SEG
    + """
void main() {
    float zm = max(zoom, 0.05);
    vec2 P = (v_texCoord - vec2(0.5)) * vec2(aspect, 1.0) / zm + pan;
    float r = length(P);
    float ang0 = atan(P.y, P.x);
    float base = floor((ang0 - spin * TAU) * float(rays) / TAU + 0.5);

    vec3 sum = vec3(0.0);
    for (int j = -NEAR_RAY; j <= NEAR_RAY; j++) {
        if (j < -near_ray || j > near_ray) continue;
        float idx = base + float(j);
        float ang = idx * TAU / float(rays) + spin * TAU;
        vec2 d = vec2(cos(ang), sin(ang));
        float p = dot(P, d);                        // along the ray
        float w = dot(P, vec2(-d.y, d.x));          // across it
        float len = rayReach(ang);
        // only the segments near the foot of the perpendicular reach this point
        float k0 = floor((p - hub) / max((len - hub) / float(segs), 1e-6));
        float row = mod(mod(idx, float(rays)) + float(rays), float(rays));
        for (int i = -NEAR_SEG; i <= NEAR_SEG; i++) {
            if (i < -near_seg || i > near_seg) continue;
            float k = k0 + float(i);
            if (k < 0.0 || k > float(segs) - 1.0) continue;
            float g0, g1, r0, r1;
            segSpan(k, len, g0, g1, r0, r1);
            // Each chunk pivots about its own middle by its own hash. Scaled
            // by the standoff rather than the chunk length: at full slider an
            // end moves by 0.6 of the height, so a chunk leans without
            // slamming the wall and printing a hot dot at one end.
            float dz = tilt_jit * (hash1(k * 7.7 + row * 3.3) - 0.5) * 1.2 * standoff;
            vec3 A = vec3(d * g0, max(rayZ(g0) - dz, 0.15 * r0));
            vec3 B = vec3(d * g1, max(rayZ(g1) + dz, 0.15 * r1));
            // the window is an index range, so whatever is about to leave it
            // must already have faded out or its exit prints a step on the wall
            float dxs = p - clamp(p, g0, g1);
            float cut = (max(float(near_seg), 1.0) - 0.5) * (len - hub)
                      / float(segs);
            float win = 1.0 - smoothstep(0.5 * cut, cut, abs(dxs));
            // led_mix is a uniform, so these branches skip for real: at the
            // ends of the slider only one model is paid for
            float hd, tl;
            segGrad(k, row, hd, tl);
            float Fc = led_mix < 0.999
                ? coneForm(vec3(P, 0.0), A, B, r0, r1, hd, tl) : 0.0;
            float Fl = led_mix > 0.001
                ? ledForm(vec3(P, 0.0), A, B, vec3(-d.y, d.x, 0.0), r0, r1,
                          hd, tl) : 0.0;
            float F = mix(Fc, Fl, led_mix);
            vec4 seg = segA(k, row);
            // chroma_dim: how much a saturated chunk under-lights the wall
            // relative to a white one
            sum += seg.rgb * (F * win * mix(1.0, seg.a, chroma_dim));
        }
    }
    // the hub is a body: it takes the wall out from under itself
    float ha = 1.5 / max(resolution.y * zm, 1.0);
    float hm = smoothstep(hub + ha, hub - ha, r);
    // sqrt so the buffer's 8 bits sit where the light does, in the dim tail.
    // Stored linearly the sideways spill quantizes into onion rings.
    gl_FragColor = vec4(sqrt(clamp(sum * (1.0 - hm), 0.0, 1.0)), 1.0 - hm);
}
"""
)

# The tubes themselves. A diffuser is brightest where the eye looks through the
# most of it, so the cross section carries a profile whose hottest part blows
# out to white, over a cylinder that turns away across the width and around the
# cut ends.
SHADER_TUBE = (
    READ_SEG
    + """
void main() {
    float zm = max(zoom, 0.05);
    vec2 P = (v_texCoord - vec2(0.5)) * vec2(aspect, 1.0) / zm + pan;
    float r = length(P);
    float ang0 = atan(P.y, P.x);
    float base = floor((ang0 - spin * TAU) * float(rays) / TAU + 0.5);
    float px = 1.5 / max(resolution.y * zm, 1.0);

    // the room's direction and its half vector do not vary per pixel or per
    // ray, so they are paid for once
    vec2 Lv = vec2(sin(shell_light * TAU), cos(shell_light * TAU));
    vec2 Hv = normalize(Lv + vec2(0.0, 1.0));
    float gloss = exp2(1.0 + shell_gloss * 7.0);

    // No tube reaches past the rim or inside the hub, and a neighbouring ray
    // can only touch this pixel within about rays * radius / pi of the centre,
    // where the rays crowd. rayReach is read at this pixel's angle but a ray
    // sits up to half a spacing off it, and rayReach swings hardest at the
    // square's corners, so the slack is the most it can climb over that half
    // spacing.
    float rmax = max(tube_r0, tube_r1);
    float rim = rayReach(ang0) + reach * square * 4.5 / float(rays) + 2.0 * rmax;
    if (r > rim || r < hub - px) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    int near = r < 1.5 * float(rays) * rmax / 3.14159 ? near_ray : 0;

    vec3 col = vec3(0.0);
    for (int j = -NEAR_RAY; j <= NEAR_RAY; j++) {
        if (j < -near || j > near) continue;
        float idx = base + float(j);
        float ang = idx * TAU / float(rays) + spin * TAU;
        vec2 d = vec2(cos(ang), sin(ang));
        float p = dot(P, d), w = dot(P, vec2(-d.y, d.x));
        float len = rayReach(ang);
        float row = mod(mod(idx, float(rays)) + float(rays), float(rays));
        float k = floor((p - hub) / max((len - hub) / float(segs), 1e-6));
        if (k < 0.0 || k > float(segs) - 1.0) continue;
        float g0, g1, r0, r1;
        segSpan(k, len, g0, g1, r0, r1);
        float t = clamp((p - g0) / max(g1 - g0, 1e-6), 0.0, 1.0);
        float rad = mix(r0, r1, t);
        // inside the segment, and inside the tube
        float along = max(max(g0 - p, p - g1), 0.0);
        float dist = length(vec2(along, w));
        float mask = 1.0 - smoothstep(rad - px, rad + px, dist);
        if (mask <= 0.0) continue;
        if (rayZ(p) + rad <= 0.0) continue;   // tilted down behind the wall
        // q is how far off the axis this bit of surface sits, nz how much of
        // it points back at the camera, nx how far it leans sideways. q comes
        // off the capsule distance, so the cut ends round like the flanks.
        float q = clamp(dist / max(rad, 1e-6), 0.0, 1.0);
        float nz = sqrt(max(1.0 - q * q, 0.0));
        float nx = clamp(w / max(rad, 1e-6), -1.0, 1.0);

        // how much diffuser the eye looks through: the chord, thinning out at
        // the silhouette and around the ends
        float thru = pow(1.0 - q * q, depth);
        float hd, tl;
        segGrad(k, row, hd, tl);
        vec3 c = segA(k, row).rgb * (thru * mix(hd, tl, t));
        // the LEDs themselves, hot through the skin when led_mix says so
        if (led_mix > 0.0) {
            float n = float(leds);
            float lt = (floor(t * n) + 0.5) / n;
            float dl = length(vec2(p - mix(g0, g1, lt), w)) / max(rad, 1e-6);
            c *= 1.0 + 2.5 * led_mix * exp(-dl * dl * 4.0);
        }
        float peak = max(max(c.r, c.g), c.b);
        c = mix(c, vec3(peak), clamp(core * peak, 0.0, 1.0));

        // A room off to one side of the cross section: wrapped diffuse for the
        // frosted body, a Blinn streak down its length. Both ride the surface
        // normal, so both curve with it.
        float body = max((nx * Lv.x + nz * Lv.y + 0.35) / 1.35, 0.0);
        float sp = pow(max(nx * Hv.x + nz * Hv.y, 0.0), gloss);
        col = max(col, (c + shell * body + shell_spec * sp) * mask);
    }
    // nothing shows through the hub
    col *= smoothstep(hub - px, hub + px, r);
    gl_FragColor = vec4(col / gain_max, 1.0);
}
"""
)

SHADER_MAIN = """
void main() {
    vec2 uv = v_texCoord;
    vec4 s = texture2D(iChannel0, uv);          // wall form factor, a = openness
    s.rgb *= s.rgb;                             // undo the wash's sqrt encode
    // the wall itself: a mono texture riding the albedo, sampled in wall
    // coordinates so it stays put under zoom and pan
    float zm = max(zoom, 0.05);
    vec2 Pw = (uv - 0.5) * vec2(aspect, 1.0) / zm + pan;
    float wt = wall_sample2(fract(Pw * wall_scale + 0.5));
    vec3 alb = albedo * mix(1.0, wt, wall_mix);
    vec3 t = texture2D(iChannel1, uv).rgb * GAIN_MAX;

    // the wall is albedo: it returns what reaches it, it does not glow
    vec3 wall = alb * (ambient * s.a + s.rgb * PI * radiance);
    vec3 col = vec3(1.0) - exp(-(wall + t * radiance) * exposure);
    // the camera's toe: below it everything crushes to true black, which keeps
    // the pools local instead of hazing the whole wall
    col = max(col - black, 0.0) / (1.0 - black);
    // the buffer feeds an sRGB display; without this encode the dim spill
    // crushes to black and the wall reads dead
    gl_FragColor = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}
"""


def p(name, val, lo, hi, step):
    return ghee.Param(name, "float", val, min=lo, max=hi, step=step)


def i(name, val, lo, hi):
    return ghee.Param(name, "int", val, min=lo, max=hi, step=1)


def build(a):
    emit = a.emit
    ns = max(0, a.near_seg)
    nr = max(0, a.near_ray)
    ml = max(1, a.max_leds)
    return ghee.Scene(
        passes=[
            ghee.Pass(name="seg", code=SHADER_SEG, size=SEG),
            ghee.Pass(
                name="wash",
                code=SHADER_WASH,
                size=a.wash or _wash_size(emit),
                channels=[ghee.Channel(buffer="seg")],
            ),
            ghee.Pass(
                name="tube",
                code=SHADER_TUBE,
                size=emit,
                channels=[ghee.Channel(buffer="seg")],
                params={
                    "depth": p("depth", 2.0, 0.05, 2.0, 0.01),
                    "core": p("core", 1.82, 0.0, 4.0, 0.02),
                    "shell": p("shell", 0.0, 0.0, 0.3, 0.005),
                    "shell_spec": p("shell_spec", 0.01, 0.0, 2.0, 0.01),
                    "shell_gloss": p("shell_gloss", 0.25, 0.0, 1.0, 0.01),
                    "shell_light": p("shell_light", -0.09, -0.25, 0.25, 0.005),
                    "gain_max": p("gain_max", 4.0, 4.0, 4.0, 1.0),
                },
            ),
            ghee.Pass(
                name="main",
                code=SHADER_MAIN,
                channels=[ghee.Channel(buffer="wash"), ghee.Channel(buffer="tube")],
                arrays=[ghee.ArrayInput("wall", WALL)],
                params={
                    "albedo": ghee.Param("albedo", "color", [0.7098, 0.7098, 0.7098]),
                    "wall_img": ghee.Param("wall_img", "string", ""),
                    "wall_mix": p("wall_mix", 1.0, 0.0, 1.0, 0.01),
                    "wall_scale": p("wall_scale", 0.45, 0.1, 8.0, 0.05),
                    "ambient": p("ambient", 0.085, 0.0, 0.5, 0.005),
                    "black": p("black", 0.06, 0.0, 0.4, 0.005),
                    "exposure": p("exposure", 1.05, 0.05, 6.0, 0.05),
                },
            ),
        ],
        common=(
            f"const int NEAR_SEG = {ns};\nconst int NEAR_RAY = {nr};\n"
            f"const int MAX_LED = {ml};\n" + SCENE_COMMON
        ),
        params={
            # the fan
            "rays": i("rays", 16, 3, 64),
            "reach": p("reach", 0.72, 0.05, 0.8, 0.005),
            "hub": p("hub", 0.053, 0.005, 0.4, 0.002),
            "square": p("square", 1.0, 0.0, 1.0, 0.01),
            "corner": p("corner", 0.125, 0.0, 0.25, 0.005),
            "spin": p("spin", 0.03, 0.0, 1.0, 0.002),
            # the tubes
            "segs": i("segs", 9, 1, 64),
            # how far the wall gathers: (2*near_seg+1) * (2*near_ray+1)
            # emitters per pixel, whatever the ray and segment counts are.
            # Trimming it clips the far spill.
            "near_seg": i("near_seg", min(1, ns), 0, ns),
            "near_ray": i("near_ray", min(1, nr), 0, nr),
            "seam": p("seam", 0.43, 0.0, 0.7, 0.01),
            # a chunk's gradient, hub end to tip end
            "head": p("head", 0.89, 0.0, 1.0, 0.01),
            "tail": p("tail", 0.16, 0.0, 1.0, 0.01),
            "ht_jit": p("ht_jit", 1.0, 0.0, 1.0, 0.01),
            "tube_r0": p("tube_r0", 0.013, 0.001, 0.06, 0.0005),
            "tube_r1": p("tube_r1", 0.014, 0.001, 0.06, 0.0005),
            "standoff": p("standoff", 0.026, 0.002, 0.2, 0.001),
            "tilt": p("tilt", 0.03, -1.2, 1.2, 0.01),
            "tilt_jit": p("tilt_jit", 1.0, 0.0, 1.0, 0.01),
            "leds": i("leds", min(2, ml), 1, ml),
            "led_mix": p("led_mix", 0.98, 0.0, 1.0, 0.01),
            "led_spread": p("led_spread", 0.0, 0.0, 1.2, 0.01),
            # the LEDs
            "radiance": p("radiance", 0.25, 0.0, 20.0, 0.05),
            "speed": p("speed", 0.9, -8.0, 8.0, 0.05),
            "snap": p("snap", 0.99, 0.0, 1.0, 0.01),
            "level_jit": p("level_jit", 1.0, 0.0, 1.0, 0.01),
            "taper": p("taper", 0.17, 0.0, 1.0, 0.01),
            "ray_jit": p("ray_jit", 1.0, 0.0, 1.0, 0.01),
            "hue": p("hue", 0.55, 0.0, 1.0, 0.005),
            "hue_span": p("hue_span", 0.2, -0.5, 0.5, 0.005),
            "hue_jit": p("hue_jit", 0.465, 0.0, 0.5, 0.005),
            "chroma_dim": p("chroma_dim", 0.78, 0.0, 1.0, 0.01),
            "sat": p("sat", 0.85, 0.0, 1.0, 0.01),
            # the view
            "zoom": p("zoom", 1.48, 0.2, 8.0, 0.01),
            "pan": ghee.Param("pan", "vec2", [0.0, 0.0], min=-1.0, max=1.0, step=0.002),
        },
    )


def _texture(size, path):
    """The wall surface.

    `path` names an image, and an unreadable one gives a soft plaster noise
    instead. The values are mono floats, and multiply into the albedo.
    """
    w, h = size
    if path:
        try:
            img = Image.open(os.path.expanduser(path)).convert("L")
            a = np.asarray(img.resize((w, h)), np.float32) / 255.0
            return a.reshape(-1)
        except OSError as e:
            # stdout belongs to the wire, so anything for a human goes to stderr
            print(f"wall_img {path!r}: {e}", file=sys.stderr)
    rng = np.random.default_rng(7)
    base = rng.random((24, 24)).astype(np.float32)
    soft = np.asarray(
        Image.fromarray(base, "F").resize((w, h), Image.BILINEAR), np.float32
    )
    grain = rng.random((h, w)).astype(np.float32)
    a = 0.75 + (soft - 0.5) * 0.4 + (grain - 0.5) * 0.06
    return np.clip(a, 0.0, 1.0).reshape(-1)


_wall = {"img": "", "pending": True}


@ghee.feed("wall")
def wall(tick):
    """The texture holds until wall_img names another file.

    This feed sends the texture once, and again on each change.
    """
    img = tick.params.get("wall_img")
    if img is not None and img != _wall["img"]:
        _wall["img"] = img
        _wall["pending"] = True
    if not _wall["pending"]:
        return None
    _wall["pending"] = False
    return _texture(WALL, _wall["img"])


ghee.run(build(args))
