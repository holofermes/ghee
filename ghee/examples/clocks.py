#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render"]
# ///
"""A grid of dials, each spinning at its own band's rate.

One buffer pass integrates a phase per band and scrolls the history up; the
image pass reads a few frames back for the motion glow. The grid sizes itself
to the frame, so the same scene fills a 128x32 strip and a 1280x800 panel
without an argument.

Origin unconfirmed: this may descend from a Shadertoy, and no source is
recorded. Credit it here if you find it.

    ghee render --demo clocks -o out.####.png --frames 60
    ghee serve --demo                         # cols and rows override the grid
"""

import ghee

ghee.args()  # the probe, this scene takes no arguments

# Slow imports come after the parse: the probe must answer inside its window.
import math  # noqa: E402
from array import array  # noqa: E402

BANDS = 64

SHADER_BUF = """
// The spin rate is read back as a phase difference across RATE_ROWS frames of
// history, so it is only unambiguous below one whole turn over that span. Past
// it the difference wraps and a fast band reads as a slow one, which lurches.
// The drive is capped to match, which also keeps the hand under the frame
// rate's own Nyquist limit.
const float RATE_ROWS = 8.0;
const float MAX_INC = 0.999 / RATE_ROWS;

// The buffer packs a float across RGB and the sampler is bilinear, so a read
// has to land on a texel centre. Blending two packed values decodes to an
// angle that belongs to neither, which shows as a clock jumping.
vec2 texel(vec2 p){ return (floor(p*iResolution.xy)+0.5)/iResolution.xy; }

float rgbToFloat(vec3 c){ return c.r + c.g/255.0 + c.b/(255.0*255.0); }
vec3 floatToRgb(float v){ float r=mod(v,1.0); float g=mod(v*255.0,1.0); r-=g/255.0;
    float b=mod(v*255.0*255.0,1.0); g-=b/255.0; return vec3(r,g,b); }

// Column = frequency band. The bottom row integrates a phase; a loud band spins
// fast. To keep the spin from lurching on spiky bass, the new increment is blended
// with the band's recent average rate (read back from the scroll history) by damp,
// so a kick ramps the spin up instead of jerking it. The rows above scroll the
// phase history up so the image pass can read a few frames back for the motion glow.
void main(){
    vec2 uv = gl_FragCoord.xy / iResolution.xy;
    float px = 1.0 / iResolution.y;
    if (uv.y < px) {
        float raw = amplitudes_at(int(uv.x * float(amplitudes_count - 1)));
        float fftP = max(0.0, raw - thres) / (1.0 - thres);
        float cur = rgbToFloat(texture2D(iChannel0, uv).rgb);
        float old = rgbToFloat(texture2D(iChannel0, texel(vec2(uv.x, RATE_ROWS/iResolution.y))).rgb);
        float recentRate = fract(cur - old + 1.0) / RATE_ROWS;
        // gain is calibrated to a 0.03 s frame
        float inc = mix(min(fftP / 255.0 * gain * (iTimeDelta / 0.03), MAX_INC), recentRate, damp);
        gl_FragColor = vec4(floatToRgb(cur + inc), 1.0);
    } else {
        gl_FragColor = vec4(texture2D(iChannel0, uv - vec2(0.0, px)).rgb, 1.0);
    }
}
"""

SHADER_MAIN = """
// The grid sizes itself: dials land near CELL pixels across and stay square,
// so one scene reads on a 128x32 strip and on a 1280x800 panel. Under
// MIN_CELL the cell stops shrinking and the outer dials crop instead, which
// keeps a dial the size it wants to be on a panel too small to hold a whole
// row of them. cols and rows above zero say it explicitly and win.
const float CELL = 160.0;
const float MIN_CELL = 32.0;
vec2 gridOf(){
    if (cols >= 1.0 && rows >= 1.0) return vec2(cols, rows);
    float r = max(1.0, floor(iResolution.y / CELL + 0.5));
    float cell = iResolution.y / r;
    if (cell < MIN_CELL) { cell = MIN_CELL; r = max(1.0, floor(iResolution.y / cell)); }
    float c = max(1.0, floor(iResolution.x / cell + 0.5));
    return vec2(c, r);
}

// The buffer packs a float across RGB and the sampler is bilinear, so a read
// has to land on a texel centre. Blending two packed values decodes to an
// angle that belongs to neither, which shows as a clock jumping.
vec2 texel(vec2 p){ return (floor(p*iResolution.xy)+0.5)/iResolution.xy; }

#define PI 3.14159265
#define TAU 6.2831853
float rgbToFloat(vec3 c){ return c.r + c.g/255.0 + c.b/(255.0*255.0); }
float clockMap(vec2 p, float value){
    float ang = mod(atan(p.x, p.y) - value, TAU)/2.0 - PI;
    return smoothstep(-PI, PI, ang);
}
void main(){
    vec2 uv = gl_FragCoord.xy / iResolution.xy;
    vec2 grid = gridOf();
    float nc = grid.x, nr = grid.y;
    int gid = int((uv.x + floor(uv.y*nr)) * nc);
    vec2 g = mod(uv*grid, 1.0) - 0.5;
    g.x /= (iResolution.y*nc)/(iResolution.x*nr);
    float sx = float(gid)/(nc*nr);
    float value = rgbToFloat(texture2D(iChannel0, texel(vec2(sx, 0.0))).rgb) * TAU;
    float prev  = rgbToFloat(texture2D(iChannel0, texel(vec2(sx, lag/iResolution.y))).rgb) * TAU;
    float cv = clockMap(g, value);
    cv += max(0.0, cv - clockMap(g, prev)) * glow;
    cv += abs(sin(value) - sin(prev)) * glow;
    cv *= smoothstep(0.0, 1.0, max(iResolution.x, iResolution.y)/8.0 * (face - length(g)));
    gl_FragColor = vec4(vec3(cv), 1.0);
}
"""


def build():
    return ghee.Scene(
        passes=[
            ghee.Pass(
                name="buf",
                fragment=SHADER_BUF,
                arrays=[ghee.ArrayInput("amplitudes", BANDS)],
                channels=[ghee.Channel(buffer="buf")],
                params={
                    "gain": ghee.Param(
                        "gain", "float", 8.0, min=1.0, max=40.0, step=0.5
                    ),
                    "thres": ghee.Param(
                        "thres", "float", 0.16, min=0.0, max=0.9, step=0.02
                    ),
                    "damp": ghee.Param(
                        "damp", "float", 0.15, min=0.0, max=0.95, step=0.05
                    ),
                },
            ),
            ghee.Pass(
                name="main",
                fragment=SHADER_MAIN,
                channels=[ghee.Channel(buffer="buf")],
                params={
                    "cols": ghee.Param(
                        "cols", "float", 0.0, min=0.0, max=16.0, step=1.0
                    ),
                    "rows": ghee.Param(
                        "rows", "float", 0.0, min=0.0, max=16.0, step=1.0
                    ),
                    "face": ghee.Param(
                        "face", "float", 0.7, min=0.1, max=1.0, step=0.05
                    ),
                    "glow": ghee.Param(
                        "glow", "float", 0.1, min=0.0, max=3.0, step=0.05
                    ),
                    "lag": ghee.Param("lag", "float", 1.0, min=1.0, max=16.0, step=1.0),
                },
            ),
        ],
    )


@ghee.feed("amplitudes")
def amplitudes(tick):
    """Levels in place of a spectrum's own.

    Each band keeps a rate of its own, so neighbouring dials drift apart
    instead of reading as one clock repeated across the grid.
    """
    t = tick.time
    out = array("f", bytes(4 * BANDS))
    for i in range(BANDS):
        x = i / (BANDS - 1)
        tilt = 0.35 + 0.45 * (1.0 - x) ** 1.6  # a spectrum leans on its low end
        swell = 0.25 * math.sin(t * (0.6 + 2.0 * x) + x * 7.0)
        out[i] = min(1.0, max(0.0, tilt + swell))
    return out


ghee.run(build())
