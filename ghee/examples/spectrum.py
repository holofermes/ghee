"""A scene with no JSON. It builds its pass in code, and feeds one array each
frame.

`gain` is one param on both sides. The shader reads it as a uniform for the bar
brightness. provide() reads it in Python for the bar height.

    ghee render --demo spectrum -o out.####.png --frames 60
    ghee serve --demo spectrum                      # drive it in the console
"""

import importlib

import numpy as np
from lib import look

from ghee import ArrayInput, Param, Pass, Scene, SceneController

# a scene is re-imported on reload but its libraries are not, so without this a
# reload after editing look.py would render the cached one
importlib.reload(look)

BARS = 48

SHADER = """
void main() {
    vec2 uv = vec2(v_texCoord.x, 1.0 - v_texCoord.y);  // y climbs off the floor
    float fi = uv.x * float(bars_size);
    int i = int(min(fi, float(bars_size - 1)));
    float h = bars_at(i);

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


class Spectrum(SceneController):
    def build(self, base):
        self.scene = Scene(
            common=look.PALETTE + look.TONE,
            passes=[
                Pass(
                    name="main",
                    code=SHADER,
                    arrays=[ArrayInput("bars", BARS)],
                    params={
                        "gain": Param("gain", "float", 0.6, min=0.0, max=1.0, step=0.01)
                    },
                )
            ],
        )
        return self.scene

    def provide(self, name, size, frame):
        if name == "bars":
            gain = self.scene.passes[0].params["gain"].value
            x = np.linspace(0.0, 1.0, size)
            t = frame * 0.05
            # look.WAVE in numpy: two sines that are not quite harmonics running
            # opposite ways, so the crest travels and never lands twice on the
            # same bar
            w = 0.28 * np.sin(x * 4.4 + t) + 0.10 * np.sin(x * 9.7 - t * 0.6)
            tilt = np.exp(-1.7 * x)  # a spectrum leans on its low end
            return (gain * (0.55 + tilt) * (0.65 + w)).astype(np.float32)
        return None
