"""A procedural scene with no JSON: it builds its pass in code and feeds a custom
array each frame. Run it with `ghee render <this file> -o out.####.png --frames N`.
"""

import numpy as np

from ghee import ArrayInput, Pass, Scene, SceneController

BARS = 48

SHADER = """
void main() {
    vec2 uv = gl_FragCoord.xy / resolution;
    int i = int(uv.x * float(bars_size));
    if (i > bars_size - 1) i = bars_size - 1;
    float h = bars_at(i);
    float lit = step(uv.y, h);
    vec3 col = mix(
        vec3(0.02, 0.0, 0.05),
        vec3(0.2 + 0.8 * (0.5 + 0.5 * sin(time * 2.0 + uv.x * 8.0)), uv.y, 1.0 - uv.y),
        lit);
    gl_FragColor = vec4(col, 1.0);
}
"""


class Spectrum(SceneController):
    def build(self, base):
        return Scene(
            passes=[Pass(name="main", code=SHADER, arrays=[ArrayInput("bars", BARS)])]
        )

    def provide(self, name, size, frame):
        if name == "bars":
            x = np.linspace(0.0, 1.0, size)
            wave = np.sin(2 * np.pi * 3 * x + frame * 0.07) * np.sin(
                np.pi * x + frame * 0.02
            )
            return (0.5 + 0.45 * wave).astype(np.float32)
        return None
