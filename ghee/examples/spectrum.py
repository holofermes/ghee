"""A procedural scene with no JSON: it builds its pass in code and feeds a custom
array each frame. Run it with `ghee render <this file> -o out.####.png --frames N`,
or `ghee serve` to drive it in the console.

The `gain` param lives in both worlds: the shader reads it as a uniform (bar
brightness) and provide() reads it in Python (bar height). One slider, both sides.
"""

import numpy as np

from ghee import ArrayInput, Param, Pass, Scene, SceneController

BARS = 48

SHADER = """
void main() {
    vec2 uv = gl_FragCoord.xy / resolution;
    int i = int(uv.x * float(bars_size));
    if (i > bars_size - 1) i = bars_size - 1;
    float h = bars_at(i);
    float lit = step(uv.y, h);
    vec3 hue = vec3(0.2 + 0.8 * (0.5 + 0.5 * sin(time * 2.0 + uv.x * 8.0)), uv.y, 1.0 - uv.y);
    vec3 col = mix(vec3(0.02, 0.0, 0.05), hue * (0.3 + 1.4 * gain), lit);
    gl_FragColor = vec4(col, 1.0);
}
"""


class Spectrum(SceneController):
    def build(self, base):
        self.scene = Scene(
            passes=[
                Pass(
                    name="main",
                    code=SHADER,
                    arrays=[ArrayInput("bars", BARS)],
                    params={
                        "gain": Param(
                            "gain", "float", 0.45, min=0.0, max=0.5, step=0.01
                        )
                    },
                )
            ]
        )
        return self.scene

    def provide(self, name, size, frame):
        if name == "bars":
            gain = self.scene.passes[0].params["gain"].value
            x = np.linspace(0.0, 1.0, size)
            wave = np.sin(2 * np.pi * 3 * x + frame * 0.07) * np.sin(
                np.pi * x + frame * 0.02
            )
            return (0.5 + gain * wave).astype(np.float32)
        return None
