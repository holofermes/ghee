"""Host-rasterized text through a 2D input: Pillow draws the string once into a
small mask, the shader scrolls and tints it. Whole-text motion and effects are
uv math in the shader; the host only supplies pixels. Add `--filters dither`
for 1-bit panels.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ghee import ArrayInput, Param, Pass, Scene, SceneController

TEXT = "GHEE"
W, H = 96, 16

SHADER = """
void main() {
    vec2 uv = v_texCoord;
    float m = text_sample2(vec2(fract(uv.x + time * speed), uv.y));
    vec3 c = mix(vec3(0.02, 0.03, 0.02), vec3(0.2, 1.0, 0.5), m);
    gl_FragColor = vec4(c, 1.0);
}
"""


class Marquee(SceneController):
    def __init__(self):
        self._mask = None

    def build(self, base):
        return Scene(
            passes=[
                Pass(
                    name="main",
                    code=SHADER,
                    arrays=[ArrayInput("text", (W, H))],
                    params={
                        "speed": Param("speed", "float", 0.15, min=0.0, max=1.0, step=0.01)
                    },
                )
            ]
        )

    def provide(self, name, size, frame):
        if self._mask is None:
            img = Image.new("L", (W, H), 0)
            d = ImageDraw.Draw(img)
            f = ImageFont.load_default()
            bb = d.textbbox((0, 0), TEXT, font=f)
            d.text(
                ((W - (bb[2] - bb[0])) // 2, (H - (bb[3] - bb[1])) // 2 - bb[1]),
                TEXT,
                fill=255,
                font=f,
            )
            self._mask = (np.asarray(img, dtype=np.float32) / 255.0).reshape(-1)
        return self._mask
