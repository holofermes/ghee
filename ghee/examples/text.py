"""Host-rasterized text through a 2D input.

Pillow draws the string once into a mask; the shader scrolls, thresholds, and
tints it. The word is host-side (set $GHEE_TEXT); the motion, crispness, and
colours are live shader knobs you tweak in the console. No dither here: text is
line art, so a threshold keeps it crisp on a 1-bit panel.

    GHEE_TEXT="hello pi" ghee serve --demo        # then edit it live
    ghee render text --display ssd1306            # to a panel, no dither
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ghee import ArrayInput, Param, Pass, Scene, SceneController

SHADER = """
void main() {
    vec2 uv = v_texCoord;
    float m = text_sample2(vec2(fract(uv.x - time * speed), uv.y));
    float g = smoothstep(threshold - 0.08, threshold + 0.08, m);
    vec3 c = mix(vec3(bg), vec3(ink_r, ink_g, ink_b), g);
    gl_FragColor = vec4(c, 1.0);
}
"""


class Text(SceneController):
    def __init__(self):
        self._mask = None

    def build(self, base):
        return Scene(
            passes=[
                Pass(
                    name="main",
                    code=SHADER,
                    arrays=[ArrayInput("text", self._bake())],
                    params={
                        "speed": Param(
                            "speed", "float", 0.1, min=-0.6, max=0.6, step=0.01
                        ),
                        "threshold": Param(
                            "threshold", "float", 0.4, min=0.05, max=0.95, step=0.01
                        ),
                        "bg": Param("bg", "float", 0.02, min=0.0, max=1.0, step=0.01),
                        "ink_r": Param(
                            "ink_r", "float", 0.2, min=0.0, max=1.0, step=0.01
                        ),
                        "ink_g": Param(
                            "ink_g", "float", 1.0, min=0.0, max=1.0, step=0.01
                        ),
                        "ink_b": Param(
                            "ink_b", "float", 0.5, min=0.0, max=1.0, step=0.01
                        ),
                    },
                )
            ]
        )

    def _bake(self):
        # embedded TrueType at a readable size; Pi OS Lite ships no other fonts
        text = os.environ.get("GHEE_TEXT", "GHEE")
        font = ImageFont.load_default(32)
        pad = 6
        scratch = ImageDraw.Draw(Image.new("L", (1, 1)))
        bb = scratch.textbbox((0, 0), text, font=font)
        w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).text((pad - bb[0], pad - bb[1]), text, fill=255, font=font)
        self._mask = (np.asarray(img, np.float32) / 255.0).reshape(-1)
        return (w, h)

    def provide(self, name, size, frame):
        return self._mask
