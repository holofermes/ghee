"""Host-rasterized text through a 2D input: Pillow draws the string once into a
mask, the shader scrolls, thresholds and tints it over a wave.

`--text` is build time because the mask is a fixed-size array; motion,
crispness and colour stay live. Line art needs no dither, and the backing wave
stays under a 1-bit panel's threshold as long as `glow` does.

    ghee serve --demo -- --text "hello there"       # then edit it live
    ghee render --demo text --display ssd1306       # to a panel, no dither
"""

import importlib

import numpy as np
from lib import look
from PIL import Image, ImageDraw, ImageFont

from ghee import UNCHANGED, Arg, ArrayInput, Param, Pass, Scene, SceneController

# a scene is re-imported on reload but its libraries are not, so without this a
# reload after editing look.py would render the cached one
importlib.reload(look)

FONT_SIZE = 28

SHADER = """
void main() {
    vec2 uv = v_texCoord;
    float aspect = resolution.x / resolution.y;
    float across = max(1.0, aspect / mask_aspect);
    float down = max(1.0, mask_aspect / aspect);
    float v = (uv.y - 0.5) * down + 0.5;
    float m = text_sample2(vec2(fract(uv.x * across - time * speed), v));
    m *= step(0.0, v) * step(v, 1.0);
    float g = smoothstep(threshold - 0.08, threshold + 0.08, m);

    float band = exp(-abs(uv.y - 0.5 - wave(uv.x, time * 0.5)) * 9.0);
    vec3 col = palette(0.14 + 0.16 * uv.x) * glow * band + ink * g * 2.0;
    gl_FragColor = vec4(tone(col, 0.02, 1.4), 1.0);
}
"""


def _scratch():
    return ImageDraw.Draw(Image.new("L", (1, 1)))


def _raster(text, font):
    """One draw at a readable size: the mask lands near 1:1 on a small panel."""
    pad = FONT_SIZE // 5
    bb = _scratch().textbbox((0, 0), text, font=font)
    img = Image.new("L", (bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad), 0)
    ImageDraw.Draw(img).text((pad - bb[0], pad - bb[1]), text, fill=255, font=font)
    return img


def _raster_bitmap(text):
    """The ~10px font, letters spaced by hand and the mask zoomed for hard
    edges. Small and soft, so it is only for Pillow builds with no sized
    load_default()."""
    pad, gap, zoom = 2, 2, 4
    font = ImageFont.load_default()
    scratch = _scratch()
    bb = scratch.textbbox((0, 0), text, font=font)
    advances = [scratch.textlength(c, font=font) for c in text]
    w = int(sum(advances)) + gap * (len(text) - 1) + 2 * pad
    h = bb[3] - bb[1] + 2 * pad
    img = Image.new("L", (w, h), 0)
    draw, x = ImageDraw.Draw(img), float(pad)
    for c, adv in zip(text, advances, strict=True):
        draw.text((x, pad - bb[1]), c, fill=255, font=font)
        x += adv + gap
    return img.resize((w * zoom, h * zoom), Image.NEAREST)


class Text(SceneController):
    ARGS = [Arg("text", default="GHEE", help="the word to rasterize")]

    def __init__(self):
        self._mask = None
        self._pending = True

    def build(self, base):
        w, h = self._bake()
        return Scene(
            common=look.PALETTE + look.WAVE + look.TONE,
            passes=[
                Pass(
                    name="main",
                    code=f"const float mask_aspect = {w / h:.6f};\n{SHADER}",
                    arrays=[ArrayInput("text", (w, h))],
                    params={
                        "speed": Param(
                            "speed", "float", 0.1, min=-0.6, max=0.6, step=0.01
                        ),
                        "threshold": Param(
                            "threshold", "float", 0.4, min=0.05, max=0.95, step=0.01
                        ),
                        "glow": Param(
                            "glow", "float", 0.3, min=0.0, max=1.0, step=0.01
                        ),
                        "ink": Param("ink", "color", [0.72, 0.92, 1.0]),
                    },
                )
            ],
        )

    def _bake(self):
        text = self.args.text
        try:
            font = ImageFont.load_default(size=FONT_SIZE)
        except TypeError:  # Pillow < 10.1 takes no size here
            img = _raster_bitmap(text)
        else:
            img = _raster(text, font)
        self._mask = (np.asarray(img, np.float32) / 255.0).reshape(-1)
        return img.size

    def setup(self, renderer):
        self._pending = True  # setup empties the engine's textures

    def provide(self, name, size, frame):
        if name != "text":
            return None
        if self._pending:
            self._pending = False
            return self._mask
        return UNCHANGED
