#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["ghee-render>=0.3.0", "pillow"]
# ///
"""Host-rasterized text through a 2D input.

Pillow draws the string once into a mask. The shader scrolls the mask, then
thresholds it and tints it over a wave.

`--text` is build time, because the mask is a fixed-size array. Motion,
crispness and color stay live. Line art needs no dither. The wave behind the
text stays under the threshold of a 1-bit panel while `glow` stays under it.

    ghee render --demo text -o out.####.png --frames 60
    ghee render --demo text --display ssd1306       # to a panel, no dither
    ghee render --demo text -- --text "hello there"
"""

import ghee

args = ghee.args(
    [ghee.Arg("text", "string", default="GHEE", help="the word to rasterize")]
)

# Slow imports come after the parse: the probe must answer inside its window.
from array import array  # noqa: E402

from lib import look  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

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
    """One draw at a readable size. The mask lands near 1:1 on a small panel."""
    pad = FONT_SIZE // 5
    bb = _scratch().textbbox((0, 0), text, font=font)
    img = Image.new("L", (bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad), 0)
    ImageDraw.Draw(img).text((pad - bb[0], pad - bb[1]), text, fill=255, font=font)
    return img


def _raster_bitmap(text):
    """The 10px font, with the letters spaced by hand.

    The mask zooms for hard edges. The result stays small and soft, so this path
    serves a Pillow build with no sized load_default() alone.
    """
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


def _bake(text):
    try:
        font = ImageFont.load_default(size=FONT_SIZE)
    except TypeError:  # Pillow < 10.1 takes no size here
        img = _raster_bitmap(text)
    else:
        img = _raster(text, font)
    return array("f", [v / 255.0 for v in img.getdata()]), img.size


MASK, (W, H) = _bake(args.text)

_sent = {"done": False}


@ghee.feed("text")
def text(tick):
    """The mask holds in the engine texture after one send."""
    if _sent["done"]:
        return None
    _sent["done"] = True
    return MASK


def build():
    return ghee.Scene(
        common=look.PALETTE + look.WAVE + look.TONE,
        passes=[
            ghee.Pass(
                name="main",
                code=f"const float mask_aspect = {W / H:.6f};\n{SHADER}",
                arrays=[ghee.ArrayInput("text", (W, H))],
                params={
                    "speed": ghee.Param(
                        "speed", "float", 0.1, min=-0.6, max=0.6, step=0.01
                    ),
                    "threshold": ghee.Param(
                        "threshold", "float", 0.4, min=0.05, max=0.95, step=0.01
                    ),
                    "glow": ghee.Param(
                        "glow", "float", 0.3, min=0.0, max=1.0, step=0.01
                    ),
                    "ink": ghee.Param("ink", "color", [0.72, 0.92, 1.0]),
                },
            )
        ],
    )


ghee.run(build())
