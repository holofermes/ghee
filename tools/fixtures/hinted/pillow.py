"""Writes the PNGs that tests/text.rs holds a hinted layer's frame to: words
in DejaVu Sans, the font of a text that names none, as Pillow draws them,
one image per size and mode.

usage: uv run --no-project --with pillow==12.3.0 python -I pillow.py

Pillow 12.3.0's wheels carry FreeType 2.14.3. Each image is 192 x 24, black,
the words drawn white at (1, y) with Pillow's default anchor, the y of each
case putting the baseline on row 18: mode "L" holds FreeType's grey coverage,
mode "1" its 1-bit render, saved as 0 and 255.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

HERE = Path(__file__).resolve().parent
FONT = HERE.parents[2] / "ghee" / "examples" / "fonts" / "DejaVuSans-Latin.ttf"
WORDS = "cpu 42% ping 12ms"
CASES = [(8, "L", 10), (12, "L", 6), (10, "1", 8), (16, "1", 3)]

assert features.version("freetype2") == "2.14.3", features.version("freetype2")
for size, mode, y in CASES:
    font = ImageFont.truetype(str(FONT), size)
    assert y + font.getmetrics()[0] == 18, (size, y)
    im = Image.new(mode, (192, 24), 0)
    d = ImageDraw.Draw(im)
    d.text((1, y), WORDS, font=font, fill=255 if mode == "L" else 1)
    im.convert("L").save(
        HERE / f"dejavu-{size}-{'grey' if mode == 'L' else 'mono'}.png", optimize=True
    )
