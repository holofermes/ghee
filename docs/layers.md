# Layers

A pass's shader fills its buffer, then the pass's `layers` paint over it, in
list order. A text layer draws words from a font, an image layer draws a
picture, and a shape layer draws a rect or a line. Each takes its settings as
keys, and ghee draws them. A pass's shader can also read the words of any text
layer and any image asset, for the per-pixel work that settings do not cover,
as [Coverage in a shader](#coverage-in-a-shader) shows. From Python, the
`draw()` of a `ghee.open` handle writes the same layers from a loop, with no
scene to write first, as [docs/python.md](python.md#drawing) shows.

This scene draws `hello` in a red to yellow gradient with a soft shadow,
fitted to the frame and centred, and puts the image `stamp.png` 16 pixels wide
in the top left corner:

```json
{
  "version": 1,
  "assets": { "stamp": { "file": "stamp.png" } },
  "passes": [{
    "name": "main",
    "fragment": "void main() { gl_FragColor = vec4(0.1, 0.12, 0.2, 1.0); }",
    "layers": [
      { "name": "title", "text": "hello",
        "color": { "from": "#ff0000", "to": "#ffff00", "angle": 90 },
        "text_shadow": {} },
      { "name": "icon", "image": "stamp", "x": 4, "y": 4, "width": 16 }
    ]
  }]
}
```

Every layer takes these keys:

| Key | Value | Default |
| --- | --- | --- |
| `name` | the layer's name, per [Names](scenes.md#names) | required |
| `visible` | `false` hides the layer | `true` |
| `x`, `y` | the pass pixel of the block's top left, y down | centred on a missing axis |
| `color` | a [colour](#colours) | white for text and a shape, no tint for an image |
| `opacity` | 0 to 1 | 1 |
| `blend` | `normal`, `add`, `multiply` or `screen` | `normal` |

A layer's colour, premultiplied by alpha, lands on the pass's buffer by its
`blend`, and alpha blends the same way: `normal` gives src + dst × (1 −
src.a), `add` gives src + dst, `multiply` gives src × dst + dst × (1 −
src.a), and `screen` gives src + dst × (1 − src). The formulas of `multiply`
and `screen` hold over an opaque buffer, such as the output of the pass's own
shader. A pass with `layers` and no GLSL of its own starts each frame
transparent and holds its layers alone, for a later pass to read through a
channel. A later pass reads the whole buffer, layers included.

## Colours

A colour is `"#rrggbb"`, `[r, g, b]` with each 0 to 1, or one of the 148
[named colors of CSS Color 4](https://www.w3.org/TR/css-color-4/#named-colors),
such as `"navy"`, in any case. A scene keeps a name as it was written.
`"transparent"` paints nothing: letters in it still draw their effects, and a
gradient to it fades out. A color param reaches its shader as three numbers,
so `transparent` reads there as black.

## Text layers

A text layer takes these keys on top of every layer's:

| Key | Value | Default |
| --- | --- | --- |
| `text` | the words | required |
| `font_size` | pixels per em | the largest size that fits |
| `font_family` | a font asset's name, a `.ttf` or `.otf` file, `DejaVu Sans` or `Pixel Operator 8` | `DejaVu Sans` |
| `text_align` | `left`, `center` or `right` | `left` |
| `render` | `auto`, `hinted`, `mono`, `mtsdf` or `slug`, as [Render](#render) describes | `auto`, `slug` in `Pixel Operator 8` |

The words lay out into a block, as wide as the longest line, with one line
height for each line, and `text_align` puts each line within the block. A
layer with no `font_size` takes the largest size at which the block fits
inside the pass, with a margin of 5% of the pass's shorter side on every
edge. On an axis with `x` or `y` given, the room runs from that pixel to the
margin at the far edge. The size follows the words, so new words fit again.

`font_size` sets the size in pixels per em, as a font size does in CSS. One
line of DejaVu Sans is 1.164 em tall, so at 48 pixels per em a line is 56
pixels tall. This status screen puts two lines in the top left corner of a
128x32 panel:

```json
"layers": [
  { "name": "cpu", "text": "cpu 12%", "font_size": 12, "x": 0, "y": 0 },
  { "name": "temp", "text": "temp 41°C", "font_size": 12, "x": 0, "y": 16 }
]
```

A text layer's `color` also takes a gradient, `{"from", "to", "angle"}`, with
CSS's angle in degrees: 0 runs up, 90 right, and 180, the default, down. The
gradient spans the block and moves with it.

A text layer's `text` is a live field: Python's `layers`, `/set`, the console
and a procedural scene set new words through it, and ghee lays them out again
with no rebuild.

## Effects

An effect is a key of a text layer that holds an object of its fields. A field
left out takes its default, so `"glow": {}` draws the default glow. Sizes are
in em, and so scale with the words.

| Effect | Fields and defaults |
| --- | --- |
| `text_shadow` | `x` 0.06, `y` 0.08, `blur` 0.15, `color` `#000000`, `strength` 0.7 |
| `glow` | `blur` 0.3, `color` `#ffa500`, `strength` 0.8 |
| `text_stroke` | `width` 0.06, `color` `#000000` |
| `karaoke` | `sung` 0, `color` `#ffad2e` |
| `wave` | `height` 0.1, `speed` 0.6 |
| `scroll` | `speed` 2 |
| `blink` | `rate` 1 |

- `text_shadow` draws a copy of the letters moved by `x` and `y`, softened
  over `blur`, at `strength` alpha.
- `glow` lights the room around the letters, from `strength` alpha at their
  outline to nothing at `blur`.
- `text_stroke` grows the letters by `width` in its colour.
- `karaoke` fills the letters in its colour up to the `sung` point, which
  counts characters and takes fractions, and lifts the letter being sung by
  up to 0.15 em as the point passes its middle.
- `wave` bobs each letter up and down by `height`, `speed` waves a second,
  each letter a tenth of a wave behind the one before it.
- `scroll` moves the block from right to left, `speed` em a second, by whole
  pixels, so on a 1-bit panel each letter keeps its pixels. The block enters
  at the pass's right edge, and starts there again once it has left at the
  left. The layer's `x` takes no part while it scrolls.
- `blink` shows the layer for the first half of each blink, `rate` blinks a
  second.

Each glyph draws its shadow, glow and stroke, in that order, before any glyph
draws its letter, so one letter's glow never covers the next letter. The
motion effects move the letters and their effects together. `"visible": false`
on an effect bypasses it and keeps its values. ghee works out how far a
layer's effects reach past its letters, and fades them out at the edge of
that reach.

A text layer of the `normal` blend fades as one: below an `opacity` of 1, its
letters draw over their effects first, and the result lands on the pass at
that opacity, so the shadow and the glow never show through the letters.

Careful: under the other blends, the effects and the letters each take the
layer's `opacity` and `blend`, so below an opacity of 1 the effects show
through the letters, and under `add` a black shadow adds nothing.

## Image layers

An image layer takes these keys on top of every layer's, and no effects:

| Key | Value | Default |
| --- | --- | --- |
| `image` | an image asset's name, or a `.png` or `.jpg` file | required |
| `width`, `height` | pass pixels | the image's own size |

One of `width` and `height` left out keeps the image's aspect, and both given
stretch it. `color` tints the image: it multiplies each pixel. An image layer
gives the passes no GLSL name of its own, and a pass's shader reads the image
as its asset's `<name>_tex`.

## Shape layers

A shape layer holds `shape`, `rect` or `line`, and takes no effects. This
scene draws a progress bar at the bottom of a 128x32 panel, a disc in the top
right corner, and a lime waveform:

```json
"layers": [
  { "name": "frame", "shape": "rect", "x": 0, "y": 24, "width": 128, "height": 8,
    "color": "transparent", "border": { "width": 1, "color": "white" } },
  { "name": "fill", "shape": "rect", "x": 2, "y": 26, "width": 60, "height": 4 },
  { "name": "dial", "shape": "rect", "x": 100, "y": 4, "width": 20, "height": 20,
    "border_radius": 10 },
  { "name": "scope", "shape": "line", "points": [[0, 16], [8, 4], [16, 28]],
    "stroke_width": 1, "color": "lime" }
]
```

| Key | Value | Default |
| --- | --- | --- |
| `width`, `height` | a rect's box in pass pixels | the pass's own |
| `border` | a rect's border, an object of `width` in pixels, `color` and `visible` | none, and in the object `width` 1 and `color` white |
| `border_radius` | a rect's corner radius in pixels | 0 |
| `points` | a line's points, each `[x, y]`, two or more | required |
| `stroke_width` | a line's thickness in pixels | 1 |

A rect's `color` fills its box, and its border lies inside the box's edge. A
radius of half the shorter side gives round ends, and more rounds no further.
A line's point is a pixel, as luma and Pillow count them, so a line of
`stroke_width` 1 from `[0, 31]` to `[127, 31]` lights row 31. A line's `x`
and `y` shift every point, 0 by default. Every edge is antialiased from the
distance to the shape, so a shape at whole pixels lights whole pixels.

Every field of a shape layer but `shape` and `blend` changes with no rebuild.
ghee draws every shape with one shader, so a shape layer that changes or
hides compiles nothing.

## Live fields

Some fields change with no rebuild: a layer's `x`, `y`, `color`, `opacity` and
`visible`, a text layer's `text`, an image layer's `image`, which takes another
image asset of the scene or an image file, which loads once and stays until
the next setup, an effect's `visible`, `color`, `strength`, `sung`, `speed`
and `rate`, and every field of a shape layer but `shape` and `blend`, its
border's in the place of an effect's. A change to any other field rebuilds the
layer. Setting `x` or `y` to none takes it back to its default, and so does
setting `color`, or a rect's `width` or `height`.

Python sets a live field through `layers`, as `h.layers.title.x = 10` on a
`ghee.open` handle or `ghee.layers.title.karaoke.sung = 12` in a procedural
scene's tick, and [docs/python.md](python.md) holds the rest. The
console sets them through `/set?layer=`, which [docs/http.md](http.md)
describes, a procedural scene through its messages, as
[docs/procedural.md](procedural.md) describes, and Rust through
`Renderer::set_layer`.

A running scene also takes a new layer and builds that layer alone, while
the other layers and passes carry on, and it takes a layer out or moves one
among the layers of its pass the same way. A handle's `draw()` in Python adds
a layer for each call that runs for the first time, takes out the layer of a
call that stops, and moves a layer to the place of its call. A procedural
scene sends message kinds 6, 9 and 10, and Rust calls `Renderer::add_layer`,
as [From Rust](rust.md) shows.

`wave`, `scroll` and `blink` move by a phase that ghee keeps for each layer.
Each frame advances the phase by the effect's `speed` or `rate` times the
frame's time step, so a new speed carries the motion on from where it is.
Under `--timestep` the phase starts at 0 with the scene.

## Text on 1-bit and 16-level panels

A 1-bit panel such as the SSD1306 lights a pixel or leaves it dark. The
kernel's ssd130x driver lights a pixel whose luma, by BT.601's weights, is 128
of 255 or more. A glow, a soft shadow and every other dim pixel go dark.

Careful: give the text layers of an on/off panel such as the SSD1306
`"render": "mono"`, since `auto` never picks it. `mono` lights the pixels that
Pillow lights on a 1-bit image, which luma.oled shows on such a panel:
FreeType hints each glyph for one bit and draws it fully on or off, so every
stroke keeps its pixels at any size. Under `auto` the panel cuts FreeType's
grey letters at 128, and a thin stroke that falls across two pixels can lose
both. `font_family: "Pixel Operator 8"` lights the font's own pixels at 8
pixels per em under its default render. This status line draws as luma.oled
draws it on a 128x32 SSD1306:

```json
{ "name": "cpu", "text": "cpu 42% ping 12ms", "font_size": 10, "x": 1, "y": 1,
  "render": "mono" }
```

On a 16-level panel such as the SSD1322, text below 128 pixels per em under
`auto` shows the levels that luma.oled shows for Pillow's text in the same
font.

Three filters bring grey back. A scene names them in `filters`, or `--filters`
adds them on the command line:

- `dither` lays a 4x4 ordered dither over the frame. Grey turns into a pattern
  of lit dots, so a glow shows as dots around the letters. The pattern holds
  still at any frame rate, and it breaks letters at every size.
- `frc` blinks each pixel on a 16-frame cycle, lit for a share of the frames
  that matches its grey, for up to 16 shades on a 1-bit panel, so letters
  keep soft edges. Sixteen shades need about 200 fps, where an outline font
  reads at every size from 8 pixels per em. At a lower rate, its `coarse`
  and `calm` params trade shades for stillness.
- `frc256` does the same on a 16-level panel such as the SSD1322: each pixel
  blinks between two neighbouring levels, for about 256 shades.

The bus sets the frame rate, and a filter costs little next to it. On a Pi 5,
a glowing clock fitted to the panel ran 300 frames at these rates:

| Panel | Bus | No filter | With the filter |
| --- | --- | --- | --- |
| SSD1306, 128x32 | I2C at 400 kHz | 86 fps | 86 fps with `frc` |
| SSD1322, 256x64 | SPI | 206 fps | 200 fps with `frc256` |

These rates were measured on a Pi 5, and not on a Pi Zero 2 W.

`frc` needs about 200 fps, which an SSD1306 reaches with its I2C bus at 1
MHz, the rate that [`ghee overlay ssd1306`](displays.md#hardware-displays) writes. With
the frame rate held at 106 fps and `calm` at 8, or at 86 fps and `calm` at
10, `frc` keeps text of 12 pixels per em as readable as at 210 fps, roughens
text of 10 and breaks text of 8.

## Render

A text layer's `render` sets how it finds the coverage of each pixel: `auto`,
the default outside the pixel font, `hinted`, `mono`, `mtsdf` or `slug`.

`hinted` draws each glyph as FreeType 2.14.3 draws it for Pillow: hinted at
the layer's size by the font's own instructions, or by FreeType's autohinter
in a font that has none, and rasterized there into a bitmap that the layer
reads one texel a pixel. Each glyph's pen lands on the nearest whole pixel and
its baseline on a whole row, after the glyph's motion, so the layer lights the
pixels that Pillow lights for the same words, font and size. `mono` draws the
glyphs that FreeType hints and draws for Pillow's 1-bit images, each pixel
fully on or off, for an on/off panel, and `auto` never picks it. A hinted
layer's fitted size rounds down to a whole pixel, and its shadow, glow and
stroke draw from an MTSDF atlas, as under `mtsdf`, moved with their letters.

`auto` draws `hinted` below 128 pixels per em and `mtsdf` from 128 up, by the
layer's size when it builds: its `font_size`, or the size that fits its words
then. New words that fit another size keep the render that the layer took.

`mtsdf` reads an atlas that holds an MTSDF, a multi-channel signed distance
field, for each glyph. Each texel of a glyph's cell holds distances to the
outline, 0.5 on the outline and rising inward. Red, green and blue each hold
the distance to some of the outline's edges, and their median keeps the
corners sharp. Alpha holds the true distance, which rounds the corners, and
the shadow, the glow and the stroke read it. A layer reads one bilinear sample
of the cell for each pixel.

The atlas's detail follows the layer's size, rounded up to 32, 48, 64, 96 or
128 texels per em, so layers of near sizes share one atlas. The size is the
layer's `font_size`, or with none the largest size at which a line of words
fits its pass, so shorter words keep their detail. A layer whose effects reach
past its letters stops at a lower step, where its cells span no more texels
than plain text's at 128: a 0.3 em glow stops at 64. The first time a font is
used at a detail, ghee generates the glyphs of the words, then a preload:
ASCII, Latin-1, the curly quotes, the en and em dash, the ellipsis, the bullet
and the euro sign. It generates them on every core, and caches the atlas in
`$XDG_CACHE_HOME/ghee/atlas`, else `~/.cache/ghee/atlas`. Any other character
follows when words first need it. New words show in their first frame when the
atlas already holds all their glyphs. Other new words wait for their glyphs,
and the layer keeps its last words on screen meanwhile. A setup waits for them,
and so does a render with `--timestep`.

`slug` reads no atlas. Each pixel of a glyph's quad casts one ray along x and
one along y, and the curves crossed by the rays give the pixel's coverage.
That coverage is exact at any size. This is Eric Lengyel's
[Slug](https://github.com/EricLengyel/Slug) method, and `_cover` runs it too.
A slug layer has no distance to the letters, so it skips `text_shadow`, `glow`
and `text_stroke` and draws its other effects.

Take `auto` for most words. A hinted layer reads one texel a pixel, as an
MTSDF layer does, and one with no shadow, glow or stroke shows new words in
their first frame, whatever characters they hold, since FreeType draws a glyph
when words first need it. On a Pi Zero 2 W, a 10 px status line in a 128x32
frame sets up in 297 ms hinted and in 399 ms under `mtsdf` with its atlas
cached, 885 ms without, and draws in 0.60 ms and 0.63 ms a frame. Take `mtsdf`
for letters that move by less than a pixel. Take `slug` for exact edges at any
size with no atlas, when the frame has time for a loop over the curves at
every pixel of every glyph. On a Pi Zero 2 W, seven lines of 216 characters at
46 pixels per em, in a 1280x800 frame, draw in 3.3 ms with `mtsdf` and in 22.0
ms with `slug`. The background pass takes 2.1 ms of each.

## Coverage in a shader

Every pass of the scene except a filter pass can read a text layer's words
with `float <name>_cover(vec2 p, float font_size)`. The function returns the
coverage of the words at `p`, from 0 to 1. `p` is in em from the block's top
left, y down like `uv`. `<name>_size` holds the block's width and
height in em. The layer's `x`, `y`, `font_size`, `color`, effects and
`visible` play no part, so a hidden layer gives a shader words that the layer
never draws itself, and ghee builds no atlas for it until it shows:

```json
"layers": [{ "name": "title", "text": "hello", "visible": false }]
```

This shader draws `title` white on black, centred in the frame at 48 pixels
per em:

```glsl
void main() {
  float font_size = 48.0;
  vec2 px = floor(uv * iResolution.xy) + 0.5;
  vec2 p = (px - 0.5 * (iResolution.xy - title_size * font_size)) / font_size;
  gl_FragColor = vec4(vec3(title_cover(p, font_size)), 1.0);
}
```

`p` sets where the words land and how large they draw. `font_size` sets the
antialiasing alone: each edge ramps from 0 to 1 over 1/`font_size` em. Pass
the pixels that an em spans at `p`, and each edge ramps over one pixel. A
larger `font_size` draws harder edges that alias, and a smaller one blurs
them. A shader that warps `p` passes the pixels per em after the warp. Where
the warp stretches the two axes unevenly, it passes the smaller of the two, as
the `lens` example does.

`_cover` runs the slug method, so its edges stay exact at any size. ghee
stores each control point to within 1/4094 of its glyph's size, 0.17 px for a
1 em glyph at 700 pixels per em. An emoji is a bitmap with no outline, so
`_cover` reads it as no coverage.

On VC4, the interpolated `uv` sits up to 0.04 px off the pixel centre. A
shader that needs exact centres, like the one above, samples at
`floor(uv * iResolution.xy) + 0.5`.

## Fonts and emoji

A font is a TrueType or OpenType font with outlines. A text layer with no
`font_family`, or with `font_family: "DejaVu Sans"`, draws from DejaVu Sans,
cut to ASCII, Latin-1, the common punctuation, the euro sign and the trade
mark sign, with its own hinting.
`font_family` takes a `.ttf` or `.otf` file, which resolves as an asset file
does, or the name of a font asset, which several layers can share.

`font_family: "Pixel Operator 8"` draws from Pixel Operator 8, a pixel font
that ghee carries, drawn on a grid of 8 px to the em. Its layer takes `render`
`slug` unless it names one, a size that fits rounds down to a whole multiple
of 8 px, 8 at the least, and the block, each pen and each baseline land on
whole pixels, so at 8 px and its multiples every pixel is fully on or off. A
`font_size` stays as given.

Emoji fall back to Noto Color Emoji. ghee looks for `NotoColorEmoji.ttf`
beside the scene, then in each directory on `$GHEE_ASSETS`, then in
`/usr/share/fonts/truetype/noto/`, where `sudo apt install
fonts-noto-color-emoji` puts it. An emoji that the layer's font lacks draws
from the first file found, and with no file found, it draws the font's
missing-glyph box. A character followed by U+FE0F, the emoji variation
selector, draws its emoji even from a font that holds a plain glyph for it, so
❤️ draws in colour. ghee reads the emoji font once, when words first need it,
and every layer shares it. Those words wait for the read, and the layer keeps
its last words on screen meanwhile.

An emoji draws its bitmap in its own colours, mipmapped, and moves, fades and
hides with its layer's letters. It takes no `text_shadow`, `glow` or
`text_stroke`, and a layer's emoji draw after every layer of its pass.

Lines break at each newline. ghee draws one glyph for each character. The
advances and the font's pair kerning space the glyphs: the pair adjustments
of the `kern` feature in its GPOS table, as HarfBuzz applies them, or, in a
font without that feature, its `kern` table. Since each character draws as
one glyph, ghee cannot lay out scripts that need shaping, such as Arabic and
the Indic scripts. Such words go in as a 2D input drawn by the host. An emoji
sequence draws one code point at a time: a flag shows its two
regional-indicator letters, a sequence joined by U+200D shows each of its
emoji, and variation selectors and the joiner take no room.

A text problem leaves the scene running, and `ghee render` still exits 0. ghee
prints the problem once as an error, under `--quiet` too, and the console
shows it. A layer whose font fails to load draws every character as an error
box, which `_cover` reads as diagonal stripes of coverage. These glyphs each
draw an error box in their place, and the rest of the words draw: a character
whose glyph has neither an outline nor a colour bitmap, a glyph past the curve
limit, and a glyph that finds no room in the atlas. New words that do not fit
leave the last words that did, and a layer whose first words do not fit draws
none.

Render the bundled text examples to PNGs. `karaoke` fills a line as it is
sung, with a glow, and mirrors it in water through a later pass. `lettering`
reads a hidden layer's words through `_cover` on a ladder of sizes. `lens`
magnifies a page through `_cover`:

```
ghee render --demo karaoke -o karaoke.png
ghee render --demo lettering -o lettering.png
ghee render --demo lens -o lens.png
```

`nowplaying` puts an artist over a scrolling title on a 128x32 or a 256x64
panel.
