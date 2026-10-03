# ghee

ghee runs your GLSL headless on the GPU and puts the frames on an SPI/I2C
panel, HDMI, image files, or an H264 stream. Multi-pass, Shadertoy-style,
any Raspberry Pi.

![a four-pass particle scene, procedural text, and the bundled dither filter, looping](https://raw.githubusercontent.com/holofermes/ghee/main/media/hero.webp)

ghee renders with GLES2 through DRM/GBM on a stock Raspberry Pi OS Lite
image, HDMI takes direct scanout. ghee teaches no GLSL,
[The Book of Shaders](https://thebookofshaders.com) and
[Shadertoy](https://www.shadertoy.com) have that covered.

## Install

The Python package carries the whole renderer, panel drivers and console
included. In a uv project, or any venv, on any 64-bit OS:

```
uv add ghee-render                # or, in a venv: pip install ghee-render
```

A Pi with no venv takes the deb from the
[releases page](https://github.com/holofermes/ghee/releases) instead. It
installs the `ghee` command and the Python package in one, for the system
python, and it is the only path for the 32-bit Pis:

```
sudo apt install ./ghee_*.deb
```

Or build everything from source: [Build](#build).

## Where does it run?

| Hardware | OS | Driver | Hardware H264 |
| --- | --- | --- | --- |
| Pi 1, Pi Zero, Pi Zero W (armv6) | Raspberry Pi OS bookworm or trixie | vc4 | yes |
| Pi 2, Pi 3, Pi Zero 2 W (arm64) | Raspberry Pi OS bookworm or trixie | vc4 | yes |
| Pi 4, Pi 400 (arm64) | Raspberry Pi OS bookworm or trixie | v3d + vc4 | yes |
| Pi 5 (arm64) | Raspberry Pi OS bookworm or trixie | v3d + vc4 | ffmpeg/libx264 |
| Desktop Linux (x86_64) | any current distro | Mesa | ffmpeg/libx264 |

ghee runs on every Pi back to the Pi 1, and builds ship for all of them.
Pi 1 through Pi 4 encode H264 on the Broadcom block. The block takes
16384 pixels or more, so ghee scales a smaller frame up by the smallest
whole factor that gets there, nearest neighbour: a 128x32 panel streams at
256x64. The Pi 5 has no encoder, so there and on desktop ghee encodes with
ffmpeg/libx264, install it if needed:

```
sudo apt install ffmpeg
```

## Build

The build needs a Rust toolchain and the GLES/DRM development headers. On
Raspberry Pi OS or Debian:

```
sudo apt install libdrm-dev libgbm-dev libegl1-mesa-dev libgles2-mesa-dev
cargo build --release
```

## Run it

The `--demo` flag provides a set of builtin [scenes](#scenes).

Render one of the JSON scenes to an image sequence of sixty frames:

```
ghee render --demo mix -o frame.####.png --frames 60
```

Render one of the Python [procedural scenes](#procedural-scenes) to a connected HDMI display
and pass custom arguments:

```
ghee render --demo bluesun --display kms -- --emit 480x300
```

Write raw frames to stdout and pipe them through ffmpeg into an mp4,
with `--timestep` pinning engine time to a fixed 1/60 s step per frame:

```
ghee render --demo bars -o - --frames 300 --timestep 0.0166667 |
  ffmpeg -f rawvideo -pix_fmt rgba -s 320x240 -framerate 60 -i - out.mp4
```

Or stream it as a hardware accelerated H264 Annex-B stream and pipe it to mpv:

```
ghee render --demo trail --h264 - --fps 30 | mpv -
```

Or scan it straight out to the HDMI display, details in
[Hardware displays](#hardware-displays):

```
ghee render --demo trail --display kms
```

Or drive an SPI or I2C panel:

```
ghee render --demo trail --display st7789
```

Or serve the web [console](#the-console) to preview and author all the
demo scenes:

```
ghee serve --demo
```

And you can combine outputs: one render can write PNGs, drive a panel and
stream H264 at the same time, and `--serve` the authoring console on top:

```
ghee render --demo bars --display st7789 -o frame.####.png      # panel and a contact sheet
ghee render --demo bars --display st7789 --serve                 # panel, and edit it live
ghee render --demo bars --display kms --serve --host 0.0.0.0     # HDMI, edit from a laptop
```

## Scenes

A scene is a JSON file with an ordered list of passes, each with its own GLSL fragment shader.
This is `bars.scene.json`, one of the bundled examples:

```json
{
  "version": 1,
  "passes": [
    {
      "name": "main",
      "shader": "bars.glsl",
      "arrays": [{ "name": "amplitudes", "size": 16 }],
      "params": {
        "base": { "type": "color", "value": [0.04, 0.04, 0.04] },
        "bar": { "type": "color", "value": [0.1, 0.9, 0.4] }
      }
    }
  ]
}
```

and `bars.glsl` next to it:

```glsl
void main() {
  vec2 uv = v_texCoord;
  int last = amplitudes_size - 1;
  float fi = uv.x * float(last);
  int i0 = int(floor(fi));
  float h = mix(amplitudes_at(i0), amplitudes_at(min(i0 + 1, last)), fract(fi));
  float m = step(uv.y, h);
  gl_FragColor = vec4(mix(base, bar, m), 1.0);
}
```

A pass can also carry its GLSL inline in a `"code"` string, and a scene can
also be a program that emits its own JSON and feeds it data, see
[Procedural scenes](#procedural-scenes).

A scene with a key that ghee does not take fails to load, and the error
names the key, where it sits, and the closest key that ghee takes, or the
object that takes it: `"fill"` on a layer names the layer's text.

Your own scenes can be looked up through the `$GHEE_SCENES` env variable.
Careful: procedural scenes are executables, and ghee runs them, only put
directories you trust there.

## How does it work

The pass model follows Shadertoy. The passes render in order, each one a
fragment shader drawing into its own buffer. A pass reads another pass
through a numbered channel sampler: a channel on an earlier pass sees the
current frame, a channel on the pass itself or a later one sees the
previous frame. ghee sends the last pass to the output.

Each pass renders at its own resolution, and samples another pass through
normalized coordinates. A pass with `"mipmaps": true` builds the mip chain of
its buffer after each frame, and every channel that reads it samples it
trilinearly, so a buffer read far away and minified, such as words in
perspective, stays calm.

A pass buffer holds 8-bit RGBA, and a channel reads the alpha that its pass
wrote. The output is opaque: a frame reads alpha 255, unless
`ghee render --alpha` keeps the output pass's alpha in RGBA frames. ghee
takes the colour of such a frame as premultiplied by that alpha, the way
layers blend it. A raw frame on `-o -` keeps it premultiplied, and a PNG file
holds it straight.

ghee compiles the GLSL of every pass behind a preamble like this one:

```glsl
uniform float time;                 // seconds since start
uniform float DT;                   // seconds the last frame took
uniform float frame;                // frames rendered, wraps at 2^20
uniform vec2  resolution;           // buffer size in pixels
varying vec2  v_texCoord;           // 0..1 across the buffer, y = 0 at the top
// gl_FragCoord maps to the varying, so its y = 0 is at the
// top too, where Shadertoy has y = 0 at the bottom

// declared array, e.g. {"name": "amplitudes", "size": 16}:
const int amplitudes_size;          // = 16
float amplitudes_at(int i);         // one element
float amplitudes_sample(float t);   // interpolated, t in 0..1, clamped edges
// arrays are bound as 1xN textures

// a 2D input, e.g. {"name": "text", "size": [96, 16]}:
float text_sample2(vec2 uv);        // bilinear, normalized uv
// the host feeds w*h floats: a mask, host-rasterized text, an image

// per channel: uniform sampler2D iChannel0..N
// per param:   uniform <type> <name>
// then the scene's common block, if it has one
```

A pass declares a channel by `"buffer"` name:

```json
{ "name": "compose", "shader": "compose.glsl", "channels": [{ "buffer": "compose" }] }
```

That pass samples its own previous frame through `iChannel0`:

```glsl
// compose.glsl: everything drawn so far, fading a little each frame
void main() {
  vec3 prev = texture2D(iChannel0, v_texCoord).rgb;
  gl_FragColor = vec4(prev * 0.95, 1.0);
}
```

A param is a typed uniform declared on the pass, with optional range
metadata:

```json
{ "params": { "gain": { "type": "float", "value": 0.45, "min": 0, "max": 1, "step": 0.01 } } }
```

The preamble turns that into `uniform float gain;`, and the console renders
every param as a live control. The types are `float`, `int`, `bool`,
`color` (a `vec3` in the shader), `vec2/3/4`, and `string`. A string param
never reaches the shader, a procedural scene gets it on its ticks, and
`options` turns its text box into a dropdown. A param declared on the scene
is shared, every pass reads the one value.

A scene declares an asset by file name, and the host decodes it and feeds
it to every pass at setup time:

```json
{ "assets": { "noise": { "file": "noise.raw", "width": 32, "height": 32, "format": "u8" } } }
```

A relative `file` resolves from the directory of the scene file, or of a
procedural scene's executable, then from each directory on `$GHEE_ASSETS`,
colon-separated. A path may be absolute or climb out with `..`, except under
the console, which reads only the files inside its scenes directory.

Every pass samples it as `uniform sampler2D noise` and reads a
`uniform vec2 noise_size` of (width, height). Sampling is `nearest`
unless the asset sets `"sampling": "linear"`.

`.png` and `.jpg` files are supported, as well as `.raw`: bare pixels
that declare `width`, `height` and `format` (`u8` or little-endian
`f32`), and `components` picks 1, 3 or 4 values per pixel, 1 by default.
A `.ttf` or `.otf` file is a font for a [text](#text), and only the text reads it.

Some names belong to ghee. `time`, `DT`, `frame`, `resolution`, `v_texCoord`
and `iChannel0` and up are built in, and every name that ghee writes into a
shader for itself starts with `ghee_`, so no param, array, asset or text takes
one of them. Each declaration makes GLSL names in the passes that it reaches:
a param its own name, an array `<name>_tex` and the readers in the preamble
above, an asset `<name>` and `<name>_size`, a text `<name>_cover` and
`<name>_size`, and layers the `glyph_` names listed under [Layers](#layers), in
every pass of the scene. A scene where two declarations make one name in a
pass fails to load, and the error names both.

## Text

ghee draws text on the GPU from the outlines of a font. A scene declares each
text in `texts`, with its words in `value`. A layer on a pass then paints the
text into the pass's buffer.

This scene holds a text with words and nothing else. ghee draws the words
white on black, in its built-in font, DejaVu Sans, and fits them to the pass:

```json
{
  "version": 1,
  "texts": { "title": { "value": "hello" } },
  "passes": [{ "name": "main", "layers": [{ "text": "title" }] }]
}
```

The words lay out into a block, as wide as the longest line, with one line
height for each line. A layer with no `font_size` and no `vertex` takes the
largest size at which the block fits inside the pass, with a margin of 5% of
the pass's shorter side on every edge, and centres the block. The size follows
the words, so new words fit again.

`font_size` sets the size in pixels per em, as a font size does in CSS. One
line of DejaVu Sans is 1.164 em tall, so at 48 pixels per em a line is 56
pixels tall. This layer draws the block at 48 pixels per em, centred in the
pass:

```json
"layers": [{ "text": "title", "font_size": 48 }]
```

`at` puts the top left of the block at a pixel of the pass, `[x, y]` with y
down. With `font_size`, the block draws at that size. Without it, the block
takes the largest size that fits right of and below that pixel, less the
margin at the right and bottom edges. This status screen puts two lines in the
top left corner of a 128x32 panel:

```json
"texts": { "cpu": { "value": "cpu 12%" }, "temp": { "value": "temp 41°C" } },
"passes": [{ "name": "main", "layers": [
  { "text": "cpu", "font_size": 12, "at": [0, 0] },
  { "text": "temp", "font_size": 12, "at": [0, 16] }
]}]
```

A glow is a second layer of the text, under the words. `pad` lets the glow
layer reach 0.3 em past the letters, and its `fragment` turns the distance
from the letters into light. `glyph_round_dist()` is that distance in pixels,
above 0 inside the letters, and `glyph_font_size` holds the pixels per em, so
`far` is the distance outside the letters in em. Both layers fit the block to
the same size, so the glow stays under its words:

```json
"layers": [
  { "text": "title", "name": "glow", "pad": 0.3,
    "fragment": "void main() { float far = -glyph_round_dist() / glyph_font_size; gl_FragColor = vec4(0.3, 0.6, 1.0, 1.0) * exp(-10.0 * max(far, 0.0)); }" },
  { "text": "title" }
]
```

[Recipes](#recipes) holds more effects: an outline, a drop shadow, a
per-letter wave and a marquee.

`font` names the text's font, a `.ttf` or `.otf` file beside the scene, as in
`"font": "DejaVuSans.ttf"`. [Fonts and words](#fonts-and-words) holds the
rest: font assets, fallback lists, colour emoji and the limits of the layout.

The words change live. `set_param`, `/set`, the console's texts panel and a
procedural scene set new words, and ghee lays them out again with no
recompile. For this, ghee declares a scene string param under each text's
name. A scene that declares its own param under a text's name fails to load.

A pass draws a text in one of two ways. A [layer](#layers) paints the words
over the pass's buffer, as one quad over each glyph, and its own GLSL places
and colours each glyph. Take a layer for words that sit on the screen, still
or moving glyph by glyph. [`<name>_cover`](#coverage-in-a-shader) gives the
pass's own shader the coverage of the words at any point. Take it when the
words feed the shader's math, such as a warp or a mask. A layer shades the
pixels of its glyphs alone, and `_cover` searches the font's curves at every
pixel that calls it.

### Layers

A pass's shader fills its buffer, then its `layers` paint texts over it, in
order. This layer puts `title` in amber at 48 pixels per em, centred in the
pass:

```json
"layers": [{
  "text": "title",
  "vertex": "vec2 glyph_place(vec2 em) { return (resolution - title_size * 48.0) * 0.5 + em * 48.0; }",
  "fragment": "void main() { float c = glyph_cover(); gl_FragColor = vec4(vec3(1.0, 0.7, 0.2) * c, c); }"
}]
```

- `text` names a scene text.
- `name` names the layer in errors and in the console: ASCII letters, digits,
  `_` and `-`, unique among the layers of its pass. A layer without a name
  goes by its text's name, `title`. A second layer of that text without a name
  goes by `title 2`, and a third by `title 3`.
- `font_size` sets the pixels per em, and `at` sets the pass pixel of the
  block's top left. Without `at` the block sits centred, and without
  `font_size` it takes the largest size that fits, as [Text](#text) shows.
- `vertex` places the glyphs in GLSL, and a layer with `vertex` takes neither
  `font_size` nor `at`. It defines `vec2 glyph_place(vec2 em)`. The function
  takes a point of the block, in em from its top left with y down, and returns
  the pass pixel where that point lands, y down like
  `v_texCoord * resolution`.
- `fragment` holds a `void main()` that writes the colour premultiplied by
  alpha. Without it, the letters draw white.
- `blend` is `over` by default, or `add` or `replace`.
- `pad` sets how far the layer reaches past the letters, in em, for a glow or
  an outline. It is 0 by default. Each glyph's quad grows that far past its
  ink, and `glyph_dist` and `glyph_round_dist` reach at least that far.
- `fade` sets a band at the outer edge of the pad, in em, over which the layer
  fades to 0. It is a quarter of `pad` by default and at most `pad`, and 0
  turns the fade off. [Recipes](#recipes) shows why a glow needs it. A layer
  of the slug fill takes no `fade`.
- `emoji` sets whether the layer paints the text's colour glyphs. By default
  the last layer of the text in its pass paints them, and the layers under it,
  such as a glow or a shadow, paint none.

`fragment` reads:

```glsl
float glyph_cover();       // coverage 0..1
float glyph_dist();        // signed distance to the outline in pixels, > 0 inside
float glyph_round_dist();  // true signed distance in pixels, round at corners
float glyph_fade;          // 1 up to pad - fade em from the letters, 0 at pad em
vec2  glyph_em;            // the point in em, as glyph_place takes it
float glyph_index;         // the character's index in the words, newlines counted
float glyph_word;          // the index of its word
float glyph_line;          // the index of its line
vec2  glyph_span;          // its advance, start and end x in em
float glyph_font_size;     // pixels per em at this glyph
```

`vertex` reads `glyph_index`, `glyph_word`, `glyph_line`, `glyph_span` and
`glyph_box`, the glyph's ink box in em as x0, y0, x1, y1. Both `vertex` and
`fragment` see the pass's preamble and the scene's common block, as the pass's
shader does.

ghee draws each glyph as one quad over its ink box. The vertex shader calls
`glyph_place` four times at each corner of the quad. The first three calls, at
the corner and a sixteenth of an em to its right and below it, measure how
many pixels an em spans there along each axis. The fourth call places the
corner, pushed out past the ink by a little over half a pixel at that scale,
or by `pad` when `pad` is larger. The GPU fills the quad flat between its four
corners, so `glyph_place` can move, turn, scale or shear each glyph, and a
bend inside one glyph draws straight. The pixels per em at the corners set
`glyph_font_size` and the antialiasing, which softens an edge over one pixel.

Each frame, a pass draws in this order. Its shader fills the buffer, or a pass
with `layers` and no GLSL of its own clears the buffer to transparent black.
Each layer then paints its glyphs, in list order. The layers that paint emoji
paint the colour glyphs after that, and every error box comes last. A layer's
colour, premultiplied by alpha, lands on the buffer by its `blend`: `over`
gives src + dst × (1 − src.a), `add` gives src + dst, and `replace` writes
src. Alpha blends the same way. Careful: `replace` writes the whole quad of
each glyph, faded or not. The first layer of a text thus puts a glow or a
shadow under the fill of every glyph. A later pass reads the whole buffer,
layers included, through a channel.

A layer that paints emoji draws each colour glyph from its bitmap, and the
layer's `fragment` never runs for them. The bitmap comes from the font's
largest strike, and mipmaps keep it smooth at text size. `vertex` places it
like any other glyph, and the layer's `blend` blends it. A shadow layer with
`"emoji": true` paints a copy of its own. ghee decodes the bitmaps into an
RGBA atlas when the words first hold them.

### Recipes

Each recipe draws a text named `title` and fits any pass, from a 128x32 panel
to HDMI. Paste its layers into a pass with no GLSL of its own, and the words
draw over black. The drop shadow brings its own pass and `common` block.

The glow in [Text](#text) needs its `pad`. Each glyph's quad reaches `pad` em
past the letters, and the glow lives in that room. A glow that has not reached
0 at the quad's edge shows the quad as a faint box. So ghee fades a padded
`mtsdf` layer to 0 over the last quarter of its pad, by the distance from the
letters, and the glow ends along the letters' shape. `fade` sets that band in
em, and `"fade": 0` turns the fade off. `glyph_fade` holds the fade's factor
in `fragment`, and ghee multiplies the layer's colour and alpha by it after
`main` returns. In a layer that does not fade, `glyph_fade` holds 1.

An outline is a layer under the words that grows the letters by the outline's
width, here 0.06 em. The second layer fills the letters black, which leaves
hollow letters in an amber outline. Fill it with a colour for outlined
letters. The outline's `pad` reaches well past its edge, so the fade leaves
the edge alone:

```json
"layers": [
  { "text": "title", "name": "outline", "pad": 0.15,
    "fragment": "void main() { float c = clamp(glyph_round_dist() + 0.06 * glyph_font_size + 0.5, 0.0, 1.0); gl_FragColor = vec4(1.0, 0.6, 0.1, 1.0) * c; }" },
  { "text": "title", "fragment": "void main() { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0) * glyph_cover(); }" }
]
```

A drop shadow is a copy of the words moved down and right, and moving glyphs
takes `vertex`. This scene's `common` block holds `place`, which sizes the
block to 0.8 of the pass and centres it, and both layers place their glyphs
with it. The shadow layer moves its glyphs 0.06 em right and 0.08 em down, and
darkens the blue background softly up to 0.15 em past the letters:

```json
"common": { "code": "vec2 place(vec2 em) { float px = 0.8 * min(resolution.x / title_size.x, resolution.y / title_size.y); return 0.5 * (resolution - title_size * px) + em * px; }" },
"passes": [{
  "name": "main",
  "code": "void main() { gl_FragColor = vec4(0.25, 0.45, 0.75, 1.0); }",
  "layers": [
    { "text": "title", "name": "shadow", "pad": 0.2,
      "vertex": "vec2 glyph_place(vec2 em) { return place(em + vec2(0.06, 0.08)); }",
      "fragment": "void main() { float far = -glyph_round_dist() / glyph_font_size; gl_FragColor = vec4(0.0, 0.0, 0.0, 0.7) * (1.0 - smoothstep(-0.05, 0.15, far)); }" },
    { "text": "title", "vertex": "vec2 glyph_place(vec2 em) { return place(em); }" }
  ]
}]
```

A per-letter wave gives each glyph its own phase. `glyph_index` counts the
characters, so this layer bobs each letter up and down by 0.1 em, a little
behind the one before it. It takes `place` from the drop shadow's `common`
block:

```json
"layers": [
  { "text": "title",
    "vertex": "vec2 glyph_place(vec2 em) { return place(em + vec2(0.0, 0.1 * sin(4.0 * time - 0.6 * glyph_index))); }" }
]
```

A marquee scrolls the words from right to left, here at 2 em a second, with
the block 0.8 of the pass tall. The words start over once they leave the pass.
`floor` moves the block by whole pixels, so on a 1-bit panel each letter keeps
its pixels as it scrolls:

```json
"layers": [
  { "text": "title",
    "vertex": "vec2 glyph_place(vec2 em) { float px = 0.8 * resolution.y / title_size.y; float trip = resolution.x + title_size.x * px; float x = floor(resolution.x - mod(time * 2.0 * px, trip)); return vec2(x, 0.5 * (resolution.y - title_size.y * px)) + em * px; }" }
]
```

`ghee/examples/nowplaying.py` puts an artist over a scrolling title on a
128x32 or a 256x64 panel.

### Text on 1-bit and 16-level panels

A 1-bit panel such as the SSD1306 lights a pixel or leaves it dark. ghee's
`ssd1306` driver lights a pixel whose grey is above 128 of 255. A letter keeps
its shape, since a pixel at its edge lights when the pixel's centre lies
inside the outline. A glow, a soft shadow and every other dim pixel go dark. A
stroke thinner than a pixel can go dark too, such as the bar of a small e at
11 pixels per em. This fill grows the letters by a tenth of a pixel and keeps
such strokes:

```json
{ "text": "title", "fragment": "void main() { gl_FragColor = vec4(clamp(glyph_dist() + 0.6, 0.0, 1.0)); }" }
```

Three filters bring grey back. A scene names them in `filters`, or `--filters`
adds them on the command line:

- `dither` lays a 4x4 ordered dither over the frame. Grey turns into a pattern
  of lit dots, so a glow shows as dots around the letters. The pattern holds
  still at any frame rate.
- `frc` blinks each pixel on a 16-frame cycle, lit for a share of the frames
  that matches its grey, for up to 16 shades on a 1-bit panel. Sixteen shades
  need about 200 fps. At a lower rate, its `coarse` and `calm` params trade
  shades for stillness.
- `frc256` does the same on a 16-level panel such as the SSD1322: each pixel
  blinks between two neighbouring levels, for about 256 shades.

The bus sets the frame rate, and a filter costs little next to it. On a Pi 5,
a glowing clock fitted to the panel ran 300 frames at these rates:

| Panel | Bus | No filter | With the filter |
| --- | --- | --- | --- |
| SSD1306, 128x32 | I2C | 86 fps | 86 fps with `frc` |
| SSD1322, 256x64 | SPI | 206 fps | 200 fps with `frc256` |

These rates were measured on a Pi 5, and not on a Pi Zero 2 W.

### Fills

A text's `fill` sets how its layers find the coverage of each pixel: `mtsdf`,
the default, or `slug`.

`mtsdf` reads an atlas that holds an MTSDF, a multi-channel signed distance
field, for each glyph. Each texel of a glyph's cell holds distances to the
outline, 0.5 on the outline and rising inward. Red, green and blue each hold
the distance to some of the outline's edges, and their median keeps the
corners sharp. Alpha holds the true distance, which rounds the corners. A
layer reads one bilinear sample of the cell for each pixel, with no mipmaps,
and scales it to pixels by the pixels per em at the glyph. `glyph_dist()`
returns the median's distance and `glyph_round_dist()` the alpha's, both in
pixels and above 0 inside. `glyph_cover()` returns
`clamp(glyph_dist() + 0.5, 0.0, 1.0)`.

The first time a font is used, ghee generates the glyphs of the words and a
preload: ASCII, Latin-1, the curly quotes, the en and em dash, the ellipsis,
the bullet and the euro sign. It generates them on every core, in 0.6 s on a
Pi Zero 2 W, and caches the atlas in `$XDG_CACHE_HOME/ghee/atlas`, else
`~/.cache/ghee/atlas`. Any other character follows when words first need it.
New words show in their first frame when the atlas already holds all their
glyphs. Other new words wait for their glyphs, and the text keeps its last
words on screen meanwhile. A setup waits for them, and so does a render with
`--timestep`. `atlas` on the text sets the texels per em of a glyph's cell, 32
by default:

```json
"texts": { "title": { "font": "sans", "atlas": { "font_size": 48 } } }
```

The field reaches past the outline as far as the largest `pad` among the
text's layers, so a glow fits with no tuning. It reaches 3 texels at least,
0.094 em at 32 texels per em. Its 8 bits span the whole reach, so a large
`pad` coarsens the edges of every layer of the text.

`slug` reads no atlas. Each pixel of a glyph's quad casts one ray along x and
one along y, and the curves crossed by the rays give the pixel's coverage.
That coverage is exact at any size and under any `glyph_place`. This is Eric
Lengyel's [Slug](https://github.com/EricLengyel/Slug) method, and `_cover`
runs it too. A slug text takes no `atlas` key, and its layers get no
distances: a layer that reads `glyph_dist`, `glyph_round_dist` or `glyph_fade`
fails to compile, and the error names `"fill": "mtsdf"`.

Take `mtsdf` for most words, and for any layer that reads a distance. It costs
one texture read a pixel. Take `slug` when new words must show in their first
frame, whatever characters they hold, and when the frame has time for a loop
over the curves at every pixel of every glyph. On a Pi Zero 2 W, seven lines
of 216 characters at 46 pixels per em, in a 1280x800 frame, draw in 3.2 ms
with `mtsdf` and in 22.7 ms with `slug`. The background pass takes 2.1 ms of
each.

### Coverage in a shader

Every pass of the scene except a filter pass can read a text's words with
`float <name>_cover(vec2 p, float font_size)`. The function returns the
coverage of the words at `p`, from 0 to 1. `p` is in em from the block's top
left, y down like `v_texCoord`. `<name>_size` holds the block's width and
height in em. This shader draws `title` white on black, centred in the frame
at 48 pixels per em:

```glsl
void main() {
  float font_size = 48.0;
  vec2 px = floor(v_texCoord * resolution) + 0.5;
  vec2 p = (px - 0.5 * (resolution - title_size * font_size)) / font_size;
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

`_cover` runs the slug fill's method, so its edges stay exact at any size.
ghee stores each control point to within 1/4094 of its glyph's size, 0.17 px
for a 1 em glyph at 700 pixels per em. A colour glyph is a bitmap with no
outline, so `_cover` reads it as no coverage, and only a layer paints it.

On VC4, the interpolated `v_texCoord` sits up to 0.04 px off the pixel centre.
A shader that needs exact centres, like the one above, samples at
`floor(v_texCoord * resolution) + 0.5`.

### Fonts and words

A font is a TrueType or OpenType font with outlines, or a colour emoji font
with PNG bitmaps in a CBDT or sbix table, such as Noto Color Emoji. A text
with no `font` draws from the built-in font: DejaVu Sans, cut to ASCII,
Latin-1, the common punctuation, the euro sign and the trade mark sign. `font`
takes a `.ttf` or `.otf` file, which resolves as an asset file does, or the
name of a font asset, which several texts can share. It also takes a list that
mixes both, `"font": ["sans", "NotoColorEmoji.ttf"]`. Each character then
draws from the first font in the list whose cmap holds it, and a character
that no font holds draws the first font's missing-glyph box. A character
followed by U+FE0F, the emoji variation selector, draws from the first font
with a colour glyph for it, so ❤️ draws in colour even when an earlier font
holds a plain heart. Line height and baseline come from the first font.

Lines break at each newline. `align` puts each line `left`, `center` or
`right` within the block, `left` by default. ghee draws one glyph for each
character. The advances and the pair kerning of the font's `kern` table space
the glyphs, and two glyphs from two fonts take no kerning between them.
Careful: a font that kerns through GPOS alone draws unkerned. Since each
character draws as one glyph, ghee cannot lay out scripts that need shaping,
such as Arabic and the Indic scripts. Such text goes in as a host-rasterized
2D input, the way the `marquee` demo draws its words. An emoji sequence draws
one code point at a time: a flag shows its two regional-indicator letters, a
sequence joined by U+200D shows each of its emoji, and variation selectors and
the joiner take no room.

A text problem leaves the scene running, and `ghee render` still exits 0. ghee
prints the problem once as an error, under `--quiet` too, and the console
shows it. A font that fails to load drops out of its text's fonts. A text with
no font left draws every character as an error box, which `_cover` reads as
diagonal stripes of coverage. These glyphs each draw an error box in their
place, and the rest of the words draw: a character whose glyph has neither an
outline nor a colour bitmap, a glyph past the curve limit, and in a layer, a
glyph too large for its atlas. New words that do not fit leave the last words
that did, and a scene whose first words do not fit draws none.

Render the bundled text examples to PNGs. `lettering` reads its words through
`_cover` on a ladder of sizes. `karaoke` sings a line in layers over a
reflection, with its emoji from Noto Color Emoji when `$GHEE_ASSETS` holds it.
`lens` magnifies a page through `_cover`:

```
ghee render --demo lettering -o lettering.png
ghee render --demo karaoke -o karaoke.png
ghee render --demo lens -o lens.png
```

## Procedural scenes

A procedural scene is an executable that supplies its own scene and data,
in any language, ghee starts it and talks to it over stdin and stdout.
`docs/procedural.md` holds the wire contract.

`pip install ghee-render` is that contract for Python: `import ghee` gives
`Scene`/`Pass`/`Param` to build the scene, `@ghee.feed` and `@ghee.tick` to
supply the data, and `ghee.run` for the loop. A PEP 723 scene declares
`ghee-render` inline under `#!/usr/bin/env -S uv run --script`.

The python shim lets you also run the python file to render directly:
`./scene.py --display --frames 300`.

`ghee/examples/` holds the bundled demos.

## The console

`ghee serve <scenes_dir>` starts the console on `127.0.0.1:8090`: a
hardware accelerated live preview of the running scene, the GLSL of each
pass and each layer editable with compile errors inline, each text's words,
font and alignment, param sliders, pass muting and a filter picker, and
edits hot-reload into the running renderer.
See [docs/http.md](docs/http.md) for its API.

![the console editing a live scene](https://raw.githubusercontent.com/holofermes/ghee/main/media/console.png)

Careful: the console has no auth and compiles any GLSL posted to it, so it
binds loopback by default. `--host 0.0.0.0` opens it to your network, do
that only on one you trust.

## Filters

A filter is a named single-pass post process, appended after the scene's
own passes. A scene can declare them:

```json
{ "passes": [ ... ], "filters": ["dither"] }
```

or the CLI can add them in order, after whatever the scene declares:

```
ghee render --demo bars --filters dither --display ssd1306
```

You can author your own `<name>.filter.json` files and add their
directories to `$GHEE_FILTERS`, colon-separated. A filter declares
assets the way a scene does, and the file resolves beside the filter.

## ISF import

`ghee isf import` converts an
[Interactive Shader Format](https://isf.video) shader into a ghee scene or
filter.

```
ghee isf import shader.fs                        # -> shader.scene.json
ghee isf import https://host/shader.fs -o s.json
ghee isf import shader.fs --as-filter            # -> shader.filter.json
ghee isf import shader.fs --size 640x480
```

ghee refuses three kinds of ISF:

- `FLOAT` pass buffers. ghee's pass buffers are 8-bit RGBA.
- `IMPORTED` external images. The import converts the shader file alone.
- a vertex shader, or a body reading a varying. ghee's vertex shader is
  fixed.

## Hardware displays

```
ghee displays                      # cards, drivers, their options, what works here
ghee render scene.json --display kms
ghee render scene.json --display st7789
ghee serve scenes/ --display st7789   # the console, driving the panel
ghee overlay st7789                   # take the kernel route instead
```

`kms` is any connected KMS display: HDMI, or an SPI/I2C panel with a kernel
driver. Use `ghee displays` to list the cards, and `--dri` to pick one.

Take the kernel route for a panel where you can: on a Pi Zero W at 240x240 the
`hero` demo renders 44% more fps than the userspace `st7789` driver does, at
40% less CPU. The kernel already drives a range of SPI and I2C panels set up
with a device-tree overlay. Use `ghee overlay` to generate what the panel
needs and the instructions to install it, the panel belongs to the kernel
driver from then on:

```
$ ghee overlay st7789 --display-opts width=240,height=240,madctl=0x00,backlight=26
dtoverlay=mipi-dbi-spi,spi0-0,speed=62500000
dtparam=width=240,height=240
dtparam=dc-gpio=25,backlight-gpio=26
dtparam=write-only
wrote panel.bin. Copy it to /lib/firmware/panel.bin, add the lines above to
/boot/firmware/config.txt, and reboot
```

Yes, two userspace panel drivers: `st7789` (a 135x240 SPI LCD) and
`ssd1306` (a 128x32 I2C 1-bit OLED). They need no overlay and no reboot,
can be hot swapped 😬, and are configurable at runtime. Extend either driver to
your own panel with `--display-opts`, which sets the driver's own options,
for example the Adafruit 1.3" TFT Bonnet, a 240x240 ST7789:

```
ghee render --demo bars --display st7789 \
  --display-opts width=240,height=240,x_offset=0,y_offset=0,madctl=0x00,backlight=26
```

## From Rust

ghee is a library and a command. A render is three calls: load or build
a scene, set it up, then render frames.

```rust
use ghee::engine::{Feed, Renderer};
use ghee::scene::{ArraySize, Scene};

let scene = Scene::load("bars.scene.json")?;
// width, height, components, DRI device (None probes), scanout, rgb565
let mut r = Renderer::new(320, 240, 4, None, false, false)?;
assert_eq!(r.setup(&scene)?, "");

let frame = r.render(Some(&mut |_name, size| match size {
    ArraySize::Linear(n) => Feed::Data(vec![0.5; n as usize]),
    ArraySize::Grid(w, h) => Feed::Data(vec![0.5; (w * h) as usize]),
}))?;
```

`render` takes one closure, which answers with the data for a named array.
The engine is thread-affine: build and use a Renderer on one thread.

## Limits

A scene holds any number of assets, texts and layers, and names of any
length. A scene that passes one of these caps fails to load or to set up,
and the error reads as shown:

- 16 textures in one pass, fewer on a GPU with fewer texture units: the
  pass's channels and arrays, each asset that its GLSL samples, and the
  curves of each text whose `_cover` it reads. A layer counts the same, plus
  the atlas of its text, or the curves under `"fill": "slug"`.
  `pass 'main' needs 17 texture units, a pass binds at most 16`, or
  `pass 'main' layer 'title' needs 17 texture units, a pass binds at most 16`.
- 16 passes, filter passes included: `17 passes, over the cap of 16`.
- 16384 on a side and 16777216 pixels for a sized pass, an array and a
  `.raw` asset: `pass 'a' size needs two positive ints [w, h] up to 16384,
  got [20000, 8]`, `pass 'a' is 8192x4096, over the 16777216 pixel cap`.
  The GPU's own texture size binds tighter, 2048 on a VC4.
- 2048 texels per em in a text's MTSDF atlas, and a field that reaches 2048
  texels at most for the largest `pad` of its layers:
  `text 't' atlas font_size needs a whole number from 1 to 2048, got 4096`,
  `pass 'main' layer 't' pad 100.0 needs an atlas range of 6400 texels at the
  atlas font_size 32 of text 't', over the cap of 2048`.

A text past one of these keeps the scene running, logs the problem once,
and the console shows it:

- a 1024x2048 data texture for the curves of a text that `_cover` or a layer
  of the slug fill reads, about 24000 characters of DejaVu Sans in lines of
  60, and 4 glyphs whose boxes overlap one spot. Words past either leave the
  last words that fit, and a scene whose first words do not fit draws none:
  `text 't': the words need a 1024x8192 texture, and a VC4 holds 1024x2048
  at most`, `text 't': more than 4 glyphs overlap at one spot`.
- 64 curves in one strip of a glyph: that texture cuts each glyph into as
  many as 64 strips across and 64 down, fewer when the words crowd it. The
  glyph draws an error box:
  `text 't': 'X' has more than 64 curves in one band`.
- a 2048x2048 atlas for the MTSDF fields of a font, and one for the colour
  glyphs of a text, room for 132 emoji of Noto Color Emoji. A glyph past
  either draws an error box: `text 't': 'A' has no field in the atlas: the
  2048x2048 atlas is full`.

The wire of a [procedural scene](#procedural-scenes) names an array or a
param in 255 bytes at most, and the Python shim raises ValueError past it.

## Known limitations

- GLES2 only: no compute, geometry or tessellation shaders. VC4 also has
  no derivatives (`fwidth`) and no multisampled FBOs.
- Everything except hardware H264 and KMS scanout reads the frame back
  with a blocking glReadPixels (GLES2 has no async readback), so bigger
  outputs cost more CPU.
- `--display kms` needs its own card: if a compositor holds it, ghee
  cannot take DRM master.
- The bundled panel drivers cover two controllers (`st7789`, `ssd1306`).

## FAQ

**`ghee render` says there is no GL device.** Point the engine at the right
node with `--dri` or `$GHEE_DRI`, and add your user to the `video` group
(plus `render` on a desktop). `ghee displays` reports what works on the
host.

**`--h264` or the console stream says there is no ffmpeg on PATH.** That
board has no Broadcom encoder, so ghee encodes with ffmpeg and libx264:
`sudo apt install ffmpeg`.

**`--display kms` cannot take DRM master.** Something else owns the card,
almost always a desktop session.

**The console preview falls back to MJPEG in a browser that has
WebCodecs.** H264 playback needs a secure context, `127.0.0.1` is one, a
plain-HTTP LAN address is not.

## Tests

```
cargo test --release
cargo run --release -- golden --manifest tools/fixtures/golden/manifest.json
uv run --no-cache --with python/target/wheels/ghee_render-*.whl --with pytest \
    pytest python/tests
```

The render tests need a GPU node, and they skip cleanly without one. The
golden check renders every scene listed in the manifest and compares the
sha256 of each frame against the pinned reference in `tools/fixtures/golden/`.

## License

MIT. The text shader ports Eric Lengyel's Slug, also MIT, and
[NOTICE](NOTICE) carries its copyright notice.

![](https://fabiopiparo.com/pixel.gif?site=ghee)
