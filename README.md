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

The file resolves beside the scene, or beside the executable of a
procedural scene, then through the `$GHEE_ASSETS` env variable,
colon-separated directories. A path may be absolute or leave the scene's
directory, except under the console, which reads only the files inside it.

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

A scene draws text on the GPU from a font. A text holds its words in
`value` and names its font, and a layer of a pass paints it. This scene draws
`hello` white on black, centred in the frame at 48 pixels to the em:

```json
{
  "version": 1,
  "texts": { "title": { "value": "hello", "font": "DejaVuSans.ttf", "align": "center" } },
  "passes": [{
    "name": "main",
    "code": "void main() { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
    "layers": [{ "text": "title", "px_per_em": 48 }]
  }]
}
```

A pass draws a text in one of two ways. A [layer](#layers) paints the words
over the pass's buffer, one quad over each glyph, placed and coloured by
GLSL of its own: take a layer for words placed on screen, still or moving
glyph by glyph. [`<name>_cover`](#coverage-in-a-shader) hands the pass's own
shader the coverage of the words at any point: take it when the words feed
the shader's math, a warp or a mask. A layer works on the pixels of its
glyphs alone, and `_cover` searches the font's curves at every pixel that
calls it.

ghee declares a scene string param under the text's name, which holds the
words: `set_param`, `/set`, the console's texts panel and a procedural
scene change them live, and ghee lays them out again with no recompile. A scene
that declares a param under a text's name fails to load.

### Layers

A pass's shader fills its buffer, then its `layers` paint texts over it in
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
  `_` and `-`, unique among the layers of its pass. A layer without one goes by
  its text's name, `title`, a second layer of that text without one by
  `title 2`, a third by `title 3`.
- `vertex` defines `vec2 glyph_place(vec2 em)`, which takes a point of the
  block, in em from its top left with y down, to the pixel of the pass where
  it lands, y down like `v_texCoord * resolution`.
- `px_per_em` takes the place of `vertex`: the block centred in the pass at
  that many pixels per em. A layer takes one of the two.
- `fragment` holds a `void main()` that writes colour premultiplied by alpha.
  Without it, the letters draw white.
- `blend` is `over` by default, `add` or `replace`.
- `pad` is how far past the letters the layer reaches, in em, for a glow or an
  outline, 0 by default. Each quad grows that far past the ink, and
  `glyph_dist` and `glyph_soft` reach at least that far.
- `emoji` sets whether the layer paints the text's colour glyphs. By default
  the last layer of the text in its pass paints them, and the layers under it,
  a glow or a shadow, paint none.

`fragment` reads:

```glsl
float glyph_cover();   // coverage 0..1
float glyph_dist();    // signed distance to the outline in pixels, > 0 inside
float glyph_soft();    // true signed distance in pixels, round at corners
vec2  glyph_em;        // the point in em, as glyph_place takes it
float glyph_index;     // the character's index in the words, newlines counted
float glyph_word;      // the index of its word
float glyph_line;      // the index of its line
vec2  glyph_span;      // its advance, start and end x in em
float glyph_ppem;      // pixels per em at this glyph
```

`vertex` reads `glyph_index`, `glyph_word`, `glyph_line`, `glyph_span` and
`glyph_box`, the glyph's ink box in em as x0, y0, x1, y1, and it sees the
pass's preamble and the scene's common block, as the pass's shader does.

ghee draws each glyph as one quad over its ink box, and `glyph_place` runs in
the vertex shader, four times at each corner of the quad. The first three
calls, at the corner and a sixteenth of an em right of it and below it,
measure how many pixels an em spans there along each axis. The last call
places the corner pushed out past the ink by a little over half a pixel at
that scale, or by `pad` when that is more. The GPU fills the quad flat between
its four corners, so `glyph_place` can move, turn, scale or shear each glyph,
and a bend inside one glyph draws straight. The pixels per em at the corners
set `glyph_ppem` and the antialiasing, an edge one pixel soft.

Each frame, a pass draws in one order. Its shader fills the buffer, or a pass
with `layers` and no GLSL of its own clears it to transparent black. Each
layer then paints its glyphs, in list order, then the colour glyphs of the
layers that paint emoji, then every error box. A layer's colour, premultiplied
by alpha, lands on what the buffer holds by its `blend`: `over` gives
src + dst × (1 − src.a), `add` gives src + dst, and `replace` writes src.
Alpha blends the same way. A first layer of a text thus puts a glow or a shadow
under the fill of every glyph, and a later pass reads the whole buffer, its
layers included, through a channel.

A layer with `emoji` paints the text's colour glyphs itself, and `fragment`
never runs for them. Each glyph shows its bitmap from the font's largest
strike, through mipmaps that keep it smooth at text size, placed by `vertex`
like any other glyph and blended by the layer's `blend`. A shadow layer with
`"emoji": true` paints a copy of its own. The bitmaps decode into an RGBA atlas
when the words first hold them.

### Fills

The text's `fill` sets how its layers find the coverage of each pixel,
`mtsdf` by default or `slug`.

`mtsdf` reads an atlas of the glyphs' MTSDFs, multi-channel signed distance
fields. Each texel of a glyph's cell holds distances to the outline, 0.5 on
it and rising inward: red, green and blue each hold the distance to some of
the outline's edges, so their median keeps the corners sharp, and alpha holds
the true distance, round at the corners. A layer takes one bilinear sample of
the cell a pixel, with no mipmaps, and scales it to pixels by the pixels per
em at the glyph. `glyph_dist()` is the median's distance and `glyph_soft()`
the alpha's, both in pixels and above 0 inside, and `glyph_cover()` is
`clamp(glyph_dist() + 0.5, 0.0, 1.0)`.

The first use of a font generates its ASCII and Latin-1 on every core, 0.6 s
on a Pi Zero 2 W, and caches the atlas in `$XDG_CACHE_HOME/ghee/atlas`, else
`~/.cache/ghee/atlas`. Any other character follows when words first need it,
and the text keeps its last words on screen until every glyph of the new
ones is in. A setup waits for them, and so does a render with `--timestep`.
`atlas` on the text sets the texels per em of a glyph, 32 by default:

```json
"texts": { "title": { "font": "sans", "atlas": { "px_per_em": 48 } } }
```

The field reaches past the outline as far as the largest `pad` of the
text's layers, so a glow fits with no tuning, and 3 texels at least, 0.094 em
at 32 texels per em. Its 8 bits span that whole reach, so a large pad
coarsens the edges of every layer of the text.

`slug` reads no atlas. Each pixel of a glyph's quad casts a ray along x and
one along y through the glyph's curves, and the curves that the rays cross
give its coverage, exact at any size and under any `glyph_place`: Eric
Lengyel's [Slug](https://github.com/EricLengyel/Slug) method, the one that
`_cover` runs. It takes no `atlas` key and gives no distances: a layer that
reads `glyph_dist` or `glyph_soft` fails to compile, and the error names
`"fill": "mtsdf"`.

Take `mtsdf` for most words, and for any layer that reads a distance: it
costs one texture read a pixel. Take `slug` when new words must show in the
frame that they arrive, since the atlas generates a character past Latin-1
at its first use and the words wait for it, and when the frame has time for a
loop over the curves at every pixel of every glyph. On a Pi Zero 2 W, seven
lines of 216 characters at 46 pixels per em, in a 1280x800 frame, draw in
3.2 ms with `mtsdf` and in 22.7 ms with `slug`, 2.1 ms of each the background
pass.

### Coverage in a shader

Every pass of the scene, filter passes excepted, reads the words of a text
through `float <name>_cover(vec2 p, float ppem)`: the coverage 0..1 of the
words at `p`, in em from the block's top left, y down like `v_texCoord`.
`<name>_size` holds the block's width and height in em. This shader draws
`title` white on black, centred in the frame at 48 pixels to the em:

```glsl
void main() {
  float ppem = 48.0;
  vec2 px = floor(v_texCoord * resolution) + 0.5;
  vec2 p = (px - 0.5 * (resolution - title_size * ppem)) / ppem;
  gl_FragColor = vec4(vec3(title_cover(p, ppem)), 1.0);
}
```

`p` sets where the words land and how large, and `ppem` sets the
antialiasing alone: each edge ramps from 0 to 1 over 1/`ppem` em, one pixel
when `ppem` holds the pixels that an em spans at `p`. A larger `ppem` draws
harder edges that alias, and a smaller one blurs them. A shader that warps
`p` passes the pixels per em after the warp, the smaller of the two where the
warp stretches the axes unevenly, as the `lens` example does.

`_cover` runs the slug fill's method, so its edges stay exact at any size.
Each control point moves by at most 1/4094 of its glyph's size, 0.17 px for a
1 em glyph at 700 pixels per em. A colour glyph, a bitmap with no outline,
has no coverage in `_cover`, and only a layer paints it.

On VC4 the interpolated `v_texCoord` sits up to 0.04 px off the pixel centre,
so a shader that needs exact centres, like the one above, samples at
`floor(v_texCoord * resolution) + 0.5`.

### Fonts and words

A font is a TrueType or OpenType font with outlines, or a colour emoji font
with PNG bitmaps in a CBDT or sbix table, such as Noto Color Emoji. `font`
takes a `.ttf` or `.otf` file, which resolves as an asset file does, or the
name of a font asset, which texts share, or a list that mixes both,
`"font": ["sans", "NotoColorEmoji.ttf"]`, and each character draws from the
first font in the list whose cmap holds it, while a character that no font has draws the
first font's missing-glyph box. A character followed by U+FE0F, the emoji
variation selector, draws from the first font with a colour glyph for it, so
❤️ draws in colour even when an earlier font holds a plain heart. Line height
and baseline come from the first font.

Lines break at each newline, and `align` sets each line `left`, `center` or
`right` within the block, `left` by default. Every character draws as one
glyph, spaced by the advances and the pair kerning of its font's `kern`
table, and two glyphs of two fonts take no kerning. Careful: a font that kerns
through GPOS alone draws unkerned. Text that needs shaping goes in as a
host-rasterized 2D input, the way the `marquee` demo draws its words. An emoji
sequence draws one code point at a time: a flag shows its two
regional-indicator letters, a sequence joined by U+200D shows each of its
emoji, and variation selectors and the joiner take no room.

A text problem leaves the scene running: ghee logs it once, and the console
shows it. A font that fails to load drops out of its text's fonts, and a text
with no font left draws every character as an error box, diagonal stripes of
coverage in `_cover`. A character whose glyph has neither an outline nor
a colour bitmap, a glyph past the curve limit, and in a layer a glyph too large
for its atlas each draw an error box in their place, and the rest of the words
draw.
Words that do not fit leave the last words that did, and a scene whose first
words do not fit draws none.

Render the bundled text examples to PNGs: `lettering` reads its words
through `_cover` on a ladder of sizes, `karaoke` sings a line in layers over a
reflection, with its emoji from Noto Color Emoji when `$GHEE_ASSETS` holds it,
and `lens` magnifies a page through `_cover`:

```
ghee render --demo lettering -o lettering.png --frames 1
ghee render --demo karaoke -o karaoke.png --frames 1
ghee render --demo lens -o lens.png --frames 1
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
  `text 't' atlas px_per_em needs a whole number from 1 to 2048, got 4096`,
  `pass 'main' layer 't' pad 100.0 needs an atlas range of 6400 texels at the
  32 px_per_em of text 't', over the cap of 2048`.

A text past one of these keeps the scene running, logs the problem once,
and the console shows it:

- a 1024x2048 data texture for the curves of a text that `_cover` or a layer
  of the slug fill reads, and 4 glyphs over one spot of it. Words past
  either leave the last words that fit, and a scene whose first words do
  not fit draws none:
  `text 't': the words need a 1024x8192 texture, and a VC4 holds 1024x2048
  at most`, `text 't': more than 4 glyphs overlap at one spot`.
- 64 curves in one band of a glyph in that texture. The glyph draws an
  error box:
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
