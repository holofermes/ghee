# Scenes

ghee teaches no GLSL: [The Book of Shaders](https://thebookofshaders.com) and
[Shadertoy](https://www.shadertoy.com) have that covered.

A scene is a JSON file that holds an ordered list of passes, each with its own
GLSL fragment shader. This is `bars.scene.json`, one of the bundled examples:

```json
{
  "version": 1,
  "passes": [
    {
      "name": "main",
      "fragment": { "file": "bars.glsl" },
      "arrays": [{ "name": "amplitudes", "size": 16 }],
      "params": {
        "base": { "type": "color", "value": [0.04, 0.04, 0.04] },
        "bar": { "type": "color", "value": [0.1, 0.9, 0.4] }
      }
    }
  ]
}
```

and `bars.glsl` beside it:

```glsl
void main() {
  int last = amplitudes_count - 1;
  float fi = uv.x * float(last);
  int i0 = int(floor(fi));
  float h = mix(amplitudes_at(i0), amplitudes_at(min(i0 + 1, last)), fract(fi));
  float m = step(uv.y, h);
  gl_FragColor = vec4(mix(base, bar, m), 1.0);
}
```

A pass's `fragment` holds its GLSL as text, or names a file beside the scene
as `{ "file": "bars.glsl" }`. A scene can also be a program that writes its
own JSON and feeds it data: see [Procedural scenes](#procedural-scenes).

ghee refuses a scene with an unknown key. The error names the key and where it
sits, and it suggests the closest known key, or the object that takes the key:
`"font"` on a layer points to `font_family`, and `"fragment"` on a layer
points to its pass. A key one edit from a key of its own object points there
first, so a scene's `"filter"` points to `filters`.

When GLSL fails to compile, `ghee render` stops with an error. The error opens
with the pass and the line of its fragment that holds the mistake, as in
`pass 'main', line 2 of its fragment:`, and the GL's own message follows. A
mistake in the common block reads `line 2 of the common block`.

ghee also looks a scene name up in the directories on `$GHEE_SCENES`, a
colon-separated list. Careful: procedural scenes are executables, and ghee
runs them. Only put directories you trust on `$GHEE_SCENES`.

## How does it work

The pass model follows Shadertoy. The passes render in order, and each pass is
a fragment shader that draws into its own buffer. A pass reads another pass
through a numbered channel sampler. A channel that reads an earlier pass sees
that pass's buffer from this frame. A channel that reads the pass itself, or a
later pass, sees the buffer from the previous frame. ghee sends the last pass
to the output, at the output's size.

A pass renders at the frame's size, at its own `"size": [w, h]` in pixels,
or at `"scale"`, a share of the frame, one number for both sides or `[x, y]`.
A pass takes one of the two. The frame is the render size turned by
`--rotate`, and a scaled side rounds to the nearest pixel, 1 at least, so
this pass blurs at half the frame's size on every screen:

```json
{ "name": "blur", "scale": 0.5, "fragment": { "file": "blur.glsl" } }
```

The last pass ignores a size or a scale, with a warning. A pass samples other
passes in normalized coordinates, and `"sampling"` on a pass picks how every
channel that reads it samples it: `"nearest"` hands each the nearest pixel
whole, `"linear"`, the default, blends the four around the point, and
`"trilinear"` builds the mip chain of the pass's buffer after each frame and
blends across it. A buffer drawn small and far away, such as words in
perspective, then stays calm.

A pass with `"visible": false` is hidden. It draws a copy of the pass before
it, or black as the first pass, its layers draw nothing, and the frame is the
last visible pass, which keeps its size or scale when hidden passes follow it.
A hidden pass keeps its params, arrays and layers, so a set and a feed still
land, and a scene that hides every pass fails to load.

A pass buffer holds 8-bit RGBA, and a channel reads the alpha that its pass
wrote. Every output frame is opaque, with alpha 255, unless
`ghee render --alpha` keeps the output pass's alpha in RGBA frames. ghee then
treats the colour as premultiplied by that alpha, as layers blend it. A raw
frame on `-o -` keeps the colour premultiplied, and a PNG file stores it
straight.

ghee puts a preamble like this one in front of the GLSL of every pass:

```glsl
uniform float iTime;                // seconds since start
uniform float iTimeDelta;           // seconds the last frame took
uniform int   iFrame;               // frame number, 0 first, wraps at 2^20
uniform vec3  iResolution;          // buffer size in pixels, z = 1.0
uniform vec3  iOutputResolution;    // frame size in pixels, z = 1.0
uniform vec3  iChannelResolution[4]; // each channel's buffer size, z = 1.0
varying vec2  uv;                   // 0..1 across the buffer, y = 0 at the top
// gl_FragCoord maps to the varying, so its y = 0 is at the
// top too, where Shadertoy has y = 0 at the bottom

// declared array, e.g. {"name": "amplitudes", "size": 16}:
const int amplitudes_count;         // = 16
float amplitudes_at(int i);         // one element
float amplitudes_sample(float t);   // interpolated, t in 0..1, clamped edges
// arrays are bound as 1xN textures

// a 2D input, e.g. {"name": "mask", "size": [96, 16]}:
float mask_sample2(vec2 uv);        // bilinear, normalized uv
// the host feeds w*h floats: a mask, host-rasterized text, an image

// per channel: uniform sampler2D iChannel0..N
// per param:   uniform <type> <name>
// then the scene's common block, if it has one
```

The scene's `common` block holds GLSL that ghee puts after the preamble of
every pass except a filter pass. It takes the two forms of a `fragment`:
`"common": "..."`, or `"common": { "file": "common.glsl" }`.

A pass declares each channel by the name of the buffer that it reads:

```json
{ "name": "compose", "fragment": { "file": "compose.glsl" }, "channels": [{ "buffer": "compose" }] }
```

That pass samples its own previous frame through `iChannel0`:

```glsl
// compose.glsl: everything drawn so far, fading a little each frame
void main() {
  vec3 prev = texture2D(iChannel0, uv).rgb;
  gl_FragColor = vec4(prev * 0.95, 1.0);
}
```

A param is a typed uniform declared on the pass, with optional range metadata:

```json
{ "params": { "gain": { "type": "float", "value": 0.45, "min": 0, "max": 1, "step": 0.01 } } }
```

The preamble turns that into `uniform float gain;`, and the console shows
every param as a live control. The types are `float`, `int`, `bool`, `color`
(a `vec3` in the shader, which takes a [colour](layers.md#colours)), `vec2`, `vec3`,
`vec4` and `string`. A string param never reaches the shader: a procedural
scene reads it on its ticks, and `options` turns its text box in the console
into a dropdown. A param declared on the scene, in place of a pass, holds one
value that every pass reads.

A scene declares an asset by its file. ghee decodes the file at setup and
binds it in every pass:

```json
{ "assets": { "noise": { "file": "noise.raw", "width": 32, "height": 32, "format": "u8" } } }
```

ghee looks a relative `file` up in the directory of the scene file, or of a
procedural scene's executable, then in each directory on `$GHEE_ASSETS`, a
colon-separated list. A path may be absolute or climb out with `..`, except
in a scene that [the console](console.md) serves.

A pass samples the asset as `uniform sampler2D noise_tex`, and reads its size
in pixels in `uniform vec2 noise_size`. Sampling is `nearest` unless the
asset sets `"sampling": "linear"`.

An asset file is a `.png`, a `.jpg`, or a `.raw` file of bare pixels. A `.raw`
file declares `width`, `height` and `format`, `u8` or little-endian `f32`, and
`components` sets 1, 3 or 4 values per pixel, 1 by default. A `.ttf` or `.otf`
file is a font for a [text layer](layers.md#text-layers)'s `font_family`, and only the
layer reads it.

### Names

Every pass, layer, asset and scene param of a scene takes a name of its own.
A pass's params and arrays take names unique within the pass that no pass,
layer, asset or scene param holds, so two passes may each declare a param
`gain`. A name holds ASCII letters, digits and `_`, starts with a letter, ends
with a letter or a digit, holds no `__`, and does not start with `gl_` or
`ghee`.

ghee writes these names into every shader: `iTime`, `iTimeDelta`, `iFrame`,
`iResolution`, `iOutputResolution`, `iChannelResolution`, `uv`, and
`iChannel0` and up. A param, an array or an asset that takes one of them fails
to load. A param that holds no string takes no GLSL keyword, such as `half`,
no name of a GLSL function, such as `mix`, and no name of a function that the
GLSL it reaches defines, `main` among them. A layer may take any name that the
rule allows.

Each declaration adds names to the GLSL of the passes that it reaches:

| Declared | GLSL names | Reaches |
| --- | --- | --- |
| a param `gain` | `gain` | its pass, or every pass for a scene param |
| an array `amplitudes` | `amplitudes_tex`, `amplitudes_count` and the readers in the preamble | its pass |
| an image asset `noise` | `noise_tex`, `noise_size` | every pass except a filter pass |
| a text layer `title` | `title_cover`, `title_size` | every pass except a filter pass |

A font asset and an image layer add no name. A scene fails to load when two
declarations add the same name to one pass, such as a param `title_size`
beside a text layer `title`, and the error names both.

## Procedural scenes

A procedural scene is an executable, in any language, that supplies its own
scene and data. ghee starts it and talks to it over stdin and stdout.
[docs/procedural.md](procedural.md) holds the wire contract.

The Python package `ghee-render` speaks that contract for Python.
`import ghee` gives `Scene`, `Pass`, `Param`, `Asset` and `Layer` to build
the scene, `@ghee.feed` and `@ghee.tick` to supply the data, `ghee.layers` to
set the layers' live fields, `ghee.draw` to draw words, pictures and shapes
from a tick, and `ghee.run` for the loop. A PEP 723 scene declares
`ghee-render` inline, under `#!/usr/bin/env -S uv run --script`, as the
bundled examples do.

ghee hands such a scene the ghee-render wheel that sits beside the `ghee`
command, in `lib/ghee` next to its `bin` directory, `/usr/lib/ghee` for the
deb: uv takes ghee-render from that wheel alone, and every other package from
its indexes. ghee puts the wheel's index first on the scene's `UV_INDEX` and
sets `UV_INDEX_STRATEGY=first-index`. A dev build with no wheel there makes uv
refuse the scene, and a release with none lets uv take the release from PyPI.
uv keeps a scene's environment while it satisfies the scene's dependencies,
so a scene of yours keeps the ghee-render of its first run until
`uv cache clean`.

A Python scene also renders on its own, with no `ghee` command:
`./scene.py --display --frames 300`. [docs/python.md](python.md)
holds the Python side.

`ghee/examples/` holds the bundled demos.

## Limits

A scene holds any number of assets and layers, and names of any length. These
caps hold, and a scene past one fails to load or to set up with the error
shown:

- 16 textures in one pass, fewer on a GPU with fewer texture units. A pass
  counts its channels and arrays, each asset that its GLSL samples, and the
  curves of each text layer whose `_cover` it reads. A layer counts its pass's
  channels and arrays, plus its own textures: the two pages of its atlas, its
  bitmaps under a hinted render, and the pages too there with a shadow, a
  glow or a stroke, its curves under `"render": "slug"`, or its image:
  `pass 'main' needs 17 texture units, a pass binds at most 16`, or
  `layer 'title' needs 17 texture units, a pass binds at most 16`.
- 16 passes, filter passes included: `17 passes, over the cap of 16`.
- 16384 pixels on a side and 16777216 pixels in all, for a pass with a size
  or a scale, an array and a `.raw` asset:
  `pass 'a' size needs two positive ints [w, h] up to 16384, got [20000, 8]`,
  `pass 'a' is 32000x240, over the 16384 pixel cap of a side`,
  `pass 'a' is 8192x4096, over the 16777216 pixel cap`.
- The GPU's own texture size, 2048 pixels on a side on a VC4, for an array
  and an image:
  `array 'a' is 4096x1, over this GPU's 2048 pixel cap of a side`,
  `asset 'sky' is 4096x1024, over this GPU's 2048 pixel cap of a side`,
  `layer 'pic': 'sky.png' is 4096x1024, over this GPU's 2048 pixel cap of a side`.

A text layer past one of these caps leaves the scene running. ghee prints the
problem once, and the console shows it:

- A 1024x2048 data texture holds the curves of a layer's words, which `_cover`
  and the slug render read: about 24000 characters of DejaVu Sans in lines of
  60. At most 4 glyphs can overlap one spot. Words past either cap leave the
  last words that fit, and a layer whose first words do not fit draws none:
  `layer 'title': the words need a 1024x8192 texture, and a VC4 holds 1024x2048 at most`,
  `layer 'title': more than 4 glyphs overlap at one spot`.
- 64 curves in one strip of a glyph. The data texture cuts each glyph into as
  many as 64 strips across and 64 down, fewer when the words crowd it. A glyph
  past the cap draws an error box:
  `layer 'title': 'X' has more than 64 curves in one band`.
- Two pages of 2048x2048 hold the MTSDF fields of a font at one detail. A
  glyph past them draws an error box, once the glyphs that no words on screen
  draw have given up their room:
  `layer 'title': 'A' has no field in the atlas: the 2 pages of the atlas, 2048x2048 each, are full`.
- An atlas of 2048x2048 holds the emoji of a layer, with room for 132 emoji of
  Noto Color Emoji. An emoji past it draws an error box, once the emoji that
  no words on screen draw have given up their room.

The wire of a [procedural scene](#procedural-scenes) names an array or a param
in 255 bytes at most, and the Python shim raises ValueError past it.

GLES2 only: no compute, geometry or tessellation shaders. VC4 also has no
derivatives (`fwidth`) and no multisampled FBOs.
