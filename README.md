# ghee

[![ci](https://github.com/holofermes/ghee/actions/workflows/ci.yml/badge.svg)](https://github.com/holofermes/ghee/actions/workflows/ci.yml)

Hardware accelerated renderer for Raspberry Pi and more!

![a four-pass particle scene, procedural text, and the bundled dither filter, looping](https://raw.githubusercontent.com/holofermes/ghee/main/media/hero.gif)

ghee compiles a scene (GLSL fragment shaders, live parameters, optionally Python driving them) and renders it straight on the GPU:

- to image files on disk
- to a SPI panel
- to an HDMI display
- as an H264 or MJPEG stream, including a live editor in your browser

Rendering is GLES2 through DRM/GBM, with direct scanout for HDMI.
The baseline target is the Raspberry Pi VC4 GPU, but it is also supported on newer generation Pis with V3D, and on any desktop Linux with Mesa.

## Scenes

A scene is JSON: an ordered list of passes, each with its own GLSL fragment shader.
This is `bars.scene.json`, one of the bundled examples:

```json
{
  "version": 1,
  "passes": [
    {
      "name": "main",
      "shader": "bars.glsl",
      "arrays": [{ "name": "amplitudes", "size": 16 }]
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
  gl_FragColor = vec4(mix(vec3(0.04), vec3(0.1, 0.9, 0.4), m), 1.0);
}
```

Optionally, a pass can also carry its GLSL inline in a `"code"` string.

A scene can also be a Python file: a `SceneController` that builds the same
passes in code and feeds them its own data per frame; see
[ghee from Python](#ghee-from-python).

## Run it

Render it to numbered PNGs (bundled scenes resolve by name; your own
directories join the lookup through `$GHEE_SCENES`):

```
ghee render bars -o frame.####.png --frames 60
```

Or stream it as H264 (raw Annex-B; pipe it anywhere):

```
ghee render trail --h264 - --fps 30 | mpv -
```

On boards with the VC4 hardware encoder (original Zero through Pi 4) the frame
goes GPU -> encoder with no CPU copy; elsewhere ffmpeg/libx264 encodes the
readback. `--bitrate` sets bits/s.

Or scan it straight out to the HDMI display; details in
[Hardware displays](#hardware-displays):

```
sudo ghee render trail --display kms
```

Or open the browser console on the bundled examples and edit the GLSL live:

```
ghee serve --demo
```

`ghee serve <your_scenes_dir>` serves your own scenes instead. The console
preview streams H264 (WebCodecs) when the browser can, MJPEG otherwise.

## Install

Building the C engine needs a compiler, CMake, and the GLES/DRM development
headers; this part is mandatory. On Raspberry Pi OS or Debian:

```
sudo apt install cmake pkg-config libdrm-dev libgbm-dev libegl1-mesa-dev libgles2-mesa-dev libjpeg-dev
```

Then use uv:

```
uvx ghee-render serve --demo    # try it
uv add ghee-render              # in a project
```

or pip, inside a venv:

```
pip install ghee-render
```

Optional extras: `ghee-render[drivers]` for the bundled SPI display drivers,
`ghee-render[test]` for the test suite.

If the GL stack is missing at install time the package still installs,
but rendering will ask for a prebuilt engine via `$GHEE_LIB`.

Note that the renderer picks its GPU node automatically, override with
`$GHEE_DRI` or `--dri`.

## How does it work

The pass model is heavily inspired by Shadertoy. Passes render in order, each
one a fragment shader drawing into its own buffer. A pass reads other passes
through numbered channel samplers: a channel bound to an earlier pass sees the
current frame, and a channel bound to the pass itself (or a later one) sees the
previous frame, and the last pass is what reaches the output.

A scene is usually a JSON file. It can also be a Python file, referred to as a
procedural scene; see [ghee from Python](#ghee-from-python) below.

Every pass's GLSL is compiled with this simple preamble:

```glsl
uniform float time;                 // seconds since start
uniform vec2  resolution;           // buffer size in pixels
varying vec2  v_texCoord;           // 0..1 across the buffer
// gl_FragCoord is remapped to the varying, so ShaderToy
// gl_FragCoord.xy / resolution math is exact

// declared array, e.g. {"name": "amplitudes", "size": 16}:
const int amplitudes_size;          // = 16
float amplitudes_at(int i);         // one element
float amplitudes_sample(float t);   // interpolated, t in 0..1
// arrays are bound as 1xN textures

// a 2D input, e.g. {"name": "text", "size": [96, 16]}:
float text_sample2(vec2 uv);        // one fetch, normalized uv
// the host feeds w*h floats: a mask, host-rasterized text, an image

// per channel: uniform sampler2D iChannel0..N
// per param:   uniform <type> <name>
```

Channels are declared on the pass by `"buffer"` name:

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

Params are typed uniforms declared on the pass, with optional range metadata:

```json
{ "params": { "gain": { "type": "float", "value": 0.45, "min": 0, "max": 1, "step": 0.01 } } }
```

which the preamble turns into `uniform float gain;`. The console renders every
param as a live control. Types: `float`, `int`, `color` (a `vec3` in the
shader), `vec2/3/4`, and `string`. A string never becomes a uniform: the
scene's controller consumes it (`on_param`) for host-side work like
rasterizing text, and `options` turns its text box into a dropdown. Params
declared on the scene instead of a pass are shared: one value, every pass.

## ghee from Python

Rendering is three calls: load a scene, set it up, render frames.

```python
from ghee import load_scene
from ghee.engine import Renderer

scene, _ = load_scene("bars.scene.json")
r = Renderer(320, 240)
r.setup(scene)
frame = r.render()  # (h, w, components) numpy array
```

When a scene declares arrays you must pass them to `render()` with a
single function returning the data for a given array:

```python
from math import sin

def my_source(size, frame):
    # size floats in 0..1
    return [0.5 + 0.5 * sin(frame / 15.0 + k) for k in range(size)]

for i in range(300):
    frame = r.render(lambda name, size: my_source(size, i))
```

A declared array with no data gets an animated sinewave placeholder.

One can build a whole procedural scene in Python with a
`SceneController` class, using `provide()` to fill the named arrays:

```python
from math import sin

from ghee import ArrayInput, Pass, Scene, SceneController


class Pulse(SceneController):
    def build(self, base):
        return Scene(passes=[Pass(
            name="main",
            code="void main() { gl_FragColor = vec4(vec3(level_at(0)), 1.0); }",
            arrays=[ArrayInput("level", 1)],
        )])

    def provide(self, name, size, frame):
        if name == "level":
            return [0.5 + 0.5 * sin(frame / 15.0)]
        return None
```

## The console

`ghee serve <scenes_dir>` (or `--demo`) starts a small web editor on
`127.0.0.1:8090`: live preview of the running scene, GLSL editing per pass with
compile errors inline, param sliders, pass muting, and a filter picker. Apply
recompiles the running renderer.

Save writes a modified copy to `<scenes_dir>/modified/` for JSON scenes; a
procedural scene saves its param values and filter list as a sidecar applied
over the scene's defaults on load.

The console embeds in Python too, with array names mapped to providers:

```python
from ghee.console.server import run

run(arrays={"amplitudes": my_source})  # my_source(size, frame) -> floats
```

![the console editing a live scene](https://raw.githubusercontent.com/holofermes/ghee/main/media/console.png)

The console has no authentication and compiles whatever GLSL is posted to it,
so it binds loopback by default. To expose it to your network pass `--host 0.0.0.0`,
do that only on a network you trust.

## Filters

A filter is a named single pass post process appended after the scene's own
passes. A scene can declare them:

```json
{ "passes": [ ... ], "filters": ["dither"] }
```

or the CLI can add them ad hoc, in order, after whatever the scene declares:

```
ghee render bars --filters dither --display ssd1306
```

The bundled `dither` is an ordered 4x4 Bayer dither, which is what makes
grayscale scenes readable on 1-bit panels. Your own `<name>.filter.json`
files join the registry through `$GHEE_FILTERS` (colon-separated
directories), and show up everywhere: the CLI, the console's picker, an
embedding app.

## Hardware displays

```
ghee displays                      # list displays and whether they work here
ghee render scene.json --display st7789
sudo ghee render scene.json --display kms
```

`kms` is the connected HDMI (or any KMS) display: the engine takes DRM master
and page-flips its render buffers straight to the connector, zero copy, paced
by the panel's own refresh. It needs root and a card nothing else is driving,
so a headless Pi is the natural home; a running desktop session already owns
the card.

The two bundled drivers are reference implementations for the exact panels
this project was built on: `st7789` (135x240 SPI LCD, pins and offsets as
wired on my Pi) and `ssd1306` (128x32 I2C 1-bit OLED). Both live behind the
`[drivers]` extra. For other panels, or the same controllers wired
differently, write a small driver class (subclass `DisplayDriver`, implement
`write(frame)`) and register it through the `ghee.drivers` entry point group;
it gets picked up by name. Making the bundled drivers configurable (size,
pins, offsets) is on the roadmap.

## Tests

```
pip install 'ghee-render[test]'
python -m pytest
```

The render tests need a GPU node and skip cleanly without one. Every bundled
example renders against a committed golden image, compared with a tolerance
that one baseline passes on both desktop Mesa and VC4.

## License

MIT.
