# ghee

[![ci](https://github.com/holofermes/ghee/actions/workflows/ci.yml/badge.svg)](https://github.com/holofermes/ghee/actions/workflows/ci.yml)

Multi-pass GLSL renderer for Raspberry Pi. Headless GLES2, out to SPI panels, HDMI, images or hardware H264.

![a four-pass particle scene, procedural text, and the bundled dither filter, looping](https://raw.githubusercontent.com/holofermes/ghee/main/media/hero.webp)

ghee compiles a scene (GLSL fragment shaders, live parameters, optionally Python
driving them) and renders it straight on the GPU:

- to image files on disk
- to a SPI panel
- to an HDMI display
- as an H264 or MJPEG stream, including a live editor in your browser

Rendering is GLES2 through DRM/GBM on a stock Lite image, with direct scanout for HDMI.

## Where does it run?

| Hardware | OS | Driver | Hardware H264 |
| --- | --- | --- | --- |
| Pi 1, Pi Zero, Pi Zero W (armv6) | Raspberry Pi OS bookworm or trixie | vc4 | yes |
| Pi 2, Pi 3, Pi Zero 2 W (arm64) | Raspberry Pi OS bookworm or trixie | vc4 | yes |
| Pi 4, Pi 400 (arm64) | Raspberry Pi OS bookworm or trixie | v3d + vc4 | yes |
| Pi 5 (arm64) | Raspberry Pi OS bookworm or trixie | v3d + vc4 | ffmpeg/libx264 |
| Desktop Linux (x86_64) | any current distro | Mesa | ffmpeg/libx264 |

The Pi 5 has no encoder block, so it falls back to ffmpeg/libx264, as do
desktops. Hardware encoding is measured on a Pi Zero W 1/2, a Pi 3 and a Pi 400.
The distributed armhf package is compiled for ARMv6, so it runs on every Pi
back to the Pi 1!

## Install

Pick the asset matching your board from
[Releases](https://github.com/holofermes/ghee/releases):

```
sudo apt install ./python3-ghee_*.deb
```

To build from source you need a compiler, CMake, and the GLES/DRM development
headers. On Raspberry Pi OS or Debian:

```
sudo apt install cmake pkg-config libdrm-dev libgbm-dev libegl1-mesa-dev libgles2-mesa-dev
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

The bundled panel drivers need three more packages which are pulled in via
`apt install` automatically. In case you work from source you'll need these:

```
sudo apt install python3-spidev python3-libgpiod python3-smbus2
```

If you are outputting an H264 stream on a Pi 5 or a desktop, ghee will shell
out to ffmpeg, but you must install it yourself like so:

```
sudo apt install ffmpeg
```

On all other Pi, 1 through 4, ghee uses the H264 hardware encoder.

## Run it

Render to numbered PNGs (`--demo` resolves the bundled scenes by name; your own
directories join the lookup through `$GHEE_SCENES`):

```
ghee render --demo bars -o frame.####.png --frames 60
```

A `.py` scene is imported to build its passes, so loading or switching to one
runs its code. Treat a scenes directory, and anything you put on `$GHEE_SCENES`,
as code you have chosen to run.

Everything after `--` goes to the scene itself, and `-- --help` lists what it
takes; see [Scene arguments](#scene-arguments):

```
ghee render --demo text -- --text "hello there"
ghee render --demo hero --display kms -- --particle-buffer 160x100 --no-dither
```

Or write raw frames to stdout and pipe them into ffmpeg. `--timestep` pins
engine time to `frame * S`, so the render is reproducible (frame N always
renders at time N*S):

```
ghee render --demo bars -o - --frames 300 --timestep 0.0166667 |
  ffmpeg -f rawvideo -pix_fmt rgba -s 320x240 -framerate 60 -i - out.mp4
```

Or stream it as H264 (raw Annex-B; pipe it anywhere):

```
ghee render --demo trail --h264 - --fps 30 | mpv -
```

On boards with the Broadcom hardware encoder (Pi 1 through Pi 4) the frame goes
GPU -> encoder with no CPU copy. The Pi 5 dropped the encoder, so there and on
desktop ffmpeg encodes the readback with libx264 and has to be on `$PATH`.

Or scan it straight out to the HDMI display; details in
[Hardware displays](#hardware-displays):

```
ghee render --demo trail --display kms
```

Or open the browser console on the bundled examples and edit the GLSL live:

```
ghee serve --demo
```

`ghee serve <your_scenes_dir>` serves your own scenes instead. The console
preview streams H264 (WebCodecs) when the browser can, MJPEG otherwise.

Those outputs are a list, not a choice: `-o`, `--display` and `--h264` combine
in one render, and `--serve` attaches the console to whatever the list is. One
render feeds them all.

```
ghee render --demo bars --display st7789 -o frame.####.png      # panel and a contact sheet
ghee render --demo bars --display st7789 --serve                 # panel, and edit it live
ghee render --demo bars --display kms --serve --host 0.0.0.0     # HDMI, edit from a laptop
```

`ghee serve <dir>` is `ghee render <dir> --serve`; pointing `--serve` at a
single scene serves that one alone.

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

Optionally, a pass can also carry its GLSL inline in a `"code"` string.

A scene can also be a Python file: a `SceneController` that builds the same
passes in code and feeds them its own data per frame; see
[ghee from Python](#ghee-from-python).

## How does it work

The pass model is heavily inspired by Shadertoy. Passes render in order, each
one a fragment shader drawing into its own buffer. A pass reads other passes
through numbered channel samplers: a channel bound to an earlier pass sees the
current frame, and a channel bound to the pass itself (or a later one) sees the
previous frame, and the last pass is what reaches the output.

A pass can render at arbitrary and indipendent resolutions, and they can sample
each other through normalized coordinates.

While a scene is usually a JSON file, it can also be built in Python file,
referred to as a procedural scene; see [ghee from Python](#ghee-from-python) below.

Every pass's GLSL is compiled with roughly this simple preamble:

```glsl
uniform float time;                 // seconds since start
uniform vec2  resolution;           // buffer size in pixels
varying vec2  v_texCoord;           // 0..1 across the buffer
// gl_FragCoord is remapped to the varying, so ShaderToy
// gl_FragCoord.xy / resolution math is exact

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
param as a live control. Types: `float`, `int`, `bool`, `color` (a `vec3` in the
shader), `vec2/3/4`, and `string`. A bool takes `true` or `false` and reaches the
shader as `uniform bool`. A string never becomes a uniform: the
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


One can build a whole procedural scene in Python with a
`SceneController` class, using `provide()` to fill the named arrays:

```python
from math import sin

from ghee import ArrayInput, Pass, Scene, SceneController


class Pulse(SceneController):
    def build(self, base):
        main = Pass(
            name="main",
            code="void main() { gl_FragColor = vec4(vec3(level_at(0)), 1.0); }",
            arrays=[ArrayInput("level", 1)],
        )
        return Scene(passes=[main])

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

`GET /frame.png` returns one lossless PNG of the next frame at the render size,
so a script can set a param and read back what it did.

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
ghee render --demo bars --filters dither --display ssd1306
```

The bundled `dither` is an ordered 4x4 Bayer dither, which is what makes
grayscale scenes readable on 1-bit panels. Your own `<name>.filter.json`
files join the registry through `$GHEE_FILTERS` (colon-separated
directories), and show up everywhere: the CLI, the console's picker, an
embedding app.

## ISF import

`ghee isf import` converts an [Interactive Shader Format](https://isf.video)
shader into a ghee scene or filter.

```
ghee isf import shader.fs                        # -> shader.scene.json
ghee isf import https://host/shader.fs -o s.json
ghee isf import shader.fs --as-filter            # -> shader.filter.json
ghee isf import shader.fs --size 640x480
```

Three kinds of ISF are refused:

- `FLOAT` pass buffers. ghee's pass buffers are 8-bit RGB.
- `IMPORTED` external images. ghee scenes have no image inputs.
- vertex shader, or a body reading a varying: ghee's vertex
  shader is fixed.

## Hardware displays

```
ghee displays                      # list drivers, their options, and what works here
ghee render scene.json --display st7789
ghee render scene.json --display kms
ghee serve scenes/ --display st7789   # the console, driving the panel
```

`--display` takes an optional value, so write the driver name straight after it
and keep any path (a scenes directory) before it.

`kms` is the connected HDMI (or any KMS) display: the engine takes DRM master
and page-flips its render buffers straight to the connector, zero copy, paced
by the panel's own refresh. It takes over a card, so a headless Pi is the
natural home; a running desktop session already owns the card.

The two bundled drivers default to the panels this project was built on:
`st7789` (135x240 SPI LCD) and `ssd1306` (128x32 I2C 1-bit OLED). You can
configure differently wired is controller with `--display-opts`, which sets 
the driver's constructor arguments:

```
ghee render --demo bars --display st7789 \
  --display-opts width=240,height=240,x_offset=0,y_offset=80,backlight=26
```

Use `ghee displays` to list every option a driver takes with its default.

For another controller, write a small driver class and register it through
the `ghee.drivers` entry point group.

## Known limitations

- GLES2 only: no compute, geometry or tessellation shaders, and on VC4 no
  derivatives (`fwidth`) and no multisampled FBOs.
- File, stream and panel output all pay a synchronous `glReadPixels`. GLES2 has
  no asynchronous readback, so the cost scales with output size.
- `--display kms` needs a card no compositor is holding, because it takes DRM
  master.
- Hardware H264 is the Broadcom encoder, which the Pi 5 does not have. There and
  on desktop the software path costs a CPU libx264 encode and needs ffmpeg
  installed.
- The bundled panel drivers cover two controllers (`st7789`, `ssd1306`).
- The web console has no authentication.

## FAQ

**A render says there is no GL device.** Point the engine at the right node
with `--dri` or `$GHEE_DRI`, and check your user is in `video` (and `render`,
on desktop). `ghee displays` reports what works on the host.

**The install failed with `C engine build failed`.** CMake or the GLES/DRM
headers are missing. Install them (see Install) and try again, or use a release
`.deb`. `GHEE_LIB` and `GHEE_SKIP_ENGINE=1` install without building it.

**`--h264` or the console stream says there is no ffmpeg on PATH.** That board
has no Broadcom encoder, so ghee encodes with ffmpeg and libx264 instead.
`sudo apt install ffmpeg`.

**`--display kms` cannot take DRM master.** Something else owns the card, almost
always a desktop session.

**Illegal instruction on a Pi 1 or an original Pi Zero.** Those are ARMv6, and
Debian's armhf is built for ARMv7. Use the `armhf` release asset, which is built
against Raspbian.

**The console preview falls back to MJPEG in a browser that has WebCodecs.**
H264 playback needs a secure context. `127.0.0.1` is one; a plain-HTTP LAN
address is not.

## Tests

The suite is not shipped in the package and must be run from a checkout:

```
git clone https://github.com/holofermes/ghee
cd ghee
pip install -e '.[test]'
python -m pytest
```

The render tests need a GPU node and skip cleanly without one. Every bundled
example renders against a committed golden image, compared with a tolerance
that one baseline passes on both desktop Mesa and VC4.

The CI runners have no GPU node, so the render and golden tests always skip
there and the badge covers the rest of the suite.

## License

MIT.

![](https://fabiopiparo.com/pixel.gif?site=ghee)
