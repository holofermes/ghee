# ghee

Multi-pass GLSL renderer for Raspberry Pi. Headless GLES2, out to SPI panels, HDMI, images or hardware H264.

![a four-pass particle scene, procedural text, and the bundled dither filter, looping](https://raw.githubusercontent.com/holofermes/ghee/main/media/hero.webp)

ghee compiles a scene of GLSL fragment shaders and renders it on the GPU:

- to image files on disk
- to a SPI panel
- to an HDMI display
- as an H264 or MJPEG stream, including a live editor in your browser

ghee renders with GLES2 through DRM/GBM on a stock Lite image. HDMI takes direct
scanout.

## Where does it run?

| Hardware | OS | Driver | Hardware H264 |
| --- | --- | --- | --- |
| Pi 1, Pi Zero, Pi Zero W (armv6) | Raspberry Pi OS bookworm or trixie | vc4 | yes |
| Pi 2, Pi 3, Pi Zero 2 W (arm64) | Raspberry Pi OS bookworm or trixie | vc4 | yes |
| Pi 4, Pi 400 (arm64) | Raspberry Pi OS bookworm or trixie | v3d + vc4 | yes |
| Pi 5 (arm64) | Raspberry Pi OS bookworm or trixie | v3d + vc4 | ffmpeg/libx264 |
| Desktop Linux (x86_64) | any current distro | Mesa | ffmpeg/libx264 |

The Pi 5 has no encoder block. The Pi 5 and a desktop encode with ffmpeg and
libx264. Tested boards for the hardware encoder: Pi Zero W, Pi Zero 2 W, Pi 3,
Pi 400. The `arm-unknown-linux-gnueabihf` target is ARMv6, and the build
compiles the C engine to match. ghee runs on every Pi back to the Pi 1.

## Build

The build needs a Rust toolchain and the GLES/DRM development headers. On
Raspberry Pi OS or Debian:

```
sudo apt install libdrm-dev libgbm-dev libegl1-mesa-dev libgles2-mesa-dev
cargo build --release
```

The build compiles the C engine into the binary. `target/release/ghee` is the
whole product. It needs no runtime library, no interpreter and no data files.

For an H264 stream on a Pi 5 or a desktop, install ffmpeg as well:

```
sudo apt install ffmpeg
```

Every other Pi, 1 through 4, uses the H264 hardware encoder.

Point `GHEE_SYSROOT` at a target sysroot with those same headers to
cross-compile.

## Run it

Render to numbered PNGs. `--demo` resolves the bundled scenes by name. Your own
directories join the lookup through `$GHEE_SCENES`:

```
ghee render --demo bars -o frame.####.png --frames 60
```

A procedural scene is an executable, and ghee runs it. Treat a scenes directory
as code you choose to run. Treat everything on `$GHEE_SCENES` the same way.

Everything after `--` goes to the scene itself. `-- --help` lists what a scene
takes; see [Procedural scenes](#procedural-scenes):

```
ghee render --demo hero -- --particle-grid 40x24
ghee render --demo bluesun --display kms -- --emit 480x300
```

Or write raw frames to stdout and pipe them into ffmpeg. `--timestep S` pins
engine time to `frame * S`. Frame N then always renders at time `N*S`, and two
runs are comparable:

```
ghee render --demo bars -o - --frames 300 --timestep 0.0166667 |
  ffmpeg -f rawvideo -pix_fmt rgba -s 320x240 -framerate 60 -i - out.mp4
```

Or stream it as H264 (raw Annex-B; pipe it anywhere):

```
ghee render --demo trail --h264 - --fps 30 | mpv -
```

A board with the Broadcom hardware encoder (Pi 1 through Pi 4) sends the frame
GPU -> encoder, with no CPU copy. The Pi 5 dropped that encoder. The Pi 5 and a
desktop encode the readback with libx264, so ffmpeg has to be on `$PATH`.

Or scan it straight out to the HDMI display; details in
[Hardware displays](#hardware-displays):

```
ghee render --demo trail --display kms
```

Or open the browser console on the bundled examples and edit the GLSL live:

```
ghee serve --demo
```

`ghee serve <your_scenes_dir>` serves your own scenes. The console preview
streams H264 through WebCodecs when the browser allows it, and MJPEG otherwise.

The outputs are a list. `-o`, `--display` and `--h264` combine in one render,
and `--serve` attaches the console to the whole list. One render feeds every
output.

```
ghee render --demo bars --display st7789 -o frame.####.png      # panel and a contact sheet
ghee render --demo bars --display st7789 --serve                 # panel, and edit it live
ghee render --demo bars --display kms --serve --host 0.0.0.0     # HDMI, edit from a laptop
```

`ghee serve <dir>` is `ghee render <dir> --serve`. `--serve` on a single scene
serves that scene alone.

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

A pass can also carry its GLSL inline in a `"code"` string.

A scene can also be a program. The program emits its own JSON and feeds the
scene its data; see [Procedural scenes](#procedural-scenes).

## How does it work

The pass model follows Shadertoy. The passes render in order. Each pass is a
fragment shader that draws into its own buffer. A pass reads another pass
through a numbered channel sampler. A channel on an earlier pass sees the
current frame. A channel on the pass itself, or on a later pass, sees the
previous frame. The last pass reaches the output.

Each pass renders at its own resolution. A pass samples another pass through
normalized coordinates.

ghee compiles the GLSL of every pass behind a preamble like this one:

```glsl
uniform float time;                 // seconds since start
uniform vec2  resolution;           // buffer size in pixels
varying vec2  v_texCoord;           // 0..1 across the buffer
// gl_FragCoord maps to the varying, so Shadertoy
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

A param is a typed uniform declared on the pass, with optional range metadata:

```json
{ "params": { "gain": { "type": "float", "value": 0.45, "min": 0, "max": 1, "step": 0.01 } } }
```

The preamble turns that into `uniform float gain;`. The console renders every
param as a live control. The types are `float`, `int`, `bool`, `color` (a `vec3`
in the shader), `vec2/3/4`, and `string`. A bool takes `true` or `false`, and
reaches the shader as `uniform bool`. A string never becomes a uniform. The
scene consumes a string for host-side work such as rasterizing text, and
`options` turns its text box into a dropdown. A param declared on the scene is
shared: every pass reads the one value.

## Procedural scenes

A procedural scene is an executable that supplies its own scene and its data.
ghee starts the scene process when the scene becomes active, then talks to it
over stdin and stdout. The process can be a program in any language.
`docs/procedural.md` holds the wire contract.

`shim/ghee.py` is that contract for Python. Copy it beside your scene. A scene
that needs packages declares them inline (PEP 723) under
`#!/usr/bin/env -S uv run --script`. The shim gives:

- `ghee.Scene / Pass / Param / ArrayInput / Channel`: build the scene
- `ghee.Arg`, `ghee.args([...])`: argument declarations, and the `-- --help` answer
- `@ghee.feed("name")`: the values of a named array, for each tick
- `@ghee.tick`: a hook for each tick, with `ghee.set_param(name, value)` and `ghee.skip()`
- `ghee.run(scene)`: the handshake, then the tick loop

`ghee/examples/` holds `bluesun.py` and `hero.py`, with the shim beside them.

## The console

`ghee serve <scenes_dir>` (or `--demo`) starts a web editor on `127.0.0.1:8090`.
The console gives a live preview of the running scene. It edits the GLSL of each
pass, and shows a compile error inline. It gives param sliders, pass muting and
a filter picker. Apply recompiles the running renderer.

Save writes a modified copy of a JSON scene to `<scenes_dir>/modified/`. A
procedural scene saves its param values and its filter list as a sidecar. ghee
applies the sidecar over the scene defaults on load.

`GET /frame.png` returns one lossless PNG of the next frame, at the render size.
A script sets a param, then reads back the result.

![the console editing a live scene](https://raw.githubusercontent.com/holofermes/ghee/main/media/console.png)

The console has no authentication, and it compiles any GLSL a client posts to
it, so it binds loopback by default. Pass `--host 0.0.0.0` to reach it from your
network. Do that only on a network you trust.

## Filters

A filter is a named single-pass post process, appended after the scene's own
passes. A scene can declare them:

```json
{ "passes": [ ... ], "filters": ["dither"] }
```

or the CLI can add them in order, after whatever the scene declares:

```
ghee render --demo bars --filters dither --display ssd1306
```

The bundled `dither` is an ordered 4x4 Bayer dither. It makes a grayscale scene
readable on a 1-bit panel. Your own `<name>.filter.json` files join the registry
through `$GHEE_FILTERS`, which takes colon-separated directories. A registered
filter shows up everywhere: the CLI, the console picker, an embedding program.

## ISF import

`ghee isf import` converts an [Interactive Shader Format](https://isf.video)
shader into a ghee scene or filter.

```
ghee isf import shader.fs                        # -> shader.scene.json
ghee isf import https://host/shader.fs -o s.json
ghee isf import shader.fs --as-filter            # -> shader.filter.json
ghee isf import shader.fs --size 640x480
```

ghee refuses three kinds of ISF:

- `FLOAT` pass buffers. ghee's pass buffers are 8-bit RGB.
- `IMPORTED` external images. ghee scenes have no image inputs.
- a vertex shader, or a body reading a varying. ghee's vertex shader is fixed.

## Hardware displays

```
ghee displays                      # list drivers, their options, and what works here
ghee render scene.json --display st7789
ghee render scene.json --display kms
ghee serve scenes/ --display st7789   # the console, driving the panel
```

`--display` takes an optional value. Write the driver name straight after it.
Keep any path, such as a scenes directory, before it.

`kms` is the connected HDMI display, or any KMS display. The engine takes DRM
master. It page-flips its render buffers straight to the connector, zero copy,
and the panel's own refresh paces the flips. The engine takes over a card, so a
headless Pi is the natural home. A running desktop session already owns the card.

The two bundled drivers are `st7789` (135x240 SPI LCD) and `ssd1306` (128x32 I2C
1-bit OLED), and their defaults fit those panels. A panel with other wiring takes
`--display-opts`, which sets the driver's own options:

```
ghee render --demo bars --display st7789 \
  --display-opts width=240,height=240,x_offset=0,y_offset=80,backlight=26
```

Run `ghee displays` to list every option a driver takes, with its default.

## From Rust

ghee is a library and a command. A render takes three calls: load or build a
scene, set it up, then render frames.

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

`render` takes one closure. The closure answers with the data for a named array.
The engine is thread-affine: build and use a Renderer on one thread.

## Known limitations

- GLES2 only: no compute, geometry or tessellation shaders. VC4 also has no
  derivatives (`fwidth`) and no multisampled FBOs.
- File, stream and panel output all pay a synchronous `glReadPixels`. GLES2 has
  no asynchronous readback, so the cost scales with output size.
- `--display kms` needs a card no compositor is holding, because it takes DRM
  master.
- Hardware H264 is the Broadcom encoder, which the Pi 5 does not have. The Pi 5
  and a desktop pay a CPU libx264 encode, and need ffmpeg.
- The bundled panel drivers cover two controllers (`st7789`, `ssd1306`).
- The web console has no authentication.

## FAQ

**A render says there is no GL device.** Point the engine at the right node with
`--dri` or `$GHEE_DRI`. Add your user to the `video` group, and to the `render`
group on a desktop. `ghee displays` reports what works on the host.

**`--h264` or the console stream says there is no ffmpeg on PATH.** That board
has no Broadcom encoder, so ghee encodes with ffmpeg and libx264.
`sudo apt install ffmpeg`.

**`--display kms` cannot take DRM master.** Something else owns the card, almost
always a desktop session.

**The console preview falls back to MJPEG in a browser that has WebCodecs.** H264
playback needs a secure context. `127.0.0.1` is a secure context, and a
plain-HTTP LAN address is not.

## Tests

```
cargo test --release
cargo run --release -- golden --manifest tools/fixtures/golden/manifest.json
python3 -m pytest tests/test_shim.py
```

The render tests need a GPU node, and they skip cleanly without one. The golden
check renders every bundled example. It compares the sha256 of each frame
against the pinned reference in `tools/fixtures/golden/`.

## License

MIT.

![](https://fabiopiparo.com/pixel.gif?site=ghee)
