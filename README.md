# ghee

Headless GLSL renderer for small Linux devices. ghee compiles a set of fragment
shaders (a scene), renders them off-screen with GLES2 through DRM/GBM, and hands
the pixels to whatever you point it at: image files on disk, a SPI panel, a live
MJPEG stream in your browser. No X, no Wayland, no window system at all.

The baseline target is the Raspberry Pi VC4 GPU (tested on a Pi Zero 2 W and a
Pi 3 A+), and the same code runs on any desktop Linux with Mesa, so you can
write scenes on your workstation and deploy them to a Pi unchanged.

## A scene in one file

A scene is JSON: an ordered list of passes, each one a GLSL fragment shader.
This is `bars.scene.json`, one of the bundled examples:

```json
{
  "version": 1,
  "passes": [
    {
      "name": "main",
      "code": "void main() { vec2 uv = v_texCoord; float fi = uv.x * 15.0; int i0 = int(floor(fi)); float h = mix(amplitudes_at(i0), amplitudes_at(min(i0 + 1, 15)), fract(fi)); float m = step(uv.y, h); gl_FragColor = vec4(mix(vec3(0.04), vec3(0.1, 0.9, 0.4), m), 1.0); }",
      "arrays": [{ "name": "amplitudes", "size": 16 }]
    }
  ]
}
```

Render it to numbered PNGs:

```
ghee render bars.scene.json -o frame.####.png --frames 60
```

Or open the browser console, which live-renders every bundled example and lets
you edit the GLSL with recompile-on-apply:

```
ghee serve
```

## Install

The Python package needs the C engine built once, which needs a compiler,
CMake, and the GLES/DRM development headers. On Raspberry Pi OS or Debian:

```
sudo apt install cmake pkg-config libdrm-dev libgbm-dev libegl1-mesa-dev libgles2-mesa-dev libjpeg-dev
pip install ghee-render
```

The distribution is named `ghee-render`; the import and the CLI are `ghee`.
Optional extras: `ghee-render[console]` for the browser console,
`ghee-render[drivers]` for the bundled SPI display drivers,
`ghee-render[test]` for the test suite.

If the GL stack is missing at install time the package still installs, but
rendering will ask for a prebuilt engine via `$GHEE_LIB`. The renderer picks
its GPU node automatically (`/dev/dri/card0` on a Pi, `renderD128` on a
desktop); override with `$GHEE_DRI` or `--dri`.

## How passes fit together

Each pass renders into its own buffer. Shaders get a small preamble; everything
below is available without declaring it:

- `time` (float) and `resolution` (vec2), plus `v_texCoord` as the 0..1 varying.
- Arrays are named float arrays the host feeds every frame, declared per pass
  as `{"name": "amplitudes", "size": 16}`. Read one element with
  `amplitudes_at(i)` or interpolate with `amplitudes_sample(t)`. They are bound
  as 1xN textures, so a lookup is a single fetch and there is no uniform-array
  size cap (VC4 has a small one).
- Channels are `iChannel0..N` samplers, each bound to a pass's output buffer by
  name. A reference to an earlier pass reads the current frame; a reference to
  the pass itself (or a later one) reads the previous frame, which is how you
  build feedback: trails, decay, ping-pong simulation. See the bundled `trail`,
  `echo`, and `twopass` examples.
- Params are typed uniforms (`float`, `int`, `vec2/3/4`) with optional
  `min`/`max`/`step`. The console turns them into sliders and updates them live
  without recompiling.

`gl_FragCoord.xy / resolution` in scene code is remapped to the exact varying in
the preamble. On VC4 the raw `gl_FragCoord` interpolates slightly off, enough to
shear a feedback pass by a third of a pixel per row, so ghee gives you the
stable coordinate and your ShaderToy-style code keeps working.

## Feeding data

ghee renders arrays; producing them is the host's job. Anything that yields
floats per frame works: audio FFT bands, sensor readings, network stats. With
nothing wired up, `ghee render` and `ghee serve` feed an animated placeholder so
every example runs out of the box.

To inject real data, embed the console (or the renderer) in your process:

```python
from ghee.console.server import run

run(arrays={"amplitudes": my_source})  # my_source(size, frame) -> floats
```

For scenes that need code, not just JSON, write a procedural scene: a `.py`
file with a `SceneController` that builds the passes and implements
`provide(name, size, frame)` for its own arrays. `examples/spectrum.py` shows
the shape, including a param that both the shader and the Python side read.

## The console

`ghee serve [scenes_dir]` starts a small web editor on `127.0.0.1:8090`: live
preview of the running scene, GLSL editing per pass with compile errors inline,
param sliders, pass muting, and a filter picker. Edits apply to the running
renderer and can be saved back to disk.

The console has no authentication and compiles whatever GLSL is posted to it,
so it binds loopback by default. `--host 0.0.0.0` exposes it to your network;
do that only on a network you trust.

## Filters

A filter is a named single-pass post-process (`dither` ships with ghee) that a
scene or a host can append after the scene's own passes:

```json
{ "passes": [ ... ], "filters": ["dither"] }
```

The bundled `dither` is an ordered 4x4 Bayer dither, which is what makes
grayscale scenes readable on 1-bit panels like the SSD1306.

## Hardware displays

```
ghee displays                      # list drivers and whether they import here
ghee render scene.json --display st7789
```

Bundled drivers: `st7789` (240x240 SPI LCD) and `ssd1306` (128x64/128x32 1-bit
OLED), both behind the `[drivers]` extra. Third-party drivers register through
the `ghee.drivers` entry point group and get picked up by name.

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
