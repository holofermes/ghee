# Changelog

## Unreleased

### Breaking

A scene, a script or a Rust host written for 0.4.0 needs these changes.

- Use Shadertoy's uniform names in a pass: `iTime` instead of `time`,
  `iTimeDelta` instead of `DT`, `iResolution` instead of `resolution`, and
  `uv` instead of `v_texCoord`. `iResolution` is a `vec3`.
- Compute the aspect as `iResolution.x / iResolution.y`. ghee no longer
  sets a param named `aspect`.
- Put the GLSL of a pass, of `common` and of a filter in `fragment`, as text
  or as `{"file": ...}`. `code` and `shader` are gone.
- Read an array's length as `<name>_count` instead of `<name>_size`.
- Name the pass in `/set` with `pass=<name>` instead of its index.
- Remove unknown keys from a scene, and keep its names inside the [naming
  rules](docs/scenes.md#names). A scene that breaks either fails to load,
  and the error names the key.
- Pass `in_pass=` to `set_param` instead of `pass_=`.
- Read the time as `tick.time`. `frame()` and `skip()` return a `ghee.Tick`
  instead of a dict.
- Pass `display=` and `serve=` to `ghee.open`. `output=` and `stats=` raise
  TypeError. `display` takes `"kms"` or None.
- Install on CPython 3.10 or newer.
- Set up SPI and I2C panels with `ghee overlay`, and draw on them with
  `--display kms` ([Outputs and displays](docs/displays.md)). ghee's own
  `st7789` and `ssd1306` drivers and `--display-opts` are gone.
- `ghee isf import` and `ghee bench` are gone.
- Fold the scenes in the console's old `modified/` folder into their scene
  files by hand. ghee no longer reads that folder.
- Send a param message with exactly as many values as its type takes. ghee
  refuses a message with too many and logs it. 0.4.0 took the first values.
- Update a Rust host for the new signatures ([Rust](docs/rust.md)):
  `SceneProcess::start` takes a `size`, `render_loop` takes the renderer,
  and a `ConsoleHost` implements `set_layer` and `save`.

### Added

- Draw text, images, rects, lines and circles over any pass with layers and
  their effects ([Layers](docs/layers.md)). Text takes DejaVu Sans, Pixel
  Operator 8, any `.ttf` or `.otf`, a list of fonts, and colour emoji.
- Draw from Python with no GLSL, in a `with screen.draw() as d:` block
  ([Python](docs/python.md)).
- Change a layer at runtime with `screen.layers`, `ghee.layers` or `/set`.
- Sample image and `.raw` data assets in a shader.
- Read `iFrame`, `iChannelResolution` and `iOutputResolution` in a pass.
- Run a Shadertoy pass as it is: a pass that defines `mainImage` instead of
  `main` gets y-up coordinates, `texture`, `texelFetch` and `tanh` ([From
  Shadertoy](docs/scenes.md#from-shadertoy)).
- Use `ghee_hash` and `ghee_noise` to get the same random values on every
  GPU.
- Set how other passes sample a pass with its `sampling`: nearest, linear or
  trilinear.
- Render a pass at a share of the frame's size with `scale`, and hide it
  with `visible`.
- Write colours as CSS names, such as `"navy"` and `"transparent"`.
- Edit layers, passes and shaders in the console, save your edits, and make
  new scenes ([Console](docs/console.md)).
- Undo and redo your console edits with Ctrl+Z and Ctrl+Shift+Z ([Undo and
  redo](docs/console.md#undo-and-redo)).
- Restore the console edits that you hadn't saved when ghee stopped. The
  first console that opens the scene asks ([Saving](docs/console.md#saving)).
- Set up an SH1106, SSD1322 or ILI9341 panel with `ghee overlay`.
- Turn the frame with `--rotate`, and set a display's brightness with
  `--brightness`.
- Use `--drop` to show the newest frame instead of waiting for the display.
- Show more greys on small panels with the `frc` and `frc256` filters
  ([Filters](docs/filters.md)).
- Keep transparency in the output with `--alpha`, premultiplied or
  unpremultiplied.
- Move ghee's cache with `$GHEE_CACHE`.
- Add, change, move and remove layers from a scene program with message
  kinds 6 to 10.
- Run a scene program in lockstep with `"lockstep": true` in its scene
  ([Procedural scenes](docs/procedural.md)).
- Try the new bundled demos: assets, bars, feedback, lens, neon, passes,
  tiltshift, hero, karaoke, nowplaying, spectrum and stats.
- The PyPI wheel carries the `ghee` command. It starts in 111 ms on a Pi
  Zero 2 W.

### Changed

- ghee repeats the last frame when nothing that the frame reads has
  changed. A still status screen costs 0.16 ms a frame on a Pi Zero 2 W,
  down from 1.32.
- ghee hands a panel each frame just in time for its bus. A program that
  draws as fast as it can reaches a Pi 5's SSD1322 in 5.5 ms, down from
  14.8.
- `ghee.args()` ignores flags that the scene doesn't declare, and prints one
  stderr line for each.
- ghee logs a refused message from a scene program, once per mistake. 0.4.0
  dropped it silently.
- After dropped ticks, a tick's `dt` covers the whole gap since the last
  tick that the program got.
- ghee rounds a half up in an int param, on the wire, in the console and in
  `set_param` alike: 2.5 gives 3, and -2.5 gives -2.

### Fixed

- A console no longer hangs on Ctrl-C or SIGTERM.
- A Pi 5's SPI or I2C panel shows the frame that the GPU drew, instead of
  stale fragments of earlier frames.
- ghee frees each scene's GPU buffers at a switch. A console on a Pi 5 ran
  out of GPU address space after enough switches, and the GPU reset.
- `dither` no longer draws wrong 4x4 blocks on a Pi Zero to Pi 3.
- ghee opens a display's card once, so the kernel's console no longer
  flashes before the first frame.
- The console refuses a scene with a filter that ghee doesn't know and
  names the filter, as `ghee render` does, instead of skipping the filter.
- The console lists a scene folder that holds only its program, as
  `ghee render` runs it.

## 0.4.0 - 2026-08-06

- `--display kms` is any connected KMS display: HDMI, or an SPI/I2C panel with
  a kernel driver.
- `ghee displays` lists every card with a connected display.
- `ghee overlay` writes the kernel's setup for a bundled panel: the
  config.txt lines, and the firmware file that overlay loads.
- A KMS display blanks and lights through its kernel driver, which takes
  the panel's backlight or its display-off command with it.
- The console takes a host filter list, applied after each scene's own.
- `--stream-fps` sets the console's /h264 encode rate, 30 by default.
- `--size` on a KMS display renders that size and the display scales it
  across the mode, so a scene can shade a quarter of the pixels and fill the
  screen.
- A scene sizes its buffers as a share of the frame: `full`, `half`,
  `quarter`, `eighth`, `sixteenth`, wherever a WxH goes.
- `--demo clocks`: a grid of dials that fits itself to the frame.
- `-- --help` says when a scene was still starting rather than calling it
  pass-through, and `--no-probe` skips the wait.
- Importing ghee no longer reads the installed metadata, which starts every
  Python scene about a second sooner on a Pi Zero.
- Minor improvements.

## 0.3.0 - 2026-08-04

Ported to Rust.

- Any executable is a scene (see `docs/procedural.md` for the wire
  contract).
- The `ghee-render` wheel carries the embedded renderer and the
  procedural-scene shim in one package.
- The deb carries the `ghee` command and the Python package in one, per
  arch.
- `docs/http.md`: the console's HTTP API.
- u8 data arrays.
- `st7789`: 12-bit colour and partial row updates.
- The H264 fanout fails fast to the client.
- `ghee bench`: the benchmark suite, in the binary.
- `golden --dump` writes the frames out.
