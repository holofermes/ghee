# Changelog

## Unreleased

### Breaking

A scene, a script or a Rust host written for 0.4.0 needs these changes.

- Use Shadertoy's uniform names in a pass: `iTime` instead of `time`,
  `iTimeDelta` instead of `DT`, `iResolution` instead of `resolution`, and
  `uv` instead of `v_texCoord`. `iResolution` is a `vec3`.
- Compute the aspect as `iResolution.x / iResolution.y`. ghee no longer
  sets a param named `aspect`.
- Count `uv` and `gl_FragCoord` y up from the bottom, as GL does. A shader
  that counted y from the top reads `1.0 - uv.y`, or `iResolution.y -
  gl_FragCoord.y` in pixels.
- Put the GLSL of a pass, of `common` and of a filter in `fragment`, as text
  or as `{"file": ...}`. `code` and `shader` are gone.
- Read an array's length as `<name>_count` instead of `<name>_size`.
- Name the pass in `/set` with `pass=<name>` instead of its index.
- Remove unknown keys from a scene, and keep its names inside the [naming
  rules](docs/scenes.md#names). A scene that breaks either fails to load.
- Pass `in_pass=` to `set_param` instead of `pass_=`.
- Read the time as `tick.time`. `frame()` and `skip()` return a `ghee.Tick`
  instead of a dict.
- Pass `display=` and `serve=` to `ghee.open`. `output=` and `stats=` raise
  TypeError. `display` takes `"kms"` or None.
- Install on CPython 3.10 or newer.
- Find the edits that the console saved in `$XDG_DATA_HOME/ghee/edits`
  instead of `$XDG_STATE_HOME/ghee`, and pass `--edits-dir` instead of
  `--state-dir` to move them. Drafts stay in `$XDG_STATE_HOME/ghee`. When
  ghee opens a scene whose saved edits it finds among the drafts, it moves
  them, once.
- Catch OSError instead of RuntimeError where a GPU or a display won't
  open. A device node that refuses its open raises the subclass of its
  errno, such as PermissionError.
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
  a `ConsoleHost` implements `set_layer` and `save`, and a
  `SceneController` gives the console its `ghee::live::Live` through
  `live()` instead of taking `on_param`. Set a `Console`'s frame size and
  its preview and stream settings on `console.hub`, and take the stream
  types from `ghee::console::hub`. Set a `Loop`'s `timestep` and `pacer` on
  its `clock`. Take a `Result` from `Card::open`, pass the display's `Card`
  in the console's `Options.display` and its size as an `Option` in `size`,
  and give an `ArraySource`'s data as a `Feed`. Stop a console through the
  `Control` in its `Options.control` instead of `server::stop()`. List a
  `Scene`'s filters and those of `Options.filters` as
  `ghee::filters::Filter`s instead of names. Open a `Renderer` with
  `Renderer::builder(width, height)` instead of `Renderer::new` and
  `Renderer::on_card`, with sizes in `u32`. Take `Ok(())` from `setup`, and
  a `SetupError` instead of the error text. Match the `CardError` of
  `Card::open` and the `SceneError` of `Scene::load`. Set `Options`'s
  folders and `dri_device` as `PathBuf`s, its `size` and `components` in
  `u32`, its `rotate` as a `Rotation`, and its frame rate cap in `fps`
  instead of `pacer`. `Card::size` gives `u32`. When `run` returns, SIGINT
  and SIGTERM go back to their handlers from before `stop_on_signals`, and a
  sink turns on before its first frame instead of after it.

### Added

- Draw text, images, rects, lines and circles over any pass with layers and
  their effects ([Layers](docs/layers.md)). Text takes DejaVu Sans, Pixel
  Operator 8, any `.ttf` or `.otf`, a list of fonts, and colour emoji.
- Draw from Python with no GLSL, in a `with screen.draw() as d:` block
  ([Python](docs/python.md)).
- Give `ghee.Layer` a Pillow image or a numpy array as its `image`, as
  `d.image` takes.
- Change a layer at runtime with `screen.layers`, `ghee.layers` or `/set`,
  and read its keys back through `screen.layers` and `ghee.layers`.
- Write a colour as CSS's `#rgb`, `#rgba` or `#rrggbbaa` too. A layer
  draws the alpha.
- Name a pass's param in `/set` as `name=main.tile`, as `set_param` takes
  it.
- Reach a console on loopback through a proxy such as `tailscale serve`:
  pass the proxy's host names to `--allow-hosts`.
- Sample image and `.raw` data assets in a shader.
- Read `iFrame`, `iChannelResolution` and `iOutputResolution` in a pass.
- Write a pass in GLSL ES 3.00 by starting it with `#version 300 es`, on
  every GPU with OpenGL ES 3 ([Passes](docs/scenes.md#passes)).
- Run a Shadertoy pass as it is: start it with `#version 300 es`, and ghee
  calls its `mainImage` with `gl_FragCoord.xy` ([From
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
- Restore the unsaved edits of the scenes that you made in the console too.
- `ghee render` draws a scene with the edits that a console saved for it.
  To draw the scene as its files hold it, pass `--original`.
- Set up an SH1106, SSD1322 or ILI9341 panel with `ghee overlay`.
- Turn the frame with `--rotate`, and set a display's brightness with
  `--brightness`.
- Use `--drop` to show the newest frame instead of waiting for the display.
- Show more greys on small panels with the `frc` and `frc16` filters
  ([Filters](docs/filters.md)).
- Start a filter at settings of your own with an entry such as
  `{"name": "frc", "calm": 8}` in a scene's `filters`, or with
  `--filters frc=calm=8` ([Changing a filter's
  settings](docs/filters.md#changing-a-filters-settings)).
- Keep transparency in the output with `--alpha`. Raw frames take colour
  premultiplied or unpremultiplied, and a PNG takes it unpremultiplied.
- Move ghee's cache with `$GHEE_CACHE`.
- Add, change, move and remove layers from a scene program with an RFC
  6902 JSON Patch in message kind 6. A patch applies whole or not at all.
- Run a scene program in lockstep with `"lockstep": true` in its scene
  ([Procedural scenes](docs/procedural.md)).
- Pass `ghee.run` a path, a dict or JSON text, as `ghee.open` takes them.
- Turn a display off and on again from Python with `screen.blank`.
- Try the new bundled demos: assets, bars, feedback, lens, neon, passes,
  tiltshift, hero, karaoke, nowplaying, spectrum and stats.
- Stop or pause a console from any thread of a Rust host with a
  `ghee::control::Control`, stop it on SIGINT and SIGTERM with
  `stop_on_signals`, and end the program by that signal with `reraise`
  ([Rust](docs/rust.md)).
- Name a Rust host's asset folders with `ghee::scene::set_asset_dirs`, and
  cap its frame rate with `ghee::output::pacer`.
- The PyPI wheel carries the `ghee` command. It starts in 111 ms on a Pi
  Zero 2 W.
- Pass `--filters ""` to draw a scene without its own filters.

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
- ghee rounds a half up in an int param, in a scene file, on the wire, in
  the console and in `set_param` alike: 2.5 gives 3, and -2.5 gives -2. A
  scene file's 2.7 gives 3 instead of 2.
- A Python scene file that you run yourself renders as `ghee render` renders
  it. Its flags take `ghee render`'s checks and help, `--stats` prints the
  same lines, and `--serve` gives the console of `ghee render --serve`,
  which saves your edits.
- A SIGTERM or a Ctrl-C ends `ghee` and a Python scene file that you run
  yourself by that signal, after ghee closes the display, so systemd counts
  the stop as clean.
- `ghee --version` prints one line, such as `ghee 0.5.0`, on stdout. A bare
  `ghee` prints its usage on stderr and exits with status 2.
- The console prints its address on stderr. `ghee.open(serve=...)` logs it
  at INFO on the `ghee` logger of Python's `logging`.
- Put a layer's key back to its default with `value=null` in `/set`
  instead of `value=none`.
- Send each edit to the console with an `If-Match` header that holds the
  scene's `ETag` instead of `rev=` and `vid=`. Each edit answers the new
  `ETag`.
- The console answers HEAD as it answers GET, and a 405 names the methods
  that the path takes in `Allow`.
- Every error line that `ghee` prints starts with `ghee: `.
- `ghee render -o -` and `ghee displays` end by SIGPIPE when the reader of
  stdout goes away, instead of exiting with status 0.
- When a scene's program ends early, ghee names its exit status or the
  signal that ended it, as in `did not complete the handshake (exit status
  3)`. ghee logs a line when it sends the program SIGTERM or SIGKILL.
- `ghee render -v` names the program that runs in place of a scene file.
- The console answers 501 to a method that HTTP doesn't define, 304 to a
  GET whose `If-None-Match` names the current `ETag`, and 412 to a GET whose
  `If-Match` names none. Its 503 carries `Retry-After`.
- A `POST /scene` that changes a param or a layer key that the scene's
  program sets answers 403, as a `/set` of it does, instead of keeping the
  program's value.
- A change that a scene's program makes to its layers gives the scene a new
  `ETag`.
- The console runs a program among your own scenes as `ghee render` runs
  it, and lists a folder there that holds only a program.
- A console that `ghee.open(serve=...)` serves locks its shaders, passes,
  channels, filters and new params in the page.

### Fixed

- A console no longer hangs on Ctrl-C or SIGTERM.
- An edit in the console of a scene with `filters` finds the scene's
  images and fonts in its folder, instead of failing to load them.
- With `--host 0.0.0.0`, the console prints the addresses that reach it,
  instead of `localhost`.
- ghee turns a display off when it lets go of it, instead of leaving it lit
  with the kernel's framebuffer.
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
- `GET /scene` holds every edit that answered before it, and the program's
  changes, under the `ETag` that names them, instead of an older scene.
- The console locks a text layer's `font_family` and `text_align` that a
  program passes, instead of taking a change that snaps back.
- The console shows a `vec2` param that holds a position as two numbers,
  instead of a direction pad that clamps it between -1 and 1.
- A page that opens on a scene whose files changed takes the new files.
- `ghee.open(display="kms")` refuses a card whose GL runs in software, as
  `ghee render` does, instead of drawing through llvmpipe.
- The console's `/stats` gives `display.format` as `rgb565` under
  `--rgb565`, instead of `rgba`.
- `--display kms` and `ghee.open(display="kms")` say why no display opened:
  the card's own error, no connected display, or GL in software.
- `Renderer::blank` does nothing on a renderer with no display, instead of
  failing.
- `fwidth`, `dFdx` and `dFdy` compile in a pass on every GPU that has them,
  with or without an `#extension` line, instead of on none
  ([Passes](docs/scenes.md#passes)).
- A compile error prints the GL's whole log, instead of its first 511
  bytes, and each error in it names the line of the pass or of the common
  block that the author wrote, instead of a line of the shader that ghee
  assembled.
- ghee refuses a frame or a pass past the GPU's texture size, 2048 pixels a
  side on a Pi Zero to Pi 3, and names the cap, instead of failing with a
  GL error.
- `--rgb565`, and `opts={"rgb565": True}` in `ghee.open`, refuse a display
  that takes no RGB565 frames, such as an SSD1306, instead of drawing
  nothing and ending with status 0.
- `--stats` prints its summary when a Ctrl-C or a SIGTERM ends the render
  too, instead of nothing.
- `/set` refuses `NaN` and infinity, and names the value as sent, instead
  of `None`. `set_param` raises ValueError for them.
- A pass that holds GLSL ES 3.00 and lacks `#version 300 es` fails with the
  line to add, instead of the GPU's errors.
- `--display kms` on an SPI or I2C panel no longer prints `MESA-LOADER:
  failed to retrieve device information`.
- `screen.layers` and `ghee.layers` read a whole number back as an int,
  instead of a float.

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
