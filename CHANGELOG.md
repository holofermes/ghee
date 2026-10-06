# Changelog

## Unreleased

### Scenes

- Declare image files in `assets`: a `.png`, a `.jpg`, or a `.raw` file of
  bare `u8` or little-endian `f32` pixels that states its `width`, `height`,
  `format` and `components`. Every pass samples an asset as
  `uniform sampler2D <name>_tex` with a `vec2 <name>_size`, nearest unless
  the asset sets `"sampling": "linear"`.
- Paint a picture over a pass with an image layer, which holds `image`, the
  name of an image asset or a `.png` or `.jpg` file, in place of `text`.
  `width` and `height` scale it in pass pixels, one of them left out keeps
  the image's aspect, and both left out take its own size. It takes `x`,
  `y`, `opacity`, `blend` and `visible` as a text layer does, and `color`,
  `"#rrggbb"` or `[r, g, b]`, tints the picture. An image layer takes no
  effects and gives a pass no GLSL name of its own, so a pass reads the
  image as the asset's `<name>_tex`.
- Resolve an asset or a font that a scene names from the directory of the
  scene file or of a procedural scene's executable, then from each directory
  on `$GHEE_ASSETS`. A path may be absolute or leave that directory, except
  under the console, and the error for a file that no directory holds names
  `$GHEE_ASSETS`.
- Read `uniform float frame`, the count of frames rendered, in every pass.
- Build the mip chain of a pass's buffer after each frame with
  `"mipmaps": true`, and sample it trilinearly through every channel that
  reads it, so a buffer read far away and minified stays calm. On a GPU
  without OES_texture_npot, a pass with mipmaps takes a power-of-two size
  alone.
- Hide a pass with `"visible": false`, which a save keeps. A hidden pass
  draws a copy of the pass before it, or black as the first pass, its layers
  draw nothing, and the frame is the last visible pass. It keeps its params
  and arrays, so a set and a feed still land, and a scene that hides every
  pass fails to load.
- Store alpha in every pass buffer, so a channel reads the alpha that its
  pass wrote. Every frame stays opaque unless `ghee render --alpha` or
  `ghee.open(..., alpha=True)` keeps the output pass's alpha: a raw frame and
  `pixels()` hold the colour premultiplied by it, and a PNG holds it
  straight. `--alpha` refuses `--components 3`.
- Declare a vec or color param as one `uniform vec2`, `vec3` or `vec4`, where
  it was a `#define` over one float uniform for each component, so a local of
  its name shadows it as it shadows any uniform, and `.bg` beside a color
  named `bg` selects the field.
- Take a color param's value as `"#rrggbb"` or as `[r, g, b]` with each 0 to
  1, in the scene, in `/set` and in `set_param`. A scene process sends a
  color as three floats in message kind 1, and the Python shim turns a
  `"#rrggbb"` into them.
- Render an output pass that sets a `size` at the screen size, with one
  warning at load, where ghee refused the scene. Its size applies again once
  a filter or a pass follows it, and a pass that hidden passes leave last
  renders at the screen size.
- Keep a color param in the form that it came in, `"#rrggbb"` or
  `[r, g, b]`, through `GET /scene`, `/set`, a save and a procedural scene's
  values sidecar. The console sends the colour picked in its swatch as
  `#rrggbb`, so it saves as hex.
- Refuse a key that ghee does not take anywhere in a scene, where the scene
  loaded and ignored it, and name the key, where it sits, and the closest key
  that ghee takes, or the object that takes it: `"code"` on a layer names its
  pass, and `"font"` names `font_family`.
- Take as the name of a pass, a layer, a param, an array or an asset ASCII
  letters, digits and `_`, with a letter first and a letter or a digit last,
  where a pass took any name, and refuse a name that holds `__` or starts
  with `gl_`, which GLSL reserves, or starts with `ghee`, which ghee keeps
  for the names that it writes into a shader. The passes, the layers, the
  assets and the scene params share one set of names, and a pass's params
  and arrays take names unique in the pass and outside that set, so two
  passes may each declare a param `gain`. A filter's pass takes a name that
  nothing in the scene holds, as `dither1` beside a pass `dither`.
- Refuse `time`, `DT`, `frame`, `resolution`, `v_texCoord` and `iChannel0` and
  up as the name of a param, an array or an asset. A layer may take any of
  them, since every GLSL name that it gives a pass carries a suffix. Refuse a
  scene where two declarations make one GLSL name in a pass, such as a param
  `title_size` beside a layer `title`, and name both.
- Take 16 passes, filter passes included, where the engine took 8, and refuse
  a scene past them with an error of its own. Take any number of assets and
  layers, a channel or an array in each of a pass's 16 texture
  units, and names of any length, where the engine took 16 arrays in a scene
  and 8 channels and 4 arrays in a pass, and cut a name past 63 bytes. The
  README states every cap and its error.
- Name the pass and the line of the author's GLSL in a compile error, as
  `pass 'main', line 2 of its code:` or `line 2 of the common block`, ahead of
  the GL's own message, on Mesa's GL and on NVIDIA's alike, and print the
  error once, where it printed the GL's log twice with a line number in the
  assembled shader.

### Text

- Draw text on the GPU from a font in text layers. A pass's `layers` paint
  over its buffer after its own shader, in list order, as one quad over each
  glyph. A layer needs a `name`, holds its words in `text`, and takes
  `font_size`, `font_family`, a `.ttf` or `.otf` file or the name of a font
  asset, and `text_align`. A layer with no `font_family` draws from the
  built-in font, DejaVu Sans cut to ASCII, Latin-1 and the common
  punctuation. ghee declares a scene string param under each layer's name,
  which holds the words, and new words from `set_param`, `/set`, the console
  or a procedural scene lay out again with no recompile.
- Place a layer's words with `x` and `y`, the pass pixel of the block's top
  left, y down, and `font_size`, their pixels per em. An axis left out
  centres the block on it. A layer with no `font_size` takes the largest size
  at which the words fit the pass, less a margin of 5% of its shorter side,
  and on a given axis the room past the given pixel, less the margin at the
  far edge. The fitted size follows the words.
- Colour a layer with `color`, `"#rrggbb"`, `[r, g, b]` or a gradient
  `{"from", "to", "angle"}` with CSS's angle, 0 up and 90 right, across the
  block, white by default. `opacity` scales it, and `blend` lays it on the
  pass's buffer as `normal`, `add`, `multiply` or `screen`, colour and alpha
  alike, where `multiply` and `screen` work over an opaque pass.
  `"visible": false` hides a layer, whose words still read through
  `<name>_cover`.
- Let a pass with `layers` leave out its GLSL: it starts each frame
  transparent and holds its layers alone, for a later pass to read through a
  channel.
- Draw effects on a text layer, each an object of its fields, a field left
  out at its default, sizes in em: `text_shadow` (`x`, `y`, `blur`, `color`,
  `strength`), `glow` (`blur`, `color`, `strength`) and `text_stroke`
  (`width`, `color`) under the letters in that order, from the distance to
  them, `karaoke` (`sung`, `color`), which fills the letters up to the sung
  point and lifts the letter being sung, and `wave` (`height`, `speed`),
  `scroll` (`speed`) and `blink` (`rate`), which move the letters and their
  effects together. Every glyph's effects draw before any letter, so a glow
  never covers the next letter. `"visible": false` bypasses an effect and
  keeps its values, the slug render skips the three that read the distance,
  and ghee works out how far a layer's quads and its atlas's field reach
  from its effects, and fades the effects out at the edge of their quads.
- Set a layer's live fields with no rebuild, `Renderer::set_layer` in the
  library: `x`, `y`, `color`, `opacity` and `visible`, an image layer's
  `image`, which takes another image asset of the scene at that image's
  size, and an effect's `visible`, `color`, `strength`, `sung`, `speed` and
  `rate`. None takes `x` and `y` back to centring and `color` back to its
  default. Every other field changes with a rebuild, so a live set of one is
  refused with the live fields named, and a set that names an effect or a
  field that the layer lacks is refused with the layer's effects or its live
  fields named.
- Move `wave`, `scroll` and `blink` by a phase that ghee keeps for each
  layer and advances each frame by the effect's `speed` or `rate` times the
  frame's time step, so a new speed carries the motion on from where it is.
  Under `--timestep` the phase runs from 0 at the scene's start.
- Draw a layer's glyphs from an MTSDF atlas by default, one texture read a
  pixel. `"render": "slug"` computes Slug's exact coverage from the font's
  curves instead, at any size and with no atlas.
- Generate a font's atlas on every core at its first use: the glyphs of the
  words first, then a preload of ASCII, Latin-1 and the common punctuation in
  the room left. Cache the atlas in `$XDG_CACHE_HOME/ghee/atlas`. New words
  show in their first frame when the atlas holds their glyphs, and otherwise
  the layer keeps its last words on screen until every glyph of the new ones
  is in. A setup and a render with `--timestep` wait for them. A full atlas
  draws an error box with an error of its own. A hidden layer builds its
  atlas when it first shows.
- Generate a layer's atlas at the detail of its size, its `font_size` or the
  size that fits its words at setup, rounded up to 32, 48, 64, 96 or 128
  texels per em, and 128 for larger text, so layers of near sizes share one
  atlas.
- Read the coverage of a layer's words at any point in any pass shader,
  filter passes excepted, with `<name>_cover(p, font_size)`, and the block's
  size in em with `<name>_size`. A shader carries the coverage reader of a
  layer only when its code or the common block names `<name>_cover`.
- Paint colour emoji in layers. An emoji that the layer's font lacks draws
  from Noto Color Emoji, the file `NotoColorEmoji.ttf` found beside the
  scene, in a directory on `$GHEE_ASSETS` or in
  `/usr/share/fonts/truetype/noto/`, which the console reads too, and with
  no such file it draws the font's missing-glyph box. A character that
  U+FE0F follows takes its emoji even from a font that holds it. ghee reads
  the emoji font once, at the first words that need it, and every layer of
  every scene shares it, and the console marks a scene stale when the font
  shows up or changes. An emoji, a glyph with a PNG bitmap in a CBDT or sbix
  table, shows the bitmap in its own colours, mipmapped, and moves, fades
  and hides with its layer's letters, with no `text_shadow`, `glow` or
  `text_stroke`. A layer's emoji draw after every layer of its pass,
  `<name>_cover` reads them as no ink, and a layer compiles their shader at
  the first words that hold one, live words too, with no new setup.
- Keep the scene running through a text problem, print it once as an error,
  which `--quiet` keeps, and show it in the console and in `/stats` as
  `text_errors`, a line for each problem that names its layer or its font.
  A font that fails to load draws an error box for each character, a glyph
  that ghee cannot draw shows an error box in its place, and words that do
  not fit leave the last words that did.
- Add `--demo lettering`, `--demo karaoke` and `--demo lens`, and rename
  `--demo text`, which rasterizes its words in Pillow, to `--demo marquee`.
  Add `ghee/examples/nowplaying.py`, an artist and a title that fit a 128x32
  or a 256x64 panel.

### Python

- Set a layer's live fields on a `ghee.open` handle with no rebuild, through
  `h.layers` by attribute or by key, `h.layers.title.x = 10` or
  `h.layers["title"]["karaoke"]["sung"] = 12`, or through
  `h.set_layer(name, field, value, effect=None)`. `text` sets a text layer's
  words and `image` an image layer's image, and None takes `x`, `y` or
  `color` back to its default. A field that changes with a rebuild, an
  unknown layer, effect or field, and a value that the field does not take
  raise ValueError, and an unknown layer's error names the scene's layers.
- Set the layers of a scene script from a tick or a feed with
  `ghee.layers`, as on a handle. Under ghee, each set goes out with the
  tick's answer, checked against the scene first, so a set that ghee would
  refuse raises ValueError. In a direct run, each set goes to the handle.
- Declare an asset with `ghee.Asset` and a text layer with
  `ghee.Layer(name, text, visible, x, y, color, opacity, blend, font_size,
  font_family, text_align, render)` and a keyword for each effect, a dict of
  its fields, or an image layer with `image`, `width` and `height`, in
  `Scene(assets=...)` and `Pass(layers=...)`. A keyword left at None leaves its key to ghee's
  default. `mipmaps=True` on `ghee.Pass` builds its mip chain, and
  `visible=False` hides the pass.
- Take a scene's and a pass's `params` as a list of `ghee.Param`. Raise
  ValueError for a dict key that differs from its param's name, where the
  param went out under the key.
- Build the scene document in the shim and leave every scene rule to ghee, so
  a mistake raises ValueError with ghee's own message in `ghee.open`, in a
  direct run and under `ghee render` alike.
- Take any buffer or iterable of numbers in `h.feed` and in a direct run's
  feeds, such as a numpy float64 array, as under ghee: a uint8 buffer goes
  out as byte / 255, and any other numbers as float32. `h.feed` took only
  uint8 and float32 buffers.
- Take a `ghee.Scene` in `ghee.open`, and an `os.PathLike` wherever the
  Python API takes a path.
- Resolve the files of a Python scene in direct mode beside its script, as
  under ghee, where they resolved in the working directory, and take
  `base_dir` on `ghee.open` for a scene given as a dict, as JSON text or as a
  `ghee.Scene`, the working directory by default.
- Write the last frame to a `.png` or a `.jpg` file with `h.save(path)`,
  numbered by its tick through a run of `#` as `ghee render -o` numbers its
  files.
- Take the flags of `ghee render` in a direct run of a Python scene: `-o`
  for image files or raw frames on stdout, `--display` and
  `--display-opts`, `--serve`, `--port` and `--host` for the console, and
  `--stats` for the timing summary, where `--output` named the display and
  `--stats` the console's port. With no `--frames`, `-o` renders one frame,
  as `ghee render` does, and a display or the console runs until stopped.
- Hand a direct run's `@ghee.tick` the string params that the console
  changed, as the wire does, through `h.changed_params()`, where the tick's
  `params` stayed empty.
- Raise ValueError from `ghee.set_param` in a Python scene for a param that
  the scene does not declare or a value that does not fit its type, under
  ghee as in direct mode, and name the param in the type error of
  `set_param` and of the console.
- Ship type stubs for `ghee.open` and `Handle`, mark the package typed, and
  type the closed string sets of the shim as `Literal`: a param's type, a
  layer's text_align, render and blend, and an arg's type.

### Procedural scenes

- Set a live field of a layer from a scene process. Kind 1 takes
  `layer.field` and `layer.effect.field`, such as `title.x` and
  `title.karaoke.sung`, with one value, 0 or 1 for `visible`, three for a
  color, seven for a gradient or none for the default, and kind 3 takes
  `title.text` with a text layer's words and `icon.image` with the name of
  an image asset. A message whose first part
  names no layer sets a param, and a field that the scene refuses logs once.
- Set a string param, and so a text layer's words, from a scene process with
  message kind 3.
- Wait under a fixed `--timestep` for a lockstep scene process to answer
  each tick, so the params and arrays sent for a tick land in that tick's
  frame, where they landed a frame later. Message kind 5 ends an answer,
  `"lockstep": true` in the handshake promises one for every tick, and ghee
  sets `GHEE_LOCKSTEP=1` in the scene process to say that it reads both.
  The Python shim sends both when it finds `GHEE_LOCKSTEP`, and under an
  older ghee it writes the wire as before.
- Log once a param message from a scene process that names no param of the
  scene or does not fit the param's type, where it changed nothing in
  silence.

### Console

- Edit each pass's layers. A pass shows its tab row, with the open pass's
  rename, eye and remove at its end, then its inputs, with a chip for each
  text layer's `_cover`, each image asset's `_tex` and each array, then its
  channels on a row of their own, then its layers, each collapsed when a
  scene or the page loads, and its shader under a line. `+ layer` adds a
  text or an image layer. A layer shows a one-row header with its
  blend, render, opacity, move, rename, eye and remove, then its settings,
  and a text layer adds `+ effect` and one row for each of its effects.
- Set a live field from the console with no rebuild, as it changes. Every
  other field applies when the cursor leaves it.
- Set a live field of a layer with `/set?layer=title&field=x&value=10`, and
  a field of one of its effects with `effect=` too, in `ghee serve`,
  `--serve` and `ghee.open(stats=)`, with no rebuild. `field=text` sets a
  text layer's words, as the scene param of the layer's name does. The value
  reads as the field takes it, `none` included, and a field that changes
  with a rebuild answers 400 with the live fields named.
- Rename a pass or a layer with its pencil, in the console's dialog, where a
  double-click on a pass's tab renamed it.
- Hide a pass with its eye, which the scene keeps as the pass's `visible`,
  where the eye muted the pass in the page alone. A hidden pass, a hidden
  layer and a bypassed effect dim the same way and stay editable.
- Show a colour as its swatch alone, in the Parameters panel too, and a
  number field with no spin buttons.
- Keep the shader open or closed under its line as the user left it, one
  choice for every scene, kept in the browser across a reload. A compile
  error opens the shader of the pass that holds it and leaves the choice.
- Show a procedural scene's passes and layers with no add, move, rename or
  remove, since its process declares them.
- Keep a hidden filter's pass hidden when the console posts the scene.
- Save each pass's `visible` and the settings and effects of each layer of a
  procedural scene in its `.values.json` sidecar, beside its params and
  filters, and apply them over the scene that the process writes at its next
  start. The process keeps which layers exist and their order, a saved layer
  that it no longer declares is skipped with a log line, and a value that
  the process sets after the start wins over the saved one.
- Take the scene that an edit is for in `scene=` on `/set`, in `name=` on
  `POST /scene` and in the body of `/save`, and refuse with 409 an edit for
  a scene other than the active one, which `/scenes` and `/stats` name as
  `active`. A page whose edit was refused shows the active scene as it runs,
  where a page left open on one scene wrote its copy into the scene that the
  console had switched to.
- List in `procedural` on `GET /scenes` the scenes that a program writes,
  an empty list when there are none.
- Build a posted scene's filters on the server: `POST /scene` rebuilds a
  filter pass from its filter, keeping its place and param values, and
  appends a filter that the `filters` list names without a pass. An unknown
  filter is a 400, and the reply is the scene as it will run.
- Parse a `/set` value by the param's declared type alone, and ignore
  `type=`, which overrode it. Name a param that the scene lacks.
- Answer `/h264` with 503 and the reason when no encoder can start, and end
  a stream whose encoder fails with a type 3 frame that carries the reason.
  Show the preview in MJPEG when the H264 stream decodes no frame in 5 s.
- Grow each GLSL editor with its code, and scroll the column or the page
  that holds it, where a long shader scrolled inside a box of fixed height.
  Keep a compile error in sight under it, and mark the error on its line
  when longer lines above it wrap.
- Edit a free-text string param over several lines, and show why the server
  refused a posted scene.
- Open a tab with a tap anywhere on it, keep each tab's width and place, and
  ask before a pass, a layer or the common block goes, in the console's
  dialog, where Escape cancels and Enter removes.
- Resolve the files of a copy that the console saved into `modified/`
  beside its original, where every scene looked one directory up.

### Displays and H264

- Find a KMS display for `ghee displays` and `--display kms` in the
  kernel's cached connector state, and probe a connector, which rereads an
  HDMI display's EDID, only when that state lists no modes.
- Draw a kernel-driven panel on `--display kms` into buffers that the GPU
  allocates and the panel imports, when the panel's card is a device apart
  from the GPU and takes them. A panel driver that reads each frame with
  the CPU (ssd130x, gud, udl) then shows what the GPU drew, where on a
  Pi 5 it showed stale fragments of earlier frames.
- Keep the picture on a KMS display whole when the display refuses a page
  flip, where the next frame drew into the buffer still on screen.
- Encode a frame under 16384 pixels in hardware, scaled up by the smallest
  whole factor that reaches 16384 with each pixel a square block, where the
  encoder refused it. A 128x32 panel streams at 256x64.

### Filters and ISF

- Declare `channels` in a filter, and read the filter's own previous frame
  through `self`.
- List a hidden filter in a scene's `filters` as
  `{"name": "dither", "visible": false}`, which a save of the console
  writes for a filter whose pass is hidden.
- Declare `assets` in a filter the way a scene does. A shipped filter's file
  is compiled in, a scanned filter's resolves beside it. Only the filter's
  own pass sees them, so they never clash with a scene's names.
- Take `frc256`'s per-pixel phase from a shipped 64x64 blue-noise mask.
- Rename an ISF input or pass target named like a built-in, such as `frame`,
  or named with the `ghee_` prefix, and rename an input's param with it,
  where an input named `time` kept the name on its param. Give the import's
  own reason when it refuses an ISF shader with `IMPORTED` images.

### Library and packages

- Read back a frame at a whole multiple of the render size with each pixel a
  square block, in `Renderer::render_into` and `render_scaled`.
- Take either rounding of v × 255 for a channel value v read back in the
  tests, since GLs round 0.5 apart: Mesa to 128, NVIDIA to 127.
- Keep the library's `assets`, `atlas`, `bench`, `colour`, `cparam`,
  `feeds`, `isf`, `report` and `sources` modules inside the crate.
- Ship NOTICE and the DejaVu font's license in the deb under
  `/usr/share/doc/ghee/`, with ghee's license in its copyright file, and
  LICENSE and NOTICE in the wheel.

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
