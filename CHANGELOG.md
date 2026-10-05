# Changelog

## Unreleased

### Scenes

- Declare image files in `assets`: a `.png`, a `.jpg`, or a `.raw` file of
  bare `u8` or little-endian `f32` pixels that states its `width`, `height`,
  `format` and `components`. Every pass samples an asset as
  `uniform sampler2D <name>_tex` with a `vec2 <name>_size`, nearest unless
  the asset sets `"sampling": "linear"`.
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
- Refuse a key that ghee does not take anywhere in a scene, where the scene
  loaded and ignored it, and name the key, where it sits, and the closest key
  that ghee takes, or the object that takes it: `"fill"` on a layer names the
  layer's text.
- Take as the name of a pass, a param, an array or an asset ASCII letters,
  digits and `_`, with a letter first and a letter or a digit last, where a
  pass took any name, and refuse a name that holds `__` or starts with
  `gl_`, which GLSL reserves, or starts with `ghee`, which ghee keeps for the
  names that it writes into a shader. The passes, the assets and the scene
  params share one set of names, and a pass's params and arrays take names
  unique in the pass and outside that set, so two passes may each declare a
  param `gain`. A filter's pass takes a name that nothing in the scene
  holds, as `dither1` beside a pass `dither`.
- Refuse `time`, `DT`, `frame`, `resolution`, `v_texCoord` and `iChannel0` and
  up as the name of a param, an array, an asset or a text, a name that starts
  with `glyph_`, which ghee keeps for the names that layers read, and a text
  name that starts with `ghee_`. Refuse a scene where two declarations make
  one GLSL name in a pass, such as a param `title_size` beside a text
  `title`, and name both.
- Take 16 passes, filter passes included, where the engine took 8, and refuse
  a scene past them with an error of its own. Take any number of assets,
  texts and layers, a channel or an array in each of a pass's 16 texture
  units, and names of any length, where the engine took 16 arrays in a scene
  and 8 channels and 4 arrays in a pass, and cut a name past 63 bytes. The
  README states every cap and its error.
- Name the pass and the line of the author's GLSL in a compile error, as
  `pass 'main', line 2 of its code:` or `line 2 of the common block`, ahead of
  the GL's own message, and print the error once, where it printed the GL's
  log twice with a line number in the assembled shader.

### Text

- Draw text on the GPU from a font. A scene declares `texts`, each with its
  words in `value`, its `font` and its `align`. A text that names no font
  draws from the built-in font, DejaVu Sans cut to ASCII, Latin-1 and the
  common punctuation. ghee declares a scene string param under each text's
  name, which holds the words, and new words from `set_param`, `/set`, the
  console or a procedural scene lay out again with no recompile. A param
  declared under a text's name refuses the scene.
- Take as a text's `font` a `.ttf` or `.otf` file, the name of a font asset,
  or a list that mixes both, and draw each character from the first font in
  the list whose cmap holds it. A character followed by U+FE0F draws from the
  first font with a colour glyph for it.
- Paint a text into a pass's buffer with the pass's `layers`, after the pass's
  own shader and in list order, as one quad over each glyph. A layer's
  `vertex` GLSL defines `glyph_place`, which puts each glyph in the pass, and
  its `fragment` GLSL colours the glyph, and both see the pass's preamble and
  the scene's common block. A layer with no `fragment` draws the words white.
  `blend` sets `normal`, `add`, `multiply` or `screen` for colour and alpha
  alike, where `multiply` and `screen` work over an opaque pass, `pad`
  reaches past the letters for a glow or an outline, and `name` names the
  layer in errors and in the console.
- Place a layer's words with no GLSL. `font_size` sets their pixels per em,
  and `at` puts their top left at a pixel of the pass, `[x, y]` with y down.
  A layer with neither fits the words inside the pass, less a margin of 5% of
  its shorter side, and centres them. A layer with `at` alone fits them right
  of and below that pixel. The fitted size follows the words, and two layers
  of one text fit to one size. A layer with `vertex` takes neither key.
- Let a pass with `layers` leave out its GLSL: it starts each frame
  transparent and holds its layers alone, for a later pass to read through a
  channel.
- Fill a layer's glyphs from an MTSDF atlas by default, one texture read a
  pixel, read the distance to the outline with `glyph_dist()` and
  `glyph_round_dist()` for outlines, glows and shadows, and the pixels per em
  with `glyph_font_size`. `"fill": "slug"` on a text computes Slug's exact
  coverage from the font's curves instead, at any size and with no atlas, and
  a slug layer that reads a distance fails with an error that names
  `"fill": "mtsdf"`. `atlas` sets the MTSDF atlas's texels per em in its
  `font_size`, and the field reaches as far as the largest `pad` of the
  text's layers, so a glow fits with no tuning.
- Fade a padded MTSDF layer to 0 at the edge of its pad, by the distance from
  the letters, so a glow ends along the letters' shape and never shows its
  quads as boxes. `fade` on a layer sets the band in em, a quarter of `pad` by
  default, and 0 turns the fade off. `glyph_fade` reads the factor in
  `fragment`. A slug layer refuses `fade`, and one that reads `glyph_fade`
  fails with an error that names `"fill": "mtsdf"`.
- Generate a font's atlas on every core at its first use: the glyphs of the
  words first, then a preload of ASCII, Latin-1 and the common punctuation in
  the room left. Cache the atlas in `$XDG_CACHE_HOME/ghee/atlas`. New words
  show in their first frame when the atlas holds their glyphs, and otherwise
  the text keeps its last words on screen until every glyph of the new ones is
  in. A setup and a render with `--timestep` wait for them. A full atlas
  draws an error box with an error of its own.
- Read the coverage of a text's words at any point in any pass shader, filter
  passes excepted, with `<name>_cover(p, font_size)`, and the block's size in
  em with `<name>_size`.
- Paint colour emoji in layers: a glyph with a PNG bitmap in a CBDT or sbix
  table, such as those of Noto Color Emoji, shows its bitmap, mipmapped and
  placed by the layer's `vertex`, and `<name>_cover` reads it as no ink. The
  last layer of a text in its pass paints them, so a shadow or a glow under
  it paints no copy, and `"emoji"` on a layer chooses otherwise.
- Keep the scene running through a text problem, print it once as an error,
  which `--quiet` keeps, and show it in the console and in `/stats` as
  `text_errors`. A font that fails to load drops out of its text's fonts, a
  glyph that ghee cannot draw shows an error box in its place, and words that
  do not fit leave the last words that did.
- Add `--demo lettering`, `--demo karaoke` and `--demo lens`, and rename
  `--demo text`, which rasterizes its words in Pillow, to `--demo marquee`.
  Add `ghee/examples/nowplaying.py`, an artist and a scrolling title that fit
  a 128x32 or a 256x64 panel.

### Python

- Declare an asset with `ghee.Asset`, a text with
  `ghee.Text(name, font, value, align, fill, atlas)`, and a layer with
  `ghee.Layer(text, fragment, vertex, font_size, at, fade, blend, pad, name,
  emoji)`, in `Scene(assets=..., texts=...)` and `Pass(layers=...)`. A text
  with no `font` draws from the built-in font, and a keyword left at None
  leaves its key to ghee's default, `pad` included. `mipmaps=True` on
  `ghee.Pass` builds its mip chain, and `visible=False` hides the pass.
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
  text's align and fill, a layer's blend and an arg's type.

### Procedural scenes

- Set a string param, and so a text's words, from a scene process with
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

- Edit a pass's layers. Each layer opens in a tab after its pass's tab,
  named by its pass and its name, `main › glow`, with its `vertex` and its
  `fragment` in editors and its text, blend and pad as fields with a hint
  each. `+` adds a layer to the open pass, and a double-click on its tab
  names it, as a pass takes its name.
- Show each text of a scene in a texts panel, with the layers and the passes
  that read it: its words, applied live, and its font and align, applied
  through the scene.
- Report the layer that holds a compile error in `/stats` as `error_layer`,
  with `part` set to `"vertex"` or `"fragment"`, and count `error_line` in
  that layer's GLSL.
- Take the scene that an edit is for in `scene=` on `/set`, in `name=` on
  `POST /scene` and in the body of `/save`, and refuse with 409 an edit for
  a scene other than the active one, which `/scenes` and `/stats` name as
  `active`. A page whose edit was refused shows the active scene as it runs,
  where a page left open on one scene wrote its copy into the scene that the
  console had switched to.
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
- Open a tab with a tap anywhere on it, keep the mute and remove buttons on
  the open tab alone, apart from its label, and ask before a pass, a layer or
  the common block goes, in a dialog of the console's own where Escape
  cancels and Enter removes.
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
