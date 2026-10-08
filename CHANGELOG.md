# Changelog

## Unreleased

### Names

A scene written for 0.4.0 needs these changes.

- Take Shadertoy's names for the uniforms of every pass: `iTime` where it
  was `time`, `iTimeDelta` where it was `DT`, and `iResolution` where it was
  `resolution`, a `vec3` now, with the size in pixels in `xy` and the pixel
  aspect, 1.0, in `z`. The varying `v_texCoord` is `uv`, with the same
  values. The old names are free for a param, an array or an asset.
- Set no param named `aspect` to the output's aspect. A pass reads
  `iResolution.x / iResolution.y`, and a pass with a size of its own takes
  the frame's shape from its scene: a Python scene reads the size that it
  renders at with `ghee.render_size()`.
- Give a pass its GLSL in `fragment`, as the text, where `code` held it, or as
  `{"file": <path>}`, a file beside the scene, where `shader` named it. The
  scene's `common` takes the same two forms, the text where
  `{"code": <text>}` held it and `{"file": <path>}` where
  `{"shader": <path>}` named it. A filter file holds its GLSL in `fragment`,
  and Python's `ghee.Pass` takes `fragment=` where it took `code=`.
- Read the length of a 1xN array as `<name>_count`, where it was
  `<name>_size`. `<name>_size` stays the size of an image asset and of a text
  layer's block.
- Name the pass of a `/set` with `pass=<name>`, where it took the pass's
  index, and leave `pass=` out for the scene param, or the param of the first
  pass that declares the name, as Python's `set_param` takes a name, where
  `pass=scene` named the scene's params. A name that no param answers to and
  a layer that the scene lacks read as Python's errors for them, which list
  the names that the scene takes.

### Rust API

A Rust host written for 0.4.0 needs these changes.

- Start a scene process with the size that it renders at, the required
  `size` of `SceneProcess::start`, where `set_render_size` set one size for
  every process and `ghee render` alone called it. `SceneProcess::start`
  also takes `reach`, how far beside the executable the scene's files
  resolve, and `edit`, a change to the scene document that the process
  writes. `SceneStore::start`, `open` and `resolve` take the size too, and
  `render_loop` hands each process its renderer's `scene_size()`, so a
  procedural scene under any host of the console reads `GHEE_SIZE`.
- Hand `render_loop` the renderer, made on the thread that calls it, and a
  `Pacer` or None, where it made its own renderer.
- Implement `ConsoleHost::set_layer`, which sets a live field of a layer for
  a `/set`, and `ConsoleHost::save`, which writes the working copy that the
  host holds, where `writable` said that a host saves and `save_scene` wrote
  the scene that a page posted.
- Build a `SceneStore` with `SceneStore::new(dir, args)` and
  `SceneStore::one(path, args)`, where both took `read_only`, and give it
  the directory of its state files in `state_dir`. `SceneStore::save`
  writes a scene's `Edits` to its state file, where it took a scene,
  `SceneStore::file` takes the place of `path`, and `store::apply_values`
  is gone with the `.values.json` sidecars.
- Return a `ConsoleResult` from `parse_query`, which holds the 400 of a
  malformed percent escape. `parse_scope` is gone, since `pass=` names the
  pass.
- Hand `server::coerce` the param's name first, which its error names.
- End an H264 stream for every viewer with `H264Fanout::fail(why)`, and
  read the reason with `H264Fanout::failure`, where `fail` took no reason
  and `is_dead` said that the stream failed.
- Hand `looper::should_draw` whether anyone takes the frames, as
  `watched`, which a `Loop` reads from `Loop::watched`: while nobody
  watches, a frame draws for a queued command and as the first frame alone.
- Hand `SceneController::on_param` the pass of the param that a host set and
  its value as a `ParamValue`, for a param of any type, where it took a
  string param's name and words, and hand `SceneController::on_layer` each
  live field of a layer that a host sets.
- Read a renderer's scene through `Renderer::with_scene`, where
  `Renderer::scene` was a public field.
- Hand `Scene::new` the scene's `assets`, `Pass::glsl` the scene's `assets`
  and the names of its text layers, and `Channel::from_dict` the name of the
  channel for its errors. A pass holds its GLSL in `Pass::fragment`, where
  `Pass::code` held it, and a color param's value is
  `ParamValue::Color(Rgb)`, where it held three numbers.
- Read the host's hardware with `console::feeds::hwinfo`, where it was
  `default_hwinfo`.
- Match `ghee::ffi` to the engine's header: `ghee_mp_add_pass` takes the
  pass's name and whether its buffer holds alpha, `Parameter` points at its
  name, and `ghee_get_timings`, `ghee_set_profiling` and `ghee_pack_frame`
  are gone. `ghee::ffi` binds every call that the engine added.
- Keep the library's `assets`, `atlas`, `colour`, `cparam`, `feeds`,
  `report` and `sources` modules inside the crate.

### Removed

- Drop the `ghee isf import` command and `ghee::isf`.
- Drop ghee's own panel drivers, `st7789` and `ssd1306`. A panel takes
  `ghee overlay`, and the kernel then drives it on `--display kms`.
  `--display` takes `kms` alone, `ghee render` and `ghee serve` take no
  `--display-opts`, and `ghee displays` lists the cards with a connected
  display.
- Take `"kms"` or None alone as the display of `ghee.open`, and `rgb565`
  alone in its `opts`.
- Drop `ghee::drivers` from the library, with the frame packers
  `engine::pack_frame`, `pack_frame_into` and `packed_size`,
  `Renderer::pack` and `Renderer::pack_into`, `stats::DEST_PANEL`, and
  `native_format` and `hwinfo` in the console's `server::Options`.
- Drop the `ghee bench` command and `ghee::bench`.
- Drop the draw and readback timing of a frame, `Renderer::set_profiling`
  and `Renderer::timings`.
- Drop the console's `<scenes>/modified/` folder, its copies of JSON scenes
  and its `.values.json` sidecars of procedural scenes: ghee reads none of
  them. Fold a copy's changes into the scene's file by hand, or redo them in
  the console and save.
- Drop the body of `POST /save`: `/save?name=` saves the working copy that
  ghee holds.
- Drop the console page's working copy in the browser's localStorage, which
  the page posted over the scene at each load, and the scene that its save
  posted. The page reads no edit that a browser kept there.
- Drop `ghee::output::AdaptivePacer`.

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
  `y`, `opacity`, `blend` and `visible` as a text layer does, and its
  `color` tints the picture. An image layer takes no effects and gives a
  pass no GLSL name of its own, so a pass reads the image as the asset's
  `<name>_tex`.
- Resolve an asset or a font that a scene names from the directory of the
  scene file or of a procedural scene's executable, then from each directory
  on `$GHEE_ASSETS`. A path may be absolute or leave that directory, except
  under the console, and the error for a file that no directory holds names
  `$GHEE_ASSETS`.
- Read `uniform int iFrame` in every pass, the number of the frame drawn: 0
  on the first frame, as in Shadertoy.
- Read the size of the buffer that each channel reads in
  `uniform vec3 iChannelResolution[4]`, as in Shadertoy: the width and height
  in `xy`, 1.0 in `z`, and a pass with more than four channels takes an entry
  for each.
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
- Render a pass at a share of the frame with `"scale"`, one number for both
  sides or `[x, y]`, beside `"size"` in pixels, and with `scale=` on
  `ghee.Pass`. The frame is the render size turned by `--rotate`, a scaled
  side rounds to the nearest pixel, 1 at least, and the output pass ignores
  a scale as it ignores a size. A pass that names both fails to load, `pass
  'blur' has both size and scale, and takes one`, and so does a scale that
  is not a positive number or two.
- Keep a color param in the form that it came in, `"#rrggbb"` or
  `[r, g, b]`, through `GET /scene`, `/set` and a save. The console sends the
  colour picked in its swatch as `#rrggbb`, so it saves as hex.
- Write each float of a scene document in the shortest form that reads back
  as the same 32-bit float, so a drawn width reads `2.5804138` in `GET
  /scene`, where it read `2.580413818359375`.
- Draw rects and lines with shape layers, which hold `"shape": "rect"` or
  `"line"` and take no effects. A rect fills `width` by `height` pass pixels,
  the pass's own by default, with its `color`, white by default, draws its
  `border`, `{"width", "color", "visible"}`, inside its edge, and rounds its
  corners by `border_radius`. A line runs through `points`, each `[x, y]` a
  pixel as Pillow counts them, `stroke_width` pixels thick, and its `x` and
  `y` shift every point. Every edge is antialiased from the distance to the
  shape. One shader draws every shape, consecutive shape layers of one blend
  in one draw, so a shape layer that changes or hides compiles nothing. Every
  field but `shape` and `blend` is live, none takes a rect's `width` or
  `height` back to the pass's own, and a rect's border takes an effect's
  place: `/set?layer=frame&effect=border&field=color`, `frame.border.color` on
  the wire, and a line's points as an array in message kind 0, `scope.points`,
  x and y in turn.
- Take CSS's colour names wherever a scene takes a colour, a layer's and an
  effect's `color`, a gradient's `from` and `to`, and a color param: the 148
  named colors of CSS Color 4 and `transparent`, in any case, which the
  scene keeps as written through `GET /scene`, `/set` and a save.
  `transparent` paints nothing, so letters in it still draw their effects
  and a gradient to it fades out, and a color param reads it as black. A
  colour that ghee refuses names every form that it takes, names among
  them.
- Refuse a key that ghee does not take anywhere in a scene, where the scene
  loaded and ignored it, and name the key, where it sits, and the closest key
  that ghee takes, or the object that takes it: `"fragment"` on a layer names
  its pass, and `"font"` names `font_family`. A key one edit from a key of
  its own object names that key first, so a scene's `"filter"` names
  `filters`.
- Take as the name of a pass, a layer, a param, an array or an asset ASCII
  letters, digits and `_`, with a letter first and a letter or a digit last,
  where a pass took any name, and refuse a name that holds `__` or starts
  with `gl_`, which GLSL reserves, or starts with `ghee`, which ghee keeps
  for the names that it writes into a shader. The passes, the layers, the
  assets and the scene params share one set of names, and a pass's params
  and arrays take names unique in the pass and outside that set, so two
  passes may each declare a param `gain`. A filter's pass takes a name that
  nothing in the scene holds, as `dither1` beside a pass `dither`.
- Refuse `iTime`, `iTimeDelta`, `iFrame`, `iResolution`, `iChannelResolution`,
  `uv` and `iChannel0` and up as the name of a param, an array or an asset. A
  layer may take any of them, since every GLSL name that it gives a pass
  carries a suffix. Refuse a scene where two declarations make one GLSL name
  in a pass, such as a param `title_size` beside a layer `title`, and name
  both.
- Refuse a GLSL keyword, such as `half`, the name of a GLSL function, such
  as `mix`, and the name of a function that the pass's GLSL or the common
  block defines, `main` among them, as the name of a param that holds no
  string, where the shader failed to compile.
- Take 16 passes, filter passes included, where the engine took 8, and refuse
  a scene past them with an error of its own. Take any number of assets and
  layers, a channel or an array in each of a pass's 16 texture
  units, and names of any length, where the engine took 16 arrays in a scene
  and 8 channels and 4 arrays in a pass, and cut a name past 63 bytes. The
  README states every cap and its error.
- Refuse an array or an image past the GPU's own texture size, 2048 pixels on
  a side on a VC4, with an error that names it, its size and the GPU's,
  where the GPU took no texture and the pass read black.
- Name the pass and the line of the author's GLSL in a compile error, as
  `pass 'main', line 2 of its fragment:` or `line 2 of the common block`,
  ahead of the GL's own message, on Mesa's GL and on NVIDIA's alike, and
  print the error once, with no blank line after it, where it printed the
  GL's log twice with a line number in the assembled shader.
- Put `compile error:` ahead of a GLSL compile or link error alone, in
  `ghee render` and `ghee.open`, where every scene that ghee refused carried
  it, and mark such an error with `Renderer::compile_error` in Rust.

### Text

- Draw text on the GPU from a font in text layers. A pass's `layers` paint
  over its buffer after its own shader, in list order, as one quad over each
  glyph. A layer needs a `name`, holds its words in `text`, and takes
  `font_size`, `font_family`, a `.ttf` or `.otf` file or the name of a font
  asset, and `text_align`. A layer with no `font_family` draws from the
  built-in font, DejaVu Sans cut to ASCII, Latin-1 and the common
  punctuation. New words in a layer's live `text`, from Python's `layers`,
  `/set`, the console or a procedural scene, lay out again with no
  recompile.
- Space a layer's glyphs by their advances and the font's pair kerning, the
  pair adjustments of the `kern` feature in its GPOS table, as HarfBuzz
  applies them, or, in a font without that feature, its `kern` table.
- Place a layer's words with `x` and `y`, the pass pixel of the block's top
  left, y down, and `font_size`, their pixels per em. An axis left out
  centres the block on it. A layer with no `font_size` takes the largest size
  at which the words fit the pass, less a margin of 5% of its shorter side,
  and on a given axis the room past the given pixel, less the margin at the
  far edge. The fitted size follows the words.
- Colour a layer with `color`, any colour that a scene takes or a gradient
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
  never covers the next letter. Below an `opacity` of 1, a layer of the
  `normal` blend draws its letters over their effects first and lays the
  result on the pass at that opacity, so the effects never show through the
  letters. `"visible": false` bypasses an effect and
  keeps its values, the slug render skips the three that read the distance,
  and ghee works out how far a layer's quads and its atlas's field reach
  from its effects, and fades the effects out at the edge of their quads.
- Set a layer's live fields with no rebuild, `Renderer::set_layer` in the
  library: `x`, `y`, `color`, `opacity` and `visible`, a text layer's
  `text`, its words, an image layer's `image`, which takes another image
  asset of the scene, or an image file that loads once and stays until the
  next setup, at that image's size, and an effect's `visible`, `color`,
  `strength`, `sung`, `speed` and `rate`. None takes `x`, `y` and `color`
  back to their defaults. Every other field changes with a rebuild, so a live
  set of one is refused with the live fields named, and a set that names an
  effect or a field that the layer lacks is refused with the layer's effects
  or its live fields named.
- Move `wave`, `scroll` and `blink` by a phase that ghee keeps for each
  layer and advances each frame by the effect's `speed` or `rate` times the
  frame's time step, so a new speed carries the motion on from where it is.
  Under `--timestep` the phase runs from 0 at the scene's start.
- Draw small text as Pillow draws it. `"render": "hinted"` hints each glyph at
  the layer's size with FreeType 2.14.3, the version in Pillow's wheels, by
  the font's own instructions or FreeType's autohinter, rasterizes it there,
  and lands its pen on a whole pixel and its baseline on a whole row, as
  Pillow does, so the layer lights the pixels that Pillow lights for the same
  words, font and size. `"render": "mono"` draws the glyphs that FreeType
  hints and draws for Pillow's 1-bit images, for an on/off panel such as the
  SSD1306. `auto`, the default outside the pixel font, never picks it: it
  takes `hinted` below 128 pixels per em, where a hinted bitmap outgrows its
  MTSDF cell, and `mtsdf` from there, by the layer's size when it builds. A
  hinted layer's shadow, glow and stroke draw from its MTSDF atlas, and a
  fitted hinted size rounds down to a whole pixel. The built-in DejaVu Sans
  carries its TrueType instructions. ghee builds FreeType from
  `vendor/freetype`, and its license reads `MIT AND FTL`.
- Draw a layer's glyphs from an MTSDF atlas under `"render": "mtsdf"`, one
  texture read a pixel. `"render": "slug"` computes Slug's exact coverage from
  the font's curves instead, at any size and with no atlas. The slug layers
  that no pass reads through `<name>_cover` share one texture of curves, which
  takes a glyph at the first words that draw it, so new words of glyphs
  already drawn upload nothing, and a layer in the pixel font takes new
  two-digit words in 0.35 ms on a Pi Zero 2 W.
- Draw consecutive shape layers and slug text layers with no effect, of one
  blend, in one draw through one program, which reads each text's settings
  from its glyphs' vertices. A glyph drawn there can land an edge pixel one
  level from where the text's own program lands it, since the program
  reads the font size from the vertex where the text's own holds a
  constant. A text whose glyphs' quads cover more than 800 pixels, or whose
  words take a texture of their own, keeps a draw of its own, and so does a
  shape of more than 3400 pixels beside a text, since on a Pi Zero 2 W that
  program costs 28 ns more for each pixel of a glyph and 5.6 ns more for
  each pixel of a shape.
- Draw a text layer from Pixel Operator 8, a second built-in font, with
  `font_family: "pixel"`, as the font's own pixels: such a layer takes
  `render` slug unless it names one, a size that fits rounded down to a
  whole multiple of 8 px, 8 at the least, and its block, each pen and each
  baseline on whole pixels, so at 8 px and its multiples every pixel is
  fully on or off. A `font_size` stays as given. A font asset named `pixel`
  stays the scene's own.
- Generate a font's atlas on every core at its first use: the glyphs of the
  words first, then a preload of ASCII, Latin-1 and the common punctuation in
  the room left. Cache the atlas in `$XDG_CACHE_HOME/ghee/atlas`. New words
  show in their first frame when the atlas holds their glyphs, and otherwise
  the layer keeps its last words on screen until every glyph of the new ones
  is in. A setup and a render with `--timestep` wait for them. A glyph of
  new words that finds the atlas full takes the room of the glyphs that no
  words on screen draw, the longest unused first, then a second page of the
  atlas, and an atlas whose two pages are full of the glyphs on screen draws
  an error box with an error of its own. An MTSDF layer binds both pages,
  two of its pass's texture units. A hidden layer builds its atlas when it
  first shows.
- Generate a layer's atlas at the detail of its size, rounded up to 32, 48,
  64, 96 or 128 texels per em, and 128 for larger text, so layers of near
  sizes share one atlas. The size is the layer's `font_size`, or with none
  the largest size at which a line of words fits its pass, so shorter words
  keep their detail. A layer whose effects reach past its letters stops at
  a lower step, where its cells span no more texels than plain text's at
  128: a 0.3 em glow stops at 64.
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
  the emoji font once, at the first words that need it, which wait for the
  read with the layer's last words on screen, and every layer of every
  scene shares it, and the console marks a scene stale when the font shows
  up or changes. An emoji, a glyph with a PNG bitmap in a CBDT or sbix
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
- Draw text in a CFF2 font that holds no variation store. ghee builds
  ttf-parser from its main branch, vendored in `vendor/ttf-parser`.
- Add `--demo lettering`, `--demo karaoke` and `--demo lens`, and rename
  `--demo text`, which rasterizes its words in Pillow, to `--demo marquee`.
  Add `ghee/examples/nowplaying.py`, an artist and a title laid out from
  `ghee.render_size()`, so one scene fits a 128x32 and a 256x64 panel.

### Python

- Hand the last frame to Pillow with `h.image()`, an RGBA image with
  straight colour. Pillow stays an optional dependency, and `image()`
  raises ImportError without it.
- Read the size that a scene renders at, as `(w, h)`, with
  `ghee.render_size()`: ghee's render size, or in a direct run its
  `--size`, else the size of the display that it drives, turned by its
  `--rotate`.
- Work out a direct run's size args given as a share, such as `half`, from
  the display that the run drives when it has no `--size`, where they took
  320x240. A run with no `-o` and no `--serve` drives the connected display,
  so its scene takes that display's size.
- Require CPython 3.10 or newer: one abi3 wheel per architecture serves 3.10
  and up, and the deb depends on python3 3.10 or newer.
- Set a layer's live fields on a `ghee.open` handle with no rebuild, through
  `h.layers` by attribute or by key, `h.layers.title.x = 10` or
  `h.layers["title"]["karaoke"]["sung"] = 12`, or through
  `h.set_layer(name, field, value, effect=None)`. `text` sets a text layer's
  words, `image` an image layer's image, and a rect's border takes an
  effect's place, as `h.layers.frame.border.color`. None takes `x`, `y` or
  `color` back to its default. A field that changes with a rebuild, an
  unknown layer, effect or field, and a value that the field does not take
  raise ValueError, and an unknown layer's error names the scene's layers.
  On a Pi Zero 2 W a set of `x` takes 20 us, a colour 59 us and a karaoke's
  `sung` 39 us.
- Set the layers of a scene script from a tick or a feed with
  `ghee.layers`, as on a handle. Under ghee, each set goes out with the
  tick's answer, checked against the scene first, so a set that ghee would
  refuse raises ValueError. In a direct run, each set goes to the handle.
- Declare an asset with `ghee.Asset`, which takes `sampling`, and a `.raw`
  file's `width`, `height`, `format` and `components`, and declare a text
  layer with `ghee.Layer(name, text, visible, x, y, color, opacity, blend,
  font_size, font_family, text_align, render)` and a keyword for each
  effect, a dict of its fields, an image layer with `image`, `width` and
  `height`, or a shape layer with `shape`, `width`, `height`, `border`,
  `border_radius`, `points` and `stroke_width`, in `Scene(assets=...)` and
  `Pass(layers=...)`. A keyword left at None leaves its key to ghee's
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
  for image files or raw frames on stdout, `--display`, `--serve`,
  `--port` and `--host` for the console,
  `--stats` for the timing summary, and `--rotate`, `--brightness` and
  `--dri`, where `--output` named the display and `--stats` the console's
  port. With no `--frames`, `-o` renders one frame,
  as `ghee render` does, and a display or the console runs until stopped.
- Scan a direct run's display out in 16-bit color with `--rgb565`, as
  `ghee render --rgb565` does.
- Return the params that the console changed since the last tick, each value
  as a scene file writes it and a text layer's words as its `text`, in the
  tick's `params`: from `h.frame()` and `h.skip()`, and in a direct run's
  `@ghee.tick`, where that tick's `params` stayed empty. The first tick holds
  them all, as a scene script's tick does under ghee, and a set from the host
  itself stays out.
- Hand the time since the tick before as the `dt` of a tick from
  `h.frame()` or `h.skip()` and of a direct run's tick, as a scene process
  takes it on the wire, where the tick held the period of the frame before
  and a direct run's first tick held 0.
- Name the display of `ghee.open` with `display=` and the console's port
  with `serve=`, the words of `ghee render`'s flags, in place of `output=`
  and `stats=`, which now raise TypeError. `dri=` picks the card, and
  `rotate=` turns the frame clockwise on its way to every output, as
  `--rotate` does.
- Open a screen with no scene: `ghee.open()` makes one pass, `main`, that
  clears to black.
- Read and set a KMS display's backlight through `h.brightness`, 0 to 1, as
  `--brightness` sets it. It reads None for a display with no backlight
  device, and a set raises ValueError there and on a screen with no KMS
  display. Closing the screen puts the earlier level back.
- Draw words, pictures, boxes, circles and lines on a screen from Python,
  with no GLSL: `with h.draw() as d:` opens one frame, `d.text`, `d.image`,
  `d.rect`, `d.circle`, `d.line` and `d.lines` each write a layer from the
  keywords of its layer and `name`, and leaving the block sends ghee what
  changed and draws the frame. A call keeps its layer by the line that makes
  it and its count from that line in the frame, or by its `name`, which
  names the layer, and a call with no name takes the console's naming,
  `text`, `text2`. A live field changes live, any other change builds that
  layer alone again, a call that runs for the first time adds its layer, a
  layer whose call stops hides, and an effect that a call stops naming is
  bypassed. Two calls that draw one layer in a frame raise ValueError. On a
  Pi Zero 2 W, with rects and two-digit words in the pixel font and half of
  the calls changed each frame, a frame of 50 calls draws at 193 fps on a
  128x32 SSD1306, and a frame of 200 calls at 181 fps, after a first frame
  of 1.3 s that adds the 200 layers.
- Measure words with `h.measure(words, font_size=None, font_family=None)`,
  and with `ghee.measure` in a scene script: the width and height in pixels
  of the block that ghee lays them out in, at the size that fits the pass
  when `font_size` is None.
- Write a text's gradient with `ghee.gradient(start, end, angle=0)`.
- Stop a direct run with ghee's one line and exit status 1 when ghee refuses
  its scene or its display, where a traceback ended it, and refuse more than
  one frame into one file name, as `ghee render` refuses it. An exception of
  the script's own keeps its traceback.
- Raise ValueError from `ghee.set_param` in a Python scene for a param that
  the scene does not declare or a value that does not fit its type, under
  ghee as in direct mode, and name the param in the type error of
  `set_param` and of the console.
- Ship type stubs for `ghee.open` and `Handle`, mark the package typed, and
  type the closed string sets of the shim as `Literal`: a param's type, a
  layer's text_align, render, blend and shape, and an arg's type.
- Draw from a scene script with `ghee.draw()` in a tick or a feed, as on a
  screen, and the frame of the tick draws the calls. Under ghee, a layer
  that a call adds or changes goes out with the tick's answer, checked
  against the scene first, so a call that ghee would refuse raises
  ValueError. `ghee.run()` with no scene hands ghee one pass, `main`, that
  clears to black, and the first tick's drawing grows it.
- Add `--demo stats`, the time and the load of the machine drawn on a
  128x32 panel through `ghee.draw()`.

### Procedural scenes

- Carry every param that changed in a tick, each value as a scene file
  writes it, where a tick carried the string params alone, so a float that
  the console sets reaches the scene's process. A param goes under its bare
  name, or as `pass.param` when the bare name sets another param. The first
  tick holds every param.
- Carry a text layer's words in a tick's `params` as its `text`, such as
  `title.text`, the name that sets them in message kind 3.
- Name the cause when a scene's script cannot start, `the file has no #!
  line` or `the interpreter '/usr/bin/python9' on its #! line does not
  exist`, where the system's error read `Exec format error` or `No such file
  or directory`.
- Count a scene process's ticks from 0 at its first tick, `frame` and
  `time` alike, where they counted from ghee's first frame. The console
  starts a new process at each switch, so the ticks start again there.
- Hand the time since the tick before as a tick's `dt`, a quarter second at
  most and 1/60 for the first tick, so the ticks' `dt` add up to the time
  passed, skipped frames included, while each tick follows the one before
  within a quarter second, and a frame drawn after a drawn frame takes it as
  its iTimeDelta, where `dt` held the period of the frame before. A pinned
  timestep is the tick's `dt` as given, where a quarter second capped it.
- Set a live field of a layer from a scene process. Kind 1 takes
  `layer.field` and `layer.effect.field`, such as `title.x` and
  `title.karaoke.sung`, with one value, 0 or 1 for `visible`, three for a
  color, seven for a gradient or none for the default, and kind 3 takes
  `title.text` with a text layer's words and `icon.image` with the name of
  an image asset or an image file. A message whose first part names no
  layer sets a param, and a field that the scene refuses logs once.
- Set a string param from a scene process with message kind 3.
- Hand a scene script the ghee-render wheel that sits beside the `ghee`
  command, in `lib/ghee` next to its `bin` directory, where uv took the
  newest ghee-render on PyPI, which a dev build outdates: ghee puts an index
  of that wheel alone first on the scene's `UV_INDEX` and sets
  `UV_INDEX_STRATEGY=first-index`, and numpy and the rest still come from
  PyPI. A dev build with no wheel there makes uv refuse every ghee-render,
  and ghee says once which directory lacks the wheel. Two wheels of ghee's
  version there keep the scene from starting. The bundled examples extract
  to a new directory for each wheel, so uv resolves them again.
- Take a colour as text in message kind 3, `#rrggbb` or a CSS colour name,
  for a color param and a layer's colour, and a text layer's gradient as its
  JSON object, so a name keeps its form and `transparent` its alpha. The
  Python shim sends a colour name, a layer's colour given as text and a
  gradient that way, and `ghee.set_param` checks a colour name through the
  native half of the package.
- Wait for a lockstep scene process to answer its first tick, with the time
  pinned or not, and under a fixed `--timestep` each tick, so the params and
  arrays sent for a tick land in that tick's frame, where they landed a frame
  later or more. Message kind 5 ends an answer, `"lockstep": true` in the
  handshake promises one for every tick, and ghee sets `GHEE_LOCKSTEP=1` in
  the scene process to say that it reads both. The Python shim sends both when
  it finds `GHEE_LOCKSTEP`, and under an older ghee it writes the wire as
  before.
- Log once a param message from a scene process that names no param of the
  scene or does not fit the param's type, where it changed nothing in
  silence.
- Add a layer to the running scene from a scene process with message kind
  6, named for the pass, its payload the UTF-8 JSON `{"index": n, "layer":
  {...}}`, and give a layer a new look with kind 7, named for the layer, its
  payload the layer's UTF-8 JSON. Every other layer and every pass carry on
  with their programs, arrays and moving effects, a look that changes live
  fields alone sets them live, and a layer that a scene file would refuse
  there logs once. ghee applies the sets of a process in the order that they
  arrive, and a message kind past 10 ends the scene.
- Give several layers new looks in one message, kind 8, its name empty and
  its payload the UTF-8 JSON list of the layers. `ghee.draw` in a scene
  that ghee runs sends a frame's looks that change live fields alone as one
  kind 8, which ghee applies as a Python screen applies its drawing, where
  it sent a kind 7 for each changed call. On a Pi Zero 2 W's 128x32
  SSD1306, a script that draws 200 calls under `ghee render`, rects and
  two-digit words with half of them changed each frame, draws at 166 fps,
  where it drew at 27.
- Take a layer out of the running scene with message kind 9, named for the
  layer, with no payload, and move a layer among the layers of its pass
  with kind 10, named for the layer, its payload the UTF-8 JSON `{"index":
  n}` of its place after the move.

### Console

- Print the console's address once its renderer starts, and end `ghee serve`
  with the renderer's error, printed once, when it cannot start, where the
  address printed first and the error twice.
- Draw only while someone watches the console: a page's preview, a viewer
  of `/stream` or `/h264`, or a `GET /frame.png`. While nobody watches, a
  console draws once for each edit and sleeps in between, where one with no
  display drew as fast as it could. A console that drives a display draws
  every frame.
- Show what ghee uses in the console's render info while it is open: the
  process's resident set, its GPU memory and how busy it keeps each queue
  of the GPU, the machine's free memory and a Pi's SoC temperature.
  `/stats?usage=1` gives them as `rss_mib`, `gpu_mib`, `gpu_busy`,
  `machine_free_mib` and `soc_celsius`, read at that request alone, and
  null where the machine does not report them, as the GPU's on a Pi with
  vc4. `--stats` prints the same figures under its summary when a render
  or a console ends, with the GPU's busy over the run.
- Edit each pass's layers. A pass shows its tab row, with the open pass's
  rename, eye and remove at its end, then its inputs, with a chip for each
  text layer's `_cover`, each image asset's `_tex` and each array, then its
  channels on a row of their own, then its layers, each collapsed when a
  scene or the page loads, and its shader under a line. The layers list
  front first: the last one in the scene's `layers` paints on top and shows
  at the top, where `+ layer` adds a text or an image layer. A layer shows a
  one-row header with its blend, render, opacity, move, rename, eye and
  remove, then its settings, and a text layer adds `+ effect` and one row for
  each of its effects.
- Set a live field from the console with no rebuild, as it changes. Every
  other field applies when the cursor leaves it.
- Set a live field of a layer with `/set?layer=title&field=x&value=10`, and a
  field of one of its effects with `effect=` too, in `ghee serve`, `--serve`
  and `ghee.open(serve=)`, with no rebuild. `field=text` sets a text layer's
  words. The value reads as the field takes it, `none` included, and a field
  that changes with a rebuild answers 400 with the live fields named.
- Rename a pass or a layer with its pencil, in the console's dialog, where a
  double-click on a pass's tab renamed it.
- Hide a pass with its eye, which the scene keeps as the pass's `visible`,
  where the eye muted the pass in the page alone. A hidden pass, a hidden
  layer and a bypassed effect dim the same way. A hidden pass and a bypassed
  effect stay editable. A hidden layer keeps its content editable, a text
  layer's words, font_family and text_align and an image layer's image, and
  locks every setting that only changes how it draws.
- Show a colour as its swatch alone, in the Parameters panel too, and a
  number field with no spin buttons.
- Show a CSS colour name in its swatch as its colour, and `transparent` in
  the "none" look, which picks a colour when tapped.
- Keep the shader open or closed under its line as the user left it, one
  choice for every scene, kept in the browser across a reload. A compile
  error opens the shader of the pass that holds it and leaves the choice.
- Show a procedural scene's passes and layers with no add, move, rename or
  remove, since its process declares them.
- Edit shape layers: `+ layer` adds a rect named `rect`, then `rect2`, and a
  shape layer shows its shape, its color, `x y`, then a rect's `width
  height`, its border with an eye and a remove, and its `border_radius`, or a
  line's points, one `x y` to a row, and its `stroke_width`. A closed shape
  layer shows its colour in its header. A rect's fill turns off with a ×,
  which writes `transparent`, and switching a layer's shape keeps the other
  shape's settings for a switch back. Every field but `shape` and `blend`
  changes live.
- Show the layers that a drawing adds as they come and change, in `ghee
  serve` with a scene script that draws and in `ghee.open(serve=)`.
  `layer_changes` on `GET /stats` counts the changes that the program behind
  the scene makes to its layers, and the page reads the scene again when the
  count moves. A layer that the scene changed takes the scene's look in the
  page, and an edit to a drawn layer lasts until the drawing next gives it a
  look.
- Keep each layer of a procedural scene through `POST /scene`: a layer of the
  running scene that the posted scene leaves out stays, in its place, so an
  apply keeps the layers that a drawing added.
- Give layers a new look through `POST /scene` in `ghee.open(serve=)`, which
  lists its scene in `procedural` and refused every post. A post that
  differs in anything but its layers and its param values answers 403.
- Keep a hidden filter's pass hidden when the console posts the scene.
- Hold a working copy of each scene that the console opens, which every
  page reads and every edit changes at once, through a switch away and back
  and a reload of the scene's files, until ghee stops. `GET /scene` answers
  the working copy, of any scene, where it read a scene other than the
  active one from disk, and `?defaults=1` answers the scene's files.
- Save the console's edits of a scene, what differs from the scene's files,
  in a state file outside the scenes folder, under `$XDG_STATE_HOME/ghee/`
  or `--state-dir DIR` on `ghee serve` and `ghee render --serve`, and lay the
  state file over the scene's files at each load. New files for a scene keep
  its saved values, and a value whose pass, layer or param is gone is
  skipped with a log line. An edit of GLSL or of a list holds the hash of
  what it was made against, and once the files change that item, the files'
  version runs and `GET /edits` lists the edit as set aside until `POST
  /aside` answers keep or default. `GET /edits` gives the working copy's
  edits and whether they are saved, and `POST /reset` takes the working copy
  back to the files. Every console that ghee runs saves, `--demo` included,
  and a console hosted by Python saves nothing. A state file that ghee
  cannot read moves to `<file>.unreadable` beside it, with a log line.
- Lay a procedural scene's saved values, GLSL, channels and filter list over
  the scene that its process writes, at each start. The process keeps which
  passes, layers and params exist and their order, and a value that it sets
  after the start stays its own, unsaved.
- Give each scene a revision, in the `ETag` of `GET /scene` and in `rev` on
  `/stats` and `/edits`. An edit that carries `rev=` and `vid=` answers 412,
  `this scene changed since you loaded it`, when another client changed the
  scene since, and a switch that carries `rev=` answers 409 when the console
  is on another scene.
- Take the scene of the last switch as the active scene everywhere, one that
  failed to compile included, where `GET /scene` with no name and the
  running document followed the last scene that compiled. ghee keeps drawing
  the last program that compiled. An edit while the active scene loads
  answers 503, `scene 'clocks' is loading`, where `/set` answered 404.
- Name the request once in a refusal, `bad /scene: pass 'main' ...`, where it
  read `bad /scene: scene: ...`.
- Start `iFrame` and `iTime` at 0 at each switch, reloads included, with
  `iTimeDelta` at 1/60 on the first frame, and keep them running through an
  apply.
- Cap the render loop's frame rate with `--fps` on `ghee serve`, and keep the
  cap that `ghee render --serve --fps` takes, where it ran uncapped.
- Serve the layer format at `GET /format`: each effect's fields with their
  defaults and whether they change live, the same for a rect's border, the
  live fields of each kind of layer, what a field left out takes, each
  shape's own keys, the built-in fonts, and the render that a text layer
  takes in each font when it names none.
- Name the board and its GPU in `GET /hwinfo` from the device tree, so a
  Pi 5 reads Raspberry Pi 5 and VideoCore VII, with its memory and, on a
  Raspberry Pi, its GPU memory and temperature.
- Mark each value that differs from its default, as `GET /edits` lists the
  working copy's edits: a param, a layer's setting, a field of an effect or
  a border, the blend, render and opacity of a layer's header, a pass's size
  and the shader's line take the mark colour on their key. A click on a
  marked key takes the value back to its default at once, and under a mouse
  the key shows a dotted underline and a `↺` after it. The shader's `↺`
  asks first, in the console's dialog. A name, a text layer's words and an
  eye take no mark.
- Take a scene back to its defaults with `back to defaults`, where `reset`
  stood, which asks first and greys out while the working copy holds no
  edit. Save lights while the working copy holds edits that the state file
  lacks, and greys out once they are saved.
- Ask once about each edit set aside when the page opens its scene, in the
  console's dialog, titled with the item, as `Default post shader updated`:
  `keep mine`, first in focus, or `update to default`. Escape and the scrim
  do nothing.
- Show `this scene changed since you loaded it · reload` when ghee refuses a
  stale edit, and read the scene and its edits again at a tap. With no edit
  in flight, the page reads the scene again when `rev` on `/stats` moves, and
  leaves alone the layer in use. `files changed · reload` keeps the edits.
- Size a pass in its size row: the frame, pixels, or a scale of the frame,
  applied on release. The row is locked on the pass that draws last and on
  a procedural scene.
- Take the effects' fields and defaults, a border's defaults, the live
  fields, what an empty field takes, each shape's keys, the built-in fonts
  and each font's render from `GET /format`, and the device's rows of
  the render info from `GET /hwinfo`. font_family lists `built-in`, `pixel`
  and the scene's font assets, and a number shows 3 decimals at most until
  it is edited.
- Give up on a `/stats` poll after 5 s, and poll again after any answer or
  none, where a hung answer stopped the page's polling.
- Show a param set through `set_param` or `/set` in `GET /scene` of
  `ghee.open(serve=)`, and publish what the host changes with the frame that
  shows it, so a script that draws twice within 0.2 s and stops shows its
  last drawing.
- Take the scene that an edit is for in `scene=` on `/set` and in `name=` on
  every other edit, and refuse with 409 an edit for a scene other than the
  active one, which `/scenes` and `/stats` name as `active`. A page whose
  edit was refused shows the active scene as it runs, where a page left open
  on one scene wrote its copy into the scene that the console had switched
  to.
- List in `procedural` on `GET /scenes` the scenes that a program writes,
  an empty list when there are none.
- Build a posted scene's filters on the server: `POST /scene` rebuilds a
  filter pass from its filter, keeping its place and param values, and
  appends a filter that the `filters` list names without a pass. An unknown
  filter is a 400, and the reply is the scene as it will run.
- Parse a `/set` value by the param's declared type alone, and ignore
  `type=`, which overrode it. Name a param that the scene lacks.
- Stream the MJPEG preview of `ghee.open(serve=)` when the frames hold
  alpha, where the stream ended before its first frame.
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
- Answer 400 to a query with a malformed percent escape, `bad query: '%zz'
  is not a percent escape`, where ghee kept the escape as text.
- Cap a chunked POST body at 1 MiB, as a body with a Content-Length is
  capped, with the same 413, `body over the limit of 1048576 bytes`, where
  ghee read a chunked body whole.
- Answer 400 with `bad body:` and the error when the read of a POST body
  fails, where ghee handled the part that it had read.

### Displays and H264

- Write the I2C bus rate and the panel's height in `ghee overlay ssd1306`,
  which needs `height=32` or `height=64`, where it wrote a 128x32 setup with
  no options. `speed` sets `dtparam=i2c_arm_baudrate`, the rate of the whole
  bus, 1000000 by default. `address` passes the panel's I2C address to the
  overlay, where the overlay took `addr` and refused any address but 0x3C.
  A 128x64 module takes the overlay's own wiring.
- Set the SPI clock of `ghee overlay st7789` with `speed`, where it took
  `baudrate`, so every panel that `ghee overlay` sets up takes its bus clock
  as `speed`.
- Pass a panel's options to `ghee overlay` with `--opts`, where it took
  `--display-opts`.
- List the panels that `ghee overlay` takes when it is handed another, where
  it sent the reader to `ghee displays`.
- Write the kernel's setup for an SH1106 with `ghee overlay sh1106` on I2C
  and `ghee overlay sh1106-spi` on SPI, for an SSD1322 with `ghee overlay
  ssd1322`, and for an ILI9341 with `ghee overlay ili9341`. The SH1106 and the
  SSD1322 take an overlay of ghee's own, written as `ghee-<panel>.dts` for
  dtc, and the ILI9341 takes panel-mipi-dbi's firmware, as the ST7789 does.
  `-o` names either file, and a panel that takes neither refuses it with
  `a panel that takes no file`.
- Turn the frame clockwise on its way to every output with `--rotate 90`,
  `180` or `270` on `ghee render` and `ghee serve`, and
  `Renderer::set_rotate` in Rust: the display, the files, the raw frames
  and the console's preview and H264 all take the turned frame, on the GPU.
  At 90 and 270 the scene renders at the output's size turned, so a 128x32
  panel gives the scene 32x128.
- Set a KMS display's backlight with `--brightness`, 0 to 1, on
  `ghee render` and `ghee serve`, and `Renderer::set_brightness` in Rust,
  which gives back the level that the device then reads, as
  `Renderer::brightness` does. ghee writes the backlight device that the
  kernel registered under the card's device, scaled onto its
  `max_brightness`: under ssd130x, the panel's contrast. A display with no
  backlight device refuses a set and reads None. `Renderer::close`, which
  ghee runs as it exits, puts the earlier level back. A `--brightness` that
  the display refuses prints its error once, and the render goes on.
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
- Submit each page flip on a KMS display from a thread of its own on a host
  with more than one CPU, and draw the next frame while the last waits on
  the GPU and the flip, so a frame reaches the glass one frame later. The
  kernel's work for a flip then starts at once on the thread's CPU, where a
  host busy with its next frame held it back, and on a host with one CPU
  ghee yields the CPU after each flip to start that work. On a Pi Zero 2 W,
  whose kernel queues that work on the CPU that submits the flip, a 128x32
  SSD1306 with 4 ms of work between frames draws at 199 fps, where it drew
  at 161. With no work between frames it draws at 193.5 fps, where it drew
  at 197, and a Python screen that draws 200 calls a frame on it draws at
  179 fps on 102% of a core, where it drew at 128 fps on 72%.
- Encode a frame under 16384 pixels in hardware, scaled up by the smallest
  whole factor that reaches 16384 with each pixel a square block, where the
  encoder refused it. A 128x32 panel streams at 256x64.
- Refuse `--display kms` on a card whose GL runs in software, `GL on
  /dev/dri/card1 runs in software; no hardware render node pairs with this
  card`, where ghee rendered through llvmpipe.
- Name the size to render at when a display has no scaler for `--size`,
  `this display is fixed at 1920x1080 with no scaler, so 640x480 does not
  fit. Render at 1920x1080`.
- Refuse an `-o` that names no `.png` or `.jpg` file before anything loads,
  `-o wants an image pattern such as frame.####.png, or '-', got 'kms'; a
  display takes --display`, and `--alpha` with a `.jpg`. A `.jpg` takes an
  RGBA frame and drops its alpha, where `-o x.jpg` failed at its first frame
  with the default `--components 4`.
- Refuse a `--size` that is not WxH with `invalid value 'banana' for
  '--size <SIZE>': wants WxH`, where the error read `invalid _size value:
  'banana'`.

### Filters

- Declare `channels` in a filter, and read the filter's own previous frame
  through `self`.
- List a hidden filter in a scene's `filters` as
  `{"name": "dither", "visible": false}`, which a save of the console
  writes for a filter whose pass is hidden.
- Declare `assets` in a filter the way a scene does. A shipped filter's file
  is compiled in, a scanned filter's resolves beside it. Only the filter's
  own pass sees them, so they never clash with a scene's names.
- Ship `frc`, a temporal dither for a 1-bit panel: each pixel blinks on a
  16-frame cycle, lit for a share of the frames that matches its grey, for
  up to 16 shades at about 200 fps. Its `coarse` and `calm` params trade
  shades for stillness at a lower rate.
- Ship `frc256`, which does the same on a 16-level panel such as the
  SSD1322, each pixel blinking between two neighbouring levels for about 256
  shades, its per-pixel phase from a shipped 64x64 blue-noise mask.
- Run `dither`, `frc` and `frc256` on one thread on VC4, where a pass on two
  threads can draw wrong 4x4 blocks.
- Resample the source in `frc256` only when it is larger than the output, so
  each pixel of a source at the output's size reaches the panel alone, where
  the Catmull-Rom kernel blurred it into its neighbours.
- Skip a filter file that holds a key that a filter does not take, and name
  the key and the closest key that a filter takes.

### Library and packages

- Read a POST body with `ConsoleAPI::read_body`, which refuses a body past
  the cap with the 413 that `precheck` gives a long Content-Length, and a
  body whose read fails with a 400.
- Restart a renderer's clock with `Renderer::restart_clock`, so the next
  frame draws as the first one does, as `iFrame` 0 at `iTime` 0 with
  `iTimeDelta` at 1/60.
- Flush the GL work before a setup deletes the last scene's pass buffers.
  Mesa's v3d and vc4 keep the draws into a buffer as a job until a read of
  the buffer or a flush submits it, and the job holds every buffer it draws
  with until then. With no display to swap, each scene that a setup
  replaced left its last frame's jobs and their buffers behind: on a Pi 5 a
  console that switched scenes at 1920x1080 held about 7 MiB more at each
  switch, until the GPU's 4 GiB address space filled, the kernel logged
  "Couldn't allocate binner overflow mem" and the GPU reset.
- Read back a frame at a whole multiple of the render size with each pixel a
  square block, in `Renderer::render_into` and `render_scaled`.
- Grow a running scene with no rebuild through `Renderer::add_layer`, which
  puts a layer, written as a scene declares it, into a pass at an index
  among its layers, and refuses one that a scene file would refuse there.
  Every other layer keeps its program, its atlas, its phases and its quads,
  and every pass its program, its arrays and its buffers. A shape compiles
  nothing, an image layer loads its picture, and a text layer builds its
  shaders and its atlas. A layer's own shaders hold no name of the layer's
  and take its settings from uniforms of its batches alone, so layers whose
  shaders come out the same, in one scene or added, draw through one
  program, and `Renderer::compiled` counts the programs that a renderer
  linked.
- Build one layer of a running scene again in its place with
  `Renderer::replace_layer`, of its kind or of another, and give a layer a
  whole new look with `Renderer::update_layer`, which sets the live fields
  that changed and builds the layer again for any other change. A key that
  the look leaves out takes its default, and an effect that it leaves out
  stays, bypassed. Every other layer keeps its program, its atlas, its
  phases and its quads, the layer's moving effects carry their phases on,
  a layer that the scene refuses changes nothing, and a layer that fails to
  draw leaves the old one drawing in its place.
- Take a layer out of a running scene with `Renderer::remove_layer`, and
  put one at another index among the layers of its pass with
  `Renderer::move_layer`, where it keeps its look, its atlas and its phases
  and compiles nothing. Every other layer and every pass keep what they
  hold. `Scene::remove_layer` and `Scene::move_layer` do the same to a scene,
  and `RenderTarget` takes both, refusing every name by default.
- Measure words with `Scene::measure` and `Renderer::measure`: the width and
  height in pixels of the block that a text layer of a pass lays them out
  in, at a `font_size` or at the size that fits the pass.
- Give layers the looks of a drawing's calls with `Renderer::update_calls`,
  each a `ghee::draw::Call` that holds its kind, its layer and the values of
  its keywords, with no look written: a call gives its layer the look that
  `update_layer` gives the same look written out, and the calls that build
  their layer again, give an image layer another image or hold a value that
  the scene refuses come back for `update_layer`.
- Write the look that a call gives its layer, as a scene declares the
  layer, with `ghee::draw::look_of`, and read a look back as its call with
  `draw::call_of`. `draw::live_of` gives a layer with the look of a call
  when the call changes the layer's live fields alone. `Arg::of_json` and
  `Arg::to_json` read and write a keyword's value as a look written as JSON
  holds it. `RenderTarget::update_calls` gives the layers of a target the
  looks of calls, as `Renderer::update_calls` does, and by default leaves
  every call for `update_layer`.
- Add a layer, set a live field of a layer and give a layer a new look on
  any `RenderTarget` with `RenderTarget::add_layer`, `set_layer` and
  `update_layer`, as the `Renderer` methods of those names do, and by
  default a target refuses each. A target's `RenderTarget::clock_restarted`
  is true once after its clock restarted, and the loop then draws the next
  frame as its first.
- Reach a scene's layers through `ghee::layer`, the layout of their words
  through `ghee::text`, the shape layers through `ghee::shape`, the batches
  that draw consecutive shapes and texts in one draw through `ghee::run`,
  the emoji font through `ghee::emoji`, CSS's colour names through
  `ghee::css`, and a KMS display's backlight through `ghee::panel`.
- Ask a `Renderer` for the longest texture side of its GPU with
  `max_texture_size`, and whether the GPU mipmaps a buffer of any size with
  `mipmaps_npot`. `Renderer::set_asset` hands the engine one image asset,
  `set_alpha` keeps the output pass's alpha, `with_layer` reads one layer of
  the scene, and `text_errors_seq` changes whenever `text_errors` does.
  `engine::display_renders` says whether GL on a card runs on the GPU, and
  `engine::quarter_turns` gives the quarter turns of a rotation in degrees.
- Build a scene whose files resolve in a directory of its own with
  `Scene::from_dict_in` and `Scene::load_in`, read its asset files with
  `Scene::load_assets` and `load_assets_in`, put a layer into a pass through
  the checks of a scene file with `Scene::insert_layer` and hand it its files
  with `Scene::load_layer`. `Scene::text_layer`, `text_layers` and
  `text_names` read its text layers, and `Scene::taken_names` the names that
  an added pass may not take. `Asset::decode` decodes an asset's bytes,
  `Asset::to_dict` writes it as a scene declares it, `is_font` tells a font
  and `local_name` gives its name in the code that declares it.
  `Pass::size_in` gives the size of a pass in a frame, and
  `Pass::has_fragment` is false for a pass of layers alone.
  `scene::no_param` and `no_layer` write the error for a name that the scene
  lacks, and `scene::short_floats` writes each float in its shortest 32-bit
  form.
- Write a console host with the `ConsoleHost` methods `scene_doc`,
  `live_doc`, `switch_by`, `edit_scene` and `procedural_names`, each with a
  default. `core::param_of` and `core::live_value` check a /set against the
  scene document, `core::other_scene` and `other_reload` give the 409 of a
  page left open on another scene, and `ConsoleError::conflict`, `stale` and
  `unavailable` give a 409, a 412 and a 503.
- Read and write a scene's state file through `console::state`: an `Edits`
  reads and writes the file with `from_file` and `to_file`, `state::diff`
  takes a scene's defaults to its working copy and `state::fold` applies
  edits over a document, `file_of` and `default_dir` place the file,
  `base_of`, `hash_json` and `hash_text` hash the items that an edit
  changes, and `move_unreadable` moves aside a file that the read refused.
  `SceneStore::state_file` names a scene's state file,
  `SceneStore::procedural_names` the scenes that a program writes, and
  `store::scene_of` builds the scene of a document.
- Serve one /h264 viewer from an `H264Fanout` with `server::serve_h264`,
  push an access unit of one stream with `H264Fanout::push_to`, and read
  `H264Fanout::began`, true once an access unit reached a viewer.
  `h264::hw_size` gives the size that the hardware encoder takes a frame at,
  and `h264::no_encoder` the reason that a stream has no encoder.
- Tell the bundled examples' directory with `console::is_examples_dir`, list
  the filters as the console's picker shows them with `filters::filter_defs`,
  build the filters of a posted scene with `filters::rebuild_filters`, and
  tell an image file's name with `output::is_image_name`.
  `SceneProcess::pin` tells a scene process that the loop pins time, and a
  lockstep process then answers each tick before the tick's frame draws.
- Take either rounding of v × 255 for a channel value v read back in the
  tests, since GLs round 0.5 apart: Mesa to 128, NVIDIA to 127.
- Report the tests that need ffmpeg as ignored under `--cfg ghee_no_ffmpeg`,
  and the test that reads the GL driver's buffer memory under
  `--cfg ghee_no_gpu_memory`, as `--cfg ghee_no_gpu` reports those that need
  a GL device, and run the tests of a connected display and of the hardware
  H264 encoder under `--cfg ghee_display` and `--cfg ghee_hw_h264`, where
  each returned early and passed.
- Ship NOTICE and the licenses of DejaVu Sans and Pixel Operator 8 in the deb
  under `/usr/share/doc/ghee/`, with ghee's license in its copyright file, and
  LICENSE and NOTICE in the wheel.
- Ship THIRD-PARTY-NOTICES, the licenses of the Rust crates that ghee
  compiles in, beside NOTICE in the deb and the wheel.
- Name holofermes as the copyright holder in the deb's copyright file.
- Ship the arch's ghee-render wheel in the deb under `/usr/lib/ghee/`, which
  ghee hands to a scene script.
- Tag the 32-bit wheel `linux_armv6l.linux_armv7l`, where it was
  `linux_armv6l` alone, so uv takes it on a 32-bit OS on a Pi 2 or newer too.

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
