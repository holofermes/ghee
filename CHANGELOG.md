# Changelog

## Unreleased

- A scene declares `assets`: image files (png, jpeg, raw u8 or f32) that
  every pass samples as `uniform sampler2D <name>` with a
  `vec2 <name>_size`, nearest unless an asset sets `"sampling": "linear"`.
- A filter def may declare `channels`, use `self` to access the previous
  frame.
- Add `uniform float frame`.
- A filter declares `assets` the way a scene does. A shipped filter's file is
  compiled in, a scanned filter's resolves beside it. Only the filter's own
  pass sees them, so they never clash with a scene's names.
- `frc256` takes its per-pixel phase from a shipped 64x64 blue-noise mask.
- `POST /scene` builds the scene's filters on the server: a filter pass is
  rebuilt from its filter, keeping its place and param values, and a filter
  the `filters` list names without a pass is appended. An unknown filter is
  a 400. The reply is the scene as it will run.
- Find a KMS display for `ghee displays` and `--display kms` in the
  kernel's cached connector state, and probe a connector, which rereads an
  HDMI display's EDID, only when that state lists no modes.
- Draw a kernel-driven panel on `--display kms` into buffers the GPU
  allocates and the panel imports, when the panel's card is a device apart
  from the GPU and takes them. A panel driver that reads each frame with
  the CPU (ssd130x, gud, udl) then shows what the GPU drew, where on a
  Pi 5 it showed stale fragments of earlier frames.
- Keep the picture on a KMS display whole when the display refuses a page
  flip, where the next frame drew into the buffer still on screen.
- Draw text on the GPU from a font. A scene declares `texts`, each with its
  words in `value` and its font, a `.ttf` or `.otf` file or a font asset,
  and every scene pass, filter passes excepted, reads `<name>_cover(p, ppem)`
  and `<name>_size`. ghee declares the scene string param of each text's
  name, which holds its words, and an edit to the param lays the words out
  again, with no recompile. A param declared under a text's name refuses
  the scene.
- Declare an asset from Python with `ghee.Asset` and a text with
  `ghee.Text(name, font, value)`, in `Scene(assets=[...], texts=[...])`.
- Add `--demo lettering`, words drawn from a font.
- Set a string param from a procedural scene with message kind 3, so its
  words reach a text over the wire.
- Load the assets of a scene given to `ghee.open` as a dict or as JSON text
  from the working directory.
- Keep the scene running through a text problem, log the problem once, and
  show it in the console. A font that fails to load draws every character as
  an error box, a glyph with no outline or past a limit draws one in its
  place, and words that do not fit leave the last words that did.
- Store alpha in every pass buffer, so a channel reads the alpha that its
  pass wrote. Keep every frame opaque unless `ghee render --alpha` or
  `ghee.open(..., alpha=True)` keeps the output pass's alpha, and refuse
  `--alpha` with `--components 3`.
- Draw a text straight into a pass's buffer with `draws`, one quad over each
  glyph: a draw's `vertex` GLSL defines `glyph_place`, which puts the glyph
  in the pass, and its `fragment` GLSL colours it. Each stage takes the
  pass's preamble and the scene's common block, as the pass's code does. A
  draw with `px_per_em` in place of `vertex` centres the words in the pass
  at that size, and a draw with no `fragment` draws them white.
- Blend a draw's alpha the way its colour blends.
- Let a pass with `draws` leave out `code`: it starts each frame transparent
  and holds its draws alone.
- Bind at most 16 textures in a pass on every GPU, and raise the cap of
  image assets and texts in a scene to 16.
- Fill a draw's glyphs from an MTSDF atlas by default, fast, and read the
  distance to the outline with `glyph_dist()` and `glyph_soft()` for
  outlines, glows and shadows. `"fill": "slug"` on a text computes Slug's
  exact coverage from the curves in each glyph's quad, exact edges at any
  size, and names `"fill": "mtsdf"` in the error of a draw that reads a
  distance. `atlas` sets the MTSDF atlas's `px_per_em`, and the atlas's
  field reaches as far as the largest `pad` of the text's draws, so a glow
  fits with no tuning.
- Generate a font's atlas on every core at its first use, cache it in
  `$XDG_CACHE_HOME/ghee/atlas`, and keep a text's last words on screen until
  the glyphs of the new ones are in. A setup and a render with `--timestep`
  wait for them.
- Draw an error box for a glyph that the atlas cannot hold, and say why.
- Count the atlas of a text drawn with the MTSDF fill against the 16 image
  assets and texts of a scene, and against the 16 textures of a pass.
- Draw a text into a pass from Python with
  `ghee.Draw(text, fragment, vertex)` in `Pass(draws=[...])`, and leave out
  the code of a pass of draws alone.
- Set a text's `fill` and `atlas` from Python on `ghee.Text`.
- Edit a pass's draws in the console. Each draw opens in a tab after its
  pass's tab, with its `vertex` and its `fragment` in editors and its text,
  blend and pad as controls, and `+` adds a draw to the open pass.
- Report the draw that holds a compile error in `/stats` as `error_draw`,
  with `part` set to `"vertex"` or `"fragment"`, and count `error_line` in
  that draw's GLSL.
- Edit a free-text string param in the console over several lines.
- Show in the console why the server refused a posted scene.
- Refuse a scene where two declarations make one GLSL name in a pass, such
  as a param `title_size` beside a text `title`, and name both in the error.
- Refuse `time`, `DT`, `frame`, `resolution`, `v_texCoord` and `iChannel0`
  and up as the name of a param, an array, an asset or a text, and any name
  that starts with `ghee_`, which ghee takes for every name that it writes
  into a shader for itself.
- Build the mip chain of a pass's buffer after each frame with
  `"mipmaps": true`, or `mipmaps=True` on `ghee.Pass`, and sample it
  trilinearly through every channel that reads it, so a layer read far away
  and minified stays calm. On a GPU without OES_texture_npot, ghee refuses
  a pass size that is no power of two.
- Rename an ISF input or pass target named like a built-in, such as `frame`,
  or named with the `ghee_` prefix, and rename an input's param with it,
  where an input named `time` kept the name on its param.
- Declare a vec or color param as one `uniform vec2`, `vec3` or `vec4`, so a
  local of its name shadows it as it shadows any uniform, and `.bg` beside a
  color named `bg` selects the field. The param was a `#define` over one
  float uniform for each component, which broke both.
- Ship NOTICE and the DejaVu font's license in the deb under
  `/usr/share/doc/ghee/`, with ghee's license in its copyright file, and
  LICENSE and NOTICE in the wheel.
- Take a list of fonts on a text, assets and files mixed,
  `"font": ["sans", "NotoColorEmoji.ttf"]`, and draw each character from the
  first font whose cmap holds it.
- Paint colour emoji in draws: a glyph with a PNG bitmap in a CBDT or sbix
  table and no outline shows its bitmap, mipmapped, placed by the draw's
  `vertex`. `<name>_cover` reads it as no ink, and a font of colour bitmaps
  alone draws its emoji in draws in place of error boxes.
- Paint a text's colour emoji from the last draw of the text in its pass
  alone, so a shadow or a glow paints no copy, and set `"emoji": true` or
  `false` on a draw, `emoji=` on `ghee.Draw`, to choose otherwise.
- Draw a text from a list of fonts in Python,
  `ghee.Text("words", ["sans", "emoji"])`.
- Encode a frame under 16384 pixels in hardware, scaled up by the smallest
  whole factor that reaches 16384 with each pixel a square block, where the
  encoder refused it. A 128x32 panel streams at 256x64.
- Read back a frame at a whole multiple of the render size with each pixel a
  square block, in `Renderer::render_into` and `render_scaled`.
- Answer `/h264` with 503 and the reason when no encoder can start, and end
  a stream whose encoder fails with a type 3 frame that carries the reason.
- Show the console preview in MJPEG when the H264 stream decodes no frame in
  5 s.
- Parse a `/set` value by the param's declared type, so a string, a bool or
  a color needs no `type=`, and name a param that the scene lacks.
- Write straight colour into a PNG of an RGBA frame, from `--alpha` and the
  console's `/frame.png`, where it held the premultiplied colour and drew
  dark fringes. A raw frame and `pixels()` keep the colour premultiplied.
- Load the assets and shader files of a scene whose directory is `/`, such
  as a dict scene opened with `/` as the working directory, where every file
  was refused as outside the directory.
- Return at once when a text's string param gets the words it holds, from
  `set_param`, the console, the wire or the Python host, where each set laid
  the words out and uploaded them again.
- Draw a character followed by U+FE0F from the first font of its text with a
  colour glyph for it, so ❤️ draws in colour when an earlier font holds a
  plain heart.
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
- Raise ValueError from `ghee.set_param` in a Python scene for a param that
  the scene does not declare or a value that does not fit its type, under
  ghee as in direct mode.
- Name the param in the type error of `set_param` and of the console, where
  the error named a param 'v'.
- Name the engine's cap of 48 layers in its error, where the error called
  them draws, and name the refused layer once, where the error named it
  twice. Give the import's own reason when it refuses an ISF shader with
  `IMPORTED` images.

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
