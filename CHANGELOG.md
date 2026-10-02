# Changelog

## Unreleased

- A scene declares `assets`: image files (png, jpeg, raw u8 or f32) that
  every pass samples as `uniform sampler2D <name>` with a
  `vec2 <name>_size`.
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
- Draw a kernel-driven panel on `--display kms` into buffers the GPU
  allocates and the panel imports, when the panel's card is a device apart
  from the GPU and takes them. A panel driver that reads each frame with
  the CPU (ssd130x, gud, udl) then shows what the GPU drew, where on a
  Pi 5 it showed stale fragments of earlier frames.
- Draw text on the GPU from a font. A scene declares `texts`, each named
  after the scene string param that holds its words and naming a `.ttf` or
  `.otf` asset as its font, and every scene pass reads `<name>_cover(p, ppem)`
  and `<name>_size`. An edit to the param lays the words out again, with no
  recompile.
- Declare an asset from Python with `ghee.Asset` and a text with `ghee.Text`,
  in `Scene(assets=[...], texts=[...])`.
- Add `--demo lettering`, words drawn from a font.
- A procedural scene sets a string param with message kind 3, so its words
  reach a text over the wire.
- `ghee.open` loads the assets of a scene given as a dict or as JSON text,
  from the working directory.

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
