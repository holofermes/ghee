# Working on ghee

ghee compiles GLSL scenes and renders them on a GPU through DRM/GBM, headless.
The output goes to any connected KMS display, HDMI or an SPI/I2C panel under a
kernel driver, to image files, to raw frames, or to H264. A KMS display takes
the frames through scanout. ghee is one Rust binary around a small C engine
(`ghee/_engine/`), and `build.rs` compiles that engine in. The target hardware
is Raspberry Pi, from a Pi 1 to a Pi 5.

## You cannot see the output

An SPI or I2C panel has no readback. An HDMI display needs a human in front of
it. Hardware tells you nothing about a change. Render to a file, and assert on
the pixels:

```
ghee render --demo bars -o out.png --size 320x240 --timestep 0 --frames 1
```

`--timestep S` pins engine time to `frame * S`. Frame N then always renders at
time `N*S`, and two runs are comparable. Without it, time is the wall clock, and
no render diffs against another.

Render before a change and after it, then compare the bytes. `cmp` over a few
scenes catches a whole class of regression that no amount of looking finds.

The golden check is that pattern, already in the repo. It renders every
scene listed in the manifest. It compares the sha256 of each frame against
a pinned reference:

```
cargo run --release -- golden --manifest tools/fixtures/golden/manifest.json
```

Ask a human when a change needs hardware. Give the human a rendered reference
image to compare against, when an image helps.

## Commands

```
cargo build --release
cargo test --release
cargo clippy --release --all-targets
cargo fmt
```

`build.rs` rebuilds the C engine when anything in `ghee/_engine/` changes, so
the engine needs no build step of its own. Cross-compiling takes `GHEE_SYSROOT`.

`python/` is one mixed crate and package: the PyO3 extension `ghee._native`,
plus the `ghee` package that wraps it and holds the procedural-scene shim
(`python/ghee/_shim.py`). One abi3 wheel per architecture serves CPython 3.10
and up. It is a crate of its own, so the commands above never build it. The
wheel is the only way to `import ghee`: nothing vendors the shim. The wheel's
`ghee` command is `cli::main` compiled into the extension, behind the entry
point `ghee._cli:main`, so a build from the sdist installs it too. The kernel
names its process `ghee` after the script, so `pkill ghee` finds it, unless
the venv's python path runs to 125 bytes or more: uv then writes a `/bin/sh`
launcher, the process is named `python`, and `pkill -f bin/ghee` finds it.

```
cd python && maturin build --release --compatibility linux
uv run --no-cache --with python/target/wheels/ghee_render-*.whl --with pytest \
    pytest python/tests
```

`--compatibility linux` leaves libEGL, libgbm and libdrm to the system, which
is where the deb takes them from. `--no-cache` is what makes uv read the wheel
it was handed: uv keys a local wheel by its name, and a rebuild keeps that name.

`ghee` hands a scene script the wheel in `lib/ghee` beside its own `bin`
directory (`src/procedural/wheel.rs`). Before the binary of a `cargo build`
runs a bundled Python demo, `maturin build --release --compatibility linux -o
../target/lib/ghee` in `python/` puts the wheel there. With no wheel there,
ghee leaves uv alone, and uv finds no ghee-render of the tree's dev version,
which each bundled demo pins (`tools/release.sh` writes the pin). The wheel's
own `ghee` looks beside the Python interpreter. Point uv at the tree's wheels
by an absolute path, since a scene runs in its own directory, and keep uv's
cache off, as above:

```
UV_FIND_LINKS=$PWD/python/target/wheels UV_NO_CACHE=1 ghee render --demo hero -o hero.png
```

The Pi wheels come from `tools/xwheels.sh`: cargo-zigbuild with the glibc
pins, then a repack of the native wheel. `tools/mksysroot.sh` makes the
sysroots it needs from the boards' own apt repositories. The deb carries the
`ghee` command, the Python package and its wheel: the release workflow
unpacks the arch's wheel into `target/pydist` and copies it into
`target/pywheel`, and the cargo-deb asset list ships the first to
dist-packages and the second to `/usr/lib/ghee`.
`.github/workflows/release.yml` runs the whole chain, wheels, debs, release
assets and PyPI.

The tree carries `X.Y.Z-dev`. `tools/release.sh <version>` writes one version
into every site, then THIRD-PARTY-NOTICES with cargo-about (`cargo install
--locked --features cli cargo-about`) from `tools/about.toml`. The release
flip is the last commit before the tag, the tag builds the wheels and the
debs, and the next commit flips to the next `-dev`.

## The surface is the CLI and plain HTTP

`ghee --help` describes the tool. `ghee render <scene> -- --help` describes a
scene. The console is JSON over HTTP, and `docs/http.md` holds the API.

A Python host takes the third surface: `ghee.open(scene, ...)`, then `feed`,
`set_param`, `frame` and `skip`. `docs/python.md` holds it.

A procedural scene you start yourself hosts itself on that same surface:
`./scene.py --display --frames 300`. `docs/procedural.md` holds it.

## Contracts

Some shapes are load-bearing, and the tests pin them. A change to one of them is
an API break.

## Tests

A test pins what a user, a scene or a board relies on, and it fails for a bug
in that and for nothing else.

What earns a test:

- A contract: the scene JSON, the GLSL names that a scene reads, the CLI, the
  HTTP API, the wire, the Python surface, an error that a user reads.
- An invariant hidden in the code: packing, encoding, the texture-unit budget,
  a GPU workaround, a lock between threads.
- A fixed bug. Its test lands in the commit of the fix.

What does not:

- A value copied from ghee's own output. Such a test pins the bug along with
  the code. Take the expected value from outside ghee: the font through
  ttf-parser, the formula, the datasheet, a stream recorded from a working
  panel.
- A round trip through ghee's own encoder and decoder. The two halves agree
  whatever the bytes mean. Decode with a third-party reader, or assert the
  bytes.
- A count or a constant that describes today: the size of a preload, the
  number of fixtures, a quality setting.
- The same behaviour again through another layer. A Python test checks that a
  call reaches the engine and that the engine's error reaches Python. The Rust
  tests own what the engine does.
- A wall-clock ratio, or the text of a source file.
- The absence of a removed feature.

Make it fail first. Break the code that a new test guards and watch the test
fail: revert the fix, delete the line, flip the condition. A test that passes
anyway is rewritten or dropped before it lands.

When a test breaks, assume the change is wrong. Edit a test's input or its
expectation only when the contract that it pins changed on purpose, and say so
in CHANGELOG.md.

GPU output:

- Assert pixels against a reference computed off the GPU: full coverage inside
  the outline, none outside, the 1.5 px band along each edge left free, a
  tolerance in levels. tests/text.rs holds the helpers.
- When a change must leave every pixel alone, render before and after on the
  same host and compare the bytes.
- A pinned sha holds on the GL that pinned it. The golden manifest is checked
  on its board before a release. Any test that reads it compares the renderer
  first and skips with the reason on another GL.
- A workaround for one GPU, like the VC4 one-thread guard, gets a test that
  finds its line in every generated shader that needs it. The rest of the
  generated GLSL is compiled and rendered, never compared as text.
- A test that needs a GL device carries
  `#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]` and fails where no
  device opens. A host without one runs the tests with
  `RUSTFLAGS="--cfg ghee_no_gpu"`, which reports them as ignored, never as
  passed. A test that needs ffmpeg takes `ghee_no_ffmpeg` the same way, and
  a test that reads the GL driver's buffer memory takes
  `ghee_no_gpu_memory`.
- A test that needs a connected display or a hardware H264 encoder carries
  `#[cfg_attr(not(ghee_display), ignore = "needs a connected display")]` or
  `not(ghee_hw_h264)`, and fails where the host lacks it. A host that has
  one runs the tests with `RUSTFLAGS="--cfg ghee_display"` or
  `--cfg ghee_hw_h264`. A test never returns early for missing hardware.

Errors: an error that a user reads is pinned whole with assert_eq, once, in
the test of the code that writes it. A contains or ends_with match lets a
doubled name through. Every other test of the same failure matches a short
key.

Names: a test name is a sentence about what a user sees, and the assertions
check that sentence. A name tells no history: no "as before", "still" or "old".

Fixtures: a fixture is the smallest input that shows the case, with a doc
comment that says where it came from and how it was cut. A recorded output
stays only where ghee has no other way to check it, like the command stream
that drove a real panel.

Flaky tests: a test that fails once with no code change has found a race, in
the test or in ghee. Find which before raising a timeout. Wait on a condition
with a deadline. A test that sets an environment variable holds the lock that
every reader of that variable holds. A test removes what it writes: a unit
test takes its directory from `crate::testing::TempDir`, and the drop removes
it.

How many: a feature brings one test per contract that it adds, one per
invariant that it hides, and one per error path that a user can reach. The
variants of one rule are rows of one table test.

When to delete:

- The contract that it pins is gone or replaced.
- Another test fails for every bug that it would catch.
- Its expected values came from a reference that no longer exists.
- It changed with every edit to its code and never failed for a bug.

## When this file is wrong

The repo wins when a command or a claim here disagrees with it. Diagnose the
difference. Do the task against the repo. Fix this file in the same change.
Never follow a stale instruction, and never work around one in silence.

## Writing a scene

A scene is either JSON or a procedural scene. The JSON is an ordered list of
passes, each with a fragment shader. A procedural scene is an executable that
emits its scene JSON and feeds its data. See `ghee/examples/`,
`python/ghee/_shim.py` and `docs/procedural.md`.

- **A scene declares its build-time parameters. It never reads the
  environment.** A scene declares its arguments (`ghee.args([...])` in the shim)
  and answers the probe with them. `ghee render <scene> -- --help` lists what a
  scene takes. Compute nothing from an argument before the parse.
- **Put shared GLSL in the scene's `common` block.** ghee puts that block in
  front of every pass. A filter pass never gets it.
- A live parameter is a `Param`. It reaches every pass through the preamble. A
  `Param` changes with no recompile.
- Per-frame data arrives as a declared array. ghee binds the array as a texture.
  The shader reads it with `name_at(i)` or `name_sample2(uv)`.
- **On VC4, a short pass that reads a texture can draw wrong 4x4 pixel
  blocks.** Mesa's vc4 compiler runs a short fragment shader in two threads
  that switch at a texture read, and a switch loses the condition flags. Its
  flag optimization can still leave a select after the switch reading a
  compare made before it: a `step`, a ternary, a comparison. Source order does
  not prevent it. Put the pass's last texture read inside an `if` that the
  compiler cannot drop, and the shader compiles single-threaded.
  `VC4_DEBUG=qpu` then shows no `sig_thread_switch` in it:

  ```glsl
  vec4 c = vec4(0.0);
  if (gl_FragCoord.x >= 0.0) c = texture2D(tex, uv);
  ```
- **A cheap reject around an expensive per-pixel loop body pays on VC4.** VC4
  skips an `if` that a whole 16-pixel block fails, and an always-false body then
  costs almost nothing. Write the reject as an `if` block. A function with an
  early return flattens into a select, and the work happens anyway.

Readback is a fixed cost. It scales with the pixel count, and not with the
shader complexity. At panel sizes the SPI or I2C write dominates the frame, and
the GPU does not.

## Conventions

A comment states a constraint the code cannot state: a hardware limit, a
protocol rule, a hidden invariant. A comment never holds the history of a
finding. A comment never argues that a change is correct. A comment never names
what it replaced. The commit message holds that.

Prose, comments, docs and commit messages alike, matches the voice this repo
already writes in: read the README, a few pages of docs/ and some module docs
before writing any.
The mechanics that are rules, not accidents: no em dashes, no "not X but Y"
constructions, no filler (simply, robust, powerful, "note that"), semicolons
rarely, no "we", plain words, one term for one thing, an actor introduced
before it is used, a fact stated once in the place the reader acts on it,
never document what the code used to do, and a speed claim takes a number on
named hardware or stays unwritten. Write nothing obvious.

An error string is an interface. An error that names a wrong cause sends the
reader down a dead end. Name the real condition, and nothing else.

Do not commit or push unless asked. Do not add a dependency without a strong
recorded reason.

## Console UI

The console web UI is one static file with no build step. It works on a phone
first and on a desktop too, in every major browser, in the light theme and in
the dark theme. Check a change at phone width and in both themes before it
lands.

- A control on a touch screen is at least 40 px tall. The page scrolls, and
  nothing is cut off at phone width.
- A setting shows its key and its value, with no grey hint text, no
  description under a field and no helper or warning line beside it. An
  empty field's placeholder names ghee's default in one word, such as `fit`
  or `auto`. A key that needs a sentence to explain it needs a better name.
- Nothing moves under the cursor. A tab keeps its width and its place, open
  or closed. Actions sit at fixed spots at the end of their row, and a click
  never shifts another control.
- Reuse the console's components: the dropdown menu, the colour control, the
  dialog and the eye. A new kind of control needs a reason that none of them
  covers.
- An eye hides a pass or a layer, or bypasses an effect. A hidden or bypassed
  thing dims the same way everywhere. A hidden pass and a bypassed effect stay
  editable. A hidden layer keeps its content editable: a text layer's words,
  font_family and text_align, which shape `<name>_cover` and `<name>_size`,
  and an image layer's image. Every setting that only changes how it draws
  dims and locks.
- The most used things come first. A pass shows its layers before its shader,
  and the shader sits closed under a line.
- A number field has no spin buttons in any browser. A colour shows its
  swatch alone, with no hex code beside it.
- Rename is a pencil that opens the console's dialog. Removing a pass or a
  layer asks first in the same dialog.
- An edit applies at once to the working copy that ghee holds of the
  scene, which every open page shows, and Save keeps it on the device, in
  the scene's state file. The page keeps no copy of its own.

## Names

A scene key, a value, a GLSL name and a Python keyword are an interface. Pick
each one for a reader who has never seen ghee.

- Where CSS has a word for a setting, use it: `color`, `opacity`,
  `font_size`, `font_family`, `text_align`, `text_shadow`, `text_stroke`,
  `border`, `border_radius`, `stroke_width`, `blend`. An underscore stands for
  CSS's hyphen, so Python takes the name as a keyword.
- Where CSS has a word for a value, use it too: a colour takes CSS's colour
  names, such as `navy` and `transparent`, in any case, beside `#rrggbb` and
  three numbers.
- Everywhere else, use a plain word: a shape layer holds `shape`, `rect` or
  `line`, and a line its `points`. A name never asks a newcomer to decode
  jargon such as pixels per em.
- `vertex` and `fragment` name shader stages and nothing else.
- A value is plain keys, never a small syntax inside a string: `x` and `y` as
  two keys, a gradient as `{"from", "to", "angle"}`, an effect as an object
  with named fields.
- A user never has to invent a name. The console and a Python drawing name a
  new layer from its kind (`text`, `text2`, `image`, `rect`), and renaming is
  a deliberate act.
- A name that reaches GLSL is GLSL-safe: ASCII letters, digits and `_`, a
  letter first, a letter or a digit last, no `__`, and no `gl_` or `ghee` at
  the start. Every GLSL name that ghee derives from a user's name carries a
  suffix, such as `_cover`, `_tex` or `_size`, and a new one does too.
