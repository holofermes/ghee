# Working on ghee

ghee compiles GLSL scenes and renders them on a GPU through DRM/GBM, headless.
The output goes to SPI/I2C panels, to any KMS display through scanout, to image
files, to raw frames, or to H264. ghee is one Rust binary
around a small C engine (`ghee/_engine/`), and `build.rs` compiles that engine
in. The target hardware is Raspberry Pi, from a Pi 1 to a Pi 5.

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

The golden check is that pattern, already in the repo. It renders the scenes
the manifest names. It compares the sha256 of each frame against a pinned
reference:

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
(`python/ghee/_shim.py`). One abi3 wheel per architecture serves CPython 3.9
and up. It is a crate of its own, so the commands above never build it. The
wheel is the only way to `import ghee`: nothing vendors the shim.

```
cd python && maturin build --release --compatibility linux
uv run --no-cache --with python/target/wheels/ghee_render-*.whl --with pytest \
    pytest python/tests
```

`--compatibility linux` leaves libEGL, libgbm and libdrm to the system, which
is where the deb takes them from. `--no-cache` is what makes uv read the wheel
it was handed: uv keys a local wheel by its name, and a rebuild keeps that name.

The Pi wheels come from `tools/xwheels.sh`: cargo-zigbuild with the glibc
pins, then a repack of the native wheel. `tools/mksysroot.sh` makes the
sysroots it needs from the boards' own apt repositories. The deb carries the
`ghee` command and the Python package in one: the release workflow unpacks
the arch's wheel into `target/pydist`, and the cargo-deb asset list ships it
to dist-packages. `.github/workflows/release.yml` runs the whole chain,
wheels, debs, release assets and PyPI.

The tree carries `X.Y.Z-dev`. `tools/release.sh <version>` writes one version
into every site. The release flip is the last commit before the tag, the tag
builds the wheels and the debs, and the next commit flips to the next `-dev`.

## The surface is the CLI and plain HTTP

`ghee --help` describes the tool. `ghee render <scene> -- --help` describes a
scene. The console is JSON over HTTP, and `docs/http.md` holds the API.

A Python host takes the third surface: `ghee.open(scene, ...)`, then `feed`,
`set_param`, `frame` and `skip`. `python/README.md` holds it.

A procedural scene you start yourself hosts itself on that same surface:
`./scene.py --output kms --frames 300`. `docs/procedural.md` holds it.

## Contracts

Some shapes are load-bearing, and the tests pin them. A change to one of them is
an API break.

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
- **On VC4, hoist the texture reads of a pass above the math that depends on
  them.** A value computed ahead of the texture reads, and consumed after them,
  comes back wrong. The error shows in 4x4 pixel blocks, in both directions.
  Every part of the expression reads correct alone, so the fault is the compiler.
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
already writes in: read the README and a few module docs before writing any.
The mechanics that are rules, not accidents: no em dashes, no "not X but Y"
constructions, no filler (simply, robust, powerful, "note that"), semicolons
rarely, no "we", plain words, one term for one thing, an actor introduced
before it is used, a fact stated once in the place the reader acts on it,
never document what the code used to do, and a speed claim takes a number on
named hardware or stays unwritten. Write nothing obvious.

An error string is an interface. An error that names a wrong cause sends the
reader down a dead end. Name the real condition, and nothing else.

Do not commit or push unless asked. Do not add a dependency without a strong
recorded reason. The console web UI is one static file with no build step. A
change to it must work in the light theme and in the dark theme. It must work on
mobile and on desktop, across all major browsers.
