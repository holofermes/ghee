# Working on ghee

ghee compiles GLSL scenes and renders them on a GPU through DRM/GBM, headless,
with output to SPI/I2C panels, HDMI via KMS scanout, image files, raw frames, or
H264. Python around a small C engine (`ghee/_engine/`). Target hardware is
Raspberry Pi, from a 2012 Pi 1 to a Pi 5.

## You cannot see the output

There is no readback from an SPI or I2C panel, and an HDMI display needs eyes on
it. Reaching for hardware to check a change is a dead end. Render to a file and
assert on pixels:

```
ghee render --demo bars -o out.png --size 320x240 --timestep 0 --frames 1
```

`--timestep S` pins engine time to `frame * S`, so frame N always renders at time
`N*S` and two runs are comparable. Without it, time is the wall clock and nothing
you render can be diffed.

For a change that should not alter output, render before and after and compare
bytes. `cmp` on a handful of scenes catches whole classes of regression that no
amount of looking would find.

`tests/goldens/` is this pattern already institutionalised. Run it with
`uv run pytest tests/test_render.py`.

When a change genuinely needs hardware, ask a human, and give them a rendered
reference image to compare against when it makes sense to do so.

## Commands

```
cd ghee && uv run pytest -q         # from this directory, not the parent
uvx ruff@0.16.0 format .
uvx ruff@0.16.0 check .
```

Python goes through `uv`. Lint and format with ruff only, config in
`pyproject.toml`.

After editing anything in `ghee/_engine/`:

```
uv pip install -e . --no-deps --force-reinstall
```

`uv run pytest` does **not** reliably recompile the engine. Without that
reinstall you will test the previous `.so` and believe a change works when it was
never built. Confirm the new binary actually contains your change (`strings`,
`objdump`) before trusting a result.

## The surface is the CLI and plain HTTP

`ghee --help` describes the tool; `ghee render <scene> -- --help` describes a
scene.The console is JSON over HTTP: `/stats`, `/scene`, `/scenes`, `/set`,
`/switch`, `/frame.png`

## Contracts

Some shapes are load-bearing and pinned by tests. Changing one is an API break,
not a refactor

## When this file is wrong

If a command or claim here disagrees with the repo, the repo wins. Diagnose,
do the task against reality, and fix this file in the same change. Do not
follow a stale instruction off a cliff, and do not silently work around it
either.

## Writing a scene

A scene is JSON (an ordered list of passes, each with a fragment shader) or a
`.py` file exposing a `SceneController` that builds those passes in code. See
`ghee/examples/`.

- **Build-time parameters are declared, not read from the environment.** A
  controller sets `ARGS = [Arg("emit", "size")]` and reads `self.args.emit` in
  `build()`. `ghee render <scene> -- --help` lists what a scene takes. Nothing
  derived from an argument may be computed at import time.
- **Shared GLSL goes in the scene's `common` block**, prepended into every pass.
  Filter passes never receive it.
- Live parameters are `Param`s; they reach every pass through the preamble and
  can be changed without a recompile.
- Per-frame data arrives as declared arrays, bound as textures, read with
  `name_at(i)` or `name_sample2(uv)`.
- **On VC4, read a pass's textures before computing values used after them.**
  The evidence: a value computed ahead of the texture reads and consumed after
  them came back wrong in 4x4 pixel blocks, both directions, while every part
  of the expression read correct alone, a compiler fault, not arithmetic.
  Hoisting the reads above the dependent math is the shape that compiles right.
- **A cheap reject around an expensive per-pixel loop body pays on VC4.** The
  evidence: an `if` a whole 16-pixel block fails is genuinely skipped, measured
  as a large cost drop for an always-false body. The reject must be an `if`
  block; written as a function with an early return it flattens into a select
  and the work happens anyway.

Performance shape, so you optimise the right thing: readback is a fixed cost that
scales with pixel count rather than shader complexity, and at panel sizes the
SPI/I2C write dominates the frame, not the GPU.

## Conventions

Comments state a constraint the code cannot: a hardware limit, a protocol rule, a
non-obvious invariant. Not the history of how something was found, not why a
change is correct, not what it replaced. That belongs in the commit message.

Prose in this repo, including docs and commit messages, avoids em dashes and
"not X but Y" constructions. Terse. Nothing obvious.

Error strings are an interface. An error that names a wrong cause sends the
reader down a dead end, so name the real condition and nothing else.

Do not commit or push unless asked. Do not add runtime dependencies. The console
web UI is one static file with no build step, and any change to it must work in
both light and dark themes as well as for mobile and desktop across all major browsers.
