# ghee tests

```
uv pip install -e '.[test]'      # pytest + Pillow (numpy comes with ghee)
python -m pytest                 # runs tests/
```

Rendering tests need a GL device. They auto-probe `$GHEE_DRI`, then `/dev/dri/card0`
(the Pi), then `/dev/dri/renderD128` (desktop Mesa); they skip if none works. Override
with `GHEE_DRI=/dev/dri/renderD128 pytest`.

## Layout

- `test_scene_model.py` — scene model, GLSL preamble, output patterns, driver registry.
  Pure Python, no GL, runs anywhere.
- `test_render.py` — render self-consistency: same input -> same pixels, different
  input -> different pixels (determinism, named array, live param update, multipass
  forward + feedback). Device-independent, no golden needed.
- `test_demos.py` — every bundled demo scene compiles, renders, and matches its golden.

## Goldens

`tests/goldens/*.png` is a committed baseline, compared fuzzily so one baseline survives
cross-GPU drift (a render fails only when more than `max_bad_frac` of channel samples
differ by more than `tol`).

- Defaults: `tol=16`, `max_bad_frac=0.05`. Override with `GHEE_GOLDEN_TOL` /
  `GHEE_GOLDEN_MAXBAD`; `tol=0` is an exact match.
- Re-record a baseline: `GHEE_REGEN=1 pytest`.
- Mesa (desktop) and VC4 (Pi) drift ~0% on the bundled demos at the default
  tolerance, so the one baseline passes on both.
