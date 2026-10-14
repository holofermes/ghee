# Filters

A filter is a named post process of one pass, and ghee appends it after the
scene's own passes. A scene declares its filters in `filters`:

```json
{ "passes": [ ... ], "filters": ["dither"] }
```

`{"name": "dither", "visible": false}` in place of a name lists the filter
with its pass hidden.

The command line adds filters in order, after the scene's own:

```
ghee render --demo bars --filters dither --display kms
```

ghee ships `dither`, `frc` and `frc256`, which [1-bit and 16-level
panels](displays.md#1-bit-and-16-level-panels) describes. Write your own
as `<name>.filter.json` files, and add their directories to `$GHEE_FILTERS`, a
colon-separated list. A filter declares assets as a scene does, and ghee
resolves the asset's file beside the filter.
