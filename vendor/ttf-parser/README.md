## ttf-parser

![Build Status](https://github.com/harfbuzz/ttf-parser/workflows/Rust/badge.svg)
[![Crates.io](https://img.shields.io/crates/v/ttf-parser.svg)](https://crates.io/crates/ttf-parser)
[![Documentation](https://docs.rs/ttf-parser/badge.svg)](https://docs.rs/ttf-parser)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88+-orange.svg)](https://www.rust-lang.org)
![Unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-brightgreen.svg)

> **This crate is in maintenance mode. Bug fixes only — no new features.**
>
> Bug reports and fixes are welcome and will be reviewed promptly. Correctness, panics and
> security issues are in scope. New table support, new API surface and performance work are
> not, and feature requests will be closed.
>
> **For new projects, we recommend [fontations](https://github.com/googlefonts/fontations)**
> (`read-fonts` and `skrifa`), which is actively developed by Google Fonts, has broader table
> support, and is the direction the Rust font ecosystem is moving.

A high-level, safe, zero-allocation font parser for
[TrueType](https://docs.microsoft.com/en-us/typography/truetype/),
[OpenType](https://docs.microsoft.com/en-us/typography/opentype/spec/), and
[AAT](https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6AATIntro.html).

Can be used as a Rust or C library.

Requires Rust 1.88 and uses edition 2024.

### Features

- A high-level API for most common properties, hiding all parsing and data resolving logic.
- A low-level, but safe API to access TrueType tables data.
- Highly configurable. You can disable most of the features, reducing binary size.
  You can also parse TrueType tables separately, without loading the whole font/face.
- Zero heap allocations.
- Zero unsafe.
- Zero dependencies.
- `no_std`/WASM compatible.
- A basic [C API](./c-api).
- Fast.
- Stateless. All parsing methods are immutable.
- Simple and maintainable code (no magic numbers).

### Safety

- The library must not panic. Any panic considered as a critical bug and should be reported.
- The library forbids unsafe code.
- No heap allocations, so crash due to OOM is not possible.
- All recursive methods have a depth limit, and the ones whose input forms a graph
  (composite glyphs, the COLRv1 paint graph, CFF subroutines) additionally bound the
  *total* work per call. A depth limit alone does not: with fan-out `b` and depth `d`,
  a small font can force `b^d` visits without ever exceeding the depth.
- Stack usage is bounded, but not tightly: outlining a composite variable glyph nests up to
  32 frames, each holding a variation-tuple buffer, for roughly 80KiB in the worst case.
- Most of arithmetic operations are checked.
- Most of numeric casts are checked.

### License

Licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license
  ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

See [CONTRIBUTING.md](./CONTRIBUTING.md) for how to build and test the library, the C API
and the benchmarks, the lint and formatting workflow, and what is expected of a change to a
parser of untrusted input.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
