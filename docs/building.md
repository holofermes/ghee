# Build

The build needs Rust, from [rustup.rs](https://rustup.rs), and the GLES/DRM
development headers. On Raspberry Pi OS or Debian:

```
sudo apt install libdrm-dev libgbm-dev libegl-dev libgles-dev
cargo build --release
```

The build puts the `ghee` command in `target/release/ghee`.

A Python scene that this build runs takes the ghee-render wheel built from
the same tree, from `target/lib/ghee`:

```
cd python && maturin build --release --compatibility linux -o ../target/lib/ghee
```

## Cross compile for the Pi

To build for a Pi on a 64-bit OS from a Debian or Ubuntu computer, add the
target, the cross compiler and the arm64 libraries:

```
rustup target add aarch64-unknown-linux-gnu
sudo dpkg --add-architecture arm64
sudo apt update
sudo apt install gcc-aarch64-linux-gnu libdrm-dev:arm64 libgbm-dev:arm64 \
    libegl-dev:arm64 libgles-dev:arm64
```

Name the linker in the `.cargo/config.toml` of the project that you build:

```toml
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"
```

Then build with `cargo build --release --target aarch64-unknown-linux-gnu`.
The binary runs on a Pi whose glibc is as new as the computer's, or newer,
such as a Pi on the same Debian release.

## Tests

```
cargo test --release
cargo run --release -- golden --manifest tools/fixtures/golden/manifest.json
uv run --no-cache --with python/target/wheels/ghee_render-*.whl --with pytest \
    pytest python/tests
```

The render tests need a GL device, and they fail on a host where none opens.
On such a host, `RUSTFLAGS="--cfg ghee_no_gpu" cargo test --release` reports
them as ignored, `--cfg ghee_no_ffmpeg` does the same for the tests that
need ffmpeg, and `--cfg ghee_no_gpu_memory` for the test that reads the GL
driver's buffer memory, which vc4 and NVIDIA's driver do not report. The
tests of a connected display and of the hardware H264 encoder run on a host
that has one, under `--cfg ghee_display` and `--cfg ghee_hw_h264`. The
golden check renders every scene in the manifest, and compares the sha256 of
each frame against the pinned reference in `tools/fixtures/golden/`.
