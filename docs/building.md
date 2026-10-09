# Build

The build needs a Rust toolchain and the GLES/DRM development headers. On
Raspberry Pi OS or Debian:

```
sudo apt install libdrm-dev libgbm-dev libegl1-mesa-dev libgles2-mesa-dev
cargo build --release
```

A Python scene that this build runs takes the ghee-render wheel built from
the same tree, from `target/lib/ghee`:

```
cd python && maturin build --release --compatibility linux -o ../target/lib/ghee
```

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
