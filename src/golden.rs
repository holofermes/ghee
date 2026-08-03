//! Golden frame check.
//!
//! The check renders every scene a manifest names. It compares each frame's
//! sha256 against the pinned reference in tools/fixtures/golden.
//! Scene paths in the manifest are relative to the repo root.
//! Run the check from the repo root.

use std::fs;

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::engine::{Feed, Renderer};
use crate::scene::{ArraySize, Scene};

#[derive(Deserialize)]
struct Manifest {
    width: i32,
    height: i32,
    components: i32,
    frames: usize,
    fps: f64,
    scenes: Map<String, Value>,
}

#[derive(Deserialize)]
struct Entry {
    path: String,
    shas: Vec<String>,
}

/// The pinned provider. The arithmetic stays in u64, so any implementation
/// reproduces the value bit for bit. The division by 65536 is exact in f32.
fn value_at(i: u64, frame: u64, namehash: u64) -> f32 {
    let raw = i
        .wrapping_mul(2654435761)
        .wrapping_add(frame.wrapping_mul(10007))
        .wrapping_add(namehash.wrapping_mul(31))
        % 65536;
    raw as f32 / 65536.0
}

fn name_hash(name: &str) -> u64 {
    name.bytes().map(u64::from).sum()
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// The index of the first frame whose hash differs. None means every frame
/// matches. `dump` writes every frame as `<name>.<frame>.png`, so a board
/// whose GPU rounds differently still compares against a reference by pixels.
fn check_scene(
    man: &Manifest,
    name: &str,
    entry: &Entry,
    dump: Option<&str>,
) -> Result<Option<usize>> {
    let scene = Scene::load(&entry.path)?;
    let mut r = Renderer::new(man.width, man.height, man.components, None, false, false)?;
    let err = r.setup(&scene)?;
    if !err.is_empty() {
        bail!("setup failed: {err}");
    }
    r.set_dt(1.0 / man.fps);
    let mut mismatch = None;
    for f in 0..man.frames {
        r.set_time(Some(f as f64 / man.fps));
        let mut provider = |name: &str, size: ArraySize| {
            let n = match size {
                ArraySize::Linear(n) => n as u64,
                ArraySize::Grid(w, h) => (w * h) as u64,
            };
            let (frame, nh) = (f as u64, name_hash(name));
            Feed::Data((0..n).map(|i| value_at(i, frame, nh)).collect())
        };
        let pixels = r.render(Some(&mut provider))?;
        if let Some(dir) = dump {
            let png = crate::output::encode_png(crate::looper::Frame {
                data: pixels,
                width: man.width,
                height: man.height,
                components: man.components,
            })?;
            fs::write(format!("{dir}/{name}.{f}.png"), png)?;
        }
        let sha = hex(&Sha256::digest(pixels));
        if mismatch.is_none() && entry.shas.get(f) != Some(&sha) {
            mismatch = Some(f);
            if dump.is_none() {
                return Ok(mismatch);
            }
        }
    }
    Ok(mismatch)
}

/// The process exit code. A mismatch in any scene gives 1.
pub fn run(manifest_path: &str, dump: Option<&str>) -> Result<i32> {
    let text = fs::read_to_string(manifest_path)
        .with_context(|| format!("cannot read manifest {manifest_path}"))?;
    let man: Manifest = serde_json::from_str(&text)
        .with_context(|| format!("{manifest_path}: not a golden manifest"))?;
    if let Some(dir) = dump {
        fs::create_dir_all(dir).with_context(|| format!("cannot create {dir}"))?;
    }
    let mut failed = false;
    for (name, raw) in &man.scenes {
        let entry: Entry = serde_json::from_value(raw.clone())
            .map_err(|e| anyhow!("{manifest_path}: scene '{name}': {e}"))?;
        match check_scene(&man, name, &entry, dump) {
            Ok(None) => println!("{name}: PASS ({} frames)", man.frames),
            Ok(Some(f)) => {
                println!("{name}: FAIL (first mismatch at frame {f})");
                failed = true;
            }
            Err(e) => {
                println!("{name}: FAIL ({e})");
                failed = true;
            }
        }
    }
    Ok(i32::from(failed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_provider_is_the_pinned_integer_math() {
        // The test computes these values by hand from the definition above.
        assert_eq!(name_hash("vals"), 118 + 97 + 108 + 115);
        assert_eq!(value_at(0, 0, 0), 0.0);
        assert_eq!(value_at(1, 0, 0), (2654435761u64 % 65536) as f32 / 65536.0);
        assert_eq!(value_at(0, 3, 0), 3.0 * 10007.0 / 65536.0);
        assert_eq!(value_at(0, 0, 2), 62.0 / 65536.0);
        // The sum wraps into 0..65536. Every value lands in [0, 1).
        for i in [0u64, 1, 7, 4095, 65535, 1 << 20] {
            let v = value_at(i, 11, name_hash("mask"));
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn hex_is_lowercase_and_padded() {
        assert_eq!(hex(&[0, 15, 16, 255]), "000f10ff");
    }

    #[test]
    fn the_provider_is_bit_exact_against_the_reference() {
        // The pinned reference holds these f32 bit patterns.
        assert_eq!(name_hash("v\u{e9}"), 482); // The hash sums the name's UTF-8 bytes.
        let bits = |name: &str, n: u64, frame: u64| -> Vec<u32> {
            (0..n)
                .map(|i| value_at(i, frame, name_hash(name)).to_bits())
                .collect()
        };
        assert_eq!(
            bits("vals", 5, 0),
            [1045702656, 1060027136, 1042395136, 1059200256, 1037987840]
        );
        assert_eq!(
            bits("vals", 5, 7),
            [1049449984, 1061182464, 1047016448, 1060355584, 1043708928]
        );
        // A grid feeds w*h values. 3x5 and 5x3 give the same run.
        let grid = bits("mask", 15, 1);
        assert_eq!(
            grid[..6],
            [1052104192, 1062509568, 1050450432, 1061682688, 1048796672, 1060855808]
        );
        assert_eq!(grid[14], 1009139712);
        assert_eq!(bits("x", 1, 1), [1045855232]);
        // The product wraps past 2^16 indices.
        assert_eq!(bits("vals", 4096, 0)[4095], 1061902592);
    }
}
