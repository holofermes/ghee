//! The ghee-render wheel that a scene process takes.
//!
//! A Python scene declares ghee-render, and uv installs it. ghee hands uv the
//! wheel built with ghee itself: the one file `ghee_render-<version>-*.whl` in
//! `lib/ghee` under the prefix of the running executable, the directory above
//! the executable's own, so `/usr/bin/ghee` takes `/usr/lib/ghee`. ghee writes
//! an index that lists that wheel alone, puts it first on the process's
//! `UV_INDEX` and sets `UV_INDEX_STRATEGY=first-index`: uv then takes
//! ghee-render from the wheel and every other package from its other indexes.
//! A ghee that finds none leaves uv alone, and uv resolves the scene's own
//! dependencies: the bundled examples pin ghee-render to ghee's version. The
//! `ghee` command of the ghee-render wheel runs in the Python interpreter, and
//! looks in the `lib/ghee` beside the interpreter.
//!
//! uv keeps one environment per script path and reuses it while it satisfies
//! the script's dependencies, so the index reaches a scene when uv resolves
//! the scene's environment.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::UNIX_EPOCH;

use anyhow::{anyhow, bail, Result};
use sha2::{Digest, Sha256};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The wheel of this ghee, and the identity that names its index: the path,
/// the size and the modification time, so a wheel built again gets a new
/// index and new environments.
struct Wheel {
    path: PathBuf,
    id: String,
}

/// The wheel of this ghee in `lib/ghee` under the prefix of the running
/// executable, read once per process.
fn own() -> &'static Result<Option<Wheel>, String> {
    static OWN: OnceLock<Result<Option<Wheel>, String>> = OnceLock::new();
    OWN.get_or_init(|| {
        let dir = std::env::current_exe()
            .ok()
            .and_then(|exe| Some(exe.parent()?.parent()?.join("lib").join("ghee")))
            .unwrap_or_else(|| PathBuf::from("lib/ghee"));
        find(&dir, VERSION).map_err(|e| e.to_string())
    })
}

/// A version as the wheel's file name spells it. maturin writes the
/// pre-release `-dev` as `.dev0`.
fn wheel_version(version: &str) -> String {
    match version.split_once('-') {
        Some((release, "dev")) => format!("{release}.dev0"),
        _ => version.to_string(),
    }
}

/// The one ghee-render wheel of `version` in `dir`. A missing directory holds
/// none.
fn find(dir: &Path, version: &str) -> Result<Option<Wheel>> {
    let head = format!("ghee_render-{}-", wheel_version(version));
    let mut found: Vec<PathBuf> = fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with(&head) && n.ends_with(".whl"))
                })
                .collect()
        })
        .unwrap_or_default();
    found.sort();
    let path = match found.as_slice() {
        [] => return Ok(None),
        [one] => one.clone(),
        several => {
            let names: Vec<String> = several
                .iter()
                .map(|p| {
                    p.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            bail!(
                "{} holds more than one ghee-render wheel of ghee {version}: {}",
                dir.display(),
                names.join(", ")
            );
        }
    };
    let meta = fs::metadata(&path)
        .map_err(|e| anyhow!("{}: {}", path.display(), crate::scene::strerror(&e)))?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    let id = format!("{}:{}:{mtime}", path.display(), meta.len());
    Ok(Some(Wheel { path, id }))
}

/// The identity of this ghee's wheel, empty when it has none. The bundled
/// examples extract to a directory named after it, so each build's examples
/// get new uv environments.
pub(crate) fn identity() -> String {
    match own() {
        Ok(Some(w)) => w.id.clone(),
        _ => String::new(),
    }
}

/// The first six bytes of the sha256 of `text`, in hex.
pub(crate) fn tag(text: &[u8]) -> String {
    Sha256::digest(text)
        .iter()
        .take(6)
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `path` as a file URL. Every byte outside the unreserved set and `/` is
/// percent-encoded, so a space never splits UV_INDEX's list.
fn file_url(path: &Path) -> String {
    use std::os::unix::ffi::OsStrExt;
    let mut url = String::from("file://");
    for &b in path.as_os_str().as_bytes() {
        if b.is_ascii_alphanumeric() || b"-._~/".contains(&b) {
            url.push(char::from(b));
        } else {
            url.push_str(&format!("%{b:02X}"));
        }
    }
    url
}

/// Writes the index that lists `wheel` alone under `cache`, in the layout of
/// PEP 503 that uv reads from a directory: `ghee-render/index.html`. The URL
/// of the index comes back.
fn write_index(wheel: &Wheel, cache: &Path) -> std::io::Result<String> {
    let root = cache.join(format!("index-{}", tag(wheel.id.as_bytes())));
    let page = root.join("ghee-render").join("index.html");
    if !page.exists() {
        fs::create_dir_all(root.join("ghee-render"))?;
        let name = wheel
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .replace('&', "&amp;")
            .replace('<', "&lt;");
        let body = format!(
            "<!DOCTYPE html>\n<html><body>\n<a href=\"{}\">{name}</a>\n</body></html>\n",
            file_url(&wheel.path)
        );
        // The page lands whole through a rename, since another ghee's scene
        // may read it meanwhile.
        let part = page.with_extension(format!("html.{}", std::process::id()));
        fs::write(&part, body)?;
        fs::rename(&part, &page)?;
    }
    Ok(file_url(&root))
}

/// `first`, then the value that `var` holds in ghee's environment, as the
/// space-separated list that uv reads.
fn before(first: &str, var: &str) -> String {
    match std::env::var(var) {
        Ok(rest) if !rest.trim().is_empty() => format!("{first} {rest}"),
        _ => first.to_string(),
    }
}

/// The variables that hand a scene process this ghee's wheel. An error keeps
/// the process from starting.
pub(crate) fn uv_env() -> Result<Vec<(&'static str, String)>> {
    match own() {
        Err(e) => Err(anyhow!("{e}")),
        Ok(Some(wheel)) => {
            static INDEX: OnceLock<Result<String, String>> = OnceLock::new();
            let url = INDEX
                .get_or_init(|| {
                    let cache = crate::console::cache_dir();
                    write_index(wheel, &cache).map_err(|e| {
                        format!(
                            "cannot write the ghee-render index under {}: {}",
                            cache.display(),
                            crate::scene::strerror(&e)
                        )
                    })
                })
                .clone()
                .map_err(|e| anyhow!(e))?;
            Ok(vec![
                ("UV_INDEX", before(&url, "UV_INDEX")),
                ("UV_INDEX_STRATEGY", "first-index".to_string()),
            ])
        }
        Ok(None) => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TempDir;

    #[test]
    fn two_wheels_of_ghees_version_name_both() {
        let dir = TempDir::new("wheels");
        for name in [
            "ghee_render-0.5.0.dev0-cp310-abi3-linux_x86_64.whl",
            "ghee_render-0.5.0.dev0-cp39-abi3-linux_x86_64.whl",
            "ghee_render-0.4.0-cp39-abi3-linux_x86_64.whl",
        ] {
            fs::write(dir.join(name), b"").unwrap();
        }
        let err = find(Path::new(&*dir), "0.5.0-dev").err().unwrap();
        assert_eq!(
            err.to_string(),
            format!(
                "{} holds more than one ghee-render wheel of ghee 0.5.0-dev: \
                 ghee_render-0.5.0.dev0-cp310-abi3-linux_x86_64.whl, \
                 ghee_render-0.5.0.dev0-cp39-abi3-linux_x86_64.whl",
                &*dir
            )
        );
    }
}
