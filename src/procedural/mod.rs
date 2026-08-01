//! A scene package is a directory `<name>/`, a `<name>.scene.json`, or a `<name>.py`.
//! The package holds one or both of these files:
//!   <name>.scene.json   the declarative shape (passes, arrays, params)
//!   <name>.py           a procedural scene (docs/procedural.md)
//!
//! ghee runs a `<name>.py`, and procedural/process.rs owns it.
//! This module loads one only to report a missing executable bit.

use std::path::Path;

use anyhow::{anyhow, bail, Result};

use crate::scene::Scene;

pub const SUFFIX: &str = ".scene.json";

/// The extension starts at the last dot of the basename. A basename of nothing
/// but dots has none.
fn splitext(path: &str) -> &str {
    let sep = path.rfind('/').map_or(0, |i| i + 1);
    match path.rfind('.') {
        Some(dot) if dot >= sep && !path[sep..dot].bytes().all(|b| b == b'.') => &path[..dot],
        _ => path,
    }
}

/// The last segment of a normalized path.
/// An empty segment drops, and a "." segment drops.
/// A ".." segment pops the segment before it, and stops at an absolute root.
fn base_name(path: &str) -> &str {
    let root = path.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => match parts.last() {
                Some(&"..") => parts.push(".."),
                None if root => {}
                None => parts.push(".."),
                _ => {
                    parts.pop();
                }
            },
            s => parts.push(s),
        }
    }
    parts.last().copied().unwrap_or(if root { "" } else { "." })
}

/// (json path, py path) for a directory, a .scene.json, or a .py.
pub fn resolve(path: &str) -> (Option<String>, Option<String>) {
    let stem = if Path::new(path).is_dir() {
        let sep = if path.ends_with('/') { "" } else { "/" };
        format!("{path}{sep}{}", base_name(path))
    } else if let Some(s) = path.strip_suffix(".py") {
        s.to_string()
    } else if let Some(s) = path.strip_suffix(SUFFIX) {
        s.to_string()
    } else {
        splitext(path).to_string()
    };
    let (j, p) = (format!("{stem}{SUFFIX}"), format!("{stem}.py"));
    (
        Path::new(&j).is_file().then_some(j),
        Path::new(&p).is_file().then_some(p),
    )
}

/// The name of a scene package: its basename without the package suffix.
pub fn scene_name(path: &str) -> String {
    let mut name = base_name(path);
    for suffix in [SUFFIX, ".py"] {
        if let Some(s) = name.strip_suffix(suffix) {
            name = s;
        }
    }
    name.to_string()
}

fn prog(path: &str) -> String {
    format!("ghee render {} --", scene_name(path))
}

/// Load a scene package. The path names a directory, a .scene.json, or a .py.
///
/// `args` are the command-line tokens after `--`.
/// A scene that declares nothing takes no tokens.
/// strict=false leaves an undeclared token alone.
/// A host that hands one set of tokens to every scene needs strict=false.
pub fn load_scene(path: &str, args: &[String], strict: bool) -> Result<Scene> {
    let (json_path, py_path) = resolve(path);
    let base = match &json_path {
        Some(j) => Some(Scene::load(j)?),
        None => None,
    };
    // procedural/process.rs starts an executable scene
    // this function loads a file, so an unexecutable .py is a dead end
    if let Some(py) = py_path {
        bail!("{py}: no executable bit; a procedural scene must be executable (chmod +x)");
    }
    if strict && !args.is_empty() {
        bail!(
            "{}: error: unrecognized arguments: {}",
            prog(path),
            args.join(" ")
        );
    }
    base.ok_or_else(|| anyhow!("no scene at {path}: need a {SUFFIX} or an executable scene"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    const JSON_SCENE: &str = r#"{"version":1,"passes":[{"name":"main","code":"void main(){}",
        "arrays":[{"name":"level","size":2}]}]}"#;

    const SCENE_PY: &str = "#!/usr/bin/env python3\n\
        import ghee\n\
        ghee.run(ghee.Scene(passes=[]))\n";

    fn tmpdir(tag: &str) -> String {
        let d = std::env::temp_dir().join(format!("ghee-procedural-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d.to_string_lossy().into_owned()
    }

    fn repo(rel: &str) -> String {
        format!("{}/{rel}", env!("CARGO_MANIFEST_DIR"))
    }

    fn no_args() -> Vec<String> {
        Vec::new()
    }

    fn args(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn json_standalone() {
        let scene = load_scene(&repo("ghee/examples/bars.scene.json"), &no_args(), true).unwrap();
        assert!(!scene.passes.is_empty());
    }

    /// A .py scene runs as a process. Without the executable bit it renders nothing.
    #[test]
    fn a_scene_py_without_the_executable_bit_says_so() {
        let d = tmpdir("controller");
        let path = format!("{d}/demo.py");
        fs::write(&path, SCENE_PY).unwrap();
        let e = load_scene(&path, &no_args(), true).unwrap_err();
        assert_eq!(
            e.to_string(),
            format!("{path}: no executable bit; a procedural scene must be executable (chmod +x)")
        );
        // with the bit, process.rs takes the file
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(crate::procedural::process::executable(&path), Some(path));
    }

    #[test]
    fn json_plus_scene_py_resolves_both() {
        let d = format!("{}/wob", tmpdir("wob"));
        fs::create_dir_all(&d).unwrap();
        fs::write(format!("{d}/wob{SUFFIX}"), JSON_SCENE).unwrap();
        fs::write(format!("{d}/wob.py"), SCENE_PY).unwrap();
        // a scene directory holds <name>.scene.json and <name>.py
        // every spelling of the package resolves to the same pair
        for spelling in [
            d.clone(),
            format!("{d}/"),
            format!("{d}/wob"),
            format!("{d}/wob.py"),
            format!("{d}/wob{SUFFIX}"),
        ] {
            assert_eq!(
                resolve(&spelling),
                (
                    Some(format!("{d}/wob{SUFFIX}")),
                    Some(format!("{d}/wob.py"))
                ),
                "{spelling}"
            );
        }
        let e = load_scene(&d, &no_args(), true).unwrap_err();
        assert!(e.to_string().contains("no executable bit"), "{e}");
    }

    #[test]
    fn a_scene_py_beside_broken_json_reports_the_json_first() {
        let d = tmpdir("broken");
        fs::write(format!("{d}/x{SUFFIX}"), "{ not json").unwrap();
        fs::write(format!("{d}/x.py"), SCENE_PY).unwrap();
        let e = load_scene(&format!("{d}/x"), &no_args(), true).unwrap_err();
        assert!(e.to_string().contains("invalid JSON"), "{e}");
    }

    #[test]
    fn a_package_that_is_neither_reports_no_scene() {
        let d = tmpdir("empty");
        assert_eq!(resolve(&format!("{d}/nope")), (None, None));
        let e = load_scene(&format!("{d}/nope"), &no_args(), true).unwrap_err();
        assert_eq!(
            e.to_string(),
            format!("no scene at {d}/nope: need a .scene.json or an executable scene")
        );
    }

    #[test]
    fn only_the_package_suffixes_resolve() {
        let d = tmpdir("suffixes");
        fs::write(format!("{d}/wob.tar{SUFFIX}"), JSON_SCENE).unwrap();
        // wob.tar.gz splits at its last dot, so the package is wob.tar
        assert_eq!(
            resolve(&format!("{d}/wob.tar.gz")),
            (Some(format!("{d}/wob.tar{SUFFIX}")), None)
        );
        assert_eq!(resolve(&format!("{d}/wob.gz")), (None, None));
        assert_eq!(resolve(&format!("{d}/.wob")), (None, None));
    }

    #[test]
    fn a_dot_segment_does_not_change_the_package() {
        let d = tmpdir("dots");
        fs::write(format!("{d}/flat{SUFFIX}"), JSON_SCENE).unwrap();
        let scene = load_scene(&format!("{d}/./flat{SUFFIX}"), &no_args(), true).unwrap();
        assert_eq!(scene.passes.len(), 1);
        assert_eq!(prog(&format!("{d}/sub/../flat.py")), "ghee render flat --");
        assert_eq!(prog(&format!("{d}/wob/")), "ghee render wob --");
    }

    #[test]
    fn a_json_scene_takes_no_args() {
        let d = tmpdir("args");
        let path = format!("{d}/flat{SUFFIX}");
        fs::write(&path, JSON_SCENE).unwrap();
        let e = load_scene(&path, &args(&["--emit", "8x8"]), true).unwrap_err();
        assert_eq!(
            e.to_string(),
            "ghee render flat --: error: unrecognized arguments: --emit 8x8"
        );
        // a host hands one set of tokens to every scene it loads
        assert!(load_scene(&path, &args(&["--emit", "8x8"]), false).is_ok());
    }

    #[test]
    fn a_missing_package_parses_its_args_first() {
        let d = tmpdir("args-missing");
        let e = load_scene(&format!("{d}/nope"), &args(&["--emit", "8x8"]), true).unwrap_err();
        assert!(
            e.to_string().contains("unrecognized arguments: --emit 8x8"),
            "{e}"
        );
    }

    #[test]
    fn base_name_normalizes_the_path() {
        for (path, want) in [
            ("a/b/", "b"),
            ("a//b", "b"),
            ("a/./b", "b"),
            ("a/b/.", "b"),
            ("a/b/..", "a"),
            ("a/b/../..", "."),
            ("/a/..", ""),
            ("/..", ""),
            ("../x", "x"),
            ("..", ".."),
            ("/", ""),
            ("", "."),
        ] {
            assert_eq!(base_name(path), want, "{path}");
        }
    }

    #[test]
    fn splitext_keeps_a_dotfile_whole() {
        assert_eq!(splitext("x/wob.tar.gz"), "x/wob.tar");
        assert_eq!(splitext("x/.hidden"), "x/.hidden");
        assert_eq!(splitext("x.d/wob"), "x.d/wob");
        assert_eq!(splitext("wob"), "wob");
    }
}

pub mod process;
