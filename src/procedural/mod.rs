//! A scene package is a directory `<name>/`, a `<name>.scene.json`, or an
//! executable. The package holds one or both of these members:
//!   <name>.scene.json   the declarative shape (passes, arrays, params)
//!   an executable       a procedural scene (docs/procedural.md)
//!
//! The execute bit marks a scene, in any language, under any name:
//! `<name>`, `<name>.py` and `<name>.sh` all belong to the package
//! `<name>`. `.scene.json` is the one special name, a document whatever
//! its mode bits say. Two executables under one stem are ambiguous, and
//! the error says so: choose one by naming the file itself.
//!
//! ghee runs the executable member, and procedural/process.rs owns it.
//! This module loads the documents.

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

/// (json path, executable path) for any spelling of the package.
///
/// The executable member is the file with the execute bit under the
/// package stem: the bare `<stem>`, or `<stem>.<ext>` under any extension.
/// A spelling that names a runnable file directly is its own choice, and a
/// named file without the bit gets the chmod hint. Two executable members
/// are ambiguous, and the error says: choose one.
pub fn resolve(path: &str) -> Result<(Option<String>, Option<String>)> {
    let stem = if Path::new(path).is_dir() {
        let sep = if path.ends_with('/') { "" } else { "/" };
        format!("{path}{sep}{}", base_name(path))
    } else if let Some(s) = path.strip_suffix(SUFFIX) {
        s.to_string()
    } else {
        splitext(path).to_string()
    };
    let j = format!("{stem}{SUFFIX}");
    let json = Path::new(&j).is_file().then_some(j);
    let named_file = Path::new(path).is_file() && !path.ends_with(SUFFIX);
    if named_file && (process::runnable(path) || path != stem) {
        return Ok((json, Some(path.to_string())));
    }
    let (mut runnable, _) = members(&stem);
    match runnable.len() {
        0 | 1 => Ok((json, runnable.pop())),
        _ => bail!(
            "ambiguous scene '{}': {}: choose one",
            scene_name(&stem),
            runnable.join(", ")
        ),
    }
}

/// The files of the package stem, the bare `<stem>` and `<stem>.<ext>`,
/// split into (with, without) the execute bit, sorted. A `.scene.json` is
/// a document and stays out.
fn members(stem: &str) -> (Vec<String>, Vec<String>) {
    let (dir, base) = match stem.rsplit_once('/') {
        Some((d, b)) => (d.to_string(), b.to_string()),
        None => (".".to_string(), stem.to_string()),
    };
    let prefix = &stem[..stem.len() - base.len()];
    let (mut with, mut without) = (Vec::new(), Vec::new());
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return (with, without);
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let is_member = name
            .strip_prefix(base.as_str())
            .is_some_and(|rest| rest.is_empty() || (rest.len() > 1 && rest.starts_with('.')));
        if !is_member || name.ends_with(SUFFIX) {
            continue;
        }
        let full = format!("{prefix}{name}");
        if !Path::new(&full).is_file() {
            continue;
        }
        if process::runnable(&full) {
            with.push(full);
        } else {
            without.push(full);
        }
    }
    with.sort();
    without.sort();
    (with, without)
}

/// True when the stem has an executable member. The scene-name lookup asks.
pub(crate) fn has_executable_member(stem: &str) -> bool {
    !members(stem).0.is_empty()
}

/// The name of a scene package: its basename without `.scene.json`, or
/// without the file's own extension.
pub fn scene_name(path: &str) -> String {
    let name = base_name(path);
    match name.strip_suffix(SUFFIX) {
        Some(s) => s.to_string(),
        None => splitext(name).to_string(),
    }
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
    let (json_path, exe) = resolve(path)?;
    let base = match &json_path {
        Some(j) => Some(Scene::load(j)?),
        None => None,
    };
    // procedural/process.rs starts an executable scene
    // this function loads documents, so an executable member is a dead end
    if let Some(x) = exe {
        if !process::runnable(&x) {
            bail!("{x}: no executable bit, and a procedural scene must be executable (chmod +x)");
        }
        bail!("{x}: an executable scene runs as a process");
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

    const JSON_SCENE: &str = r#"{"version":1,"passes":[{"name":"main","fragment":"void main(){}",
        "arrays":[{"name":"level","size":2}]}]}"#;

    const SCENE_PY: &str = "#!/usr/bin/env python3\n\
        import ghee\n\
        ghee.run(ghee.Scene(passes=[]))\n";

    fn tmpdir(tag: &str) -> crate::testing::TempDir {
        crate::testing::TempDir::new(&format!("procedural-{tag}"))
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
            format!(
                "{path}: no executable bit, and a procedural scene must be executable (chmod +x)"
            )
        );
        // with the bit, process.rs takes the file
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(crate::procedural::process::executable(&path), Some(path));
    }

    #[test]
    fn json_plus_executable_resolves_both() {
        let root = tmpdir("wob");
        let d = format!("{root}/wob");
        fs::create_dir_all(&d).unwrap();
        fs::write(format!("{d}/wob{SUFFIX}"), JSON_SCENE).unwrap();
        let py = format!("{d}/wob.py");
        fs::write(&py, SCENE_PY).unwrap();
        fs::set_permissions(&py, fs::Permissions::from_mode(0o755)).unwrap();
        // every spelling of the package resolves to the same pair
        for spelling in [
            d.clone(),
            format!("{d}/"),
            format!("{d}/wob"),
            format!("{d}/wob.py"),
            format!("{d}/wob{SUFFIX}"),
        ] {
            assert_eq!(
                resolve(&spelling).unwrap(),
                (Some(format!("{d}/wob{SUFFIX}")), Some(py.clone())),
                "{spelling}"
            );
        }
        // the runnable member belongs to process.rs, and the load says so
        let e = load_scene(&d, &no_args(), true).unwrap_err();
        assert!(e.to_string().contains("runs as a process"), "{e}");
    }

    #[test]
    fn a_module_without_the_bit_is_not_a_scene() {
        let root = tmpdir("module");
        let d = format!("{root}/module");
        fs::create_dir_all(&d).unwrap();
        fs::write(format!("{d}/mod{SUFFIX}"), JSON_SCENE).unwrap();
        fs::write(format!("{d}/mod.py"), "# a module the scenes import\n").unwrap();
        assert_eq!(
            resolve(&format!("{d}/mod")).unwrap(),
            (Some(format!("{d}/mod{SUFFIX}")), None)
        );
        assert!(load_scene(&format!("{d}/mod"), &no_args(), true).is_ok());
    }

    #[test]
    fn an_executable_is_a_scene_under_any_name() {
        let d = tmpdir("names");
        for name in ["wave.sh", "bare"] {
            let p = format!("{d}/{name}");
            fs::write(&p, "#!/bin/sh\nexit 0\n").unwrap();
            fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
        }
        assert_eq!(
            resolve(&format!("{d}/wave")).unwrap(),
            (None, Some(format!("{d}/wave.sh")))
        );
        assert_eq!(
            resolve(&format!("{d}/bare")).unwrap(),
            (None, Some(format!("{d}/bare")))
        );
        assert_eq!(scene_name(&format!("{d}/wave.sh")), "wave");
        assert_eq!(scene_name(&format!("{d}/bare")), "bare");
    }

    #[test]
    fn two_executables_error_choose_one() {
        let d = tmpdir("ambiguous");
        for name in ["twin.py", "twin.sh"] {
            let p = format!("{d}/{name}");
            fs::write(&p, "#!/bin/sh\nexit 0\n").unwrap();
            fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
        }
        let e = resolve(&format!("{d}/twin")).unwrap_err().to_string();
        assert!(e.contains("ambiguous scene 'twin'"), "{e}");
        assert!(e.contains("twin.py") && e.contains("twin.sh"), "{e}");
        assert!(e.contains("choose one"), "{e}");
        // naming the file is the choice
        assert_eq!(
            resolve(&format!("{d}/twin.sh")).unwrap(),
            (None, Some(format!("{d}/twin.sh")))
        );
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
        assert_eq!(resolve(&format!("{d}/nope")).unwrap(), (None, None));
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
            resolve(&format!("{d}/wob.tar.gz")).unwrap(),
            (Some(format!("{d}/wob.tar{SUFFIX}")), None)
        );
        assert_eq!(resolve(&format!("{d}/wob.gz")).unwrap(), (None, None));
        assert_eq!(resolve(&format!("{d}/.wob")).unwrap(), (None, None));
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
pub(crate) mod wheel;
