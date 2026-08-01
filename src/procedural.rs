//! Port of ghee/procedural.py (see that file for the reference implementation).
//!
//! A scene package is a directory `<name>/`, a `<name>.scene.json`, or a
//! `<name>.py`, holding any of:
//!   <name>.scene.json   the declarative shape (passes, arrays, params)
//!   <name>.py           a SceneController subclass, or a load() factory
//!
//! There is no Python interpreter in this build: a package's controller is
//! still resolved, so a scene that needs one says so, but it cannot run.
//! Nothing here builds a scene in code, feeds it per-frame arrays, or declares
//! build-time arguments, all of which live on the controller.

use std::path::Path;

use anyhow::{anyhow, bail, Result};

use crate::scene::Scene;

pub const SUFFIX: &str = ".scene.json";

/// os.path.splitext: the extension starts at the last dot of the basename, and
/// a basename of nothing but dots has none.
fn splitext(path: &str) -> &str {
    let sep = path.rfind('/').map_or(0, |i| i + 1);
    match path.rfind('.') {
        Some(dot) if dot >= sep && !path[sep..dot].bytes().all(|b| b == b'.') => &path[..dot],
        _ => path,
    }
}

/// os.path.basename(os.path.normpath(path)): empty and "." segments drop, ".."
/// pops the segment before it and dies at an absolute root.
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

fn prog(path: &str) -> String {
    let mut name = base_name(path);
    for suffix in [SUFFIX, ".py"] {
        if let Some(s) = name.strip_suffix(suffix) {
            name = s;
        }
    }
    format!("ghee render {name} --")
}

/// Load a scene package. Accepts a directory, a .scene.json, or a .py.
///
/// `args` are the command-line tokens after `--`. A scene declaring nothing
/// takes no args; strict=false leaves undeclared tokens alone, for a host
/// handing one set of args to every scene it loads.
pub fn load_scene(path: &str, args: &[String], strict: bool) -> Result<Scene> {
    let (json_path, py_path) = resolve(path);
    let base = match &json_path {
        Some(j) => Some(Scene::load(j)?),
        None => None,
    };
    if let Some(py) = py_path {
        bail!("{py}: Python scene controllers are unsupported in this build");
    }
    if strict && !args.is_empty() {
        bail!(
            "{}: error: unrecognized arguments: {}",
            prog(path),
            args.join(" ")
        );
    }
    base.ok_or_else(|| anyhow!("no scene at {path}: need a {SUFFIX} or a controller with build()"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const JSON_SCENE: &str = r#"{"version":1,"passes":[{"name":"main","code":"void main(){}",
        "arrays":[{"name":"level","size":2}]}]}"#;

    const PROVIDE_ONLY_PY: &str = "from ghee import SceneController\n\
        \n\
        class Feed(SceneController):\n\
        \x20   def provide(self, name, size, frame):\n\
        \x20       return [1.0] * size\n";

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

    #[test]
    fn a_controller_is_unsupported() {
        let d = tmpdir("controller");
        fs::write(format!("{d}/demo.py"), PROVIDE_ONLY_PY).unwrap();
        let e = load_scene(&format!("{d}/demo.py"), &no_args(), true).unwrap_err();
        assert_eq!(
            e.to_string(),
            format!("{d}/demo.py: Python scene controllers are unsupported in this build")
        );
    }

    #[test]
    fn a_bundled_controller_example_is_unsupported() {
        for name in ["spectrum.py", "text.py"] {
            let path = repo(&format!("ghee/examples/{name}"));
            let e = load_scene(&path, &no_args(), true).unwrap_err();
            assert_eq!(
                e.to_string(),
                format!("{path}: Python scene controllers are unsupported in this build")
            );
        }
    }

    #[test]
    fn json_plus_controller_resolves_both() {
        let d = format!("{}/wob", tmpdir("wob"));
        fs::create_dir_all(&d).unwrap();
        fs::write(format!("{d}/wob{SUFFIX}"), JSON_SCENE).unwrap();
        fs::write(format!("{d}/wob.py"), PROVIDE_ONLY_PY).unwrap();
        // a scene dir holds <name>.scene.json + <name>.py; every spelling of
        // the package resolves to the same pair
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
        assert!(e.to_string().contains("unsupported in this build"), "{e}");
    }

    #[test]
    fn a_controller_beside_broken_json_reports_the_json_first() {
        let d = tmpdir("broken");
        fs::write(format!("{d}/x{SUFFIX}"), "{ not json").unwrap();
        fs::write(format!("{d}/x.py"), PROVIDE_ONLY_PY).unwrap();
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
            format!("no scene at {d}/nope: need a .scene.json or a controller with build()")
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
        // a host handing one set of args to every scene it loads
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
    fn base_name_matches_normpath() {
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
