//! The ghee command as a user runs it: what it prints, and how it exits.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A directory of one test's own under the temp directory. The drop removes
/// it, with everything that the test wrote in it.
struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("ghee-cli-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        Dir(d)
    }

    fn join(&self, rel: &str) -> PathBuf {
        self.0.join(rel)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// `ghee render` of the scene JSON `scene` with the flags `args`, one frame
/// to a PNG in `dir`, with no $GHEE_ASSETS and the atlas cache in `dir`.
fn render(dir: &Path, scene: &str, args: &[&str]) -> Output {
    let path = dir.join("s.scene.json");
    fs::write(&path, scene).unwrap();
    Command::new(env!("CARGO_BIN_EXE_ghee"))
        .arg("render")
        .arg(&path)
        .args(["-o", &dir.join("out.png").to_string_lossy()])
        .args(args)
        .env("GHEE_ASSETS", "")
        .env("XDG_CACHE_HOME", dir)
        .output()
        .unwrap()
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn quiet_still_says_what_is_wrong_with_a_text_layer() {
    let dir = Dir::new("quiet");
    let scene = r#"{"passes": [{"name": "main",
        "code": "void main(){ gl_FragColor = vec4(t_cover(v_texCoord, 20.0)); }",
        "layers": [{"name": "t", "text": "hi", "font_family": "missing.ttf", "visible": false}]}]}"#;
    let out = render(&dir.0, scene, &["-q"]);
    assert!(out.status.success(), "{out:?}");
    let err = String::from_utf8(out.stderr).unwrap();
    // The font's problem, and nothing else.
    assert_eq!(err.lines().count(), 1, "{err}");
    assert!(err.contains("missing.ttf"), "{err}");
    assert!(dir.join("out.png").is_file());
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_compile_error_prints_once_naming_the_pass_and_the_authors_line() {
    let dir = Dir::new("compile");
    let scene = r#"{"passes": [{"name": "main",
        "code": "void main() {\n  float x = 1.0;\n  gl_FragColor = vec4(nope);\n}"}]}"#;
    let out = render(&dir.0, scene, &[]);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    let err = String::from_utf8(out.stderr).unwrap();
    // Nothing ahead of it, and one copy of the GL's log.
    let log = err
        .strip_prefix("compile error: pass 'main', line 3 of its code: ")
        .unwrap_or_else(|| panic!("{err}"));
    assert!(!log.contains("compile error"), "{err}");
}
