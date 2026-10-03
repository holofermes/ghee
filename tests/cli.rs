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
fn quiet_still_says_what_is_wrong_with_a_text() {
    let dir = Dir::new("quiet");
    let scene = r#"{"texts": {"t": {"value": "hi", "font": "missing.ttf"}},
        "passes": [{"name": "main",
                    "code": "void main(){ gl_FragColor = vec4(t_cover(v_texCoord, 20.0)); }"}]}"#;
    let out = render(&dir.0, scene, &["-q"]);
    assert!(out.status.success(), "{out:?}");
    let err = String::from_utf8(out.stderr).unwrap();
    // The font's problem, and nothing else.
    assert_eq!(err.lines().count(), 1, "{err}");
    assert!(err.contains("missing.ttf"), "{err}");
    assert!(dir.join("out.png").is_file());
}
