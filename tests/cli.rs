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
        "code": "void main(){ gl_FragColor = vec4(t_cover(uv, 20.0)); }",
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

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_sized_output_pass_says_once_that_it_renders_at_the_screen_size_and_quiet_says_nothing() {
    let dir = Dir::new("sized");
    let scene = r#"{"passes": [{"name": "main", "size": [1024, 256],
        "code": "void main(){ gl_FragColor = vec4(1.0); }"}]}"#;
    let out = render(&dir.0, scene, &["--size", "256x64"]);
    assert!(out.status.success(), "{out:?}");
    let err = String::from_utf8(out.stderr).unwrap();
    let said: Vec<&str> = err.lines().filter(|l| l.contains("output pass")).collect();
    assert_eq!(
        said,
        [
            "  careful  output pass 'main' sets a size (1024x256) and renders at the screen size \
          (256x64); to keep its size, add a filter or a pass after it that reads it through a \
          channel"
        ],
        "{err}"
    );
    let out = render(&dir.0, scene, &["--size", "256x64", "-q"]);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(String::from_utf8(out.stderr).unwrap(), "");
}

/// Each pixel of the scene holds its column in red and its row from the top
/// in green.
const POSITION: &str = r#"{"passes": [{"name": "main",
    "code": "void main(){ vec2 p = floor(uv * iResolution.xy); gl_FragColor = vec4(p / 255.0, 0.0, 1.0); }"}]}"#;

/// The PNG's size and its RGBA rows, decoded by the png crate.
fn decode(path: &Path) -> ((u32, u32), Vec<u8>) {
    let mut reader = png::Decoder::new(fs::File::open(path).unwrap())
        .read_info()
        .unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!(info.color_type, png::ColorType::Rgba);
    buf.truncate(info.buffer_size());
    ((info.width, info.height), buf)
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn rotate_turns_the_written_frame_and_keeps_its_size() {
    let dir = Dir::new("rotate");
    let out = render(&dir.0, POSITION, &["--size", "8x4", "--rotate", "90"]);
    assert!(out.status.success(), "{out:?}");
    let ((w, h), px) = decode(&dir.join("out.png"));
    assert_eq!((w, h), (8, 4));
    // the scene is 4x8, and its bottom-left pixel lands top left
    assert_eq!(&px[..2], &[0, 7]);
    // its top-left pixel lands top right
    assert_eq!(&px[7 * 4..7 * 4 + 2], &[0, 0]);
}
