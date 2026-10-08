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
        "fragment": "void main(){ gl_FragColor = vec4(t_cover(uv, 20.0)); }",
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
        "fragment": "void main() {\n  float x = 1.0;\n  gl_FragColor = vec4(nope);\n}"}]}"#;
    let out = render(&dir.0, scene, &[]);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    let err = String::from_utf8(out.stderr).unwrap();
    // Nothing ahead of it, and one copy of the GL's log.
    let log = err
        .strip_prefix("compile error: pass 'main', line 3 of its fragment: ")
        .unwrap_or_else(|| panic!("{err}"));
    assert!(!log.contains("compile error"), "{err}");
    assert_eq!(log, format!("{}\n", log.trim_end()), "{err:?}");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_refusal_other_than_a_compile_error_prints_alone() {
    let dir = Dir::new("refusal");
    let scene = r#"{"passes": [{"name": "main", "channels": [{"buffer": "typo"}],
        "fragment": "void main(){ gl_FragColor = vec4(1.0); }"}]}"#;
    let out = render(&dir.0, scene, &[]);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    assert_eq!(
        String::from_utf8(out.stderr).unwrap(),
        "pass 'main': channel 0 binds to buffer 'typo', which is not a pass in this scene\n"
    );
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_sized_output_pass_says_once_that_it_renders_at_the_screen_size_and_quiet_says_nothing() {
    let dir = Dir::new("sized");
    let scene = r#"{"passes": [{"name": "main", "size": [1024, 256],
        "fragment": "void main(){ gl_FragColor = vec4(1.0); }"}]}"#;
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
    "fragment": "void main(){ vec2 p = floor(uv * iResolution.xy); gl_FragColor = vec4(p / 255.0, 0.0, 1.0); }"}]}"#;

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

/// A scene process that writes the $GHEE_SIZE that it starts with into
/// `size`, beside it, and draws black.
const SIZE_SCRIPT: &str = "#!/bin/sh\nprintf '%s' \"$GHEE_SIZE\" > size\n\
    printf '%s\\n' '{\"version\": 1, \"passes\": [{\"name\": \"main\", \
    \"fragment\": \"void main(){ gl_FragColor = vec4(0.0); }\"}]}'\n\
    while read -r l; do :; done\n";

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_scene_process_starts_with_the_size_that_it_renders_at_turned_by_rotate() {
    use std::os::unix::fs::PermissionsExt;
    let dir = Dir::new("size");
    let path = dir.join("sized");
    fs::write(&path, SIZE_SCRIPT).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    for (rotate, want) in [("0", "32x24"), ("90", "24x32")] {
        let out = Command::new(env!("CARGO_BIN_EXE_ghee"))
            .arg("render")
            .arg(&path)
            .args(["-o", &dir.join("out.png").to_string_lossy()])
            .args(["--size", "32x24", "--rotate", rotate, "--frames", "1"])
            .env("GHEE_ASSETS", "")
            .env("XDG_CACHE_HOME", &dir.0)
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        assert_eq!(fs::read_to_string(dir.join("size")).unwrap(), want);
    }
}

/// A scene process that writes the uv variables that it starts with into
/// `uv`, beside it, one `NAME=value` line each, takes no arguments, and
/// draws black.
const UV_SCRIPT: &str = r#"#!/bin/sh
{
    echo "UV_INDEX=$UV_INDEX"
    echo "UV_INDEX_STRATEGY=$UV_INDEX_STRATEGY"
    echo "UV_NO_BINARY_PACKAGE=$UV_NO_BINARY_PACKAGE"
    echo "UV_NO_BUILD_PACKAGE=$UV_NO_BUILD_PACKAGE"
} > uv
if [ "$1" = --ghee-args ]; then
    echo '{"args": []}'
    exit 0
fi
printf '%s\n' '{"version": 1, "passes": [{"name": "main", "fragment": "void main(){ gl_FragColor = vec4(0.0); }"}]}'
while read -r l; do :; done
"#;

/// A copy of ghee in `bin/` of a new directory, with `lib/ghee/` beside it
/// and UV_SCRIPT as `scene.sh`: the directory, ghee and the scene.
fn ghee_beside_lib(tag: &str) -> (Dir, PathBuf, PathBuf) {
    use std::os::unix::fs::PermissionsExt;
    let dir = Dir::new(tag);
    fs::create_dir_all(dir.join("bin")).unwrap();
    fs::create_dir_all(dir.join("lib/ghee")).unwrap();
    let ghee = dir.join("bin/ghee");
    fs::copy(env!("CARGO_BIN_EXE_ghee"), &ghee).unwrap();
    let scene = dir.join("scene.sh");
    fs::write(&scene, UV_SCRIPT).unwrap();
    fs::set_permissions(&scene, fs::Permissions::from_mode(0o755)).unwrap();
    (dir, ghee, scene)
}

/// The wheel of ghee's version in `lib/ghee/` of `dir`, empty. maturin
/// writes the version's `-dev` as `.dev0` in the wheel's name.
fn wheel_of_ghee(dir: &Dir) -> PathBuf {
    let version = env!("CARGO_PKG_VERSION").replace("-dev", ".dev0");
    let wheel = dir.join(&format!(
        "lib/ghee/ghee_render-{version}-cp310-abi3-linux_x86_64.whl"
    ));
    fs::write(&wheel, b"").unwrap();
    wheel
}

/// The variables that UV_SCRIPT wrote in `dir`, by name.
fn uv_vars(dir: &Dir) -> std::collections::HashMap<String, String> {
    fs::read_to_string(dir.join("uv"))
        .unwrap()
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// The links of the ghee-render page of the index at `url`, a file URL.
fn index_links(url: &str) -> Vec<String> {
    let index = url
        .strip_prefix("file://")
        .unwrap_or_else(|| panic!("{url}"));
    let page = fs::read_to_string(Path::new(index).join("ghee-render/index.html")).unwrap();
    page.split("href=\"")
        .skip(1)
        .filter_map(|s| s.split('"').next())
        .map(str::to_string)
        .collect()
}

#[test]
fn a_scene_process_takes_ghee_render_from_the_wheel_beside_ghee() {
    let (dir, ghee, scene) = ghee_beside_lib("wheel");
    // `ghee render <scene> -- --help` runs the scene to ask what it takes
    let probe = || {
        let out = Command::new(&ghee)
            .arg("render")
            .arg(&scene)
            .args(["--", "--help"])
            .env("XDG_CACHE_HOME", &dir.0)
            .env_remove("UV_INDEX")
            .env_remove("UV_INDEX_STRATEGY")
            .env_remove("UV_NO_BINARY_PACKAGE")
            .env_remove("UV_NO_BUILD_PACKAGE")
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        (uv_vars(&dir), String::from_utf8(out.stderr).unwrap())
    };

    // the wheel of ghee's version: an index that lists it alone
    let wheel = wheel_of_ghee(&dir);
    let (uv, _) = probe();
    let wheel = fs::canonicalize(&wheel).unwrap();
    assert_eq!(
        index_links(&uv["UV_INDEX"]),
        [format!("file://{}", wheel.display())]
    );
    assert_eq!(uv["UV_INDEX_STRATEGY"], "first-index");

    // a wheel of another version alone: a dev build refuses every
    // ghee-render, which PyPI has only older, and a release leaves uv alone
    fs::remove_file(&wheel).unwrap();
    fs::write(
        dir.join("lib/ghee/ghee_render-0.0.1-cp310-abi3-linux_x86_64.whl"),
        b"",
    )
    .unwrap();
    let (uv, err) = probe();
    assert_eq!(
        (uv["UV_INDEX"].as_str(), uv["UV_INDEX_STRATEGY"].as_str()),
        ("", "")
    );
    if env!("CARGO_PKG_VERSION").contains('-') {
        assert_eq!(uv["UV_NO_BINARY_PACKAGE"], "ghee-render");
        assert_eq!(uv["UV_NO_BUILD_PACKAGE"], "ghee-render");
        let lib = fs::canonicalize(dir.join("lib/ghee")).unwrap();
        assert_eq!(
            err,
            format!(
                "  careful  {} holds no ghee-render wheel of ghee {}, so uv refuses a scene \
                 that declares ghee-render\n",
                lib.display(),
                env!("CARGO_PKG_VERSION")
            )
        );
    } else {
        assert_eq!(uv["UV_NO_BINARY_PACKAGE"], "");
        assert_eq!(uv["UV_NO_BUILD_PACKAGE"], "");
    }
}

#[test]
fn an_empty_or_relative_xdg_cache_home_puts_the_wheel_index_under_home() {
    let (dir, ghee, scene) = ghee_beside_lib("wheel-home");
    wheel_of_ghee(&dir);
    for xdg in ["", "cache"] {
        let out = Command::new(&ghee)
            .arg("render")
            .arg(&scene)
            .args(["--", "--help"])
            .current_dir(&dir.0)
            .env("XDG_CACHE_HOME", xdg)
            .env("HOME", &dir.0)
            .env_remove("UV_INDEX")
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        let uv = uv_vars(&dir);
        let under = format!("file://{}/.cache/ghee/index-", dir.0.display());
        assert!(uv["UV_INDEX"].starts_with(&under), "{xdg:?}: {uv:?}");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_started_scene_takes_the_index_of_the_wheel_beside_ghee_ahead_of_its_own() {
    let (dir, ghee, scene) = ghee_beside_lib("wheel-start");
    let wheel = fs::canonicalize(wheel_of_ghee(&dir)).unwrap();
    let out = Command::new(&ghee)
        .arg("render")
        .arg(&scene)
        .args(["-o", &dir.join("out.png").to_string_lossy()])
        .args(["--size", "8x8", "--frames", "1"])
        .env("GHEE_ASSETS", "")
        .env("XDG_CACHE_HOME", &dir.0)
        .env("UV_INDEX", "file:///elsewhere")
        .env_remove("UV_INDEX_STRATEGY")
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    let uv = uv_vars(&dir);
    let (first, rest) = uv["UV_INDEX"]
        .split_once(' ')
        .unwrap_or_else(|| panic!("{uv:?}"));
    assert_eq!(index_links(first), [format!("file://{}", wheel.display())]);
    assert_eq!(rest, "file:///elsewhere");
    assert_eq!(uv["UV_INDEX_STRATEGY"], "first-index");
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
