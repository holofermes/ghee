//! The console: the web page and the HTTP API that show and edit a running
//! scene.
//!
//! core is the API, with no transport under it. server serves the API and the
//! page over HTTP, and hands each edit to a render loop. store holds a
//! directory of scenes and the working copy of each. state holds the edits of
//! one scene and reads and writes its state file. feeds reads the facts about
//! the host for the page's render info.

pub mod core;
pub mod feeds;
pub mod server;
pub mod state;
pub mod store;

/// The web assets, compiled in. A binary carries no package directory. Each
/// name is the path under ghee/console/web/, which is what a URL asks for.
pub const WEB: [(&str, &[u8]); 5] = [
    (
        "index.html",
        include_bytes!("../../ghee/console/web/index.html"),
    ),
    (
        "h264client.js",
        include_bytes!("../../ghee/console/web/h264client.js"),
    ),
    ("sw.js", include_bytes!("../../ghee/console/web/sw.js")),
    (
        "manifest.json",
        include_bytes!("../../ghee/console/web/manifest.json"),
    ),
    (
        "icon.svg",
        include_bytes!("../../ghee/console/web/icon.svg"),
    ),
];

/// The content type and the bytes for a name in WEB. An unknown suffix gets
/// application/octet-stream.
pub fn static_asset(name: &str) -> (&'static str, &'static [u8]) {
    let ctype = match name.rsplit_once('.').map(|(_, ext)| ext) {
        Some("html") => "text/html",
        Some("js") => "text/javascript",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("css") => "text/css",
        Some("png") => "image/png",
        _ => "application/octet-stream",
    };
    let body = WEB
        .iter()
        .find(|(n, _)| *n == name)
        .map_or(&[][..], |(_, b)| *b);
    (ctype, body)
}

/// The bundled example scenes, compiled in. Scene loading reads files, so ghee
/// extracts this set to a cache directory first. A procedural example imports
/// ghee from the ghee-render package, which uv installs.
pub const EXAMPLES: [(&str, &[u8]); 42] = [
    (
        "assets.glsl",
        include_bytes!("../../ghee/examples/assets.glsl"),
    ),
    (
        "assets.scene.json",
        include_bytes!("../../ghee/examples/assets.scene.json"),
    ),
    ("bars.glsl", include_bytes!("../../ghee/examples/bars.glsl")),
    (
        "bars.scene.json",
        include_bytes!("../../ghee/examples/bars.scene.json"),
    ),
    (
        "feedback.glsl",
        include_bytes!("../../ghee/examples/feedback.glsl"),
    ),
    (
        "feedback.scene.json",
        include_bytes!("../../ghee/examples/feedback.scene.json"),
    ),
    ("fonts/DejaVuSans-Latin.ttf", crate::text::DEJAVU_FONT),
    (
        "fonts/LICENSE-DejaVu.txt",
        include_bytes!("../../ghee/examples/fonts/LICENSE-DejaVu.txt"),
    ),
    ("fonts/PixelOperator8.ttf", crate::text::PIXEL_FONT),
    (
        "fonts/LICENSE-PixelOperator.txt",
        include_bytes!("../../ghee/examples/fonts/LICENSE-PixelOperator.txt"),
    ),
    (
        "fonts/DepartureMono-Regular.otf",
        include_bytes!("../../ghee/examples/fonts/DepartureMono-Regular.otf"),
    ),
    (
        "fonts/LICENSE-DepartureMono.txt",
        include_bytes!("../../ghee/examples/fonts/LICENSE-DepartureMono.txt"),
    ),
    (
        "fonts/Monogram.ttf",
        include_bytes!("../../ghee/examples/fonts/Monogram.ttf"),
    ),
    (
        "fonts/LICENSE-Monogram.txt",
        include_bytes!("../../ghee/examples/fonts/LICENSE-Monogram.txt"),
    ),
    (
        "fonts/TitlePixel.ttf",
        include_bytes!("../../ghee/examples/fonts/TitlePixel.ttf"),
    ),
    (
        "fonts/LICENSE-PressStart2P.txt",
        include_bytes!("../../ghee/examples/fonts/LICENSE-PressStart2P.txt"),
    ),
    (
        "fonts/NotoColorEmoji-karaoke.ttf",
        include_bytes!("../../ghee/examples/fonts/NotoColorEmoji-karaoke.ttf"),
    ),
    (
        "fonts/LICENSE-NotoColorEmoji.txt",
        include_bytes!("../../ghee/examples/fonts/LICENSE-NotoColorEmoji.txt"),
    ),
    (
        "fonts/GrenzeGotisch-lens.ttf",
        include_bytes!("../../ghee/examples/fonts/GrenzeGotisch-lens.ttf"),
    ),
    (
        "fonts/LICENSE-GrenzeGotisch.txt",
        include_bytes!("../../ghee/examples/fonts/LICENSE-GrenzeGotisch.txt"),
    ),
    (
        "fonts/OldStandard-lens.ttf",
        include_bytes!("../../ghee/examples/fonts/OldStandard-lens.ttf"),
    ),
    (
        "fonts/OldStandard-Bold-lens.ttf",
        include_bytes!("../../ghee/examples/fonts/OldStandard-Bold-lens.ttf"),
    ),
    (
        "fonts/OldStandard-Italic-lens.ttf",
        include_bytes!("../../ghee/examples/fonts/OldStandard-Italic-lens.ttf"),
    ),
    (
        "fonts/LICENSE-OldStandard.txt",
        include_bytes!("../../ghee/examples/fonts/LICENSE-OldStandard.txt"),
    ),
    (
        "passes3-city.glsl",
        include_bytes!("../../ghee/examples/passes3-city.glsl"),
    ),
    (
        "passes3.glsl",
        include_bytes!("../../ghee/examples/passes3.glsl"),
    ),
    (
        "passes3.scene.json",
        include_bytes!("../../ghee/examples/passes3.scene.json"),
    ),
    (
        "passes2-sign.glsl",
        include_bytes!("../../ghee/examples/passes2-sign.glsl"),
    ),
    (
        "passes2.glsl",
        include_bytes!("../../ghee/examples/passes2.glsl"),
    ),
    (
        "passes2.scene.json",
        include_bytes!("../../ghee/examples/passes2.scene.json"),
    ),
    ("hero.py", include_bytes!("../../ghee/examples/hero.py")),
    ("host.py", include_bytes!("../../ghee/examples/host.py")),
    (
        "karaoke.py",
        include_bytes!("../../ghee/examples/karaoke.py"),
    ),
    ("lens.glsl", include_bytes!("../../ghee/examples/lens.glsl")),
    (
        "lens.scene.json",
        include_bytes!("../../ghee/examples/lens.scene.json"),
    ),
    (
        "lib/look.py",
        include_bytes!("../../ghee/examples/lib/look.py"),
    ),
    (
        "nowplaying.py",
        include_bytes!("../../ghee/examples/nowplaying.py"),
    ),
    (
        "passes.glsl",
        include_bytes!("../../ghee/examples/passes.glsl"),
    ),
    (
        "passes.scene.json",
        include_bytes!("../../ghee/examples/passes.scene.json"),
    ),
    (
        "spectrum.py",
        include_bytes!("../../ghee/examples/spectrum.py"),
    ),
    ("stamp.png", include_bytes!("../../ghee/examples/stamp.png")),
    ("stats.py", include_bytes!("../../ghee/examples/stats.py")),
];

/// The bundled example scenes on disk. $GHEE_EXAMPLES wins. Otherwise this
/// extracts the compiled-in set under the user cache. None means no cache
/// directory takes a write.
pub fn examples_dir() -> Option<String> {
    if let Some(d) = std::env::var("GHEE_EXAMPLES")
        .ok()
        .filter(|d| std::path::Path::new(d).is_dir())
    {
        return Some(d);
    }
    extract_examples().ok()
}

/// True when `dir` is the bundled examples directory, $GHEE_EXAMPLES or the
/// directory that examples_dir() extracts them to. The comparison resolves
/// symlinks, and extracts nothing.
pub fn is_examples_dir(dir: &std::path::Path) -> bool {
    let given = std::env::var("GHEE_EXAMPLES").ok();
    let wheel = crate::procedural::wheel::identity();
    is_examples_in(dir, given.as_deref(), &cache_home(), &wheel)
}

/// is_examples_dir() with `given` for $GHEE_EXAMPLES, and the examples of
/// `wheel` under the cache directory `base`.
fn is_examples_in(
    dir: &std::path::Path,
    given: Option<&str>,
    base: &std::path::Path,
    wheel: &str,
) -> bool {
    let examples = match given.filter(|d| std::path::Path::new(d).is_dir()) {
        Some(d) => std::path::PathBuf::from(d),
        None => examples_in(base, wheel),
    };
    match (dir.canonicalize(), examples.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// $XDG_CACHE_HOME, else ~/.cache, else the temp directory. An empty or
/// relative $XDG_CACHE_HOME counts as unset, as the XDG spec says.
pub(crate) fn cache_home() -> std::path::PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
}

fn extract_examples() -> std::io::Result<String> {
    extract_examples_in(&cache_home(), &crate::procedural::wheel::identity())
}

/// The directory of the examples under the cache directory `base`. Its name
/// comes from the examples, ghee's version and `wheel`, the identity of the
/// ghee-render wheel that ghee hands a scene: uv keeps one environment per
/// script path, and a new path makes it resolve each example again.
fn examples_in(base: &std::path::Path, wheel: &str) -> std::path::PathBuf {
    let mut key = Vec::new();
    for (name, body) in &EXAMPLES {
        key.extend_from_slice(name.as_bytes());
        key.push(0);
        key.extend_from_slice(body);
    }
    key.extend_from_slice(env!("CARGO_PKG_VERSION").as_bytes());
    key.push(0);
    key.extend_from_slice(wheel.as_bytes());
    let tag = crate::procedural::wheel::tag(&key);
    base.join("ghee").join(format!("examples-{tag}"))
}

/// extract_examples() under the cache directory `base`, into examples_in().
fn extract_examples_in(base: &std::path::Path, wheel: &str) -> std::io::Result<String> {
    let dir = examples_in(base, wheel);
    let done = dir.join(".complete");
    if !done.exists() {
        // Once per build, and it lands on whoever runs --demo first.
        crate::report::progress("examples", format!("unpacking to {}", dir.display()));
        for (name, body) in &EXAMPLES {
            let path = dir.join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, body)?;
            // ghee runs a procedural scene, so an extracted one needs the execute
            // bit. The shebang marks a scene. A module beside it must stay
            // without the bit, or a scan takes it for a scene.
            #[cfg(unix)]
            if body.starts_with(b"#!") {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))?;
            }
        }
        std::fs::write(done, b"")?;
        crate::report::progress_done();
        prune_examples(&dir);
    }
    Ok(dir.to_string_lossy().into_owned())
}

/// Remove every examples directory beside `kept` that another build or
/// version of ghee extracted.
fn prune_examples(kept: &std::path::Path) {
    let (Some(parent), Some(name)) = (kept.parent(), kept.file_name()) else {
        return;
    };
    for e in std::fs::read_dir(parent).into_iter().flatten().flatten() {
        let n = e.file_name();
        if n != name && n.to_string_lossy().starts_with("examples-") {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

/// $GHEE_EXAMPLES on the repo's examples, for a test that reaches
/// examples_dir(). The variable is process-wide, so every such test holds
/// this, and the drop puts the variable back.
#[cfg(test)]
pub(crate) struct ExamplesEnv {
    saved: Option<std::ffi::OsString>,
    _held: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
pub(crate) fn repo_examples() -> ExamplesEnv {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let held = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let saved = std::env::var_os("GHEE_EXAMPLES");
    std::env::set_var(
        "GHEE_EXAMPLES",
        concat!(env!("CARGO_MANIFEST_DIR"), "/ghee/examples"),
    );
    ExamplesEnv { saved, _held: held }
}

#[cfg(test)]
impl Drop for ExamplesEnv {
    fn drop(&mut self) {
        match self.saved.take() {
            Some(v) => std::env::set_var("GHEE_EXAMPLES", v),
            None => std::env::remove_var("GHEE_EXAMPLES"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_examples_extract_and_resolve() {
        let cache = std::env::temp_dir().join(format!("ghee-examples-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cache);
        let out = extract_examples_in(&cache, "").unwrap();
        let out = std::path::Path::new(&out);
        assert!(out.join("bars.scene.json").is_file());
        assert!(out.join("lib/look.py").is_file());
        assert!(out.join("hero.py").is_file());
        assert!(out.join("stats.py").is_file());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::metadata(out.join("hero.py"))
                .unwrap()
                .permissions();
            assert_eq!(perms.mode() & 0o111, 0o111);
        }
        let scene = crate::scene::Scene::load(out.join("bars.scene.json").to_str().unwrap());
        assert!(scene.is_ok());
        std::fs::remove_dir_all(&cache).unwrap();
    }

    #[test]
    fn a_directory_of_scenes_is_told_from_the_examples_with_nothing_extracted() {
        let cache = crate::testing::TempDir::new("examples-told");
        let base = std::path::Path::new(&*cache);
        let scenes = crate::testing::TempDir::new("examples-scenes");
        let dir = std::path::Path::new(&*scenes);
        assert!(!is_examples_in(dir, None, base, "w"));
        assert_eq!(std::fs::read_dir(base).unwrap().count(), 0);
        let out = extract_examples_in(base, "w").unwrap();
        assert!(is_examples_in(std::path::Path::new(&out), None, base, "w"));
        assert!(is_examples_in(dir, Some(&scenes), base, "w"));
    }

    #[test]
    fn a_wheel_built_again_extracts_the_examples_to_a_new_directory() {
        let cache = crate::testing::TempDir::new("examples-wheel");
        let base = std::path::Path::new(&*cache);
        let wheel = "/usr/lib/ghee/ghee_render-0.5.0-cp310-abi3-linux_aarch64.whl:4096";
        let first = extract_examples_in(base, &format!("{wheel}:1")).unwrap();
        let again = extract_examples_in(base, &format!("{wheel}:2")).unwrap();
        assert_ne!(first, again);
    }

    #[test]
    fn the_cache_keeps_the_examples_of_the_last_build_alone() {
        let cache = crate::testing::TempDir::new("examples-pruned");
        let base = std::path::Path::new(&*cache);
        let other = base.join("ghee/other");
        std::fs::create_dir_all(&other).unwrap();
        let first = extract_examples_in(base, "1").unwrap();
        let again = extract_examples_in(base, "2").unwrap();
        assert!(!std::path::Path::new(&first).exists());
        assert!(std::path::Path::new(&again)
            .join("bars.scene.json")
            .is_file());
        assert!(other.is_dir());
    }
}
