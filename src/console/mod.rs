//! The console modules.
//!
//! core is the API. store holds the scenes. feeds fills in the host info.
//! server mounts them on a render loop.

pub mod core;
pub mod feeds;
pub mod server;
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
/// extracts this set to a content-keyed cache directory first. A procedural
/// example imports ghee from the ghee-render package, which pip installs.
pub const EXAMPLES: [(&str, &[u8]); 19] = [
    ("bars.glsl", include_bytes!("../../ghee/examples/bars.glsl")),
    (
        "bars.scene.json",
        include_bytes!("../../ghee/examples/bars.scene.json"),
    ),
    (
        "bluesun.py",
        include_bytes!("../../ghee/examples/bluesun.py"),
    ),
    (
        "blur.scene.json",
        include_bytes!("../../ghee/examples/blur.scene.json"),
    ),
    ("clocks.py", include_bytes!("../../ghee/examples/clocks.py")),
    (
        "echo.scene.json",
        include_bytes!("../../ghee/examples/echo.scene.json"),
    ),
    (
        "fonts/DejaVuSans-Latin.ttf",
        include_bytes!("../../ghee/examples/fonts/DejaVuSans-Latin.ttf"),
    ),
    (
        "fonts/LICENSE-DejaVu.txt",
        include_bytes!("../../ghee/examples/fonts/LICENSE-DejaVu.txt"),
    ),
    ("hero.py", include_bytes!("../../ghee/examples/hero.py")),
    (
        "lettering.py",
        include_bytes!("../../ghee/examples/lettering.py"),
    ),
    (
        "lib/look.py",
        include_bytes!("../../ghee/examples/lib/look.py"),
    ),
    (
        "mix.scene.json",
        include_bytes!("../../ghee/examples/mix.scene.json"),
    ),
    (
        "spectrum.py",
        include_bytes!("../../ghee/examples/spectrum.py"),
    ),
    (
        "stamp.glsl",
        include_bytes!("../../ghee/examples/stamp.glsl"),
    ),
    ("stamp.png", include_bytes!("../../ghee/examples/stamp.png")),
    (
        "stamp.scene.json",
        include_bytes!("../../ghee/examples/stamp.scene.json"),
    ),
    ("text.py", include_bytes!("../../ghee/examples/text.py")),
    (
        "trail.scene.json",
        include_bytes!("../../ghee/examples/trail.scene.json"),
    ),
    (
        "twopass.scene.json",
        include_bytes!("../../ghee/examples/twopass.scene.json"),
    ),
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

fn extract_examples() -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for (name, body) in &EXAMPLES {
        h.update(name.as_bytes());
        h.update([0]);
        h.update(body);
    }
    let tag: String = h
        .finalize()
        .iter()
        .take(6)
        .map(|b| format!("{b:02x}"))
        .collect();
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    let dir = base.join("ghee").join(format!("examples-{tag}"));
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
    }
    Ok(dir.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_examples_extract_and_resolve() {
        let dir = examples_dir().unwrap();
        let dir = std::path::Path::new(&dir);
        assert!(dir.join("bars.scene.json").is_file());
        assert!(dir.join("lib/look.py").is_file());
        // examples_dir answers with $GHEE_EXAMPLES when it is set, so read the
        // extraction itself
        let out = extract_examples().unwrap();
        let out = std::path::Path::new(&out);
        assert!(out.join("hero.py").is_file());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::metadata(out.join("hero.py"))
                .unwrap()
                .permissions();
            assert_eq!(perms.mode() & 0o111, 0o111);
        }
        let scene = crate::scene::Scene::load(dir.join("bars.scene.json").to_str().unwrap());
        assert!(scene.is_ok());
    }
}
