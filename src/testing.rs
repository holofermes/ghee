//! What the unit tests share.

use std::ffi::OsStr;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;
use std::thread;
use std::time::Duration;

/// A patience longer than any run. A timed wait given it waits for its
/// event, and the test runner ends a run that hangs.
pub(crate) const NEVER: Duration = Duration::from_secs(u32::MAX as u64);

/// Poll until `done` holds.
pub(crate) fn until(mut done: impl FnMut() -> bool) {
    while !done() {
        thread::sleep(Duration::from_millis(1));
    }
}

/// The calling thread's clock, standing at the time given until a sleep or
/// `pass` moves it. The drop puts the machine's clock back.
pub(crate) struct Clock;

impl Clock {
    pub(crate) fn at(t: f64) -> Clock {
        crate::stats::set_clock(Some(t));
        Clock
    }

    pub(crate) fn pass(&self, seconds: f64) {
        crate::stats::sleep(seconds);
    }
}

impl Drop for Clock {
    fn drop(&mut self) {
        crate::stats::set_clock(None);
    }
}

/// An empty directory of one test's own under the temp directory. The drop
/// removes it, with everything that the test wrote in it.
pub(crate) struct TempDir(String);

impl TempDir {
    pub(crate) fn new(tag: &str) -> TempDir {
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        let d = std::env::temp_dir().join(format!("ghee-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        TempDir(d.to_string_lossy().into_owned())
    }

    pub(crate) fn join(&self, rel: impl AsRef<Path>) -> PathBuf {
        Path::new(&self.0).join(rel)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

impl std::ops::Deref for TempDir {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TempDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<Path> for TempDir {
    fn as_ref(&self) -> &Path {
        Path::new(&self.0)
    }
}

impl AsRef<OsStr> for TempDir {
    fn as_ref(&self) -> &OsStr {
        OsStr::new(&self.0)
    }
}

impl AsRef<str> for TempDir {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// A fork that lands while another thread holds a write descriptor on a
/// fresh executable hands that descriptor to the child, and the exec of that
/// file answers ETXTBSY until it closes. Writing a script takes this lock
/// exclusively and spawning one takes it shared, so the two never overlap,
/// and concurrent spawns still run together.
pub(crate) static SCRIPTS: RwLock<()> = RwLock::new(());

/// Write the executable `name` in `dir`, holding `body`, and return its path.
pub(crate) fn script(dir: &str, name: &str, body: &str) -> String {
    use std::os::unix::fs::PermissionsExt;
    let _write = SCRIPTS.write().unwrap();
    let path = format!("{dir}/{name}");
    std::fs::write(&path, body).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

/// A png of `w` by `h` grey RGB pixels.
pub(crate) fn png(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().unwrap();
    writer
        .write_image_data(&vec![128; (w * h * 3) as usize])
        .unwrap();
    drop(writer);
    out
}
