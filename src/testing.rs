//! What the unit tests share.

use std::ffi::OsStr;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

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
