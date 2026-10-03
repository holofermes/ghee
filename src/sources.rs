//! A scene load reads files. record() notes one file. collect() gathers
//! them. A Snapshot polls that set, so a running scene sees an edit on disk.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::stats::monotonic;

/// seconds between stale()'s own disk checks
pub const STALE_INTERVAL: f64 = 1.0;

thread_local! {
    static COLLECTING: RefCell<Option<HashSet<PathBuf>>> = const { RefCell::new(None) };
}

/// An absolute path. It joins onto the working directory and normalises with
/// no disk read. A symlink stays unresolved, under the load's own spelling.
fn abspath(path: &Path) -> PathBuf {
    let joined = match env::current_dir() {
        Ok(cwd) if path.is_relative() => cwd.join(path),
        _ => path.to_path_buf(),
    };
    let mut out: Vec<Component<'_>> = Vec::new();
    for c in joined.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => match out.last() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                Some(Component::RootDir) => {}
                _ => out.push(c),
            },
            _ => out.push(c),
        }
    }
    out.iter().collect()
}

/// Note one file read by the scene load.
pub fn record(path: impl AsRef<Path>) {
    COLLECTING.with(|c| {
        if let Some(into) = c.borrow_mut().as_mut() {
            into.insert(abspath(path.as_ref()));
        }
    });
}

/// Gather what record() sees on this thread while `f` runs. Collects nest: the
/// outer collect resumes when the inner one ends.
pub fn collect<T>(f: impl FnOnce() -> T) -> (T, HashSet<PathBuf>) {
    let prev = COLLECTING.with(|c| c.borrow_mut().replace(HashSet::new()));
    let out = f();
    let paths = COLLECTING.with(|c| {
        let mut b = c.borrow_mut();
        let paths = b.take().unwrap_or_default();
        *b = prev;
        paths
    });
    (out, paths)
}

fn mtime(path: &Path) -> i128 {
    fs::metadata(path).map_or(-1, |m| {
        i128::from(m.mtime()) * 1_000_000_000 + i128::from(m.mtime_nsec())
    })
}

fn digest(path: &Path) -> String {
    fs::read(path).map_or_else(
        |_| String::new(),
        |bytes| {
            Sha256::digest(&bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect()
        },
    )
}

/// A compiled scene's files, with the mtime and digest of the compile.
pub struct Snapshot {
    interval: f64,
    clock: Box<dyn Fn() -> f64 + Send + Sync>,
    files: BTreeMap<PathBuf, (i128, String)>,
    stale: bool,
    checked: f64,
}

impl Snapshot {
    /// The files behind a scene. stale() checks them once a STALE_INTERVAL at
    /// most.
    pub fn new(paths: impl IntoIterator<Item = impl AsRef<Path>>) -> Snapshot {
        Snapshot::with(paths, STALE_INTERVAL, Box::new(monotonic))
    }

    pub fn with(
        paths: impl IntoIterator<Item = impl AsRef<Path>>,
        interval: f64,
        clock: Box<dyn Fn() -> f64 + Send + Sync>,
    ) -> Snapshot {
        let files = paths
            .into_iter()
            .map(|p| {
                let p = p.as_ref().to_path_buf();
                let seen = (mtime(&p), digest(&p));
                (p, seen)
            })
            .collect();
        let checked = clock();
        Snapshot {
            interval,
            clock,
            files,
            stale: false,
            checked,
        }
    }

    /// True when a file changed since the compile read it. Inside
    /// `interval` the last check answers.
    pub fn stale(&mut self) -> bool {
        let now = (self.clock)();
        if now - self.checked < self.interval {
            return self.stale;
        }
        self.checked = now;
        self.stale = self.diverged();
        self.stale
    }

    fn diverged(&mut self) -> bool {
        let mut out = false;
        for (path, seen) in &mut self.files {
            let fresh = mtime(path);
            if fresh == seen.0 {
                continue;
            }
            if digest(path) == seen.1 {
                seen.0 = fresh; // a copy: same content, new mtime
            } else {
                out = true;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::fs::FileTimes;
    use std::time::Duration;

    use super::*;

    const CODE: &str = "void main(){ gl_FragColor = vec4(1.0); }";

    fn tmpdir(tag: &str) -> crate::testing::TempDir {
        crate::testing::TempDir::new(&format!("sources-{tag}"))
    }

    /// A new mtime over identical content. A file copy leaves this behind.
    fn touch(path: &Path) {
        let m = fs::metadata(path).unwrap();
        let times = FileTimes::new()
            .set_accessed(m.accessed().unwrap())
            .set_modified(m.modified().unwrap() + Duration::from_secs(1));
        fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_times(times)
            .unwrap();
    }

    /// Write, then push the mtime past what a snapshot holds. A stat cannot
    /// see a rewrite inside one timestamp tick.
    fn write(path: &Path, text: &str) {
        fs::write(path, text).unwrap();
        touch(path);
    }

    /// A file in a directory of its own, which lives as long as the TempDir.
    fn watched(tag: &str) -> (crate::testing::TempDir, PathBuf) {
        let d = tmpdir(tag);
        let p = d.join("watched.glsl");
        fs::write(&p, CODE).unwrap();
        (d, p)
    }

    #[test]
    fn nothing_is_recorded_outside_a_collect() {
        record("/tmp/uncollected.glsl"); // no collector, so record() does nothing
        let ((), paths) = collect(|| ());
        assert!(paths.is_empty());
    }

    #[test]
    fn a_collect_gathers_what_the_load_read() {
        let ((), paths) = collect(|| {
            record("/tmp/one.glsl");
            record("/tmp/sub/../one.glsl"); // the same file, spelled twice
            record("/tmp/two.glsl");
        });
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(Path::new("/tmp/one.glsl")));
    }

    #[test]
    fn a_relative_path_is_recorded_absolute() {
        let ((), paths) = collect(|| record("watched.glsl"));
        let want = env::current_dir().unwrap().join("watched.glsl");
        assert_eq!(paths, HashSet::from([want]));
    }

    #[test]
    fn an_inner_collect_hands_the_outer_one_back() {
        let (inner, outer) = collect(|| {
            record("/tmp/outer.glsl");
            let ((), inner) = collect(|| record("/tmp/inner.glsl"));
            inner
        });
        assert_eq!(inner, HashSet::from([PathBuf::from("/tmp/inner.glsl")]));
        assert_eq!(outer, HashSet::from([PathBuf::from("/tmp/outer.glsl")]));
    }

    fn snap(paths: &[&Path]) -> Snapshot {
        let paths: Vec<PathBuf> = paths.iter().map(|p| p.to_path_buf()).collect();
        Snapshot::with(paths, 0.0, Box::new(monotonic))
    }

    #[test]
    fn untouched_files_are_not_stale() {
        let (_d, p) = watched("untouched");
        assert!(!snap(&[&p]).stale());
    }

    #[test]
    fn changed_content_is_stale() {
        let (_d, p) = watched("changed");
        let mut s = snap(&[&p]);
        write(&p, &CODE.replace("1.0", "0.5"));
        assert!(s.stale());
    }

    #[test]
    fn a_copy_that_only_rewrote_the_mtime_is_not_stale() {
        let (_d, p) = watched("copied");
        let mut s = snap(&[&p]);
        touch(&p);
        assert!(!s.stale());
    }

    #[test]
    fn a_copied_file_is_not_hashed_twice() {
        // the snapshot takes the new mtime, so a copied tree costs one hash
        // per file
        let (_d, p) = watched("rehash");
        let mut s = snap(&[&p]);
        touch(&p);
        assert!(!s.stale());
        assert_eq!(s.files[&p].0, mtime(&p));
    }

    #[test]
    fn a_file_that_disappeared_is_stale() {
        let (_d, p) = watched("gone");
        let mut s = snap(&[&p]);
        fs::remove_file(&p).unwrap();
        assert!(s.stale());
    }

    #[test]
    fn a_scene_with_no_files_behind_it_is_never_stale() {
        assert!(!snap(&[]).stale());
    }

    #[test]
    fn an_unchanged_file_is_never_hashed_again() {
        let (_d, p) = watched("unchanged");
        let mut s = snap(&[&p]);
        let seen = s.files[&p].clone();
        for _ in 0..5 {
            assert!(!s.stale());
        }
        assert_eq!(s.files[&p], seen);
    }

    #[test]
    fn disk_is_read_at_most_once_an_interval() {
        let (_d, p) = watched("interval");
        let now = std::sync::Arc::new(std::sync::Mutex::new(1000.0f64));
        let clock = std::sync::Arc::clone(&now);
        let mut s = Snapshot::with([&p], 10.0, Box::new(move || *clock.lock().unwrap()));
        write(&p, "void main(){ discard; }");
        assert!(!s.stale()); // inside the interval, the last answer stands
        *now.lock().unwrap() += 11.0;
        assert!(s.stale());
    }
}
