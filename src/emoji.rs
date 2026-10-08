//! The emoji fallback: Noto Color Emoji, for the emoji that a text layer's
//! font lacks.
//!
//! A scene load looks for FILE beside the scene, then in each directory on
//! `$GHEE_ASSETS`, then in SYSTEM_DIR, and hands every text layer of the
//! scene the first file found, as an EmojiFont. The process holds one
//! EmojiFont for each file, and the EmojiFont reads its file once, when the
//! words of a layer first need it, so every layer of every scene shares one
//! copy of the font. A caller that must not wait takes ready(), which reads
//! the file on a thread of its own.

use std::fmt;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

use ttf_parser::Face;

use crate::scene::{join, normpath};
use crate::text::Font;

/// The file of Noto Color Emoji.
pub const FILE: &str = "NotoColorEmoji.ttf";
/// The system's directory of Noto fonts, searched last.
pub const SYSTEM_DIR: &str = "/usr/share/fonts/truetype/noto";
/// The font's name in an error.
pub const NAME: &str = "Noto Color Emoji";

/// Noto Color Emoji at one path.
pub struct EmojiFont {
    path: String,
    bytes: OnceLock<Font>,
    /// Set by the first ready(), which starts the read.
    reading: AtomicBool,
}

impl EmojiFont {
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The bytes of the font, read at the first call, or why they failed
    /// to load.
    pub fn font(&self) -> Font {
        self.bytes
            .get_or_init(|| {
                let what = format!("{NAME} ({})", self.path);
                let bytes = std::fs::read(&self.path)
                    .map_err(|e| format!("{what}: {}", crate::scene::strerror(&e)))?;
                match Face::parse(&bytes, 0) {
                    Ok(_) => Ok(Arc::new(bytes)),
                    Err(e) => Err(format!("{what}: unreadable font: {e}")),
                }
            })
            .clone()
    }

    /// The bytes of the font, or why they failed to load, once read. None
    /// while a thread reads them, which the first call starts. A font()
    /// meanwhile waits for that read.
    pub fn ready(self: &Arc<Self>) -> Option<Font> {
        if let Some(f) = self.bytes.get() {
            return Some(f.clone());
        }
        if !self.reading.swap(true, Ordering::AcqRel) {
            let font = Arc::clone(self);
            thread::Builder::new()
                .name("ghee-emoji".into())
                .spawn(move || drop(font.font()))
                .expect("spawn the emoji font's read");
        }
        self.bytes.get().cloned()
    }
}

impl fmt::Debug for EmojiFont {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "EmojiFont({:?})", self.path)
    }
}

impl PartialEq for EmojiFont {
    fn eq(&self, other: &EmojiFont) -> bool {
        self.path == other.path
    }
}

/// The size and the mtime of a file, or None for a file that is not there.
type Stamp = Option<(u64, i64, i64)>;

fn stamp(path: &str) -> Stamp {
    let m = std::fs::metadata(path).ok()?;
    Some((m.len(), m.mtime(), m.mtime_nsec()))
}

/// The EmojiFont of the file at `path`. Every call returns the same one
/// while the file keeps its size and its mtime.
pub(crate) fn at(path: &str) -> Arc<EmojiFont> {
    static FONTS: Mutex<Vec<(Stamp, Arc<EmojiFont>)>> = Mutex::new(Vec::new());
    let now = stamp(path);
    let mut fonts = FONTS.lock().unwrap_or_else(|e| e.into_inner());
    let held = fonts.iter().position(|(_, f)| f.path == path);
    if let Some(i) = held.filter(|i| fonts[*i].0 == now) {
        return Arc::clone(&fonts[i].1);
    }
    let font = Arc::new(EmojiFont {
        path: path.to_string(),
        bytes: OnceLock::new(),
        reading: AtomicBool::new(false),
    });
    match held {
        Some(i) => fonts[i] = (now, Arc::clone(&font)),
        None => fonts.push((now, Arc::clone(&font))),
    }
    font
}

/// The font of the emoji of a scene in `dir`: FILE in `dir`, in each
/// directory of `more`, or in `system`, the first found. A scene depends on
/// each path tried, so FILE showing up at one of them marks the scene stale.
pub(crate) fn find(dir: &str, more: &[String], system: &str) -> Option<Arc<EmojiFont>> {
    let dirs = std::iter::once(dir)
        .chain(more.iter().map(String::as_str))
        .chain([system]);
    for d in dirs {
        let path = normpath(&join(d, FILE));
        crate::sources::record(&path);
        if Path::new(&path).is_file() {
            return Some(at(&path));
        }
    }
    None
}
