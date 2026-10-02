//! Glyph atlases for the MTSDF fill.
//!
//! An MTSDF holds a glyph as distances to its outline, 0.5 on the outline and
//! above it inside the glyph. The median of red, green and blue keeps the
//! corners sharp, and alpha holds the true distance, which outlines, glows and
//! shadows read. fdsm generates each glyph's field on the CPU, and the field
//! takes one cell of an RGBA8 atlas.
//!
//! A [`Generator`] fills the atlas of one font at one px/em and range, and
//! starts nothing until the caller first requests characters. Its worker
//! thread then loads the atlas from the disk cache, or generates the
//! [`preload`] set on every core, and after that generates each other
//! requested glyph, one at a time. [`Generator::request`] answers at once with
//! the characters still pending, so the caller can hold new words until their
//! glyphs are ready, and [`Generator::update`] hands over the texels that
//! changed, for the upload.
//!
//! The cache keeps one file per font sha256, px/em and range in
//! `$XDG_CACHE_HOME/ghee/atlas/`, else `~/.cache/ghee/atlas/`.

use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::fs;
use std::io;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::{self, JoinHandle};

use anyhow::{anyhow, bail, Result};
use fdsm::bezier::scanline::FillRule;
use fdsm::bezier::{Point, Segment};
use fdsm::correct_error::{correct_error_mtsdf, ErrorCorrectionConfig};
use fdsm::generate::generate_mtsdf;
use fdsm::render::correct_sign_mtsdf;
use fdsm::shape::{Contour, Shape};
use image::{ImageBuffer, Rgba};
use sha2::{Digest, Sha256};

/// Pixels per em of a cell, unless a text sets its own.
pub const PX_EM: u32 = 32;
/// The distance range in texels, unless a text sets its own. A field spans
/// half of it on each side of the outline.
pub const RANGE: u32 = 6;
/// VC4's GL_MAX_TEXTURE_SIZE. An atlas grows to this on each side and no
/// further.
pub const MAX_SIZE: u32 = 2048;
/// An atlas starts this size on each side, and doubles its shorter side as it
/// fills.
const MIN_SIZE: u32 = 128;
/// Edge coloring takes a turn of more than asin(0.03), 1.7 degrees, between two
/// segments for a corner.
const SIN_CORNER: f64 = 0.03;
/// A fixed edge coloring seed gives a glyph the same field on every run.
const COLOR_SEED: u64 = 69_441_337_420;
/// The cache file's layout. A change to the layout or to the generation takes
/// the next number.
const FORMAT: u32 = 1;
const MAGIC: &[u8; 8] = b"gheemsdf";

/// The characters a generator makes at first use: printable ASCII, Latin-1,
/// the curly quotes, the en and em dash, the ellipsis, the bullet and the euro
/// sign. 198 in all.
pub fn preload() -> impl Iterator<Item = char> {
    ('!'..='~')
        .chain('\u{a1}'..='\u{ff}')
        .chain("\u{2018}\u{2019}\u{201c}\u{201d}\u{2013}\u{2014}\u{2026}\u{2022}\u{20ac}".chars())
}

/// A quadratic Bézier curve: start, control and end point, in em, y up. A line
/// is {start, end, end}.
pub type Curve = [[f64; 2]; 3];

/// The font a generator draws its glyphs from.
pub trait Font: Send + Sync {
    /// The font file. Its sha256 keys the cache.
    fn bytes(&self) -> &[u8];
    /// The glyph the layout draws for `ch`, None when it draws nothing for it.
    fn glyph(&self, ch: char) -> Option<u16>;
    /// The closed contours of glyph `gid`, none for a glyph without ink.
    fn outline(&self, gid: u16) -> Result<Vec<Vec<Curve>>>;
}

/// One glyph's MTSDF, before it takes a cell.
#[derive(Debug)]
pub struct Field {
    pub width: u32,
    pub height: u32,
    /// RGBA8, rows from the bottom of the glyph up.
    pub texels: Vec<u8>,
    /// The ink box in em from the glyph's origin, y up: x0, y0, x1, y1.
    pub ink: [f64; 4],
    /// Texels from the field's left and bottom edges to the ink box. The right
    /// and top edges leave at least as many.
    pub pad: f64,
}

/// The MTSDF of a glyph's closed contours at `px_em` pixels per em, its
/// distances spanning `range` texels. None when the contours hold no curve.
pub fn field(contours: &[Vec<Curve>], px_em: u32, range: u32) -> Result<Option<Field>> {
    let (s, r) = (f64::from(px_em), f64::from(range));
    let pad = r / 2.0 + 1.0;
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in contours.iter().flatten().flatten() {
        if !p[0].is_finite() || !p[1].is_finite() {
            bail!("its outline has a point at infinity or NaN");
        }
        for a in 0..2 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let at = |q: [f64; 2]| Point::new((q[0] - lo[0]) * s + pad, (q[1] - lo[1]) * s + pad);
    // A curve whose control point sits on an end has no direction there, and
    // edge coloring needs one, so it becomes the line it draws.
    let contours: Vec<Contour> = contours
        .iter()
        .map(|c| Contour {
            segments: c
                .iter()
                .filter_map(|q| {
                    if q[1] == q[0] || q[1] == q[2] {
                        (q[0] != q[2]).then(|| Segment::line(at(q[0]), at(q[2])))
                    } else {
                        Some(Segment::quad(at(q[0]), at(q[1]), at(q[2])))
                    }
                })
                .collect(),
        })
        .filter(|c| !c.segments.is_empty())
        .collect();
    if contours.is_empty() {
        return Ok(None);
    }
    let w = ((hi[0] - lo[0]) * s + 2.0 * pad).ceil();
    let h = ((hi[1] - lo[1]) * s + 2.0 * pad).ceil();
    if w > f64::from(MAX_SIZE) || h > f64::from(MAX_SIZE) {
        bail!("its {w}x{h} cell is larger than the {MAX_SIZE}x{MAX_SIZE} atlas");
    }
    let (w, h) = (w as u32, h as u32);
    let colored = Shape::edge_coloring_simple(Shape { contours }, SIN_CORNER, COLOR_SEED);
    let prepared = colored.prepare();
    let mut img: ImageBuffer<Rgba<f32>, Vec<f32>> = ImageBuffer::new(w, h);
    generate_mtsdf(&prepared, r, &mut img);
    // Error correction reads the corners, so it runs before the sign fix.
    correct_error_mtsdf(
        &mut img,
        &colored,
        &prepared,
        r,
        &ErrorCorrectionConfig::default(),
    );
    correct_sign_mtsdf(&mut img, &prepared, FillRule::Nonzero);
    let texels = img
        .as_raw()
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect();
    Ok(Some(Field {
        width: w,
        height: h,
        texels,
        ink: [lo[0], lo[1], hi[0], hi[1]],
        pad,
    }))
}

/// Where a glyph's field sits in the atlas.
///
/// A point p in em from the glyph's origin samples the atlas at texel
/// coordinate `rect.xy + pad + (p - ink.xy) * px_em`, where texel i spans
/// i to i + 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    /// x, y, width and height, in texels.
    pub rect: [u32; 4],
    /// The ink box in em from the glyph's origin, y up: x0, y0, x1, y1.
    pub ink: [f64; 4],
    /// Texels from the cell's left and bottom edges to the ink box. The right
    /// and top edges leave at least as many.
    pub pad: f64,
}

impl Cell {
    /// The cell's box in the texture coordinates of a `width` x `height`
    /// atlas: u0, v0, u1, v1.
    pub fn uv(&self, width: u32, height: u32) -> [f64; 4] {
        let [x, y, w, h] = self.rect.map(f64::from);
        let (aw, ah) = (f64::from(width), f64::from(height));
        [x / aw, y / ah, (x + w) / aw, (y + h) / ah]
    }
}

/// A row of cells as tall as the first cell put on it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Shelf {
    y: u32,
    height: u32,
    /// Texels taken from the left edge.
    used: u32,
}

/// Glyph cells packed on shelves into one RGBA8 texture.
#[derive(Debug)]
pub struct Atlas {
    pub width: u32,
    pub height: u32,
    /// RGBA8, row-major, row 0 at texture coordinate v = 0.
    pub texels: Vec<u8>,
    /// Each glyph's cell, by glyph id.
    pub cells: HashMap<u16, Cell>,
    /// Bottom to top, with no gap between two.
    shelves: Vec<Shelf>,
    /// Rows changed since the last update, as runs from a first row to an end.
    dirty: Vec<[u32; 2]>,
    /// The size changed since the last update.
    resized: bool,
}

impl Default for Atlas {
    /// An empty MIN_SIZE x MIN_SIZE atlas, whose first update holds all of it.
    fn default() -> Atlas {
        Atlas {
            width: MIN_SIZE,
            height: MIN_SIZE,
            texels: vec![0; (MIN_SIZE * MIN_SIZE * 4) as usize],
            cells: HashMap::new(),
            shelves: Vec::new(),
            dirty: Vec::new(),
            resized: true,
        }
    }
}

/// Texels for the GPU copy of an atlas.
#[derive(Debug)]
pub struct Update {
    pub width: u32,
    pub height: u32,
    /// The atlas changed size. `rows` holds all of it, for a new texture, and
    /// every cell's uv moved.
    pub resized: bool,
    /// The rows that changed, each run the full width of the atlas.
    pub rows: Vec<Rows>,
}

/// A run of full-width rows, from row `y` up.
#[derive(Debug)]
pub struct Rows {
    pub y: u32,
    pub texels: Vec<u8>,
}

impl Atlas {
    /// Put `field` in a free cell for glyph `gid`, growing the atlas when no
    /// shelf has room.
    pub fn insert(&mut self, gid: u16, field: &Field) -> Result<Cell> {
        let (w, h) = (field.width, field.height);
        if w > MAX_SIZE || h > MAX_SIZE {
            bail!("its {w}x{h} cell is larger than the {MAX_SIZE}x{MAX_SIZE} atlas");
        }
        if field.texels.len() != w as usize * h as usize * 4 {
            bail!(
                "a {w}x{h} field holds {} bytes in place of {}",
                field.texels.len(),
                w as usize * h as usize * 4
            );
        }
        let (x, y) = loop {
            if let Some(at) = self.place(w, h) {
                break at;
            }
            if !self.grow() {
                bail!("the {MAX_SIZE}x{MAX_SIZE} atlas is full");
            }
        };
        let (aw, fw) = (self.width as usize, w as usize * 4);
        for row in 0..h as usize {
            let at = ((y as usize + row) * aw + x as usize) * 4;
            self.texels[at..at + fw].copy_from_slice(&field.texels[row * fw..(row + 1) * fw]);
        }
        self.mark(y, y + h);
        let cell = Cell {
            rect: [x, y, w, h],
            ink: field.ink,
            pad: field.pad,
        };
        self.cells.insert(gid, cell);
        Ok(cell)
    }

    /// The texels that changed since the last call, None when nothing did.
    pub fn update(&mut self) -> Option<Update> {
        let resized = std::mem::take(&mut self.resized);
        let runs = std::mem::take(&mut self.dirty);
        let runs = if resized {
            vec![[0, self.height]]
        } else {
            runs
        };
        if runs.is_empty() {
            return None;
        }
        let row = self.width as usize * 4;
        Some(Update {
            width: self.width,
            height: self.height,
            resized,
            rows: runs
                .iter()
                .map(|&[a, b]| Rows {
                    y: a,
                    texels: self.texels[a as usize * row..b as usize * row].to_vec(),
                })
                .collect(),
        })
    }

    /// The first row above every shelf.
    fn top(&self) -> u32 {
        self.shelves.last().map_or(0, |s| s.y + s.height)
    }

    /// A free corner for a w x h cell: on the shelf with the least height to
    /// spare, else on a new shelf.
    fn place(&mut self, w: u32, h: u32) -> Option<(u32, u32)> {
        let width = self.width;
        if let Some(s) = self
            .shelves
            .iter_mut()
            .filter(|s| s.height >= h && s.used + w <= width)
            .min_by_key(|s| s.height - h)
        {
            let x = s.used;
            s.used += w;
            return Some((x, s.y));
        }
        let top = self.top();
        if w > width || top + h > self.height {
            return None;
        }
        self.shelves.push(Shelf {
            y: top,
            height: h,
            used: w,
        });
        Some((0, top))
    }

    /// Double the shorter side, the width of a square. Cells keep their
    /// texels. False with both sides at MAX_SIZE.
    fn grow(&mut self) -> bool {
        let (w, h) = (self.width as usize, self.height as usize);
        if self.width < MAX_SIZE && (self.width <= self.height || self.height == MAX_SIZE) {
            let mut texels = vec![0; w * 2 * h * 4];
            for (row, src) in self.texels.chunks_exact(w * 4).enumerate() {
                texels[row * w * 8..row * w * 8 + w * 4].copy_from_slice(src);
            }
            self.texels = texels;
            self.width *= 2;
        } else if self.height < MAX_SIZE {
            self.texels.resize(w * h * 2 * 4, 0);
            self.height *= 2;
        } else {
            return false;
        }
        self.resized = true;
        self.dirty.clear();
        true
    }

    /// Rows `from` to `to` changed.
    fn mark(&mut self, from: u32, to: u32) {
        if self.resized {
            return;
        }
        self.dirty.push([from, to]);
        self.dirty.sort_unstable();
        let mut runs: Vec<[u32; 2]> = Vec::with_capacity(self.dirty.len());
        for r in self.dirty.drain(..) {
            match runs.last_mut() {
                Some(last) if r[0] <= last[1] => last[1] = last[1].max(r[1]),
                _ => runs.push(r),
            }
        }
        self.dirty = runs;
    }
}

/// How a generator sizes its cells, and where it caches the atlas.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Pixels per em of a cell.
    pub px_em: u32,
    /// The distance range in texels. A field spans half of it on each side of
    /// the outline.
    pub range: u32,
    /// The cache directory. None takes `$XDG_CACHE_HOME/ghee/atlas`, else
    /// `~/.cache/ghee/atlas`.
    pub cache: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            px_em: PX_EM,
            range: RANGE,
            cache: None,
        }
    }
}

/// One font's atlas at one px/em and range, filled by a worker thread.
pub struct Generator {
    font: Arc<dyn Font>,
    config: Config,
    /// Set by the first request.
    started: OnceLock<Started>,
}

struct Started {
    shared: Arc<Shared>,
    worker: JoinHandle<()>,
}

/// What the caller and the worker share.
struct Shared {
    state: Mutex<State>,
    /// The worker sleeps on it until a request or a stop.
    wake: Condvar,
    /// Set under the state's lock, so a sleeping worker never misses it.
    stop: AtomicBool,
}

struct State {
    atlas: Atlas,
    /// Every character requested so far, true once its glyph is in the atlas,
    /// failed, or draws nothing.
    chars: HashMap<char, bool>,
    /// Requested characters the worker has yet to take.
    queue: Vec<char>,
    /// Errors the caller has yet to take.
    errors: Vec<String>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Generator {
    /// A generator for `font`. It starts nothing until the first request.
    pub fn new(font: Arc<dyn Font>, config: Config) -> Result<Generator> {
        if config.px_em == 0 || config.range == 0 {
            bail!(
                "an atlas needs a px/em and a range of 1 or more, got {} and {}",
                config.px_em,
                config.range
            );
        }
        Ok(Generator {
            font,
            config,
            started: OnceLock::new(),
        })
    }

    /// Ask for the glyphs of `chars`, and get back at once the characters
    /// still pending, none when every glyph is ready. The first request starts
    /// the worker.
    pub fn request(&self, chars: impl IntoIterator<Item = char>) -> Vec<char> {
        let started = self.started.get_or_init(|| self.start());
        let mut st = lock(&started.shared.state);
        let mut pending = Vec::new();
        let mut queued = false;
        for ch in chars {
            match st.chars.get(&ch) {
                Some(true) => {}
                Some(false) => pending.push(ch),
                None => {
                    st.chars.insert(ch, false);
                    st.queue.push(ch);
                    pending.push(ch);
                    queued = true;
                }
            }
        }
        drop(st);
        if queued {
            started.shared.wake.notify_one();
        }
        pending.sort_unstable();
        pending.dedup();
        pending
    }

    /// The errors of the glyphs that failed since the last call, each naming
    /// its character. A failed glyph stays out of the atlas and counts as
    /// ready.
    pub fn errors(&self) -> Vec<String> {
        self.started
            .get()
            .map(|s| std::mem::take(&mut lock(&s.shared.state).errors))
            .unwrap_or_default()
    }

    /// `f` on the atlas as it stands, None before the first request. The
    /// worker waits meanwhile.
    pub fn with<R>(&self, f: impl FnOnce(&Atlas) -> R) -> Option<R> {
        let s = self.started.get()?;
        Some(f(&lock(&s.shared.state).atlas))
    }

    /// The texels that changed since the last call, None when nothing did.
    pub fn update(&self) -> Option<Update> {
        lock(&self.started.get()?.shared.state).atlas.update()
    }

    fn start(&self) -> Started {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                atlas: Atlas::default(),
                chars: HashMap::new(),
                queue: Vec::new(),
                errors: Vec::new(),
            }),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
        });
        let (sh, font, config) = (shared.clone(), self.font.clone(), self.config.clone());
        let worker = thread::Builder::new()
            .name("ghee-atlas".into())
            .spawn(move || work(&sh, &*font, &config))
            .expect("spawn the atlas worker");
        Started { shared, worker }
    }
}

impl Drop for Generator {
    /// Stop the worker. It finishes the glyph in hand on each core first.
    fn drop(&mut self) {
        let Some(s) = self.started.take() else {
            return;
        };
        {
            let _st = lock(&s.shared.state);
            s.shared.stop.store(true, Ordering::Relaxed);
        }
        s.shared.wake.notify_all();
        let _ = s.worker.join();
    }
}

/// The worker: the cache or the preload first, then each request as it comes.
fn work(shared: &Shared, font: &dyn Font, config: &Config) {
    let key = Key {
        font: Sha256::digest(font.bytes()).into(),
        px_em: config.px_em,
        range: config.range,
    };
    let mut file = match cache_dir(config) {
        Ok(dir) => Some(dir.join(key.file_name())),
        Err(e) => {
            warn_memory(e);
            None
        }
    };
    let cached = file
        .as_deref()
        .and_then(|f| fs::read(f).ok())
        .and_then(|b| decode(&key, &b));
    let mut preloading = cached.is_none();
    if let Some(atlas) = cached {
        lock(&shared.state).atlas = atlas;
    }
    let mut none = Cellless::default();
    loop {
        let batch = {
            let mut st = lock(&shared.state);
            while st.queue.is_empty() && !shared.stop.load(Ordering::Relaxed) {
                st = shared.wake.wait(st).unwrap_or_else(|e| e.into_inner());
            }
            if shared.stop.load(Ordering::Relaxed) {
                return;
            }
            std::mem::take(&mut st.queue)
        };
        let asked: Vec<(char, Option<u16>)> = batch.iter().map(|&c| (c, font.glyph(c))).collect();
        let extra: Vec<Option<u16>> = if preloading {
            preload().map(|c| font.glyph(c)).collect()
        } else {
            Vec::new()
        };
        let mut jobs: Vec<u16> = Vec::new();
        {
            let mut st = lock(&shared.state);
            let mut seen = HashSet::new();
            for g in asked.iter().map(|a| a.1).chain(extra).flatten() {
                if !st.atlas.cells.contains_key(&g) && !none.holds(g) && seen.insert(g) {
                    jobs.push(g);
                }
            }
            for &(c, g) in &asked {
                if g.is_none_or(|g| !seen.contains(&g)) {
                    st.chars.insert(c, true);
                    if let Some(g) = g {
                        none.tell(&mut st, g, c);
                    }
                }
            }
        }
        let mut added = false;
        if std::mem::take(&mut preloading) {
            let mut made = make_all(font, &jobs, config, &shared.stop);
            if shared.stop.load(Ordering::Relaxed) {
                return;
            }
            // Tallest first packs the shelves tightest.
            made.sort_by_key(|(gid, f)| {
                let h = f
                    .as_ref()
                    .ok()
                    .and_then(|f| f.as_ref())
                    .map_or(0, |f| f.height);
                (std::cmp::Reverse(h), *gid)
            });
            let mut st = lock(&shared.state);
            for (gid, f) in made {
                added |= settle(&mut st, &asked, gid, f, &mut none);
            }
        } else {
            for &gid in &jobs {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                let f = make(font, gid, config);
                added |= settle(&mut lock(&shared.state), &asked, gid, f, &mut none);
            }
        }
        if let (true, Some(path)) = (added, &file) {
            let bytes = encode(&key, &lock(&shared.state).atlas);
            if let Err(e) = save(path, &bytes) {
                let dir = path.parent().unwrap_or(path).display();
                warn_memory(format!("cannot write the atlas cache in {dir}: {e}"));
                file = None;
            }
        }
    }
}

/// The glyphs a worker made that hold no cell.
#[derive(Default)]
struct Cellless {
    /// Glyphs without ink.
    blank: HashSet<u16>,
    /// Each failed glyph's error, and whether a request has heard it.
    failed: HashMap<u16, (String, bool)>,
}

impl Cellless {
    fn holds(&self, gid: u16) -> bool {
        self.blank.contains(&gid) || self.failed.contains_key(&gid)
    }

    /// Hand glyph `gid`'s error to the caller, named after `ch`, the first
    /// time a request asks for it. A preloaded glyph that failed stays quiet
    /// until then.
    fn tell(&mut self, st: &mut State, gid: u16, ch: char) {
        if let Some((e, told)) = self.failed.get_mut(&gid) {
            if !*told {
                *told = true;
                st.errors
                    .push(format!("glyph {ch:?} (U+{:04X}): {e}", u32::from(ch)));
            }
        }
    }
}

/// Put a made glyph in the atlas, or its error in `none`, and mark the
/// characters that asked for it ready. True when it took a cell.
fn settle(
    st: &mut State,
    asked: &[(char, Option<u16>)],
    gid: u16,
    made: Result<Option<Field>>,
    none: &mut Cellless,
) -> bool {
    let took = match made.and_then(|f| f.map(|f| st.atlas.insert(gid, &f)).transpose()) {
        Ok(cell) => cell.is_some(),
        Err(e) => {
            none.failed.insert(gid, (format!("{e:#}"), false));
            false
        }
    };
    if !took && !none.failed.contains_key(&gid) {
        none.blank.insert(gid);
    }
    for &(c, g) in asked {
        if g == Some(gid) {
            st.chars.insert(c, true);
            none.tell(st, gid, c);
        }
    }
    took
}

/// One glyph's field. A panic inside fdsm fails this glyph alone.
fn make(font: &dyn Font, gid: u16, config: &Config) -> Result<Option<Field>> {
    catch_unwind(AssertUnwindSafe(|| {
        field(&font.outline(gid)?, config.px_em, config.range)
    }))
    .unwrap_or_else(|_| Err(anyhow!("the MTSDF generator panicked on its outline")))
}

/// The field of every glyph in `jobs`, on every core. Glyphs left after a stop
/// stay out.
fn make_all(
    font: &dyn Font,
    jobs: &[u16],
    config: &Config,
    stop: &AtomicBool,
) -> Vec<(u16, Result<Option<Field>>)> {
    let cores = thread::available_parallelism().map_or(1, |n| n.get());
    let next = AtomicUsize::new(0);
    thread::scope(|s| {
        let hands: Vec<_> = (0..cores.min(jobs.len()))
            .map(|_| {
                s.spawn(|| {
                    let mut got = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= jobs.len() || stop.load(Ordering::Relaxed) {
                            return got;
                        }
                        got.push((jobs[i], make(font, jobs[i], config)));
                    }
                })
            })
            .collect();
        hands
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_default())
            .collect()
    })
}

/// Say once per process that atlases stay in memory, and why.
fn warn_memory(why: impl Display) {
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        crate::report::warn(format!("{why}, so glyph atlases stay in memory"));
    }
}

/// Config::cache, else $XDG_CACHE_HOME/ghee/atlas, else ~/.cache/ghee/atlas.
fn cache_dir(config: &Config) -> std::result::Result<PathBuf, &'static str> {
    if let Some(d) = &config.cache {
        return Ok(d.clone());
    }
    let var = |k| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let base = var("XDG_CACHE_HOME")
        .or_else(|| var("HOME").map(|h| h.join(".cache")))
        .ok_or("neither $XDG_CACHE_HOME nor $HOME is set")?;
    Ok(base.join("ghee").join("atlas"))
}

/// What a cache file holds an atlas for.
#[derive(Clone, Debug, PartialEq)]
struct Key {
    /// The font file's sha256.
    font: [u8; 32],
    px_em: u32,
    range: u32,
}

impl Key {
    fn file_name(&self) -> String {
        let hex: String = self.font.iter().map(|b| format!("{b:02x}")).collect();
        format!("{hex}-{}-{}-{FORMAT}.mtsdf", self.px_em, self.range)
    }
}

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

/// The cache file, little-endian: MAGIC, FORMAT, px/em, range, the font's
/// sha256, the atlas width and height, the rows under the top shelf, the
/// shelves (a count, then y, height and used width each), the cells (a count,
/// then gid, rect, ink and pad each), the texels of those rows, and the sha256
/// of everything before it.
fn encode(key: &Key, a: &Atlas) -> Vec<u8> {
    let rows = a.top();
    let used = rows as usize * a.width as usize * 4;
    let mut b = Vec::with_capacity(used + 64 * a.cells.len() + 1024);
    b.extend_from_slice(MAGIC);
    for v in [FORMAT, key.px_em, key.range] {
        put_u32(&mut b, v);
    }
    b.extend_from_slice(&key.font);
    for v in [a.width, a.height, rows, a.shelves.len() as u32] {
        put_u32(&mut b, v);
    }
    for s in &a.shelves {
        for v in [s.y, s.height, s.used] {
            put_u32(&mut b, v);
        }
    }
    let mut cells: Vec<_> = a.cells.iter().collect();
    cells.sort_unstable_by_key(|(gid, _)| **gid);
    put_u32(&mut b, cells.len() as u32);
    for (gid, c) in cells {
        put_u32(&mut b, u32::from(*gid));
        for v in c.rect {
            put_u32(&mut b, v);
        }
        for v in c.ink.iter().chain([&c.pad]) {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    b.extend_from_slice(&a.texels[..used]);
    let digest = Sha256::digest(&b);
    b.extend_from_slice(&digest);
    b
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.0.len() < n {
            return None;
        }
        let (a, rest) = self.0.split_at(n);
        self.0 = rest;
        Some(a)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn f64(&mut self) -> Option<f64> {
        Some(f64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
}

/// The atlas in a cache file, None for a file that is torn, corrupt, or holds
/// another key.
fn decode(key: &Key, bytes: &[u8]) -> Option<Atlas> {
    let (body, digest) = bytes.split_at(bytes.len().checked_sub(32)?);
    if Sha256::digest(body).as_slice() != digest {
        return None;
    }
    let mut r = Reader(body);
    let ours = r.take(8)? == MAGIC
        && r.u32()? == FORMAT
        && r.u32()? == key.px_em
        && r.u32()? == key.range
        && r.take(32)? == key.font;
    let (width, height, rows) = (r.u32()?, r.u32()?, r.u32()?);
    let side = |v: u32| v.is_power_of_two() && (MIN_SIZE..=MAX_SIZE).contains(&v);
    if !ours || !side(width) || !side(height) || rows > height {
        return None;
    }
    let mut shelves = Vec::new();
    let mut top = 0;
    for _ in 0..r.u32()? {
        let s = Shelf {
            y: r.u32()?,
            height: r.u32()?,
            used: r.u32()?,
        };
        if s.y != top || s.used > width {
            return None;
        }
        top = s.y.checked_add(s.height)?;
        shelves.push(s);
    }
    if top != rows {
        return None;
    }
    let mut cells = HashMap::new();
    for _ in 0..r.u32()? {
        let gid = u16::try_from(r.u32()?).ok()?;
        let rect = [r.u32()?, r.u32()?, r.u32()?, r.u32()?];
        let ink = [r.f64()?, r.f64()?, r.f64()?, r.f64()?];
        let pad = r.f64()?;
        let inside =
            rect[0].checked_add(rect[2])? <= width && rect[1].checked_add(rect[3])? <= rows;
        let finite = ink.iter().chain([&pad]).all(|v| v.is_finite());
        if !inside || !finite || cells.insert(gid, Cell { rect, ink, pad }).is_some() {
            return None;
        }
    }
    let used = r.take(rows as usize * width as usize * 4)?;
    if !r.0.is_empty() {
        return None;
    }
    let mut texels = vec![0; width as usize * height as usize * 4];
    texels[..used.len()].copy_from_slice(used);
    Some(Atlas {
        width,
        height,
        texels,
        cells,
        shelves,
        dirty: Vec::new(),
        resized: true,
    })
}

/// Write `bytes` to `path` through a temporary file and a rename, so a reader
/// finds the old file or the new one, whole.
fn save(path: &Path, bytes: &[u8]) -> io::Result<()> {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(dir)?;
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let tmp = dir.join(format!(
        ".{name}.{}.{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let written = fs::write(&tmp, bytes).and_then(|()| fs::rename(&tmp, path));
    if written.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const SANS: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tools/fixtures/fonts/DejaVuSans-ASCII.ttf"
    );

    /// A fresh path under the temp directory, left uncreated.
    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ghee-atlas-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        let _ = fs::remove_file(&d);
        d
    }

    fn cached(dir: &Path) -> Config {
        Config {
            cache: Some(dir.to_path_buf()),
            ..Config::default()
        }
    }

    fn key(font: &dyn Font, config: &Config) -> Key {
        Key {
            font: Sha256::digest(font.bytes()).into(),
            px_em: config.px_em,
            range: config.range,
        }
    }

    /// A TrueType glyph's contours as curves in em.
    struct Contours {
        scale: f64,
        start: [f64; 2],
        at: [f64; 2],
        out: Vec<Vec<Curve>>,
    }

    impl Contours {
        fn push(&mut self, c: Curve) {
            self.out.last_mut().unwrap().push(c);
            self.at = c[2];
        }
    }

    impl ttf_parser::OutlineBuilder for Contours {
        fn move_to(&mut self, x: f32, y: f32) {
            self.start = [f64::from(x) * self.scale, f64::from(y) * self.scale];
            self.at = self.start;
            self.out.push(Vec::new());
        }

        fn line_to(&mut self, x: f32, y: f32) {
            let p = [f64::from(x) * self.scale, f64::from(y) * self.scale];
            self.push([self.at, p, p]);
        }

        fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
            let c = [f64::from(x1) * self.scale, f64::from(y1) * self.scale];
            let p = [f64::from(x) * self.scale, f64::from(y) * self.scale];
            self.push([self.at, c, p]);
        }

        fn curve_to(&mut self, _: f32, _: f32, _: f32, _: f32, _: f32, _: f32) {
            panic!("the fixture font draws quadratics");
        }

        fn close(&mut self) {
            if self.at != self.start {
                let s = self.start;
                self.push([self.at, s, s]);
            }
        }
    }

    /// DejaVu Sans cut to printable ASCII. Counts the outlines asked of it.
    struct Ttf {
        data: Vec<u8>,
        outlines: AtomicUsize,
    }

    impl Ttf {
        fn new() -> Ttf {
            Ttf {
                data: fs::read(SANS).unwrap(),
                outlines: AtomicUsize::new(0),
            }
        }

        fn face(&self) -> ttf_parser::Face<'_> {
            ttf_parser::Face::parse(&self.data, 0).unwrap()
        }
    }

    impl Font for Ttf {
        fn bytes(&self) -> &[u8] {
            &self.data
        }

        fn glyph(&self, ch: char) -> Option<u16> {
            self.face().glyph_index(ch).map(|g| g.0)
        }

        fn outline(&self, gid: u16) -> Result<Vec<Vec<Curve>>> {
            self.outlines.fetch_add(1, Ordering::Relaxed);
            let face = self.face();
            let mut b = Contours {
                scale: 1.0 / f64::from(face.units_per_em()),
                start: [0.0; 2],
                at: [0.0; 2],
                out: Vec::new(),
            };
            face.outline_glyph(ttf_parser::GlyphId(gid), &mut b);
            Ok(b.out)
        }
    }

    /// A character draws its code point as a 0.5 x 0.7 em box. W is 100 em wide,
    /// a space has no ink. `tag` stands in for the font file.
    struct Boxes {
        tag: [u8; 1],
        outlines: AtomicUsize,
    }

    fn boxes(tag: u8) -> Arc<Boxes> {
        Arc::new(Boxes {
            tag: [tag],
            outlines: AtomicUsize::new(0),
        })
    }

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Vec<Curve>> {
        let p = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        vec![(0..4)
            .map(|i| [p[i], p[(i + 1) % 4], p[(i + 1) % 4]])
            .collect()]
    }

    impl Font for Boxes {
        fn bytes(&self) -> &[u8] {
            &self.tag
        }

        fn glyph(&self, ch: char) -> Option<u16> {
            u16::try_from(u32::from(ch)).ok()
        }

        fn outline(&self, gid: u16) -> Result<Vec<Vec<Curve>>> {
            self.outlines.fetch_add(1, Ordering::Relaxed);
            Ok(match char::from_u32(u32::from(gid)) {
                Some(' ') => Vec::new(),
                Some('W') => rect(0.0, 0.0, 100.0, 0.7),
                _ => rect(0.05, 0.0, 0.55, 0.7),
            })
        }
    }

    /// Request `chars` until every glyph is ready.
    fn wait(g: &Generator, chars: &str) {
        let t0 = Instant::now();
        while !g.request(chars.chars()).is_empty() {
            assert!(
                t0.elapsed() < Duration::from_secs(60),
                "{chars:?} still pending"
            );
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// A w x h field whose texels name `gid`.
    fn pattern(gid: u16, w: u32, h: u32) -> Field {
        Field {
            width: w,
            height: h,
            texels: (0..w * h * 4)
                .map(|i| (i.wrapping_mul(31) ^ u32::from(gid).wrapping_mul(97)) as u8)
                .collect(),
            ink: [0.0, 0.0, 1.0, 1.0],
            pad: 4.0,
        }
    }

    /// The texels of `cell` in `a`, rows bottom to top.
    fn texels_of(a: &Atlas, cell: &Cell) -> Vec<u8> {
        let [x, y, w, h] = cell.rect.map(|v| v as usize);
        (y..y + h)
            .flat_map(|row| {
                let at = (row * a.width as usize + x) * 4;
                a.texels[at..at + w * 4].to_vec()
            })
            .collect()
    }

    #[test]
    fn the_preload_is_198_distinct_characters() {
        let all: HashSet<char> = preload().collect();
        assert_eq!((preload().count(), all.len()), (198, 198));
    }

    // Fields

    #[test]
    fn a_field_pads_its_ink_box_and_reads_half_on_the_outline() {
        // One side starts with its control point on its start point.
        let mut sq = rect(0.0, 0.0, 0.5, 0.5);
        sq[0][1] = [[0.5, 0.0], [0.5, 0.0], [0.5, 0.5]];
        let f = field(&sq, 32, 6).unwrap().unwrap();
        assert_eq!(f.pad, 4.0);
        assert_eq!(f.ink, [0.0, 0.0, 0.5, 0.5]);
        assert_eq!((f.width, f.height), (24, 24));
        let at = |x: usize, y: usize| &f.texels[(y * 24 + x) * 4..(y * 24 + x) * 4 + 4];
        let median = |t: &[u8]| t[0].min(t[1]).max(t[0].max(t[1]).min(t[2]));
        // Texel 4's centre lies 0.5 texels inside the left edge: 0.5 + 0.5 / 6.
        for y in 8..16 {
            assert_eq!(at(4, y)[3], 149, "alpha at (4, {y})");
            assert!(median(at(4, y)).abs_diff(149) <= 1, "median at (4, {y})");
            assert!(median(at(3, y)).abs_diff(106) <= 1, "median at (3, {y})");
        }
        assert_eq!(at(12, 12)[3], 255);
        assert!(median(at(12, 12)) == 255 && median(at(0, 0)) == 0 && at(0, 0)[3] == 0);
    }

    #[test]
    fn an_outline_without_curves_has_no_field() {
        assert!(field(&[], 32, 6).unwrap().is_none());
        let p = [0.25, 0.25];
        assert!(field(&[vec![[p, p, p]], vec![]], 32, 6).unwrap().is_none());
    }

    #[test]
    fn a_field_past_the_atlas_or_off_the_plane_fails() {
        let e = field(&rect(0.0, 0.0, 100.0, 0.7), 32, 6).unwrap_err();
        assert_eq!(
            e.to_string(),
            "its 3208x31 cell is larger than the 2048x2048 atlas"
        );
        let e = field(&rect(0.0, 0.0, f64::NAN, 1.0), 32, 6).unwrap_err();
        assert_eq!(e.to_string(), "its outline has a point at infinity or NaN");
    }

    // Packing

    #[test]
    fn cells_never_overlap_and_keep_their_texels() {
        let mut a = Atlas::default();
        let mut fields = Vec::new();
        let mut seed = 1u32;
        for gid in 0..400u16 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let f = pattern(gid, 8 + (seed >> 8) % 40, 8 + (seed >> 20) % 48);
            a.insert(gid, &f).unwrap();
            fields.push(f);
        }
        assert!(a.width.is_power_of_two() && a.height.is_power_of_two());
        assert!(a.width <= MAX_SIZE && a.height <= a.width);
        let cells: Vec<Cell> = (0..400).map(|g| a.cells[&g]).collect();
        for (i, c) in cells.iter().enumerate() {
            let [x, y, w, h] = c.rect;
            assert!(x + w <= a.width && y + h <= a.height, "{i} {:?}", c.rect);
            for d in &cells[..i] {
                let [dx, dy, dw, dh] = d.rect;
                let apart = x + w <= dx || dx + dw <= x || y + h <= dy || dy + dh <= y;
                assert!(apart, "{:?} overlaps {:?}", c.rect, d.rect);
            }
            assert!(texels_of(&a, c) == fields[i].texels, "cell {i}");
            assert_eq!((c.ink, c.pad), (fields[i].ink, fields[i].pad));
        }
    }

    #[test]
    fn the_atlas_doubles_its_shorter_side_and_cells_stay_put() {
        let mut a = Atlas::default();
        assert_eq!((a.width, a.height), (MIN_SIZE, MIN_SIZE));
        let first = a.insert(0, &pattern(0, 100, 100)).unwrap();
        assert_eq!(first.rect, [0, 0, 100, 100]);
        a.update();
        a.insert(1, &pattern(1, 100, 100)).unwrap();
        assert_eq!((a.width, a.height), (256, 128));
        let u = a.update().unwrap();
        assert!(u.resized && u.rows.len() == 1 && u.rows[0].y == 0);
        assert_eq!(u.rows[0].texels, a.texels);
        a.insert(2, &pattern(2, 100, 100)).unwrap();
        assert_eq!((a.width, a.height), (256, 256));
        assert_eq!(a.cells[&0], first);
        assert!(texels_of(&a, &first) == pattern(0, 100, 100).texels);
        assert_eq!(first.uv(256, 256), [0.0, 0.0, 100.0 / 256.0, 100.0 / 256.0]);
    }

    #[test]
    fn the_atlas_stops_at_2048_and_a_cell_too_large_fails_alone() {
        let mut a = Atlas::default();
        for gid in 0..16 {
            a.insert(gid, &pattern(gid, 512, 512)).unwrap();
        }
        assert_eq!((a.width, a.height), (2048, 2048));
        let e = a.insert(16, &pattern(16, 512, 512)).unwrap_err();
        assert_eq!(e.to_string(), "the 2048x2048 atlas is full");
        let e = a.insert(17, &pattern(17, 2049, 1)).unwrap_err();
        assert_eq!(
            e.to_string(),
            "its 2049x1 cell is larger than the 2048x2048 atlas"
        );
        assert_eq!((a.cells.len(), a.width, a.height), (16, 2048, 2048));
    }

    #[test]
    fn an_update_carries_the_changed_rows_once() {
        let mut a = Atlas::default();
        let u = a.update().unwrap();
        assert!(u.resized && u.rows[0].texels.len() == 128 * 128 * 4);
        assert!(a.update().is_none());
        a.insert(5, &pattern(5, 20, 30)).unwrap();
        let low = a.insert(6, &pattern(6, 20, 10)).unwrap();
        assert_eq!(low.rect, [20, 0, 20, 10]);
        let u = a.update().unwrap();
        assert!(!u.resized && u.rows.len() == 1 && u.rows[0].y == 0);
        assert_eq!(u.rows[0].texels, a.texels[..30 * 128 * 4]);
        let tall = a.insert(7, &pattern(7, 20, 40)).unwrap();
        assert_eq!(tall.rect, [0, 30, 20, 40]);
        let u = a.update().unwrap();
        assert!(!u.resized && u.rows.len() == 1 && u.rows[0].y == 30);
        assert_eq!(u.rows[0].texels, a.texels[30 * 128 * 4..70 * 128 * 4]);
        assert!(a.update().is_none());
    }

    // The cache

    fn filled(n: u16) -> Atlas {
        let mut a = Atlas::default();
        for gid in 0..n {
            let f = pattern(gid, 10 + u32::from(gid) % 13, 12 + u32::from(gid) % 9);
            a.insert(gid * 3, &f).unwrap();
        }
        a
    }

    const KEY: Key = Key {
        font: [7; 32],
        px_em: 32,
        range: 6,
    };

    #[test]
    fn the_cache_round_trips_and_packing_carries_on() {
        let mut a = filled(120);
        let bytes = encode(&KEY, &a);
        let mut b = decode(&KEY, &bytes).unwrap();
        assert_eq!((b.width, b.height), (a.width, a.height));
        assert!(b.texels == a.texels);
        assert_eq!((&b.cells, &b.shelves), (&a.cells, &a.shelves));
        assert!(b.update().unwrap().resized);
        let next = pattern(999, 17, 15);
        assert_eq!(a.insert(999, &next).unwrap(), b.insert(999, &next).unwrap());
        assert_eq!(
            KEY.file_name(),
            format!("{}-32-6-{FORMAT}.mtsdf", "07".repeat(32))
        );
    }

    #[test]
    fn a_cache_file_for_another_key_or_torn_or_flipped_reads_as_none() {
        let bytes = encode(&KEY, &filled(40));
        for other in [
            Key {
                font: [8; 32],
                ..KEY
            },
            Key { px_em: 48, ..KEY },
            Key { range: 4, ..KEY },
        ] {
            assert!(decode(&other, &bytes).is_none());
        }
        assert!(decode(&KEY, &bytes[..bytes.len() - 1]).is_none());
        assert!(decode(&KEY, &bytes[..100]).is_none());
        assert!(decode(&KEY, &[]).is_none());
        let mut flipped = bytes.clone();
        flipped[300] ^= 1;
        assert!(decode(&KEY, &flipped).is_none());
    }

    #[test]
    fn a_cache_write_is_whole_to_every_reader() {
        let dir = tmp("atomic");
        let path = dir.join(KEY.file_name());
        let small = encode(&KEY, &filled(10));
        let mut big = Atlas::default();
        big.insert(0, &pattern(0, 300, 300)).unwrap();
        let big = encode(&KEY, &big);
        assert!(big.len() > 500_000);
        save(&path, &small).unwrap();
        let done = AtomicUsize::new(0);
        let reads = thread::scope(|s| {
            for k in 0..4 {
                let (path, small, big, done) = (&path, &small, &big, &done);
                s.spawn(move || {
                    for i in 0..25 {
                        save(path, if (i + k) % 2 == 0 { small } else { big }).unwrap();
                    }
                    done.fetch_add(1, Ordering::Relaxed);
                });
            }
            let mut reads = 0;
            while done.load(Ordering::Relaxed) < 4 {
                let b = fs::read(&path).unwrap();
                assert!(
                    decode(&KEY, &b).is_some(),
                    "read {reads}: {} bytes",
                    b.len()
                );
                reads += 1;
            }
            reads
        });
        assert!(reads > 10, "{reads} reads");
        let left: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(left, [std::ffi::OsString::from(KEY.file_name())]);
        fs::remove_dir_all(&dir).unwrap();
    }

    // The generator

    #[test]
    fn nothing_starts_before_the_first_request() {
        let dir = tmp("lazy");
        let font = boxes(1);
        let g = Generator::new(font.clone(), cached(&dir)).unwrap();
        assert!(g.started.get().is_none());
        assert!(g.with(|_| ()).is_none() && g.update().is_none() && g.errors().is_empty());
        assert!(!dir.exists());
        assert_eq!(font.outlines.load(Ordering::Relaxed), 0);
        assert_eq!(g.request("baab".chars()), ['a', 'b']);
        assert!(g.started.get().is_some());
        wait(&g, "ab");
        drop(g);
        assert!(dir.join(key(&*font, &cached(&dir)).file_name()).is_file());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn requested_glyphs_become_ready_and_the_preload_comes_along() {
        let dir = tmp("ready");
        let font = Arc::new(Ttf::new());
        let g = Generator::new(font.clone(), cached(&dir)).unwrap();
        let words = "Hello, world";
        assert_eq!(
            g.request(words.chars()),
            [' ', ',', 'H', 'd', 'e', 'l', 'o', 'r', 'w']
        );
        wait(&g, words);
        assert!(g.errors().is_empty());
        let gid = |ch| font.glyph(ch).unwrap();
        g.with(|a| {
            assert!(!a.cells.contains_key(&gid(' ')));
            for ch in ('!'..='~').chain(words.chars().filter(|c| *c != ' ')) {
                assert!(a.cells.contains_key(&gid(ch)), "{ch:?}");
            }
        })
        .unwrap();
        let u = g.update().unwrap();
        assert!(u.resized);
        assert_eq!(u.rows[0].texels.len() as u32, u.width * u.height * 4);
        assert!(g.update().is_none());
        drop(g);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_cache_hit_draws_without_generating() {
        let dir = tmp("hit");
        let made = {
            let g = Generator::new(Arc::new(Ttf::new()), cached(&dir)).unwrap();
            wait(&g, "ghee");
            g.with(|a| (a.cells.clone(), a.texels.clone())).unwrap()
        };
        let font = Arc::new(Ttf::new());
        let g = Generator::new(font.clone(), cached(&dir)).unwrap();
        wait(&g, "ghee");
        assert_eq!(font.outlines.load(Ordering::Relaxed), 0);
        assert!(g.with(|a| (a.cells.clone(), a.texels.clone())).unwrap() == made);
        drop(g);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_corrupt_or_foreign_cache_file_regenerates() {
        let dir = tmp("corrupt");
        fs::create_dir_all(&dir).unwrap();
        let font = boxes(2);
        let k = key(&*font, &cached(&dir));
        let path = dir.join(k.file_name());
        let foreign = encode(
            &Key {
                range: 4,
                ..k.clone()
            },
            &filled(5),
        );
        for junk in [b"not an atlas".to_vec(), foreign] {
            fs::write(&path, junk).unwrap();
            font.outlines.store(0, Ordering::Relaxed);
            let g = Generator::new(font.clone(), cached(&dir)).unwrap();
            wait(&g, "abc");
            assert!(g.errors().is_empty());
            drop(g);
            assert!(font.outlines.load(Ordering::Relaxed) >= 3);
            assert!(decode(&k, &fs::read(&path).unwrap()).is_some());
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_glyph_too_large_fails_alone_naming_its_character() {
        let dir = tmp("large");
        let g = Generator::new(boxes(3), cached(&dir)).unwrap();
        wait(&g, "A");
        // W failed in the preload, and says so once a request asks for it.
        assert!(g.errors().is_empty());
        wait(&g, "AWB");
        assert_eq!(
            g.errors(),
            ["glyph 'W' (U+0057): its 3208x31 cell is larger than the 2048x2048 atlas"]
        );
        wait(&g, "W");
        assert!(g.errors().is_empty());
        g.with(|a| {
            assert!(a.cells.contains_key(&('A' as u16)) && a.cells.contains_key(&('B' as u16)));
            assert!(!a.cells.contains_key(&('W' as u16)));
        })
        .unwrap();
        drop(g);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_glyph_past_the_preload_fills_on_demand() {
        let dir = tmp("demand");
        let font = boxes(4);
        let g = Generator::new(font.clone(), cached(&dir)).unwrap();
        wait(&g, "a");
        let before = font.outlines.load(Ordering::Relaxed);
        assert_eq!(g.request("a\u{3042}".chars()), ['\u{3042}']);
        wait(&g, "\u{3042}");
        assert_eq!(font.outlines.load(Ordering::Relaxed), before + 1);
        assert!(g.with(|a| a.cells.contains_key(&0x3042)).unwrap());
        let u = g.update().unwrap();
        assert!(u.rows.iter().map(|r| r.texels.len()).sum::<usize>() > 0);
        drop(g);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_unwritable_cache_keeps_the_atlas_in_memory() {
        let file = tmp("unwritable");
        fs::write(&file, b"").unwrap();
        let g = Generator::new(boxes(5), cached(&file.join("atlas"))).unwrap();
        wait(&g, "abc");
        wait(&g, "\u{3042}");
        assert!(g.errors().is_empty());
        assert!(g.with(|a| a.cells.len()).unwrap() > 190);
        drop(g);
        fs::remove_file(&file).unwrap();
    }

    #[test]
    fn a_dropped_generator_stops_its_preload() {
        let dir = tmp("drop");
        let g = Generator::new(Arc::new(Ttf::new()), cached(&dir)).unwrap();
        g.request("x".chars());
        let t0 = Instant::now();
        drop(g);
        assert!(t0.elapsed() < Duration::from_secs(5));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_zero_px_em_or_range_is_refused() {
        for (px_em, range) in [(0, 6), (32, 0)] {
            let config = Config {
                px_em,
                range,
                cache: None,
            };
            let e = Generator::new(boxes(6), config).err().unwrap();
            assert_eq!(
                e.to_string(),
                format!("an atlas needs a px/em and a range of 1 or more, got {px_em} and {range}")
            );
        }
    }
}
