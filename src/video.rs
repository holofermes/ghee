//! Video: the frames of a capture device, a camera or any V4L2 node that
//! streams MJPEG or YUYV, for video layers and video assets.
//!
//! A [`Capture`] streams one device on a thread of its own, through the
//! engine's V4L2 calls (ghee/_engine/video.c), and every renderer that shows
//! the device shares it. The thread decodes the newest frame the device
//! holds once the last one it decoded was taken, so it decodes no faster
//! than the renderers draw, and it runs only while no other thread wants the
//! CPU, so on a one-core Pi it takes the time that the renderer leaves.
//!
//! A frame comes out as RGBA rows, the bottom row first, as GL reads a
//! texture. MJPEG decodes at the smallest scale of 1, 1/2, 1/4 or 1/8 whose
//! picture still covers the size the frames are drawn at, which skips most
//! of the work of a large camera frame. YUYV averages square blocks of
//! pixels down to that size.
//!
//! A device that is missing, or that fails, gives the reason as its problem,
//! and the thread opens it again every second until it streams.

use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::ffi;
use crate::scene::py_str_repr;

/// MJPEG as V4L2 names it.
const MJPEG: u32 = u32::from_le_bytes(*b"MJPG");
/// YUYV 4:2:2 as V4L2 names it.
const YUYV: u32 = u32::from_le_bytes(*b"YUYV");

/// The frame rate under which a mode is too slow to pick while a faster
/// mode of the device exists.
const MIN_FPS: f64 = 15.0;

/// How far apart two aspects may lie and still show the same picture.
const SAME_ASPECT: f64 = 0.02;

/// How long one wait for a frame lasts, which bounds how long a stop takes.
const WAIT_MS: c_int = 200;

/// How long a device may stream nothing before that is its problem.
const SILENT: Duration = Duration::from_secs(3);

/// How long a missing or failed device waits before the next open.
const RETRY: Duration = Duration::from_secs(1);

/// The size of a video before its first frame, at the aspect of most
/// cameras.
pub(crate) const FIRST_SIZE: (u32, u32) = (64, 36);

/// A capture device that a video can show.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    /// The path that a scene names it by: a link in /dev/v4l/by-id, which
    /// stays the same across plugs and boots, where one leads to it, else
    /// its /dev/videoN node.
    pub device: String,
    /// The name that the device gives itself.
    pub name: String,
}

/// True when `video`, a video layer's `video`, names a device by its path
/// rather than a video asset, whose name holds no `/`.
pub fn is_device(video: &str) -> bool {
    video.starts_with('/')
}

/// The capture devices that a video can show, as devices_in() finds them in
/// /dev.
pub fn devices() -> Vec<Device> {
    devices_in(Path::new("/dev"), probe)
}

/// The name of the capture node at `path` that streams a format ghee
/// decodes, None for any other path.
fn probe(path: &str) -> Option<String> {
    let cpath = CString::new(path).ok()?;
    let mut name = [0 as c_char; 64];
    let found = unsafe { ffi::ghee_capture_probe(cpath.as_ptr(), name.as_mut_ptr(), 64) };
    (found == 1).then(|| text(&name).trim().to_string())
}

/// The capture devices under `dev`, each by the first of its links in
/// `dev`/v4l/by-id in name order, or by its own node with no link, the nodes
/// in number order. `probe` names a node that a video can show, and refuses
/// any other.
fn devices_in(dev: &Path, probe: impl Fn(&str) -> Option<String>) -> Vec<Device> {
    let mut nodes: Vec<(u32, String)> = std::fs::read_dir(dev)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let n = name.strip_prefix("video")?.parse().ok()?;
            Some((n, dev.join(&name).to_string_lossy().into_owned()))
        })
        .collect();
    nodes.sort();
    let mut links: Vec<(String, std::path::PathBuf)> = std::fs::read_dir(dev.join("v4l/by-id"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            let target = std::fs::canonicalize(&path).ok()?;
            Some((path.to_string_lossy().into_owned(), target))
        })
        .collect();
    links.sort();
    let mut out = Vec::new();
    for (_, node) in nodes {
        let Some(name) = probe(&node) else {
            continue;
        };
        let real = std::fs::canonicalize(&node).ok();
        let link = links
            .iter()
            .find(|(_, target)| Some(target) == real.as_ref())
            .map(|(link, _)| link.clone());
        out.push(Device {
            device: link.unwrap_or(node),
            name,
        });
    }
    out
}

/// The text of a C string that a call wrote into `buf`.
fn text(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// One mode that a device streams in.
pub(crate) type Mode = ffi::GheeCaptureMode;

/// The mode to stream in for frames drawn at `need`. Of the modes at
/// MIN_FPS or faster, or of every mode when none is, and of those with the
/// aspect of the largest, which shows the whole picture: the smallest that
/// covers `need`, a faster one first and YUYV before MJPEG at the same size,
/// since it costs less to decode, else the largest. None when there is no
/// mode.
pub(crate) fn pick_mode(modes: &[Mode], need: (u32, u32)) -> Option<Mode> {
    let area = |m: &Mode| i64::from(m.width) * i64::from(m.height);
    let fast: Vec<&Mode> = modes.iter().filter(|m| m.fps >= MIN_FPS).collect();
    let pool: Vec<&Mode> = match fast.is_empty() {
        true => modes.iter().collect(),
        false => fast,
    };
    let largest = pool.iter().max_by_key(|m| area(m))?;
    let aspect = |m: &Mode| f64::from(m.width) / f64::from(m.height.max(1));
    let whole = aspect(largest);
    let pool: Vec<&Mode> = pool
        .into_iter()
        .filter(|m| (aspect(m) / whole - 1.0).abs() <= SAME_ASPECT)
        .collect();
    let covers = |m: &&&Mode| m.width as u32 >= need.0 && m.height as u32 >= need.1;
    let cheaper = |a: &&&Mode, b: &&&Mode| {
        area(a)
            .cmp(&area(b))
            .then(b.fps.total_cmp(&a.fps))
            .then((a.fourcc != YUYV).cmp(&(b.fourcc != YUYV)))
    };
    let pick = match pool.iter().filter(covers).min_by(cheaper) {
        Some(m) => m,
        None => pool.iter().max_by(|a, b| cheaper(a, b))?,
    };
    Some(**pick)
}

/// The divisor of 1, 2, 4 or 8 that scales a `size` frame down the most
/// while it still covers `need` on both axes.
fn scale_for(size: (u32, u32), need: (u32, u32)) -> u32 {
    [8, 4, 2]
        .into_iter()
        .find(|d| size.0.div_ceil(*d) >= need.0 && size.1.div_ceil(*d) >= need.1)
        .unwrap_or(1)
}

/// A decoded frame: RGBA rows of `width` pixels, the bottom row first.
#[derive(Debug)]
pub(crate) struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// Counts the frames that the capture decoded, from 1.
    pub seq: u64,
}

/// The pixels of an MJPEG frame, at the smallest scale of 1, 1/2, 1/4 or 1/8
/// that covers `need`.
fn decode_mjpeg(bytes: &[u8], need: (u32, u32)) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    let mut d = jpeg_decoder::Decoder::new(bytes);
    d.read_info()?;
    let Some(info) = d.info() else {
        anyhow::bail!("no frame header");
    };
    let size = (u32::from(info.width), u32::from(info.height));
    // The decoder takes the first scale that covers either axis, so it is
    // asked for the scale that covers both.
    let k = scale_for(size, need);
    d.scale(size.0.div_ceil(k) as u16, size.1.div_ceil(k) as u16)?;
    let px = d.decode()?;
    let Some(info) = d.info() else {
        anyhow::bail!("no frame header");
    };
    let (w, h) = (u32::from(info.width), u32::from(info.height));
    let rgba = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => {
            bottom_first(&px, w as usize * 3, |p: &[u8; 3]| [p[0], p[1], p[2], 255])
        }
        jpeg_decoder::PixelFormat::L8 => {
            bottom_first(&px, w as usize, |p: &[u8; 1]| [p[0], p[0], p[0], 255])
        }
        f => anyhow::bail!("a frame of {f:?}, and ghee reads RGB or grey"),
    };
    Ok((w, h, rgba))
}

/// The pixels of `rows`, each `row` bytes long, in the opposite order of
/// rows, each pixel as `rgba` makes it of its `N` bytes.
fn bottom_first<const N: usize>(
    rows: &[u8],
    row: usize,
    rgba: impl Fn(&[u8; N]) -> [u8; 4],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows.len() / N * 4);
    for r in rows.chunks_exact(row).rev() {
        out.extend(r.as_chunks::<N>().0.iter().flat_map(&rgba));
    }
    out
}

/// RGB of one BT.601 limited-range YCbCr pixel, as UVC cameras send YUYV.
fn yuv_rgb(y: i32, u: i32, v: i32) -> [u8; 3] {
    let c = 298 * (y - 16);
    let (d, e) = (u - 128, v - 128);
    let px = |x: i32| ((x + 128) >> 8).clamp(0, 255) as u8;
    [px(c + 409 * e), px(c - 100 * d - 208 * e), px(c + 516 * d)]
}

/// The pixels of a YUYV frame of `w` x `h`, its rows `bpl` bytes apart, with
/// each block of k x k pixels averaged into one, k the largest whole number
/// that still covers `need`.
fn decode_yuyv(
    bytes: &[u8],
    w: u32,
    h: u32,
    bpl: usize,
    need: (u32, u32),
) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    let bpl = bpl.max(w as usize * 2);
    if bytes.len() < bpl * (h as usize - 1) + w as usize * 2 {
        anyhow::bail!("a YUYV frame of {} bytes, short of {w}x{h}", bytes.len());
    }
    let k = (w / need.0.max(1)).min(h / need.1.max(1)).max(1);
    let (ow, oh) = (w / k, h / k);
    let mut rgba = Vec::with_capacity(ow as usize * oh as usize * 4);
    let n = (k * k) as i32;
    for oy in (0..oh).rev() {
        for ox in 0..ow {
            let (mut ys, mut us, mut vs) = (0i32, 0i32, 0i32);
            for y in oy * k..oy * k + k {
                let row = &bytes[y as usize * bpl..];
                for x in ox * k..ox * k + k {
                    let pair = (x as usize & !1) * 2;
                    ys += i32::from(row[x as usize * 2]);
                    us += i32::from(row[pair + 1]);
                    vs += i32::from(row[pair + 3]);
                }
            }
            let [r, g, b] = yuv_rgb(ys / n, us / n, vs / n);
            rgba.extend([r, g, b, 255]);
        }
    }
    Ok((ow, oh, rgba))
}

/// Yellow and black stripes of `w` x `h` pixels, RGBA with the bottom row
/// first: the look of a text's error box, which a video shows while it has
/// no frame. The stripes run at 45 degrees, up to the right, an eighth of the
/// height wide along a row.
pub(crate) fn stripes(w: u32, h: u32) -> Vec<u8> {
    let width = (h / 8).max(2) as i64;
    let mut out = Vec::with_capacity(w as usize * h as usize * 4);
    for y in 0..i64::from(h) {
        for x in 0..i64::from(w) {
            let yellow = (x - y).div_euclid(width) % 2 == 0;
            out.extend(if yellow {
                [255, 204, 0, 255]
            } else {
                [0, 0, 0, 255]
            });
        }
    }
    out
}

/// What a capture's thread and its renderers share.
#[derive(Default)]
struct State {
    /// The size each user draws the frames at, by the user.
    needs: Vec<(u64, (u32, u32))>,
    /// The newest frame decoded.
    frame: Option<Arc<Frame>>,
    /// True once a user took the newest frame, or before the first.
    taken: bool,
    /// Why the device shows no frame now, None while it streams.
    problem: Option<String>,
}

struct Shared {
    stop: AtomicBool,
    state: Mutex<State>,
    wake: Condvar,
}

impl Shared {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The size the frames are drawn at: the largest of the users' sizes on
    /// each axis.
    fn need(&self) -> (u32, u32) {
        let s = self.state();
        s.needs
            .iter()
            .fold((1, 1), |(w, h), (_, n)| (w.max(n.0), h.max(n.1)))
    }

    fn set_problem(&self, path: &str, problem: Option<String>) {
        let mut s = self.state();
        if s.problem == problem {
            return;
        }
        if let Some(p) = &problem {
            crate::report::error(format!("video {}: {p}", py_str_repr(path)));
        }
        s.problem = problem;
    }
}

/// One device's stream, shared by every renderer that shows it.
pub(crate) struct Capture {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

/// The captures that run, by their device's node.
static CAPTURES: Mutex<Vec<(String, Weak<Capture>)>> = Mutex::new(Vec::new());

/// The node that `path` leads to, past its links, or `path` itself when it
/// leads nowhere now.
fn node_of(path: &str) -> String {
    std::fs::canonicalize(path)
        .map_or_else(|_| path.to_string(), |p| p.to_string_lossy().into_owned())
}

/// Counts the users of every capture, for their ids.
static USERS: AtomicU64 = AtomicU64::new(1);

impl Capture {
    /// The capture of the device at `path`: the one that runs on its node,
    /// whatever link led to it, else a new one, which starts its thread.
    pub(crate) fn of(path: &str) -> Arc<Capture> {
        let node = node_of(path);
        let mut all = CAPTURES.lock().unwrap_or_else(|e| e.into_inner());
        all.retain(|(_, c)| c.strong_count() > 0);
        if let Some(c) = all
            .iter()
            .find(|(n, _)| *n == node)
            .and_then(|(_, c)| c.upgrade())
        {
            return c;
        }
        let shared = Arc::new(Shared {
            stop: AtomicBool::new(false),
            state: Mutex::new(State {
                taken: true,
                ..State::default()
            }),
            wake: Condvar::new(),
        });
        let (p, s) = (path.to_string(), shared.clone());
        let thread = std::thread::Builder::new()
            .name("ghee-video".to_string())
            .spawn(move || run(&p, &s))
            .ok();
        let c = Arc::new(Capture { shared, thread });
        all.push((node, Arc::downgrade(&c)));
        c
    }

    /// A new user id for need().
    pub(crate) fn user() -> u64 {
        USERS.fetch_add(1, Ordering::Relaxed)
    }

    /// Set the size that `user` draws the frames at, or take the user off
    /// with None.
    pub(crate) fn need(&self, user: u64, size: Option<(u32, u32)>) {
        let mut s = self.shared.state();
        s.needs.retain(|(u, _)| *u != user);
        if let Some(n) = size {
            s.needs.push((user, (n.0.max(1), n.1.max(1))));
        }
        drop(s);
        self.shared.wake.notify_all();
    }

    /// The newest frame when it is newer than frame `seen`.
    pub(crate) fn newer(&self, seen: u64) -> Option<Arc<Frame>> {
        let mut s = self.shared.state();
        let f = s.frame.clone().filter(|f| f.seq > seen)?;
        s.taken = true;
        drop(s);
        self.shared.wake.notify_all();
        Some(f)
    }

    /// Why the device shows no frame now, None while it streams.
    pub(crate) fn problem(&self) -> Option<String> {
        self.shared.state().problem.clone()
    }
}

#[cfg(test)]
impl Capture {
    /// A capture of `path` with no device behind it, which shows the frames
    /// that a test hands it with show().
    pub(crate) fn fake(path: &str) -> Arc<Capture> {
        let c = Arc::new(Capture {
            shared: Arc::new(Shared {
                stop: AtomicBool::new(false),
                state: Mutex::new(State {
                    taken: true,
                    ..State::default()
                }),
                wake: Condvar::new(),
            }),
            thread: None,
        });
        let mut all = CAPTURES.lock().unwrap_or_else(|e| e.into_inner());
        all.retain(|(p, w)| p != path && w.strong_count() > 0);
        all.push((path.to_string(), Arc::downgrade(&c)));
        c
    }

    /// Show `rgb`, RGB rows of `w` x `h` pixels, the top row first, as the
    /// newest frame.
    pub(crate) fn show(&self, w: u32, h: u32, rgb: &[u8]) {
        let mut s = self.shared.state();
        let seq = s.frame.as_ref().map_or(0, |f| f.seq) + 1;
        s.frame = Some(Arc::new(Frame {
            width: w,
            height: h,
            rgba: bottom_first(rgb, w as usize * 3, |p: &[u8; 3]| [p[0], p[1], p[2], 255]),
            seq,
        }));
        s.taken = false;
        s.problem = None;
    }

    /// The size that the users draw the frames at, None with no user.
    pub(crate) fn wanted(&self) -> Option<(u32, u32)> {
        if self.shared.state().needs.is_empty() {
            return None;
        }
        Some(self.shared.need())
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.wake.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// An open device, closed at its drop.
struct Open(*mut ffi::GheeCapture);

impl Drop for Open {
    fn drop(&mut self) {
        unsafe { ffi::ghee_capture_close(self.0) };
    }
}

/// The thread of a capture: stream the device while it has a user, until
/// the capture stops, opening it again a second after each failure.
fn run(path: &str, shared: &Shared) {
    unsafe { ffi::ghee_thread_idle() };
    let stopped = || shared.stop.load(Ordering::Relaxed);
    while !stopped() {
        if shared.state().needs.is_empty() {
            let s = shared.state();
            drop(
                shared
                    .wake
                    .wait_while(s, |s| s.needs.is_empty() && !stopped()),
            );
            continue;
        }
        let problem = match stream(path, shared) {
            Ok(()) => continue,
            Err(why) => why,
        };
        shared.set_problem(path, Some(problem));
        let s = shared.state();
        drop(shared.wake.wait_timeout_while(s, RETRY, |_| !stopped()));
    }
}

/// Stream the device at `path` in the mode that its users' sizes pick,
/// until the capture stops, its last user leaves or the users' sizes pick
/// another mode. Err holds why the device streams no more.
fn stream(path: &str, shared: &Shared) -> Result<(), String> {
    let mut err = [0 as c_char; 256];
    let cpath = CString::new(path).map_err(|_| "a path holds a NUL".to_string())?;
    let handle = unsafe { ffi::ghee_capture_open(cpath.as_ptr(), err.as_mut_ptr(), 256) };
    if handle.is_null() {
        return Err(text(&err));
    }
    let cap = Open(handle);
    let mut modes = [Mode::default(); 64];
    let n = unsafe { ffi::ghee_capture_modes(cap.0, modes.as_mut_ptr(), 64) };
    let modes = &modes[..n.max(0) as usize];
    let need = shared.need();
    let Some(mode) = pick_mode(modes, need) else {
        return Err("the device streams neither MJPEG nor YUYV".to_string());
    };
    let started = unsafe {
        ffi::ghee_capture_start(
            cap.0,
            mode.fourcc,
            mode.width,
            mode.height,
            mode.fps,
            err.as_mut_ptr(),
            256,
        )
    };
    if started < 0 {
        return Err(text(&err));
    }
    let (mut fourcc, mut w, mut h, mut bpl) = (0u32, 0, 0, 0);
    unsafe { ffi::ghee_capture_format(cap.0, &mut fourcc, &mut w, &mut h, &mut bpl) };
    let format = if fourcc == MJPEG { "MJPEG" } else { "YUYV" };
    crate::report::detail(
        "video",
        format!(
            "{}: {format} {w}x{h} at {} fps",
            py_str_repr(path),
            mode.fps.round()
        ),
    );
    let mut last = Instant::now();
    let mut seq = shared.state().frame.as_ref().map_or(0, |f| f.seq);
    while !shared.stop.load(Ordering::Relaxed) {
        if shared.state().needs.is_empty() {
            return Ok(());
        }
        let now_need = shared.need();
        if now_need != need && pick_mode(modes, now_need) != Some(mode) {
            return Ok(());
        }
        let (mut data, mut len) = (std::ptr::null(), 0usize);
        let got = unsafe {
            ffi::ghee_capture_next(cap.0, WAIT_MS, &mut data, &mut len, err.as_mut_ptr(), 256)
        };
        if got < 0 {
            return Err(text(&err));
        }
        if got == 0 {
            if last.elapsed() >= SILENT {
                shared.set_problem(path, Some(format!("no frame for {} s", SILENT.as_secs())));
            }
            continue;
        }
        last = Instant::now();
        if !shared.state().taken {
            continue;
        }
        let bytes = unsafe { std::slice::from_raw_parts(data, len) };
        let decoded = match fourcc {
            MJPEG => decode_mjpeg(bytes, now_need),
            _ => decode_yuyv(bytes, w as u32, h as u32, bpl as usize, now_need),
        };
        unsafe { ffi::ghee_capture_release(cap.0) };
        // A camera sends a broken frame now and then, and the next one
        // shows.
        let Ok((fw, fh, rgba)) = decoded else {
            continue;
        };
        seq += 1;
        {
            let mut s = shared.state();
            s.frame = Some(Arc::new(Frame {
                width: fw,
                height: fh,
                rgba,
                seq,
            }));
            s.taken = false;
        }
        shared.set_problem(path, None);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mode(fourcc: u32, width: c_int, height: c_int, fps: f64) -> Mode {
        Mode {
            fourcc,
            width,
            height,
            fps,
        }
    }

    #[test]
    fn a_video_streams_the_cheapest_mode_that_covers_its_size() {
        let webcam = [
            mode(MJPEG, 1920, 1080, 30.0),
            mode(MJPEG, 1280, 720, 30.0),
            mode(MJPEG, 640, 480, 30.0),
            mode(YUYV, 1920, 1080, 5.0),
            mode(YUYV, 1280, 720, 10.0),
            mode(YUYV, 640, 360, 30.0),
            mode(YUYV, 640, 480, 30.0),
        ];
        let cases = [
            // a panel: YUYV at the smallest 16:9 size
            ((128, 64), mode(YUYV, 640, 360, 30.0)),
            // past every YUYV mode fast enough: MJPEG
            ((1280, 700), mode(MJPEG, 1280, 720, 30.0)),
            // past every mode: the largest
            ((3840, 2160), mode(MJPEG, 1920, 1080, 30.0)),
        ];
        for (need, want) in cases {
            assert_eq!(pick_mode(&webcam, need), Some(want), "{need:?}");
        }
        let slow = [mode(YUYV, 640, 480, 5.0), mode(YUYV, 320, 240, 10.0)];
        assert_eq!(
            pick_mode(&slow, (128, 64)),
            Some(mode(YUYV, 320, 240, 10.0))
        );
        assert_eq!(pick_mode(&[], (128, 64)), None);
    }

    #[test]
    fn mjpeg_decodes_at_the_smallest_scale_that_covers_the_size() {
        let px = vec![200u8; 64 * 32 * 3];
        let mut jpeg = Vec::new();
        jpeg_encoder::Encoder::new(&mut jpeg, 90)
            .encode(&px, 64, 32, jpeg_encoder::ColorType::Rgb)
            .unwrap();
        let cases = [
            ((8, 4), (8, 4)),
            ((9, 4), (16, 8)),
            ((17, 1), (32, 16)),
            ((33, 4), (64, 32)),
            ((128, 64), (64, 32)),
        ];
        for (need, size) in cases {
            let (w, h, rgba) = decode_mjpeg(&jpeg, need).unwrap();
            assert_eq!((w, h), size, "{need:?}");
            assert_eq!(rgba.len(), (w * h * 4) as usize);
        }
    }

    #[test]
    fn yuyv_turns_into_bt601_limited_range_rgb() {
        // ITU-R BT.601 at limited range: black, white and the red of the
        // 75% colour bars, (0.75, 0, 0).
        let cases = [
            ((16, 128, 128), [0, 0, 0]),
            ((235, 128, 128), [255, 255, 255]),
            ((65, 100, 212), [191, 0, 0]),
        ];
        for ((y, u, v), want) in cases {
            let got = yuv_rgb(y, u, v);
            for c in 0..3 {
                assert!(
                    (i32::from(got[c]) - want[c]).abs() <= 2,
                    "{y} {u} {v}: {got:?}"
                );
            }
        }
    }

    #[test]
    fn a_yuyv_frame_averages_blocks_and_holds_its_bottom_row_first() {
        // 4x2 pixels, the top row white, the bottom row black, with 2 bytes
        // of padding a row.
        let row = |y: u8| [y, 128, y, 128, y, 128, y, 128, 0, 0];
        let frame: Vec<u8> = row(235).into_iter().chain(row(16)).collect();
        let (w, h, rgba) = decode_yuyv(&frame, 4, 2, 10, (4, 2)).unwrap();
        assert_eq!((w, h), (4, 2));
        assert_eq!(&rgba[..4], &[0, 0, 0, 255]);
        assert_eq!(&rgba[16..20], &[255, 255, 255, 255]);
        let (w, h, rgba) = decode_yuyv(&frame, 4, 2, 10, (2, 1)).unwrap();
        assert_eq!((w, h), (2, 1));
        assert!(
            rgba.as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[..3].iter().all(|c| (126..=129).contains(c))),
            "{rgba:?}"
        );
    }

    #[test]
    fn the_stripes_are_yellow_and_black_at_45_degrees() {
        let px = stripes(16, 16);
        let at = |x: usize, y: usize| &px[(y * 16 + x) * 4..][..4];
        assert_eq!(at(0, 0), [255, 204, 0, 255]);
        assert_eq!(at(2, 0), [0, 0, 0, 255]);
        // up to the right: a step right and a step up keep the colour
        assert_eq!(at(3, 1), at(2, 0));
        assert_eq!(at(1, 1), at(0, 0));
    }

    #[test]
    fn a_device_goes_by_its_by_id_link_or_by_its_node() {
        let dir = crate::testing::TempDir::new("video-devices");
        let dev: &Path = dir.as_ref();
        for n in ["video0", "video1", "video10", "video2"] {
            std::fs::write(dev.join(n), b"").unwrap();
        }
        let by_id = dev.join("v4l/by-id");
        std::fs::create_dir_all(&by_id).unwrap();
        std::os::unix::fs::symlink("../../video0", by_id.join("usb-cam-video-index0")).unwrap();
        std::os::unix::fs::symlink("../../video1", by_id.join("usb-cam-video-index1")).unwrap();
        // video1 is the camera's metadata node and video10 a codec, which
        // the probe refuses.
        let found = devices_in(dev, |p| match p.rsplit('/').next() {
            Some("video0") => Some("cam".to_string()),
            Some("video2") => Some("hdmi".to_string()),
            _ => None,
        });
        let at = |p: &str| dev.join(p).to_string_lossy().into_owned();
        assert_eq!(
            found,
            vec![
                Device {
                    device: at("v4l/by-id/usb-cam-video-index0"),
                    name: "cam".to_string()
                },
                Device {
                    device: at("video2"),
                    name: "hdmi".to_string()
                },
            ]
        );
    }

    #[test]
    fn a_device_path_starts_with_a_slash() {
        for (video, device) in [("/dev/video0", true), ("cam", false), ("", false)] {
            assert_eq!(is_device(video), device, "{video}");
        }
    }

    #[test]
    fn a_missing_device_says_why_and_shows_no_frame() {
        let c = Capture::of("/nonexistent/video9");
        c.need(Capture::user(), Some((64, 32)));
        let deadline = Instant::now() + Duration::from_secs(5);
        while c.problem().is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(c.problem().as_deref(), Some("No such file or directory"));
        assert!(c.newer(0).is_none());
    }
}
