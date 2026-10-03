//! The sinks themselves. The Output and Pacer traits live in ghee::looper,
//! beside the loop that drives them.
//!
//! ImageSequence encodes PNG and JPEG itself. ghee links no image library.
//! `quality` in save_opts reaches the JPEG encoder. The encoder ignores an
//! option it has no use for.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use anyhow::{bail, Context, Result};
use jpeg_encoder::{ColorType, Encoder};
use serde_json::{Map, Value};

use crate::looper::{Frame, Output, OwnedFrame, Pacer};
use crate::stats::{DEST_FILE, DEST_KMS, DEST_PIPE};

/// frame.####.jpg + 7 -> frame.0007.jpg. A printf field (frame.%04d.jpg) also
/// works. A plain filename comes back as it is (one frame, or an overwrite).
pub fn expand_pattern(pattern: &str, index: i64) -> Result<String> {
    if let Some(start) = pattern.find('#') {
        let width = pattern[start..].bytes().take_while(|b| *b == b'#').count();
        return Ok(format!(
            "{}{:0width$}{}",
            &pattern[..start],
            index,
            &pattern[start + width..]
        ));
    }
    if pattern.contains('%') {
        return percent_format(pattern, index);
    }
    Ok(pattern.to_string())
}

/// The integer conversions of a printf-style `pattern % index`. A frame number
/// reaches its filename this way.
fn percent_format(pattern: &str, index: i64) -> Result<String> {
    let src = pattern.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(src.len() + 8);
    let mut used = false;
    let mut i = 0;
    while i < src.len() {
        if src[i] != b'%' {
            out.push(src[i]);
            i += 1;
            continue;
        }
        i += 1;
        if src.get(i) == Some(&b'%') {
            out.push(b'%');
            i += 1;
            continue;
        }
        let mut spec = Spec::default();
        while let Some(f) = src.get(i) {
            match f {
                b'0' => spec.zero = true,
                b'-' => spec.left = true,
                b' ' => spec.space = true,
                b'+' => spec.plus = true,
                _ => break,
            }
            i += 1;
        }
        while src.get(i).is_some_and(u8::is_ascii_digit) {
            spec.width = spec.width * 10 + usize::from(src[i] - b'0');
            i += 1;
        }
        if src.get(i) == Some(&b'.') {
            i += 1;
            let mut p = 0;
            while src.get(i).is_some_and(u8::is_ascii_digit) {
                p = p * 10 + usize::from(src[i] - b'0');
                i += 1;
            }
            spec.precision = Some(p);
        }
        while matches!(src.get(i), Some(b'h' | b'l' | b'L')) {
            i += 1;
        }
        let Some(&conv) = src.get(i) else {
            bail!("incomplete format");
        };
        if !b"diuxXos".contains(&conv) {
            bail!(
                "unsupported format character '{}' (0x{conv:x}) at index {i}",
                conv as char
            );
        }
        if used {
            bail!("not enough arguments for format string");
        }
        used = true;
        i += 1;
        out.extend_from_slice(spec.render(conv, index).as_bytes());
    }
    if !used {
        bail!("not all arguments converted during string formatting");
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// A parsed conversion: the flags, width and precision ahead of its character.
/// `#` never reaches here. A pattern that holds one takes the ####-run branch.
#[derive(Default)]
struct Spec {
    zero: bool,
    left: bool,
    space: bool,
    plus: bool,
    width: usize,
    precision: Option<usize>,
}

impl Spec {
    fn render(&self, conv: u8, index: i64) -> String {
        if conv == b's' {
            let mut body = index.to_string();
            if let Some(p) = self.precision {
                body.truncate(p);
            }
            return self.pad("", &body, false);
        }
        let n = index.unsigned_abs();
        let mut body = match conv {
            b'x' => format!("{n:x}"),
            b'X' => format!("{n:X}"),
            b'o' => format!("{n:o}"),
            _ => n.to_string(),
        };
        if let Some(p) = self.precision {
            body = format!("{body:0>p$}");
        }
        let sign = if index < 0 {
            "-"
        } else if self.plus {
            "+"
        } else if self.space {
            " "
        } else {
            ""
        };
        self.pad(sign, &body, self.zero)
    }

    fn pad(&self, head: &str, body: &str, zero: bool) -> String {
        let pad = self.width.saturating_sub(head.len() + body.len());
        if pad == 0 {
            format!("{head}{body}")
        } else if self.left {
            format!("{head}{body}{}", " ".repeat(pad))
        } else if zero {
            format!("{head}{}{body}", "0".repeat(pad))
        } else {
            format!("{}{head}{body}", " ".repeat(pad))
        }
    }
}

/// Hold a display loop at an explicit frame rate.
pub struct FixedPacer {
    pub period: f64,
}

impl FixedPacer {
    pub fn new(fps: f64) -> FixedPacer {
        FixedPacer { period: 1.0 / fps }
    }
}

impl Pacer for FixedPacer {
    fn sleep_for(&mut self, _write_s: f64, frame_s: f64) -> f64 {
        (self.period - frame_s).max(0.0)
    }

    fn idle_period(&self) -> f64 {
        self.period
    }
}

/// KMS. The renderer page-flips the final buffer to the connector itself
/// (Renderer::present). This sink holds nothing and never takes a frame. It
/// names the destination and puts the loop on present().
pub struct Scanout;

impl Output for Scanout {
    fn dest(&self) -> &str {
        DEST_KMS
    }

    fn scanout(&self) -> bool {
        true
    }

    fn persists(&self) -> bool {
        true
    }

    fn wants(&self, _frame: u64) -> bool {
        false
    }

    fn write(&mut self, _frame: Frame<'_>) -> Result<()> {
        Ok(())
    }
}

enum Job {
    Frame(i64, OwnedFrame),
    Stop,
}

/// Write each frame to its own numbered file. Encoding and disk I/O run on one
/// worker thread, so the render loop never waits on the encoder.
///
/// The pool path takes no per-frame copy. acquire() hands out a preallocated
/// buffer. The renderer reads the GPU straight into it (render_into). submit()
/// queues it. The worker encodes, writes and recycles the buffer.
///
/// The pool grows to depth+1 buffers, so acquire() blocks only when the writer
/// is `depth` frames behind. depth is the grace for a slow writer. The depth+1
/// cap is the memory bound on a small board. A depth of the whole frame count
/// keeps the render clear of every wait. The writer drains at the end.
pub struct ImageSequence {
    #[allow(dead_code)]
    pub pattern: String,
    pub index: i64,
    cap: usize,
    allocated: usize,
    jobs: Sender<Job>,
    free: Receiver<OwnedFrame>,
    recycle: Sender<OwnedFrame>,
    err: Arc<Mutex<Option<String>>>,
    worker: Option<JoinHandle<()>>,
}

impl ImageSequence {
    /// The defaults: start 0, no save options, depth 16.
    #[allow(dead_code)]
    pub fn new(pattern: &str) -> ImageSequence {
        ImageSequence::with_opts(pattern, 0, None, 16)
    }

    pub fn with_opts(
        pattern: &str,
        start: i64,
        save_opts: Option<Map<String, Value>>,
        depth: usize,
    ) -> ImageSequence {
        let (jobs, inbox) = channel::<Job>();
        let (recycle, free) = channel::<OwnedFrame>();
        let err = Arc::new(Mutex::new(None));
        let worker = {
            let (pattern, opts) = (pattern.to_string(), save_opts.unwrap_or_default());
            let (recycle, err) = (recycle.clone(), Arc::clone(&err));
            std::thread::spawn(move || run(&inbox, &recycle, &err, &pattern, &opts))
        };
        ImageSequence {
            pattern: pattern.to_string(),
            index: start,
            cap: depth + 1,
            allocated: 0,
            jobs,
            free,
            recycle,
            err,
            worker: Some(worker),
        }
    }

    fn failure(&self) -> Option<String> {
        self.err.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Stop the writer once it drains. Report the failure it holds.
    fn finish(&mut self) -> Result<()> {
        let _ = self.jobs.send(Job::Stop);
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                bail!("the image writer thread died");
            }
        }
        match self.failure() {
            Some(e) => bail!(e),
            None => Ok(()),
        }
    }
}

/// The writer thread: encode, write, recycle. The buffer belongs to the worker
/// alone between submit and recycle, so the encoder reads it in place. After a
/// failure the worker keeps draining and recycling, so a blocked acquire()
/// never starves. The error reaches the caller on the next acquire(), the next
/// submit(), or on close().
fn run(
    inbox: &Receiver<Job>,
    recycle: &Sender<OwnedFrame>,
    err: &Arc<Mutex<Option<String>>>,
    pattern: &str,
    opts: &Map<String, Value>,
) {
    while let Ok(Job::Frame(index, buf)) = inbox.recv() {
        // the write never holds the lock, because acquire() reads it every
        // frame from the render thread
        let failed = err.lock().unwrap_or_else(|e| e.into_inner()).is_some();
        if !failed {
            if let Err(e) = save(pattern, index, buf.view(), opts) {
                *err.lock().unwrap_or_else(|e| e.into_inner()) = Some(format!("{e:#}"));
            }
        }
        let _ = recycle.send(buf);
    }
}

fn save(pattern: &str, index: i64, frame: Frame<'_>, opts: &Map<String, Value>) -> Result<()> {
    let path = expand_pattern(pattern, index)?;
    let bytes = encode(&path, frame, opts)?;
    fs::write(&path, bytes).with_context(|| path.clone())
}

impl Output for ImageSequence {
    fn dest(&self) -> &str {
        DEST_FILE
    }

    fn pooled(&self) -> bool {
        true
    }

    /// A buffer of `shape` to render into. It takes a recycled buffer first.
    /// It grows the pool to its cap next. It blocks for a recycled buffer
    /// last, which is the render's only wait. Single-producer: only the render
    /// loop calls it.
    fn acquire(&mut self, shape: (i32, i32, i32)) -> Result<OwnedFrame> {
        if let Some(e) = self.failure() {
            bail!(e);
        }
        if let Ok(buf) = self.free.try_recv() {
            return Ok(buf);
        }
        if self.allocated < self.cap {
            self.allocated += 1;
            return Ok(OwnedFrame::of(shape));
        }
        Ok(self.free.recv()?)
    }

    /// Queue a filled buffer for encoding under the next numbered name.
    fn submit(&mut self, buf: OwnedFrame) -> Result<()> {
        if let Some(e) = self.failure() {
            let _ = self.recycle.send(buf);
            bail!(e);
        }
        // after close() no worker is left, so the frame goes nowhere
        let _ = self.jobs.send(Job::Frame(self.index, buf));
        self.index += 1;
        Ok(())
    }

    /// Copy a caller-owned frame into a pooled buffer and submit it.
    fn write(&mut self, frame: Frame<'_>) -> Result<()> {
        let mut buf = self.acquire((frame.height, frame.width, frame.components))?;
        buf.data.copy_from_slice(frame.data);
        self.submit(buf)
    }

    fn close(&mut self) -> Result<()> {
        self.finish()
    }
}

impl Drop for ImageSequence {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

/// Write each frame's exact bytes to a binary stream, no header, one frame
/// after another (h*w*components, row-major). For piping raw video to ffmpeg,
/// mpv, or a recorder.
pub struct RawStream<W: Write> {
    pub stream: W,
}

impl<W: Write> RawStream<W> {
    pub fn new(stream: W) -> RawStream<W> {
        RawStream { stream }
    }
}

impl<W: Write> Output for RawStream<W> {
    fn dest(&self) -> &str {
        DEST_PIPE
    }

    fn write(&mut self, frame: Frame<'_>) -> Result<()> {
        self.stream.write_all(frame.data)?;
        Ok(self.stream.flush()?)
    }
}

/// The extension that picks the format: everything after the last dot of the
/// file name, lowercased. A name without a dot gives "".
fn suffix(path: &str) -> String {
    Path::new(path)
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default()
}

fn encode(path: &str, frame: Frame<'_>, opts: &Map<String, Value>) -> Result<Vec<u8>> {
    match suffix(path).as_str() {
        ".png" => encode_png(frame),
        ".jpg" | ".jpeg" => {
            let quality = opts.get("quality").and_then(Value::as_u64).unwrap_or(75);
            encode_jpeg(frame, quality as u8)
        }
        ext => bail!("unknown file extension: {ext}"),
    }
}

/// A frame as baseline JPEG. JPEG holds no alpha channel, so RGBA fails.
pub fn encode_jpeg(frame: Frame<'_>, quality: u8) -> Result<Vec<u8>> {
    let color = match frame.components {
        3 => ColorType::Rgb,
        4 => bail!("cannot write mode RGBA as JPEG"),
        n => bail!("a {n}-component frame is neither RGB nor RGBA"),
    };
    let mut out = Vec::new();
    Encoder::new(&mut out, quality).encode(
        frame.data,
        frame.width as u16,
        frame.height as u16,
        color,
    )?;
    Ok(out)
}

const CRC32: [u32; 256] = crc32_table();

const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 == 0 {
                c >> 1
            } else {
                0xedb8_8320 ^ (c >> 1)
            };
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut c = 0xffff_ffffu32;
    for b in bytes {
        c = CRC32[((c ^ u32::from(*b)) & 0xff) as usize] ^ (c >> 8);
    }
    c ^ 0xffff_ffff
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    // 5552 bytes is the most a u32 b sums without an overflow
    for block in bytes.chunks(5552) {
        for byte in block {
            a += u32::from(*byte);
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

/// A zlib stream of stored (uncompressed) deflate blocks. Any PNG decoder
/// reads it. The file is larger than a compressed one.
fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + raw.len() / 65535 * 5 + 11);
    out.extend_from_slice(&[0x78, 0x01]);
    let mut blocks = raw.chunks(65535).peekable();
    if blocks.peek().is_none() {
        out.extend_from_slice(&[0x01, 0x00, 0x00, 0xff, 0xff]);
    }
    while let Some(block) = blocks.next() {
        let len = block.len() as u16;
        out.push(u8::from(blocks.peek().is_none()));
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(block);
    }
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    out
}

fn png_chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    let mut body = Vec::with_capacity(4 + data.len());
    body.extend_from_slice(tag);
    body.extend_from_slice(data);
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc32(&body).to_be_bytes());
}

/// A frame as an 8-bit RGB or RGBA PNG. An RGBA frame holds colour
/// premultiplied by alpha, and the PNG holds it straight.
pub fn encode_png(frame: Frame<'_>) -> Result<Vec<u8>> {
    let color = match frame.components {
        3 => 2u8,
        4 => 6u8,
        n => bail!("a {n}-component frame is neither RGB nor RGBA"),
    };
    let (width, height) = (frame.width as usize, frame.height as usize);
    let stride = width * frame.components as usize;
    let mut raw = Vec::with_capacity(height * (stride + 1));
    for row in frame.data.chunks(stride).take(height) {
        raw.push(0); // filter type: none
        let start = raw.len();
        raw.extend_from_slice(row);
        if frame.components == 4 {
            unpremultiply(&mut raw[start..]);
        }
    }
    let mut head = Vec::with_capacity(13);
    head.extend_from_slice(&(width as u32).to_be_bytes());
    head.extend_from_slice(&(height as u32).to_be_bytes());
    head.extend_from_slice(&[8, color, 0, 0, 0]);
    let mut out = Vec::with_capacity(raw.len() + 64);
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
    png_chunk(&mut out, b"IHDR", &head);
    png_chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    png_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

/// Divide each colour of RGBA pixels by its alpha, rounded to the nearest
/// byte. An opaque pixel keeps its bytes, and a transparent one turns black.
fn unpremultiply(px: &mut [u8]) {
    for p in px.chunks_exact_mut(4) {
        let a = u32::from(p[3]);
        if a == 255 {
            continue;
        }
        for c in &mut p[..3] {
            *c = match a {
                0 => 0,
                _ => ((u32::from(*c) * 255 + a / 2) / a).min(255) as u8,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread::sleep;
    use std::time::Duration;

    use super::*;

    fn tmpdir(tag: &str) -> crate::testing::TempDir {
        crate::testing::TempDir::new(&format!("output-{tag}"))
    }

    fn at(dir: impl AsRef<Path>, name: &str) -> String {
        dir.as_ref().join(name).to_string_lossy().into_owned()
    }

    /// n frames of (8, 6, 4), each a flat value with an opaque alpha.
    fn frames(n: usize, step: u8) -> Vec<OwnedFrame> {
        (0..n)
            .map(|i| {
                let mut buf = OwnedFrame::of((8, 6, 4));
                for px in buf.data.chunks_mut(4) {
                    px.copy_from_slice(&[i as u8 * step, i as u8 * step, i as u8 * step, 255]);
                }
                buf
            })
            .collect()
    }

    #[test]
    fn a_hash_run_is_the_frame_number() {
        assert_eq!(
            expand_pattern("frame.####.jpg", 7).unwrap(),
            "frame.0007.jpg"
        );
        assert_eq!(expand_pattern("f.#.png", 12).unwrap(), "f.12.png");
        assert_eq!(expand_pattern("#/f.##.png", 3).unwrap(), "3/f.##.png"); // the first run
    }

    #[test]
    fn a_printf_field_is_the_frame_number_too() {
        assert_eq!(
            expand_pattern("frame.%04d.jpg", 7).unwrap(),
            "frame.0007.jpg"
        );
        assert_eq!(expand_pattern("f.%d.png", 7).unwrap(), "f.7.png");
        assert_eq!(expand_pattern("f.%x.png", 255).unwrap(), "f.ff.png");
        assert_eq!(expand_pattern("100%%.%02d.png", 7).unwrap(), "100%.07.png");
    }

    /// The pinned answer of `pattern % index`, spec by spec.
    #[test]
    fn every_conversion_matches_its_pinned_reference() {
        let specs = [
            "%d", "%i", "%u", "%x", "%X", "%o", "%s", "%4d", "%-4d", "%04d", "%+d", "% d", "%.3d",
            "%5.3d", "%08.3d", "%-8.3d", "%05s", "%.2s", "%+04d", "%ld",
        ];
        let want = [
            (
                7,
                [
                    "7", "7", "7", "7", "7", "7", "7", "   7", "7   ", "0007", "+7", " 7", "007",
                    "  007", "00000007", "007     ", "    7", "7", "+007", "7",
                ],
            ),
            (
                -7,
                [
                    "-7", "-7", "-7", "-7", "-7", "-7", "-7", "  -7", "-7  ", "-007", "-7", "-7",
                    "-007", " -007", "-0000007", "-007    ", "   -7", "-7", "-007", "-7",
                ],
            ),
            (
                0,
                [
                    "0", "0", "0", "0", "0", "0", "0", "   0", "0   ", "0000", "+0", " 0", "000",
                    "  000", "00000000", "000     ", "    0", "0", "+000", "0",
                ],
            ),
            (
                255,
                [
                    "255", "255", "255", "ff", "FF", "377", "255", " 255", "255 ", "0255", "+255",
                    " 255", "255", "  255", "00000255", "255     ", "  255", "25", "+255", "255",
                ],
            ),
        ];
        for (value, row) in want {
            for (spec, want) in specs.iter().zip(row) {
                assert_eq!(
                    percent_format(spec, value).unwrap(),
                    want,
                    "{spec} % {value}"
                );
            }
        }
    }

    #[test]
    fn a_plain_name_is_returned_as_it_was() {
        assert_eq!(expand_pattern("out.png", 4).unwrap(), "out.png");
        assert_eq!(expand_pattern("-", 4).unwrap(), "-");
    }

    #[test]
    fn a_malformed_field_carries_the_recorded_message() {
        for (pattern, msg) in [
            (
                "f.%q.png",
                "unsupported format character 'q' (0x71) at index 3",
            ),
            ("f.%d.%d.png", "not enough arguments for format string"),
            (
                "100%%.png",
                "not all arguments converted during string formatting",
            ),
            ("f.png%", "incomplete format"),
        ] {
            let e = expand_pattern(pattern, 7).unwrap_err().to_string();
            assert_eq!(e, msg);
        }
    }

    #[test]
    fn fixed_pacer_idle_period() {
        assert_eq!(FixedPacer::new(30.0).idle_period(), 1.0 / 30.0);
    }

    #[test]
    fn a_pacer_sleeps_out_the_rest_of_the_period() {
        let mut p = FixedPacer::new(50.0);
        assert!((p.sleep_for(0.001, 0.010) - 0.010).abs() < 1e-9);
        assert_eq!(p.sleep_for(0.001, 0.5), 0.0); // an overlong frame never sleeps
    }

    #[test]
    fn every_sink_carries_its_own_destination() {
        let dir = tmpdir("dest");
        assert_eq!(
            ImageSequence::new(&at(&dir, "d.####.png")).dest(),
            DEST_FILE
        );
        assert_eq!(RawStream::new(Vec::new()).dest(), DEST_PIPE);
        assert_eq!(Scanout.dest(), DEST_KMS);
    }

    #[test]
    fn the_scanout_never_takes_a_frame() {
        assert!(!Scanout.wants(0));
        assert!(Scanout.scanout() && Scanout.persists());
    }

    #[test]
    fn image_sequence_write_byte_identical() {
        let dir = tmpdir("identical");
        let want = frames(4, 25);
        let mut seq = ImageSequence::new(&at(&dir, "w.####.png"));
        for f in &want {
            seq.write(f.view()).unwrap();
        }
        seq.close().unwrap();
        for (i, f) in want.iter().enumerate() {
            let got = fs::read(dir.join(format!("w.{i:04}.png"))).unwrap();
            assert_eq!(got, encode_png(f.view()).unwrap());
        }
    }

    #[test]
    fn image_sequence_pool_recycles() {
        // a shallow depth recycles the buffers every couple of frames. Each
        // file still matches a synchronous save of that frame byte for byte.
        let dir = tmpdir("pool");
        let want = frames(10, 20);
        let mut seq = ImageSequence::with_opts(&at(&dir, "p.####.png"), 0, None, 2);
        for f in &want {
            let mut buf = seq.acquire(f.shape()).unwrap();
            buf.data.copy_from_slice(&f.data); // stands in for render_into(out=buf)
            seq.submit(buf).unwrap();
        }
        assert!(seq.allocated <= 3); // depth + 1 is the memory bound
        seq.close().unwrap();
        for (i, f) in want.iter().enumerate() {
            let got = fs::read(dir.join(format!("p.{i:04}.png"))).unwrap();
            assert_eq!(got, encode_png(f.view()).unwrap());
        }
    }

    #[test]
    fn worker_error_surfaces_on_write() {
        let dir = tmpdir("err-write").join("no-such-dir");
        let mut seq = ImageSequence::new(&at(&dir, "e.####.png"));
        let mut caught = None;
        for _ in 0..100 {
            if let Err(e) = seq.write(OwnedFrame::of((8, 6, 4)).view()) {
                caught = Some(e.to_string());
                break;
            }
            sleep(Duration::from_millis(2));
        }
        assert!(caught.is_some_and(|e| e.contains("No such file or directory")));
    }

    #[test]
    fn worker_error_surfaces_on_close() {
        let dir = tmpdir("err-close").join("no-such-dir");
        let mut seq = ImageSequence::new(&at(&dir, "e.####.png"));
        seq.write(OwnedFrame::of((8, 6, 4)).view()).unwrap();
        let e = seq.close().unwrap_err().to_string();
        assert!(e.contains("No such file or directory"), "{e}");
    }

    #[test]
    fn an_unknown_extension_is_refused_by_name() {
        let frame = OwnedFrame::of((2, 2, 3));
        let opts = Map::new();
        for (path, msg) in [
            ("out.tiff", "unknown file extension: .tiff"),
            ("out", "unknown file extension: "),
        ] {
            let e = encode(path, frame.view(), &opts).unwrap_err().to_string();
            assert_eq!(e, msg);
        }
    }

    #[test]
    fn a_jpeg_takes_its_quality_and_refuses_an_alpha_channel() {
        let rgb = OwnedFrame::of((16, 16, 3));
        let mut opts = Map::new();
        opts.insert("quality".to_string(), Value::from(20));
        let small = encode("f.jpg", rgb.view(), &opts).unwrap();
        opts.insert("quality".to_string(), Value::from(95));
        let large = encode("f.jpeg", rgb.view(), &opts).unwrap();
        assert!(small.len() < large.len());
        assert_eq!(&small[..2], &[0xff, 0xd8]); // SOI
        let rgba = OwnedFrame::of((16, 16, 4));
        let e = encode("f.jpg", rgba.view(), &opts).unwrap_err().to_string();
        assert_eq!(e, "cannot write mode RGBA as JPEG");
    }

    #[test]
    fn a_png_is_a_signature_ihdr_idat_and_iend() {
        let mut buf = OwnedFrame::of((3, 2, 3));
        buf.data.copy_from_slice(&[
            1, 2, 3, 4, 5, 6, // row 0
            7, 8, 9, 10, 11, 12, // row 1
            13, 14, 15, 16, 17, 18, // row 2
        ]);
        let png = encode_png(buf.view()).unwrap();
        assert_eq!(&png[..8], &[0x89, 80, 78, 71, 0x0d, 0x0a, 0x1a, 0x0a]);
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[16..24], &[0, 0, 0, 2, 0, 0, 0, 3]); // 2x3
        assert_eq!(&png[24..29], &[8, 2, 0, 0, 0]); // 8-bit rgb, no interlace
        assert_eq!(&png[png.len() - 8..], b"IEND\xae\x42\x60\x82");
        // every chunk's crc matches
        let mut i = 8;
        while i < png.len() {
            let len = u32::from_be_bytes(png[i..i + 4].try_into().unwrap()) as usize;
            let body = &png[i + 4..i + 8 + len];
            let want = u32::from_be_bytes(png[i + 8 + len..i + 12 + len].try_into().unwrap());
            assert_eq!(crc32(body), want);
            i += 12 + len;
        }
        assert_eq!(i, png.len());
    }

    #[test]
    fn a_png_over_64k_spans_several_stored_blocks() {
        // one stored deflate block holds 65535 bytes, so a large frame spans
        // several of them
        let mut buf = OwnedFrame::of((200, 200, 3));
        for (i, b) in buf.data.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let dir = tmpdir("large");
        let path = at(&dir, "large.png");
        fs::write(&path, encode_png(buf.view()).unwrap()).unwrap();
        let png = fs::read(&path).unwrap();
        let len = u32::from_be_bytes(png[33..37].try_into().unwrap()) as usize;
        let idat = &png[41..41 + len];
        assert_eq!(&idat[..2], &[0x78, 0x01]);
        let mut raw = Vec::new();
        let mut i = 2;
        loop {
            let last = idat[i] == 1;
            let n = u16::from_le_bytes(idat[i + 1..i + 3].try_into().unwrap()) as usize;
            assert_eq!(
                u16::from_le_bytes(idat[i + 3..i + 5].try_into().unwrap()),
                !(n as u16)
            );
            raw.extend_from_slice(&idat[i + 5..i + 5 + n]);
            i += 5 + n;
            if last {
                break;
            }
        }
        assert_eq!(i + 4, idat.len()); // the adler32 closes the stream
        assert_eq!(
            u32::from_be_bytes(idat[i..].try_into().unwrap()),
            adler32(&raw)
        );
        assert_eq!(raw.len(), 200 * (200 * 3 + 1));
        for (row, want) in raw.chunks(601).zip(buf.data.chunks(600)) {
            assert_eq!(row[0], 0); // filter: none
            assert_eq!(&row[1..], want);
        }
    }

    /// The pixels of a PNG that encode_png wrote: stored blocks, no filter.
    fn png_pixels(png: &[u8], stride: usize) -> Vec<u8> {
        let len = u32::from_be_bytes(png[33..37].try_into().unwrap()) as usize;
        let idat = &png[41..41 + len];
        let n = u16::from_le_bytes(idat[3..5].try_into().unwrap()) as usize;
        idat[7..2 + 5 + n]
            .chunks(stride + 1)
            .flat_map(|row| row[1..].to_vec())
            .collect()
    }

    #[test]
    fn a_png_holds_straight_alpha_and_a_raw_stream_the_frame_as_drawn() {
        let mut buf = OwnedFrame::of((1, 5, 4));
        let drawn = [
            10, 20, 30, 255, // opaque
            20, 10, 0, 40, // premultiplied by 40/255
            64, 128, 32, 128, // premultiplied by 128/255
            9, 9, 9, 0, // transparent
            90, 60, 30, 30, // more colour than alpha, as an add blend leaves it
        ];
        buf.data.copy_from_slice(&drawn);
        let png = encode_png(buf.view()).unwrap();
        assert_eq!(
            png_pixels(&png, 20),
            [10, 20, 30, 255, 128, 64, 0, 40, 128, 255, 64, 128, 0, 0, 0, 0, 255, 255, 255, 30]
        );
        let mut rs = RawStream::new(Vec::new());
        rs.write(buf.view()).unwrap();
        assert_eq!(rs.stream, drawn);
    }

    #[test]
    fn a_raw_stream_writes_the_frames_end_to_end() {
        let mut rs = RawStream::new(Vec::new());
        let one = OwnedFrame::of((4, 3, 4));
        for _ in 0..3 {
            rs.write(one.view()).unwrap();
        }
        assert_eq!(rs.stream.len(), 3 * 4 * 3 * 4); // no header, no padding
    }
}
