//! H264 output: the encoders for the frames that ghee renders, and the framing
//! that carries their stream to a file, a pipe or the console's player.
//!
//! [`make_encoder`] picks one of two encoders, and an [`Encoder`] holds
//! either:
//!
//! - [`HardwareH264`] encodes on the video block of a Pi 1 through a Pi 4,
//!   through the V4L2 memory-to-memory codec device (bcm2835-codec,
//!   /dev/video11). The GPU copies the final pass into a buffer that the codec
//!   reads straight from memory, so no frame passes through the CPU.
//! - [`SoftwareH264`] pipes the frames read back from the GPU into an ffmpeg
//!   process that encodes with libx264, on a Pi 5, which has no video encoder,
//!   and on a desktop.
//!
//! Both write Annex-B access units, the NAL units of a frame each behind a
//! start code, so `--h264`, the console's /h264 endpoint and mpv all take the
//! same bytes. [`H264Stream`] writes them to a file or a pipe. The console
//! sends them to a browser in [`wire_frame`]s, after the decoder config of
//! [`build_config`].

use std::ffi::CStr;
use std::io::{self, Read, Write};
use std::os::raw::{c_char, c_int};
use std::os::unix::fs::PermissionsExt;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::ptr;
use std::slice;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use serde::Serialize;

use crate::engine::Renderer;
use crate::ffi;
use crate::looper::{Frame, Output, RenderTarget};
use crate::stats::DEST_H264;

pub const NAL_SLICE: u8 = 1;
pub const NAL_IDR: u8 = 5;
pub const NAL_SEI: u8 = 6;
pub const NAL_SPS: u8 = 7;
pub const NAL_PPS: u8 = 8;
pub const NAL_AUD: u8 = 9;

const START_CODE: &[u8] = &[0x00, 0x00, 0x01];

/// WebCodecs VideoDecoderConfig.colorSpace.
#[derive(Clone, Copy, Serialize)]
pub struct ColorSpace {
    pub primaries: &'static str,
    pub transfer: &'static str,
    pub matrix: &'static str,
    #[serde(rename = "fullRange")]
    pub full_range: bool,
}

// The colour space that the console names to the browser's decoder for each
// encoder. The Pi's video block converts RGB to YUV with a fixed BT.601
// full-range matrix and writes no colour description into the stream, and a
// browser with no hint guesses BT.709 limited range. x264 tags its stream
// BT.601 limited range.
pub const HW_COLOR: ColorSpace = ColorSpace {
    primaries: "bt709",
    transfer: "iec61966-2-1",
    matrix: "smpte170m",
    full_range: true,
};
pub const SW_COLOR: ColorSpace = ColorSpace {
    primaries: "bt709",
    transfer: "bt709",
    matrix: "smpte170m",
    full_range: false,
};

/// The first index of `needle` in `hay` at or after `from`.
fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    let tail = hay.get(from..)?;
    tail.windows(needle.len())
        .position(|w| w == needle)
        .map(|i| i + from)
}

/// Split an Annex-B access unit into raw NAL units, without their start
/// codes. The split looks for the 3-byte start code 00 00 01, so the first 00
/// of a 4-byte start code lands at the end of the NAL unit before it. Each
/// NAL unit drops the zero bytes at its end, which removes that 00 and any
/// trailing_zero_8bits padding alike.
pub fn split_annexb(au: &[u8]) -> Vec<&[u8]> {
    let mut nals = Vec::new();
    let mut i = find(au, START_CODE, 0);
    while let Some(at) = i {
        let start = at + 3;
        let j = find(au, START_CODE, start);
        let end = j.unwrap_or(au.len());
        let nal = &au[start..end];
        let zeros = nal.iter().rev().take_while(|b| **b == 0).count();
        let nal = &nal[..nal.len() - zeros];
        if !nal.is_empty() {
            nals.push(nal);
        }
        i = j;
    }
    nals
}

/// Length-prefixed (AVCC) concatenation of NAL units, for VideoDecoder.
pub fn avcc(nals: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for nal in nals {
        out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
        out.extend_from_slice(nal);
    }
    out
}

#[derive(Serialize)]
struct DecoderConfig {
    codec: String,
    description: String,
    #[serde(rename = "codedWidth")]
    coded_width: i32,
    #[serde(rename = "codedHeight")]
    coded_height: i32,
    #[serde(rename = "colorSpace", skip_serializing_if = "Option::is_none")]
    color_space: Option<ColorSpace>,
}

/// WebCodecs decoder config JSON: codec string + avcC description.
pub fn build_config(sps: &[u8], pps: &[u8], w: i32, h: i32, color: Option<ColorSpace>) -> Vec<u8> {
    let mut avcc_box = vec![1, sps[1], sps[2], sps[3], 0xFF, 0xE1];
    avcc_box.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    avcc_box.extend_from_slice(sps);
    avcc_box.push(1);
    avcc_box.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    avcc_box.extend_from_slice(pps);
    let cfg = DecoderConfig {
        codec: format!("avc1.{:02x}{:02x}{:02x}", sps[1], sps[2], sps[3]),
        description: crate::assets::base64(&avcc_box),
        coded_width: w,
        coded_height: h,
        color_space: color,
    };
    serde_json::to_vec(&cfg).expect("decoder config serialises")
}

/// The wire's error type: the payload is the UTF-8 reason the stream ends.
pub const WIRE_ERROR: u8 = 3;

/// Browser wire framing: [u8 type][u32 BE len][payload].
pub fn wire_frame(t: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + payload.len());
    out.push(t);
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    out.extend_from_slice(payload);
    out
}

// bcm2835-codec refuses an input buffer under 64KiB ("buffer size mismatch
// sizeimage N < min size 65536" in dmesg). It reads 4 bytes a pixel.
pub const HW_MIN_PIXELS: i32 = 65536 / 4;

/// The size the hardware encoder takes a w x h frame at: the frame scaled up
/// by the smallest whole factor that reaches HW_MIN_PIXELS. The blit into the
/// encoder's buffer repeats each pixel as a square block.
pub fn hw_size(w: i32, h: i32) -> (i32, i32) {
    if w <= 0 || h <= 0 {
        return (w, h);
    }
    let mut k = 1;
    while i64::from(w * k) * i64::from(h * k) < i64::from(HW_MIN_PIXELS) {
        k += 1;
    }
    (w * k, h * k)
}

/// The reason a stream has no encoder, from the reason of each lane.
pub fn no_encoder(hardware: &str, software: &str) -> String {
    format!("no H264 encoder. hardware: {hardware}. software: {software}")
}

/// An access unit and its keyframe flag. None while nothing is ready.
pub type Au<'a> = (Option<&'a [u8]>, bool);

/// Zero-copy V4L2 encoder bound to a Renderer's GL context. Every method issues
/// GL calls, so every method runs on the renderer's thread. The encoder never
/// outlives its renderer.
pub struct HardwareH264 {
    scene: *mut ffi::GheeScene,
    /// The encoded size, hw_size() of the frame size.
    pub w: i32,
    pub h: i32,
    handle: *mut ffi::GheeEncoder,
}

impl HardwareH264 {
    /// An encoder for w x h frames. It encodes at hw_size(w, h).
    pub fn new(
        renderer: &Renderer,
        w: i32,
        h: i32,
        fps: i32,
        bitrate: i32,
    ) -> Result<HardwareH264> {
        let (w, h) = hw_size(w, h);
        let mut err: *mut c_char = ptr::null_mut();
        let handle =
            unsafe { ffi::ghee_encoder_create(renderer.handle(), w, h, fps, bitrate, &mut err) };
        if handle.is_null() {
            bail!("{}", err_text(err));
        }
        Ok(HardwareH264 {
            scene: renderer.handle(),
            w,
            h,
            handle,
        })
    }

    fn last_error(&self) -> String {
        let e = unsafe { ffi::ghee_last_error(self.scene) };
        if e.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
        }
    }

    fn collect(&self, rc: c_int, nal: *const u8, len: c_int, key: c_int) -> Result<Au<'_>> {
        if rc < 0 {
            bail!("hardware encode failed: {}", self.last_error());
        }
        if rc > 0 {
            return Ok((None, false)); // the pipeline primes: no output yet
        }
        Ok((
            Some(unsafe { slice::from_raw_parts(nal, len as usize) }),
            key != 0,
        ))
    }

    /// Encode the frame left by the last render() or present(). One render
    /// feeds the screen and the stream.
    pub fn encode_last(&mut self) -> Result<Au<'_>> {
        let mut nal: *const u8 = ptr::null();
        let (mut len, mut key): (c_int, c_int) = (0, 0);
        let rc = unsafe {
            ffi::ghee_encoder_encode_last(self.scene, self.handle, &mut nal, &mut len, &mut key)
        };
        self.collect(rc, nal, len, key)
    }

    pub fn force_keyframe(&mut self) {
        unsafe { ffi::ghee_encoder_force_keyframe(self.handle) };
    }

    /// One access unit still owed, None once the encoder reports drained.
    fn flush(&mut self) -> Result<Option<Vec<u8>>> {
        if self.handle.is_null() {
            return Ok(None);
        }
        let mut nal: *const u8 = ptr::null();
        let (mut len, mut key): (c_int, c_int) = (0, 0);
        let rc = unsafe {
            ffi::ghee_encoder_flush(self.scene, self.handle, &mut nal, &mut len, &mut key)
        };
        if rc < 0 {
            bail!("hardware flush failed: {}", self.last_error());
        }
        if rc > 0 {
            return Ok(None);
        }
        Ok(Some(
            unsafe { slice::from_raw_parts(nal, len as usize) }.to_vec(),
        ))
    }

    /// End of stream: the access units still owed, one per flush.
    /// It runs until the encoder reports drained. A capture that skips it ends
    /// mid-GOP and loses its last frames.
    pub fn drain(&mut self) -> HardwareDrain<'_> {
        HardwareDrain { enc: Some(self) }
    }

    pub fn close(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::ghee_encoder_destroy(self.scene, self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

impl Drop for HardwareH264 {
    fn drop(&mut self) {
        self.close();
    }
}

pub struct HardwareDrain<'a> {
    enc: Option<&'a mut HardwareH264>,
}

impl Iterator for HardwareDrain<'_> {
    type Item = Result<Vec<u8>>;

    fn next(&mut self) -> Option<Result<Vec<u8>>> {
        match self.enc.as_mut()?.flush() {
            Ok(Some(au)) => Some(Ok(au)),
            Ok(None) => {
                self.enc = None;
                None
            }
            Err(e) => {
                self.enc = None;
                Some(Err(e))
            }
        }
    }
}

/// The text of an engine error, or a generic one when the engine set none.
pub(crate) fn err_text(err: *mut c_char) -> String {
    if err.is_null() {
        "encoder create failed".to_string()
    } else {
        unsafe { CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned()
    }
}

/// True when PATH holds an executable file of that name.
fn on_path(program: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| {
        dir.join(program)
            .metadata()
            .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    })
}

/// How long SoftwareH264 waits for ffmpeg to drain and exit.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// ffmpeg/libx264 subprocess. feed() takes the readback frames. get_au()
/// returns complete access units. x264 runs with aud=1, so an AU boundary is
/// explicit.
pub struct SoftwareH264 {
    pub w: i32,
    pub h: i32,
    proc: Child,
    stdin: Option<ChildStdin>,
    rx: Receiver<Vec<u8>>,
}

impl SoftwareH264 {
    pub fn new(w: i32, h: i32, fps: i32, bitrate: i32, components: i32) -> Result<SoftwareH264> {
        if !on_path("ffmpeg") {
            bail!(
                "no ffmpeg on PATH (needed for software H264; \
                 install it with: sudo apt install ffmpeg)"
            );
        }
        let pix = if components == 4 { "rgba" } else { "rgb24" };
        let gop = fps * 2;
        let mut proc = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                pix,
                "-s",
                &format!("{w}x{h}"),
                "-r",
                &fps.to_string(),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-pix_fmt",
                "yuv420p", // 4:4:4 breaks WebCodecs and hardware decoders
                // pin the RGB->YUV matrix and range, then tag the stream to
                // match. A decoder guesses an untagged stream's colorimetry.
                "-vf",
                "scale=out_color_matrix=bt601:out_range=tv",
                "-colorspace",
                "smpte170m",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-color_range",
                "tv",
                "-x264-params",
                &format!("aud=1:keyint={gop}:min-keyint={gop}"),
                "-b:v",
                &bitrate.to_string(),
                "-f",
                "h264",
                "-",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let stdin = proc.stdin.take();
        let stdout = proc.stdout.take();
        let (tx, rx) = sync_channel(256);
        if let Some(stdout) = stdout {
            thread::spawn(move || pump(stdout, &tx));
        }
        Ok(SoftwareH264 {
            w,
            h,
            proc,
            stdin,
            rx,
        })
    }

    /// One raw frame, matching w/h/components.
    pub fn feed(&mut self, rgb: &[u8]) -> io::Result<()> {
        match self.stdin.as_mut() {
            Some(stdin) => stdin.write_all(rgb),
            None => Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "encoder input closed",
            )),
        }
    }

    /// (au or None, is_key). None means nothing arrives inside the timeout.
    pub fn get_au(&mut self, timeout: Duration) -> (Option<Vec<u8>>, bool) {
        let Ok(au) = self.rx.recv_timeout(timeout) else {
            return (None, false);
        };
        let is_key = split_annexb(&au).iter().any(|x| x[0] & 0x1F == NAL_IDR);
        (Some(au), is_key)
    }

    /// The pipe takes no mid-stream control. A viewer joins at the next GOP.
    #[allow(dead_code)]
    pub fn force_keyframe(&mut self) {}

    /// Close input and yield every remaining access unit.
    pub fn drain(&mut self, timeout: Duration) -> SoftwareDrain<'_> {
        self.stdin = None;
        SoftwareDrain {
            rx: &self.rx,
            timeout,
        }
    }

    pub fn close(&mut self, timeout: Duration) -> Result<()> {
        for _ in self.drain(timeout) {}
        // std has no wait-with-timeout: poll to the deadline, then kill.
        let deadline = Instant::now() + timeout;
        loop {
            if self.proc.try_wait()?.is_some() {
                return Ok(());
            }
            if Instant::now() >= deadline {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        self.proc.kill()?;
        self.proc.wait()?;
        Ok(())
    }
}

pub struct SoftwareDrain<'a> {
    rx: &'a Receiver<Vec<u8>>,
    timeout: Duration,
}

impl Iterator for SoftwareDrain<'_> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Vec<u8>> {
        self.rx.recv_timeout(self.timeout).ok()
    }
}

/// Split the raw stream into access units at AUD start codes.
fn pump(mut stdout: ChildStdout, tx: &SyncSender<Vec<u8>>) {
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let n = match stdout.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        buf.extend_from_slice(&chunk[..n]);
        while let Some(first) = find_aud(&buf, 0) {
            let Some(next) = find_aud(&buf, first + 4) else {
                break;
            };
            if tx.send(buf[first..next].to_vec()).is_err() {
                return;
            }
            buf.drain(..next);
        }
    }
    // the closing flush leaves one AU after EOF
    if !buf.is_empty() && find_aud(&buf, 0).is_some() {
        let _ = tx.send(buf);
    }
}

fn find_aud(buf: &[u8], start: usize) -> Option<usize> {
    let mut i = start;
    loop {
        let at = find(buf, START_CODE, i)?;
        if at + 3 >= buf.len() {
            return None;
        }
        if buf[at + 3] & 0x1F == NAL_AUD {
            return Some(at);
        }
        i = at + 3;
    }
}

pub enum Encoder {
    Hardware(HardwareH264),
    Software(SoftwareH264),
}

impl Encoder {
    #[allow(dead_code)]
    pub fn force_keyframe(&mut self) {
        match self {
            Encoder::Hardware(e) => e.force_keyframe(),
            Encoder::Software(e) => e.force_keyframe(),
        }
    }

    pub fn drain(&mut self) -> Drain<'_> {
        match self {
            Encoder::Hardware(e) => Drain::Hardware(e.drain()),
            Encoder::Software(e) => Drain::Software(e.drain(DRAIN_TIMEOUT)),
        }
    }

    pub fn close(&mut self) -> Result<()> {
        match self {
            Encoder::Hardware(e) => {
                e.close();
                Ok(())
            }
            Encoder::Software(e) => e.close(DRAIN_TIMEOUT),
        }
    }
}

pub enum Drain<'a> {
    Hardware(HardwareDrain<'a>),
    Software(SoftwareDrain<'a>),
}

impl Iterator for Drain<'_> {
    type Item = Result<Vec<u8>>;

    fn next(&mut self) -> Option<Result<Vec<u8>>> {
        match self {
            Drain::Hardware(d) => d.next(),
            Drain::Software(d) => d.next().map(Ok),
        }
    }
}

/// Hardware when the board has it, software otherwise. Fails only when neither
/// exists.
pub fn make_encoder(
    renderer: &Renderer,
    w: i32,
    h: i32,
    fps: i32,
    bitrate: i32,
    components: i32,
) -> Result<Encoder> {
    let hw_err = match HardwareH264::new(renderer, w, h, fps, bitrate) {
        Ok(enc) => return Ok(Encoder::Hardware(enc)),
        Err(e) => e,
    };
    match SoftwareH264::new(w, h, fps, bitrate, components) {
        Ok(enc) => Ok(Encoder::Software(enc)),
        Err(sw_err) => bail!(no_encoder(&hw_err.to_string(), &sw_err.to_string())),
    }
}

/// An Annex-B H264 stream as a sink: a file, or a pipe to mpv or ffmpeg. The
/// hardware lane encodes the frame just drawn and never reads it back.
/// The software lane takes the readback.
///
/// Every frame encodes. H264 delta coding makes a static frame cheap, so this
/// sink holds no last frame and skips no clean frame.
pub struct H264Stream {
    stream: Box<dyn Write + Send>,
    enc: Encoder,
    pub hw: bool,
    /// The encoded size.
    pub size: (i32, i32),
    timeout: Duration,
}

impl H264Stream {
    pub fn new(
        stream: Box<dyn Write + Send>,
        renderer: &Renderer,
        w: i32,
        h: i32,
        fps: i32,
        bitrate: i32,
        components: i32,
    ) -> Result<H264Stream> {
        let enc = make_encoder(renderer, w, h, fps, bitrate, components)?;
        let (hw, size) = match &enc {
            Encoder::Hardware(e) => (true, (e.w, e.h)),
            Encoder::Software(e) => (false, (e.w, e.h)),
        };
        Ok(H264Stream {
            stream,
            enc,
            hw,
            size,
            timeout: Duration::from_secs(2), // the first AU waits out the pipeline priming
        })
    }
}

/// True when the reader at the far end of the pipe is gone.
fn dropped(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset
    )
}

impl Output for H264Stream {
    fn dest(&self) -> &str {
        DEST_H264
    }

    fn wants(&self, _frame: u64) -> bool {
        !self.hw // the hardware encoder takes no readback: mirror() encodes from the GPU
    }

    fn write(&mut self, frame: Frame<'_>) -> Result<()> {
        let H264Stream {
            stream,
            enc,
            timeout,
            ..
        } = self;
        if let Encoder::Software(sw) = enc {
            sw.feed(frame.data)?;
            if let (Some(au), _) = sw.get_au(*timeout) {
                stream.write_all(&au)?;
                stream.flush()?;
            }
            *timeout = Duration::from_millis(500);
        }
        Ok(())
    }

    fn mirror(&mut self, _renderer: &dyn RenderTarget) -> Result<()> {
        let H264Stream { stream, enc, .. } = self;
        if let Encoder::Hardware(hw) = enc {
            if let (Some(au), _) = hw.encode_last()? {
                stream.write_all(au)?;
                stream.flush()?;
            }
        }
        Ok(())
    }

    /// Drain what the encoder still owes. The stream belongs to its opener.
    fn close(&mut self) -> Result<()> {
        let H264Stream { stream, enc, .. } = self;
        let mut wrote = Ok(());
        for au in enc.drain() {
            match au {
                Ok(au) => {
                    if let Err(e) = stream.write_all(&au) {
                        if !dropped(&e) {
                            wrote = Err(e.into());
                        }
                        break;
                    }
                }
                Err(e) => {
                    wrote = Err(e);
                    break;
                }
            }
        }
        if wrote.is_ok() {
            if let Err(e) = stream.flush() {
                if !dropped(&e) {
                    wrote = Err(e.into());
                }
            }
        }
        let closed = enc.close();
        wrote.and(closed)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::{Mutex, MutexGuard};

    use serde_json::Value;

    use super::*;
    use crate::engine::probe_dri_devices;
    use crate::testing::NEVER;

    const SC: &[u8] = &[0x00, 0x00, 0x00, 0x01];

    /// One GL device, one Renderer at a time.
    static GPU: Mutex<()> = Mutex::new(());

    /// The renderer these tests share.
    fn gpu() -> (MutexGuard<'static, ()>, Renderer) {
        let guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let mut errs = Vec::new();
        for dev in probe_dri_devices() {
            match Renderer::builder(64, 48).device(&dev).build() {
                Ok(r) => return (guard, r),
                Err(e) => errs.push(format!("{dev}: {e}")),
            }
        }
        panic!(
            "no GL device opens ({}). A host without one runs the tests under \
             RUSTFLAGS=\"--cfg ghee_no_gpu\", which reports them as ignored",
            errs.join("; ")
        );
    }

    /// Fails the test where ffmpeg or ffprobe is missing.
    fn need_ffmpeg() {
        assert!(
            on_path("ffmpeg") && on_path("ffprobe"),
            "needs ffmpeg and ffprobe on PATH. A host without them runs the tests \
             under RUSTFLAGS=\"--cfg ghee_no_ffmpeg\", which reports these as ignored"
        );
    }

    #[test]
    fn split_annexb_basic() {
        let (sps, pps, idr): (&[u8], &[u8], &[u8]) = (
            &[0x67, 0x42, 0x00, 0x1e],
            &[0x68, 0xce, 0x06, 0xe2],
            &[0x65, 0x88, 0x80],
        );
        let au = [SC, sps, START_CODE, pps, SC, idr].concat();
        let nals = split_annexb(&au);
        assert_eq!(nals, vec![sps, pps, idr]);
        assert_eq!(
            nals.iter().map(|x| x[0] & 0x1F).collect::<Vec<u8>>(),
            vec![NAL_SPS, 8, NAL_IDR]
        );
    }

    #[test]
    fn split_annexb_garbage_prefix_and_padding() {
        let idr: &[u8] = &[0x65, 0x01, 0x02];
        let au = [&[0xff, 0xfe][..], SC, idr, &[0x00, 0x00]].concat();
        assert_eq!(split_annexb(&au), vec![idr]);
        assert!(split_annexb(b"no start code at all").is_empty());
    }

    #[test]
    fn avcc_length_prefixes() {
        let out = avcc(&[&[0x65, 0x01], &[0x41, 0x02, 0x03]]);
        assert_eq!(
            out,
            b"\x00\x00\x00\x02\x65\x01\x00\x00\x00\x03\x41\x02\x03".to_vec()
        );
    }

    #[test]
    fn build_config_codec_string() {
        let sps: &[u8] = &[0x67, 0x42, 0xC0, 0x1E, 0xAA];
        let pps: &[u8] = &[0x68, 0xCE, 0x06, 0xE2];
        let cfg: Value = serde_json::from_slice(&build_config(sps, pps, 320, 240, None)).unwrap();
        assert_eq!(cfg["codec"], "avc1.42c01e");
        assert_eq!(cfg["codedWidth"], 320);
        assert_eq!(cfg["codedHeight"], 240);
        let raw = crate::assets::unbase64(cfg["description"].as_str().unwrap()).unwrap();
        assert_eq!(raw[0], 1); // avcC version
        assert!(find(&raw, sps, 0).is_some() && find(&raw, pps, 0).is_some());
    }

    #[test]
    fn build_config_colorspace() {
        let sps: &[u8] = &[0x67, 0x42, 0xC0, 0x1E, 0xAA];
        let pps: &[u8] = &[0x68, 0xCE, 0x06, 0xE2];
        let plain: Value = serde_json::from_slice(&build_config(sps, pps, 320, 240, None)).unwrap();
        assert!(plain.get("colorSpace").is_none());
        let hinted: Value =
            serde_json::from_slice(&build_config(sps, pps, 320, 240, Some(HW_COLOR))).unwrap();
        assert_eq!(hinted["colorSpace"]["matrix"], "smpte170m");
        assert_eq!(hinted["colorSpace"]["fullRange"], true);
    }

    #[test]
    fn wire_frame_framing() {
        let f = wire_frame(1, b"abc");
        assert_eq!(f[0], 1);
        assert_eq!(u32::from_be_bytes(f[1..5].try_into().unwrap()), 3);
        assert_eq!(&f[5..], b"abc");
    }

    /// The wire's config payload, as a fresh /h264 viewer parses it.
    #[test]
    fn the_config_payload_is_a_decoder_config() {
        let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1e];
        let pps: &[u8] = &[0x68, 0xce, 0x06, 0xe2];
        let msg = wire_frame(0, &build_config(sps, pps, 320, 240, None));
        let (t, n, payload) = (
            msg[0],
            u32::from_be_bytes(msg[1..5].try_into().unwrap()),
            &msg[5..],
        );
        assert_eq!(t, 0);
        assert_eq!(n as usize, payload.len());
        let cfg: Value = serde_json::from_slice(payload).unwrap();
        for key in ["codec", "description", "codedWidth", "codedHeight"] {
            assert!(cfg.get(key).is_some(), "{key}");
        }
        assert_eq!(
            (cfg["codedWidth"].as_i64(), cfg["codedHeight"].as_i64()),
            (Some(320), Some(240))
        );
    }

    /// A keyframe access unit carries its own SPS/PPS. A viewer's config and
    /// its first frame come from the same bytes.
    #[test]
    fn a_keyframe_access_unit_carries_its_parameter_sets() {
        let au = [
            SC,
            &[0x67, 0x42, 0x00, 0x1e],
            SC,
            &[0x68, 0xce, 0x06, 0xe2],
            SC,
            &[0x65, 0x88, 0x80],
        ]
        .concat();
        let nals = split_annexb(&au);
        let types: Vec<u8> = nals.iter().map(|x| x[0] & 0x1F).collect();
        assert_eq!(types, vec![NAL_SPS, NAL_PPS, NAL_IDR]);
        assert!(types.contains(&NAL_IDR));
        let delta = [SC, &[0x41, 0x88, 0x80]].concat();
        assert!(!split_annexb(&delta).iter().any(|x| x[0] & 0x1F == NAL_IDR));
    }

    fn gradient_frames(w: i32, h: i32, n: i32) -> Vec<Vec<u8>> {
        (0..n)
            .map(|i| {
                let green = (255 * i / 1.max(n - 1)) as u8;
                let mut f = vec![0u8; (w * h * 4) as usize];
                for px in f.chunks_mut(4) {
                    px[1] = green; // green ramps over time
                    px[3] = 255;
                }
                f
            })
            .collect()
    }

    /// Feed `input` to a program's stdin, take its stdout.
    fn run(program: &str, args: &[&str], input: &[u8]) -> Vec<u8> {
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let data = input.to_vec();
        let writer = thread::spawn(move || {
            let _ = stdin.write_all(&data);
        });
        let out = child.wait_with_output().unwrap();
        writer.join().unwrap();
        out.stdout
    }

    fn ffprobe(entries: &str, stream: &[u8]) -> String {
        let out = run(
            "ffprobe",
            &[
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                entries,
                "-of",
                "csv=p=0",
                "-",
            ],
            stream,
        );
        String::from_utf8_lossy(&out).trim().to_string()
    }

    #[test]
    #[cfg_attr(ghee_no_ffmpeg, ignore = "needs ffmpeg")]
    fn software_roundtrip_decodes() {
        need_ffmpeg();
        let (w, h, n) = (160, 120, 30);
        let mut enc = SoftwareH264::new(w, h, 30, 1_000_000, 4).unwrap();
        for f in gradient_frames(w, h, n) {
            enc.feed(&f).unwrap();
        }
        let aus: Vec<Vec<u8>> = enc.drain(NEVER).collect();
        enc.close(DRAIN_TIMEOUT).unwrap();
        let stream = aus.concat();
        // zerolatency gives one AU per frame once flushed
        assert_eq!(aus.len(), n as usize);
        // the first AU carries the config and a keyframe
        let types0: Vec<u8> = split_annexb(&aus[0]).iter().map(|x| x[0] & 0x1F).collect();
        assert!(types0.contains(&NAL_SPS) && types0.contains(&NAL_IDR));
        assert!(aus
            .iter()
            .all(|au| split_annexb(au)[0][0] & 0x1F == NAL_AUD));
        // the whole stream decodes back to n frames at the right size
        assert_eq!(
            ffprobe("stream=width,height,nb_read_frames", &stream),
            format!("{w},{h},{n}")
        );
    }

    #[test]
    #[cfg_attr(ghee_no_ffmpeg, ignore = "needs ffmpeg")]
    fn software_stream_is_tagged() {
        need_ffmpeg();
        let mut enc = SoftwareH264::new(160, 120, 30, 1_000_000, 4).unwrap();
        let mut f = vec![0u8; 160 * 120 * 4];
        for px in f.chunks_mut(4) {
            px[3] = 255;
        }
        for _ in 0..5 {
            enc.feed(&f).unwrap();
        }
        let stream = enc.drain(NEVER).collect::<Vec<_>>().concat();
        enc.close(DRAIN_TIMEOUT).unwrap();
        assert_eq!(
            ffprobe("stream=color_space,color_range", &stream),
            "tv,smpte170m"
        );
    }

    #[test]
    #[cfg_attr(ghee_no_ffmpeg, ignore = "needs ffmpeg")]
    fn software_red_stays_red() {
        need_ffmpeg();
        // green hides channel swaps and matrix errors, red catches them
        let (w, h) = (160usize, 120usize);
        let mut enc = SoftwareH264::new(w as i32, h as i32, 30, 2_000_000, 4).unwrap();
        let mut f = vec![0u8; w * h * 4];
        for px in f.chunks_mut(4) {
            px[0] = 255;
            px[3] = 255;
        }
        for _ in 0..10 {
            enc.feed(&f).unwrap();
        }
        let stream = enc.drain(NEVER).collect::<Vec<_>>().concat();
        enc.close(DRAIN_TIMEOUT).unwrap();
        let raw = run(
            "ffmpeg",
            &[
                "-v",
                "error",
                "-i",
                "-",
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            &stream,
        );
        let last = &raw[raw.len() - w * h * 3..];
        let mut mean = [0f64; 3];
        for px in last.chunks(3) {
            for (c, m) in mean.iter_mut().enumerate() {
                *m += f64::from(px[c]);
            }
        }
        let n = (w * h) as f64;
        let mean = [mean[0] / n, mean[1] / n, mean[2] / n];
        assert!(
            mean[0] > 200.0 && mean[1] < 50.0 && mean[2] < 50.0,
            "{mean:?}"
        );
    }

    #[test]
    #[cfg_attr(ghee_no_ffmpeg, ignore = "needs ffmpeg")]
    fn software_content_survives() {
        need_ffmpeg();
        // a brightness ramp in and a brightness ramp out: the pixels reach
        // the codec
        let (w, h, n) = (160usize, 120usize, 20usize);
        let mut enc = SoftwareH264::new(w as i32, h as i32, 30, 2_000_000, 4).unwrap();
        for f in gradient_frames(w as i32, h as i32, n as i32) {
            enc.feed(&f).unwrap();
        }
        let stream = enc.drain(NEVER).collect::<Vec<_>>().concat();
        enc.close(DRAIN_TIMEOUT).unwrap();
        let raw = run(
            "ffmpeg",
            &[
                "-v", "error", "-i", "-", "-f", "rawvideo", "-pix_fmt", "gray", "-",
            ],
            &stream,
        );
        assert_eq!(raw.len(), w * h * n);
        let mean = |f: &[u8]| f.iter().map(|b| f64::from(*b)).sum::<f64>() / (w * h) as f64;
        assert!(mean(&raw[..w * h]) < 60.0 && mean(&raw[raw.len() - w * h..]) > 120.0);
    }

    #[test]
    fn a_frame_under_the_codec_minimum_encodes_at_a_whole_multiple() {
        assert_eq!(hw_size(128, 32), (256, 64)); // exactly the minimum
        assert_eq!(hw_size(100, 40), (300, 120)); // x2 is 16000 pixels, short of it
        assert_eq!(hw_size(64, 48), (192, 144));
        assert_eq!(hw_size(1, 1), (128, 128));
        assert_eq!(hw_size(128, 128), (128, 128));
        assert_eq!(hw_size(320, 240), (320, 240));
        for (w, h) in [(128, 32), (100, 40), (7, 3), (2, 2000)] {
            let (ew, eh) = hw_size(w, h);
            let k = ew / w;
            assert_eq!((ew, eh), (w * k, h * k));
            assert!(ew * eh >= HW_MIN_PIXELS);
            assert!(k == 1 || (w * (k - 1)) * (h * (k - 1)) < HW_MIN_PIXELS);
        }
    }

    #[test]
    #[cfg_attr(
        any(ghee_no_gpu, not(ghee_hw_h264)),
        ignore = "needs a GL device and a hardware H264 encoder"
    )]
    fn a_tiny_frame_reaches_the_hardware_encoder_scaled_up() {
        assert!(Path::new("/dev/video11").exists(), "needs /dev/video11");
        let (_g, mut r) = gpu();
        let scene = crate::scene::Scene::from_dict(
            &serde_json::json!({"version": 1, "passes": [{"name": "main",
                "fragment": "void main(){ gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }"}]}),
            None,
            None,
        )
        .unwrap();
        r.setup(&scene).unwrap();
        let mut enc = HardwareH264::new(&r, 64, 48, 30, 1_000_000).unwrap();
        assert_eq!((enc.w, enc.h), (192, 144));
        let mut key = None;
        for _ in 0..30 {
            r.draw(None).unwrap();
            if let (Some(au), true) = enc.encode_last().unwrap() {
                key = Some(au.to_vec());
                break;
            }
        }
        let key = key.expect("no keyframe in 30 frames");
        let types: Vec<u8> = split_annexb(&key).iter().map(|x| x[0] & 0x1F).collect();
        assert!(
            types.contains(&NAL_SPS) && types.contains(&NAL_IDR),
            "{types:?}"
        );
        enc.close();
    }

    #[test]
    #[cfg_attr(
        any(ghee_no_gpu, not(ghee_hw_h264)),
        ignore = "needs a GL device and a hardware H264 encoder"
    )]
    fn a_hardware_jpeg_of_any_width_keeps_red_black_and_white() {
        let (_g, mut r) = gpu();
        let scene = crate::scene::Scene::from_dict(
            &serde_json::json!({"version": 1, "passes": [{"name": "main",
                "fragment": "void main(){ gl_FragColor = uv.x < 0.5 ? vec4(1.0, 0.0, 0.0, 1.0) \
                             : vec4(vec3(step(0.5, uv.y)), 1.0); }"}]}),
            None,
            None,
        )
        .unwrap();
        r.setup(&scene).unwrap();
        // rows of 72 pixels, 288 bytes, are no multiple of 64 bytes
        let mut enc = crate::output::HardwareJpeg::new(&r, 72, 48, 75).unwrap();
        let mut jpg = None;
        for _ in 0..10 {
            r.draw(None).unwrap();
            if let Some(j) = enc.encode_last().unwrap() {
                jpg = Some(j.to_vec());
                break;
            }
        }
        let jpg = jpg.expect("no JPEG in 10 frames");
        enc.close();
        let mut d = jpeg_decoder::Decoder::new(&jpg[..]);
        let px = d.decode().unwrap();
        let info = d.info().unwrap();
        assert_eq!((info.width, info.height), (72, 48));
        // each probe sits inside a 16 x 16 block of one colour, y down
        let at = |x: usize, y: usize| px[(y * 72 + x) * 3..][..3].to_vec();
        let near =
            |got: Vec<u8>, want: [u8; 3]| got.iter().zip(want).all(|(g, w)| g.abs_diff(w) <= 6);
        assert!(near(at(16, 24), [255, 0, 0]), "red {:?}", at(16, 24));
        assert!(near(at(48, 8), [255, 255, 255]), "white {:?}", at(48, 8));
        assert!(near(at(48, 40), [0, 0, 0]), "black {:?}", at(48, 40));
    }

    #[test]
    #[cfg_attr(
        any(ghee_no_gpu, ghee_no_ffmpeg, ghee_hw_h264),
        ignore = "needs a GL device, ffmpeg and a host with no hardware H264 encoder"
    )]
    fn hardware_unavailable_falls_back() {
        assert!(
            !Path::new("/dev/video11").exists(),
            "this host has /dev/video11, so it runs the tests under \
             RUSTFLAGS=\"--cfg ghee_hw_h264\""
        );
        need_ffmpeg();
        let (_g, r) = gpu();
        assert!(HardwareH264::new(&r, 64, 64, 30, 4_000_000).is_err());
        let mut enc = make_encoder(&r, 64, 64, 30, 4_000_000, 4).unwrap();
        assert!(matches!(enc, Encoder::Software(_)));
        enc.close().unwrap();
    }
}
