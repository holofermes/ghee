//! ghee console render server.
//!
//! This attaches the console to a render loop (looper). The console owns the
//! active scene, the scene list and the command queue the loop drains. It serves
//! the JSON API and the compiled-in web assets. The loop belongs to the caller,
//! so a render runs without a browser, and a browser costs the caller no panel.
//!
//! API:
//!   GET  /stream[?vid=]     MJPEG, one part per rendered frame
//!   GET  /h264[?vid=]       framed H264 for WebCodecs ([u8 type][u32 len]:
//!                           0=config JSON, 1=key AU, 2=delta AU, AVCC). One
//!                           shared encoder feeds every viewer. Hardware
//!                           zero-copy needs /dev/video11, else ffmpeg/libx264.
//!   POST /ping?vid=         heartbeat for the streams opened under that vid
//!   GET  /frame.png         one lossless PNG of the next frame, at the render
//!                           size. The request rides the command queue, so the
//!                           frame carries every /set that returned before it.
//!                           409 while the scene holds an error. 503 when the
//!                           loop draws nothing.
//!   GET  /stats             {fps, frame_ms, stream_fps, out, error, error_line,
//!                           error_pass, active, display, stale}
//!   GET  /scenes            {scenes:[name...], active}
//!   GET  /filters           {name: filter def} registry for the "+ filter" picker
//!   GET  /scene?name=       resolved scene JSON (shader files inlined, filters
//!                           expanded). The active name answers with the document
//!                           that renders now, applied edits and /set values
//!                           included. Any other name answers from disk.
//!   POST /set?pass=&name=&value=   live uniform, no recompile (pass=scene = scene-scoped)
//!   POST /next /prev        switch active scene
//!   POST /switch?name=      switch to a named scene. The active name reloads it,
//!                           which is how a stale scene reads its files again.
//!   POST /scene             full scene JSON -> recompile active (apply working copy)
//!   POST /save              write the scene as a modified copy in <scenes>/modified/
//!   GET  /<file>            static file from the compiled-in web assets
//!
//! One Console holds this module's state. The render loop, the encoder threads
//! and the request handlers share it. run() also parks it in a process global,
//! so stop() ends a console from a signal handler.
//!
//! run() draws on the thread that called it. It accepts requests on a thread of
//! its own. A panel driver binds its SPI and GPIO handles to one thread. The
//! loop must run where the caller built its sinks.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::io::Write;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use serde_json::{Map, Value};
use tiny_http::{Header, Request, Server, StatusCode};

use crate::console::core::{
    parse_query, ConsoleAPI, ConsoleError, ConsoleHost, ConsoleResult, Headers, Query, Response,
    Scope,
};
use crate::console::examples_dir;
use crate::console::feeds::default_hwinfo;
use crate::console::store::SceneStore;
use crate::engine::{ErrorPass, Feed, Provider, Renderer};
use crate::feeds::demo_feed;
use crate::h264::{
    avcc, build_config, split_annexb, wire_frame, ColorSpace, HardwareH264, SoftwareH264, HW_COLOR,
    NAL_IDR, NAL_PPS, NAL_SEI, NAL_SLICE, NAL_SPS, SW_COLOR,
};
use crate::looper::{ControllerSlot, Frame, Loop, Output, RenderTarget, SceneController};
use crate::output::{encode_jpeg, encode_png, Scanout};
use crate::scene::{ArraySize, Param, ParamValue, Scene};
use crate::sources::{collect, Snapshot};
use crate::stats::{monotonic, FrameRate, DEST_H264, DEST_KMS, DEST_MJPEG};

/// JPEG quality for the MJPEG preview
pub const PREVIEW_QUALITY: u8 = 75;

/// The producer and a viewer run at the same rate, so a backlog never drains on
/// its own. Past this many queued frames the fanout drops the client, and the
/// client resyncs at the next keyframe.
pub const H264_LAG_FRAMES: usize = 10;

/// The bits per pixel per frame the stream aims for.
pub const H264_BPP: f64 = 0.15;
pub const H264_BITRATE_CEILING: i64 = 80_000_000;
/// The software encode cadence. Hardware follows the display or the loop.
pub const H264_FPS: i32 = 30;

/// How long GET /frame.png waits for the render loop to answer its request.
pub const FRAME_TIMEOUT: f64 = 4.0;

/// A page opens /h264?vid=<id> or /stream?vid=<id>. It repeats POST
/// /ping?vid=<id> while it watches. Consumption proves nothing: a page frozen in
/// a browser's back/forward cache stays connected and keeps draining its socket.
/// Only the ping stopping marks a departure. The deadline runs on its own clock,
/// so a congested link never drops a viewer.
///
/// The cadence the console page pings at.
#[allow(dead_code)]
pub const PING_INTERVAL: f64 = 3.0;
/// Three intervals, so two lost pings are not a departure.
pub const VIEWER_DEADLINE: f64 = 10.0;

/// run() sets this. stop() ends that console from anywhere.
static CONSOLE: Mutex<Option<Arc<Console>>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The render-queue item for a /set. Scope::Scene targets the scene's shared
/// params, and the loop applies that item to every pass. Anything else names a
/// pass index.
pub fn set_item(scope: Scope, name: &str, value: Value) -> Command {
    match scope {
        Scope::Scene => Command::SceneParam(name.to_string(), value),
        Scope::Pass(i) => Command::Param(i, name.to_string(), value),
    }
}

/// The param's declared type wins over the wire type. An int rounds. A bool
/// becomes a real bool. A color or a vec arrives as a list and passes through.
pub fn coerce(ptype: &str, val: &Value) -> Result<ParamValue> {
    let built = if val.is_array() {
        Param::new("v", ptype, val.clone(), None, None, None, None)
    } else if ptype == "int" {
        Param::new(
            "v",
            ptype,
            Value::from(round_half_even(as_f64(val)) as i64),
            None,
            None,
            None,
            None,
        )
    } else if ptype == "bool" {
        Param::new("v", ptype, Value::from(truthy(val)), None, None, None, None)
    } else {
        Param::new("v", ptype, val.clone(), None, None, None, None)
    };
    Ok(built?.value)
}

fn as_f64(v: &Value) -> f64 {
    match v {
        Value::Bool(b) => f64::from(u8::from(*b)),
        _ => v.as_f64().unwrap_or_default(),
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Null => false,
        _ => v.as_f64().unwrap_or_default() != 0.0,
    }
}

/// Round a float. A half goes to the even side.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// What the render thread drains off the command queue.
pub enum Command {
    Switch(String),
    Param(i64, String, Value),
    SceneParam(String, Value),
    /// GET /frame.png: arm the still where this item lands in queue order.
    Shot,
    /// A working copy to compile onto the running renderer.
    SceneObj(Box<Scene>),
    /// Draw one frame, for a stream that just opened or closed.
    Preview,
    Stop,
}

/// The command queue the render loop drains.
#[derive(Default)]
pub struct Commands {
    items: Mutex<VecDeque<Command>>,
    ready: Condvar,
}

impl Commands {
    fn put(&self, item: Command) {
        lock(&self.items).push_back(item);
        self.ready.notify_all();
    }

    fn take(&self) -> Option<Command> {
        lock(&self.items).pop_front()
    }

    fn is_empty(&self) -> bool {
        lock(&self.items).is_empty()
    }

    /// Sleep until a command arrives. The command stays on the queue.
    fn wait(&self) {
        let mut items = lock(&self.items);
        while items.is_empty() {
            items = self
                .ready
                .wait(items)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

/// One readback from the render thread, row-major (height, width, components).
pub struct RawFrame {
    pub data: Vec<u8>,
    pub width: i32,
    pub height: i32,
    pub components: i32,
}

impl RawFrame {
    fn of(frame: Frame<'_>) -> RawFrame {
        RawFrame {
            data: frame.data.to_vec(),
            width: frame.width,
            height: frame.height,
            components: frame.components,
        }
    }

    fn view(&self) -> Frame<'_> {
        Frame {
            data: &self.data,
            width: self.width,
            height: self.height,
            components: self.components,
        }
    }
}

/// The latest preview frame handed to the encoder thread.
#[derive(Default)]
struct Raw {
    rgb: Option<Arc<RawFrame>>,
    seq: u64,
}

/// The latest JPEG the console serves to the MJPEG viewers.
#[derive(Default)]
struct Latest {
    jpg: Option<Arc<Vec<u8>>>,
    seq: u64,
}

/// What /stats publishes about the running scene, minus the rates.
#[derive(Default)]
struct Stat {
    frame_ms: f64,
    error: String,
    active: Option<String>,
    display: Value,
    error_line: Option<i64>,
    error_pass: Option<ErrorPass>,
}

/// A source for one declared array: the host's data for this frame.
pub type ArraySource = Box<dyn Fn(ArraySize, u64) -> Vec<f32> + Send + Sync>;

/// This module's state. The render loop and the request threads share it.
pub struct Console {
    // configuration: run() sets it before the threads start
    pub store: SceneStore,
    pub web: bool,
    pub w: i32,
    pub h: i32,
    /// Scan out to a connected KMS display as well.
    pub display: bool,
    /// Scan out in 16-bit color. This halves the GPU store and the scanout bytes.
    pub rgb565: bool,
    /// The longest side of the browser preview. The encoder thread downscales it.
    pub preview_max: i32,
    /// The browser preview refresh cap. The display and the panel stay at full
    /// speed.
    pub preview_fps: f64,
    /// The cap on the h264 stream's longest side. 0 means full size.
    pub stream_max: i32,
    pub bitrate: i64,
    /// The frame components the renderer reads back. A panel that packs RGB asks
    /// for 3.
    pub components: i32,
    /// 0 renders until stop(). A count ends the console with the render.
    pub frames: u64,
    /// Pin engine time to frame * this, for a reproducible render.
    pub timestep: Option<f64>,
    /// The DRI device override. None probes for one.
    pub dri: Option<String>,
    /// The /stats destinations of the caller's sinks. A sink itself belongs to
    /// the render thread.
    pub sink_dests: Vec<String>,
    /// A panel's own pixel format. None takes the readback's.
    pub native_format: Option<String>,
    /// The interval a stale check reads disk at. None takes the default window.
    pub stale_interval: Option<f64>,
    arrays: Vec<(String, ArraySource)>,
    hwinfo: Box<dyn Fn() -> Value + Send + Sync>,

    // state
    stat: Mutex<Stat>,
    /// The document that renders now, under the name it loaded as.
    running: Mutex<Option<(Option<String>, Value)>>,
    /// The files the running scene was built from. Only a load off disk replaces
    /// this. An applied working copy leaves the scene stale when its files moved
    /// on.
    sources: Mutex<Option<Snapshot>>,
    raw: Mutex<Raw>,
    raw_ready: Condvar,
    latest: Mutex<Latest>,
    latest_ready: Condvar,
    nclients: Mutex<i64>,
    q: Commands,
    /// The loop iterations that drew.
    frame_rate: Arc<Mutex<FrameRate>>,
    /// The frames handed to viewers, whichever codec.
    stream_rate: Arc<Mutex<FrameRate>>,
    pub fanout: H264Fanout,
    pub shot: Arc<Shot>,
    pub viewers: Heartbeat,
    stopping: Arc<AtomicBool>,
    /// A failure the console cannot render through. run() returns it.
    fatal: Mutex<String>,
    /// Hardware encoding failed for this stream, so the software lane owns it.
    /// The last viewer to leave clears it. A stall or a busy codec belongs to
    /// the stream it happened in, and the software lane costs a readback.
    h264_hw_dead: AtomicBool,
    httpd: Mutex<Option<Arc<Server>>>,
}

impl Console {
    /// A console over a scene store, with the defaults run() overwrites.
    pub fn new(store: SceneStore) -> Console {
        let stopping = Arc::new(AtomicBool::new(false));
        let stream_rate = Arc::new(Mutex::new(FrameRate::new()));
        Console {
            store,
            web: true,
            w: 320,
            h: 240,
            display: false,
            rgb565: false,
            preview_max: 480,
            preview_fps: 15.0,
            stream_max: 0,
            bitrate: 10_000_000,
            components: 4,
            frames: 0,
            timestep: None,
            dri: None,
            sink_dests: Vec::new(),
            native_format: None,
            stale_interval: None,
            arrays: Vec::new(),
            hwinfo: Box::new(default_hwinfo),
            stat: Mutex::new(Stat {
                display: Value::Object(Map::new()),
                ..Stat::default()
            }),
            running: Mutex::new(None),
            sources: Mutex::new(None),
            raw: Mutex::new(Raw::default()),
            raw_ready: Condvar::new(),
            latest: Mutex::new(Latest::default()),
            latest_ready: Condvar::new(),
            nclients: Mutex::new(0),
            q: Commands::default(),
            frame_rate: Arc::new(Mutex::new(FrameRate::new())),
            fanout: H264Fanout::new(Arc::clone(&stream_rate)),
            stream_rate,
            shot: Shot::new(Arc::clone(&stopping)),
            viewers: Heartbeat::new(VIEWER_DEADLINE),
            stopping,
            fatal: Mutex::new(String::new()),
            h264_hw_dead: AtomicBool::new(false),
            httpd: Mutex::new(None),
        }
    }

    pub fn stopping(&self) -> bool {
        self.stopping.load(Ordering::SeqCst)
    }

    /// The MJPEG viewer count. A handler thread changes it. The render and
    /// encoder threads read it.
    pub fn count_clients(&self, delta: i64) -> i64 {
        let mut n = lock(&self.nclients);
        *n += delta;
        *n
    }

    pub fn nclients(&self) -> i64 {
        *lock(&self.nclients)
    }

    /// Where the frames go now: the scanout, the caller's sinks, and each stream
    /// with a viewer.
    fn out(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        if self.display {
            out.push(DEST_KMS.to_string());
        }
        out.extend(self.sink_dests.iter().cloned());
        if self.fanout.nclients() > 0 {
            out.push(DEST_H264.to_string());
        }
        if self.nclients() > 0 {
            out.push(DEST_MJPEG.to_string());
        }
        out
    }

    /// The pixel format the frames end in. A panel's own wins over the
    /// readback's.
    fn format(&self) -> String {
        self.native_format
            .clone()
            .unwrap_or_else(|| "rgba".to_string())
    }

    /// The /stats payload. Each rate computes on read, so an idle loop's number
    /// falls with the clock.
    pub fn stats(&self) -> Value {
        let mut d = Map::new();
        {
            let stat = lock(&self.stat);
            d.insert("frame_ms".to_string(), Value::from(stat.frame_ms));
            d.insert("error".to_string(), Value::from(stat.error.clone()));
            d.insert(
                "active".to_string(),
                stat.active.clone().map_or(Value::Null, Value::from),
            );
            d.insert("display".to_string(), stat.display.clone());
            d.insert(
                "error_line".to_string(),
                stat.error_line.map_or(Value::Null, Value::from),
            );
            d.insert(
                "error_pass".to_string(),
                stat.error_pass.map_or(Value::Null, ErrorPass::to_json),
            );
        }
        let stale = lock(&self.sources).as_mut().is_some_and(Snapshot::stale);
        d.insert(
            "fps".to_string(),
            Value::from(lock(&self.frame_rate).value(None)),
        );
        d.insert(
            "stream_fps".to_string(),
            Value::from(lock(&self.stream_rate).value(None)),
        );
        d.insert("out".to_string(), Value::from(self.out()));
        d.insert("stale".to_string(), Value::from(stale));
        Value::Object(d)
    }

    /// Drop what the scene before published: the rate windows, the last frame's
    /// cost, the compile error.
    fn reset_stats(&self) {
        lock(&self.frame_rate).reset(None);
        lock(&self.stream_rate).reset(None);
        let mut stat = lock(&self.stat);
        stat.frame_ms = 0.0;
        stat.error = String::new();
        stat.error_line = None;
        stat.error_pass = None;
    }

    /// Republish the running document from the scene the loop holds. The render
    /// thread calls this after a load, an applied scene that compiled, or a /set
    /// that landed.
    fn publish(&self, scene: &Scene) {
        let active = lock(&self.stat).active.clone();
        *lock(&self.running) = Some((active, scene.to_dict()));
    }

    fn snapshot(&self, paths: impl IntoIterator<Item = impl AsRef<std::path::Path>>) -> Snapshot {
        match self.stale_interval {
            None => Snapshot::new(paths),
            Some(interval) => Snapshot::with(paths, interval, Box::new(monotonic)),
        }
    }

    /// Queue an item that changes what renders. The queue takes it before the
    /// count, so a still joins only a frame that carries it.
    fn command(&self, item: Command) {
        self.q.put(item);
        self.shot.touch();
    }

    /// Software H264 reads back the same frames as the MJPEG preview, so its
    /// viewers count as readback consumers. Hardware samples the GPU texture
    /// instead.
    fn software_h264(&self) -> bool {
        self.h264_hw_dead.load(Ordering::SeqCst) && self.fanout.nclients() > 0
    }

    /// Shut the server down, and signal the render loop and the encode threads.
    /// run() calls it on the way out. A host that runs the console in a thread
    /// calls it to end one. It waits for no thread, because a request handler
    /// and a signal handler both call it. It ends run()'s own loop, so run()
    /// closes its sinks before it returns.
    pub fn stop(&self) {
        self.stopping.store(true, Ordering::SeqCst);
        self.q.put(Command::Stop); // wake a render loop parked on the command queue
        self.raw_ready.notify_all();
        self.latest_ready.notify_all();
        let httpd = lock(&self.httpd).take();
        if let Some(server) = httpd {
            server.unblock();
        }
    }

    /// Record a failure the console cannot render through, and stop. run()
    /// returns it, instead of leaving a server with no renderer.
    fn fail(&self, msg: &str) {
        *lock(&self.fatal) = msg.to_string();
        lock(&self.stat).error = msg.to_string();
        eprintln!("ghee console: {msg}");
        self.stop();
    }
}

/// Shut down the console run() started.
pub fn stop() {
    let console = lock(&CONSOLE).clone();
    if let Some(console) = console {
        console.stop();
    }
}

// ---------------------------------------------------------------- preview

/// Encode the preview JPEGs, off the render thread. The render loop drops its
/// readback here. This downscales only a frame another consumer sized, so the
/// display and the panel never wait. Raw frames also feed the h264 path, but a
/// JPEG needs an attached viewer.
pub fn encoder_loop(console: &Console) {
    let mut last_seq = u64::MAX;
    while !console.stopping() {
        let rgb = {
            let mut raw = lock(&console.raw);
            if console.nclients() == 0 || raw.seq == last_seq {
                raw = console
                    .raw_ready
                    .wait_timeout(raw, Duration::from_millis(250))
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
            }
            let fresh = console.nclients() > 0 && raw.seq != last_seq;
            let rgb = if fresh { raw.rgb.clone() } else { None };
            last_seq = raw.seq;
            rgb
        };
        let Some(rgb) = rgb else { continue };
        let (w, h) = (rgb.width, rgb.height);
        let mut pixels = to_rgb(&rgb);
        let scale = f64::from(console.preview_max) / f64::from(w.max(h));
        let (mut pw, mut ph) = (w, h);
        if scale < 1.0 {
            // the frame is sized for the software H264 stream, or for a bigger host
            pw = 1.max(round_half_even(f64::from(w) * scale) as i32);
            ph = 1.max(round_half_even(f64::from(h) * scale) as i32);
            pixels = resize_rgb(&pixels, w, h, pw, ph);
        }
        let frame = Frame {
            data: &pixels,
            width: pw,
            height: ph,
            components: 3,
        };
        match encode_jpeg(frame, PREVIEW_QUALITY) {
            Ok(jpg) => {
                let mut latest = lock(&console.latest);
                latest.jpg = Some(Arc::new(jpg));
                latest.seq += 1;
                console.latest_ready.notify_all();
            }
            Err(e) => eprintln!("preview: {e}"),
        }
        lock(&console.stream_rate).tick(None);
    }
}

/// The RGB bytes of an RGB(A) readback, which is what a JPEG needs.
fn to_rgb(frame: &RawFrame) -> Vec<u8> {
    if frame.components == 3 {
        return frame.data.clone();
    }
    frame
        .data
        .chunks(frame.components as usize)
        .flat_map(|px| px.iter().take(3).copied())
        .collect()
}

/// Box-average downscale of an RGB frame.
fn resize_rgb(rgb: &[u8], w: i32, h: i32, nw: i32, nh: i32) -> Vec<u8> {
    let (w, h, nw, nh) = (w as usize, h as usize, nw as usize, nh as usize);
    let mut out = vec![0u8; nw * nh * 3];
    for y in 0..nh {
        let y0 = y * h / nh;
        let y1 = (((y + 1) * h).div_ceil(nh)).max(y0 + 1).min(h);
        for x in 0..nw {
            let x0 = x * w / nw;
            let x1 = (((x + 1) * w).div_ceil(nw)).max(x0 + 1).min(w);
            let mut sums = [0u32; 3];
            let mut n = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let i = (sy * w + sx) * 3;
                    for (c, sum) in sums.iter_mut().enumerate() {
                        *sum += u32::from(rgb[i + c]);
                    }
                    n += 1;
                }
            }
            let o = (y * nw + x) * 3;
            for c in 0..3 {
                out[o + c] = (sums[c] / n.max(1)) as u8;
            }
        }
    }
    out
}

// ---------------------------------------------------------------- h264

/// One viewer's queue of wire-framed access units.
pub struct H264Client {
    state: Mutex<ClientQueue>,
    ready: Condvar,
}

#[derive(Default)]
struct ClientQueue {
    q: VecDeque<Arc<Vec<u8>>>,
    started: bool,
}

impl H264Client {
    fn new() -> H264Client {
        H264Client {
            state: Mutex::new(ClientQueue::default()),
            ready: Condvar::new(),
        }
    }

    /// The fanout wire-frames frame and config once, and every client shares
    /// them.
    fn offer(&self, frame: &Arc<Vec<u8>>, config: Option<&Arc<Vec<u8>>>) {
        let mut state = lock(&self.state);
        if state.started && state.q.len() >= H264_LAG_FRAMES {
            state.started = false; // lagging: drop the rest of this GOP
            state.q.clear();
        }
        if !state.started {
            if let Some(config) = config {
                state.q.push_back(Arc::clone(config));
                state.q.push_back(Arc::clone(frame));
                state.started = true;
            }
            self.ready.notify_all();
            return;
        }
        if state.q.len() >= 32 {
            state.started = false; // backstop: the lag check above fires first
        } else {
            state.q.push_back(Arc::clone(frame));
        }
        self.ready.notify_all();
    }

    /// The next blob for this viewer, or None when the wait runs out.
    pub fn get(&self, timeout: Duration) -> Option<Arc<Vec<u8>>> {
        let mut state = lock(&self.state);
        if state.q.is_empty() {
            state = self
                .ready
                .wait_timeout(state, timeout)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        state.q.pop_front()
    }
}

/// One shared H264 stream for every /h264 viewer. A new viewer starts at a
/// keyframe: hardware forces one on join, and software waits for the next GOP.
///
/// rate counts the pushes. A host that serves another codec as well passes the
/// counter it publishes as stream_fps, so one number covers every viewer.
pub struct H264Fanout {
    clients: Mutex<Vec<Arc<H264Client>>>,
    cfg: Mutex<Option<Vec<u8>>>,
    /// The encoded dimensions. Whoever starts the encoder sets them.
    size: Mutex<(i32, i32)>,
    /// The decoder colorSpace hint (HW_COLOR or SW_COLOR). Same owner as size.
    color: Mutex<Option<ColorSpace>>,
    want_key: AtomicBool,
    rate: Arc<Mutex<FrameRate>>,
}

impl H264Fanout {
    pub fn new(rate: Arc<Mutex<FrameRate>>) -> H264Fanout {
        H264Fanout {
            clients: Mutex::new(Vec::new()),
            cfg: Mutex::new(None),
            size: Mutex::new((0, 0)),
            color: Mutex::new(None),
            want_key: AtomicBool::new(false),
            rate,
        }
    }

    pub fn nclients(&self) -> usize {
        lock(&self.clients).len()
    }

    #[allow(dead_code)]
    pub fn stream_fps(&self) -> f64 {
        lock(&self.rate).value(None)
    }

    pub fn add(&self) -> Arc<H264Client> {
        let c = Arc::new(H264Client::new());
        lock(&self.clients).push(Arc::clone(&c));
        self.want_key.store(true, Ordering::SeqCst);
        c
    }

    pub fn remove(&self, c: &Arc<H264Client>) {
        lock(&self.clients).retain(|x| !Arc::ptr_eq(x, c));
    }

    pub fn push(&self, au: &[u8]) {
        let nals = split_annexb(au);
        let sps = nals.iter().find(|x| x[0] & 0x1F == NAL_SPS);
        let pps = nals.iter().find(|x| x[0] & 0x1F == NAL_PPS);
        if let (Some(sps), Some(pps)) = (sps, pps) {
            let (w, h) = *lock(&self.size);
            *lock(&self.cfg) = Some(build_config(sps, pps, w, h, *lock(&self.color)));
        }
        // Keep the VCL and SEI NALs. The config carries the parameter sets.
        let keep: Vec<&[u8]> = nals
            .iter()
            .filter(|x| matches!(x[0] & 0x1F, NAL_SLICE | NAL_IDR | NAL_SEI))
            .copied()
            .collect();
        let chunk = avcc(&keep);
        if chunk.is_empty() {
            return;
        }
        let is_key = nals.iter().any(|x| x[0] & 0x1F == NAL_IDR);
        let frame = Arc::new(wire_frame(u8::from(!is_key) + 1, &chunk));
        let config = match (is_key, lock(&self.cfg).as_ref()) {
            (true, Some(cfg)) => Some(Arc::new(wire_frame(0, cfg))),
            _ => None,
        };
        let clients = lock(&self.clients);
        for c in clients.iter() {
            c.offer(&frame, config.as_ref());
        }
        if !clients.is_empty() {
            lock(&self.rate).tick(None);
        }
        // A waiting client rejoins at the next keyframe. Keep asking for one
        // until every client streams.
        if clients.iter().any(|c| !lock(&c.state).started) {
            self.want_key.store(true, Ordering::SeqCst);
        }
    }
}

/// The stream dimensions: fit the longest side and keep the aspect. yuv420 needs
/// even dimensions.
pub fn stream_size(w: i32, h: i32, longest: i32) -> (i32, i32) {
    let s = if longest > 0 {
        (f64::from(longest) / f64::from(w.max(h))).min(1.0)
    } else {
        1.0
    };
    let even = |v: i32| 2.max((round_half_even(f64::from(v) * s / 2.0) as i32) * 2);
    (even(w), even(h))
}

pub fn stream_bitrate(w: i32, h: i32, fps: i32, floor: i64) -> i64 {
    let derived = f64::from(w) * f64::from(h) * f64::from(fps.max(1)) * H264_BPP;
    floor.max((derived as i64).min(H264_BITRATE_CEILING))
}

/// Software H264, for a board with no /dev/video11. It encodes the same readback
/// frames as the MJPEG preview, off the render thread.
pub fn h264_sw_loop(console: &Console) {
    let mut enc: Option<SoftwareH264> = None;
    let mut size = (0, 0);
    let mut last_seq = u64::MAX;
    while !console.stopping() {
        if !console.software_h264() {
            if let Some(mut e) = enc.take() {
                let _ = e.close(Duration::from_secs(1));
            }
            let raw = lock(&console.raw);
            let _ = console
                .raw_ready
                .wait_timeout(raw, Duration::from_millis(250));
            continue;
        }
        let rgb = {
            let mut raw = lock(&console.raw);
            if raw.seq == last_seq {
                raw = console
                    .raw_ready
                    .wait_timeout(raw, Duration::from_millis(250))
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
            }
            let rgb = if raw.seq == last_seq {
                None
            } else {
                raw.rgb.clone()
            };
            last_seq = raw.seq;
            rgb
        };
        let Some(rgb) = rgb else { continue };
        let (w, h) = (rgb.width, rgb.height);
        if enc.is_some() && size != (w, h) {
            if let Some(mut e) = enc.take() {
                let _ = e.close(Duration::from_secs(1));
            }
        }
        if enc.is_none() {
            match SoftwareH264::new(
                w,
                h,
                H264_FPS,
                stream_bitrate(w, h, H264_FPS, console.bitrate) as i32,
                rgb.components,
            ) {
                Ok(e) => {
                    enc = Some(e);
                    size = (w, h);
                    *lock(&console.fanout.size) = (w, h);
                    *lock(&console.fanout.color) = Some(SW_COLOR);
                }
                Err(e) => {
                    eprintln!("h264: software encoder unavailable: {e}");
                    thread::sleep(Duration::from_secs(5));
                    continue;
                }
            }
        }
        let e = enc.as_mut().expect("the encoder was just created");
        if let Err(err) = e.feed(&rgb.data) {
            // a dead encoder process must not kill this thread
            eprintln!("h264: software encoder died ({err}); respawning");
            if let Some(mut e) = enc.take() {
                let _ = e.close(Duration::from_secs(1));
            }
            continue;
        }
        if let (Some(au), _) = e.get_au(Duration::from_secs(1)) {
            console.fanout.push(&au);
        }
    }
    if let Some(mut e) = enc.take() {
        let _ = e.close(Duration::from_secs(1));
    }
}

// ---------------------------------------------------------------- sinks

/// How long a clean scene sleeps before it polls again. Whoever consumes frames
/// sets it: the display's refresh, else the preview cap, else the last measured
/// render period. None means nobody consumes frames, so block on the command
/// queue instead.
pub fn skip_interval(
    display_hz: i32,
    nclients: i64,
    preview_fps: f64,
    render_period: f64,
) -> Option<f64> {
    if display_hz > 0 {
        return Some(1.0 / f64::from(display_hz));
    }
    if nclients <= 0 {
        return None;
    }
    if preview_fps > 0.0 {
        return Some(1.0 / preview_fps);
    }
    Some(render_period)
}

/// One whole frame for a client that asks for a picture (GET /frame.png).
///
/// A request is a generation. request() hands one out, and the handler queues a
/// token. The loop arms the shot where that token lands, and the next frame
/// answers every generation asked for so far. A /set that returned before the
/// request drains before the arm, so that frame carries it. A request made after
/// the arm waits for the following draw.
///
/// A request made while another one waits joins it, so overlapping requests cost
/// one frame. A command queued since the arm ends that sharing. The frame in
/// flight is one command behind what the caller already saw land, so the request
/// asks anew.
pub struct Shot {
    state: Mutex<ShotState>,
    published: Condvar,
    stopping: Arc<AtomicBool>,
}

#[derive(Default)]
struct ShotState {
    /// The generations handed out.
    asked: u64,
    /// The generations the next frame answers.
    armed: u64,
    /// The generations the published frame answers.
    done: u64,
    /// The requests that wait for an answer now.
    holders: i64,
    /// A command arrived after the arm.
    moved: bool,
    frame: Option<Arc<RawFrame>>,
}

impl Shot {
    pub fn new(stopping: Arc<AtomicBool>) -> Arc<Shot> {
        Arc::new(Shot {
            state: Mutex::new(ShotState::default()),
            published: Condvar::new(),
            stopping,
        })
    }

    pub fn request(&self) -> u64 {
        let mut s = lock(&self.state);
        if s.holders > 0 && !s.moved {
            return s.asked;
        }
        s.asked += 1;
        s.asked
    }

    /// Count a request until it has its answer. A request made meanwhile shares
    /// its frame.
    pub fn hold(&self) {
        lock(&self.state).holders += 1;
    }

    pub fn release(&self) {
        lock(&self.state).holders -= 1;
    }

    /// A command is on the queue, so the frame in flight predates it.
    pub fn touch(&self) {
        lock(&self.state).moved = true;
    }

    fn arm(&self) {
        let mut s = lock(&self.state);
        s.armed = s.asked;
        s.moved = false;
    }

    /// The frame that answers this generation, or None when none arrives in
    /// time. A shutdown wakes the wait, because nothing more publishes.
    pub fn wait(&self, gen: u64, timeout: f64) -> Option<Arc<RawFrame>> {
        let end = monotonic() + timeout;
        let mut s = lock(&self.state);
        while s.done < gen {
            let left = end - monotonic();
            if left <= 0.0 || self.stopping.load(Ordering::SeqCst) {
                return None;
            }
            s = self
                .published
                .wait_timeout(s, Duration::from_secs_f64(left.min(0.25)))
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        s.frame.clone()
    }

    pub fn wants(&self) -> bool {
        let s = lock(&self.state);
        s.armed > s.done
    }

    fn write(&self, frame: Arc<RawFrame>) {
        let mut s = lock(&self.state);
        s.frame = Some(frame);
        s.done = s.armed;
        self.published.notify_all();
    }
}

/// The shot as an output sink.
struct FrameShot {
    shot: Arc<Shot>,
}

impl Output for FrameShot {
    fn dest(&self) -> &str {
        "" // a still is not a destination; see Console::out
    }

    fn persists(&self) -> bool {
        true // nothing here waits on a frame, so a clean scene may skip
    }

    fn wants(&self, _frame: u64) -> bool {
        self.shot.wants()
    }

    fn readback(&self) -> Option<(i32, i32)> {
        None // the render size, read back directly
    }

    fn write(&mut self, frame: Frame<'_>) -> Result<()> {
        self.shot.write(Arc::new(RawFrame::of(frame)));
        Ok(())
    }
}

/// The console as an output sink. The browser preview and the software H264
/// encoder share one throttled, downscaled readback, and the hardware H264 mirror
/// runs here too. This reads the console's viewer counts every frame, so what it
/// asks the loop for follows the viewers.
struct ConsoleSink {
    console: Arc<Console>,
    renderer: Rc<RefCell<Renderer>>,
    /// The hardware encoder. The render thread owns it, because it issues GL
    /// calls.
    enc: Option<HardwareH264>,
    /// When the preview last refreshed.
    last: Cell<f64>,
    now: Cell<f64>,
    /// The display refresh. A hardware encoder runs at this rate.
    hz: i32,
}

impl Output for ConsoleSink {
    fn dest(&self) -> &str {
        "" // the console's destinations are its viewers; see Console::out
    }

    fn persists(&self) -> bool {
        true // a browser holds the last frame it got
    }

    fn wants(&self, _frame: u64) -> bool {
        // Refresh the browser preview at preview_fps at most, and only while a
        // viewer watches. The display and the panel still render every frame.
        // The software encoder has its own cadence (H264_FPS). The fastest
        // active consumer wins.
        if self.console.shot.wants() {
            // The still reads this frame back whole. Taking it here resizes the
            // software encoder to the render size and back again.
            return false;
        }
        let mut rates: Vec<f64> = Vec::new();
        if self.console.nclients() > 0 {
            rates.push(self.console.preview_fps); // <= 0 means every frame
        }
        if self.console.software_h264() {
            rates.push(f64::from(H264_FPS));
        }
        self.now.set(monotonic());
        let fastest = rates.iter().copied().fold(f64::MIN, f64::max);
        !rates.is_empty()
            && (rates.iter().any(|r| *r <= 0.0)
                || (self.now.get() - self.last.get()) >= 1.0 / fastest)
    }

    fn readback(&self) -> Option<(i32, i32)> {
        // One readback feeds both consumers, so whoever is attached sizes it.
        // The MJPEG preview downscales a bigger frame. The software encoder
        // cannot recover a smaller one.
        let (w, h) = (self.console.w, self.console.h);
        let scale = (f64::from(self.console.preview_max) / f64::from(w.max(h))).min(1.0);
        let pw = 1.max(round_half_even(f64::from(w) * scale) as i32);
        let ph = 1.max(round_half_even(f64::from(h) * scale) as i32);
        if self.console.software_h264() {
            let (sw, sh) = stream_size(w, h, self.console.stream_max);
            return Some((pw.max(sw), ph.max(sh)));
        }
        Some((pw, ph))
    }

    fn write(&mut self, frame: Frame<'_>) -> Result<()> {
        self.last.set(self.now.get());
        let mut raw = lock(&self.console.raw);
        raw.rgb = Some(Arc::new(RawFrame::of(frame)));
        raw.seq += 1;
        self.console.raw_ready.notify_all();
        Ok(())
    }

    /// The hardware H264 mirror: encode the frame just drawn, with no readback.
    /// The engine's keep_last holds the final pass in its FBO while the encoder
    /// exists. encode_last always has a texture to sample.
    fn mirror(&mut self, _renderer: &dyn RenderTarget) -> Result<()> {
        let console = Arc::clone(&self.console);
        if console.fanout.nclients() == 0 {
            console.h264_hw_dead.store(false, Ordering::SeqCst); // the next stream tries hardware again
            if let Some(mut enc) = self.enc.take() {
                enc.close(); // direct scanout returns while nobody watches
            }
            return Ok(());
        }
        if self.enc.is_none() && !console.h264_hw_dead.load(Ordering::SeqCst) {
            let (w, h) = stream_size(console.w, console.h, console.stream_max);
            let hz = if self.hz > 0 { self.hz } else { 30 };
            let bitrate = stream_bitrate(w, h, hz, console.bitrate) as i32;
            match HardwareH264::new(&self.renderer.borrow(), w, h, hz, bitrate) {
                Ok(enc) => {
                    self.enc = Some(enc);
                    *lock(&console.fanout.size) = (w, h);
                    *lock(&console.fanout.color) = Some(HW_COLOR);
                }
                Err(e) => {
                    // h264_sw_loop takes this stream over
                    console.h264_hw_dead.store(true, Ordering::SeqCst);
                    eprintln!("h264: {e}; using software encoder");
                }
            }
        }
        let Some(enc) = self.enc.as_mut() else {
            return Ok(());
        };
        if console.fanout.want_key.swap(false, Ordering::SeqCst) {
            enc.force_keyframe();
        }
        match enc.encode_last() {
            Ok((Some(au), _)) => console.fanout.push(au),
            Ok((None, _)) => {}
            Err(e) => {
                let mut enc = self.enc.take().expect("the encoder is still here");
                enc.close();
                console.h264_hw_dead.store(true, Ordering::SeqCst);
                eprintln!("h264: {e}; using software encoder");
            }
        }
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        if let Some(mut enc) = self.enc.take() {
            enc.close();
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- render loop

/// The renderer the loop drives. The console sink samples it for the hardware
/// encoder. Both live on the render thread, because GL is thread-affine.
struct SharedRenderer(Rc<RefCell<Renderer>>);

impl RenderTarget for SharedRenderer {
    fn width(&self) -> i32 {
        self.0.borrow().width
    }

    fn height(&self) -> i32 {
        self.0.borrow().height
    }

    fn components(&self) -> i32 {
        self.0.borrow().components
    }

    fn set_dt(&self, value: f64) {
        self.0.borrow().set_dt(value);
    }

    fn set_time(&self, t: f64) {
        self.0.borrow().set_time(Some(t));
    }

    fn set_param(
        &self,
        pass: Option<&str>,
        name: &str,
        ty: &str,
        value: &ParamValue,
    ) -> Result<bool> {
        RenderTarget::set_param(&*self.0.borrow(), pass, name, ty, value)
    }

    fn draw(&mut self, provider: Option<Provider<'_>>) -> Result<()> {
        self.0.borrow_mut().draw(provider)
    }

    fn render_into(
        &mut self,
        provider: Option<Provider<'_>>,
        out: &mut [u8],
        out_size: Option<(i32, i32)>,
    ) -> Result<()> {
        self.0.borrow_mut().render_into(provider, out, out_size)
    }

    fn present(
        &mut self,
        provider: Option<Provider<'_>>,
        preview: Option<(&mut [u8], (i32, i32))>,
    ) -> Result<()> {
        self.0.borrow_mut().present(provider, preview)
    }
}

/// Create the render-thread Renderer. It probes the DRI devices, so the console
/// runs on a Pi (card0) and on desktop Mesa (renderD128). components defaults to
/// 4, because desktop Mesa refuses an RGB glReadPixels. A panel sink asks for 3,
/// which is what it packs from.
fn make_renderer(console: &Console) -> Result<Renderer> {
    if console.display {
        let dev = console
            .dri
            .clone()
            .or_else(|| std::env::var("GHEE_DRI").ok().filter(|d| !d.is_empty()))
            .unwrap_or_else(|| "/dev/dri/card0".to_string());
        return Renderer::new(
            console.w,
            console.h,
            console.components,
            Some(&dev),
            true,
            console.rgb565,
        );
    }
    let devices = match &console.dri {
        Some(d) => vec![d.clone()],
        None => crate::engine::probe_dri_devices(),
    };
    let mut errs: Vec<String> = Vec::new();
    for dev in devices {
        match Renderer::new(
            console.w,
            console.h,
            console.components,
            Some(&dev),
            false,
            false,
        ) {
            Ok(r) => return Ok(r),
            Err(e) => errs.push(format!("{dev}: {e}")),
        }
    }
    bail!("no GL device for the console:\n{}", errs.join("\n"))
}

/// What the render thread holds between commands.
#[derive(Default)]
struct Running {
    scene: Option<Scene>,
}

/// Returns the compile error, or "" when this scene renders now.
fn set_scene(
    console: &Console,
    renderer: &RefCell<Renderer>,
    st: &RefCell<Running>,
    scene: Scene,
    name: Option<&str>,
    snap: Option<Snapshot>,
) -> String {
    let err = match renderer.borrow_mut().setup(&scene) {
        Ok(err) => err,
        Err(e) => e.to_string(),
    };
    if !err.is_empty() {
        {
            let mut stat = lock(&console.stat);
            stat.error = err.clone();
            stat.error_line = renderer.borrow().error_line;
            stat.error_pass = renderer.borrow().error_pass;
        }
        if let Some(last) = st.borrow().scene.as_ref() {
            let _ = renderer.borrow_mut().setup(last); // keep the last good scene rendering
        }
        return err;
    }
    st.borrow_mut().scene = Some(scene);
    console.reset_stats();
    if let Some(name) = name {
        lock(&console.stat).active = Some(name.to_string());
    }
    // Store the snapshot before the publish. A /stats between the two must not
    // read the new scene against the snapshot before it.
    if let Some(snap) = snap {
        *lock(&console.sources) = Some(snap);
    }
    let held = st.borrow();
    if let Some(scene) = held.scene.as_ref() {
        console.publish(scene);
    }
    String::new()
}

/// Load a scene by name, and put what feeds it in the controller slot. The last
/// scene's process stops first, so two procedural scenes never overlap.
fn load(
    console: &Console,
    renderer: &RefCell<Renderer>,
    st: &RefCell<Running>,
    ctrl: &ControllerSlot,
    name: &str,
) {
    *ctrl.borrow_mut() = None;
    let (built, read) = collect(|| console.store.start(name));
    match built {
        Ok((scene, process)) => {
            let snap = console.snapshot(read);
            let err = set_scene(console, renderer, st, scene, Some(name), Some(snap));
            if err.is_empty() {
                *ctrl.borrow_mut() = process.map(|f| Box::new(f) as Box<dyn SceneController>);
            }
        }
        Err(e) => lock(&console.stat).error = format!("load {name}: {e}"),
    }
}

/// Apply every queued command, and report whether one landed. A queued command
/// forces a draw this iteration, so an edit always shows.
fn drain(
    console: &Console,
    renderer: &RefCell<Renderer>,
    st: &RefCell<Running>,
    ctrl: &ControllerSlot,
) -> bool {
    let mut force = false;
    let mut edited = false; // a /set landed, so the running document must follow
    while !console.q.is_empty() {
        let Some(item) = console.q.take() else { break };
        force = true;
        match item {
            Command::Switch(name) => load(console, renderer, st, ctrl, &name),
            Command::Param(pi, name, value) => {
                if set_pass_param(renderer, st, ctrl, pi, &name, &value) {
                    edited = true;
                }
            }
            Command::SceneParam(name, value) => {
                if set_scene_param(renderer, st, ctrl, &name, &value) {
                    edited = true;
                }
            }
            // in queue order, so the armed frame carries a /set ahead of it
            Command::Shot => console.shot.arm(),
            Command::SceneObj(scene) => {
                set_scene(console, renderer, st, *scene, None, None);
            }
            Command::Preview | Command::Stop => {}
        }
    }
    if edited {
        let held = st.borrow();
        if let Some(scene) = held.scene.as_ref() {
            console.publish(scene);
        }
    }
    force
}

fn set_pass_param(
    renderer: &RefCell<Renderer>,
    st: &RefCell<Running>,
    ctrl: &ControllerSlot,
    pi: i64,
    name: &str,
    value: &Value,
) -> bool {
    let mut held = st.borrow_mut();
    let Some(scene) = held.scene.as_mut() else {
        return false;
    };
    let applied = || -> Result<()> {
        let pass = scene
            .passes
            .get_mut(usize::try_from(pi).unwrap_or(usize::MAX))
            .ok_or_else(|| anyhow!("list index out of range"))?;
        let ty = pass
            .params
            .get(name)
            .ok_or_else(|| anyhow!("'{name}'"))?
            .ty
            .clone();
        let v = coerce(&ty, value)?;
        if ty != "string" {
            renderer.borrow().set_param(&pass.name, name, &ty, &v)?;
        }
        told(ctrl, name, &v);
        pass.params
            .get_mut(name)
            .ok_or_else(|| anyhow!("'{name}'"))?
            .value = v;
        Ok(())
    }();
    match applied {
        Ok(()) => true,
        Err(e) => {
            eprintln!("set param {name}: {e}");
            false
        }
    }
}

/// Tell the scene's process about a string param the console changed. The
/// process reads it on the next tick.
fn told(ctrl: &ControllerSlot, name: &str, value: &ParamValue) {
    let ParamValue::Str(text) = value else {
        return;
    };
    if let Some(c) = ctrl.borrow_mut().as_mut() {
        c.on_param(name, text);
    }
}

fn set_scene_param(
    renderer: &RefCell<Renderer>,
    st: &RefCell<Running>,
    ctrl: &ControllerSlot,
    name: &str,
    value: &Value,
) -> bool {
    let mut held = st.borrow_mut();
    let Some(scene) = held.scene.as_mut() else {
        return false;
    };
    let applied = || -> Result<()> {
        let ty = scene
            .params
            .get(name)
            .ok_or_else(|| anyhow!("'{name}'"))?
            .ty
            .clone();
        let v = coerce(&ty, value)?;
        if ty != "string" {
            renderer.borrow().set_scene_param(name, &ty, &v)?;
        }
        told(ctrl, name, &v);
        scene
            .params
            .get_mut(name)
            .ok_or_else(|| anyhow!("'{name}'"))?
            .value = v;
        Ok(())
    }();
    match applied {
        Ok(()) => true,
        Err(e) => {
            eprintln!("set scene param {name}: {e}");
            false
        }
    }
}

/// Run the console's render loop on this thread. The console adds the command
/// queue, the published scene and the stale snapshot. A frame error must not
/// kill the thread.
///
/// GL is thread-affine, so this creates the Renderer on the thread that renders.
/// The sinks belong to the caller, and this draws to them from here. A driver
/// bound to the calling thread works as a sink.
pub fn render_loop(
    console: &Arc<Console>,
    mut sinks: Vec<Box<dyn Output>>,
    on_stats: Option<StatsHook>,
) {
    let renderer = match make_renderer(console) {
        Ok(r) => Rc::new(RefCell::new(r)),
        Err(e) => return console.fail(&format!("renderer: {e}")),
    };
    let hz = if console.display {
        renderer.borrow().display_vrefresh()
    } else {
        0
    };
    let st = RefCell::new(Running::default());
    // what feeds the running scene, when the scene brings something
    let ctrl: ControllerSlot = RefCell::new(None);
    let mut shared = SharedRenderer(Rc::clone(&renderer));
    let mut scanout = Scanout;
    let mut console_sink = ConsoleSink {
        console: Arc::clone(console),
        renderer: Rc::clone(&renderer),
        enc: None,
        last: Cell::new(0.0),
        now: Cell::new(0.0),
        hz,
    };
    let mut shot_sink = FrameShot {
        shot: Arc::clone(&console.shot),
    };
    let mut reported = String::new();
    let times: RefCell<Vec<f64>> = RefCell::new(Vec::new());
    let timed = on_stats.is_some();
    let (elapsed, out): (f64, Vec<String>);
    {
        let mut refs: Vec<&mut dyn Output> = Vec::new();
        if console.display {
            refs.push(&mut scanout);
        }
        refs.extend(sinks.iter_mut().map(|s| &mut **s as &mut dyn Output));
        refs.push(&mut console_sink);
        refs.push(&mut shot_sink);
        let mut render = Loop::new(&mut shared, refs);
        render.frames = console.frames;
        render.timestep = console.timestep;
        render.rate = Some(Arc::clone(&console.frame_rate));
        render.rest = 0.003; // yield to the handler and encoder threads
        let feed = provider(console);
        render.provider = Some(Box::new(feed));
        render.controller = Some(&ctrl);
        render.poll = Some(Box::new(|| drain(console, &renderer, &st, &ctrl)));
        render.idle_hook = Some(Box::new(|period| {
            let interval = skip_interval(
                hz,
                console.nclients() + console.fanout.nclients() as i64,
                console.preview_fps,
                period.unwrap_or(0.0),
            );
            match interval {
                None => {
                    console.q.wait(); // nobody consumes frames, so sleep until a command
                    0.0
                }
                Some(interval) => interval,
            }
        }));
        render.stopping = Some(Box::new(|| console.stopping()));
        render.on_frame = Some(Box::new(|_loop, cost| {
            lock(&console.stat).frame_ms = cost * 1000.0;
            if timed {
                times.borrow_mut().push(cost);
            }
        }));
        render.on_error = Some(Box::new(|e| {
            // One bad frame must not take the thread with it. Report it, and
            // keep rendering the last scene that worked.
            let msg = format!("render: {e}");
            lock(&console.stat).error = msg.clone();
            if msg != reported {
                eprintln!("ghee console: {msg}");
                reported = msg;
            }
            true
        }));
        let first = console.store.names().first().cloned();
        if let Some(name) = first {
            load(console, &renderer, &st, &ctrl, &name);
        }
        if let Err(e) = render.run() {
            eprintln!("ghee console: render: {e}");
        }
        elapsed = render.elapsed;
        out = render.out().iter().map(|d| (*d).to_string()).collect();
    }
    // The scene's process stops with the loop that drove it.
    *ctrl.borrow_mut() = None;
    if console.display {
        let _ = scanout.close();
    }
    for s in sinks.iter_mut() {
        let _ = s.close();
    }
    let _ = console_sink.close();
    let _ = shot_sink.close();
    if let Some(report) = on_stats {
        report(&times.borrow(), elapsed, &out);
    }
    // The loop ended, so nothing renders to serve.
    console.stop();
}

/// The host's array source for a name, else the placeholder feed.
fn provider(console: &Console) -> impl FnMut(&str, ArraySize, u64) -> Feed + '_ {
    |name: &str, size: ArraySize, frame: u64| match console.arrays.iter().find(|(n, _)| n == name) {
        Some((_, src)) => Feed::Data(src(size, frame)),
        None => Feed::Data(demo_feed(name, size, frame)),
    }
}

/// This module's render loop, as a console host. A handler thread reads the
/// published state and queues items only. The render thread owns the renderer
/// and drains the queue, so nothing here touches GL.
pub struct RenderLoopHost {
    console: Arc<Console>,
}

impl RenderLoopHost {
    pub fn new(console: Arc<Console>) -> RenderLoopHost {
        RenderLoopHost { console }
    }
}

impl ConsoleHost for RenderLoopHost {
    fn writable(&self) -> bool {
        !self.console.store.read_only
    }

    fn save_refused(&self) -> String {
        "demo scenes are read-only; serve your own directory to save".to_string()
    }

    fn stats(&self) -> ConsoleResult<Value> {
        Ok(self.console.stats())
    }

    fn scene_names(&self) -> ConsoleResult<Vec<String>> {
        Ok(self.console.store.names())
    }

    fn active_name(&self) -> ConsoleResult<Option<String>> {
        Ok(lock(&self.console.stat).active.clone())
    }

    fn scene_def(&self, name: Option<&str>) -> ConsoleResult<Value> {
        let active = lock(&self.console.stat).active.clone();
        let name = name.map(str::to_string).or(active).unwrap_or_default();
        if !self.console.store.contains(&name) {
            return Err(ConsoleError::not_found("no scene"));
        }
        if let Some((cur, doc)) = lock(&self.console.running).as_ref() {
            if cur.as_deref() == Some(name.as_str()) {
                return Ok(doc.clone());
            }
        }
        let scene = self
            .console
            .store
            .resolve(&name)
            .map_err(|e| ConsoleError::server_error(format!("build {name}: {e}")))?;
        Ok(scene.to_dict())
    }

    fn hwinfo(&self) -> ConsoleResult<Value> {
        Ok((self.console.hwinfo)())
    }

    fn set_param(&self, scope: Scope, name: &str, value: Value) -> ConsoleResult<()> {
        self.console.command(set_item(scope, name, value));
        Ok(())
    }

    fn switch(&self, name: Option<&str>) -> ConsoleResult<String> {
        let name = name.unwrap_or_default();
        if !self.console.store.contains(name) {
            return Err(ConsoleError::not_found("no scene"));
        }
        self.console.command(Command::Switch(name.to_string()));
        Ok(name.to_string())
    }

    fn apply_scene(&self, scene: Scene) -> ConsoleResult<()> {
        self.console.command(Command::SceneObj(Box::new(scene)));
        Ok(())
    }

    fn save_scene(&self, name: &str, scene: Scene) -> ConsoleResult<()> {
        self.console.store.save(name, scene)
    }
}

// ---------------------------------------------------------------- heartbeat

/// One stream connection, and when its client last reported in. A connection
/// with no vid never expires, because nothing pings for it. That keeps
/// `curl /h264` and a page older than the heartbeat streaming.
pub struct Viewer {
    vid: String,
    last: Mutex<f64>,
}

/// The stream viewers that must report in.
///
/// This holds each connection, not one entry per vid. A page that reopens its
/// stream leaves nothing behind that outlives the connection it belonged to.
pub struct Heartbeat {
    pub deadline: f64,
    by_vid: Mutex<Vec<(String, Vec<Arc<Viewer>>)>>,
}

impl Heartbeat {
    pub fn new(deadline: f64) -> Heartbeat {
        Heartbeat {
            deadline,
            by_vid: Mutex::new(Vec::new()),
        }
    }

    pub fn join(&self, vid: Option<&str>) -> Arc<Viewer> {
        let v = Arc::new(Viewer {
            vid: vid.unwrap_or_default().to_string(),
            last: Mutex::new(monotonic()),
        });
        if !v.vid.is_empty() {
            let mut groups = lock(&self.by_vid);
            match groups.iter_mut().find(|(id, _)| *id == v.vid) {
                Some((_, group)) => group.push(Arc::clone(&v)),
                None => groups.push((v.vid.clone(), vec![Arc::clone(&v)])),
            }
        }
        v
    }

    pub fn leave(&self, v: &Arc<Viewer>) {
        if v.vid.is_empty() {
            return;
        }
        let mut groups = lock(&self.by_vid);
        let Some(i) = groups.iter().position(|(id, _)| *id == v.vid) else {
            return;
        };
        groups[i].1.retain(|x| !Arc::ptr_eq(x, v));
        if groups[i].1.is_empty() {
            groups.remove(i);
        }
    }

    /// Mark every connection under this vid alive. Returns how many. A page that
    /// pings before its stream opens gets zero.
    pub fn ping(&self, vid: Option<&str>) -> usize {
        let now = monotonic();
        let vid = vid.unwrap_or_default();
        let group: Vec<Arc<Viewer>> = lock(&self.by_vid)
            .iter()
            .find(|(id, _)| id == vid)
            .map(|(_, g)| g.clone())
            .unwrap_or_default();
        for v in &group {
            *lock(&v.last) = now;
        }
        group.len()
    }

    pub fn gone(&self, v: &Viewer) -> bool {
        !v.vid.is_empty() && (monotonic() - *lock(&v.last)) > self.deadline
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        lock(&self.by_vid).is_empty()
    }
}

// ---------------------------------------------------------------- transport

/// The first line only. A status body stays one line.
fn line(msg: &str) -> &str {
    msg.split('\n').next().unwrap_or("")
}

fn headers_of(request: &Request) -> Headers {
    let mut headers = Headers::new();
    for h in request.headers() {
        headers.insert(h.field.as_str().as_str(), h.value.as_str());
    }
    headers
}

fn send(request: Request, resp: &Response) {
    let mut out =
        tiny_http::Response::from_data(resp.body.clone()).with_status_code(StatusCode(resp.status));
    if let Ok(h) = Header::from_bytes(&b"Content-Type"[..], resp.content_type.as_bytes()) {
        out = out.with_header(h);
    }
    let _ = request.respond(out);
}

/// The status line and headers a stream writes onto the socket itself. The
/// stream frames the parts that follow.
fn stream_head(w: &mut dyn Write, ctype: &str, extra: &[&str]) -> std::io::Result<()> {
    let mut head = format!("HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\n");
    for line in extra {
        head.push_str(line);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    w.write_all(head.as_bytes())?;
    w.flush()
}

/// The blocking transport. It serves the streams itself, and it hands the rest
/// to the API.
fn handle(console: &Arc<Console>, api: &ConsoleAPI<RenderLoopHost>, mut request: Request) {
    let method = request.method().as_str().to_string();
    let url = request.url().to_string();
    let (path, qs) = url.split_once('?').unwrap_or((url.as_str(), ""));
    let headers = headers_of(&request);
    if let Some(resp) = api.precheck(&method, path, &headers) {
        return send(request, &resp);
    }
    let query = parse_query(qs);
    if method == "GET" {
        match path {
            "/stream" => return stream(console, request, &query),
            "/h264" => return h264(console, request, &query),
            "/frame.png" => return frame_png(console, request),
            _ => {}
        }
        let resp = api.dispatch("GET", path, &query, &headers, b"");
        return send(request, &resp);
    }
    let mut body: Vec<u8> = Vec::new();
    if method == "POST" {
        let _ = request.as_reader().read_to_end(&mut body);
        if path == "/ping" {
            console.viewers.ping(query.get("vid").map(String::as_str));
            return send(request, &Response::text(200, "ok"));
        }
    }
    let resp = api.dispatch(&method, path, &query, &headers, &body);
    send(request, &resp);
}

fn stream(console: &Arc<Console>, request: Request, query: &Query) {
    let mut w = request.into_writer();
    if stream_head(
        &mut w,
        "multipart/x-mixed-replace; boundary=frame",
        &["Cache-Control: no-cache"],
    )
    .is_err()
    {
        return;
    }
    let viewer = console.viewers.join(query.get("vid").map(String::as_str));
    console.count_clients(1);
    console.q.put(Command::Preview); // wake an idle scene so it draws a frame
    let mut last_seq = u64::MAX;
    while !console.stopping() {
        if console.viewers.gone(&viewer) {
            break;
        }
        let (jpg, seq) = {
            let mut latest = lock(&console.latest);
            if latest.seq == last_seq {
                latest = console
                    .latest_ready
                    .wait_timeout(latest, Duration::from_secs(1))
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
            }
            (latest.jpg.clone(), latest.seq)
        };
        let Some(jpg) = jpg.filter(|_| seq != last_seq) else {
            continue;
        };
        last_seq = seq;
        let part = format!(
            "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
            jpg.len()
        );
        if w.write_all(part.as_bytes()).is_err()
            || w.write_all(&jpg).is_err()
            || w.write_all(b"\r\n").is_err()
            || w.flush().is_err()
        {
            break;
        }
    }
    console.count_clients(-1);
    console.viewers.leave(&viewer);
}

fn h264(console: &Arc<Console>, request: Request, query: &Query) {
    let mut w = request.into_writer();
    if stream_head(
        &mut w,
        "application/octet-stream",
        &["Cache-Control: no-store", "Transfer-Encoding: chunked"],
    )
    .is_err()
    {
        return;
    }
    let c = console.fanout.add();
    let viewer = console.viewers.join(query.get("vid").map(String::as_str));
    console.q.put(Command::Preview); // wake an idle scene so the stream starts
    while !console.stopping() {
        if console.viewers.gone(&viewer) {
            break;
        }
        let Some(blob) = c.get(Duration::from_secs(1)) else {
            continue;
        };
        let head = format!("{:X}\r\n", blob.len());
        if w.write_all(head.as_bytes()).is_err()
            || w.write_all(&blob).is_err()
            || w.write_all(b"\r\n").is_err()
            || w.flush().is_err()
        {
            break;
        }
    }
    console.fanout.remove(&c);
    console.viewers.leave(&viewer);
    // The encoder tears down on the next drawn frame, and a clean scene skips
    // that frame. An open encoder holds keep_last, which costs the display its
    // direct scanout while nobody watches.
    console.q.put(Command::Preview);
}

fn frame_png(console: &Arc<Console>, request: Request) {
    // A scene that does not compile has no frame. Say so, instead of answering
    // with the scene before it.
    let error = lock(&console.stat).error.clone();
    if !error.is_empty() {
        return send(
            request,
            &Response::text(409, &format!("scene error: {}", line(&error))),
        );
    }
    let gen = console.shot.request();
    console.shot.hold();
    console.q.put(Command::Shot); // forces a draw, and wakes a loop parked on the queue
    let frame = console.shot.wait(gen, FRAME_TIMEOUT);
    let png = frame.as_ref().map(|f| encode_png(f.view()));
    console.shot.release();
    let png = match png {
        Some(Ok(png)) => png,
        other => {
            let fatal = lock(&console.fatal).clone();
            let why = match (other, fatal.is_empty()) {
                (Some(Err(e)), _) => e.to_string(),
                (_, false) => fatal,
                _ => format!("nothing drawn in {FRAME_TIMEOUT}s"),
            };
            return send(
                request,
                &Response::text(503, &format!("no frame: {}", line(&why))),
            );
        }
    };
    let mut out = tiny_http::Response::from_data(png).with_status_code(StatusCode(200));
    for (name, value) in [("Content-Type", "image/png"), ("Cache-Control", "no-store")] {
        if let Ok(h) = Header::from_bytes(name.as_bytes(), value.as_bytes()) {
            out = out.with_header(h);
        }
    }
    let _ = request.respond(out);
}

// ---------------------------------------------------------------- run

/// The render calls this once at the end, with every frame's cost, the wall time
/// and the destinations the frames reached. It runs on the render thread.
pub type StatsHook = Box<dyn FnOnce(&[f64], f64, &[String])>;

/// What run() serves. Every field has a default, so a caller sets only what it
/// needs.
pub struct Options {
    /// The array sources the scenes read per frame.
    pub arrays: Vec<(String, ArraySource)>,
    /// The host info for the device-info readout.
    pub hwinfo: Option<Box<dyn Fn() -> Value + Send + Sync>>,
    pub scenes_dir: Option<String>,
    pub size: (i32, i32),
    pub port: u16,
    /// Loopback by default. The console has no auth, and it compiles POSTed
    /// GLSL. Pass "0.0.0.0" only on a network you trust.
    pub host: String,
    pub display: bool,
    pub rgb565: bool,
    pub preview_max: i32,
    pub preview_fps: f64,
    pub stream_max: i32,
    pub bitrate: i64,
    /// The build-time arguments for every scene the console loads.
    pub scene_args: Vec<String>,
    /// A prebuilt store, instead of one over scenes_dir.
    pub store: Option<SceneStore>,
    /// The output sinks the frames also reach, so a console drives a panel or
    /// writes files. `display` covers the KMS scanout instead, because that one
    /// is a mode the renderer is created in. run() writes these from its own
    /// thread, so a sink bound to the thread that built it belongs here.
    pub sinks: Vec<Box<dyn Output>>,
    /// The pixel format a panel sink packs to, for /stats.
    pub native_format: Option<String>,
    pub components: i32,
    /// 0 serves until stop(). A count ends the console with the render.
    pub frames: u64,
    pub timestep: Option<f64>,
    pub dri_device: Option<String>,
    /// The summary hook. This hook makes the loop keep a cost per frame, and a
    /// run without it keeps none.
    pub on_stats: Option<StatsHook>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            arrays: Vec::new(),
            hwinfo: None,
            scenes_dir: None,
            size: (320, 240),
            port: 8090,
            host: "127.0.0.1".to_string(),
            display: false,
            rgb565: false,
            preview_max: 480,
            preview_fps: 15.0,
            stream_max: 0,
            bitrate: 10_000_000,
            scene_args: Vec::new(),
            store: None,
            sinks: Vec::new(),
            native_format: None,
            components: 4,
            frames: 0,
            timestep: None,
            dri_device: None,
            on_stats: None,
        }
    }
}

/// Start the console and serve until stop() (or Ctrl-C).
pub fn run(opts: Options) -> Result<()> {
    let Options {
        arrays,
        hwinfo,
        scenes_dir,
        size,
        port,
        host,
        display,
        rgb565,
        preview_max,
        preview_fps,
        stream_max,
        bitrate,
        scene_args,
        store,
        sinks,
        native_format,
        components,
        frames,
        timestep,
        dri_device,
        on_stats,
    } = opts;
    let store = match store {
        Some(store) => store,
        None => {
            let read_only = scenes_dir.is_none();
            let dir = match scenes_dir {
                Some(dir) => dir,
                None => examples_dir().ok_or_else(|| {
                    anyhow!("no scenes directory (set GHEE_EXAMPLES for the bundled scenes)")
                })?,
            };
            SceneStore::new(&dir, read_only, scene_args)
        }
    };
    let names = store.names();
    if names.is_empty() {
        bail!("no .scene.json in {}", store.dir);
    }
    let mut console = Console::new(store);
    console.w = size.0;
    console.h = size.1;
    console.display = display;
    console.rgb565 = rgb565;
    console.preview_max = preview_max;
    console.preview_fps = preview_fps;
    console.stream_max = stream_max;
    console.bitrate = bitrate;
    console.components = components;
    console.frames = frames;
    console.timestep = timestep;
    console.dri = dri_device;
    console.sink_dests = sinks
        .iter()
        .map(|s| s.dest().to_string())
        .filter(|d| !d.is_empty())
        .collect();
    console.native_format = native_format;
    console.arrays = arrays;
    if let Some(hwinfo) = hwinfo {
        console.hwinfo = hwinfo;
    }
    let format = console.format();
    {
        let mut stat = lock(&console.stat);
        stat.active = Some(names[0].clone());
        stat.display =
            serde_json::json!({"w": size.0, "h": size.1, "format": format, "rotation": 0});
    }
    let console = Arc::new(console);
    console.reset_stats();
    *lock(&CONSOLE) = Some(Arc::clone(&console));
    let server = Server::http((host.as_str(), port)).map_err(|e| anyhow!("{e}"))?;
    let server = Arc::new(server);
    *lock(&console.httpd) = Some(Arc::clone(&server));
    for worker in [encoder_loop as fn(&Console), h264_sw_loop as fn(&Console)] {
        let console = Arc::clone(&console);
        thread::spawn(move || worker(&console));
    }
    let serving = {
        let console = Arc::clone(&console);
        let api = Arc::new(
            ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console))).with_web(console.web),
        );
        thread::spawn(move || {
            for request in server.incoming_requests() {
                let (console, api) = (Arc::clone(&console), Arc::clone(&api));
                thread::spawn(move || handle(&console, &api, request));
            }
        })
    };
    println!(
        "ghee console: {} scenes [{}] on {host}:{port} (scenes={}, preview {}x{} {format})",
        names.len(),
        names
            .iter()
            .map(|n| format!("'{n}'"))
            .collect::<Vec<_>>()
            .join(", "),
        console.store.dir,
        size.0,
        size.1,
    );
    render_loop(&console, sinks, on_stats);
    // The loop is done, so nothing is left to serve. stop() unblocks the accept
    // thread, and the handlers it spawned end on the same flag.
    console.stop();
    let _ = serving.join();
    let fatal = lock(&console.fatal).clone();
    if !fatal.is_empty() {
        bail!("{fatal}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    const SCENE: &str = r#"{"version": 1,
        "passes": [{"name": "main", "code": "void main(){ gl_FragColor = vec4(1.0); }"}]}"#;
    const SC: &[u8] = &[0, 0, 0, 1];

    fn tmp_console(tag: &str) -> Arc<Console> {
        let d = std::env::temp_dir().join(format!("ghee-console-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("one.scene.json"), SCENE).unwrap();
        Arc::new(Console::new(SceneStore::new(
            &d.to_string_lossy(),
            true,
            Vec::new(),
        )))
    }

    /// Whether a GL device opens. `run` builds its own renderer, so a test
    /// that calls it reports the skip and passes where no device opens. The
    /// answer is cached: one probe opens one renderer for the whole process.
    fn have_gl() -> bool {
        static GL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *GL.get_or_init(|| {
            let mut errs = Vec::new();
            for dev in crate::engine::probe_dri_devices() {
                match crate::engine::Renderer::new(8, 8, 4, Some(&dev), false, false) {
                    Ok(_) => return true,
                    Err(e) => errs.push(format!("{dev}: {e}")),
                }
            }
            eprintln!("no GL device for the console: {}", errs.join("; "));
            false
        })
    }

    fn key_au() -> Vec<u8> {
        let mut au = Vec::new();
        for nal in [
            &[0x67, 0x42, 0x00, 0x1e][..],
            &[0x68, 0xce, 0x06, 0xe2][..],
            &[0x65, 0x88, 0x80][..],
        ] {
            au.extend_from_slice(SC);
            au.extend_from_slice(nal);
        }
        au
    }

    // /set on the wire

    #[test]
    fn set_item_scene_scope() {
        // the literal pass="scene" routes to a distinct scene-scoped queue item
        match set_item(Scope::Scene, "speed", Value::from(0.5)) {
            Command::SceneParam(name, v) => {
                assert_eq!((name.as_str(), v), ("speed", Value::from(0.5)))
            }
            _ => panic!("not a scene param"),
        }
    }

    #[test]
    fn set_item_pass_scope() {
        match set_item(Scope::Pass(2), "k", Value::from(0.5)) {
            Command::Param(i, name, v) => {
                assert_eq!((i, name.as_str(), v), (2, "k", Value::from(0.5)));
            }
            _ => panic!("not a pass param"),
        }
        match set_item(Scope::Pass(0), "tint", serde_json::json!([1.0, 0.0, 0.0])) {
            Command::Param(i, name, v) => {
                assert_eq!((i, name.as_str()), (0, "tint"));
                assert_eq!(v, serde_json::json!([1.0, 0.0, 0.0]));
            }
            _ => panic!("not a pass param"),
        }
    }

    #[test]
    fn a_coerced_value_takes_the_declared_type() {
        assert_eq!(
            coerce("int", &Value::from(2.6)).unwrap(),
            ParamValue::Int(3)
        );
        assert_eq!(
            coerce("int", &Value::from(2.5)).unwrap(),
            ParamValue::Int(2)
        ); // half to even
        assert_eq!(
            coerce("bool", &Value::from(1.0)).unwrap(),
            ParamValue::Bool(true)
        );
        assert_eq!(
            coerce("float", &Value::from(0.5)).unwrap(),
            ParamValue::Float(0.5)
        );
        assert_eq!(
            coerce("color", &serde_json::json!([1.0, 0.0, 0.5])).unwrap(),
            ParamValue::Color([1.0, 0.0, 0.5])
        );
        assert!(coerce("color", &serde_json::json!([1.0, 0.0])).is_err());
    }

    // stream sizing

    #[test]
    fn a_panel_sized_stream_keeps_the_configured_rate() {
        assert_eq!(stream_bitrate(128, 32, 30, 10_000_000), 10_000_000);
        assert_eq!(stream_bitrate(320, 240, 30, 10_000_000), 10_000_000);
    }

    #[test]
    fn a_2k_stream_is_lifted_off_its_pixel_rate() {
        let bps = stream_bitrate(2048, 2048, 30, 10_000_000);
        let want = f64::from(2048 * 2048) * 30.0 * H264_BPP;
        assert!((bps as f64 - want).abs() < 1.0, "{bps}");
        assert!(bps > 4_000_000);
    }

    #[test]
    fn the_derived_rate_stops_at_the_ceiling() {
        assert_eq!(
            stream_bitrate(4096, 4096, 60, 10_000_000),
            H264_BITRATE_CEILING
        );
    }

    #[test]
    fn a_configured_rate_above_the_ceiling_still_wins() {
        assert_eq!(stream_bitrate(2048, 2048, 30, 100_000_000), 100_000_000);
    }

    #[test]
    fn a_stream_keeps_its_aspect_and_stays_even() {
        assert_eq!(stream_size(320, 240, 0), (320, 240));
        assert_eq!(stream_size(320, 240, 160), (160, 120));
        assert_eq!(stream_size(101, 33, 0), (100, 32));
        assert_eq!(stream_size(2, 2, 1), (2, 2));
    }

    #[test]
    fn the_skip_interval_follows_whoever_consumes_frames() {
        assert_eq!(skip_interval(60, 0, 15.0, 0.5), Some(1.0 / 60.0));
        assert_eq!(skip_interval(0, 0, 15.0, 0.5), None);
        assert_eq!(skip_interval(0, 1, 15.0, 0.5), Some(1.0 / 15.0));
        assert_eq!(skip_interval(0, 1, 0.0, 0.5), Some(0.5));
    }

    // the preview

    #[test]
    fn the_preview_quality_is_not_the_old_mush() {
        // q50 on a 480px frame looks mushy once a browser enlarges it
        let quality = u32::from(PREVIEW_QUALITY);
        assert!(quality > 50);
    }

    fn test_frame(w: i32, h: i32) -> Arc<RawFrame> {
        let mut data = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                data[i] = (x % 256) as u8;
                data[i + 1] = (y % 256) as u8;
                data[i + 2] = ((x * y) % 256) as u8;
                data[i + 3] = 255;
            }
        }
        Arc::new(RawFrame {
            data,
            width: w,
            height: h,
            components: 4,
        })
    }

    /// (width, height) out of a baseline JPEG's frame header.
    fn jpeg_size(jpg: &[u8]) -> (i32, i32) {
        let mut i = 2;
        while i + 9 < jpg.len() {
            let marker = jpg[i + 1];
            let len = usize::from(u16::from_be_bytes([jpg[i + 2], jpg[i + 3]]));
            if (0xC0..=0xCF).contains(&marker) && marker != 0xC4 && marker != 0xC8 {
                let h = i32::from(u16::from_be_bytes([jpg[i + 5], jpg[i + 6]]));
                let w = i32::from(u16::from_be_bytes([jpg[i + 7], jpg[i + 8]]));
                return (w, h);
            }
            i += 2 + len;
        }
        (0, 0)
    }

    fn feed_encoder(console: &Arc<Console>, frame: Arc<RawFrame>) -> Arc<Vec<u8>> {
        {
            let mut raw = lock(&console.raw);
            raw.rgb = Some(frame);
            raw.seq += 1;
            console.raw_ready.notify_all();
        }
        let end = monotonic() + 5.0;
        while monotonic() < end {
            if let Some(jpg) = lock(&console.latest).jpg.clone() {
                return jpg;
            }
            thread::sleep(Duration::from_millis(20));
        }
        panic!("the encoder produced no preview frame");
    }

    #[test]
    fn a_frame_bigger_than_the_preview_cap_is_downscaled() {
        let mut console = tmp_console("preview");
        Arc::get_mut(&mut console).unwrap().preview_max = 120;
        console.count_clients(1); // the encoder only works while a viewer is counted
        let worker = {
            let console = Arc::clone(&console);
            thread::spawn(move || encoder_loop(&console))
        };
        let jpg = feed_encoder(&console, test_frame(480, 240));
        assert_eq!(jpeg_size(&jpg), (120, 60));
        console.stop();
        console.raw_ready.notify_all();
        worker.join().unwrap();
    }

    #[test]
    fn a_frame_inside_the_cap_keeps_its_size() {
        let console = tmp_console("preview-small");
        console.count_clients(1);
        let worker = {
            let console = Arc::clone(&console);
            thread::spawn(move || encoder_loop(&console))
        };
        let jpg = feed_encoder(&console, test_frame(96, 96));
        assert_eq!(jpeg_size(&jpg), (96, 96));
        console.stop();
        console.raw_ready.notify_all();
        worker.join().unwrap();
    }

    // the h264 fanout

    #[test]
    fn fanout_wires_each_frame_once() {
        let fanout = H264Fanout::new(Arc::new(Mutex::new(FrameRate::new())));
        let (a, b) = (fanout.add(), fanout.add());
        fanout.push(&key_au());
        let (a_cfg, a_key) = (
            a.get(Duration::ZERO).unwrap(),
            a.get(Duration::ZERO).unwrap(),
        );
        let (b_cfg, b_key) = (
            b.get(Duration::ZERO).unwrap(),
            b.get(Duration::ZERO).unwrap(),
        );
        assert!(Arc::ptr_eq(&a_cfg, &b_cfg) && Arc::ptr_eq(&a_key, &b_key));
        let cfg = lock(&fanout.cfg).clone().unwrap();
        assert_eq!(*a_cfg, wire_frame(0, &cfg));
    }

    #[test]
    fn a_client_that_left_takes_no_more_frames() {
        let fanout = H264Fanout::new(Arc::new(Mutex::new(FrameRate::new())));
        let c = fanout.add();
        fanout.remove(&c);
        assert_eq!(fanout.nclients(), 0);
        fanout.push(&key_au());
        assert!(c.get(Duration::ZERO).is_none());
    }

    #[test]
    fn the_fanout_counts_what_it_hands_to_viewers() {
        let fanout = H264Fanout::new(Arc::new(Mutex::new(FrameRate::new())));
        fanout.add();
        for _ in 0..4 {
            fanout.push(&key_au());
        }
        assert!(fanout.stream_fps() > 0.0);
    }

    #[test]
    fn nothing_is_streamed_to_nobody() {
        let fanout = H264Fanout::new(Arc::new(Mutex::new(FrameRate::new())));
        fanout.push(&key_au());
        assert_eq!(fanout.stream_fps(), 0.0);
    }

    #[test]
    fn one_counter_covers_both_codecs() {
        let shared = Arc::new(Mutex::new(FrameRate::new()));
        let fanout = H264Fanout::new(Arc::clone(&shared));
        fanout.add();
        fanout.push(&key_au());
        assert!(Arc::ptr_eq(&fanout.rate, &shared));
        // the JPEG path ticks the same counter the console publishes
        let console = tmp_console("one-counter");
        assert!(Arc::ptr_eq(&console.fanout.rate, &console.stream_rate));
    }

    #[test]
    fn a_lagging_client_is_dropped_to_the_next_keyframe() {
        let fanout = H264Fanout::new(Arc::new(Mutex::new(FrameRate::new())));
        let c = fanout.add();
        fanout.push(&key_au());
        let delta = [SC, &[0x41, 0x9a, 0x00]].concat();
        for _ in 0..H264_LAG_FRAMES + 4 {
            fanout.push(&delta);
        }
        let queued = lock(&c.state).q.len();
        assert!(queued <= H264_LAG_FRAMES + 2, "{queued}");
        assert!(!lock(&c.state).started);
        assert!(fanout.want_key.load(Ordering::SeqCst));
    }

    // the still

    #[test]
    fn the_generations_answer_one_draw_each() {
        let shot = Shot::new(Arc::new(AtomicBool::new(false)));
        let frame = Arc::new(RawFrame {
            data: vec![0; 16],
            width: 2,
            height: 2,
            components: 4,
        });
        let (a, b) = (shot.request(), shot.request());
        shot.arm();
        assert!(shot.wants());
        shot.write(Arc::clone(&frame));
        assert!(!shot.wants());
        assert!(Arc::ptr_eq(&shot.wait(a, 0.0).unwrap(), &frame));
        assert!(Arc::ptr_eq(&shot.wait(b, 0.0).unwrap(), &frame));
        let c = shot.request(); // asked after the frame published, so it gets no answer
        assert!(shot.wait(c, 0.0).is_none());
    }

    #[test]
    fn two_requests_in_flight_share_one_generation() {
        let shot = Shot::new(Arc::new(AtomicBool::new(false)));
        let a = shot.request();
        shot.hold();
        let b = shot.request();
        assert_eq!(a, b);
        shot.touch(); // a command landed, so the frame in flight is behind
        assert_eq!(shot.request(), a + 1);
        shot.release();
    }

    #[test]
    fn a_shot_wait_gives_up_when_the_console_stops() {
        let stopping = Arc::new(AtomicBool::new(true));
        let shot = Shot::new(stopping);
        let gen = shot.request();
        assert!(shot.wait(gen, 5.0).is_none());
    }

    // viewers

    #[test]
    fn a_connection_that_named_no_vid_never_expires() {
        let hb = Heartbeat::new(0.0);
        let v = hb.join(None);
        thread::sleep(Duration::from_millis(10));
        assert!(!hb.gone(&v));
    }

    #[test]
    fn a_named_connection_expires_once_its_pings_stop() {
        let hb = Heartbeat::new(0.05);
        let v = hb.join(Some("abc"));
        assert!(!hb.gone(&v));
        thread::sleep(Duration::from_millis(80));
        assert!(hb.gone(&v));
    }

    #[test]
    fn a_ping_puts_a_named_connection_back_inside_the_deadline() {
        let hb = Heartbeat::new(0.05);
        let v = hb.join(Some("abc"));
        thread::sleep(Duration::from_millis(80));
        assert_eq!(hb.ping(Some("abc")), 1);
        assert!(!hb.gone(&v));
    }

    #[test]
    fn every_connection_under_one_vid_is_kept_alive_together() {
        // a page that falls back from h264 to MJPEG opens a second stream under
        // the id it already pings with
        let hb = Heartbeat::new(0.05);
        let (a, b) = (hb.join(Some("abc")), hb.join(Some("abc")));
        thread::sleep(Duration::from_millis(80));
        assert_eq!(hb.ping(Some("abc")), 2);
        assert!(!hb.gone(&a) && !hb.gone(&b));
    }

    #[test]
    fn a_connection_that_left_is_not_pinged_for_any_more() {
        let hb = Heartbeat::new(0.05);
        let (a, b) = (hb.join(Some("abc")), hb.join(Some("abc")));
        hb.leave(&a);
        assert_eq!(hb.ping(Some("abc")), 1);
        hb.leave(&b);
        assert_eq!(hb.ping(Some("abc")), 0);
        assert!(hb.is_empty());
    }

    #[test]
    fn a_ping_for_an_id_with_no_stream_matches_nothing() {
        let hb = Heartbeat::new(0.05);
        assert_eq!(hb.ping(Some("nobody")), 0);
        assert_eq!(hb.ping(None), 0);
    }

    #[test]
    fn the_deadline_leaves_room_for_lost_pings() {
        let (deadline, interval) = (VIEWER_DEADLINE, PING_INTERVAL);
        assert!(deadline >= 3.0 * interval);
    }

    // viewer counts

    #[test]
    fn client_count_is_atomic() {
        let console = tmp_console("counts");
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let console = Arc::clone(&console);
                thread::spawn(move || {
                    for _ in 0..200 {
                        console.count_clients(1);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(console.nclients(), 8 * 200);
        for _ in 0..8 * 200 {
            console.count_clients(-1);
        }
        assert_eq!(console.nclients(), 0);
    }

    // what /stats publishes

    #[test]
    fn stats_carries_every_key_the_page_polls() {
        let console = tmp_console("stats");
        let stats = console.stats();
        for key in [
            "fps",
            "frame_ms",
            "stream_fps",
            "out",
            "error",
            "error_line",
            "error_pass",
            "active",
            "display",
            "stale",
        ] {
            assert!(stats.get(key).is_some(), "{key}");
        }
        assert_eq!(stats["stale"], Value::from(false));
        assert_eq!(stats["out"], serde_json::json!([]));
    }

    #[test]
    fn a_stream_viewer_shows_up_in_the_destinations() {
        let console = tmp_console("dests");
        console.count_clients(1);
        assert_eq!(console.stats()["out"], serde_json::json!(["mjpeg"]));
        let c = console.fanout.add();
        assert_eq!(console.stats()["out"], serde_json::json!(["h264", "mjpeg"]));
        console.fanout.remove(&c);
        console.count_clients(-1);
        assert_eq!(console.stats()["out"], serde_json::json!([]));
    }

    // the host

    #[test]
    fn the_host_refuses_a_scene_it_does_not_have() {
        let console = tmp_console("host");
        let host = RenderLoopHost::new(Arc::clone(&console));
        assert_eq!(host.scene_names(), Ok(vec!["one".to_string()]));
        assert_eq!(
            host.scene_def(Some("nope")).err().map(|e| e.status),
            Some(404)
        );
        assert_eq!(host.switch(Some("nope")).err().map(|e| e.status), Some(404));
        assert!(!host.writable()); // a read-only store
    }

    #[test]
    fn a_switch_rides_the_command_queue() {
        let console = tmp_console("switch");
        let host = RenderLoopHost::new(Arc::clone(&console));
        assert_eq!(host.switch(Some("one")), Ok("one".to_string()));
        match console.q.take() {
            Some(Command::Switch(name)) => assert_eq!(name, "one"),
            _ => panic!("no switch was queued"),
        }
    }

    #[test]
    fn a_scene_def_off_disk_is_the_file() {
        let console = tmp_console("def");
        let host = RenderLoopHost::new(Arc::clone(&console));
        let doc = host.scene_def(Some("one")).unwrap();
        assert_eq!(doc["passes"][0]["name"], "main");
    }

    // the loop run() drives

    /// A sink that never leaves the thread that built it. A panel driver's SPI
    /// and GPIO handles work the same way.
    struct Bound(Rc<Cell<u32>>);

    impl Output for Bound {
        fn dest(&self) -> &str {
            crate::stats::DEST_PANEL
        }

        fn persists(&self) -> bool {
            true
        }

        fn write(&mut self, _frame: Frame<'_>) -> Result<()> {
            self.0.set(self.0.get() + 1);
            Ok(())
        }
    }

    #[test]
    fn the_console_draws_to_a_thread_bound_sink_on_the_callers_thread() {
        if !have_gl() {
            return;
        }
        let console = tmp_console("bound"); // for its scene directory
        let drawn = Rc::new(Cell::new(0));
        let ran = run(Options {
            store: Some(SceneStore::new(&console.store.dir, true, Vec::new())),
            port: free_port(),
            size: (32, 24),
            frames: 3,
            sinks: vec![Box::new(Bound(Rc::clone(&drawn)))],
            native_format: Some("rgb565".to_string()),
            ..Options::default()
        });
        assert!(ran.is_ok(), "{:?}", ran.err());
        assert_eq!(drawn.get(), 3);
    }

    /// What the stats hook hands back: frame count, wall time, destinations.
    type Summary = Rc<RefCell<Option<(usize, f64, Vec<String>)>>>;

    /// --stats needs this to print its line after a bounded serve run.
    #[test]
    fn a_bounded_console_run_hands_back_its_timing() {
        if !have_gl() {
            return;
        }
        let console = tmp_console("stats");
        let summary: Summary = Rc::new(RefCell::new(None));
        let got = Rc::clone(&summary);
        let ran = run(Options {
            store: Some(SceneStore::new(&console.store.dir, true, Vec::new())),
            port: free_port(),
            size: (32, 24),
            frames: 3,
            sinks: vec![Box::new(Bound(Rc::new(Cell::new(0))))],
            on_stats: Some(Box::new(move |times, wall, out| {
                *got.borrow_mut() = Some((times.len(), wall, out.to_vec()));
            })),
            ..Options::default()
        });
        assert!(ran.is_ok(), "{:?}", ran.err());
        let (n, wall, out) = summary.borrow().clone().expect("no summary");
        assert_eq!(n, 3);
        assert!(wall > 0.0);
        assert_eq!(out, [crate::stats::DEST_PANEL]);
    }

    fn free_port() -> u16 {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    }
}
