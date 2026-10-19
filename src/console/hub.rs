//! What every console serves, whichever host draws its frames.
//!
//! A host is the side that draws: the render loop of `ghee serve`, or a
//! session that a Python program drives. The host publishes into a `Hub`:
//! each frame that a viewer wants, the scene that runs, and the numbers
//! behind /stats. The hub's threads and the request threads read it and
//! never touch the renderer.
//!
//! The hub serves the HTTP transport around the console API: the MJPEG
//! preview, one JPEG a frame for every viewer, the H264 stream, one encoder
//! for every viewer, the still that /frame.png answers, and the viewers'
//! heartbeat. The hardware H264 lane issues GL calls, so it runs on the
//! host's thread (`HwLane`). The software lane encodes the published frames
//! on a thread of its own.

use std::collections::VecDeque;
use std::io::Write;
use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde_json::{Map, Value};
use tiny_http::{Header, Request, Server, StatusCode};

use crate::console::core::{parse_query, ConsoleAPI, ConsoleHost, Headers, Query, Response};
use crate::engine::{ErrorPass, Renderer};
use crate::h264::{
    avcc, build_config, no_encoder, split_annexb, wire_frame, ColorSpace, HardwareH264,
    SoftwareH264, HW_COLOR, NAL_IDR, NAL_PPS, NAL_SEI, NAL_SLICE, NAL_SPS, SW_COLOR, WIRE_ERROR,
};
use crate::live::{Changes, Live};
use crate::looper::Frame;
use crate::output::{encode_jpeg, encode_png};
use crate::scene::{short_floats, Scene};
use crate::stats::{monotonic, FrameRate, DEST_H264, DEST_MJPEG};

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

/// How long GET /frame.png waits for the host to answer its request.
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

/// The shortest time in seconds between two publishes of the running
/// document for the changes that a scene's own code makes to it. Writing the
/// document takes time with each layer, and the code may change a layer
/// every frame.
pub const LAYERS_PUBLISH_EVERY: f64 = 0.2;

/// Encodes with no output before the hardware lane is declared dead.
const HW_STARVE_LIMIT: u32 = 90;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Round a float. A half goes to the even side.
pub(crate) fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// One readback from the host's thread, row-major (height, width, components).
pub struct RawFrame {
    pub data: Vec<u8>,
    pub width: i32,
    pub height: i32,
    pub components: i32,
}

impl RawFrame {
    pub fn of(frame: Frame<'_>) -> RawFrame {
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

/// The latest preview frame handed to the encoder threads.
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
    /// Why the looks of the last apply failed, joined by newlines, in a host
    /// whose apply sets looks one layer at a time. The scene still draws.
    apply_error: String,
    /// The renderer's text_errors(), a line each.
    texts: Vec<String>,
    display: Value,
    error_line: Option<i64>,
    error_pass: Option<ErrorPass>,
    gpu: Value,
}

/// The /stats `gpu` object: what the renderer's GPU can do.
pub fn gpu_caps(renderer: &Renderer) -> Value {
    serde_json::json!({"trilinear_npot": renderer.trilinear_npot()})
}

/// The /stats `display.format` of a frame.
pub fn frame_format(rgb565: bool, components: i32) -> &'static str {
    match (rgb565, components) {
        (true, _) => "rgb565",
        (false, 3) => "rgb",
        _ => "rgba",
    }
}

/// A param's value that the host set: the param's pass, None for the
/// scene's, its name and its value.
pub type ParamSet = (Option<String>, String, Value);

/// The scene that renders, as the host last published it whole, and what
/// the program changed in it since.
struct Published {
    name: Option<String>,
    /// The generation of the working copy behind it, in a host that keeps one.
    generation: u64,
    doc: Value,
    /// Each layer that the program changed since, whole, by its name.
    layers: Map<String, Value>,
    /// Each param that the program set since.
    params: Vec<ParamSet>,
}

impl Published {
    /// The scene with the program's changes laid over it.
    fn now(&self) -> Value {
        let mut doc = self.doc.clone();
        if self.layers.is_empty() && self.params.is_empty() {
            return doc;
        }
        let value = |name: &str| format!("/params/{name}/value");
        let passes = doc.get_mut("passes").and_then(Value::as_array_mut);
        for pass in passes.into_iter().flatten() {
            let of = pass.get("name").and_then(Value::as_str).map(str::to_string);
            for (_, name, v) in self.params.iter().filter(|(p, ..)| p.is_some() && *p == of) {
                if let Some(slot) = pass.pointer_mut(&value(name)) {
                    slot.clone_from(v);
                }
            }
            let layers = pass.get_mut("layers").and_then(Value::as_array_mut);
            for layer in layers.into_iter().flatten() {
                let name = layer.get("name").and_then(Value::as_str);
                if let Some(new) = name.and_then(|n| self.layers.get(n)) {
                    layer.clone_from(new);
                }
            }
        }
        for (_, name, v) in self.params.iter().filter(|(p, ..)| p.is_none()) {
            if let Some(slot) = doc.pointer_mut(&value(name)) {
                slot.clone_from(v);
            }
        }
        doc
    }
}

/// What a host knows of its scene for /stats.
pub struct SceneStats {
    pub active: Option<String>,
    /// The files behind the scene changed since it loaded.
    pub stale: bool,
    /// What the scene's program set, `Locks::to_json()`.
    pub locked: Value,
    /// The scene's revision, in a host that keeps one.
    pub rev: Option<u64>,
}

/// What a console serves, shared by the host's thread, the encoder threads
/// and the request threads.
pub struct Hub {
    // configuration: the host sets it before the threads start
    /// The size of the frames that the host draws.
    pub w: i32,
    pub h: i32,
    /// The longest side of the browser preview. The encoder thread downscales it.
    pub preview_max: i32,
    /// The browser preview refresh cap. The display stays at full speed.
    pub preview_fps: f64,
    /// The cap on the h264 stream's longest side. 0 means full size.
    pub stream_max: i32,
    /// The rate that the hardware H264 lane encodes at.
    pub stream_hz: i32,
    pub bitrate: i64,
    /// The shortest time in seconds between two publishes of the program's
    /// changes.
    pub publish_every: f64,
    /// Where the host puts every frame: the scanout and the host's sinks.
    pub dests: Vec<String>,
    /// Say on stderr why a stream changed lanes or ended.
    pub verbose: bool,

    // state
    stat: Mutex<Stat>,
    published: Mutex<Option<Published>>,
    /// When the program's changes last went into the published document.
    changes_published: Mutex<f64>,
    /// The times that the scene's own code changed its layers in the
    /// document: a layer added or given a new look, or a field set.
    layer_changes: AtomicU64,
    drawn: AtomicU64,
    skipped: AtomicU64,
    raw: Mutex<Raw>,
    raw_ready: Condvar,
    latest: Mutex<Latest>,
    latest_ready: Condvar,
    /// When the preview last took a frame.
    previewed: Mutex<f64>,
    nclients: Mutex<i64>,
    /// The frames drawn.
    frame_rate: Mutex<FrameRate>,
    /// The frames handed to viewers, whichever codec.
    stream_rate: Arc<Mutex<FrameRate>>,
    pub fanout: H264Fanout,
    pub shot: Arc<Shot>,
    pub viewers: Heartbeat,
    stopping: Arc<AtomicBool>,
    /// A failure the host cannot draw through.
    fatal: Mutex<String>,
    /// The stream whose hardware encoding failed, which the software lane
    /// owns. A stall or a busy codec belongs to the stream it happened in, so
    /// the next stream tries hardware again, and the software lane costs a
    /// readback.
    hw_dead: Mutex<Option<u64>>,
    /// Why the hardware lane failed this stream.
    hw_why: Mutex<String>,
    httpd: Mutex<Option<Arc<Server>>>,
}

impl Default for Hub {
    fn default() -> Hub {
        Hub::new()
    }
}

impl Hub {
    /// A hub with the defaults of `ghee serve`.
    pub fn new() -> Hub {
        let stopping = Arc::new(AtomicBool::new(false));
        let stream_rate = Arc::new(Mutex::new(FrameRate::new()));
        Hub {
            w: 320,
            h: 240,
            preview_max: 480,
            preview_fps: 15.0,
            stream_max: 0,
            stream_hz: H264_FPS,
            bitrate: 10_000_000,
            publish_every: LAYERS_PUBLISH_EVERY,
            dests: Vec::new(),
            verbose: true,
            stat: Mutex::new(Stat {
                display: Value::Object(Map::new()),
                gpu: Value::Object(Map::new()),
                ..Stat::default()
            }),
            published: Mutex::new(None),
            changes_published: Mutex::new(f64::NEG_INFINITY),
            layer_changes: AtomicU64::new(0),
            drawn: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
            raw: Mutex::new(Raw::default()),
            raw_ready: Condvar::new(),
            latest: Mutex::new(Latest::default()),
            latest_ready: Condvar::new(),
            previewed: Mutex::new(0.0),
            nclients: Mutex::new(0),
            frame_rate: Mutex::new(FrameRate::new()),
            fanout: H264Fanout::new(Arc::clone(&stream_rate)),
            stream_rate,
            shot: Shot::new(Arc::clone(&stopping)),
            viewers: Heartbeat::new(VIEWER_DEADLINE),
            stopping,
            fatal: Mutex::new(String::new()),
            hw_dead: Mutex::new(None),
            hw_why: Mutex::new(String::new()),
            httpd: Mutex::new(None),
        }
    }

    pub fn stopping(&self) -> bool {
        self.stopping.load(Ordering::SeqCst)
    }

    fn say(&self, msg: &str) {
        if self.verbose {
            eprintln!("ghee console: {msg}");
        }
    }

    /// The MJPEG viewer count. A handler thread changes it. The host and
    /// encoder threads read it.
    pub fn count_clients(&self, delta: i64) -> i64 {
        let mut n = lock(&self.nclients);
        *n += delta;
        *n
    }

    pub fn nclients(&self) -> i64 {
        *lock(&self.nclients)
    }

    // ------------------------------------------------------------ stats

    /// Where the frames go now: the host's destinations, and each stream
    /// with a viewer.
    pub fn out(&self) -> Vec<String> {
        let mut out = self.dests.clone();
        if self.fanout.nclients() > 0 {
            out.push(DEST_H264.to_string());
        }
        if self.nclients() > 0 {
            out.push(DEST_MJPEG.to_string());
        }
        out
    }

    /// The /stats payload. Each rate computes on read, so an idle host's
    /// number falls with the clock.
    pub fn stats(&self, scene: SceneStats) -> Value {
        let mut d = Map::new();
        {
            let stat = lock(&self.stat);
            let error = match stat.error.is_empty() {
                true => &stat.apply_error,
                false => &stat.error,
            };
            d.insert("frame_ms".to_string(), Value::from(stat.frame_ms));
            d.insert("error".to_string(), Value::from(error.clone()));
            d.insert("text_errors".to_string(), Value::from(stat.texts.clone()));
            d.insert("display".to_string(), stat.display.clone());
            d.insert(
                "error_line".to_string(),
                stat.error_line.map_or(Value::Null, Value::from),
            );
            d.insert(
                "error_pass".to_string(),
                stat.error_pass.map_or(Value::Null, ErrorPass::to_json),
            );
            d.insert("gpu".to_string(), stat.gpu.clone());
        }
        d.insert(
            "active".to_string(),
            scene.active.map_or(Value::Null, Value::from),
        );
        d.insert(
            "fps".to_string(),
            Value::from(lock(&self.frame_rate).value(None)),
        );
        d.insert(
            "stream_fps".to_string(),
            Value::from(lock(&self.stream_rate).value(None)),
        );
        d.insert("out".to_string(), Value::from(self.out()));
        d.insert("stale".to_string(), Value::from(scene.stale));
        d.insert(
            "frames".to_string(),
            Value::from(self.drawn.load(Ordering::SeqCst)),
        );
        d.insert(
            "skipped".to_string(),
            Value::from(self.skipped.load(Ordering::SeqCst)),
        );
        d.insert(
            "layer_changes".to_string(),
            Value::from(self.layer_changes.load(Ordering::SeqCst)),
        );
        d.insert("locked".to_string(), scene.locked);
        d.insert(
            "rev".to_string(),
            scene.rev.map_or(Value::Null, Value::from),
        );
        Value::Object(d)
    }

    /// Drop what the scene before published: the rate windows, the last frame's
    /// cost, the compile error.
    pub fn reset(&self) {
        lock(&self.frame_rate).reset(None);
        lock(&self.stream_rate).reset(None);
        let mut stat = lock(&self.stat);
        stat.frame_ms = 0.0;
        stat.error = String::new();
        stat.error_line = None;
        stat.error_pass = None;
    }

    /// The frame's size, its format and its turn in degrees, for /stats.
    pub fn set_display(&self, format: &str, rotation: i64) {
        lock(&self.stat).display = serde_json::json!({
            "w": self.w, "h": self.h, "format": format, "rotation": rotation
        });
    }

    pub fn set_gpu(&self, gpu: Value) {
        lock(&self.stat).gpu = gpu;
    }

    /// The scene's error, with the pass and the line that hold it when it
    /// failed to compile. "" clears it.
    pub fn set_error(&self, error: &str, line: Option<i64>, pass: Option<ErrorPass>) {
        let mut stat = lock(&self.stat);
        stat.error = error.to_string();
        stat.error_line = line;
        stat.error_pass = pass;
    }

    /// The error of the running scene, "" when it draws.
    pub fn error(&self) -> String {
        lock(&self.stat).error.clone()
    }

    /// Why the looks of the last apply failed, "" when every look took.
    pub fn set_apply_error(&self, why: &str) {
        lock(&self.stat).apply_error = why.to_string();
    }

    /// The problems of the scene's text layers, a line each.
    pub fn set_text_errors(&self, lines: Vec<String>) {
        lock(&self.stat).texts = lines;
    }

    /// A failure that the host can't draw through. /frame.png answers with it.
    pub fn set_fatal(&self, why: &str) {
        *lock(&self.fatal) = why.to_string();
    }

    pub fn fatal(&self) -> String {
        lock(&self.fatal).clone()
    }

    /// Count a frame drawn, and what it cost in seconds.
    pub fn frame_done(&self, cost: f64) {
        lock(&self.frame_rate).tick(None);
        lock(&self.stat).frame_ms = cost * 1000.0;
        self.drawn.fetch_add(1, Ordering::SeqCst);
    }

    /// Count a frame skipped.
    pub fn skipped(&self) {
        self.skipped.fetch_add(1, Ordering::SeqCst);
    }

    // ------------------------------------------------------------ the scene

    /// Publish `doc`, the document of scene `name`, built from the working
    /// copy of `generation`.
    pub fn publish_scene(&self, name: Option<&str>, generation: u64, doc: Value) {
        *lock(&self.published) = Some(Published {
            name: name.map(str::to_string),
            generation,
            doc,
            layers: Map::new(),
            params: Vec::new(),
        });
    }

    /// No scene runs.
    pub fn clear_scene(&self) {
        *lock(&self.published) = None;
    }

    /// The running document of scene `name`, built from the working copy of
    /// `generation`, with the program's changes laid over it. None when the
    /// host published another.
    pub fn scene_of(&self, name: &str, generation: u64) -> Option<Value> {
        let held = lock(&self.published);
        let p = held.as_ref()?;
        (p.name.as_deref() == Some(name) && p.generation == generation).then(|| p.now())
    }

    /// The running document, with the program's changes laid over it.
    pub fn scene(&self) -> Option<Value> {
        lock(&self.published).as_ref().map(Published::now)
    }

    /// Publish what changed in the scene that `renderer` runs as scene `name`
    /// of `generation`. The whole document goes when `whole` is true, or when
    /// the program added, removed or moved a layer. Else each layer and param
    /// that the program changed goes, laid over the document. Without
    /// `whole`, the program's changes wait in `live` until `publish_every`
    /// passed since they last went. A change by the program, or `by_host`,
    /// counts in /stats, so a page reads the scene again. True when one
    /// counted.
    pub fn publish_live(
        &self,
        renderer: &Renderer,
        live: Option<&mut Live>,
        whole: bool,
        by_host: bool,
        name: Option<&str>,
        generation: u64,
    ) -> bool {
        if !whole && live.as_deref().is_some_and(Live::changed) {
            let now = monotonic();
            let mut last = lock(&self.changes_published);
            if now - *last < self.publish_every {
                return false;
            }
            *last = now;
        }
        let Changes {
            layers,
            params,
            grown,
        } = live.map(Live::take_changes).unwrap_or_default();
        let changed = grown || !(layers.is_empty() && params.is_empty());
        if whole || grown {
            if let Some(doc) = renderer.with_scene(Scene::to_dict) {
                self.publish_scene(name, generation, doc);
            }
        } else if changed {
            self.publish_changes(renderer, layers, params);
        }
        if changed || by_host {
            self.layer_changes.fetch_add(1, Ordering::SeqCst);
        }
        changed || by_host
    }

    /// Lay the layers named `changed`, whole, and the values of the params of
    /// `set`, as `renderer` holds them, over the published document.
    fn publish_changes(
        &self,
        renderer: &Renderer,
        changed: Vec<String>,
        set: Vec<(Option<String>, String)>,
    ) {
        let mut params = Vec::new();
        for (pass, name) in set {
            let value = renderer.with_scene(|s| {
                let params = match pass.as_deref() {
                    Some(p) => &s.passes.iter().find(|q| q.name == p)?.params,
                    None => &s.params,
                };
                Some(params.get(&name)?.value.to_json())
            });
            if let Some(mut value) = value.flatten() {
                short_floats(&mut value);
                params.push((pass, name, value));
            }
        }
        let layers: Vec<Value> = changed
            .iter()
            .filter_map(|name| {
                renderer.with_layer(name, |_, l| {
                    let mut doc = l.to_dict();
                    short_floats(&mut doc);
                    doc
                })
            })
            .collect();
        let mut held = lock(&self.published);
        let Some(held) = held.as_mut() else {
            return;
        };
        for layer in layers {
            if let Some(name) = layer.get("name").and_then(Value::as_str) {
                held.layers.insert(name.to_string(), layer);
            }
        }
        for (pass, name, value) in params {
            match held
                .params
                .iter_mut()
                .find(|(p, n, _)| *p == pass && *n == name)
            {
                Some(slot) => slot.2 = value,
                None => held.params.push((pass, name, value)),
            }
        }
    }

    // ------------------------------------------------------------ frames

    /// The size of the H264 stream: the cap on the longest side, never past
    /// the frame's size.
    pub fn stream_size(&self) -> (i32, i32) {
        encode_size(self.w, self.h, self.stream_max)
    }

    /// True when the preview or the software H264 lane takes this frame: a
    /// viewer watches, and the fastest of their rates is due. A still takes
    /// the frame whole instead.
    pub fn wants_preview(&self) -> bool {
        if self.shot.wants() {
            return false;
        }
        let mut rates: Vec<f64> = Vec::new();
        if self.nclients() > 0 {
            rates.push(self.preview_fps); // <= 0 means every frame
        }
        if self.software_h264() {
            rates.push(f64::from(H264_FPS));
        }
        let fastest = rates.iter().copied().fold(f64::MIN, f64::max);
        !rates.is_empty()
            && (rates.iter().any(|r| *r <= 0.0)
                || (monotonic() - *lock(&self.previewed)) >= 1.0 / fastest)
    }

    /// True when any viewer needs the host to draw: the preview, a still, or
    /// the hardware H264 lane, which encodes on the GPU after a draw.
    pub fn wants_draw(&self) -> bool {
        self.wants_preview() || self.shot.wants() || self.fanout.nclients() > 0
    }

    /// The size that the preview reads frames back at. One readback feeds
    /// the preview and the software encoder, so whoever watches sizes it.
    /// The preview downscales a bigger frame. The software encoder can't
    /// recover a smaller one.
    pub fn preview_size(&self) -> (i32, i32) {
        let (w, h) = (self.w, self.h);
        let scale = (f64::from(self.preview_max) / f64::from(w.max(h))).min(1.0);
        let pw = 1.max(round_half_even(f64::from(w) * scale) as i32);
        let ph = 1.max(round_half_even(f64::from(h) * scale) as i32);
        if self.software_h264() {
            let (sw, sh) = self.stream_size();
            return (pw.max(sw), ph.max(sh));
        }
        (pw, ph)
    }

    /// Hand a frame to the preview and the software encoder.
    pub fn publish_frame(&self, frame: Frame<'_>) {
        *lock(&self.previewed) = monotonic();
        let mut raw = lock(&self.raw);
        raw.rgb = Some(Arc::new(RawFrame::of(frame)));
        raw.seq += 1;
        self.raw_ready.notify_all();
    }

    // ------------------------------------------------------------ h264

    /// Software H264 reads back the same frames as the MJPEG preview, so its
    /// viewers count as readback consumers. Hardware samples the GPU texture
    /// instead.
    pub fn software_h264(&self) -> bool {
        self.software_stream().is_some()
    }

    /// The stream that the software lane owns, while a viewer watches it.
    pub fn software_stream(&self) -> Option<u64> {
        let stream = self.fanout.stream();
        let dead = *lock(&self.hw_dead) == Some(stream);
        (dead && self.fanout.nclients() > 0).then_some(stream)
    }

    /// True when hardware encoding failed for the current stream.
    pub fn hardware_dead(&self) -> bool {
        *lock(&self.hw_dead) == Some(self.fanout.stream())
    }

    /// The hardware lane failed. Before the stream's first access unit the
    /// software lane takes the stream, and after it the stream ends.
    pub fn hardware_failed(&self, why: &str) {
        if self.fanout.began() {
            self.say(&format!("h264: {why}"));
            self.fanout.fail(why);
            return;
        }
        *lock(&self.hw_why) = why.to_string();
        *lock(&self.hw_dead) = Some(self.fanout.stream());
        self.say(&format!("h264: {why}, using software encoder"));
    }

    // ------------------------------------------------------------ the end

    /// Hold the HTTP server, so stop() unblocks it.
    pub fn hold_server(&self, server: Arc<Server>) {
        *lock(&self.httpd) = Some(server);
    }

    /// Shut the server down, and signal the encode threads. It waits for no
    /// thread, because a request handler and a signal handler both call it.
    pub fn stop(&self) {
        self.stopping.store(true, Ordering::SeqCst);
        self.raw_ready.notify_all();
        self.latest_ready.notify_all();
        let httpd = lock(&self.httpd).take();
        if let Some(server) = httpd {
            server.unblock();
        }
    }
}

/// The side of a console that draws its frames, which the transport asks for
/// a frame.
pub trait Drawer: Send + Sync + 'static {
    fn hub(&self) -> &Hub;

    /// A stream opened or closed: draw a frame soon, for a host that rests
    /// while nobody watches.
    fn wake(&self) {}

    /// /frame.png asked for a still: arm it after every command queued
    /// before it.
    fn shot(&self) {}
}

/// Start the hub's threads: the preview encoder and the software H264 lane.
pub fn spawn_encoders<D: Drawer>(drawer: &Arc<D>) -> Vec<JoinHandle<()>> {
    let preview = {
        let drawer = Arc::clone(drawer);
        thread::spawn(move || encoder_loop(drawer.hub()))
    };
    let software = {
        let drawer = Arc::clone(drawer);
        thread::spawn(move || h264_sw_loop(drawer.hub()))
    };
    vec![preview, software]
}

/// Serve the console's requests on `server`, a thread for each, until the
/// hub stops.
pub fn serve<D: Drawer, H: ConsoleHost + Send + Sync + 'static>(
    server: Arc<Server>,
    drawer: Arc<D>,
    api: Arc<ConsoleAPI<H>>,
) -> JoinHandle<()> {
    drawer.hub().hold_server(Arc::clone(&server));
    thread::spawn(move || {
        for request in server.incoming_requests() {
            let (drawer, api) = (Arc::clone(&drawer), Arc::clone(&api));
            thread::spawn(move || handle(&*drawer, &api, request));
        }
    })
}

/// The console's addresses as URLs. A host that names every interface gives
/// one URL for each address of this machine that reaches it: every IPv4
/// address for 0.0.0.0, and every IPv6 address but a link-local one, then
/// every IPv4 address, for `::`.
pub fn urls(host: &str, port: u16) -> Vec<String> {
    let url = |ip: IpAddr| match ip {
        IpAddr::V4(ip) => format!("http://{ip}:{port}/"),
        IpAddr::V6(ip) => format!("http://[{ip}]:{port}/"),
    };
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(ip)) if ip.is_unspecified() => interface_addresses()
            .into_iter()
            .filter(IpAddr::is_ipv4)
            .map(url)
            .collect(),
        Ok(IpAddr::V6(ip)) if ip.is_unspecified() => {
            let all = interface_addresses();
            let v6 = all.iter().filter(|a| match a {
                IpAddr::V6(a) => (a.segments()[0] & 0xffc0) != 0xfe80,
                IpAddr::V4(_) => false,
            });
            v6.chain(all.iter().filter(|a| a.is_ipv4()))
                .copied()
                .map(url)
                .collect()
        }
        Ok(ip) => vec![url(ip)],
        Err(_) => vec![format!("http://{host}:{port}/")],
    }
}

/// The addresses of this machine's interfaces that are up, in the kernel's
/// order.
fn interface_addresses() -> Vec<IpAddr> {
    use std::ffi::{c_int, c_uint, c_void};

    #[repr(C)]
    struct IfAddrs {
        next: *mut IfAddrs,
        name: *const c_void,
        flags: c_uint,
        addr: *const u16,
        netmask: *const c_void,
        ifu: *const c_void,
        data: *const c_void,
    }
    extern "C" {
        fn getifaddrs(ifap: *mut *mut IfAddrs) -> c_int;
        fn freeifaddrs(ifa: *mut IfAddrs);
    }
    const IFF_UP: c_uint = 1;
    const AF_INET: u16 = 2;
    const AF_INET6: u16 = 10;

    let mut out = Vec::new();
    let mut head: *mut IfAddrs = std::ptr::null_mut();
    if unsafe { getifaddrs(&mut head) } != 0 {
        return out;
    }
    let mut at = head;
    while !at.is_null() {
        let ifa = unsafe { &*at };
        at = ifa.next;
        if ifa.flags & IFF_UP == 0 || ifa.addr.is_null() {
            continue;
        }
        // sockaddr_in holds the address at byte 4, sockaddr_in6 at byte 8.
        let sa = ifa.addr.cast::<u8>();
        match unsafe { ifa.addr.read_unaligned() } {
            AF_INET => {
                let a = unsafe { sa.add(4).cast::<[u8; 4]>().read_unaligned() };
                out.push(IpAddr::from(a));
            }
            AF_INET6 => {
                let a = unsafe { sa.add(8).cast::<[u8; 16]>().read_unaligned() };
                out.push(IpAddr::from(a));
            }
            _ => {}
        }
    }
    unsafe { freeifaddrs(head) };
    out
}

/// The port that `server` listens on, else `port`.
pub fn bound_port(server: &Server, port: u16) -> u16 {
    server.server_addr().to_ip().map_or(port, |a| a.port())
}

/// The line that a console prints when it starts: its address, its scenes,
/// where they come from, and the frame.
pub fn banner(
    urls: &[String],
    names: &[String],
    from: &str,
    size: (i32, i32),
    format: &str,
) -> String {
    let count = match names.len() {
        0 => "no scene".to_string(),
        1 => "1 scene".to_string(),
        n => format!("{n} scenes"),
    };
    let names = names
        .iter()
        .map(|n| format!("'{n}'"))
        .collect::<Vec<_>>()
        .join(", ");
    let from = match from {
        "" => String::new(),
        f => format!("{f}, "),
    };
    format!(
        "ghee console: {} {count} [{names}] ({from}preview {}x{} {format})",
        urls.join(" "),
        size.0,
        size.1
    )
}

// ---------------------------------------------------------------- preview

/// Encode the preview JPEGs, off the host's thread. The host drops its
/// readback here. This downscales a frame bigger than the preview cap, so the
/// display never waits. Raw frames also feed the h264 path, but a JPEG needs
/// an attached viewer.
pub fn encoder_loop(hub: &Hub) {
    let mut last_seq = u64::MAX;
    while !hub.stopping() {
        let rgb = {
            let mut raw = lock(&hub.raw);
            if hub.nclients() == 0 || raw.seq == last_seq {
                raw = hub
                    .raw_ready
                    .wait_timeout(raw, Duration::from_millis(250))
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
            }
            let fresh = hub.nclients() > 0 && raw.seq != last_seq;
            let rgb = if fresh { raw.rgb.clone() } else { None };
            last_seq = raw.seq;
            rgb
        };
        let Some(rgb) = rgb else { continue };
        let (w, h) = (rgb.width, rgb.height);
        let mut pixels = to_rgb(&rgb);
        let scale = f64::from(hub.preview_max) / f64::from(w.max(h));
        let (mut pw, mut ph) = (w, h);
        if scale < 1.0 {
            // the frame is sized for the software H264 stream, or for a bigger host
            pw = 1.max(round_half_even(f64::from(w) * scale) as i32);
            ph = 1.max(round_half_even(f64::from(h) * scale) as i32);
            pixels = resize(&pixels, (w, h), 3, (pw, ph));
        }
        let frame = Frame {
            data: &pixels,
            width: pw,
            height: ph,
            components: 3,
        };
        match encode_jpeg(frame, PREVIEW_QUALITY) {
            Ok(jpg) => {
                let mut latest = lock(&hub.latest);
                latest.jpg = Some(Arc::new(jpg));
                latest.seq += 1;
                hub.latest_ready.notify_all();
            }
            Err(e) => hub.say(&format!("preview: {e}")),
        }
        lock(&hub.stream_rate).tick(None);
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

/// Box-average resample of a frame of `c` components a pixel.
fn resize(data: &[u8], from: (i32, i32), c: usize, to: (i32, i32)) -> Vec<u8> {
    let (w, h) = (from.0 as usize, from.1 as usize);
    let (nw, nh) = (to.0 as usize, to.1 as usize);
    let mut out = vec![0u8; nw * nh * c];
    let mut sums = vec![0u32; c];
    for y in 0..nh {
        let y0 = y * h / nh;
        let y1 = (((y + 1) * h).div_ceil(nh)).max(y0 + 1).min(h);
        for x in 0..nw {
            let x0 = x * w / nw;
            let x1 = (((x + 1) * w).div_ceil(nw)).max(x0 + 1).min(w);
            sums.fill(0);
            let mut n = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let i = (sy * w + sx) * c;
                    for (k, sum) in sums.iter_mut().enumerate() {
                        *sum += u32::from(data[i + k]);
                    }
                    n += 1;
                }
            }
            let o = (y * nw + x) * c;
            for k in 0..c {
                out[o + k] = (sums[k] / n.max(1)) as u8;
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
    /// The reason this stream ends. Every viewer ends with it, and the last
    /// viewer to leave clears it.
    failure: Mutex<Option<String>>,
    /// An access unit reached a viewer since the first one joined.
    began: AtomicBool,
    /// The number of the current stream. The last viewer to leave ends a
    /// stream, and the next viewer joins the next number. It changes under
    /// the clients lock.
    stream: AtomicU64,
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
            failure: Mutex::new(None),
            began: AtomicBool::new(false),
            stream: AtomicU64::new(0),
            rate,
        }
    }

    /// End this stream for every viewer, with the reason. The first reason
    /// stands. The next stream tries every lane again.
    pub fn fail(&self, why: &str) {
        lock(&self.failure).get_or_insert_with(|| why.to_string());
        for c in lock(&self.clients).iter() {
            let _queue = lock(&c.state);
            c.ready.notify_all();
        }
    }

    pub fn failure(&self) -> Option<String> {
        lock(&self.failure).clone()
    }

    /// True once an access unit has reached a viewer of this stream. A lane
    /// that fails after that ends the stream.
    pub fn began(&self) -> bool {
        self.began.load(Ordering::SeqCst)
    }

    pub fn nclients(&self) -> usize {
        lock(&self.clients).len()
    }

    /// The number of the current stream.
    pub fn stream(&self) -> u64 {
        self.stream.load(Ordering::SeqCst)
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
        let mut clients = lock(&self.clients);
        clients.retain(|x| !Arc::ptr_eq(x, c));
        if clients.is_empty() {
            *lock(&self.failure) = None;
            self.began.store(false, Ordering::SeqCst);
            self.stream.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// The encoder that starts a stream states its size and its colorSpace
    /// hint. push() puts them in the config a joining viewer decodes with.
    pub fn opened(&self, w: i32, h: i32, color: ColorSpace) {
        *lock(&self.size) = (w, h);
        *lock(&self.color) = Some(color);
    }

    /// True once per keyframe request. Hardware forces one, and the software
    /// lane waits for the next GOP.
    pub fn wants_key(&self) -> bool {
        self.want_key.swap(false, Ordering::SeqCst)
    }

    pub fn push(&self, au: &[u8]) {
        self.push_in(None, au);
    }

    /// push() from a lane that encodes for one stream. An access unit that
    /// outlives `stream` reaches no viewer of the next one, and the result is
    /// false.
    pub fn push_to(&self, stream: u64, au: &[u8]) -> bool {
        self.push_in(Some(stream), au)
    }

    fn push_in(&self, stream: Option<u64>, au: &[u8]) -> bool {
        let clients = lock(&self.clients);
        if stream.is_some_and(|s| s != self.stream()) {
            return false;
        }
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
            return true;
        }
        let is_key = nals.iter().any(|x| x[0] & 0x1F == NAL_IDR);
        let frame = Arc::new(wire_frame(u8::from(!is_key) + 1, &chunk));
        let config = match (is_key, lock(&self.cfg).as_ref()) {
            (true, Some(cfg)) => Some(Arc::new(wire_frame(0, cfg))),
            _ => None,
        };
        for c in clients.iter() {
            c.offer(&frame, config.as_ref());
        }
        if !clients.is_empty() {
            self.began.store(true, Ordering::SeqCst);
            lock(&self.rate).tick(None);
        }
        // A waiting client rejoins at the next keyframe. Keep asking for one
        // until every client streams.
        if clients.iter().any(|c| !lock(&c.state).started) {
            self.want_key.store(true, Ordering::SeqCst);
        }
        true
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

/// What an encoder takes: stream_size(), never past the frame's size.
pub fn encode_size(w: i32, h: i32, longest: i32) -> (i32, i32) {
    let (sw, sh) = stream_size(w, h, longest);
    (sw.min((w & !1).max(2)), sh.min((h & !1).max(2)))
}

pub fn stream_bitrate(w: i32, h: i32, fps: i32, floor: i64) -> i64 {
    let derived = f64::from(w) * f64::from(h) * f64::from(fps.max(1)) * H264_BPP;
    floor.max((derived as i64).min(H264_BITRATE_CEILING))
}

/// Software H264, for a board with no /dev/video11. It encodes the frames
/// that the preview takes, at the stream's size, off the host's thread.
pub fn h264_sw_loop(hub: &Hub) {
    let mut enc: Option<SoftwareH264> = None;
    let mut size = (0, 0);
    // the stream that the encoder encodes for
    let mut owned = 0;
    let mut last_seq = u64::MAX;
    while !hub.stopping() {
        let stream = hub
            .software_stream()
            .filter(|_| hub.fanout.failure().is_none());
        let Some(stream) = stream else {
            if let Some(mut e) = enc.take() {
                let _ = e.close(Duration::from_secs(1));
            }
            let raw = lock(&hub.raw);
            let _ = hub.raw_ready.wait_timeout(raw, Duration::from_millis(250));
            continue;
        };
        let rgb = {
            let mut raw = lock(&hub.raw);
            if raw.seq == last_seq {
                raw = hub
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
        let (w, h) = hub.stream_size();
        if enc.is_some() && (size != (w, h) || owned != stream) {
            if let Some(mut e) = enc.take() {
                let _ = e.close(Duration::from_secs(1));
            }
        }
        if enc.is_none() {
            match SoftwareH264::new(
                w,
                h,
                H264_FPS,
                stream_bitrate(w, h, H264_FPS, hub.bitrate) as i32,
                rgb.components,
            ) {
                Ok(e) => {
                    enc = Some(e);
                    size = (w, h);
                    owned = stream;
                    hub.fanout.opened(w, h, SW_COLOR);
                }
                Err(e) => {
                    let why = no_encoder(&lock(&hub.hw_why), &e.to_string());
                    hub.say(&format!("h264: {why}"));
                    hub.fanout.fail(&why);
                    continue;
                }
            }
        }
        let e = enc.as_mut().expect("the encoder was just created");
        let from = (rgb.width, rgb.height);
        let fed = match from == (w, h) {
            true => e.feed(&rgb.data),
            false => e.feed(&resize(&rgb.data, from, rgb.components as usize, (w, h))),
        };
        if let Err(err) = fed {
            let why = format!("software encoder died: {err}");
            hub.say(&format!("h264: {why}"));
            hub.fanout.fail(&why);
            if let Some(mut e) = enc.take() {
                let _ = e.close(Duration::from_secs(1));
            }
            continue;
        }
        if let (Some(au), _) = e.get_au(Duration::from_secs(1)) {
            if !hub.fanout.push_to(owned, &au) {
                if let Some(mut e) = enc.take() {
                    let _ = e.close(Duration::from_secs(1));
                }
            }
        }
    }
    if let Some(mut e) = enc.take() {
        let _ = e.close(Duration::from_secs(1));
    }
}

/// The hardware H264 lane: encode each frame just drawn, with no readback.
/// The engine's keep_last holds the final pass in its FBO while the encoder
/// exists, so encode_last() always has a texture to sample. It issues GL
/// calls, so it runs on the host's thread.
#[derive(Default)]
pub struct HwLane {
    enc: Option<HardwareH264>,
    /// Encodes since the last access unit. An encoder can open at a size it
    /// never emits for, with no error to catch.
    starved: u32,
}

impl HwLane {
    /// Encode the frame that `renderer` just drew for the /h264 viewers of
    /// `hub`. A failure goes to the hub, which hands the stream to the
    /// software lane or ends it, and never to the frame.
    pub fn mirror(&mut self, hub: &Hub, renderer: &Renderer) {
        if hub.fanout.nclients() == 0 || hub.fanout.failure().is_some() {
            // an open encoder costs a display its direct scanout
            self.close();
            return;
        }
        if self.enc.is_none() && !hub.hardware_dead() {
            let (w, h) = hub.stream_size();
            // The stream encodes at its own rate, not the display's. A panel
            // that asks for 63Hz asks the encoder for a rate it refuses at
            // this size, and the input port then fails to enable: the whole
            // stream drops to software for a rate that no viewer needed.
            let hz = hub.stream_hz.max(1);
            let bitrate = stream_bitrate(w, h, hz, hub.bitrate) as i32;
            match HardwareH264::new(renderer, w, h, hz, bitrate) {
                Ok(enc) => {
                    hub.fanout.opened(enc.w, enc.h, HW_COLOR);
                    self.enc = Some(enc);
                    self.starved = 0;
                }
                Err(e) => hub.hardware_failed(&e.to_string()),
            }
        }
        let Some(enc) = self.enc.as_mut() else {
            return;
        };
        if hub.fanout.wants_key() {
            enc.force_keyframe();
        }
        let failed = match enc.encode_last() {
            Ok((Some(au), _)) => {
                self.starved = 0;
                hub.fanout.push(au);
                None
            }
            Ok((None, _)) => {
                self.starved += 1;
                (self.starved >= HW_STARVE_LIMIT)
                    .then(|| format!("hardware encoder gave no output in {HW_STARVE_LIMIT} frames"))
            }
            Err(e) => Some(e.to_string()),
        };
        if let Some(why) = failed {
            self.close();
            hub.hardware_failed(&why);
        }
    }

    pub fn close(&mut self) {
        if let Some(mut enc) = self.enc.take() {
            enc.close();
        }
    }
}

// ---------------------------------------------------------------- the still

/// One whole frame for a client that asks for a picture (GET /frame.png).
///
/// A request is a generation. request() hands one out, and the handler asks
/// the host for a still. The host arms the shot after every command queued
/// before it, and the next frame answers every generation asked for so far.
/// A /set that returned before the request lands before the arm, so that
/// frame carries it. A request made after the arm waits for the following
/// draw.
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

    /// The next frame answers every generation asked for so far.
    pub fn arm(&self) {
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

    pub fn write(&self, frame: Arc<RawFrame>) {
        let mut s = lock(&self.state);
        s.frame = Some(frame);
        s.done = s.armed;
        self.published.notify_all();
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
    for (name, value) in &resp.headers {
        if let Ok(h) = Header::from_bytes(name.as_bytes(), value.as_bytes()) {
            out = out.with_header(h);
        }
    }
    let _ = request.respond(out);
}

/// The status line and headers a stream writes onto the socket itself. The
/// stream frames the parts that follow.
pub fn stream_head(w: &mut dyn Write, ctype: &str, extra: &[&str]) -> std::io::Result<()> {
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
pub fn handle<D: Drawer + ?Sized, H: ConsoleHost>(
    drawer: &D,
    api: &ConsoleAPI<H>,
    mut request: Request,
) {
    let method = request.method().as_str().to_string();
    let url = request.url().to_string();
    let (path, qs) = url.split_once('?').unwrap_or((url.as_str(), ""));
    let headers = headers_of(&request);
    if let Some(resp) = api.precheck(&method, path, &headers) {
        return send(request, &resp);
    }
    let query = match parse_query(qs) {
        Ok(q) => q,
        Err(e) => return send(request, &Response::text(e.status, &e.message)),
    };
    if method == "HEAD" {
        let streamed = match path {
            "/stream" => Some("multipart/x-mixed-replace; boundary=frame"),
            "/h264" => Some("application/octet-stream"),
            _ => None,
        };
        if let Some(ctype) = streamed {
            let mut head = Response::text(200, "");
            head.content_type = ctype.to_string();
            return send(request, &head);
        }
    }
    if method == "GET" || method == "HEAD" {
        match path {
            "/stream" => return stream(drawer, request, &query),
            "/h264" => return h264(drawer, request, &query),
            "/frame.png" => return frame_png(drawer, request),
            _ => {}
        }
        let resp = api.dispatch(&method, path, &query, &headers, b"");
        return send(request, &resp);
    }
    if method == "POST" && path == "/upload" {
        let resp = api.upload(&query, request.as_reader());
        return send(request, &resp);
    }
    let mut body: Vec<u8> = Vec::new();
    if method == "POST" {
        body = match api.read_body(request.as_reader()) {
            Ok(body) => body,
            Err(refused) => return send(request, &refused),
        };
        if path == "/ping" {
            drawer
                .hub()
                .viewers
                .ping(query.get("vid").map(String::as_str));
            return send(request, &Response::text(200, "ok"));
        }
    }
    let resp = api.dispatch(&method, path, &query, &headers, &body);
    send(request, &resp);
}

/// One JPEG part for each JPEG that the preview encodes.
fn stream<D: Drawer + ?Sized>(drawer: &D, request: Request, query: &Query) {
    let hub = drawer.hub();
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
    let viewer = hub.viewers.join(query.get("vid").map(String::as_str));
    hub.count_clients(1);
    drawer.wake();
    let mut last_seq = u64::MAX;
    while !hub.stopping() {
        if hub.viewers.gone(&viewer) {
            break;
        }
        let (jpg, seq) = {
            let mut latest = lock(&hub.latest);
            if latest.seq == last_seq {
                latest = hub
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
    hub.count_clients(-1);
    hub.viewers.leave(&viewer);
}

/// One chunk of a chunked response.
fn write_chunk(w: &mut dyn Write, blob: &[u8]) -> std::io::Result<()> {
    w.write_all(format!("{:X}\r\n", blob.len()).as_bytes())?;
    w.write_all(blob)?;
    w.write_all(b"\r\n")?;
    w.flush()
}

/// Serve one /h264 viewer from the fanout while `alive` holds. The status line
/// waits for the viewer's first blob, so a stream that cannot start answers
/// 503 with the reason. A stream that fails later ends on a WIRE_ERROR frame.
pub fn serve_h264(request: Request, fanout: &H264Fanout, c: &H264Client, alive: &dyn Fn() -> bool) {
    let first = loop {
        if let Some(why) = fanout.failure() {
            return send(request, &Response::text(503, &why));
        }
        if !alive() {
            return send(
                request,
                &Response::text(503, "the stream closed before its first frame"),
            );
        }
        if let Some(blob) = c.get(Duration::from_secs(1)) {
            break blob;
        }
    };
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
    let mut next = Some(first);
    loop {
        let blob = match next.take() {
            Some(blob) => blob,
            None => {
                if let Some(why) = fanout.failure() {
                    let _ = write_chunk(&mut w, &wire_frame(WIRE_ERROR, why.as_bytes()));
                    break;
                }
                if !alive() {
                    break;
                }
                match c.get(Duration::from_secs(1)) {
                    Some(blob) => blob,
                    None => continue,
                }
            }
        };
        if write_chunk(&mut w, &blob).is_err() {
            break;
        }
    }
    // The terminal chunk ends the response. Without it, keep-alive parks the
    // socket and a viewer waits forever on an abandoned stream. A clean end is
    // what sends the client to its MJPEG fallback.
    let _ = w.write_all(b"0\r\n\r\n");
    let _ = w.flush();
}

/// Framed H264 for WebCodecs, docs/http.md holds the wire. The encoder runs
/// while a viewer holds this request open.
fn h264<D: Drawer + ?Sized>(drawer: &D, request: Request, query: &Query) {
    let hub = drawer.hub();
    let c = hub.fanout.add();
    let viewer = hub.viewers.join(query.get("vid").map(String::as_str));
    drawer.wake();
    let alive = || !hub.stopping() && !hub.viewers.gone(&viewer);
    serve_h264(request, &hub.fanout, &c, &alive);
    hub.fanout.remove(&c);
    hub.viewers.leave(&viewer);
    // The encoder tears down on the next drawn frame, and a clean scene skips
    // that frame. An open encoder holds keep_last, which costs the display its
    // direct scanout while nobody watches.
    drawer.wake();
}

/// One lossless PNG of the next drawn frame.
fn frame_png<D: Drawer + ?Sized>(drawer: &D, request: Request) {
    let hub = drawer.hub();
    // A scene that does not compile has no frame. Say so, instead of answering
    // with the scene before it.
    let error = hub.error();
    if !error.is_empty() {
        return send(
            request,
            &Response::text(409, &format!("scene error: {}", line(&error))),
        );
    }
    let gen = hub.shot.request();
    hub.shot.hold();
    drawer.shot();
    let frame = hub.shot.wait(gen, FRAME_TIMEOUT);
    let png = frame.as_ref().map(|f| encode_png(f.view()));
    hub.shot.release();
    let png = match png {
        Some(Ok(png)) => png,
        other => {
            let fatal = hub.fatal();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{until, Clock, NEVER};

    #[test]
    fn a_console_on_every_interface_prints_the_addresses_that_reach_it() {
        let all = urls("0.0.0.0", 8090);
        assert!(
            all.contains(&"http://127.0.0.1:8090/".to_string()),
            "{all:?}"
        );
        assert!(all
            .iter()
            .all(|u| !u.contains("localhost") && !u.contains('[')));
        let v6 = urls("::", 8090);
        assert!(v6.iter().all(|u| !u.starts_with("http://[fe80")), "{v6:?}");
        assert!(v6.contains(&"http://127.0.0.1:8090/".to_string()), "{v6:?}");
        assert_eq!(urls("127.0.0.1", 1), ["http://127.0.0.1:1/"]);
        assert_eq!(urls("::1", 1), ["http://[::1]:1/"]);
        assert_eq!(urls("pi.local", 1), ["http://pi.local:1/"]);
    }

    const SC: &[u8] = &[0, 0, 0, 1];

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
    fn a_configured_rate_above_the_ceiling_wins() {
        assert_eq!(stream_bitrate(2048, 2048, 30, 100_000_000), 100_000_000);
    }

    #[test]
    fn a_stream_keeps_its_aspect_and_stays_even() {
        assert_eq!(stream_size(320, 240, 0), (320, 240));
        assert_eq!(stream_size(320, 240, 160), (160, 120));
        assert_eq!(stream_size(101, 33, 0), (100, 32));
        assert_eq!(stream_size(2, 2, 1), (2, 2));
    }

    // the preview

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

    fn feed_encoder(hub: &Arc<Hub>, frame: Arc<RawFrame>) -> Arc<Vec<u8>> {
        {
            let mut raw = lock(&hub.raw);
            raw.rgb = Some(frame);
            raw.seq += 1;
            hub.raw_ready.notify_all();
        }
        until(|| lock(&hub.latest).jpg.is_some());
        lock(&hub.latest).jpg.clone().unwrap()
    }

    #[test]
    fn a_frame_bigger_than_the_preview_cap_is_downscaled() {
        let mut hub = Hub::new();
        hub.preview_max = 120;
        let hub = Arc::new(hub);
        hub.count_clients(1); // the encoder only works while a viewer is counted
        let worker = {
            let hub = Arc::clone(&hub);
            thread::spawn(move || encoder_loop(&hub))
        };
        let jpg = feed_encoder(&hub, test_frame(480, 240));
        assert_eq!(jpeg_size(&jpg), (120, 60));
        hub.stop();
        hub.raw_ready.notify_all();
        worker.join().unwrap();
    }

    #[test]
    fn a_frame_inside_the_cap_keeps_its_size() {
        let hub = Arc::new(Hub::new());
        hub.count_clients(1);
        let worker = {
            let hub = Arc::clone(&hub);
            thread::spawn(move || encoder_loop(&hub))
        };
        let jpg = feed_encoder(&hub, test_frame(96, 96));
        assert_eq!(jpeg_size(&jpg), (96, 96));
        hub.stop();
        hub.raw_ready.notify_all();
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
        // the JPEG path ticks the counter published as stream_fps
        let hub = Arc::new(Hub::new());
        assert!(Arc::ptr_eq(&hub.fanout.rate, &hub.stream_rate));
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

    #[test]
    fn a_failure_wakes_a_waiting_viewer_and_the_first_reason_stands() {
        let fanout = Arc::new(H264Fanout::new(Arc::new(Mutex::new(FrameRate::new()))));
        let c = fanout.add();
        let woke = Arc::new(AtomicBool::new(false));
        // A fail before the viewer waits wakes nothing, so the fails repeat
        // until the viewer wakes.
        let failer = {
            let (fanout, woke) = (Arc::clone(&fanout), Arc::clone(&woke));
            thread::spawn(move || {
                while !woke.load(Ordering::SeqCst) {
                    fanout.fail("first");
                    thread::yield_now();
                }
                fanout.fail("second");
            })
        };
        assert!(c.get(NEVER).is_none());
        woke.store(true, Ordering::SeqCst);
        failer.join().unwrap();
        assert_eq!(fanout.failure().as_deref(), Some("first"));
    }

    #[test]
    fn the_last_viewer_to_leave_clears_the_failure_and_the_start() {
        let fanout = H264Fanout::new(Arc::new(Mutex::new(FrameRate::new())));
        let (a, b) = (fanout.add(), fanout.add());
        assert!(!fanout.began());
        fanout.push(&key_au());
        assert!(fanout.began());
        fanout.fail("gone");
        fanout.remove(&a);
        assert!(fanout.failure().is_some() && fanout.began());
        fanout.remove(&b);
        assert!(fanout.failure().is_none() && !fanout.began());
    }

    #[test]
    fn a_late_unit_of_an_ended_stream_never_ends_the_next_one() {
        let hub = Arc::new(Hub::new());
        let first = hub.fanout.add();
        hub.hardware_failed("no hardware encoder");
        let owned = hub.software_stream().unwrap();
        hub.fanout.remove(&first);
        let next = hub.fanout.add();
        // the software encoder of the first stream hands over one more unit
        assert!(!hub.fanout.push_to(owned, &key_au()));
        assert!(next.get(Duration::ZERO).is_none());
        assert!(!hub.fanout.began());
        // the next stream tries hardware first, and the software lane takes it
        // when hardware fails
        assert!(!hub.software_h264());
        hub.hardware_failed("no hardware encoder");
        assert_eq!(hub.fanout.failure(), None);
        assert_eq!(hub.software_stream(), Some(hub.fanout.stream()));
        assert!(hub.fanout.push_to(hub.fanout.stream(), &key_au()));
        assert!(next.get(Duration::ZERO).is_some());
    }

    /// One GET /h264 served by serve_h264, as the raw bytes on the socket.
    /// `run` drives the fanout and reads the viewer's queue while the request
    /// is served.
    fn h264_exchange(
        fanout: Arc<H264Fanout>,
        run: impl FnOnce(&H264Fanout, &H264Client) + Send + 'static,
    ) -> Vec<u8> {
        use std::io::Read;
        use std::net::TcpStream;
        let server = Server::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        let client = thread::spawn(move || {
            let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
            s.write_all(b"GET /h264 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
                .unwrap();
            let mut out = Vec::new();
            let _ = s.read_to_end(&mut out); // the stream ends on its own terminal chunk
            out
        });
        let request = server.recv().unwrap();
        let c = fanout.add();
        let driver = {
            let (fanout, c) = (Arc::clone(&fanout), Arc::clone(&c));
            thread::spawn(move || run(&fanout, &c))
        };
        serve_h264(request, &fanout, &c, &|| true);
        fanout.remove(&c);
        driver.join().unwrap();
        client.join().unwrap()
    }

    /// The body of a chunked response, and whether its terminal chunk arrived.
    fn dechunk(body: &[u8]) -> (Vec<u8>, bool) {
        let mut out = Vec::new();
        let mut at = 0;
        while let Some(end) = find(body, b"\r\n", at) {
            let n =
                usize::from_str_radix(std::str::from_utf8(&body[at..end]).unwrap(), 16).unwrap();
            if n == 0 {
                return (out, true);
            }
            out.extend_from_slice(&body[end + 2..end + 2 + n]);
            at = end + 2 + n + 2;
        }
        (out, false)
    }

    fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
        hay.get(from..)?
            .windows(needle.len())
            .position(|w| w == needle)
            .map(|i| i + from)
    }

    #[test]
    fn a_stream_that_cannot_start_answers_503_with_the_reason() {
        let fanout = Arc::new(H264Fanout::new(Arc::new(Mutex::new(FrameRate::new()))));
        let why = "no H264 encoder. hardware: none here. software: no ffmpeg";
        let out = h264_exchange(fanout, move |f, _| f.fail(why));
        let text = String::from_utf8_lossy(&out);
        assert!(text.starts_with("HTTP/1.1 503"), "{text}");
        assert!(text.ends_with(why), "{text}");
    }

    #[test]
    fn a_stream_that_fails_after_its_start_ends_on_an_error_frame() {
        let fanout = Arc::new(H264Fanout::new(Arc::new(Mutex::new(FrameRate::new()))));
        let why = "hardware encode failed: stalled";
        let out = h264_exchange(fanout, move |f, c| {
            f.opened(256, 64, HW_COLOR);
            f.push(&key_au());
            // the stream starts once the viewer took the config and the key
            until(|| lock(&c.state).q.is_empty());
            f.fail(why);
        });
        let head_end = find(&out, b"\r\n\r\n", 0).unwrap();
        let head = String::from_utf8_lossy(&out[..head_end]);
        assert!(head.starts_with("HTTP/1.1 200"), "{head}");
        let (body, ended) = dechunk(&out[head_end + 4..]);
        assert!(ended);
        let mut msgs = Vec::new();
        let mut at = 0;
        while at < body.len() {
            let n = u32::from_be_bytes(body[at + 1..at + 5].try_into().unwrap()) as usize;
            msgs.push((body[at], body[at + 5..at + 5 + n].to_vec()));
            at += 5 + n;
        }
        let types: Vec<u8> = msgs.iter().map(|m| m.0).collect();
        assert_eq!(types, vec![0, 1, WIRE_ERROR]);
        let cfg: Value = serde_json::from_slice(&msgs[0].1).unwrap();
        assert_eq!(
            (cfg["codedWidth"].as_i64(), cfg["codedHeight"].as_i64()),
            (Some(256), Some(64))
        );
        assert_eq!(msgs[2].1, why.as_bytes());
    }

    #[test]
    fn the_page_ends_its_stream_on_the_error_frame() {
        let js = include_str!("../../ghee/console/web/h264client.js");
        assert!(js.contains(&format!("type === {WIRE_ERROR}")));
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
        assert!(shot.wait(gen, NEVER.as_secs_f64()).is_none());
    }

    // viewers

    #[test]
    fn a_connection_that_named_no_vid_never_expires() {
        let clock = Clock::at(0.0);
        let hb = Heartbeat::new(0.0);
        let v = hb.join(None);
        clock.pass(10.0);
        assert!(!hb.gone(&v));
    }

    #[test]
    fn a_named_connection_expires_once_its_pings_stop() {
        let clock = Clock::at(0.0);
        let hb = Heartbeat::new(0.05);
        let v = hb.join(Some("abc"));
        clock.pass(0.05);
        assert!(!hb.gone(&v));
        clock.pass(0.01);
        assert!(hb.gone(&v));
    }

    #[test]
    fn a_ping_puts_a_named_connection_back_inside_the_deadline() {
        let clock = Clock::at(0.0);
        let hb = Heartbeat::new(0.05);
        let v = hb.join(Some("abc"));
        clock.pass(0.08);
        assert_eq!(hb.ping(Some("abc")), 1);
        clock.pass(0.04);
        assert!(!hb.gone(&v));
    }

    #[test]
    fn every_connection_under_one_vid_is_kept_alive_together() {
        // a page that falls back from h264 to MJPEG opens a second stream under
        // its existing ping id
        let clock = Clock::at(0.0);
        let hb = Heartbeat::new(0.05);
        let (a, b) = (hb.join(Some("abc")), hb.join(Some("abc")));
        clock.pass(0.08);
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
        let hub = Arc::new(Hub::new());
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let hub = Arc::clone(&hub);
                thread::spawn(move || {
                    for _ in 0..200 {
                        hub.count_clients(1);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(hub.nclients(), 8 * 200);
        for _ in 0..8 * 200 {
            hub.count_clients(-1);
        }
        assert_eq!(hub.nclients(), 0);
    }
}
