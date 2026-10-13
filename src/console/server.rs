//! ghee console render server.
//!
//! A host is the program that runs the renderer: the `ghee` command, or a
//! program that embedded it. The host runs a render loop (looper), one
//! thread that draws every frame and hands it to the host's sinks: the
//! display, files, a pipe. This module attaches the console to that loop.
//!
//! The console serves the HTTP API (`docs/http.md`) and the web page on a
//! thread of its own. A request handler never touches the renderer: it
//! queues a command, a param set, a scene switch, an edited scene, and the
//! render thread applies the queue between frames.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::ffi::c_int;
use std::io::Write;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use serde_json::{Map, Value};
use tiny_http::{Header, Request, Server, StatusCode};

use crate::console::core::{
    other_reload, other_scene, parse_query, Aside, ConsoleAPI, ConsoleError, ConsoleHost,
    ConsoleResult, Edit, Headers, LayerSet, Query, Response, Scope,
};
use crate::console::state::{self, Edits, Item};
use crate::console::store::{self, Carried, Opened, SceneStore};
use crate::console::{examples_dir, is_examples_dir};
use crate::draw::Call;
use crate::engine::{Card, ErrorPass, Feed, Given, Provider, Renderer, Taken};
use crate::feeds::demo_feed;
use crate::h264::{
    avcc, build_config, no_encoder, split_annexb, wire_frame, ColorSpace, HardwareH264,
    SoftwareH264, HW_COLOR, NAL_IDR, NAL_PPS, NAL_SEI, NAL_SLICE, NAL_SPS, SW_COLOR, WIRE_ERROR,
};
use crate::layer::Address;
use crate::looper::{ControllerSlot, Frame, Loop, Output, Pacer, RenderTarget, SceneController};
use crate::output::{encode_jpeg, encode_png, Scanout};
use crate::procedural::process::Locks;
use crate::scene::{ArraySize, Param, ParamValue, Reach, Scene};
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

/// How long GET /scene and GET /edits wait for the active scene's load. A
/// procedural scene's program takes up to 30 s to hand over its scene.
pub const LOAD_TIMEOUT: Duration = Duration::from_secs(35);

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
/// document for the changes that a scene's own code makes to its layers.
/// Writing the document takes time with each layer, and the code may change
/// a layer every frame.
pub const LAYERS_PUBLISH_EVERY: f64 = 0.2;

/// run() sets this. stop() ends that console from anywhere.
static CONSOLE: Mutex<Option<Arc<Console>>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

extern "C" {
    fn malloc_trim(pad: usize) -> c_int;
}

/// Give the heap that a scene setup freed back to the system. glibc keeps
/// freed memory in the process until a trim.
fn trim_heap() {
    unsafe { malloc_trim(0) };
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
/// `name` names the param in an error.
pub fn coerce(name: &str, ptype: &str, val: &Value) -> Result<ParamValue> {
    let built = if val.is_array() {
        Param::new(name, ptype, val.clone(), None, None, None, None)
    } else if ptype == "int" {
        Param::new(
            name,
            ptype,
            Value::from(round_half_even(as_f64(val)) as i64),
            None,
            None,
            None,
            None,
        )
    } else if ptype == "bool" {
        Param::new(
            name,
            ptype,
            Value::from(truthy(val)),
            None,
            None,
            None,
            None,
        )
    } else {
        Param::new(name, ptype, val.clone(), None, None, None, None)
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
    /// A scene to load, with the number of the switch that asked for it and
    /// the vid of the page that asked.
    Switch(String, u64, String),
    Param(i64, String, Value),
    SceneParam(String, Value),
    Layer(LayerSet),
    /// GET /frame.png: arm the still where this item lands in queue order.
    Shot,
    /// A working copy to compile onto the running renderer, with its
    /// generation, and true when its passes start from black, as on a reset.
    SceneObj(Box<Scene>, u64, bool),
    /// Draw one frame, for a stream that just opened or closed.
    Preview,
    /// No scene: the renderer draws black.
    Clear,
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

/// The active scene: the one that the last switch asked for, which an edit
/// must name, and how far the render thread got with its load.
#[derive(Default)]
struct Active {
    name: Option<String>,
    /// The switches asked for.
    asked: u64,
    /// The last switch whose load ended, whether or not its scene compiled.
    loaded: u64,
    /// Why that load made no working copy.
    failed: Option<String>,
}

impl Active {
    fn is_loading(&self) -> bool {
        self.name.is_some() && self.asked != self.loaded
    }

    /// The name of the active scene, or why an edit cannot land on it: the
    /// 503 for an edit that arrives before its load ended, which would land
    /// on the scene before it, and the load's own error.
    fn editable(&self) -> ConsoleResult<String> {
        let Some(name) = &self.name else {
            return Err(ConsoleError::not_found("no scene"));
        };
        if self.is_loading() {
            return Err(ConsoleError::unavailable(format!(
                "scene '{name}' is loading"
            )));
        }
        if let Some(why) = &self.failed {
            return Err(ConsoleError::server_error(why.clone()));
        }
        Ok(name.clone())
    }
}

/// One scene as the console holds it since its last load: its defaults, its
/// working copy, and what its state file holds.
struct WorkingCopy {
    opened: Opened,
    /// opened.defaults.to_dict(), which every diff reads.
    defaults_doc: Value,
    /// The scene's revision: every change applied through the API counts.
    rev: u64,
    /// The page that made the last change, by its vid, and the revision of
    /// the last change that another client made.
    by: String,
    before: u64,
    /// The changes to the working copy that the render thread compiles anew:
    /// a load, an apply, a reset, a kept item. A procedural scene's running
    /// document carries this working copy once it carries this count.
    generation: u64,
}

impl WorkingCopy {
    fn new(opened: Opened, rev: u64, generation: u64) -> WorkingCopy {
        WorkingCopy {
            defaults_doc: opened.defaults.to_dict(),
            opened,
            rev,
            by: String::new(),
            before: 0,
            generation,
        }
    }

    /// Count a change made by the page `vid`.
    fn bump(&mut self, vid: &str) {
        if self.by != vid {
            self.before = self.rev;
            self.by = vid.to_string();
        }
        self.rev += 1;
    }

    /// Whether another client changed the scene after revision `rev`, which
    /// the page `vid` last read. A page's own changes never make its `rev`
    /// stale, so a slider's sets, sent before the last one answers, all land.
    fn stale(&self, rev: u64, vid: &str) -> bool {
        match self.by == vid {
            true => self.before > rev,
            false => self.rev > rev,
        }
    }

    fn working_doc(&self) -> Value {
        self.opened.working.to_dict()
    }

    /// The working copy's edits, with no value of what `locks` holds.
    fn edits(&self, locks: Option<&Locks>) -> Edits {
        let mut edits = state::diff(
            &self.defaults_doc,
            &self.working_doc(),
            self.opened.procedural,
        );
        if let Some(locks) = locks {
            leave_out(&mut edits.values, locks);
        }
        edits
    }

    /// What a save writes: the working copy's edits, the items set aside,
    /// and beneath them the saved edits that make no scene.
    fn to_save(&self, locks: Option<&Locks>) -> Edits {
        let mut edits = self.edits(locks);
        edits.items.extend(self.opened.aside.iter().cloned());
        for b in &self.opened.broken {
            edits.add_beneath(&b.edit);
        }
        edits
    }

    fn unsaved(&self, locks: Option<&Locks>) -> bool {
        let edits = self.to_save(locks);
        match &self.opened.saved {
            Some(saved) => !edits.same(saved),
            None => !edits.is_empty(),
        }
    }

    fn carried(&self) -> Carried {
        Carried {
            edits: self.to_save(None),
            saved: self.opened.saved.clone(),
        }
    }

    /// GET /edits.
    fn edits_json(&self, locks: Option<&Locks>) -> Value {
        let edits = self.edits(locks);
        serde_json::json!({
            "rev": self.rev,
            "values": edits.values,
            "items": edits.items.iter().map(Item::to_json).collect::<Vec<_>>(),
            "unsaved": self.unsaved(locks),
            "aside": self.opened.aside.iter().map(Item::key).collect::<Vec<_>>(),
            "broken": self.opened.broken.iter()
                .map(|b| serde_json::json!({"what": b.edit.what(), "why": b.why}))
                .collect::<Vec<_>>(),
        })
    }

    /// The document that a page reads and a /set checks against: the
    /// working copy, or for a procedural scene the document that runs, which
    /// holds what its program drew and set, once it carries this working copy.
    fn live_doc(&self, name: &str, running: &Option<Published>) -> Value {
        match running {
            Some(p)
                if self.opened.procedural
                    && p.name.as_deref() == Some(name)
                    && p.generation == self.generation =>
            {
                p.doc.clone()
            }
            _ => self.working_doc(),
        }
    }
}

/// Take out of `values`, the values of a scene's edits, each param, layer
/// and live field that `locks` holds, and the removal of an effect that
/// holds such a field.
pub fn leave_out(values: &mut Map<String, Value>, locks: &Locks) {
    fn take(map: &mut Map<String, Value>, mut out: impl FnMut(&str, &mut Value) -> bool) -> bool {
        let had = map.len();
        map.retain(|k, v| !out(k, v));
        had > 0 && map.is_empty()
    }
    if let Some(Value::Object(params)) = values.get_mut("params") {
        if take(params, |name, _| locks.param(None, name)) {
            values.remove("params");
        }
    }
    if let Some(Value::Object(passes)) = values.get_mut("passes") {
        let gone = take(passes, |pass, v| {
            let Some(Value::Object(params)) = v.get_mut("params") else {
                return false;
            };
            if !take(params, |name, _| locks.param(Some(pass), name)) {
                return false;
            }
            let Value::Object(fields) = v else {
                return false;
            };
            fields.remove("params");
            fields.is_empty()
        });
        if gone {
            values.remove("passes");
        }
    }
    if let Some(Value::Object(layers)) = values.get_mut("layers") {
        let gone = take(layers, |layer, v| {
            let Value::Object(fields) = v else {
                return locks.layer(layer);
            };
            locks.layer(layer)
                || take(fields, |key, value| {
                    let own = Address {
                        layer,
                        effect: None,
                        field: key,
                    };
                    match value {
                        _ if locks.field(&own) => true,
                        Value::Null => locks.effect(layer, key),
                        Value::Object(sub) => take(sub, |field, _| {
                            let at = Address {
                                layer,
                                effect: Some(key),
                                field,
                            };
                            locks.field(&at)
                        }),
                        _ => false,
                    }
                })
        });
        if gone {
            values.remove("layers");
        }
    }
}

/// The document of the scene that the render thread set last.
struct Published {
    name: Option<String>,
    /// The generation of the working copy behind it.
    generation: u64,
    doc: Value,
}

/// A source for one declared array: the host's data for this frame.
pub type ArraySource = Box<dyn Fn(ArraySize, u64) -> Vec<f32> + Send + Sync>;

/// The item of POST /aside that names the saved edits that make no scene.
pub const BROKEN: &str = "edits";

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
    /// The degrees clockwise that every output turns the frame.
    pub rotate: i64,
    /// The backlight level that the display takes at the start, 0 to 1.
    pub brightness: Option<f64>,
    /// The longest side of the browser preview. The encoder thread downscales it.
    pub preview_max: i32,
    /// The browser preview refresh cap. The display stays at full speed.
    pub preview_fps: f64,
    /// The cap on the h264 stream's longest side. 0 means full size.
    pub stream_max: i32,
    pub stream_hz: i32,
    pub bitrate: i64,
    /// The frame components the renderer reads back: 3 for RGB, 4 for RGBA.
    pub components: i32,
    /// The readback keeps the output pass's alpha.
    pub alpha: bool,
    /// 0 renders until stop(). A count ends the console with the render.
    pub frames: u64,
    /// Pin engine time to frame * this, for a reproducible render.
    pub timestep: Option<f64>,
    /// The DRI device override. None probes for one.
    pub dri: Option<String>,
    /// The /stats destinations of the caller's sinks. A sink itself belongs to
    /// the render thread.
    pub sink_dests: Vec<String>,
    /// The interval a stale check reads disk at. None takes the default window.
    pub stale_interval: Option<f64>,
    arrays: Vec<(String, ArraySource)>,

    // state
    stat: Mutex<Stat>,
    active: Mutex<Active>,
    /// A handler holds this over a switch, and over an edit's check of the
    /// active scene and the edit, so an edit queued behind a switch never
    /// lands on the scene that the switch brought.
    editing: Mutex<()>,
    /// Signalled at the end of each load.
    loaded: Condvar,
    /// The working copy of each scene that the console loaded since it
    /// started, by name.
    copies: Mutex<HashMap<String, WorkingCopy>>,
    /// The `rev` of each working copy in `copies`. /stats reads it here,
    /// since an apply loads its assets and a save writes its file with
    /// `copies` held.
    revs: Mutex<HashMap<String, u64>>,
    /// What the process of the scene named beside it set, while it runs.
    locks: Mutex<Option<(String, Arc<Mutex<Locks>>)>>,
    /// The document of the active scene as the render thread last set it,
    /// under the scene's name. A scene that failed to compile is published
    /// too, while the renderer keeps drawing the last one that compiled.
    running: Mutex<Option<Published>>,
    /// The times that the scene's own code changed its layers in the
    /// document: a layer added or given a new look, or a field set.
    layer_changes: AtomicU64,
    /// The scene_size() of the renderer that render_loop draws with, from the
    /// start of the loop.
    scene_size: Mutex<Option<(i32, i32)>>,
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
    /// The stream whose hardware encoding failed, which the software lane
    /// owns. A stall or a busy codec belongs to the stream it happened in, so
    /// the next stream tries hardware again, and the software lane costs a
    /// readback.
    h264_hw_dead: Mutex<Option<u64>>,
    /// Why the hardware lane failed this stream.
    h264_hw_why: Mutex<String>,
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
            rotate: 0,
            brightness: None,
            preview_max: 480,
            preview_fps: 15.0,
            stream_max: 0,
            stream_hz: 30,
            bitrate: 10_000_000,
            components: 4,
            alpha: false,
            frames: 0,
            timestep: None,
            dri: None,
            sink_dests: Vec::new(),
            stale_interval: None,
            arrays: Vec::new(),
            stat: Mutex::new(Stat {
                display: Value::Object(Map::new()),
                gpu: Value::Object(Map::new()),
                ..Stat::default()
            }),
            active: Mutex::new(Active::default()),
            editing: Mutex::new(()),
            loaded: Condvar::new(),
            copies: Mutex::new(HashMap::new()),
            revs: Mutex::new(HashMap::new()),
            locks: Mutex::new(None),
            running: Mutex::new(None),
            layer_changes: AtomicU64::new(0),
            scene_size: Mutex::new(None),
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
            h264_hw_dead: Mutex::new(None),
            h264_hw_why: Mutex::new(String::new()),
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

    /// The /stats payload. Each rate computes on read, so an idle loop's number
    /// falls with the clock.
    pub fn stats(&self) -> Value {
        let mut d = Map::new();
        {
            let stat = lock(&self.stat);
            d.insert("frame_ms".to_string(), Value::from(stat.frame_ms));
            d.insert("error".to_string(), Value::from(stat.error.clone()));
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
        // `rev` is the revision of the scene that `active` names.
        let active = lock(&self.active).name.clone();
        d.insert(
            "active".to_string(),
            active.clone().map_or(Value::Null, Value::from),
        );
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
        d.insert(
            "layer_changes".to_string(),
            Value::from(self.layer_changes.load(Ordering::SeqCst)),
        );
        let locked = active
            .as_deref()
            .and_then(|name| self.with_locks(name, |locks| locks.map(Locks::to_json)));
        d.insert(
            "locked".to_string(),
            locked.unwrap_or_else(|| Locks::default().to_json()),
        );
        let rev = active.and_then(|name| lock(&self.revs).get(&name).copied());
        d.insert("rev".to_string(), rev.map_or(Value::Null, Value::from));
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

    /// Republish the running document of scene `name`, built from the
    /// working copy of `generation`. The render thread calls this after a
    /// load, an applied scene, or a /set that landed.
    fn publish(&self, name: Option<&str>, generation: u64, scene: &Scene) {
        *lock(&self.running) = Some(Published {
            name: name.map(str::to_string),
            generation,
            doc: scene.to_dict(),
        });
    }

    /// Hold `opened` as the working copy of scene `name`, loaded for the page
    /// `vid`, with the next revision and generation, and return that
    /// generation.
    fn install(&self, name: &str, opened: Opened, vid: &str) -> u64 {
        let mut copies = lock(&self.copies);
        let mut copy = WorkingCopy::new(opened, 0, 0);
        if let Some(old) = copies.get(name) {
            (copy.rev, copy.by, copy.before) = (old.rev, old.by.clone(), old.before);
            copy.generation = old.generation;
        }
        self.bump(name, &mut copy, vid);
        copy.generation += 1;
        let generation = copy.generation;
        copies.insert(name.to_string(), copy);
        generation
    }

    /// Count a change to `copy`, the working copy of scene `name`, made by
    /// the page `vid`.
    fn bump(&self, name: &str, copy: &mut WorkingCopy, vid: &str) {
        copy.bump(vid);
        lock(&self.revs).insert(name.to_string(), copy.rev);
    }

    /// Run `f` on the working copy of the active scene, or say why no edit
    /// lands on it.
    fn with_active<T>(
        &self,
        f: impl FnOnce(&str, &mut WorkingCopy) -> ConsoleResult<T>,
    ) -> ConsoleResult<T> {
        let name = lock(&self.active).editable()?;
        let mut copies = lock(&self.copies);
        let copy = copies
            .get_mut(&name)
            .ok_or_else(|| ConsoleError::not_found("no scene"))?;
        f(&name, copy)
    }

    /// Run `f` on the working copy of scene `name`, or of the active scene:
    /// the one that the console holds, after the load of the active scene
    /// ends, or else one built from the scene's files and its state file.
    /// Its revision is 0 when the console never loaded it.
    fn with_copy<T>(
        &self,
        name: Option<&str>,
        f: impl FnOnce(&str, &WorkingCopy) -> T,
    ) -> ConsoleResult<T> {
        let name = {
            let mut active = lock(&self.active);
            let name = name
                .map(str::to_string)
                .or_else(|| active.name.clone())
                .unwrap_or_default();
            let deadline = Instant::now() + LOAD_TIMEOUT;
            while active.is_loading() && active.name.as_deref() == Some(name.as_str()) {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() || self.stopping() {
                    return Err(ConsoleError::unavailable(format!(
                        "scene '{name}' is loading"
                    )));
                }
                active = self
                    .loaded
                    .wait_timeout(active, left.min(Duration::from_millis(250)))
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
            }
            name
        };
        if !self.store.contains(&name) {
            return Err(ConsoleError::not_found("no scene"));
        }
        if let Some(copy) = lock(&self.copies).get(&name) {
            return Ok(f(&name, copy));
        }
        // before the render loop starts, the scene_size() of the renderer that
        // make_renderer() builds
        let size = lock(&self.scene_size).unwrap_or(match self.rotate {
            90 | 270 => (self.h, self.w),
            _ => (self.w, self.h),
        });
        let (opened, _) = self
            .store
            .open(&name, None, size)
            .map_err(|e| ConsoleError::server_error(format!("build {name}: {e}")))?;
        Ok(f(&name, &WorkingCopy::new(opened, 0, 0)))
    }

    /// Run `f` on what the process of scene `name` set, None when no
    /// process of that scene runs.
    fn with_locks<T>(&self, name: &str, f: impl FnOnce(Option<&Locks>) -> T) -> T {
        let slot = lock(&self.locks);
        match slot.as_ref() {
            Some((at, locks)) if at == name => f(Some(&lock(locks))),
            _ => f(None),
        }
    }

    /// Make `name` the active scene, and number the switch for the load
    /// that the render thread runs.
    fn ask(&self, name: &str) -> u64 {
        let mut active = lock(&self.active);
        active.name = Some(name.to_string());
        active.asked += 1;
        active.asked
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
        self.software_stream().is_some()
    }

    /// The stream that the software lane owns, while a viewer watches it.
    fn software_stream(&self) -> Option<u64> {
        let stream = self.fanout.stream();
        let dead = *lock(&self.h264_hw_dead) == Some(stream);
        (dead && self.fanout.nclients() > 0).then_some(stream)
    }

    /// True when hardware encoding failed for the current stream.
    fn hardware_dead(&self) -> bool {
        *lock(&self.h264_hw_dead) == Some(self.fanout.stream())
    }

    /// The hardware lane failed. Before the stream's first access unit the
    /// software lane takes the stream, and after it the stream ends.
    fn hardware_failed(&self, why: &str) {
        if self.fanout.began() {
            eprintln!("h264: {why}");
            self.fanout.fail(why);
            return;
        }
        *lock(&self.h264_hw_why) = why.to_string();
        *lock(&self.h264_hw_dead) = Some(self.fanout.stream());
        eprintln!("h264: {why}, using software encoder");
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
/// display never waits. Raw frames also feed the h264 path, but a JPEG needs
/// an attached viewer.
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

pub fn stream_bitrate(w: i32, h: i32, fps: i32, floor: i64) -> i64 {
    let derived = f64::from(w) * f64::from(h) * f64::from(fps.max(1)) * H264_BPP;
    floor.max((derived as i64).min(H264_BITRATE_CEILING))
}

/// Software H264, for a board with no /dev/video11. It encodes the same readback
/// frames as the MJPEG preview, off the render thread.
pub fn h264_sw_loop(console: &Console) {
    let mut enc: Option<SoftwareH264> = None;
    let mut size = (0, 0);
    // the stream that the encoder encodes for
    let mut owned = 0;
    let mut last_seq = u64::MAX;
    while !console.stopping() {
        let stream = console
            .software_stream()
            .filter(|_| console.fanout.failure().is_none());
        let Some(stream) = stream else {
            if let Some(mut e) = enc.take() {
                let _ = e.close(Duration::from_secs(1));
            }
            let raw = lock(&console.raw);
            let _ = console
                .raw_ready
                .wait_timeout(raw, Duration::from_millis(250));
            continue;
        };
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
                stream_bitrate(w, h, H264_FPS, console.bitrate) as i32,
                rgb.components,
            ) {
                Ok(e) => {
                    enc = Some(e);
                    size = (w, h);
                    owned = stream;
                    console.fanout.opened(w, h, SW_COLOR);
                }
                Err(e) => {
                    let why = no_encoder(&lock(&console.h264_hw_why), &e.to_string());
                    eprintln!("h264: {why}");
                    console.fanout.fail(&why);
                    continue;
                }
            }
        }
        let e = enc.as_mut().expect("the encoder was just created");
        if let Err(err) = e.feed(&rgb.data) {
            let why = format!("software encoder died: {err}");
            eprintln!("h264: {why}");
            console.fanout.fail(&why);
            if let Some(mut e) = enc.take() {
                let _ = e.close(Duration::from_secs(1));
            }
            continue;
        }
        if let (Some(au), _) = e.get_au(Duration::from_secs(1)) {
            if !console.fanout.push_to(owned, &au) {
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
        "" // a still is not a destination, see Console::out
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
    /// Encodes since the last access unit. An encoder can open at a size it
    /// never emits for, with no error to catch.
    starved: u32,
    /// When the preview last refreshed.
    last: Cell<f64>,
    now: Cell<f64>,
}

/// Encodes with no output before the hardware lane is declared dead.
const HW_STARVE_LIMIT: u32 = 90;

impl Output for ConsoleSink {
    fn dest(&self) -> &str {
        "" // the console's destinations are its viewers, see Console::out
    }

    fn persists(&self) -> bool {
        true // a browser holds the last frame it got
    }

    fn wants(&self, _frame: u64) -> bool {
        // Refresh the browser preview at preview_fps at most, and only while a
        // viewer watches. The display still renders every frame.
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
            if let Some(mut enc) = self.enc.take() {
                enc.close(); // direct scanout returns while nobody watches
            }
            return Ok(());
        }
        if console.fanout.failure().is_some() {
            if let Some(mut enc) = self.enc.take() {
                enc.close();
            }
            return Ok(());
        }
        if self.enc.is_none() && !console.hardware_dead() {
            let (w, h) = stream_size(console.w, console.h, console.stream_max);
            // The stream encodes at its own rate, not the display's. A panel
            // that asks for 63Hz asks the encoder for a rate it refuses at
            // this size, and the input port then fails to enable: the whole
            // stream drops to software for a rate that no viewer needed.
            let hz = console.stream_hz.max(1);
            let bitrate = stream_bitrate(w, h, hz, console.bitrate) as i32;
            match HardwareH264::new(&self.renderer.borrow(), w, h, hz, bitrate) {
                Ok(enc) => {
                    console.fanout.opened(enc.w, enc.h, HW_COLOR);
                    self.enc = Some(enc);
                    self.starved = 0;
                }
                Err(e) => console.hardware_failed(&e.to_string()),
            }
        }
        let Some(enc) = self.enc.as_mut() else {
            return Ok(());
        };
        if console.fanout.wants_key() {
            enc.force_keyframe();
        }
        let failed = match enc.encode_last() {
            Ok((Some(au), _)) => {
                self.starved = 0;
                console.fanout.push(au);
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
            if let Some(mut enc) = self.enc.take() {
                enc.close();
            }
            console.hardware_failed(&why);
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
///
/// The scene's controller changes layers through it: a procedural scene's
/// drawing adds a layer or gives one a new look, and its ticks set fields.
/// Each change that the renderer takes lands in the running document too.
struct SharedRenderer {
    renderer: Rc<RefCell<Renderer>>,
    running: Rc<RefCell<Running>>,
}

impl SharedRenderer {
    /// Copy layer `name`, with its place, from the renderer's scene into the
    /// running document.
    fn mirror(&self, name: &str) {
        let found = self.renderer.borrow().with_scene(|s| {
            let (p, i) = s.place_of(name)?;
            let pass = &s.passes[p];
            Some((pass.name.clone(), i, pass.layers[i].clone()))
        });
        let Some((pass, index, layer)) = found.flatten() else {
            return;
        };
        let mut held = self.running.borrow_mut();
        let Some(scene) = held.scene.as_mut() else {
            return;
        };
        let kept = match scene.place_of(name) {
            Some((p, i)) => {
                scene.passes[p].layers[i] = layer;
                Ok(())
            }
            None => scene.insert_layer(&pass, index, layer),
        };
        if let Err(e) = kept {
            eprintln!("ghee console: layer {name}: {e}");
            return;
        }
        held.changed = true;
    }
}

impl RenderTarget for SharedRenderer {
    fn width(&self) -> i32 {
        self.renderer.borrow().width
    }

    fn height(&self) -> i32 {
        self.renderer.borrow().height
    }

    fn components(&self) -> i32 {
        self.renderer.borrow().components
    }

    fn set_dt(&self, value: f64) {
        self.renderer.borrow().set_dt(value);
    }

    fn set_time(&self, t: f64) {
        self.renderer.borrow().set_time(Some(t));
    }

    fn clock_restarted(&self) -> bool {
        RenderTarget::clock_restarted(&*self.renderer.borrow())
    }

    fn set_param(
        &self,
        pass: Option<&str>,
        name: &str,
        ty: &str,
        value: &ParamValue,
    ) -> Result<bool> {
        RenderTarget::set_param(&*self.renderer.borrow(), pass, name, ty, value)
    }

    fn set_array(&self, name: &str, data: &Feed) -> Result<bool> {
        self.renderer.borrow().set_array(name, data)
    }

    fn set_layer(&self, at: &Address, given: Given) -> Result<()> {
        self.renderer.borrow().set_layer(at, given)?;
        self.mirror(at.layer);
        Ok(())
    }

    fn add_layer(&mut self, pass: &str, index: usize, layer: &Value) -> Result<()> {
        self.renderer.borrow_mut().add_layer(pass, index, layer)?;
        if let Some(name) = layer.get("name").and_then(Value::as_str) {
            self.mirror(name);
        }
        Ok(())
    }

    fn update_layer(&mut self, layer: &Value) -> Result<bool> {
        let changed = self.renderer.borrow_mut().update_layer(layer)?;
        if let (true, Some(name)) = (changed, layer.get("name").and_then(Value::as_str)) {
            self.mirror(name);
        }
        Ok(changed)
    }

    fn update_calls(&mut self, calls: &[Call]) -> Result<Taken> {
        let taken = self.renderer.borrow_mut().update_calls(calls)?;
        for &k in &taken.changed {
            self.mirror(&calls[k].layer);
        }
        Ok(taken)
    }

    fn remove_layer(&mut self, name: &str) -> Result<()> {
        self.renderer.borrow_mut().remove_layer(name)?;
        let mut held = self.running.borrow_mut();
        if let Some(scene) = held.scene.as_mut() {
            if scene.remove_layer(name).is_ok() {
                held.changed = true;
            }
        }
        Ok(())
    }

    fn move_layer(&mut self, name: &str, index: usize) -> Result<()> {
        self.renderer.borrow_mut().move_layer(name, index)?;
        let mut held = self.running.borrow_mut();
        let Some(scene) = held.scene.as_mut() else {
            return Ok(());
        };
        match scene.move_layer(name, index) {
            Ok(()) => held.changed = true,
            Err(e) => eprintln!("ghee console: layer {name}: {e}"),
        }
        Ok(())
    }

    fn draw(&mut self, provider: Option<Provider<'_>>) -> Result<()> {
        self.renderer.borrow_mut().draw(provider)
    }

    fn blank(&self, on: bool) -> Result<()> {
        self.renderer.borrow().blank(on)
    }

    fn render_into(
        &mut self,
        provider: Option<Provider<'_>>,
        out: &mut [u8],
        out_size: Option<(i32, i32)>,
    ) -> Result<()> {
        self.renderer
            .borrow_mut()
            .render_into(provider, out, out_size)
    }

    fn present(
        &mut self,
        provider: Option<Provider<'_>>,
        preview: Option<(&mut [u8], (i32, i32))>,
    ) -> Result<()> {
        self.renderer.borrow_mut().present(provider, preview)
    }
}

/// Create the render-thread Renderer. It probes the DRI devices, so the console
/// runs on a Pi (card0) and on desktop Mesa (renderD128). A display takes
/// `card` when the caller holds it open.
fn make_renderer(console: &Console, card: Option<Card>) -> Result<Renderer> {
    let mut r = open_renderer(console, card)?;
    r.set_alpha(console.alpha);
    r.set_rotate(console.rotate)?;
    if let Some(level) = console.brightness {
        if let Err(e) = r.set_brightness(level) {
            eprintln!("ghee console: {e:#}");
        }
    }
    Ok(r)
}

fn open_renderer(console: &Console, card: Option<Card>) -> Result<Renderer> {
    if console.display {
        if let Some(card) = card {
            return Renderer::on_card(card, console.w, console.h, console.rgb565);
        }
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
    /// The scene that the renderer draws, the last one that compiled.
    scene: Option<Scene>,
    /// The name of that scene, and the generation of its working copy.
    name: Option<String>,
    generation: u64,
    /// The name of the scene that the last load asked for. A /set is for
    /// this scene, so it lands only while `scene` is this scene.
    loaded: Option<String>,
    /// The scene's own code changed a layer since the last publish.
    changed: bool,
}

/// Set `scene`, the scene of the last load, on the renderer. Returns the
/// compile error, or "" when this scene renders now. A scene that fails to
/// compile is published all the same, so a page opens it to fix it. `keep`
/// makes `scene` an edit of the running scene: its passes go on from the
/// running passes' buffers, as the last good scene does when a failure sets it
/// again.
fn set_scene(
    console: &Console,
    renderer: &RefCell<Renderer>,
    st: &RefCell<Running>,
    scene: Scene,
    generation: u64,
    snap: Option<Snapshot>,
    keep: bool,
) -> String {
    let built = match keep {
        true => renderer.borrow_mut().rebuild(&scene),
        false => renderer.borrow_mut().setup(&scene),
    };
    let err = match built {
        Ok(err) => err,
        Err(e) => e.to_string(),
    };
    let name = st.borrow().loaded.clone();
    // Store the snapshot before the publish. A /stats between the two must not
    // read the new scene against the snapshot before it.
    if let Some(snap) = snap {
        *lock(&console.sources) = Some(snap);
    }
    if !err.is_empty() {
        {
            let mut stat = lock(&console.stat);
            stat.error = err.clone();
            stat.error_line = renderer.borrow().error_line;
            stat.error_pass = renderer.borrow().error_pass;
        }
        if let Some(last) = st.borrow().scene.as_ref() {
            let _ = renderer.borrow_mut().rebuild(last); // keep the last good scene rendering
        }
        console.publish(name.as_deref(), generation, &scene);
        trim_heap();
        return err;
    }
    console.reset_stats();
    console.publish(name.as_deref(), generation, &scene);
    let mut held = st.borrow_mut();
    held.scene = Some(scene);
    held.name = name;
    held.generation = generation;
    trim_heap();
    String::new()
}

/// Load a scene by name for switch number `switch`, and put what feeds it in
/// the controller slot. The last scene's process stops first, so two
/// procedural scenes never overlap.
fn load(
    console: &Console,
    renderer: &RefCell<Renderer>,
    st: &RefCell<Running>,
    ctrl: &ControllerSlot,
    name: &str,
    switch: u64,
    vid: &str,
) {
    *ctrl.borrow_mut() = None;
    *lock(&console.locks) = None;
    st.borrow_mut().loaded = Some(name.to_string());
    // A switch away and back, and a reload, keep the edits made since the
    // last save, and check them against the files as they are now.
    let carried = lock(&console.copies).get(name).map(WorkingCopy::carried);
    let size = renderer.borrow().scene_size();
    let (built, read) = collect(|| console.store.open(name, carried, size));
    let failed = match built {
        Ok((opened, mut process)) => {
            let snap = console.snapshot(read);
            let working = opened.working.clone();
            let generation = console.install(name, opened, vid);
            let err = set_scene(
                console,
                renderer,
                st,
                working,
                generation,
                Some(snap),
                false,
            );
            renderer.borrow().restart_clock();
            if let (Some(f), Some(_)) = (process.as_mut(), console.timestep) {
                f.pin();
            }
            if err.is_empty() {
                if let Some(f) = process.as_mut() {
                    let locks = Arc::new(Mutex::new(Locks::default()));
                    f.lock_into(Arc::clone(&locks));
                    *lock(&console.locks) = Some((name.to_string(), locks));
                }
                *ctrl.borrow_mut() = process.map(|f| Box::new(f) as Box<dyn SceneController>);
            }
            None
        }
        Err(e) => {
            let why = format!("load {name}: {e}");
            lock(&console.stat).error = why.clone();
            Some(why)
        }
    };
    let mut active = lock(&console.active);
    if switch >= active.loaded {
        active.loaded = switch;
        active.failed = failed;
    }
    console.loaded.notify_all();
}

/// Leave the renderer on no scene, drawing black.
fn clear(
    console: &Console,
    renderer: &RefCell<Renderer>,
    st: &RefCell<Running>,
    ctrl: &ControllerSlot,
) {
    *ctrl.borrow_mut() = None;
    *lock(&console.locks) = None;
    *lock(&console.sources) = None;
    *lock(&console.running) = None;
    *st.borrow_mut() = Running::default();
    let black = serde_json::json!({"version": 1, "passes": [{"name": "main",
        "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }"}]});
    match Scene::from_dict(&black, None, None).and_then(|s| renderer.borrow_mut().setup(&s)) {
        Ok(err) if err.is_empty() => {}
        Ok(err) => eprintln!("ghee console: no scene: {err}"),
        Err(e) => eprintln!("ghee console: no scene: {e}"),
    }
    console.reset_stats();
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
        let drawn = {
            let held = st.borrow();
            held.name == held.loaded
        };
        match item {
            Command::Switch(name, switch, vid) => {
                load(console, renderer, st, ctrl, &name, switch, &vid);
            }
            Command::Param(..) | Command::SceneParam(..) | Command::Layer(_) if !drawn => {}
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
            Command::Layer(set) => {
                if set_layer(renderer, st, ctrl, &set) {
                    edited = true;
                }
            }
            // in queue order, so the armed frame carries a /set ahead of it
            Command::Shot => console.shot.arm(),
            Command::SceneObj(scene, generation, fresh) => {
                let mut scene = *scene;
                // A procedural scene's code owns which layers exist, so an
                // apply keeps a layer that the code added after the page
                // read the scene.
                let kept = ctrl.borrow().is_some()
                    && st
                        .borrow()
                        .scene
                        .as_ref()
                        .is_some_and(|running| keep_layers(&mut scene, running));
                let built =
                    set_scene(console, renderer, st, scene, generation, None, !fresh).is_empty();
                // a failed setup sets the last scene up again
                if let Some(c) = ctrl.borrow_mut().as_mut() {
                    c.rebuilt();
                }
                if built && kept {
                    st.borrow_mut().changed = true;
                    publish_changes(console, st);
                }
            }
            Command::Clear => clear(console, renderer, st, ctrl),
            Command::Preview | Command::Stop => {}
        }
    }
    if edited {
        let held = st.borrow();
        if let Some(scene) = held.scene.as_ref() {
            console.publish(held.name.as_deref(), held.generation, scene);
        }
    }
    force
}

/// Put into `posted`, a scene applied over the procedural scene `running`,
/// each layer of `running` that `posted` leaves out, in its place among the
/// layers of its pass, with its words. True when one went in.
fn keep_layers(posted: &mut Scene, running: &Scene) -> bool {
    let mut kept = false;
    for pass in &running.passes {
        for (i, layer) in pass.layers.iter().enumerate() {
            let name = layer.name();
            if posted.layer(name).is_some() {
                continue;
            }
            let Some(at) = posted.passes.iter().position(|p| p.name == pass.name) else {
                continue;
            };
            let index = i.min(posted.passes[at].layers.len());
            if let Err(e) = posted.insert_layer(&pass.name, index, layer.clone()) {
                eprintln!("ghee console: apply: layer {name}: {e}");
                continue;
            }
            kept = true;
        }
    }
    kept
}

/// Publish the running document when the scene's own code changed a layer
/// in it, and count the change for /stats.
fn publish_changes(console: &Console, st: &RefCell<Running>) {
    let mut held = st.borrow_mut();
    if !std::mem::take(&mut held.changed) {
        return;
    }
    if let Some(scene) = held.scene.as_ref() {
        console.publish(held.name.as_deref(), held.generation, scene);
        console.layer_changes.fetch_add(1, Ordering::SeqCst);
    }
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
            .ok_or_else(|| anyhow!("the scene has no pass {pi}"))?;
        let missing = || anyhow!("pass '{}' has no param '{name}'", pass.name);
        let ty = pass.params.get(name).ok_or_else(missing)?.ty.clone();
        let v = coerce(name, &ty, value)?;
        if ty != "string" {
            renderer.borrow().set_param(&pass.name, name, &ty, &v)?;
        }
        told(ctrl, Some(&pass.name), name, &v);
        let gone = || anyhow!("pass '{}' has no param '{name}'", pass.name);
        pass.params.get_mut(name).ok_or_else(gone)?.value = v;
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

/// Tell the scene's process about a param of pass `pass`, None for the
/// scene's, that the console changed. The process reads it on the next tick.
fn told(ctrl: &ControllerSlot, pass: Option<&str>, name: &str, value: &ParamValue) {
    if let Some(c) = ctrl.borrow_mut().as_mut() {
        c.on_param(pass, name, value);
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
        let missing = || anyhow!("the scene has no param '{name}'");
        let ty = scene.params.get(name).ok_or_else(missing)?.ty.clone();
        let v = coerce(name, &ty, value)?;
        renderer.borrow().set_scene_param(name, &ty, &v)?;
        told(ctrl, None, name, &v);
        scene.params.get_mut(name).ok_or_else(missing)?.value = v;
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

/// Set a live field of a layer on the renderer and in the running document,
/// and tell the scene's process.
fn set_layer(
    renderer: &RefCell<Renderer>,
    st: &RefCell<Running>,
    ctrl: &ControllerSlot,
    set: &LayerSet,
) -> bool {
    let mut held = st.borrow_mut();
    let Some(scene) = held.scene.as_mut() else {
        return false;
    };
    let at = set.at();
    let applied = renderer
        .borrow()
        .set_layer(&at, Given::Json(&set.value))
        .and_then(|()| scene.set_layer(&at, &set.value));
    match applied {
        Ok(()) => {
            if let Some(c) = ctrl.borrow_mut().as_mut() {
                c.on_layer(&at, &set.value);
            }
            true
        }
        Err(e) => {
            eprintln!("set layer {}: {e}", set.layer);
            false
        }
    }
}

/// Publish the renderer's text problems for /stats, when they changed since
/// the count `seen`.
fn publish_text_errors(console: &Console, renderer: &Renderer, seen: &Cell<u64>) {
    let seq = renderer.text_errors_seq();
    if seq != seen.get() {
        seen.set(seq);
        lock(&console.stat).texts = renderer.text_errors();
    }
}

/// Run the console's render loop on this thread. The console adds the command
/// queue, the published scene and the stale snapshot. A frame error must not
/// kill the thread.
///
/// GL is thread-affine, so the caller creates the Renderer on the thread that
/// calls this. The sinks belong to the caller, and this draws to them from
/// here.
pub fn render_loop(
    console: &Arc<Console>,
    renderer: Rc<RefCell<Renderer>>,
    mut sinks: Vec<Box<dyn Output>>,
    pacer: Option<Box<dyn Pacer>>,
    on_stats: Option<StatsHook>,
) {
    let hz = if console.display {
        renderer.borrow().display_vrefresh()
    } else {
        0
    };
    *lock(&console.scene_size) = Some(renderer.borrow().scene_size());
    lock(&console.stat).gpu = gpu_caps(&renderer.borrow());
    let st = Rc::new(RefCell::new(Running::default()));
    // what feeds the running scene, when the scene brings something
    let ctrl: ControllerSlot = RefCell::new(None);
    let mut shared = SharedRenderer {
        renderer: Rc::clone(&renderer),
        running: Rc::clone(&st),
    };
    let mut scanout = Scanout;
    let mut console_sink = ConsoleSink {
        console: Arc::clone(console),
        renderer: Rc::clone(&renderer),
        enc: None,
        starved: 0,
        last: Cell::new(0.0),
        now: Cell::new(0.0),
    };
    let mut shot_sink = FrameShot {
        shot: Arc::clone(&console.shot),
    };
    let mut reported = String::new();
    let times: RefCell<Vec<f64>> = RefCell::new(Vec::new());
    let text_errors = Cell::new(renderer.borrow().text_errors_seq());
    let published = Cell::new(f64::NEG_INFINITY);
    let timed = on_stats.is_some();
    // A display or a sink of the caller's takes every frame. Without one, the
    // frames reach the viewers alone.
    let drives = console.display || !sinks.is_empty();
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
        render.pacer = pacer;
        // the loop rests so the handler and encoder threads run
        render.rest = 0.003;
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
        render.watched = Some(Box::new(move || {
            drives || console.nclients() > 0 || console.fanout.nclients() > 0
        }));
        render.stopping = Some(Box::new(|| console.stopping()));
        render.on_frame = Some(Box::new(|_loop, cost| {
            lock(&console.stat).frame_ms = cost * 1000.0;
            let now = monotonic();
            if now - published.get() >= LAYERS_PUBLISH_EVERY && st.borrow().changed {
                publish_changes(console, &st);
                published.set(now);
            }
            publish_text_errors(console, &renderer.borrow(), &text_errors);
            if timed {
                times.borrow_mut().push(cost);
            }
        }));
        render.on_error = Some(Box::new(|e| {
            // One bad frame must not take the thread with it. Report it, and
            // keep rendering the last scene that worked. /stats reaches a
            // browser, so it repeats the stderr already on the terminal.
            let msg = format!("render: {e}");
            lock(&console.stat).error =
                format!("render: {}", crate::procedural::process::with_stderr(e));
            if msg != reported {
                eprintln!("ghee console: {msg}");
                reported = msg;
            }
            true
        }));
        let first = {
            let active = lock(&console.active);
            active.name.clone().map(|name| (name, active.asked))
        };
        match first {
            Some((name, switch)) => load(console, &renderer, &st, &ctrl, &name, switch, ""),
            None => clear(console, &renderer, &st, &ctrl),
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
    fn stats(&self) -> ConsoleResult<Value> {
        Ok(self.console.stats())
    }

    /// The list is the directories' as they are now.
    fn scene_names(&self) -> ConsoleResult<Vec<String>> {
        self.console.store.rescan();
        Ok(self.console.store.names())
    }

    fn procedural_names(&self) -> ConsoleResult<Vec<String>> {
        Ok(self.console.store.procedural_names())
    }

    fn mine_names(&self) -> ConsoleResult<Vec<String>> {
        Ok(self.console.store.mine_names())
    }

    fn my_scenes(&self) -> Option<String> {
        self.console.store.mine.clone()
    }

    /// A procedural scene has no scene file to copy: its program writes it.
    fn new_scene(&self, name: &str, from: Option<&str>) -> ConsoleResult<()> {
        let console = &self.console;
        let store = &console.store;
        let (doc, dir) = match from {
            None => (store::blank(), None),
            Some(from) => {
                store.rescan();
                if !store.contains(from) {
                    return Err(ConsoleError::not_found(format!("no scene '{from}'")));
                }
                if store.is_procedural(from) {
                    return Err(ConsoleError::read_only(format!(
                        "a program writes scene '{from}', so it has no scene file to copy"
                    )));
                }
                console.with_copy(Some(from), |_, copy| {
                    let root = copy.opened.defaults.root.as_ref();
                    (copy.working_doc(), root.map(|r| r.dir.clone()))
                })?
            }
        };
        let _editing = lock(&console.editing);
        store.create(name, &doc, dir.as_deref(), &store::files_of(&doc))
    }

    /// The working copy goes with its scene, and a rename of the active
    /// scene loads it from its new place.
    fn rename_scene(&self, name: &str, to: &str, vid: &str) -> ConsoleResult<()> {
        let console = &self.console;
        let _editing = lock(&console.editing);
        console.store.rename(name, to)?;
        {
            let mut copies = lock(&console.copies);
            if let Some(copy) = copies.remove(name) {
                copies.insert(to.to_string(), copy);
            }
        }
        {
            let mut revs = lock(&console.revs);
            if let Some(rev) = revs.remove(name) {
                revs.insert(to.to_string(), rev);
            }
        }
        if lock(&console.active).name.as_deref() == Some(name) {
            let switch = console.ask(to);
            console.command(Command::Switch(to.to_string(), switch, vid.to_string()));
        }
        Ok(())
    }

    /// With the active scene gone, the console takes the first scene, or
    /// none.
    fn delete_scene(&self, name: &str) -> ConsoleResult<()> {
        let console = &self.console;
        let _editing = lock(&console.editing);
        console.store.delete(name)?;
        lock(&console.copies).remove(name);
        lock(&console.revs).remove(name);
        if lock(&console.active).name.as_deref() != Some(name) {
            return Ok(());
        }
        match console.store.names().into_iter().next() {
            Some(next) => {
                let switch = console.ask(&next);
                console.command(Command::Switch(next, switch, String::new()));
            }
            None => {
                let mut active = lock(&console.active);
                active.name = None;
                active.failed = None;
                drop(active);
                console.command(Command::Clear);
            }
        }
        Ok(())
    }

    fn upload_dir(&self, name: &str) -> ConsoleResult<std::path::PathBuf> {
        self.console
            .store
            .own_dir(name)
            .map(std::path::PathBuf::from)
    }

    fn active_name(&self) -> ConsoleResult<Option<String>> {
        Ok(lock(&self.console.active).name.clone())
    }

    fn scene_def(&self, name: Option<&str>) -> ConsoleResult<Value> {
        self.scene_doc(name, false).map(|(doc, _)| doc)
    }

    fn scene_doc(&self, name: Option<&str>, defaults: bool) -> ConsoleResult<(Value, Option<u64>)> {
        let running = &self.console.running;
        self.console.with_copy(name, |name, copy| {
            let doc = match defaults {
                true => copy.defaults_doc.clone(),
                false => copy.live_doc(name, &lock(running)),
            };
            (doc, Some(copy.rev))
        })
    }

    fn edits(&self, name: Option<&str>) -> ConsoleResult<Value> {
        let console = &self.console;
        console.with_copy(name, |name, copy| {
            console.with_locks(name, |locks| copy.edits_json(locks))
        })
    }

    /// A param that the scene's program set answers 403.
    fn set_param(&self, scope: Scope, name: &str, value: Value) -> ConsoleResult<()> {
        let console = &self.console;
        console.with_active(|scene, copy| {
            let working = &mut copy.opened.working;
            let refusal =
                |pass| console.with_locks(scene, |l| l.and_then(|l| l.param_refusal(pass, name)));
            let held = match scope {
                Scope::Scene => refusal(None),
                Scope::Pass(i) => working
                    .passes
                    .get(usize::try_from(i).unwrap_or(usize::MAX))
                    .and_then(|p| refusal(Some(&p.name))),
            };
            if let Some(why) = held {
                return Err(ConsoleError::read_only(why));
            }
            let params = match scope {
                Scope::Scene => Some(&mut working.params),
                Scope::Pass(i) => working
                    .passes
                    .get_mut(usize::try_from(i).unwrap_or(usize::MAX))
                    .map(|p| &mut p.params),
            };
            if let Some(p) = params.and_then(|p| p.get_mut(name)) {
                p.value = coerce(name, &p.ty, &value)
                    .map_err(|e| ConsoleError::bad_request(format!("bad /set: {e}")))?;
            }
            Ok(())
        })?;
        self.console.command(set_item(scope, name, value));
        Ok(())
    }

    /// A field that the scene's program set answers 403, and so does every
    /// field of a layer that it added or gave a look.
    fn set_layer(&self, set: LayerSet) -> ConsoleResult<()> {
        let console = &self.console;
        console.with_active(|scene, copy| {
            let at = set.at();
            let locked = console.with_locks(scene, |l| l.and_then(|l| l.field_refusal(&at)));
            if let Some(why) = locked {
                return Err(ConsoleError::read_only(why));
            }
            let working = &mut copy.opened.working;
            // A layer that a procedural scene's program drew is the program's.
            if working.layer(&set.layer).is_some() {
                working
                    .set_layer(&set.at(), &set.value)
                    .map_err(|e| ConsoleError::bad_request(format!("bad /set: {e}")))?;
            }
            Ok(())
        })?;
        self.console.command(Command::Layer(set));
        Ok(())
    }

    /// The /set lands on the active scene, so it checks against the document
    /// that a page reads of it.
    fn live_doc(&self) -> ConsoleResult<Value> {
        let running = &self.console.running;
        self.console
            .with_active(|name, copy| Ok(copy.live_doc(name, &lock(running))))
    }

    fn switch(&self, name: Option<&str>) -> ConsoleResult<String> {
        self.switch_by(name, &Edit::default())
    }

    fn switch_by(&self, name: Option<&str>, by: &Edit) -> ConsoleResult<String> {
        let name = name.unwrap_or_default();
        let _editing = lock(&self.console.editing);
        if let Some(scene) = by.scene {
            other_reload(scene, self.active_name()?.as_deref())?;
        }
        if !self.console.store.contains(name) {
            self.console.store.rescan();
        }
        if !self.console.store.contains(name) {
            return Err(ConsoleError::not_found("no scene"));
        }
        let switch = self.console.ask(name);
        self.console.command(Command::Switch(
            name.to_string(),
            switch,
            by.vid.to_string(),
        ));
        Ok(name.to_string())
    }

    /// Every edit gives the active scene a new revision.
    fn edit_scene(
        &self,
        at: &Edit,
        edit: &mut dyn FnMut() -> ConsoleResult<()>,
    ) -> ConsoleResult<()> {
        let _editing = lock(&self.console.editing);
        if let Some(scene) = at.scene {
            other_scene(scene, self.active_name()?.as_deref())?;
        }
        let name = lock(&self.console.active).editable()?;
        let stale = lock(&self.console.copies)
            .get(&name)
            .zip(at.rev)
            .is_some_and(|(copy, rev)| copy.stale(rev, at.vid));
        if stale {
            return Err(ConsoleError::stale(
                "this scene changed since you loaded it",
            ));
        }
        edit()?;
        if let Some(copy) = lock(&self.console.copies).get_mut(&name) {
            self.console.bump(&name, copy, at.vid);
        }
        Ok(())
    }

    fn apply_scene(&self, mut scene: Scene) -> ConsoleResult<()> {
        let running = &self.console.running;
        self.console.with_active(|name, copy| {
            // A posted scene names its asset files only. They resolve where
            // the scene's own files do, where GET /scene took them from.
            let (dir, reach) = match &copy.opened.defaults.root {
                Some(root) => (root.dir.clone(), root.reach),
                None => (self.console.store.dir.clone(), Reach::Inside),
            };
            scene
                .load_assets_in(&dir, reach)
                .map_err(|e| ConsoleError::bad_request(format!("bad /scene: {e}")))?;
            if copy.opened.procedural {
                // The program's own drawings and sets stay its own: the
                // working copy takes what the page changed in what runs.
                let mut changed =
                    state::diff(&copy.live_doc(name, &lock(running)), &scene.to_dict(), true);
                self.console.with_locks(name, |locks| {
                    if let Some(locks) = locks {
                        leave_out(&mut changed.values, locks);
                    }
                });
                let doc = state::fold(&copy.defaults_doc, &copy.working_doc(), &changed);
                match store::scene_of(&copy.opened.defaults, &doc, name) {
                    Ok(working) => copy.opened.working = working,
                    Err(e) => eprintln!(
                        "ghee console: {name}: the working copy keeps no edit of this apply: {e}"
                    ),
                }
            } else {
                copy.opened.working = scene.clone();
            }
            // A new edit of an item set aside answers it, so a save writes
            // the new edit alone.
            let edited = copy.edits(None).items;
            copy.opened.aside.retain(|a| {
                !edited
                    .iter()
                    .any(|i| i.is(&a.item, a.pass.as_deref(), a.param.as_deref()))
            });
            copy.generation += 1;
            self.console
                .command(Command::SceneObj(Box::new(scene), copy.generation, false));
            Ok(())
        })
    }

    /// A scene of the console's own saves whole, and its defaults are then
    /// the scene as saved.
    fn save(&self, _name: Option<&str>) -> ConsoleResult<()> {
        let console = &self.console;
        console.with_active(|name, copy| {
            if console.store.is_mine(name) {
                console.store.write(name, &copy.working_doc())?;
                copy.opened.defaults = copy.opened.working.clone();
                copy.defaults_doc = copy.opened.defaults.to_dict();
                copy.opened.saved = None;
                let size = lock(&console.scene_size).unwrap_or((console.w, console.h));
                let (_, read) = collect(|| console.store.resolve(name, size));
                *lock(&console.sources) = Some(console.snapshot(read));
                return Ok(());
            }
            let edits = console.with_locks(name, |locks| copy.to_save(locks));
            console.store.save(name, &edits)?;
            copy.opened.saved = (!edits.is_empty()).then_some(edits);
            Ok(())
        })
    }

    fn reset(&self, _name: Option<&str>) -> ConsoleResult<()> {
        self.console.with_active(|_, copy| {
            copy.opened.working = copy.opened.defaults.clone();
            copy.generation += 1;
            let defaults = Box::new(copy.opened.defaults.clone());
            self.console
                .command(Command::SceneObj(defaults, copy.generation, true));
            Ok(())
        })
    }

    /// Keep takes the default item's hash as the edit's base and applies the
    /// edit, and default drops it. Either writes the state file at once, as
    /// it was with that one item answered. The item BROKEN answers the saved
    /// edits that make no scene: keep leaves them waiting, and default drops
    /// them from the state file.
    fn answer(&self, _name: Option<&str>, aside: &Aside, keep: bool) -> ConsoleResult<()> {
        let store = &self.console.store;
        if aside.item == BROKEN {
            return self.console.with_active(|name, copy| {
                if copy.opened.broken.is_empty() {
                    return Err(ConsoleError::not_found("no such item is set aside"));
                }
                if keep {
                    return Ok(());
                }
                let mut saved = copy.opened.saved.clone().unwrap_or_default();
                for b in &copy.opened.broken {
                    saved.remove(&b.edit);
                }
                store.save(name, &saved)?;
                copy.opened.saved = (!saved.is_empty()).then_some(saved);
                copy.opened.broken.clear();
                Ok(())
            });
        }
        self.console.with_active(|name, copy| {
            let is = |i: &Item| i.is(&aside.item, aside.pass.as_deref(), aside.param.as_deref());
            let at = copy
                .opened
                .aside
                .iter()
                .position(is)
                .ok_or_else(|| ConsoleError::not_found("no such item is set aside"))?;
            let mut item = copy.opened.aside[at].clone();
            let mut saved = copy.opened.saved.clone().unwrap_or_default();
            saved.items.retain(|i| !is(i));
            let mut working = None;
            if keep {
                item.base = state::base_of(&copy.defaults_doc, &item).flatten();
                let only = Edits {
                    values: serde_json::Map::new(),
                    items: vec![item.clone()],
                };
                let doc = state::fold(&copy.defaults_doc, &copy.working_doc(), &only);
                working = Some(
                    store::scene_of(&copy.opened.defaults, &doc, name)
                        .map_err(|e| ConsoleError::server_error(format!("{name}: {e}")))?,
                );
                saved.items.push(item);
                saved
                    .items
                    .sort_by_key(|i| state::KINDS.iter().position(|k| *k == i.item));
            }
            store.save(name, &saved)?;
            copy.opened.aside.remove(at);
            copy.opened.saved = (!saved.is_empty()).then_some(saved);
            if let Some(working) = working {
                copy.opened.working = working;
                copy.generation += 1;
                let scene = Box::new(copy.opened.working.clone());
                self.console
                    .command(Command::SceneObj(scene, copy.generation, false));
            }
            Ok(())
        })
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
fn handle(console: &Arc<Console>, api: &ConsoleAPI<RenderLoopHost>, mut request: Request) {
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

fn h264(console: &Arc<Console>, request: Request, query: &Query) {
    let c = console.fanout.add();
    let viewer = console.viewers.join(query.get("vid").map(String::as_str));
    console.q.put(Command::Preview); // wake an idle scene so the stream starts
    let alive = || !console.stopping() && !console.viewers.gone(&viewer);
    serve_h264(request, &console.fanout, &c, &alive);
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
    pub scenes_dir: Option<String>,
    /// The directory of the scenes that the console makes, listed after
    /// those of `scenes_dir`. None makes none.
    pub my_scenes: Option<String>,
    /// Take uploads while the console listens on a network.
    pub uploads: bool,
    /// Where the scenes' state files live. None takes the store's own, else
    /// state::default_dir().
    pub state_dir: Option<std::path::PathBuf>,
    pub size: (i32, i32),
    pub port: u16,
    /// Loopback by default. The console has no auth, and it compiles POSTed
    /// GLSL. Pass "0.0.0.0" only on a network you trust.
    pub host: String,
    pub display: bool,
    pub rgb565: bool,
    /// The degrees clockwise that every output turns the frame: 0, 90, 180
    /// or 270. `size` stays the output's.
    pub rotate: i64,
    /// The backlight level that the KMS display takes at the start, 0 to 1.
    pub brightness: Option<f64>,
    pub preview_max: i32,
    pub preview_fps: f64,
    pub stream_max: i32,
    pub stream_hz: i32,
    pub bitrate: i64,
    /// The build-time arguments for every scene the console loads.
    pub scene_args: Vec<String>,
    /// A prebuilt store, instead of one over scenes_dir.
    pub store: Option<SceneStore>,
    /// Filters every scene ends with, after its own. A host whose panel needs
    /// one sets them here.
    pub filters: Vec<String>,
    /// The output sinks the frames also reach, so a console writes files.
    /// `display` covers the KMS scanout instead, because that one is a mode
    /// the renderer is created in. run() writes these from its own thread, so
    /// a sink bound to the thread that built it belongs here.
    pub sinks: Vec<Box<dyn Output>>,
    /// Paces the render loop, as a display's fixed rate does. None draws as
    /// fast as the sinks and the viewers take frames.
    pub pacer: Option<Box<dyn Pacer>>,
    pub components: i32,
    /// Keep the output pass's alpha in 4-component frames.
    pub alpha: bool,
    /// 0 serves until stop(). A count ends the console with the render.
    pub frames: u64,
    pub timestep: Option<f64>,
    pub dri_device: Option<String>,
    /// The display's card, held open since the caller's probe, which the
    /// renderer takes in place of opening `dri_device`.
    pub card: Option<Card>,
    /// The summary hook. This hook makes the loop keep a cost per frame, and a
    /// run without it keeps none.
    pub on_stats: Option<StatsHook>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            arrays: Vec::new(),
            scenes_dir: None,
            my_scenes: None,
            uploads: false,
            state_dir: None,
            size: (320, 240),
            port: 8090,
            host: "127.0.0.1".to_string(),
            display: false,
            rgb565: false,
            rotate: 0,
            brightness: None,
            preview_max: 480,
            preview_fps: 15.0,
            stream_max: 0,
            stream_hz: 30,
            bitrate: 10_000_000,
            scene_args: Vec::new(),
            store: None,
            filters: Vec::new(),
            sinks: Vec::new(),
            pacer: None,
            components: 4,
            alpha: false,
            frames: 0,
            timestep: None,
            dri_device: None,
            card: None,
            on_stats: None,
        }
    }
}

/// The store over `scenes_dir`, or over the bundled examples when it is None,
/// and over `mine`, the console's own scenes. The bundled examples keep their
/// state files as demos, whichever way a caller names their directory.
fn dir_store(
    scenes_dir: Option<String>,
    mine: Option<String>,
    args: Vec<String>,
) -> Result<SceneStore> {
    let demo = scenes_dir.is_none();
    let dir = match scenes_dir {
        Some(dir) => dir,
        None => examples_dir().ok_or_else(|| {
            anyhow!("no scenes directory (set GHEE_EXAMPLES for the bundled scenes)")
        })?,
    };
    let mut store = SceneStore::with_mine(&dir, mine.as_deref(), args);
    store.demo = demo || is_examples_dir(std::path::Path::new(&dir));
    Ok(store)
}

/// The console's address as a URL that a browser on this machine opens.
fn url(host: &str, port: u16) -> String {
    let host = match host.parse::<std::net::IpAddr>() {
        Ok(ip) if ip.is_unspecified() => "localhost".to_string(),
        Ok(std::net::IpAddr::V6(ip)) => format!("[{ip}]"),
        _ => host.to_string(),
    };
    format!("http://{host}:{port}/")
}

/// Start the console and serve until stop() (or Ctrl-C).
pub fn run(opts: Options) -> Result<()> {
    let Options {
        arrays,
        scenes_dir,
        my_scenes,
        uploads,
        state_dir,
        size,
        port,
        host,
        display,
        rgb565,
        rotate,
        brightness,
        preview_max,
        preview_fps,
        stream_max,
        stream_hz: stream_fps,
        bitrate,
        scene_args,
        store,
        sinks,
        pacer,
        components,
        alpha,
        frames,
        timestep,
        dri_device,
        card,
        on_stats,
        filters,
    } = opts;
    let mut store = match store {
        Some(store) => store,
        None => dir_store(scenes_dir, my_scenes, scene_args)?,
    };
    let names = store.names();
    store.filters = filters;
    store.state_dir = state_dir
        .or_else(|| store.state_dir.take())
        .or_else(state::default_dir);
    let mut console = Console::new(store);
    console.w = size.0;
    console.h = size.1;
    console.display = display;
    console.rgb565 = rgb565;
    console.rotate = rotate;
    console.brightness = brightness;
    console.preview_max = preview_max;
    console.preview_fps = preview_fps;
    console.stream_max = stream_max;
    console.stream_hz = stream_fps;
    console.bitrate = bitrate;
    console.components = components;
    console.alpha = alpha;
    console.frames = frames;
    console.timestep = timestep;
    console.dri = dri_device;
    console.sink_dests = sinks
        .iter()
        .map(|s| s.dest().to_string())
        .filter(|d| !d.is_empty())
        .collect();
    console.arrays = arrays;
    lock(&console.stat).display =
        serde_json::json!({"w": size.0, "h": size.1, "format": "rgba", "rotation": rotate});
    if let Some(first) = names.first() {
        console.ask(first);
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
            ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)))
                .with_web(console.web)
                .listening_on(&host)
                .with_uploads(uploads),
        );
        thread::spawn(move || {
            for request in server.incoming_requests() {
                let (console, api) = (Arc::clone(&console), Arc::clone(&api));
                thread::spawn(move || handle(&console, &api, request));
            }
        })
    };

    match make_renderer(&console, card) {
        Ok(r) => {
            let count = match names.len() {
                0 => "no scene".to_string(),
                1 => "1 scene".to_string(),
                n => format!("{n} scenes"),
            };
            let mine = match console.store.mine.as_deref() {
                Some(m) if m != console.store.dir => format!(", my scenes={m}"),
                _ => String::new(),
            };
            println!(
                "ghee console: {} {count} [{}] (scenes={}{mine}, preview {}x{} rgba)",
                url(&host, port),
                names
                    .iter()
                    .map(|n| format!("'{n}'"))
                    .collect::<Vec<_>>()
                    .join(", "),
                console.store.dir,
                size.0,
                size.1,
            );
            render_loop(&console, Rc::new(RefCell::new(r)), sinks, pacer, on_stats);
        }
        Err(e) => {
            lock(&console.stat).error = e.to_string();
            *lock(&console.fatal) = e.to_string();
        }
    }
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
        "passes": [{"name": "main", "fragment": "void main(){ gl_FragColor = vec4(1.0); }"}]}"#;
    const SC: &[u8] = &[0, 0, 0, 1];

    /// Make `scene` the working copy of the active scene `name`, as a load
    /// ends.
    fn opened(console: &Console, name: &str, scene: Scene) {
        let opened = Opened {
            defaults: scene.clone(),
            working: scene,
            aside: Vec::new(),
            broken: Vec::new(),
            saved: None,
            procedural: false,
        };
        ready(console, name, opened);
    }

    /// Make `opened` the working copy of the active scene `name`, as a load
    /// ends.
    fn ready(console: &Console, name: &str, opened: Opened) {
        let switch = console.ask(name);
        console.install(name, opened, "");
        lock(&console.active).loaded = switch;
    }

    /// A store of clocks, one pass main with the params cols and rows, and
    /// its state directory.
    fn clocks(
        tag: &str,
    ) -> (
        crate::testing::TempDir,
        crate::testing::TempDir,
        Arc<Console>,
    ) {
        let d = crate::testing::TempDir::new(&format!("console-{tag}"));
        let doc = serde_json::json!({"version": 1, "passes": [{"name": "main",
            "fragment": "void main(){ gl_FragColor = vec4(1.0); }",
            "params": {"cols": {"type": "int", "value": 7},
                       "rows": {"type": "int", "value": 13}}}]});
        fs::write(d.join("clocks.scene.json"), doc.to_string()).unwrap();
        let states = crate::testing::TempDir::new(&format!("console-{tag}-state"));
        let mut store = SceneStore::new(&d, Vec::new());
        store.state_dir = Some(std::path::PathBuf::from(&*states));
        (d, states, Arc::new(Console::new(store)))
    }

    /// The state file of clocks in the directory `d`.
    fn clocks_file(d: &str, states: &str) -> std::path::PathBuf {
        std::path::Path::new(states).join(format!("scenes{d}/clocks.json"))
    }

    /// One request of a page: its status, its body and its ETag.
    fn ask(
        api: &ConsoleAPI<RenderLoopHost>,
        method: &str,
        path: &str,
        q: &str,
        body: &str,
    ) -> (u16, String, Option<String>) {
        let headers = Headers::from([("Content-Type", "application/json"), ("Host", "localhost")]);
        let r = api.dispatch(
            method,
            path,
            &parse_query(q).unwrap(),
            &headers,
            body.as_bytes(),
        );
        let etag = r
            .headers
            .iter()
            .find(|(k, _)| k == "ETag")
            .map(|(_, v)| v.clone());
        (
            r.status,
            String::from_utf8_lossy(&r.body).into_owned(),
            etag,
        )
    }

    fn json_of(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn two_params_save_alone_and_back_to_defaults_removes_the_state_file() {
        let (d, states, console) = clocks("two-params");
        let (opened, _) = console.store.open("clocks", None, (32, 24)).unwrap();
        ready(&console, "clocks", opened);
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let (_, _, first) = ask(&api, "GET", "/scene", "", "");
        for q in [
            "scene=clocks&pass=main&name=cols&value=5",
            "scene=clocks&pass=main&name=rows&value=4",
        ] {
            assert_eq!(ask(&api, "POST", "/set", q, "").0, 200);
        }
        let (_, scene, etag) = ask(&api, "GET", "/scene", "name=clocks", "");
        assert_eq!(json_of(&scene)["passes"][0]["params"]["cols"]["value"], 5);
        assert!(etag.is_some() && etag != first, "{etag:?} {first:?}");
        let (_, defaults, _) = ask(&api, "GET", "/scene", "name=clocks&defaults=1", "");
        assert_eq!(
            json_of(&defaults)["passes"][0]["params"]["cols"]["value"],
            7
        );
        let values = serde_json::json!({"passes": {"main": {"params": {"cols": 5, "rows": 4}}}});
        let edits = json_of(&ask(&api, "GET", "/edits", "name=clocks", "").1);
        assert_eq!(
            (&edits["values"], &edits["items"], &edits["unsaved"]),
            (&values, &serde_json::json!([]), &Value::from(true))
        );
        assert_eq!(ask(&api, "POST", "/save", "name=clocks", "").0, 200);
        let file = clocks_file(&d, &states);
        assert_eq!(
            json_of(&fs::read_to_string(&file).unwrap()),
            serde_json::json!({"version": 1, "scene": format!("{d}/clocks.scene.json"),
                               "values": values})
        );
        let edits = json_of(&ask(&api, "GET", "/edits", "", "").1);
        assert_eq!(edits["unsaved"], false);
        assert_eq!(ask(&api, "POST", "/reset", "name=clocks", "").0, 200);
        let edits = json_of(&ask(&api, "GET", "/edits", "", "").1);
        assert_eq!(
            (&edits["values"], &edits["unsaved"]),
            (&serde_json::json!({}), &Value::from(true))
        );
        assert_eq!(ask(&api, "POST", "/save", "name=clocks", "").0, 200);
        assert!(!file.exists());
    }

    #[test]
    fn a_shader_edit_whose_default_changed_waits_for_keep_mine_or_update_to_default() {
        let mine = "void main(){ gl_FragColor = vec4(0.5); }";
        let default = "void main(){ gl_FragColor = vec4(1.0); }";
        for answer in ["keep", "default"] {
            let (d, states, console) = clocks(answer);
            let file = clocks_file(&d, &states);
            fs::create_dir_all(file.parent().unwrap()).unwrap();
            let saved = serde_json::json!({"version": 1,
                "values": {"passes": {"main": {"params": {"cols": 5}}}},
                "items": [{"item": "fragment", "pass": "main", "base": "0123456789abcdef",
                           "value": mine}]});
            fs::write(&file, saved.to_string()).unwrap();
            let (opened, _) = console.store.open("clocks", None, (32, 24)).unwrap();
            ready(&console, "clocks", opened);
            let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
            let fragment = |api: &ConsoleAPI<RenderLoopHost>| {
                json_of(&ask(api, "GET", "/scene", "", "").1)["passes"][0]["fragment"].clone()
            };
            assert_eq!(fragment(&api), default);
            let edits = json_of(&ask(&api, "GET", "/edits", "", "").1);
            assert_eq!(
                edits["aside"],
                serde_json::json!([{"item": "fragment", "pass": "main"}])
            );
            assert_eq!(edits["unsaved"], false);
            let q = format!("name=clocks&item=fragment&pass=main&answer={answer}");
            assert_eq!(ask(&api, "POST", "/aside", &q, "").0, 200);
            let written = json_of(&fs::read_to_string(&file).unwrap());
            let edits = json_of(&ask(&api, "GET", "/edits", "", "").1);
            assert_eq!(
                (&edits["aside"], &edits["unsaved"]),
                (&serde_json::json!([]), &Value::from(false))
            );
            match answer {
                "keep" => {
                    // sha256sum of the default shader's text
                    assert_eq!(written["items"][0]["base"], "758cc820f81e647b");
                    assert_eq!(fragment(&api), mine);
                }
                _ => {
                    assert!(written.get("items").is_none(), "{written}");
                    assert_eq!(fragment(&api), default);
                }
            }
            assert_eq!(written["values"], saved["values"]);
            assert_eq!(
                ask(&api, "POST", "/aside", &q, ""),
                (404, "no such item is set aside".to_string(), None)
            );
        }
    }

    /// clocks of clocks(), and the console's own scenes in a directory of
    /// their own.
    fn with_mine(
        tag: &str,
    ) -> (
        crate::testing::TempDir,
        crate::testing::TempDir,
        Arc<Console>,
    ) {
        let (d, states, console) = clocks(tag);
        let m = crate::testing::TempDir::new(&format!("console-{tag}-mine"));
        let mut store = SceneStore::with_mine(&d, Some(&m), Vec::new());
        store.state_dir = console.store.state_dir.clone();
        drop(states);
        (d, m, Arc::new(Console::new(store)))
    }

    #[test]
    fn a_new_scene_copies_the_working_copy_and_saves_whole_with_no_state_file() {
        let (_d, m, console) = with_mine("own-new");
        let (opened, _) = console.store.open("clocks", None, (32, 24)).unwrap();
        ready(&console, "clocks", opened);
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let set = |scene: &str, value: u32| {
            let q = format!("scene={scene}&pass=main&name=cols&value={value}");
            ask(&api, "POST", "/set", &q, "").0
        };
        assert_eq!(set("clocks", 5), 200);
        let made = ask(&api, "POST", "/new", "name=copy&from=clocks", "");
        assert_eq!((made.0, made.1.as_str()), (200, r#"{"name":"copy"}"#));
        let scenes = json_of(&ask(&api, "GET", "/scenes", "", "").1);
        assert_eq!(scenes["scenes"], serde_json::json!(["clocks", "copy"]));
        assert_eq!(scenes["mine"], serde_json::json!(["copy"]));
        let (opened, _) = console.store.open("copy", None, (32, 24)).unwrap();
        assert_eq!(
            opened.defaults.passes[0]
                .params
                .get("cols")
                .unwrap()
                .value
                .to_json(),
            5
        );
        ready(&console, "copy", opened);
        assert_eq!(set("copy", 3), 200);
        assert!(json_of(&ask(&api, "GET", "/edits", "", "").1)["unsaved"] == true);
        assert_eq!(ask(&api, "POST", "/save", "name=copy", "").0, 200);
        let file = m.join("copy/copy.scene.json");
        let written = json_of(&fs::read_to_string(&file).unwrap());
        assert_eq!(written["passes"][0]["params"]["cols"]["value"], 3);
        assert_eq!(
            written["passes"][0]["fragment"],
            serde_json::json!({"file": "main.glsl"})
        );
        let edits = json_of(&ask(&api, "GET", "/edits", "", "").1);
        assert_eq!(
            (&edits["values"], &edits["unsaved"]),
            (&serde_json::json!({}), &Value::from(false))
        );
        assert_eq!(console.store.state_file("copy").unwrap(), None);
        assert_eq!(console.stats()["stale"], false);
    }

    #[test]
    fn a_new_scene_from_a_procedural_scene_is_refused() {
        let (d, _m, console) = with_mine("own-program");
        crate::testing::script(&d, "prog.py", "#!/bin/sh\nexit 1\n");
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        assert_eq!(
            ask(&api, "POST", "/new", "name=copy&from=prog", ""),
            (
                403,
                "a program writes scene 'prog', so it has no scene file to copy".to_string(),
                None
            )
        );
        assert_eq!(ask(&api, "POST", "/new", "name=copy&from=nope", "").0, 404);
        assert_eq!(ask(&api, "POST", "/new", "name=blank", "").0, 200);
        assert_eq!(console.store.mine_names(), ["blank"]);
    }

    #[test]
    fn the_active_scene_renamed_or_removed_takes_the_console_along() {
        let (_d, m, console) = with_mine("own-move");
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        assert_eq!(ask(&api, "POST", "/new", "name=one", "").0, 200);
        let (opened, _) = console.store.open("one", None, (32, 24)).unwrap();
        ready(&console, "one", opened);
        let renamed = ask(&api, "POST", "/rename", "name=one&to=two", "");
        assert_eq!((renamed.0, renamed.1.as_str()), (200, r#"{"name":"two"}"#));
        assert_eq!(console.stats()["active"], "two");
        assert!(m.join("two/two.scene.json").is_file());
        assert_eq!(ask(&api, "POST", "/rename", "name=clocks&to=x", "").0, 403);
        assert_eq!(ask(&api, "POST", "/delete", "name=two", "").0, 200);
        assert_eq!(console.stats()["active"], "clocks");
        assert!(!m.join("two").exists());
    }

    /// A saved image layer whose asset the scene lacks.
    #[test]
    fn saved_edits_that_make_no_scene_wait_in_the_state_file_until_dropped() {
        let (d, states, console) = clocks("broken");
        let file = clocks_file(&d, &states);
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        let layers = serde_json::json!({"item": "layers", "pass": "main",
            "base": state::hash_json(&serde_json::json!([])),
            "value": [{"name": "image", "image": "plaster"}]});
        let saved = serde_json::json!({"version": 1,
            "values": {"passes": {"main": {"params": {"cols": 5}}}}, "items": [layers]});
        fs::write(&file, saved.to_string()).unwrap();
        let (opened, _) = console.store.open("clocks", None, (32, 24)).unwrap();
        ready(&console, "clocks", opened);
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let scene = json_of(&ask(&api, "GET", "/scene", "", "").1);
        assert_eq!(scene["passes"][0]["params"]["cols"]["value"], 5);
        assert!(scene["passes"][0].get("layers").is_none(), "{scene}");
        let edits = json_of(&ask(&api, "GET", "/edits", "", "").1);
        assert_eq!(edits["unsaved"], false);
        assert_eq!(edits["broken"][0]["what"], "pass 'main''s layers");
        assert!(
            edits["broken"][0]["why"]
                .as_str()
                .unwrap()
                .starts_with("layer 'image' image 'plaster'"),
            "{edits}"
        );
        let set = "scene=clocks&pass=main&name=cols&value=6";
        assert_eq!(ask(&api, "POST", "/set", set, "").0, 200);
        assert_eq!(ask(&api, "POST", "/save", "name=clocks", "").0, 200);
        let written = json_of(&fs::read_to_string(&file).unwrap());
        assert_eq!(written["items"], serde_json::json!([layers]));
        assert_eq!(written["values"]["passes"]["main"]["params"]["cols"], 6);
        let answer = |a: &str| {
            ask(
                &api,
                "POST",
                "/aside",
                &format!("name=clocks&item=edits&answer={a}"),
                "",
            )
        };
        assert_eq!(answer("keep").0, 200);
        assert_eq!(json_of(&fs::read_to_string(&file).unwrap()), written);
        assert_eq!(answer("default").0, 200);
        let written = json_of(&fs::read_to_string(&file).unwrap());
        assert!(written.get("items").is_none(), "{written}");
        assert_eq!(written["values"]["passes"]["main"]["params"]["cols"], 6);
        let edits = json_of(&ask(&api, "GET", "/edits", "", "").1);
        assert_eq!(
            (&edits["broken"], &edits["unsaved"]),
            (&serde_json::json!([]), &Value::from(false))
        );
        assert_eq!(answer("default").0, 404);
    }

    #[test]
    fn a_new_edit_of_a_shader_set_aside_answers_it_and_saves_alone() {
        let mine = "void main(){ gl_FragColor = vec4(0.5); }";
        let newer = "void main(){ gl_FragColor = vec4(0.25); }";
        let (d, states, console) = clocks("newer");
        let file = clocks_file(&d, &states);
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        let saved = serde_json::json!({"version": 1,
            "items": [{"item": "fragment", "pass": "main", "base": "0123456789abcdef",
                       "value": mine}]});
        fs::write(&file, saved.to_string()).unwrap();
        let (opened, _) = console.store.open("clocks", None, (32, 24)).unwrap();
        ready(&console, "clocks", opened);
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let mut doc = json_of(&ask(&api, "GET", "/scene", "", "").1);
        doc["passes"][0]["fragment"] = Value::from(newer);
        let posted = ask(&api, "POST", "/scene", "name=clocks", &doc.to_string());
        assert_eq!(posted.0, 200, "{}", posted.1);
        let edits = json_of(&ask(&api, "GET", "/edits", "", "").1);
        assert_eq!(edits["aside"], serde_json::json!([]));
        assert_eq!(ask(&api, "POST", "/save", "name=clocks", "").0, 200);
        let written = json_of(&fs::read_to_string(&file).unwrap());
        let items = written["items"].as_array().unwrap();
        assert_eq!(items.len(), 1, "{written}");
        assert_eq!(items[0]["value"], newer);
    }

    /// The revision of the scene that a GET /scene answers, from its ETag.
    fn rev_of(api: &ConsoleAPI<RenderLoopHost>) -> u64 {
        let etag = ask(api, "GET", "/scene", "", "").2.unwrap();
        etag.trim_matches('"').parse().unwrap()
    }

    #[test]
    fn an_edit_that_another_client_made_stale_answers_412_and_a_pages_own_never_do() {
        let (_d, _states, console) = clocks("stale");
        let (opened, _) = console.store.open("clocks", None, (32, 24)).unwrap();
        ready(&console, "clocks", opened);
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let read = rev_of(&api);
        assert_eq!(console.stats()["rev"], read);
        let set = |name: &str, value: u32, rev: Option<u64>, vid: &str| {
            let mut q = format!("scene=clocks&pass=main&name={name}&value={value}&vid={vid}");
            if let Some(rev) = rev {
                q.push_str(&format!("&rev={rev}"));
            }
            ask(&api, "POST", "/set", &q, "")
        };
        // page a's sets, each sent before the last one answers
        for value in [1, 2] {
            assert_eq!(set("cols", value, Some(read), "a").0, 200);
        }
        assert_eq!(
            set("rows", 4, Some(read), "b"),
            (
                412,
                "this scene changed since you loaded it".to_string(),
                None
            )
        );
        let rows = |api| {
            json_of(&ask(api, "GET", "/scene", "", "").1)["passes"][0]["params"]["rows"]["value"]
                .clone()
        };
        assert_eq!(rows(&api), 13);
        // an edit with no rev, as curl sends it
        assert_eq!(set("rows", 5, None, "").0, 200);
        assert_eq!(set("cols", 3, Some(read), "a").0, 412);
        assert_eq!(console.stats()["rev"], rev_of(&api));
    }

    #[test]
    fn a_reload_from_a_page_left_on_another_scene_answers_409() {
        let (_d, console) = good_and_bad("reload");
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        api.host.switch(Some("good")).unwrap();
        assert_eq!(
            ask(&api, "POST", "/switch", "name=bad&rev=1&vid=a", ""),
            (
                409,
                "the console is on scene 'good', and this reload is for scene 'bad'".to_string(),
                None
            )
        );
        assert_eq!(console.stats()["active"], "good");
        assert_eq!(
            ask(&api, "POST", "/switch", "name=good&rev=1&vid=a", "").0,
            200
        );
        assert_eq!(ask(&api, "POST", "/switch", "name=bad", "").0, 200);
        assert_eq!(console.stats()["active"], "bad");
    }

    #[test]
    fn an_apply_of_a_procedural_scene_leaves_what_its_program_drew_and_set_out_of_its_edits() {
        let (_d, console) = tmp_console("program");
        let doc = |x: f64, color: Option<&str>, bar: Option<f64>| {
            let mut title = serde_json::json!({"name": "title", "text": "hi", "x": x});
            if let Some(c) = color {
                title["color"] = Value::from(c);
            }
            let mut layers = vec![title];
            if let Some(w) = bar {
                layers.push(serde_json::json!({"name": "bar", "shape": "rect", "width": w}));
            }
            serde_json::json!({"version": 1, "passes": [{"name": "main",
                "fragment": "void main(){ gl_FragColor = vec4(1.0); }", "layers": layers}]})
        };
        let scene = |d: Value| Scene::from_dict(&d, None, None).unwrap();
        ready(
            &console,
            "one",
            Opened {
                defaults: scene(doc(1.0, None, None)),
                working: scene(doc(1.0, None, None)),
                aside: Vec::new(),
                broken: Vec::new(),
                saved: None,
                procedural: true,
            },
        );
        // the program set title's x and drew bar
        console.publish(Some("one"), 1, &scene(doc(20.0, None, Some(4.0))));
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let posted = doc(20.0, Some("#ff0000"), Some(9.0)).to_string();
        assert_eq!(ask(&api, "POST", "/scene", "name=one", &posted).0, 200);
        let edits = json_of(&ask(&api, "GET", "/edits", "", "").1);
        assert_eq!(
            (&edits["values"], &edits["items"]),
            (
                &serde_json::json!({"layers": {"title": {"color": "#ff0000"}}}),
                &serde_json::json!([])
            )
        );
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_switch_away_and_back_keeps_the_edits_made_since_the_last_save() {
        let renderer = RefCell::new(gpu(8, 8));
        let (_d, console) = good_and_bad("away");
        let st = RefCell::new(Running::default());
        let ctrl: ControllerSlot = RefCell::new(None);
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        api.host.switch(Some("good")).unwrap();
        drain(&console, &renderer, &st, &ctrl);
        assert_eq!(
            ask(&api, "POST", "/set", "scene=good&name=k&value=0.5", "").0,
            200
        );
        for name in ["bad", "good"] {
            api.host.switch(Some(name)).unwrap();
            drain(&console, &renderer, &st, &ctrl);
        }
        let edits = json_of(&ask(&api, "GET", "/edits", "", "").1);
        assert_eq!(
            edits["values"],
            serde_json::json!({"passes": {"main": {"params": {"k": 0.5}}}})
        );
        let drawn = st.borrow().scene.as_ref().unwrap().passes[0]
            .params
            .get("k")
            .unwrap()
            .value
            .to_json();
        assert_eq!(drawn, Value::from(0.5));
    }

    /// The running document, as the render thread published it last.
    fn running_doc(console: &Console) -> Value {
        lock(&console.running).as_ref().unwrap().doc.clone()
    }

    /// A console over a directory of one scene, which lives as long as the
    /// TempDir.
    fn tmp_console(tag: &str) -> (crate::testing::TempDir, Arc<Console>) {
        let d = crate::testing::TempDir::new(&format!("console-{tag}"));
        fs::write(d.join("one.scene.json"), SCENE).unwrap();
        let console = Arc::new(Console::new(SceneStore::new(&d, Vec::new())));
        (d, console)
    }

    /// A renderer on the first GL device that opens.
    fn gpu(w: i32, h: i32) -> Renderer {
        let mut errs = Vec::new();
        for dev in crate::engine::probe_dri_devices() {
            match Renderer::new(w, h, 4, Some(&dev), false, false) {
                Ok(r) => return r,
                Err(e) => errs.push(format!("{dev}: {e}")),
            }
        }
        panic!(
            "no GL device opens ({}). A host without one runs the tests under \
             RUSTFLAGS=\"--cfg ghee_no_gpu\", which reports them as ignored",
            errs.join("; ")
        );
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
        // a scene param routes to a distinct scene-scoped queue item
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
            coerce("k", "int", &Value::from(2.6)).unwrap(),
            ParamValue::Int(3)
        );
        assert_eq!(
            coerce("k", "int", &Value::from(2.5)).unwrap(),
            ParamValue::Int(2)
        ); // half to even
        assert_eq!(
            coerce("k", "bool", &Value::from(1.0)).unwrap(),
            ParamValue::Bool(true)
        );
        assert_eq!(
            coerce("k", "float", &Value::from(0.5)).unwrap(),
            ParamValue::Float(0.5)
        );
        assert_eq!(
            coerce("k", "color", &serde_json::json!([1.0, 0.0, 0.5])).unwrap(),
            ParamValue::Color(crate::layer::Rgb::numbers([1.0, 0.0, 0.5]))
        );
        assert!(coerce("k", "color", &serde_json::json!([1.0, 0.0])).is_err());
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

    #[test]
    fn the_skip_interval_follows_whoever_consumes_frames() {
        assert_eq!(skip_interval(60, 0, 15.0, 0.5), Some(1.0 / 60.0));
        assert_eq!(skip_interval(0, 0, 15.0, 0.5), None);
        assert_eq!(skip_interval(0, 1, 15.0, 0.5), Some(1.0 / 15.0));
        assert_eq!(skip_interval(0, 1, 0.0, 0.5), Some(0.5));
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
        let (_d, mut console) = tmp_console("preview");
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
        let (_d, console) = tmp_console("preview-small");
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
        // the JPEG path ticks the counter published as stream_fps
        let (_d, console) = tmp_console("one-counter");
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

    #[test]
    fn a_failure_wakes_a_waiting_viewer_and_the_first_reason_stands() {
        let fanout = Arc::new(H264Fanout::new(Arc::new(Mutex::new(FrameRate::new()))));
        let c = fanout.add();
        let failer = {
            let fanout = Arc::clone(&fanout);
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(100));
                fanout.fail("first");
                fanout.fail("second");
            })
        };
        let t0 = monotonic();
        assert!(c.get(Duration::from_secs(5)).is_none());
        assert!(monotonic() - t0 < 2.0);
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
        let (_d, console) = tmp_console("late-au");
        let first = console.fanout.add();
        console.hardware_failed("no hardware encoder");
        let owned = console.software_stream().unwrap();
        console.fanout.remove(&first);
        let next = console.fanout.add();
        // the software encoder of the first stream hands over one more unit
        assert!(!console.fanout.push_to(owned, &key_au()));
        assert!(next.get(Duration::ZERO).is_none());
        assert!(!console.fanout.began());
        // the next stream tries hardware first, and the software lane takes it
        // when hardware fails
        assert!(!console.software_h264());
        console.hardware_failed("no hardware encoder");
        assert_eq!(console.fanout.failure(), None);
        assert_eq!(console.software_stream(), Some(console.fanout.stream()));
        assert!(console.fanout.push_to(console.fanout.stream(), &key_au()));
        assert!(next.get(Duration::ZERO).is_some());
    }

    /// One GET /h264 served by serve_h264, as the raw bytes on the socket.
    /// `run` drives the fanout while the request is served.
    fn h264_exchange(
        fanout: Arc<H264Fanout>,
        run: impl FnOnce(&H264Fanout) + Send + 'static,
    ) -> Vec<u8> {
        use std::io::Read;
        use std::net::TcpStream;
        let server = Server::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        let client = thread::spawn(move || {
            let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
            s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            s.write_all(b"GET /h264 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
                .unwrap();
            let mut out = Vec::new();
            let _ = s.read_to_end(&mut out); // the stream ends on its own terminal chunk
            out
        });
        let request = server.recv().unwrap();
        let c = fanout.add();
        let driver = {
            let fanout = Arc::clone(&fanout);
            thread::spawn(move || run(&fanout))
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

    /// One raw request through handle(), with a body cap of 10 bytes, and
    /// the raw answer.
    fn exchange(raw: &'static [u8]) -> String {
        use std::io::Read;
        use std::net::TcpStream;
        let (_d, console) = tmp_console("transport");
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console))).with_max_body(10);
        let server = Server::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        let client = thread::spawn(move || {
            let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
            s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            s.write_all(raw).unwrap();
            let mut out = Vec::new();
            let _ = s.read_to_end(&mut out); // the server closes the connection
            out
        });
        handle(&console, &api, server.recv().unwrap());
        String::from_utf8_lossy(&client.join().unwrap()).into_owned()
    }

    #[test]
    fn a_query_with_a_malformed_percent_escape_answers_400() {
        let out =
            exchange(b"GET /stats?a=%zz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        assert!(out.starts_with("HTTP/1.1 400"), "{out}");
        assert!(
            out.ends_with("bad query: '%zz' is not a percent escape"),
            "{out}"
        );
    }

    #[test]
    fn a_chunked_body_past_the_cap_answers_413() {
        let out = exchange(
            b"POST /ping HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\
              Transfer-Encoding: chunked\r\n\r\n6\r\n012345\r\n5\r\n67890\r\n0\r\n\r\n",
        );
        assert!(out.starts_with("HTTP/1.1 413"), "{out}");
        assert!(out.ends_with("body over the limit of 10 bytes"), "{out}");
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
        let out = h264_exchange(fanout, move |f| {
            thread::sleep(Duration::from_millis(100));
            f.fail(why);
        });
        let text = String::from_utf8_lossy(&out);
        assert!(text.starts_with("HTTP/1.1 503"), "{text}");
        assert!(text.ends_with(why), "{text}");
    }

    #[test]
    fn a_stream_that_fails_after_its_start_ends_on_an_error_frame() {
        let fanout = Arc::new(H264Fanout::new(Arc::new(Mutex::new(FrameRate::new()))));
        let why = "hardware encode failed: stalled";
        let out = h264_exchange(fanout, move |f| {
            thread::sleep(Duration::from_millis(100));
            f.opened(256, 64, HW_COLOR);
            f.push(&key_au());
            thread::sleep(Duration::from_millis(100));
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
        // its existing ping id
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
        let (_d, console) = tmp_console("counts");
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
    fn stats_carries_every_key_polled_by_the_page() {
        let (_d, console) = tmp_console("stats");
        let stats = console.stats();
        for key in [
            "fps",
            "frame_ms",
            "stream_fps",
            "out",
            "error",
            "error_line",
            "error_pass",
            "text_errors",
            "active",
            "display",
            "stale",
            "layer_changes",
            "rev",
            "locked",
            "gpu",
        ] {
            assert!(stats.get(key).is_some(), "{key}");
        }
        assert_eq!(stats["stale"], Value::from(false));
        assert_eq!(stats["out"], serde_json::json!([]));
    }

    /// An apply loads its assets and a save writes its file with the working
    /// copies held, and a poll of /stats answers through either.
    #[test]
    fn stats_answers_while_the_working_copies_are_held() {
        let (_d, console) = tmp_console("stats-held");
        let scene = console.store.resolve("one", (32, 24)).unwrap();
        opened(&console, "one", scene);
        let held = lock(&console.copies);
        let (tx, rx) = std::sync::mpsc::channel();
        let polled = Arc::clone(&console);
        thread::spawn(move || tx.send(polled.stats()["rev"].clone()));
        let rev = rx.recv_timeout(Duration::from_secs(5));
        drop(held);
        assert_eq!(rev, Ok(Value::from(1u64)));
    }

    /// `serve` over the bundled examples' directory keeps their state files
    /// by name, as `serve --demo` does, and any other directory by path.
    #[test]
    fn the_examples_directory_keeps_its_state_files_by_name() {
        let _e = crate::console::repo_examples();
        let states = std::path::Path::new("/states");
        let state_file = |dir: Option<String>, name: &str| {
            let mut store = dir_store(dir, None, Vec::new()).unwrap();
            store.state_dir = Some(states.to_path_buf());
            store.state_file(name).unwrap().unwrap()
        };
        let demo = states.join("demo/bars.json");
        assert_eq!(state_file(None, "bars"), demo);
        assert_eq!(state_file(Some("ghee/examples".to_string()), "bars"), demo);
        let (d, _) = tmp_console("not-examples");
        assert_eq!(
            state_file(Some(d.to_string()), "one"),
            states.join(format!("scenes{d}/one.json"))
        );
    }

    #[test]
    fn a_stream_viewer_shows_up_in_the_destinations() {
        let (_d, console) = tmp_console("dests");
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
        let (_d, console) = tmp_console("host");
        let host = RenderLoopHost::new(Arc::clone(&console));
        assert_eq!(host.scene_names(), Ok(vec!["one".to_string()]));
        assert_eq!(
            host.scene_def(Some("nope")).err().map(|e| e.status),
            Some(404)
        );
        assert_eq!(host.switch(Some("nope")).err().map(|e| e.status), Some(404));
    }

    #[test]
    fn a_switch_rides_the_command_queue() {
        let (_d, console) = tmp_console("switch");
        let host = RenderLoopHost::new(Arc::clone(&console));
        assert_eq!(host.switch(Some("one")), Ok("one".to_string()));
        match console.q.take() {
            Some(Command::Switch(name, _, _)) => assert_eq!(name, "one"),
            _ => panic!("no switch was queued"),
        }
    }

    #[test]
    fn an_edit_behind_a_switch_names_the_scene_that_the_switch_asked_for() {
        let d = crate::testing::TempDir::new("console-edit");
        for name in ["one", "two"] {
            fs::write(d.join(format!("{name}.scene.json")), SCENE).unwrap();
        }
        let console = Arc::new(Console::new(SceneStore::new(&d, Vec::new())));
        let host = RenderLoopHost::new(Arc::clone(&console));
        host.switch(Some("one")).unwrap();
        host.switch(Some("two")).unwrap();
        // No render thread drains the queue: the switch alone moves the scene
        // that an edit must name.
        assert_eq!(host.active_name(), Ok(Some("two".to_string())));
        assert_eq!(console.stats()["active"], "two");
        let mut ran = 0;
        let refused = host.edit_scene(
            &Edit {
                scene: Some("one"),
                ..Edit::default()
            },
            &mut || {
                ran += 1;
                Ok(())
            },
        );
        assert_eq!(refused.map_err(|e| e.status), Err(409));
        let mut active = lock(&console.active);
        active.loaded = active.asked;
        drop(active);
        host.edit_scene(
            &Edit {
                scene: Some("two"),
                ..Edit::default()
            },
            &mut || {
                ran += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(ran, 1);
    }

    #[test]
    fn an_edit_while_the_active_scene_loads_answers_503() {
        let (_d, console) = tmp_console("loading");
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        api.host.switch(Some("one")).unwrap();
        let set = api.dispatch(
            "POST",
            "/set",
            &parse_query("scene=one&name=k&value=1").unwrap(),
            &Headers::from([("Host", "localhost")]),
            b"",
        );
        let json = Headers::from([("Content-Type", "application/json"), ("Host", "localhost")]);
        let apply = api.dispatch(
            "POST",
            "/scene",
            &parse_query("name=one").unwrap(),
            &json,
            SCENE.as_bytes(),
        );
        for r in [set, apply] {
            assert_eq!(
                (r.status, String::from_utf8_lossy(&r.body).as_ref()),
                (503, "scene 'one' is loading")
            );
        }
    }

    /// The scenes good and bad, each with the param k, and bad's shader
    /// does not compile.
    fn good_and_bad(tag: &str) -> (crate::testing::TempDir, Arc<Console>) {
        let d = crate::testing::TempDir::new(&format!("console-{tag}"));
        let doc = |fragment: &str| {
            serde_json::json!({"version": 1, "passes": [{"name": "main", "fragment": fragment,
                "params": {"k": {"type": "float", "value": 0.0}}}]})
            .to_string()
        };
        fs::write(
            d.join("bad.scene.json"),
            doc("void main(){ gl_FragColor = nope; }"),
        )
        .unwrap();
        fs::write(
            d.join("good.scene.json"),
            doc("void main(){ gl_FragColor = vec4(k); }"),
        )
        .unwrap();
        let console = Arc::new(Console::new(SceneStore::new(&d, Vec::new())));
        (d, console)
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_scene_that_fails_to_compile_is_the_active_scene_and_takes_its_fix() {
        let renderer = RefCell::new(gpu(8, 8));
        let (_d, console) = good_and_bad("failed");
        let st = RefCell::new(Running::default());
        let ctrl: ControllerSlot = RefCell::new(None);
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let fragment = |doc: Value| doc["passes"][0]["fragment"].as_str().unwrap().to_string();
        api.host.switch(Some("good")).unwrap();
        drain(&console, &renderer, &st, &ctrl);
        api.host.switch(Some("bad")).unwrap();
        drain(&console, &renderer, &st, &ctrl);
        assert!(!console.stats()["error"].as_str().unwrap().is_empty());
        // The page opens the scene that failed, to fix it there.
        assert!(fragment(api.host.scene_def(None).unwrap()).contains("nope"));
        // A set for the scene that failed leaves the scene that draws alone.
        let set = api.dispatch(
            "POST",
            "/set",
            &parse_query("scene=bad&name=k&value=1").unwrap(),
            &Headers::from([("Host", "localhost")]),
            b"",
        );
        assert_eq!(set.status, 200);
        drain(&console, &renderer, &st, &ctrl);
        let k = |s: &RefCell<Running>| {
            s.borrow().scene.as_ref().unwrap().passes[0]
                .params
                .get("k")
                .unwrap()
                .value
                .to_json()
        };
        assert_eq!(k(&st), Value::from(0.0));
        let fix = r#"{"version": 1, "passes": [{"name": "main",
            "fragment": "void main(){ gl_FragColor = vec4(0.5); }"}]}"#;
        let json = Headers::from([("Content-Type", "application/json"), ("Host", "localhost")]);
        let r = api.dispatch(
            "POST",
            "/scene",
            &parse_query("name=bad").unwrap(),
            &json,
            fix.as_bytes(),
        );
        assert_eq!(r.status, 200);
        drain(&console, &renderer, &st, &ctrl);
        assert_eq!(console.stats()["error"], "");
        assert!(fragment(api.host.scene_def(Some("bad")).unwrap()).contains("0.5"));
        assert!(fragment(api.host.scene_def(Some("good")).unwrap()).contains("vec4(k)"));
    }

    #[test]
    fn a_scene_def_off_disk_is_the_file() {
        let (_d, console) = tmp_console("def");
        let host = RenderLoopHost::new(Arc::clone(&console));
        let doc = host.scene_def(Some("one")).unwrap();
        assert_eq!(doc["passes"][0]["name"], "main");
    }

    // a text

    /// The words of `label` stretched over the frame, in the fixture font.
    fn text_scene(words: &str) -> Scene {
        let doc = serde_json::json!({
            "version": 1,
            "assets": {"sans": {"file": "DejaVuSans-ASCII.ttf"}},
            "passes": [{"name": "main", "fragment": "void main(){ \
                float c = label_cover(uv * label_size, iResolution.y / label_size.y); \
                gl_FragColor = vec4(c, c, c, 1.0); }",
                "layers": [{"name": "label", "text": words, "font_family": "sans",
                            "visible": false}]}],
        });
        let mut sc = Scene::from_dict(&doc, None, None).unwrap();
        sc.load_assets(concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts"))
            .unwrap();
        sc
    }

    #[test]
    fn a_set_takes_its_type_from_the_running_scene() {
        let (_d, console) = tmp_console("settype");
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let set = |q: &str| {
            api.dispatch(
                "POST",
                "/set",
                &parse_query(q).unwrap(),
                &Headers::from([("Host", "localhost")]),
                b"",
            )
        };
        let r = set("layer=label&field=text&value=O");
        assert_eq!((r.status, r.body.as_slice()), (404, b"no scene".as_slice()));
        opened(&console, "one", text_scene("I"));
        assert_eq!(set("layer=label&field=text&value=hello").status, 200);
        match console.q.take() {
            Some(Command::Layer(set)) => {
                assert_eq!(set, text_set("hello"));
            }
            _ => panic!("no layer set queued"),
        }
        assert_eq!(set("name=nope&value=1").status, 400);
        assert!(console.q.take().is_none());
    }

    /// The set of the `text` of layer `label` to `words`.
    fn text_set(words: &str) -> LayerSet {
        LayerSet {
            layer: "label".to_string(),
            effect: None,
            field: "text".to_string(),
            value: Value::from(words),
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_posted_text_relays_its_words() {
        let renderer = RefCell::new(gpu(32, 24));
        let (_d, console) = tmp_console("text");
        let st = RefCell::new(Running::default());
        let ctrl: ControllerSlot = RefCell::new(None);
        let frame = || {
            renderer.borrow().set_time(Some(0.0));
            renderer.borrow_mut().render(None).unwrap().to_vec()
        };
        let set = |sc| set_scene(&console, &renderer, &st, sc, 0, None, false);
        assert_eq!(set(text_scene("I")), "");
        let i = frame();
        opened(&console, "one", text_scene("I"));
        // what a POST /set queues
        let host = RenderLoopHost::new(Arc::clone(&console));
        host.set_layer(text_set("O")).unwrap();
        assert!(drain(&console, &renderer, &st, &ctrl));
        let o = frame();
        assert!(o != i, "the frame kept the old words");
        let running = running_doc(&console);
        assert_eq!(running["passes"][0]["layers"][0]["text"], "O");
        // the relayed words draw what a scene built on them draws
        assert_eq!(set(text_scene("O")), "");
        assert!(frame() == o, "the relayed words drew another frame");
    }

    /// A controller that records each set that the console hands it.
    struct Told(Rc<RefCell<Vec<String>>>);

    impl SceneController for Told {
        fn on_param(&mut self, pass: Option<&str>, name: &str, value: &ParamValue) {
            let told = format!("param {pass:?} {name} {value:?}");
            self.0.borrow_mut().push(told);
        }

        fn on_layer(&mut self, at: &Address, value: &Value) {
            let told = format!("layer {}.{} {value}", at.layer, at.field);
            self.0.borrow_mut().push(told);
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn the_scenes_process_hears_of_each_param_and_each_text_that_the_console_sets() {
        let renderer = RefCell::new(gpu(32, 24));
        let (_d, console) = tmp_console("told");
        let st = RefCell::new(Running::default());
        let seen = Rc::new(RefCell::new(Vec::new()));
        let ctrl: ControllerSlot = RefCell::new(Some(Box::new(Told(Rc::clone(&seen)))));
        let mut sc = text_scene("I");
        sc.params
            .insert("gain".to_string(), Param::float("gain", 0.0));
        sc.passes[0]
            .params
            .insert("amt".to_string(), Param::float("amt", 0.0));
        assert_eq!(
            set_scene(&console, &renderer, &st, sc.clone(), 0, None, false),
            ""
        );
        opened(&console, "one", sc);
        let host = RenderLoopHost::new(Arc::clone(&console));
        host.set_param(Scope::Scene, "gain", Value::from(0.5))
            .unwrap();
        host.set_param(Scope::Pass(0), "amt", Value::from(0.25))
            .unwrap();
        host.set_layer(text_set("O")).unwrap();
        assert!(drain(&console, &renderer, &st, &ctrl));
        assert_eq!(
            *seen.borrow(),
            [
                "param None gain Float(0.5)",
                "param Some(\"main\") amt Float(0.25)",
                "layer label.text \"O\"",
            ]
        );
    }

    /// A scene of the layer `label`, "I" in the fixture font, at x.
    fn placed_scene(x: f64) -> Scene {
        let mut doc = text_scene("I").to_dict();
        let label = &mut doc["passes"][0]["layers"][0];
        label["x"] = Value::from(x);
        label["visible"] = Value::from(true);
        let mut sc = Scene::from_dict(&doc, None, None).unwrap();
        sc.load_assets(concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts"))
            .unwrap();
        sc
    }

    fn label_x(value: f64) -> LayerSet {
        LayerSet {
            layer: "label".to_string(),
            effect: None,
            field: "x".to_string(),
            value: Value::from(value),
        }
    }

    #[test]
    fn a_layer_set_takes_its_field_from_the_running_scene() {
        let (_d, console) = tmp_console("setlayer");
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let set = |q: &str| {
            api.dispatch(
                "POST",
                "/set",
                &parse_query(q).unwrap(),
                &Headers::from([("Host", "localhost")]),
                b"",
            )
        };
        let r = set("layer=label&field=x&value=4");
        assert_eq!((r.status, r.body.as_slice()), (404, b"no scene".as_slice()));
        opened(&console, "one", placed_scene(2.0));
        assert_eq!(set("layer=label&field=x&value=4").status, 200);
        match console.q.take() {
            Some(Command::Layer(set)) => assert_eq!(set, label_x(4.0)),
            _ => panic!("no layer field queued"),
        }
        assert_eq!(set("layer=nope&field=x&value=1").status, 400);
        assert!(console.q.take().is_none());
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_posted_layer_field_reaches_the_renderer_and_the_running_scene() {
        let renderer = RefCell::new(gpu(32, 24));
        let (_d, console) = tmp_console("layer");
        let st = RefCell::new(Running::default());
        let ctrl: ControllerSlot = RefCell::new(None);
        let frame = || {
            renderer.borrow().set_time(Some(0.0));
            renderer.borrow_mut().render(None).unwrap().to_vec()
        };
        let set = |sc| set_scene(&console, &renderer, &st, sc, 0, None, false);
        assert_eq!(set(placed_scene(2.0)), "");
        let before = frame();
        opened(&console, "one", placed_scene(2.0));
        // what a POST /set queues
        let host = RenderLoopHost::new(Arc::clone(&console));
        host.set_layer(label_x(12.0)).unwrap();
        assert!(drain(&console, &renderer, &st, &ctrl));
        let live = frame();
        assert!(live != before, "the frame kept the layer where it was");
        let running = running_doc(&console);
        assert_eq!(running["passes"][0]["layers"][0]["x"], 12.0);
        assert_eq!(set(placed_scene(12.0)), "");
        assert!(frame() == live, "the live field drew another frame");
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_saved_layer_field_starts_a_procedural_scene_and_a_tick_sets_it_over() {
        // The code of proc places the layer title at x 1, its state file at
        // x 8, and each answer of the process at x 20.
        let d = crate::testing::TempDir::new("console-proc");
        let doc = |x: f64| {
            serde_json::json!({"version": 1, "passes": [{"name": "main",
                "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
                "layers": [{"name": "title", "text": "I", "font_size": 16, "x": x, "y": 4}]}]})
        };
        let mut handshake = doc(1.0);
        handshake["lockstep"] = Value::from(true);
        // kind 1, title.x, one value: 20.0, then kind 5
        let answer = r"\001\007title.x\001\000\000\000\000\000\240\101\005";
        let body = format!(
            "#!/bin/sh\nprintf '%s\\n' '{handshake}'\nwhile read -r l; do printf '{answer}'; done\n"
        );
        let path = crate::testing::script(&d, "proc", &body);
        let states = crate::testing::TempDir::new("console-states");
        let saved = serde_json::json!({"version": 1, "values": {"layers": {"title": {"x": 8}}}});
        let file = state::file_of(std::path::Path::new(&*states), &path, None);
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, saved.to_string()).unwrap();
        let mut store = SceneStore::new(&d, Vec::new());
        store.state_dir = Some(std::path::PathBuf::from(&*states));
        let console = Console::new(store);
        let renderer = RefCell::new(gpu(48, 24));
        let st = RefCell::new(Running::default());
        let ctrl: ControllerSlot = RefCell::new(None);
        let frame = || {
            renderer.borrow().set_time(Some(0.0));
            renderer.borrow_mut().render(None).unwrap().to_vec()
        };
        let built = |x: f64| {
            let sc = Scene::from_dict(&doc(x), None, None).unwrap();
            assert_eq!(renderer.borrow_mut().setup(&sc).unwrap(), "");
            frame()
        };
        let (at8, at20) = (built(8.0), built(20.0));
        {
            let _spawn = crate::testing::SCRIPTS.read().unwrap();
            load(&console, &renderer, &st, &ctrl, "proc", 0, "");
        }
        assert!(frame() == at8, "the saved x places the layer");
        // The loop's tick and update, until the process answers.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        for n in 0.. {
            let mut slot = ctrl.borrow_mut();
            let running = slot.as_mut().expect("the scene runs");
            running.tick(n, 0.0, 1.0 / 60.0).unwrap();
            running.update(&mut *renderer.borrow_mut(), n, 0.0).unwrap();
            drop(slot);
            if frame() == at20 || std::time::Instant::now() > deadline {
                break;
            }
        }
        assert!(frame() == at20, "the tick's x places the layer");
    }

    /// A sink that keeps the last frame that it takes.
    struct Grab(Rc<RefCell<Vec<u8>>>);

    impl Output for Grab {
        fn write(&mut self, frame: Frame<'_>) -> Result<()> {
            let mut held = self.0.borrow_mut();
            held.clear();
            held.extend_from_slice(frame.data);
            Ok(())
        }
    }

    /// One wire message as the octal escapes of a script's printf.
    fn wire(kind: u8, name: &str, count: usize, payload: &[u8]) -> String {
        let mut bytes = vec![kind, name.len() as u8];
        bytes.extend(name.as_bytes());
        bytes.extend((count as u32).to_le_bytes());
        bytes.extend(payload);
        bytes.iter().map(|b| format!("\\{b:03o}")).collect()
    }

    /// The red and the green of a pixel of a frame, and whether a pixel of it
    /// is blue.
    type Drawn = (u8, u8, bool);

    /// A console operation: what it is, what runs it and the frame after it.
    type Op<'a> = (&'static str, Box<dyn Fn() + 'a>, Drawn);

    /// The fragment of pass tint, which draws main with the param `amt` in
    /// green.
    const TINT: &str =
        "void main(){ vec4 c = texture2D(iChannel0, uv); gl_FragColor = vec4(c.r, amt, c.b, 1.0); }";

    /// A lockstep scene process whose pass main draws its array `level` in
    /// red, under its pass tint. It answers its first tick alone: `level` as
    /// the byte 128, `amt` at 0.75, and the blue rect `dot` added to main.
    fn sent_once_script() -> String {
        let doc = serde_json::json!({"version": 1, "lockstep": true, "passes": [
            {"name": "main", "arrays": [{"name": "level", "size": 1}],
             "fragment": "void main(){ gl_FragColor = vec4(level_at(0), 0.0, 0.0, 1.0); }"},
            {"name": "tint", "channels": [{"buffer": "main"}],
             "params": {"amt": {"type": "float", "value": 0.0}}, "fragment": TINT}]});
        let dot = r#"{"index": 0, "layer": {"name": "dot", "shape": "rect", "x": 0, "y": 0, "width": 4, "height": 4, "color": "blue"}}"#;
        let answer = [
            wire(4, "level", 1, &[128]),
            wire(1, "amt", 1, &0.75f32.to_le_bytes()),
            wire(6, "main", dot.len(), dot.as_bytes()),
        ]
        .concat();
        format!(
            "#!/bin/sh\nprintf '%s\\n' '{doc}'\nread -r l\nprintf '{answer}\\005'\n\
             while read -r l; do printf '\\005'; done\n"
        )
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn every_console_rebuild_keeps_what_a_scene_process_sent_once() {
        let d = crate::testing::TempDir::new("console-sent");
        let path = crate::testing::script(&d, "proc", &sent_once_script());
        fs::write(d.join("other.scene.json"), SCENE).unwrap();
        // a saved edit of tint's fragment, made over another default, which
        // waits for an answer
        let states = crate::testing::TempDir::new("console-sent-states");
        let file = state::file_of(std::path::Path::new(&*states), &path, None);
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        let saved = serde_json::json!({"version": 1, "items": [{"item": "fragment",
            "pass": "tint", "base": "0123456789abcdef", "value": format!("{TINT} // kept")}]});
        fs::write(&file, saved.to_string()).unwrap();
        let mut store = SceneStore::new(&d, Vec::new());
        store.state_dir = Some(std::path::PathBuf::from(&*states));
        let console = Arc::new(Console::new(store));
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let renderer = Rc::new(RefCell::new(gpu(16, 8)));
        let st = Rc::new(RefCell::new(Running::default()));
        let ctrl: ControllerSlot = RefCell::new(None);
        let got = Rc::new(RefCell::new(Vec::new()));
        // One iteration of the console's loop. Its frame as the red and the
        // green of a pixel right of dot, and whether a pixel is dot's blue.
        let frame = || {
            let _spawn = crate::testing::SCRIPTS.read().unwrap();
            let mut shared = SharedRenderer {
                renderer: Rc::clone(&renderer),
                running: Rc::clone(&st),
            };
            let mut grab = Grab(Rc::clone(&got));
            let mut lp = Loop::new(&mut shared, vec![&mut grab]);
            lp.frames = 1;
            lp.controller = Some(&ctrl);
            lp.poll = Some(Box::new(|| drain(&console, &renderer, &st, &ctrl)));
            lp.run().unwrap();
            drop(lp);
            publish_changes(&console, &st);
            let px = got.borrow();
            let blue = px.as_chunks::<4>().0.iter().any(|p| p[2] > 200);
            (px[12 * 4], px[12 * 4 + 1], blue)
        };
        let doc = || json_of(&ask(&api, "GET", "/scene", "name=proc", "").1);
        let post = |doc: &Value| {
            let posted = ask(&api, "POST", "/scene", "name=proc", &doc.to_string());
            assert_eq!(posted.0, 200, "{}", posted.1);
        };
        let with = |pass: usize, key: &str, value: Value| {
            let mut d = doc();
            d["passes"][pass][key] = value;
            d
        };
        let near = |(r, g, blue): Drawn, want: Drawn| {
            r.abs_diff(want.0) <= 2 && g.abs_diff(want.1) <= 2 && blue == want.2
        };
        let sent = (128, 191, true);
        api.host.switch(Some("proc")).unwrap();
        let first = frame();
        assert!(near(first, sent), "the process's first answer: {first:?}");
        let ops: Vec<Op> = vec![
            (
                "an apply of the scene as it runs",
                Box::new(|| post(&doc())),
                sent,
            ),
            (
                "a kept edit that waited for its answer",
                Box::new(|| {
                    let q = "name=proc&item=fragment&pass=tint&answer=keep";
                    assert_eq!(ask(&api, "POST", "/aside", q, "").0, 200);
                }),
                sent,
            ),
            (
                "an apply that hides tint",
                Box::new(|| post(&with(1, "visible", false.into()))),
                (128, 0, true),
            ),
            (
                "an apply that shows tint again",
                Box::new(|| post(&with(1, "visible", true.into()))),
                sent,
            ),
            (
                "an apply that fails to compile",
                Box::new(|| post(&with(0, "fragment", "void main(){ nope; }".into()))),
                sent,
            ),
            (
                "a reset to the defaults",
                Box::new(|| assert_eq!(ask(&api, "POST", "/reset", "name=proc", "").0, 200)),
                sent,
            ),
            (
                "a switch away and back",
                Box::new(|| {
                    api.host.switch(Some("other")).unwrap();
                    frame();
                    api.host.switch(Some("proc")).unwrap();
                }),
                sent,
            ),
        ];
        let mut lost = Vec::new();
        for (op, run, want) in &ops {
            run();
            let drawn = frame();
            let layers = doc()["passes"][0]["layers"].clone();
            let dot = layers
                .as_array()
                .is_some_and(|l| l.iter().any(|l| l["name"] == "dot"));
            if !near(drawn, *want) || !dot {
                lost.push(format!(
                    "{op}: drew {drawn:?} for {want:?}, layers {layers}"
                ));
            }
        }
        assert!(lost.is_empty(), "{}", lost.join("\n"));
    }

    /// A lockstep scene process with the param `amt` and the text layer
    /// `title` in its pass main. It answers its first tick with nothing, and
    /// its second with `amt` at 0.75, `title.x` at 20 and the rect `dot`
    /// added to main.
    fn sets_later_script() -> String {
        let doc = serde_json::json!({"version": 1, "lockstep": true, "passes": [
            {"name": "main", "params": {"amt": {"type": "float", "value": 0.0}},
             "fragment": "void main(){ gl_FragColor = vec4(amt, 0.0, 0.0, 1.0); }",
             "layers": [{"name": "title", "text": "I", "font_size": 8, "x": 1, "y": 1}]}]});
        let dot =
            r#"{"index": 1, "layer": {"name": "dot", "shape": "rect", "width": 2, "height": 2}}"#;
        let answer = [
            wire(1, "amt", 1, &0.75f32.to_le_bytes()),
            wire(1, "title.x", 1, &20.0f32.to_le_bytes()),
            wire(6, "main", dot.len(), dot.as_bytes()),
        ]
        .concat();
        format!(
            "#!/bin/sh\nprintf '%s\\n' '{doc}'\nread -r l\nprintf '\\005'\nread -r l\n\
             printf '{answer}\\005'\nwhile read -r l; do printf '\\005'; done\n"
        )
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn what_a_scene_process_sets_locks_from_its_first_set_until_the_process_ends() {
        let d = crate::testing::TempDir::new("console-locks");
        let path = crate::testing::script(&d, "proc", &sets_later_script());
        fs::write(d.join("other.scene.json"), SCENE).unwrap();
        let states = crate::testing::TempDir::new("console-locks-states");
        let mut store = SceneStore::new(&d, Vec::new());
        store.state_dir = Some(std::path::PathBuf::from(&*states));
        let console = Arc::new(Console::new(store));
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let renderer = Rc::new(RefCell::new(gpu(16, 8)));
        let st = Rc::new(RefCell::new(Running::default()));
        let ctrl: ControllerSlot = RefCell::new(None);
        let got = Rc::new(RefCell::new(Vec::new()));
        let frame = || {
            let _spawn = crate::testing::SCRIPTS.read().unwrap();
            let mut shared = SharedRenderer {
                renderer: Rc::clone(&renderer),
                running: Rc::clone(&st),
            };
            let mut grab = Grab(Rc::clone(&got));
            let mut lp = Loop::new(&mut shared, vec![&mut grab]);
            lp.frames = 1;
            lp.controller = Some(&ctrl);
            lp.poll = Some(Box::new(|| drain(&console, &renderer, &st, &ctrl)));
            lp.run().unwrap();
            drop(lp);
            publish_changes(&console, &st);
        };
        let locked = || console.stats()["locked"].clone();
        let none = serde_json::json!({"params": [], "fields": [], "layers": []});
        let set = |q: &str| {
            let (status, body, _) = ask(&api, "POST", "/set", &format!("scene=proc&{q}"), "");
            (status, body)
        };
        api.host.switch(Some("proc")).unwrap();
        frame();
        assert_eq!(locked(), none, "before the process sets anything");
        assert_eq!(set("pass=main&name=amt&value=0.25").0, 200);
        let all =
            serde_json::json!({"params": ["main.amt"], "fields": ["title.x"], "layers": ["dot"]});
        let deadline = Instant::now() + Duration::from_secs(10);
        while locked() != all && Instant::now() < deadline {
            frame();
        }
        assert_eq!(locked(), all);
        assert_eq!(
            set("pass=main&name=amt&value=0.5"),
            (403, "the scene's program sets param 'main.amt'".to_string())
        );
        assert_eq!(
            set("layer=title&field=x&value=3"),
            (
                403,
                "the scene's program sets layer field 'title.x'".to_string()
            )
        );
        assert_eq!(
            set("layer=dot&field=x&value=3"),
            (403, "the scene's program sets layer 'dot'".to_string())
        );
        assert_eq!(set("layer=title&field=y&value=3").0, 200);
        // a page one step behind posts the values from before the sets
        let mut stale = json_of(&ask(&api, "GET", "/scene", "name=proc", "").1);
        stale["passes"][0]["params"]["amt"]["value"] = Value::from(0.9);
        for layer in stale["passes"][0]["layers"].as_array_mut().unwrap() {
            layer["x"] = Value::from(9);
            if layer["name"] == "title" {
                layer["y"] = Value::from(6);
            }
        }
        let posted = ask(&api, "POST", "/scene", "name=proc", &stale.to_string());
        assert_eq!(posted.0, 200, "{}", posted.1);
        let only_y = serde_json::json!({"layers": {"title": {"y": 6.0}}});
        let edits = json_of(&ask(&api, "GET", "/edits", "name=proc", "").1);
        assert_eq!(edits["values"], only_y);
        assert_eq!(ask(&api, "POST", "/save", "name=proc", "").0, 200);
        let file = state::file_of(std::path::Path::new(&*states), &path, None);
        let saved = json_of(&fs::read_to_string(&file).unwrap());
        assert_eq!(saved["values"], only_y);
        assert_eq!(ask(&api, "POST", "/reset", "name=proc", "").0, 200);
        frame();
        assert_eq!(locked(), all, "back to the defaults");
        api.host.switch(Some("other")).unwrap();
        frame();
        api.host.switch(Some("proc")).unwrap();
        frame();
        assert_eq!(locked(), none, "a new process");
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_feedback_pass_counts_on_through_each_rebuild_and_starts_over_on_a_reset_or_a_switch() {
        // acc adds 1 to its red each frame, so the red of the frame counts the
        // frames since acc's buffers were new.
        let d = crate::testing::TempDir::new("console-count");
        let copy = "void main(){ gl_FragColor = texture2D(iChannel0, uv); }";
        let doc = serde_json::json!({"version": 1, "passes": [
            {"name": "acc", "channels": [{"buffer": "acc"}],
             "fragment": "void main(){ gl_FragColor = texture2D(iChannel0, uv) \
                          + vec4(1.0 / 255.0, 0.0, 0.0, 0.0); }"},
            {"name": "post", "channels": [{"buffer": "acc"}], "fragment": copy}]});
        fs::write(d.join("count.scene.json"), doc.to_string()).unwrap();
        fs::write(d.join("other.scene.json"), SCENE).unwrap();
        let states = crate::testing::TempDir::new("console-count-states");
        let mut store = SceneStore::new(&d, Vec::new());
        store.state_dir = Some(std::path::PathBuf::from(&*states));
        // a saved edit of post's fragment, made over another default, which
        // waits for an answer
        let file = store.state_file("count").unwrap().unwrap();
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        let saved = serde_json::json!({"version": 1, "items": [{"item": "fragment",
            "pass": "post", "base": "0123456789abcdef", "value": format!("{copy} // kept")}]});
        fs::write(&file, saved.to_string()).unwrap();
        let console = Arc::new(Console::new(store));
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let renderer = Rc::new(RefCell::new(gpu(16, 8)));
        let st = Rc::new(RefCell::new(Running::default()));
        let ctrl: ControllerSlot = RefCell::new(None);
        let got = Rc::new(RefCell::new(Vec::new()));
        // One iteration of the console's loop, and the red of its frame.
        let frame = || {
            let mut shared = SharedRenderer {
                renderer: Rc::clone(&renderer),
                running: Rc::clone(&st),
            };
            let mut grab = Grab(Rc::clone(&got));
            let mut lp = Loop::new(&mut shared, vec![&mut grab]);
            lp.frames = 1;
            lp.controller = Some(&ctrl);
            lp.poll = Some(Box::new(|| drain(&console, &renderer, &st, &ctrl)));
            lp.run().unwrap();
            drop(lp);
            got.borrow()[0]
        };
        let doc = || json_of(&ask(&api, "GET", "/scene", "name=count", "").1);
        let post = |doc: &Value| {
            let posted = ask(&api, "POST", "/scene", "name=count", &doc.to_string());
            assert_eq!(posted.0, 200, "{}", posted.1);
        };
        let with = |pass: usize, key: &str, value: Value| {
            let mut d = doc();
            d["passes"][pass][key] = value;
            d
        };
        api.host.switch(Some("count")).unwrap();
        let mut count = frame();
        assert_eq!(count, 1, "the first frame");
        // what each operation does, and whether acc counts on through it
        type Counts<'a> = (&'static str, Box<dyn Fn() + 'a>, bool);
        let ops: Vec<Counts> = vec![
            (
                "an apply of the scene as it runs",
                Box::new(|| post(&doc())),
                true,
            ),
            (
                "a kept edit that waited for its answer",
                Box::new(|| {
                    let q = "name=count&item=fragment&pass=post&answer=keep";
                    assert_eq!(ask(&api, "POST", "/aside", q, "").0, 200);
                }),
                true,
            ),
            (
                "an apply that hides post",
                Box::new(|| post(&with(1, "visible", false.into()))),
                true,
            ),
            (
                "an apply that shows post again",
                Box::new(|| post(&with(1, "visible", true.into()))),
                true,
            ),
            (
                "an apply that fails to compile",
                Box::new(|| post(&with(0, "fragment", "void main(){ nope; }".into()))),
                true,
            ),
            (
                "a reset to the defaults",
                Box::new(|| assert_eq!(ask(&api, "POST", "/reset", "name=count", "").0, 200)),
                false,
            ),
            (
                "a switch away and back",
                Box::new(|| {
                    api.host.switch(Some("other")).unwrap();
                    frame();
                    api.host.switch(Some("count")).unwrap();
                }),
                false,
            ),
        ];
        let mut wrong = Vec::new();
        for (op, run, counts_on) in &ops {
            run();
            let want = if *counts_on { count + 1 } else { 1 };
            count = frame();
            if count != want {
                wrong.push(format!("{op}: counted {count} for {want}"));
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn posted_words_past_the_texture_keep_the_last_words_and_show_why() {
        let renderer = RefCell::new(gpu(32, 24));
        let (_d, console) = tmp_console("toolong");
        let st = RefCell::new(Running::default());
        let ctrl: ControllerSlot = RefCell::new(None);
        let seen = Cell::new(renderer.borrow().text_errors_seq());
        let frame = || {
            renderer.borrow().set_time(Some(0.0));
            renderer.borrow_mut().render(None).unwrap().to_vec()
        };
        // /stats as (error, text_errors).
        let errors = || {
            publish_text_errors(&console, &renderer.borrow(), &seen);
            let stats = console.stats();
            (stats["error"].clone(), stats["text_errors"].clone())
        };
        assert_eq!(
            set_scene(&console, &renderer, &st, text_scene("O"), 0, None, false),
            ""
        );
        let o = frame();
        assert_eq!(errors(), (Value::from(""), serde_json::json!([])));
        opened(&console, "one", text_scene("O"));
        let host = RenderLoopHost::new(Arc::clone(&console));
        let long = "I".repeat(400_000);
        host.set_layer(text_set(&long)).unwrap();
        assert!(drain(&console, &renderer, &st, &ctrl));
        assert!(frame() == o, "the last words went");
        let (error, texts) = errors();
        assert_eq!(error, "", "a text problem is no scene error");
        let e = texts[0].as_str().unwrap();
        assert!(e.starts_with("layer 'label': ") && e.contains("VC4"), "{e}");
        // The document holds the words posted.
        let running = running_doc(&console);
        assert_eq!(running["passes"][0]["layers"][0]["text"], long.as_str());
        host.set_layer(text_set("I")).unwrap();
        assert!(drain(&console, &renderer, &st, &ctrl));
        assert_eq!(errors(), (Value::from(""), serde_json::json!([])));
    }

    // a scene's own layers

    /// The layers of pass main of the running document, as the console
    /// serves it, every number as a float.
    fn running_layers(console: &Console) -> Value {
        floats(&running_doc(console)["passes"][0]["layers"])
    }

    /// `v` with every number as a float, as the document writes a number.
    fn floats(v: &Value) -> Value {
        match v {
            Value::Number(n) => Value::from(n.as_f64().unwrap()),
            Value::Array(a) => Value::Array(a.iter().map(floats).collect()),
            Value::Object(o) => {
                Value::Object(o.iter().map(|(k, v)| (k.clone(), floats(v))).collect())
            }
            v => v.clone(),
        }
    }

    fn changes(console: &Console) -> Value {
        console.stats()["layer_changes"].clone()
    }

    /// A change that a scene's code makes through the loop's target, what
    /// it is, and the layers of the running document after it.
    type Step<'a> = (&'static str, Box<dyn Fn(&mut SharedRenderer) + 'a>, Value);

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn the_layers_that_a_scenes_code_changes_reach_the_running_document() {
        let renderer = Rc::new(RefCell::new(gpu(32, 24)));
        let (_d, console) = tmp_console("drawn");
        let st = Rc::new(RefCell::new(Running::default()));
        let scene = Scene::from_dict(&serde_json::from_str(SCENE).unwrap(), None, None).unwrap();
        assert_eq!(
            set_scene(&console, &renderer, &st, scene, 0, None, false),
            ""
        );
        // what a procedural scene's drawing and ticks hand the loop's target
        let mut target = SharedRenderer {
            renderer: Rc::clone(&renderer),
            running: Rc::clone(&st),
        };
        let bar = serde_json::json!({"name": "bar", "shape": "rect", "width": 4});
        let words = serde_json::json!({"name": "label", "text": "hi", "font_size": 8});
        let steps: [Step<'_>; 6] = [
            (
                "added",
                Box::new(|t| t.add_layer("main", 0, &bar).unwrap()),
                serde_json::json!([bar]),
            ),
            (
                "a text added after it",
                Box::new(|t| t.add_layer("main", 1, &words).unwrap()),
                serde_json::json!([bar, words]),
            ),
            (
                "a new look",
                Box::new(|t| {
                    let look = serde_json::json!({"name": "bar", "shape": "rect", "width": 9});
                    assert!(t.update_layer(&look).unwrap());
                }),
                serde_json::json!([{"name": "bar", "shape": "rect", "width": 9}, words]),
            ),
            (
                "a field set",
                Box::new(|t| {
                    let at = Address {
                        layer: "bar",
                        effect: None,
                        field: "visible",
                    };
                    t.set_layer(&at, Given::Numbers(&[0.0])).unwrap();
                }),
                serde_json::json!([
                    {"name": "bar", "visible": false, "shape": "rect", "width": 9},
                    words
                ]),
            ),
            (
                "new words",
                Box::new(|t| {
                    let at = Address {
                        layer: "label",
                        effect: None,
                        field: "text",
                    };
                    t.set_layer(&at, Given::Text("ho")).unwrap();
                }),
                serde_json::json!([
                    {"name": "bar", "visible": false, "shape": "rect", "width": 9},
                    {"name": "label", "text": "ho", "font_size": 8}
                ]),
            ),
            (
                "a field of a text whose words changed",
                Box::new(|t| {
                    let at = Address {
                        layer: "label",
                        effect: None,
                        field: "x",
                    };
                    t.set_layer(&at, Given::Numbers(&[3.0])).unwrap();
                }),
                serde_json::json!([
                    {"name": "bar", "visible": false, "shape": "rect", "width": 9},
                    {"name": "label", "text": "ho", "x": 3, "font_size": 8}
                ]),
            ),
        ];
        for (n, (what, step, layers)) in steps.iter().enumerate() {
            step(&mut target);
            publish_changes(&console, &st);
            assert_eq!(running_layers(&console), floats(layers), "{what}");
            assert_eq!(changes(&console), Value::from(n + 1), "{what}");
        }
        // a frame with no change publishes nothing new
        publish_changes(&console, &st);
        assert_eq!(changes(&console), Value::from(steps.len()));
    }

    /// A process stand-in: a scene with something in the controller slot is
    /// procedural.
    struct Code;

    impl SceneController for Code {}

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn an_apply_keeps_a_layer_of_a_procedural_scene_that_it_leaves_out() {
        let doc = |layers: Value| {
            let mut d: Value = serde_json::from_str(SCENE).unwrap();
            d["passes"][0]["layers"] = layers;
            Scene::from_dict(&d, None, None).unwrap()
        };
        let a = serde_json::json!({"name": "a", "shape": "rect", "width": 2});
        let b = serde_json::json!({"name": "b", "shape": "line", "points": [[0, 0], [4, 4]]});
        let c = serde_json::json!({"name": "c", "shape": "rect", "width": 6});
        let posted = serde_json::json!({"name": "c", "shape": "rect", "width": 5});
        for (code, kept) in [
            (true, serde_json::json!([a, b, posted])),
            (false, serde_json::json!([posted])),
        ] {
            let renderer = RefCell::new(gpu(32, 24));
            let (_d, console) = tmp_console("keep");
            let st = RefCell::new(Running::default());
            let ctrl: ControllerSlot = RefCell::new(None);
            let running = doc(serde_json::json!([a, b, c]));
            assert_eq!(
                set_scene(&console, &renderer, &st, running, 0, None, false),
                ""
            );
            if code {
                *ctrl.borrow_mut() = Some(Box::new(Code));
            }
            let before = changes(&console);
            console.command(Command::SceneObj(
                Box::new(doc(serde_json::json!([posted]))),
                0,
                false,
            ));
            assert!(drain(&console, &renderer, &st, &ctrl));
            assert_eq!(running_layers(&console), floats(&kept), "code {code}");
            let names: Vec<String> = renderer
                .borrow()
                .with_scene(|s| s.layers().map(|l| l.name().to_string()).collect())
                .unwrap();
            let want: Vec<&str> = kept
                .as_array()
                .unwrap()
                .iter()
                .map(|l| l["name"].as_str().unwrap())
                .collect();
            assert_eq!(names, want, "code {code}");
            // a page reads the scene again for the layers that it lacked
            let moved = changes(&console) != before;
            assert_eq!(moved, code, "code {code}");
        }
    }

    // run()'s own loop

    /// A sink that never leaves the thread that built it.
    struct Bound(Rc<Cell<u32>>);

    impl Output for Bound {
        fn dest(&self) -> &str {
            "bound"
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
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn the_console_draws_to_a_thread_bound_sink_on_the_callers_thread() {
        drop(gpu(8, 8)); // run() opens a renderer of its own on that device
        let (_d, console) = tmp_console("bound"); // for its scene directory
        let drawn = Rc::new(Cell::new(0));
        let ran = run(Options {
            store: Some(SceneStore::new(&console.store.dir, Vec::new())),
            port: free_port(),
            size: (32, 24),
            frames: 3,
            sinks: vec![Box::new(Bound(Rc::clone(&drawn)))],
            ..Options::default()
        });
        assert!(ran.is_ok(), "{:?}", ran.err());
        assert_eq!(drawn.get(), 3);
    }

    /// What the stats hook hands back: frame count, wall time, destinations.
    type Summary = Rc<RefCell<Option<(usize, f64, Vec<String>)>>>;

    /// --stats needs this to print its line after a bounded serve run.
    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_bounded_console_run_hands_back_its_timing() {
        drop(gpu(8, 8)); // run() opens a renderer of its own on that device
        let (_d, console) = tmp_console("stats");
        let summary: Summary = Rc::new(RefCell::new(None));
        let got = Rc::clone(&summary);
        let ran = run(Options {
            store: Some(SceneStore::new(&console.store.dir, Vec::new())),
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
        assert_eq!(out, ["bound"]);
    }

    /// A sink that reads GET /stats from the console on `port` at the first
    /// frame that it takes.
    struct StatsAt(u16, Rc<RefCell<Value>>);

    impl Output for StatsAt {
        fn write(&mut self, _frame: Frame<'_>) -> Result<()> {
            use std::io::Read;
            if self.1.borrow().is_null() {
                let mut s = std::net::TcpStream::connect(("127.0.0.1", self.0))?;
                s.write_all(
                    b"GET /stats HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
                )?;
                let mut out = String::new();
                s.read_to_string(&mut out)?;
                let body = out.split_once("\r\n\r\n").map_or("", |(_, b)| b);
                *self.1.borrow_mut() = serde_json::from_str(body)?;
            }
            Ok(())
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn stats_say_how_far_the_console_turns_the_frame() {
        drop(gpu(8, 8)); // run() opens a renderer of its own on that device
        let (_d, console) = tmp_console("rotate");
        let port = free_port();
        let stats = Rc::new(RefCell::new(Value::Null));
        let ran = run(Options {
            store: Some(SceneStore::new(&console.store.dir, Vec::new())),
            port,
            size: (32, 24),
            rotate: 90,
            frames: 1,
            sinks: vec![Box::new(StatsAt(port, Rc::clone(&stats)))],
            ..Options::default()
        });
        assert!(ran.is_ok(), "{:?}", ran.err());
        assert_eq!(
            stats.borrow()["display"],
            serde_json::json!({"w": 32, "h": 24, "format": "rgba", "rotation": 90})
        );
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_switch_starts_i_frame_at_0_and_an_apply_keeps_it_running() {
        let renderer = RefCell::new(gpu(8, 8));
        let d = crate::testing::TempDir::new("console-clock");
        let count = r#"{"version": 1, "passes": [{"name": "main",
            "fragment": "void main(){ gl_FragColor = vec4(float(iFrame) / 255.0, 0.0, 0.0, 1.0); }"}]}"#;
        fs::write(d.join("count.scene.json"), count).unwrap();
        let console = Arc::new(Console::new(SceneStore::new(&d, Vec::new())));
        let st = RefCell::new(Running::default());
        let ctrl: ControllerSlot = RefCell::new(None);
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        // iFrame in levels of red
        let frame = || f64::from(renderer.borrow_mut().render(None).unwrap()[0]);
        let switch = || {
            api.host.switch(Some("count")).unwrap();
            drain(&console, &renderer, &st, &ctrl);
        };
        switch();
        let drawn: Vec<f64> = (0..3).map(|_| frame()).collect();
        assert_eq!(drawn, [0.0, 1.0, 2.0]);
        switch();
        assert_eq!(frame(), 0.0, "a reload starts the clock again");
        assert_eq!(ask(&api, "POST", "/scene", "name=count", count).0, 200);
        drain(&console, &renderer, &st, &ctrl);
        assert_eq!(frame(), 1.0, "an apply keeps the clock running");
    }

    /// A scene process that writes the $GHEE_SIZE that it starts with into
    /// `size`, beside it, and draws black.
    const SIZE_SCRIPT: &str = "#!/bin/sh\nprintf '%s' \"$GHEE_SIZE\" > size\n\
        printf '%s\\n' '{\"version\": 1, \"passes\": [{\"name\": \"main\", \
        \"fragment\": \"void main(){ gl_FragColor = vec4(0.0); }\"}]}'\n\
        while read -r l; do :; done\n";

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_scene_process_starts_with_the_size_that_the_console_renders_it_at() {
        let d = crate::testing::TempDir::new("console-size");
        crate::testing::script(&d, "sized", SIZE_SCRIPT);
        crate::testing::script(&d, "unloaded", SIZE_SCRIPT);
        // The console's own size stays at its default, so the process takes
        // the renderer's.
        let mut console = Console::new(SceneStore::new(&d, Vec::new()));
        console.frames = 1;
        let console = Arc::new(console);
        console.ask("sized");
        let mut renderer = gpu(32, 24);
        renderer.set_rotate(90).unwrap();
        let sinks: Vec<Box<dyn Output>> = vec![Box::new(Bound(Rc::new(Cell::new(0))))];
        render_loop(&console, Rc::new(RefCell::new(renderer)), sinks, None, None);
        assert_eq!(fs::read_to_string(d.join("size")).unwrap(), "24x32");
        // a scene that the console never loaded, read through GET /scene
        fs::remove_file(d.join("size")).unwrap();
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        assert_eq!(ask(&api, "GET", "/scene", "name=unloaded", "").0, 200);
        assert_eq!(fs::read_to_string(d.join("size")).unwrap(), "24x32");
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn stats_say_whether_the_gpu_samples_a_size_that_is_not_a_power_of_two_trilinearly() {
        let d = crate::testing::TempDir::new("console-gpu");
        fs::write(d.join("one.scene.json"), SCENE).unwrap();
        let mut console = Console::new(SceneStore::new(&d, Vec::new()));
        console.frames = 1;
        let console = Arc::new(console);
        console.ask("one");
        let renderer = gpu(8, 8);
        let npot = renderer.trilinear_npot();
        let sinks: Vec<Box<dyn Output>> = vec![Box::new(Bound(Rc::new(Cell::new(0))))];
        render_loop(&console, Rc::new(RefCell::new(renderer)), sinks, None, None);
        assert_eq!(
            console.stats()["gpu"],
            serde_json::json!({"trilinear_npot": npot})
        );
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_console_that_nobody_watches_draws_its_first_frame_alone() {
        for viewers in [0, 1] {
            let d = crate::testing::TempDir::new("console-watched");
            fs::write(d.join("one.scene.json"), SCENE).unwrap();
            let mut console = Console::new(SceneStore::new(&d, Vec::new()));
            (console.w, console.h) = (8, 8);
            let console = Arc::new(console);
            console.ask("one");
            console.count_clients(viewers);
            let summary: Summary = Rc::new(RefCell::new(None));
            let got = Rc::clone(&summary);
            let stopper = {
                let console = Arc::clone(&console);
                thread::spawn(move || {
                    // the first frame, then time for many more
                    let deadline = Instant::now() + Duration::from_secs(10);
                    while lock(&console.stat).frame_ms == 0.0 && Instant::now() < deadline {
                        thread::sleep(Duration::from_millis(5));
                    }
                    thread::sleep(Duration::from_millis(200));
                    console.stop();
                })
            };
            let renderer = Rc::new(RefCell::new(gpu(8, 8)));
            let hook: StatsHook = Box::new(move |times, wall, out| {
                *got.borrow_mut() = Some((times.len(), wall, out.to_vec()));
            });
            render_loop(&console, renderer, Vec::new(), None, Some(hook));
            stopper.join().unwrap();
            let (n, _, _) = summary.borrow().clone().expect("no summary");
            if viewers == 0 {
                assert_eq!(n, 1);
            } else {
                assert!(n > 1, "{n} frames with a viewer");
            }
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_console_with_a_frame_rate_cap_draws_no_faster_than_it() {
        drop(gpu(8, 8)); // run() opens a renderer of its own on that device
        let (_d, console) = tmp_console("capped");
        let summary: Summary = Rc::new(RefCell::new(None));
        let got = Rc::clone(&summary);
        let ran = run(Options {
            store: Some(SceneStore::new(&console.store.dir, Vec::new())),
            port: free_port(),
            size: (32, 24),
            frames: 5,
            sinks: vec![Box::new(Bound(Rc::new(Cell::new(0))))],
            pacer: Some(Box::new(crate::output::FixedPacer::new(20.0))),
            on_stats: Some(Box::new(move |times, wall, out| {
                *got.borrow_mut() = Some((times.len(), wall, out.to_vec()));
            })),
            ..Options::default()
        });
        assert!(ran.is_ok(), "{:?}", ran.err());
        let (n, wall, _) = summary.borrow().clone().expect("no summary");
        // five frames at 20 a second leave four periods of 50 ms between them
        assert_eq!(n, 5);
        assert!(wall >= 0.19, "{wall}");
    }

    fn free_port() -> u16 {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    }
}
