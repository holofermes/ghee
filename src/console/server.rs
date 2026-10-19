//! ghee console render server.
//!
//! A host is the program that runs the renderer: the `ghee` command, or a
//! program that embedded it. The host runs a render loop (looper), one
//! thread that draws every frame and hands it to the host's sinks: the
//! display, files, a pipe. This module attaches the console to that loop.
//!
//! The console serves the HTTP API (`docs/http.md`) and the web page on a
//! thread of its own, through the hub (console::hub). A request handler never
//! touches the renderer: it queues a command, a param set, a scene switch, an
//! edited scene, and the render thread applies the queue between frames.

use std::cell::{Cell, RefCell, RefMut};
use std::collections::{HashMap, VecDeque};
use std::ffi::c_int;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use serde_json::{Map, Value};
use tiny_http::Server;

use crate::console::core::{
    other_reload, other_scene, Aside, ConsoleAPI, ConsoleError, ConsoleHost, ConsoleResult, Edit,
    LayerSet, Scope,
};
use crate::console::hub::{
    banner, bound_port, frame_format, gpu_caps, serve, spawn_encoders, urls, Drawer, Hub, HwLane,
    RawFrame, SceneStats, Shot,
};
use crate::console::state::{self, Edits, Item};
use crate::console::store::{self, Carried, Opened, SceneStore};
use crate::console::{examples_dir, is_examples_dir};
use crate::control::Control;
use crate::draw::Call;
use crate::engine::{Card, Feed, Given, Provider, Renderer, Rotation, Taken, DEFAULT_SIZE};
use crate::feeds::demo_feed;
use crate::layer::Address;
use crate::live::Live;
use crate::looper::{ControllerSlot, Frame, Loop, Output, Pacer, RenderTarget, SceneController};
use crate::output::Scanout;
use crate::procedural::locks::Locks;
use crate::scene::{coerce, ArraySize, ParamValue, Reach, Scene};
use crate::sources::{collect, Snapshot};
use crate::stats::{monotonic, DEST_KMS};

/// How long GET /scene and GET /edits wait for the active scene's load. A
/// procedural scene's program takes up to 30 s to hand over its scene.
pub const LOAD_TIMEOUT: Duration = Duration::from_secs(35);

/// How long after a scene's last change its draft is written.
pub const DRAFT_AFTER: Duration = Duration::from_secs(1);

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

/// What the render thread drains off the command queue.
pub enum Command {
    /// A scene to load, with the number of the switch that asked for it.
    Switch(String, u64),
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
    items: Mutex<Queue>,
    ready: Condvar,
    /// The count of commands that the render loop applied and published.
    applied: Mutex<u64>,
    settled: Condvar,
}

/// The queued commands, and the count of commands put and taken since the
/// console started.
#[derive(Default)]
struct Queue {
    items: VecDeque<Command>,
    put: u64,
    taken: u64,
}

impl Commands {
    fn put(&self, item: Command) {
        let mut q = lock(&self.items);
        q.items.push_back(item);
        q.put += 1;
        self.ready.notify_all();
    }

    fn take(&self) -> Option<Command> {
        let mut q = lock(&self.items);
        let item = q.items.pop_front();
        q.taken += u64::from(item.is_some());
        item
    }

    fn is_empty(&self) -> bool {
        lock(&self.items).items.is_empty()
    }

    /// Sleep until a command arrives. The command stays on the queue.
    fn wait(&self) {
        let mut q = lock(&self.items);
        while q.items.is_empty() {
            q = self
                .ready
                .wait(q)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    fn sent(&self) -> u64 {
        lock(&self.items).put
    }

    /// The render loop applied each command that it took, and published
    /// the document that they made.
    fn settle(&self) {
        let taken = lock(&self.items).taken;
        *lock(&self.applied) = taken;
        self.settled.notify_all();
    }

    /// Wait until the render loop applied the first `count` commands, until
    /// `deadline`, or until `stopping` says the console stops. False when
    /// it gave up.
    fn wait_applied(&self, count: u64, deadline: Instant, stopping: impl Fn() -> bool) -> bool {
        let mut applied = lock(&self.applied);
        while *applied < count {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() || stopping() {
                return false;
            }
            applied = self
                .settled
                .wait_timeout(applied, left.min(Duration::from_millis(250)))
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        true
    }
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
            generation,
        }
    }

    /// Count a change.
    fn bump(&mut self) {
        self.rev += 1;
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
            draft: self.opened.draft.clone(),
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
            "draft": self.opened.draft.is_some(),
        })
    }

    /// The document that a page reads and a /set checks against: the
    /// working copy, or for a procedural scene the document that runs, which
    /// holds what its program drew and set, once it carries this working copy.
    fn live_doc(&self, name: &str, hub: &Hub) -> Value {
        let running = match self.opened.procedural {
            true => hub.scene_of(name, self.generation),
            false => None,
        };
        running.unwrap_or_else(|| self.working_doc())
    }
}

/// Take out of `values`, the values of a scene's edits, each param and
/// field of a layer that `locks` holds, and the removal of an effect that
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
                return false;
            };
            take(fields, |key, value| {
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

/// A source for one declared array: its size and the frame number give this
/// frame's data, as the engine's provider gives it.
pub type ArraySource = Box<dyn FnMut(ArraySize, u64) -> Feed>;

/// The item of POST /aside that names the saved edits that make no scene.
pub const BROKEN: &str = "edits";

/// This module's state. The render loop and the request threads share it.
pub struct Console {
    // configuration: run() sets it before the threads start
    pub store: SceneStore,
    pub web: bool,
    /// Scan out to a connected KMS display as well.
    pub display: bool,
    /// Scan out in 16-bit color. This halves the GPU store and the scanout bytes.
    pub rgb565: bool,
    /// The display skips to the newest frame, and the render keeps its own pace.
    pub drop: bool,
    /// The turn clockwise of the frame on its way to every output.
    pub rotate: Rotation,
    /// The backlight level that the display takes at the start, 0 to 1.
    pub brightness: Option<f64>,
    /// The frame components the renderer reads back: 3 for RGB, 4 for RGBA.
    pub components: i32,
    /// The readback keeps the output pass's alpha.
    pub alpha: bool,
    /// 0 renders until its control stops it. A count ends the console with
    /// the render.
    pub frames: u64,
    /// Pin engine time to frame * this, for a reproducible render.
    pub timestep: Option<f64>,
    /// The DRI device override. None probes for one.
    pub dri: Option<PathBuf>,
    /// The interval a stale check reads disk at. None takes the default window.
    pub stale_interval: Option<f64>,
    /// What stops and pauses the render.
    pub control: Control,
    /// The frame's size, the preview, the streams and /stats.
    pub hub: Hub,

    // state
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
    /// When each scene whose draft is due changed last, by name.
    drafts: Mutex<HashMap<String, Instant>>,
    /// What the process of the scene named beside it set, while it runs.
    locks: Mutex<Option<(String, Arc<Mutex<Locks>>)>>,
    /// The scene_size() of the renderer that render_loop draws with, from the
    /// start of the loop.
    scene_size: Mutex<Option<(i32, i32)>>,
    /// The files the running scene was built from. Only a load off disk replaces
    /// this. An applied working copy leaves the scene stale when its files moved
    /// on.
    sources: Mutex<Option<Snapshot>>,
    q: Commands,
    /// A render loop drains `q`.
    draining: AtomicBool,
}

impl Console {
    /// A console over a scene store, with the defaults run() overwrites.
    pub fn new(store: SceneStore) -> Console {
        Console {
            store,
            web: true,
            display: false,
            rgb565: false,
            drop: false,
            rotate: Rotation::Deg0,
            brightness: None,
            components: 4,
            alpha: false,
            frames: 0,
            timestep: None,
            dri: None,
            stale_interval: None,
            control: Control::new(),
            hub: Hub::new(),
            active: Mutex::new(Active::default()),
            editing: Mutex::new(()),
            loaded: Condvar::new(),
            copies: Mutex::new(HashMap::new()),
            revs: Mutex::new(HashMap::new()),
            drafts: Mutex::new(HashMap::new()),
            locks: Mutex::new(None),
            scene_size: Mutex::new(None),
            sources: Mutex::new(None),
            q: Commands::default(),
            draining: AtomicBool::new(false),
        }
    }

    pub fn stopping(&self) -> bool {
        self.hub.stopping()
    }

    /// The /stats payload.
    pub fn stats(&self) -> Value {
        // `rev` is the revision of the scene that `active` names.
        let active = lock(&self.active).name.clone();
        let stale = lock(&self.sources).as_mut().is_some_and(Snapshot::stale);
        let locked = active
            .as_deref()
            .and_then(|name| self.with_locks(name, |locks| locks.map(Locks::to_json)));
        let rev = active
            .as_ref()
            .and_then(|name| lock(&self.revs).get(name).copied());
        self.hub.stats(SceneStats {
            active,
            stale,
            locked: locked.unwrap_or_else(|| Locks::default().to_json()),
            rev,
        })
    }
    /// Hold `opened` as the working copy of scene `name`, with the next
    /// revision and generation, and return that generation.
    fn install(&self, name: &str, opened: Opened) -> u64 {
        let mut copies = lock(&self.copies);
        let mut copy = WorkingCopy::new(opened, 0, 0);
        if let Some(old) = copies.get(name) {
            copy.rev = old.rev;
            copy.generation = old.generation;
        }
        self.bump(name, &mut copy);
        copy.generation += 1;
        let generation = copy.generation;
        copies.insert(name.to_string(), copy);
        generation
    }

    /// Count a change to `copy`, the working copy of scene `name`.
    fn bump(&self, name: &str, copy: &mut WorkingCopy) {
        copy.bump();
        lock(&self.revs).insert(name.to_string(), copy.rev);
        lock(&self.drafts).insert(name.to_string(), Instant::now());
    }

    /// Bring the draft of each scene that changed DRAFT_AFTER ago or more, or
    /// with `all` of each scene that changed, in line with its working copy:
    /// the working copy's edits while they differ from the state file, else
    /// no draft. A scene whose draft waits for the console keeps it.
    fn write_drafts(&self, all: bool) {
        let due: Vec<String> = {
            let mut drafts = lock(&self.drafts);
            let now = Instant::now();
            let due: Vec<String> = drafts
                .iter()
                .filter(|(_, at)| all || now.duration_since(**at) >= DRAFT_AFTER)
                .map(|(name, _)| name.clone())
                .collect();
            for name in &due {
                drafts.remove(name);
            }
            due
        };
        for name in due {
            let copies = lock(&self.copies);
            let Some(copy) = copies.get(&name) else {
                continue;
            };
            if copy.opened.draft.is_some() {
                continue;
            }
            let (unsaved, edits) =
                self.with_locks(&name, |locks| (copy.unsaved(locks), copy.to_save(locks)));
            let wrote = match unsaved {
                true => self.store.write_draft(&name, &edits),
                false => self.store.remove_draft(&name),
            };
            if let Err(e) = wrote {
                crate::report::error(format!("{name}: its unsaved edits: {e}"));
            }
        }
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
        let size = lock(&self.scene_size).unwrap_or(match self.rotate.sideways() {
            true => (self.hub.h, self.hub.w),
            false => (self.hub.w, self.hub.h),
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
        self.hub.shot.touch();
    }

    /// Shut the server down, and signal the render loop and the encode threads.
    /// run() calls it on the way out, and Control::stop() through it. It waits
    /// for no thread, because a request handler calls it. It ends run()'s own
    /// loop, so run() closes its sinks before it returns.
    pub fn stop(&self) {
        self.hub.stop();
        self.q.put(Command::Stop); // wake a render loop parked on the command queue
    }
}

impl Drawer for Console {
    fn hub(&self) -> &Hub {
        &self.hub
    }

    /// Wake a scene that rests on the command queue, so it draws a frame.
    fn wake(&self) {
        self.q.put(Command::Preview);
    }

    /// The still rides the command queue, which forces a draw.
    fn shot(&self) {
        self.q.put(Command::Shot);
    }
}

/// Write each draft that is due, until the console stops.
pub fn draft_loop(console: &Console) {
    while !console.stopping() {
        thread::sleep(Duration::from_millis(250));
        console.write_drafts(false);
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

/// The shot as an output sink.
struct FrameShot {
    shot: Arc<Shot>,
}

impl Output for FrameShot {
    fn dest(&self) -> &str {
        "" // a still is not a destination, see Hub::out
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
/// encoder share one throttled, downscaled readback, and the hardware H264 lane
/// runs here too. This reads the console's viewer counts every frame, so what it
/// asks the loop for follows the viewers.
struct ConsoleSink {
    console: Arc<Console>,
    renderer: Rc<RefCell<Renderer>>,
    lane: HwLane,
}

impl Output for ConsoleSink {
    fn dest(&self) -> &str {
        "" // the console's destinations are its viewers, see Hub::out
    }

    fn persists(&self) -> bool {
        true // a browser holds the last frame it got
    }

    fn wants(&self, _frame: u64) -> bool {
        self.console.hub.wants_preview()
    }

    fn readback(&self) -> Option<(i32, i32)> {
        Some(self.console.hub.preview_size())
    }

    fn write(&mut self, frame: Frame<'_>) -> Result<()> {
        self.console.hub.publish_frame(frame);
        Ok(())
    }

    fn mirror(&mut self, _renderer: &dyn RenderTarget) -> Result<()> {
        self.lane.mirror(&self.console.hub, &self.renderer.borrow());
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.lane.close();
        Ok(())
    }
}

// ---------------------------------------------------------------- render loop

/// The renderer the loop drives. The console sink samples it for the hardware
/// encoder. Both live on the render thread, because GL is thread-affine.
struct SharedRenderer {
    renderer: Rc<RefCell<Renderer>>,
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
        self.renderer.borrow().set_layer(at, given)
    }

    fn add_layer(&mut self, pass: &str, index: usize, layer: &Value) -> Result<()> {
        self.renderer.borrow_mut().add_layer(pass, index, layer)
    }

    fn update_layer(&mut self, layer: &Value) -> Result<bool> {
        self.renderer.borrow_mut().update_layer(layer)
    }

    fn layer(&self, name: &str) -> Option<Value> {
        RenderTarget::layer(&*self.renderer.borrow(), name)
    }

    fn pass_layers(&self, pass: &str) -> Option<Vec<String>> {
        RenderTarget::pass_layers(&*self.renderer.borrow(), pass)
    }

    fn update_calls(&mut self, calls: &[Call]) -> Result<Taken> {
        self.renderer.borrow_mut().update_calls(calls)
    }

    fn remove_layer(&mut self, name: &str) -> Result<()> {
        self.renderer.borrow_mut().remove_layer(name)
    }

    fn move_layer(&mut self, name: &str, index: usize) -> Result<()> {
        self.renderer.borrow_mut().move_layer(name, index)
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

/// Create the render-thread Renderer: on `card`, the display's, else on the
/// GL device that `console.dri` names, else on the first one that opens.
fn make_renderer(console: &Console, card: Option<Card>) -> Result<Renderer> {
    let (w, h) = (console.hub.w, console.hub.h);
    let mut r = match card {
        Some(card) => Renderer::builder(w as u32, h as u32)
            .display(card)
            .rgb565(console.rgb565)
            .build()?,
        None => {
            let b = Renderer::builder(w as u32, h as u32).components(console.components as u32);
            match &console.dri {
                Some(dri) => b.device(dri),
                None => b,
            }
            .build()?
        }
    };
    if console.drop {
        r.set_drop(true)?;
    }
    r.set_alpha(console.alpha);
    r.set_rotate(console.rotate)?;
    if let Some(level) = console.brightness {
        if let Err(e) = r.set_brightness(level) {
            crate::report::error(format!("{e:#}"));
        }
    }
    Ok(r)
}

/// What the render thread holds between commands. The renderer's scene is
/// the running document.
#[derive(Default)]
struct Running {
    /// The name of the scene that the renderer draws, the last one that
    /// compiled, and the generation of its working copy.
    name: Option<String>,
    generation: u64,
    /// The name of the scene that the last load asked for. A /set is for
    /// this scene, so it lands only while `scene` is this scene.
    loaded: Option<String>,
    /// An apply kept layers that the scene's own code added, since the last
    /// publish.
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
    // the last good scene, which a failed setup sets up again
    let last = match st.borrow().name {
        Some(_) => renderer.borrow().with_scene(Scene::clone),
        None => None,
    };
    let built = match keep {
        true => renderer.borrow_mut().rebuild(&scene),
        false => renderer.borrow_mut().setup(&scene),
    };
    let (err, line, pass) = match built {
        Ok(()) => (String::new(), None, None),
        Err(e) => (e.to_string(), e.line(), e.pass()),
    };
    let name = st.borrow().loaded.clone();
    // Store the snapshot before the publish. A /stats between the two must not
    // read the new scene against the snapshot before it.
    if let Some(snap) = snap {
        *lock(&console.sources) = Some(snap);
    }
    if !err.is_empty() {
        console.hub.set_error(&err, line, pass);
        if let Some(last) = last {
            let _ = renderer.borrow_mut().rebuild(&last);
        }
        console
            .hub
            .publish_scene(name.as_deref(), generation, scene.to_dict());
        trim_heap();
        return err;
    }
    console.hub.reset();
    console
        .hub
        .publish_scene(name.as_deref(), generation, scene.to_dict());
    let mut held = st.borrow_mut();
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
            let generation = console.install(name, opened);
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
                if let Some(f) = process.as_ref() {
                    *lock(&console.locks) = Some((name.to_string(), f.locks()));
                }
                *ctrl.borrow_mut() = process.map(|f| Box::new(f) as Box<dyn SceneController>);
            }
            None
        }
        Err(e) => {
            let why = format!("load {name}: {e}");
            console.hub.set_error(&why, None, None);
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
    console.hub.clear_scene();
    *st.borrow_mut() = Running::default();
    let black = serde_json::json!({"version": 1, "passes": [{"name": "main",
        "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }"}]});
    let built =
        Scene::from_dict(&black, None, None).and_then(|s| Ok(renderer.borrow_mut().setup(&s)?));
    if let Err(e) = built {
        crate::report::error(format!("no scene: {e}"));
    }
    console.hub.reset();
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
            Command::Switch(name, switch) => {
                load(console, renderer, st, ctrl, &name, switch);
            }
            Command::Param(..) | Command::SceneParam(..) | Command::Layer(_) if !drawn => {}
            Command::Param(pi, name, value) => {
                if set_pass_param(renderer, ctrl, pi, &name, &value) {
                    edited = true;
                }
            }
            Command::SceneParam(name, value) => {
                if set_scene_param(renderer, ctrl, &name, &value) {
                    edited = true;
                }
            }
            Command::Layer(set) => {
                if set_layer(renderer, ctrl, &set) {
                    edited = true;
                }
            }
            // in queue order, so the armed frame carries a /set ahead of it
            Command::Shot => console.hub.shot.arm(),
            Command::SceneObj(scene, generation, fresh) => {
                let mut scene = *scene;
                // A procedural scene's code owns which layers exist, so an
                // apply keeps a layer that the code added after the page
                // read the scene.
                let kept = ctrl.borrow().is_some()
                    && renderer
                        .borrow()
                        .with_scene(|running| keep_layers(&mut scene, running))
                        .unwrap_or(false);
                if let Some(live) = ctrl.borrow_mut().as_mut().and_then(|c| c.live()) {
                    if let Err(e) = live.console_scene(&*renderer.borrow(), &mut scene) {
                        crate::report::error(format!("apply: {e}"));
                    }
                }
                let built =
                    set_scene(console, renderer, st, scene, generation, None, !fresh).is_empty();
                // a failed setup sets the last scene up again
                if let Some(c) = ctrl.borrow_mut().as_mut() {
                    c.rebuilt();
                }
                if built && kept {
                    st.borrow_mut().changed = true;
                    publish(console, renderer, st, ctrl, false);
                }
            }
            Command::Clear => clear(console, renderer, st, ctrl),
            Command::Preview | Command::Stop => {}
        }
    }
    if edited {
        publish(console, renderer, st, ctrl, true);
    }
    if force {
        console.q.settle();
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
                crate::report::error(format!("apply: layer {name}: {e}"));
                continue;
            }
            kept = true;
        }
    }
    kept
}

/// The live state of the scene that the controller in `ctrl` runs.
fn live_of(ctrl: &ControllerSlot) -> Option<RefMut<'_, Live>> {
    RefMut::filter_map(ctrl.borrow_mut(), |c| c.as_mut().and_then(|c| c.live())).ok()
}

/// Publish the running document, the renderer's scene: whole with `whole`,
/// else what the scene's own code changed in it. A change by that code, or
/// an apply that kept layers that the code added, counts for /stats and
/// gives the scene a new revision.
fn publish(
    console: &Console,
    renderer: &RefCell<Renderer>,
    st: &RefCell<Running>,
    ctrl: &ControllerSlot,
    whole: bool,
) {
    let kept = std::mem::take(&mut st.borrow_mut().changed);
    let held = st.borrow();
    let mut live = live_of(ctrl);
    // A reader takes the revision and the document under this lock, so one
    // revision names one document.
    let mut copies = lock(&console.copies);
    let counted = console.hub.publish_live(
        &renderer.borrow(),
        live.as_deref_mut(),
        whole || kept,
        kept,
        held.name.as_deref(),
        held.generation,
    );
    let copy = held
        .name
        .as_deref()
        .and_then(|n| Some((n, copies.get_mut(n)?)));
    if let (true, Some((name, copy))) = (counted, copy) {
        copy.bump();
        lock(&console.revs).insert(name.to_string(), copy.rev);
    }
}

/// True when the scene's own code changed the running document since the
/// last publish, or an apply kept layers that the code added.
fn changed(st: &RefCell<Running>, ctrl: &ControllerSlot) -> bool {
    st.borrow().changed || live_of(ctrl).is_some_and(|live| live.changed())
}

/// Set the param `name` of pass `pass`, None for the scene's, as a console
/// sets it: through the live state of the scene that a controller runs.
fn console_param(
    renderer: &RefCell<Renderer>,
    ctrl: &ControllerSlot,
    pass: Option<&str>,
    name: &str,
    value: &Value,
) -> Result<()> {
    let ty = renderer.borrow().with_scene(|s| {
        let params = match pass {
            Some(pass) => &s.passes.iter().find(|p| p.name == pass)?.params,
            None => &s.params,
        };
        Some(params.get(name)?.ty.clone())
    });
    let Some(ty) = ty.flatten() else {
        match pass {
            Some(pass) => bail!("pass '{pass}' has no param '{name}'"),
            None => bail!("the scene has no param '{name}'"),
        }
    };
    let v = coerce(name, &ty, value)?;
    let r = renderer.borrow();
    match live_of(ctrl) {
        Some(mut live) => live.console_param(&*r, pass, name, &ty, &v),
        None => RenderTarget::set_param(&*r, pass, name, &ty, &v).map(drop),
    }
}

fn set_pass_param(
    renderer: &RefCell<Renderer>,
    ctrl: &ControllerSlot,
    pi: i64,
    name: &str,
    value: &Value,
) -> bool {
    let pass = renderer.borrow().with_scene(|s| {
        s.passes
            .get(usize::try_from(pi).unwrap_or(usize::MAX))
            .map(|p| p.name.clone())
    });
    let applied = match pass.flatten() {
        Some(pass) => console_param(renderer, ctrl, Some(&pass), name, value),
        None => Err(anyhow!("the scene has no pass {pi}")),
    };
    match applied {
        Ok(()) => true,
        Err(e) => {
            crate::report::error(format!("set param {name}: {e}"));
            false
        }
    }
}

fn set_scene_param(
    renderer: &RefCell<Renderer>,
    ctrl: &ControllerSlot,
    name: &str,
    value: &Value,
) -> bool {
    match console_param(renderer, ctrl, None, name, value) {
        Ok(()) => true,
        Err(e) => {
            crate::report::error(format!("set scene param {name}: {e}"));
            false
        }
    }
}

/// Set a key of a layer as a console sets it: through the live state of the
/// scene that a controller runs.
fn set_layer(renderer: &RefCell<Renderer>, ctrl: &ControllerSlot, set: &LayerSet) -> bool {
    let at = set.at();
    let mut r = renderer.borrow_mut();
    let applied = match live_of(ctrl) {
        Some(mut live) => live.console_layer(&mut *r, &at, &set.value),
        None => crate::live::put_key(&mut *r, &at, Given::Json(&set.value)).map(|_| ()),
    };
    match applied {
        Ok(()) => true,
        Err(e) => {
            crate::report::error(format!("set layer {}: {e}", set.layer));
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
        console.hub.set_text_errors(renderer.text_errors());
    }
}

/// Run the console's render loop on this thread. The console adds the command
/// queue, the published scene and the stale snapshot. A frame error must not
/// kill the thread.
///
/// GL is thread-affine, so the caller creates the Renderer on the thread that
/// calls this. The sinks belong to the caller, and this draws to them from
/// here. `arrays` feed the arrays of the scenes by name.
pub fn render_loop(
    console: &Arc<Console>,
    renderer: Rc<RefCell<Renderer>>,
    mut sinks: Vec<Box<dyn Output>>,
    mut arrays: Vec<(String, ArraySource)>,
    pacer: Option<Box<dyn Pacer>>,
    on_stats: Option<StatsHook>,
) {
    let hz = if console.display {
        renderer.borrow().display_vrefresh()
    } else {
        0
    };
    *lock(&console.scene_size) = Some(renderer.borrow().scene_size());
    console.hub.set_gpu(gpu_caps(&renderer.borrow()));
    console.draining.store(true, Ordering::SeqCst);
    let st = Rc::new(RefCell::new(Running::default()));
    // what feeds the running scene, when the scene brings something
    let ctrl: ControllerSlot = RefCell::new(None);
    let mut shared = SharedRenderer {
        renderer: Rc::clone(&renderer),
    };
    let mut scanout = Scanout;
    let mut console_sink = ConsoleSink {
        console: Arc::clone(console),
        renderer: Rc::clone(&renderer),
        lane: HwLane::default(),
    };
    let mut shot_sink = FrameShot {
        shot: Arc::clone(&console.hub.shot),
    };
    let mut reported = String::new();
    let times: RefCell<Vec<f64>> = RefCell::new(Vec::new());
    let text_errors = Cell::new(renderer.borrow().text_errors_seq());
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
        render.clock.timestep = console.timestep;
        render.clock.pacer = pacer;
        // the loop rests so the handler and encoder threads run
        render.rest = 0.003;
        let feed = provider(&mut arrays);
        render.provider = Some(Box::new(feed));
        render.controller = Some(&ctrl);
        render.poll = Some(Box::new(|| drain(console, &renderer, &st, &ctrl)));
        render.idle_hook = Some(Box::new(|period| {
            // a change that waited for publish_every goes out before the
            // loop rests, so the document never lags behind the last frame
            if changed(&st, &ctrl) {
                publish(console, &renderer, &st, &ctrl, true);
            }
            let hub = &console.hub;
            hub.skipped();
            let interval = skip_interval(
                hz,
                hub.nclients() + hub.fanout.nclients() as i64,
                hub.preview_fps,
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
            drives || console.hub.nclients() > 0 || console.hub.fanout.nclients() > 0
        }));
        render.stopping = Some(Box::new(|| console.stopping()));
        render.held = Some(Box::new(|| console.control.paused()));
        render.on_frame = Some(Box::new(|_loop, cost| {
            console.hub.frame_done(cost);
            if changed(&st, &ctrl) {
                publish(console, &renderer, &st, &ctrl, false);
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
            let why = format!("render: {}", crate::procedural::process::with_stderr(e));
            console.hub.set_error(&why, None, None);
            if msg != reported {
                crate::report::error(&msg);
                reported = msg;
            }
            true
        }));
        let first = {
            let active = lock(&console.active);
            active.name.clone().map(|name| (name, active.asked))
        };
        match first {
            Some((name, switch)) => load(console, &renderer, &st, &ctrl, &name, switch),
            None => clear(console, &renderer, &st, &ctrl),
        }
        if let Err(e) = render.run() {
            crate::report::error(format!("render: {e}"));
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
    console.draining.store(false, Ordering::SeqCst);
    console.stop();
}

/// The host's array source for a name, else the placeholder feed.
fn provider(arrays: &mut [(String, ArraySource)]) -> impl FnMut(&str, ArraySize, u64) -> Feed + '_ {
    |name: &str, size: ArraySize, frame: u64| match arrays.iter_mut().find(|(n, _)| n == name) {
        Some((_, src)) => src(size, frame),
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
    fn rename_scene(&self, name: &str, to: &str) -> ConsoleResult<()> {
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
            console.command(Command::Switch(to.to_string(), switch));
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
                console.command(Command::Switch(next, switch));
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

    /// The read waits until the render loop applied every edit that
    /// answered before it, and holds off the edits that come meanwhile, so
    /// the document is the one that its revision names.
    fn scene_doc(&self, name: Option<&str>, defaults: bool) -> ConsoleResult<(Value, Option<u64>)> {
        let console = &self.console;
        let hub = &console.hub;
        let _editing = lock(&console.editing);
        let sent = console.q.sent();
        let deadline = Instant::now() + LOAD_TIMEOUT;
        if console.draining.load(Ordering::SeqCst)
            && !console
                .q
                .wait_applied(sent, deadline, || console.stopping())
        {
            return Err(ConsoleError::unavailable("the scene is still taking edits"));
        }
        console.with_copy(name, |name, copy| {
            let doc = match defaults {
                true => copy.defaults_doc.clone(),
                false => copy.live_doc(name, hub),
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
                    .put_key(&set.at(), &set.value)
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
        let hub = &self.console.hub;
        self.console
            .with_active(|name, copy| Ok(copy.live_doc(name, hub)))
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
        self.console
            .command(Command::Switch(name.to_string(), switch));
        Ok(name.to_string())
    }

    /// Every edit gives the active scene a new revision.
    fn edit_scene(
        &self,
        at: &Edit,
        edit: &mut dyn FnMut() -> ConsoleResult<()>,
    ) -> ConsoleResult<Option<u64>> {
        let _editing = lock(&self.console.editing);
        if let Some(scene) = at.scene {
            other_scene(scene, self.active_name()?.as_deref())?;
        }
        let name = lock(&self.console.active).editable()?;
        let stale = lock(&self.console.copies)
            .get(&name)
            .is_some_and(|copy| !at.matches(copy.rev));
        if stale {
            return Err(ConsoleError::stale(
                "this scene changed since you loaded it",
            ));
        }
        edit()?;
        Ok(lock(&self.console.copies).get_mut(&name).map(|copy| {
            self.console.bump(&name, copy);
            copy.rev
        }))
    }

    fn apply_scene(&self, mut scene: Scene) -> ConsoleResult<()> {
        let hub = &self.console.hub;
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
                let mut changed = state::diff(&copy.live_doc(name, hub), &scene.to_dict(), true);
                self.console.with_locks(name, |locks| {
                    if let Some(locks) = locks {
                        leave_out(&mut changed.values, locks);
                    }
                });
                let doc = state::fold(&copy.defaults_doc, &copy.working_doc(), &changed);
                match store::scene_of(&copy.opened.defaults, &doc, name) {
                    Ok(working) => copy.opened.working = working,
                    Err(e) => crate::report::error(format!(
                        "{name}: the working copy keeps no edit of this apply: {e}"
                    )),
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

    /// A scene of the console's own saves its edits into its files, and its
    /// defaults are then the scene that its files hold.
    fn save(&self, _name: Option<&str>) -> ConsoleResult<()> {
        let console = &self.console;
        console.with_active(|name, copy| {
            if console.store.is_mine(name) {
                console
                    .store
                    .write(name, &copy.defaults_doc, &copy.edits(None))?;
                let size = lock(&console.scene_size).unwrap_or((console.hub.w, console.hub.h));
                let (resolved, read) = collect(|| console.store.resolve(name, size));
                copy.opened.defaults = resolved?;
                copy.defaults_doc = copy.opened.defaults.to_dict();
                copy.opened.saved = None;
                *lock(&console.sources) = Some(console.snapshot(read));
                return Ok(());
            }
            let edits = console.with_locks(name, |locks| copy.to_save(locks));
            console.store.save(name, &edits)?;
            copy.opened.saved = (!edits.is_empty()).then_some(edits);
            if copy.opened.draft.is_none() {
                console
                    .store
                    .remove_draft(name)
                    .map_err(ConsoleError::server_error)?;
            }
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

    /// Restore lays the draft's edits over the defaults, as a load lays a
    /// state file's, and its items whose default changed are set aside.
    /// Drop removes the draft.
    fn draft(&self, _name: Option<&str>, restore: bool) -> ConsoleResult<()> {
        let store = &self.console.store;
        self.console.with_active(|name, copy| {
            let Some(draft) = copy.opened.draft.clone() else {
                return Err(ConsoleError::not_found("no unsaved edits wait"));
            };
            if !restore {
                store
                    .remove_draft(name)
                    .map_err(ConsoleError::server_error)?;
                copy.opened.draft = None;
                return Ok(());
            }
            let (working, aside, broken) = store::working(name, &copy.opened.defaults, &draft);
            copy.opened.working = working;
            copy.opened.aside = aside;
            copy.opened.broken = broken;
            copy.opened.draft = None;
            copy.generation += 1;
            let scene = Box::new(copy.opened.working.clone());
            self.console
                .command(Command::SceneObj(scene, copy.generation, false));
            Ok(())
        })
    }
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
    pub scenes_dir: Option<PathBuf>,
    /// The directory of the scenes that the console makes, listed after
    /// those of `scenes_dir`. None makes none.
    pub my_scenes: Option<PathBuf>,
    /// Take uploads while the console listens on a network.
    pub uploads: bool,
    /// Where the scenes' drafts live. None takes the store's own, else
    /// state::default_dir().
    pub state_dir: Option<PathBuf>,
    /// Where the scenes' saved edits live. None takes the store's own, else
    /// state::default_edits_dir().
    pub edits_dir: Option<PathBuf>,
    /// The render size. None takes the mode of `display`, else
    /// engine::DEFAULT_SIZE.
    pub size: Option<(u32, u32)>,
    pub port: u16,
    /// Loopback by default. The console has no auth, and it compiles POSTed
    /// GLSL. Pass "0.0.0.0" only on a network you trust.
    pub host: String,
    /// The host names that a console on loopback answers beside loopback's
    /// own, for a proxy that passes its own name on.
    pub allowed_hosts: Vec<String>,
    /// The KMS display that the frames scan out to, from Card::open().
    pub display: Option<Card>,
    pub rgb565: bool,
    /// The display skips to the newest frame, and the render keeps its own
    /// pace: Renderer::set_drop.
    pub drop: bool,
    /// The turn clockwise of the frame on its way to every output. `size`
    /// stays the output's.
    pub rotate: Rotation,
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
    /// one sets them here. A filter that the scene lists too keeps the
    /// scene's place and takes the values set here.
    pub filters: Vec<crate::filters::Filter>,
    /// The output sinks the frames also reach, so a console writes files.
    /// `display` covers the KMS scanout instead, because that one is a mode
    /// the renderer is created in. run() writes these from its own thread, so
    /// a sink bound to the thread that built it belongs here.
    pub sinks: Vec<Box<dyn Output>>,
    /// The most frames a second that the console draws. None and 0 draw at
    /// the display's refresh, or as fast as the sinks and the viewers take
    /// frames with no display.
    pub fps: Option<f64>,
    /// The bytes of a pixel in a frame: 3 for RGB, 4 for RGBA. A display
    /// reads RGBA back.
    pub components: u32,
    /// Keep the output pass's alpha in 4-component frames.
    pub alpha: bool,
    /// 0 serves until `control` stops it. A count ends the console with the
    /// render.
    pub frames: u64,
    pub timestep: Option<f64>,
    /// The GL device of a render with no display. None takes the first one
    /// that opens.
    pub dri_device: Option<PathBuf>,
    /// The summary hook. This hook makes the loop keep a cost per frame, and a
    /// run without it keeps none.
    pub on_stats: Option<StatsHook>,
    /// What stops and pauses the console. Keep a clone to stop it from
    /// another thread.
    pub control: Control,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            arrays: Vec::new(),
            scenes_dir: None,
            my_scenes: None,
            uploads: false,
            state_dir: None,
            edits_dir: None,
            size: None,
            port: 8090,
            host: "127.0.0.1".to_string(),
            allowed_hosts: Vec::new(),
            display: None,
            rgb565: false,
            drop: false,
            rotate: Rotation::Deg0,
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
            fps: None,
            components: 4,
            alpha: false,
            frames: 0,
            timestep: None,
            dri_device: None,
            on_stats: None,
            control: Control::new(),
        }
    }
}

/// The store over `scenes_dir`, or over the bundled examples when it is None,
/// and over `mine`, the console's own scenes. The bundled examples keep their
/// state files as demos, whichever way a caller names their directory.
fn dir_store(
    scenes_dir: Option<PathBuf>,
    mine: Option<PathBuf>,
    args: Vec<String>,
) -> Result<SceneStore> {
    let demo = scenes_dir.is_none();
    let mine = mine.map(utf8).transpose()?;
    let dir = match scenes_dir {
        Some(dir) => utf8(dir)?,
        None => examples_dir().ok_or_else(|| {
            anyhow!("no scenes directory (set GHEE_EXAMPLES for the bundled scenes)")
        })?,
    };
    let mut store = SceneStore::with_mine(&dir, mine.as_deref(), args);
    store.demo = demo || is_examples_dir(std::path::Path::new(&dir));
    Ok(store)
}

/// `path` as a String. The store names its folders in UTF-8.
fn utf8(path: PathBuf) -> Result<String> {
    path.into_os_string()
        .into_string()
        .map_err(|p| anyhow!("{}: not a UTF-8 path", p.to_string_lossy()))
}

/// Start the console and serve until `control` stops it, or until the
/// render ends.
pub fn run(opts: Options) -> Result<()> {
    let Options {
        arrays,
        scenes_dir,
        my_scenes,
        uploads,
        state_dir,
        edits_dir,
        size,
        port,
        host,
        allowed_hosts,
        display,
        rgb565,
        drop,
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
        fps,
        components,
        alpha,
        frames,
        timestep,
        dri_device,
        on_stats,
        filters,
        control,
    } = opts;
    let _released = crate::control::Released(control.clone());
    crate::filters::check_filters(&filters, &crate::filters::load_filters(&[]))?;
    let mut store = match store {
        Some(store) => store,
        None => dir_store(scenes_dir, my_scenes, scene_args)?,
    };
    let names = store.names();
    store.filters = filters;
    let state_dir = state_dir
        .or_else(|| store.state_dir.take())
        .or_else(state::default_dir);
    let edits_dir = edits_dir
        .or_else(|| store.edits_dir.take())
        .or_else(state::default_edits_dir);
    store.keep_in(state_dir, edits_dir);
    let size = size
        .or_else(|| display.as_ref().map(Card::size))
        .map_or(DEFAULT_SIZE, |(w, h)| (w as i32, h as i32));
    let pacer = crate::output::pacer(fps, crate::output::Sinks::Display);
    let mut console = Console::new(store);
    console.display = display.is_some();
    console.rgb565 = rgb565;
    console.drop = drop;
    console.rotate = rotate;
    console.brightness = brightness;
    console.components = components as i32;
    console.alpha = alpha;
    console.frames = frames;
    console.timestep = timestep;
    console.dri = dri_device;
    console.control = control;
    let hub = &mut console.hub;
    (hub.w, hub.h) = size;
    hub.preview_max = preview_max;
    hub.preview_fps = preview_fps;
    hub.stream_max = stream_max;
    hub.stream_hz = stream_fps;
    hub.bitrate = bitrate;
    if console.display {
        hub.dests.push(DEST_KMS.to_string());
    }
    hub.dests.extend(
        sinks
            .iter()
            .map(|s| s.dest().to_string())
            .filter(|d| !d.is_empty()),
    );
    let format = frame_format(display.is_some() && rgb565, components as i32);
    hub.set_display(format, i64::from(rotate.degrees()));
    if let Some(first) = names.first() {
        console.ask(first);
    }
    let console = Arc::new(console);
    console.hub.reset();
    {
        let weak = Arc::downgrade(&console);
        console.control.attach(Box::new(move || {
            if let Some(console) = weak.upgrade() {
                match console.control.stopped() {
                    true => console.stop(),
                    false => console.wake(),
                }
            }
        }));
    }
    if console.control.stopped() {
        console.stop();
    }
    let server = Server::http((host.as_str(), port)).map_err(|e| anyhow!("{e}"))?;
    let address = urls(&host, bound_port(&server, port));
    spawn_encoders(&console);
    {
        let console = Arc::clone(&console);
        thread::spawn(move || draft_loop(&console));
    }
    let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)))
        .with_web(console.web)
        .listening_on(&host)
        .answering(allowed_hosts)
        .with_uploads(uploads);
    let serving = serve(Arc::new(server), Arc::clone(&console), Arc::new(api));

    match make_renderer(&console, display) {
        Ok(r) => {
            let mut from = format!("scenes={}", console.store.dir);
            if let Some(m) = console.store.mine.as_deref() {
                if m != console.store.dir {
                    from.push_str(&format!(", my scenes={m}"));
                }
            }
            eprintln!("{}", banner(&address, &names, &from, size, format));
            let r = Rc::new(RefCell::new(r));
            render_loop(&console, r, sinks, arrays, pacer, on_stats);
        }
        Err(e) => {
            console.hub.set_error(&e.to_string(), None, None);
            console.hub.set_fatal(&e.to_string());
        }
    }
    // The loop is done, so nothing is left to serve. stop() unblocks the accept
    // thread, and the handlers it spawned end on the same flag.
    console.stop();
    let _ = serving.join();
    console.write_drafts(true);
    console.control.detach();
    let fatal = console.hub.fatal();
    if !fatal.is_empty() {
        bail!("{fatal}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::console::core::{parse_query, Headers};
    use crate::scene::Param;
    use crate::testing::{until, Clock};

    const SCENE: &str = r#"{"version": 1,
        "passes": [{"name": "main", "fragment": "void main(){ gl_FragColor = vec4(1.0); }"}]}"#;

    /// Make `scene` the working copy of the active scene `name`, as a load
    /// ends.
    fn opened(console: &Console, name: &str, scene: Scene) {
        let opened = Opened {
            defaults: scene.clone(),
            working: scene,
            aside: Vec::new(),
            broken: Vec::new(),
            saved: None,
            draft: None,
            procedural: false,
        };
        ready(console, name, opened);
    }

    /// Make `opened` the working copy of the active scene `name`, as a load
    /// ends.
    fn ready(console: &Console, name: &str, opened: Opened) {
        let switch = console.ask(name);
        console.install(name, opened);
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
        store.edits_dir = store.state_dir.clone();
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
        ask_if(api, method, path, q, body, None)
    }

    /// ask(), with `if_match` as the request's If-Match header.
    fn ask_if(
        api: &ConsoleAPI<RenderLoopHost>,
        method: &str,
        path: &str,
        q: &str,
        body: &str,
        if_match: Option<&str>,
    ) -> (u16, String, Option<String>) {
        let mut headers =
            Headers::from([("Content-Type", "application/json"), ("Host", "localhost")]);
        if let Some(tags) = if_match {
            headers.insert("If-Match", tags);
        }
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
            "scene=clocks&name=main.rows&value=4",
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
    fn a_scene_posted_back_takes_its_assets_from_its_folder_with_or_without_filters() {
        for filters in [serde_json::json!([]), serde_json::json!(["frc"])] {
            let d = crate::testing::TempDir::new("console-asset-folder");
            fs::create_dir_all(d.join("word")).unwrap();
            fs::write(d.join("word/pic.png"), crate::testing::png(2, 2)).unwrap();
            let mut doc: Value = serde_json::from_str(SCENE).unwrap();
            doc["assets"] = serde_json::json!({"pic": {"file": "pic.png"}});
            doc["filters"] = filters.clone();
            fs::write(d.join("word/word.scene.json"), doc.to_string()).unwrap();
            let console = Arc::new(Console::new(SceneStore::new(&d, Vec::new())));
            let (opened, _) = console.store.open("word", None, (32, 24)).unwrap();
            ready(&console, "word", opened);
            let api = ConsoleAPI::new(RenderLoopHost::new(console));
            let (_, scene, _) = ask(&api, "GET", "/scene", "", "");
            let (status, body, _) = ask(&api, "POST", "/scene", "", &scene);
            assert_eq!(status, 200, "{filters}: {body}");
        }
    }

    /// A console over the clocks of clocks() in `d` and its state directory
    /// `states`, as a new ghee starts it, with clocks loaded.
    fn clocks_again(d: &str, states: &str) -> ConsoleAPI<RenderLoopHost> {
        let mut store = SceneStore::new(d, Vec::new());
        store.state_dir = Some(std::path::PathBuf::from(states));
        store.edits_dir = store.state_dir.clone();
        let console = Arc::new(Console::new(store));
        let (opened, _) = console.store.open("clocks", None, (32, 24)).unwrap();
        ready(&console, "clocks", opened);
        ConsoleAPI::new(RenderLoopHost::new(console))
    }

    #[test]
    fn unsaved_edits_wait_after_a_stop_until_a_page_restores_or_drops_them() {
        for answer in ["restore", "drop"] {
            let (d, states, _) = clocks(&format!("draft-{answer}"));
            let api = clocks_again(&d, &states);
            let q = "scene=clocks&pass=main&name=cols&value=5";
            assert_eq!(ask(&api, "POST", "/set", q, "").0, 200);
            api.host.console.write_drafts(true);
            let draft = std::path::Path::new(&*states).join(format!("scenes{d}/clocks.draft.json"));
            assert_eq!(
                json_of(&fs::read_to_string(&draft).unwrap()),
                serde_json::json!({"version": 1, "scene": format!("{d}/clocks.scene.json"),
                    "values": {"passes": {"main": {"params": {"cols": 5}}}}})
            );
            let api = clocks_again(&d, &states);
            let cols = |api: &ConsoleAPI<RenderLoopHost>| {
                json_of(&ask(api, "GET", "/scene", "", "").1)["passes"][0]["params"]["cols"]
                    ["value"]
                    .clone()
            };
            let edits = json_of(&ask(&api, "GET", "/edits", "", "").1);
            assert_eq!(
                (cols(&api), &edits["draft"]),
                (Value::from(7), &Value::from(true))
            );
            let q = format!("name=clocks&answer={answer}");
            assert_eq!(ask(&api, "POST", "/draft", &q, "").0, 200);
            let edits = json_of(&ask(&api, "GET", "/edits", "", "").1);
            match answer {
                "restore" => {
                    assert_eq!(cols(&api), 5);
                    assert_eq!(edits["unsaved"], true);
                }
                _ => {
                    assert_eq!(cols(&api), 7);
                    assert!(!draft.exists());
                }
            }
            assert_eq!(edits["draft"], false);
            assert_eq!(
                ask(&api, "POST", "/draft", &q, ""),
                (404, "no unsaved edits wait".to_string(), None)
            );
        }
    }

    #[test]
    fn a_save_or_a_working_copy_that_matches_the_state_file_again_removes_the_draft() {
        let (d, states, _) = clocks("draft-gone");
        let api = clocks_again(&d, &states);
        let draft = std::path::Path::new(&*states).join(format!("scenes{d}/clocks.draft.json"));
        let set = |cols: i32| {
            let q = format!("scene=clocks&pass=main&name=cols&value={cols}");
            assert_eq!(ask(&api, "POST", "/set", &q, "").0, 200);
            api.host.console.write_drafts(true);
            draft.exists()
        };
        assert!(set(5));
        assert!(!set(7));
        assert!(set(5));
        assert_eq!(ask(&api, "POST", "/save", "name=clocks", "").0, 200);
        assert!(!draft.exists());
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
        store.edits_dir = store.state_dir.clone();
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
    fn a_save_of_an_own_scene_that_a_person_wrote_keeps_its_file_names_and_layout() {
        let (_d, m, console) = with_mine("own-art");
        let dir = m.join("art");
        fs::create_dir_all(&dir).unwrap();
        let text = r#"{
  "version": 1,
  "passes": [
    {"name": "main", "fragment": {"file": "art-main.glsl"},
     "params": {"cols": {"type": "int", "value": 7, "min": 1, "max": 9}}}
  ]
}
"#;
        fs::write(dir.join("art.scene.json"), text).unwrap();
        fs::write(
            dir.join("art-main.glsl"),
            "void main(){ gl_FragColor = vec4(1.0); }",
        )
        .unwrap();
        console.store.rescan();
        let (opened, _) = console.store.open("art", None, (32, 24)).unwrap();
        ready(&console, "art", opened);
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let q = "scene=art&pass=main&name=cols&value=3";
        assert_eq!(ask(&api, "POST", "/set", q, "").0, 200);
        assert_eq!(ask(&api, "POST", "/save", "name=art", "").0, 200);
        assert_eq!(
            fs::read_to_string(dir.join("art.scene.json")).unwrap(),
            text.replace("\"value\": 7", "\"value\": 3")
        );
        let mut files: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        files.sort();
        assert_eq!(files, ["art-main.glsl", "art.scene.json"]);
        let edits = json_of(&ask(&api, "GET", "/edits", "", "").1);
        assert_eq!(edits["unsaved"], false);
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

    /// RFC 9110, section 13.1.1: an edit lands when its If-Match names the
    /// scene's ETag, and each edit answers the ETag that it makes.
    #[test]
    fn an_edit_whose_if_match_names_an_older_etag_answers_412() {
        let (_d, _states, console) = clocks("stale");
        let (opened, _) = console.store.open("clocks", None, (32, 24)).unwrap();
        ready(&console, "clocks", opened);
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let read = format!("\"{}\"", rev_of(&api));
        assert_eq!(console.stats()["rev"], rev_of(&api));
        let set = |name: &str, value: u32, tags: Option<&str>| {
            let q = format!("scene=clocks&pass=main&name={name}&value={value}");
            ask_if(&api, "POST", "/set", &q, "", tags)
        };
        // a page's edits in turn, each with the ETag of the answer before
        let first = set("cols", 1, Some(&read));
        assert_eq!(first.0, 200);
        let first = first.2.unwrap();
        let second = set("cols", 2, Some(&first));
        assert_eq!(second.0, 200);
        let second = second.2.unwrap();
        // another client's edit, with the ETag that it read
        assert_eq!(
            set("rows", 4, Some(&read)),
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
        assert_eq!(set("rows", 4, Some(&format!("W/{second}"))).0, 412);
        let listed = set("rows", 4, Some(&format!("{read}, {second}")));
        assert_eq!(listed.0, 200);
        assert_eq!(set("rows", 5, Some("*")).0, 200);
        // an edit with no If-Match, as curl sends it
        assert_eq!(set("rows", 6, None).0, 200);
        assert_eq!(set("cols", 3, Some(&listed.2.unwrap())).0, 412);
        assert_eq!(console.stats()["rev"], rev_of(&api));
        assert_eq!(rows(&api), 6);
    }

    #[test]
    fn a_reload_from_a_page_left_on_another_scene_answers_409() {
        let (_d, console) = good_and_bad("reload");
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        api.host.switch(Some("good")).unwrap();
        assert_eq!(
            ask_if(&api, "POST", "/switch", "name=bad", "", Some("\"1\"")),
            (
                409,
                "the console is on scene 'good', and this reload is for scene 'bad'".to_string(),
                None
            )
        );
        assert_eq!(console.stats()["active"], "good");
        assert_eq!(
            ask_if(&api, "POST", "/switch", "name=good", "", Some("\"1\"")).0,
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
                draft: None,
                procedural: true,
            },
        );
        // the program set title's x and drew bar
        console
            .hub
            .publish_scene(Some("one"), 1, scene(doc(20.0, None, Some(4.0))).to_dict());
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
        let drawn = renderer
            .borrow()
            .with_scene(|s| s.passes[0].params.get("k").unwrap().value.to_json());
        assert_eq!(drawn, Some(Value::from(0.5)));
    }

    /// The running document, as the render thread published it last.
    fn running_doc(console: &Console) -> Value {
        console.hub.scene().unwrap()
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
            match Renderer::builder(w as u32, h as u32).device(&dev).build() {
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
    fn the_skip_interval_follows_whoever_consumes_frames() {
        assert_eq!(skip_interval(60, 0, 15.0, 0.5), Some(1.0 / 60.0));
        assert_eq!(skip_interval(0, 0, 15.0, 0.5), None);
        assert_eq!(skip_interval(0, 1, 15.0, 0.5), Some(1.0 / 15.0));
        assert_eq!(skip_interval(0, 1, 0.0, 0.5), Some(0.5));
    }

    /// One raw request through handle(), with a body cap of 10 bytes, and
    /// the raw answer.
    fn exchange(raw: &'static [u8]) -> String {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        let (_d, console) = tmp_console("transport");
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console))).with_max_body(10);
        let server = Server::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        let client = thread::spawn(move || {
            let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
            s.write_all(raw).unwrap();
            let mut out = Vec::new();
            let _ = s.read_to_end(&mut out); // the server closes the connection
            out
        });
        crate::console::hub::handle(&*console, &api, server.recv().unwrap());
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
            "frames",
            "skipped",
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
        let rev = rx.recv();
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
            let mut store = dir_store(dir.map(PathBuf::from), None, Vec::new()).unwrap();
            store.state_dir = Some(states.to_path_buf());
            store.edits_dir = store.state_dir.clone();
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
        console.hub.count_clients(1);
        assert_eq!(console.stats()["out"], serde_json::json!(["mjpeg"]));
        let c = console.hub.fanout.add();
        assert_eq!(console.stats()["out"], serde_json::json!(["h264", "mjpeg"]));
        console.hub.fanout.remove(&c);
        console.hub.count_clients(-1);
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
            Some(Command::Switch(name, _)) => assert_eq!(name, "one"),
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
        let k = renderer
            .borrow()
            .with_scene(|s| s.passes[0].params.get("k").unwrap().value.to_json());
        assert_eq!(k, Some(Value::from(0.0)));
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

    /// A controller that keeps the live state of its scene.
    struct Told(Live);

    impl SceneController for Told {
        fn live(&mut self) -> Option<&mut Live> {
            Some(&mut self.0)
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn the_scenes_process_hears_of_each_param_and_each_text_that_the_console_sets() {
        let renderer = RefCell::new(gpu(32, 24));
        let (_d, console) = tmp_console("told");
        let st = RefCell::new(Running::default());
        let mut sc = text_scene("I");
        sc.params
            .insert("gain".to_string(), Param::float("gain", 0.0));
        sc.passes[0]
            .params
            .insert("amt".to_string(), Param::float("amt", 0.0));
        let mut live = Live::new(&sc);
        live.tick_params();
        let ctrl: ControllerSlot = RefCell::new(Some(Box::new(Told(live))));
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
        let ticked = live_of(&ctrl).unwrap().tick_params();
        assert_eq!(
            Value::Object(ticked),
            serde_json::json!({"gain": 0.5, "amt": 0.25, "label.text": "O"})
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
        store.edits_dir = store.state_dir.clone();
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
            renderer.borrow_mut().setup(&sc).unwrap();
            frame()
        };
        let (at8, at20) = (built(8.0), built(20.0));
        {
            let _spawn = crate::testing::SCRIPTS.read().unwrap();
            load(&console, &renderer, &st, &ctrl, "proc", 0);
        }
        assert!(frame() == at8, "the saved x places the layer");
        // The loop's tick and update, until the process answers.
        let mut n = 0;
        until(|| {
            let mut slot = ctrl.borrow_mut();
            let running = slot.as_mut().expect("the scene runs");
            running.tick(n, 0.0, 1.0 / 60.0).unwrap();
            running.update(&mut *renderer.borrow_mut(), n, 0.0).unwrap();
            drop(slot);
            n += 1;
            frame() == at20
        });
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
        let dot = r#"[{"op": "add", "path": "/main/0", "value": {"name": "dot", "shape": "rect", "x": 0, "y": 0, "width": 4, "height": 4, "color": "blue"}}]"#;
        let answer = [
            wire(4, "level", 1, &[128]),
            wire(1, "amt", 1, &0.75f32.to_le_bytes()),
            wire(6, "", dot.len(), dot.as_bytes()),
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
        store.edits_dir = store.state_dir.clone();
        let mut console = Console::new(store);
        console.hub.publish_every = 0.0;
        let console = Arc::new(console);
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
            };
            let mut grab = Grab(Rc::clone(&got));
            let mut lp = Loop::new(&mut shared, vec![&mut grab]);
            lp.frames = 1;
            lp.controller = Some(&ctrl);
            lp.poll = Some(Box::new(|| drain(&console, &renderer, &st, &ctrl)));
            lp.run().unwrap();
            drop(lp);
            publish(&console, &renderer, &st, &ctrl, false);
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
        let dot = r#"[{"op": "add", "path": "/main/1", "value": {"name": "dot", "shape": "rect", "width": 2, "height": 2}}]"#;
        let answer = [
            wire(1, "amt", 1, &0.75f32.to_le_bytes()),
            wire(1, "title.x", 1, &20.0f32.to_le_bytes()),
            wire(6, "", dot.len(), dot.as_bytes()),
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
        store.edits_dir = store.state_dir.clone();
        let mut console = Console::new(store);
        console.hub.publish_every = 0.0;
        let console = Arc::new(console);
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let renderer = Rc::new(RefCell::new(gpu(16, 8)));
        let st = Rc::new(RefCell::new(Running::default()));
        let ctrl: ControllerSlot = RefCell::new(None);
        let got = Rc::new(RefCell::new(Vec::new()));
        let frame = || {
            let _spawn = crate::testing::SCRIPTS.read().unwrap();
            let mut shared = SharedRenderer {
                renderer: Rc::clone(&renderer),
            };
            let mut grab = Grab(Rc::clone(&got));
            let mut lp = Loop::new(&mut shared, vec![&mut grab]);
            lp.frames = 1;
            lp.controller = Some(&ctrl);
            lp.poll = Some(Box::new(|| drain(&console, &renderer, &st, &ctrl)));
            lp.run().unwrap();
            drop(lp);
            publish(&console, &renderer, &st, &ctrl, false);
        };
        let locked = || console.stats()["locked"].clone();
        let none = serde_json::json!({"params": [], "fields": []});
        let set = |q: &str| {
            let (status, body, _) = ask(&api, "POST", "/set", &format!("scene=proc&{q}"), "");
            (status, body)
        };
        api.host.switch(Some("proc")).unwrap();
        frame();
        assert_eq!(locked(), none, "before the process sets anything");
        assert_eq!(set("pass=main&name=amt&value=0.25").0, 200);
        let all = serde_json::json!({"params": ["main.amt"],
                                     "fields": ["dot.height", "dot.width", "title.x"]});
        while locked() != all {
            frame();
        }
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
            set("layer=dot&field=width&value=3"),
            (
                403,
                "the scene's program sets layer field 'dot.width'".to_string()
            )
        );
        assert_eq!(set("layer=title&field=y&value=3").0, 200);
        // a page one step behind posts the values from before the sets
        let mut stale = json_of(&ask(&api, "GET", "/scene", "name=proc", "").1);
        stale["passes"][0]["params"]["amt"]["value"] = Value::from(0.9);
        for layer in stale["passes"][0]["layers"].as_array_mut().unwrap() {
            match layer["name"].as_str() {
                Some("title") => {
                    layer["x"] = Value::from(9);
                    layer["y"] = Value::from(6);
                }
                _ => layer["width"] = Value::from(9),
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

    /// RFC 9110, section 8.8.3: a strong ETag names one document. A change
    /// by the scene's process gives the scene a new revision, and the
    /// document that GET /scene answers under it holds the change.
    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_change_by_the_scene_process_moves_the_etag_and_shows_in_the_scene() {
        let d = crate::testing::TempDir::new("console-rev");
        crate::testing::script(&d, "proc", &sets_later_script());
        let mut console = Console::new(SceneStore::new(&d, Vec::new()));
        console.hub.publish_every = 0.0;
        let console = Arc::new(console);
        let api = ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console)));
        let renderer = Rc::new(RefCell::new(gpu(16, 8)));
        let st = Rc::new(RefCell::new(Running::default()));
        let ctrl: ControllerSlot = RefCell::new(None);
        let frame = || {
            let _spawn = crate::testing::SCRIPTS.read().unwrap();
            let mut shared = SharedRenderer {
                renderer: Rc::clone(&renderer),
            };
            let mut lp = Loop::new(&mut shared, vec![]);
            lp.frames = 1;
            lp.controller = Some(&ctrl);
            lp.poll = Some(Box::new(|| drain(&console, &renderer, &st, &ctrl)));
            lp.run().unwrap();
            drop(lp);
            publish(&console, &renderer, &st, &ctrl, false);
        };
        let read = || {
            let (_, body, etag) = ask(&api, "GET", "/scene", "name=proc", "");
            (json_of(&body), etag.unwrap())
        };
        let has_dot = |doc: &Value| {
            doc["passes"][0]["layers"]
                .as_array()
                .is_some_and(|l| l.iter().any(|l| l["name"] == "dot"))
        };
        api.host.switch(Some("proc")).unwrap();
        frame();
        let (before, old) = read();
        assert!(!has_dot(&before));
        until(|| {
            frame();
            has_dot(&read().0)
        });
        let (after, new) = read();
        assert_ne!(new, old, "{after}");
        let set = ask_if(
            &api,
            "POST",
            "/set",
            "name=main.amt&value=0.5",
            "",
            Some(&old),
        );
        assert_eq!(set.0, 412);
    }

    /// A GET /scene after a /set reads the set, once the render loop that
    /// runs on another thread applied it.
    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_read_after_a_set_holds_the_set_under_the_etag_that_the_set_answered() {
        let d = crate::testing::TempDir::new("console-read");
        crate::testing::script(&d, "proc", &sets_later_script());
        let console = Arc::new(Console::new(SceneStore::new(&d, Vec::new())));
        let api = Arc::new(ConsoleAPI::new(RenderLoopHost::new(Arc::clone(&console))));
        let renderer = Rc::new(RefCell::new(gpu(16, 8)));
        let st = Rc::new(RefCell::new(Running::default()));
        let ctrl: ControllerSlot = RefCell::new(None);
        api.host.switch(Some("proc")).unwrap();
        {
            let _spawn = crate::testing::SCRIPTS.read().unwrap();
            drain(&console, &renderer, &st, &ctrl);
        }
        console.draining.store(true, Ordering::SeqCst);
        let client = {
            let api = Arc::clone(&api);
            thread::spawn(move || {
                (1..=5)
                    .map(|i| {
                        let value = f64::from(i) / 10.0;
                        let q = format!("name=main.amt&value={value}");
                        let set = ask(&api, "POST", "/set", &q, "");
                        let (_, body, etag) = ask(&api, "GET", "/scene", "name=proc", "");
                        let read = json_of(&body)["passes"][0]["params"]["amt"]["value"].clone();
                        (set.2, etag, (read.as_f64().unwrap() - value).abs() < 1e-6)
                    })
                    .collect::<Vec<_>>()
            })
        };
        while !client.is_finished() {
            drain(&console, &renderer, &st, &ctrl);
            thread::sleep(Duration::from_millis(2));
        }
        for (set, read, holds) in client.join().unwrap() {
            assert!(holds && set == read, "set {set:?}, read {read:?}");
        }
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
        store.edits_dir = store.state_dir.clone();
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
        let (_d, mut console) = tmp_console("drawn");
        Arc::get_mut(&mut console).unwrap().hub.publish_every = 0.0;
        let st = Rc::new(RefCell::new(Running::default()));
        let scene = Scene::from_dict(&serde_json::from_str(SCENE).unwrap(), None, None).unwrap();
        let ctrl: ControllerSlot = RefCell::new(Some(Box::new(Told(Live::new(&scene)))));
        assert_eq!(
            set_scene(&console, &renderer, &st, scene, 0, None, false),
            ""
        );
        // what a procedural scene's drawing and ticks hand its live state
        let mut target = SharedRenderer {
            renderer: Rc::clone(&renderer),
        };
        let live = || live_of(&ctrl).unwrap();
        let bar = serde_json::json!({"name": "bar", "shape": "rect", "width": 4});
        let words = serde_json::json!({"name": "label", "text": "hi", "font_size": 8});
        let steps: [Step<'_>; 6] = [
            (
                "added",
                Box::new(|t| live().add_layer(t, "main", 0, &bar).unwrap()),
                serde_json::json!([bar]),
            ),
            (
                "a text added after it",
                Box::new(|t| live().add_layer(t, "main", 1, &words).unwrap()),
                serde_json::json!([bar, words]),
            ),
            (
                "a new look",
                Box::new(|t| {
                    let look = serde_json::json!({"name": "bar", "shape": "rect", "width": 9});
                    assert!(live().update_layer(t, &look).unwrap());
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
                    live().set_layer(t, &at, Given::Numbers(&[0.0])).unwrap();
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
                    live().set_layer(t, &at, Given::Text("ho")).unwrap();
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
                    live().set_layer(t, &at, Given::Numbers(&[3.0])).unwrap();
                }),
                serde_json::json!([
                    {"name": "bar", "visible": false, "shape": "rect", "width": 9},
                    {"name": "label", "text": "ho", "x": 3, "font_size": 8}
                ]),
            ),
        ];
        for (n, (what, step, layers)) in steps.iter().enumerate() {
            step(&mut target);
            publish(&console, &renderer, &st, &ctrl, false);
            assert_eq!(running_layers(&console), floats(layers), "{what}");
            assert_eq!(changes(&console), Value::from(n + 1), "{what}");
        }
        // a frame with no change publishes nothing new
        publish(&console, &renderer, &st, &ctrl, false);
        assert_eq!(changes(&console), Value::from(steps.len()));
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_scenes_code_reaches_the_running_document_at_most_every_publish_every_and_an_edit_at_once()
    {
        let renderer = Rc::new(RefCell::new(gpu(32, 24)));
        let (_d, mut console) = tmp_console("throttled");
        Arc::get_mut(&mut console).unwrap().hub.publish_every = 1.0;
        let clock = Clock::at(0.0);
        let st = Rc::new(RefCell::new(Running::default()));
        let scene = Scene::from_dict(&serde_json::from_str(SCENE).unwrap(), None, None).unwrap();
        let ctrl: ControllerSlot = RefCell::new(Some(Box::new(Told(Live::new(&scene)))));
        assert_eq!(
            set_scene(&console, &renderer, &st, scene, 0, None, false),
            ""
        );
        let mut target = SharedRenderer {
            renderer: Rc::clone(&renderer),
        };
        let bar = |width: i32| serde_json::json!({"name": "bar", "shape": "rect", "width": width});
        let live = || live_of(&ctrl).unwrap();
        live().add_layer(&mut target, "main", 0, &bar(4)).unwrap();
        publish(&console, &renderer, &st, &ctrl, false);
        assert_eq!(
            running_layers(&console),
            floats(&serde_json::json!([bar(4)]))
        );
        assert_eq!(changes(&console), Value::from(1));
        assert!(live().update_layer(&mut target, &bar(9)).unwrap());
        clock.pass(0.5);
        publish(&console, &renderer, &st, &ctrl, false);
        assert_eq!(
            running_layers(&console),
            floats(&serde_json::json!([bar(4)]))
        );
        assert_eq!(changes(&console), Value::from(1));
        publish(&console, &renderer, &st, &ctrl, true);
        assert_eq!(
            running_layers(&console),
            floats(&serde_json::json!([bar(9)]))
        );
        assert_eq!(changes(&console), Value::from(2));
        assert!(live().update_layer(&mut target, &bar(5)).unwrap());
        clock.pass(0.5);
        publish(&console, &renderer, &st, &ctrl, false);
        assert_eq!(
            running_layers(&console),
            floats(&serde_json::json!([bar(5)]))
        );
        assert_eq!(changes(&console), Value::from(3));
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

    /// A sink that keeps the red of the first pixel of every frame.
    struct Reds(Rc<RefCell<Vec<u8>>>);

    impl Output for Reds {
        fn write(&mut self, frame: Frame<'_>) -> Result<()> {
            self.0.borrow_mut().push(frame.data[0]);
            Ok(())
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn an_array_source_sends_bytes_and_then_says_unchanged() {
        drop(gpu(8, 8)); // run() opens a renderer of its own on that device
        let d = crate::testing::TempDir::new("console-arrays");
        let scene = r#"{"version": 1, "passes": [{"name": "main",
            "arrays": [{"name": "level", "size": 1}],
            "fragment": "void main(){ gl_FragColor = vec4(level_at(0), 0.0, 0.0, 1.0); }"}]}"#;
        fs::write(d.join("level.scene.json"), scene).unwrap();
        let reds = Rc::new(RefCell::new(Vec::new()));
        let level: ArraySource = Box::new(|_size, frame| match frame {
            0 => Feed::DataU8(vec![255]),
            _ => Feed::Unchanged,
        });
        let ran = run(Options {
            store: Some(SceneStore::new(&d, Vec::new())),
            port: free_port(),
            size: Some((8, 8)),
            frames: 3,
            arrays: vec![("level".to_string(), level)],
            sinks: vec![Box::new(Reds(Rc::clone(&reds)))],
            ..Options::default()
        });
        assert!(ran.is_ok(), "{:?}", ran.err());
        assert_eq!(*reds.borrow(), [255, 255, 255]);
    }

    /// A sink that pauses the console at its first frame, and lets it go as
    /// the pause turns it off.
    struct PauseAt {
        wrote: Rc<Cell<u32>>,
        offs: Rc<Cell<u32>>,
        control: Control,
    }

    impl Output for PauseAt {
        fn persists(&self) -> bool {
            true
        }

        fn write(&mut self, _frame: Frame<'_>) -> Result<()> {
            self.wrote.set(self.wrote.get() + 1);
            if self.wrote.get() == 1 {
                self.control.pause(true);
            }
            Ok(())
        }

        fn turn_off(&mut self) -> Result<()> {
            self.offs.set(self.offs.get() + 1);
            if self.offs.get() == 1 {
                self.control.pause(false);
            }
            Ok(())
        }
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_paused_console_turns_its_sinks_off_and_draws_on_after_it() {
        drop(gpu(8, 8)); // run() opens a renderer of its own on that device
        let (_d, console) = tmp_console("pause");
        let (wrote, offs) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
        let control = Control::new();
        let ran = run(Options {
            store: Some(SceneStore::new(&console.store.dir, Vec::new())),
            port: free_port(),
            size: Some((32, 24)),
            frames: 3,
            sinks: vec![Box::new(PauseAt {
                wrote: Rc::clone(&wrote),
                offs: Rc::clone(&offs),
                control: control.clone(),
            })],
            control: control.clone(),
            ..Options::default()
        });
        assert!(ran.is_ok(), "{:?}", ran.err());
        assert_eq!(wrote.get(), 3);
        assert_eq!(offs.get(), 2, "once for the pause, once on the way out");
        assert!(!control.paused());
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_console_stopped_before_it_runs_ends_at_once() {
        drop(gpu(8, 8)); // run() opens a renderer of its own on that device
        let (_d, console) = tmp_console("stopped");
        let control = Control::new();
        control.stop();
        let drawn = Rc::new(Cell::new(0));
        let ran = run(Options {
            store: Some(SceneStore::new(&console.store.dir, Vec::new())),
            port: free_port(),
            size: Some((8, 8)),
            sinks: vec![Box::new(Bound(Rc::clone(&drawn)))],
            control,
            ..Options::default()
        });
        assert!(ran.is_ok(), "{:?}", ran.err());
        assert!(drawn.get() <= 1, "{} frames", drawn.get());
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
            size: Some((32, 24)),
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
            size: Some((32, 24)),
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

    #[test]
    fn a_console_run_with_a_filter_that_no_filter_answers_fails_before_it_serves() {
        let (_d, console) = tmp_console("unknown-filter");
        let port = free_port();
        let ran = run(Options {
            store: Some(SceneStore::new(&console.store.dir, Vec::new())),
            port,
            filters: vec![crate::filters::Filter::named("nope")],
            ..Options::default()
        });
        let e = ran.unwrap_err().to_string();
        assert!(e.starts_with("unknown filter: nope (known: "), "{e}");
        assert!(std::net::TcpListener::bind(("127.0.0.1", port)).is_ok());
    }

    /// A sink that reads GET /stats from the console on `port` at the first
    /// frame that it takes.
    struct StatsAt(u16, Rc<RefCell<Value>>);

    impl Output for StatsAt {
        fn write(&mut self, _frame: Frame<'_>) -> Result<()> {
            use std::io::{Read, Write};
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
    fn a_run_puts_back_the_signal_handlers_from_before_it() {
        extern "C" {
            fn signal(sig: c_int, handler: usize) -> usize;
        }
        const SIG_DFL: usize = 0;
        drop(gpu(8, 8)); // run() opens a renderer of its own on that device
        let (_d, console) = tmp_console("signals");
        let control = Control::new();
        crate::control::stop_on_signals(&control).unwrap();
        let ran = run(Options {
            store: Some(SceneStore::new(&console.store.dir, Vec::new())),
            port: free_port(),
            size: Some((8, 8)),
            frames: 1,
            control,
            ..Options::default()
        });
        assert!(ran.is_ok(), "{:?}", ran.err());
        for sig in [crate::control::SIGINT, crate::control::SIGTERM] {
            assert_eq!(unsafe { signal(sig, SIG_DFL) }, SIG_DFL, "{sig}");
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
            size: Some((32, 24)),
            rotate: Rotation::Deg90,
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
        renderer.set_rotate(Rotation::Deg90).unwrap();
        let sinks: Vec<Box<dyn Output>> = vec![Box::new(Bound(Rc::new(Cell::new(0))))];
        render_loop(
            &console,
            Rc::new(RefCell::new(renderer)),
            sinks,
            Vec::new(),
            None,
            None,
        );
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
        render_loop(
            &console,
            Rc::new(RefCell::new(renderer)),
            sinks,
            Vec::new(),
            None,
            None,
        );
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
            (console.hub.w, console.hub.h) = (8, 8);
            let console = Arc::new(console);
            console.ask("one");
            console.hub.count_clients(viewers);
            let summary: Summary = Rc::new(RefCell::new(None));
            let got = Rc::clone(&summary);
            let stopper = {
                let console = Arc::clone(&console);
                thread::spawn(move || {
                    // a second frame with a viewer, a skipped one without
                    let (frames, skipped) = if viewers == 0 { (1, 1) } else { (2, 0) };
                    until(|| {
                        let stats = console.stats();
                        stats["frames"].as_u64() >= Some(frames)
                            && stats["skipped"].as_u64() >= Some(skipped)
                    });
                    console.stop();
                })
            };
            let renderer = Rc::new(RefCell::new(gpu(8, 8)));
            let hook: StatsHook = Box::new(move |times, wall, out| {
                *got.borrow_mut() = Some((times.len(), wall, out.to_vec()));
            });
            render_loop(&console, renderer, Vec::new(), Vec::new(), None, Some(hook));
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
        let _clock = Clock::at(0.0);
        let summary: Summary = Rc::new(RefCell::new(None));
        let got = Rc::clone(&summary);
        let ran = run(Options {
            store: Some(SceneStore::new(&console.store.dir, Vec::new())),
            port: free_port(),
            size: Some((32, 24)),
            frames: 5,
            sinks: vec![Box::new(Bound(Rc::new(Cell::new(0))))],
            fps: Some(20.0),
            on_stats: Some(Box::new(move |times, wall, out| {
                *got.borrow_mut() = Some((times.len(), wall, out.to_vec()));
            })),
            ..Options::default()
        });
        assert!(ran.is_ok(), "{:?}", ran.err());
        let (n, wall, _) = summary.borrow().clone().expect("no summary");
        // five frames at 20 a second, each held to its 50 ms
        assert_eq!(n, 5);
        assert!((wall - 0.25).abs() < 1e-9, "{wall}");
    }

    fn free_port() -> u16 {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    }
}
