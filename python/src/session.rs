//! One scene, one renderer, one output. The host asks for each frame.
//!
//! The scene loading and the pacing are ghee's own.
//! A frame follows ghee::looper: `t` is the wall clock since the first frame,
//! iTimeDelta is the period since the last frame drawn, and MAX_DT caps it.
//! A tick's dt is the time since the tick before, which MAX_DT caps too.
//!
//! The engine is thread-affine, so a Session belongs to the thread that built
//! it.

use std::collections::HashMap;
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;

use ghee::console::server::{coerce, gpu_caps, H264_FPS};
use ghee::draw::{Arg, Call, CallKind};
use ghee::engine::{probe_display, quarter_turns, Feed, Given, Refused, Renderer};
use ghee::filters::{apply_filters, load_filters};
use ghee::h264::{HardwareH264, HW_COLOR};
use ghee::layer::Address;
use ghee::looper::{Frame, Pacer, RenderTarget, MAX_DT};
use ghee::output::{encode_jpeg, encode_png, expand_pattern, FixedPacer};
use ghee::procedural::{load_scene, process, scene_name};
use ghee::scene::{no_layer, no_param, short_floats, ArraySize, Scene};
use ghee::stats::{monotonic, DEST_KMS};
use serde_json::{Map, Value};

use crate::serve::{encode_bitrate, encode_size, Console, ParamSet};

/// A render with no display takes this size, as in the CLI.
pub const DEFAULT_SIZE: (i32, i32) = (320, 240);

/// The KMS scanout, as `--display` names it.
pub const KMS: &str = "kms";

/// The quality of a saved JPEG, `ghee render`'s own.
const JPEG_QUALITY: u8 = 75;

/// A failure reported to a host. `scene` marks what the caller wrote: the
/// scene, the GLSL, a param name, a feed. Anything else is the system.
pub struct Fault {
    pub scene: bool,
    pub message: String,
}

impl Fault {
    pub fn scene(message: impl Into<String>) -> Fault {
        Fault {
            scene: true,
            message: message.into(),
        }
    }

    pub fn system(message: impl Into<String>) -> Fault {
        Fault {
            scene: false,
            message: message.into(),
        }
    }
}

pub type Outcome<T> = Result<T, Fault>;

fn scene_fault(e: anyhow::Error) -> Fault {
    Fault::scene(format!("{e:#}"))
}

fn system_fault(e: anyhow::Error) -> Fault {
    Fault::system(format!("{e:#}"))
}

/// A refusal of the caller's value as a fault of the caller's, anything
/// else as the system's.
fn refusal(e: anyhow::Error) -> Fault {
    match e.downcast_ref::<Refused>() {
        Some(refused) => Fault::scene(refused.0.clone()),
        None => system_fault(e),
    }
}

/// What one frame or one skip reports: what the loop hands a scene each
/// tick. `params` holds the params that changed since the last tick, and the
/// first tick holds every param.
#[derive(Clone)]
pub struct Tick {
    pub frame: u64,
    pub time: f64,
    pub dt: f64,
    pub params: Map<String, Value>,
}

/// An iteration that begin() opened: its clock, the period since the last
/// frame drawn, and its tick.
struct Opened {
    t0: f64,
    period: Option<f64>,
    tick: Tick,
}

/// Data waiting for the next frame. U8 values reach the texture as byte / 255.
pub enum Fed {
    F32(Vec<f32>),
    U8(Vec<u8>),
}

impl Fed {
    pub fn count(&self) -> usize {
        match self {
            Fed::F32(v) => v.len(),
            Fed::U8(v) => v.len(),
        }
    }
}

/// How many values one declared array takes.
pub fn array_count(size: ArraySize) -> usize {
    match size {
        ArraySize::Linear(n) => n.max(0) as usize,
        ArraySize::Grid(w, h) => (w.max(0) * h.max(0)) as usize,
    }
}

/// One live param, as a host addresses it.
#[derive(Clone)]
pub struct ParamRef {
    /// the name on the wire: `pass.name`, or `name`
    pub wire: String,
    /// the pass that holds it. None is scene scope.
    pub pass: Option<String>,
    /// the uniform itself, without the pass
    pub uniform: String,
    pub ty: String,
}

/// What the scene declares. A host feeds the arrays, sets the params and the
/// live fields of the layers, and a console addresses the passes by index.
///
/// A param answers to `pass.name` and to `name`. A bare name is the scene
/// param, else the param of the first pass that declares it. This is the
/// addressing of the procedural wire.
pub struct Index {
    pub arrays: Vec<(String, ArraySize)>,
    pub params: Vec<ParamRef>,
    pub passes: Vec<String>,
    /// Each layer's name, and true for a text layer.
    pub layers: Vec<(String, bool)>,
    /// The place in `params` of each param, by its name on the wire.
    wires: HashMap<String, usize>,
    /// True for a text layer, by the layer's name.
    texts: HashMap<String, bool>,
}

impl Index {
    pub fn of(scene: &Scene) -> Index {
        let mut arrays: Vec<(String, ArraySize)> = Vec::new();
        for p in &scene.passes {
            for a in &p.arrays {
                match arrays.iter_mut().find(|(n, _)| *n == a.name) {
                    Some(slot) => slot.1 = a.size,
                    None => arrays.push((a.name.clone(), a.size)),
                }
            }
        }
        let layers: Vec<(String, bool)> = scene
            .layers()
            .map(|l| (l.name().to_string(), l.as_text().is_some()))
            .collect();
        let texts: HashMap<String, bool> = layers.iter().cloned().collect();
        let mut params: Vec<ParamRef> = Vec::new();
        let scoped = std::iter::once((None, &scene.params)).chain(
            scene
                .passes
                .iter()
                .map(|p| (Some(p.name.as_str()), &p.params)),
        );
        for (pass, map) in scoped {
            for (name, p) in map.iter() {
                if let Some(pass) = pass {
                    params.push(ParamRef {
                        wire: format!("{pass}.{name}"),
                        pass: Some(pass.to_string()),
                        uniform: name.clone(),
                        ty: p.ty.clone(),
                    });
                }
                if params.iter().any(|r| r.wire == *name) {
                    continue;
                }
                params.push(ParamRef {
                    wire: name.clone(),
                    pass: pass.map(str::to_string),
                    uniform: name.clone(),
                    ty: p.ty.clone(),
                });
            }
        }
        Index {
            wires: params
                .iter()
                .enumerate()
                .map(|(k, r)| (r.wire.clone(), k))
                .collect(),
            texts,
            arrays,
            params,
            passes: scene.passes.iter().map(|p| p.name.clone()).collect(),
            layers,
        }
    }

    /// True for a text layer named `name`, false for a layer of another
    /// kind, None when the scene has no such layer.
    pub fn text(&self, name: &str) -> Option<bool> {
        self.texts.get(name).copied()
    }

    /// The param named `wire` on the wire.
    fn wired(&self, wire: &str) -> Option<&ParamRef> {
        self.wires.get(wire).map(|&k| &self.params[k])
    }

    /// The name that a tick gives the param `name` of pass `pass`, None for
    /// the scene's: the bare name when it addresses that param, else
    /// `pass.name`.
    fn ticked(&self, pass: Option<&str>, name: &str) -> Option<&str> {
        match (self.wired(name), pass) {
            (Some(r), _) if r.pass.as_deref() == pass => Some(&r.wire),
            (_, Some(pass)) => self
                .wired(&format!("{pass}.{name}"))
                .map(|r| r.wire.as_str()),
            _ => None,
        }
    }

    /// True for a text layer named `name`, false for a layer of another
    /// kind. A layer that the scene lacks is a fault of the caller's.
    pub fn layer(&self, name: &str) -> Outcome<bool> {
        self.text(name).ok_or_else(|| {
            Fault::scene(no_layer(name, self.layers.iter().map(|(n, _)| n.as_str())))
        })
    }

    /// The param named by a host. `pass` addresses one pass, and a dotted name
    /// carries its own.
    pub fn param(&self, pass: Option<&str>, name: &str) -> Outcome<&ParamRef> {
        let wire = match pass {
            Some(pass) => format!("{pass}.{name}"),
            None => name.to_string(),
        };
        self.wired(&wire).ok_or_else(|| {
            Fault::scene(no_param(&wire, self.params.iter().map(|r| r.wire.as_str())))
        })
    }

    fn array(&self, name: &str) -> Outcome<ArraySize> {
        match self.arrays.iter().find(|(n, _)| n == name) {
            Some((_, size)) => Ok(*size),
            None => Err(Fault::scene(format!(
                "feed '{name}' is not an array this scene declares ({})",
                names(self.arrays.iter().map(|(n, _)| n.as_str()))
            ))),
        }
    }
}

fn names<'a>(all: impl Iterator<Item = &'a str>) -> String {
    let all: Vec<&str> = all.collect();
    match all.is_empty() {
        true => "none".to_string(),
        false => all.join(", "),
    }
}

/// Where the frames go.
enum Sink {
    None,
    /// The renderer scans out to the connected display itself.
    Kms,
}

/// The scene named by a host: a path, or a JSON document. A document's files
/// resolve in `dir`.
pub enum Source {
    Path(String),
    Doc { doc: Value, dir: String },
}

/// The host's own h264 stream cap.
#[derive(Clone, Copy)]
pub enum StreamMax {
    Pixels(i32),
    /// Of the render size's longest side.
    Fraction(f64),
}

/// What open() takes, once the host's own values are parsed.
pub struct Opening {
    pub source: Source,
    /// KMS, or None for headless.
    pub display: Option<String>,
    pub opts: Vec<(String, Value)>,
    /// The DRI card or render node.
    pub dri: Option<String>,
    /// The degrees that the frame turns clockwise on its way out.
    pub rotate: i64,
    pub size: Option<(i32, i32)>,
    pub fps: Option<f64>,
    pub timestep: Option<f64>,
    /// The console's bind address and port.
    pub serve: Option<(String, u16)>,
    pub stream_max: Option<StreamMax>,
    pub verbose: bool,
    /// Keep the output pass's alpha in the frames.
    pub alpha: bool,
}

pub struct Session {
    renderer: Renderer,
    sink: Sink,
    index: Arc<Index>,
    pacer: Option<Box<dyn Pacer>>,
    console: Option<Console>,
    /// The h264 cap for the longest side. 0 streams the render size.
    stream_longest: i32,
    /// The console's hardware H264 encoder. It issues GL calls, so it belongs
    /// to this thread, and it never outlives the renderer.
    h264: Option<HardwareH264>,
    /// The rate that the H264 stream declares: the display's refresh, else the
    /// console's own cadence.
    hz: i32,
    pending: Vec<(String, Fed)>,
    buf: Vec<u8>,
    timestep: Option<f64>,
    started: Option<f64>,
    prev: Option<f64>,
    period: Option<f64>,
    /// The clock of the last tick, of a frame or of a skip.
    ticked: Option<f64>,
    frame: u64,
    /// The tick of the frame drawn last, None before the first.
    drawn: Option<u64>,
    /// The renderer's text_errors_seq() at the console's last look.
    text_errors: u64,
    /// The value of each param, by the name that a tick gives it, as a scene
    /// file writes it, and the words of each text layer as its `text`, such
    /// as `title.text`.
    values: Map<String, Value>,
    /// Each value at the last tick.
    sent: Map<String, Value>,
    /// The iteration that begin() opened, for the next frame() or skip().
    opened: Option<Opened>,
    /// The layers that the host changed since the console last took them,
    /// noted only while a console runs.
    changed: Vec<String>,
    /// The params that the host set since the console last took them: each
    /// one's pass, None for the scene's, and its name.
    set: Vec<(Option<String>, String)>,
    /// The host changed which layers the scene holds, or a layer's kind,
    /// since the console last took the scene.
    grown: bool,
}

/// The value of each param of `scene`, by the name that a tick gives it, and
/// the words of each text layer as its `text`.
fn param_values(scene: &Scene, index: &Index) -> Map<String, Value> {
    let mut out = Map::new();
    for l in scene.text_layers() {
        out.insert(format!("{}.text", l.name), Value::from(l.text.clone()));
    }
    for r in index
        .params
        .iter()
        .filter(|r| index.ticked(r.pass.as_deref(), &r.uniform) == Some(&r.wire))
    {
        let params = match &r.pass {
            None => Some(&scene.params),
            Some(p) => scene
                .passes
                .iter()
                .find(|x| x.name == *p)
                .map(|x| &x.params),
        };
        if let Some(p) = params.and_then(|ps| ps.get(&r.uniform)) {
            out.insert(r.wire.clone(), p.value.to_json());
        }
    }
    out
}

/// Load a scene as `ghee render` loads one, filters included.
fn load(source: Source) -> Outcome<(Scene, String)> {
    let (mut scene, name) = match source {
        Source::Path(path) => {
            if let Some(exe) = process::executable(&path) {
                return Err(Fault::scene(format!(
                    "{exe}: a procedural scene runs itself and feeds its own data; \
                     open a .scene.json"
                )));
            }
            let scene = load_scene(&path, &[], true).map_err(scene_fault)?;
            (scene, scene_name(&path))
        }
        Source::Doc { doc, dir } => {
            let mut scene = Scene::from_dict(&doc, Some(&dir), None).map_err(scene_fault)?;
            scene.load_assets(&dir).map_err(scene_fault)?;
            (scene, "scene".to_string())
        }
    };
    let wanted = scene.filters.clone();
    let reg = load_filters(&[]);
    let unknown: Vec<&str> = wanted
        .iter()
        .filter(|n| !reg.contains_key(n.as_str()))
        .map(String::as_str)
        .collect();
    if !unknown.is_empty() {
        return Err(Fault::scene(format!(
            "unknown filter: {} (known: {})",
            unknown.join(", "),
            reg.keys().cloned().collect::<Vec<_>>().join(", ")
        )));
    }
    apply_filters(&mut scene, &wanted, Some(&reg)).map_err(scene_fault)?;
    Ok((scene, name))
}

/// The output named by `display`, with the options in `opts`.
fn open_sink(display: Option<&str>, opts: &[(String, Value)]) -> Outcome<Sink> {
    let Some(name) = display else {
        if !opts.is_empty() {
            return Err(Fault::scene("opts without display"));
        }
        return Ok(Sink::None);
    };
    if name != KMS {
        return Err(Fault::scene(format!("no display '{name}' (takes: kms)")));
    }
    if let Some((k, _)) = opts.iter().find(|(k, _)| k != "rgb565") {
        return Err(Fault::scene(format!(
            "opts: kms has no option {k} (takes: rgb565)"
        )));
    }
    Ok(Sink::Kms)
}

/// (size, components) of the frame. A connected display on card `dri`, else
/// on the first card with one, states its mode, which wins over the host's own
/// size.
fn geometry(
    sink: &Sink,
    size: Option<(i32, i32)>,
    dri: Option<&str>,
) -> Outcome<((i32, i32), i32)> {
    match sink {
        Sink::Kms => match probe_display(dri).or(size) {
            Some(size) => Ok((size, 4)),
            None => Err(Fault::system(
                "no connected display to size from; pass width and height",
            )),
        },
        Sink::None => Ok((size.unwrap_or(DEFAULT_SIZE), 4)),
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

/// What a drawing paints on in a scene: the scene's last pass before its
/// filters, with the names of its layers in their order, then every name
/// that the scene holds, and the names of all its layers.
#[derive(Default)]
pub struct Drawing {
    pub pass: String,
    pub layers: Vec<String>,
    pub taken: Vec<String>,
    pub all: Vec<String>,
}

/// The name of the last pass of `scene` before its filters.
pub fn draw_pass(scene: &Scene) -> String {
    scene
        .passes
        .iter()
        .rev()
        .find(|p| p.filter.is_empty())
        .map(|p| p.name.clone())
        .unwrap_or_default()
}

/// What a drawing paints on in `scene`.
pub fn drawing(scene: &Scene) -> Drawing {
    let pass = draw_pass(scene);
    let layers = scene
        .passes
        .iter()
        .find(|p| p.name == pass)
        .map(|p| p.layers.iter().map(|l| l.name().to_string()).collect())
        .unwrap_or_default();
    Drawing {
        pass,
        layers,
        taken: scene
            .taken_names()
            .into_iter()
            .map(str::to_string)
            .collect(),
        all: scene.layers().map(|l| l.name().to_string()).collect(),
    }
}

impl Session {
    pub fn open(opening: Opening) -> Outcome<Session> {
        let Opening {
            source,
            display,
            opts,
            dri,
            rotate,
            size,
            fps,
            timestep,
            serve,
            stream_max,
            verbose,
            alpha,
        } = opening;
        quarter_turns(rotate).map_err(scene_fault)?;
        let (scene, name) = load(source)?;
        let sink = open_sink(display.as_deref(), &opts)?;
        let ((w, h), components) = geometry(&sink, size, dri.as_deref())?;
        let kms = matches!(sink, Sink::Kms);
        let rgb565 = opts
            .iter()
            .find(|(k, _)| k == "rgb565")
            .is_some_and(|(_, v)| truthy(v));
        let mut renderer =
            Renderer::new(w, h, components, dri.as_deref(), kms, rgb565).map_err(system_fault)?;
        renderer.set_alpha(alpha);
        renderer.set_rotate(rotate).map_err(system_fault)?;
        let err = renderer.setup(&scene).map_err(system_fault)?;
        if !err.is_empty() {
            // the command's own line, without the engine's trailing newline
            let kind = if renderer.compile_error {
                "compile error: "
            } else {
                ""
            };
            return Err(Fault::scene(format!("{kind}{}", err.trim_end())));
        }
        let index = Arc::new(Index::of(&scene));
        let values = param_values(&scene, &index);
        let out = match &sink {
            Sink::None => Vec::new(),
            Sink::Kms => vec![DEST_KMS.to_string()],
        };
        let format = match &sink {
            _ if rgb565 => "rgb565".to_string(),
            _ if components == 3 => "rgb".to_string(),
            _ => "rgba".to_string(),
        };
        if verbose {
            eprintln!(
                "ghee: {name} {w}x{h} {format} -> {}",
                if out.is_empty() {
                    "nothing".to_string()
                } else {
                    out.join("+")
                }
            );
        }
        let longest = match stream_max {
            None => 0,
            Some(StreamMax::Pixels(n)) => n,
            Some(StreamMax::Fraction(f)) => ((f64::from(w.max(h)) * f).round() as i32).max(2),
        };
        let console = match serve {
            None => None,
            Some((host, port)) => Some(
                Console::start(
                    &host,
                    port,
                    &name,
                    &scene,
                    Arc::clone(&index),
                    (w, h),
                    &format,
                    rotate,
                    gpu_caps(&renderer),
                    out,
                    longest,
                    verbose,
                )
                .map_err(system_fault)?,
            ),
        };
        let hz = match kms {
            true => renderer.display_vrefresh(),
            false => 0,
        };
        Ok(Session {
            renderer,
            sink,
            stream_longest: longest,
            index,
            pacer: fps
                .filter(|f| *f != 0.0)
                .map(|f| Box::new(FixedPacer::new(f)) as Box<dyn Pacer>),
            console,
            h264: None,
            hz,
            pending: Vec::new(),
            buf: vec![0u8; (i64::from(h) * i64::from(w) * i64::from(components)) as usize],
            timestep,
            started: None,
            prev: None,
            period: None,
            ticked: None,
            frame: 0,
            drawn: None,
            text_errors: u64::MAX,
            values,
            sent: Map::new(),
            opened: None,
            changed: Vec::new(),
            set: Vec::new(),
            grown: false,
        })
    }

    pub fn size(&self) -> (i32, i32) {
        (self.renderer.width, self.renderer.height)
    }

    pub fn components(&self) -> i32 {
        self.renderer.components
    }

    /// The GL that renders, GL_RENDERER and GL_VERSION in one string.
    pub fn gl(&self) -> String {
        self.renderer.renderer()
    }

    /// Open this iteration, once: apply the console's sets, and give the tick
    /// that the next frame() or skip() takes and returns. Its dt is the
    /// pinned step, else the time since the tick before, which MAX_DT caps,
    /// else the engine's first 1/60, so the ticks' dt add up to the time
    /// passed while no gap passes MAX_DT. A frame drawn after a drawn one
    /// draws with that dt.
    pub fn begin(&mut self) -> Outcome<Tick> {
        if let Some(o) = &self.opened {
            return Ok(o.tick.clone());
        }
        self.drain()?;
        let params = self.changed();
        let t0 = monotonic();
        let t = self.clock(t0);
        let period = self.prev.map(|prev| t0 - prev);
        let since = self.ticked.replace(t0).map(|ticked| t0 - ticked);
        let tick = Tick {
            frame: self.frame,
            time: t,
            dt: match self.timestep {
                Some(step) => step,
                None => since.map_or(1.0 / 60.0, |s| s.min(MAX_DT)),
            },
            params,
        };
        self.opened = Some(Opened {
            t0,
            period,
            tick: tick.clone(),
        });
        Ok(tick)
    }

    /// The iteration that begin() opened, opened now when none is.
    fn take_opened(&mut self) -> Outcome<Opened> {
        if self.opened.is_none() {
            self.begin()?;
        }
        Ok(self.opened.take().expect("begin() opened the iteration"))
    }

    /// The values that changed since the last tick. The first tick holds
    /// every value.
    fn changed(&mut self) -> Map<String, Value> {
        let mut out = Map::new();
        for (k, v) in &self.values {
            if self.sent.get(k) != Some(v) {
                out.insert(k.clone(), v.clone());
            }
        }
        // `sent` holds no key that `values` lacks, so the changed values
        // bring it level.
        for (k, v) in &out {
            self.sent.insert(k.clone(), v.clone());
        }
        out
    }

    /// This iteration's t. Unpinned it is the wall clock since the first frame:
    /// a scene synced to outside time must not drift with the host's pace.
    fn clock(&mut self, t0: f64) -> f64 {
        let started = *self.started.get_or_insert(t0);
        match self.timestep {
            Some(step) => self.frame as f64 * step,
            None => t0 - started,
        }
    }

    /// How long a skipped frame waits.
    fn idle(&self) -> f64 {
        if let Some(p) = self.pacer.as_ref() {
            return p.idle_period();
        }
        match self.period {
            Some(p) if p != 0.0 => p,
            _ => 1.0 / 60.0,
        }
    }

    pub fn frame(&mut self) -> Outcome<Tick> {
        let Opened { t0, period, tick } = self.take_opened()?;
        let t = tick.time;
        self.drain()?;
        if period.is_some() {
            self.period = period;
        }
        self.prev = Some(t0);
        // The one place a frame feeds iTimeDelta: the measured period, or the
        // pinned step. A draw after a skip stretch measures the whole gap, so
        // MAX_DT caps it.
        if let Some(step) = self.timestep {
            self.renderer.set_dt(step);
        } else if let Some(period) = self.period {
            self.renderer.set_dt(period.min(MAX_DT));
        }
        if self.timestep.is_some() {
            self.renderer.set_time(Some(t));
        }
        let want = self.console.as_ref().is_some_and(Console::wants_frame);
        let drew = self.draw(want)?;
        self.mirror();
        self.drawn = Some(self.frame);
        self.frame += 1;
        let end = monotonic();
        if let Some(c) = self.console.as_ref() {
            c.frame_done(end - t0);
            let seq = self.renderer.text_errors_seq();
            if seq != self.text_errors {
                self.text_errors = seq;
                c.text_errors(&self.renderer.text_errors());
            }
            if want {
                let (w, h) = self.size();
                c.publish(&self.buf, w, h, self.renderer.components);
            }
        }
        if let Some(pacer) = self.pacer.as_mut() {
            let now = monotonic();
            nap(pacer.sleep_for(now - drew, now - t0));
        }
        Ok(tick)
    }

    /// Render the passes and hand the frame to the output. Returns the clock at
    /// the end of the render.
    fn draw(&mut self, want: bool) -> Outcome<f64> {
        let Session {
            renderer,
            sink,
            pending,
            buf,
            ..
        } = self;
        let (w, h) = (renderer.width, renderer.height);
        // The provider answers each declared array once. An array with no feed
        // keeps what the texture holds.
        let mut provider = |name: &str, _size: ArraySize| -> Feed {
            match pending.iter().position(|(n, _)| n == name) {
                Some(i) => match pending.remove(i).1 {
                    Fed::F32(v) => Feed::Data(v),
                    Fed::U8(v) => Feed::DataU8(v),
                },
                None => Feed::Unchanged,
            }
        };
        match sink {
            Sink::Kms => {
                // present() has no direct full-frame readback, so a preview
                // states its size.
                let preview = want.then(|| (&mut buf[..], (w, h)));
                renderer
                    .present(Some(&mut provider), preview)
                    .map_err(system_fault)?;
                Ok(monotonic())
            }
            Sink::None => {
                renderer
                    .render_into(Some(&mut provider), buf, None)
                    .map_err(system_fault)?;
                Ok(monotonic())
            }
        }
    }

    /// The console's hardware H264 mirror: encode the frame just drawn, with no
    /// readback. The engine holds the final pass in its FBO while the encoder
    /// exists, so encode_last() always has a texture to sample. A failure goes
    /// to Console::hardware_failed and never to the host's frame, because a
    /// stream is no reason to fail the frame.
    fn mirror(&mut self) {
        let Session {
            renderer,
            console,
            h264,
            hz,
            stream_longest,
            ..
        } = self;
        let Some(console) = console.as_ref() else {
            return;
        };
        if !console.hardware_h264() {
            if let Some(mut enc) = h264.take() {
                enc.close(); // an open encoder costs a display its direct scanout
            }
            return;
        }
        if h264.is_none() {
            let (w, h) = encode_size(renderer.width, renderer.height, *stream_longest);
            let fps = if *hz > 0 { *hz } else { H264_FPS };
            match HardwareH264::new(renderer, w, h, fps, encode_bitrate(w, h, fps)) {
                Ok(enc) => {
                    console.h264_opened(enc.w, enc.h, HW_COLOR);
                    *h264 = Some(enc);
                }
                Err(e) => return console.hardware_failed(&e.to_string()),
            }
        }
        let enc = h264.as_mut().expect("the encoder was just created");
        if console.wants_key() {
            enc.force_keyframe();
        }
        match enc.encode_last() {
            Ok((Some(au), _)) => console.push_au(au),
            Ok((None, _)) => {}
            Err(e) => {
                let why = e.to_string();
                if let Some(mut enc) = h264.take() {
                    enc.close();
                }
                console.hardware_failed(&why);
            }
        }
    }

    /// This tick needs no render. The display keeps its last frame. A console
    /// reader still gets one: while a viewer wants a frame, the skip draws, as
    /// the CLI queues a preview for its console. A preview draw that fails
    /// falls back to a plain skip, and the host's own frame() reports it.
    pub fn skip(&mut self) -> Outcome<Tick> {
        let tick = self.begin()?;
        if self.console.as_ref().is_some_and(Console::wants_draw) && self.frame().is_ok() {
            if self.pacer.is_none() {
                nap(self.idle());
            }
            return Ok(tick);
        }
        self.opened = None;
        self.frame += 1;
        if let Some(c) = self.console.as_ref() {
            c.skipped();
        }
        nap(self.idle());
        Ok(tick)
    }

    /// The last frame, row-major (height, width, components).
    pub fn pixels(&self) -> Outcome<&[u8]> {
        if matches!(self.sink, Sink::Kms) {
            return Err(Fault::system(
                "kms scans out on the GPU; no frame is read back",
            ));
        }
        if self.frame == 0 {
            return Err(Fault::system("nothing is drawn yet"));
        }
        Ok(&self.buf)
    }

    /// Write the last frame to an image file: a .png, or a .jpg, which holds
    /// no alpha. A run of `#` or a printf field in `path` takes the tick of
    /// that frame, as `ghee render -o` numbers its files.
    pub fn save(&self, path: &str) -> Outcome<()> {
        let data = self.pixels()?;
        let index = self.drawn.unwrap_or_default() as i64;
        let path = expand_pattern(path, index).map_err(scene_fault)?;
        let (width, height) = self.size();
        let components = self.renderer.components;
        let lower = path.to_ascii_lowercase();
        let bytes = if lower.ends_with(".png") {
            encode_png(Frame {
                data,
                width,
                height,
                components,
            })
        } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
            encode_jpeg(
                Frame {
                    data,
                    width,
                    height,
                    components,
                },
                JPEG_QUALITY,
            )
        } else {
            return Err(Fault::scene(format!(
                "save writes a .png or a .jpg file, got {path}"
            )));
        };
        let bytes = bytes.map_err(system_fault)?;
        std::fs::write(&path, bytes).map_err(|e| Fault::system(format!("{path}: {e}")))
    }

    /// Set one live param, with no recompile.
    pub fn set_param(&mut self, pass: Option<&str>, name: &str, value: &Value) -> Outcome<()> {
        let p = self.index.param(pass, name)?.clone();
        let v = coerce(&p.wire, &p.ty, value).map_err(scene_fault)?;
        RenderTarget::set_param(&self.renderer, p.pass.as_deref(), &p.uniform, &p.ty, &v)
            .map_err(system_fault)?;
        let (pass, uniform) = (p.pass.clone(), p.uniform.clone());
        // The host set this value itself, so no tick reports it.
        self.held(pass.as_deref(), &uniform, v.to_json(), true);
        if let Some(c) = &self.console {
            c.locks().set_param(pass.as_deref(), &uniform);
            self.set.push((pass, uniform));
        }
        Ok(())
    }

    /// Set the live field `field` of layer `layer`, or of its effect
    /// `effect`, with no rebuild. The field `text` of a text layer sets its
    /// words. A layer, a field or a value that the scene refuses is a fault
    /// of the caller's.
    pub fn set_layer(
        &mut self,
        layer: &str,
        effect: Option<&str>,
        field: &str,
        value: &Value,
    ) -> Outcome<()> {
        let text = self.index.layer(layer)?;
        let at = Address {
            layer,
            effect,
            field,
        };
        self.renderer
            .set_layer(&at, Given::Json(value))
            .map_err(refusal)?;
        // The host set these words itself, so no tick reports them.
        if let (true, None, "text") = (text, effect, field) {
            self.held_words(layer, value.clone(), true);
        }
        if let Some(c) = &self.console {
            c.locks().set_field(&at);
        }
        self.note(layer);
        Ok(())
    }

    /// Add `layer`, a layer as a scene declares it, at `index` among the
    /// layers of pass `pass`, with no rebuild. Its words, when it has them,
    /// are the host's own.
    pub fn add_layer(&mut self, pass: &str, index: usize, layer: &Value) -> Outcome<()> {
        self.renderer
            .add_layer(pass, index, layer)
            .map_err(refusal)?;
        self.took(layer);
        self.whole(layer.get("name").and_then(Value::as_str));
        self.grown = true;
        Ok(())
    }

    /// Take layer `name` out of the scene, with no rebuild.
    pub fn remove_layer(&mut self, name: &str) -> Outcome<()> {
        self.renderer.remove_layer(name).map_err(refusal)?;
        if let Some(index) = self.renderer.with_scene(Index::of) {
            self.index = Arc::new(index);
        }
        let text = format!("{name}.text");
        self.values.remove(&text);
        self.sent.remove(&text);
        self.grown = true;
        Ok(())
    }

    /// Put layer `name` at `index` among the layers of its pass, with no
    /// rebuild.
    pub fn move_layer(&mut self, name: &str, index: usize) -> Outcome<()> {
        self.renderer.move_layer(name, index).map_err(refusal)?;
        self.grown = true;
        Ok(())
    }

    /// Give the layer of the name of `layer` its look, as
    /// Renderer::update_layer() does. False when the layer had it already.
    pub fn update_layer(&mut self, layer: &Value) -> Outcome<bool> {
        let changed = self.renderer.update_layer(layer).map_err(refusal)?;
        let name = layer.get("name").and_then(Value::as_str);
        if changed {
            self.took(layer);
            if let Some(name) = name {
                self.note(name);
            }
        }
        self.whole(name);
        Ok(changed)
    }

    /// Give the layers of `calls` their looks, as Renderer::update_calls()
    /// does, and hold the words of each text call that changed its layer as
    /// the host's own. Returns the indexes of the calls left for
    /// update_layer().
    pub fn update_calls(&mut self, calls: &[Call]) -> Outcome<Vec<usize>> {
        let taken = self.renderer.update_calls(calls).map_err(refusal)?;
        for call in taken.changed.iter().map(|&k| &calls[k]) {
            if let (CallKind::Text, Some(Arg::Str(words))) = (call.kind, call.args.first()) {
                self.held_words(&call.layer, Value::from(words.as_str()), true);
            }
        }
        for &k in &taken.changed {
            self.note(&calls[k].layer);
        }
        for (k, call) in calls.iter().enumerate() {
            if !taken.left.contains(&k) {
                self.whole(Some(&call.layer));
            }
        }
        Ok(taken.left)
    }

    /// Lock layer `name` whole in a console: the host added it or gave it a
    /// look.
    fn whole(&self, name: Option<&str>) {
        if let (Some(c), Some(name)) = (&self.console, name) {
            c.locks().set_layer(name);
        }
    }

    /// Record that the host changed layer `name`, for a console to take.
    fn note(&mut self, name: &str) {
        if self.console.is_some() {
            self.changed.push(name.to_string());
        }
    }

    /// Index the scene again after the layer `layer` joined it or changed
    /// its kind, and hold the words of a text layer as the host's own, which
    /// no tick reports.
    fn took(&mut self, layer: &Value) {
        let Some(name) = layer.get("name").and_then(Value::as_str) else {
            return;
        };
        let text = self.renderer.with_layer(name, |_, l| l.as_text().is_some());
        if self.index.text(name) != text {
            if let Some(index) = self.renderer.with_scene(Index::of) {
                self.index = Arc::new(index);
            }
            self.grown = true;
        }
        let words = self
            .renderer
            .with_layer(name, |_, l| l.as_text().map(|l| l.text.clone()));
        match words.flatten() {
            Some(words) => self.held_words(name, Value::from(words), true),
            None => {
                let text = format!("{name}.text");
                self.values.remove(&text);
                self.sent.remove(&text);
            }
        }
    }

    /// What a drawing paints on in the scene, as drawing() gives it.
    pub fn drawing(&self) -> Drawing {
        self.renderer.with_scene(drawing).unwrap_or_default()
    }

    /// The name of the scene's last pass before its filters.
    fn draw_pass(&self) -> String {
        self.renderer.with_scene(draw_pass).unwrap_or_default()
    }

    /// The width and height in pixels of the block that `words` lay out in
    /// on the pass that a drawing paints on, as Renderer::measure() gives
    /// them.
    pub fn measure(
        &self,
        words: &str,
        font_size: Option<f64>,
        font_family: Option<&str>,
    ) -> Outcome<(f64, f64)> {
        let pass = self.draw_pass();
        let [w, h] = self
            .renderer
            .measure(&pass, words, font_size, font_family)
            .map_err(refusal)?;
        Ok((w, h))
    }

    /// The backlight level of the display, 0 to 1. None for a display with
    /// no backlight device, and with no KMS display.
    pub fn brightness(&self) -> Outcome<Option<f64>> {
        self.renderer.brightness().map_err(system_fault)
    }

    /// Set the backlight of the display, 0 to 1. A level out of range, and a
    /// display with no backlight device, are faults of the caller's.
    pub fn set_brightness(&self, level: f64) -> Outcome<f64> {
        self.renderer.set_brightness(level).map_err(refusal)
    }

    /// Hold `value` for the param `name` of `pass`, under the name that a
    /// tick gives it, and as already handed out when `sent`.
    fn held(&mut self, pass: Option<&str>, name: &str, value: Value, sent: bool) {
        if let Some(ticked) = self.index.ticked(pass, name).map(str::to_string) {
            if sent {
                self.sent.insert(ticked.clone(), value.clone());
            }
            self.values.insert(ticked, value);
        }
    }

    /// Hold `words` as the words of text layer `layer`, under its `text`, and
    /// as already handed out when `sent`.
    fn held_words(&mut self, layer: &str, words: Value, sent: bool) {
        let text = format!("{layer}.text");
        if sent {
            self.sent.insert(text.clone(), words.clone());
        }
        self.values.insert(text, words);
    }

    /// Hold data for the next frame. That frame consumes it once.
    pub fn feed(&mut self, name: &str, data: Fed) -> Outcome<()> {
        let want = array_count(self.index.array(name)?);
        if data.count() != want {
            return Err(Fault::scene(format!(
                "array '{name}' takes {want} values, got {}",
                data.count()
            )));
        }
        match self.pending.iter_mut().find(|(n, _)| n == name) {
            Some(slot) => slot.1 = data,
            None => self.pending.push((name.to_string(), data)),
        }
        Ok(())
    }

    /// What a console queued, applied before this frame draws: params, live
    /// fields and the looks of applies. The console then takes the scene,
    /// when it changed.
    fn drain(&mut self) -> Outcome<()> {
        let Some(console) = self.console.as_ref() else {
            return Ok(());
        };
        let layers = console.take_layers();
        let looks = console.take_looks();
        let params = console.take_params();
        let edited = !(layers.is_empty() && looks.is_empty() && params.is_empty());
        for (pass, name, ty, value) in params {
            RenderTarget::set_param(&self.renderer, pass.as_deref(), &name, &ty, &value)
                .map_err(system_fault)?;
            self.held(pass.as_deref(), &name, value.to_json(), false);
        }
        for set in layers {
            self.renderer
                .set_layer(&set.at(), Given::Json(&set.value))
                .map_err(system_fault)?;
            // A text layer alone takes a `text`.
            if let (None, "text") = (set.effect.as_deref(), set.field.as_str()) {
                self.held_words(&set.layer, set.value.clone(), false);
            }
        }
        let applied = !looks.is_empty();
        let mut failed = Vec::new();
        for look in looks {
            match self.renderer.replace_layer(&look) {
                Ok(()) => self.took(&look),
                Err(e) => failed.push(e.to_string()),
            }
        }
        if let (true, Some(c)) = (applied, self.console.as_ref()) {
            c.look_error(&failed.join("\n"));
        }
        let by_host = self.grown || !(self.changed.is_empty() && self.set.is_empty());
        let (changed, set) = (
            std::mem::take(&mut self.changed),
            std::mem::take(&mut self.set),
        );
        if edited || self.grown {
            let scene = self.renderer.with_scene(Scene::to_dict);
            if let (Some(c), Some(scene)) = (self.console.as_ref(), scene) {
                c.publish_scene(scene, Arc::clone(&self.index), by_host);
            }
            self.grown = false;
        } else if by_host {
            let (layers, params) = self.changes(changed, set);
            if let Some(c) = self.console.as_ref() {
                c.publish_changes(layers, params);
            }
        }
        Ok(())
    }

    /// The layers named `changed`, whole, and the values of the params of
    /// `set`, as the renderer's scene holds them.
    fn changes(
        &self,
        mut changed: Vec<String>,
        set: Vec<(Option<String>, String)>,
    ) -> (Vec<Value>, Vec<ParamSet>) {
        let mut params = Vec::new();
        for (pass, name) in set {
            let value = self.renderer.with_scene(|s| {
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
        changed.sort_unstable();
        changed.dedup();
        let layers = changed
            .iter()
            .filter_map(|name| {
                self.renderer.with_layer(name, |_, l| {
                    let mut doc = l.to_dict();
                    short_floats(&mut doc);
                    doc
                })
            })
            .collect();
        (layers, params)
    }

    /// Drop everything the session holds.
    pub fn close(&mut self) {
        if let Some(c) = self.console.take() {
            c.stop();
        }
        // the encoder holds engine resources, so it goes before the renderer
        if let Some(mut enc) = self.h264.take() {
            enc.close();
        }
        self.sink = Sink::None;
        self.renderer.close();
    }
}

fn nap(seconds: f64) {
    if seconds > 0.0 {
        sleep(Duration::from_secs_f64(seconds));
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.close();
    }
}
