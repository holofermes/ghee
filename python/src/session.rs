//! One scene, one renderer, one output. The host asks for each frame.
//!
//! The scene loading, the driver construction and the pacing are ghee's own.
//! A frame follows ghee::looper: `t` is the wall clock since the first frame,
//! DT is the measured frame period, and MAX_DT caps it.
//!
//! The engine is thread-affine, so a Session belongs to the thread that built
//! it.

use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;

use ghee::console::server::{coerce, H264_FPS};
use ghee::drivers::{available_drivers, driver_options, get_driver, DisplayDriver};
use ghee::engine::{probe_display, Feed, Renderer};
use ghee::filters::{apply_filters, load_filters};
use ghee::h264::{HardwareH264, HW_COLOR};
use ghee::looper::{Frame, Pacer, RenderTarget, MAX_DT};
use ghee::output::{encode_jpeg, encode_png, expand_pattern, FixedPacer};
use ghee::procedural::{load_scene, process, scene_name};
use ghee::scene::{ArraySize, ParamValue, Scene};
use ghee::stats::{monotonic, DEST_KMS};
use serde_json::{Map, Value};

use crate::serve::{encode_bitrate, encode_size, Console};

/// A render with no panel and no display takes this size, as in the CLI.
pub const DEFAULT_SIZE: (i32, i32) = (320, 240);

/// The KMS scanout, as `--display` names it.
pub const KMS: &str = "kms";

/// The quality of a saved JPEG, `ghee render`'s own.
const JPEG_QUALITY: u8 = 75;

/// A failure reported to a host. `scene` marks what the caller wrote: the
/// scene, the GLSL, a param name, a feed. Anything else is the driver or the
/// system.
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

/// What one frame or one skip reports: the numbers that the loop hands a scene
/// each tick.
#[derive(Clone, Copy)]
pub struct Tick {
    pub frame: u64,
    pub time: f64,
    pub dt: f64,
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
pub struct ParamRef {
    /// the name on the wire: `pass.name`, or `name`
    pub wire: String,
    /// the pass that holds it. None is scene scope.
    pub pass: Option<String>,
    /// the uniform itself, without the pass
    pub uniform: String,
    pub ty: String,
}

/// What the scene declares. A host feeds the arrays and sets the params, and a
/// console addresses the passes by index.
///
/// A param answers to `pass.name` and to `name`. A bare name is the scene
/// param, else the param of the first pass that declares it. This is the
/// addressing of the procedural wire.
pub struct Index {
    pub arrays: Vec<(String, ArraySize)>,
    pub params: Vec<ParamRef>,
    pub passes: Vec<String>,
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
            arrays,
            params,
            passes: scene.passes.iter().map(|p| p.name.clone()).collect(),
        }
    }

    /// The param named by a host. `pass` addresses one pass, and a dotted name
    /// carries its own.
    pub fn param(&self, pass: Option<&str>, name: &str) -> Outcome<&ParamRef> {
        let wire = match pass {
            Some(pass) => format!("{pass}.{name}"),
            None => name.to_string(),
        };
        self.params.iter().find(|r| r.wire == wire).ok_or_else(|| {
            Fault::scene(format!(
                "param '{wire}' is not a param this scene declares ({})",
                names(self.params.iter().map(|r| r.wire.as_str()))
            ))
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
    Panel(Box<dyn DisplayDriver>),
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
    pub output: Option<String>,
    pub opts: Vec<(String, Value)>,
    pub size: Option<(i32, i32)>,
    pub fps: Option<f64>,
    pub timestep: Option<f64>,
    /// The console's bind address and port.
    pub stats: Option<(String, u16)>,
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
    frame: u64,
    /// The tick of the frame drawn last, None before the first.
    drawn: Option<u64>,
    lit: bool,
    /// The renderer's text_errors_seq() at the console's last look.
    text_errors: u64,
    /// The words of each string param, by the bare name that the wire gives
    /// it.
    strings: Map<String, Value>,
    /// The words of each string param at the last changed_params().
    sent: Map<String, Value>,
}

/// The words of each string param of `scene`, by the bare name that `index`
/// gives it.
fn string_params(scene: &Scene, index: &Index) -> Map<String, Value> {
    let mut out = Map::new();
    for r in index
        .params
        .iter()
        .filter(|r| r.ty == "string" && r.wire == r.uniform)
    {
        let params = match &r.pass {
            None => Some(&scene.params),
            Some(p) => scene
                .passes
                .iter()
                .find(|x| x.name == *p)
                .map(|x| &x.params),
        };
        if let Some(ParamValue::Str(words)) =
            params.and_then(|ps| ps.get(&r.uniform)).map(|p| &p.value)
        {
            out.insert(r.wire.clone(), Value::from(words.clone()));
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

/// The driver named by `output`, built from `opts` as --display-opts does.
fn open_sink(output: Option<&str>, opts: &[(String, Value)]) -> Outcome<Sink> {
    let Some(name) = output else {
        if !opts.is_empty() {
            return Err(Fault::scene("opts without output"));
        }
        return Ok(Sink::None);
    };
    if name == KMS {
        if let Some((k, _)) = opts.iter().find(|(k, _)| k != "rgb565") {
            return Err(Fault::scene(format!(
                "opts: kms has no option {k} (takes: rgb565)"
            )));
        }
        return Ok(Sink::Kms);
    }
    let Ok(spec) = get_driver(name) else {
        let have: Vec<&str> = available_drivers();
        return Err(Fault::scene(format!(
            "no display driver '{name}' (this build has {}, and kms)",
            have.join(", ")
        )));
    };
    // check before the driver runs: a constructor claims the GPIO lines and the
    // SPI bus first
    let known = driver_options(spec);
    let unknown: Vec<&str> = opts
        .iter()
        .map(|(k, _)| k.as_str())
        .filter(|k| !known.iter().any(|o| o.name == *k))
        .collect();
    if !unknown.is_empty() {
        return Err(Fault::scene(format!(
            "opts: {name} has no option {} (takes: {})",
            unknown.join(", "),
            known.iter().map(|o| o.name).collect::<Vec<_>>().join(", ")
        )));
    }
    Ok(Sink::Panel((spec.open)(opts).map_err(system_fault)?))
}

/// (size, components) of the render. A panel states its own geometry and pixel
/// layout, and a connected display states its mode. Both win over the host's
/// own size.
fn geometry(sink: &Sink, size: Option<(i32, i32)>) -> Outcome<((i32, i32), i32)> {
    match sink {
        Sink::Kms => match probe_display(None).or(size) {
            Some(size) => Ok((size, 4)),
            None => Err(Fault::system(
                "no connected display to size from; pass width and height",
            )),
        },
        Sink::Panel(d) => Ok(((d.width(), d.height()), d.components())),
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

impl Session {
    pub fn open(opening: Opening) -> Outcome<Session> {
        let Opening {
            source,
            output,
            opts,
            size,
            fps,
            timestep,
            stats,
            stream_max,
            verbose,
            alpha,
        } = opening;
        let (scene, name) = load(source)?;
        let sink = open_sink(output.as_deref(), &opts)?;
        let ((w, h), components) = geometry(&sink, size)?;
        let kms = matches!(sink, Sink::Kms);
        let rgb565 = opts
            .iter()
            .find(|(k, _)| k == "rgb565")
            .is_some_and(|(_, v)| truthy(v));
        let mut renderer =
            Renderer::new(w, h, components, None, kms, rgb565).map_err(system_fault)?;
        renderer.set_alpha(alpha);
        let err = renderer.setup(&scene).map_err(system_fault)?;
        if !err.is_empty() {
            // the command's own line, without the engine's trailing newline
            return Err(Fault::scene(format!("compile error: {}", err.trim_end())));
        }
        let index = Arc::new(Index::of(&scene));
        let strings = string_params(&scene, &index);
        let out = match &sink {
            Sink::None => Vec::new(),
            Sink::Kms => vec![DEST_KMS.to_string()],
            Sink::Panel(d) => vec![d.dest().to_string()],
        };
        let format = match &sink {
            Sink::Panel(d) => d.native_format().to_string(),
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
        let console = match stats {
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
            frame: 0,
            drawn: None,
            lit: false,
            text_errors: u64::MAX,
            strings,
            sent: Map::new(),
        })
    }

    pub fn size(&self) -> (i32, i32) {
        (self.renderer.width, self.renderer.height)
    }

    pub fn components(&self) -> i32 {
        self.renderer.components
    }

    /// This iteration's tick, drawn or skipped. DT is the pinned step, else the
    /// last measured period.
    fn tick(&self, t: f64) -> Tick {
        Tick {
            frame: self.frame,
            time: t,
            dt: self
                .timestep
                .or(self.period)
                .unwrap_or(1.0 / 60.0)
                .min(MAX_DT),
        }
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
        let t0 = monotonic();
        let t = self.clock(t0);
        let tick = self.tick(t);
        self.drain()?;
        if let Some(prev) = self.prev {
            self.period = Some(t0 - prev);
        }
        self.prev = Some(t0);
        // The one place a frame feeds DT: the measured period, or the pinned
        // step. A draw after a skip stretch measures the whole gap, so MAX_DT
        // caps it.
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
        if !self.lit {
            // the panel now holds a frame to light
            self.lit = true;
            if let Sink::Panel(d) = &mut self.sink {
                d.turn_on().map_err(system_fault)?;
            }
        }
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
    /// the end of the render, before the panel write.
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
            Sink::Panel(d) => {
                renderer
                    .render_into(Some(&mut provider), buf, None)
                    .map_err(system_fault)?;
                let drew = monotonic();
                d.write(buf).map_err(system_fault)?;
                Ok(drew)
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

    /// This tick needs no render. The panel keeps its last frame. A console
    /// reader still gets one: while a viewer wants a frame, the skip draws, as
    /// the CLI queues a preview for its console. A preview draw that fails
    /// falls back to a plain skip; the host's own frame() reports it.
    pub fn skip(&mut self) -> Tick {
        if self.console.as_ref().is_some_and(Console::wants_draw) {
            if let Ok(tick) = self.frame() {
                if self.pacer.is_none() {
                    nap(self.idle());
                }
                return tick;
            }
        }
        let t0 = monotonic();
        let t = self.clock(t0);
        let tick = self.tick(t);
        self.frame += 1;
        if let Some(c) = self.console.as_ref() {
            c.skipped();
        }
        nap(self.idle());
        tick
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
            let rgb: Vec<u8> = match components {
                4 => data
                    .chunks_exact(4)
                    .flat_map(|p| &p[..3])
                    .copied()
                    .collect(),
                _ => data.to_vec(),
            };
            encode_jpeg(
                Frame {
                    data: &rgb,
                    width,
                    height,
                    components: 3,
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

    /// Set one live param, with no recompile. A string param of a text's name
    /// lays the text out again.
    pub fn set_param(&mut self, pass: Option<&str>, name: &str, value: &Value) -> Outcome<()> {
        let p = self.index.param(pass, name)?;
        let v = coerce(&p.wire, &p.ty, value).map_err(scene_fault)?;
        // A string param fails on a text texture or quads that the engine
        // refuses, and both report as scene faults.
        let fault = match p.ty.as_str() {
            "string" => scene_fault,
            _ => system_fault,
        };
        RenderTarget::set_param(&self.renderer, p.pass.as_deref(), &p.uniform, &p.ty, &v)
            .map_err(fault)?;
        // The host wrote these words itself, so changed_params() leaves them
        // out.
        if let ParamValue::Str(words) = &v {
            let pass = p.pass.clone();
            self.held_string(pass.as_deref(), &p.uniform.clone(), words, true);
        }
        Ok(())
    }

    /// Hold `words` for the string param `name` of `pass`, under its bare
    /// name when it has one, and as already handed out when `sent`.
    fn held_string(&mut self, pass: Option<&str>, name: &str, words: &str, sent: bool) {
        let bare = self
            .index
            .params
            .iter()
            .any(|r| r.wire == name && r.pass.as_deref() == pass);
        if bare {
            self.strings.insert(name.to_string(), Value::from(words));
            if sent {
                self.sent.insert(name.to_string(), Value::from(words));
            }
        }
    }

    /// The string params that changed since the last call, by their bare
    /// names, with their words: the console's sets, which this applies first.
    /// The first call holds every string param, as the first tick of the
    /// wire does.
    pub fn changed_params(&mut self) -> Outcome<Map<String, Value>> {
        self.drain()?;
        let mut out = Map::new();
        for (k, v) in &self.strings {
            if self.sent.get(k) != Some(v) {
                out.insert(k.clone(), v.clone());
            }
        }
        self.sent.clone_from(&self.strings);
        Ok(out)
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

    /// The params queued by a console, applied before this frame draws.
    fn drain(&mut self) -> Outcome<()> {
        let Some(console) = self.console.as_ref() else {
            return Ok(());
        };
        for (pass, name, ty, value) in console.take_params() {
            RenderTarget::set_param(&self.renderer, pass.as_deref(), &name, &ty, &value)
                .map_err(system_fault)?;
            if let ParamValue::Str(words) = &value {
                self.held_string(pass.as_deref(), &name, words, false);
            }
        }
        Ok(())
    }

    /// Blank the panel, turn it off, and drop everything the session holds.
    pub fn close(&mut self) -> Outcome<()> {
        if let Some(c) = self.console.take() {
            c.stop();
        }
        // the encoder holds engine resources, so it goes before the renderer
        if let Some(mut enc) = self.h264.take() {
            enc.close();
        }
        let mut closing = Ok(());
        if let Sink::Panel(d) = &mut self.sink {
            for step in [
                DisplayDriver::clear,
                DisplayDriver::turn_off,
                DisplayDriver::close,
            ] {
                if let Err(e) = step(d.as_mut()) {
                    closing = Err(system_fault(e));
                }
            }
        }
        self.sink = Sink::None;
        self.renderer.close();
        closing
    }
}

fn nap(seconds: f64) {
    if seconds > 0.0 {
        sleep(Duration::from_secs_f64(seconds));
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
