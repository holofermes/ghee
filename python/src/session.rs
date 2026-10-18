//! One scene, one renderer, one output. The host asks for each frame.
//!
//! The scene loading, the console and the frame clock, ghee::looper::Clock,
//! are ghee's own. `t` counts from the first frame.
//!
//! The engine is thread-affine, so a Session belongs to the thread that built
//! it.

use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::Duration;

use ghee::console::hub::{frame_format, gpu_caps, HwLane, RawFrame};
use ghee::draw::Call;
use ghee::engine::{quarter_turns, Card, Feed, Given, Refused, Renderer};
use ghee::filters::apply_filters;
use ghee::layer::Address;
use ghee::live::Live;
use ghee::looper::{Clock, Frame, Moment, Pacer, RenderTarget};
use ghee::output::{encode_jpeg, encode_png_as, expand_pattern, AlphaForm, FixedPacer};
use ghee::procedural::{load_scene, process, scene_name};
use ghee::scene::{coerce, truthy, ArraySize, Scene};
use ghee::stats::{monotonic, DEST_KMS};
use serde_json::{Map, Value};

use crate::serve::{Console, Serving};

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

impl From<Refused> for Fault {
    fn from(refused: Refused) -> Fault {
        Fault::scene(refused.0)
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

/// An iteration that begin() opened: its moment and its tick.
struct Opened {
    moment: Moment,
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
    /// Keep the output pass's alpha in the frames, in this form. None keeps
    /// every frame opaque.
    pub alpha: Option<AlphaForm>,
    /// The display skips to the newest frame: Renderer::set_drop.
    pub drop: bool,
}

pub struct Session {
    renderer: Renderer,
    sink: Sink,
    /// The scene's live state: what it declares, what the host set and what
    /// a console changed.
    live: Live,
    clock: Clock<'static>,
    console: Option<Console>,
    /// The console's hardware H264 lane. It issues GL calls, so it belongs to
    /// this thread, and it never outlives the renderer.
    h264: HwLane,
    pending: Vec<(String, Fed)>,
    buf: Vec<u8>,
    alpha: Option<AlphaForm>,
    frame: u64,
    /// The tick of the frame drawn last, None before the first.
    drawn: Option<u64>,
    /// The renderer's text_errors_seq() at the console's last look.
    text_errors: u64,
    /// The iteration that begin() opened, for the next frame() or skip().
    opened: Option<Opened>,
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
    apply_filters(&mut scene, &wanted, None).map_err(scene_fault)?;
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

/// The card that display_size() opened, under the `dri` that it was asked
/// for, held for the open() that follows.
static PROBED: Mutex<Option<(Option<String>, Card)>> = Mutex::new(None);

/// The mode size of the display that open() with display="kms" and `dri`
/// draws on. Its card stays open for that open().
pub fn display_size(dri: Option<&str>) -> Option<(i32, i32)> {
    let mut held = PROBED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((asked, card)) = held.as_ref() {
        if asked.as_deref() == dri {
            return Some(card.size());
        }
    }
    let card = Card::open(dri)?;
    let size = card.size();
    *held = Some((dri.map(str::to_string), card));
    Some(size)
}

/// The card of a display="kms" open on `dri`: the one that display_size()
/// holds for it, else the card opened now.
fn display_card(dri: Option<&str>) -> Option<Card> {
    let held = PROBED.lock().unwrap_or_else(|e| e.into_inner()).take();
    match held {
        Some((asked, card)) if asked.as_deref() == dri => Some(card),
        _ => Card::open(dri),
    }
}

/// (size, components) of the frame. A connected display on `card` states
/// its mode, which wins over the host's own size.
fn geometry(
    sink: &Sink,
    size: Option<(i32, i32)>,
    card: Option<&Card>,
) -> Outcome<((i32, i32), i32)> {
    match sink {
        Sink::Kms => match card.map(Card::size).or(size) {
            Some(size) => Ok((size, 4)),
            None => Err(Fault::system(
                "no connected display to size from; pass width and height",
            )),
        },
        Sink::None => Ok((size.unwrap_or(DEFAULT_SIZE), 4)),
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
            drop,
        } = opening;
        quarter_turns(rotate).map_err(scene_fault)?;
        let (scene, name) = load(source)?;
        let sink = open_sink(display.as_deref(), &opts)?;
        if drop && matches!(sink, Sink::None) {
            return Err(Fault::scene("drop without display"));
        }
        let card = match sink {
            Sink::Kms => display_card(dri.as_deref()),
            Sink::None => None,
        };
        if let Some(card) = &card {
            card.check_renders().map_err(system_fault)?;
        }
        let ((w, h), components) = geometry(&sink, size, card.as_ref())?;
        let kms = matches!(sink, Sink::Kms);
        let rgb565 = opts
            .iter()
            .find(|(k, _)| k == "rgb565")
            .is_some_and(|(_, v)| truthy(v));
        let mut renderer = match card {
            Some(card) => Renderer::on_card(card, w, h, rgb565),
            None => Renderer::new(w, h, components, dri.as_deref(), kms, rgb565),
        }
        .map_err(system_fault)?;
        if drop {
            renderer.set_drop(true).map_err(system_fault)?;
        }
        renderer.set_alpha(alpha.is_some());
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
        let live = Live::new(&scene);
        let out = match &sink {
            Sink::None => Vec::new(),
            Sink::Kms => vec![DEST_KMS.to_string()],
        };
        let format = frame_format(rgb565, components);
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
                    Serving {
                        name: &name,
                        scene: &scene,
                        index: Arc::clone(live.index()),
                        locks: live.locks(),
                        size: (w, h),
                        format,
                        rotate,
                        gpu: gpu_caps(&renderer),
                        out,
                        stream_max: longest,
                        verbose,
                    },
                )
                .map_err(system_fault)?,
            ),
        };
        let mut clock = Clock::new(timestep);
        clock.pacer = fps
            .filter(|f| *f != 0.0)
            .map(|f| Box::new(FixedPacer::new(f)) as Box<dyn Pacer>);
        Ok(Session {
            renderer,
            sink,
            live,
            clock,
            console,
            h264: HwLane::default(),
            pending: Vec::new(),
            buf: vec![0u8; (i64::from(h) * i64::from(w) * i64::from(components)) as usize],
            alpha,
            frame: 0,
            drawn: None,
            text_errors: u64::MAX,
            opened: None,
        })
    }

    pub fn size(&self) -> (i32, i32) {
        (self.renderer.width, self.renderer.height)
    }

    pub fn components(&self) -> i32 {
        self.renderer.components
    }

    /// The form that the frames' kept alpha takes on the way out, None when
    /// every frame is opaque.
    pub fn alpha(&self) -> Option<AlphaForm> {
        self.alpha
    }

    /// The GL that renders, GL_RENDERER and GL_VERSION in one string.
    pub fn gl(&self) -> String {
        self.renderer.renderer()
    }

    /// Open this iteration, once: apply the console's sets, and give the tick
    /// that the next frame() or skip() takes and returns, as Clock::begin()
    /// times it.
    pub fn begin(&mut self) -> Outcome<Tick> {
        if let Some(o) = &self.opened {
            return Ok(o.tick.clone());
        }
        self.drain()?;
        let params = self.live.tick_params();
        let moment = self.clock.begin(self.frame, monotonic());
        let tick = Tick {
            frame: self.frame,
            time: moment.t,
            dt: moment.dt,
            params,
        };
        self.opened = Some(Opened {
            moment,
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

    pub fn frame(&mut self) -> Outcome<Tick> {
        let Opened { moment, tick } = self.take_opened()?;
        self.drain()?;
        self.clock
            .drawn(&moment, &self.renderer as &dyn RenderTarget);
        if self.drawn.is_none() {
            self.renderer.wait_for_words().map_err(system_fault)?;
        }
        let hub = self.console.as_ref().map(Console::hub);
        let shot = hub.is_some_and(|h| h.shot.wants());
        let preview = match hub {
            Some(h) if h.wants_preview() => Some(h.preview_size()),
            _ => None,
        };
        let want = if shot { Some(self.size()) } else { preview };
        let (drew, read) = self.draw(want)?;
        if let Some(hub) = self.console.as_ref().map(Console::hub) {
            self.h264.mirror(hub, &self.renderer);
        }
        let components = self.renderer.components;
        self.drawn = Some(self.frame);
        self.frame += 1;
        let end = monotonic();
        if let Some(hub) = self.console.as_ref().map(Console::hub) {
            hub.frame_done(end - moment.t0);
            let seq = self.renderer.text_errors_seq();
            if seq != self.text_errors {
                self.text_errors = seq;
                hub.set_text_errors(self.renderer.text_errors());
            }
            if let Some((width, height)) = read.filter(|_| want.is_some()) {
                let frame = Frame {
                    data: &self.buf[..(width * height * components) as usize],
                    width,
                    height,
                    components,
                };
                match shot {
                    true => hub.shot.write(Arc::new(RawFrame::of(frame))),
                    false => hub.publish_frame(frame),
                }
            }
        }
        self.clock.pace(&moment, drew);
        Ok(tick)
    }

    /// Render the passes and hand the frame to the output. A display reads
    /// the frame back at the size that `want` asks, and a render with no
    /// display reads back the whole frame. Returns the clock at the end of the
    /// render, and the size of the frame read back.
    fn draw(&mut self, want: Option<(i32, i32)>) -> Outcome<(f64, Option<(i32, i32)>)> {
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
                let preview = want.map(|size| (&mut buf[..], size));
                renderer
                    .present(Some(&mut provider), preview)
                    .map_err(system_fault)?;
                Ok((monotonic(), want))
            }
            Sink::None => {
                renderer
                    .render_into(Some(&mut provider), buf, None)
                    .map_err(system_fault)?;
                Ok((monotonic(), Some((w, h))))
            }
        }
    }

    /// This tick needs no render. The display keeps its last frame. A console
    /// reader still gets one: while a viewer wants a frame, the skip draws, as
    /// the CLI queues a preview for its console. A preview draw that fails
    /// falls back to a plain skip, and the host's own frame() reports it.
    pub fn skip(&mut self) -> Outcome<Tick> {
        let tick = self.begin()?;
        let wants = self.console.as_ref().is_some_and(|c| c.hub().wants_draw());
        if wants && self.frame().is_ok() {
            if self.clock.pacer.is_none() {
                nap(self.clock.idle());
            }
            return Ok(tick);
        }
        self.opened = None;
        self.frame += 1;
        if let Some(c) = self.console.as_ref() {
            c.hub().skipped();
        }
        nap(self.clock.idle());
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
            encode_png_as(
                Frame {
                    data,
                    width,
                    height,
                    components,
                },
                self.alpha.is_some_and(AlphaForm::image_premultiplied),
            )
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
        let p = self.live.index().param(pass, name)?.clone();
        let v = coerce(&p.wire, &p.ty, value).map_err(scene_fault)?;
        self.live
            .set_param(&self.renderer, &p, v)
            .map_err(system_fault)
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
        self.live.index().layer(layer)?;
        let at = Address {
            layer,
            effect,
            field,
        };
        self.live
            .set_layer(&self.renderer, &at, Given::Json(value))
            .map_err(refusal)
    }

    /// Add `layer`, a layer as a scene declares it, at `index` among the
    /// layers of pass `pass`, with no rebuild. Its words, when it has them,
    /// are the host's own, and so are its keys.
    pub fn add_layer(&mut self, pass: &str, index: usize, layer: &Value) -> Outcome<()> {
        self.live
            .add_layer(&mut self.renderer, pass, index, layer)
            .map_err(refusal)
    }

    /// Take layer `name` out of the scene, with no rebuild.
    pub fn remove_layer(&mut self, name: &str) -> Outcome<()> {
        self.live
            .remove_layer(&mut self.renderer, name)
            .map_err(refusal)
    }

    /// Put layer `name` at `index` among the layers of its pass, with no
    /// rebuild.
    pub fn move_layer(&mut self, name: &str, index: usize) -> Outcome<()> {
        self.live
            .move_layer(&mut self.renderer, name, index)
            .map_err(refusal)
    }

    /// Give the layer of the name of `layer` its look, as Live::update_layer()
    /// gives it. False when the layer had that look already.
    pub fn update_layer(&mut self, layer: &Value) -> Outcome<bool> {
        self.live
            .update_layer(&mut self.renderer, layer)
            .map_err(refusal)
    }

    /// Give the layers of `calls` their looks, as Live::update_calls() gives
    /// them. Returns the indexes of the calls left for update_layer().
    pub fn update_calls(&mut self, calls: &[Call]) -> Outcome<Vec<usize>> {
        self.live
            .update_calls(&mut self.renderer, calls)
            .map_err(refusal)
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
        font_family: &[&str],
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

    /// Hold data for the next frame. That frame consumes it once.
    pub fn feed(&mut self, name: &str, data: Fed) -> Outcome<()> {
        let want = array_count(self.live.index().array(name)?);
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
    /// fields and the looks of applies. A still that /frame.png asked for
    /// before them takes the next frame. The console then takes the scene,
    /// when it changed.
    fn drain(&mut self) -> Outcome<()> {
        let Some(console) = self.console.as_ref() else {
            return Ok(());
        };
        console.hub().shot.arm();
        let layers = console.take_layers();
        let looks = console.take_looks();
        let params = console.take_params();
        let edited = !(layers.is_empty() && looks.is_empty() && params.is_empty());
        for (pass, name, ty, value) in params {
            self.live
                .console_param(&self.renderer, pass.as_deref(), &name, &ty, &value)
                .map_err(system_fault)?;
        }
        for set in layers {
            self.live
                .console_layer(&self.renderer, &set.at(), &set.value)
                .map_err(system_fault)?;
        }
        let applied = !looks.is_empty();
        let mut failed = Vec::new();
        for look in looks {
            let look = self.live.console_look(&self.renderer, look);
            match self.renderer.replace_layer(&look) {
                Ok(()) => self.live.took(&look),
                Err(e) => failed.push(e.to_string()),
            }
        }
        if let Some(c) = self.console.as_ref() {
            if applied {
                c.hub().set_apply_error(&failed.join("\n"));
            }
            c.publish(&self.renderer, &mut self.live, edited);
        }
        Ok(())
    }

    /// Drop everything the session holds.
    pub fn close(&mut self) {
        if let Some(c) = self.console.take() {
            c.stop();
        }
        // the encoder holds engine resources, so it goes before the renderer
        self.h264.close();
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
