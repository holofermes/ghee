//! The `ghee` command line.
//!
//! ```text
//! ghee render scene.json -o frame.####.jpg --frames 500  # image sequence
//! ghee render scene.json -o - --frames 300 | ffmpeg ...  # raw frames to a pipe
//! ghee render scene.json --display st7789                # to a panel, until Ctrl-C
//! ghee render scene.json --h264 - | mpv -                # H264 to a pipe
//! ghee render scene.json --display st7789 --serve        # panel plus the console
//! ghee serve scenes/                                     # editor console
//! ghee serve --demo                                      # console on the bundled examples
//! ghee render --demo bars -o frame.png                   # a bundled example by name
//! ghee displays                                          # list display drivers
//! ghee overlay st7789                                     # kernel setup for a panel
//! ```

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::c_int;
use std::fs::File;
use std::io::{stdout, Stdout, Write};
use std::path::Path;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use anyhow::{bail, Context, Result};
use clap::error::{ContextKind, ErrorKind};
use clap::{Args, Parser, Subcommand};
use serde_json::{Map, Value};

use crate::bench::{self, BenchArgs};
use crate::console::examples_dir;
use crate::console::server;
use crate::console::store::SceneStore;
use crate::drivers::{available_drivers, driver_options, get_driver, DisplayDriver};
use crate::engine::{display_renders, probe_display, probe_display_device, Feed, Renderer};
use crate::feeds::demo_feed;
use crate::filters::{apply_filters, load_filters};
use crate::h264::H264Stream;
use crate::looper::{ControllerSlot, Frame, Loop, Output, OwnedFrame, Pacer, SceneController};
use crate::output::{expand_pattern, FixedPacer, ImageSequence, RawStream, Scanout};
use crate::overlay;
use crate::procedural::process::{self, SceneProcess};
use crate::procedural::{load_scene, scene_name};
use crate::report;
use crate::scene::{Reach, Scene};
use crate::stats::rate;
use crate::usage;

/// main() prints this message as written, status 1. main() puts the command
/// name in front of any other error.
#[derive(Debug)]
struct Exit(String);

impl std::fmt::Display for Exit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Exit {}

fn exit(message: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(Exit(message.into()))
}

/// True when an optional string flag carries text. Unset and "" are false.
fn given(v: &Option<String>) -> bool {
    v.as_deref().is_some_and(|s| !s.is_empty())
}

fn text(v: &Option<String>) -> &str {
    v.as_deref().unwrap_or_default()
}

/// --rotate's degrees, one of the turns that a renderer takes.
fn rotation(s: &str) -> Result<i64, String> {
    match s.trim().parse::<i64>() {
        Ok(d) if crate::engine::ROTATIONS.contains(&d) => Ok(d),
        _ => Err(format!("'{s}' is not 0, 90, 180 or 270")),
    }
}

/// --brightness's level, 0 to 1.
fn level(s: &str) -> Result<f64, String> {
    match s.trim().parse::<f64>() {
        Ok(v) if (0.0..=1.0).contains(&v) => Ok(v),
        _ => Err(format!("'{s}' is not a level from 0 to 1")),
    }
}

fn size(s: &str) -> Result<(i32, i32), String> {
    let bad = || format!("invalid _size value: '{s}'");
    let lower = s.to_lowercase();
    let mut parts = lower.split('x');
    let (Some(w), Some(h), None) = (parts.next(), parts.next(), parts.next()) else {
        return Err(bad());
    };
    match (w.trim().parse::<i32>(), h.trim().parse::<i32>()) {
        (Ok(w), Ok(h)) => Ok((w, h)),
        _ => Err(bad()),
    }
}

/// What a k=v value holds: a number, True/False/None, or a quoted string.
/// Anything else gives None, and the caller keeps it as text.
fn literal_eval(s: &str) -> Option<Value> {
    match s {
        "True" => return Some(Value::Bool(true)),
        "False" => return Some(Value::Bool(false)),
        "None" => return Some(Value::Null),
        _ => {}
    }
    if let Some(quote) = s.chars().next().filter(|c| *c == '\'' || *c == '"') {
        let inner = s.get(1..s.len().saturating_sub(1))?;
        return match s.len() >= 2 && s.ends_with(quote) && !inner.contains(quote) {
            true => Some(Value::from(inner)),
            false => None,
        };
    }
    // An integer literal takes a sign, a base prefix and _ separators.
    // serde_json already reads a float literal.
    let body = s.strip_prefix(['+', '-']).unwrap_or(s);
    let (digits, radix) = match body.get(..2).map(str::to_lowercase).as_deref() {
        Some("0x") => (&body[2..], 16),
        Some("0o") => (&body[2..], 8),
        Some("0b") => (&body[2..], 2),
        _ => (body, 10),
    };
    let plain = digits.replace('_', "");
    if !plain.is_empty() && plain.bytes().all(|b| b.is_ascii_alphanumeric()) {
        if let Ok(n) = i64::from_str_radix(&plain, radix) {
            return Some(Value::from(if s.starts_with('-') { -n } else { n }));
        }
    }
    if radix != 10 {
        return None;
    }
    serde_json::from_str::<Value>(s)
        .ok()
        .filter(Value::is_number)
}

/// "k=v,k2=v2" -> a map. Each value goes through literal_eval (90 -> int, True
/// -> bool), and literal_eval leaves a value it refuses as the raw string.
/// `what` names the flag behind an error.
fn parse_kv(s: &str, what: &str) -> Result<Map<String, Value>> {
    let mut opts = Map::new();
    for pair in s.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let Some((k, v)) = pair.split_once('=') else {
            return Err(exit(format!("{what} wants k=v pairs, got '{pair}'")));
        };
        let v = v.trim();
        opts.insert(
            k.trim().to_string(),
            literal_eval(v).unwrap_or_else(|| Value::from(v)),
        );
    }
    Ok(opts)
}

/// (ghee's own argv, the scene's tokens). The first `--` ends ghee's flags.
/// Everything past it belongs to the scene, further `--` included.
fn split_scene_args(argv: &[String]) -> (Vec<String>, Vec<String>) {
    match argv.iter().position(|a| a == "--") {
        None => (argv.to_vec(), Vec::new()),
        Some(i) => (argv[..i].to_vec(), argv[i + 1..].to_vec()),
    }
}

/// dir joined with name. An absolute name replaces the directory.
fn join(dir: &str, name: &str) -> String {
    if name.starts_with('/') || dir.is_empty() {
        return name.to_string();
    }
    if dir.ends_with('/') {
        format!("{dir}{name}")
    } else {
        format!("{dir}/{name}")
    }
}

/// A path as given, or a scene name from $GHEE_SCENES (colon-separated
/// directories). --demo resolves from the bundled examples alone. No file in
/// the working directory shadows a demo name. A process is a file like any
/// other, so the name resolves to the executable itself.
fn find_scene(arg: &str, demo: bool) -> Result<String> {
    fn hit(stem: &str) -> bool {
        Path::new(stem).exists()
            || Path::new(&format!("{stem}{}", crate::procedural::SUFFIX)).exists()
            || crate::procedural::has_executable_member(stem)
    }
    if demo {
        let Some(dir) = examples_dir() else {
            return Err(exit("no demo scenes: no writable cache location"));
        };
        let p = join(&dir, arg);
        if hit(&p) {
            return Ok(p);
        }
        return Err(exit(format!("no demo scene '{arg}'")));
    }
    if hit(arg) {
        return Ok(arg.to_string());
    }
    let env = std::env::var("GHEE_SCENES").unwrap_or_default();
    let dirs: Vec<String> = env
        .split(':')
        .filter(|d| !d.is_empty())
        .map(String::from)
        .collect();
    for d in &dirs {
        let p = join(d, arg);
        if hit(&p) {
            return Ok(p);
        }
    }
    let searched: Vec<&str> = std::iter::once("cwd")
        .chain(dirs.iter().map(String::as_str))
        .collect();
    Err(exit(format!(
        "no scene '{arg}' (searched: {})",
        searched.join(", ")
    )))
}

/// (name, path) for each scene loaded here: one for a render, and every scene
/// in the console's directory.
fn scene_set(a: &Ns) -> Result<Vec<(String, String)>> {
    let one = |path: String| vec![(scene_name(&path), path)];
    if !a.serve {
        return Ok(one(find_scene(text(&a.scene), a.demo)?));
    }
    let dir = match &a.scene {
        Some(scene) => find_scene(scene, a.demo)?,
        None => examples_dir().unwrap_or_default(),
    };
    if !Path::new(&dir).is_dir() {
        return Ok(one(dir));
    }
    Ok(SceneStore::new(&dir, Vec::new()).scenes)
}

/// `-- --help` lists what each scene takes, grouped per scene. This runs each
/// scene with `--ghee-args` to ask, and nothing else in ghee ever does. Any
/// other token goes to the scene untouched.
fn scene_args_help(a: &Ns) -> Result<()> {
    if !process::wants_help(&a.scene_args) {
        return Ok(());
    }
    let set = scene_set(a)?;
    let prog = match a.cmd {
        "serve" => "ghee serve --".to_string(),
        _ => format!("ghee render {} --", set.first().map_or("", |(n, _)| n)),
    };
    print!(
        "{}",
        process::help(&process::declare(&set, !a.no_probe), &prog)
    );
    std::process::exit(0);
}

/// The scene of this run, and the process behind it. The filters apply in
/// order: the scene's own first, then --filters.
fn scene_of(a: &Ns) -> Result<(Scene, Option<SceneProcess>)> {
    let path = find_scene(text(&a.scene), a.demo)?;
    report::detail("scene", &path);
    let (mut scene, mut process) = match process::executable(&path) {
        Some(exe) => {
            let (scene, fed) = SceneProcess::start(&exe, &a.scene_args, Reach::Any, |_| {})?;
            (scene, Some(fed))
        }
        None => (load_scene(&path, &a.scene_args, true)?, None),
    };
    let mut names = scene.filters.clone();
    names.extend(
        text(&a.filters)
            .split(',')
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .map(String::from),
    );
    let mut wanted: Vec<String> = Vec::new();
    for n in names {
        if !wanted.contains(&n) {
            wanted.push(n);
        }
    }
    let reg = load_filters(&[]);
    let unknown: Vec<&str> = wanted
        .iter()
        .filter(|n| !reg.contains_key(n.as_str()))
        .map(String::as_str)
        .collect();
    if !unknown.is_empty() {
        return Err(exit(format!(
            "unknown filter: {} (known: {})",
            unknown.join(", "),
            reg.keys().cloned().collect::<Vec<_>>().join(", ")
        )));
    }
    apply_filters(&mut scene, &wanted, Some(&reg))?;
    // the process addresses the params of the final scene, filters included
    if let Some(fed) = process.as_mut() {
        fed.bind(&scene);
        if a.timestep.is_some() {
            fed.pin();
        }
    }
    Ok((scene, process))
}

/// What --display names: the KMS scanout, or the driver instance.
/// --display-opts builds the driver.
enum Panel {
    None,
    Kms,
    Driver(Box<dyn DisplayDriver + Send>),
}

/// Panel::Kms, after checking GL on the card runs on hardware. A display-only
/// card (a panel under a kernel display driver) still gets hardware GL: Mesa
/// pairs it with the render node and imports each frame back for the flip.
/// Software GL here therefore means no usable render node at all, and ghee
/// stops rather than render through llvmpipe. With no connected display this
/// stays Panel::Kms, and geometry() reports that with its own error.
fn kms_panel(a: &Ns) -> Result<Panel> {
    let Some((dev, _, _)) = probe_display_device(a.dri.as_deref()) else {
        return Ok(Panel::Kms);
    };
    if !display_renders(&dev) {
        return Err(exit(format!(
            "{}: GL on {dev} runs in software; no hardware render node pairs with this card",
            a.cmd
        )));
    }
    Ok(Panel::Kms)
}

fn panel_of(a: &Ns) -> Result<Panel> {
    if !given(&a.display) {
        if given(&a.display_opts) {
            return Err(exit(format!("{}: --display-opts without --display", a.cmd)));
        }
        return Ok(Panel::None);
    }
    let name = text(&a.display);
    if name == "kms" {
        if given(&a.display_opts) {
            return Err(exit(format!(
                "{}: --display-opts with --display kms; kms has no driver",
                a.cmd
            )));
        }
        return kms_panel(a);
    }
    let Ok(spec) = get_driver(name) else {
        // --display takes an optional value, so it reads a path after it as a
        // driver name
        let hint = if Path::new(name).is_dir() {
            ", and a scenes directory goes before it"
        } else {
            ""
        };
        return Err(exit(format!(
            "{}: no display driver '{name}' (see: ghee displays){hint}",
            a.cmd
        )));
    };
    if !given(&a.display_opts) {
        return Ok(Panel::Driver((spec.open)(&[])?));
    }
    let opts = parse_kv(text(&a.display_opts), &format!("{}: --display-opts", a.cmd))?;
    // check before the driver runs: a constructor claims the GPIO lines and
    // the SPI bus first. It reaches a keyword it does not take after that.
    let known = driver_options(spec);
    let unknown: Vec<&str> = opts
        .keys()
        .filter(|k| !known.iter().any(|o| o.name == k.as_str()))
        .map(String::as_str)
        .collect();
    if !unknown.is_empty() {
        return Err(exit(format!(
            "{}: --display-opts: {name} has no option {} (takes: {})",
            a.cmd,
            unknown.join(", "),
            known.iter().map(|o| o.name).collect::<Vec<_>>().join(", ")
        )));
    }
    let pairs: Vec<(String, Value)> = opts.into_iter().collect();
    Ok(Panel::Driver((spec.open)(&pairs)?))
}

/// A scene builds against this size, worked out before any device opens.
/// geometry() gives the same answer once the panel is up, and a scene needs it
/// earlier than that: it sizes its own buffers as it builds, so `half` and
/// `quarter` have to mean something already. Reading a mode or a driver's
/// declared width costs nothing and takes no DRM master, so asking early is
/// free. None where the size cannot be known without opening the device, and a
/// scene that wanted a fraction then says so itself.
fn prospective_size(a: &Ns) -> Option<(i32, i32)> {
    if !given(&a.display) {
        return Some(a.size.unwrap_or(DEFAULT_SIZE));
    }
    let name = text(&a.display);
    if name == "kms" {
        return a.size.or_else(|| probe_display(a.dri.as_deref()));
    }
    let spec = get_driver(name).ok()?;
    let opts = parse_kv(text(&a.display_opts), "--display-opts").unwrap_or_default();
    let pairs: Vec<(String, Value)> = opts.into_iter().collect();
    (spec.size)(&pairs).ok()
}

/// (size, components) of the render.
/// A scanout takes --size and scales it across the mode on the way out, or the
/// mode itself when there is no --size. A panel takes its own geometry and
/// pixel layout, and its wire carries no scaler, so --size means nothing there.
/// Everything else takes --size.
fn geometry(a: &Ns, panel: &Panel) -> Result<((i32, i32), i32)> {
    match panel {
        // ghee_display_create renders RGBA whatever --components asks for, so
        // the readback writes four bytes a pixel.
        Panel::Kms => match a.size.or_else(|| probe_display(a.dri.as_deref())) {
            Some(size) => Ok((size, 4)),
            None => Err(exit(format!(
                "{}: no connected display to size from, pass --size WxH",
                a.cmd
            ))),
        },
        Panel::Driver(d) => match a.size {
            Some(_) => Err(exit(format!(
                "{}: --display {} is fixed at {}x{}, drop --size",
                a.cmd,
                text(&a.display),
                d.width(),
                d.height()
            ))),
            None => Ok(((d.width(), d.height()), d.components())),
        },
        Panel::None => Ok((a.size.unwrap_or(DEFAULT_SIZE), a.components)),
    }
}

/// The -o sink: raw frames to stdout, or numbered image files.
enum FileSink {
    None,
    Raw(RawStream<Stdout>),
    Images(ImageSequence),
}

fn file_sink(a: &Ns) -> Result<FileSink> {
    if text(&a.output) == "-" {
        return Ok(FileSink::Raw(RawStream::new(stdout())));
    }
    if !given(&a.output) {
        return Ok(FileSink::None);
    }
    let pattern = text(&a.output);
    if a.frames.is_some_and(|n| n > 1) && expand_pattern(pattern, 0)? == pattern {
        return Err(exit(
            "render: multiple frames into one filename; number them with \
             a # run (frame.####.png)",
        ));
    }
    let save_opts = match given(&a.save_opts) {
        true => Some(parse_kv(
            text(&a.save_opts),
            &format!("{}: --save-opts", a.cmd),
        )?),
        false => None,
    };
    Ok(FileSink::Images(ImageSequence::with_opts(
        pattern,
        0,
        save_opts,
        usize::try_from(a.write_buffer).unwrap_or(0),
    )))
}

/// Nothing paces a file or a pipe: that render runs as fast as it draws. A
/// panel sink blocks acquire() when the wire is a frame behind, so the bus
/// paces an unpaced loop. A KMS present waits out the display refresh. An
/// H264 stream holds a fixed rate for its timestamps.
fn pacer(a: &Ns, panel: &Panel) -> Option<Box<dyn Pacer>> {
    let fixed = a
        .fps
        .filter(|f| *f != 0.0)
        .map(|f| Box::new(FixedPacer::new(f)) as Box<dyn Pacer>);
    match panel {
        Panel::Kms | Panel::Driver(_) => fixed,
        Panel::None if given(&a.h264) => Some(Box::new(FixedPacer::new(f64::from(h264_fps(a))))),
        Panel::None => None,
    }
}

fn h264_fps(a: &Ns) -> i32 {
    match a.fps.filter(|f| *f != 0.0) {
        Some(f) => f as i32,
        None => 30,
    }
}

/// --frames as given. Without it an image render draws one frame, and a
/// display, a stream or the console runs endlessly.
fn frames_of(a: &Ns) -> i64 {
    match a.frames {
        Some(n) => n,
        None => i64::from(!(given(&a.display) || given(&a.h264) || a.serve)),
    }
}

enum PanelJob {
    Frame(OwnedFrame),
    On,
    Off,
    Clear,
    Stop,
}

/// A driver writes to a physical panel (drivers::base) over a bus that takes
/// milliseconds a frame. The loop reaches a driver through this Output. The
/// bus write runs on the sink's own thread, so the loop renders frame N+1
/// while the wire carries frame N.
///
/// The pool holds two buffers: one on the wire, one under the render.
/// acquire() blocks when the wire is a full frame behind, so an unpaced loop
/// settles at the bus rate, and no frame is dropped.
struct PanelSink {
    dest: &'static str,
    persists: bool,
    native_format: &'static str,
    allocated: usize,
    jobs: Sender<PanelJob>,
    free: Receiver<OwnedFrame>,
    recycle: Sender<OwnedFrame>,
    err: Arc<Mutex<Option<String>>>,
    worker: Option<JoinHandle<()>>,
}

impl PanelSink {
    fn new(driver: Box<dyn DisplayDriver + Send>) -> PanelSink {
        let (jobs, inbox) = channel::<PanelJob>();
        let (recycle, free) = channel::<OwnedFrame>();
        let err = Arc::new(Mutex::new(None));
        let (dest, persists, native_format) =
            (driver.dest(), driver.persists(), driver.native_format());
        let worker = {
            let (recycle, err) = (recycle.clone(), Arc::clone(&err));
            std::thread::spawn(move || run_panel(driver, &inbox, &recycle, &err))
        };
        PanelSink {
            dest,
            persists,
            native_format,
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
        let _ = self.jobs.send(PanelJob::Stop);
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                bail!("the panel writer thread died");
            }
        }
        match self.failure() {
            Some(e) => bail!(e),
            None => Ok(()),
        }
    }
}

/// The panel thread: write, recycle. The buffer belongs to the worker alone
/// between submit and recycle, so the driver reads it in place. After a
/// failure the worker keeps draining and recycling, so a blocked acquire()
/// never starves. The error reaches the caller on the next acquire(), the
/// next submit(), or on close().
fn run_panel(
    mut driver: Box<dyn DisplayDriver + Send>,
    inbox: &Receiver<PanelJob>,
    recycle: &Sender<OwnedFrame>,
    err: &Arc<Mutex<Option<String>>>,
) {
    let note = |r: Result<()>| {
        if let Err(e) = r {
            let mut slot = err.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() {
                *slot = Some(format!("{e:#}"));
            }
        }
    };
    loop {
        match inbox.recv() {
            Ok(PanelJob::Frame(buf)) => {
                // the write never holds the lock, because acquire() reads it
                // every frame from the render thread
                let failed = err.lock().unwrap_or_else(|e| e.into_inner()).is_some();
                if !failed {
                    note(driver.write(&buf.data));
                }
                let _ = recycle.send(buf);
            }
            Ok(PanelJob::On) => note(driver.turn_on()),
            Ok(PanelJob::Off) => note(driver.turn_off()),
            Ok(PanelJob::Clear) => note(driver.clear()),
            Ok(PanelJob::Stop) | Err(_) => break,
        }
    }
    note(driver.close());
}

impl Output for PanelSink {
    fn dest(&self) -> &str {
        self.dest
    }

    fn persists(&self) -> bool {
        self.persists
    }

    fn pooled(&self) -> bool {
        true
    }

    /// A buffer of `shape` to render into. It takes the recycled buffer
    /// first. It grows the pool to two next. It blocks for the recycled
    /// buffer last, which is the render's only wait. Single-producer: only
    /// the render loop calls it.
    fn acquire(&mut self, shape: (i32, i32, i32)) -> Result<OwnedFrame> {
        if let Some(e) = self.failure() {
            bail!(e);
        }
        if let Ok(buf) = self.free.try_recv() {
            return Ok(buf);
        }
        if self.allocated < 2 {
            self.allocated += 1;
            return Ok(OwnedFrame::of(shape));
        }
        Ok(self.free.recv()?)
    }

    /// Queue a filled buffer for the wire.
    fn submit(&mut self, buf: OwnedFrame) -> Result<()> {
        if let Some(e) = self.failure() {
            let _ = self.recycle.send(buf);
            bail!(e);
        }
        // after close() no worker is left, so the frame goes nowhere
        let _ = self.jobs.send(PanelJob::Frame(buf));
        Ok(())
    }

    /// Copy a caller-owned frame into a pooled buffer and submit it.
    fn write(&mut self, frame: Frame<'_>) -> Result<()> {
        let mut buf = self.acquire((frame.height, frame.width, frame.components))?;
        buf.data.copy_from_slice(frame.data);
        self.submit(buf)
    }

    fn turn_on(&mut self) -> Result<()> {
        let _ = self.jobs.send(PanelJob::On);
        Ok(())
    }

    fn turn_off(&mut self) -> Result<()> {
        let _ = self.jobs.send(PanelJob::Off);
        Ok(())
    }

    fn clear(&mut self) -> Result<()> {
        let _ = self.jobs.send(PanelJob::Clear);
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.finish()
    }
}

impl Drop for PanelSink {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

/// The sinks in flag order: the display, then -o, then --h264.
fn sink_list<'a>(
    scanout: &'a mut Option<Scanout>,
    panel: &'a mut Option<PanelSink>,
    file: &'a mut FileSink,
    h264: &'a mut Option<H264Stream>,
) -> Vec<&'a mut dyn Output> {
    let mut sinks: Vec<&mut dyn Output> = Vec::new();
    if let Some(s) = scanout.as_mut() {
        sinks.push(s);
    }
    if let Some(s) = panel.as_mut() {
        sinks.push(s);
    }
    match file {
        FileSink::Raw(s) => sinks.push(s),
        FileSink::Images(s) => sinks.push(s),
        FileSink::None => {}
    }
    if let Some(s) = h264.as_mut() {
        sinks.push(s);
    }
    sinks
}

/// The signal that ends the run, 0 for none. SIGTERM (timeout, systemd stop)
/// and SIGINT unwind the run. The run turns its sinks off, so a panel holds no
/// stale frame.
static SIGNALLED: AtomicI32 = AtomicI32::new(0);

/// The render size with nothing to take one from: no display, no --size.
const DEFAULT_SIZE: (i32, i32) = (320, 240);

// Linux signal numbers. The handler is a plain function pointer, so the
// install needs neither SIG_DFL nor SIG_IGN.
const SIGINT: c_int = 2;
const SIGTERM: c_int = 15;

extern "C" {
    fn signal(sig: c_int, handler: extern "C" fn(c_int)) -> usize;
}

extern "C" fn on_signal(sig: c_int) {
    SIGNALLED.store(sig, Ordering::SeqCst);
    // a console blocks in its own server, and this is the way out
    server::stop();
}

fn catch_signals() {
    unsafe {
        signal(SIGINT, on_signal);
        signal(SIGTERM, on_signal);
    }
}

/// True when the reader at the far end of the pipe is gone.
fn broken_pipe(e: &anyhow::Error) -> bool {
    e.chain().any(|c| {
        c.downcast_ref::<std::io::Error>().is_some_and(|io| {
            matches!(
                io.kind(),
                std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset
            )
        })
    })
}

/// One render into as many sinks as the flags ask for.
fn cmd_render(a: &Ns) -> Result<()> {
    // before anything opens: the tokens after `--` own their own --help
    scene_args_help(a)?;
    if a.serve && given(&a.filters) {
        return Err(exit(format!(
            "{}: --filters with --serve; the console has a filter picker",
            a.cmd
        )));
    }
    if a.serve && given(&a.h264) {
        return Err(exit(format!(
            "{}: --h264 with --serve; the console serves /h264 already",
            a.cmd
        )));
    }
    if text(&a.output) == "-" && text(&a.h264) == "-" {
        return Err(exit(format!(
            "{}: -o - and --h264 - both want stdout",
            a.cmd
        )));
    }
    if a.alpha && a.components == 3 {
        return Err(exit(format!(
            "{}: --alpha with --components 3; an RGB frame holds no alpha",
            a.cmd
        )));
    }
    report::apply(a);
    // the scene loads first: it owns the tokens after `--` and their --help. A
    // rejected scene fails before a panel opens. It builds its own buffers as
    // it loads, so it learns the render size before that.
    if let Some((w, h)) = prospective_size(a) {
        let turned = a.rotate == 90 || a.rotate == 270;
        crate::procedural::process::set_render_size(if turned { (h, w) } else { (w, h) });
    }
    let scene = if a.serve { None } else { Some(scene_of(a)?) };
    let panel = panel_of(a)?;
    let kms = matches!(panel, Panel::Kms);
    if a.brightness.is_some() && !kms {
        return Err(exit(format!("{}: --brightness needs --display kms", a.cmd)));
    }
    if matches!(panel, Panel::None) && !given(&a.output) && !given(&a.h264) && !a.serve {
        return Err(exit(format!(
            "{}: pass -o PATTERN, --display NAME, --h264 PATH or --serve",
            a.cmd
        )));
    }
    let dri = a.dri.clone();
    let ((w, h), components) = geometry(a, &panel)?;
    let pace = pacer(a, &panel);
    let mut scanout = kms.then_some(Scanout);
    let mut sink = match panel {
        Panel::Driver(d) => Some(PanelSink::new(d)),
        _ => None,
    };
    let mut file = file_sink(a)?;
    if a.serve {
        // The console draws on this thread, so it takes the sinks as they are.
        // A panel driver's bus handles live on the sink's writer thread. The
        // scanout is never a sink here. It is a mode of the Renderer itself.
        let mut sinks: Vec<Box<dyn Output>> = Vec::new();
        let mut native_format = None;
        if let Some(panel) = sink {
            native_format = Some(panel.native_format.to_string());
            sinks.push(Box::new(panel));
        }
        match file {
            FileSink::Raw(s) => sinks.push(Box::new(s)),
            FileSink::Images(s) => sinks.push(Box::new(s)),
            FileSink::None => {}
        }
        return serve(a, sinks, native_format, (w, h), components, kms, dri);
    }
    let Some((scene, process)) = scene else {
        unreachable!("loaded above for every render that is not --serve")
    };
    let controller: ControllerSlot =
        RefCell::new(process.map(|f| Box::new(f) as Box<dyn SceneController>));
    let mut r = Renderer::new(w, h, components, dri.as_deref(), kms, a.rgb565)?;
    r.set_alpha(a.alpha);
    r.set_rotate(a.rotate)?;
    if let Some(level) = a.brightness {
        let took = r
            .set_brightness(level)
            .map_err(|e| exit(format!("{}: {e:#}", a.cmd)))?;
        report::detail("backlight", format!("{took:.3}"));
    }
    let err = r.setup(&scene)?;
    if !err.is_empty() {
        return Err(exit(format!("compile error: {err}")));
    }
    let mut h264 = None;
    if given(&a.h264) {
        let path = text(&a.h264);
        let fps = h264_fps(a);
        let stream: Box<dyn Write + Send> = match path {
            "-" => Box::new(stdout()),
            _ => Box::new(File::create(path).with_context(|| path.to_string())?),
        };
        let stream = H264Stream::new(stream, &r, w, h, fps, a.bitrate, components)?;
        let (sw, sh) = stream.size;
        eprintln!(
            "h264: {} {sw}x{sh}@{fps} {}bps -> {path}",
            if stream.hw {
                "hardware zero-copy"
            } else {
                "software libx264"
            },
            a.bitrate
        );
        h264 = Some(stream);
    }
    let times: RefCell<Vec<f64>> = RefCell::new(Vec::new());
    // a terminal watches the run, a pipe and a service unit do not
    let live = report::tty() && report::level() >= report::NORMAL;
    let keep = a.stats;
    let engines = keep.then(usage::Engines::read).flatten();
    let want = frames_of(a);
    let (ran, drawn, elapsed, out) = {
        let sinks = sink_list(&mut scanout, &mut sink, &mut file, &mut h264);
        let mut lp = Loop::new(&mut r, sinks);
        lp.controller = Some(&controller);
        lp.provider = Some(Box::new(|name, size, frame| {
            Feed::Data(demo_feed(name, size, frame))
        }));
        // 0 is the endless run, so the frame budget spells no other count for
        // "render nothing". A negative --frames stops before the first frame.
        lp.frames = want.max(0) as u64;
        lp.timestep = a.timestep;
        lp.pacer = pace;
        lp.stopping = Some(Box::new(move || {
            want < 0 || SIGNALLED.load(Ordering::SeqCst) != 0
        }));
        if keep || live {
            lp.on_frame = Some(Box::new(|lp, cost| {
                if keep {
                    times.borrow_mut().push(cost);
                }
                if live {
                    report::frame(cost, lp.drawn, lp.frames, lp.frame - lp.drawn, || {
                        lp.out().join("+")
                    });
                }
            }));
        }
        let ran = lp.run();
        report::frames_done();
        let out: Vec<String> = lp.out().iter().map(|d| (*d).to_string()).collect();
        (ran, lp.drawn, lp.elapsed, out)
    };
    // before the renderer lets its buffers go
    let used = match keep {
        true => usage_now(engines.as_ref()),
        false => Usage::default(),
    };
    let pipe = text(&a.output) == "-" || text(&a.h264) == "-";
    let ran = match ran {
        Err(e) if pipe && broken_pipe(&e) => Ok(()),
        other => other,
    };
    let mut closed = Ok(());
    for s in sink_list(&mut scanout, &mut sink, &mut file, &mut h264) {
        closed = closed.and(s.close());
    }
    r.close();
    if SIGNALLED.load(Ordering::SeqCst) == SIGTERM {
        std::process::exit(143);
    }
    ran?;
    closed?;
    if given(&a.output) && text(&a.output) != "-" {
        report::step("wrote", format!("{drawn} frame(s) to {}", text(&a.output)));
    }
    if given(&a.h264) && text(&a.h264) != "-" {
        report::step("wrote", format!("{drawn} frame(s) to {}", text(&a.h264)));
    }
    if a.stats {
        print_stats(&times.borrow(), elapsed, &out, &used);
    }
    Ok(())
}

/// Attach the console to this render. A scene serves itself alone. A directory
/// (or --demo) serves everything in it.
#[allow(clippy::too_many_arguments)]
fn serve(
    a: &Ns,
    sinks: Vec<Box<dyn Output>>,
    native_format: Option<String>,
    size: (i32, i32),
    components: i32,
    kms: bool,
    dri: Option<String>,
) -> Result<()> {
    let (mut scenes_dir, mut store) = (None, None);
    if let Some(scene) = &a.scene {
        let path = find_scene(scene, a.demo)?;
        if Path::new(&path).is_dir() {
            scenes_dir = Some(path);
        } else {
            store = Some(one_scene_store(a, &path)?);
        }
    }
    let ran = server::run(server::Options {
        scenes_dir,
        state_dir: a.state_dir.clone(),
        store,
        size,
        components,
        alpha: a.alpha,
        sinks,
        native_format,
        display: kms,
        rgb565: a.rgb565,
        rotate: a.rotate,
        brightness: a.brightness,
        port: a.port,
        host: a.host.clone(),
        preview_max: a.preview_max,
        preview_fps: f64::from(a.preview_fps),
        stream_max: a.stream_max,
        stream_hz: a.stream_fps,
        bitrate: i64::from(a.bitrate),
        scene_args: a.scene_args.clone(),
        frames: frames_of(a).max(0) as u64,
        timestep: a.timestep,
        dri_device: dri,
        on_stats: a.stats.then(serve_stats),
        pacer: console_pacer(a),
        ..Default::default()
    });
    if SIGNALLED.load(Ordering::SeqCst) == SIGTERM {
        std::process::exit(143);
    }
    ran
}

/// The console's frame rate cap: --fps, whatever takes the frames.
fn console_pacer(a: &Ns) -> Option<Box<dyn Pacer>> {
    a.fps
        .filter(|f| *f != 0.0)
        .map(|f| Box::new(FixedPacer::new(f)) as Box<dyn Pacer>)
}

/// A store for the single scene `render --serve` names. A bundled example
/// keeps its state file as a demo, whichever way a command reaches it.
fn one_scene_store(a: &Ns, path: &str) -> Result<SceneStore> {
    let mut store =
        SceneStore::one(path, a.scene_args.clone()).map_err(|e| exit(format!("{}: {e}", a.cmd)))?;
    store.demo = in_examples(path);
    Ok(store)
}

/// True when a scene sits in the bundled examples directory. The comparison
/// resolves symlinks.
fn in_examples(path: &str) -> bool {
    let dir = match Path::new(path).parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => std::env::current_dir().unwrap_or_default(),
    };
    let Some(examples) = examples_dir() else {
        return false;
    };
    match (dir.canonicalize(), Path::new(&examples).canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// `serve` is `render --serve` over a directory of scenes.
fn cmd_serve(a: &Ns) -> Result<()> {
    if a.demo && given(&a.scene) {
        return Err(exit("serve: pass a scenes directory or --demo, not both"));
    }
    if !a.demo && !given(&a.scene) {
        if given(&a.display) && Path::new(text(&a.display)).is_dir() {
            // --display takes an optional value, so it took the directory
            return Err(exit(format!(
                "serve: --display took '{}' as the display name; \
                 put the scenes directory before it",
                text(&a.display)
            )));
        }
        return Err(exit(
            "serve: pass a scenes directory, or --demo for the bundled examples",
        ));
    }
    cmd_render(a)
}

/// --stats: one summary line after a render. ms/frame is the per-frame cost
/// without the paced sleeps. fps is frames over elapsed time, and the console
/// publishes the same number.
fn stats_line(times: &[f64], wall: f64, out: &[String]) -> String {
    let n = times.len();
    let total: f64 = times.iter().sum();
    let min = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    format!(
        "stats: {n} frame(s) in {wall:.2}s, {:.2} ms/frame (min {:.2}, max {:.2}), {:.1} fps -> {}",
        1000.0 * total / n as f64,
        1000.0 * min,
        1000.0 * max,
        rate(n as u64, wall),
        out.join("+")
    )
}

/// What this process used at the end of a run, where the machine reports it
/// (crate::usage).
#[derive(Default)]
struct Usage {
    rss: Option<u64>,
    gpu: Option<u64>,
    /// The percent of the run that each queue of the GPU was busy.
    busy: Option<BTreeMap<String, f64>>,
    machine_free: Option<u64>,
    soc_celsius: Option<f64>,
}

/// The process's usage now, its GPU busy since the reading `start`.
fn usage_now(start: Option<&usage::Engines>) -> Usage {
    let busy = start
        .zip(usage::Engines::read())
        .map(|(s, now)| now.busy_since(s));
    Usage {
        rss: usage::rss(),
        gpu: usage::gpu(),
        busy,
        machine_free: usage::machine_free(),
        soc_celsius: usage::soc_celsius(),
    }
}

/// --stats: the line under the summary, with each figure of `u` that the
/// machine reports.
fn usage_line(u: &Usage) -> String {
    let mib = |b: u64| format!("{:.1} MiB", usage::mib(b));
    let mut parts = Vec::new();
    if let Some(b) = u.rss {
        parts.push(format!("rss {}", mib(b)));
    }
    if let Some(b) = u.gpu {
        parts.push(format!("gpu {}", mib(b)));
    }
    if let Some(busy) = u.busy.as_ref().filter(|b| !b.is_empty()) {
        let queues: Vec<String> = busy.iter().map(|(q, p)| format!("{q} {p:.1}%")).collect();
        parts.push(format!("gpu busy {}", queues.join(" ")));
    }
    if let Some(b) = u.machine_free {
        parts.push(format!("machine free {:.0} MiB", usage::mib(b)));
    }
    if let Some(c) = u.soc_celsius {
        parts.push(format!("soc {c:.1} °C"));
    }
    format!("usage: {}", parts.join(", "))
}

fn print_stats(times: &[f64], wall: f64, out: &[String], used: &Usage) {
    if times.is_empty() {
        return;
    }
    report::summary(stats_line(times, wall, out));
    report::summary(usage_line(used));
}

/// Where a serve run's frames go. A console with no panel and no file is still
/// a destination.
fn serve_out(out: &[String]) -> Vec<String> {
    if out.is_empty() {
        return vec!["console".to_string()];
    }
    out.to_vec()
}

/// The same summary line, for the end of the console's render.
fn serve_stats() -> server::StatsHook {
    let start = usage::Engines::read();
    Box::new(move |times, wall, out| {
        print_stats(times, wall, &serve_out(out), &usage_now(start.as_ref()))
    })
}

/// One line of a listing. A reader that closes the pipe ends the command,
/// the way `ghee displays | head` does.
fn line(text: &str) {
    if let Err(e) = writeln!(stdout(), "{text}") {
        if e.kind() == std::io::ErrorKind::BrokenPipe {
            std::process::exit(0);
        }
        eprintln!("ghee: {e}");
        std::process::exit(1);
    }
}

/// `ghee overlay` writes the kernel's own setup for a panel: the config.txt
/// lines that bring the card up, and the panel's firmware file. The panel
/// then answers `--display kms` like any other display.
fn cmd_overlay(a: &OverlayArgs) -> Result<()> {
    let Some(panel) = overlay::panel(&a.driver) else {
        return Err(exit(format!(
            "overlay: no display driver '{}' (see: ghee displays)",
            a.driver
        )));
    };
    let opts = match given(&a.display_opts) {
        true => parse_kv(text(&a.display_opts), "overlay: --display-opts")?,
        false => Map::new(),
    };
    // Only the options that describe the panel reach a kernel setup: the
    // wire format and the update strategy are the driver's own.
    let known = panel.options;
    let driver = driver_options(get_driver(&a.driver)?);
    let takes = || known.join(", ");
    let takes_key = |k: &str| known.contains(&k);
    let own: Vec<&str> = opts
        .keys()
        .map(String::as_str)
        .filter(|k| !takes_key(k) && driver.iter().any(|o| o.name == *k))
        .collect();
    if !own.is_empty() {
        return Err(exit(format!(
            "overlay: the userspace driver alone acts on {} (the kernel setup takes: {})",
            own.join(", "),
            takes()
        )));
    }
    let unknown: Vec<&str> = opts
        .keys()
        .map(String::as_str)
        .filter(|k| !takes_key(k) && !driver.iter().any(|o| o.name == *k))
        .collect();
    if !unknown.is_empty() {
        return Err(exit(format!(
            "overlay: no option {} on {} (takes: {})",
            unknown.join(", "),
            a.driver,
            takes()
        )));
    }
    let pairs: Vec<(String, Value)> = opts.into_iter().collect();
    let setup = (panel.setup)(&pairs).map_err(|e| exit(format!("overlay: {e}")))?;
    for text in &setup.lines {
        line(text);
    }
    match setup.firmware {
        Some(blob) => {
            let path = text(&a.output);
            let path = match path.is_empty() {
                true => "panel.bin",
                false => path,
            };
            File::create(path)
                .and_then(|mut f| f.write_all(&blob))
                .with_context(|| path.to_string())?;
            eprintln!(
                "wrote {path}. Copy it to /lib/firmware/panel.bin, add the lines \
                 above to /boot/firmware/config.txt, and reboot"
            );
        }
        None => {
            if given(&a.output) {
                return Err(exit(format!(
                    "overlay: -o with {}, a panel that takes no firmware file",
                    a.driver
                )));
            }
            eprintln!("add the lines above to /boot/firmware/config.txt, and reboot");
        }
    }
    Ok(())
}

fn cmd_displays() -> Result<()> {
    // every card with a connected display, since an SPI panel under a
    // kernel display driver is a card beside the HDMI one
    let mut connected = 0;
    for dev in crate::engine::probe_dri_cards() {
        if let Some((w, h)) = probe_display(Some(&dev)) {
            line(&format!("{:12} {dev} connected, {w}x{h}", "kms"));
            connected += 1;
        }
    }
    if connected == 0 {
        line(&format!("{:12} no connected display", "kms"));
    }
    for name in available_drivers() {
        line(&format!("{name:12} ok"));
        let opts: Vec<String> = driver_options(get_driver(name)?)
            .iter()
            .map(|o| match o.default {
                Some(v) => format!("{}={v}", o.name),
                None => format!("{}=None", o.name),
            })
            .collect();
        if !opts.is_empty() {
            // a driver's own defaults: an SPI or I2C panel answers no probe
            line(&format!(
                "{:12} defaults: --display-opts {}",
                "",
                opts.join(",")
            ));
        }
    }
    Ok(())
}

// ------------------------------------------------------------ the parser ----

#[derive(Parser)]
#[command(
    name = "ghee",
    version,
    disable_help_subcommand = true,
    disable_version_flag = true
)]
struct Cli {
    // long only: clap's own short -V stays off
    #[arg(
        long,
        action = clap::ArgAction::Version,
        help = "show program's version number and exit"
    )]
    version: Option<bool>,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    #[command(
        about = "render a scene to images, a display, a stream, or several at once",
        // a negative number is a value here
        allow_negative_numbers = true,
        after_help = "anything after `--` goes to the scene itself \
                      (ghee render --demo hero -- --no-dither), and `-- --help` lists what it takes"
    )]
    Render(RenderArgs),

    #[command(
        about = "run the editor console",
        allow_negative_numbers = true,
        after_help = "the console is plain HTTP: GET /stats /scene /scenes /frame.png \
                      /stream /h264, POST /set /switch /next /prev /scene /save /ping. \
                      The console passes anything after `--` to every scene it loads, \
                      each scene keeps the arguments it takes, and `-- --help` lists \
                      them per scene"
    )]
    Serve(ServeArgs),

    #[command(about = "list connected cards, display drivers, and their options")]
    Displays,

    #[command(
        about = "write the kernel's setup for a panel, so it answers --display kms",
        after_help = "the panel then belongs to the kernel driver, and --display <NAME> \
                      no longer reaches it"
    )]
    Overlay(OverlayArgs),

    /// The benchmark suite is a development tool, so it stays out of
    /// `ghee --help`.
    #[command(hide = true, about = "run the benchmark suite")]
    Bench(BenchArgs),
}

/// What every render takes, whichever command asks for it.
#[derive(Args)]
struct RenderFlags {
    #[arg(
        long,
        value_name = "NAME",
        num_args = 0..=1,
        default_missing_value = "kms",
        help = "render to a hardware display: an SPI/I2C driver name, or bare \
                (or 'kms') for the connected KMS display, HDMI or a kernel-driven \
                panel (zero-copy scanout, needs DRM master)"
    )]
    display: Option<String>,

    #[arg(
        long,
        value_name = "K=V,...",
        help = "options for the --display driver, e.g. width=240,height=240,\
                y_offset=80,backlight=26. `ghee displays` lists what each one takes"
    )]
    display_opts: Option<String>,

    #[arg(
        long,
        help = "display scanout in 16-bit color: halves the GPU store and scanout \
                bandwidth (worth it on VC4-class boards), slight banding in gradients"
    )]
    rgb565: bool,

    #[arg(
        long,
        value_name = "DEGREES",
        default_value_t = 0,
        value_parser = rotation,
        help = "turn the frame clockwise on its way to every output: 0, 90, 180 or \
                270. At 90 and 270 the scene renders at the output's size turned"
    )]
    rotate: i64,

    #[arg(
        long,
        value_name = "LEVEL",
        value_parser = level,
        help = "set the --display kms panel's backlight, 0 to 1, through the \
                kernel's backlight device"
    )]
    brightness: Option<f64>,

    #[arg(
        long,
        help = "with `-- --help`, list the scenes without asking each one what \
                it takes"
    )]
    no_probe: bool,

    #[arg(short, long, help = "errors only")]
    quiet: bool,

    #[arg(short, long, help = "the paths and formats behind each step")]
    verbose: bool,

    #[arg(
        long,
        help = "frame count (default 1 for images, loop for a display, a stream \
                or the console)"
    )]
    frames: Option<i64>,

    #[arg(
        long,
        value_name = "S",
        help = "pin engine time to frame*S seconds instead of the wall clock, so \
                renders are reproducible (frame N always renders at time N*S)"
    )]
    timestep: Option<f64>,

    #[arg(
        long,
        value_parser = size,
        help = "WxH to render at, a display or panel sets its own (default 320x240)"
    )]
    size: Option<(i32, i32)>,

    #[arg(
        long,
        help = "DRI device, e.g. /dev/dri/renderD128. With --display it picks \
                which card's display to drive"
    )]
    dri: Option<String>,
}

/// The console's own flags, bar --bitrate. --bitrate has a different default
/// per command, so each command declares it.
#[derive(Args)]
struct ConsoleFlags {
    #[arg(long, default_value_t = 8090)]
    port: u16,

    #[arg(
        long,
        value_name = "DIR",
        help = "keep the console's saved edits in DIR \
                (default $XDG_STATE_HOME/ghee, else ~/.local/state/ghee)"
    )]
    state_dir: Option<std::path::PathBuf>,

    #[arg(
        long,
        default_value = "127.0.0.1",
        help = "bind address, 0.0.0.0 exposes the console (no auth) to the network"
    )]
    host: String,

    #[arg(
        long,
        default_value_t = 480,
        help = "longest side of the browser preview in px (downscaled, default 480)"
    )]
    preview_max: i32,

    #[arg(
        long,
        default_value_t = 15,
        help = "browser preview refresh cap, the display stays full speed (default 15)"
    )]
    preview_fps: i32,

    #[arg(
        long,
        default_value_t = 0,
        help = "cap the /h264 stream's longest side (GPU downscale, \
                default 0 = full render size)"
    )]
    stream_max: i32,

    #[arg(
        long,
        default_value_t = 30,
        help = "encode the /h264 stream at this frame rate (default 30)"
    )]
    stream_fps: i32,
}

#[derive(Args)]
struct RenderArgs {
    #[arg(help = "scene file, or a name from $GHEE_SCENES")]
    scene: String,

    #[arg(long, help = "look the scene name up in the bundled examples too")]
    demo: bool,

    #[arg(
        short,
        long,
        help = "image pattern (frame.####.jpg), or '-' for raw RGB(A) frames to \
                stdout (h*w*components with no header, pipe to ffmpeg/mpv)"
    )]
    output: Option<String>,

    #[arg(
        long,
        value_name = "PATH",
        help = "write an Annex-B H264 stream to PATH ('-' = stdout, pipe to mpv). \
                Hardware zero-copy on Pi 1..Pi 4, ffmpeg/libx264 elsewhere"
    )]
    h264: Option<String>,

    #[arg(
        long,
        help = "also run the editor console on this render (see: ghee serve)"
    )]
    serve: bool,

    #[arg(
        long,
        default_value_t = 4,
        value_parser = clap::value_parser!(i32).range(3..=4),
        help = "3=RGB, 4=RGBA (desktop Mesa can only read back RGBA)"
    )]
    components: i32,

    #[arg(
        long,
        help = "keep the output pass's alpha in RGBA frames, which are opaque \
                without it"
    )]
    alpha: bool,

    #[arg(
        long,
        value_name = "K=V,...",
        help = "extra options passed to the image encoder, e.g. quality=90 or \
                lossless=True (ignored for -o -)"
    )]
    save_opts: Option<String>,

    #[arg(
        long,
        default_value_t = 16,
        value_name = "N",
        help = "frames buffered for the async image writer. The render waits only \
                when the writer falls N frames behind, at a memory cost of \
                N * w * h * components bytes"
    )]
    write_buffer: i64,

    #[arg(
        long,
        help = "fixed display frame rate (default: adapt to the panel's own speed)"
    )]
    fps: Option<f64>,

    #[arg(
        long,
        help = "comma-separated filter names applied after the scene's own"
    )]
    filters: Option<String>,

    #[arg(
        long,
        help = "print a timing summary and what ghee used when the render ends"
    )]
    stats: bool,

    #[command(flatten)]
    render: RenderFlags,

    #[command(flatten)]
    console: ConsoleFlags,

    #[arg(
        long,
        default_value_t = 4_000_000,
        help = "H264 bitrate in bits/s: the --h264 stream's rate, or the floor \
                for the console's /h264 (default 4000000)"
    )]
    bitrate: i32,
}

#[derive(Args)]
struct ServeArgs {
    #[arg(help = "scenes directory")]
    scenes: Option<String>,

    #[arg(long, help = "serve the bundled example scenes")]
    demo: bool,

    #[arg(
        long,
        help = "cap the render's frame rate (default: as fast as the display and the \
                viewers take frames)"
    )]
    fps: Option<f64>,

    #[command(flatten)]
    render: RenderFlags,

    #[command(flatten)]
    console: ConsoleFlags,

    #[arg(
        long,
        default_value_t = 10_000_000,
        help = "H264 bitrate in bits/s: the --h264 stream's rate, or the floor \
                for the console's /h264 (default 10000000)"
    )]
    bitrate: i32,
}

#[derive(Args)]
struct OverlayArgs {
    #[arg(help = "display driver name (see: ghee displays)")]
    driver: String,

    #[arg(
        long,
        value_name = "K=V,...",
        help = "the panel's options, as `ghee render --display-opts` takes them"
    )]
    display_opts: Option<String>,

    #[arg(
        short,
        long,
        help = "where to write the firmware file (default panel.bin)"
    )]
    output: Option<String>,
}

/// render and serve both resolve to this namespace. serve is render --serve
/// over a directory. It carries render's whole namespace, and every flag it
/// does not offer keeps its default.
pub struct Ns {
    cmd: &'static str,
    scene: Option<String>,
    demo: bool,
    output: Option<String>,
    h264: Option<String>,
    serve: bool,
    components: i32,
    alpha: bool,
    save_opts: Option<String>,
    write_buffer: i64,
    fps: Option<f64>,
    filters: Option<String>,
    stats: bool,
    display: Option<String>,
    display_opts: Option<String>,
    rgb565: bool,
    rotate: i64,
    brightness: Option<f64>,
    frames: Option<i64>,
    timestep: Option<f64>,
    size: Option<(i32, i32)>,
    dri: Option<String>,
    port: u16,
    host: String,
    state_dir: Option<std::path::PathBuf>,
    preview_max: i32,
    preview_fps: i32,
    stream_max: i32,
    stream_fps: i32,
    bitrate: i32,
    no_probe: bool,
    pub quiet: bool,
    pub verbose: bool,
    scene_args: Vec<String>,
}

impl RenderArgs {
    fn ns(self, scene_args: Vec<String>) -> Ns {
        Ns {
            cmd: "render",
            scene: Some(self.scene),
            demo: self.demo,
            output: self.output,
            h264: self.h264,
            serve: self.serve,
            components: self.components,
            alpha: self.alpha,
            save_opts: self.save_opts,
            write_buffer: self.write_buffer,
            fps: self.fps,
            filters: self.filters,
            stats: self.stats,
            no_probe: self.render.no_probe,
            quiet: self.render.quiet,
            verbose: self.render.verbose,
            display: self.render.display,
            display_opts: self.render.display_opts,
            rgb565: self.render.rgb565,
            rotate: self.render.rotate,
            brightness: self.render.brightness,
            frames: self.render.frames,
            timestep: self.render.timestep,
            size: self.render.size,
            dri: self.render.dri,
            port: self.console.port,
            host: self.console.host,
            state_dir: self.console.state_dir,
            preview_max: self.console.preview_max,
            preview_fps: self.console.preview_fps,
            stream_max: self.console.stream_max,
            stream_fps: self.console.stream_fps,
            bitrate: self.bitrate,
            scene_args,
        }
    }
}

impl ServeArgs {
    fn ns(self, scene_args: Vec<String>) -> Ns {
        Ns {
            cmd: "serve",
            // --demo leaves it None: the bundled examples
            scene: self.scenes,
            demo: self.demo,
            output: None,
            h264: None,
            serve: true,
            no_probe: self.render.no_probe,
            quiet: self.render.quiet,
            verbose: self.render.verbose,
            components: 4,
            alpha: false,
            save_opts: None,
            write_buffer: 16,
            fps: self.fps,
            filters: None,
            stats: false,
            display: self.render.display,
            display_opts: self.render.display_opts,
            rgb565: self.render.rgb565,
            rotate: self.render.rotate,
            brightness: self.render.brightness,
            frames: self.render.frames,
            timestep: self.render.timestep,
            size: self.render.size,
            dri: self.render.dri,
            port: self.console.port,
            host: self.console.host,
            state_dir: self.console.state_dir,
            preview_max: self.console.preview_max,
            preview_fps: self.console.preview_fps,
            stream_max: self.console.stream_max,
            stream_fps: self.console.stream_fps,
            bitrate: self.bitrate,
            scene_args,
        }
    }
}

/// The command name behind a failure report, and the result of that command.
fn dispatch(cli: Cli, scene_args: Vec<String>) -> (&'static str, Result<()>) {
    let cmd = match &cli.cmd {
        Command::Render(_) => "render",
        Command::Serve(_) => "serve",
        Command::Displays => "displays",
        Command::Overlay(_) => "overlay",
        Command::Bench(_) => "bench",
    };
    if !scene_args.is_empty() && !matches!(cli.cmd, Command::Render(_) | Command::Serve(_)) {
        return (
            cmd,
            Err(exit(format!("{cmd}: takes no scene arguments after --"))),
        );
    }
    let ran = match cli.cmd {
        Command::Render(a) => cmd_render(&a.ns(scene_args)),
        Command::Serve(a) => cmd_serve(&a.ns(scene_args)),
        Command::Displays => cmd_displays(),
        Command::Overlay(a) => cmd_overlay(&a),
        Command::Bench(a) => bench::run(&a),
    };
    (cmd, ran)
}

/// The top-level parser reports a token when no subcommand claims it. The
/// line names `ghee` itself and the status is 2.
fn clap_status(e: &clap::Error) -> i32 {
    if e.kind() == ErrorKind::UnknownArgument {
        let arg = e
            .get(ContextKind::InvalidArg)
            .map(|v| v.to_string())
            .unwrap_or_default();
        eprintln!("ghee: error: unrecognized arguments: {arg}");
        return 2;
    }
    let _ = e.print();
    i32::from(e.use_stderr()) * 2
}

/// The command line's exit status. A scene, engine, encoder or missing
/// dependency failure reaches the user as one line, with no stack. An argument
/// error reads the same way. $GHEE_TRACEBACK asks for the whole error chain.
pub fn main(argv: Vec<String>) -> i32 {
    // A bare `ghee` and `ghee --version` greet, everything else works.
    if argv.is_empty() || argv == ["--version"] || argv == ["-V"] {
        eprint!("{}", report::splash(env!("CARGO_PKG_VERSION")));
        return 0;
    }
    let (own, scene_args) = split_scene_args(&argv);
    let cli = match Cli::try_parse_from(std::iter::once("ghee".to_string()).chain(own)) {
        Ok(cli) => cli,
        Err(e) => return clap_status(&e),
    };
    catch_signals();
    let (cmd, ran) = dispatch(cli, scene_args);
    match ran {
        Ok(()) => 0,
        Err(e) => {
            if std::env::var("GHEE_TRACEBACK").is_ok_and(|v| !v.is_empty()) {
                eprintln!("{e:?}");
            } else if let Some(line) = e.downcast_ref::<Exit>() {
                eprintln!("{line}");
            } else {
                eprintln!("{cmd}: {e:#}");
            }
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const SCENE: &str = r#"{"version":1,"passes":[{"name":"main",
        "fragment":"void main(){ gl_FragColor = vec4(0.25,0.5,0.75,1.0); }"}]}"#;

    fn argv(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|s| (*s).to_string()).collect()
    }

    fn parse(tokens: &[&str]) -> (Cli, Vec<String>) {
        let (own, scene_args) = split_scene_args(&argv(tokens));
        let cli = Cli::try_parse_from(std::iter::once("ghee".to_string()).chain(own)).unwrap();
        (cli, scene_args)
    }

    /// The namespace of a parsed command line, without running it.
    fn ns(tokens: &[&str]) -> Ns {
        let (cli, scene_args) = parse(tokens);
        match cli.cmd {
            Command::Render(a) => a.ns(scene_args),
            Command::Serve(a) => a.ns(scene_args),
            _ => panic!("not a render command"),
        }
    }

    fn run(tokens: &[&str]) -> Result<()> {
        let (cli, scene_args) = parse(tokens);
        dispatch(cli, scene_args).1
    }

    /// The message of a call that has to fail. Neither Panel nor FileSink is
    /// Debug, so unwrap_err() cannot ask for it.
    fn failure<T>(r: Result<T>) -> String {
        match r {
            Ok(_) => panic!("expected a failure"),
            Err(e) => e.to_string(),
        }
    }

    fn tmpdir(tag: &str) -> crate::testing::TempDir {
        crate::testing::TempDir::new(&format!("cli-{tag}"))
    }

    /// A scene file in a directory of its own, which lives as long as the
    /// TempDir.
    fn scene_file(tag: &str) -> (crate::testing::TempDir, String) {
        let d = tmpdir(tag);
        let path = format!("{d}/s.scene.json");
        fs::write(&path, SCENE).unwrap();
        (d, path)
    }

    #[test]
    fn brightness_takes_a_level_for_a_kms_display() {
        let (_d, path) = scene_file("brightness");
        let e = run(&["render", &path, "-o", "x.png", "--brightness", "0.5"]).unwrap_err();
        assert_eq!(e.to_string(), "render: --brightness needs --display kms");
        assert_eq!(
            level("1.5").unwrap_err(),
            "'1.5' is not a level from 0 to 1"
        );
        assert_eq!(
            ns(&["serve", "--demo", "--brightness", "0.25"]).brightness,
            Some(0.25)
        );
    }

    #[test]
    fn rotate_takes_the_four_turns() {
        for d in ["0", "90", "180", "270"] {
            let a = ns(&["render", "s.scene.json", "-o", "x.png", "--rotate", d]);
            assert_eq!(a.rotate.to_string(), d);
        }
        assert_eq!(ns(&["serve", "--demo"]).rotate, 0);
        assert_eq!(rotation("45").unwrap_err(), "'45' is not 0, 90, 180 or 270");
    }

    #[test]
    fn parse_kv_types() {
        let d = parse_kv(
            "quality=90,lossless=True,method=fast,note=hi there",
            "--save-opts",
        )
        .unwrap();
        assert_eq!(d["quality"], Value::from(90));
        assert_eq!(d["lossless"], Value::Bool(true));
        assert_eq!(d["method"], Value::from("fast"));
        assert_eq!(d["note"], Value::from("hi there"));
        assert!(d["quality"].is_i64());
    }

    /// One parser, two flags: the error says which flag took the value.
    #[test]
    fn parse_kv_malformed_exits_naming_the_flag_that_took_it() {
        for (s, what) in [
            ("quality=90,nope", "render: --save-opts"),
            ("width=240,nope", "serve: --display-opts"),
        ] {
            let e = parse_kv(s, what).unwrap_err();
            assert_eq!(e.to_string(), format!("{what} wants k=v pairs, got 'nope'"));
            assert!(e.downcast_ref::<Exit>().is_some());
        }
    }

    #[test]
    fn a_value_that_is_no_literal_stays_a_string() {
        let d = parse_kv(
            "backlight=None,rst= 25 ,name='panel',rate=1.5,mask=0x1f",
            "x",
        )
        .unwrap();
        assert_eq!(d["backlight"], Value::Null);
        assert_eq!(d["rst"], Value::from(25));
        assert_eq!(d["name"], Value::from("panel"));
        assert_eq!(d["rate"], Value::from(1.5));
        assert_eq!(d["mask"], Value::from(31));
        assert_eq!(literal_eval("wide"), None);
        assert_eq!(literal_eval("-4"), Some(Value::from(-4)));
        assert_eq!(literal_eval("1_0"), Some(Value::from(10)));
    }

    #[test]
    fn split_leaves_gheeflags_alone() {
        let (mine, theirs) = split_scene_args(&argv(&[
            "render", "text", "--size", "64x48", "--", "--text", "hi", "--", "-x",
        ]));
        assert_eq!(mine, argv(&["render", "text", "--size", "64x48"]));
        assert_eq!(theirs, argv(&["--text", "hi", "--", "-x"]));
        assert_eq!(
            split_scene_args(&argv(&["render", "text", "--size", "64x48"])),
            (argv(&["render", "text", "--size", "64x48"]), Vec::new())
        );
    }

    #[test]
    fn commands_without_scenes_reject_scene_args() {
        let e = run(&["displays", "--", "--emit", "8x8"]).unwrap_err();
        assert_eq!(e.to_string(), "displays: takes no scene arguments after --");
    }

    #[test]
    fn a_flag_that_ghee_does_not_take_is_refused_by_name() {
        let e = Cli::try_parse_from(argv(&["ghee", "render", "text", "--nosuchflag"]))
            .err()
            .unwrap();
        assert_eq!(e.kind(), ErrorKind::UnknownArgument);
        assert_eq!(
            e.get(ContextKind::InvalidArg)
                .map(|v| v.to_string())
                .as_deref(),
            Some("--nosuchflag")
        );
    }

    #[test]
    fn a_size_is_wxh() {
        assert_eq!(size("64x48"), Ok((64, 48)));
        assert_eq!(size("64X48"), Ok((64, 48)));
        assert_eq!(size("-4x9"), Ok((-4, 9)));
        for bad in ["banana", "16", "0x9x2", ""] {
            assert!(size(bad).is_err(), "{bad}");
        }
        assert_eq!(ns(&["render", "s", "--size", "80x60"]).size, Some((80, 60)));
        assert_eq!(ns(&["render", "s"]).size, None);
    }

    #[test]
    fn a_panel_render_loops_until_stopped_and_an_image_render_is_one_frame() {
        assert_eq!(frames_of(&ns(&["render", "s", "-o", "x.png"])), 1);
        for endless in [
            vec!["render", "s", "--display", "st7789"],
            vec!["render", "s", "--h264", "out.h264"],
            vec!["render", "s", "--serve"],
        ] {
            assert_eq!(frames_of(&ns(&endless)), 0, "{endless:?}");
        }
        assert_eq!(
            frames_of(&ns(&[
                "render",
                "s",
                "--display",
                "st7789",
                "--frames",
                "9"
            ])),
            9
        );
    }

    #[test]
    fn display_opts_belong_to_a_panel_driver() {
        for (extra, msg) in [
            (vec![], "render: --display-opts without --display"),
            (
                vec!["--display", "kms"],
                "render: --display-opts with --display kms; kms has no driver",
            ),
        ] {
            let mut tokens = vec!["render", "s", "-o", "x.png", "--display-opts", "width=40"];
            tokens.extend(extra);
            assert_eq!(failure(panel_of(&ns(&tokens))), msg);
        }
    }

    #[test]
    fn an_unknown_display_opt_names_the_drivers_own_options() {
        let a = ns(&[
            "render",
            "s",
            "--display",
            "st7789",
            "--display-opts",
            "widht=40",
        ]);
        let e = failure(panel_of(&a));
        assert!(
            e.starts_with("render: --display-opts: st7789 has no option widht"),
            "{e}"
        );
        assert!(e.contains("(takes: width, height, x_offset"), "{e}");
    }

    #[test]
    fn a_display_that_is_a_directory_says_where_the_directory_goes() {
        let d = tmpdir("dir-display");
        assert_eq!(
            failure(panel_of(&ns(&["serve", "--display", &d]))),
            format!(
                "serve: no display driver '{d}' (see: ghee displays), \
                 and a scenes directory goes before it"
            )
        );
    }

    #[test]
    fn two_sinks_cannot_both_claim_stdout() {
        let e = run(&["render", "s", "-o", "-", "--h264", "-"]).unwrap_err();
        assert_eq!(e.to_string(), "render: -o - and --h264 - both want stdout");
    }

    #[test]
    fn alpha_is_refused_with_three_components() {
        let e = run(&["render", "s", "-o", "-", "--alpha", "--components", "3"]).unwrap_err();
        assert_eq!(
            e.to_string(),
            "render: --alpha with --components 3; an RGB frame holds no alpha"
        );
    }

    #[test]
    fn a_render_with_nowhere_to_go_says_so() {
        let (_d, path) = scene_file("nowhere");
        assert_eq!(
            run(&["render", &path]).unwrap_err().to_string(),
            "render: pass -o PATTERN, --display NAME, --h264 PATH or --serve"
        );
    }

    #[test]
    fn filters_and_h264_are_refused_with_serve() {
        for (extra, msg) in [
            (
                vec!["--filters", "dither"],
                "render: --filters with --serve; the console has a filter picker",
            ),
            (
                vec!["--h264", "x.h264"],
                "render: --h264 with --serve; the console serves /h264 already",
            ),
        ] {
            let mut tokens = vec!["render", "s", "--serve"];
            tokens.extend(extra);
            assert_eq!(run(&tokens).unwrap_err().to_string(), msg);
        }
    }

    #[test]
    fn serve_wants_a_directory_or_demo_and_not_both() {
        assert_eq!(
            run(&["serve"]).unwrap_err().to_string(),
            "serve: pass a scenes directory, or --demo for the bundled examples"
        );
        assert_eq!(
            run(&["serve", "scenes/", "--demo"])
                .unwrap_err()
                .to_string(),
            "serve: pass a scenes directory or --demo, not both"
        );
    }

    #[test]
    fn serve_names_the_directory_swallowed_by_a_bare_display() {
        let d = tmpdir("swallowed");
        assert_eq!(
            run(&["serve", "--display", &d]).unwrap_err().to_string(),
            format!(
                "serve: --display took '{d}' as the display name; \
                 put the scenes directory before it"
            )
        );
    }

    /// --demo is what puts the bundled examples on the search path.
    #[test]
    fn a_bundled_name_only_resolves_with_demo() {
        let _e = crate::console::repo_examples();
        assert!(
            failure(find_scene("bars", false)).starts_with("no scene 'bars' (searched: cwd"),
            "resolved without --demo"
        );
        assert_eq!(
            find_scene("bars", true).unwrap(),
            concat!(env!("CARGO_MANIFEST_DIR"), "/ghee/examples/bars")
        );
    }

    #[test]
    fn a_scene_path_is_taken_as_written() {
        let _e = crate::console::repo_examples();
        let (_d, path) = scene_file("asis");
        let stem = path.trim_end_matches(".scene.json");
        assert_eq!(find_scene(&path, false).unwrap(), path);
        assert_eq!(find_scene(stem, false).unwrap(), stem);
        assert_eq!(join("/a/b", "c"), "/a/b/c");
        assert_eq!(join("/a/b/", "c"), "/a/b/c");
        assert_eq!(join("/a/b", "/c"), "/c");
    }

    /// `overlay` refuses before it writes anything.
    #[test]
    fn overlay_names_what_it_cannot_set_up() {
        for (tokens, msg) in [
            (
                vec!["overlay", "does-not-exist"],
                "overlay: no display driver 'does-not-exist' (see: ghee displays)",
            ),
            (
                vec![
                    "overlay",
                    "ssd1306",
                    "--display-opts",
                    "height=32",
                    "-o",
                    "panel.bin",
                ],
                "overlay: -o with ssd1306, a panel that takes no firmware file",
            ),
            (
                vec!["overlay", "st7789", "--display-opts", "widht=240"],
                "overlay: no option widht on st7789 (takes: width, height, \
                 x_offset, y_offset, dc, rst, backlight, baudrate, spi_bus, spi_device, \
                 madctl)",
            ),
            // the wire format and the update strategy never reach a kernel setup
            (
                vec![
                    "overlay",
                    "st7789",
                    "--display-opts",
                    "color_bits=12,partial=0",
                ],
                "overlay: the userspace driver alone acts on color_bits, partial (the \
                 kernel setup takes: width, height, x_offset, y_offset, dc, rst, \
                 backlight, baudrate, spi_bus, spi_device, madctl)",
            ),
            (
                vec!["overlay", "ssd1306", "--display-opts", "bus=0"],
                "overlay: the userspace driver alone acts on bus (the kernel setup takes: \
                 height, address, speed)",
            ),
        ] {
            assert_eq!(run(&tokens).unwrap_err().to_string(), msg, "{tokens:?}");
        }
    }

    #[test]
    fn the_stats_line_names_every_destination() {
        let out = ["panel".to_string(), "file".to_string()];
        assert_eq!(
            stats_line(&[0.010, 0.030, 0.020], 2.0, &out),
            "stats: 3 frame(s) in 2.00s, 20.00 ms/frame \
             (min 10.00, max 30.00), 1.5 fps -> panel+file"
        );
    }

    #[test]
    fn the_usage_line_gives_each_figure_that_the_machine_reports() {
        let pi5 = Usage {
            rss: Some(83 << 20 | 1 << 19),
            gpu: Some(12 << 20),
            busy: Some(BTreeMap::from([
                ("bin".to_string(), 2.14),
                ("render".to_string(), 19.9),
            ])),
            machine_free: Some(14892 << 20),
            soc_celsius: Some(52.31),
        };
        assert_eq!(
            usage_line(&pi5),
            "usage: rss 83.5 MiB, gpu 12.0 MiB, gpu busy bin 2.1% render 19.9%, \
             machine free 14892 MiB, soc 52.3 °C"
        );
        let vc4 = Usage {
            rss: Some(40 << 20),
            machine_free: Some(300 << 20),
            soc_celsius: Some(48.0),
            ..Usage::default()
        };
        assert_eq!(
            usage_line(&vc4),
            "usage: rss 40.0 MiB, machine free 300 MiB, soc 48.0 °C"
        );
    }

    #[test]
    fn a_serve_run_names_the_console_when_nothing_else_took_the_frames() {
        assert_eq!(serve_out(&[]), ["console"]);
        let panel = ["panel".to_string()];
        assert_eq!(serve_out(&panel), panel);
    }

    #[test]
    fn multiple_frames_into_one_filename_is_refused() {
        let a = ns(&["render", "s", "-o", "one.png", "--frames", "4"]);
        let e = failure(file_sink(&a));
        assert!(
            e.starts_with("render: multiple frames into one filename"),
            "{e}"
        );
        // one frame into one name overwrites. A # run numbers the files.
        for tokens in [
            vec!["render", "s", "-o", "one.png"],
            vec!["render", "s", "-o", "f.####.png", "--frames", "4"],
        ] {
            assert!(matches!(
                file_sink(&ns(&tokens)).unwrap(),
                FileSink::Images(_)
            ));
        }
        assert!(matches!(
            file_sink(&ns(&["render", "s", "-o", "-"])).unwrap(),
            FileSink::Raw(_)
        ));
        assert!(matches!(
            file_sink(&ns(&["render", "s"])).unwrap(),
            FileSink::None
        ));
    }

    #[test]
    fn geometry_takes_size_then_the_panels_own() {
        assert_eq!(
            geometry(&ns(&["render", "s"]), &Panel::None).unwrap(),
            ((320, 240), 4)
        );
        let a = ns(&["render", "s", "--size", "640x480", "--components", "3"]);
        assert_eq!(geometry(&a, &Panel::None).unwrap(), ((640, 480), 3));
    }

    #[test]
    fn serve_keeps_renders_defaults_for_the_flags_it_does_not_offer() {
        let a = ns(&["serve", "scenes/"]);
        assert_eq!(a.cmd, "serve");
        assert_eq!(a.scene.as_deref(), Some("scenes/"));
        assert!(a.serve && !a.stats);
        assert_eq!((a.port, a.bitrate), (8090, 10_000_000));
        assert_eq!((a.preview_max, a.preview_fps, a.stream_max), (480, 15, 0));
        assert_eq!(a.host, "127.0.0.1");
        assert_eq!((a.components, a.write_buffer), (4, 16));
        assert!(a.output.is_none() && a.h264.is_none() && a.filters.is_none());
        assert_eq!(frames_of(&a), 0);
    }

    #[test]
    fn render_keeps_its_own_console_defaults() {
        let a = ns(&["render", "s", "-o", "x.png"]);
        assert_eq!((a.port, a.bitrate), (8090, 4_000_000));
        assert_eq!((a.components, a.write_buffer), (4, 16));
        assert!(!a.serve && !a.demo && !a.rgb565);
        assert_eq!(ns(&["render", "s", "--bitrate", "123"]).bitrate, 123);
    }

    #[test]
    fn serve_takes_every_flag_it_always_took() {
        let a = ns(&[
            "serve",
            "scenes/",
            "--port",
            "9",
            "--host",
            "0.0.0.0",
            "--size",
            "80x60",
            "--preview-fps",
            "5",
            "--preview-max",
            "120",
            "--stream-max",
            "320",
            "--bitrate",
            "123",
            "--rgb565",
            "--state-dir",
            "/srv/state",
            "--",
            "--text",
            "hi",
        ]);
        assert_eq!(
            (a.port, a.host.as_str(), a.size),
            (9, "0.0.0.0", Some((80, 60)))
        );
        assert_eq!(a.state_dir, Some(std::path::PathBuf::from("/srv/state")));
        assert_eq!((a.preview_fps, a.preview_max, a.stream_max), (5, 120, 320));
        assert_eq!((a.bitrate, a.rgb565), (123, true));
        assert_eq!(a.scene_args, argv(&["--text", "hi"]));
    }

    #[test]
    fn a_bare_display_is_the_scanout() {
        let a = ns(&["render", "s", "--display"]);
        assert_eq!(a.display.as_deref(), Some("kms"));
        assert!(matches!(panel_of(&a).unwrap(), Panel::Kms));
        assert_eq!(
            ns(&["render", "s", "--display", "st7789"])
                .display
                .as_deref(),
            Some("st7789")
        );
    }

    #[test]
    fn a_panel_gives_a_scene_its_size_before_it_opens() {
        let ssd1306 = get_driver("ssd1306").unwrap();
        let a = ns(&["render", "s.py", "--display", "ssd1306"]);
        assert_eq!(prospective_size(&a), (ssd1306.size)(&[]).ok());
        assert!(prospective_size(&a).is_some());
    }

    #[test]
    fn a_console_takes_the_frame_rate_cap_it_is_given() {
        for (tokens, capped) in [
            (vec!["serve", "scenes/", "--fps", "24"], true),
            (vec!["render", "s", "--serve", "--fps", "106"], true),
            (vec!["serve", "scenes/", "--fps", "0"], false),
            (vec!["serve", "scenes/"], false),
        ] {
            assert_eq!(console_pacer(&ns(&tokens)).is_some(), capped, "{tokens:?}");
        }
    }

    #[test]
    fn a_pacer_follows_the_sink() {
        assert!(pacer(&ns(&["render", "s", "-o", "x.png"]), &Panel::None).is_none());
        assert!(pacer(&ns(&["render", "s", "--h264", "x"]), &Panel::None).is_some());
        assert!(pacer(&ns(&["render", "s", "--display"]), &Panel::Kms).is_none());
        let paced = ns(&["render", "s", "--display", "--fps", "24"]);
        assert!(pacer(&paced, &Panel::Kms).is_some());
        assert_eq!(h264_fps(&ns(&["render", "s"])), 30);
        assert_eq!(h264_fps(&ns(&["render", "s", "--fps", "24.9"])), 24);
        assert_eq!(h264_fps(&ns(&["render", "s", "--fps", "0"])), 30);
    }

    #[test]
    fn the_scene_loaded_for_a_render_carries_its_filters() {
        let (_d, path) = scene_file("filters");
        let (scene, process) = scene_of(&ns(&["render", &path, "--filters", "dither"])).unwrap();
        assert_eq!(scene.passes.len(), 2);
        assert_eq!(scene.passes[1].name, "dither");
        assert!(process.is_none()); // a JSON scene feeds itself
        let e = failure(scene_of(&ns(&["render", &path, "--filters", "nope"])));
        assert!(e.starts_with("unknown filter: nope (known: dither"), "{e}");
    }

    #[test]
    fn serve_mounts_the_directory_it_was_given() {
        // the console reports an empty directory before it binds anything.
        // main() puts the command name in front.
        let d = tmpdir("serve-dir");
        assert_eq!(
            failure(run(&["serve", &d])),
            format!("no .scene.json in {d}")
        );
    }

    #[test]
    fn render_serve_on_one_scene_serves_that_scene_alone() {
        let _e = crate::console::repo_examples();
        let (_d, path) = scene_file("one");
        let a = ns(&["render", &path, "--serve"]);
        let store = one_scene_store(&a, &path).unwrap();
        assert_eq!(store.names(), ["s"]);
        assert!(!store.demo);
        assert!(!in_examples(&path));
    }

    /// A bundled scene served on its own keeps its state file as a demo,
    /// whichever way a command reaches it.
    #[test]
    fn a_bundled_scene_keeps_its_state_file_as_a_demo() {
        let _e = crate::console::repo_examples();
        let path = "ghee/examples/bars.scene.json";
        assert!(in_examples(path));
        let a = ns(&["render", "--demo", "bars", "--serve"]);
        assert!(one_scene_store(&a, path).unwrap().demo);
    }
}
