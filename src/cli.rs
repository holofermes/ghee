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
//! ghee isf import shader.fs -o shader.scene.json         # import an ISF shader
//! ghee displays                                          # list display drivers
//! ```

use std::cell::RefCell;
use std::ffi::c_int;
use std::fs::File;
use std::io::{stdout, Stdout, Write};
use std::path::Path;
use std::sync::atomic::{AtomicI32, Ordering};

use anyhow::{Context, Result};
use clap::error::{ContextKind, ErrorKind};
use clap::{Args, Parser, Subcommand};
use serde_json::{Map, Value};

use crate::bench::{self, BenchArgs};
use crate::console::examples_dir;
use crate::console::server;
use crate::console::store::SceneStore;
use crate::drivers::{available_drivers, driver_options, get_driver, DisplayDriver};
use crate::engine::{probe_display, Feed, Renderer};
use crate::feeds::demo_feed;
use crate::filters::{apply_filters, load_filters};
use crate::h264::H264Stream;
use crate::isf::{read_isf, slug, to_filter, to_scene};
use crate::looper::{ControllerSlot, Frame, Loop, Output, Pacer, SceneController};
use crate::output::{expand_pattern, AdaptivePacer, FixedPacer, ImageSequence, RawStream, Scanout};
use crate::procedural::process::{self, SceneProcess};
use crate::procedural::{load_scene, scene_name};
use crate::scene::Scene;
use crate::stats::rate;

/// A message main() prints as written, status 1. main() puts the command name
/// in front of any other error.
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
/// -> bool). A value literal_eval refuses stays the raw string. `what` names
/// the flag an error comes from.
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

/// (name, path) for every scene this command loads: the scene it renders, or
/// each scene the console mounts.
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
    Ok(SceneStore::new(&dir, true, Vec::new()).scenes)
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
    print!("{}", process::help(&process::declare(&set), &prog));
    std::process::exit(0);
}

/// The scene this run renders, and the process behind it. The filters apply in
/// order: the ones the scene declares, then the ones --filters adds.
fn scene_of(a: &Ns) -> Result<(Scene, Option<SceneProcess>)> {
    let path = find_scene(text(&a.scene), a.demo)?;
    let (mut scene, mut process) = match process::executable(&path) {
        Some(exe) => {
            let (scene, fed) = SceneProcess::start(&exe, &a.scene_args)?;
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
    }
    Ok((scene, process))
}

/// What --display names: the KMS scanout, or the driver instance.
/// --display-opts builds the driver.
enum Panel {
    None,
    Kms,
    Driver(Box<dyn DisplayDriver>),
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
        return Ok(Panel::Kms);
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

/// (size, components) the render runs at. A scanout takes the connected mode.
/// A panel takes its own geometry and pixel layout. Both win over --size.
/// Everything else takes --size.
fn geometry(a: &Ns, panel: &Panel) -> Result<((i32, i32), i32)> {
    match panel {
        Panel::Kms => match probe_display(a.dri.as_deref()).or(a.size) {
            Some(size) => Ok((size, a.components)),
            None => Err(exit(format!(
                "{}: no connected display to size from; pass --size WxH",
                a.cmd
            ))),
        },
        Panel::Driver(d) => Ok(((d.width(), d.height()), d.components())),
        Panel::None => Ok((a.size.unwrap_or((320, 240)), a.components)),
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
/// panel paces on its own bus speed. The KMS flip paces itself at the display
/// refresh. An H264 stream holds a fixed rate for its timestamps.
fn pacer(a: &Ns, panel: &Panel) -> Option<Box<dyn Pacer>> {
    let fixed = a
        .fps
        .filter(|f| *f != 0.0)
        .map(|f| Box::new(FixedPacer::new(f)) as Box<dyn Pacer>);
    match panel {
        Panel::Kms => fixed,
        Panel::Driver(_) => {
            fixed.or_else(|| Some(Box::new(AdaptivePacer::new())))
        }
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

/// A driver writes to a physical panel (drivers::base). The loop reaches a
/// driver through this Output.
struct PanelSink(Box<dyn DisplayDriver>);

impl Output for PanelSink {
    fn dest(&self) -> &str {
        self.0.dest()
    }

    fn persists(&self) -> bool {
        self.0.persists()
    }

    fn write(&mut self, frame: Frame<'_>) -> Result<()> {
        self.0.write(frame.data)
    }

    fn turn_on(&mut self) -> Result<()> {
        self.0.turn_on()
    }

    fn turn_off(&mut self) -> Result<()> {
        self.0.turn_off()
    }

    fn clear(&mut self) -> Result<()> {
        self.0.clear()
    }

    fn close(&mut self) -> Result<()> {
        self.0.close()
    }
}

/// The sinks in the order the flags add them up: the display, then -o, then
/// --h264.
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

/// One render into however many sinks the flags ask for.
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
    // the scene loads first: it owns the tokens after `--` and their --help. A
    // scene the loader rejects fails before a panel opens.
    let scene = if a.serve { None } else { Some(scene_of(a)?) };
    let panel = panel_of(a)?;
    let kms = matches!(panel, Panel::Kms);
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
        Panel::Driver(d) => Some(PanelSink(d)),
        _ => None,
    };
    let mut file = file_sink(a)?;
    if a.serve {
        // The console draws on this thread, so it takes the sinks as they are.
        // A panel driver's SPI and GPIO handles never leave the thread that
        // opened them. The scanout is never a sink here. It is a mode of the
        // Renderer itself.
        let mut sinks: Vec<Box<dyn Output>> = Vec::new();
        let mut native_format = None;
        if let Some(panel) = sink {
            native_format = Some(panel.0.native_format().to_string());
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
        eprintln!(
            "h264: {} {w}x{h}@{fps} {}bps -> {path}",
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
        if a.stats {
            lp.on_frame = Some(Box::new(|_, cost| times.borrow_mut().push(cost)));
        }
        let ran = lp.run();
        let out: Vec<String> = lp.out().iter().map(|d| (*d).to_string()).collect();
        (ran, lp.drawn, lp.elapsed, out)
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
        println!("wrote {drawn} frame(s) to {}", text(&a.output));
    }
    if given(&a.h264) && text(&a.h264) != "-" {
        eprintln!("wrote {drawn} frame(s) to {}", text(&a.h264));
    }
    if a.stats {
        print_stats(&times.borrow(), elapsed, &out);
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
        store,
        size,
        components,
        sinks,
        native_format,
        display: kms,
        rgb565: a.rgb565,
        port: a.port,
        host: a.host.clone(),
        preview_max: a.preview_max,
        preview_fps: f64::from(a.preview_fps),
        stream_max: a.stream_max,
        bitrate: i64::from(a.bitrate),
        scene_args: a.scene_args.clone(),
        frames: frames_of(a).max(0) as u64,
        timestep: a.timestep,
        dri_device: dri,
        on_stats: a.stats.then(serve_stats),
        ..Default::default()
    });
    if SIGNALLED.load(Ordering::SeqCst) == SIGTERM {
        std::process::exit(143);
    }
    ran
}

/// A store for the single scene `render --serve` names. A bundled example
/// stays read-only, whichever way a command reaches it.
fn one_scene_store(a: &Ns, path: &str) -> Result<SceneStore> {
    SceneStore::one(path, in_examples(path), a.scene_args.clone())
        .map_err(|e| exit(format!("{}: {e}", a.cmd)))
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
/// without the paced sleeps. fps is frames over elapsed time, the same number
/// the console publishes.
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

fn print_stats(times: &[f64], wall: f64, out: &[String]) {
    if times.is_empty() {
        return;
    }
    eprintln!("{}", stats_line(times, wall, out));
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
    Box::new(|times, wall, out| print_stats(times, wall, &serve_out(out)))
}

fn cmd_isf_import(a: &ImportArgs) -> Result<()> {
    let (source, name, vs) =
        read_isf(&a.source).map_err(|e| exit(format!("isf import: {}: {e}", a.source)))?;
    let (body, what, warnings) = if a.as_filter {
        let (out, warnings) = to_filter(&source, &name, a.size, vs).map_err(isf_failed)?;
        let named = out.get("name").and_then(Value::as_str).unwrap_or_default();
        let what = format!("filter '{named}'");
        (serde_json::to_string_pretty(&out)?, what, warnings)
    } else {
        let (scene, warnings) = to_scene(&source, &name, a.size, vs).map_err(isf_failed)?;
        let what = format!("scene, {} pass(es)", scene.passes.len());
        (
            serde_json::to_string_pretty(&scene.to_dict())?,
            what,
            warnings,
        )
    };
    for w in warnings {
        eprintln!("isf import: {name}: {w}");
    }
    let path = match a.output.as_deref().filter(|p| !p.is_empty()) {
        Some(p) => p.to_string(),
        None => format!(
            "{}.{}.json",
            slug(&name),
            if a.as_filter { "filter" } else { "scene" }
        ),
    };
    let mut f = File::create(&path).with_context(|| path.clone())?;
    f.write_all(body.as_bytes())?;
    f.write_all(b"\n")?;
    println!("wrote {what} to {path}");
    Ok(())
}

fn isf_failed(e: anyhow::Error) -> anyhow::Error {
    exit(format!("isf import: {e:#}"))
}

fn cmd_displays() -> Result<()> {
    // every card with a connected display, since an SPI panel under a
    // kernel display driver is a card beside the HDMI one
    let mut connected = 0;
    for dev in crate::engine::probe_dri_cards() {
        if let Some((w, h)) = probe_display(Some(&dev)) {
            println!("{:12} {dev} connected, {w}x{h}", "kms");
            connected += 1;
        }
    }
    if connected == 0 {
        println!("{:12} no connected display", "kms");
    }
    for (name, status) in available_drivers() {
        println!("{name:12} {status}");
        if status != "ok" {
            continue;
        }
        let opts: Vec<String> = driver_options(get_driver(name)?)
            .iter()
            .map(|o| match o.default {
                Some(v) => format!("{}={v}", o.name),
                None => format!("{}=None", o.name),
            })
            .collect();
        if !opts.is_empty() {
            // a driver's own defaults: an SPI or I2C panel answers no probe
            println!("{:12} defaults: --display-opts {}", "", opts.join(","));
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
                      (ghee render --demo hero -- --no-dither); `-- --help` lists what it takes"
    )]
    Render(RenderArgs),

    #[command(
        about = "run the editor console",
        allow_negative_numbers = true,
        after_help = "the console is plain HTTP: GET /stats /scene /scenes /frame.png \
                      /stream /h264, POST /set /switch /next /prev /scene /save /ping. \
                      Anything after `--` goes to every scene the console loads, and \
                      each keeps what it takes; `-- --help` lists what takes what"
    )]
    Serve(ServeArgs),

    #[command(about = "Interactive Shader Format (ISF) shaders")]
    Isf(IsfArgs),

    #[command(about = "list connected cards, display drivers, and their options")]
    Displays,

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
                (or 'kms') for the connected KMS display, HDMI or a panel the \
                kernel drives (zero-copy scanout; needs DRM master)"
    )]
    display: Option<String>,

    #[arg(
        long,
        value_name = "K=V,...",
        help = "options for the --display driver, e.g. width=240,height=240,\
                y_offset=80,backlight=26; `ghee displays` lists what each one takes"
    )]
    display_opts: Option<String>,

    #[arg(
        long,
        help = "display scanout in 16-bit color: halves the GPU store and scanout \
                bandwidth (worth it on VC4-class boards); slight banding in gradients"
    )]
    rgb565: bool,

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
        help = "WxH to render at; a display or panel sets its own (default 320x240)"
    )]
    size: Option<(i32, i32)>,

    #[arg(
        long,
        help = "DRI device, e.g. /dev/dri/renderD128; with --display it picks \
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
        default_value = "127.0.0.1",
        help = "bind address; 0.0.0.0 exposes the console (no auth) to the network"
    )]
    host: String,

    #[arg(
        long,
        default_value_t = 480,
        help = "longest side of the browser preview in px (downscaled; default 480)"
    )]
    preview_max: i32,

    #[arg(
        long,
        default_value_t = 15,
        help = "browser preview refresh cap; the display stays full speed (default 15)"
    )]
    preview_fps: i32,

    #[arg(
        long,
        default_value_t = 0,
        help = "cap the /h264 stream's longest side (GPU downscale; \
                default 0 = full render size)"
    )]
    stream_max: i32,
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
                stdout (h*w*components, no header; pipe to ffmpeg/mpv)"
    )]
    output: Option<String>,

    #[arg(
        long,
        value_name = "PATH",
        help = "write an Annex-B H264 stream to PATH ('-' = stdout, pipe to mpv); \
                hardware zero-copy on Pi 1..Pi 4, ffmpeg/libx264 elsewhere"
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
        value_name = "K=V,...",
        help = "extra options passed to the image encoder, e.g. quality=90 or \
                lossless=True (ignored for -o -)"
    )]
    save_opts: Option<String>,

    #[arg(
        long,
        default_value_t = 16,
        value_name = "N",
        help = "frames buffered for the async image writer; the render only waits \
                when the writer falls N frames behind; memory cost is N * w * h * \
                components bytes"
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

    #[arg(long, help = "print a timing summary when the render ends")]
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
struct IsfArgs {
    #[command(subcommand)]
    isf_cmd: IsfCommand,
}

#[derive(Subcommand)]
enum IsfCommand {
    #[command(
        about = "convert an ISF shader to a scene or filter",
        allow_negative_numbers = true
    )]
    Import(ImportArgs),
}

#[derive(Args)]
struct ImportArgs {
    #[arg(help = "a .fs file, or an http(s) URL to one")]
    source: String,

    #[arg(
        short,
        long,
        help = "output path (default: the shader's name, .scene.json or .filter.json)"
    )]
    output: Option<String>,

    #[arg(
        long,
        default_value = "320x240",
        value_parser = isf_size,
        help = "WxH the pass WIDTH/HEIGHT equations are evaluated against, since \
                ghee pass sizes are fixed at import (default 320x240)"
    )]
    size: (i64, i64),

    #[arg(
        long,
        help = "write a ghee filter (a single-pass ISF whose image input becomes \
                iChannel0, the frame so far) instead of a scene"
    )]
    as_filter: bool,
}

/// ghee::isf evaluates its size equations in i64.
fn isf_size(s: &str) -> Result<(i64, i64), String> {
    size(s).map(|(w, h)| (i64::from(w), i64::from(h)))
}

/// The namespace both render and serve resolve to. serve is render --serve
/// over a directory. It carries render's whole namespace, and every flag it
/// does not offer keeps its default.
struct Ns {
    cmd: &'static str,
    scene: Option<String>,
    demo: bool,
    output: Option<String>,
    h264: Option<String>,
    serve: bool,
    components: i32,
    save_opts: Option<String>,
    write_buffer: i64,
    fps: Option<f64>,
    filters: Option<String>,
    stats: bool,
    display: Option<String>,
    display_opts: Option<String>,
    rgb565: bool,
    frames: Option<i64>,
    timestep: Option<f64>,
    size: Option<(i32, i32)>,
    dri: Option<String>,
    port: u16,
    host: String,
    preview_max: i32,
    preview_fps: i32,
    stream_max: i32,
    bitrate: i32,
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
            save_opts: self.save_opts,
            write_buffer: self.write_buffer,
            fps: self.fps,
            filters: self.filters,
            stats: self.stats,
            display: self.render.display,
            display_opts: self.render.display_opts,
            rgb565: self.render.rgb565,
            frames: self.render.frames,
            timestep: self.render.timestep,
            size: self.render.size,
            dri: self.render.dri,
            port: self.console.port,
            host: self.console.host,
            preview_max: self.console.preview_max,
            preview_fps: self.console.preview_fps,
            stream_max: self.console.stream_max,
            bitrate: self.bitrate,
            scene_args,
        }
    }
}

impl ServeArgs {
    fn ns(self, scene_args: Vec<String>) -> Ns {
        Ns {
            cmd: "serve",
            // --demo leaves it None: the bundled examples, read-only
            scene: self.scenes,
            demo: self.demo,
            output: None,
            h264: None,
            serve: true,
            components: 4,
            save_opts: None,
            write_buffer: 16,
            fps: None,
            filters: None,
            stats: false,
            display: self.render.display,
            display_opts: self.render.display_opts,
            rgb565: self.render.rgb565,
            frames: self.render.frames,
            timestep: self.render.timestep,
            size: self.render.size,
            dri: self.render.dri,
            port: self.console.port,
            host: self.console.host,
            preview_max: self.console.preview_max,
            preview_fps: self.console.preview_fps,
            stream_max: self.console.stream_max,
            bitrate: self.bitrate,
            scene_args,
        }
    }
}

/// The command name a failure reports under, and the result of that command.
fn dispatch(cli: Cli, scene_args: Vec<String>) -> (&'static str, Result<()>) {
    let cmd = match &cli.cmd {
        Command::Render(_) => "render",
        Command::Serve(_) => "serve",
        Command::Isf(_) => "isf",
        Command::Displays => "displays",
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
        Command::Isf(a) => match &a.isf_cmd {
            IsfCommand::Import(i) => cmd_isf_import(i),
        },
        Command::Displays => cmd_displays(),
        Command::Bench(a) => bench::run(&a),
    };
    (cmd, ran)
}

/// The top-level parser reports a token no subcommand claims. The line names
/// `ghee` itself and the status is 2.
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
        "code":"void main(){ gl_FragColor = vec4(0.25,0.5,0.75,1.0); }"}]}"#;

    fn argv(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|s| (*s).to_string()).collect()
    }

    fn parse(tokens: &[&str]) -> (Cli, Vec<String>) {
        let (own, scene_args) = split_scene_args(&argv(tokens));
        let cli = Cli::try_parse_from(std::iter::once("ghee".to_string()).chain(own)).unwrap();
        (cli, scene_args)
    }

    /// The namespace a command line resolves to, without running it.
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

    fn tmpdir(tag: &str) -> String {
        let d = std::env::temp_dir().join(format!("ghee-cli-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d.to_string_lossy().into_owned()
    }

    fn scene_file(tag: &str) -> String {
        let path = format!("{}/s.scene.json", tmpdir(tag));
        fs::write(&path, SCENE).unwrap();
        path
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
    fn own_flags_are_still_validated() {
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
    fn an_unknown_display_opt_names_the_options_the_driver_takes() {
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
    fn a_render_with_nowhere_to_go_says_so() {
        let path = scene_file("nowhere");
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
    fn serve_names_the_directory_a_bare_display_swallowed() {
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
        std::env::set_var("GHEE_EXAMPLES", "ghee/examples");
        assert!(
            failure(find_scene("bars", false)).starts_with("no scene 'bars' (searched: cwd"),
            "resolved without --demo"
        );
        assert_eq!(find_scene("bars", true).unwrap(), "ghee/examples/bars");
    }

    #[test]
    fn a_scene_path_is_taken_as_written() {
        let path = scene_file("asis");
        let stem = path.trim_end_matches(".scene.json");
        assert_eq!(find_scene(&path, false).unwrap(), path);
        assert_eq!(find_scene(stem, false).unwrap(), stem);
        assert_eq!(join("/a/b", "c"), "/a/b/c");
        assert_eq!(join("/a/b/", "c"), "/a/b/c");
        assert_eq!(join("/a/b", "/c"), "/c");
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
            "--",
            "--text",
            "hi",
        ]);
        assert_eq!(
            (a.port, a.host.as_str(), a.size),
            (9, "0.0.0.0", Some((80, 60)))
        );
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
    fn the_scene_the_render_loads_carries_its_filters() {
        let path = scene_file("filters");
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
        let path = scene_file("one");
        let a = ns(&["render", &path, "--serve"]);
        let store = one_scene_store(&a, &path).unwrap();
        assert_eq!(store.names(), ["s"]);
        assert!(!store.read_only);
        assert!(!in_examples(&path));
    }

    /// A bundled scene served on its own stays read-only, whichever way a
    /// command reaches it.
    #[test]
    fn a_bundled_scene_is_read_only() {
        std::env::set_var("GHEE_EXAMPLES", "ghee/examples");
        let path = "ghee/examples/bars.scene.json";
        assert!(in_examples(path));
        let a = ns(&["render", "--demo", "bars", "--serve"]);
        assert!(one_scene_store(&a, path).unwrap().read_only);
    }
}
