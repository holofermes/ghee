//! The `ghee` command line.
//!
//! ```text
//! ghee render scene.json -o frame.####.jpg --frames 500  # image sequence
//! ghee render scene.json -o - --frames 300 | ffmpeg ...  # raw frames to a pipe
//! ghee render scene.json --display kms                   # to the connected display, until Ctrl-C
//! ghee render scene.json --h264 - | mpv -                # H264 to a pipe
//! ghee render scene.json --display kms --serve           # the display plus the console
//! ghee serve scenes/                                     # editor console
//! ghee serve --demo                                      # console on the bundled examples
//! ghee render --demo bars -o frame.png                   # a bundled example by name
//! ghee displays                                          # list the connected displays
//! ghee overlay st7789                                    # kernel setup for a panel
//! ```

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{stdout, Stdout, Write};
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::error::{ContextKind, ErrorKind};
use clap::{Args, CommandFactory, Parser, Subcommand};
use serde_json::{Map, Value};

use crate::console::server;
use crate::console::state;
use crate::console::store::{self, SceneStore};
use crate::console::{examples_dir, is_examples_dir};
use crate::control::{self, Control};
use crate::engine::{probe_display, Card, Feed, Renderer, SetupError, DEFAULT_SIZE};
use crate::feeds::demo_feed;
use crate::filters::{self, apply_filters, Filter};
use crate::golden;
use crate::h264::H264Stream;
use crate::looper::{ControllerSlot, Loop, Output, Pacer, SceneController};
use crate::output::{
    self, expand_pattern, is_image_name, AlphaForm, ImageSequence, RawStream, Scanout, Sinks,
};
use crate::overlay;
use crate::procedural::process::{self, Program, SceneProcess};
use crate::procedural::{load_scene, scene_name};
use crate::report;
use crate::scene::{Pass, Reach, Scene};
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
        _ => Err("wants 0, 90, 180 or 270".to_string()),
    }
}

/// --brightness's level, 0 to 1.
fn level(s: &str) -> Result<f64, String> {
    match s.trim().parse::<f64>() {
        Ok(v) if (0.0..=1.0).contains(&v) => Ok(v),
        _ => Err("wants a level from 0 to 1".to_string()),
    }
}

fn size(s: &str) -> Result<(i32, i32), String> {
    let bad = || "wants WxH".to_string();
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
        None if a.demo => examples_dir().unwrap_or_default(),
        // the console's own scenes are scene files, which take no args
        None => return Ok(Vec::new()),
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
    let prog = match a.cmd.as_str() {
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
/// order: the scene's own first, then --filters. A scene process sizes
/// itself on the display's card, which `card` holds from then on.
fn scene_of(a: &Ns, card: &mut Option<Card>) -> Result<(Scene, Option<SceneProcess>)> {
    let path = match &a.here {
        Some(here) => here.path.clone(),
        None => find_scene(text(&a.scene), a.demo)?,
    };
    report::detail("scene", &path);
    let program = a.here.as_ref().map(|h| Arc::clone(&h.program));
    let exe = process::executable(&path).filter(|_| program.is_none());
    let (mut scene, mut process) = if program.is_some() || exe.is_some() {
        let turned = a.rotate == 90 || a.rotate == 270;
        let (w, h) = prospective_size(a, card)?;
        let size = if turned { (h, w) } else { (w, h) };
        let (scene, fed) = match (program, exe) {
            (Some(program), _) => SceneProcess::here(&path, program, Reach::Any, size, |_| {})?,
            (None, Some(exe)) => {
                SceneProcess::start(&exe, &a.scene_args, Reach::Any, size, |_| {})?
            }
            (None, None) => unreachable!("one of the two is there"),
        };
        (scene, Some(fed))
    } else {
        (load_scene(&path, &a.scene_args, true)?, None)
    };
    let given = Filter::parse_list(text(&a.filters))
        .map_err(|e| exit(format!("{}: --filters: {e}", a.cmd)))?;
    let wanted = filters::chain(&scene.filters, &given);
    apply_filters(&mut scene, &wanted, None).map_err(|e| match e.is::<filters::Unknown>() {
        true => exit(e.to_string()),
        false => e,
    })?;
    if !a.original {
        scene = with_saved_edits(a, &path, scene)?;
    }
    // the process addresses the params of the final scene, filters included
    if let Some(fed) = process.as_mut() {
        fed.bind(&scene);
        if a.timestep.is_some() {
            fed.pin();
        }
    }
    Ok((scene, process))
}

/// `scene`, the scene of `path`, with the edits that a console saved for it
/// laid on, as a console opens it.
fn with_saved_edits(a: &Ns, path: &str, scene: Scene) -> Result<Scene> {
    let mut store = match &a.here {
        Some(here) => SceneStore::here(path, Arc::clone(&here.program), Vec::new()),
        None => one_scene_store(a, path)?,
    };
    store.keep_in(
        a.state_dir.clone().or_else(state::default_dir),
        a.edits_dir.clone().or_else(state::default_edits_dir),
    );
    let Some(name) = store.names().into_iter().next() else {
        return Ok(scene);
    };
    match store.read_state(&name) {
        Some(edits) => Ok(store::working(&name, &scene, &edits).0),
        None => Ok(scene),
    }
}

/// What --display names: the KMS scanout, with the mode size of its
/// display, or nothing.
enum Panel {
    None,
    Kms((i32, i32)),
}

fn panel_of(a: &Ns, card: &mut Option<Card>) -> Result<Panel> {
    if !given(&a.display) {
        return Ok(Panel::None);
    }
    kms_named(a)?;
    let (w, h) = display_card(a, card)?.size();
    Ok(Panel::Kms((w as i32, h as i32)))
}

/// The card of --display in `card`, opened at the first call and held from
/// then on.
fn display_card<'c>(a: &Ns, card: &'c mut Option<Card>) -> Result<&'c Card> {
    let open = match card.take() {
        Some(open) => open,
        None => Card::open(a.dri.as_deref()).map_err(|e| exit(format!("{}: {e}", a.cmd)))?,
    };
    Ok(card.insert(open))
}

/// Err unless the given --display names kms.
fn kms_named(a: &Ns) -> Result<()> {
    let name = text(&a.display);
    if name == "kms" {
        return Ok(());
    }
    // --display takes an optional value, so it reads a path after it as a
    // display name
    let hint = if Path::new(name).is_dir() {
        ", and a scenes directory goes before it"
    } else {
        ""
    };
    Err(exit(format!(
        "{}: no display '{name}' (takes: kms){hint}",
        a.cmd
    )))
}

/// A scene builds against this size, worked out before the renderer opens,
/// as geometry() works it out. A scene process needs it before it writes its
/// scene.
fn prospective_size(a: &Ns, card: &mut Option<Card>) -> Result<(i32, i32)> {
    Ok(geometry(a, &panel_of(a, card)?).0)
}

/// (size, components) of the render.
/// A scanout takes --size and scales it across the mode of its display on
/// the way out, or the mode itself when there is no --size.
/// Everything else takes --size.
fn geometry(a: &Ns, panel: &Panel) -> ((i32, i32), i32) {
    match panel {
        // ghee_display_create renders RGBA whatever --components asks for, so
        // the readback writes four bytes a pixel.
        Panel::Kms(mode) => (a.size.unwrap_or(*mode), 4),
        Panel::None => (a.size.unwrap_or(DEFAULT_SIZE), a.components),
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
        return Ok(FileSink::Raw(RawStream::with_alpha(
            stdout(),
            a.alpha.unwrap_or_default(),
        )));
    }
    if !given(&a.output) {
        return Ok(FileSink::None);
    }
    let pattern = text(&a.output);
    if a.frames.is_some_and(|n| n > 1) && expand_pattern(pattern, 0)? == pattern {
        return Err(exit(format!(
            "{}: multiple frames into one filename; number them with \
             a # run (frame.####.png)",
            a.cmd
        )));
    }
    let save_opts = match given(&a.save_opts) {
        true => Some(parse_kv(
            text(&a.save_opts),
            &format!("{}: --save-opts", a.cmd),
        )?),
        false => None,
    };
    if a.alpha == Some(AlphaForm::Premultiplied) {
        return Err(exit(format!(
            "{}: --alpha premultiplied writes raw frames, -o -; an image file holds \
             alpha unpremultiplied",
            a.cmd
        )));
    }
    Ok(FileSink::Images(ImageSequence::with_opts(
        pattern,
        0,
        save_opts,
        usize::try_from(a.write_buffer).unwrap_or(0),
    )))
}

/// The pacer of the render: output::pacer() over what takes its frames.
fn pacer(a: &Ns, panel: &Panel) -> Option<Box<dyn Pacer>> {
    let sinks = match panel {
        Panel::Kms(_) => Sinks::Display,
        Panel::None if given(&a.h264) => Sinks::Stream(h264_fps(a)),
        Panel::None => Sinks::Files,
    };
    output::pacer(a.fps, sinks)
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

/// The sinks in flag order: the display, then -o, then --h264.
fn sink_list<'a>(
    scanout: &'a mut Option<Scanout>,
    file: &'a mut FileSink,
    h264: &'a mut Option<H264Stream>,
) -> Vec<&'a mut dyn Output> {
    let mut sinks: Vec<&mut dyn Output> = Vec::new();
    if let Some(s) = scanout.as_mut() {
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
    render(a, None)
}

/// cmd_render(), with `card` open already for --display.
fn render(a: &Ns, card: Option<Card>) -> Result<()> {
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
    if a.alpha.is_some() && a.components == 3 {
        return Err(exit(format!(
            "{}: --alpha with --components 3; an RGB frame holds no alpha",
            a.cmd
        )));
    }
    let pattern = text(&a.output);
    if given(&a.output) && pattern != "-" {
        if !is_image_name(pattern) {
            return Err(exit(format!(
                "{}: -o wants an image pattern such as frame.####.png, or '-', \
                 got '{pattern}'; a display takes --display",
                a.cmd
            )));
        }
        if a.alpha.is_some() && !pattern.to_lowercase().ends_with(".png") {
            return Err(exit(format!(
                "{}: --alpha with a .jpg; a JPEG holds no alpha",
                a.cmd
            )));
        }
    }
    report::apply(a);
    // the scene loads first: it owns the tokens after `--` and their --help. A
    // rejected scene fails before a display opens.
    let mut card = card;
    let scene = if a.serve {
        None
    } else {
        Some(scene_of(a, &mut card)?)
    };
    let panel = panel_of(a, &mut card)?;
    let kms = matches!(panel, Panel::Kms(_));
    if a.brightness.is_some() && !kms {
        return Err(exit(format!("{}: --brightness needs --display kms", a.cmd)));
    }
    if a.drop && !kms {
        return Err(exit(format!("{}: --drop needs --display kms", a.cmd)));
    }
    if matches!(panel, Panel::None) && !given(&a.output) && !given(&a.h264) && !a.serve {
        return Err(exit(format!(
            "{}: pass -o PATTERN, --display, --h264 PATH or --serve",
            a.cmd
        )));
    }
    let dri = a.dri.clone();
    let ((w, h), components) = geometry(a, &panel);
    let pace = pacer(a, &panel);
    let mut scanout = kms.then_some(Scanout);
    let mut file = file_sink(a)?;
    if a.serve {
        // The console draws on this thread, so it takes the sinks as they are.
        // The scanout is never a sink here. It is a mode of the Renderer
        // itself.
        let mut sinks: Vec<Box<dyn Output>> = Vec::new();
        match file {
            FileSink::Raw(s) => sinks.push(Box::new(s)),
            FileSink::Images(s) => sinks.push(Box::new(s)),
            FileSink::None => {}
        }
        return serve(a, sinks, (w, h), components, dri, card.filter(|_| kms));
    }
    let Some((scene, process)) = scene else {
        unreachable!("loaded above for every render that is not --serve")
    };
    let controller: ControllerSlot =
        RefCell::new(process.map(|f| Box::new(f) as Box<dyn SceneController>));
    let mut r = match card {
        Some(card) => Renderer::builder(w as u32, h as u32)
            .display(card)
            .rgb565(a.rgb565)
            .build()?,
        None => {
            let b = Renderer::builder(w as u32, h as u32).components(components as u32);
            match dri {
                Some(dri) => b.device(dri),
                None => b,
            }
            .build()?
        }
    };
    if a.drop {
        r.set_drop(true)?;
    }
    r.set_alpha(a.alpha.is_some());
    r.set_rotate(a.rotate)?;
    if let Some(level) = a.brightness {
        match r.set_brightness(level) {
            Ok(took) => report::detail("backlight", format!("{took:.3}")),
            Err(e) => report::error(format!("{}: {e:#}", a.cmd)),
        }
    }
    match r.setup(&scene) {
        Ok(()) => {}
        Err(e @ SetupError::Engine(_)) => return Err(e.into()),
        Err(e @ SetupError::Compile { .. }) => return Err(exit(format!("compile error: {e}"))),
        Err(e) => return Err(exit(e.to_string())),
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
        let sinks = sink_list(&mut scanout, &mut file, &mut h264);
        let mut lp = Loop::new(&mut r, sinks);
        lp.controller = Some(&controller);
        lp.provider = Some(Box::new(|name, size, frame| {
            Feed::Data(demo_feed(name, size, frame))
        }));
        // 0 is the endless run, so the frame budget spells no other count for
        // "render nothing". A negative --frames stops before the first frame.
        lp.frames = want.max(0) as u64;
        lp.clock.timestep = a.timestep;
        lp.clock.pacer = pace;
        let control = a.control.clone();
        lp.stopping = Some(Box::new(move || want < 0 || control.stopped()));
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
    for s in sink_list(&mut scanout, &mut file, &mut h264) {
        closed = closed.and(s.close());
    }
    r.close();
    if a.control.signal().is_some() {
        return Ok(());
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

/// Attach the console to this render, on the display of `card` when it
/// holds one. A scene serves itself alone. A directory (or --demo) serves
/// everything in it.
fn serve(
    a: &Ns,
    sinks: Vec<Box<dyn Output>>,
    size: (i32, i32),
    components: i32,
    dri: Option<String>,
    card: Option<Card>,
) -> Result<()> {
    let (mut scenes_dir, mut store) = (None, None);
    let my_scenes = match a.cmd.as_str() {
        "serve" => Some(my_scenes(a)?),
        _ => None,
    };
    if let Some(here) = &a.here {
        store = Some(SceneStore::here(
            &here.path,
            Arc::clone(&here.program),
            Vec::new(),
        ));
    } else if let Some(scene) = &a.scene {
        let path = find_scene(scene, a.demo)?;
        if Path::new(&path).is_dir() {
            scenes_dir = Some(path);
        } else {
            store = Some(one_scene_store(a, &path)?);
        }
    } else if !a.demo {
        scenes_dir = my_scenes.clone();
    }
    let ran = server::run(server::Options {
        scenes_dir,
        my_scenes,
        uploads: a.allow_uploads,
        state_dir: a.state_dir.clone(),
        edits_dir: a.edits_dir.clone(),
        store,
        size: Some(size),
        components,
        alpha: a.alpha.is_some(),
        sinks,
        display: card,
        rgb565: a.rgb565,
        drop: a.drop,
        rotate: a.rotate,
        brightness: a.brightness,
        port: a.port,
        host: a.host.clone(),
        allowed_hosts: a.allow_hosts.clone(),
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
        control: a.control.clone(),
        ..Default::default()
    });
    match a.control.signal() {
        Some(_) => Ok(()),
        None => ran,
    }
}

/// The directory of the scenes that `ghee serve` makes: --my-scenes, else
/// $GHEE_MY_SCENES, else $XDG_DATA_HOME/ghee/scenes, else
/// ~/.local/share/ghee/scenes.
fn my_scenes(a: &Ns) -> Result<String> {
    let given = a.my_scenes.clone().or_else(|| {
        std::env::var_os("GHEE_MY_SCENES")
            .filter(|d| !d.is_empty())
            .map(Into::into)
    });
    let dir = given.or_else(|| {
        std::env::var_os("XDG_DATA_HOME")
            .map(std::path::PathBuf::from)
            .filter(|d| d.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".local/share")))
            .map(|d| d.join("ghee/scenes"))
    });
    let dir = dir.ok_or_else(|| {
        exit("serve: no directory for your scenes: set HOME or XDG_DATA_HOME, or pass --my-scenes")
    })?;
    Ok(dir.to_string_lossy().into_owned())
}

/// The console's frame rate cap: --fps, whatever takes the frames.
fn console_pacer(a: &Ns) -> Option<Box<dyn Pacer>> {
    output::pacer(a.fps, Sinks::Display)
}

/// A store for the single scene `render --serve` names. A bundled example
/// keeps its state file as a demo, whichever way a command reaches it.
fn one_scene_store(a: &Ns, path: &str) -> Result<SceneStore> {
    let mut store =
        SceneStore::one(path, a.scene_args.clone()).map_err(|e| exit(format!("{}: {e}", a.cmd)))?;
    store.demo = in_examples(path);
    Ok(store)
}

/// True when a scene sits in the bundled examples directory.
fn in_examples(path: &str) -> bool {
    let dir = match Path::new(path).parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => std::env::current_dir().unwrap_or_default(),
    };
    is_examples_dir(&dir)
}

/// `serve` is `render --serve` over a directory of scenes.
fn cmd_serve(a: &Ns) -> Result<()> {
    if a.demo && given(&a.scene) {
        return Err(exit("serve: pass a scenes directory or --demo, not both"));
    }
    if !a.demo && !given(&a.scene) && given(&a.display) && Path::new(text(&a.display)).is_dir() {
        // --display takes an optional value, so it took the directory
        return Err(exit(format!(
            "serve: --display took '{}' as the display name; \
             put the scenes directory before it",
            text(&a.display)
        )));
    }
    cmd_render(a)
}

/// --stats: one summary line after a render. ms/frame is the per-frame cost
/// without the paced sleeps. fps is the frames drawn over the whole run, the
/// stretches that drew nothing included, and the console publishes the same
/// number.
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

/// Where a serve run's frames go. A console with no display and no file is still
/// a destination.
fn serve_out(out: &[String]) -> Vec<String> {
    if out.is_empty() {
        return vec!["console".to_string()];
    }
    out.to_vec()
}

/// The same summary line, for the end of the console's render.
fn serve_stats() -> server::StatsHook {
    // The console opens its GL after this, so its queues start at 0 here.
    let start = usage::Engines::read().unwrap_or_else(usage::Engines::none_yet);
    Box::new(move |times, wall, out| {
        print_stats(times, wall, &serve_out(out), &usage_now(Some(&start)))
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
/// lines that bring the card up, and the file that those lines need, the
/// panel's firmware or an overlay's source. The panel then answers
/// `--display kms` like any other display.
fn cmd_overlay(a: &OverlayArgs) -> Result<()> {
    let Some(panel) = overlay::panel(&a.panel) else {
        return Err(exit(format!(
            "overlay: no panel '{}' (takes: {})",
            a.panel,
            overlay::panels().join(", ")
        )));
    };
    let opts = match given(&a.opts) {
        true => parse_kv(text(&a.opts), "overlay: --opts")?,
        false => Map::new(),
    };
    let unknown: Vec<&str> = opts
        .keys()
        .map(String::as_str)
        .filter(|k| !panel.options.contains(k))
        .collect();
    if !unknown.is_empty() {
        return Err(exit(format!(
            "overlay: no option {} on {} (takes: {})",
            unknown.join(", "),
            a.panel,
            panel.options.join(", ")
        )));
    }
    let pairs: Vec<(String, Value)> = opts.into_iter().collect();
    let setup = (panel.setup)(&pairs).map_err(|e| exit(format!("overlay: {e}")))?;
    for text in &setup.lines {
        line(text);
    }
    let Some(file) = setup.file else {
        if given(&a.output) {
            return Err(exit(format!(
                "overlay: -o with {}, a panel that takes no file",
                a.panel
            )));
        }
        eprintln!("add the lines above to /boot/firmware/config.txt, and reboot");
        return Ok(());
    };
    let (bytes, default) = match &file {
        overlay::File::Firmware(blob) => (blob.clone(), "panel.bin".to_string()),
        overlay::File::Overlay { name, source } => {
            (source.as_bytes().to_vec(), format!("{name}.dts"))
        }
    };
    let path = match given(&a.output) {
        true => text(&a.output),
        false => default.as_str(),
    };
    File::create(path)
        .and_then(|mut f| f.write_all(&bytes))
        .with_context(|| path.to_string())?;
    let install = match file {
        overlay::File::Firmware(_) => "Copy it to /lib/firmware/panel.bin".to_string(),
        overlay::File::Overlay { name, .. } => format!(
            "Compile it with `sudo dtc -@ -I dts -O dtb -o \
             /boot/firmware/overlays/{name}.dtbo {path}`"
        ),
    };
    eprintln!(
        "wrote {path}. {install}, add the lines above to /boot/firmware/config.txt, and reboot"
    );
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
    Ok(())
}

// ------------------------------------------------------------ the parser ----

#[derive(Parser)]
#[command(
    name = "ghee",
    version,
    arg_required_else_help = true,
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

    #[command(about = "list the cards with a connected display")]
    Displays,

    #[command(
        about = "write the kernel's setup for a panel, so it answers --display kms",
        after_help = "Careful: up to Linux 7.3, and in the Pi's 6.18 kernel, the ssd130x \
                      driver leaves an SSD1322 dark. The ssd1322 overlay is for a kernel \
                      whose ssd130x drives it"
    )]
    Overlay(OverlayArgs),
}

/// What every render takes, whichever command asks for it.
#[derive(Args)]
struct RenderFlags {
    #[arg(
        long,
        value_name = "kms",
        num_args = 0..=1,
        default_missing_value = "kms",
        help = "render to the connected KMS display, HDMI or a kernel-driven \
                panel: bare, or 'kms' (zero-copy scanout, needs DRM master)"
    )]
    display: Option<String>,

    #[arg(
        long,
        help = "display scanout in 16-bit color: halves the GPU store and scanout \
                bandwidth (worth it on VC4-class boards), slight banding in gradients"
    )]
    rgb565: bool,

    #[arg(
        long,
        help = "show the newest frame on the --display kms display each time it frees, \
                and skip the frames that waited: the render stops waiting for the \
                display, so --fps paces it"
    )]
    drop: bool,

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
        help = "WxH to render at (default: the display's mode, else 320x240)"
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
    #[arg(long, default_value_t = 8090, help = "the console's port")]
    port: u16,

    #[arg(
        long,
        value_name = "DIR",
        help = "keep the console's saved edits in DIR \
                (default $XDG_DATA_HOME/ghee/edits, else ~/.local/share/ghee/edits)"
    )]
    edits_dir: Option<std::path::PathBuf>,

    #[arg(
        long,
        value_name = "DIR",
        help = "keep the console's drafts, its unsaved edits, in DIR \
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
        value_name = "NAMES",
        value_delimiter = ',',
        help = "comma-separated host names that a console on loopback answers \
                beside localhost, 127.0.0.1 and [::1], for a proxy such as \
                tailscale serve"
    )]
    allow_hosts: Vec<String>,

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
        help = "3=RGB, 4=RGBA"
    )]
    components: i32,

    #[arg(
        long,
        value_name = "FORM",
        num_args = 0..=1,
        value_parser = ["premultiplied", "unpremultiplied"],
        help = "keep the output pass's alpha in RGBA frames, which are opaque \
                without it. Bare, raw frames hold colour premultiplied and \
                PNGs unpremultiplied; a FORM names the form of raw frames, \
                and a PNG takes unpremultiplied alone"
    )]
    alpha: Option<Option<String>>,

    #[arg(
        long,
        value_name = "K=V,...",
        help = "options for the image encoder: quality=90 sets a JPEG's quality, \
                75 by default (ignored for -o -)"
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
        help = "fixed display frame rate (default: the display's own refresh)"
    )]
    fps: Option<f64>,

    #[arg(
        long,
        help = "filters applied after the scene's own, in ffmpeg's filtergraph syntax, \
                e.g. frc=calm=8:coarse=1,dither"
    )]
    filters: Option<String>,

    #[arg(
        long,
        help = "print a timing summary and what ghee used when the render ends"
    )]
    stats: bool,

    #[arg(
        long,
        conflicts_with = "serve",
        help = "draw the scene as its files hold it, without the edits that a \
                console saved for it"
    )]
    original: bool,

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
    #[arg(help = "scenes directory (default: your own scenes, see --my-scenes)")]
    scenes: Option<String>,

    #[arg(long, help = "serve the bundled example scenes")]
    demo: bool,

    #[arg(
        long,
        value_name = "DIR",
        help = "keep the scenes that you make in the console in DIR, one directory each \
                (default $GHEE_MY_SCENES, else $XDG_DATA_HOME/ghee/scenes, else \
                ~/.local/share/ghee/scenes)"
    )]
    my_scenes: Option<std::path::PathBuf>,

    #[arg(
        long,
        help = "take uploaded images and fonts while --host opens the console to a network"
    )]
    allow_uploads: bool,

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
    #[arg(
        value_name = "PANEL",
        help = "the panel's controller: st7789, ssd1306, sh1106 (on I2C), sh1106-spi, \
                ssd1322 or ili9341"
    )]
    panel: String,

    #[arg(
        long,
        value_name = "K=V,...",
        help = "the panel's options, e.g. width=240,height=240,backlight=26. \
                A wrong one lists what the panel takes"
    )]
    opts: Option<String>,

    #[arg(
        short,
        long,
        help = "where to write the panel's file: its firmware (default panel.bin), or \
                its overlay's source (default ghee-<panel>.dts)"
    )]
    output: Option<String>,
}

/// render and serve both resolve to this namespace. serve is render --serve
/// over a directory. It carries render's whole namespace, and every flag it
/// does not offer keeps its default.
pub struct Ns {
    /// The command's name, which each of its error lines starts with.
    cmd: String,
    scene: Option<String>,
    demo: bool,
    output: Option<String>,
    h264: Option<String>,
    serve: bool,
    components: i32,
    /// None keeps every frame opaque.
    alpha: Option<AlphaForm>,
    save_opts: Option<String>,
    write_buffer: i64,
    fps: Option<f64>,
    filters: Option<String>,
    stats: bool,
    display: Option<String>,
    rgb565: bool,
    drop: bool,
    rotate: i64,
    brightness: Option<f64>,
    frames: Option<i64>,
    timestep: Option<f64>,
    size: Option<(i32, i32)>,
    dri: Option<String>,
    port: u16,
    host: String,
    allow_hosts: Vec<String>,
    state_dir: Option<std::path::PathBuf>,
    edits_dir: Option<std::path::PathBuf>,
    /// Draw the scene without the edits that a console saved for it.
    original: bool,
    my_scenes: Option<std::path::PathBuf>,
    allow_uploads: bool,
    preview_max: i32,
    preview_fps: i32,
    stream_max: i32,
    stream_fps: i32,
    bitrate: i32,
    no_probe: bool,
    pub quiet: bool,
    pub verbose: bool,
    scene_args: Vec<String>,
    /// The program in this process that writes the scene, for a scene that
    /// renders itself.
    here: Option<Here>,
    /// What stops the render: SIGINT and SIGTERM.
    control: Control,
}

impl Ns {
    /// This namespace, its render stopped through `control`.
    fn under(mut self, control: &Control) -> Ns {
        self.control = control.clone();
        self
    }
}

impl RenderArgs {
    fn ns(self, scene_args: Vec<String>) -> Ns {
        Ns {
            cmd: "render".to_string(),
            scene: Some(self.scene),
            demo: self.demo,
            output: self.output,
            h264: self.h264,
            serve: self.serve,
            components: self.components,
            alpha: self.alpha.map(|form| match form.as_deref() {
                Some(name) => AlphaForm::named(name).expect("clap checks the name"),
                None => AlphaForm::Each,
            }),
            save_opts: self.save_opts,
            write_buffer: self.write_buffer,
            fps: self.fps,
            filters: self.filters,
            stats: self.stats,
            no_probe: self.render.no_probe,
            quiet: self.render.quiet,
            verbose: self.render.verbose,
            display: self.render.display,
            rgb565: self.render.rgb565,
            drop: self.render.drop,
            rotate: self.render.rotate,
            brightness: self.render.brightness,
            frames: self.render.frames,
            timestep: self.render.timestep,
            size: self.render.size,
            dri: self.render.dri,
            port: self.console.port,
            host: self.console.host,
            allow_hosts: self.console.allow_hosts,
            state_dir: self.console.state_dir,
            edits_dir: self.console.edits_dir,
            original: self.original,
            my_scenes: None,
            allow_uploads: false,
            preview_max: self.console.preview_max,
            preview_fps: self.console.preview_fps,
            stream_max: self.console.stream_max,
            stream_fps: self.console.stream_fps,
            bitrate: self.bitrate,
            scene_args,
            here: None,
            control: Control::new(),
        }
    }
}

impl ServeArgs {
    fn ns(self, scene_args: Vec<String>) -> Ns {
        Ns {
            cmd: "serve".to_string(),
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
            alpha: None,
            save_opts: None,
            write_buffer: 16,
            fps: self.fps,
            filters: None,
            stats: false,
            display: self.render.display,
            rgb565: self.render.rgb565,
            drop: self.render.drop,
            rotate: self.render.rotate,
            brightness: self.render.brightness,
            frames: self.render.frames,
            timestep: self.render.timestep,
            size: self.render.size,
            dri: self.render.dri,
            port: self.console.port,
            host: self.console.host,
            allow_hosts: self.console.allow_hosts,
            state_dir: self.console.state_dir,
            edits_dir: self.console.edits_dir,
            original: false,
            my_scenes: self.my_scenes,
            allow_uploads: self.allow_uploads,
            preview_max: self.console.preview_max,
            preview_fps: self.console.preview_fps,
            stream_max: self.console.stream_max,
            stream_fps: self.console.stream_fps,
            bitrate: self.bitrate,
            scene_args,
            here: None,
            control: Control::new(),
        }
    }
}

/// The command name behind a failure report, and the result of that command.
fn dispatch(cli: Cli, scene_args: Vec<String>, control: &Control) -> (&'static str, Result<()>) {
    let cmd = match &cli.cmd {
        Command::Render(_) => "render",
        Command::Serve(_) => "serve",
        Command::Displays => "displays",
        Command::Overlay(_) => "overlay",
    };
    if !scene_args.is_empty() && !matches!(cli.cmd, Command::Render(_) | Command::Serve(_)) {
        return (
            cmd,
            Err(exit(format!("{cmd}: takes no scene arguments after --"))),
        );
    }
    let ran = match cli.cmd {
        Command::Render(a) => cmd_render(&a.ns(scene_args).under(control)),
        Command::Serve(a) => cmd_serve(&a.ns(scene_args).under(control)),
        Command::Displays => cmd_displays(),
        Command::Overlay(a) => cmd_overlay(&a),
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

/// Two checks on the engine itself. They stay out of `ghee --help`.
const EXTRAS: &str =
    "usage: ghee smoke [dri] [components] | ghee golden --manifest PATH [--dump DIR | --pin]";

/// The status of `ghee smoke` or `ghee golden`, None for any other command.
fn extras(argv: &[String]) -> Option<i32> {
    let rest: Vec<&str> = argv.iter().skip(1).map(String::as_str).collect();
    let ran = match argv.first().map(String::as_str) {
        Some("smoke") => smoke(&rest).map(|()| 0),
        Some("golden") => golden(&rest),
        _ => return None,
    };
    Some(ran.unwrap_or_else(|e| {
        eprintln!("{e}");
        1
    }))
}

/// Render one flat pass at a known colour and read it back. One command covers
/// the engine, the scene assembly and the readback.
fn smoke(args: &[&str]) -> Result<()> {
    let dri = args.first().copied().unwrap_or("/dev/dri/renderD128");
    let components: i32 = match args.get(1) {
        Some(c) => c.parse()?,
        None => 3,
    };
    let mut r = Renderer::builder(64, 64)
        .components(components as u32)
        .device(dri)
        .build()?;
    let scene = Scene::of(vec![Pass::of(
        "main",
        "void main(){ gl_FragColor = vec4(0.25,0.5,0.75,1.0); }",
    )?])?;
    if let Err(e) = r.setup(&scene) {
        bail!("scene build failed: {e}");
    }
    let frame = r.render(None)?;
    println!("RUST_SMOKE_OK pixel={},{},{}", frame[0], frame[1], frame[2]);
    Ok(())
}

fn golden(args: &[&str]) -> Result<i32> {
    match args {
        ["--manifest", path] => golden::run(path, None, false),
        ["--manifest", path, "--dump", dir] => golden::run(path, Some(dir), false),
        ["--manifest", path, "--pin"] => golden::run(path, None, true),
        _ => bail!("{EXTRAS}"),
    }
}

/// The command line's exit status. A scene, engine, encoder or missing
/// dependency failure reaches the user as one line, with no stack. An argument
/// error reads the same way. $GHEE_TRACEBACK asks for the whole error chain.
pub fn main(argv: Vec<String>) -> i32 {
    if let Some(code) = extras(&argv) {
        return code;
    }
    let (own, scene_args) = split_scene_args(&argv);
    let cli = match Cli::try_parse_from(std::iter::once("ghee".to_string()).chain(own)) {
        Ok(cli) => cli,
        Err(e) => return clap_status(&e),
    };
    let control = Control::new();
    let held = match control::catch(&control) {
        Ok(held) => held,
        Err(e) => {
            eprintln!("ghee: {e}");
            return 1;
        }
    };
    let (cmd, ran) = dispatch(cli, scene_args, &control);
    control::release(&control, held);
    status(&control, cmd, ran)
}

/// The exit status of command `cmd` that ended with `ran` under `control`:
/// 1 for an error, which this prints. After SIGINT or SIGTERM this sends the
/// signal again, and the process ends by it; 128 plus the signal's number is
/// the status when a handler of the caller's takes it instead.
fn status(control: &Control, cmd: &str, ran: Result<()>) -> i32 {
    if let Some(sig) = control.signal() {
        control::reraise(sig);
        return 128 + sig;
    }
    match ran {
        Ok(()) => 0,
        Err(e) => {
            say_failure(cmd, &e);
            1
        }
    }
}

/// Print `e`, the failure of command `cmd`, as the command line prints it.
pub fn say_failure(cmd: &str, e: &anyhow::Error) {
    if std::env::var("GHEE_TRACEBACK").is_ok_and(|v| !v.is_empty()) {
        eprintln!("{e:?}");
    } else if let Some(line) = e.downcast_ref::<Exit>() {
        eprintln!("{line}");
    } else {
        eprintln!("{cmd}: {e:#}");
    }
}

// ------------------------------------------------- a scene that renders itself

/// The flags of `ghee render` that a scene's program takes when it renders
/// itself, by their long names.
pub const DIRECT_FLAGS: [&str; 16] = [
    "output",
    "display",
    "rgb565",
    "drop",
    "serve",
    "port",
    "host",
    "size",
    "fps",
    "stats",
    "frames",
    "timestep",
    "rotate",
    "brightness",
    "dri",
    "original",
];

/// A flag of `ghee render`, as its --help shows it.
pub struct Flag {
    pub long: String,
    pub short: Option<char>,
    /// The flag takes a value: always, never, or as `--display` takes one.
    pub takes: Takes,
    pub help: String,
    /// The name of its value.
    pub metavar: Option<String>,
    pub default: Option<String>,
}

/// Whether a flag takes a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Takes {
    Never,
    Always,
    Maybe,
}

/// The flags of DIRECT_FLAGS, in that order.
pub fn direct_flags() -> Vec<Flag> {
    let cli = Cli::command();
    let render = cli
        .find_subcommand("render")
        .expect("ghee has a render command");
    DIRECT_FLAGS
        .iter()
        .filter_map(|id| render.get_arguments().find(|a| a.get_id() == *id))
        .map(|a| {
            let takes = match a.get_num_args() {
                _ if !a.get_action().takes_values() => Takes::Never,
                Some(r) if r.min_values() == 0 => Takes::Maybe,
                _ => Takes::Always,
            };
            Flag {
                long: a.get_long().unwrap_or_default().to_string(),
                short: a.get_short(),
                takes,
                help: a.get_help().map(ToString::to_string).unwrap_or_default(),
                metavar: a
                    .get_value_names()
                    .and_then(|v| v.first())
                    .map(ToString::to_string),
                default: a
                    .get_default_values()
                    .first()
                    .filter(|_| takes != Takes::Never)
                    .map(|v| v.to_string_lossy().into_owned()),
            }
        })
        .collect()
}

/// The program of a scene that renders itself: the path of its file, and
/// the program, which runs in this process.
pub struct Here {
    pub path: String,
    pub program: Arc<dyn Program>,
}

/// The namespace of a render of `here` with `flags`, flags of DIRECT_FLAGS
/// alone, or the line that says why the flags make none. With no -o and no
/// --serve the frames go to the display.
fn direct_ns(prog: &str, flags: &[String], here: Option<Here>) -> Result<Ns, String> {
    let path = here.as_ref().map_or("-", |h| h.path.as_str());
    let argv = ["ghee", "render", path]
        .into_iter()
        .map(String::from)
        .chain(flags.iter().cloned());
    let cli = Cli::try_parse_from(argv).map_err(|e| clap_line(&e))?;
    let Command::Render(r) = cli.cmd else {
        unreachable!("the command line names render")
    };
    let mut a = r.ns(Vec::new());
    a.cmd = prog.to_string();
    if !given(&a.output) && !given(&a.display) && !a.serve {
        a.display = Some("kms".to_string());
    }
    a.here = here;
    Ok(a)
}

/// Why `flags`, flags of DIRECT_FLAGS alone, make no render: the line, as
/// `argument --rotate: ...`. None when they make one.
pub fn direct_check(flags: &[String]) -> Option<String> {
    direct_ns("", flags, None).err()
}

/// One line for a flag that clap refuses.
fn clap_line(e: &clap::Error) -> String {
    let arg = e.get(ContextKind::InvalidArg).map(|v| v.to_string());
    let value = e.get(ContextKind::InvalidValue).map(|v| v.to_string());
    let why = std::error::Error::source(e).map(ToString::to_string);
    match (arg, value, why) {
        (Some(arg), Some(value), Some(why)) => {
            let flag = arg.split(' ').next().unwrap_or_default();
            format!("argument {flag}: {why}, got '{value}'")
        }
        _ => {
            let text = e.to_string();
            let first = text.lines().next().unwrap_or_default();
            first.trim_start_matches("error: ").to_string()
        }
    }
}

/// Render `here` as `ghee render` renders a scene, with `flags`, flags of
/// DIRECT_FLAGS alone, and `card` open already for the display. Returns the
/// exit status of `ghee render`, or the failure, which say_failure() prints
/// as `ghee render` prints it, each line starting with `prog`. A SIGINT or
/// SIGTERM that stops the render goes again, after the render's cleanup, to
/// the handler that held it before the render.
pub fn direct(prog: &str, flags: &[String], here: Here, card: Option<Card>) -> Result<i32> {
    let a = match direct_ns(prog, flags, Some(here)) {
        Ok(a) => a,
        Err(why) => {
            eprintln!("{prog}: error: {why}");
            return Ok(2);
        }
    };
    let held = control::catch(&a.control)?;
    let ran = render(&a, card);
    control::release(&a.control, held);
    if let Some(sig) = a.control.signal() {
        control::reraise(sig);
        return Ok(0);
    }
    ran.map(|()| 0)
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
        dispatch(cli, scene_args, &Control::new()).1
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
            ns(&["serve", "--demo", "--brightness", "0.25"]).brightness,
            Some(0.25)
        );
    }

    #[test]
    fn drop_takes_a_kms_display() {
        let (_d, path) = scene_file("drop");
        let e = run(&["render", &path, "-o", "x.png", "--drop"]).unwrap_err();
        assert_eq!(e.to_string(), "render: --drop needs --display kms");
        assert!(ns(&["serve", "--demo", "--display", "--drop"]).drop);
    }

    #[test]
    fn rotate_takes_the_four_turns() {
        for d in ["0", "90", "180", "270"] {
            let a = ns(&["render", "s.scene.json", "-o", "x.png", "--rotate", d]);
            assert_eq!(a.rotate.to_string(), d);
        }
        assert_eq!(ns(&["serve", "--demo"]).rotate, 0);
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
            ("width=240,nope", "overlay: --opts"),
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
    fn a_refused_flag_value_is_named_once() {
        for (flag, value, said) in [
            (
                "--size",
                "banana",
                "error: invalid value 'banana' for '--size <SIZE>': wants WxH",
            ),
            (
                "--rotate",
                "45",
                "error: invalid value '45' for '--rotate <DEGREES>': wants 0, 90, 180 or 270",
            ),
            (
                "--brightness",
                "1.5",
                "error: invalid value '1.5' for '--brightness <LEVEL>': wants a level from 0 to 1",
            ),
        ] {
            let e = Cli::try_parse_from(argv(&["ghee", "render", "s", flag, value]))
                .err()
                .unwrap();
            assert_eq!(e.to_string().lines().next(), Some(said));
        }
    }

    #[test]
    fn a_display_render_loops_until_stopped_and_an_image_render_is_one_frame() {
        assert_eq!(frames_of(&ns(&["render", "s", "-o", "x.png"])), 1);
        for endless in [
            vec!["render", "s", "--display", "kms"],
            vec!["render", "s", "--h264", "out.h264"],
            vec!["render", "s", "--serve"],
        ] {
            assert_eq!(frames_of(&ns(&endless)), 0, "{endless:?}");
        }
        assert_eq!(
            frames_of(&ns(&["render", "s", "--display", "kms", "--frames", "9"])),
            9
        );
    }

    #[test]
    fn a_display_other_than_kms_is_refused_and_a_directory_says_where_it_goes() {
        let d = tmpdir("dir-display");
        assert_eq!(
            failure(panel_of(
                &ns(&["render", "s", "--display", "hdmi"]),
                &mut None
            )),
            "render: no display 'hdmi' (takes: kms)"
        );
        assert_eq!(
            failure(panel_of(&ns(&["serve", "--display", &d]), &mut None)),
            format!("serve: no display '{d}' (takes: kms), and a scenes directory goes before it")
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
    fn an_output_that_names_no_image_file_is_refused() {
        let e = run(&["render", "s", "-o", "kms"]).unwrap_err();
        assert_eq!(
            e.to_string(),
            "render: -o wants an image pattern such as frame.####.png, or '-', \
             got 'kms'; a display takes --display"
        );
    }

    #[test]
    fn alpha_is_refused_with_a_jpeg() {
        let e = run(&["render", "s", "-o", "f.####.JPG", "--alpha"]).unwrap_err();
        assert_eq!(
            e.to_string(),
            "render: --alpha with a .jpg; a JPEG holds no alpha"
        );
    }

    #[test]
    fn a_render_with_nowhere_to_go_says_so() {
        let (_d, path) = scene_file("nowhere");
        assert_eq!(
            run(&["render", &path]).unwrap_err().to_string(),
            "render: pass -o PATTERN, --display, --h264 PATH or --serve"
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
    fn serve_takes_a_directory_or_demo_and_not_both() {
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
                "overlay: no panel 'does-not-exist' (takes: st7789, ssd1306, sh1106, \
                 sh1106-spi, ssd1322, ili9341)",
            ),
            (
                vec![
                    "overlay",
                    "ssd1306",
                    "--opts",
                    "height=32",
                    "-o",
                    "panel.bin",
                ],
                "overlay: -o with ssd1306, a panel that takes no file",
            ),
            (
                vec!["overlay", "sh1106", "--opts", "dc=24"],
                "overlay: no option dc on sh1106 (takes: address, speed)",
            ),
            (
                vec!["overlay", "st7789", "--opts", "widht=240"],
                "overlay: no option widht on st7789 (takes: width, height, \
                 x_offset, y_offset, dc, rst, backlight, speed, spi_bus, spi_device, \
                 madctl)",
            ),
        ] {
            assert_eq!(run(&tokens).unwrap_err().to_string(), msg, "{tokens:?}");
        }
    }

    /// An overlay's source lands where -o names it, ready for dtc.
    #[test]
    fn overlay_writes_the_overlay_source_where_o_names_it() {
        let d = tmpdir("overlay");
        let path = format!("{d}/oled.dts");
        run(&["overlay", "sh1106-spi", "-o", &path]).unwrap();
        let source = fs::read_to_string(&path).unwrap();
        assert!(source.starts_with("/dts-v1/;\n/plugin/;\n"), "{source}");
    }

    #[test]
    fn the_stats_line_names_every_destination() {
        let out = ["kms".to_string(), "file".to_string()];
        assert_eq!(
            stats_line(&[0.010, 0.030, 0.020], 2.0, &out),
            "stats: 3 frame(s) in 2.00s, 20.00 ms/frame \
             (min 10.00, max 30.00), 1.5 fps -> kms+file"
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
        let file = ["file".to_string()];
        assert_eq!(serve_out(&file), file);
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
    fn geometry_takes_size_else_the_display_mode_else_the_default() {
        assert_eq!(
            geometry(&ns(&["render", "s"]), &Panel::None),
            ((320, 240), 4)
        );
        let a = ns(&["render", "s", "--size", "640x480", "--components", "3"]);
        assert_eq!(geometry(&a, &Panel::None), ((640, 480), 3));
        let kms = Panel::Kms((256, 64));
        assert_eq!(geometry(&ns(&["render", "s"]), &kms), ((256, 64), 4));
        let a = ns(&["render", "s", "--size", "128x32"]);
        assert_eq!(geometry(&a, &kms), ((128, 32), 4));
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
            "--edits-dir",
            "/srv/edits",
            "--allow-hosts",
            "pi.tail.ts.net,pi.local",
            "--",
            "--text",
            "hi",
        ]);
        assert_eq!(
            (a.port, a.host.as_str(), a.size),
            (9, "0.0.0.0", Some((80, 60)))
        );
        assert_eq!(a.state_dir, Some(std::path::PathBuf::from("/srv/state")));
        assert_eq!(a.edits_dir, Some(std::path::PathBuf::from("/srv/edits")));
        assert_eq!(a.allow_hosts, ["pi.tail.ts.net", "pi.local"]);
        assert_eq!((a.preview_fps, a.preview_max, a.stream_max), (5, 120, 320));
        assert_eq!((a.bitrate, a.rgb565), (123, true));
        assert_eq!(a.scene_args, argv(&["--text", "hi"]));
    }

    #[test]
    fn a_bare_display_is_the_scanout() {
        let a = ns(&["render", "s", "--display"]);
        assert_eq!(a.display.as_deref(), Some("kms"));
        assert!(kms_named(&a).is_ok());
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
        assert!(pacer(&ns(&["render", "s", "--display"]), &Panel::Kms((8, 8))).is_none());
        let paced = ns(&["render", "s", "--display", "--fps", "24"]);
        assert!(pacer(&paced, &Panel::Kms((8, 8))).is_some());
        assert_eq!(h264_fps(&ns(&["render", "s"])), 30);
        assert_eq!(h264_fps(&ns(&["render", "s", "--fps", "24.9"])), 24);
        assert_eq!(h264_fps(&ns(&["render", "s", "--fps", "0"])), 30);
    }

    #[test]
    fn the_scene_loaded_for_a_render_carries_its_filters() {
        let (_d, path) = scene_file("filters");
        let (scene, process) =
            scene_of(&ns(&["render", &path, "--filters", "dither"]), &mut None).unwrap();
        assert_eq!(scene.passes.len(), 2);
        assert_eq!(scene.passes[1].name, "dither");
        assert!(process.is_none()); // a JSON scene feeds itself
        let (scene, _) = scene_of(
            &ns(&["render", &path, "--filters", "dither=levels=4,frc=calm=8"]),
            &mut None,
        )
        .unwrap();
        let value = |i: usize, p: &str| scene.passes[i].params.get(p).unwrap().value.to_json();
        assert_eq!(
            (value(1, "levels"), value(2, "calm")),
            (serde_json::json!(4.0), serde_json::json!(8.0))
        );
        let e = failure(scene_of(
            &ns(&["render", &path, "--filters", "frc=calm"]),
            &mut None,
        ));
        assert_eq!(
            e,
            "render: --filters: filter 'frc=calm': 'calm' needs the form param=value"
        );
        let e = failure(scene_of(
            &ns(&["render", &path, "--filters", "nope"]),
            &mut None,
        ));
        assert!(e.starts_with("unknown filter: nope (known: dither"), "{e}");
    }

    /// A console's saved edit reaches a render and the console alike, and
    /// --original leaves it off.
    #[test]
    fn a_render_lays_on_the_edits_that_a_console_saved_unless_original() {
        let d = tmpdir("saved");
        let path = format!("{d}/s.scene.json");
        let gain = r#"{"version":1,"passes":[{"name":"main",
            "fragment":"void main(){ gl_FragColor = vec4(gain); }",
            "params":{"gain":{"type":"float","value":0.5}}}]}"#;
        fs::write(&path, gain).unwrap();
        let states = tmpdir("saved-state");
        let mut console = SceneStore::one(&path, Vec::new()).unwrap();
        console.state_dir = Some(std::path::PathBuf::from(&*states));
        console.edits_dir = console.state_dir.clone();
        let defaults = console.resolve("s", (8, 8)).unwrap().to_dict();
        let mut working = defaults.clone();
        working["passes"][0]["params"]["gain"]["value"] = Value::from(0.75);
        let edits = state::diff(&defaults, &working, false);
        console.save("s", &edits).unwrap();
        let gain_of = |scene: &Scene| scene.passes[0].params.get("gain").unwrap().value.to_json();
        let (opened, _) = console.open("s", None, (8, 8)).unwrap();
        assert_eq!(gain_of(&opened.working), 0.75);
        let render = |extra: &[&str]| {
            let mut t = vec!["render", path.as_str(), "--edits-dir", &states];
            t.extend(extra);
            scene_of(&ns(&t), &mut None).unwrap().0
        };
        assert_eq!(gain_of(&render(&[])), 0.75);
        assert_eq!(gain_of(&render(&["--original"])), 0.5);
        let e = Cli::try_parse_from(argv(&["ghee", "render", &path, "--original", "--serve"]))
            .err()
            .unwrap();
        assert_eq!(e.kind(), ErrorKind::ArgumentConflict);
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

    /// A program in this process: one red pass, which answers each tick
    /// with nothing.
    struct Red;

    impl Program for Red {
        fn start(&self, _size: (i32, i32)) -> Result<Value> {
            Ok(serde_json::from_str(SCENE)?)
        }

        fn answer(&self, _tick: &str) -> Result<Vec<u8>> {
            Ok(vec![5])
        }
    }

    #[test]
    fn a_scene_that_renders_itself_takes_its_flags_from_ghee_render() {
        let flags = direct_flags();
        let longs: Vec<&str> = flags.iter().map(|f| f.long.as_str()).collect();
        assert_eq!(longs, DIRECT_FLAGS);
        let output = &flags[0];
        assert_eq!((output.short, output.takes), (Some('o'), Takes::Always));
        assert!(output.help.starts_with("image pattern"), "{}", output.help);
        assert_eq!(flags[1].takes, Takes::Maybe);
        assert_eq!(flags[2].takes, Takes::Never);
        assert_eq!(flags[5].default.as_deref(), Some("8090"));
        assert_eq!(direct_check(&argv(&["--rotate=90", "-o", "f.png"])), None);
        assert_eq!(
            direct_check(&argv(&["--rotate=45"])).unwrap(),
            "argument --rotate: wants 0, 90, 180 or 270, got '45'"
        );
        let a = direct_ns("feeder.py", &[], None).unwrap();
        assert_eq!((a.cmd.as_str(), text(&a.display)), ("feeder.py", "kms"));
        assert!(direct_ns("x", &argv(&["--serve"]), None)
            .unwrap()
            .display
            .is_none());
    }

    #[test]
    #[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
    fn a_scene_that_renders_itself_renders_as_ghee_render() {
        let d = tmpdir("here");
        let here = || Here {
            path: format!("{d}/red.py"),
            program: Arc::new(Red),
        };
        let frame = format!("{d}/f.png");
        let flags = argv(&["-o", &frame, "--size", "8x4"]);
        assert_eq!(direct("red.py", &flags, here(), None).unwrap(), 0);
        let png = fs::read(&frame).unwrap();
        assert_eq!(&png[1..4], b"PNG");
        let flags = argv(&["-o", &frame, "--frames", "2"]);
        let a = direct_ns("red.py", &flags, Some(here())).unwrap();
        let e = render(&a, None).unwrap_err().to_string();
        assert!(
            e.starts_with("red.py: multiple frames into one filename"),
            "{e}"
        );
    }
}
