//! Procedural scenes (docs/procedural.md).
//! A procedural scene is an executable that supplies a scene and its data.
//!
//! ghee starts the process when the scene becomes active. The process
//! writes one scene JSON line, then binary messages that set arrays and
//! live params, and ghee writes one JSON line to its stdin for each render
//! tick.
//!
//! The render thread never blocks on the process.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::ffi::c_int;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use serde_json::{Map, Value};

use crate::draw::{self, Arg, Call, CallKind};
use crate::engine::{Feed, Given, Refused};
use crate::layer::{Address, Layer, Rgb};
use crate::looper::{RenderTarget, SceneController};
use crate::procedural::resolve;
use crate::scene::{py_repr, ArraySize, ParamValue, Reach, Scene};

/// `-- --help` waits this long for a scene's declarations line.
/// Nothing else runs a scene with `--ghee-args`.
const PROBE_WAIT: Duration = Duration::from_secs(2);

/// The handshake waits this long for the scene line.
const HANDSHAKE_WAIT: Duration = Duration::from_secs(30);

/// A pinned tick waits this long for a lockstep process to end its answer.
const ANSWER_WAIT: Duration = Duration::from_secs(30);

/// The stop sequence: closed stdin, then SIGTERM, then SIGKILL.
const TERM_AFTER: Duration = Duration::from_secs(2);
const KILL_AFTER: Duration = Duration::from_secs(1);

/// A stop waits this long for the pipe threads to end on their own.
const SETTLE: Duration = Duration::from_millis(200);

/// How long the error message of a dead process waits for the stderr
/// drain thread to finish. The process is already reaped and the drain
/// ends at EOF, so the wait covers thread scheduling alone. Without it, a
/// loaded machine builds the message before the drain hands over the
/// stderr tail.
const EPITAPH_WAIT: Duration = Duration::from_secs(1);

/// The error state keeps this much of the process stderr.
const STDERR_TAIL: usize = 8192;

/// The cap on one message's value count, far above any real scene's data.
/// A count past it means a corrupt stream, and the reader refuses the
/// message instead of allocating a huge buffer for garbage.
const MAX_VALUES: usize = 1 << 22;

/// ghee sets this env var in every scene process it starts. The shim's
/// run() reads it to tell a ghee-started run from a direct one.
const WIRE: &str = "GHEE_WIRE";

/// ghee's render size, as WxH, in every scene process it starts.
/// A scene sizes its own buffers against it, so `half` and `quarter` mean
/// something before a single pass is built.
const SIZE: &str = "GHEE_SIZE";

// The kind of a message from the scene process, its first byte, numbered as
// docs/procedural.md lists them: kind 5 is END_TICK.
const SET_ARRAY: u8 = 0;
const SET_PARAM: u8 = 1;
const CLEAN: u8 = 2;
const SET_STRING: u8 = 3;
const SET_ARRAY_U8: u8 = 4;
const END_TICK: u8 = 5;
const ADD_LAYER: u8 = 6;
const UPDATE_LAYER: u8 = 7;
const UPDATE_LAYERS: u8 = 8;
const REMOVE_LAYER: u8 = 9;
const MOVE_LAYER: u8 = 10;

const SIGTERM: c_int = 15;

extern "C" {
    fn kill(pid: c_int, sig: c_int) -> c_int;
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ------------------------------------------------------------ what a path is

/// True for a regular file anyone may execute.
pub(crate) fn runnable(path: &str) -> bool {
    fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

/// The executable behind a scene path.
/// A runnable path names itself, and a package spelling names its
/// executable member. None means the path holds no process.
pub fn executable(path: &str) -> Option<String> {
    // a .scene.json is a document whatever its mode bits say
    if runnable(path) && !path.ends_with(crate::procedural::SUFFIX) {
        return Some(path.to_string());
    }
    match resolve(path) {
        Ok((_, Some(x))) if runnable(&x) => Some(x),
        _ => None,
    }
}

/// The process working directory, and the program path for any directory.
fn where_to_run(path: &str) -> Result<(String, String)> {
    let full = fs::canonicalize(path)
        .map_err(|e| anyhow!("{path}: {e}"))?
        .to_string_lossy()
        .into_owned();
    let dir = match Path::new(&full).parent() {
        Some(d) => d.to_string_lossy().into_owned(),
        None => "/".to_string(),
    };
    Ok((full, dir))
}

/// Why `program` did not start. For a script the OS names the wrong cause:
/// ENOEXEC when the file has no `#!` line, ENOENT when the interpreter on
/// its `#!` line is missing.
fn why_not_started(program: &str, e: &std::io::Error) -> String {
    const ENOENT: i32 = 2;
    const ENOEXEC: i32 = 8;
    let mut head = [0u8; 256];
    let n = fs::File::open(program)
        .and_then(|mut f| f.read(&mut head))
        .unwrap_or(0);
    let head = &head[..n];
    match (e.raw_os_error(), head.strip_prefix(b"#!")) {
        (Some(ENOEXEC), None) if !head.starts_with(b"\x7fELF") => {
            "the file has no #! line".to_string()
        }
        (Some(ENOENT), Some(rest)) => {
            let line = rest.split(|&b| b == b'\n').next().unwrap_or_default();
            let line = String::from_utf8_lossy(line);
            let interpreter = line
                .trim_start_matches([' ', '\t'])
                .split([' ', '\t'])
                .next()
                .unwrap_or_default();
            if Path::new(interpreter).exists() {
                return crate::scene::strerror(e);
            }
            format!(
                "the interpreter '{}' on its #! line does not exist",
                interpreter.escape_debug()
            )
        }
        _ => crate::scene::strerror(e),
    }
}

// ------------------------------------------------------- what a scene takes

/// One argument declared by a process.
#[derive(Clone, Debug, PartialEq)]
pub struct ArgDecl {
    pub name: String,
    pub ty: String,
    pub default: Value,
    pub help: String,
}

/// What a scene said when ghee asked what arguments it takes.
#[derive(Clone, Debug)]
pub enum Takes {
    /// The declarations it answered with, empty for a scene that takes none.
    Args(Vec<ArgDecl>),
    /// It answered with something else, so its arguments are its own.
    PassThrough,
    /// It was still starting when the wait ran out.
    Starting,
    /// Nothing asked it.
    Unasked,
    /// ghee could not run it to ask, for the reason held.
    Failed(String),
}

/// A scene in the `-- --help` listing.
#[derive(Clone, Debug)]
pub struct Declared {
    pub name: String,
    pub args: Takes,
}

/// What a process declares, read now.
/// Only `-- --help` calls this. A render and a serve never probe.
/// `wait` is how long the probe waits for the line.
fn declarations(path: &str, wait: Duration) -> Takes {
    match probe(path, wait) {
        Answer::Line(line) => match parse_declarations(&line) {
            Some(args) => Takes::Args(args),
            None => Takes::PassThrough,
        },
        Answer::Silent => Takes::PassThrough,
        Answer::Starting => Takes::Starting,
        Answer::Failed(why) => Takes::Failed(why),
    }
}

/// What came back from `--ghee-args`.
enum Answer {
    Line(String),
    /// It ran and said nothing, so it takes its arguments its own way.
    Silent,
    /// It was still running when the wait ran out, so nothing is known.
    Starting,
    /// ghee could not run it, for the reason held.
    Failed(String),
}

/// Run `<process> --ghee-args` and read the one line it answers with.
fn probe(path: &str, wait: Duration) -> Answer {
    let Ok((program, dir)) = where_to_run(path) else {
        return Answer::Silent;
    };
    // The probe runs the scene as its start does: uv makes the scene's
    // environment at the first run, and every later run reuses it.
    let uv = match crate::procedural::wheel::uv_env() {
        Ok(uv) => uv,
        Err(e) => return Answer::Failed(e.to_string()),
    };
    let Ok(mut child) = Command::new(program)
        .arg("--ghee-args")
        .current_dir(dir)
        .envs(uv)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    else {
        return Answer::Silent;
    };
    let Some(out) = child.stdout.take() else {
        return Answer::Silent;
    };
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut line = Vec::new();
        if BufReader::new(out).read_until(b'\n', &mut line).is_ok() {
            let _ = tx.send(String::from_utf8_lossy(&line).into_owned());
        }
    });
    let answer = rx.recv_timeout(wait).ok();
    // Still running when the wait ran out is a scene still starting. Gone
    // already, with nothing on stdout, is a scene that takes no arguments
    // this way.
    let running = matches!(child.try_wait(), Ok(None));
    let _ = child.kill();
    let _ = child.wait();
    match (answer, running) {
        (Some(line), _) => Answer::Line(line),
        (None, true) => Answer::Starting,
        (None, false) => Answer::Silent,
    }
}

/// `{"args": [{"name": ..., "type": ..., "default": ..., "help": ...}]}`.
/// The parser drops an entry without a name.
/// Any other line is a pass-through.
fn parse_declarations(line: &str) -> Option<Vec<ArgDecl>> {
    let doc: Value = serde_json::from_str(line.trim()).ok()?;
    let items = doc.get("args")?.as_array()?;
    let mut out: Vec<ArgDecl> = Vec::new();
    for item in items {
        let Some(name) = item.get("name").and_then(Value::as_str) else {
            continue;
        };
        out.push(ArgDecl {
            name: name.to_string(),
            ty: item
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("string")
                .to_string(),
            default: item.get("default").cloned().unwrap_or(Value::Null),
            help: item
                .get("help")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        });
    }
    Some(out)
}

/// What each scene of a set declares, for `-- --help`.
/// A scene that holds no process declares nothing.
/// Each executable scene answers `--ghee-args` here, and nothing else asks.
/// `ask` false skips every probe, which is the point of --no-probe: a scene
/// slow to start costs the wait once per scene, and a listing of ten costs it
/// ten times.
pub fn declare(scenes: &[(String, String)], ask: bool) -> Vec<Declared> {
    let jobs = match ask {
        false => 1,
        true => thread::available_parallelism().map_or(1, |n| n.get().min(4)),
    };
    let mut out: Vec<Declared> = Vec::with_capacity(scenes.len());
    for batch in scenes.chunks(jobs) {
        let waiting: Vec<&str> = batch.iter().map(|(name, _)| name.as_str()).collect();
        crate::report::progress("scenes", format!("asking {}", waiting.join(" ")));
        let mut handles = Vec::new();
        for (name, path) in batch {
            let (name, exe) = (name.clone(), executable(path));
            handles.push(thread::spawn(move || Declared {
                name,
                args: match exe {
                    Some(_) if !ask => Takes::Unasked,
                    Some(exe) => declarations(&exe, PROBE_WAIT),
                    None => Takes::Args(Vec::new()),
                },
            }));
        }
        for h in handles {
            if let Ok(d) = h.join() {
                out.push(d);
            }
        }
    }
    crate::report::progress_done();
    out
}

// -------------------------------------------------------------------- help

fn metavar(ty: &str) -> &'static str {
    match ty {
        "int" => "N",
        "float" => "X",
        "size" => "WxH",
        "bool" => "",
        _ => "STR",
    }
}

/// The default as the help writes it. A size takes WxH, and a string stays bare.
fn shown(decl: &ArgDecl) -> Option<String> {
    match &decl.default {
        Value::Null => None,
        Value::Array(a) if decl.ty == "size" && a.len() == 2 => Some(format!("{}x{}", a[0], a[1])),
        Value::String(s) => Some(s.clone()),
        v => Some(v.to_string()),
    }
}

/// One line per declared argument, grouped under the scene that takes it.
pub fn help(scenes: &[Declared], prog: &str) -> String {
    let mut out = format!("usage: {prog} [ARGS]\n");
    for scene in scenes {
        let args = match &scene.args {
            Takes::Args(args) if !args.is_empty() => args,
            Takes::Args(_) => {
                out.push_str(&format!("\n{}: no arguments\n", scene.name));
                continue;
            }
            Takes::PassThrough => {
                out.push_str(&format!("\n{}: pass-through\n", scene.name));
                continue;
            }
            Takes::Starting => {
                out.push_str(&format!(
                    "\n{}: still starting after {}s, so its arguments went unread. \
                     --no-probe skips the wait\n",
                    scene.name,
                    PROBE_WAIT.as_secs()
                ));
                continue;
            }
            Takes::Unasked => {
                out.push_str(&format!("\n{}: not asked\n", scene.name));
                continue;
            }
            Takes::Failed(why) => {
                out.push_str(&format!("\n{}: {why}\n", scene.name));
                continue;
            }
        };
        out.push_str(&format!("\n{}:\n", scene.name));
        for a in args {
            let flag = match a.ty.as_str() {
                "bool" => format!("--{}, --no-{}", a.name, a.name),
                _ => format!("--{} {}", a.name, metavar(&a.ty)),
            };
            let mut text = a.help.clone();
            if let Some(d) = shown(a) {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&format!("(default: {d})"));
            }
            out.push_str(&format!("  {flag:<24}{text}\n"));
        }
    }
    out
}

/// True when the tokens are the help request itself.
/// Any other token belongs to the scene, `--help` beside it included.
pub fn wants_help(tokens: &[String]) -> bool {
    matches!(tokens, [one] if one == "--help" || one == "-h")
}

// ------------------------------------------------------------- the mailboxes

/// One param message: the numbers of kind 1, or the text of kind 3.
enum ParamData {
    F32(Vec<f32>),
    Str(String),
}

/// One message that sets something by name: a param or a live field of a
/// layer (kinds 1 and 3), a layer added to the pass of that name (kind 6),
/// the new look of the layer of that name (kind 7), the new looks of
/// several layers (kind 8), each of those three as its JSON text, the layer
/// of that name taken out (kind 9), or the layer of that name moved, as the
/// JSON text of its new place (kind 10).
enum Set {
    Param(ParamData),
    Layer(String),
    Look(String),
    Looks(String),
    Remove,
    Move(String),
}

impl Set {
    /// True when this set, the newer, leaves nothing of `older`, a set of
    /// the same name.
    fn overrides(&self, older: &Set) -> bool {
        matches!(
            (self, older),
            (Set::Param(_), Set::Param(_)) | (Set::Look(_), Set::Look(_))
        )
    }
}

/// A param value that the process set: the name that its message gave the
/// param, the param's pass, None for the scene's, its type and the value.
struct Kept {
    name: String,
    pass: Option<String>,
    ty: String,
    value: ParamValue,
}

/// The uniform of the param `name`: a dotted name carries its pass, and the
/// uniform is the tail.
fn uniform(name: &str) -> &str {
    name.rsplit_once('.').map_or(name, |(_, p)| p)
}

/// The param `name` of pass `pass`, None for a scene param, as Locks names
/// it.
fn param_name(pass: Option<&str>, name: &str) -> String {
    match pass {
        Some(pass) => format!("{pass}.{name}"),
        None => name.to_string(),
    }
}

/// The live field `at` as a message names it.
pub(crate) fn dotted(at: &Address) -> String {
    match at.effect {
        Some(effect) => format!("{}.{effect}.{}", at.layer, at.field),
        None => format!("{}.{}", at.layer, at.field),
    }
}

/// What the program of a scene set, a scene process or a host of the
/// console: each param, a scene param by its name and a param of a pass as
/// `pass.name`, each live field of a layer as `layer.field` or
/// `layer.effect.field`, and each layer that the program added or gave a
/// look, every field of it. A console refuses to set what the program set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Locks {
    params: HashSet<String>,
    fields: HashSet<String>,
    layers: HashSet<String>,
}

impl Locks {
    /// The program set the param `name` of pass `pass`, None for a scene
    /// param.
    pub fn set_param(&mut self, pass: Option<&str>, name: &str) {
        self.params.insert(param_name(pass, name));
    }

    /// The program set the live field `at`.
    pub fn set_field(&mut self, at: &Address) {
        self.fields.insert(dotted(at));
    }

    /// The program added layer `name`, or gave it a look.
    pub fn set_layer(&mut self, name: &str) {
        self.layers.insert(name.to_string());
    }

    /// Whether the program set the param `name` of pass `pass`, None for a
    /// scene param.
    pub fn param(&self, pass: Option<&str>, name: &str) -> bool {
        self.params.contains(&param_name(pass, name))
    }

    /// Whether the program set the live field `at`, or its whole layer.
    pub fn field(&self, at: &Address) -> bool {
        self.layer(at.layer) || self.fields.contains(&dotted(at))
    }

    /// Whether the program set a field of the effect `effect` of layer
    /// `layer`.
    pub fn effect(&self, layer: &str, effect: &str) -> bool {
        let at = format!("{layer}.{effect}.");
        self.fields.iter().any(|f| f.starts_with(&at))
    }

    /// Whether the program set every field of layer `name`.
    pub fn layer(&self, name: &str) -> bool {
        self.layers.contains(name)
    }

    /// Why a console refuses to set the param `name` of pass `pass`, None
    /// for a scene param, or None when the program never set it.
    pub fn param_refusal(&self, pass: Option<&str>, name: &str) -> Option<String> {
        self.param(pass, name).then(|| {
            format!(
                "the scene's program sets param '{}'",
                param_name(pass, name)
            )
        })
    }

    /// Why a console refuses to set the live field `at`, or None when the
    /// program never set it or its layer.
    pub fn field_refusal(&self, at: &Address) -> Option<String> {
        match (self.layer(at.layer), self.fields.contains(&dotted(at))) {
            (true, _) => Some(format!("the scene's program sets layer '{}'", at.layer)),
            (false, true) => Some(format!(
                "the scene's program sets layer field '{}'",
                dotted(at)
            )),
            (false, false) => None,
        }
    }

    /// `{params, fields, layers}`, each a sorted list of names.
    pub fn to_json(&self) -> Value {
        let sorted = |set: &HashSet<String>| {
            let mut names: Vec<&String> = set.iter().collect();
            names.sort();
            Value::from(names.into_iter().cloned().collect::<Vec<_>>())
        };
        serde_json::json!({
            "params": sorted(&self.params),
            "fields": sorted(&self.fields),
            "layers": sorted(&self.layers),
        })
    }
}

/// What the reader thread takes off stdout between two drains.
#[derive(Default)]
struct Inbox {
    arrays: Vec<(String, Feed)>,
    /// The sets in the order of arrival, as latest() queues them.
    params: Vec<(String, Set)>,
    /// The process calls these ticks clean.
    clean: u32,
    /// The answers that the process ended with kind 5, one a tick.
    ended: u64,
    /// stdout ended, so the process is gone.
    done: bool,
    /// A message ghee cannot read. The stream never resynchronizes.
    fault: String,
}

fn newest<T>(into: &mut Vec<(String, T)>, name: String, values: T) {
    match into.iter_mut().find(|(n, _)| *n == name) {
        Some(slot) => slot.1 = values,
        None => into.push((name, values)),
    }
}

/// Queue `set` of `name` last, and drop each older set of that name that it
/// overrides. The queue then ends in the newest state of every layer, when
/// the answers to several ticks arrive between two frames: a layer hidden
/// by one answer and drawn again by the next ends up shown.
fn latest(into: &mut Vec<(String, Set)>, name: String, set: Set) {
    into.retain(|(n, older)| *n != name || !set.overrides(older));
    into.push((name, set));
}

/// One render tick on its way to the process stdin.
struct Tick {
    frame: u64,
    time: f64,
    dt: f64,
    params: Map<String, Value>,
}

impl Tick {
    fn line(&self) -> String {
        let mut d = Map::new();
        d.insert("frame".to_string(), Value::from(self.frame));
        d.insert("time".to_string(), Value::from(self.time));
        d.insert("dt".to_string(), Value::from(self.dt));
        d.insert("params".to_string(), Value::Object(self.params.clone()));
        format!("{}\n", Value::Object(d))
    }
}

/// The one tick that waits for the writer.
/// A new tick absorbs the untaken one.
/// A slow process loses ticks, and never a param change.
#[derive(Default)]
struct Outbox {
    pending: Option<Tick>,
    closed: bool,
}

// --------------------------------------------------------------- the failure

/// What a dead scene process leaves.
/// The drain writes the stderr of the process to the terminal as it arrives.
/// Display names the condition alone, so a terminal reads that stderr once.
/// `full()` adds the tail, for a reader that saw no terminal.
#[derive(Clone, Debug)]
pub struct SceneFailure {
    /// The scene, and what happened to it.
    why: String,
    /// The last of the process stderr.
    stderr: String,
}

impl std::fmt::Display for SceneFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.why)
    }
}

impl std::error::Error for SceneFailure {}

impl SceneFailure {
    /// The condition, then the stderr of the process.
    /// The console error state carries this, because a browser reads no
    /// terminal.
    pub fn full(&self) -> String {
        match self.stderr.is_empty() {
            true => self.why.clone(),
            false => format!("{}\n{}", self.why, self.stderr),
        }
    }
}

/// The text of an error for a reader that saw no terminal.
/// A dead scene process adds the stderr already written by the drain.
pub fn with_stderr(e: &anyhow::Error) -> String {
    match e.downcast_ref::<SceneFailure>() {
        Some(failure) => failure.full(),
        None => format!("{e:#}"),
    }
}

// ------------------------------------------------------------------ the process

pub struct SceneProcess {
    /// The path the caller named, for the error messages.
    pub name: String,
    child: Child,
    inbox: Arc<Mutex<Inbox>>,
    /// Woken with the inbox when an answer ends or stdout ends.
    answered: Arc<Condvar>,
    outbox: Arc<(Mutex<Outbox>, Condvar)>,
    stderr: Arc<Mutex<String>>,
    /// The stderr drain. SceneFailure waits for it before it reads the tail.
    errors: Option<JoinHandle<()>>,
    /// The tick writer. It holds the process stdin, so it ends the run.
    writer: Option<JoinHandle<()>>,
    threads: Vec<JoinHandle<()>>,
    /// The last data of each array, and whether it waits for its upload.
    data: HashMap<String, (Feed, bool)>,
    /// An array of `data` waits for its upload.
    unsent: bool,
    /// The sets that wait for a renderer, in the order of arrival.
    incoming: Vec<(String, Set)>,
    /// param name -> (the pass that holds it, its type). None is scene scope.
    params: HashMap<String, (Option<String>, String)>,
    /// The names of the scene's layers, which address their live fields as
    /// `layer.field` and `layer.effect.field`, each true for a text layer.
    layers: HashMap<String, bool>,
    /// The params as they stand, by the name that a tick gives each, a text
    /// layer's words as its `text` too, each value as a scene file writes
    /// it, and the last values read by the process.
    values: HashMap<String, Value>,
    sent: HashMap<String, Value>,
    /// The last value that the process set of each param, by the name that
    /// a tick gives the param.
    kept: HashMap<String, Kept>,
    /// The live fields that the process set, by the names of their
    /// messages, and the layers that it added or gave a look.
    fields: HashSet<String>,
    whole: HashSet<String>,
    /// Where each first set lands, for a console that locks them, emptied
    /// when the process ends.
    locks: Option<Arc<Mutex<Locks>>>,
    /// The scene was set up again, so update() sets every kept value again.
    rebuilt: bool,
    /// Clean ticks from the process, not yet spent on a skip.
    clean: u32,
    /// The process died. The scene stays in this error state.
    dead: Option<SceneFailure>,
    /// The handshake said `"lockstep": true`: the process ends its answer to
    /// every tick with kind 5.
    lockstep: bool,
    /// The stop sequence's waits, TERM_AFTER and KILL_AFTER.
    term_after: Duration,
    kill_after: Duration,
    /// The loop pins time to a fixed step, so every tick waits for the
    /// answer of a lockstep process, the first one waiting either way.
    pinned: bool,
    /// The ticks handed to the writer, each one written to the process.
    offered: u64,
    /// The loop's frame and time at the first tick offered, which the
    /// process reads as frame 0 at time 0.
    origin: Option<(u64, f64)>,
    /// The params already logged as refused, so each logs once.
    refused: HashSet<String>,
}

impl SceneProcess {
    /// Start the process and take the handshake.
    /// The scene belongs to the process.
    /// The caller compiles the scene, then binds this process to it.
    /// `args` are the tokens after `--`, as they came. The process parses them,
    /// and a process that refuses one dies with its own message on stderr.
    /// The scene's files resolve beside the executable, as far as `reach`
    /// allows. `size` is the size that the scene renders at, which the
    /// process reads before it writes its scene. `edit` changes the scene
    /// document that the process wrote before ghee builds it, and a scene
    /// that the edit breaks runs as the process wrote it, with a warning.
    pub fn start(
        path: &str,
        args: &[String],
        reach: Reach,
        (w, h): (i32, i32),
        edit: impl FnOnce(&mut Value),
    ) -> Result<(Scene, SceneProcess)> {
        let (program, dir) = where_to_run(path)?;
        let uv = crate::procedural::wheel::uv_env().map_err(|e| anyhow!("{path}: {e}"))?;
        let mut child = Command::new(&program)
            .args(args)
            .current_dir(&dir)
            .envs(uv)
            .env(SIZE, format!("{w}x{h}"))
            .env(WIRE, "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("{path}: {}", why_not_started(&program, &e)))?;
        crate::sources::record(path);
        let inbox = Arc::new(Mutex::new(Inbox::default()));
        let answered = Arc::new(Condvar::new());
        let stderr = Arc::new(Mutex::new(String::new()));
        let outbox = Arc::new((Mutex::new(Outbox::default()), Condvar::new()));
        let (tx, rx): (Sender<String>, Receiver<String>) = mpsc::channel();
        let errors = drain_stderr(take_stderr(&mut child)?, Arc::clone(&stderr));
        let threads = vec![read_stream(
            take_stdout(&mut child)?,
            tx,
            Arc::clone(&inbox),
            Arc::clone(&answered),
        )];
        let stdin = take_stdin(&mut child)?;
        let mut process = SceneProcess {
            name: path.to_string(),
            child,
            inbox,
            answered,
            outbox,
            stderr,
            errors: Some(errors),
            writer: None,
            threads,
            data: HashMap::new(),
            unsent: false,
            incoming: Vec::new(),
            params: HashMap::new(),
            layers: HashMap::new(),
            values: HashMap::new(),
            sent: HashMap::new(),
            kept: HashMap::new(),
            fields: HashSet::new(),
            whole: HashSet::new(),
            locks: None,
            rebuilt: false,
            clean: 0,
            dead: None,
            lockstep: false,
            term_after: TERM_AFTER,
            kill_after: KILL_AFTER,
            pinned: false,
            offered: 0,
            origin: None,
            refused: HashSet::new(),
        };
        // The wait here is the long one: the process starts an interpreter,
        // imports what it needs and builds its scene before it answers. Seconds
        // of it on a Pi Zero, and silence reads as a hang.
        let name = Path::new(path)
            .file_name()
            .map_or(path, |n| n.to_str().unwrap_or(path));
        crate::report::progress("scene", format!("starting {name}"));
        let line = match rx.recv_timeout(HANDSHAKE_WAIT) {
            Ok(line) => line,
            Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => {
                crate::report::progress_done();
                let why = process.epitaph("did not complete the handshake");
                return Err(anyhow!(why));
            }
        };
        crate::report::progress_done();
        let mut doc: Value = serde_json::from_str(line.trim()).map_err(|e| {
            anyhow!(process.epitaph(&format!("handshake line is not a scene object: {e}")))
        })?;
        // "lockstep" belongs to the handshake, and the scene takes no such key.
        let lockstep = doc.as_object_mut().and_then(|o| o.remove("lockstep"));
        let mut edited = doc.clone();
        edit(&mut edited);
        let mut scene = match Scene::from_dict_in(&edited, &dir, reach, Some(path)) {
            Ok(scene) => scene,
            Err(e) if edited != doc => {
                crate::report::warn(format!("{e}, so the scene runs as its process wrote it"));
                Scene::from_dict_in(&doc, &dir, reach, Some(path))?
            }
            Err(e) => return Err(e),
        };
        scene.load_assets_in(&dir, reach)?;
        process.lockstep = lockstep == Some(Value::Bool(true));
        process.writer = Some(write_ticks(stdin, Arc::clone(&process.outbox)));
        Ok((scene, process))
    }

    /// The loop pins time to a fixed step. A lockstep process then answers
    /// each tick before the tick's frame draws, where it otherwise answers
    /// the first one alone first.
    pub fn pin(&mut self) {
        self.pinned = true;
    }

    /// Record in `locks` each param, live field and layer that the process
    /// sets from now on, once, at its first set, and empty `locks` when the
    /// process ends.
    pub(crate) fn lock_into(&mut self, locks: Arc<Mutex<Locks>>) {
        self.locks = Some(locks);
    }

    /// The process set the live field of the message `name`.
    fn set_field(&mut self, name: &str) {
        if self.fields.contains(name) {
            return;
        }
        self.fields.insert(name.to_string());
        if let (Some(locks), Some(at)) = (&self.locks, Address::dotted(name)) {
            lock(locks).set_field(&at);
        }
    }

    /// The process added layer `name`, or gave it a look.
    fn set_whole(&mut self, name: &str) {
        if self.whole.contains(name) {
            return;
        }
        self.whole.insert(name.to_string());
        if let Some(locks) = &self.locks {
            lock(locks).set_layer(name);
        }
    }

    /// Wait until the process ended its answer to every tick offered, or
    /// until stdout ends.
    fn await_answers(&mut self, frame: u64) -> Result<()> {
        let deadline = Instant::now() + ANSWER_WAIT;
        let mut held = lock(&self.inbox);
        while held.ended < self.offered && !held.done && held.fault.is_empty() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(anyhow!(
                    "{}: did not end its answer to tick {frame} within {} s",
                    self.name,
                    ANSWER_WAIT.as_secs()
                ));
            }
            held = self
                .answered
                .wait_timeout(held, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        Ok(())
    }

    /// Log a param message that changes nothing, once for each name.
    fn refuse(&mut self, name: &str, why: &str) {
        if self.refused.insert(name.to_string()) {
            crate::report::warn(format!("{}: {why}, so it changes nothing", self.name));
        }
    }

    /// Log a message that the renderer refused, once for each name, and
    /// hand on any other failure.
    fn refuse_or_fail(&mut self, name: &str, e: anyhow::Error) -> Result<()> {
        let Some(refused) = e.downcast_ref::<Refused>() else {
            return Err(e);
        };
        let why = refused.0.clone();
        self.refuse(name, &why);
        Ok(())
    }

    /// Add the layer of a kind 6 message to pass `pass`, `text` its JSON.
    fn add_layer(&mut self, renderer: &mut dyn RenderTarget, pass: &str, text: &str) -> Result<()> {
        let (index, layer) = match added(pass, text) {
            Ok(found) => found,
            Err(why) => {
                self.refuse(pass, &why);
                return Ok(());
            }
        };
        let name = layer.get("name").and_then(Value::as_str).unwrap_or(pass);
        let name = name.to_string();
        match renderer.add_layer(pass, index, &layer) {
            Ok(()) => {
                self.took(&layer);
                self.set_whole(&name);
            }
            Err(e) => self.refuse_or_fail(&name, e)?,
        }
        Ok(())
    }

    /// Give layer `name` the look of a kind 7 message, `text` its JSON.
    fn update_layer(
        &mut self,
        renderer: &mut dyn RenderTarget,
        name: &str,
        text: &str,
    ) -> Result<()> {
        match look_of(name, text) {
            Ok(look) => self.give_look(renderer, name, &look),
            Err(why) => {
                self.refuse(name, &why);
                Ok(())
            }
        }
    }

    /// Give layer `name` the look `look`.
    fn give_look(
        &mut self,
        renderer: &mut dyn RenderTarget,
        name: &str,
        look: &Value,
    ) -> Result<()> {
        match renderer.update_layer(look) {
            Ok(changed) => {
                if changed {
                    self.took(look);
                }
                self.set_whole(name);
            }
            Err(e) => self.refuse_or_fail(name, e)?,
        }
        Ok(())
    }

    /// Give the layers of a kind 8 message their looks, `text` its JSON. The
    /// looks that draw::call_of() reads go to the renderer as calls, at
    /// once, and each call that the renderer leaves, and each other look,
    /// takes its look as kind 7 gives it, in the order of the list.
    fn update_layers(&mut self, renderer: &mut dyn RenderTarget, text: &str) -> Result<()> {
        let looks = match looks_of(text) {
            Ok(looks) => looks,
            Err(why) => {
                self.refuse("kind 8", &why);
                return Ok(());
            }
        };
        let mut calls = Vec::with_capacity(looks.len());
        let mut at = Vec::with_capacity(looks.len());
        let mut slow = Vec::new();
        for (k, look) in looks.iter().enumerate() {
            match draw::call_of(look) {
                Some(call) => {
                    calls.push(call);
                    at.push(k);
                }
                None => slow.push(k),
            }
        }
        if !calls.is_empty() {
            match renderer.update_calls(&calls) {
                Ok(taken) => {
                    for &k in &taken.changed {
                        self.took_call(&calls[k]);
                    }
                    let mut taken_here = vec![true; calls.len()];
                    for &i in &taken.left {
                        taken_here[i] = false;
                    }
                    for (call, here) in calls.iter().zip(taken_here) {
                        if here {
                            self.set_whole(&call.layer);
                        }
                    }
                    slow.extend(taken.left.iter().map(|&i| at[i]));
                }
                Err(e) => return self.refuse_or_fail("kind 8", e),
            }
        }
        slow.sort_unstable();
        for k in slow {
            let name = looks[k]["name"].as_str().unwrap_or_default().to_string();
            self.give_look(renderer, &name, &looks[k])?;
        }
        Ok(())
    }

    /// Take layer `name` out of the scene, for a kind 9 message.
    fn remove_layer(&mut self, renderer: &mut dyn RenderTarget, name: &str) -> Result<()> {
        if let Err(e) = renderer.remove_layer(name) {
            return self.refuse_or_fail(name, e);
        }
        self.layers.remove(name);
        let text = format!("{name}.text");
        self.values.remove(&text);
        self.sent.remove(&text);
        Ok(())
    }

    /// Put layer `name` in the place of a kind 10 message, `text` its JSON.
    fn move_layer(
        &mut self,
        renderer: &mut dyn RenderTarget,
        name: &str,
        text: &str,
    ) -> Result<()> {
        match place_of(name, text) {
            Ok(index) => match renderer.move_layer(name, index) {
                Ok(()) => Ok(()),
                Err(e) => self.refuse_or_fail(name, e),
            },
            Err(why) => {
                self.refuse(name, &why);
                Ok(())
            }
        }
    }

    /// Hold the words of `call`, a call that the renderer took, as the
    /// process's own, which no tick carries back.
    fn took_call(&mut self, call: &Call) {
        let (CallKind::Text, Some(Arg::Str(words))) = (call.kind, call.args.first()) else {
            return;
        };
        let text = format!("{}.text", call.layer);
        self.values
            .insert(text.clone(), Value::from(words.as_str()));
        self.sent.insert(text, Value::from(words.as_str()));
    }

    /// Index `layer`, a layer that the renderer took from the process: its
    /// live fields, and the words of a text layer, the process's own, which
    /// no tick carries back.
    fn took(&mut self, layer: &Value) {
        let Ok(layer) = Layer::from_dict("the layer", layer) else {
            return;
        };
        let name = layer.name().to_string();
        let was_text = self.layers.insert(name.clone(), layer.as_text().is_some());
        let text = format!("{name}.text");
        match layer.as_text() {
            Some(l) => {
                self.values
                    .insert(text.clone(), Value::from(l.text.clone()));
                self.sent.insert(text, Value::from(l.text.clone()));
            }
            None if was_text == Some(true) => {
                self.values.remove(&text);
                self.sent.remove(&text);
            }
            None => {}
        }
    }

    /// Index the params of the compiled scene.
    /// The set holds the handshake params, plus any param added by a filter
    /// or a saved value.
    /// The first tick carries every param, and the words of each text layer
    /// as its `text`.
    pub fn bind(&mut self, scene: &Scene) {
        self.params.clear();
        self.layers = scene
            .layers()
            .map(|l| (l.name().to_string(), l.as_text().is_some()))
            .collect();
        self.values.clear();
        self.sent.clear();
        for l in scene.text_layers() {
            self.values
                .insert(format!("{}.text", l.name), Value::from(l.text.clone()));
        }
        let scoped: Vec<(Option<&str>, &crate::scene::ParamMap)> =
            std::iter::once((None, &scene.params))
                .chain(
                    scene
                        .passes
                        .iter()
                        .map(|p| (Some(p.name.as_str()), &p.params)),
                )
                .collect();
        for (pass, params) in scoped {
            for (name, p) in params.iter() {
                // "pass.param" addresses one pass
                // a bare name goes to the first pass that declares it
                if let Some(pass) = pass {
                    self.params.insert(
                        format!("{pass}.{name}"),
                        (Some(pass.to_string()), p.ty.clone()),
                    );
                }
                if self.params.contains_key(name) {
                    if let Some(pass) = pass {
                        self.values
                            .insert(format!("{pass}.{name}"), p.value.to_json());
                    }
                    continue;
                }
                self.params
                    .insert(name.clone(), (pass.map(str::to_string), p.ty.clone()));
                self.values.insert(name.clone(), p.value.to_json());
            }
        }
    }

    /// The name that a tick gives the param `name` of pass `pass`, None for
    /// the scene's: the bare name when it addresses that param, else
    /// `pass.name`.
    fn tick_name(&self, pass: Option<&str>, name: &str) -> Option<String> {
        match (self.params.get(name), pass) {
            (Some((at, _)), _) if at.as_deref() == pass => Some(name.to_string()),
            (_, Some(pass)) => Some(format!("{pass}.{name}")),
            _ => None,
        }
    }

    /// Take everything the reader holds. The newest data for a name wins.
    fn drain(&mut self) {
        let (arrays, params, clean, done, fault) = {
            let mut inbox = lock(&self.inbox);
            (
                std::mem::take(&mut inbox.arrays),
                std::mem::take(&mut inbox.params),
                std::mem::take(&mut inbox.clean),
                inbox.done,
                inbox.fault.clone(),
            )
        };
        for (name, values) in arrays {
            self.data.insert(name, (values, true));
            self.unsent = true;
        }
        for (name, set) in params {
            latest(&mut self.incoming, name, set);
        }
        self.clean = self.clean.saturating_add(clean);
        if self.dead.is_none() && (done || !fault.is_empty()) {
            let why = match fault.is_empty() {
                true => "the scene process exited".to_string(),
                false => fault,
            };
            self.dead = Some(self.epitaph(&why));
            if let Some(locks) = self.locks.take() {
                *lock(&locks) = Locks::default();
            }
        }
    }

    /// Set every kept param value again. A value that the scene no longer
    /// takes is logged and dropped.
    fn set_kept(&mut self, renderer: &mut dyn RenderTarget) {
        let mut refused = Vec::new();
        for (ticked, k) in &self.kept {
            if let Err(e) = renderer.set_param(k.pass.as_deref(), uniform(&k.name), &k.ty, &k.value)
            {
                refused.push((ticked.clone(), format!("{e:#}")));
            }
        }
        for (ticked, why) in refused {
            if let Some(k) = self.kept.remove(&ticked) {
                self.refuse(&k.name, &why);
            }
        }
    }

    /// What a dead process leaves: the condition, and its stderr beside it.
    fn epitaph(&mut self, why: &str) -> SceneFailure {
        self.stop();
        if let Some(t) = self.errors.take() {
            wait_until(EPITAPH_WAIT, || t.is_finished());
            self.threads.push(t);
        }
        SceneFailure {
            why: format!("{}: {why}", self.name),
            stderr: lock(&self.stderr).trim_end().to_string(),
        }
    }

    /// One JSON line for this tick, carrying the params that changed.
    fn offer(&mut self, frame: u64, time: f64, dt: f64) {
        let mut params = Map::new();
        for (name, value) in &self.values {
            if self.sent.get(name) == Some(value) {
                continue;
            }
            params.insert(name.clone(), value.clone());
        }
        self.sent.clone_from(&self.values);
        let (slot, wake) = &*self.outbox;
        let mut out = lock(slot);
        if out.closed {
            return;
        }
        match out.pending.take() {
            Some(old) => {
                for (name, value) in old.params {
                    params.entry(name).or_insert(value);
                }
            }
            None => self.offered += 1,
        }
        out.pending = Some(Tick {
            frame,
            time,
            dt,
            params,
        });
        wake.notify_all();
    }

    /// The stop sequence. stdin closes first.
    /// SIGTERM follows two seconds later.
    /// SIGKILL follows one second after SIGTERM.
    fn stop(&mut self) {
        {
            let (slot, wake) = &*self.outbox;
            lock(slot).closed = true;
            wake.notify_all();
        }
        // the writer drops the process stdin as it ends, which ends the ticks
        // a writer blocked on a full pipe ends with the process
        if let Some(w) = self.writer.take() {
            wait_until(SETTLE, || w.is_finished());
            self.threads.push(w);
        }
        if !self.reaped(self.term_after) {
            unsafe { kill(self.child.id() as c_int, SIGTERM) };
            if !self.reaped(self.kill_after) {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
        // a child of the process can hold the pipes open after the process ends
        // nothing here may hold the render thread that long, so the threads go unjoined
        wait_until(SETTLE, || self.threads.iter().all(JoinHandle::is_finished));
        self.threads.clear();
    }

    /// Wait for the process to exit, and reap it.
    /// False means the process outlives the wait.
    fn reaped(&mut self, patience: Duration) -> bool {
        let child = &mut self.child;
        wait_until(patience, || {
            matches!(child.try_wait(), Ok(Some(_)) | Err(_))
        })
    }
}

impl Drop for SceneProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

impl SceneController for SceneProcess {
    /// Read what arrived, then hand the process this tick.
    /// The loop calls this once per iteration, drawn or skipped.
    fn tick(&mut self, frame: u64, t: f64, dt: f64) -> Result<()> {
        self.drain();
        if let Some(dead) = &self.dead {
            return Err(anyhow!(dead.clone()));
        }
        // A lockstep process answers its first tick before that frame draws,
        // so the frame shows the data of its own tick. Later ticks wait only
        // with the time pinned.
        let first = self.offered == 0;
        let (f0, t0) = *self.origin.get_or_insert((frame, t));
        let frame = frame.saturating_sub(f0);
        self.offer(frame, t - t0, dt);
        if self.lockstep && (self.pinned || first) {
            self.await_answers(frame)?;
            self.drain();
            if let Some(dead) = &self.dead {
                return Err(anyhow!(dead.clone()));
            }
        }
        Ok(())
    }

    /// update() hands the renderer each array, once for each message and
    /// again after each setup.
    fn provide(&mut self, _name: &str, _size: ArraySize, _frame: u64) -> Option<Feed> {
        Some(Feed::Unchanged)
    }

    fn update(&mut self, renderer: &mut dyn RenderTarget, _frame: u64, _t: f64) -> Result<()> {
        if std::mem::take(&mut self.rebuilt) {
            self.set_kept(renderer);
        }
        for (name, set) in std::mem::take(&mut self.incoming) {
            let data = match set {
                Set::Layer(text) => {
                    self.add_layer(renderer, &name, &text)?;
                    continue;
                }
                Set::Look(text) => {
                    self.update_layer(renderer, &name, &text)?;
                    continue;
                }
                Set::Looks(text) => {
                    self.update_layers(renderer, &text)?;
                    continue;
                }
                Set::Remove => {
                    self.remove_layer(renderer, &name)?;
                    continue;
                }
                Set::Move(text) => {
                    self.move_layer(renderer, &name, &text)?;
                    continue;
                }
                Set::Param(data) => data,
            };
            // A first part that names a layer addresses a live field of it.
            // Names are unique across the scene, so no pass has that name.
            let layer = Address::dotted(&name).filter(|at| self.layers.contains_key(at.layer));
            if let Some(at) = layer {
                let given = match &data {
                    ParamData::F32(values) => Given::Numbers(values),
                    ParamData::Str(text) => Given::Text(text),
                };
                if let Err(e) = renderer.set_layer(&at, given) {
                    self.refuse_or_fail(&name, e)?;
                    continue;
                }
                self.set_field(&name);
                // The process wrote these words itself, so no tick carries
                // them back.
                if let (Some(held), ParamData::Str(text)) = (self.values.get_mut(&name), data) {
                    *held = Value::from(text);
                    self.sent.insert(name, held.clone());
                }
                continue;
            }
            let Some((pass, ty)) = self.params.get(&name).cloned() else {
                let why = format!("param '{name}' is not a param this scene declares");
                self.refuse(&name, &why);
                continue;
            };
            let value = match data {
                ParamData::F32(values) => as_param(&ty, &values).ok_or_else(|| match ty.as_str() {
                    "string" => format!("string param '{name}' takes text, got numbers"),
                    _ => format!(
                        "{ty} param '{name}' takes {}, got {}",
                        match arity(&ty) {
                            1 => "a number".to_string(),
                            n => format!("{n} numbers"),
                        },
                        values.len()
                    ),
                }),
                ParamData::Str(text) => match ty.as_str() {
                    "string" => Ok(ParamValue::Str(text)),
                    "color" => Rgb::from_text(&text).map(ParamValue::Color).ok_or_else(|| {
                        format!(
                            "color param '{name}' takes '#rrggbb' or a CSS colour name as \
                             text, got {}",
                            py_repr(&Value::from(text))
                        )
                    }),
                    _ => Err(format!("{ty} param '{name}' takes numbers, got text")),
                },
            };
            let value = match value {
                Ok(v) => v,
                Err(why) => {
                    self.refuse(&name, &why);
                    continue;
                }
            };
            renderer.set_param(pass.as_deref(), uniform(&name), &ty, &value)?;
            let Some(ticked) = self.tick_name(pass.as_deref(), uniform(&name)) else {
                continue;
            };
            // The process set this value itself, so no tick carries it back.
            if let Some(held) = self.values.get_mut(&ticked) {
                *held = value.to_json();
                match self.sent.get_mut(&ticked) {
                    Some(sent) => sent.clone_from(held),
                    None => {
                        self.sent.insert(ticked.clone(), held.clone());
                    }
                }
            }
            let kept = Kept {
                name,
                pass,
                ty,
                value,
            };
            match self.kept.entry(ticked) {
                Entry::Occupied(mut held) => {
                    held.insert(kept);
                }
                Entry::Vacant(slot) => {
                    let first = slot.insert(kept);
                    if let Some(locks) = &self.locks {
                        lock(locks).set_param(first.pass.as_deref(), uniform(&first.name));
                    }
                }
            }
        }
        if !std::mem::take(&mut self.unsent) {
            return Ok(());
        }
        // An array message whose name addresses a layer, such as
        // `scope.points`, sets that live field of the layer.
        let fields: Vec<String> = self
            .data
            .keys()
            .filter(|n| Address::dotted(n).is_some_and(|at| self.layers.contains_key(at.layer)))
            .cloned()
            .collect();
        for name in fields {
            let at = Address::dotted(&name).expect("the name was just matched");
            match self.data.remove(&name) {
                Some((Feed::Data(values), _)) => {
                    match renderer.set_layer(&at, Given::Numbers(&values)) {
                        Ok(()) => self.set_field(&name),
                        Err(e) => self.refuse_or_fail(&name, e)?,
                    }
                }
                Some((Feed::DataU8(_), _)) => {
                    let why = format!(
                        "layer '{}' {} takes floats in message kind 0, got bytes",
                        at.layer, at.field
                    );
                    self.refuse(&name, &why);
                }
                Some((Feed::Unchanged, _)) | None => {}
            }
        }
        for (name, (data, unsent)) in &mut self.data {
            if std::mem::take(unsent) {
                renderer.set_array(name, data)?;
            }
        }
        Ok(())
    }

    fn dirty(&mut self, _frame: u64, _t: f64) -> bool {
        match self.clean {
            0 => true,
            n => {
                self.clean = n - 1;
                false
            }
        }
    }

    fn on_param(&mut self, pass: Option<&str>, name: &str, value: &ParamValue) {
        let Some(ticked) = self.tick_name(pass, name) else {
            return;
        };
        if let Some(held) = self.values.get_mut(&ticked) {
            *held = value.to_json();
        }
    }

    fn rebuilt(&mut self) {
        for (_, unsent) in self.data.values_mut() {
            *unsent = true;
        }
        self.unsent = !self.data.is_empty();
        self.rebuilt = !self.kept.is_empty();
    }

    fn on_layer(&mut self, at: &Address, value: &Value) {
        if let (None, "text") = (at.effect, at.field) {
            if let Some(held) = self.values.get_mut(&format!("{}.text", at.layer)) {
                held.clone_from(value);
            }
        }
    }
}

/// How many numbers a param of type `ty` takes.
fn arity(ty: &str) -> usize {
    match ty {
        "color" => 3,
        "vec2" | "vec3" | "vec4" => ty[3..].parse().unwrap_or(1),
        _ => 1,
    }
}

/// The floats of a param message, as the param's own type.
/// None means the message is short, or the param takes text.
fn as_param(ty: &str, values: &[f32]) -> Option<ParamValue> {
    let at = |i: usize| values.get(i).map(|v| f64::from(*v));
    match ty {
        "float" => Some(ParamValue::Float(at(0)?)),
        "int" => Some(ParamValue::Int(at(0)?.round() as i64)),
        "bool" => Some(ParamValue::Bool(at(0)? != 0.0)),
        "color" => Some(ParamValue::Color(Rgb::numbers([at(0)?, at(1)?, at(2)?]))),
        "vec2" | "vec3" | "vec4" => {
            let n = ty[3..].parse::<usize>().ok()?;
            (values.len() >= n)
                .then(|| ParamValue::Vec(values[..n].iter().map(|v| f64::from(*v)).collect()))
        }
        _ => None,
    }
}

/// The place and the layer of a kind 6 message to pass `pass`, read from
/// `text`, its JSON, or why the message carries none.
fn added(pass: &str, text: &str) -> Result<(usize, Value), String> {
    let doc: Value = serde_json::from_str(text)
        .map_err(|e| format!("kind 6 to pass '{pass}' is no JSON: {e}"))?;
    let index = doc
        .get("index")
        .and_then(Value::as_u64)
        .and_then(|i| usize::try_from(i).ok());
    let layer = doc.get("layer").filter(|l| l.is_object());
    match (index, layer, doc.as_object().map(Map::len)) {
        (Some(index), Some(layer), Some(2)) => Ok((index, layer.clone())),
        _ => Err(format!(
            "kind 6 to pass '{pass}' needs {{\"index\": n, \"layer\": {{...}}}}, n a whole \
             number from 0"
        )),
    }
}

/// The place of a kind 10 message to layer `layer`, read from `text`, its
/// JSON, or why the message carries none.
fn place_of(layer: &str, text: &str) -> Result<usize, String> {
    let doc: Value = serde_json::from_str(text)
        .map_err(|e| format!("kind 10 to layer '{layer}' is no JSON: {e}"))?;
    let index = doc
        .get("index")
        .and_then(Value::as_u64)
        .and_then(|i| usize::try_from(i).ok());
    match (index, doc.as_object().map(Map::len)) {
        (Some(index), Some(1)) => Ok(index),
        _ => Err(format!(
            "kind 10 to layer '{layer}' needs {{\"index\": n}}, n a whole number from 0"
        )),
    }
}

/// The look of a kind 7 message to layer `layer`, read from `text`, its
/// JSON, or why the message carries none.
fn look_of(layer: &str, text: &str) -> Result<Value, String> {
    let doc: Value = serde_json::from_str(text)
        .map_err(|e| format!("kind 7 to layer '{layer}' is no JSON: {e}"))?;
    match doc.get("name").and_then(Value::as_str) {
        Some(name) if name == layer => Ok(doc),
        _ => Err(format!(
            "kind 7 to layer '{layer}' holds no layer named '{layer}'"
        )),
    }
}

/// The looks of a kind 8 message, read from `text`, its JSON, or why the
/// message carries none.
fn looks_of(text: &str) -> Result<Vec<Value>, String> {
    let doc: Value = serde_json::from_str(text).map_err(|e| format!("kind 8 is no JSON: {e}"))?;
    match doc {
        Value::Array(looks)
            if looks
                .iter()
                .all(|l| l.get("name").is_some_and(Value::is_string)) =>
        {
            Ok(looks)
        }
        _ => Err("kind 8 needs a list of layers, each with its name".to_string()),
    }
}

/// Poll until the condition holds or the patience runs out.
fn wait_until(patience: Duration, mut done: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + patience;
    loop {
        if done() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        thread::sleep(Duration::from_millis(5));
    }
}

// ------------------------------------------------------------- the three pipes

fn take_stdout(child: &mut Child) -> Result<ChildStdout> {
    child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("no process stdout"))
}

fn take_stderr(child: &mut Child) -> Result<ChildStderr> {
    child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("no process stderr"))
}

fn take_stdin(child: &mut Child) -> Result<ChildStdin> {
    child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("no process stdin"))
}

/// Keep the tail of the process stderr.
/// An empty pipe keeps the process from blocking on a write.
fn drain_stderr(mut err: ChildStderr, into: Arc<Mutex<String>>) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut buf = [0u8; 4096];
        loop {
            match err.read(&mut buf) {
                Ok(0) | Err(_) => return,
                Ok(n) => {
                    // pass the scene's own words through: prints reach the
                    // terminal live, and the tail below feeds a reader with no
                    // terminal (SceneFailure::full)
                    eprint!("{}", String::from_utf8_lossy(&buf[..n]));
                    let mut held = lock(&into);
                    held.push_str(&String::from_utf8_lossy(&buf[..n]));
                    while held.len() > STDERR_TAIL {
                        let cut = held.len() - STDERR_TAIL;
                        let at = (cut..held.len())
                            .find(|i| held.is_char_boundary(*i))
                            .unwrap_or(held.len());
                        *held = held[at..].to_string();
                    }
                }
            }
        }
    })
}

/// The handshake line, then the binary messages.
fn read_stream(
    out: ChildStdout,
    first: Sender<String>,
    inbox: Arc<Mutex<Inbox>>,
    answered: Arc<Condvar>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut out = BufReader::new(out);
        let mut line = Vec::new();
        match out.read_until(b'\n', &mut line) {
            Ok(n) if n > 0 && line.ends_with(b"\n") => {
                let _ = first.send(String::from_utf8_lossy(&line).into_owned());
            }
            _ => {
                lock(&inbox).done = true;
                answered.notify_all();
                return;
            }
        }
        if let Err(why) = messages(&mut out, &inbox, &answered) {
            lock(&inbox).fault = why;
        }
        lock(&inbox).done = true;
        answered.notify_all();
    })
}

/// kind: u8, name_len: u8, name: UTF-8, count: u32 LE, values: count x f32 LE.
/// Kind 4 carries count u8 bytes instead, and the value is byte / 255.
/// Kinds 3 and 6 to 10 carry count bytes of UTF-8 text.
/// Kind 2 (clean) and kind 5 (the end of a tick's answer) are the single byte.
fn messages(
    out: &mut impl BufRead,
    inbox: &Mutex<Inbox>,
    answered: &Condvar,
) -> Result<(), String> {
    loop {
        let mut kind = [0u8; 1];
        if out.read_exact(&mut kind).is_err() {
            return Ok(()); // stdout ended between messages
        }
        if kind[0] == CLEAN {
            let mut held = lock(inbox);
            held.clean = held.clean.saturating_add(1);
            continue;
        }
        if kind[0] == END_TICK {
            lock(inbox).ended += 1;
            answered.notify_all();
            continue;
        }
        if kind[0] > MOVE_LAYER {
            return Err(format!("wrote message kind {} (0 to 10 only)", kind[0]));
        }
        let mut len = [0u8; 1];
        let mut count = [0u8; 4];
        if out.read_exact(&mut len).is_err() {
            return Ok(());
        }
        let mut name = vec![0u8; len[0] as usize];
        if out.read_exact(&mut name).is_err() || out.read_exact(&mut count).is_err() {
            return Ok(());
        }
        let count = u32::from_le_bytes(count) as usize;
        if count > MAX_VALUES {
            return Err(format!(
                "wrote a message of {count} values, over the {MAX_VALUES} cap"
            ));
        }
        let width = match kind[0] {
            SET_ARRAY | SET_PARAM => 4,
            _ => 1,
        };
        let mut body = vec![0u8; count * width];
        if out.read_exact(&mut body).is_err() {
            return Ok(());
        }
        let name = String::from_utf8_lossy(&name).into_owned();
        if kind[0] == SET_ARRAY_U8 {
            newest(&mut lock(inbox).arrays, name, Feed::DataU8(body));
            continue;
        }
        if matches!(
            kind[0],
            SET_STRING | ADD_LAYER | UPDATE_LAYER | UPDATE_LAYERS | REMOVE_LAYER | MOVE_LAYER
        ) {
            let text = String::from_utf8_lossy(&body).into_owned();
            let set = match kind[0] {
                ADD_LAYER => Set::Layer(text),
                UPDATE_LAYER => Set::Look(text),
                UPDATE_LAYERS => Set::Looks(text),
                REMOVE_LAYER => Set::Remove,
                MOVE_LAYER => Set::Move(text),
                _ => Set::Param(ParamData::Str(text)),
            };
            latest(&mut lock(inbox).params, name, set);
            continue;
        }
        let values: Vec<f32> = body
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut held = lock(inbox);
        match kind[0] {
            SET_ARRAY => newest(&mut held.arrays, name, Feed::Data(values)),
            _ => latest(&mut held.params, name, Set::Param(ParamData::F32(values))),
        }
    }
}

/// Write the pending tick.
/// Close stdin when the process stops.
/// A full pipe blocks this thread alone, and never the render thread.
fn write_ticks(mut stdin: ChildStdin, outbox: Arc<(Mutex<Outbox>, Condvar)>) -> JoinHandle<()> {
    thread::spawn(move || {
        let (slot, wake) = &*outbox;
        loop {
            let tick = {
                let mut out = lock(slot);
                while out.pending.is_none() && !out.closed {
                    out = wake.wait(out).unwrap_or_else(|e| e.into_inner());
                }
                match out.pending.take() {
                    Some(tick) => tick,
                    None => return, // closed
                }
            };
            if stdin.write_all(tick.line().as_bytes()).is_err() {
                return; // the process closed its stdin, or is gone
            }
            let _ = stdin.flush();
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{script, SCRIPTS};

    /// A process that answers the probe, then hands over a scene.
    /// It feeds one array and one param for each tick.
    /// It appends every tick it reads to ticks.jsonl in its own directory.
    /// The test reads that file, so the process must run in that directory.
    const SCENE_SCRIPT: &str = r#"#!/bin/sh
# f32 <a> <b>: the four little-endian bytes of a/b as an IEEE 754 single.
f32() {
    a=$1 b=$2
    if [ "$a" -eq 0 ]; then printf '\000\000\000\000'; return; fi
    e=0
    while [ "$a" -ge $((b * 2)) ]; do b=$((b * 2)); e=$((e + 1)); done
    while [ "$a" -lt "$b" ]; do a=$((a * 2)); e=$((e - 1)); done
    s=$(( ((a << 24) + b) / (b * 2) ))
    if [ "$s" -ge $((1 << 24)) ]; then s=$((1 << 23)); e=$((e + 1)); fi
    bits=$(( (127 + e) << 23 | (s - (1 << 23)) ))
    printf "$(printf '\\%03o\\%03o\\%03o\\%03o' $((bits & 255)) $((bits >> 8 & 255)) $((bits >> 16 & 255)) $((bits >> 24 & 255)))"
}

if [ "$1" = "--ghee-args" ]; then
    : > probed
    printf '%s\n' '{"args": [{"name": "emit", "type": "size", "default": [4, 2], "help": "the array shape"}, {"name": "loud", "type": "bool", "default": false, "help": ""}]}'
    exit 0
fi
printf '%s\n' '{"version": 1, "params": {"text": {"type": "string", "value": "GHEE"}}, "passes": [{"name": "main", "fragment": "void main(){ gl_FragColor = vec4(level_at(0)); }", "arrays": [{"name": "level", "size": 4}], "params": {"amt": {"type": "float", "value": 0.0}}}]}'

while IFS= read -r line; do
    printf '%s\n' "$line" >> ticks.jsonl
    n=${line#*\"frame\":}; n=${n%%,*}; n=${n%%\}*}
    printf '\000\005level\004\000\000\000'
    f32 "$n" 1; f32 $((n + 1)) 1; f32 $((n + 2)) 1; f32 $((n + 3)) 1
    printf '\001\003amt\001\000\000\000'
    f32 "$n" 10
    if [ $((n % 2)) -eq 1 ]; then printf '\002'; fi
done
printf 'eof\n' >> ticks.jsonl
"#;

    /// A process that ignores stdin. It writes its data, then waits for the stop.
    const FREE: &str = r#"#!/bin/sh
printf '%s\n' '{"version": 1, "passes": [{"name": "main", "fragment": "void main(){ gl_FragColor = vec4(1.0); }", "arrays": [{"name": "level", "size": 2}]}]}'
# kind 0, level, the floats 7.0 and 8.0
printf '\000\005level\002\000\000\000\000\000\340\100\000\000\000\101'
sleep 60
"#;

    /// A process that sends one array as kind 4 bytes and one as kind 0 floats.
    const U8_SCRIPT: &str = r#"#!/bin/sh
printf '%s\n' '{"version": 1, "passes": [{"name": "main", "fragment": "void main(){ gl_FragColor = vec4(1.0); }", "arrays": [{"name": "mask", "size": [2, 2]}, {"name": "state", "size": 2}]}]}'
# kind 4, mask, the bytes 0 64 128 255
printf '\004\004mask\004\000\000\000\000\100\200\377'
# kind 0, state, the floats 0.25 and 1.0
printf '\000\005state\002\000\000\000\000\000\200\076\000\000\200\077'
sleep 60
"#;

    fn tmpdir(tag: &str) -> crate::testing::TempDir {
        crate::testing::TempDir::new(&format!("proc-{tag}"))
    }

    fn no_args() -> Vec<String> {
        Vec::new()
    }

    fn start_scene(path: &str, args: &[String]) -> Result<(Scene, SceneProcess)> {
        let _spawn = SCRIPTS.read().unwrap();
        SceneProcess::start(path, args, Reach::Any, (32, 24), |_| {})
    }

    /// declarations(), spawn-locked like start_scene.
    fn declared(path: &str, wait: Duration) -> Takes {
        let _spawn = SCRIPTS.read().unwrap();
        declarations(path, wait)
    }

    /// A wait that a test runs out on purpose: the stop of a process that
    /// never reads stdin, the probe of a scene still starting.
    const SOON: Duration = Duration::from_millis(50);

    /// start_scene() for a process that never reads stdin, so its stop goes
    /// to the signals after SOON.
    fn start_deaf(path: &str, args: &[String]) -> Result<(Scene, SceneProcess)> {
        let (scene, mut f) = start_scene(path, args)?;
        (f.term_after, f.kill_after) = (SOON, SOON);
        Ok((scene, f))
    }

    /// The error of a call that must fail.
    /// A running process is not Debug, so unwrap_err() does not compile here.
    fn failure<T>(r: Result<T>) -> anyhow::Error {
        match r {
            Ok(_) => panic!("expected a failure"),
            Err(e) => e,
        }
    }

    fn args(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|s| (*s).to_string()).collect()
    }

    /// A renderer stand-in that records the params set by a process, and the
    /// arrays that it uploads. `feeds` names the arrays that it declares.
    #[derive(Default)]
    struct Stub {
        set: std::cell::RefCell<Vec<(Option<String>, String, ParamValue)>>,
        /// The live fields of layers set, as their wire names and values.
        layers: std::cell::RefCell<Vec<(String, String)>>,
        feeds: Vec<(String, ArraySize)>,
        bytes: std::cell::RefCell<Vec<(String, Vec<u8>)>>,
        floats: std::cell::RefCell<Vec<(String, Vec<f32>)>>,
        /// Every param set, live set, layer added, look given, layer taken out
        /// and layer moved, in order.
        log: std::cell::RefCell<Vec<String>>,
    }

    impl RenderTarget for Stub {
        fn width(&self) -> i32 {
            8
        }

        fn height(&self) -> i32 {
            8
        }

        fn components(&self) -> i32 {
            4
        }

        fn set_dt(&self, _value: f64) {}

        fn set_time(&self, _t: f64) {}

        fn draw(&mut self, _provider: Option<crate::engine::Provider<'_>>) -> Result<()> {
            Ok(())
        }

        fn render_into(
            &mut self,
            _provider: Option<crate::engine::Provider<'_>>,
            _out: &mut [u8],
            _out_size: Option<(i32, i32)>,
        ) -> Result<()> {
            Ok(())
        }

        fn set_array(&self, name: &str, data: &Feed) -> Result<bool> {
            if !self.feeds.iter().any(|(n, _)| n == name) {
                return Ok(false);
            }
            match data {
                Feed::DataU8(v) => self.bytes.borrow_mut().push((name.to_string(), v.clone())),
                Feed::Data(v) => self.floats.borrow_mut().push((name.to_string(), v.clone())),
                Feed::Unchanged => {}
            }
            Ok(true)
        }

        fn present(
            &mut self,
            _provider: Option<crate::engine::Provider<'_>>,
            _preview: Option<(&mut [u8], (i32, i32))>,
        ) -> Result<()> {
            Ok(())
        }

        fn set_param(
            &self,
            pass: Option<&str>,
            name: &str,
            _ty: &str,
            value: &ParamValue,
        ) -> Result<bool> {
            self.set
                .borrow_mut()
                .push((pass.map(str::to_string), name.to_string(), value.clone()));
            self.log
                .borrow_mut()
                .push(format!("param {name} {value:?}"));
            Ok(true)
        }

        /// Records the layer, and refuses a layer named `refused`.
        fn add_layer(&mut self, pass: &str, index: usize, layer: &Value) -> Result<()> {
            if layer["name"] == "refused" {
                return Err(Refused("a layer the scene refuses".to_string()).into());
            }
            self.log
                .borrow_mut()
                .push(format!("add {pass} {index} {layer}"));
            Ok(())
        }

        fn update_layer(&mut self, layer: &Value) -> Result<bool> {
            self.log.borrow_mut().push(format!("look {layer}"));
            Ok(true)
        }

        fn remove_layer(&mut self, name: &str) -> Result<()> {
            self.log.borrow_mut().push(format!("remove {name}"));
            Ok(())
        }

        /// Records the move, and refuses a place past 2.
        fn move_layer(&mut self, name: &str, index: usize) -> Result<()> {
            if index > 2 {
                return Err(Refused("a place past the pass's layers".to_string()).into());
            }
            self.log.borrow_mut().push(format!("move {name} {index}"));
            Ok(())
        }

        /// Records the layers of the calls, leaves a call to a layer whose
        /// name starts with `slow`, and finds a layer whose name starts with
        /// `same` at the look of its call already.
        fn update_calls(&mut self, calls: &[Call]) -> Result<crate::engine::Taken> {
            let names: Vec<&str> = calls.iter().map(|c| c.layer.as_str()).collect();
            self.log
                .borrow_mut()
                .push(format!("calls {}", names.join(" ")));
            let (left, changed) = (0..calls.len())
                .filter(|&k| !names[k].starts_with("same"))
                .partition(|&k| names[k].starts_with("slow"));
            Ok(crate::engine::Taken { left, changed })
        }

        /// Records the field and the value, and refuses a font_size as the
        /// scene does.
        fn set_layer(&self, at: &Address, given: Given) -> Result<()> {
            if at.field == "font_size" {
                return Err(Refused("a rebuild field".to_string()).into());
            }
            let name = match at.effect {
                Some(e) => format!("{}.{e}.{}", at.layer, at.field),
                None => format!("{}.{}", at.layer, at.field),
            };
            self.log.borrow_mut().push(format!("set {name} {given:?}"));
            self.layers.borrow_mut().push((name, format!("{given:?}")));
            Ok(())
        }
    }

    /// Run ticks until the process answers, or give up.
    /// Ticks until `done` holds, for up to 10 s: a scene's interpreter can take
    /// seconds to start on a loaded machine.
    fn pump(f: &mut SceneProcess, mut done: impl FnMut(&mut SceneProcess) -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut frame = 0u64;
        while Instant::now() < deadline {
            if f.tick(frame, frame as f64 / 60.0, 1.0 / 60.0).is_err() {
                return false;
            }
            if done(f) {
                return true;
            }
            frame += 1;
            thread::sleep(Duration::from_millis(10));
        }
        false
    }

    #[test]
    fn the_help_probe_reads_the_declared_arguments() {
        let d = tmpdir("probe");
        let path = script(&d, "feed", SCENE_SCRIPT);
        let Takes::Args(decls) = declared(&path, PROBE_WAIT) else {
            panic!("the scene declared nothing");
        };
        assert_eq!(decls.len(), 2);
        assert_eq!(decls[0].name, "emit");
        assert_eq!(decls[0].ty, "size");
        assert_eq!(decls[0].default, serde_json::json!([4, 2]));
        assert_eq!(decls[1].ty, "bool");
        // nothing is cached, so an edited scene answers with what it says now
        script(
            &d,
            "feed",
            "#!/bin/sh\necho '{\"args\": [{\"name\": \"gain\", \"type\": \"float\"}]}'\n",
        );
        let Takes::Args(after) = declared(&path, PROBE_WAIT) else {
            panic!("the edited scene declared nothing");
        };
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].name, "gain");
    }

    /// A scene still starting when the wait runs out is reported as that, so
    /// the listing can say why it is bare and how to skip the wait.
    #[test]
    fn a_scene_still_starting_when_the_wait_ends_says_so() {
        let d = tmpdir("slow");
        let path = script(&d, "slow", "#!/bin/sh\nsleep 30\n");
        let started = Instant::now();
        assert!(matches!(declared(&path, SOON), Takes::Starting));
        assert!(started.elapsed() >= SOON, "gave up before the wait");
        assert!(started.elapsed() < SOON + Duration::from_secs(3));
    }

    /// A scene that runs and answers nothing takes its arguments its own way,
    /// and it is not still starting: it is already gone when the wait ends.
    #[test]
    fn a_scene_that_answers_nothing_is_pass_through() {
        let d = tmpdir("passthrough");
        let path = script(&d, "quiet", "#!/bin/sh\nexit 0\n");
        assert!(matches!(declared(&path, PROBE_WAIT), Takes::PassThrough));
    }

    /// --no-probe asks nothing at all, so a scene still starting costs no wait.
    #[test]
    fn no_probe_asks_nothing() {
        let d = tmpdir("unasked");
        let path = script(&d, "slow", "#!/bin/sh\nsleep 30\n");
        let set = vec![("slow".to_string(), path)];
        let started = Instant::now();
        let declared = declare(&set, false);
        assert!(matches!(declared[0].args, Takes::Unasked));
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(help(&declared, "ghee serve --").contains("slow: not asked"));
    }

    /// The start runs the scene itself. Nothing runs `--ghee-args` first.
    #[test]
    fn a_start_costs_no_probe() {
        let d = tmpdir("no-probe");
        let path = script(&d, "feed", SCENE_SCRIPT);
        let (_scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.stop();
        // The fixture writes this file from its --ghee-args branch.
        assert!(!Path::new(&format!("{d}/probed")).exists());
    }

    #[test]
    fn an_executable_scene_is_a_procedural_scene_and_a_document_is_not() {
        let d = tmpdir("kind");
        let process = script(&d, "run.sh", "#!/bin/sh\nexit 0\n");
        assert_eq!(executable(&process), Some(process.clone()));
        let plain = format!("{d}/flat.scene.json");
        fs::write(&plain, "{}").unwrap();
        assert_eq!(executable(&plain), None);
        // a .py of a scene package carries the bit
        let py = script(&d, "pack.py", "#!/bin/sh\nexit 0\n");
        assert_eq!(executable(&format!("{d}/pack")), Some(py));
        fs::set_permissions(format!("{d}/pack.py"), fs::Permissions::from_mode(0o644)).unwrap();
        assert_eq!(executable(&format!("{d}/pack")), None);
    }

    /// A scene process reads the mark and answers the wire. A scene the user
    /// starts sees none, and hosts itself on the bindings.
    /// The shell runs from /bin, so the mark file takes the whole path.
    #[test]
    fn the_scene_process_carries_the_wire_mark() {
        let d = tmpdir("wire");
        let mark = format!("{d}/mark");
        let body = format!(
            "printf '%s %s' \"$GHEE_WIRE\" \"${{GHEE_LOCKSTEP-unset}}\" > {mark}\n\
             printf '{{\"passes\":[{{\"name\":\"main\",\"fragment\":\"void main(){{}}\"}}]}}\\n'\n\
             sleep 5\n"
        );
        let (_scene, mut f) = SceneProcess::start(
            "/bin/sh",
            &args(&["-c", &body]),
            Reach::Any,
            (32, 24),
            |_| {},
        )
        .unwrap();
        (f.term_after, f.kill_after) = (SOON, SOON);
        assert!(wait_until(SETTLE, || fs::read_to_string(&mark).is_ok()));
        assert_eq!(fs::read_to_string(&mark).unwrap(), "1 unset");
        drop(f);
    }

    #[test]
    fn the_handshake_line_becomes_the_scene() {
        let d = tmpdir("handshake");
        let path = script(&d, "feed", SCENE_SCRIPT);
        let (scene, process) = start_scene(&path, &no_args()).unwrap();
        assert_eq!(scene.passes.len(), 1);
        assert_eq!(scene.passes[0].arrays[0].name, "level");
        assert_eq!(scene.params.get("text").unwrap().ty, "string");
        drop(process);
    }

    /// The drain writes the process stderr to the terminal as it arrives.
    /// The message names the condition, and repeats none of that stderr.
    /// A reader with no terminal asks for the stderr with it.
    #[test]
    fn a_scene_that_dies_without_a_handshake_names_the_condition() {
        let d = tmpdir("nohandshake");
        let path = script(
            &d,
            "broken",
            "#!/bin/sh\necho 'boom in the process' >&2\nexit 3\n",
        );
        let e = failure(start_scene(&path, &no_args()));
        let shown = e.to_string();
        assert!(shown.contains("did not complete the handshake"), "{shown}");
        assert!(!shown.contains("boom in the process"), "{shown}");
        let full = with_stderr(&e);
        assert_eq!(full, format!("{shown}\nboom in the process"));
    }

    #[test]
    fn a_script_that_cannot_start_names_its_first_line() {
        let d = tmpdir("nostart");
        for (name, body, why) in [
            ("bare", "echo hi\n", "the file has no #! line"),
            (
                "gone",
                "#!/nonexistent/sh -e\necho hi\n",
                "the interpreter '/nonexistent/sh' on its #! line does not exist",
            ),
            (
                "crlf",
                "#!/bin/sh\r\necho hi\r\n",
                "the interpreter '/bin/sh\\r' on its #! line does not exist",
            ),
        ] {
            let path = script(&d, name, body);
            let e = failure(start_scene(&path, &no_args()));
            assert_eq!(e.to_string(), format!("{path}: {why}"), "{name}");
        }
    }

    #[test]
    fn messages_set_arrays_params_and_the_clean_flag() {
        let d = tmpdir("messages");
        let path = script(&d, "feed", SCENE_SCRIPT);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        assert!(pump(&mut f, |f| f.data.contains_key("level")));
        assert!(!f.incoming.is_empty(), "no param message");
        let mut stub = Stub {
            feeds: vec![("level".to_string(), ArraySize::Linear(4))],
            ..Stub::default()
        };
        f.update(&mut stub, 0, 0.0).unwrap();
        assert!(f.incoming.is_empty());
        let floats = stub.floats.borrow().clone();
        let [(name, values)] = floats.as_slice() else {
            panic!("no data for the declared array: {floats:?}");
        };
        assert_eq!((name.as_str(), values.len()), ("level", 4));
        assert_eq!(values[1] - values[0], 1.0);
        // the texture holds the array, so the engine never uploads it again
        for name in ["other", "level"] {
            assert!(matches!(
                f.provide(name, ArraySize::Linear(4), 0),
                Some(Feed::Unchanged)
            ));
        }
        f.update(&mut stub, 1, 0.0).unwrap();
        assert_eq!(stub.floats.borrow().len(), 1);
        let set = stub.set.borrow();
        let (pass, name, _) = set.first().expect("no param reached the renderer");
        assert_eq!((pass.as_deref(), name.as_str()), (Some("main"), "amt"));
        drop(set);
        // every second tick is clean, so the loop takes a skip
        assert!(pump(&mut f, |f| f.clean > 0));
        assert!(!f.dirty(9, 0.1));
    }

    /// The tick lines the process logs, once the file holds `want` of them.
    fn ticks(dir: &str, want: usize) -> Vec<Value> {
        let path = format!("{dir}/ticks.jsonl");
        let read = || fs::read_to_string(&path).unwrap_or_default();
        assert!(
            wait_until(Duration::from_secs(5), || read().lines().count() >= want),
            "the process read {} ticks, not {want}",
            read().lines().count()
        );
        read()
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    }

    #[test]
    fn the_ticks_reach_the_scene_in_its_own_directory() {
        let d = tmpdir("ticks");
        let path = script(&d, "feed", SCENE_SCRIPT);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        f.tick(0, 0.0, 1.0 / 60.0).unwrap();
        let first = ticks(&d, 1).remove(0);
        assert_eq!(first["frame"], Value::from(0));
        assert_eq!(first["dt"], Value::from(1.0 / 60.0));
        assert_eq!(first["time"], Value::from(0.0));
        // the first tick carries every param
        assert_eq!(
            first["params"],
            serde_json::json!({"text": "GHEE", "amt": 0.0})
        );
        f.tick(1, 0.5, 0.5).unwrap();
        let second = ticks(&d, 2).remove(1);
        assert_eq!(second["frame"], Value::from(1));
        assert_eq!(second["params"], serde_json::json!({}));
        drop(f);
        // stdin closed, and the process ended on it
        let log = fs::read_to_string(format!("{d}/ticks.jsonl")).unwrap();
        assert!(log.trim_end().ends_with("eof"), "{log}");
    }

    #[test]
    fn a_process_started_late_in_a_run_counts_its_ticks_from_0() {
        let d = tmpdir("late");
        let path = script(&d, "feed", SCENE_SCRIPT);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        f.tick(300, 5.0, 1.0 / 60.0).unwrap();
        ticks(&d, 1);
        f.tick(301, 5.5, 0.5).unwrap();
        let got: Vec<(Value, Value)> = ticks(&d, 2)
            .iter()
            .map(|t| (t["frame"].clone(), t["time"].clone()))
            .collect();
        assert_eq!(
            got,
            [
                (Value::from(0), Value::from(0.0)),
                (Value::from(1), Value::from(0.5))
            ]
        );
    }

    #[test]
    fn a_param_that_the_console_changed_reaches_the_next_tick() {
        let d = tmpdir("onparam");
        let path = script(&d, "feed", SCENE_SCRIPT);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        f.tick(0, 0.0, 0.016).unwrap();
        ticks(&d, 1);
        f.on_param(None, "text", &ParamValue::Str("WOW".to_string()));
        f.on_param(Some("main"), "amt", &ParamValue::Float(0.5));
        // not a param of this scene, and no scene param amt
        f.on_param(None, "nosuch", &ParamValue::Str("x".to_string()));
        f.on_param(None, "amt", &ParamValue::Float(0.7));
        f.tick(1, 0.016, 0.016).unwrap();
        let second = ticks(&d, 2).remove(1);
        assert_eq!(
            second["params"],
            serde_json::json!({"text": "WOW", "amt": 0.5})
        );
    }

    /// A process with two passes that each declare the float param `amt`,
    /// which writes each tick that it reads into ticks.jsonl.
    const SHARED_NAME: &str = r#"#!/bin/sh
printf '%s\n' '{"version": 1, "passes": [{"name": "first", "params": {"amt": {"type": "float", "value": 0.0}}, "fragment": "void main(){}"}, {"name": "second", "params": {"amt": {"type": "float", "value": 0.25}}, "fragment": "void main(){}"}]}'
while IFS= read -r line; do printf '%s\n' "$line" >> ticks.jsonl; done
"#;

    #[test]
    fn a_tick_names_the_param_of_a_second_pass_that_shares_its_name_as_pass_param() {
        let d = tmpdir("shared");
        let path = script(&d, "shared", SHARED_NAME);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        f.tick(0, 0.0, 0.016).unwrap();
        ticks(&d, 1);
        f.on_param(Some("second"), "amt", &ParamValue::Float(0.5));
        f.tick(1, 0.016, 0.016).unwrap();
        let seen = ticks(&d, 2);
        assert_eq!(
            seen[0]["params"],
            serde_json::json!({"amt": 0.0, "second.amt": 0.25})
        );
        assert_eq!(seen[1]["params"], serde_json::json!({"second.amt": 0.5}));
    }

    #[test]
    fn a_scene_that_ignores_stdin_runs_free() {
        let d = tmpdir("free");
        let path = script(&d, "free", FREE);
        let (scene, mut f) = start_deaf(&path, &no_args()).unwrap();
        f.bind(&scene);
        assert!(pump(&mut f, |f| f.data.contains_key("level")));
        let mut stub = Stub {
            feeds: vec![("level".to_string(), ArraySize::Linear(2))],
            ..Stub::default()
        };
        f.update(&mut stub, 0, 0.0).unwrap();
        assert_eq!(
            stub.floats.borrow().as_slice(),
            [("level".to_string(), vec![7.0f32, 8.0])]
        );
    }

    #[test]
    fn a_kind_4_message_reaches_the_render_target_as_bytes() {
        let d = tmpdir("u8");
        let path = script(&d, "mask", U8_SCRIPT);
        let (scene, mut f) = start_deaf(&path, &no_args()).unwrap();
        f.bind(&scene);
        assert!(pump(&mut f, |f| f.data.len() == 2));
        let mut stub = Stub {
            feeds: vec![
                ("mask".to_string(), ArraySize::Grid(2, 2)),
                ("state".to_string(), ArraySize::Linear(2)),
            ],
            ..Stub::default()
        };
        f.update(&mut stub, 0, 0.0).unwrap();
        // the bytes reach the target as bytes, and the floats stay floats
        assert_eq!(
            stub.bytes.borrow().as_slice(),
            [("mask".to_string(), vec![0u8, 64, 128, 255])]
        );
        assert_eq!(
            stub.floats.borrow().as_slice(),
            [("state".to_string(), vec![0.25f32, 1.0])]
        );
        // the texture holds the mask now, so the next frame uploads nothing,
        // and a setup empties it, so the frame after one uploads it again
        f.update(&mut stub, 1, 0.0).unwrap();
        assert_eq!(stub.bytes.borrow().len(), 1);
        f.rebuilt();
        f.update(&mut stub, 2, 0.0).unwrap();
        assert_eq!(stub.bytes.borrow().len(), 2);
        assert_eq!(stub.floats.borrow().len(), 2);
    }

    /// A process that sets its string param as kind 3 text, after a kind 1
    /// message to the same name, sets its float param `amp`, and sends kind 3
    /// text to the float param `gain`.
    const STRING_SCRIPT: &str = r#"#!/bin/sh
printf '%s\n' '{"version": 1, "params": {"words": {"type": "string", "value": "GHEE"}, "gain": {"type": "float", "value": 0.0}, "amp": {"type": "float", "value": 0.0}}, "passes": [{"name": "main", "fragment": "void main(){}"}]}'
# kind 1, words, the float 1.0
printf '\001\005words\001\000\000\000\000\000\200\077'
# kind 1, amp, the float 0.5
printf '\001\003amp\001\000\000\000\000\000\000\077'
# kind 3, gain, the text "x"
printf '\003\004gain\001\000\000\000x'
# kind 3, words, the text "hi"
printf '\003\005words\002\000\000\000hi'
while IFS= read -r line; do printf '%s\n' "$line" >> ticks.jsonl; done
"#;

    #[test]
    fn a_param_that_the_process_sets_reaches_no_tick() {
        let d = tmpdir("string");
        let path = script(&d, "words", STRING_SCRIPT);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        f.tick(0, 0.0, 1.0 / 60.0).unwrap();
        ticks(&d, 1);
        let text = |f: &SceneProcess| {
            f.incoming
                .iter()
                .any(|(n, v)| n == "words" && matches!(v, Set::Param(ParamData::Str(_))))
        };
        assert!(wait_until(Duration::from_secs(5), || {
            f.drain();
            text(&f)
        }));
        let mut stub = Stub::default();
        f.update(&mut stub, 0, 0.0).unwrap();
        // the newest message for a name wins, and text sets no float param
        assert_eq!(
            stub.set.borrow().as_slice(),
            [
                (None, "amp".to_string(), ParamValue::Float(0.5)),
                (None, "words".to_string(), ParamValue::Str("hi".to_string()))
            ]
        );
        assert_eq!(f.refused, HashSet::from(["gain".to_string()]));
        f.tick(1, 0.0, 1.0 / 60.0).unwrap();
        ticks(&d, 2);
        f.on_param(None, "words", &ParamValue::Str("GHEE".to_string()));
        f.tick(2, 0.0, 1.0 / 60.0).unwrap();
        let seen = ticks(&d, 3);
        assert_eq!(
            seen[0]["params"],
            serde_json::json!({"words": "GHEE", "gain": 0.0, "amp": 0.0})
        );
        assert_eq!(seen[1]["params"], serde_json::json!({}));
        assert_eq!(seen[2]["params"], serde_json::json!({"words": "GHEE"}));
    }

    /// A process that sets the color param `tint` to a colour name as kind 3
    /// text, and `bg` to text that names no colour.
    const COLOUR_SCRIPT: &str = r#"#!/bin/sh
printf '%s\n' '{"version": 1, "params": {"tint": {"type": "color", "value": [0, 0, 0]}, "bg": {"type": "color", "value": [0, 0, 0]}}, "passes": [{"name": "main", "fragment": "void main(){}"}]}'
# kind 3, tint, the text "Navy"
printf '\003\004tint\004\000\000\000Navy'
# kind 3, bg, the text "rde"
printf '\003\002bg\003\000\000\000rde'
while IFS= read -r line; do :; done
"#;

    #[test]
    fn a_kind_3_message_sets_a_color_param_to_a_colour_name_as_written() {
        let d = tmpdir("colour");
        let path = script(&d, "colour", COLOUR_SCRIPT);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        assert!(wait_until(Duration::from_secs(5), || {
            f.drain();
            f.incoming.len() == 2
        }));
        let mut stub = Stub::default();
        f.update(&mut stub, 0, 0.0).unwrap();
        let set = stub.set.borrow();
        let [(None, name, ParamValue::Color(c))] = set.as_slice() else {
            panic!("{set:?}");
        };
        // CSS's navy is #000080.
        assert_eq!((name.as_str(), c.rgb), ("tint", [0.0, 0.0, 128.0 / 255.0]));
        assert_eq!(c.to_json(), serde_json::json!("Navy"));
        assert_eq!(f.refused, HashSet::from(["bg".to_string()]));
    }

    /// A process that sends a param that the scene lacks, text to a float
    /// param, and one number to a color.
    const REFUSED_SCRIPT: &str = r#"#!/bin/sh
printf '%s\n' '{"version": 1, "params": {"gain": {"type": "float", "value": 0.0}, "tint": {"type": "color", "value": [0, 0, 0]}}, "passes": [{"name": "main", "fragment": "void main(){}"}]}'
# kind 1, nope, the float 1.0
printf '\001\004nope\001\000\000\000\000\000\200\077'
# kind 3, gain, the text "x"
printf '\003\004gain\001\000\000\000x'
# kind 1, tint, the float 1.0
printf '\001\004tint\001\000\000\000\000\000\200\077'
while IFS= read -r line; do :; done
"#;

    #[test]
    fn a_param_message_that_fits_no_param_is_refused_and_logged_once() {
        let d = tmpdir("refused");
        let path = script(&d, "refused", REFUSED_SCRIPT);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        assert!(wait_until(Duration::from_secs(5), || {
            f.drain();
            f.incoming.len() == 3
        }));
        let mut stub = Stub::default();
        f.update(&mut stub, 0, 0.0).unwrap();
        assert!(stub.set.borrow().is_empty());
        let mut refused: Vec<&str> = f.refused.iter().map(String::as_str).collect();
        refused.sort_unstable();
        assert_eq!(refused, ["gain", "nope", "tint"]);
    }

    /// A process in lockstep. It answers each tick 0.2 s late with the frame
    /// number as the text of `word`, then kind 5.
    const LOCKSTEP_SCRIPT: &str = r#"#!/bin/sh
printf '%s\n' '{"version": 1, "lockstep": true, "params": {"word": {"type": "string", "value": ""}}, "passes": [{"name": "main", "fragment": "void main(){}"}]}'
while IFS= read -r line; do
    n=${line#*\"frame\":}; n=${n%%,*}; n=${n%%\}*}
    sleep 0.2
    printf '\003\004word\001\000\000\000%s\005' "$n"
done
"#;

    #[test]
    fn a_first_or_pinned_tick_waits_for_a_lockstep_answer_and_a_later_one_draws_what_arrived() {
        let d = tmpdir("lockstep");
        let path = script(&d, "lockstep", LOCKSTEP_SCRIPT);
        let word = |stub: &Stub| match stub.set.borrow().last() {
            Some((_, _, ParamValue::Str(w))) => Some(w.clone()),
            _ => None,
        };
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        f.pin();
        let stub = &mut Stub::default();
        for frame in 0..3u64 {
            f.tick(frame, frame as f64 / 60.0, 1.0 / 60.0).unwrap();
            f.update(stub, frame, 0.0).unwrap();
            assert_eq!(word(stub), Some(frame.to_string()));
        }
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        let stub = &mut Stub::default();
        f.tick(0, 0.0, 1.0 / 60.0).unwrap();
        f.update(stub, 0, 0.0).unwrap();
        assert_eq!(word(stub), Some("0".to_string()));
        f.tick(1, 1.0 / 60.0, 1.0 / 60.0).unwrap();
        f.update(stub, 1, 0.0).unwrap();
        assert_eq!(word(stub), Some("0".to_string()));
        assert!(wait_until(Duration::from_secs(5), || {
            f.drain();
            !f.incoming.is_empty()
        }));
        f.update(stub, 2, 0.0).unwrap();
        assert_eq!(word(stub), Some("1".to_string()));
    }

    #[test]
    fn a_pinned_tick_never_waits_on_a_process_that_is_not_lockstep() {
        let d = tmpdir("free-pinned");
        let path = script(&d, "free", FREE);
        let (scene, mut f) = start_deaf(&path, &no_args()).unwrap();
        f.bind(&scene);
        f.pin();
        let t = Instant::now();
        for frame in 0..3u64 {
            f.tick(frame, 0.0, 1.0 / 60.0).unwrap();
        }
        assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
    }

    #[test]
    fn a_scene_that_exits_mid_run_is_an_error_that_holds_its_stderr() {
        let d = tmpdir("died");
        let path = script(
            &d,
            "shortlived",
            "#!/bin/sh\nprintf '{\"passes\":[{\"name\":\"main\",\"fragment\":\"void main(){}\"}]}\\n'\n\
             echo 'the process gave up' >&2\nexit 1\n",
        );
        let (_scene, mut f) = start_scene(&path, &no_args()).unwrap();
        let mut last = Ok(());
        for frame in 0..100u64 {
            last = f.tick(frame, 0.0, 0.016);
            if last.is_err() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let e = last.unwrap_err();
        let shown = e.to_string();
        assert!(shown.contains("the scene process exited"), "{shown}");
        // the terminal already holds the stderr, so the message repeats none
        assert!(!shown.contains("the process gave up"), "{shown}");
        assert!(with_stderr(&e).contains("the process gave up"), "{e}");
        // the error state holds, so every later tick reports the same message
        let again = f.tick(1, 0.0, 0.016).unwrap_err();
        assert_eq!(again.to_string(), shown);
        assert_eq!(with_stderr(&again), with_stderr(&e));
    }

    #[test]
    fn closed_stdin_stops_a_scene_and_a_stubborn_one_is_signalled() {
        let d = tmpdir("stop");
        let quick = script(&d, "quick", SCENE_SCRIPT);
        let (_scene, f) = start_scene(&quick, &no_args()).unwrap();
        let started = Instant::now();
        drop(f);
        assert!(
            started.elapsed() < TERM_AFTER,
            "stdin never closed: {:?}",
            started.elapsed()
        );
        // an ignored signal stays ignored across exec, and the exec leaves no
        // child to hold the pipes past the kill
        let stubborn = script(
            &d,
            "stubborn",
            "#!/bin/sh\ntrap '' TERM\nprintf '{\"passes\":[{\"name\":\"main\",\
             \"fragment\":\"void main(){}\"}]}\\n'\nexec sleep 10\n",
        );
        let (_scene, mut f) = start_scene(&stubborn, &no_args()).unwrap();
        let (term, kill) = (SOON * 4, SOON * 2);
        (f.term_after, f.kill_after) = (term, kill);
        let started = Instant::now();
        drop(f);
        let took = started.elapsed();
        assert!(took >= term + kill, "killed early: {took:?}");
        assert!(took < term + kill + Duration::from_secs(3), "{took:?}");
    }

    // the arguments of a set of scenes

    fn decl(name: &str, ty: &str, default: Value, help: &str) -> ArgDecl {
        ArgDecl {
            name: name.to_string(),
            ty: ty.to_string(),
            default,
            help: help.to_string(),
        }
    }

    fn hero() -> Declared {
        Declared {
            name: "hero".to_string(),
            args: Takes::Args(vec![
                decl("emit", "size", serde_json::json!([480, 300]), "the wall"),
                decl("loud", "bool", Value::from(false), "shout"),
            ]),
        }
    }

    fn plain(name: &str) -> Declared {
        Declared {
            name: name.to_string(),
            args: Takes::Args(Vec::new()),
        }
    }

    fn raw() -> Declared {
        Declared {
            name: "raw".to_string(),
            args: Takes::PassThrough,
        }
    }

    #[test]
    fn the_help_groups_the_arguments_under_the_scene_that_takes_them() {
        let text = help(&[hero(), plain("bars"), raw()], "ghee serve --");
        let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines[0], "usage: ghee serve -- [ARGS]");
        assert_eq!(lines[1], "hero:");
        assert_eq!(
            lines[2],
            format!("  {:<24}{}", "--emit WxH", "the wall (default: 480x300)")
        );
        assert_eq!(
            lines[3],
            format!("  {:<24}{}", "--loud, --no-loud", "shout (default: false)")
        );
        assert_eq!(lines[4], "bars: no arguments");
        assert_eq!(lines[5], "raw: pass-through");
        assert_eq!(lines.len(), 6);
    }

    #[test]
    fn only_the_help_request_alone_asks_for_the_help() {
        assert!(wants_help(&args(&["--help"])) && wants_help(&args(&["-h"])));
        // a scene owns every other spelling, its own --help included
        for tokens in [
            args(&["--emit", "8x8"]),
            args(&["--emit", "8x8", "--help"]),
            args(&["--help", "--emit"]),
            args(&[]),
        ] {
            assert!(!wants_help(&tokens), "{tokens:?}");
        }
    }

    #[test]
    fn the_help_runs_each_scene_of_the_set_on_demand() {
        let d = tmpdir("declare");
        // a shell answer, because the set already holds a scene that spends the
        // whole wait, and a loaded machine must not race an interpreter start
        let path = script(
            &d,
            "feed",
            "#!/bin/sh\necho '{\"args\": [{\"name\": \"emit\", \"type\": \"size\"},\
             {\"name\": \"loud\", \"type\": \"bool\"}]}'\n",
        );
        let quiet = script(&d, "quiet", "#!/bin/sh\nexit 0\n");
        let json = format!("{d}/flat.scene.json");
        fs::write(&json, "{}").unwrap();
        let set = vec![
            ("feed".to_string(), path),
            ("quiet".to_string(), quiet),
            ("flat".to_string(), json),
        ];
        let declared = declare(&set, true);
        let Takes::Args(args) = &declared[0].args else {
            panic!("feed declared nothing: {:?}", declared[0].args);
        };
        assert_eq!(args.len(), 2);
        let text = help(&declared, "ghee serve --");
        assert!(text.contains("--emit WxH"), "{text}");
        // a scene that answers nothing, and a scene with no process at all
        assert!(text.contains("quiet: pass-through"), "{text}");
        assert!(text.contains("flat: no arguments"), "{text}");
    }

    /// An unknown token reaches the scene. The scene refuses it, and
    /// its own words reach the caller through the handshake failure.
    #[test]
    fn an_unknown_token_reaches_the_scene() {
        let d = tmpdir("tokens");
        let path = script(
            &d,
            "picky",
            "#!/bin/sh\ntest \"$1\" = --emit || { echo \"picky: unrecognized: $1\" >&2; exit 2; }\n\
             printf '{\"passes\":[{\"name\":\"main\",\"fragment\":\"void main(){}\"}]}\\n'\nsleep 30\n",
        );
        let e = failure(start_scene(&path, &args(&["--bogus"])));
        assert!(
            e.to_string().contains("did not complete the handshake"),
            "{e}"
        );
        assert!(
            with_stderr(&e).contains("picky: unrecognized: --bogus"),
            "{e}"
        );
        // the scene's own token passes through the same way
        start_deaf(&path, &args(&["--emit"])).unwrap();
    }

    #[test]
    fn an_unreadable_message_kind_ends_the_scene_process() {
        let d = tmpdir("badkind");
        let path = script(
            &d,
            "garbage",
            "#!/bin/sh\nprintf '{\"passes\":[{\"name\":\"main\",\"fragment\":\"void main(){}\"}]}\\n'\n\
             printf '\\013'\nsleep 5\n",
        );
        let (_scene, mut f) = start_deaf(&path, &no_args()).unwrap();
        let mut last = Ok(());
        for frame in 0..100u64 {
            last = f.tick(frame, 0.0, 0.016);
            if last.is_err() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let e = last.unwrap_err().to_string();
        assert!(e.contains("wrote message kind 11 (0 to 10 only)"), "{e}");
    }

    const DOTTED_SCRIPT: &str = r#"#!/bin/sh
printf '%s\n' '{"version": 1, "params": {}, "passes": [{"name": "draw", "fragment": "void main(){}", "params": {"color": {"type": "float", "value": 0.2}}}, {"name": "dither", "fragment": "void main(){}", "params": {"color": {"type": "float", "value": 0.0}}}]}'
# kind 1: dither.color = 0.65, then color = 0.1
printf '\001\014dither.color\001\000\000\000\146\146\046\077'
printf '\001\005color\001\000\000\000\315\314\314\075'
# the first tick can reach the process before ghee reads both messages, and an
# exit then ends the run early, so the process waits for ghee to close stdin
while IFS= read -r line; do :; done
"#;

    #[test]
    fn a_dotted_name_addresses_the_param_of_one_pass() {
        let d = tmpdir("dotted");
        let path = script(&d, "feed", DOTTED_SCRIPT);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        assert!(pump(&mut f, |f| f.incoming.len() >= 2));
        let mut stub = Stub::default();
        f.update(&mut stub, 0, 0.0).unwrap();
        let set = stub.set.borrow();
        let routed: Vec<(Option<&str>, &str, &ParamValue)> = set
            .iter()
            .map(|(p, n, v)| (p.as_deref(), n.as_str(), v))
            .collect();
        assert!(routed
            .iter()
            .any(|(p, n, v)| (*p, *n) == (Some("dither"), "color")
                && matches!(v, ParamValue::Float(f) if (*f - 0.65).abs() < 1e-6)));
        // a bare colliding name goes to the first pass that declares it
        assert!(routed
            .iter()
            .any(|(p, n, _)| (*p, *n) == (Some("draw"), "color")));
    }

    /// The printf line that writes a message of `kind` for `name`, with the
    /// count `count` and the payload `payload`, as the wire frames it.
    fn message(kind: u8, name: &str, count: u32, payload: &[u8]) -> String {
        let mut bytes = vec![kind, name.len() as u8];
        bytes.extend(name.as_bytes());
        bytes.extend(count.to_le_bytes());
        bytes.extend(payload);
        let octal: String = bytes.iter().map(|b| format!("\\{b:03o}")).collect();
        format!("printf '{octal}'\n")
    }

    /// A kind 1 message of `values`.
    fn numbers(name: &str, values: &[f32]) -> String {
        let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        message(1, name, values.len() as u32, &payload)
    }

    #[test]
    fn a_first_part_that_names_a_layer_sets_its_live_field_and_a_pass_param_stays() {
        let d = tmpdir("layers");
        let pic = crate::output::encode_png(crate::looper::Frame {
            data: &[255; 4],
            width: 1,
            height: 1,
            components: 4,
        })
        .unwrap();
        fs::write(format!("{d}/pic.png"), pic).unwrap();
        let doc = serde_json::json!({
            "version": 1,
            "assets": {"pic": {"file": "pic.png"}},
            "passes": [{"name": "draw", "fragment": "void main(){}",
                        "params": {"color": {"type": "float", "value": 0.2}},
                        "layers": [{"name": "title", "text": "hi", "karaoke": {}},
                                   {"name": "icon", "image": "pic"}]}],
        });
        let body = [
            "#!/bin/sh\n".to_string(),
            format!("printf '%s\\n' '{doc}'\n"),
            numbers("title.x", &[5.0]),
            numbers("title.karaoke.sung", &[2.5]),
            message(3, "icon.image", 3, b"pic"),
            numbers("title.font_size", &[20.0]),
            numbers("draw.color", &[0.65]),
            numbers("nope.x", &[1.0]),
            numbers("title.karaoke.sung.more", &[1.0]),
            "while IFS= read -r line; do :; done\n".to_string(),
        ]
        .concat();
        let path = script(&d, "feed", &body);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        assert!(pump(&mut f, |f| f.incoming.len() >= 7));
        let mut stub = Stub::default();
        f.update(&mut stub, 0, 0.0).unwrap();
        let fields = |n: &str, v: &str| (n.to_string(), v.to_string());
        assert_eq!(
            *stub.layers.borrow(),
            [
                fields("title.x", "Numbers([5.0])"),
                fields("title.karaoke.sung", "Numbers([2.5])"),
                fields("icon.image", "Text(\"pic\")"),
            ]
        );
        assert_eq!(
            *stub.set.borrow(),
            [(
                Some("draw".to_string()),
                "color".to_string(),
                ParamValue::Float(f64::from(0.65f32))
            )]
        );
        let mut refused: Vec<&String> = f.refused.iter().collect();
        refused.sort();
        assert_eq!(
            refused,
            ["nope.x", "title.font_size", "title.karaoke.sung.more"]
        );
    }

    #[test]
    fn an_array_message_that_names_a_layer_sets_its_live_field() {
        let d = tmpdir("points");
        let doc = serde_json::json!({
            "version": 1,
            "passes": [{"name": "draw", "fragment": "void main(){}",
                        "arrays": [{"name": "level", "size": 2}],
                        "layers": [
                            {"name": "scope", "shape": "line", "points": [[0, 0], [1, 1]]},
                            {"name": "wave", "shape": "line", "points": [[0, 0], [1, 1]]}]}],
        });
        let floats =
            |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
        let body = [
            "#!/bin/sh\n".to_string(),
            format!("printf '%s\\n' '{doc}'\n"),
            message(0, "scope.points", 4, &floats(&[0.0, 16.0, 8.0, 4.0])),
            message(4, "wave.points", 4, &[0, 16, 8, 4]),
            message(0, "level", 2, &floats(&[0.5, 1.0])),
            "while IFS= read -r line; do :; done\n".to_string(),
        ]
        .concat();
        let path = script(&d, "feed", &body);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        assert!(pump(&mut f, |f| f.data.len() >= 3));
        let mut stub = Stub {
            feeds: vec![("level".to_string(), ArraySize::Linear(2))],
            ..Stub::default()
        };
        f.update(&mut stub, 0, 0.0).unwrap();
        assert_eq!(
            *stub.layers.borrow(),
            [(
                "scope.points".to_string(),
                "Numbers([0.0, 16.0, 8.0, 4.0])".to_string()
            )]
        );
        assert_eq!(f.refused, HashSet::from(["wave.points".to_string()]));
        // A declared array stays an array.
        assert_eq!(
            stub.floats.borrow().as_slice(),
            [("level".to_string(), vec![0.5f32, 1.0])]
        );
    }

    #[test]
    fn a_text_layers_words_are_its_text_on_the_wire_both_ways() {
        let d = tmpdir("words");
        let doc = serde_json::json!({
            "version": 1,
            "passes": [{"name": "draw", "fragment": "void main(){}",
                        "layers": [{"name": "title", "text": "hi"}]}],
        });
        let body = [
            "#!/bin/sh\n".to_string(),
            format!("printf '%s\\n' '{doc}'\n"),
            message(3, "title.text", 5, b"la la"),
            "while IFS= read -r line; do printf '%s\\n' \"$line\" >> ticks.jsonl; done\n"
                .to_string(),
        ]
        .concat();
        let path = script(&d, "feed", &body);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        f.tick(0, 0.0, 1.0 / 60.0).unwrap();
        ticks(&d, 1);
        assert!(wait_until(Duration::from_secs(5), || {
            f.drain();
            !f.incoming.is_empty()
        }));
        let mut stub = Stub::default();
        f.update(&mut stub, 0, 0.0).unwrap();
        assert!(stub.set.borrow().is_empty());
        assert_eq!(
            *stub.layers.borrow(),
            [("title.text".to_string(), "Text(\"la la\")".to_string())]
        );
        assert!(f.refused.is_empty());
        // the process's own words stay with it, and the console's reach it
        f.tick(1, 0.0, 1.0 / 60.0).unwrap();
        ticks(&d, 2);
        let at = Address {
            layer: "title",
            effect: None,
            field: "text",
        };
        f.on_layer(&at, &Value::from("from the console"));
        f.tick(2, 0.0, 1.0 / 60.0).unwrap();
        let seen: Vec<Value> = ticks(&d, 3)
            .into_iter()
            .map(|t| t["params"].clone())
            .collect();
        assert_eq!(
            seen,
            [
                serde_json::json!({"title.text": "hi"}),
                serde_json::json!({}),
                serde_json::json!({"title.text": "from the console"}),
            ]
        );
    }

    /// A process that adds the text layer `title` to pass `main`, sets its
    /// words and its x, gives it a new look, adds the line `scope`, adds a
    /// layer that the renderer refuses, sends a kind 6 message that holds no
    /// layer, then sets the points of `scope` in a kind 0 message.
    fn grow_script() -> String {
        let title = r#"{"index": 1, "layer": {"name": "title", "text": "hi"}}"#;
        let look = r#"{"name": "title", "text": "ho", "font_size": 20}"#;
        let scope = r#"{"index": 2, "layer": {"name": "scope", "shape": "line", "points": [[0, 0], [1, 1]]}}"#;
        let refused = r#"{"index": 0, "layer": {"name": "refused", "shape": "rect"}}"#;
        let points: Vec<u8> = [0f32, 1.0, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let doc = serde_json::json!({
            "version": 1,
            "passes": [{"name": "main", "fragment": "void main(){}",
                        "layers": [{"name": "dot", "shape": "rect"}]}],
        });
        [
            "#!/bin/sh\n".to_string(),
            format!("printf '%s\\n' '{doc}'\n"),
            message(6, "main", title.len() as u32, title.as_bytes()),
            message(3, "title.text", 2, b"ho"),
            numbers("title.x", &[4.0]),
            message(7, "title", look.len() as u32, look.as_bytes()),
            message(6, "main", scope.len() as u32, scope.as_bytes()),
            message(6, "main", refused.len() as u32, refused.as_bytes()),
            message(6, "main", 4, b"nope"),
            message(0, "scope.points", 4, &points),
            "while IFS= read -r line; do printf '%s\\n' \"$line\" >> ticks.jsonl; done\n"
                .to_string(),
        ]
        .concat()
    }

    #[test]
    fn a_kind_6_message_adds_a_layer_that_later_messages_address() {
        let d = tmpdir("grow");
        let path = script(&d, "grow", &grow_script());
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        assert!(wait_until(Duration::from_secs(5), || {
            f.drain();
            f.data.contains_key("scope.points")
        }));
        let mut stub = Stub::default();
        f.update(&mut stub, 0, 0.0).unwrap();
        let title = r#"{"name":"title","text":"hi"}"#;
        let look = r#"{"name":"title","text":"ho","font_size":20}"#;
        let scope = r#"{"name":"scope","shape":"line","points":[[0,0],[1,1]]}"#;
        assert_eq!(
            *stub.log.borrow(),
            [
                format!("add main 1 {title}"),
                "set title.text Text(\"ho\")".to_string(),
                "set title.x Numbers([4.0])".to_string(),
                format!("look {look}"),
                format!("add main 2 {scope}"),
                "set scope.points Numbers([0.0, 1.0, 2.0, 3.0])".to_string(),
            ]
        );
        let mut refused: Vec<&String> = f.refused.iter().collect();
        refused.sort();
        assert_eq!(refused, ["main", "refused"]);
        // the console's words for the added layer reach the process, and
        // the process's own words stay with it
        let at = Address {
            layer: "title",
            effect: None,
            field: "text",
        };
        f.on_layer(&at, &Value::from("from the console"));
        f.tick(0, 0.0, 1.0 / 60.0).unwrap();
        let seen = ticks(&d, 1);
        assert_eq!(
            seen[0]["params"],
            serde_json::json!({"title.text": "from the console"})
        );
    }

    #[test]
    fn a_layer_message_that_holds_no_layer_says_what_it_needs() {
        assert_eq!(
            added(
                "main",
                r#"{"index": -1, "layer": {"name": "t", "text": "hi"}}"#
            ),
            Err(
                "kind 6 to pass 'main' needs {\"index\": n, \"layer\": {...}}, n a whole number \
                 from 0"
                    .to_string()
            )
        );
        for text in [r#"{"index": 0}"#, r#"{"index": 0, "layer": {}, "more": 1}"#] {
            let e = added("main", text).unwrap_err();
            assert!(e.starts_with("kind 6 to pass 'main' needs"), "{e}");
        }
        let e = added("main", "nope").unwrap_err();
        assert!(e.starts_with("kind 6 to pass 'main' is no JSON: "), "{e}");
        assert_eq!(
            look_of("title", r#"{"name": "other", "text": "hi"}"#),
            Err("kind 7 to layer 'title' holds no layer named 'title'".to_string())
        );
        let e = look_of("title", "[1]").unwrap_err();
        assert!(
            e.starts_with("kind 7 to layer 'title' holds no layer"),
            "{e}"
        );
        assert_eq!(
            place_of("title", r#"{"index": -1}"#),
            Err(
                "kind 10 to layer 'title' needs {\"index\": n}, n a whole number from 0"
                    .to_string()
            )
        );
        for text in [r#"{}"#, r#"{"index": 0, "more": 1}"#] {
            let e = place_of("title", text).unwrap_err();
            assert!(e.starts_with("kind 10 to layer 'title' needs"), "{e}");
        }
        let e = place_of("title", "nope").unwrap_err();
        assert!(
            e.starts_with("kind 10 to layer 'title' is no JSON: "),
            "{e}"
        );
    }

    #[test]
    fn a_kind_9_message_takes_a_layer_out_and_a_kind_10_message_moves_one() {
        let d = tmpdir("place");
        let doc = serde_json::json!({
            "version": 1,
            "passes": [{"name": "main", "fragment": "void main(){}",
                        "layers": [{"name": "dot", "shape": "rect"},
                                   {"name": "bar", "shape": "rect"},
                                   {"name": "title", "text": "hi"}]}],
        });
        let (first, past) = (r#"{"index": 0}"#, r#"{"index": 3}"#);
        let body = [
            "#!/bin/sh\n".to_string(),
            format!("printf '%s\\n' '{doc}'\n"),
            message(10, "bar", first.len() as u32, first.as_bytes()),
            message(9, "title", 0, b""),
            message(10, "dot", past.len() as u32, past.as_bytes()),
            numbers("bar.x", &[1.0]),
            "while IFS= read -r line; do :; done\n".to_string(),
        ]
        .concat();
        let path = script(&d, "place", &body);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        assert!(wait_until(Duration::from_secs(5), || {
            f.drain();
            f.incoming.iter().any(|(n, _)| n == "bar.x")
        }));
        let mut stub = Stub::default();
        f.update(&mut stub, 0, 0.0).unwrap();
        assert_eq!(
            *stub.log.borrow(),
            [
                "move bar 0".to_string(),
                "remove title".to_string(),
                "set bar.x Numbers([1.0])".to_string(),
            ]
        );
        assert_eq!(f.refused.iter().collect::<Vec<_>>(), ["dot"]);
        // the words of the layer that left go out with no tick
        assert!(!f.values.contains_key("title.text"));
    }

    #[test]
    fn a_kind_8_message_gives_each_layer_its_look_and_the_calls_go_at_once() {
        // title's look reads as a call, the renderer leaves slow's call,
        // and odd's width is no key of a text call: the last two go as
        // looks, in the order of the list.
        let d = tmpdir("looks");
        let doc = serde_json::json!({
            "version": 1,
            "passes": [{"name": "main", "fragment": "void main(){}",
                        "layers": [{"name": "title", "text": "hi"},
                                   {"name": "slow", "shape": "rect"},
                                   {"name": "odd", "text": "o"},
                                   {"name": "same", "shape": "rect"}]}],
        });
        let looks = r#"[{"name": "title", "text": "ho", "x": 4},
            {"name": "slow", "shape": "rect", "x": 2},
            {"name": "odd", "text": "o", "width": 3},
            {"name": "same", "shape": "rect"}]"#;
        let body = [
            "#!/bin/sh\n".to_string(),
            format!("printf '%s\\n' '{doc}'\n"),
            message(8, "", looks.len() as u32, looks.as_bytes()),
            numbers("title.y", &[1.0]),
            "while IFS= read -r line; do :; done\n".to_string(),
        ]
        .concat();
        let path = script(&d, "looks", &body);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        let locks = Arc::new(Mutex::new(Locks::default()));
        f.lock_into(Arc::clone(&locks));
        assert!(wait_until(Duration::from_secs(5), || {
            f.drain();
            f.incoming.iter().any(|(n, _)| n == "title.y")
        }));
        let mut stub = Stub::default();
        f.update(&mut stub, 0, 0.0).unwrap();
        assert_eq!(
            *stub.log.borrow(),
            [
                "calls title slow same".to_string(),
                r#"look {"name":"slow","shape":"rect","x":2}"#.to_string(),
                r#"look {"name":"odd","text":"o","width":3}"#.to_string(),
                "set title.y Numbers([1.0])".to_string(),
            ]
        );
        // the process's own words, which no tick carries back
        assert_eq!(f.sent.get("title.text"), Some(&Value::from("ho")));
        for name in ["title", "slow", "odd", "same"] {
            assert!(lock(&locks).layer(name), "{name}");
        }
    }

    #[test]
    fn the_locks_of_a_scene_process_end_when_it_exits() {
        let d = tmpdir("locks");
        let doc = r#"{"version": 1, "params": {"amt": {"type": "float", "value": 0}}, "passes": [{"name": "main", "fragment": "void main(){}"}]}"#;
        let body = [
            "#!/bin/sh\n".to_string(),
            format!("printf '%s\\n' '{doc}'\n"),
            numbers("amt", &[0.5]),
            "read -r line\n".to_string(),
        ]
        .concat();
        let path = script(&d, "locks", &body);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        let locks = Arc::new(Mutex::new(Locks::default()));
        f.lock_into(Arc::clone(&locks));
        assert!(wait_until(Duration::from_secs(5), || {
            f.drain();
            f.incoming.iter().any(|(n, _)| n == "amt")
        }));
        f.update(&mut Stub::default(), 0, 0.0).unwrap();
        assert!(lock(&locks).param(None, "amt"));
        // the script exits once it reads a tick
        assert!(wait_until(Duration::from_secs(5), || f
            .tick(1, 0.0, 1.0 / 60.0)
            .is_err()));
        assert_eq!(*lock(&locks), Locks::default());
    }

    #[test]
    fn a_kind_8_message_that_holds_no_list_of_layers_says_what_it_needs() {
        let e = looks_of("nope").unwrap_err();
        assert!(e.starts_with("kind 8 is no JSON: "), "{e}");
        for text in [
            r#"{"name": "t"}"#,
            r#"[{"text": "hi"}]"#,
            r#"[{"name": 1}]"#,
        ] {
            assert_eq!(
                looks_of(text),
                Err("kind 8 needs a list of layers, each with its name".to_string())
            );
        }
    }

    #[test]
    fn the_answers_of_several_ticks_leave_each_layer_in_its_newest_state() {
        let d = tmpdir("newest");
        let doc = serde_json::json!({
            "version": 1,
            "passes": [{"name": "main", "fragment": "void main(){}",
                        "layers": [{"name": "dot", "shape": "rect"},
                                   {"name": "bar", "shape": "rect"}]}],
        });
        let drawn = |x: u32| format!(r#"{{"name": "dot", "shape": "rect", "x": {x}}}"#);
        let (first, again) = (drawn(1), drawn(2));
        let body = [
            "#!/bin/sh\n".to_string(),
            format!("printf '%s\\n' '{doc}'\n"),
            // one tick draws dot, the next hides it, the next draws it again
            message(7, "dot", first.len() as u32, first.as_bytes()),
            numbers("dot.visible", &[0.0]),
            message(7, "dot", again.len() as u32, again.as_bytes()),
            numbers("bar.x", &[1.0]),
            numbers("bar.x", &[2.0]),
            "while IFS= read -r line; do :; done\n".to_string(),
        ]
        .concat();
        let path = script(&d, "newest", &body);
        let (scene, mut f) = start_scene(&path, &no_args()).unwrap();
        f.bind(&scene);
        let last = |(n, s): &(String, Set)| {
            n == "bar.x" && matches!(s, Set::Param(ParamData::F32(v)) if v[..] == [2.0])
        };
        assert!(wait_until(Duration::from_secs(5), || {
            f.drain();
            f.incoming.iter().any(last)
        }));
        let mut stub = Stub::default();
        f.update(&mut stub, 0, 0.0).unwrap();
        assert_eq!(
            *stub.log.borrow(),
            [
                "set dot.visible Numbers([0.0])".to_string(),
                r#"look {"name":"dot","shape":"rect","x":2}"#.to_string(),
                "set bar.x Numbers([2.0])".to_string(),
            ]
        );
    }
}
