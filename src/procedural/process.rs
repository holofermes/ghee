//! Procedural scenes (docs/procedural.md).
//! A procedural scene is an executable that supplies a scene and its data.
//!
//! ghee starts the process when the scene becomes active.
//! The process writes one scene JSON line.
//! The process then writes binary messages that set arrays and live params.
//! ghee writes one JSON line to the process stdin for each render tick.
//!
//! The render thread never blocks on the process.
//! A reader thread takes the messages into a mailbox.
//! A writer thread takes the ticks out of a second mailbox.
//! The frame path reads the one mailbox and writes the other.

use std::collections::HashMap;
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

use crate::engine::Feed;
use crate::looper::{RenderTarget, SceneController};
use crate::procedural::resolve;
use crate::scene::{ArraySize, ParamValue, Scene};

/// `-- --help` waits this long for a scene's declarations line.
/// Nothing else runs a scene with `--ghee-args`.
const PROBE_WAIT: Duration = Duration::from_secs(2);

/// The handshake waits this long for the scene line.
/// A first `uv run` resolves packages before the process runs a statement.
const HANDSHAKE_WAIT: Duration = Duration::from_secs(30);

/// The stop sequence: closed stdin, then SIGTERM, then SIGKILL.
const TERM_AFTER: Duration = Duration::from_secs(2);
const KILL_AFTER: Duration = Duration::from_secs(1);

/// A stop waits this long for the pipe threads to end on their own.
const SETTLE: Duration = Duration::from_millis(200);

/// The message of a dead process waits this long for its stderr.
/// The drain thread ends at EOF, and the process is already reaped here.
/// The wait covers scheduling alone.
/// A loaded machine drops the stderr from the message without the wait.
const EPITAPH_WAIT: Duration = Duration::from_secs(1);

/// The error state keeps this much of the process stderr.
const STDERR_TAIL: usize = 8192;

/// A message count no scene has.
/// The reader refuses such a message instead of allocating for it.
const MAX_VALUES: usize = 1 << 22;

/// The mark of a scene process ghee started.
/// A scene the user starts sees no such variable, and hosts itself instead.
const WIRE: &str = "GHEE_WIRE";

const SET_ARRAY: u8 = 0;
const SET_PARAM: u8 = 1;
const CLEAN: u8 = 2;
const SET_ARRAY_U8: u8 = 4;

const SIGTERM: c_int = 15;

extern "C" {
    fn kill(pid: c_int, sig: c_int) -> c_int;
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ------------------------------------------------------------ what a path is

/// True for a regular file anyone may execute.
fn runnable(path: &str) -> bool {
    fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

/// The executable a scene path names.
/// A scene package names its `<name>.py`, and any other path names itself.
/// None means the path holds no process.
pub fn executable(path: &str) -> Option<String> {
    if let (_, Some(py)) = resolve(path) {
        if runnable(&py) {
            return Some(py);
        }
    }
    // a .scene.json is a document whatever its mode bits say
    if runnable(path) && !path.ends_with(crate::procedural::SUFFIX) {
        return Some(path.to_string());
    }
    None
}

/// The directory the process runs in, and the program path for any directory.
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

// ------------------------------------------------------- what a scene takes

/// One argument a process declares.
#[derive(Clone, Debug, PartialEq)]
pub struct ArgDecl {
    pub name: String,
    pub ty: String,
    pub default: Value,
    pub help: String,
}

/// A scene in the `-- --help` listing.
/// `args` is None for a pass-through process.
/// A pass-through process answers nothing, and ghee lists no argument for it.
#[derive(Clone, Debug)]
pub struct Declared {
    pub name: String,
    pub args: Option<Vec<ArgDecl>>,
}

/// What a process declares, read now. None is a pass-through.
/// Only `-- --help` calls this. A render and a serve never probe.
fn declarations(path: &str) -> Option<Vec<ArgDecl>> {
    probe(path).and_then(|line| parse_declarations(&line))
}

/// Run `<process> --ghee-args` and read the one line it answers with.
fn probe(path: &str) -> Option<String> {
    let (program, dir) = where_to_run(path).ok()?;
    let mut child = Command::new(program)
        .arg("--ghee-args")
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let out = child.stdout.take()?;
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut line = Vec::new();
        if BufReader::new(out).read_until(b'\n', &mut line).is_ok() {
            let _ = tx.send(String::from_utf8_lossy(&line).into_owned());
        }
    });
    let answer = rx.recv_timeout(PROBE_WAIT).ok();
    let _ = child.kill();
    let _ = child.wait();
    answer
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
pub fn declare(scenes: &[(String, String)]) -> Vec<Declared> {
    scenes
        .iter()
        .map(|(name, path)| Declared {
            name: name.clone(),
            args: match executable(path) {
                Some(exe) => declarations(&exe),
                None => Some(Vec::new()),
            },
        })
        .collect()
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
        let Some(args) = &scene.args else {
            out.push_str(&format!("\n{}: pass-through\n", scene.name));
            continue;
        };
        if args.is_empty() {
            out.push_str(&format!("\n{}: no arguments\n", scene.name));
            continue;
        }
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

/// One array message, in the form the process wrote it.
/// The bytes reach the engine as bytes, so a mask pays for no conversion.
enum ArrayData {
    F32(Vec<f32>),
    U8(Vec<u8>),
}

/// What the reader thread takes off stdout between two drains.
#[derive(Default)]
struct Inbox {
    arrays: Vec<(String, ArrayData)>,
    params: Vec<(String, Vec<f32>)>,
    /// The ticks the process calls clean.
    clean: u32,
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
/// A new tick absorbs the tick the writer has not taken.
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
/// A dead scene process adds the stderr the drain already wrote.
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
    outbox: Arc<(Mutex<Outbox>, Condvar)>,
    stderr: Arc<Mutex<String>>,
    /// The stderr drain. An epitaph waits for it before it reads the tail.
    errors: Option<JoinHandle<()>>,
    /// The tick writer. It holds the process stdin, so it ends the run.
    writer: Option<JoinHandle<()>>,
    threads: Vec<JoinHandle<()>>,
    /// Array data no frame takes yet, newest for each name.
    data: HashMap<String, ArrayData>,
    /// Param values that wait for a renderer.
    incoming: Vec<(String, Vec<f32>)>,
    /// param name -> (the pass that holds it, its type). None is scene scope.
    params: HashMap<String, (Option<String>, String)>,
    /// The string params as they stand, and the last values the process read.
    strings: HashMap<String, String>,
    sent: HashMap<String, String>,
    /// Ticks the process calls clean that the loop has not skipped.
    clean: u32,
    /// The process died. The scene stays in this error state.
    dead: Option<SceneFailure>,
}

impl SceneProcess {
    /// Start the process and take the handshake.
    /// The scene belongs to the process.
    /// The caller compiles the scene, then binds this process to it.
    /// `args` are the tokens after `--`, as they came. The process parses them,
    /// and a process that refuses one dies with its own message on stderr.
    pub fn start(path: &str, args: &[String]) -> Result<(Scene, SceneProcess)> {
        let (program, dir) = where_to_run(path)?;
        let mut child = Command::new(program)
            .args(args)
            .current_dir(&dir)
            .env(WIRE, "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("{path}: {e}"))?;
        crate::sources::record(path);
        let inbox = Arc::new(Mutex::new(Inbox::default()));
        let stderr = Arc::new(Mutex::new(String::new()));
        let outbox = Arc::new((Mutex::new(Outbox::default()), Condvar::new()));
        let (tx, rx): (Sender<String>, Receiver<String>) = mpsc::channel();
        let errors = drain_stderr(take_stderr(&mut child)?, Arc::clone(&stderr));
        let threads = vec![read_stream(
            take_stdout(&mut child)?,
            tx,
            Arc::clone(&inbox),
        )];
        let stdin = take_stdin(&mut child)?;
        let mut process = SceneProcess {
            name: path.to_string(),
            child,
            inbox,
            outbox,
            stderr,
            errors: Some(errors),
            writer: None,
            threads,
            data: HashMap::new(),
            incoming: Vec::new(),
            params: HashMap::new(),
            strings: HashMap::new(),
            sent: HashMap::new(),
            clean: 0,
            dead: None,
        };
        let line = match rx.recv_timeout(HANDSHAKE_WAIT) {
            Ok(line) => line,
            Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => {
                let why = process.epitaph("did not complete the handshake");
                return Err(anyhow!(why));
            }
        };
        let doc: Value = serde_json::from_str(line.trim()).map_err(|e| {
            anyhow!(process.epitaph(&format!("handshake line is not a scene object: {e}")))
        })?;
        let scene = Scene::from_dict(&doc, Some(&dir), Some(path))?;
        process.writer = Some(write_ticks(stdin, Arc::clone(&process.outbox)));
        Ok((scene, process))
    }

    /// Index the params of the compiled scene.
    /// The set holds the handshake params, plus the params filters and saved
    /// values add.
    /// The first tick carries every string param.
    pub fn bind(&mut self, scene: &Scene) {
        self.params.clear();
        self.strings.clear();
        self.sent.clear();
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
                    continue;
                }
                self.params
                    .insert(name.clone(), (pass.map(str::to_string), p.ty.clone()));
                if let ParamValue::Str(s) = &p.value {
                    self.strings.insert(name.clone(), s.clone());
                }
            }
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
            self.data.insert(name, values);
        }
        for (name, values) in params {
            newest(&mut self.incoming, name, values);
        }
        self.clean = self.clean.saturating_add(clean);
        if self.dead.is_none() && (done || !fault.is_empty()) {
            let why = match fault.is_empty() {
                true => "the scene process exited".to_string(),
                false => fault,
            };
            self.dead = Some(self.epitaph(&why));
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

    /// One JSON line for this tick, carrying the string params that changed.
    fn offer(&mut self, frame: u64, time: f64, dt: f64) {
        let mut params = Map::new();
        for (name, value) in &self.strings {
            if self.sent.get(name).is_some_and(|s| s == value) {
                continue;
            }
            params.insert(name.clone(), Value::from(value.clone()));
        }
        self.sent.clone_from(&self.strings);
        let (slot, wake) = &*self.outbox;
        let mut out = lock(slot);
        if out.closed {
            return;
        }
        if let Some(old) = out.pending.take() {
            for (name, value) in old.params {
                params.entry(name).or_insert(value);
            }
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
        if !self.reaped(TERM_AFTER) {
            unsafe { kill(self.child.id() as c_int, SIGTERM) };
            if !self.reaped(KILL_AFTER) {
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
        self.offer(frame, t, dt);
        Ok(())
    }

    /// Data reaches the engine texture once.
    /// A name no later message sets keeps what the texture holds.
    /// A process that sends a mask once pays for one upload.
    fn provide(&mut self, name: &str, _size: ArraySize, _frame: u64) -> Option<Feed> {
        Some(match self.data.remove(name) {
            Some(ArrayData::F32(values)) => Feed::Data(values),
            Some(ArrayData::U8(values)) => Feed::DataU8(values),
            None => Feed::Unchanged,
        })
    }

    fn update(&mut self, renderer: &mut dyn RenderTarget, _frame: u64, _t: f64) -> Result<()> {
        for (name, values) in std::mem::take(&mut self.incoming) {
            let Some((pass, ty)) = self.params.get(&name) else {
                continue; // a param this scene does not hold
            };
            let Some(value) = as_param(ty, &values) else {
                continue;
            };
            // a dotted name carries its pass, and the uniform is the tail
            let param = name.rsplit_once('.').map_or(name.as_str(), |(_, p)| p);
            renderer.set_param(pass.as_deref(), param, ty, &value)?;
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

    fn on_param(&mut self, name: &str, value: &str) {
        if self.strings.contains_key(name) {
            self.strings.insert(name.to_string(), value.to_string());
        }
    }
}

/// The floats of a param message, as the param's own type.
/// None means the message is short, or the param is host-side.
fn as_param(ty: &str, values: &[f32]) -> Option<ParamValue> {
    let at = |i: usize| values.get(i).map(|v| f64::from(*v));
    match ty {
        "float" => Some(ParamValue::Float(at(0)?)),
        "int" => Some(ParamValue::Int(at(0)?.round() as i64)),
        "bool" => Some(ParamValue::Bool(at(0)? != 0.0)),
        "color" => Some(ParamValue::Color([at(0)?, at(1)?, at(2)?])),
        "vec2" | "vec3" | "vec4" => {
            let n = ty[3..].parse::<usize>().ok()?;
            (values.len() >= n)
                .then(|| ParamValue::Vec(values[..n].iter().map(|v| f64::from(*v)).collect()))
        }
        _ => None,
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
                return;
            }
        }
        if let Err(why) = messages(&mut out, &inbox) {
            lock(&inbox).fault = why;
        }
        lock(&inbox).done = true;
    })
}

/// kind: u8, name_len: u8, name: UTF-8, count: u32 LE, values: count x f32 LE.
/// Kind 4 carries count u8 bytes instead, and the value is byte / 255.
/// Kind 2 (clean) is the single byte.
fn messages(out: &mut impl BufRead, inbox: &Mutex<Inbox>) -> Result<(), String> {
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
        if !matches!(kind[0], SET_ARRAY | SET_PARAM | SET_ARRAY_U8) {
            return Err(format!(
                "wrote message kind {} (0, 1, 2 or 4 only)",
                kind[0]
            ));
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
        let width = if kind[0] == SET_ARRAY_U8 { 1 } else { 4 };
        let mut body = vec![0u8; count * width];
        if out.read_exact(&mut body).is_err() {
            return Ok(());
        }
        let name = String::from_utf8_lossy(&name).into_owned();
        if kind[0] == SET_ARRAY_U8 {
            newest(&mut lock(inbox).arrays, name, ArrayData::U8(body));
            continue;
        }
        let values: Vec<f32> = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut held = lock(inbox);
        match kind[0] {
            SET_ARRAY => newest(&mut held.arrays, name, ArrayData::F32(values)),
            _ => newest(&mut held.params, name, values),
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
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A process that answers the probe, then hands over a scene.
    /// It feeds one array and one param for each tick.
    /// It appends every tick it reads to ticks.jsonl in its own directory.
    /// The test reads that file, so the process must run in that directory.
    const SCENE_SCRIPT: &str = r#"#!/usr/bin/env python3
import json, struct, sys

ARGS = {"args": [{"name": "emit", "type": "size", "default": [4, 2],
                  "help": "the array shape"},
                 {"name": "loud", "type": "bool", "default": False, "help": ""}]}

SCENE = {"version": 1, "params": {"text": {"type": "string", "value": "GHEE"}},
         "passes": [{"name": "main", "code": "void main(){ gl_FragColor = vec4(level_at(0)); }",
                     "arrays": [{"name": "level", "size": 4}],
                     "params": {"amt": {"type": "float", "value": 0.0}}}]}

if "--ghee-args" in sys.argv:
    sys.stdout.write(json.dumps(ARGS) + "\n")
    sys.exit(0)

out = sys.stdout.buffer
out.write((json.dumps(SCENE) + "\n").encode())
out.flush()


def msg(kind, name, values):
    out.write(struct.pack("<BB", kind, len(name)) + name.encode())
    out.write(struct.pack("<I", len(values)))
    out.write(struct.pack("<%df" % len(values), *values))
    out.flush()


for line in sys.stdin:
    tick = json.loads(line)
    with open("ticks.jsonl", "a") as f:
        f.write(line)
    n = tick["frame"]
    msg(0, "level", [n, n + 1, n + 2, n + 3])
    msg(1, "amt", [n / 10.0])
    if n % 2:
        out.write(bytes([2]))
        out.flush()
with open("ticks.jsonl", "a") as f:
    f.write("eof\n")
"#;

    /// A process that ignores stdin. It writes its data, then waits for the stop.
    const FREE: &str = r#"#!/usr/bin/env python3
import json, struct, sys, time

SCENE = {"version": 1,
         "passes": [{"name": "main", "code": "void main(){ gl_FragColor = vec4(1.0); }",
                     "arrays": [{"name": "level", "size": 2}]}]}
out = sys.stdout.buffer
out.write((json.dumps(SCENE) + "\n").encode())
out.write(struct.pack("<BB", 0, 5) + b"level" + struct.pack("<I", 2))
out.write(struct.pack("<2f", 7.0, 8.0))
out.flush()
time.sleep(60)
"#;

    /// A process that sends one array as kind 4 bytes and one as kind 0 floats.
    const U8_SCRIPT: &str = r#"#!/usr/bin/env python3
import json, struct, sys, time

SCENE = {"version": 1,
         "passes": [{"name": "main", "code": "void main(){ gl_FragColor = vec4(1.0); }",
                     "arrays": [{"name": "mask", "size": [2, 2]},
                                {"name": "state", "size": 2}]}]}
out = sys.stdout.buffer
out.write((json.dumps(SCENE) + "\n").encode())
out.write(struct.pack("<BB", 4, 4) + b"mask" + struct.pack("<I", 4))
out.write(bytes([0, 64, 128, 255]))
out.write(struct.pack("<BB", 0, 5) + b"state" + struct.pack("<I", 2))
out.write(struct.pack("<2f", 0.25, 1.0))
out.flush()
time.sleep(60)
"#;

    fn tmpdir(tag: &str) -> String {
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        let d = std::env::temp_dir().join(format!("ghee-proc-{}-{tag}{n}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d.to_string_lossy().into_owned()
    }

    fn script(dir: &str, name: &str, body: &str) -> String {
        let path = format!("{dir}/{name}");
        fs::write(&path, body).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn no_args() -> Vec<String> {
        Vec::new()
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

    /// A renderer stand-in that records the params a process sets, and the
    /// arrays a provider hands it. `feeds` names what render_into asks for.
    #[derive(Default)]
    struct Stub {
        set: std::cell::RefCell<Vec<(Option<String>, String, ParamValue)>>,
        feeds: Vec<(String, ArraySize)>,
        bytes: std::cell::RefCell<Vec<(String, Vec<u8>)>>,
        floats: std::cell::RefCell<Vec<(String, Vec<f32>)>>,
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
            provider: Option<crate::engine::Provider<'_>>,
            _out: &mut [u8],
            _out_size: Option<(i32, i32)>,
        ) -> Result<()> {
            let Some(p) = provider else { return Ok(()) };
            for (name, size) in &self.feeds {
                match p(name, *size) {
                    Feed::DataU8(v) => self.bytes.borrow_mut().push((name.clone(), v)),
                    Feed::Data(v) => self.floats.borrow_mut().push((name.clone(), v)),
                    Feed::Unchanged => {}
                }
            }
            Ok(())
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
            Ok(true)
        }
    }

    /// Run ticks until the process answers, or give up.
    fn pump(f: &mut SceneProcess, mut done: impl FnMut(&mut SceneProcess) -> bool) -> bool {
        for frame in 0..200u64 {
            if f.tick(frame, frame as f64 / 60.0, 1.0 / 60.0).is_err() {
                return false;
            }
            if done(f) {
                return true;
            }
            thread::sleep(Duration::from_millis(10));
        }
        false
    }

    #[test]
    fn the_help_probe_reads_the_declared_arguments() {
        let d = tmpdir("probe");
        let path = script(&d, "feed", SCENE_SCRIPT);
        let decls = declarations(&path).unwrap();
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
        let after = declarations(&path).unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].name, "gain");
    }

    #[test]
    fn a_scene_that_does_not_answer_the_help_probe_is_pass_through() {
        let d = tmpdir("passthrough");
        let path = script(&d, "quiet", "#!/bin/sh\nsleep 30\n");
        let started = Instant::now();
        assert_eq!(declarations(&path), None);
        assert!(started.elapsed() >= PROBE_WAIT, "gave up before the wait");
        assert!(started.elapsed() < PROBE_WAIT + Duration::from_secs(3));
    }

    /// The start runs the scene itself. Nothing runs `--ghee-args` first.
    #[test]
    fn a_start_costs_no_probe() {
        let d = tmpdir("no-probe");
        let path = script(&d, "feed", SCENE_SCRIPT);
        let started = Instant::now();
        let (_scene, mut f) = SceneProcess::start(&path, &no_args()).unwrap();
        assert!(started.elapsed() < PROBE_WAIT, "{:?}", started.elapsed());
        f.stop();
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
            "printf %s \"$GHEE_WIRE\" > {mark}\n\
             printf '{{\"passes\":[{{\"name\":\"main\",\"code\":\"void main(){{}}\"}}]}}\\n'\n\
             sleep 5\n"
        );
        let (_scene, f) = SceneProcess::start("/bin/sh", &args(&["-c", &body])).unwrap();
        assert!(wait_until(SETTLE, || fs::read_to_string(&mark).is_ok()));
        assert_eq!(fs::read_to_string(&mark).unwrap(), "1");
        drop(f);
    }

    #[test]
    fn the_handshake_line_becomes_the_scene() {
        let d = tmpdir("handshake");
        let path = script(&d, "feed", SCENE_SCRIPT);
        let (scene, process) = SceneProcess::start(&path, &no_args()).unwrap();
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
        let e = failure(SceneProcess::start(&path, &no_args()));
        let shown = e.to_string();
        assert!(shown.contains("did not complete the handshake"), "{shown}");
        assert!(!shown.contains("boom in the process"), "{shown}");
        let full = with_stderr(&e);
        assert_eq!(full, format!("{shown}\nboom in the process"));
    }

    #[test]
    fn messages_set_arrays_params_and_the_clean_flag() {
        let d = tmpdir("messages");
        let path = script(&d, "feed", SCENE_SCRIPT);
        let (scene, mut f) = SceneProcess::start(&path, &no_args()).unwrap();
        f.bind(&scene);
        assert!(pump(&mut f, |f| f.data.contains_key("level")));
        let Some(Feed::Data(values)) = f.provide("level", ArraySize::Linear(4), 0) else {
            panic!("no data for the declared array");
        };
        assert_eq!(values.len(), 4);
        assert_eq!(values[1] - values[0], 1.0);
        // a name no message sets keeps what the texture holds
        // data the engine takes never uploads twice
        for name in ["other", "level"] {
            assert!(matches!(
                f.provide(name, ArraySize::Linear(4), 0),
                Some(Feed::Unchanged)
            ));
        }
        assert!(!f.incoming.is_empty(), "no param message");
        let mut stub = Stub::default();
        f.update(&mut stub, 0, 0.0).unwrap();
        assert!(f.incoming.is_empty());
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
        let (scene, mut f) = SceneProcess::start(&path, &no_args()).unwrap();
        f.bind(&scene);
        f.tick(0, 0.0, 1.0 / 60.0).unwrap();
        let first = ticks(&d, 1).remove(0);
        assert_eq!(first["frame"], Value::from(0));
        assert_eq!(first["dt"], Value::from(1.0 / 60.0));
        assert_eq!(first["time"], Value::from(0.0));
        // the first tick carries every string param
        assert_eq!(first["params"]["text"], Value::from("GHEE"));
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
    fn a_changed_string_param_reaches_the_next_tick() {
        let d = tmpdir("onparam");
        let path = script(&d, "feed", SCENE_SCRIPT);
        let (scene, mut f) = SceneProcess::start(&path, &no_args()).unwrap();
        f.bind(&scene);
        f.tick(0, 0.0, 0.016).unwrap();
        ticks(&d, 1);
        f.on_param("text", "WOW");
        f.on_param("nosuch", "x"); // not a param of this scene
        f.tick(1, 0.016, 0.016).unwrap();
        let second = ticks(&d, 2).remove(1);
        assert_eq!(second["params"], serde_json::json!({"text": "WOW"}));
    }

    #[test]
    fn a_scene_that_ignores_stdin_runs_free() {
        let d = tmpdir("free");
        let path = script(&d, "free", FREE);
        let (scene, mut f) = SceneProcess::start(&path, &no_args()).unwrap();
        f.bind(&scene);
        assert!(pump(&mut f, |f| f.data.contains_key("level")));
        let Some(Feed::Data(values)) = f.provide("level", ArraySize::Linear(2), 0) else {
            panic!("no data");
        };
        assert_eq!(values, vec![7.0, 8.0]);
    }

    #[test]
    fn a_kind_4_message_reaches_the_render_target_as_bytes() {
        let d = tmpdir("u8");
        let path = script(&d, "mask", U8_SCRIPT);
        let (scene, mut f) = SceneProcess::start(&path, &no_args()).unwrap();
        f.bind(&scene);
        assert!(pump(&mut f, |f| f.data.len() == 2));
        let mut stub = Stub {
            feeds: vec![
                ("mask".to_string(), ArraySize::Grid(2, 2)),
                ("state".to_string(), ArraySize::Linear(2)),
            ],
            ..Stub::default()
        };
        let mut frame = [0u8; 8 * 8 * 4];
        {
            let mut feed = |name: &str, size: ArraySize| f.provide(name, size, 0).unwrap();
            stub.render_into(Some(&mut feed), &mut frame, None).unwrap();
        }
        // the bytes reach the target as bytes, and the floats stay floats
        assert_eq!(
            stub.bytes.borrow().as_slice(),
            [("mask".to_string(), vec![0u8, 64, 128, 255])]
        );
        assert_eq!(
            stub.floats.borrow().as_slice(),
            [("state".to_string(), vec![0.25f32, 1.0])]
        );
        // the texture holds the mask now, so the next frame uploads nothing
        assert!(matches!(
            f.provide("mask", ArraySize::Grid(2, 2), 1),
            Some(Feed::Unchanged)
        ));
    }

    #[test]
    fn a_scene_that_exits_mid_run_is_an_error_that_holds_its_stderr() {
        let d = tmpdir("died");
        let path = script(
            &d,
            "shortlived",
            "#!/bin/sh\nprintf '{\"passes\":[{\"name\":\"main\",\"code\":\"void main(){}\"}]}\\n'\n\
             echo 'the process gave up' >&2\nexit 1\n",
        );
        let (_scene, mut f) = SceneProcess::start(&path, &no_args()).unwrap();
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
        let (_scene, f) = SceneProcess::start(&quick, &no_args()).unwrap();
        let started = Instant::now();
        drop(f);
        assert!(
            started.elapsed() < TERM_AFTER,
            "stdin never closed: {:?}",
            started.elapsed()
        );
        let stubborn = script(
            &d,
            "stubborn",
            "#!/bin/sh\ntrap '' TERM\nprintf '{\"passes\":[{\"name\":\"main\",\
             \"code\":\"void main(){}\"}]}\\n'\nsleep 10\n",
        );
        let (_scene, f) = SceneProcess::start(&stubborn, &no_args()).unwrap();
        let started = Instant::now();
        drop(f);
        let took = started.elapsed();
        assert!(took >= TERM_AFTER + KILL_AFTER, "killed early: {took:?}");
        assert!(
            took < TERM_AFTER + KILL_AFTER + Duration::from_secs(3),
            "{took:?}"
        );
    }

    // the arguments a set of scenes takes

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
            args: Some(vec![
                decl("emit", "size", serde_json::json!([480, 300]), "the wall"),
                decl("loud", "bool", Value::from(false), "shout"),
            ]),
        }
    }

    fn plain(name: &str) -> Declared {
        Declared {
            name: name.to_string(),
            args: Some(Vec::new()),
        }
    }

    fn raw() -> Declared {
        Declared {
            name: "raw".to_string(),
            args: None,
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
        let quiet = script(&d, "quiet", "#!/bin/sh\nsleep 30\n");
        let json = format!("{d}/flat.scene.json");
        fs::write(&json, "{}").unwrap();
        let set = vec![
            ("feed".to_string(), path),
            ("quiet".to_string(), quiet),
            ("flat".to_string(), json),
        ];
        let declared = declare(&set);
        assert_eq!(declared[0].args.as_ref().unwrap().len(), 2);
        let text = help(&declared, "ghee serve --");
        assert!(text.contains("--emit WxH"), "{text}");
        // a scene that answers nothing in the wait, and a scene with no process
        assert!(text.contains("quiet: pass-through"), "{text}");
        assert!(text.contains("flat: no arguments"), "{text}");
    }

    /// A token ghee cannot know reaches the scene. The scene refuses it, and
    /// its own words reach the caller through the handshake failure.
    #[test]
    fn an_unknown_token_reaches_the_scene() {
        let d = tmpdir("tokens");
        let path = script(
            &d,
            "picky",
            "#!/bin/sh\ntest \"$1\" = --emit || { echo \"picky: unrecognized: $1\" >&2; exit 2; }\n\
             printf '{\"passes\":[{\"name\":\"main\",\"code\":\"void main(){}\"}]}\\n'\nsleep 30\n",
        );
        let e = failure(SceneProcess::start(&path, &args(&["--bogus"])));
        assert!(
            e.to_string().contains("did not complete the handshake"),
            "{e}"
        );
        assert!(
            with_stderr(&e).contains("picky: unrecognized: --bogus"),
            "{e}"
        );
        // the token the scene knows passes through the same way
        SceneProcess::start(&path, &args(&["--emit"])).unwrap();
    }

    #[test]
    fn a_message_kind_ghee_cannot_read_ends_the_scene_process() {
        let d = tmpdir("badkind");
        let path = script(
            &d,
            "garbage",
            "#!/bin/sh\nprintf '{\"passes\":[{\"name\":\"main\",\"code\":\"void main(){}\"}]}\\n'\n\
             printf '\\007'\nsleep 5\n",
        );
        let (_scene, mut f) = SceneProcess::start(&path, &no_args()).unwrap();
        let mut last = Ok(());
        for frame in 0..100u64 {
            last = f.tick(frame, 0.0, 0.016);
            if last.is_err() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let e = last.unwrap_err().to_string();
        assert!(
            e.contains("wrote message kind 7 (0, 1, 2 or 4 only)"),
            "{e}"
        );
    }

    const DOTTED_SCRIPT: &str = r#"#!/usr/bin/env python3
import json, struct, sys

SCENE = {"version": 1, "params": {},
         "passes": [{"name": "draw", "code": "void main(){}",
                     "params": {"color": {"type": "float", "value": 0.2}}},
                    {"name": "dither", "code": "void main(){}",
                     "params": {"color": {"type": "float", "value": 0.0}}}]}

out = sys.stdout.buffer
out.write((json.dumps(SCENE) + "\n").encode())

def msg(name, v):
    n = name.encode()
    out.write(bytes([1, len(n)]) + n + struct.pack("<I", 1) + struct.pack("<f", v))

msg("dither.color", 0.65)
msg("color", 0.1)
out.flush()
sys.stdin.readline()
"#;

    #[test]
    fn a_dotted_name_addresses_the_param_of_one_pass() {
        let d = tmpdir("dotted");
        let path = script(&d, "feed", DOTTED_SCRIPT);
        let (scene, mut f) = SceneProcess::start(&path, &no_args()).unwrap();
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
}
