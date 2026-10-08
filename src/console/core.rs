//! The console API, with no transport under it.
//!
//! A host implements ConsoleHost over what it renders, and hands that to a
//! ConsoleAPI. The transport turns each of its own requests into (method, path,
//! query, headers, body). It turns the Response back into what it serves:
//!
//! ```text
//! let api = ConsoleAPI::new(MyHost::new()).with_web(true);
//! let refused = api.precheck(method, path, headers);  // before reading a body
//! let body = api.read_body(reader)?;                   // a 413 past the cap
//! let resp = api.dispatch(method, path, query, headers, body);
//! ```
//!
//! query maps a name to its first value, and parse_query() builds one from a raw
//! query string, or the 400 for a malformed percent escape. A headers lookup
//! ignores case.
//!
//! The transport keeps the streams (/stream, /h264) and its own routes. Route
//! them before dispatch reaches the static-file fallback.
//!
//! Every refusal lives here, so a new host inherits it. The refusals are the
//! Origin check on a mutating request, the body cap, POST-only state changes,
//! the JSON content type, an edit of a scene other than the active one and the
//! static root check.
//!
//! A host that keeps each scene's working copy and its defaults says so
//! through scene_doc(), edits(), save(), reset() and answer(). The defaults of
//! those suit a host whose scene is written in code: scene_doc() serves
//! scene_def() and refuses the defaults, edits() answers no edit, and the
//! others refuse.
//!
//! An API holds no mutable state, so any number of threads may call it at once.
//! A host states its own thread contract for set_param, switch and apply_scene.

use std::collections::BTreeMap;
use std::fmt;
use std::io::Read;
use std::sync::Mutex;

use serde_json::{Map, Value};

use crate::console::{static_asset, WEB};
use crate::filters::{filter_defs, rebuild_filters};
use crate::layer::{Address, Form, Layer, Live, Rgb};
use crate::scene::{no_layer, no_param, not_an_image, py_repr, Scene};
use crate::usage;

/// The console's cap on a request body. A POSTed scene is JSON text.
pub const MAX_BODY: usize = 1 << 20;

/// The paths that change the renderer. They take POST only. A hostile page
/// reaches a GET through <img src>, and that request carries no Origin to
/// refuse.
pub const POST_ONLY: [&str; 7] = [
    "/set", "/switch", "/next", "/prev", "/save", "/reset", "/aside",
];

const VEC_ARITY: [(&str, usize); 3] = [("vec2", 2), ("vec3", 3), ("vec4", 4)];
const BOOLS: [(&str, bool); 4] = [("true", true), ("1", true), ("false", false), ("0", false)];

/// A query string, as a name to its first value.
pub type Query = BTreeMap<String, String>;

/// The request headers, as the transport hands them over. A lookup ignores
/// case.
#[derive(Clone, Debug, Default)]
pub struct Headers(Vec<(String, String)>);

impl Headers {
    pub fn new() -> Headers {
        Headers(Vec::new())
    }

    pub fn insert(&mut self, name: &str, value: &str) {
        self.0.push((name.to_string(), value.to_string()));
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

impl<const N: usize> From<[(&str, &str); N]> for Headers {
    fn from(pairs: [(&str, &str); N]) -> Headers {
        let mut h = Headers::new();
        for (k, v) in pairs {
            h.insert(k, v);
        }
        h
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Response {
    pub status: u16,
    pub body: Vec<u8>,
    pub content_type: String,
    /// Headers beside the content type, such as a scene's ETag.
    pub headers: Vec<(String, String)>,
}

impl Response {
    pub fn text(status: u16, msg: &str) -> Response {
        Response {
            status,
            body: msg.as_bytes().to_vec(),
            content_type: "text/plain".to_string(),
            headers: Vec::new(),
        }
    }

    pub fn json(obj: &Value) -> Response {
        Response {
            status: 200,
            body: obj.to_string().into_bytes(),
            content_type: "application/json".to_string(),
            headers: Vec::new(),
        }
    }
}

/// A refusal from a host or from the API. The client gets this status, with the
/// message as the body.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsoleError {
    pub status: u16,
    pub message: String,
}

impl ConsoleError {
    pub fn bad_request(msg: impl Into<String>) -> ConsoleError {
        ConsoleError {
            status: 400,
            message: msg.into(),
        }
    }

    pub fn read_only(msg: impl Into<String>) -> ConsoleError {
        ConsoleError {
            status: 403,
            message: msg.into(),
        }
    }

    pub fn not_found(msg: impl Into<String>) -> ConsoleError {
        ConsoleError {
            status: 404,
            message: msg.into(),
        }
    }

    pub fn conflict(msg: impl Into<String>) -> ConsoleError {
        ConsoleError {
            status: 409,
            message: msg.into(),
        }
    }

    pub fn stale(msg: impl Into<String>) -> ConsoleError {
        ConsoleError {
            status: 412,
            message: msg.into(),
        }
    }

    pub fn unsupported_media(msg: impl Into<String>) -> ConsoleError {
        ConsoleError {
            status: 415,
            message: msg.into(),
        }
    }

    pub fn unavailable(msg: impl Into<String>) -> ConsoleError {
        ConsoleError {
            status: 503,
            message: msg.into(),
        }
    }

    pub fn server_error(msg: impl Into<String>) -> ConsoleError {
        ConsoleError {
            status: 500,
            message: msg.into(),
        }
    }
}

impl fmt::Display for ConsoleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ConsoleError {}

pub type ConsoleResult<T> = Result<T, ConsoleError>;

/// A live field of a layer that a /set names, with its value as the scene
/// format writes the field.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerSet {
    pub layer: String,
    pub effect: Option<String>,
    pub field: String,
    pub value: Value,
}

impl LayerSet {
    pub fn at(&self) -> Address<'_> {
        Address {
            layer: &self.layer,
            effect: self.effect.as_deref(),
            field: &self.field,
        }
    }
}

/// What a /set changes: a param, or a live field of a layer.
enum Set {
    Param(Scope, String, Value),
    Layer(LayerSet),
}

/// Where a /set lands: the scene's shared params, or the params of the pass
/// at an index of the scene that renders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Scene,
    Pass(i64),
}

/// A state-changing request must carry this server's own Origin. A request with
/// no Origin passes, because curl, the CLI and a script send none.
pub fn origin_allowed(origin: &str, host: Option<&str>) -> bool {
    if origin.is_empty() {
        return true;
    }
    netloc(origin).to_lowercase() == host.unwrap_or("").to_lowercase()
}

/// The host part of a URL: what follows "//", up to the path, query or
/// fragment.
fn netloc(url: &str) -> &str {
    let rest = match url.find(':') {
        Some(i) if is_scheme(&url[..i]) => &url[i + 1..],
        _ => url,
    };
    let Some(rest) = rest.strip_prefix("//") else {
        return "";
    };
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    &rest[..end]
}

fn is_scheme(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c))
}

/// A raw query string, as a name to its first value. The console never repeats a
/// key.
///
/// A field with a blank value is dropped, so `?value=` carries no value. A `%`
/// without two hex digits after it answers 400.
pub fn parse_query(qs: &str) -> ConsoleResult<Query> {
    let mut out = Query::new();
    for field in qs.split('&').filter(|f| !f.is_empty()) {
        let Some((name, value)) = field.split_once('=') else {
            continue;
        };
        if value.is_empty() {
            continue;
        }
        let name = unquote_plus(name)?;
        let value = unquote_plus(value)?;
        out.entry(name).or_insert(value);
    }
    Ok(out)
}

/// Decode a query field. "+" is a space and %XX is a byte.
fn unquote_plus(s: &str) -> ConsoleResult<String> {
    let bytes = s.as_bytes();
    let hex = |at: usize| bytes.get(at).and_then(|b| char::from(*b).to_digit(16));
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' => {
                let (Some(hi), Some(lo)) = (hex(i + 1), hex(i + 2)) else {
                    let escape: String = s[i..].chars().take(3).collect();
                    return Err(ConsoleError::bad_request(format!(
                        "bad query: {} is not a percent escape",
                        py_repr(&Value::from(escape))
                    )));
                };
                out.push((hi * 16 + lo) as u8);
                i += 3;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// Parse a float. Whitespace around the number is fine. Any other text fails.
fn py_float(raw: &str) -> Result<f64, String> {
    let s = raw.trim().to_ascii_lowercase();
    let text = if s.trim_start_matches(['+', '-']) == "infinity" {
        s.replace("infinity", "inf")
    } else {
        s
    };
    text.parse::<f64>()
        .map_err(|_| format!("{} is not a number", py_repr(&Value::from(raw))))
}

/// The value set_param expects, from a /set raw query string.
///
/// A "color" is "#rrggbb" or a CSS colour name, which stays as written, or
/// "r,g,b", which gives three floats. A "vecK" is K comma-separated floats. A "bool" is true or
/// false (1 or 0), and the scene model demands a real bool. Anything else is
/// one float. Malformed input errs.
pub fn parse_param_value(ptype: &str, raw: &str) -> Result<Value, String> {
    if ptype == "bool" {
        let key = raw.trim().to_lowercase();
        return BOOLS
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| Value::from(*v))
            .ok_or_else(|| {
                format!(
                    "bool needs true or false, got {}",
                    py_repr(&Value::from(raw))
                )
            });
    }
    if ptype == "color" {
        let text = raw.trim();
        if Rgb::from_text(text).is_some() {
            return Ok(Value::from(text));
        }
        let rgb = floats(raw).ok().filter(|p| p.len() == 3);
        return rgb.map(Value::from).ok_or_else(|| {
            format!(
                "color needs '#rrggbb', a CSS colour name or three comma-separated floats, got {}",
                py_repr(&Value::from(raw))
            )
        });
    }
    if let Some((_, n)) = VEC_ARITY.iter().find(|(t, _)| *t == ptype) {
        let parts = floats(raw)?;
        if parts.len() != *n {
            return Err(format!(
                "{ptype} needs {n} comma-separated floats, got {}",
                py_repr(&Value::from(raw))
            ));
        }
        return Ok(Value::from(parts));
    }
    if ptype == "string" {
        return Ok(Value::from(raw));
    }
    Ok(Value::from(py_float(raw)?))
}

fn floats(raw: &str) -> Result<Vec<f64>, String> {
    raw.split(',').map(py_float).collect()
}

/// The web asset named by a URL path, or None. The assets are compiled in, so a
/// path that climbs out of the root gets None.
pub fn static_path(urlpath: &str) -> Option<&'static str> {
    let urlpath = match urlpath {
        "" | "/" => "/index.html",
        p => p,
    };
    let mut parts: Vec<&str> = Vec::new();
    for seg in urlpath.trim_start_matches('/').split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            s => parts.push(s),
        }
    }
    let name = parts.join("/");
    WEB.iter().find(|(n, _)| *n == name).map(|(n, _)| *n)
}

/// What the console needs from whatever renders.
///
/// Implement the methods your host has. Mount a ConsoleAPI on your server. Any
/// method may return a ConsoleError, and the client gets its status and its
/// message.
pub trait ConsoleHost {
    /// Why save() refuses, for a host that keeps no state file.
    fn save_refused(&self) -> String {
        "this console saves no scene".to_string()
    }

    /// {fps, frame_ms, stream_fps, out, error, error_line, error_pass,
    /// text_errors, active, display, stale, layer_changes, rev}.
    ///
    /// fps counts completed render-loop frames per elapsed second, over a
    /// trailing window (stats::FrameRate). frame_ms is one loop iteration, with
    /// no paced sleep in it. stream_fps counts the frames handed to viewers per
    /// elapsed second, every codec in one number. out names where the frames go
    /// now (stats::DEST_*). error_pass is the pass index that holds a compile
    /// error, or "common" for the scene's common block, and error_line the line
    /// inside the pass or the block. text_errors holds the problems of the
    /// scene's text layers, a line each. stale is true when the files behind
    /// the active scene changed since the compile. A host whose scenes have no
    /// files answers false for stale. layer_changes counts the changes that
    /// the program behind the scene made to its layers in scene_def(), so a
    /// page reads the scene again when it moves. rev is the active scene's
    /// revision, in a host that keeps one.
    ///
    /// A poll loop in the browser reads this, so it must not block.
    fn stats(&self) -> ConsoleResult<Value>;

    fn scene_names(&self) -> ConsoleResult<Vec<String>>;

    /// The names among scene_names() whose scene a program writes. The
    /// program keeps which layers exist and their order, so the page adds,
    /// moves and removes none of their layers. /scenes lists them under
    /// "procedural".
    fn procedural_names(&self) -> ConsoleResult<Vec<String>> {
        Ok(Vec::new())
    }

    fn active_name(&self) -> ConsoleResult<Option<String>>;

    /// The resolved scene JSON for a name, or for the active scene when name is
    /// None. Resolved means each GLSL file as its text and filters expanded.
    fn scene_def(&self, name: Option<&str>) -> ConsoleResult<Value>;

    /// GET /scene: scene_def(), or the scene's defaults with `defaults`,
    /// with the scene's revision when the host keeps one.
    fn scene_doc(&self, name: Option<&str>, defaults: bool) -> ConsoleResult<(Value, Option<u64>)> {
        if defaults {
            return Err(ConsoleError::not_found(
                "this console's scene has no defaults",
            ));
        }
        Ok((self.scene_def(name)?, None))
    }

    /// GET /edits: `{rev, values, items, unsaved, aside}`, the working copy's
    /// edits as save() writes them, whether they differ from the state file,
    /// and the items set aside.
    fn edits(&self, _name: Option<&str>) -> ConsoleResult<Value> {
        Ok(
            serde_json::json!({"rev": null, "values": {}, "items": [], "unsaved": false,
                              "aside": []}),
        )
    }

    /// Write the working copy's edits to the scene's state file.
    fn save(&self, _name: Option<&str>) -> ConsoleResult<()> {
        Err(ConsoleError::read_only(self.save_refused()))
    }

    /// Take the working copy back to the defaults.
    fn reset(&self, _name: Option<&str>) -> ConsoleResult<()> {
        Err(ConsoleError::read_only(
            "this console's scene has no defaults to go back to",
        ))
    }

    /// Answer an item set aside: keep it over the new default, or drop it.
    fn answer(&self, _name: Option<&str>, _item: &Aside, _keep: bool) -> ConsoleResult<()> {
        Err(ConsoleError::not_found("no such item is set aside"))
    }

    /// Each filter name and its definition, for the "+ filter" picker.
    fn filters(&self) -> ConsoleResult<Value> {
        Ok(Value::Object(filter_defs()))
    }

    /// The board, its GPU and its memory, as crate::console::feeds reads
    /// them.
    fn hwinfo(&self) -> ConsoleResult<Value> {
        Ok(crate::console::feeds::hwinfo())
    }

    /// Set a live uniform, with no recompile, to a value that param_of()
    /// typed against live_doc().
    fn set_param(&self, scope: Scope, name: &str, value: Value) -> ConsoleResult<()>;

    /// Set a live field of a layer, with no rebuild, to a value that
    /// live_value() checked against live_doc().
    fn set_layer(&self, set: LayerSet) -> ConsoleResult<()>;

    /// The scene document that a /set checks its param or its layer's field
    /// and its value against: the scene that renders.
    fn live_doc(&self) -> ConsoleResult<Value> {
        self.scene_def(None)
    }

    /// Make name active and return the new active name.
    fn switch(&self, name: Option<&str>) -> ConsoleResult<String>;

    /// switch() for a request that `by` describes. A switch that names a
    /// scene in `by` is a page's reload of that scene, and refuses with a 409
    /// while the console is on another scene.
    fn switch_by(&self, name: Option<&str>, by: &Edit) -> ConsoleResult<String> {
        if let Some(scene) = by.scene {
            other_reload(scene, self.active_name()?.as_deref())?;
        }
        self.switch(name)
    }

    /// Move to the next or previous of scene_names().
    fn step(&self, direction: &str) -> ConsoleResult<String> {
        let names = self.scene_names()?;
        if names.is_empty() {
            return Err(ConsoleError::not_found("no scenes"));
        }
        let active = self.active_name()?;
        let i = active
            .as_deref()
            .and_then(|a| names.iter().position(|n| n == a))
            .unwrap_or(0);
        let step = if direction == "next" {
            1
        } else {
            names.len() - 1
        };
        let name = names[(i + step) % names.len()].clone();
        self.switch(Some(&name))
    }

    /// Recompile the running renderer onto a validated Scene.
    fn apply_scene(&self, scene: Scene) -> ConsoleResult<()>;

    /// Run `edit`, a change to the scene that `at` names or to its values,
    /// while that scene is the active scene, else refuse it as other_scene()
    /// does. No scene edits the active one. A host whose switch() queues the
    /// switch holds one lock over its switch and over this check and the
    /// edit, so an edit queued behind a switch never lands on the scene that
    /// the switch brought. A host that keeps revisions refuses an edit whose
    /// revision another client's change made stale.
    fn edit_scene(
        &self,
        at: &Edit,
        edit: &mut dyn FnMut() -> ConsoleResult<()>,
    ) -> ConsoleResult<()> {
        if let Some(scene) = at.scene {
            other_scene(scene, self.active_name()?.as_deref())?;
        }
        edit()
    }
}

/// What a request that changes a scene says of itself: the scene that it is
/// for, the revision of that scene that its page last read, and the page's
/// id, the one that its streams use.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Edit<'a> {
    pub scene: Option<&'a str>,
    pub rev: Option<u64>,
    pub vid: &'a str,
}

/// An item set aside, as POST /aside names it.
#[derive(Clone, Debug, PartialEq)]
pub struct Aside {
    pub item: String,
    pub pass: Option<String>,
    pub param: Option<String>,
}

/// Routes one transport-free request onto a ConsoleHost.
pub struct ConsoleAPI<H: ConsoleHost> {
    pub host: H,
    /// Serve the compiled-in web assets. false leaves that to the transport.
    pub web: bool,
    pub post_only: Vec<String>,
    pub max_body: usize,
    /// The GPU's engine times at the last /stats that asked for usage.
    engines: Mutex<Option<usage::Engines>>,
}

/// A figure to a tenth.
fn tenth(v: f64) -> Value {
    Value::from((v * 10.0).round() / 10.0)
}

/// A count of bytes in MiB to a tenth, null when unknown.
fn mib(bytes: Option<u64>) -> Value {
    bytes.map_or(Value::Null, |b| tenth(usage::mib(b)))
}

impl<H: ConsoleHost> ConsoleAPI<H> {
    pub fn new(host: H) -> ConsoleAPI<H> {
        ConsoleAPI {
            host,
            web: false,
            post_only: POST_ONLY.iter().map(|p| (*p).to_string()).collect(),
            max_body: MAX_BODY,
            engines: Mutex::new(None),
        }
    }

    pub fn with_web(mut self, web: bool) -> ConsoleAPI<H> {
        self.web = web;
        self
    }

    /// Add to a host's /stats what the process uses, every host alike:
    /// `rss_mib`, `gpu_mib`, `gpu_busy`, `machine_free_mib` and
    /// `soc_celsius` (crate::usage). The files behind them are read here
    /// alone, so nothing reads them while no request asks. `gpu_busy` covers
    /// the time since the last request that asked, and holds null for each
    /// queue in the first.
    fn add_usage(&self, stats: &mut Value) {
        let Value::Object(d) = stats else {
            return;
        };
        d.insert("rss_mib".to_string(), mib(usage::rss()));
        d.insert("gpu_mib".to_string(), mib(usage::gpu()));
        let now = usage::Engines::read();
        let mut last = self.engines.lock().unwrap_or_else(|e| e.into_inner());
        let busy = match (&now, last.as_ref()) {
            (None, _) => Value::Null,
            (Some(now), Some(before)) => Value::Object(
                now.busy_since(before)
                    .into_iter()
                    .map(|(q, share)| (q, tenth(share)))
                    .collect(),
            ),
            (Some(now), None) => {
                Value::Object(now.ran().map(|q| (q.to_string(), Value::Null)).collect())
            }
        };
        *last = now;
        d.insert("gpu_busy".to_string(), busy);
        d.insert("machine_free_mib".to_string(), mib(usage::machine_free()));
        d.insert(
            "soc_celsius".to_string(),
            usage::soc_celsius().map_or(Value::Null, tenth),
        );
    }

    #[allow(dead_code)]
    pub fn with_post_only(mut self, paths: &[&str]) -> ConsoleAPI<H> {
        self.post_only = paths.iter().map(|p| (*p).to_string()).collect();
        self
    }

    #[allow(dead_code)]
    pub fn with_max_body(mut self, max_body: usize) -> ConsoleAPI<H> {
        self.max_body = max_body;
        self
    }

    /// The refusals that must land before the transport reads a body. Returns a
    /// Response to send instead of handling the request, or None. dispatch runs
    /// this again, so a transport that skips it still gets the refusal.
    pub fn precheck(&self, method: &str, path: &str, headers: &Headers) -> Option<Response> {
        if method == "POST" {
            if !origin_allowed(headers.get("Origin").unwrap_or(""), headers.get("Host")) {
                return Some(Response::text(403, "cross-origin request"));
            }
            let raw = match headers.get("Content-Length").unwrap_or("0") {
                "" => "0",
                v => v,
            };
            if !raw.bytes().all(|b| b.is_ascii_digit()) {
                return Some(Response::text(
                    400,
                    &format!("bad Content-Length {}", py_repr(&Value::from(raw))),
                ));
            }
            let n = raw.parse::<u128>().unwrap_or(u128::MAX);
            if n > self.max_body as u128 {
                return Some(self.too_large());
            }
        }
        if method == "GET" && self.post_only.iter().any(|p| p == path) {
            return Some(Response::text(405, &format!("{path} is POST only")));
        }
        None
    }

    /// The body of a POST that precheck() let through, read from `r` up to
    /// the body cap. A longer body answers the 413 of precheck(), which a
    /// chunked body, sent with no Content-Length, meets only here, and a
    /// read that fails answers 400 with the reason.
    pub fn read_body(&self, r: &mut dyn Read) -> Result<Vec<u8>, Response> {
        let mut body = Vec::new();
        if let Err(e) = r.take(self.max_body as u64 + 1).read_to_end(&mut body) {
            return Err(Response::text(400, &format!("bad body: {e}")));
        }
        if body.len() > self.max_body {
            return Err(self.too_large());
        }
        Ok(body)
    }

    fn too_large(&self) -> Response {
        Response::text(
            413,
            &format!("body over the limit of {} bytes", self.max_body),
        )
    }

    pub fn dispatch(
        &self,
        method: &str,
        path: &str,
        query: &Query,
        headers: &Headers,
        body: &[u8],
    ) -> Response {
        if let Some(refused) = self.precheck(method, path, headers) {
            return refused;
        }
        let out = match method {
            "GET" => self.get(path, query),
            "POST" => self.post(path, query, headers, body),
            _ => return Response::text(405, &format!("{method} is not allowed")),
        };
        out.unwrap_or_else(|e| Response::text(e.status, &e.message))
    }

    fn get(&self, path: &str, query: &Query) -> ConsoleResult<Response> {
        let h = &self.host;
        match path {
            "/stats" => {
                let mut stats = h.stats()?;
                if query.get("usage").map(String::as_str) == Some("1") {
                    self.add_usage(&mut stats);
                }
                Ok(Response::json(&stats))
            }
            "/scenes" => {
                let mut d = Map::new();
                d.insert("scenes".to_string(), Value::from(h.scene_names()?));
                d.insert(
                    "active".to_string(),
                    h.active_name()?.map_or(Value::Null, Value::from),
                );
                let procedural = h.procedural_names()?;
                d.insert("procedural".to_string(), Value::from(procedural));
                Ok(Response::json(&Value::Object(d)))
            }
            "/hwinfo" => Ok(Response::json(&h.hwinfo()?)),
            "/filters" => Ok(Response::json(&h.filters()?)),
            "/scene" => {
                let defaults = query.get("defaults").map(String::as_str) == Some("1");
                let (doc, rev) = h.scene_doc(query.get("name").map(String::as_str), defaults)?;
                let mut r = Response::json(&doc);
                if let Some(rev) = rev {
                    r.headers.push(("ETag".to_string(), format!("\"{rev}\"")));
                }
                Ok(r)
            }
            "/edits" => Ok(Response::json(
                &h.edits(query.get("name").map(String::as_str))?,
            )),
            "/format" => Ok(Response::json(&crate::layer::format())),
            _ => Ok(self.static_file(path)),
        }
    }

    fn post(
        &self,
        path: &str,
        query: &Query,
        headers: &Headers,
        body: &[u8],
    ) -> ConsoleResult<Response> {
        let h = &self.host;
        let name = query.get("name").map(String::as_str);
        let rev = match query.get("rev") {
            None => None,
            Some(r) => Some(r.parse::<u64>().map_err(|_| {
                ConsoleError::bad_request(format!(
                    "bad {path}: rev needs a revision number, got {}",
                    py_repr(&Value::from(r.as_str()))
                ))
            })?),
        };
        let vid = query.get("vid").map_or("", String::as_str);
        let at = |scene| Edit { scene, rev, vid };
        if path == "/set" {
            let mut set = || {
                let set = parse_set(h, query)?
                    .map_err(|e| ConsoleError::bad_request(format!("bad /set: {e}")))?;
                match set {
                    Set::Param(scope, name, value) => h.set_param(scope, &name, value),
                    Set::Layer(set) => h.set_layer(set),
                }
            };
            h.edit_scene(&at(query.get("scene").map(String::as_str)), &mut set)?;
            return Ok(Response::text(200, "ok"));
        }
        if path == "/next" || path == "/prev" {
            return Ok(Response::json(&active_json(h.step(&path[1..])?)));
        }
        if path == "/switch" {
            let reload = at(rev.and(name));
            let active = h.switch_by(name, &reload)?;
            return Ok(Response::json(&active_json(active)));
        }
        if path == "/scene" {
            let mut doc = Value::Null;
            let mut apply = || {
                let mut scene = scene_of(&json_body(headers, body, "/scene")?, "/scene")?;
                rebuild_filters(&mut scene, None)
                    .map_err(|e| ConsoleError::bad_request(format!("bad /scene: {e}")))?;
                // The scene as it will run, so the console shows each filter
                // pass as the filter built it.
                doc = scene.to_dict();
                h.apply_scene(scene)
            };
            h.edit_scene(&at(name), &mut apply)?;
            return Ok(Response::json(&doc));
        }
        if path == "/save" {
            h.edit_scene(&at(name), &mut || h.save(name))?;
            return Ok(Response::text(200, "ok"));
        }
        if path == "/reset" {
            h.edit_scene(&at(name), &mut || h.reset(name))?;
            return Ok(Response::text(200, "ok"));
        }
        if path == "/aside" {
            let (aside, keep) = parse_aside(query)
                .map_err(|e| ConsoleError::bad_request(format!("bad /aside: {e}")))?;
            h.edit_scene(&at(name), &mut || h.answer(name, &aside, keep))?;
            return Ok(Response::text(200, "ok"));
        }
        Ok(Response::text(404, "no"))
    }

    fn static_file(&self, path: &str) -> Response {
        if !self.web {
            return Response::text(404, "not found");
        }
        let Some(name) = static_path(path) else {
            return Response::text(404, "not found");
        };
        let (ctype, body) = static_asset(name);
        Response {
            status: 200,
            body: body.to_vec(),
            content_type: ctype.to_string(),
            headers: Vec::new(),
        }
    }
}

/// The item that a POST /aside answers, and whether its answer keeps it.
fn parse_aside(query: &Query) -> Result<(Aside, bool), String> {
    let item = query.get("item").ok_or("the query has no 'item'")?;
    let keep = match query.get("answer").map(String::as_str) {
        Some("keep") => true,
        Some("default") => false,
        Some(a) => {
            return Err(format!(
                "answer needs keep or default, got {}",
                py_repr(&Value::from(a))
            ))
        }
        None => return Err("the query has no 'answer'".to_string()),
    };
    let aside = Aside {
        item: item.clone(),
        pass: query.get("pass").cloned(),
        param: query.get("param").cloned(),
    };
    Ok((aside, keep))
}

/// What a /set changes, or the first thing wrong with it. A query that names
/// a `layer` sets one of its live fields, and its value parses by the field.
/// Any other query sets the param that param_of() finds, and its value
/// parses by the param's type. The outer error is the host's own.
fn parse_set(host: &impl ConsoleHost, query: &Query) -> ConsoleResult<Result<Set, String>> {
    if let Some(layer) = query.get("layer") {
        let parts = || -> Result<(&String, &String), String> {
            let raw = query.get("value").ok_or("the query has no 'value'")?;
            let field = query.get("field").ok_or("the query has no 'field'")?;
            Ok((field, raw))
        };
        let (field, raw) = match parts() {
            Ok(p) => p,
            Err(e) => return Ok(Err(e)),
        };
        let effect = query.get("effect").map(String::as_str);
        let at = Address {
            layer,
            effect,
            field,
        };
        let doc = host.live_doc()?;
        return Ok(live_value(&doc, &at, raw).map(|value| {
            Set::Layer(LayerSet {
                layer: layer.clone(),
                effect: effect.map(str::to_string),
                field: field.clone(),
                value,
            })
        }));
    }
    let parts = || -> Result<(&String, &String), String> {
        let raw = query.get("value").ok_or("the query has no 'value'")?;
        let name = query.get("name").ok_or("the query has no 'name'")?;
        Ok((name, raw))
    };
    let (name, raw) = match parts() {
        Ok(p) => p,
        Err(e) => return Ok(Err(e)),
    };
    let doc = host.live_doc()?;
    let pass = query.get("pass").map(String::as_str);
    Ok(param_of(&doc, pass, name).and_then(|(scope, ty)| {
        parse_param_value(&ty, raw).map(|value| Set::Param(scope, name.clone(), value))
    }))
}

/// The param of the scene document `doc` that `pass` and `name` address, as
/// Python's set_param addresses one: `pass` names the pass that holds it, and
/// with no pass, `name` is the scene param, else the param of the first pass
/// that declares it. Its place and its type, or the error that set_param
/// gives for the same names.
pub fn param_of(doc: &Value, pass: Option<&str>, name: &str) -> Result<(Scope, String), String> {
    let passes = doc_passes(doc);
    let typed = |params: Option<&Value>| -> Option<String> {
        Some(params?.get(name)?.get("type")?.as_str()?.to_string())
    };
    let at = |i: usize| Scope::Pass(i64::try_from(i).unwrap_or(i64::MAX));
    let found = match pass {
        Some(p) => passes
            .iter()
            .position(|d| d.get("name").and_then(Value::as_str) == Some(p))
            .and_then(|i| Some((at(i), typed(passes[i].get("params"))?))),
        None => typed(doc.get("params"))
            .map(|ty| (Scope::Scene, ty))
            .or_else(|| {
                passes
                    .iter()
                    .enumerate()
                    .find_map(|(i, d)| Some((at(i), typed(d.get("params"))?)))
            }),
    };
    found.ok_or_else(|| {
        let wire = match pass {
            Some(p) => format!("{p}.{name}"),
            None => name.to_string(),
        };
        no_param(&wire, param_wires(doc).iter().map(String::as_str))
    })
}

/// Every name that addresses a param of the scene document `doc`, in the
/// order that Python lists them: each scene param, then each param of each
/// pass as `pass.name`, and by its name alone where no name before it is the
/// same.
fn param_wires(doc: &Value) -> Vec<String> {
    let names = |params: Option<&Value>| -> Vec<String> {
        params
            .and_then(Value::as_object)
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default()
    };
    let mut out = names(doc.get("params"));
    for p in doc_passes(doc) {
        let pass = p.get("name").and_then(Value::as_str).unwrap_or_default();
        for n in names(p.get("params")) {
            out.push(format!("{pass}.{n}"));
            if !out.contains(&n) {
                out.push(n);
            }
        }
    }
    out
}

/// The passes of the scene document `doc`.
fn doc_passes(doc: &Value) -> &[Value] {
    doc.get("passes")
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

/// The value of `raw`, a /set value, for the live field `at` of the scene
/// document `doc`, checked as the scene checks a live value, or why the scene
/// takes none.
pub fn live_value(doc: &Value, at: &Address, raw: &str) -> Result<Value, String> {
    let named = Value::from(at.layer);
    let Some(d) = doc_layers(doc).find(|l| l.get("name") == Some(&named)) else {
        let names = doc_layers(doc).filter_map(|l| l.get("name")?.as_str());
        return Err(no_layer(at.layer, names));
    };
    let mut layer = Layer::from_dict("a layer", d).map_err(|e| e.to_string())?;
    let live = layer.live(at.effect, at.field).map_err(|e| e.to_string())?;
    let value = parse_live_value(&live, raw)?;
    if live.form == Form::Image && !crate::layer::is_image_file(raw) {
        let file = doc
            .get("assets")
            .and_then(|a| a.get(raw))
            .and_then(|a| a.get("file"))
            .and_then(Value::as_str);
        if file.is_none_or(crate::text::is_font_file) {
            return Err(not_an_image(&live.what, &value));
        }
    }
    layer
        .set_live(at.effect, at.field, &value)
        .map_err(|e| e.to_string())?;
    Ok(value)
}

/// `raw`, a /set value, as the live field `live` takes it: a number, true or
/// false, a colour as `#rrggbb`, a CSS colour name, three comma-separated
/// floats or a gradient object in JSON, the name of an image asset, a text
/// layer's words, a line's points as a JSON list, or `none` for the default
/// of a field that takes none.
fn parse_live_value(live: &Live, raw: &str) -> Result<Value, String> {
    let what = &live.what;
    let s = raw.trim();
    let got = py_repr(&Value::from(raw));
    let or_none = |none: bool| if none { " or none" } else { "" };
    match live.form {
        Form::Number { none: true } | Form::Colour { none: true, .. } if s == "none" => {
            Ok(Value::Null)
        }
        Form::Number { none } => py_float(raw)
            .map(Value::from)
            .map_err(|_| format!("{what} needs a number{}, got {got}", or_none(none))),
        Form::Flag => BOOLS
            .iter()
            .find(|(k, _)| *k == s.to_lowercase())
            .map(|(_, v)| Value::from(*v))
            .ok_or_else(|| format!("{what} needs true or false, got {got}")),
        Form::Colour { gradient, none } => {
            if Rgb::from_text(s).is_some() {
                return Ok(Value::from(s));
            }
            if gradient && s.starts_with('{') {
                if let Ok(v @ Value::Object(_)) = serde_json::from_str::<Value>(s) {
                    return Ok(v);
                }
            }
            if let Some(rgb) = floats(raw).ok().filter(|p| p.len() == 3) {
                return Ok(Value::from(rgb));
            }
            let forms = match (gradient, none) {
                (true, true) => {
                    "'#rrggbb', a CSS colour name, three comma-separated numbers, a gradient \
                     object or none"
                }
                (false, true) => {
                    "'#rrggbb', a CSS colour name, three comma-separated numbers or none"
                }
                _ => "'#rrggbb', a CSS colour name or three comma-separated numbers",
            };
            Err(format!("{what} needs {forms}, got {got}"))
        }
        Form::Image | Form::Words => Ok(Value::from(raw)),
        Form::Points => match serde_json::from_str::<Value>(s) {
            Ok(v @ Value::Array(_)) => Ok(v),
            _ => Err(format!(
                "{what} needs a JSON list of points, each [x, y], got {got}"
            )),
        },
    }
}

/// The layers of every pass of the scene document `doc`.
fn doc_layers(doc: &Value) -> impl Iterator<Item = &Value> {
    doc.get("passes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|p| p.get("layers")?.as_array())
        .flatten()
}

/// The 409 for an edit of scene `scene` while `active` is the active scene,
/// as a page left open on one scene meets after the console switched.
pub fn other_scene(scene: &str, active: Option<&str>) -> ConsoleResult<()> {
    match active {
        Some(a) if a == scene => Ok(()),
        Some(a) => Err(ConsoleError::conflict(format!(
            "the console is on scene '{a}', and this edit is for scene '{scene}'"
        ))),
        None => Err(ConsoleError::conflict(format!(
            "the console is on no scene, and this edit is for scene '{scene}'"
        ))),
    }
}

/// The 409 for a page's reload of scene `scene` while `active` is the active
/// scene: a page left open on one scene never switches the console back.
pub fn other_reload(scene: &str, active: Option<&str>) -> ConsoleResult<()> {
    match active {
        Some(a) if a != scene => Err(ConsoleError::conflict(format!(
            "the console is on scene '{a}', and this reload is for scene '{scene}'"
        ))),
        _ => Ok(()),
    }
}

fn active_json(active: String) -> Value {
    let mut d = Map::new();
    d.insert("active".to_string(), Value::from(active));
    Value::Object(d)
}

fn json_body(headers: &Headers, body: &[u8], what: &str) -> ConsoleResult<Value> {
    let ct = headers
        .get("Content-Type")
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_lowercase();
    if ct != "application/json" {
        return Err(ConsoleError::unsupported_media(format!(
            "expected application/json, got {}",
            py_repr(&Value::from(ct))
        )));
    }
    serde_json::from_slice(body).map_err(|e| ConsoleError::bad_request(format!("bad {what}: {e}")))
}

/// Validate a posted scene into a Scene. The renderer never gets an
/// unvalidated document. The scene's errors start with `what`, the request.
fn scene_of(d: &Value, what: &str) -> ConsoleResult<Scene> {
    Scene::from_dict(d, None, Some(what)).map_err(|e| ConsoleError::bad_request(format!("bad {e}")))
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    const SCENE: &str = r#"{"version": 1,
        "params": {"tint": {"type": "float", "value": 0.0},
                   "title": {"type": "string", "value": "hi"}},
        "assets": {"pic": {"file": "pic.png"}, "sans": {"file": "sans.ttf"}},
        "passes": [{"name": "main", "fragment": "void main(){ gl_FragColor = vec4(1.0); }",
                    "layers": [{"name": "words", "text": "ho", "font_family": "sans.ttf",
                                "karaoke": {}},
                               {"name": "icon", "image": "icon.png"},
                               {"name": "frame", "shape": "rect"},
                               {"name": "scope", "shape": "line", "points": [[0, 0], [1, 1]]}],
                    "params": {"k": {"type": "float", "value": 0.0},
                               "on": {"type": "bool", "value": false},
                               "n": {"type": "int", "value": 0},
                               "c": {"type": "color", "value": [0.0, 0.0, 0.0]},
                               "v": {"type": "vec2", "value": [0.0, 0.0]}}}]}"#;

    fn scene_json() -> Value {
        serde_json::from_str(SCENE).unwrap()
    }

    fn json_ct() -> Headers {
        Headers::from([("Content-Type", "application/json")])
    }

    /// One recorded call on the fake host.
    #[derive(Debug, PartialEq)]
    enum Call {
        SceneDef(Option<String>),
        SetParam(Scope, String, Value),
        SetLayer(LayerSet),
        Switch(String),
        ApplyScene(Scene),
        Save(Option<String>),
        Reset(Option<String>),
        Answer(Aside, bool),
    }

    /// A host that records instead of rendering.
    struct FakeHost {
        names: RefCell<Vec<String>>,
        active: RefCell<String>,
        calls: RefCell<Vec<Call>>,
        boom: bool,
    }

    impl FakeHost {
        fn new() -> FakeHost {
            FakeHost {
                names: RefCell::new(["a", "b", "c"].iter().map(|s| (*s).to_string()).collect()),
                active: RefCell::new("a".to_string()),
                calls: RefCell::new(Vec::new()),
                boom: false,
            }
        }
    }

    impl ConsoleHost for FakeHost {
        fn stats(&self) -> ConsoleResult<Value> {
            let mut d = Map::new();
            d.insert("fps".to_string(), Value::from(12.0));
            d.insert(
                "active".to_string(),
                Value::from(self.active.borrow().clone()),
            );
            Ok(Value::Object(d))
        }

        fn scene_names(&self) -> ConsoleResult<Vec<String>> {
            Ok(self.names.borrow().clone())
        }

        fn procedural_names(&self) -> ConsoleResult<Vec<String>> {
            Ok(vec!["c".to_string()])
        }

        fn active_name(&self) -> ConsoleResult<Option<String>> {
            Ok(Some(self.active.borrow().clone()))
        }

        fn scene_def(&self, name: Option<&str>) -> ConsoleResult<Value> {
            self.calls
                .borrow_mut()
                .push(Call::SceneDef(name.map(str::to_string)));
            if self.boom {
                return Err(ConsoleError::server_error("build x: kaboom"));
            }
            if let Some(n) = name {
                if !self.names.borrow().iter().any(|x| x == n) {
                    return Err(ConsoleError::not_found("no scene"));
                }
            }
            let mut d = scene_json();
            d["name"] =
                Value::from(name.map_or_else(|| self.active.borrow().clone(), str::to_string));
            Ok(d)
        }

        fn filters(&self) -> ConsoleResult<Value> {
            Ok(serde_json::json!({"dither": {"fragment": "x"}}))
        }

        fn hwinfo(&self) -> ConsoleResult<Value> {
            Ok(serde_json::json!({"model": "fake"}))
        }

        fn set_param(&self, scope: Scope, name: &str, value: Value) -> ConsoleResult<()> {
            self.calls
                .borrow_mut()
                .push(Call::SetParam(scope, name.to_string(), value));
            Ok(())
        }

        fn set_layer(&self, set: LayerSet) -> ConsoleResult<()> {
            self.calls.borrow_mut().push(Call::SetLayer(set));
            Ok(())
        }

        fn switch(&self, name: Option<&str>) -> ConsoleResult<String> {
            let name = name.unwrap_or("");
            if !self.names.borrow().iter().any(|x| x == name) {
                return Err(ConsoleError::not_found("no scene"));
            }
            name.clone_into(&mut self.active.borrow_mut());
            self.calls.borrow_mut().push(Call::Switch(name.to_string()));
            Ok(name.to_string())
        }

        fn apply_scene(&self, scene: Scene) -> ConsoleResult<()> {
            self.calls.borrow_mut().push(Call::ApplyScene(scene));
            Ok(())
        }

        fn save(&self, name: Option<&str>) -> ConsoleResult<()> {
            self.calls
                .borrow_mut()
                .push(Call::Save(name.map(str::to_string)));
            Ok(())
        }

        fn reset(&self, name: Option<&str>) -> ConsoleResult<()> {
            self.calls
                .borrow_mut()
                .push(Call::Reset(name.map(str::to_string)));
            Ok(())
        }

        fn answer(&self, _name: Option<&str>, aside: &Aside, keep: bool) -> ConsoleResult<()> {
            self.calls
                .borrow_mut()
                .push(Call::Answer(aside.clone(), keep));
            Ok(())
        }
    }

    /// A host with the required methods only. It shows what the trait defaults
    /// answer.
    struct Bare;

    impl ConsoleHost for Bare {
        fn stats(&self) -> ConsoleResult<Value> {
            Ok(Value::Null)
        }
        fn scene_names(&self) -> ConsoleResult<Vec<String>> {
            Ok(Vec::new())
        }
        fn active_name(&self) -> ConsoleResult<Option<String>> {
            Ok(None)
        }
        fn scene_def(&self, _name: Option<&str>) -> ConsoleResult<Value> {
            Ok(Value::Null)
        }
        fn set_param(&self, _scope: Scope, _name: &str, _value: Value) -> ConsoleResult<()> {
            Ok(())
        }
        fn set_layer(&self, _set: LayerSet) -> ConsoleResult<()> {
            Ok(())
        }
        fn switch(&self, _name: Option<&str>) -> ConsoleResult<String> {
            Ok(String::new())
        }
        fn apply_scene(&self, _scene: Scene) -> ConsoleResult<()> {
            Ok(())
        }
    }

    fn api() -> ConsoleAPI<FakeHost> {
        ConsoleAPI::new(FakeHost::new())
    }

    fn query(pairs: &[(&str, &str)]) -> Query {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    fn body(r: &Response) -> String {
        String::from_utf8_lossy(&r.body).into_owned()
    }

    fn as_json(r: &Response) -> Value {
        serde_json::from_slice(&r.body).unwrap()
    }

    fn get(api: &ConsoleAPI<FakeHost>, path: &str, q: &[(&str, &str)]) -> Response {
        api.dispatch("GET", path, &query(q), &Headers::new(), b"")
    }

    fn post(api: &ConsoleAPI<FakeHost>, path: &str, q: &[(&str, &str)]) -> Response {
        api.dispatch("POST", path, &query(q), &Headers::new(), b"")
    }

    // reads

    #[test]
    fn stats_comes_from_the_host() {
        let api = api();
        let r = get(&api, "/stats", &[]);
        assert_eq!(
            (r.status, r.content_type.as_str()),
            (200, "application/json")
        );
        assert_eq!(as_json(&r), api.host.stats().unwrap());
    }

    #[test]
    fn stats_asked_for_usage_adds_what_the_process_uses() {
        let api = api();
        let mut got = as_json(&get(&api, "/stats", &[("usage", "1")]));
        let d = got.as_object_mut().unwrap();
        for key in ["rss_mib", "machine_free_mib"] {
            assert!(d.remove(key).unwrap().as_f64().unwrap() > 0.0, "{key}");
        }
        for key in ["gpu_mib", "soc_celsius"] {
            let v = d.remove(key).unwrap();
            assert!(v.is_null() || v.is_number(), "{key}: {v}");
        }
        // the first answer holds no share yet for any queue
        let busy = d.remove("gpu_busy").unwrap();
        assert!(
            busy.is_null() || busy.as_object().unwrap().values().all(Value::is_null),
            "{busy}"
        );
        assert_eq!(got, api.host.stats().unwrap());
        let again = as_json(&get(&api, "/stats", &[("usage", "1")]));
        let busy = &again["gpu_busy"];
        assert!(
            busy.is_null()
                || busy
                    .as_object()
                    .unwrap()
                    .values()
                    .all(|v| v.is_null() || v.as_f64().is_some_and(|p| p >= 0.0)),
            "{busy}"
        );
    }

    #[test]
    fn scenes_lists_names_active_and_the_procedural_ones() {
        assert_eq!(
            as_json(&get(&api(), "/scenes", &[])),
            serde_json::json!({"scenes": ["a", "b", "c"], "active": "a", "procedural": ["c"]})
        );
        let bare =
            ConsoleAPI::new(Bare).dispatch("GET", "/scenes", &Query::new(), &Headers::new(), b"");
        assert_eq!(
            as_json(&bare),
            serde_json::json!({"scenes": [], "active": null, "procedural": []})
        );
    }

    #[test]
    fn scene_without_a_name_asks_for_the_active_one() {
        let api = api();
        get(&api, "/scene", &[]);
        assert_eq!(*api.host.calls.borrow(), vec![Call::SceneDef(None)]);
    }

    #[test]
    fn scene_passes_the_name_through() {
        let api = api();
        assert_eq!(as_json(&get(&api, "/scene", &[("name", "b")]))["name"], "b");
        assert_eq!(
            *api.host.calls.borrow(),
            vec![Call::SceneDef(Some("b".to_string()))]
        );
    }

    #[test]
    fn an_unknown_scene_is_404() {
        assert_eq!(get(&api(), "/scene", &[("name", "nope")]).status, 404);
    }

    #[test]
    fn a_host_error_keeps_its_status() {
        let mut api = api();
        api.host.boom = true;
        let r = get(&api, "/scene", &[]);
        assert_eq!(r.status, 500);
        assert!(body(&r).contains("kaboom"));
    }

    #[test]
    fn filters_and_hwinfo_come_from_the_host() {
        let api = api();
        assert_eq!(
            as_json(&get(&api, "/filters", &[])),
            serde_json::json!({"dither": {"fragment": "x"}})
        );
        assert_eq!(
            as_json(&get(&api, "/hwinfo", &[])),
            serde_json::json!({"model": "fake"})
        );
    }

    #[test]
    fn a_host_with_no_working_copies_reports_no_edit() {
        assert_eq!(
            as_json(&ConsoleAPI::new(Bare).dispatch(
                "GET",
                "/edits",
                &Query::new(),
                &Headers::new(),
                b""
            )),
            serde_json::json!({"rev": null, "values": {}, "items": [], "unsaved": false,
                               "aside": []})
        );
    }

    #[test]
    fn defaults_supply_the_registry_and_ghees_own_hwinfo() {
        let (host, own) = (Bare.hwinfo().unwrap(), crate::console::feeds::hwinfo());
        assert_eq!((&host["model"], &host["gpu"]), (&own["model"], &own["gpu"]));
        assert!(Bare.filters().unwrap().is_object()); // ghee's own filter registry
    }

    /// The effects and their defaults as the README's table gives them, and
    /// the live fields as docs/http.md lists them for /set.
    #[test]
    fn format_serves_the_effects_and_the_live_fields_that_ghee_draws() {
        let r = get(&api(), "/format", &[]);
        let f = as_json(&r);
        let defaults: Vec<(String, Vec<(String, Value)>)> = f["effects"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, fields)| {
                let pairs = fields
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| (d["key"].as_str().unwrap().to_string(), d["default"].clone()))
                    .collect();
                (k.clone(), pairs)
            })
            .collect();
        let row = |k: &str, fields: &[(&str, Value)]| {
            (
                k.to_string(),
                fields
                    .iter()
                    .map(|(n, v)| ((*n).to_string(), v.clone()))
                    .collect::<Vec<_>>(),
            )
        };
        use serde_json::json;
        assert_eq!(
            defaults,
            vec![
                row(
                    "text_shadow",
                    &[
                        ("x", json!(0.06)),
                        ("y", json!(0.08)),
                        ("blur", json!(0.15)),
                        ("color", json!("#000000")),
                        ("strength", json!(0.7))
                    ]
                ),
                row(
                    "glow",
                    &[
                        ("blur", json!(0.3)),
                        ("color", json!("#ffa500")),
                        ("strength", json!(0.8))
                    ]
                ),
                row(
                    "text_stroke",
                    &[("width", json!(0.06)), ("color", json!("#000000"))]
                ),
                row(
                    "karaoke",
                    &[("sung", json!(0.0)), ("color", json!("#ffad2e"))]
                ),
                row("wave", &[("height", json!(0.1)), ("speed", json!(0.6))]),
                row("scroll", &[("speed", json!(2.0))]),
                row("blink", &[("rate", json!(1.0))]),
            ]
        );
        let live: Vec<String> = f["effects"]
            .as_object()
            .unwrap()
            .values()
            .flat_map(|fields| fields.as_array().unwrap().iter())
            .filter(|d| d["live"] == true)
            .map(|d| d["key"].as_str().unwrap().to_string())
            .collect();
        for key in &live {
            assert!(
                ["color", "strength", "sung", "speed", "rate"].contains(&key.as_str()),
                "{key}"
            );
        }
        let every = ["x", "y", "color", "opacity", "visible"];
        let kinds: [(&str, &[&str]); 4] = [
            ("text", &["text"]),
            ("image", &["image"]),
            ("rect", &["width", "height", "border_radius"]),
            ("line", &["points", "stroke_width"]),
        ];
        for (kind, own) in kinds {
            let mut want: Vec<&str> = every.to_vec();
            want.extend(own);
            want.sort_unstable();
            let mut got: Vec<&str> = f["live"][kind]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            got.sort_unstable();
            assert_eq!(got, want, "{kind}");
        }
        assert_eq!(f["fonts"], json!({"built-in": null, "pixel": "pixel"}));
    }

    /// A rect's border, the keys of each shape and the render of a text
    /// layer that names none, as the README's tables give them.
    #[test]
    fn format_serves_the_border_the_keys_of_each_shape_and_the_render_of_each_font() {
        use serde_json::json;
        let f = as_json(&get(&api(), "/format", &[]));
        assert_eq!(
            f["border"],
            json!([{"key": "width", "default": 1.0, "live": true},
                   {"key": "color", "default": "#ffffff", "live": true}])
        );
        assert_eq!(
            f["shapes"],
            json!({"rect": ["width", "height", "border", "border_radius"],
                   "line": ["points", "stroke_width"]})
        );
        assert_eq!(
            f["render"],
            json!({"default": "auto", "fonts": {"pixel": "slug"}})
        );
    }

    // writes

    #[test]
    fn set_routes_the_scene_scope() {
        let api = api();
        post(&api, "/set", &[("name", "tint"), ("value", "0.5")]);
        assert_eq!(
            *api.host.calls.borrow(),
            vec![
                Call::SceneDef(None),
                Call::SetParam(Scope::Scene, "tint".to_string(), Value::from(0.5))
            ]
        );
    }

    #[test]
    fn set_parses_the_value_by_the_declared_type() {
        let main = Some("main");
        for (pass, name, raw, want) in [
            (None, "title", "hello", Value::from("hello")),
            (None, "title", "1.5", Value::from("1.5")),
            (main, "on", "true", Value::from(true)),
            (main, "on", "false", Value::from(false)),
            (main, "k", "2", Value::from(2.0)),
            (main, "k", "0.5", Value::from(0.5)),
            (main, "n", "5", Value::from(5.0)),
            (main, "c", "1,0,0.5", serde_json::json!([1.0, 0.0, 0.5])),
            (main, "v", "0.1,0.2", serde_json::json!([0.1, 0.2])),
            // a bare name that no scene param holds is the first pass's
            (None, "k", "0.25", Value::from(0.25)),
        ] {
            let api = api();
            let mut q = vec![("name", name), ("value", raw)];
            q.extend(pass.map(|p| ("pass", p)));
            let r = post(&api, "/set", &q);
            assert_eq!((r.status, body(&r).as_str()), (200, "ok"), "{name}");
            let scope = match (pass, name) {
                (None, "title") => Scope::Scene,
                _ => Scope::Pass(0),
            };
            assert_eq!(
                api.host.calls.borrow().last(),
                Some(&Call::SetParam(scope, name.to_string(), want)),
                "{name}"
            );
        }
    }

    /// The /set of `field` of `layer`, or of its effect `effect`, to `value`.
    fn layer_query<'a>(
        layer: &'a str,
        effect: Option<&'a str>,
        field: &'a str,
        value: &'a str,
    ) -> Vec<(&'a str, &'a str)> {
        let mut q = vec![("layer", layer), ("field", field), ("value", value)];
        q.extend(effect.map(|e| ("effect", e)));
        q
    }

    #[test]
    fn a_layer_set_parses_its_value_by_the_field() {
        let gradient = r##"{"from": "#ff0000", "to": "#0000ff", "angle": 90}"##;
        for (layer, effect, field, raw, want) in [
            ("words", None, "x", "10", serde_json::json!(10.0)),
            ("words", None, "y", "none", Value::Null),
            ("words", None, "opacity", "0.5", serde_json::json!(0.5)),
            ("words", None, "visible", "false", Value::from(false)),
            ("words", None, "color", " #FF0000 ", Value::from("#FF0000")),
            (
                "words",
                None,
                "color",
                "DarkOrange",
                Value::from("DarkOrange"),
            ),
            (
                "words",
                Some("karaoke"),
                "color",
                "transparent",
                Value::from("transparent"),
            ),
            (
                "words",
                None,
                "color",
                "1,0,0.5",
                serde_json::json!([1.0, 0.0, 0.5]),
            ),
            (
                "words",
                None,
                "color",
                gradient,
                serde_json::from_str(gradient).unwrap(),
            ),
            ("words", None, "color", "none", Value::Null),
            (
                "words",
                Some("karaoke"),
                "sung",
                "2.5",
                serde_json::json!(2.5),
            ),
            ("words", Some("karaoke"), "visible", "0", Value::from(false)),
            ("icon", None, "color", "none", Value::Null),
            ("icon", None, "image", "pic", Value::from("pic")),
            (
                "scope",
                None,
                "points",
                "[[0, 16], [8, 4.5]]",
                serde_json::json!([[0, 16], [8, 4.5]]),
            ),
            ("frame", Some("border"), "color", "Red", Value::from("Red")),
            ("frame", None, "width", "none", Value::Null),
            ("words", None, "text", " la la ", Value::from(" la la ")),
        ] {
            let api = api();
            let r = post(&api, "/set", &layer_query(layer, effect, field, raw));
            assert_eq!((r.status, body(&r).as_str()), (200, "ok"), "{field} {raw}");
            let set = LayerSet {
                layer: layer.to_string(),
                effect: effect.map(str::to_string),
                field: field.to_string(),
                value: want,
            };
            assert_eq!(api.host.calls.borrow().last(), Some(&Call::SetLayer(set)));
        }
    }

    #[test]
    fn a_malformed_layer_set_never_reaches_the_host() {
        let q = |l, e, f, v| layer_query(l, e, f, v);
        let some = Some("karaoke");
        for (query, why) in [
            (
                vec![("layer", "words"), ("value", "1")],
                "the query has no 'field'".to_string(),
            ),
            (
                vec![("layer", "words"), ("field", "x")],
                "the query has no 'value'".to_string(),
            ),
            (
                q("nope", None, "x", "1"),
                "layer 'nope' is not a layer this scene declares (words, icon, frame, scope)"
                    .to_string(),
            ),
            (
                q("words", None, "x", "left"),
                "layer 'words' x needs a number or none, got 'left'".to_string(),
            ),
            (
                q("words", None, "opacity", "none"),
                "layer 'words' opacity needs a number, got 'none'".to_string(),
            ),
            (
                q("words", None, "visible", "yes"),
                "layer 'words' visible needs true or false, got 'yes'".to_string(),
            ),
            (
                q("words", None, "color", "1,0"),
                "layer 'words' color needs '#rrggbb', a CSS colour name, three comma-separated \
                 numbers, a gradient object or none, got '1,0'"
                    .to_string(),
            ),
            (
                q("icon", None, "color", r#"{"from": [1, 0, 0]}"#),
                "layer 'icon' color needs '#rrggbb', a CSS colour name, three comma-separated \
                 numbers or none, got '{\"from\": [1, 0, 0]}'"
                    .to_string(),
            ),
            (
                q("words", some, "color", "none"),
                "layer 'words' karaoke color needs '#rrggbb', a CSS colour name or three \
                 comma-separated numbers, got 'none'"
                    .to_string(),
            ),
            (
                q("icon", None, "image", "sans"),
                "layer 'icon' image needs an image asset of the scene or a .png or .jpg file, \
                 got 'sans'"
                    .to_string(),
            ),
            (
                q("words", None, "color", r##"{"from": "#ff0000"}"##),
                "layer 'words' color has no 'to'".to_string(),
            ),
            (
                q("words", None, "font_size", "20"),
                "layer 'words' font_size changes with a rebuild".to_string(),
            ),
        ] {
            let api = api();
            let r = post(&api, "/set", &query);
            assert_eq!(r.status, 400, "{query:?}");
            // The errors of the scene's own checks hold more than the key.
            match why.ends_with("rebuild") {
                true => assert!(body(&r).starts_with(&format!("bad /set: {why}"))),
                false => assert_eq!(body(&r), format!("bad /set: {why}")),
            }
            assert!(!api
                .host
                .calls
                .borrow()
                .iter()
                .any(|c| matches!(c, Call::SetLayer(..))));
        }
    }

    #[test]
    fn a_set_of_an_undeclared_param_names_it_as_python_does() {
        let wires = "tint, title, main.k, k, main.on, on, main.n, n, main.c, c, main.v, v";
        for (pass, name, wire) in [
            (None, "nope", "nope"),
            (Some("main"), "nope", "main.nope"),
            (Some("other"), "k", "other.k"),
            (Some("scene"), "tint", "scene.tint"),
            // a text's words take the layer's address alone
            (None, "words", "words"),
        ] {
            let api = api();
            let mut q = vec![("name", name), ("value", "1")];
            q.extend(pass.map(|p| ("pass", p)));
            let r = post(&api, "/set", &q);
            let want =
                format!("bad /set: param '{wire}' is not a param this scene declares ({wires})");
            assert_eq!((r.status, body(&r)), (400, want));
            assert_eq!(*api.host.calls.borrow(), vec![Call::SceneDef(None)]);
        }
    }

    #[test]
    fn a_set_carries_the_host_error_of_its_scene_lookup() {
        let mut host = FakeHost::new();
        host.boom = true;
        let api = ConsoleAPI::new(host);
        let r = post(
            &api,
            "/set",
            &[("pass", "main"), ("name", "k"), ("value", "1")],
        );
        assert_eq!((r.status, body(&r).as_str()), (500, "build x: kaboom"));
    }

    #[test]
    fn a_malformed_set_never_reaches_the_host() {
        for (q, why) in [
            (
                vec![("pass", "main"), ("name", "k")],
                "the query has no 'value'",
            ),
            (
                vec![("pass", "main"), ("value", "1")],
                "the query has no 'name'",
            ),
            (
                vec![("pass", "main"), ("name", "c"), ("value", "1,0")],
                "color needs '#rrggbb', a CSS colour name or three comma-separated floats, got \
                 '1,0'",
            ),
            (
                vec![("pass", "main"), ("name", "on"), ("value", "yes")],
                "bool needs true or false, got 'yes'",
            ),
            (
                vec![("pass", "main"), ("name", "k"), ("value", "notanumber")],
                "'notanumber' is not a number",
            ),
        ] {
            let api = api();
            let r = post(&api, "/set", &q);
            assert_eq!((r.status, body(&r)), (400, format!("bad /set: {why}")));
            assert!(!api
                .host
                .calls
                .borrow()
                .iter()
                .any(|c| matches!(c, Call::SetParam(..))));
        }
    }

    #[test]
    fn an_edit_whose_rev_is_no_number_is_400() {
        let api = api();
        let r = post(
            &api,
            "/set",
            &[("name", "tint"), ("value", "1"), ("rev", "x")],
        );
        assert_eq!(
            (r.status, body(&r).as_str()),
            (400, "bad /set: rev needs a revision number, got 'x'")
        );
        assert!(api.host.calls.borrow().is_empty());
    }

    #[test]
    fn switch_returns_the_new_active() {
        let api = api();
        assert_eq!(
            as_json(&post(&api, "/switch", &[("name", "c")])),
            serde_json::json!({"active": "c"})
        );
        assert_eq!(
            *api.host.calls.borrow(),
            vec![Call::Switch("c".to_string())]
        );
    }

    #[test]
    fn switching_to_nothing_is_404() {
        let api = api();
        assert_eq!(post(&api, "/switch", &[]).status, 404);
        assert!(api.host.calls.borrow().is_empty());
    }

    #[test]
    fn next_and_prev_wrap_through_the_host_names() {
        let api = api();
        for want in ["b", "c", "a"] {
            assert_eq!(
                as_json(&post(&api, "/next", &[])),
                serde_json::json!({ "active": want })
            );
        }
        assert_eq!(
            as_json(&post(&api, "/prev", &[])),
            serde_json::json!({"active": "c"})
        );
    }

    #[test]
    fn step_with_no_scenes_is_404() {
        let api = api();
        api.host.names.borrow_mut().clear();
        assert_eq!(post(&api, "/next", &[]).status, 404);
    }

    #[test]
    fn a_posted_scene_arrives_validated() {
        let api = api();
        let r = api.dispatch(
            "POST",
            "/scene",
            &Query::new(),
            &json_ct(),
            SCENE.as_bytes(),
        );
        assert_eq!(r.status, 200);
        let doc: Value = serde_json::from_str(&body(&r)).unwrap();
        assert_eq!(doc["passes"][0]["name"], "main");
        let calls = api.host.calls.borrow();
        match &calls[0] {
            Call::ApplyScene(scene) => assert_eq!(scene.passes[0].name, "main"),
            c => panic!("{c:?}"),
        }
    }

    #[test]
    fn an_invalid_scene_is_400_and_never_reaches_the_host() {
        let api = api();
        let r = api.dispatch(
            "POST",
            "/scene",
            &Query::new(),
            &json_ct(),
            br#"{"passes": "not-a-list"}"#,
        );
        assert_eq!(r.status, 400);
        assert!(body(&r).contains("passes"), "{}", body(&r));
        assert!(api.host.calls.borrow().is_empty());
    }

    #[test]
    fn a_refused_scene_names_its_request_once() {
        let api = api();
        let posted =
            br#"{"passes": [{"name": "main", "fragment": "void main(){}", "chanels": []}]}"#;
        let r = api.dispatch("POST", "/scene", &Query::new(), &json_ct(), posted);
        assert_eq!(
            (r.status, body(&r).as_str()),
            (
                400,
                "bad /scene: pass 'main' takes no 'chanels', the closest key is 'channels'"
            )
        );
    }

    #[test]
    fn a_scene_naming_a_fragment_file_is_400() {
        // A posted scene has no base directory. A fragment file reads a file
        // off disk, so the model refuses it and the API passes that on.
        let api = api();
        let r = api.dispatch(
            "POST",
            "/scene",
            &Query::new(),
            &json_ct(),
            br#"{"passes": [{"name": "main", "fragment": {"file": "/etc/hostname"}}]}"#,
        );
        assert_eq!(r.status, 400);
        assert!(body(&r).contains("names the file"), "{}", body(&r));
        assert!(api.host.calls.borrow().is_empty());
    }

    #[test]
    fn unparseable_json_is_400() {
        let api = api();
        let r = api.dispatch("POST", "/scene", &Query::new(), &json_ct(), b"{oops");
        assert_eq!(r.status, 400);
        assert!(api.host.calls.borrow().is_empty());
    }

    #[test]
    fn a_body_without_a_json_content_type_is_415() {
        let api = api();
        let r = api.dispatch(
            "POST",
            "/scene",
            &Query::new(),
            &Headers::new(),
            SCENE.as_bytes(),
        );
        assert_eq!(r.status, 415);
        assert!(body(&r).contains("application/json"));
        assert!(api.host.calls.borrow().is_empty());
    }

    #[test]
    fn a_host_that_keeps_no_state_file_refuses_a_save_and_a_reset() {
        let api = ConsoleAPI::new(Bare);
        for (path, why) in [
            ("/save", "this console saves no scene"),
            (
                "/reset",
                "this console's scene has no defaults to go back to",
            ),
        ] {
            let r = api.dispatch("POST", path, &Query::new(), &Headers::new(), b"");
            assert_eq!((r.status, body(&r).as_str()), (403, why));
        }
        let r = api.dispatch(
            "GET",
            "/scene",
            &query(&[("defaults", "1")]),
            &Headers::new(),
            b"",
        );
        assert_eq!(
            (r.status, body(&r).as_str()),
            (404, "this console's scene has no defaults")
        );
    }

    #[test]
    fn a_save_reaches_the_host_with_the_scene_it_names() {
        let api = api();
        let r = post(&api, "/save", &[("name", "a")]);
        assert_eq!(r.status, 200);
        assert_eq!(
            *api.host.calls.borrow(),
            vec![Call::Save(Some("a".to_string()))]
        );
    }

    #[test]
    fn an_answer_names_its_item_and_keeps_or_drops_it() {
        let kept = api();
        let r = post(
            &kept,
            "/aside",
            &[("item", "fragment"), ("pass", "main"), ("answer", "keep")],
        );
        assert_eq!(r.status, 200);
        let main = Aside {
            item: "fragment".to_string(),
            pass: Some("main".to_string()),
            param: None,
        };
        assert_eq!(*kept.host.calls.borrow(), vec![Call::Answer(main, true)]);
        for (q, why) in [
            (
                vec![("item", "fragment"), ("answer", "maybe")],
                "answer needs keep or default, got 'maybe'",
            ),
            (vec![("item", "fragment")], "the query has no 'answer'"),
            (vec![("answer", "keep")], "the query has no 'item'"),
        ] {
            let api = api();
            let r = post(&api, "/aside", &q);
            assert_eq!((r.status, body(&r)), (400, format!("bad /aside: {why}")));
            assert!(api.host.calls.borrow().is_empty());
        }
    }

    #[test]
    fn an_unrouted_post_is_404() {
        let api = api();
        assert_eq!(
            api.dispatch("POST", "/nope", &Query::new(), &json_ct(), b"{}")
                .status,
            404
        );
    }

    #[test]
    fn an_unknown_method_is_405() {
        assert_eq!(
            api()
                .dispatch("DELETE", "/stats", &Query::new(), &Headers::new(), b"")
                .status,
            405
        );
    }

    /// Each request that edits a scene, naming scene `name`: a /set, a posted
    /// scene, a save, a reset and an answer.
    fn edits(api: &ConsoleAPI<FakeHost>, name: &str) -> [Response; 5] {
        [
            post(
                api,
                "/set",
                &[("scene", name), ("name", "tint"), ("value", "0.5")],
            ),
            api.dispatch(
                "POST",
                "/scene",
                &query(&[("name", name)]),
                &json_ct(),
                SCENE.as_bytes(),
            ),
            post(api, "/save", &[("name", name)]),
            post(api, "/reset", &[("name", name)]),
            post(
                api,
                "/aside",
                &[("name", name), ("item", "common"), ("answer", "keep")],
            ),
        ]
    }

    #[test]
    fn an_edit_for_a_scene_other_than_the_active_one_is_409_and_never_reaches_the_host() {
        let api = api();
        for r in edits(&api, "b") {
            assert_eq!(
                (r.status, body(&r).as_str()),
                (
                    409,
                    "the console is on scene 'a', and this edit is for scene 'b'"
                )
            );
        }
        // A scene def is a read, and none of the edits lands.
        assert!(api
            .host
            .calls
            .borrow()
            .iter()
            .all(|c| matches!(c, Call::SceneDef(_))));
        for r in edits(&api, "a") {
            assert_eq!(r.status, 200, "{}", body(&r));
        }
        let landed = api
            .host
            .calls
            .borrow()
            .iter()
            .filter(|c| !matches!(c, Call::SceneDef(_)))
            .count();
        assert_eq!(landed, 5);
    }

    // refusals: precheck decides them, dispatch enforces them again

    #[test]
    fn state_changing_paths_are_post_only() {
        let api = api();
        for path in [
            "/set", "/switch", "/next", "/prev", "/save", "/reset", "/aside",
        ] {
            let r = get(
                &api,
                path,
                &[("pass", "main"), ("name", "k"), ("value", "1")],
            );
            assert_eq!((r.status, path), (405, path));
            assert!(body(&r).contains("POST"));
        }
        assert!(api.host.calls.borrow().is_empty());
    }

    #[test]
    fn a_host_declaring_its_own_get_routes_mutates_through_post() {
        // A host that keeps GET /set for its scripts relabels the request as a
        // POST. The core keeps its refusal.
        let api = ConsoleAPI::new(FakeHost::new()).with_post_only(&[]);
        assert!(api.precheck("GET", "/set", &Headers::new()).is_none());
        post(
            &api,
            "/set",
            &[("pass", "main"), ("name", "k"), ("value", "1")],
        );
        assert_eq!(
            *api.host.calls.borrow(),
            vec![
                Call::SceneDef(None),
                Call::SetParam(Scope::Pass(0), "k".to_string(), Value::from(1.0))
            ]
        );
    }

    #[test]
    fn a_cross_origin_post_is_refused() {
        let api = api();
        let headers = Headers::from([
            ("Content-Type", "application/json"),
            ("Host", "127.0.0.1:8090"),
            ("Origin", "http://evil.example"),
        ]);
        let r = api.dispatch("POST", "/scene", &Query::new(), &headers, SCENE.as_bytes());
        assert_eq!(r.status, 403);
        assert!(body(&r).contains("cross-origin"));
        assert!(api.host.calls.borrow().is_empty());
    }

    #[test]
    fn a_same_origin_post_is_accepted() {
        let api = api();
        let headers = Headers::from([
            ("Content-Type", "application/json"),
            ("Host", "127.0.0.1:8090"),
            ("Origin", "http://127.0.0.1:8090"),
        ]);
        assert_eq!(
            api.dispatch("POST", "/scene", &Query::new(), &headers, SCENE.as_bytes())
                .status,
            200
        );
    }

    #[test]
    fn a_get_carrying_an_origin_is_not_refused() {
        // A read is safe. Only a mutating request gets the Origin check.
        let headers = Headers::from([("Origin", "http://evil.example")]);
        assert!(api().precheck("GET", "/stats", &headers).is_none());
    }

    #[test]
    fn an_oversized_body_is_413_before_it_is_read() {
        let api = api();
        let over = (MAX_BODY + 1).to_string();
        let headers = Headers::from([
            ("Content-Type", "application/json"),
            ("Content-Length", over.as_str()),
        ]);
        let r = api.precheck("POST", "/scene", &headers).unwrap();
        assert_eq!(r.status, 413);
        assert_eq!(body(&r), "body over the limit of 1048576 bytes");
        assert_eq!(
            api.dispatch("POST", "/scene", &Query::new(), &headers, b"{}")
                .status,
            413
        );
        assert!(api.host.calls.borrow().is_empty());
    }

    #[test]
    fn a_bad_content_length_is_400() {
        let headers = Headers::from([("Content-Length", "not-a-number")]);
        assert_eq!(
            api().precheck("POST", "/scene", &headers).unwrap().status,
            400
        );
    }

    #[test]
    fn the_body_cap_is_configurable() {
        let api = ConsoleAPI::new(FakeHost::new()).with_max_body(10);
        assert_eq!(
            api.precheck("POST", "/scene", &Headers::from([("Content-Length", "11")]))
                .unwrap()
                .status,
            413
        );
        assert!(api
            .precheck("POST", "/scene", &Headers::from([("Content-Length", "10")]))
            .is_none());
    }

    // static files

    #[test]
    fn static_serves_the_bundled_page() {
        let api = ConsoleAPI::new(FakeHost::new()).with_web(true);
        let r = get(&api, "/", &[]);
        assert_eq!(r.status, 200);
        assert!(body(&r).contains("<title>ghee</title>"));
        assert_eq!(r.content_type, "text/html");
    }

    #[test]
    fn no_web_root_serves_nothing() {
        assert_eq!(get(&api(), "/index.html", &[]).status, 404);
    }

    #[test]
    fn static_path_refuses_traversal_and_prefix_siblings() {
        assert_eq!(static_path("/"), Some("index.html"));
        assert_eq!(static_path("/../webXYZ/secret.txt"), None);
        assert_eq!(static_path("/../../etc/passwd"), None);
        assert_eq!(static_path("/missing.js"), None);
    }

    // the pure helpers

    #[test]
    fn origin_helper() {
        assert!(origin_allowed("", Some("h"))); // curl, a script, the CLI send none
        assert!(origin_allowed("http://127.0.0.1:1", Some("127.0.0.1:1")));
        assert!(!origin_allowed("http://evil.example", Some("127.0.0.1:1")));
        assert!(!origin_allowed("null", Some("127.0.0.1:1")));
        assert!(!origin_allowed("http://evil.example", None)); // no Host to match
    }

    #[test]
    fn parse_query_takes_the_first_value() {
        assert_eq!(
            parse_query("a=1&b=x&a=2").unwrap(),
            query(&[("a", "1"), ("b", "x")])
        );
        assert_eq!(parse_query("").unwrap(), Query::new());
        assert_eq!(
            parse_query("a=%41+b%2b%c3%a9").unwrap(),
            query(&[("a", "A b+é")])
        );
    }

    #[test]
    fn a_malformed_percent_escape_is_400() {
        let e = parse_query("a=%1é").unwrap_err();
        assert_eq!(e.status, 400);
        assert_eq!(e.message, "bad query: '%1é' is not a percent escape");
        for (qs, escape) in [("a=%zz", "'%zz'"), ("a=1%", "'%'"), ("%4=1", "'%4'")] {
            let e = parse_query(qs).unwrap_err();
            assert!(
                e.message.starts_with(&format!("bad query: {escape} ")),
                "{qs}"
            );
        }
    }

    #[test]
    fn a_set_value_parses_by_its_type() {
        use serde_json::json;
        // An int parses as a float here. The render thread rounds it with the
        // scene's own param type, so the wire value stays a plain number.
        for (ty, raw, want) in [
            ("float", "0.5", Ok(json!(0.5))),
            ("float", "3", Ok(json!(3.0))),
            ("int", "5", Ok(json!(5.0))),
            ("float", "notanumber", Err("'notanumber' is not a number")),
            ("bool", "true", Ok(json!(true))),
            ("bool", "TRUE", Ok(json!(true))),
            ("bool", " true ", Ok(json!(true))),
            ("bool", "1", Ok(json!(true))),
            ("bool", "false", Ok(json!(false))),
            ("bool", "False", Ok(json!(false))),
            ("bool", "0", Ok(json!(false))),
            ("bool", "yes", Err("bool needs true or false, got 'yes'")),
            ("bool", "", Err("bool needs true or false, got ''")),
            ("bool", "2", Err("bool needs true or false, got '2'")),
            ("bool", "0.0", Err("bool needs true or false, got '0.0'")),
            ("color", "1,0,0.5", Ok(json!([1.0, 0.0, 0.5]))),
            ("color", " #0A0b0C ", Ok(json!("#0A0b0C"))),
            ("color", " NavY ", Ok(json!("NavY"))),
            ("color", "transparent", Ok(json!("transparent"))),
            (
                "color",
                "1,0",
                Err("color needs '#rrggbb', a CSS colour name or three comma-separated floats, got '1,0'"),
            ),
            (
                "color",
                "1,0,0,0",
                Err("color needs '#rrggbb', a CSS colour name or three comma-separated floats, got '1,0,0,0'"),
            ),
            (
                "color",
                "1,x,0",
                Err("color needs '#rrggbb', a CSS colour name or three comma-separated floats, got '1,x,0'"),
            ),
            (
                "color",
                "#ff80",
                Err("color needs '#rrggbb', a CSS colour name or three comma-separated floats, got '#ff80'"),
            ),
            (
                "color",
                "#ff800g",
                Err("color needs '#rrggbb', a CSS colour name or three comma-separated floats, got '#ff800g'"),
            ),
            ("vec2", "0.1,0.2", Ok(json!([0.1, 0.2]))),
            ("vec3", "1,0,0.5", Ok(json!([1.0, 0.0, 0.5]))),
            ("vec4", "0,0.25,0.5,1", Ok(json!([0.0, 0.25, 0.5, 1.0]))),
            (
                "vec2",
                "0.1",
                Err("vec2 needs 2 comma-separated floats, got '0.1'"),
            ),
            (
                "vec2",
                "0.1,0.2,0.3",
                Err("vec2 needs 2 comma-separated floats, got '0.1,0.2,0.3'"),
            ),
            (
                "vec3",
                "0.1,0.2",
                Err("vec3 needs 3 comma-separated floats, got '0.1,0.2'"),
            ),
            (
                "vec4",
                "0.1,0.2,0.3",
                Err("vec4 needs 4 comma-separated floats, got '0.1,0.2,0.3'"),
            ),
            ("vec3", "0.1,x,0.3", Err("'x' is not a number")),
            ("string", "GHEE", Ok(json!("GHEE"))),
            ("string", "hello world", Ok(json!("hello world"))),
        ] {
            let got = parse_param_value(ty, raw);
            assert_eq!(got, want.map_err(str::to_string), "{ty} {raw:?}");
        }
    }

    #[test]
    fn a_body_whose_read_fails_answers_400_with_the_reason() {
        struct Cut(bool);
        impl Read for Cut {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if std::mem::replace(&mut self.0, true) {
                    return Err(std::io::Error::other("the client hung up"));
                }
                buf[..2].copy_from_slice(b"{\"");
                Ok(2)
            }
        }
        let refused = api().read_body(&mut Cut(false)).unwrap_err();
        assert_eq!(refused.status, 400);
        assert_eq!(refused.body, b"bad body: the client hung up");
    }

    #[test]
    fn a_step_from_an_unknown_active_starts_at_the_first_name() {
        let api = api();
        "gone".clone_into(&mut api.host.active.borrow_mut());
        assert_eq!(
            as_json(&post(&api, "/next", &[])),
            serde_json::json!({"active": "b"})
        );
    }
}
