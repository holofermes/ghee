//! The console API, with no transport under it.
//!
//! A host implements ConsoleHost over what it renders, and hands that to a
//! ConsoleAPI. The transport turns each of its own requests into (method, path,
//! query, headers, body). It turns the Response back into what it serves:
//!
//! ```text
//! let api = ConsoleAPI::new(MyHost::new()).with_web(true);
//! let refused = api.precheck(method, path, headers);  // before reading a body
//! let resp = api.dispatch(method, path, query, headers, body);
//! ```
//!
//! query maps a name to its first value, and parse_query() builds one from a raw
//! query string. A headers lookup ignores case.
//!
//! The transport keeps the streams (/stream, /h264) and its own routes. Route
//! them before dispatch reaches the static-file fallback.
//!
//! Every refusal lives here, so a new host inherits it. The refusals are the
//! Origin check on a mutating request, the body cap, POST-only state changes,
//! the JSON content type, an edit of a scene other than the active one and the
//! static root check.
//!
//! An API holds no mutable state, so any number of threads may call it at once.
//! A host states its own thread contract for set_param, switch and apply_scene.

use std::collections::BTreeMap;
use std::fmt;

use serde_json::{Map, Value};

use crate::console::{static_asset, WEB};
use crate::filters::{filter_defs, rebuild_filters};
use crate::layer::{Address, Form, Layer, Live, Rgb};
use crate::scene::{not_an_image, py_repr, Scene};

/// The console's cap on a request body. A POSTed scene is JSON text.
pub const MAX_BODY: usize = 1 << 20;

/// The paths that change the renderer. They take POST only. A hostile page
/// reaches a GET through <img src>, and that request carries no Origin to
/// refuse.
pub const POST_ONLY: [&str; 4] = ["/set", "/switch", "/next", "/prev"];

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
}

impl Response {
    pub fn text(status: u16, msg: &str) -> Response {
        Response {
            status,
            body: msg.as_bytes().to_vec(),
            content_type: "text/plain".to_string(),
        }
    }

    pub fn json(obj: &Value) -> Response {
        Response {
            status: 200,
            body: obj.to_string().into_bytes(),
            content_type: "application/json".to_string(),
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

    pub fn unsupported_media(msg: impl Into<String>) -> ConsoleError {
        ConsoleError {
            status: 415,
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

/// A /set target. The literal "scene" names the scene's shared params. Anything
/// else is a pass index.
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
/// A field with a blank value is dropped, so `?value=` carries no value.
pub fn parse_query(qs: &str) -> Query {
    let mut out = Query::new();
    for field in qs.split('&').filter(|f| !f.is_empty()) {
        let Some((name, value)) = field.split_once('=') else {
            continue;
        };
        if value.is_empty() {
            continue;
        }
        out.entry(unquote_plus(name))
            .or_insert_with(|| unquote_plus(value));
    }
    out
}

/// Decode a query field. "+" is a space and %XX is a byte. An invalid escape
/// stays as it is.
fn unquote_plus(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => match u8::from_str_radix(&s[i + 1..i + 3], 16) {
                Ok(b) => {
                    out.push(b);
                    i += 3;
                }
                Err(_) => {
                    out.push(b'%');
                    i += 1;
                }
            },
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
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

/// Parse a decimal integer. Whitespace around it is fine.
fn py_int(raw: &str) -> Option<i64> {
    let s = raw.trim();
    let digits = s.strip_prefix(['+', '-']).unwrap_or(s);
    let decimal = !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit());
    s.parse::<i64>().ok().filter(|_| decimal)
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

/// The /set pass= target. The literal "scene" names the scene's shared params.
/// Anything else is a pass index, and a non-integer errs.
pub fn parse_scope(raw: &str) -> Result<Scope, String> {
    if raw == "scene" {
        return Ok(Scope::Scene);
    }
    py_int(raw).map(Scope::Pass).ok_or_else(|| {
        format!(
            "pass needs 'scene' or a pass index, got {}",
            py_repr(&Value::from(raw))
        )
    })
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
    /// /save is refused unless a host opts in. save_refused carries the text.
    fn writable(&self) -> bool {
        false
    }

    fn save_refused(&self) -> String {
        "this console cannot save".to_string()
    }

    /// {fps, frame_ms, stream_fps, out, error, error_line, error_pass,
    /// text_errors, active, display, stale, layer_changes}.
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
    /// page reads the scene again when it moves.
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

    /// Each filter name and its definition, for the "+ filter" picker.
    fn filters(&self) -> ConsoleResult<Value> {
        Ok(Value::Object(filter_defs()))
    }

    fn hwinfo(&self) -> ConsoleResult<Value> {
        Ok(Value::Object(Map::new()))
    }

    /// Set a live uniform, with no recompile. scope is a pass index, or Scene
    /// for a scene-scoped param.
    fn set_param(&self, scope: Scope, name: &str, value: Value) -> ConsoleResult<()>;

    /// Set a live field of a layer, with no rebuild, to a value that
    /// live_value() checked against live_doc().
    fn set_layer(&self, set: LayerSet) -> ConsoleResult<()>;

    /// The scene document that a /set of a layer's field checks the field and
    /// the value against: the scene that renders.
    fn live_doc(&self) -> ConsoleResult<Value> {
        self.scene_def(None)
    }

    /// The type declared by the running scene for a param, or None when its
    /// scope holds no param of that name. A /set value parses by it.
    fn param_type(&self, scope: Scope, name: &str) -> ConsoleResult<Option<String>> {
        Ok(declared_type(&self.scene_def(None)?, scope, name))
    }

    /// Make name active and return the new active name.
    fn switch(&self, name: Option<&str>) -> ConsoleResult<String>;

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

    /// Run `edit`, a change to scene `scene` or to its values, while `scene`
    /// is the active scene, else refuse it as other_scene() does. A host whose
    /// switch() queues the switch holds one lock over its switch and over this
    /// check and the edit, so an edit queued behind a switch never lands on
    /// the scene that the switch brought.
    fn edit_scene(
        &self,
        scene: &str,
        edit: &mut dyn FnMut() -> ConsoleResult<()>,
    ) -> ConsoleResult<()> {
        other_scene(scene, self.active_name()?.as_deref())?;
        edit()
    }

    fn save_scene(&self, _name: &str, _scene: Scene) -> ConsoleResult<()> {
        Err(ConsoleError::read_only(self.save_refused()))
    }
}

/// Routes one transport-free request onto a ConsoleHost.
pub struct ConsoleAPI<H: ConsoleHost> {
    pub host: H,
    /// Serve the compiled-in web assets. false leaves that to the transport.
    pub web: bool,
    pub post_only: Vec<String>,
    pub max_body: usize,
}

impl<H: ConsoleHost> ConsoleAPI<H> {
    pub fn new(host: H) -> ConsoleAPI<H> {
        ConsoleAPI {
            host,
            web: false,
            post_only: POST_ONLY.iter().map(|p| (*p).to_string()).collect(),
            max_body: MAX_BODY,
        }
    }

    pub fn with_web(mut self, web: bool) -> ConsoleAPI<H> {
        self.web = web;
        self
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
                return Some(Response::text(
                    413,
                    &format!("body of {n} bytes, limit is {}", self.max_body),
                ));
            }
        }
        if method == "GET" && self.post_only.iter().any(|p| p == path) {
            return Some(Response::text(405, &format!("{path} is POST only")));
        }
        None
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
            "/stats" => Ok(Response::json(&h.stats()?)),
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
            "/scene" => Ok(Response::json(
                &h.scene_def(query.get("name").map(String::as_str))?,
            )),
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
        if path == "/set" {
            let mut set = || {
                let set = parse_set(h, query)?
                    .map_err(|e| ConsoleError::bad_request(format!("bad /set: {e}")))?;
                match set {
                    Set::Param(scope, name, value) => h.set_param(scope, &name, value),
                    Set::Layer(set) => h.set_layer(set),
                }
            };
            match query.get("scene") {
                Some(scene) => h.edit_scene(scene, &mut set)?,
                None => set()?,
            }
            return Ok(Response::text(200, "ok"));
        }
        if path == "/next" || path == "/prev" {
            return Ok(Response::json(&active_json(h.step(&path[1..])?)));
        }
        if path == "/switch" {
            let active = h.switch(query.get("name").map(String::as_str))?;
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
            match query.get("name") {
                Some(name) => h.edit_scene(name, &mut apply)?,
                None => apply()?,
            }
            return Ok(Response::json(&doc));
        }
        if path == "/save" {
            let d = json_body(headers, body, "/save")?;
            if !h.writable() {
                return Err(ConsoleError::read_only(h.save_refused()));
            }
            let name = d
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| ConsoleError::bad_request("bad /save: 'name'"))?
                .to_string();
            let mut scene = Some(scene_of(d.get("scene").unwrap_or(&Value::Null), "/save")?);
            h.edit_scene(&name, &mut || {
                h.save_scene(&name, scene.take().expect("one save a request"))
            })?;
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
        }
    }
}

/// What a /set changes, or the first thing wrong with it. A query that names
/// a `layer` sets one of its live fields, and its value parses by the field.
/// The `text` of a text layer is its words, the scene param of the layer's
/// name. Any other query sets a param, and its value parses by the type that
/// the host reports for the param. The outer error is the host's own.
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
        if effect.is_none() && field == "text" && text_layer(&doc, layer) {
            return param_set(host, Scope::Scene, layer.clone(), raw);
        }
        return Ok(live_value(&doc, &at, raw).map(|value| {
            Set::Layer(LayerSet {
                layer: layer.clone(),
                effect: effect.map(str::to_string),
                field: field.clone(),
                value,
            })
        }));
    }
    let parts = || -> Result<(Scope, String, &String), String> {
        let raw = query.get("value").ok_or("the query has no 'value'")?;
        let scope = parse_scope(query.get("pass").ok_or("the query has no 'pass'")?)?;
        let name = query.get("name").ok_or("the query has no 'name'")?;
        Ok((scope, name.clone(), raw))
    };
    let (scope, name, raw) = match parts() {
        Ok(p) => p,
        Err(e) => return Ok(Err(e)),
    };
    param_set(host, scope, name, raw)
}

/// The set of the param `name` of `scope` to `raw`, parsed by the type that
/// the host reports for the param.
fn param_set(
    host: &impl ConsoleHost,
    scope: Scope,
    name: String,
    raw: &str,
) -> ConsoleResult<Result<Set, String>> {
    let Some(ty) = host.param_type(scope, &name)? else {
        return Ok(Err(undeclared(scope, &name)));
    };
    Ok(parse_param_value(&ty, raw).map(|value| Set::Param(scope, name, value)))
}

/// True when the scene document `doc` holds a text layer named `name`.
fn text_layer(doc: &Value, name: &str) -> bool {
    let named = Value::from(name);
    doc_layers(doc).any(|l| {
        l.get("name") == Some(&named)
            && Layer::from_dict("a layer", l).is_ok_and(|l| l.as_text().is_some())
    })
}

/// The value of `raw`, a /set value, for the live field `at` of the scene
/// document `doc`, checked as the scene checks a live value, or why the scene
/// takes none.
pub fn live_value(doc: &Value, at: &Address, raw: &str) -> Result<Value, String> {
    let named = Value::from(at.layer);
    let Some(d) = doc_layers(doc).find(|l| l.get("name") == Some(&named)) else {
        return Err(format!("the scene has no layer '{}'", at.layer));
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
/// floats or a gradient object in JSON, the name of an image asset, a line's
/// points as a JSON list, or `none` for the default of a field that takes
/// none.
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
        Form::Image => Ok(Value::from(raw)),
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

fn undeclared(scope: Scope, name: &str) -> String {
    match scope {
        Scope::Scene => format!("the scene has no param '{name}'"),
        Scope::Pass(i) => format!("pass {i} has no param '{name}'"),
    }
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

/// The type of a param in a scene document, or None when the scope holds no
/// param of that name. ghee declares a scene string param under each text
/// layer's name, and none for an image layer, which holds an `image`, or a
/// shape layer, which holds a `shape`.
pub fn declared_type(doc: &Value, scope: Scope, name: &str) -> Option<String> {
    let text = |l: &Value| l.get("image").is_none() && l.get("shape").is_none();
    if scope == Scope::Scene
        && doc_layers(doc).any(|l| l.get("name") == Some(&Value::from(name)) && text(l))
    {
        return Some("string".to_string());
    }
    let params = match scope {
        Scope::Scene => doc.get("params"),
        Scope::Pass(i) => doc
            .get("passes")?
            .get(usize::try_from(i).ok()?)?
            .get("params"),
    };
    Some(params?.get(name)?.get("type")?.as_str()?.to_string())
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
/// unvalidated document.
fn scene_of(d: &Value, what: &str) -> ConsoleResult<Scene> {
    Scene::from_dict(d, None, None)
        .map_err(|e| ConsoleError::bad_request(format!("bad {what}: {e}")))
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
        SaveScene(String, Scene),
    }

    /// A host that records instead of rendering.
    struct FakeHost {
        names: RefCell<Vec<String>>,
        active: RefCell<String>,
        calls: RefCell<Vec<Call>>,
        writable: bool,
        boom: bool,
    }

    impl FakeHost {
        fn new() -> FakeHost {
            FakeHost {
                names: RefCell::new(["a", "b", "c"].iter().map(|s| (*s).to_string()).collect()),
                active: RefCell::new("a".to_string()),
                calls: RefCell::new(Vec::new()),
                writable: false,
                boom: false,
            }
        }
    }

    impl ConsoleHost for FakeHost {
        fn writable(&self) -> bool {
            self.writable
        }

        fn save_refused(&self) -> String {
            "nope".to_string()
        }

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

        fn save_scene(&self, name: &str, scene: Scene) -> ConsoleResult<()> {
            self.calls
                .borrow_mut()
                .push(Call::SaveScene(name.to_string(), scene));
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
    fn defaults_supply_the_registry_and_no_hwinfo() {
        assert_eq!(Bare.hwinfo().unwrap(), serde_json::json!({}));
        assert!(Bare.filters().unwrap().is_object()); // ghee's own filter registry
    }

    // writes

    #[test]
    fn set_routes_the_scene_scope() {
        let api = api();
        post(
            &api,
            "/set",
            &[("pass", "scene"), ("name", "tint"), ("value", "0.5")],
        );
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
        for (pass, name, raw, want) in [
            ("scene", "title", "hello", Value::from("hello")),
            ("scene", "title", "1.5", Value::from("1.5")),
            // a text's words
            ("scene", "words", "2", Value::from("2")),
            ("0", "on", "true", Value::from(true)),
            ("0", "on", "false", Value::from(false)),
            ("0", "k", "2", Value::from(2.0)),
            ("0", "k", "0.5", Value::from(0.5)),
            ("0", "n", "5", Value::from(5.0)),
            ("0", "c", "1,0,0.5", serde_json::json!([1.0, 0.0, 0.5])),
            ("0", "v", "0.1,0.2", serde_json::json!([0.1, 0.2])),
        ] {
            let api = api();
            let r = post(
                &api,
                "/set",
                &[("pass", pass), ("name", name), ("value", raw)],
            );
            assert_eq!((r.status, body(&r).as_str()), (200, "ok"), "{name}");
            let scope = parse_scope(pass).unwrap();
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
    fn a_text_layers_text_sets_the_words_param_of_its_name() {
        let api = api();
        let r = post(&api, "/set", &layer_query("words", None, "text", "la la"));
        assert_eq!((r.status, body(&r).as_str()), (200, "ok"));
        assert_eq!(
            api.host.calls.borrow().last(),
            Some(&Call::SetParam(
                Scope::Scene,
                "words".to_string(),
                Value::from("la la")
            ))
        );
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
                "the scene has no layer 'nope'".to_string(),
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
    fn a_set_of_an_undeclared_param_names_it() {
        for (pass, name, want) in [
            ("scene", "nope", "bad /set: the scene has no param 'nope'"),
            ("0", "nope", "bad /set: pass 0 has no param 'nope'"),
            ("3", "nope", "bad /set: pass 3 has no param 'nope'"),
            // an image layer has no words
            ("scene", "icon", "bad /set: the scene has no param 'icon'"),
        ] {
            let api = api();
            let r = post(
                &api,
                "/set",
                &[("pass", pass), ("name", name), ("value", "1")],
            );
            assert_eq!((r.status, body(&r).as_str()), (400, want));
            assert_eq!(*api.host.calls.borrow(), vec![Call::SceneDef(None)]);
        }
    }

    #[test]
    fn a_set_carries_the_host_error_of_its_type_lookup() {
        let mut host = FakeHost::new();
        host.boom = true;
        let api = ConsoleAPI::new(host);
        let r = post(
            &api,
            "/set",
            &[("pass", "0"), ("name", "k"), ("value", "1")],
        );
        assert_eq!((r.status, body(&r).as_str()), (500, "build x: kaboom"));
    }

    #[test]
    fn a_malformed_set_never_reaches_the_host() {
        for (q, why) in [
            (
                vec![("pass", "0"), ("name", "k")],
                "the query has no 'value'",
            ),
            (
                vec![("pass", "0"), ("value", "1")],
                "the query has no 'name'",
            ),
            (
                vec![("name", "k"), ("value", "1")],
                "the query has no 'pass'",
            ),
            (
                vec![("pass", "nope"), ("name", "k"), ("value", "1")],
                "pass needs 'scene' or a pass index, got 'nope'",
            ),
            (
                vec![("pass", "0"), ("name", "c"), ("value", "1,0")],
                "color needs '#rrggbb', a CSS colour name or three comma-separated floats, got \
                 '1,0'",
            ),
            (
                vec![("pass", "0"), ("name", "on"), ("value", "yes")],
                "bool needs true or false, got 'yes'",
            ),
            (
                vec![("pass", "0"), ("name", "k"), ("value", "notanumber")],
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
        for path in ["/scene", "/save"] {
            let api = api();
            let r = api.dispatch(
                "POST",
                path,
                &Query::new(),
                &Headers::new(),
                SCENE.as_bytes(),
            );
            assert_eq!(r.status, 415, "{path}");
            assert!(body(&r).contains("application/json"));
            assert!(api.host.calls.borrow().is_empty());
        }
    }

    #[test]
    fn save_is_refused_unless_the_host_opts_in() {
        let api = api();
        let posted = serde_json::json!({"name": "a", "scene": scene_json()}).to_string();
        let r = api.dispatch(
            "POST",
            "/save",
            &Query::new(),
            &json_ct(),
            posted.as_bytes(),
        );
        assert_eq!((r.status, body(&r).as_str()), (403, "nope"));
        assert!(api.host.calls.borrow().is_empty());
    }

    #[test]
    fn a_writable_host_gets_the_name_and_a_scene() {
        let mut api = api();
        api.host.writable = true;
        let posted = serde_json::json!({"name": "a", "scene": scene_json()}).to_string();
        let r = api.dispatch(
            "POST",
            "/save",
            &Query::new(),
            &json_ct(),
            posted.as_bytes(),
        );
        assert_eq!(r.status, 200);
        let calls = api.host.calls.borrow();
        match &calls[0] {
            Call::SaveScene(name, _) => assert_eq!(name, "a"),
            c => panic!("{c:?}"),
        }
    }

    #[test]
    fn a_save_without_a_scene_is_400() {
        let mut api = api();
        api.host.writable = true;
        let r = api.dispatch(
            "POST",
            "/save",
            &Query::new(),
            &json_ct(),
            br#"{"name": "a"}"#,
        );
        assert_eq!(r.status, 400);
        assert!(api.host.calls.borrow().is_empty());
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
    /// scene and a save.
    fn edits(api: &ConsoleAPI<FakeHost>, name: &str) -> [Response; 3] {
        let save = serde_json::json!({"name": name, "scene": scene_json()}).to_string();
        [
            post(
                api,
                "/set",
                &[
                    ("scene", name),
                    ("pass", "scene"),
                    ("name", "tint"),
                    ("value", "0.5"),
                ],
            ),
            api.dispatch(
                "POST",
                "/scene",
                &query(&[("name", name)]),
                &json_ct(),
                SCENE.as_bytes(),
            ),
            api.dispatch("POST", "/save", &Query::new(), &json_ct(), save.as_bytes()),
        ]
    }

    #[test]
    fn an_edit_for_a_scene_other_than_the_active_one_is_409_and_never_reaches_the_host() {
        let mut api = api();
        api.host.writable = true;
        for r in edits(&api, "b") {
            assert_eq!(
                (r.status, body(&r).as_str()),
                (
                    409,
                    "the console is on scene 'a', and this edit is for scene 'b'"
                )
            );
        }
        // A scene def is a read, and none of the three edits lands.
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
        assert_eq!(landed, 3);
    }

    // refusals: precheck decides them, dispatch enforces them again

    #[test]
    fn state_changing_paths_are_post_only() {
        let api = api();
        for path in ["/set", "/switch", "/next", "/prev"] {
            let r = get(&api, path, &[("pass", "0"), ("name", "k"), ("value", "1")]);
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
            &[("pass", "0"), ("name", "k"), ("value", "1")],
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
        assert!(body(&r).contains(&MAX_BODY.to_string()));
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
        assert_eq!(parse_query("a=1&b=x&a=2"), query(&[("a", "1"), ("b", "x")]));
        assert_eq!(parse_query(""), Query::new());
    }

    #[test]
    fn parse_scope_reads_a_pass_index() {
        assert_eq!(parse_scope("scene"), Ok(Scope::Scene));
        assert_eq!(parse_scope(" +3 "), Ok(Scope::Pass(3)));
        for raw in ["notanint", "3.0", ""] {
            assert_eq!(
                parse_scope(raw),
                Err(format!("pass needs 'scene' or a pass index, got '{raw}'"))
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
    fn a_step_from_an_unknown_active_starts_at_the_first_name() {
        let api = api();
        "gone".clone_into(&mut api.host.active.borrow_mut());
        assert_eq!(
            as_json(&post(&api, "/next", &[])),
            serde_json::json!({"active": "b"})
        );
    }
}
