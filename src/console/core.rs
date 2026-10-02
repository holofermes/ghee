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
//! the JSON content type and the static root check.
//!
//! An API holds no mutable state, so any number of threads may call it at once.
//! A host states its own thread contract for set_param, switch and apply_scene.

use std::collections::BTreeMap;
use std::fmt;

use serde_json::{Map, Value};

use crate::console::{static_asset, WEB};
use crate::filters::{filter_defs, rebuild_filters};
use crate::scene::{py_repr, Scene};

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
    text.parse::<f64>().map_err(|_| {
        format!(
            "could not convert string to float: {}",
            py_repr(&Value::from(raw))
        )
    })
}

/// Parse a decimal integer. Whitespace around it is fine.
fn py_int(raw: &str) -> Result<i64, String> {
    let s = raw.trim();
    let digits = s.strip_prefix(['+', '-']).unwrap_or(s);
    let decimal = !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit());
    match s.parse::<i64>() {
        Ok(v) if decimal => Ok(v),
        _ => Err(format!(
            "invalid literal for int() with base 10: {}",
            py_repr(&Value::from(raw))
        )),
    }
}

/// The value set_param expects, from a /set raw query string.
///
/// A "color" is "r,g,b" and gives three floats. A "vecK" is K comma-separated
/// floats. A "bool" is true or false (1 or 0), and the scene model demands a
/// real bool. Anything else is one float. Malformed input errs.
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
        let parts = floats(raw)?;
        if parts.len() != 3 {
            return Err(format!(
                "color needs three comma-separated floats, got {}",
                py_repr(&Value::from(raw))
            ));
        }
        return Ok(Value::from(parts));
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
    Ok(Scope::Pass(py_int(raw)?))
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
    /// error_draw, active, display, stale}.
    ///
    /// fps counts completed render-loop frames per elapsed second, over a
    /// trailing window (stats::FrameRate). frame_ms is one loop iteration, with
    /// no paced sleep in it. stream_fps counts the frames handed to viewers per
    /// elapsed second, every codec in one number. out names where the frames go
    /// now (stats::DEST_*). error_pass is the pass index that holds a compile
    /// error, or "common" for the scene's common block. error_draw is the draw
    /// of that pass that holds it, {index, part}, where part names the draw's
    /// "code" or its "place". error_line is the line inside the draw's part,
    /// else inside the pass. stale is true when the files behind the active
    /// scene changed since the compile. A host whose scenes have no files
    /// answers false for stale.
    ///
    /// A poll loop in the browser reads this, so it must not block.
    fn stats(&self) -> ConsoleResult<Value>;

    fn scene_names(&self) -> ConsoleResult<Vec<String>>;

    fn active_name(&self) -> ConsoleResult<Option<String>>;

    /// The resolved scene JSON for a name, or for the active scene when name is
    /// None. Resolved means shader files inlined and filters expanded.
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
            let (scope, name, value) = match parse_set(query) {
                Ok(parts) => parts,
                Err(e) => return Ok(Response::text(400, &format!("bad /set: {e}"))),
            };
            h.set_param(scope, &name, value)?;
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
            let mut scene = scene_of(&json_body(headers, body, "/scene")?, "/scene")?;
            rebuild_filters(&mut scene, None)
                .map_err(|e| ConsoleError::bad_request(format!("bad /scene: {e}")))?;
            // The scene as it will run, so the console shows each filter pass
            // as the filter built it.
            let doc = scene.to_dict();
            h.apply_scene(scene)?;
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
            let scene = scene_of(d.get("scene").unwrap_or(&Value::Null), "/save")?;
            h.save_scene(&name, scene)?;
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

/// The (scope, name, value) of a /set, or the first thing wrong with it.
fn parse_set(query: &Query) -> Result<(Scope, String, Value), String> {
    let raw = query.get("value").ok_or("'value'")?;
    let value = parse_param_value(query.get("type").map_or("float", String::as_str), raw)?;
    let scope = parse_scope(query.get("pass").ok_or("'pass'")?)?;
    let name = query.get("name").ok_or("'name'")?;
    Ok((scope, name.clone(), value))
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
        "passes": [{"name": "main", "code": "void main(){ gl_FragColor = vec4(1.0); }"}]}"#;

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
            Ok(serde_json::json!({"dither": {"code": "x"}}))
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
    fn scenes_lists_names_and_active() {
        assert_eq!(
            as_json(&get(&api(), "/scenes", &[])),
            serde_json::json!({"scenes": ["a", "b", "c"], "active": "a"})
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
            serde_json::json!({"dither": {"code": "x"}})
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
    fn set_reaches_the_host_with_a_typed_value() {
        for (ptype, raw, want) in [
            ("float", "0.5", Value::from(0.5)),
            ("int", "5", Value::from(5.0)),
            ("color", "1,0,0.5", serde_json::json!([1.0, 0.0, 0.5])),
            ("vec2", "0.1,0.2", serde_json::json!([0.1, 0.2])),
            ("string", "GHEE", Value::from("GHEE")),
            ("bool", "true", Value::from(true)),
            ("bool", "false", Value::from(false)),
        ] {
            let api = api();
            let r = post(
                &api,
                "/set",
                &[
                    ("pass", "0"),
                    ("name", "k"),
                    ("type", ptype),
                    ("value", raw),
                ],
            );
            assert_eq!((r.status, body(&r).as_str()), (200, "ok"), "{ptype}");
            assert_eq!(
                *api.host.calls.borrow(),
                vec![Call::SetParam(Scope::Pass(0), "k".to_string(), want)],
                "{ptype}"
            );
        }
    }

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
            vec![Call::SetParam(
                Scope::Scene,
                "tint".to_string(),
                Value::from(0.5)
            )]
        );
    }

    #[test]
    fn a_malformed_set_never_reaches_the_host() {
        for q in [
            vec![("pass", "0"), ("name", "k")],  // no value
            vec![("pass", "0"), ("value", "1")], // no name
            vec![("name", "k"), ("value", "1")], // no pass
            vec![("pass", "nope"), ("name", "k"), ("value", "1")],
            vec![
                ("pass", "0"),
                ("name", "k"),
                ("type", "color"),
                ("value", "1,0"),
            ],
            vec![
                ("pass", "0"),
                ("name", "k"),
                ("type", "bool"),
                ("value", "yes"),
            ],
            vec![("pass", "0"), ("name", "k"), ("value", "notanumber")],
        ] {
            let api = api();
            let r = post(&api, "/set", &q);
            assert_eq!(r.status, 400, "{q:?}");
            assert!(body(&r).contains("bad /set"), "{}", body(&r));
            assert!(api.host.calls.borrow().is_empty());
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
    fn a_scene_naming_a_shader_file_is_400() {
        // A posted scene has no base directory. A "shader" path reads a file off
        // disk, so the model refuses it and the API passes that on.
        let api = api();
        let r = api.dispatch(
            "POST",
            "/scene",
            &Query::new(),
            &json_ct(),
            br#"{"passes": [{"name": "main", "shader": "/etc/hostname"}]}"#,
        );
        assert_eq!(r.status, 400);
        assert!(body(&r).contains("shader"), "{}", body(&r));
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
    fn a_host_declaring_its_own_get_routes_still_mutates_through_post() {
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
            vec![Call::SetParam(
                Scope::Pass(0),
                "k".to_string(),
                Value::from(1.0)
            )]
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
        assert_eq!(parse_scope("3"), Ok(Scope::Pass(3)));
        assert!(parse_scope("notanint").is_err());
    }

    #[test]
    fn bool_parses_both_spellings_and_refuses_anything_else() {
        for raw in ["true", "TRUE", " true ", "1"] {
            assert_eq!(
                parse_param_value("bool", raw),
                Ok(Value::from(true)),
                "{raw}"
            );
        }
        for raw in ["false", "False", "0"] {
            assert_eq!(
                parse_param_value("bool", raw),
                Ok(Value::from(false)),
                "{raw}"
            );
        }
        for raw in ["yes", "", "2", "0.0"] {
            assert!(parse_param_value("bool", raw).is_err(), "{raw}");
        }
    }

    // console request helpers

    #[test]
    fn parse_float() {
        assert_eq!(parse_param_value("float", "0.5"), Ok(Value::from(0.5)));
        assert_eq!(parse_param_value("float", "3"), Ok(Value::from(3.0)));
    }

    #[test]
    fn parse_int_rides_the_float_path() {
        // An int param parses as a float here. The render thread rounds it with
        // the scene's own param type, so the wire value stays a plain number.
        assert_eq!(parse_param_value("int", "5"), Ok(Value::from(5.0)));
    }

    #[test]
    fn parse_color() {
        assert_eq!(
            parse_param_value("color", "1,0,0.5"),
            Ok(serde_json::json!([1.0, 0.0, 0.5]))
        );
    }

    #[test]
    fn parse_color_wrong_length() {
        assert!(parse_param_value("color", "1,0").is_err());
        assert!(parse_param_value("color", "1,0,0,0").is_err());
    }

    #[test]
    fn parse_rejects_non_numbers() {
        assert!(parse_param_value("float", "notanumber").is_err());
        assert!(parse_param_value("color", "1,x,0").is_err());
    }

    #[test]
    fn parse_vec() {
        assert_eq!(
            parse_param_value("vec2", "0.1,0.2"),
            Ok(serde_json::json!([0.1, 0.2]))
        );
        assert_eq!(
            parse_param_value("vec3", "1,0,0.5"),
            Ok(serde_json::json!([1.0, 0.0, 0.5]))
        );
        assert_eq!(
            parse_param_value("vec4", "0,0.25,0.5,1"),
            Ok(serde_json::json!([0.0, 0.25, 0.5, 1.0]))
        );
    }

    #[test]
    fn parse_vec_wrong_length() {
        for (ty, raw) in [
            ("vec2", "0.1"),
            ("vec2", "0.1,0.2,0.3"),
            ("vec3", "0.1,0.2"),
            ("vec4", "0.1,0.2,0.3"),
        ] {
            assert!(parse_param_value(ty, raw).is_err(), "{ty} {raw}");
        }
    }

    #[test]
    fn parse_vec_rejects_non_numbers() {
        assert!(parse_param_value("vec3", "0.1,x,0.3").is_err());
    }

    #[test]
    fn parse_string() {
        assert_eq!(parse_param_value("string", "GHEE"), Ok(Value::from("GHEE")));
        assert_eq!(
            parse_param_value("string", "hello world"),
            Ok(Value::from("hello world"))
        );
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
