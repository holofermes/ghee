//! The ghee console over a session the host drives.
//!
//! The console API is ghee's own (console::core). This module is the transport
//! and a read-only host: it serves the scene, the stats and the preview, and it
//! sets live params.
//!
//! No request thread ever touches the engine. A /set lands on a queue that the
//! next frame drains, and a viewer reads the last published frame. The host
//! owns the render, so this console switches no scene and compiles none.
//!
//! /h264 is the CLI console's stream: one encoder feeds every viewer, and it
//! runs only while a viewer watches. The hardware lane issues GL calls, so the
//! session drives it on the host's thread (session::Session::mirror). The
//! software lane runs here, on the published frames.

use std::borrow::Cow;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{anyhow, Result};
use ghee::console::core::{
    parse_query, ConsoleAPI, ConsoleError, ConsoleHost, ConsoleResult, Headers, LayerSet, Query,
    Response, Scope,
};
use ghee::console::server::{
    coerce, serve_h264, stream_bitrate, stream_size, H264Fanout, Heartbeat, Options, H264_FPS,
    VIEWER_DEADLINE,
};
use ghee::h264::{no_encoder, ColorSpace, SoftwareH264, SW_COLOR};
use ghee::looper::Frame;
use ghee::output::{encode_jpeg, encode_png};
use ghee::scene::{ParamValue, Scene};
use ghee::stats::{monotonic, FrameRate, DEST_H264};
use serde_json::{Map, Value};
use tiny_http::{Header, Request, Server, StatusCode};

use crate::session::Index;

/// JPEG quality for the preview, as the ghee console encodes it.
const PREVIEW_QUALITY: u8 = 75;

/// How long GET /frame.png waits for the host to draw.
const FRAME_TIMEOUT: f64 = 4.0;

/// A bare port binds to this address. The console has no auth, so it stays on
/// loopback until a host names another address.
pub const LOOPBACK: &str = "127.0.0.1";

/// What the encoder takes: the host's stream cap, never past the render size.
/// yuv420 takes even dimensions only. Both consoles answer /h264 with the same
/// size and the same bitrate.
pub fn encode_size(w: i32, h: i32, longest: i32) -> (i32, i32) {
    let (sw, sh) = stream_size(w, h, longest);
    (sw.min(w & !1), sh.min(h & !1))
}

/// The CLI console's bitrate for this size at this pace.
pub fn encode_bitrate(w: i32, h: i32, fps: i32) -> i32 {
    stream_bitrate(w, h, fps, Options::default().bitrate) as i32
}

/// A param queued by a request for the next frame.
pub type Queued = (Option<String>, String, String, ParamValue);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The last published frame, and the number that says it is new.
#[derive(Default)]
struct Snap {
    data: Vec<u8>,
    width: i32,
    height: i32,
    components: i32,
    seq: u64,
}

/// What the render thread publishes and the request threads read.
pub struct Shared {
    index: Arc<Index>,
    name: String,
    scene: Value,
    display: Value,
    out: Vec<String>,
    queue: Mutex<Vec<Queued>>,
    /// The live fields of layers queued by requests for the next frame.
    layers: Mutex<Vec<LayerSet>>,
    rate: Mutex<FrameRate>,
    frame_ms: Mutex<f64>,
    /// The renderer's text_errors(), a line each.
    texts: Mutex<Vec<String>>,
    drawn: AtomicU64,
    skipped: AtomicU64,
    latest: Mutex<Snap>,
    ready: Condvar,
    viewers: AtomicI64,
    shots: AtomicI64,
    /// One shared H264 stream for every /h264 viewer.
    fanout: H264Fanout,
    /// The /h264 viewers that report in, so a page in a back/forward cache
    /// stops the encoder.
    watchers: Heartbeat,
    /// The stream whose hardware encoding failed, which the software lane
    /// owns. The next stream tries hardware again.
    hw_dead: Mutex<Option<u64>>,
    /// Why the hardware lane failed this stream.
    hw_why: Mutex<String>,
    /// The h264 cap for the longest side. 0 streams the render size.
    stream_longest: i32,
    verbose: bool,
    stopping: AtomicBool,
}

impl Shared {
    fn stopping(&self) -> bool {
        self.stopping.load(Ordering::SeqCst)
    }

    /// The software lane owns this stream, so it takes the published frames.
    /// Hardware samples the GPU on the host's thread instead.
    fn software_h264(&self) -> bool {
        self.software_stream().is_some()
    }

    /// The stream that the software lane owns, while a viewer watches it.
    fn software_stream(&self) -> Option<u64> {
        let stream = self.fanout.stream();
        let dead = *lock(&self.hw_dead) == Some(stream);
        (dead && self.fanout.nclients() > 0).then_some(stream)
    }

    /// The latest frame new to the caller, once one lands.
    fn next_frame(&self, seen: u64, timeout: f64) -> Option<(Vec<u8>, i32, i32, i32, u64)> {
        let deadline = monotonic() + timeout;
        let mut latest = lock(&self.latest);
        while latest.seq == seen && !self.stopping() {
            let left = deadline - monotonic();
            if left <= 0.0 {
                return None;
            }
            latest = self
                .ready
                .wait_timeout(latest, Duration::from_secs_f64(left.min(1.0)))
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        if latest.seq == seen {
            return None;
        }
        Some((
            latest.data.clone(),
            latest.width,
            latest.height,
            latest.components,
            latest.seq,
        ))
    }
}

/// The console API runs on this read-only host.
struct Host(Arc<Shared>);

impl ConsoleHost for Host {
    fn stats(&self) -> ConsoleResult<Value> {
        let s = &self.0;
        let mut d = Map::new();
        d.insert("fps".to_string(), Value::from(lock(&s.rate).value(None)));
        d.insert("frame_ms".to_string(), Value::from(*lock(&s.frame_ms)));
        d.insert("stream_fps".to_string(), Value::from(s.fanout.stream_fps()));
        let mut out = s.out.clone();
        if s.fanout.nclients() > 0 {
            out.push(DEST_H264.to_string());
        }
        d.insert("out".to_string(), Value::from(out));
        d.insert("error".to_string(), Value::from(""));
        d.insert(
            "text_errors".to_string(),
            Value::from(lock(&s.texts).clone()),
        );
        d.insert("error_line".to_string(), Value::Null);
        d.insert("error_pass".to_string(), Value::Null);
        d.insert("active".to_string(), Value::from(s.name.clone()));
        d.insert("display".to_string(), s.display.clone());
        d.insert("stale".to_string(), Value::from(false));
        d.insert(
            "frames".to_string(),
            Value::from(s.drawn.load(Ordering::SeqCst)),
        );
        d.insert(
            "skipped".to_string(),
            Value::from(s.skipped.load(Ordering::SeqCst)),
        );
        Ok(Value::Object(d))
    }

    fn scene_names(&self) -> ConsoleResult<Vec<String>> {
        Ok(vec![self.0.name.clone()])
    }

    fn active_name(&self) -> ConsoleResult<Option<String>> {
        Ok(Some(self.0.name.clone()))
    }

    fn scene_def(&self, name: Option<&str>) -> ConsoleResult<Value> {
        match name {
            Some(n) if n != self.0.name => Err(ConsoleError::not_found(format!("no scene '{n}'"))),
            _ => Ok(self.0.scene.clone()),
        }
    }

    fn set_param(&self, scope: Scope, name: &str, value: Value) -> ConsoleResult<()> {
        let pass = match scope {
            Scope::Scene => None,
            Scope::Pass(i) => Some(
                self.0
                    .index
                    .passes
                    .get(usize::try_from(i).unwrap_or(usize::MAX))
                    .ok_or_else(|| ConsoleError::bad_request(format!("no pass {i}")))?
                    .as_str(),
            ),
        };
        let p = self
            .0
            .index
            .param(pass, name)
            .map_err(|e| ConsoleError::bad_request(e.message))?;
        let coerced =
            coerce(&p.wire, &p.ty, &value).map_err(|e| ConsoleError::bad_request(e.to_string()))?;
        lock(&self.0.queue).push((p.pass.clone(), p.uniform.clone(), p.ty.clone(), coerced));
        Ok(())
    }

    fn set_layer(&self, set: LayerSet) -> ConsoleResult<()> {
        lock(&self.0.layers).push(set);
        Ok(())
    }

    fn switch(&self, _name: Option<&str>) -> ConsoleResult<String> {
        Err(ConsoleError::read_only(
            "this console renders one scene, its host's own",
        ))
    }

    fn apply_scene(&self, _scene: Scene) -> ConsoleResult<()> {
        Err(ConsoleError::read_only(
            "this console compiles no scene; its host owns the render",
        ))
    }

    fn save_refused(&self) -> String {
        "this console saves no scene".to_string()
    }
}

pub struct Console {
    shared: Arc<Shared>,
    server: Arc<Server>,
    accepting: Option<JoinHandle<()>>,
    encoding: Option<JoinHandle<()>>,
}

impl Console {
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        host: &str,
        port: u16,
        name: &str,
        scene: &Scene,
        index: Arc<Index>,
        size: (i32, i32),
        format: &str,
        out: Vec<String>,
        stream_longest: i32,
        verbose: bool,
    ) -> Result<Console> {
        let shared = Arc::new(Shared {
            index,
            name: name.to_string(),
            scene: scene.to_dict(),
            display: serde_json::json!({
                "w": size.0, "h": size.1, "format": format, "rotation": 0
            }),
            out,
            queue: Mutex::new(Vec::new()),
            layers: Mutex::new(Vec::new()),
            rate: Mutex::new(FrameRate::new()),
            frame_ms: Mutex::new(0.0),
            texts: Mutex::new(Vec::new()),
            drawn: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
            latest: Mutex::new(Snap::default()),
            ready: Condvar::new(),
            viewers: AtomicI64::new(0),
            shots: AtomicI64::new(0),
            fanout: H264Fanout::new(Arc::new(Mutex::new(FrameRate::new()))),
            watchers: Heartbeat::new(VIEWER_DEADLINE),
            hw_dead: Mutex::new(None),
            hw_why: Mutex::new(String::new()),
            stream_longest,
            verbose,
            stopping: AtomicBool::new(false),
        });
        let server = Arc::new(Server::http((host, port)).map_err(|e| anyhow!("{e}"))?);
        let accepting = {
            let (shared, server) = (Arc::clone(&shared), Arc::clone(&server));
            let api = Arc::new(ConsoleAPI::new(Host(Arc::clone(&shared))).with_web(true));
            thread::spawn(move || {
                for request in server.incoming_requests() {
                    let (shared, api) = (Arc::clone(&shared), Arc::clone(&api));
                    thread::spawn(move || handle(&shared, &api, request));
                }
            })
        };
        let encoding = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || h264_sw_loop(&shared))
        };
        if verbose {
            eprintln!("ghee console: scene '{name}' on {host}:{port}");
        }
        Ok(Console {
            shared,
            server,
            accepting: Some(accepting),
            encoding: Some(encoding),
        })
    }

    /// True when a viewer or a still is waiting for a frame. Nothing reads the
    /// frame back while nobody watches. A software H264 viewer counts, because
    /// that lane encodes the readback. Hardware samples the GPU instead.
    pub fn wants_frame(&self) -> bool {
        let s = &self.shared;
        s.viewers.load(Ordering::SeqCst) > 0
            || s.shots.load(Ordering::SeqCst) > 0
            || s.software_h264()
    }

    /// True when any console reader needs the host to draw. The hardware
    /// H264 lane counts here and not in wants_frame: it encodes on the GPU
    /// after a draw, with no readback to wait for.
    pub fn wants_draw(&self) -> bool {
        self.wants_frame() || self.shared.fanout.nclients() > 0
    }

    /// True while the hardware lane owns a /h264 stream.
    pub fn hardware_h264(&self) -> bool {
        let s = &self.shared;
        if s.fanout.nclients() == 0 {
            return false;
        }
        *lock(&s.hw_dead) != Some(s.fanout.stream()) && s.fanout.failure().is_none()
    }

    /// The hardware lane failed. Before the stream's first access unit the
    /// software lane takes the stream, and after it the stream ends.
    pub fn hardware_failed(&self, why: &str) {
        let s = &self.shared;
        if s.fanout.began() {
            if s.verbose {
                eprintln!("ghee console: h264: {why}");
            }
            s.fanout.fail(why);
            return;
        }
        *lock(&s.hw_why) = why.to_string();
        *lock(&s.hw_dead) = Some(s.fanout.stream());
        if s.verbose {
            eprintln!("ghee console: h264: {why}, using software encoder");
        }
    }

    /// The size and the colorSpace hint of the host's encoder.
    pub fn h264_opened(&self, w: i32, h: i32, color: ColorSpace) {
        self.shared.fanout.opened(w, h, color);
    }

    /// True once per keyframe request. A viewer joins on a keyframe.
    pub fn wants_key(&self) -> bool {
        self.shared.fanout.wants_key()
    }

    /// One access unit for every viewer.
    pub fn push_au(&self, au: &[u8]) {
        self.shared.fanout.push(au);
    }

    pub fn frame_done(&self, cost: f64) {
        let s = &self.shared;
        lock(&s.rate).tick(None);
        *lock(&s.frame_ms) = cost * 1000.0;
        s.drawn.fetch_add(1, Ordering::SeqCst);
    }

    /// The problems of the scene's texts, a line each, for /stats.
    pub fn text_errors(&self, lines: &[String]) {
        *lock(&self.shared.texts) = lines.to_vec();
    }

    pub fn skipped(&self) {
        self.shared.skipped.fetch_add(1, Ordering::SeqCst);
    }

    pub fn publish(&self, data: &[u8], width: i32, height: i32, components: i32) {
        let mut latest = lock(&self.shared.latest);
        latest.data.clear();
        latest.data.extend_from_slice(data);
        latest.width = width;
        latest.height = height;
        latest.components = components;
        latest.seq += 1;
        drop(latest);
        self.shared.ready.notify_all();
    }

    pub fn take_params(&self) -> Vec<Queued> {
        std::mem::take(&mut lock(&self.shared.queue))
    }

    pub fn take_layers(&self) -> Vec<LayerSet> {
        std::mem::take(&mut lock(&self.shared.layers))
    }

    pub fn stop(mut self) {
        self.shared.stopping.store(true, Ordering::SeqCst);
        self.shared.ready.notify_all();
        self.server.unblock();
        for t in [self.accepting.take(), self.encoding.take()]
            .into_iter()
            .flatten()
        {
            let _ = t.join();
        }
    }
}

/// Software H264 for the /h264 viewers, on a host with no hardware encoder. It
/// encodes the published frames, off the host's thread.
fn h264_sw_loop(shared: &Arc<Shared>) {
    let mut enc: Option<SoftwareH264> = None;
    let mut size = (0, 0);
    // the stream that the encoder encodes for
    let mut owned = 0;
    let mut seen = 0;
    while !shared.stopping() {
        let stream = shared
            .software_stream()
            .filter(|_| shared.fanout.failure().is_none());
        let Some(stream) = stream else {
            if let Some(mut e) = enc.take() {
                let _ = e.close(Duration::from_secs(1));
            }
            let latest = lock(&shared.latest);
            let _ = shared
                .ready
                .wait_timeout(latest, Duration::from_millis(250));
            continue;
        };
        let Some((data, w, h, components, seq)) = shared.next_frame(seen, 0.25) else {
            continue;
        };
        seen = seq;
        let (ew, eh) = encode_size(w, h, shared.stream_longest);
        if ew < 2 || eh < 2 {
            continue; // under 2 px a side there is nothing even to encode
        }
        if enc.is_some() && (size != (ew, eh) || owned != stream) {
            if let Some(mut e) = enc.take() {
                let _ = e.close(Duration::from_secs(1));
            }
        }
        if enc.is_none() {
            match SoftwareH264::new(
                ew,
                eh,
                H264_FPS,
                encode_bitrate(ew, eh, H264_FPS),
                components,
            ) {
                Ok(e) => {
                    enc = Some(e);
                    size = (ew, eh);
                    owned = stream;
                    shared.fanout.opened(ew, eh, SW_COLOR);
                }
                Err(e) => {
                    let why = no_encoder(&lock(&shared.hw_why), &e.to_string());
                    if shared.verbose {
                        eprintln!("ghee console: h264: {why}");
                    }
                    shared.fanout.fail(&why);
                    continue;
                }
            }
        }
        let e = enc.as_mut().expect("the encoder was just created");
        if let Err(err) = e.feed(&crop(&data, w, h, components, (ew, eh))) {
            let why = format!("software encoder died: {err}");
            if shared.verbose {
                eprintln!("ghee console: h264: {why}");
            }
            shared.fanout.fail(&why);
            if let Some(mut e) = enc.take() {
                let _ = e.close(Duration::from_secs(1));
            }
            continue;
        }
        if let (Some(au), _) = e.get_au(Duration::from_secs(1)) {
            if !shared.fanout.push_to(owned, &au) {
                if let Some(mut e) = enc.take() {
                    let _ = e.close(Duration::from_secs(1));
                }
            }
        }
    }
    if let Some(mut e) = enc.take() {
        let _ = e.close(Duration::from_secs(1));
    }
}

/// The frame at the encoded size. An odd render loses its last row or column,
/// because yuv420 takes even dimensions only.
fn crop(data: &[u8], w: i32, h: i32, c: i32, to: (i32, i32)) -> Cow<'_, [u8]> {
    if to == (w, h) {
        return Cow::Borrowed(data);
    }
    let (stride, keep) = ((w * c) as usize, (to.0 * c) as usize);
    Cow::Owned(
        data.chunks(stride)
            .take(to.1 as usize)
            .flat_map(|row| &row[..keep])
            .copied()
            .collect(),
    )
}

fn headers_of(request: &Request) -> Headers {
    let mut headers = Headers::new();
    for h in request.headers() {
        headers.insert(h.field.as_str().as_str(), h.value.as_str());
    }
    headers
}

fn send(request: Request, resp: &Response) {
    let mut out =
        tiny_http::Response::from_data(resp.body.clone()).with_status_code(StatusCode(resp.status));
    if let Ok(h) = Header::from_bytes(&b"Content-Type"[..], resp.content_type.as_bytes()) {
        out = out.with_header(h);
    }
    let _ = request.respond(out);
}

fn handle(shared: &Arc<Shared>, api: &ConsoleAPI<Host>, mut request: Request) {
    let method = request.method().as_str().to_string();
    let url = request.url().to_string();
    let (path, qs) = url.split_once('?').unwrap_or((url.as_str(), ""));
    let headers = headers_of(&request);
    if let Some(resp) = api.precheck(&method, path, &headers) {
        return send(request, &resp);
    }
    let query: Query = parse_query(qs);
    if method == "GET" {
        match path {
            "/stream" => return stream(shared, request),
            "/h264" => return h264(shared, request, &query),
            "/frame.png" => return frame_png(shared, request),
            _ => {}
        }
        return send(request, &api.dispatch("GET", path, &query, &headers, b""));
    }
    let mut body: Vec<u8> = Vec::new();
    if method == "POST" {
        let _ = request.as_reader().read_to_end(&mut body);
        if path == "/ping" {
            shared.watchers.ping(query.get("vid").map(String::as_str));
            return send(request, &Response::text(200, "ok"));
        }
    }
    let resp = api.dispatch(&method, path, &query, &headers, &body);
    send(request, &resp);
}

/// One JPEG part per published frame.
fn stream(shared: &Arc<Shared>, request: Request) {
    let mut w = request.into_writer();
    let head = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; \
                boundary=frame\r\nCache-Control: no-cache\r\n\r\n";
    if w.write_all(head.as_bytes()).is_err() || w.flush().is_err() {
        return;
    }
    shared.viewers.fetch_add(1, Ordering::SeqCst);
    let mut seen = 0;
    while !shared.stopping() {
        let Some((data, width, height, components, seq)) = shared.next_frame(seen, 1.0) else {
            continue;
        };
        seen = seq;
        let Ok(jpg) = encode_jpeg(
            Frame {
                data: &data,
                width,
                height,
                components,
            },
            PREVIEW_QUALITY,
        ) else {
            break;
        };
        let part = format!(
            "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
            jpg.len()
        );
        if w.write_all(part.as_bytes()).is_err()
            || w.write_all(&jpg).is_err()
            || w.write_all(b"\r\n").is_err()
            || w.flush().is_err()
        {
            break;
        }
    }
    shared.viewers.fetch_sub(1, Ordering::SeqCst);
}

/// Framed H264 for WebCodecs, docs/http.md holds the wire. The encoder runs
/// while a viewer holds this request open.
fn h264(shared: &Arc<Shared>, request: Request, query: &Query) {
    let c = shared.fanout.add();
    let viewer = shared.watchers.join(query.get("vid").map(String::as_str));
    let alive = || !shared.stopping() && !shared.watchers.gone(&viewer);
    serve_h264(request, &shared.fanout, &c, &alive);
    shared.fanout.remove(&c);
    shared.watchers.leave(&viewer);
}

/// One lossless PNG of the next drawn frame.
fn frame_png(shared: &Arc<Shared>, request: Request) {
    let seen = lock(&shared.latest).seq;
    shared.shots.fetch_add(1, Ordering::SeqCst);
    let frame = shared.next_frame(seen, FRAME_TIMEOUT);
    shared.shots.fetch_sub(1, Ordering::SeqCst);
    let png = frame.map(|(data, width, height, components, _)| {
        encode_png(Frame {
            data: &data,
            width,
            height,
            components,
        })
    });
    let png = match png {
        Some(Ok(png)) => png,
        Some(Err(e)) => {
            return send(request, &Response::text(503, &format!("no frame: {e}")));
        }
        None => {
            return send(
                request,
                &Response::text(503, &format!("no frame drawn in {FRAME_TIMEOUT}s")),
            );
        }
    };
    let mut out = tiny_http::Response::from_data(png).with_status_code(StatusCode(200));
    for (name, value) in [("Content-Type", "image/png"), ("Cache-Control", "no-store")] {
        if let Ok(h) = Header::from_bytes(name.as_bytes(), value.as_bytes()) {
            out = out.with_header(h);
        }
    }
    let _ = request.respond(out);
}
