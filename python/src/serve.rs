//! The ghee console over a session that its program drives.
//!
//! The console API is ghee's own (console::core), and so is everything that a
//! viewer reads: the preview, the streams, the still and /stats
//! (console::hub). This module is the console's host over a session: it
//! serves the session's scene, sets live params and layer keys, and an apply
//! gives layers a new look.
//!
//! No request thread ever touches the engine. A /set or an apply lands on a
//! queue that the session drains before its next frame. The program owns the
//! render: which layers exist and their order, the passes, the params and the
//! assets. So this console switches no scene, compiles none, and lists its
//! scene as one that a program writes.

use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use anyhow::{anyhow, Result};
use ghee::console::core::{ConsoleAPI, ConsoleError, ConsoleHost, ConsoleResult, LayerSet, Scope};
use ghee::console::hub::{
    banner, bound_port, serve, spawn_encoders, urls, Drawer, Hub, SceneStats,
};
use ghee::console::server::leave_out;
use ghee::console::state;
use ghee::engine::Renderer;
use ghee::live::{Index, Live};
use ghee::procedural::locks::Locks;
use ghee::scene::{coerce, ParamValue, Scene};
use serde_json::Value;
use tiny_http::Server;

/// A bare port binds to this address. The console has no auth, so it stays on
/// loopback until a host names another address.
pub const LOOPBACK: &str = "127.0.0.1";

/// A param queued by a request for the next frame.
pub type Queued = (Option<String>, String, String, ParamValue);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What the session publishes and the request threads read.
pub struct Shared {
    hub: Hub,
    index: Mutex<Arc<Index>>,
    name: String,
    /// The new looks of layers, from applies, for the next frame.
    looks: Mutex<Vec<Value>>,
    queue: Mutex<Vec<Queued>>,
    /// The layer keys queued by requests for the next frame.
    layers: Mutex<Vec<LayerSet>>,
    /// What the host set, which this console refuses to set.
    locks: Arc<Mutex<Locks>>,
}

/// The session draws when its program asks, and arms a still when it drains
/// the console's queues.
impl Drawer for Shared {
    fn hub(&self) -> &Hub {
        &self.hub
    }
}

/// The console API runs on this host.
struct Host(Arc<Shared>);

impl ConsoleHost for Host {
    fn stats(&self) -> ConsoleResult<Value> {
        let s = &self.0;
        Ok(s.hub.stats(SceneStats {
            active: Some(s.name.clone()),
            stale: false,
            locked: lock(&s.locks).to_json(),
            rev: None,
        }))
    }

    fn scene_names(&self) -> ConsoleResult<Vec<String>> {
        Ok(vec![self.0.name.clone()])
    }

    /// The host keeps which layers its scene has and their order.
    fn procedural_names(&self) -> ConsoleResult<Vec<String>> {
        Ok(vec![self.0.name.clone()])
    }

    fn active_name(&self) -> ConsoleResult<Option<String>> {
        Ok(Some(self.0.name.clone()))
    }

    fn scene_def(&self, name: Option<&str>) -> ConsoleResult<Value> {
        match name {
            Some(n) if n != self.0.name => Err(ConsoleError::not_found(format!("no scene '{n}'"))),
            _ => Ok(self.0.hub.scene().unwrap_or_default()),
        }
    }

    fn set_param(&self, scope: Scope, name: &str, value: Value) -> ConsoleResult<()> {
        let index = Arc::clone(&lock(&self.0.index));
        let pass = match scope {
            Scope::Scene => None,
            Scope::Pass(i) => Some(
                index
                    .passes
                    .get(usize::try_from(i).unwrap_or(usize::MAX))
                    .ok_or_else(|| ConsoleError::bad_request(format!("no pass {i}")))?
                    .as_str(),
            ),
        };
        let p = index
            .param(pass, name)
            .map_err(|e| ConsoleError::bad_request(e.0))?;
        if let Some(why) = lock(&self.0.locks).param_refusal(p.pass.as_deref(), &p.uniform) {
            return Err(ConsoleError::read_only(why));
        }
        let coerced =
            coerce(&p.wire, &p.ty, &value).map_err(|e| ConsoleError::bad_request(e.to_string()))?;
        lock(&self.0.queue).push((p.pass.clone(), p.uniform.clone(), p.ty.clone(), coerced));
        self.0.hub.shot.touch();
        Ok(())
    }

    fn set_layer(&self, set: LayerSet) -> ConsoleResult<()> {
        if let Some(why) = lock(&self.0.locks).field_refusal(&set.at()) {
            return Err(ConsoleError::read_only(why));
        }
        lock(&self.0.layers).push(set);
        self.0.hub.shot.touch();
        Ok(())
    }

    fn switch(&self, _name: Option<&str>) -> ConsoleResult<String> {
        Err(ConsoleError::read_only(
            "this console renders one scene, its host's own",
        ))
    }

    /// An apply gives each layer of the scene the look that `scene` gives
    /// it, and changes nothing else: a layer that `scene` leaves out keeps
    /// its look, and one that the scene lacks is skipped. What the host set
    /// keeps the host's values.
    fn apply_scene(&self, scene: Scene) -> ConsoleResult<()> {
        let posted = scene.to_dict();
        let running = self.0.hub.scene().unwrap_or_default();
        if frame_of(&posted) != frame_of(&running) {
            return Err(ConsoleError::read_only(
                "this console gives layers a new look and changes nothing else; its host \
                 owns the passes, the params and the assets",
            ));
        }
        let mut changed = state::diff(&running, &posted, true);
        leave_out(&mut changed.values, &lock(&self.0.locks));
        let posted = state::fold(&running, &running, &changed);
        let now: Vec<&Value> = layers_of(&running).collect();
        let looks: Vec<Value> = layers_of(&posted)
            .filter(|l| {
                let name = l.get("name");
                now.iter().any(|n| n.get("name") == name && *n != *l)
            })
            .cloned()
            .collect();
        lock(&self.0.looks).extend(looks);
        self.0.hub.shot.touch();
        Ok(())
    }

    fn save_refused(&self) -> String {
        "this console saves no scene".to_string()
    }
}

/// What a console takes from the session that hosts it.
pub struct Serving<'a> {
    pub name: &'a str,
    pub scene: &'a Scene,
    pub index: Arc<Index>,
    pub locks: Arc<Mutex<Locks>>,
    pub size: (i32, i32),
    pub format: &'a str,
    pub rotate: i64,
    pub gpu: Value,
    /// Where the session puts every frame.
    pub out: Vec<String>,
    /// The cap on the H264 stream's longest side. 0 streams the frame's size.
    pub stream_max: i32,
    pub verbose: bool,
}

pub struct Console {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl Console {
    /// Serve the console on `host` and `port`, and print its address.
    pub fn start(host: &str, port: u16, serving: Serving<'_>) -> Result<Console> {
        let mut hub = Hub::new();
        (hub.w, hub.h) = serving.size;
        hub.stream_max = serving.stream_max;
        hub.dests = serving.out;
        hub.verbose = serving.verbose;
        hub.set_display(serving.format, serving.rotate);
        hub.set_gpu(serving.gpu);
        hub.publish_scene(Some(serving.name), 0, serving.scene.to_dict());
        let shared = Arc::new(Shared {
            hub,
            index: Mutex::new(serving.index),
            name: serving.name.to_string(),
            looks: Mutex::new(Vec::new()),
            queue: Mutex::new(Vec::new()),
            layers: Mutex::new(Vec::new()),
            locks: serving.locks,
        });
        let server = Server::http((host, port)).map_err(|e| anyhow!("{e}"))?;
        let address = urls(host, bound_port(&server, port));
        let api = ConsoleAPI::new(Host(Arc::clone(&shared)))
            .with_web(true)
            .listening_on(host);
        let mut threads = spawn_encoders(&shared);
        threads.push(serve(Arc::new(server), Arc::clone(&shared), Arc::new(api)));
        let names = [serving.name.to_string()];
        println!(
            "{}",
            banner(&address, &names, "", serving.size, serving.format)
        );
        Ok(Console { shared, threads })
    }

    pub fn hub(&self) -> &Hub {
        &self.shared.hub
    }

    pub fn take_params(&self) -> Vec<Queued> {
        std::mem::take(&mut lock(&self.shared.queue))
    }

    pub fn take_layers(&self) -> Vec<LayerSet> {
        std::mem::take(&mut lock(&self.shared.layers))
    }

    /// The looks that applies gave layers since the last take.
    pub fn take_looks(&self) -> Vec<Value> {
        std::mem::take(&mut lock(&self.shared.looks))
    }

    /// Publish what changed in the scene that `renderer` runs, as
    /// Hub::publish_live() does, the whole scene after a console's `edited`
    /// sets, and keep the index of its params and layers.
    pub fn publish(&self, renderer: &Renderer, live: &mut Live, edited: bool) {
        let s = &self.shared;
        s.hub
            .publish_live(renderer, Some(live), edited, false, Some(&s.name), 0);
        let mut index = lock(&s.index);
        if !Arc::ptr_eq(&index, live.index()) {
            *index = Arc::clone(live.index());
        }
    }

    pub fn stop(mut self) {
        self.shared.hub.stop();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

/// The layers of every pass of the scene document `doc`.
fn layers_of(doc: &Value) -> impl Iterator<Item = &Value> {
    doc.get("passes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|p| p.get("layers")?.as_array())
        .flatten()
}

/// The scene document `doc` less what an apply here may change and what
/// changes as the scene runs: the layers, and the value of each param.
fn frame_of(doc: &Value) -> Value {
    fn values(params: Option<&mut Value>) {
        if let Some(Value::Object(m)) = params {
            for p in m.values_mut() {
                if let Value::Object(p) = p {
                    p.remove("value");
                }
            }
        }
    }
    let mut d = doc.clone();
    values(d.get_mut("params"));
    if let Some(Value::Array(passes)) = d.get_mut("passes") {
        for p in passes.iter_mut().filter_map(Value::as_object_mut) {
            p.remove("layers");
            values(p.get_mut("params"));
        }
    }
    d
}
