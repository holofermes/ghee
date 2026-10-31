//! A running scene as its host holds it: what the scene declares, what its
//! program set and locked, what a console changed, and what the next tick
//! reports.
//!
//! A host renders the scene and holds one Live beside the renderer. The
//! scene's program reaches the scene through the program's calls of a Live,
//! and a console through the console's calls. Each call changes the
//! renderer, then keeps by the rules of procedural::locks what changed.
//! The renderer's scene is the scene's document.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::Result;
use serde_json::{Map, Value};

use crate::draw::{call_of, Arg, Call, CallKind};
use crate::engine::{Given, Refused};
use crate::layer::{rebuilt_look, Address, Layer};
use crate::looper::RenderTarget;
use crate::procedural::locks::{call_keys, look_keys, Locks, Tuned};
use crate::scene::{no_layer, no_param, ArraySize, ParamValue, Scene};

/// One live param, as a host addresses it.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamRef {
    /// the name on the wire: `pass.name`, or `name`
    pub wire: String,
    /// the pass that holds it. None is scene scope.
    pub pass: Option<String>,
    /// the uniform itself, without the pass
    pub uniform: String,
    pub ty: String,
}

/// What the scene declares. A host feeds the arrays, sets the params and the
/// live fields of the layers, and a console addresses the passes by index.
///
/// A param answers to `pass.name` and to `name`. A bare name is the scene
/// param, else the param of the first pass that declares it. This is the
/// addressing of the procedural wire.
#[derive(Clone, Debug, Default)]
pub struct Index {
    pub arrays: Vec<(String, ArraySize)>,
    pub params: Vec<ParamRef>,
    pub passes: Vec<String>,
    /// Each layer's name, and true for a text layer.
    pub layers: Vec<(String, bool)>,
    /// The place in `params` of each param, by its name on the wire.
    wires: HashMap<String, usize>,
    /// True for a text layer, by the layer's name.
    texts: HashMap<String, bool>,
}

impl Index {
    pub fn of(scene: &Scene) -> Index {
        let mut arrays: Vec<(String, ArraySize)> = Vec::new();
        for p in &scene.passes {
            for a in &p.arrays {
                match arrays.iter_mut().find(|(n, _)| *n == a.name) {
                    Some(slot) => slot.1 = a.size,
                    None => arrays.push((a.name.clone(), a.size)),
                }
            }
        }
        let layers: Vec<(String, bool)> = scene
            .layers()
            .map(|l| (l.name().to_string(), l.as_text().is_some()))
            .collect();
        let texts: HashMap<String, bool> = layers.iter().cloned().collect();
        let mut params: Vec<ParamRef> = Vec::new();
        let scoped = std::iter::once((None, &scene.params)).chain(
            scene
                .passes
                .iter()
                .map(|p| (Some(p.name.as_str()), &p.params)),
        );
        for (pass, map) in scoped {
            for (name, p) in map.iter() {
                if let Some(pass) = pass {
                    params.push(ParamRef {
                        wire: format!("{pass}.{name}"),
                        pass: Some(pass.to_string()),
                        uniform: name.clone(),
                        ty: p.ty.clone(),
                    });
                }
                if params.iter().any(|r| r.wire == *name) {
                    continue;
                }
                params.push(ParamRef {
                    wire: name.clone(),
                    pass: pass.map(str::to_string),
                    uniform: name.clone(),
                    ty: p.ty.clone(),
                });
            }
        }
        Index {
            wires: params
                .iter()
                .enumerate()
                .map(|(k, r)| (r.wire.clone(), k))
                .collect(),
            texts,
            arrays,
            params,
            passes: scene.passes.iter().map(|p| p.name.clone()).collect(),
            layers,
        }
    }

    /// True for a text layer named `name`, false for a layer of another
    /// kind, None when the scene has no such layer.
    pub fn text(&self, name: &str) -> Option<bool> {
        self.texts.get(name).copied()
    }

    /// The param named `wire` on the wire.
    pub fn wired(&self, wire: &str) -> Option<&ParamRef> {
        self.wires.get(wire).map(|&k| &self.params[k])
    }

    /// The name that a tick gives the param `name` of pass `pass`, None for
    /// the scene's: the bare name when it addresses that param, else
    /// `pass.name`.
    pub fn ticked(&self, pass: Option<&str>, name: &str) -> Option<&str> {
        match (self.wired(name), pass) {
            (Some(r), _) if r.pass.as_deref() == pass => Some(&r.wire),
            (_, Some(pass)) => self
                .wired(&format!("{pass}.{name}"))
                .map(|r| r.wire.as_str()),
            _ => None,
        }
    }

    /// True for a text layer named `name`, false for a layer of another
    /// kind. A layer that the scene lacks is refused.
    pub fn layer(&self, name: &str) -> Result<bool, Refused> {
        self.text(name)
            .ok_or_else(|| Refused(no_layer(name, self.layers.iter().map(|(n, _)| n.as_str()))))
    }

    /// The param named by a host. `pass` addresses one pass, and a dotted name
    /// carries its own. A param that the scene lacks is refused.
    pub fn param(&self, pass: Option<&str>, name: &str) -> Result<&ParamRef, Refused> {
        let wire = match pass {
            Some(pass) => format!("{pass}.{name}"),
            None => name.to_string(),
        };
        self.wired(&wire)
            .ok_or_else(|| Refused(no_param(&wire, self.params.iter().map(|r| r.wire.as_str()))))
    }

    /// The size of the array `name`. An array that the scene lacks is
    /// refused.
    pub fn array(&self, name: &str) -> Result<ArraySize, Refused> {
        match self.arrays.iter().find(|(n, _)| n == name) {
            Some((_, size)) => Ok(*size),
            None => {
                let all: Vec<&str> = self.arrays.iter().map(|(n, _)| n.as_str()).collect();
                Err(Refused(format!(
                    "feed '{name}' is not an array this scene declares ({})",
                    match all.is_empty() {
                        true => "none".to_string(),
                        false => all.join(", "),
                    }
                )))
            }
        }
    }

    /// The layer `name` joined the scene or changed its kind, `text` true
    /// for a text layer. True when the index changed.
    fn took(&mut self, name: &str, text: bool) -> bool {
        if self.texts.insert(name.to_string(), text) == Some(text) {
            return false;
        }
        match self.layers.iter_mut().find(|(n, _)| n == name) {
            Some(slot) => slot.1 = text,
            None => self.layers.push((name.to_string(), text)),
        }
        true
    }

    /// The layer `name` left the scene.
    fn left(&mut self, name: &str) {
        self.texts.remove(name);
        self.layers.retain(|(n, _)| n != name);
    }
}

/// A param value that the program set, which a rebuild of the scene sets
/// again.
#[derive(Clone)]
struct Kept {
    pass: Option<String>,
    uniform: String,
    ty: String,
    value: ParamValue,
}

/// What the program changed in the scene since a console last took it.
#[derive(Debug, Default, PartialEq)]
pub struct Changes {
    /// The layers whose looks or live fields changed, by name.
    pub layers: Vec<String>,
    /// The params set, each with its pass, None for the scene's.
    pub params: Vec<(Option<String>, String)>,
    /// A layer joined the scene, left it, moved, or changed its kind.
    pub grown: bool,
}

impl Changes {
    pub fn is_empty(&self) -> bool {
        !self.grown && self.layers.is_empty() && self.params.is_empty()
    }
}

/// The live state of one running scene.
#[derive(Clone)]
pub struct Live {
    index: Arc<Index>,
    /// The value of each param, by the name that a tick gives it, as a scene
    /// file writes it, and the words of each text layer as its `text`, such
    /// as `title.text`.
    values: Map<String, Value>,
    /// Each value at the last tick.
    sent: Map<String, Value>,
    /// The last value that the program set of each param, by the name that a
    /// tick gives the param.
    kept: HashMap<String, Kept>,
    /// What the program set, which a console refuses to set.
    locks: Arc<Mutex<Locks>>,
    /// The keys of layers that a console changed.
    tuned: Tuned,
    /// The look of each layer of the scene's own, as the scene declared it.
    own: HashMap<String, Value>,
    /// The layers that the program added.
    added: HashSet<String>,
    /// The scene was set up again, so restore() sets every kept value again.
    rebuilt: bool,
    layers: BTreeSet<String>,
    params: BTreeSet<(Option<String>, String)>,
    grown: bool,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The value of each param of `scene`, by the name that a tick gives it, and
/// the words of each text layer as its `text`.
fn values_of(scene: &Scene, index: &Index) -> Map<String, Value> {
    let mut out = Map::new();
    for l in scene.text_layers() {
        out.insert(format!("{}.text", l.name), Value::from(l.text.clone()));
    }
    for r in index
        .params
        .iter()
        .filter(|r| index.ticked(r.pass.as_deref(), &r.uniform) == Some(&r.wire))
    {
        let params = match &r.pass {
            None => Some(&scene.params),
            Some(p) => scene
                .passes
                .iter()
                .find(|x| x.name == *p)
                .map(|x| &x.params),
        };
        if let Some(p) = params.and_then(|ps| ps.get(&r.uniform)) {
            out.insert(r.wire.clone(), p.value.to_json());
        }
    }
    out
}

/// The name of `look`, a layer as a scene declares it.
fn name_of(look: &Value) -> Option<&str> {
    look.get("name").and_then(Value::as_str)
}

/// Set the key `at` of a layer of `renderer` to `given`: a live field with no
/// rebuild, and any other key that the layer takes in its look by building
/// the layer again alone. The layer's new look when it was built again.
pub fn put_key(
    renderer: &mut dyn RenderTarget,
    at: &Address,
    given: Given,
) -> Result<Option<Value>> {
    let Err(e) = renderer.set_layer(at, given) else {
        return Ok(None);
    };
    let Some(look) = renderer
        .layer(at.layer)
        .and_then(|now| rebuilt_look(&now, at, given.to_json()))
    else {
        return Err(e);
    };
    renderer.update_layer(&look)?;
    Ok(Some(look))
}

impl Default for Live {
    /// The live state of a scene that declares nothing, until bind().
    fn default() -> Live {
        Live {
            index: Arc::new(Index::default()),
            values: Map::new(),
            sent: Map::new(),
            kept: HashMap::new(),
            locks: Arc::new(Mutex::new(Locks::default())),
            tuned: Tuned::default(),
            own: HashMap::new(),
            added: HashSet::new(),
            rebuilt: false,
            layers: BTreeSet::new(),
            params: BTreeSet::new(),
            grown: false,
        }
    }
}

impl Live {
    /// The live state of `scene`, which a host has just set up. The first
    /// tick reports every value.
    pub fn new(scene: &Scene) -> Live {
        let mut live = Live::default();
        live.bind(scene);
        live
    }

    /// Index `scene`, the scene that the host builds, filters and saved
    /// values included, and take its values. The next tick reports every
    /// value. What the program set and what a console changed stay.
    pub fn bind(&mut self, scene: &Scene) {
        self.own
            .retain(|name, _| scene.layers().any(|l| l.name() == name));
        for l in scene.layers() {
            let name = l.name();
            if !self.added.contains(name) && !self.own.contains_key(name) {
                self.own.insert(name.to_string(), l.to_dict());
            }
        }
        let index = Index::of(scene);
        self.values = values_of(scene, &index);
        self.sent.clear();
        self.index = Arc::new(index);
    }

    pub fn index(&self) -> &Arc<Index> {
        &self.index
    }

    /// What the program set, as a console reads it.
    pub fn locks(&self) -> Arc<Mutex<Locks>> {
        Arc::clone(&self.locks)
    }

    /// A copy of the state, which put_back() restores.
    pub fn save(&self) -> (Live, Locks) {
        (self.clone(), lock(&self.locks).clone())
    }

    /// Restore the state that save() copied. A console that holds the locks
    /// reads the restored ones.
    pub fn put_back(&mut self, (saved, locks): (Live, Locks)) {
        let held = Arc::clone(&self.locks);
        *self = saved;
        *lock(&held) = locks;
        self.locks = held;
    }

    /// The program ended, so a console sets anything again.
    pub fn unlock(&mut self) {
        *lock(&self.locks) = Locks::default();
    }

    // ------------------------------------------------------ the program's

    /// Set the param `p` to `value`, with no recompile. The program set it:
    /// it locks, no tick reports it, and a rebuild sets it again.
    pub fn set_param(
        &mut self,
        renderer: &dyn RenderTarget,
        p: &ParamRef,
        value: ParamValue,
    ) -> Result<()> {
        renderer.set_param(p.pass.as_deref(), &p.uniform, &p.ty, &value)?;
        let pass = p.pass.as_deref();
        lock(&self.locks).set_param(pass, &p.uniform);
        self.params.insert((p.pass.clone(), p.uniform.clone()));
        let Some(ticked) = self.index.ticked(pass, &p.uniform).map(str::to_string) else {
            return Ok(());
        };
        self.hold(&ticked, value.to_json(), true);
        let kept = Kept {
            pass: p.pass.clone(),
            uniform: p.uniform.clone(),
            ty: p.ty.clone(),
            value,
        };
        self.kept.insert(ticked, kept);
        Ok(())
    }

    /// Set the field `at` to `given`: a live field with no rebuild, and any
    /// other key of the layer by building the layer again alone. The program
    /// set it: it locks, and new words of a text layer reach no tick.
    pub fn set_layer(
        &mut self,
        renderer: &mut dyn RenderTarget,
        at: &Address,
        given: Given,
    ) -> Result<()> {
        if let Some(look) = put_key(renderer, at, given)? {
            self.took(&look);
        }
        if let (None, "text") = (at.effect, at.field) {
            let words = match given {
                Given::Json(v) => Some(v.clone()),
                Given::Text(t) => Some(Value::from(t)),
                Given::Numbers(_) => None,
            };
            if let Some(words) = words {
                self.hold(&format!("{}.text", at.layer), words, true);
            }
        }
        lock(&self.locks).set_field(at);
        self.layers.insert(at.layer.to_string());
        Ok(())
    }

    /// Add `look`, a layer as a scene declares it, at `index` among the
    /// layers of pass `pass`, with no rebuild. Its keys lock.
    pub fn add_layer(
        &mut self,
        renderer: &mut dyn RenderTarget,
        pass: &str,
        index: usize,
        look: &Value,
    ) -> Result<()> {
        renderer.add_layer(pass, index, look)?;
        self.took(look);
        if let Some(name) = name_of(look) {
            self.added.insert(name.to_string());
            lock(&self.locks).set_keys(name, look_keys(look));
            self.layers.insert(name.to_string());
        }
        self.grown = true;
        Ok(())
    }

    /// Give the layer of the name of `look` that look, where each key that
    /// `look` leaves out and a console changed keeps the console's value,
    /// and each other key that it leaves out takes the scene's own value
    /// for a layer of the scene's own. The keys of `look` lock. False when the
    /// layer had that look already.
    pub fn update_layer(&mut self, renderer: &mut dyn RenderTarget, look: &Value) -> Result<bool> {
        let given = self.filled(&*renderer, look);
        let changed = renderer.update_layer(&given)?;
        if changed {
            self.took(&given);
        }
        if let Some(name) = name_of(look) {
            lock(&self.locks).set_keys(name, look_keys(look));
            if changed {
                self.layers.insert(name.to_string());
            }
        }
        Ok(changed)
    }

    /// Give the layers of `calls` their looks, where a key that a call leaves
    /// out takes the console's value, else the scene's own, as
    /// update_layer() fills a look. The keys that each call passes lock.
    /// Returns the indexes of the calls that the renderer leaves for
    /// update_layer().
    pub fn update_calls(
        &mut self,
        renderer: &mut dyn RenderTarget,
        calls: &[Call],
    ) -> Result<Vec<usize>> {
        let kept = self.tuned.calls(calls, |n| renderer.layer(n));
        let kept = self.own_calls(kept.as_deref().unwrap_or(calls)).or(kept);
        let taken = renderer.update_calls(kept.as_deref().unwrap_or(calls))?;
        for &k in &taken.changed {
            let call = &calls[k];
            if let (CallKind::Text, Some(Arg::Str(words))) = (call.kind, call.args.first()) {
                let text = format!("{}.text", call.layer);
                self.hold(&text, Value::from(words.as_str()), true);
            }
            self.layers.insert(call.layer.clone());
        }
        let mut locks = lock(&self.locks);
        for (k, call) in calls.iter().enumerate() {
            if !taken.left.contains(&k) {
                locks.set_keys(&call.layer, call_keys(call));
            }
        }
        Ok(taken.left)
    }

    /// `look`, a look that the program gives a layer whole, as
    /// update_layer() hands it to the renderer.
    pub fn filled(&self, renderer: &dyn RenderTarget, look: &Value) -> Value {
        let kept = self.tuned.look(look, |n| renderer.layer(n));
        let kept = self.own_look(kept.as_ref().unwrap_or(look)).or(kept);
        kept.unwrap_or_else(|| look.clone())
    }

    /// `look` with each key that it leaves out at the scene's own value, for
    /// a layer of the scene's own of the same kind. None when no key fills.
    /// A drawn layer shows unless the program hides it.
    fn own_look(&self, look: &Value) -> Option<Value> {
        let own = self.own.get(name_of(look)?)?.as_object()?;
        if own.get("shape") != look.get("shape")
            || own.contains_key("image") != look.get("image").is_some()
        {
            return None;
        }
        let mut kept = look.clone();
        let into = kept.as_object_mut()?;
        let mut any = false;
        for (k, v) in own.iter().filter(|(k, _)| *k != "visible") {
            if !into.contains_key(k) {
                into.insert(k.clone(), v.clone());
                any = true;
            }
        }
        any.then_some(kept)
    }

    /// `calls` with each key that a call leaves out at the scene's own
    /// value, as own_look() fills a look. None when no key fills.
    fn own_calls(&self, calls: &[Call]) -> Option<Vec<Call>> {
        let mut kept = calls.to_vec();
        let mut any = false;
        for call in &mut kept {
            let Some(own) = self.own.get(&call.layer).and_then(call_of) else {
                continue;
            };
            if own.kind != call.kind {
                continue;
            }
            for ((key, arg), was) in call.kind.keys().iter().zip(&mut call.args).zip(own.args) {
                if *arg == Arg::None && was != Arg::None && *key != "visible" {
                    *arg = was;
                    any = true;
                }
            }
        }
        any.then_some(kept)
    }

    /// Take layer `name` out of the scene, with no rebuild.
    pub fn remove_layer(&mut self, renderer: &mut dyn RenderTarget, name: &str) -> Result<()> {
        renderer.remove_layer(name)?;
        Arc::make_mut(&mut self.index).left(name);
        let text = format!("{name}.text");
        self.values.remove(&text);
        self.sent.remove(&text);
        self.tuned.forget(name);
        self.own.remove(name);
        self.added.remove(name);
        self.layers.remove(name);
        self.grown = true;
        Ok(())
    }

    /// Put layer `name` at `index` among the layers of its pass, with no
    /// rebuild.
    pub fn move_layer(
        &mut self,
        renderer: &mut dyn RenderTarget,
        name: &str,
        index: usize,
    ) -> Result<()> {
        renderer.move_layer(name, index)?;
        self.grown = true;
        Ok(())
    }

    /// Index `look`, a look that the renderer took for a layer, and hold the
    /// words of a text layer as the host's own, which no tick reports.
    pub fn took(&mut self, look: &Value) {
        let Ok(layer) = Layer::from_dict("the layer", look) else {
            return;
        };
        let name = layer.name().to_string();
        let text = format!("{name}.text");
        match layer.as_text() {
            Some(l) => self.hold(&text, Value::from(l.text.clone()), true),
            None => {
                self.values.remove(&text);
                self.sent.remove(&text);
            }
        }
        if Arc::make_mut(&mut self.index).took(&name, layer.as_text().is_some()) {
            self.grown = true;
        }
    }

    /// Hold `value` under `name`, the name that a tick gives it, and as
    /// already reported when `sent`.
    fn hold(&mut self, name: &str, value: Value, sent: bool) {
        if sent {
            self.sent.insert(name.to_string(), value.clone());
        }
        self.values.insert(name.to_string(), value);
    }

    // ------------------------------------------------------ the console's

    /// A console sets the param `name` of pass `pass`, None for the scene's,
    /// of type `ty`. The next tick reports it.
    pub fn console_param(
        &mut self,
        renderer: &dyn RenderTarget,
        pass: Option<&str>,
        name: &str,
        ty: &str,
        value: &ParamValue,
    ) -> Result<()> {
        renderer.set_param(pass, name, ty, value)?;
        if let Some(ticked) = self.index.ticked(pass, name).map(str::to_string) {
            self.hold(&ticked, value.to_json(), false);
        }
        Ok(())
    }

    /// A console sets the key `at` to `value`, written as the scene format
    /// writes the key, as put_key() sets it. The key counts as the console's,
    /// and the next tick reports new words of a text layer.
    pub fn console_layer(
        &mut self,
        renderer: &mut dyn RenderTarget,
        at: &Address,
        value: &Value,
    ) -> Result<()> {
        if let Some(look) = put_key(renderer, at, Given::Json(value))? {
            self.took(&look);
        }
        self.tuned.set(at);
        let text = format!("{}.text", at.layer);
        if let (None, "text", true) = (at.effect, at.field, self.values.contains_key(&text)) {
            self.hold(&text, value.clone(), false);
        }
        Ok(())
    }

    /// `look`, the look that a console gives a layer whole, with each key
    /// that the program set at its running value. Each other key that `look`
    /// changes counts as the console's.
    pub fn console_look(&mut self, renderer: &dyn RenderTarget, look: Value) -> Value {
        let Some(now) = name_of(&look).and_then(|n| renderer.layer(n)) else {
            return look;
        };
        let locks = lock(&self.locks).clone();
        self.tuned.applied(look, &now, &locks)
    }

    /// `posted`, a scene that a console sets up in place of the running
    /// scene, with each layer's look as console_look() gives it. A layer
    /// that fails to take its look stays as posted, and the error says why.
    pub fn console_scene(
        &mut self,
        renderer: &dyn RenderTarget,
        posted: &mut Scene,
    ) -> Result<(), String> {
        let names: Vec<String> = posted.layers().map(|l| l.name().to_string()).collect();
        let mut failed = Vec::new();
        for name in names {
            let Some(layer) = posted.layer(&name) else {
                continue;
            };
            let look = layer.to_dict();
            let kept = self.console_look(renderer, look.clone());
            if kept == look {
                continue;
            }
            let put = Layer::from_dict("the layer", &kept).and_then(|mut layer| {
                posted.load_layer(&mut layer)?;
                posted.replace_layer(layer)
            });
            if let Err(e) = put {
                failed.push(format!("layer {name}: {e:#}"));
            }
        }
        match failed.is_empty() {
            true => Ok(()),
            false => Err(failed.join("\n")),
        }
    }

    /// The host set the scene up again, so the next restore() sets each
    /// value that the program set again.
    pub fn rebuilt(&mut self) {
        self.rebuilt = !self.kept.is_empty();
    }

    /// After a rebuild, set each value that the program set again. Returns
    /// each value that the scene no longer takes, by the name that a tick
    /// gives its param, with the reason, and forgets it.
    pub fn restore(&mut self, renderer: &dyn RenderTarget) -> Vec<(String, String)> {
        if !std::mem::take(&mut self.rebuilt) {
            return Vec::new();
        }
        let mut refused = Vec::new();
        for (ticked, k) in &self.kept {
            if let Err(e) = renderer.set_param(k.pass.as_deref(), &k.uniform, &k.ty, &k.value) {
                refused.push((ticked.clone(), format!("{e:#}")));
            }
        }
        for (ticked, _) in &refused {
            self.kept.remove(ticked);
        }
        refused
    }

    // ------------------------------------------------------ the ticks

    /// The values that changed since the last tick. The first tick holds
    /// every value.
    pub fn tick_params(&mut self) -> Map<String, Value> {
        let mut out = Map::new();
        for (k, v) in &self.values {
            if self.sent.get(k) != Some(v) {
                out.insert(k.clone(), v.clone());
            }
        }
        for (k, v) in &out {
            self.sent.insert(k.clone(), v.clone());
        }
        out
    }

    /// True when the program changed the scene since take_changes().
    pub fn changed(&self) -> bool {
        self.grown || !(self.layers.is_empty() && self.params.is_empty())
    }

    /// What the program changed in the scene since the last take.
    pub fn take_changes(&mut self) -> Changes {
        Changes {
            layers: std::mem::take(&mut self.layers).into_iter().collect(),
            params: std::mem::take(&mut self.params).into_iter().collect(),
            grown: std::mem::take(&mut self.grown),
        }
    }
}
