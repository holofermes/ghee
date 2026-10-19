//! Kind 6 of the wire: an RFC 6902 JSON Patch over the scene's layers.
//!
//! The patch's document is the scene's layers, an object of each pass's name
//! and the list of its layers in the order that they draw, each layer as a
//! scene declares it, with the keys at their defaults left out. The pointer
//! `""` names the whole document, `/main` the layers of pass `main`,
//! `/main/2` the third of them, `/main/-` the end of that list, `/main/2/x` a
//! key of the layer, and `/main/2/glow/strength` a field of its effect
//! (RFC 6901).
//!
//! draft() applies a patch to a copy of the document, as RFC 6902 applies
//! it, and checks the names of the result. A patch that fails there changes
//! nothing, and one that passes comes out as the changes that the running
//! scene takes.

use std::collections::HashSet;

use serde_json::{json, Map, Value};

/// A place that a pointer names below a pass.
#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    pub pass: String,
    /// The layer's place among the pass's layers, None for `-`, past the
    /// last.
    pub index: Option<usize>,
    /// A key of the layer, then a field of its effect. None names the layer.
    pub keys: Vec<String>,
}

impl Place {
    /// The pointer, as the patch wrote it.
    pub fn pointer(&self) -> String {
        let index = self.index.map_or("-".to_string(), |i| i.to_string());
        let mut out = format!("/{}/{index}", escape(&self.pass));
        for k in &self.keys {
            out.push('/');
            out.push_str(&escape(k));
        }
        out
    }
}

/// What a pointer names.
#[derive(Clone, Debug, PartialEq)]
pub enum Pointer {
    /// The whole document.
    Doc,
    /// The layers of a pass.
    Pass(String),
    /// A layer, a key of it, or a field of its effect.
    At(Place),
}

impl Pointer {
    /// The pointer, as the patch wrote it.
    pub fn text(&self) -> String {
        match self {
            Pointer::Doc => String::new(),
            Pointer::Pass(pass) => format!("/{}", escape(pass)),
            Pointer::At(place) => place.pointer(),
        }
    }
}

/// One operation of a patch, as RFC 6902 section 4 names it.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Add(Pointer, Value),
    Remove(Pointer),
    Replace(Pointer, Value),
    Move { from: Pointer, path: Pointer },
    Copy { from: Pointer, path: Pointer },
    Test(Pointer, Value),
}

impl Op {
    /// The operation's `path`.
    pub fn path(&self) -> &Pointer {
        match self {
            Op::Add(at, _) | Op::Remove(at) | Op::Replace(at, _) | Op::Test(at, _) => at,
            Op::Move { path, .. } | Op::Copy { path, .. } => path,
        }
    }
}

/// The operations of the patch `text`, or why it holds none.
pub fn ops_of(text: &str) -> Result<Vec<Op>, String> {
    let doc: Value =
        serde_json::from_str(text).map_err(|e| format!("kind 6 holds no JSON: {e}"))?;
    let Value::Array(items) = doc else {
        return Err("kind 6 needs a JSON Patch, a list of operations".to_string());
    };
    items.iter().enumerate().map(|(k, o)| op_of(k, o)).collect()
}

fn op_of(k: usize, o: &Value) -> Result<Op, String> {
    let at = |what: &str| format!("kind 6 operation {k}: {what}");
    let Value::Object(o) = o else {
        return Err(at("needs an object"));
    };
    let text = |key: &str| {
        o.get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| at(&format!("needs '{key}'")))
    };
    let place = |key: &str, end: bool| -> Result<Pointer, String> {
        let pointer = pointer_of(text(key)?).map_err(|why| at(&why))?;
        if let Pointer::At(p) = &pointer {
            if p.index.is_none() && (!end || !p.keys.is_empty()) {
                return Err(at(&format!(
                    "'{key}' takes '-' only as the end of a pass's layers"
                )));
            }
        }
        Ok(pointer)
    };
    let value = || o.get("value").cloned().ok_or_else(|| at("needs 'value'"));
    Ok(match text("op")? {
        "add" => Op::Add(place("path", true)?, value()?),
        "remove" => Op::Remove(place("path", false)?),
        "replace" => Op::Replace(place("path", false)?, value()?),
        "move" => Op::Move {
            from: place("from", false)?,
            path: place("path", true)?,
        },
        "copy" => Op::Copy {
            from: place("from", false)?,
            path: place("path", true)?,
        },
        "test" => Op::Test(place("path", false)?, value()?),
        op => return Err(at(&format!("has no op '{op}'"))),
    })
}

/// What `pointer` names: the document, a pass, or a place in a pass and up
/// to two keys.
fn pointer_of(pointer: &str) -> Result<Pointer, String> {
    if pointer.is_empty() {
        return Ok(Pointer::Doc);
    }
    let Some(rest) = pointer.strip_prefix('/') else {
        return Err(format!("pointer '{pointer}' needs to start with '/'"));
    };
    let tokens: Vec<String> = rest.split('/').map(unescape).collect();
    let (pass, index, keys) = match tokens.as_slice() {
        [pass] => return Ok(Pointer::Pass(pass.clone())),
        [pass, index, keys @ ..] if keys.len() <= 2 => (pass, index, keys),
        _ => {
            return Err(format!(
                "pointer '{pointer}' needs a pass, a layer's place, and at most a key and a \
                 field: /main/0/glow/strength"
            ))
        }
    };
    let index = match index.as_str() {
        "-" => None,
        i if i == "0" || (!i.starts_with('0') && i.bytes().all(|b| b.is_ascii_digit())) => Some(
            i.parse()
                .map_err(|_| format!("pointer '{pointer}': no place {i}"))?,
        ),
        i => return Err(format!("pointer '{pointer}': '{i}' is no layer's place")),
    };
    Ok(Pointer::At(Place {
        pass: pass.clone(),
        index,
        keys: keys.to_vec(),
    }))
}

/// A reference token as RFC 6901 section 4 reads it: `~1` is `/`, then `~0`
/// is `~`.
fn unescape(token: &str) -> String {
    token.replace("~1", "/").replace("~0", "~")
}

fn escape(token: &str) -> String {
    token.replace('~', "~0").replace('/', "~1")
}

/// True when `a` and `b` are equal as RFC 6902 section 4.6 compares JSON:
/// numbers by their value, objects by their members in any order.
pub fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w)))
        }
        _ => a == b,
    }
}

// ------------------------------------------------------------ the draft

/// The running scene, as a patch reads it.
pub trait Layers {
    /// The names of the scene's passes, in their order.
    fn passes(&self) -> Vec<String>;
    /// The names of the layers of pass `pass`, in the order that they draw.
    /// None when the scene has no such pass.
    fn pass_layers(&self, pass: &str) -> Option<Vec<String>>;
    /// The layer `name`, as a scene declares it.
    fn layer(&self, name: &str) -> Option<Value>;
    /// The look that a layer of the scene takes when a patch replaces it
    /// whole with `look`.
    fn filled(&self, look: &Value) -> Value;
}

/// Who a layer of the document is.
#[derive(Clone, Debug, PartialEq)]
pub enum Who {
    /// The layer of the running scene of that name.
    Own(String),
    /// A layer that the patch adds.
    New,
}

#[derive(Clone, Debug)]
enum Look {
    /// The look that the layer runs with.
    Running,
    /// The look that a replace gave the layer, before filled() fills it.
    Given(Value),
    Known(Value),
}

/// A layer of the document.
#[derive(Clone, Debug)]
pub struct Entry {
    pub who: Who,
    look: Look,
    /// The operation that added the layer or last moved it.
    pub by: Option<usize>,
}

impl Entry {
    /// The look of a layer that the patch adds.
    pub fn new_look(&self) -> Option<&Value> {
        match (&self.who, &self.look) {
            (Who::New, Look::Known(look)) => Some(look),
            _ => None,
        }
    }
}

/// A pass that a patch reached.
#[derive(Clone, Debug)]
pub struct Pass {
    pub name: String,
    /// The names of its layers before the patch, in their order.
    pub before: Vec<String>,
    /// Its layers after the patch, in their order.
    pub after: Vec<Entry>,
}

impl Pass {
    /// True when the pass holds the layers that it held, in their order.
    pub fn kept_order(&self) -> bool {
        self.after.len() == self.before.len()
            && self
                .after
                .iter()
                .zip(&self.before)
                .all(|(e, n)| matches!(&e.who, Who::Own(o) if o == n))
    }
}

/// A change to a layer of the running scene, made by the operation `by`.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// Set the key or the effect's field `keys` to `value`, null for its
    /// default.
    Key {
        by: usize,
        layer: String,
        keys: Vec<String>,
        value: Value,
    },
    /// Give the layer `look`, which carries its name.
    Look {
        by: usize,
        layer: String,
        look: Value,
    },
}

impl Step {
    /// The layer that the step changes.
    pub fn layer(&self) -> &str {
        match self {
            Step::Key { layer, .. } | Step::Look { layer, .. } => layer,
        }
    }
}

/// What a patch does to the running scene.
#[derive(Clone, Debug)]
pub struct Draft {
    /// Each pass that the patch reached.
    pub passes: Vec<Pass>,
    /// Each layer of the scene that leaves it, with the operation that took
    /// it out.
    pub removed: Vec<(String, usize)>,
    /// The changes to the layers of the scene, in the order of the patch.
    pub steps: Vec<Step>,
}

impl Draft {
    /// How many changes the scene takes, a pass's new order counted once.
    pub fn changes(&self) -> usize {
        let placed = self.passes.iter().filter(|p| !p.kept_order()).count();
        let added = self
            .passes
            .iter()
            .flat_map(|p| &p.after)
            .filter(|e| e.who == Who::New)
            .count();
        self.removed.len() + self.steps.len() + added + placed
    }
}

/// Why a patch changes nothing: the operation that failed, None for the
/// patch as a whole, and the reason.
#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    pub by: Option<usize>,
    pub why: String,
}

impl Refusal {
    /// The refusal as ghee logs it, the operation named by its place in
    /// `ops` and its path.
    pub fn line(&self, ops: &[Op]) -> String {
        match self.by.and_then(|k| Some((k, ops.get(k)?))) {
            Some((k, op)) => format!(
                "kind 6 operation {k} at '{}': {}",
                op.path().text(),
                self.why
            ),
            None => format!("kind 6: {}", self.why),
        }
    }
}

/// Apply `ops` to a copy of the document of `scene`, in their order, as RFC
/// 6902 applies them, then check that the result names each layer once.
pub fn draft(scene: &dyn Layers, ops: &[Op]) -> Result<Draft, Refusal> {
    let mut d = Drafting {
        scene,
        draft: Draft {
            passes: Vec::new(),
            removed: Vec::new(),
            steps: Vec::new(),
        },
    };
    for (k, op) in ops.iter().enumerate() {
        d.op(k, op).map_err(|why| Refusal { by: Some(k), why })?;
    }
    d.check()?;
    Ok(d.draft)
}

struct Drafting<'a> {
    scene: &'a dyn Layers,
    draft: Draft,
}

impl Drafting<'_> {
    fn op(&mut self, k: usize, op: &Op) -> Result<(), String> {
        match op {
            Op::Add(at, value) => self.add(k, at, value.clone()),
            Op::Remove(at) => self.remove(k, at).map(drop),
            Op::Replace(at, value) => {
                self.get(at)?;
                self.put(k, at, value.clone())
            }
            Op::Move { from, path } => self.move_to(k, from, path),
            Op::Copy { from, path } => {
                let value = self
                    .get(from)
                    .map_err(|why| format!("from '{}': {why}", from.text()))?;
                self.add(k, path, value)
            }
            Op::Test(at, value) => match self.get(at)? {
                now if same(&now, value) => Ok(()),
                now => Err(format!("the test found {now}")),
            },
        }
    }

    /// The place of pass `name` among the passes reached, reached now when
    /// the patch had not reached it.
    fn pass(&mut self, name: &str) -> Result<usize, String> {
        if let Some(i) = self.draft.passes.iter().position(|p| p.name == name) {
            return Ok(i);
        }
        let before = self
            .scene
            .pass_layers(name)
            .ok_or_else(|| format!("the scene has no pass '{name}'"))?;
        let after = before
            .iter()
            .map(|n| Entry {
                who: Who::Own(n.clone()),
                look: Look::Running,
                by: None,
            })
            .collect();
        self.draft.passes.push(Pass {
            name: name.to_string(),
            before,
            after,
        });
        Ok(self.draft.passes.len() - 1)
    }

    /// The pass and the place of the layer at `place`.
    fn layer_at(&mut self, place: &Place) -> Result<(usize, usize), String> {
        let i = self.pass(&place.pass)?;
        let j = place
            .index
            .filter(|&j| j < self.draft.passes[i].after.len())
            .ok_or_else(|| format!("pass '{}' has no layer there", place.pass))?;
        Ok((i, j))
    }

    fn look(&self, e: &Entry) -> Value {
        match (&e.look, &e.who) {
            (Look::Known(look), _) => look.clone(),
            (Look::Given(look), _) => self.scene.filled(look),
            (Look::Running, Who::Own(name)) => self
                .scene
                .layer(name)
                .unwrap_or_else(|| json!({ "name": name })),
            (Look::Running, Who::New) => unreachable!("a new layer holds its look"),
        }
    }

    /// The value at `at`, or why none is there.
    fn get(&mut self, at: &Pointer) -> Result<Value, String> {
        match at {
            Pointer::Doc => {
                let mut doc = Map::new();
                for pass in self.scene.passes() {
                    let layers = self.get(&Pointer::Pass(pass.clone()))?;
                    doc.insert(pass, layers);
                }
                Ok(Value::Object(doc))
            }
            Pointer::Pass(pass) => {
                let i = self.pass(pass)?;
                let looks = self.draft.passes[i].after.iter().map(|e| self.look(e));
                Ok(Value::Array(looks.collect()))
            }
            Pointer::At(place) => {
                let (i, j) = self.layer_at(place)?;
                let mut value = self.look(&self.draft.passes[i].after[j]);
                let name = name_of(&value);
                for (n, key) in place.keys.iter().enumerate() {
                    let held = place.keys[..=n].join("/");
                    value = value
                        .get(key.as_str())
                        .cloned()
                        .ok_or_else(|| format!("layer '{name}' holds no '{held}'"))?;
                }
                Ok(value)
            }
        }
    }

    /// RFC 6902 section 4.1: a layer goes in at its place, and a key's
    /// layer or effect holds it.
    fn add(&mut self, k: usize, at: &Pointer, value: Value) -> Result<(), String> {
        let Pointer::At(place) = at else {
            return self.put(k, at, value);
        };
        if let [effect, _] = place.keys.as_slice() {
            let (i, j) = self.layer_at(place)?;
            let look = self.look(&self.draft.passes[i].after[j]);
            if !look.get(effect.as_str()).is_some_and(Value::is_object) {
                return Err(format!("layer '{}' holds no '{effect}'", name_of(&look)));
            }
        }
        if !place.keys.is_empty() {
            return self.put(k, at, value);
        }
        let i = self.pass(&place.pass)?;
        let after = &mut self.draft.passes[i].after;
        let j = place.index.unwrap_or(after.len());
        if j > after.len() {
            return Err(format!("pass '{}' has no place {j}", place.pass));
        }
        if !value.is_object() {
            return Err("a layer is an object".to_string());
        }
        after.insert(
            j,
            Entry {
                who: Who::New,
                look: Look::Known(value),
                by: Some(k),
            },
        );
        Ok(())
    }

    /// Set the value at `at`, which holds one or which add() checked.
    fn put(&mut self, k: usize, at: &Pointer, value: Value) -> Result<(), String> {
        match at {
            Pointer::Doc => {
                let Value::Object(doc) = value else {
                    return Err("the document is an object of each pass's layers".to_string());
                };
                let passes = self.scene.passes();
                if let Some(pass) = doc.keys().find(|p| !passes.contains(p)) {
                    return Err(format!("the scene has no pass '{pass}'"));
                }
                if let Some(pass) = passes.iter().find(|p| !doc.contains_key(*p)) {
                    return Err(format!("pass '{pass}' stays in the scene"));
                }
                for (pass, layers) in doc {
                    self.put(k, &Pointer::Pass(pass), layers)?;
                }
                Ok(())
            }
            Pointer::Pass(pass) => {
                let i = self.pass(pass)?;
                let Value::Array(looks) = value else {
                    return Err(format!("pass '{pass}' takes a list of layers"));
                };
                self.put_pass(k, i, looks)
            }
            Pointer::At(place) => {
                let (i, j) = self.layer_at(place)?;
                match place.keys.as_slice() {
                    [] => self.put_layer(k, i, j, value),
                    keys => self.put_key(k, i, j, keys, value),
                }
            }
        }
    }

    /// Give pass `i` the layers `looks`. A look of the name of a layer of
    /// the pass replaces that layer, and each other look adds a layer.
    fn put_pass(&mut self, k: usize, i: usize, looks: Vec<Value>) -> Result<(), String> {
        let mut left = std::mem::take(&mut self.draft.passes[i].after);
        let mut after = Vec::with_capacity(looks.len());
        for look in looks {
            if !look.is_object() {
                return Err("a layer is an object".to_string());
            }
            let own = look.get("name").and_then(Value::as_str).and_then(|n| {
                left.iter()
                    .position(|e| matches!(&e.who, Who::Own(o) if o == n))
            });
            let entry = match own {
                Some(at) => {
                    let mut entry = left.remove(at);
                    if let Who::Own(layer) = &entry.who {
                        self.draft.steps.push(Step::Look {
                            by: k,
                            layer: layer.clone(),
                            look: look.clone(),
                        });
                    }
                    entry.look = Look::Given(look);
                    entry.by = Some(k);
                    entry
                }
                None => Entry {
                    who: Who::New,
                    look: Look::Known(look),
                    by: Some(k),
                },
            };
            after.push(entry);
        }
        for e in left {
            if let Who::Own(name) = e.who {
                self.draft.removed.push((name, k));
            }
        }
        self.draft.passes[i].after = after;
        Ok(())
    }

    /// Replace the layer at `j` of pass `i` with `look`. A layer of the
    /// scene keeps its name.
    fn put_layer(&mut self, k: usize, i: usize, j: usize, look: Value) -> Result<(), String> {
        let Value::Object(mut o) = look else {
            return Err("a layer is an object".to_string());
        };
        let entry = &mut self.draft.passes[i].after[j];
        let Who::Own(name) = &entry.who else {
            entry.look = Look::Known(Value::Object(o));
            return Ok(());
        };
        match o.get("name") {
            None => {
                o.insert("name".to_string(), Value::from(name.as_str()));
            }
            Some(n) if n == name.as_str() => {}
            Some(_) => return Err(format!("layer '{name}' keeps its name")),
        }
        let look = Value::Object(o);
        self.draft.steps.push(Step::Look {
            by: k,
            layer: name.clone(),
            look: look.clone(),
        });
        entry.look = Look::Given(look);
        Ok(())
    }

    /// Set the key or the effect's field `keys` of the layer at `j` of pass
    /// `i` to `value`, and take it out for null, its default. A layer of the
    /// scene keeps its name.
    fn put_key(
        &mut self,
        k: usize,
        i: usize,
        j: usize,
        keys: &[String],
        value: Value,
    ) -> Result<(), String> {
        let who = self.draft.passes[i].after[j].who.clone();
        let mut look = self.look(&self.draft.passes[i].after[j]);
        let name = name_of(&look);
        if let (Who::Own(own), [key]) = (&who, keys) {
            match key == "name" {
                true if value == own.as_str() => return Ok(()),
                true => return Err(format!("layer '{own}' keeps its name")),
                false => {}
            }
        }
        let (into, field) = match keys {
            [field] => (look.as_object_mut(), field),
            [effect, field] => (
                look.get_mut(effect.as_str()).and_then(Value::as_object_mut),
                field,
            ),
            _ => unreachable!("a pointer holds at most a key and a field"),
        };
        let into = into.ok_or_else(|| format!("layer '{name}' holds no '{}'", keys[0]))?;
        match value.is_null() {
            true => into.remove(field.as_str()),
            false => into.insert(field.clone(), value.clone()),
        };
        self.draft.passes[i].after[j].look = Look::Known(look);
        if let Who::Own(layer) = who {
            self.draft.steps.push(Step::Key {
                by: k,
                layer,
                keys: keys.to_vec(),
                value,
            });
        }
        Ok(())
    }

    /// RFC 6902 section 4.2: take out the value at `at`, which holds one.
    fn remove(&mut self, k: usize, at: &Pointer) -> Result<Value, String> {
        let place = match at {
            Pointer::Doc => return Err("the whole document stays".to_string()),
            Pointer::Pass(pass) => {
                self.pass(pass)?;
                return Err(format!("pass '{pass}' stays in the scene"));
            }
            Pointer::At(place) => place,
        };
        let value = self.get(at)?;
        let (i, j) = self.layer_at(place)?;
        if !place.keys.is_empty() {
            self.put_key(k, i, j, &place.keys, Value::Null)?;
            return Ok(value);
        }
        if let Who::Own(name) = self.draft.passes[i].after.remove(j).who {
            self.draft.removed.push((name, k));
        }
        Ok(value)
    }

    /// RFC 6902 section 4.4. A layer of the scene moves within its own
    /// pass, and stays itself.
    fn move_to(&mut self, k: usize, from: &Pointer, path: &Pointer) -> Result<(), String> {
        let from_at = |why: String| format!("from '{}': {why}", from.text());
        if path.text().starts_with(&format!("{}/", from.text())) {
            return Err("'path' lies inside 'from'".to_string());
        }
        let (Pointer::At(f), Pointer::At(p)) = (from, path) else {
            let value = self.remove(k, from).map_err(from_at)?;
            return self.add(k, path, value);
        };
        match (f.keys.is_empty(), p.keys.is_empty()) {
            (true, true) => {}
            (true, false) => return Err("a layer moves to a place among layers".to_string()),
            _ => {
                let value = self.remove(k, from).map_err(from_at)?;
                return self.add(k, path, value);
            }
        }
        let (i, j) = self.layer_at(f).map_err(from_at)?;
        if matches!(self.draft.passes[i].after[j].who, Who::Own(_)) && f.pass != p.pass {
            return Err("a layer moves within its own pass".to_string());
        }
        let entry = self.draft.passes[i].after.remove(j);
        let t = self.pass(&p.pass)?;
        let after = &mut self.draft.passes[t].after;
        let to = p.index.unwrap_or(after.len());
        if to > after.len() {
            return Err(format!("pass '{}' has no place {to}", p.pass));
        }
        after.insert(
            to,
            Entry {
                by: Some(k),
                ..entry
            },
        );
        Ok(())
    }

    /// Each layer that the patch adds takes a name that no other layer of
    /// the result holds.
    fn check(&self) -> Result<(), Refusal> {
        let passes = &self.draft.passes;
        let mut names: HashSet<&str> = passes
            .iter()
            .flat_map(|p| &p.after)
            .filter_map(|e| match &e.who {
                Who::Own(name) => Some(name.as_str()),
                Who::New => None,
            })
            .collect();
        let reached: HashSet<&str> = passes
            .iter()
            .flat_map(|p| &p.before)
            .map(String::as_str)
            .collect();
        for e in passes.iter().flat_map(|p| &p.after) {
            let Some(name) = e
                .new_look()
                .and_then(|l| l.get("name"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            let elsewhere = !reached.contains(name) && self.scene.layer(name).is_some();
            if !names.insert(name) || elsewhere {
                return Err(Refusal {
                    by: e.by,
                    why: format!("the result declares layer '{name}' twice"),
                });
            }
        }
        Ok(())
    }
}

/// The name of `look`, a layer as a scene declares it.
fn name_of(look: &Value) -> String {
    look.get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn at(pass: &str, index: Option<usize>, keys: &[&str]) -> Pointer {
        Pointer::At(Place {
            pass: pass.to_string(),
            index,
            keys: keys.iter().map(|k| (*k).to_string()).collect(),
        })
    }

    #[test]
    fn a_patch_reads_each_operation_of_rfc_6902() {
        let text = json!([
            {"op": "add", "path": "/main/-", "value": {"name": "t", "text": "hi"}},
            {"op": "replace", "path": "/main/0/x", "value": null},
            {"op": "remove", "path": "/main/1/glow/strength"},
            {"op": "move", "from": "/main/1", "path": "/main/0"},
            {"op": "copy", "from": "/main/0/x", "path": "/main/1/y"},
            {"op": "test", "path": "/m~1a~0in/10", "value": {"name": "t"}},
            {"op": "test", "path": "", "value": {}},
            {"op": "replace", "path": "/main", "value": []},
        ])
        .to_string();
        assert_eq!(
            ops_of(&text).unwrap(),
            vec![
                Op::Add(at("main", None, &[]), json!({"name": "t", "text": "hi"})),
                Op::Replace(at("main", Some(0), &["x"]), Value::Null),
                Op::Remove(at("main", Some(1), &["glow", "strength"])),
                Op::Move {
                    from: at("main", Some(1), &[]),
                    path: at("main", Some(0), &[]),
                },
                Op::Copy {
                    from: at("main", Some(0), &["x"]),
                    path: at("main", Some(1), &["y"]),
                },
                Op::Test(at("m/a~in", Some(10), &[]), json!({"name": "t"})),
                Op::Test(Pointer::Doc, json!({})),
                Op::Replace(Pointer::Pass("main".to_string()), json!([])),
            ]
        );
        assert_eq!(at("m/a~in", Some(10), &[]).text(), "/m~1a~0in/10");
        assert_eq!(Pointer::Pass("a/b".to_string()).text(), "/a~1b");
    }

    #[test]
    fn a_patch_that_rfc_6902_refuses_says_why() {
        for (patch, why) in [
            (
                json!({"op": "add"}),
                "kind 6 needs a JSON Patch, a list of operations",
            ),
            (
                json!([{"path": "/main/0"}]),
                "kind 6 operation 0: needs 'op'",
            ),
            (
                json!([{"op": "add", "path": "/main/0"}]),
                "kind 6 operation 0: needs 'value'",
            ),
            (
                json!([{"op": "remove", "path": "/main/-"}]),
                "kind 6 operation 0: 'path' takes '-' only as the end of a pass's layers",
            ),
            (
                json!([{"op": "replace", "path": "/main/01", "value": 1}]),
                "kind 6 operation 0: pointer '/main/01': '01' is no layer's place",
            ),
            (
                json!([{"op": "remove", "path": "main/0"}]),
                "kind 6 operation 0: pointer 'main/0' needs to start with '/'",
            ),
            (
                json!([{"op": "remove", "path": "/main/0/a/b/c"}]),
                "kind 6 operation 0: pointer '/main/0/a/b/c' needs a pass, a layer's place, \
                 and at most a key and a field: /main/0/glow/strength",
            ),
            (
                json!([{"op": "merge", "path": "/main/0", "value": {}}]),
                "kind 6 operation 0: has no op 'merge'",
            ),
        ] {
            assert_eq!(ops_of(&patch.to_string()), Err(why.to_string()), "{patch}");
        }
    }

    #[test]
    fn json_compares_as_rfc_6902_compares_it() {
        assert!(same(
            &json!({"a": 1, "b": [2.0]}),
            &json!({"b": [2], "a": 1.0})
        ));
        assert!(!same(&json!({"a": 1}), &json!({"a": 1, "b": null})));
        assert!(!same(&json!("1"), &json!(1)));
    }

    /// A scene of the passes `main` and `top`, each layer as given.
    struct Scene(Vec<(&'static str, Vec<Value>)>);

    fn scene() -> Scene {
        Scene(vec![
            (
                "main",
                vec![
                    json!({"name": "title", "text": "hi", "x": 2}),
                    json!({"name": "dot", "shape": "circle", "glow": {"strength": 0.5}}),
                    json!({"name": "floor", "shape": "rect"}),
                ],
            ),
            ("top", vec![json!({"name": "tag", "text": "t"})]),
        ])
    }

    impl Layers for Scene {
        fn passes(&self) -> Vec<String> {
            self.0.iter().map(|(p, _)| (*p).to_string()).collect()
        }

        fn pass_layers(&self, pass: &str) -> Option<Vec<String>> {
            let (_, layers) = self.0.iter().find(|(p, _)| *p == pass)?;
            Some(layers.iter().map(name_of).collect())
        }

        fn layer(&self, name: &str) -> Option<Value> {
            self.0
                .iter()
                .flat_map(|(_, l)| l)
                .find(|l| l["name"] == name)
                .cloned()
        }

        /// A key left out keeps the layer's `x`.
        fn filled(&self, look: &Value) -> Value {
            let mut look = look.clone();
            let x = self
                .layer(&name_of(&look))
                .and_then(|l| l.get("x").cloned());
            if let (Some(x), None) = (x, look.get("x")) {
                look["x"] = x;
            }
            look
        }
    }

    fn draft_of(ops: Value) -> Result<Draft, Refusal> {
        draft(&scene(), &ops_of(&ops.to_string()).unwrap())
    }

    fn refusal(by: usize, why: &str) -> Result<(), Refusal> {
        Err(Refusal {
            by: Some(by),
            why: why.to_string(),
        })
    }

    #[test]
    fn a_patch_that_fails_at_any_operation_changes_nothing() {
        let failed = draft_of(json!([
            {"op": "move", "from": "/main/2", "path": "/main/0"},
            {"op": "test", "path": "/main/0/name", "value": "WRONG"},
        ]));
        assert_eq!(failed.map(drop), refusal(1, r#"the test found "floor""#));
        let ops = ops_of(r#"[{"op": "test", "path": "/main/0/name", "value": "WRONG"}]"#).unwrap();
        let line = Refusal {
            by: Some(0),
            why: "w".to_string(),
        }
        .line(&ops);
        assert_eq!(line, "kind 6 operation 0 at '/main/0/name': w");
    }

    #[test]
    fn replace_and_remove_need_the_key_and_add_needs_its_layer_or_effect() {
        for (op, why) in [
            (
                json!({"op": "replace", "path": "/main/0/opacity", "value": 0.5}),
                "layer 'title' holds no 'opacity'",
            ),
            (
                json!({"op": "remove", "path": "/main/0/y"}),
                "layer 'title' holds no 'y'",
            ),
            (
                json!({"op": "replace", "path": "/main/1/glow/radius", "value": 1}),
                "layer 'dot' holds no 'glow/radius'",
            ),
            (
                json!({"op": "add", "path": "/main/0/glow/strength", "value": 1}),
                "layer 'title' holds no 'glow'",
            ),
            (
                json!({"op": "remove", "path": "/main/3"}),
                "pass 'main' has no layer there",
            ),
            (
                json!({"op": "add", "path": "/main/4", "value": {"name": "n"}}),
                "pass 'main' has no place 4",
            ),
            (
                json!({"op": "copy", "from": "/main/1/x", "path": "/main/0/x"}),
                "from '/main/1/x': layer 'dot' holds no 'x'",
            ),
            (
                json!({"op": "replace", "path": "/main/0/name", "value": "t"}),
                "layer 'title' keeps its name",
            ),
            (
                json!({"op": "move", "from": "/main/0", "path": "/top/0"}),
                "a layer moves within its own pass",
            ),
            (
                json!({"op": "remove", "path": ""}),
                "the whole document stays",
            ),
            (
                json!({"op": "remove", "path": "/top"}),
                "pass 'top' stays in the scene",
            ),
        ] {
            assert_eq!(draft_of(json!([op])).map(drop), refusal(0, why), "{op}");
        }
        let d = draft_of(json!([
            {"op": "add", "path": "/main/0/opacity", "value": 0.5},
            {"op": "replace", "path": "/main/0/opacity", "value": 0.25},
            {"op": "add", "path": "/main/1/glow/strength", "value": null},
            {"op": "test", "path": "/main/1/glow", "value": {}},
            {"op": "remove", "path": "/main/0/x"},
            {"op": "replace", "path": "/main/0/name", "value": "title"},
        ]))
        .unwrap();
        let key = |by, layer: &str, keys: &[&str], value| Step::Key {
            by,
            layer: layer.to_string(),
            keys: keys.iter().map(|k| (*k).to_string()).collect(),
            value,
        };
        assert_eq!(
            d.steps,
            [
                key(0, "title", &["opacity"], json!(0.5)),
                key(1, "title", &["opacity"], json!(0.25)),
                key(2, "dot", &["glow", "strength"], Value::Null),
                key(4, "title", &["x"], Value::Null),
            ]
        );
        assert_eq!(d.changes(), 4);
    }

    #[test]
    fn names_are_checked_on_the_result() {
        let copied = json!({"op": "copy", "from": "/main/1", "path": "/main/-"});
        assert_eq!(
            draft_of(json!([copied])).map(drop),
            refusal(0, "the result declares layer 'dot' twice")
        );
        assert_eq!(
            draft_of(json!([{"op": "add", "path": "/main/0", "value": {"name": "tag"}}])).map(drop),
            refusal(0, "the result declares layer 'tag' twice")
        );
        let d = draft_of(json!([copied, {"op": "remove", "path": "/main/1"}])).unwrap();
        assert_eq!(d.removed, [("dot".to_string(), 1)]);
        let main = &d.passes[0];
        assert_eq!(
            main.after.iter().map(|e| e.who.clone()).collect::<Vec<_>>(),
            [
                Who::Own("title".to_string()),
                Who::Own("floor".to_string()),
                Who::New
            ]
        );
        assert_eq!(
            main.after[2].new_look(),
            Some(&json!({"name": "dot", "shape": "circle", "glow": {"strength": 0.5}}))
        );
        // a copy takes a new name before the result
        let renamed = draft_of(json!([
            copied,
            {"op": "replace", "path": "/main/3/name", "value": "dot2"},
        ]))
        .unwrap();
        assert_eq!(
            renamed.passes[0].after[3].new_look().unwrap()["name"],
            "dot2"
        );
    }

    #[test]
    fn the_empty_pointer_is_the_whole_document_and_a_pass_its_layers() {
        let d = draft_of(json!([
            {"op": "test", "path": "", "value": {
                "main": [{"name": "title", "text": "hi", "x": 2},
                         {"name": "dot", "shape": "circle", "glow": {"strength": 0.5}},
                         {"name": "floor", "shape": "rect"}],
                "top": [{"name": "tag", "text": "t"}]}},
            {"op": "replace", "path": "/main", "value": [
                {"name": "floor", "shape": "rect", "y": 1},
                {"name": "new", "shape": "rect"}]},
            {"op": "test", "path": "/main/0", "value": {"name": "floor", "shape": "rect", "y": 1}},
        ]))
        .unwrap();
        assert_eq!(
            d.removed,
            [("title".to_string(), 1), ("dot".to_string(), 1)]
        );
        assert_eq!(
            d.steps,
            [Step::Look {
                by: 1,
                layer: "floor".to_string(),
                look: json!({"name": "floor", "shape": "rect", "y": 1}),
            }]
        );
        assert_eq!(
            draft_of(json!([{"op": "replace", "path": "", "value": {"main": []}}])).map(drop),
            refusal(0, "pass 'top' stays in the scene")
        );
    }

    #[test]
    fn a_replaced_layer_reads_back_filled_and_named() {
        let d = draft_of(json!([
            {"op": "replace", "path": "/main/0", "value": {"text": "ho"}},
            {"op": "test", "path": "/main/0", "value": {"name": "title", "text": "ho", "x": 2}},
            {"op": "move", "from": "/main/0", "path": "/main/-"},
        ]))
        .unwrap();
        assert!(!d.passes[0].kept_order());
        assert_eq!(d.passes[0].after[2].by, Some(2));
    }
}
