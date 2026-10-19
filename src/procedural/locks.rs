//! What a scene's program sets, which a console locks, and what a console
//! changes, which the program's next look of a layer keeps.
//!
//! A program is a scene process or a host of the console. Each param and
//! each key of a layer that it sends locks in the console: a key of a look
//! or of a call, or a live field. An effect sent whole is one key. A look or
//! a call from the program that leaves out a key that a console changed
//! keeps the console's value of it. Any other key left out takes the value
//! that the scene declares for a layer of its own, else its default.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::draw::{Arg, Call};
use crate::layer::Address;
use crate::scene::short_floats;

/// The param `name` of pass `pass`, None for a scene param, as Locks names
/// it.
fn param_name(pass: Option<&str>, name: &str) -> String {
    match pass {
        Some(pass) => format!("{pass}.{name}"),
        None => name.to_string(),
    }
}

/// The live field `at` as a message names it.
fn dotted(at: &Address) -> String {
    match at.effect {
        Some(effect) => format!("{}.{effect}.{}", at.layer, at.field),
        None => format!("{}.{}", at.layer, at.field),
    }
}

/// The keys that `look`, a layer as a scene declares it, gives its layer:
/// each key but its `name` and its `shape`, an effect as one key.
pub fn look_keys(look: &Value) -> Vec<&str> {
    look.as_object()
        .into_iter()
        .flatten()
        .map(|(k, _)| k.as_str())
        .filter(|k| !matches!(*k, "name" | "shape"))
        .collect()
}

/// The keys that `call` gives its layer, an effect as one key.
pub fn call_keys(call: &Call) -> Vec<&'static str> {
    call.kind
        .keys()
        .iter()
        .zip(&call.args)
        .filter(|(_, a)| **a != Arg::None)
        .map(|(k, _)| *k)
        .collect()
}

/// What the program of a scene set: each param, a scene param by its name
/// and a param of a pass as `pass.name`, each field of a layer as
/// `layer.field` or `layer.effect.field`, and each effect whole as
/// `layer.effect`. A console refuses to set what the program set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Locks {
    params: HashSet<String>,
    fields: HashSet<String>,
}

impl Locks {
    /// The program set the param `name` of pass `pass`, None for a scene
    /// param.
    pub fn set_param(&mut self, pass: Option<&str>, name: &str) {
        self.params.insert(param_name(pass, name));
    }

    /// The program set the field `at`, or the effect that `at.field` names
    /// when `at` has no effect.
    pub fn set_field(&mut self, at: &Address) {
        self.fields.insert(dotted(at));
    }

    /// The program gave layer `layer` the keys `keys`, each a field or an
    /// effect.
    pub fn set_keys<'a>(&mut self, layer: &str, keys: impl IntoIterator<Item = &'a str>) {
        for field in keys {
            self.set_field(&Address {
                layer,
                effect: None,
                field,
            });
        }
    }

    /// Whether the program set the param `name` of pass `pass`, None for a
    /// scene param.
    pub fn param(&self, pass: Option<&str>, name: &str) -> bool {
        self.params.contains(&param_name(pass, name))
    }

    /// Whether the program set the field `at`, or its whole effect.
    pub fn field(&self, at: &Address) -> bool {
        self.field_set(at).is_some()
    }

    /// The name of what the program set that holds the field `at`: the
    /// field or its whole effect. None when the program set neither.
    fn field_set(&self, at: &Address) -> Option<String> {
        let whole = at.effect.map(|e| format!("{}.{e}", at.layer));
        [Some(dotted(at)), whole]
            .into_iter()
            .flatten()
            .find(|f| self.fields.contains(f))
    }

    /// Whether the program set the effect `effect` of layer `layer` whole,
    /// or a field of it.
    pub fn effect(&self, layer: &str, effect: &str) -> bool {
        let whole = format!("{layer}.{effect}");
        let at = format!("{whole}.");
        self.fields.contains(&whole) || self.fields.iter().any(|f| f.starts_with(&at))
    }

    /// Why a console refuses to remove the effect `effect` of layer
    /// `layer`, or None when the program set neither it nor a field of it.
    pub fn effect_refusal(&self, layer: &str, effect: &str) -> Option<String> {
        let whole = format!("{layer}.{effect}");
        let at = format!("{whole}.");
        let mut set: Vec<&String> = self
            .fields
            .iter()
            .filter(|f| **f == whole || f.starts_with(&at))
            .collect();
        set.sort();
        set.first()
            .map(|f| format!("the scene's program sets layer field '{f}'"))
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

    /// Why a console refuses to set the field `at`, or None when the
    /// program never set it.
    pub fn field_refusal(&self, at: &Address) -> Option<String> {
        self.field_set(at)
            .map(|f| format!("the scene's program sets layer field '{f}'"))
    }

    /// `{params, fields}`, each a sorted list of names.
    pub fn to_json(&self) -> Value {
        let sorted = |set: &HashSet<String>| {
            let mut names: Vec<&String> = set.iter().collect();
            names.sort();
            Value::from(names.into_iter().cloned().collect::<Vec<_>>())
        };
        serde_json::json!({
            "params": sorted(&self.params),
            "fields": sorted(&self.fields),
        })
    }
}

/// The keys of each layer that a console changed, a field or an effect by
/// its name.
#[derive(Clone, Debug, Default)]
pub struct Tuned {
    layers: HashMap<String, HashSet<String>>,
}

/// The kind of the layer `look`, as a scene declares it: its shape, and
/// whether it shows an image.
fn kind_of(look: &Value) -> (Option<&Value>, bool) {
    (look.get("shape"), look.get("image").is_some())
}

impl Tuned {
    /// A console set the field `at`.
    pub fn set(&mut self, at: &Address) {
        let key = at.effect.unwrap_or(at.field);
        self.layers
            .entry(at.layer.to_string())
            .or_default()
            .insert(key.to_string());
    }

    /// Layer `layer` left the scene.
    pub fn forget(&mut self, layer: &str) {
        self.layers.remove(layer);
    }

    /// `look`, the look that the program gives a layer, with each key that
    /// it leaves out and a console changed at its value in `now`, the layer
    /// as it stands, read by name. None when `look` leaves out no such key,
    /// or when it gives the layer another kind.
    pub fn look(&self, look: &Value, now: impl FnOnce(&str) -> Option<Value>) -> Option<Value> {
        let name = look.get("name")?.as_str()?;
        let tuned = self.layers.get(name)?;
        let now = now(name)?;
        if kind_of(&now) != kind_of(look) {
            return None;
        }
        let mut kept = look.clone();
        let into = kept.as_object_mut()?;
        let mut any = false;
        for key in tuned {
            if let (false, Some(v)) = (into.contains_key(key), now.get(key)) {
                into.insert(key.clone(), v.clone());
                any = true;
            }
        }
        any.then_some(kept)
    }

    /// `calls` with each key that a call leaves out and a console changed at
    /// its value in `now`, the layer as it stands, read by name. None when
    /// no call leaves out such a key.
    pub fn calls(&self, calls: &[Call], now: impl Fn(&str) -> Option<Value>) -> Option<Vec<Call>> {
        let kept: Vec<Option<Call>> = calls.iter().map(|c| self.call(c, &now)).collect();
        kept.iter().any(Option::is_some).then(|| {
            calls
                .iter()
                .zip(kept)
                .map(|(c, k)| k.unwrap_or_else(|| c.clone()))
                .collect()
        })
    }

    /// `call` with each key that it leaves out and a console changed at its
    /// value in `now`, the layer as it stands, read by name. None when
    /// `call` leaves out no such key.
    fn call(&self, call: &Call, now: impl FnOnce(&str) -> Option<Value>) -> Option<Call> {
        let tuned = self.layers.get(&call.layer)?;
        let now = now(&call.layer)?;
        let mut kept = call.clone();
        let mut any = false;
        for (key, arg) in call.kind.keys().iter().zip(&mut kept.args) {
            if let (Arg::None, true, Some(v)) = (&*arg, tuned.contains(*key), now.get(*key)) {
                *arg = Arg::of_json(v);
                any = true;
            }
        }
        any.then_some(kept)
    }

    /// `look`, the look that a console gives a layer whole, with each key
    /// that `locks` holds at its value in `now`, the layer as it stands.
    /// Each other key whose value differs from its value in `now` becomes a
    /// key that the console changed.
    pub fn applied(&mut self, mut look: Value, now: &Value, locks: &Locks) -> Value {
        let Some(name) = look.get("name").and_then(Value::as_str).map(str::to_string) else {
            return look;
        };
        let (Some(now), Some(into)) = (now.as_object(), look.as_object_mut()) else {
            return look;
        };
        let mut keys: Vec<String> = now.keys().chain(into.keys()).cloned().collect();
        keys.sort_unstable();
        keys.dedup();
        let mut changed = Vec::new();
        for key in keys {
            let at = Address {
                layer: &name,
                effect: None,
                field: &key,
            };
            if locks.field(&at) {
                match now.get(&key) {
                    Some(v) => into.insert(key, v.clone()),
                    None => into.remove(&key),
                };
                continue;
            }
            let (mut a, mut b) = (now.get(&key).cloned(), into.get(&key).cloned());
            a.iter_mut().chain(b.iter_mut()).for_each(short_floats);
            if a != b {
                changed.push(key);
            }
        }
        if !changed.is_empty() {
            self.layers.entry(name).or_default().extend(changed);
        }
        look
    }
}
