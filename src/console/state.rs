//! A scene's edits, and the state file that keeps them.
//!
//! The console edits a working copy of each scene: the scene's defaults with
//! every edit made since. Its edits are what differs from the defaults, in two
//! parts. The values are a JSON merge patch (RFC 7386) over the scene seen by
//! name: `params` holds the scene params' values, `passes` each pass's
//! `params`, `visible`, `size` and `scale`, `layers` each layer's fields and
//! effects, and `assets` each asset whole. The items are the code and
//! structure edits, each whole, with the hash of the same item in the
//! defaults it was made against.
//!
//! A load applies the items, then the values, over the defaults. An item whose
//! default changed since its edit is set aside, and the default runs until the
//! console answers. A value or an item whose pass, layer or param is gone is
//! skipped. Each of these logs one line.
//!
//! Both sides of a diff are scene documents as Scene::to_dict() writes them,
//! with each GLSL file as its text and the filters expanded.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::layer::Layer;
use crate::scene::{py_repr, Asset, Param, Pass};

/// The format of a state file.
pub const VERSION: i64 = 1;

/// The kinds of item, in the order a load applies them.
pub const KINDS: [&str; 7] = [
    "passes", "layers", "fragment", "channels", "common", "param", "filters",
];

/// One code or structure edit.
#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    /// One of KINDS.
    pub item: String,
    pub pass: Option<String>,
    pub param: Option<String>,
    /// The first 16 hex digits of the SHA-256 of the same item in the
    /// defaults that the edit was made against. None when they had none.
    pub base: Option<String>,
    pub value: Value,
}

impl Item {
    fn new(item: &str, pass: Option<&str>, base: Option<String>, value: Value) -> Item {
        Item {
            item: item.to_string(),
            pass: pass.map(str::to_string),
            param: None,
            base,
            value,
        }
    }

    /// Whether this is the item that `item`, `pass` and `param` name.
    pub fn is(&self, item: &str, pass: Option<&str>, param: Option<&str>) -> bool {
        self.item == item && self.pass.as_deref() == pass && self.param.as_deref() == param
    }

    /// `{item, pass, param}`, with the keys its kind takes.
    pub fn key(&self) -> Value {
        let mut d = Map::new();
        d.insert("item".to_string(), Value::from(self.item.as_str()));
        if let Some(p) = &self.pass {
            d.insert("pass".to_string(), Value::from(p.as_str()));
        }
        if let Some(p) = &self.param {
            d.insert("param".to_string(), Value::from(p.as_str()));
        }
        Value::Object(d)
    }

    pub fn to_json(&self) -> Value {
        let Value::Object(mut d) = self.key() else {
            unreachable!("key() is an object")
        };
        d.insert(
            "base".to_string(),
            self.base.clone().map_or(Value::Null, Value::from),
        );
        d.insert("value".to_string(), self.value.clone());
        Value::Object(d)
    }

    fn from_json(v: &Value) -> Result<Item, String> {
        let text = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
        let item = text("item").filter(|k| KINDS.contains(&k.as_str()));
        let Some(item) = item else {
            return Err(format!("an item needs an 'item' of {}", KINDS.join(", ")));
        };
        let base = match v.get("base") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(b) => return Err(format!("item {item}: 'base' {} is no hash", py_repr(b))),
        };
        Ok(Item {
            item,
            pass: text("pass"),
            param: text("param"),
            base,
            value: v.get("value").cloned().unwrap_or(Value::Null),
        })
    }

    /// What the item edits, for a log line.
    fn what(&self) -> String {
        let pass = self.pass.as_deref().unwrap_or_default();
        let param = self.param.as_deref().unwrap_or_default();
        match self.item.as_str() {
            "fragment" => format!("pass '{pass}''s shader"),
            "channels" => format!("pass '{pass}''s channels"),
            "layers" => format!("pass '{pass}''s layers"),
            "common" => "the common shader".to_string(),
            "passes" => "the passes".to_string(),
            "filters" => "the filters".to_string(),
            _ if self.pass.is_some() => format!("pass '{pass}''s param '{param}'"),
            _ => format!("param '{param}'"),
        }
    }
}

/// A scene's edits: the values and the items.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Edits {
    pub values: Map<String, Value>,
    pub items: Vec<Item>,
}

impl Edits {
    pub fn is_empty(&self) -> bool {
        self.values.is_empty() && self.items.is_empty()
    }

    /// The same edits as `other`, the items in any order, with numbers
    /// compared as 32-bit floats, which is how a scene holds them. No two
    /// items of one Edits are the same item.
    pub fn same(&self, other: &Edits) -> bool {
        same(
            &Value::Object(self.values.clone()),
            &Value::Object(other.values.clone()),
        ) && self.items.len() == other.items.len()
            && self.items.iter().all(|a| {
                other.items.iter().any(|b| {
                    a.is(&b.item, b.pass.as_deref(), b.param.as_deref())
                        && a.base == b.base
                        && same(&a.value, &b.value)
                })
            })
    }

    /// The state file's document for the scene file `scene`.
    pub fn to_file(&self, scene: &str) -> Value {
        let mut d = Map::new();
        d.insert("version".to_string(), Value::from(VERSION));
        d.insert("scene".to_string(), Value::from(scene));
        if !self.values.is_empty() {
            d.insert("values".to_string(), Value::Object(self.values.clone()));
        }
        if !self.items.is_empty() {
            d.insert(
                "items".to_string(),
                self.items.iter().map(Item::to_json).collect(),
            );
        }
        Value::Object(d)
    }

    /// Each edit alone: each asset, which a layer of an item may name, each
    /// item, then the value of each scene param, the values of each pass and
    /// those of each layer.
    pub fn units(&self) -> Vec<Edits> {
        let mut out = Vec::new();
        let (assets, rest): (Vec<_>, Vec<_>) =
            self.values.iter().partition(|(k, _)| *k == "assets");
        let values = |pairs: Vec<(&String, &Value)>, out: &mut Vec<Edits>| {
            for (key, v) in pairs {
                let Value::Object(m) = v else {
                    out.push(Edits {
                        values: Map::from_iter([(key.clone(), v.clone())]),
                        items: Vec::new(),
                    });
                    continue;
                };
                for (name, v) in m {
                    let one = Map::from_iter([(name.clone(), v.clone())]);
                    out.push(Edits {
                        values: Map::from_iter([(key.clone(), Value::Object(one))]),
                        items: Vec::new(),
                    });
                }
            }
        };
        values(assets, &mut out);
        out.extend(self.items.iter().map(|i| Edits {
            values: Map::new(),
            items: vec![i.clone()],
        }));
        values(rest, &mut out);
        out
    }

    /// What one edit of units() edits, for a person to read.
    pub fn what(&self) -> String {
        if let Some(item) = self.items.first() {
            return item.what();
        }
        let Some((key, v)) = self.values.iter().next() else {
            return "nothing".to_string();
        };
        let name = v
            .as_object()
            .and_then(|m| m.keys().next())
            .map_or("", String::as_str);
        match key.as_str() {
            "params" => format!("param '{name}'"),
            "passes" => format!("pass '{name}'"),
            "layers" => format!("layer '{name}'"),
            "assets" => format!("asset '{name}'"),
            _ => format!("the values' '{key}'"),
        }
    }

    /// Lay `under` beneath these edits: each of its items that these edits
    /// do not hold, and each of its values that these edits do not set.
    pub fn add_beneath(&mut self, under: &Edits) {
        for item in &under.items {
            let mine = |i: &Item| i.is(&item.item, item.pass.as_deref(), item.param.as_deref());
            if !self.items.iter().any(mine) {
                self.items.push(item.clone());
            }
        }
        self.items
            .sort_by_key(|i| KINDS.iter().position(|k| *k == i.item));
        let mut values = under.values.clone();
        compose(&mut values, &self.values);
        self.values = values;
    }

    /// Take out each item and each value that `unit`, one edit of units(),
    /// holds.
    pub fn remove(&mut self, unit: &Edits) {
        self.items.retain(|i| {
            !unit
                .items
                .iter()
                .any(|u| i.is(&u.item, u.pass.as_deref(), u.param.as_deref()))
        });
        for (key, v) in &unit.values {
            let Value::Object(names) = v else {
                self.values.remove(key);
                continue;
            };
            if let Some(Value::Object(m)) = self.values.get_mut(key) {
                for name in names.keys() {
                    m.remove(name);
                }
                if m.is_empty() {
                    self.values.remove(key);
                }
            }
        }
    }

    pub fn from_file(d: &Value) -> Result<Edits, String> {
        match d.get("version") {
            Some(v) if v.as_i64() == Some(VERSION) => {}
            v => {
                return Err(format!(
                    "version {} is not {VERSION}",
                    v.map_or("none".to_string(), py_repr)
                ))
            }
        }
        let values = match d.get("values") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(o)) => o.clone(),
            Some(v) => return Err(format!("'values' needs an object, got {}", py_repr(v))),
        };
        let items = match d.get("items") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(a)) => a.iter().map(Item::from_json).collect::<Result<_, _>>()?,
            Some(v) => return Err(format!("'items' needs a list, got {}", py_repr(v))),
        };
        Ok(Edits { values, items })
    }
}

// ---------------------------------------------------------------- files

/// The state directory when no other is given: `$XDG_STATE_HOME/ghee`, else
/// `~/.local/state/ghee`. None when neither variable is set.
pub fn default_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_STATE_HOME")
        .filter(|d| Path::new(d).is_absolute())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))?;
    Some(base.join("ghee"))
}

/// The state file of the scene file `scene`, an absolute path, in the state
/// directory `dir`: the scene's path under `scenes/`, with `.json` in place
/// of the scene file's extension. A bundled demo's sits under `demo/`, by the
/// scene's name.
pub fn file_of(dir: &Path, scene: &str, demo: Option<&str>) -> PathBuf {
    if let Some(name) = demo {
        return dir.join("demo").join(format!("{name}.json"));
    }
    let rel = scene.trim_start_matches('/');
    let stem = match rel.strip_suffix(crate::procedural::SUFFIX) {
        Some(s) => s.to_string(),
        None => {
            let p = Path::new(rel);
            match p.extension() {
                Some(_) => p.with_extension("").to_string_lossy().into_owned(),
                None => rel.to_string(),
            }
        }
    };
    dir.join("scenes").join(format!("{stem}.json"))
}

/// The draft beside the state file `path`: `clocks.draft.json` beside
/// `clocks.json`.
pub fn draft_of(path: &Path) -> PathBuf {
    path.with_extension("draft.json")
}

/// The edits that the state file `path` holds. None when there is no file.
pub fn read(path: &Path) -> Result<Option<Edits>, String> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let d: Value = serde_json::from_str(&text)
        .map_err(|e| format!("{}: invalid JSON: {e}", path.display()))?;
    Edits::from_file(&d)
        .map(Some)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Move the state file `path`, which read() refused, to `<path>.unreadable`
/// beside it, or to `<path>.unreadable.2` and on when that name is taken, so
/// the next write leaves it whole. Returns where it went.
pub fn move_unreadable(path: &Path) -> Result<PathBuf, String> {
    let mut n = 1;
    let to = loop {
        let mut to = path.as_os_str().to_owned();
        to.push(".unreadable");
        if n > 1 {
            to.push(format!(".{n}"));
        }
        let to = PathBuf::from(to);
        if fs::symlink_metadata(&to).is_err() {
            break to;
        }
        n += 1;
    };
    fs::rename(path, &to).map_err(|e| format!("{}: {e}", to.display()))?;
    Ok(to)
}

/// Write `edits` of the scene file `scene` to the state file `path`, whole,
/// through a temporary file and a rename. No edit removes the file.
pub fn write(path: &Path, scene: &str, edits: &Edits) -> Result<(), String> {
    if edits.is_empty() {
        return remove(path);
    }
    write_draft(path, scene, edits)
}

/// Write `edits` of the scene file `scene` to the draft `path`, whole, in the
/// state file's format. A draft with no edit stays a file: it takes the
/// scene back to its defaults.
pub fn write_draft(path: &Path, scene: &str, edits: &Edits) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{}: {e}", path.display());
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(io)?;
    }
    let text = serde_json::to_string_pretty(&edits.to_file(scene)).map_err(|e| e.to_string())?;
    write_file(path, (text + "\n").as_bytes()).map_err(io)
}

/// Remove the file `path`, which may be gone already.
pub fn remove(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            Err(format!("{}: {e}", path.display()))
        }
        _ => Ok(()),
    }
}

/// Write `bytes` to `path` whole, through a temporary file beside it and a
/// rename.
pub fn write_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(".{}.tmp", std::process::id()));
    // The rename takes only bytes that reached the disk, so a power cut
    // leaves the old file or the new one, whole.
    let wrote = fs::File::create(&tmp)
        .and_then(|mut f| {
            f.write_all(bytes)?;
            f.sync_all()
        })
        .and_then(|()| fs::rename(&tmp, path));
    if wrote.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    wrote
}

// ---------------------------------------------------------------- the doc

/// The first 16 hex digits of the SHA-256 of `bytes`.
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A GLSL item's hash: its text as UTF-8.
pub fn hash_text(text: &str) -> String {
    hash(text.as_bytes())
}

/// A list's hash: its compact JSON.
pub fn hash_json(v: &Value) -> String {
    hash(v.to_string().as_bytes())
}

/// JSON equality, with each number compared as the 32-bit float that a
/// scene holds.
pub fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            x.as_f64().map(|v| v as f32) == y.as_f64().map(|v| v as f32)
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| same(a, b))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w)))
        }
        _ => a == b,
    }
}

fn same_opt(a: Option<&Value>, b: Option<&Value>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => same(a, b),
        (None, None) => true,
        _ => false,
    }
}

/// The merge patch that takes `from` to `to`, two objects.
fn patch_of(from: &Map<String, Value>, to: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::new();
    for (k, v) in to {
        match (from.get(k), v) {
            (Some(Value::Object(a)), Value::Object(b)) => {
                let inner = patch_of(a, b);
                if !inner.is_empty() {
                    out.insert(k.clone(), Value::Object(inner));
                }
            }
            (Some(a), b) if same(a, b) => {}
            _ => {
                out.insert(k.clone(), v.clone());
            }
        }
    }
    for k in from.keys().filter(|k| !to.contains_key(*k)) {
        out.insert(k.clone(), Value::Null);
    }
    out
}

/// Lay the merge patch `over` on the merge patch `under`, so that `under`
/// patches what both did, and `over` wins a key that both set. A null of
/// `over` stays, since it removes its key.
fn compose(under: &mut Map<String, Value>, over: &Map<String, Value>) {
    for (k, v) in over {
        match (under.get_mut(k), v) {
            (Some(Value::Object(u)), Value::Object(o)) => compose(u, o),
            _ => {
                under.insert(k.clone(), v.clone());
            }
        }
    }
}

/// RFC 7386: lay `patch` over `target`.
fn merge(target: &mut Value, patch: &Value) {
    let Value::Object(p) = patch else {
        *target = patch.clone();
        return;
    };
    if !target.is_object() {
        *target = Value::Object(Map::new());
    }
    let Value::Object(t) = target else {
        unreachable!("made an object above")
    };
    for (k, v) in p {
        if v.is_null() {
            t.remove(k);
        } else {
            merge(t.entry(k.clone()).or_insert(Value::Null), v);
        }
    }
}

fn passes(doc: &Value) -> &[Value] {
    doc.get("passes")
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

fn name_of(v: &Value) -> &str {
    v.get("name").and_then(Value::as_str).unwrap_or_default()
}

fn is_filter(pass: &Value) -> bool {
    pass.get("filter")
        .and_then(Value::as_str)
        .is_some_and(|f| !f.is_empty())
}

/// The scene's own passes: every pass but a filter's.
fn own(doc: &Value) -> impl Iterator<Item = &Value> {
    passes(doc).iter().filter(|p| !is_filter(p))
}

fn pass<'a>(doc: &'a Value, name: &str) -> Option<&'a Value> {
    passes(doc).iter().find(|p| name_of(p) == name)
}

fn own_pass<'a>(doc: &'a Value, name: &str) -> Option<&'a Value> {
    own(doc).find(|p| name_of(p) == name)
}

fn pass_mut<'a>(doc: &'a mut Value, name: &str) -> Option<&'a mut Map<String, Value>> {
    doc.get_mut("passes")?
        .as_array_mut()?
        .iter_mut()
        .find(|p| name_of(p) == name)?
        .as_object_mut()
}

fn layers(pass: &Value) -> &[Value] {
    pass.get("layers")
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

fn layer<'a>(doc: &'a Value, name: &str) -> Option<&'a Value> {
    passes(doc)
        .iter()
        .flat_map(layers)
        .find(|l| name_of(l) == name)
}

fn names(list: &[Value]) -> Value {
    list.iter().map(|v| Value::from(name_of(v))).collect()
}

fn own_names(doc: &Value) -> Value {
    own(doc).map(|p| Value::from(name_of(p))).collect()
}

/// The filters that the scene ends with, by the filter's name, in order.
fn filter_list(doc: &Value) -> Value {
    passes(doc)
        .iter()
        .filter(|p| is_filter(p))
        .map(|p| p["filter"].clone())
        .collect()
}

fn assets(doc: &Value) -> Option<&Map<String, Value>> {
    doc.get("assets").and_then(Value::as_object)
}

fn params(of: &Value) -> Option<&Map<String, Value>> {
    of.get("params").and_then(Value::as_object)
}

fn fragment(pass: &Value) -> &str {
    pass.get("fragment")
        .and_then(Value::as_str)
        .unwrap_or_default()
}

fn channels(pass: &Value) -> Value {
    pass.get("channels")
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()))
}

/// The hash of the item in `doc` that `item` names, as its base reads it.
/// Outer None when the item's pass is gone.
pub fn base_of(doc: &Value, item: &Item) -> Option<Option<String>> {
    let pass = item.pass.as_deref();
    let of_pass = || pass.and_then(|p| own_pass(doc, p));
    Some(match item.item.as_str() {
        "passes" => Some(hash_json(&own_names(doc))),
        "filters" => Some(hash_json(&filter_list(doc))),
        "common" => doc.get("common").and_then(Value::as_str).map(hash_text),
        "fragment" => Some(hash_text(fragment(of_pass()?))),
        "channels" => Some(hash_json(&channels(of_pass()?))),
        "layers" => Some(hash_json(&names(layers(of_pass()?)))),
        _ => {
            let owner = match pass {
                Some(_) => of_pass()?,
                None => doc,
            };
            let param = item.param.as_deref().unwrap_or_default();
            params(owner).and_then(|p| p.get(param)).map(hash_json)
        }
    })
}

/// The edits that take the scene document `defaults` to `working`. A
/// procedural scene's program keeps which passes, layers and params exist,
/// so its edits hold no list of them and no added param.
pub fn diff(defaults: &Value, working: &Value, procedural: bool) -> Edits {
    let d = defaults;
    let w = working;
    let mut items = Vec::new();
    if !procedural {
        let (dn, wn) = (own_names(d), own_names(w));
        if !same(&dn, &wn) {
            let had = |n: &str| own_pass(d, n).is_some();
            let value = own(w)
                .map(|p| match had(name_of(p)) {
                    true => Value::from(name_of(p)),
                    false => p.clone(),
                })
                .collect();
            items.push(Item::new("passes", None, Some(hash_json(&dn)), value));
        }
    }
    let mut pass_values = Map::new();
    for wp in passes(w) {
        let name = name_of(wp);
        let Some(dp) = pass(d, name).filter(|dp| is_filter(dp) == is_filter(wp)) else {
            continue;
        };
        let at = Some(name);
        if !is_filter(wp) {
            if fragment(dp) != fragment(wp) {
                let value = Value::from(fragment(wp));
                items.push(Item::new(
                    "fragment",
                    at,
                    Some(hash_text(fragment(dp))),
                    value,
                ));
            }
            if !same(&channels(dp), &channels(wp)) {
                let base = Some(hash_json(&channels(dp)));
                items.push(Item::new("channels", at, base, channels(wp)));
            }
            let (dl, wl) = (names(layers(dp)), names(layers(wp)));
            if !procedural && dl != wl {
                let value = layers(wp)
                    .iter()
                    .map(|l| match layer(d, name_of(l)) {
                        Some(_) => Value::from(name_of(l)),
                        None => l.clone(),
                    })
                    .collect();
                items.push(Item::new("layers", at, Some(hash_json(&dl)), value));
            }
        }
        let mut values = Map::new();
        let (dps, wps) = (params(dp), params(wp));
        let mut changed = Map::new();
        for (n, wv) in wps.into_iter().flatten() {
            match dps.and_then(|p| p.get(n)) {
                Some(dv) => {
                    if !same_opt(dv.get("value"), wv.get("value")) {
                        changed.insert(n.clone(), wv["value"].clone());
                    }
                }
                None if !procedural => items.push(Item {
                    param: Some(n.clone()),
                    ..Item::new("param", at, None, wv.clone())
                }),
                None => {}
            }
        }
        if !changed.is_empty() {
            values.insert("params".to_string(), Value::Object(changed));
        }
        for (key, absent) in [
            ("visible", Value::from(true)),
            ("sampling", Value::from("linear")),
        ] {
            let value = |p: &Value| {
                p.get(key)
                    .filter(|v| !v.is_null())
                    .unwrap_or(&absent)
                    .clone()
            };
            if value(dp) != value(wp) {
                values.insert(key.to_string(), value(wp));
            }
        }
        for key in ["size", "scale"] {
            if !same_opt(dp.get(key), wp.get(key)) {
                values.insert(key.to_string(), wp.get(key).cloned().unwrap_or(Value::Null));
            }
        }
        if !values.is_empty() {
            pass_values.insert(name.to_string(), Value::Object(values));
        }
    }
    let mut layer_values = Map::new();
    for wl in passes(w).iter().flat_map(layers) {
        let name = name_of(wl);
        let (Some(Value::Object(dl)), Value::Object(wo)) = (layer(d, name), wl) else {
            continue;
        };
        let mut patch = patch_of(dl, wo);
        patch.remove("name");
        if !patch.is_empty() {
            layer_values.insert(name.to_string(), Value::Object(patch));
        }
    }
    let empty = Map::new();
    let (da, wa) = (assets(d).unwrap_or(&empty), assets(w).unwrap_or(&empty));
    let mut asset_values: Map<String, Value> = wa
        .iter()
        .filter(|(n, wv)| !da.get(*n).is_some_and(|dv| same(dv, wv)))
        .map(|(n, wv)| (n.clone(), wv.clone()))
        .collect();
    for n in da.keys().filter(|n| !wa.contains_key(*n)) {
        asset_values.insert(n.clone(), Value::Null);
    }
    let mut scene_values = Map::new();
    for (n, wv) in params(w).into_iter().flatten() {
        match params(d).and_then(|p| p.get(n)) {
            Some(dv) => {
                if !same_opt(dv.get("value"), wv.get("value")) {
                    scene_values.insert(n.clone(), wv["value"].clone());
                }
            }
            None if !procedural => items.push(Item {
                param: Some(n.clone()),
                ..Item::new("param", None, None, wv.clone())
            }),
            None => {}
        }
    }
    if !same_opt(d.get("common"), w.get("common")) {
        let base = d.get("common").and_then(Value::as_str).map(hash_text);
        let value = w.get("common").cloned().unwrap_or(Value::Null);
        items.push(Item::new("common", None, base, value));
    }
    let (df, wf) = (filter_list(d), filter_list(w));
    if df != wf {
        items.push(Item::new("filters", None, Some(hash_json(&df)), wf));
    }
    let mut values = Map::new();
    for (key, m) in [
        ("params", scene_values),
        ("passes", pass_values),
        ("layers", layer_values),
        ("assets", asset_values),
    ] {
        if !m.is_empty() {
            values.insert(key.to_string(), Value::Object(m));
        }
    }
    items.sort_by_key(|i| KINDS.iter().position(|k| *k == i.item));
    Edits { values, items }
}

/// What a load made of the defaults and the edits.
#[derive(Debug, Default)]
pub struct Applied {
    pub doc: Value,
    /// The items whose default changed since their edit.
    pub aside: Vec<Item>,
    /// One line for each item set aside and each value or item skipped.
    pub notes: Vec<String>,
}

/// Apply `edits` over `defaults`, a scene document, for the scene `scene`:
/// the items, then the values.
pub fn apply(defaults: &Value, edits: &Edits, scene: &str) -> Applied {
    lay(defaults, defaults, edits, scene, true)
}

/// Apply `edits`, whatever their bases, over `doc`, the working copy of
/// `defaults`, and skip what has no target with no line.
pub fn fold(defaults: &Value, doc: &Value, edits: &Edits) -> Value {
    lay(defaults, doc, edits, "", false).doc
}

fn lay(defaults: &Value, doc: &Value, edits: &Edits, scene: &str, check: bool) -> Applied {
    let d = defaults;
    let mut out = Applied {
        doc: doc.clone(),
        ..Applied::default()
    };
    let note = |line: String| {
        if check {
            out_note(scene, line)
        } else {
            None
        }
    };
    let mut notes = Vec::new();
    for item in &edits.items {
        let base = base_of(d, item);
        let Some(base) = base else {
            notes.extend(note(format!(
                "the scene has no pass '{}', its saved edit of {} is skipped",
                item.pass.as_deref().unwrap_or_default(),
                item.what()
            )));
            continue;
        };
        if check && base != item.base {
            notes.extend(note(format!(
                "{} changed since its saved edit, which waits for the console",
                item.what()
            )));
            out.aside.push(item.clone());
            continue;
        }
        if let Err(why) = put(d, &mut out.doc, item) {
            notes.extend(note(why));
        }
    }
    for (key, v) in &edits.values {
        let Value::Object(m) = v else { continue };
        for (name, v) in m {
            let why = match key.as_str() {
                "params" => put_param(&mut out.doc, None, name, v)
                    .err()
                    .into_iter()
                    .collect(),
                "passes" => put_pass(&mut out.doc, name, v),
                "layers" => put_layer(&mut out.doc, name, v).err().into_iter().collect(),
                "assets" => put_asset(&mut out.doc, name, v).err().into_iter().collect(),
                _ => vec![format!(
                    "the scene's values take no '{key}', which are skipped"
                )],
            };
            for line in why {
                notes.extend(note(line));
            }
        }
    }
    out.notes = notes;
    out
}

fn out_note(scene: &str, line: String) -> Option<String> {
    Some(format!("{scene}: {line}"))
}

/// Apply `item` to `doc`, with `defaults` for the default passes and layers
/// that a list names.
fn put(defaults: &Value, doc: &mut Value, item: &Item) -> Result<(), String> {
    let gone = |pass: &str| {
        format!(
            "the scene has no pass '{pass}', its saved edit of {} is skipped",
            item.what()
        )
    };
    let pass = item.pass.clone().unwrap_or_default();
    let list = |v: &Value| v.as_array().cloned().unwrap_or_default();
    match item.item.as_str() {
        "passes" => {
            let mut kept: Vec<Value> = Vec::new();
            for entry in list(&item.value) {
                match entry.as_str() {
                    Some(n) => kept.extend(own_pass(defaults, n).cloned()),
                    None => kept.push(entry),
                }
            }
            let filters = passes(doc).iter().filter(|p| is_filter(p)).cloned();
            kept.extend(filters);
            doc["passes"] = Value::Array(kept);
        }
        "filters" => {
            let kept: Vec<Value> = own(doc).cloned().collect();
            doc["passes"] = Value::Array(kept);
            doc["filters"] = Value::Array(list(&item.value));
        }
        "common" => match &item.value {
            Value::Null => {
                doc.as_object_mut().map(|o| o.remove("common"));
            }
            v => doc["common"] = v.clone(),
        },
        "fragment" | "channels" | "layers" => {
            let value = match item.item.as_str() {
                "layers" => Value::Array(
                    list(&item.value)
                        .into_iter()
                        .filter_map(|e| match e.as_str() {
                            Some(n) => layer(defaults, n).cloned(),
                            None => Some(e),
                        })
                        .collect(),
                ),
                _ => item.value.clone(),
            };
            let p = pass_mut(doc, &pass).ok_or_else(|| gone(&pass))?;
            match value {
                Value::Null => p.remove(&item.item),
                v => p.insert(item.item.clone(), v),
            };
        }
        _ => {
            let param = item.param.clone().unwrap_or_default();
            let owner = match &item.pass {
                Some(_) => pass_mut(doc, &pass).ok_or_else(|| gone(&pass))?,
                None => doc.as_object_mut().ok_or("the scene is no object")?,
            };
            let params = owner
                .entry("params")
                .or_insert_with(|| Value::Object(Map::new()));
            if let Value::Object(p) = params {
                p.insert(param, item.value.clone());
            }
        }
    }
    Ok(())
}

/// Set the value of param `name` of pass `pass`, or of the scene, in `doc`.
fn put_param(doc: &mut Value, pass: Option<&str>, name: &str, v: &Value) -> Result<(), String> {
    let (owner, where_) = match pass {
        Some(p) => (
            pass_mut(doc, p).map(|o| o.get_mut("params")),
            format!("pass '{p}'"),
        ),
        None => (
            doc.as_object_mut().map(|o| o.get_mut("params")),
            "the scene".to_string(),
        ),
    };
    let skipped = || format!("{where_} has no param '{name}', its saved value is skipped");
    let decl = owner
        .flatten()
        .and_then(|p| p.get_mut(name))
        .ok_or_else(skipped)?;
    let ty = decl.get("type").and_then(Value::as_str).unwrap_or_default();
    let numeric = ty == "float" || ty == "int";
    let refused = || {
        format!(
            "{where_}'s param '{name}' takes no saved value {}",
            py_repr(v)
        )
    };
    if (numeric && !matches!(v, Value::Number(_))) || (ty == "bool" && !v.is_boolean()) {
        return Err(refused());
    }
    let mut next = decl.clone();
    next["value"] = v.clone();
    Param::from_dict(name, &next).map_err(|e| format!("{}: {e}", refused()))?;
    *decl = next;
    Ok(())
}

/// Lay the saved values of pass `name` over it in `doc`, and say why each
/// one that does not apply is skipped.
fn put_pass(doc: &mut Value, name: &str, v: &Value) -> Vec<String> {
    let gone = || format!("the scene has no pass '{name}', its saved values are skipped");
    if pass(doc, name).is_none() {
        return vec![gone()];
    }
    let mut why = Vec::new();
    if let Some(Value::Object(ps)) = v.get("params") {
        for (n, pv) in ps {
            if let Err(e) = put_param(doc, Some(name), n, pv) {
                why.push(e);
            }
        }
    }
    let keys: Vec<(&String, &Value)> = v
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(k, _)| ["visible", "size", "scale", "sampling"].contains(&k.as_str()))
        .collect();
    if let (false, Some(p)) = (keys.is_empty(), pass_mut(doc, name)) {
        let mut next = Value::Object(p.clone());
        for (k, kv) in keys {
            merge(&mut next[k.as_str()], kv);
            if kv.is_null() {
                next.as_object_mut().map(|o| o.remove(k.as_str()));
            }
        }
        match Pass::from_dict(&next, None) {
            Ok(_) => *p = next.as_object().cloned().unwrap_or_default(),
            Err(e) => why.push(format!("pass '{name}' takes no saved values: {e}")),
        }
    }
    why
}

/// Set asset `name` of `doc` to `v`, whole, or remove it for a null.
fn put_asset(doc: &mut Value, name: &str, v: &Value) -> Result<(), String> {
    let Some(d) = doc.as_object_mut() else {
        return Err("the scene is no object".to_string());
    };
    if v.is_null() {
        if let Some(Value::Object(a)) = d.get_mut("assets") {
            a.remove(name);
            if a.is_empty() {
                d.remove("assets");
            }
        }
        return Ok(());
    }
    Asset::from_dict(name, v).map_err(|e| format!("asset '{name}' takes no saved value: {e}"))?;
    let assets = d
        .entry("assets")
        .or_insert_with(|| Value::Object(Map::new()));
    if let Value::Object(a) = assets {
        a.insert(name.to_string(), v.clone());
    }
    Ok(())
}

/// Lay the saved fields and effects of layer `name` over it in `doc`.
fn put_layer(doc: &mut Value, name: &str, v: &Value) -> Result<(), String> {
    let found = doc
        .get_mut("passes")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(|p| p.get_mut("layers")?.as_array_mut())
        .flatten()
        .find(|l| name_of(l) == name);
    let Some(l) = found else {
        return Err(format!(
            "the scene has no layer '{name}', its saved values are skipped"
        ));
    };
    let mut next = l.clone();
    merge(&mut next, v);
    next["name"] = Value::from(name);
    Layer::from_dict("a layer", &next)
        .map_err(|e| format!("layer '{name}' takes no saved values: {e}"))?;
    *l = next;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The default shader of clocks. Its first 16 hex digits of SHA-256,
    /// from sha256sum, are 758cc820f81e647b.
    const SHADER: &str = "void main(){ gl_FragColor = vec4(1.0); }";

    /// A scene clocks: one pass main with the params cols and rows.
    fn clocks() -> Value {
        json!({"version": 1, "passes": [{"name": "main", "fragment": SHADER,
            "params": {"cols": {"type": "int", "value": 7},
                       "rows": {"type": "int", "value": 13}}}]})
    }

    #[test]
    fn a_state_file_mirrors_the_scene_files_path() {
        let dir = Path::new("/home/user/.local/state/ghee");
        for (scene, demo, want) in [
            (
                "/home/user/scenes/clocks/clocks.scene.json",
                None,
                "scenes/home/user/scenes/clocks/clocks.json",
            ),
            ("/srv/stage/stage.py", None, "scenes/srv/stage/stage.json"),
            ("/srv/stage/run", None, "scenes/srv/stage/run.json"),
            (
                "/cache/examples/wavy.scene.json",
                Some("wavy"),
                "demo/wavy.json",
            ),
        ] {
            assert_eq!(file_of(dir, scene, demo), dir.join(want), "{scene}");
        }
    }

    #[test]
    fn edits_of_the_same_items_in_another_order_are_the_same_edits() {
        let fragment = |pass: &str, base: &str| {
            Item::new(
                "fragment",
                Some(pass),
                Some(base.to_string()),
                json!("void main(){}"),
            )
        };
        let edits = |items: Vec<Item>| Edits {
            values: Map::new(),
            items,
        };
        let one = edits(vec![fragment("a", "1"), fragment("b", "2")]);
        assert!(one.same(&edits(vec![fragment("b", "2"), fragment("a", "1")])));
        assert!(!one.same(&edits(vec![fragment("b", "1"), fragment("a", "2")])));
    }

    #[test]
    fn two_params_moved_save_as_their_values_alone_and_load_back() {
        let mut moved = clocks();
        moved["passes"][0]["params"]["cols"]["value"] = json!(5);
        moved["passes"][0]["params"]["rows"]["value"] = json!(4);
        let file = json!({"version": 1, "scene": "/s/clocks/clocks.scene.json",
            "values": {"passes": {"main": {"params": {"cols": 5, "rows": 4}}}}});
        assert_eq!(
            diff(&clocks(), &moved, false).to_file("/s/clocks/clocks.scene.json"),
            file
        );
        let loaded = apply(&clocks(), &Edits::from_file(&file).unwrap(), "clocks");
        assert_eq!(loaded.doc, moved);
        assert!(loaded.notes.is_empty() && loaded.aside.is_empty());
    }

    #[test]
    fn a_state_file_that_cannot_be_written_leaves_no_temporary_file() {
        let dir = crate::testing::TempDir::new("state-unwritten");
        let path = dir.join("clocks.json");
        fs::create_dir(&path).unwrap();
        let file = json!({"version": 1,
            "values": {"passes": {"main": {"params": {"cols": 5}}}}});
        let edits = Edits::from_file(&file).unwrap();
        write(&path, "/s/clocks/clocks.scene.json", &edits).unwrap_err();
        let left: Vec<_> = fs::read_dir(&*dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(left, ["clocks.json"]);
    }

    #[test]
    fn a_value_whose_param_is_gone_is_skipped_with_one_line() {
        let mut renamed = clocks();
        let rows = renamed["passes"][0]["params"]
            .as_object_mut()
            .unwrap()
            .remove("rows")
            .unwrap();
        renamed["passes"][0]["params"]["lines"] = rows;
        let file = json!({"version": 1,
            "values": {"passes": {"main": {"params": {"cols": 5, "rows": 4}}}}});
        let loaded = apply(&renamed, &Edits::from_file(&file).unwrap(), "clocks");
        assert_eq!(
            loaded.notes,
            ["clocks: pass 'main' has no param 'rows', its saved value is skipped"]
        );
        assert_eq!(loaded.doc["passes"][0]["params"]["cols"]["value"], 5);
    }

    #[test]
    fn a_shader_edit_waits_when_its_default_changed() {
        let mine = "void main(){ gl_FragColor = vec4(0.5); }";
        let mut edited = clocks();
        edited["passes"][0]["fragment"] = json!(mine);
        let edits = diff(&clocks(), &edited, false);
        assert_eq!(
            edits.items,
            [Item {
                item: "fragment".to_string(),
                pass: Some("main".to_string()),
                param: None,
                base: Some("758cc820f81e647b".to_string()),
                value: json!(mine),
            }]
        );
        assert_eq!(apply(&clocks(), &edits, "clocks").doc, edited);
        let mut deployed = clocks();
        deployed["passes"][0]["fragment"] = json!("void main(){ gl_FragColor = vec4(0.25); }");
        let loaded = apply(&deployed, &edits, "clocks");
        assert_eq!(loaded.doc, deployed);
        assert_eq!(loaded.aside, edits.items);
        assert_eq!(
            loaded.notes,
            ["clocks: pass 'main''s shader changed since its saved edit, which waits for the console"]
        );
    }

    #[test]
    fn a_list_edit_names_the_default_layers_and_hashes_their_names() {
        let mut d = clocks();
        d["passes"][0]["layers"] = json!([{"name": "title", "text": "hi", "x": 1}]);
        let mut w = d.clone();
        w["passes"][0]["layers"] = json!([
            {"name": "bar", "shape": "rect", "width": 4},
            {"name": "title", "text": "hi", "x": 3}
        ]);
        let edits = diff(&d, &w, false);
        // sha256sum of the compact JSON ["title"]
        assert_eq!(edits.items[0].base.as_deref(), Some("db9f1b137477d577"));
        assert_eq!(
            edits.items[0].value,
            json!([{"name": "bar", "shape": "rect", "width": 4}, "title"])
        );
        assert_eq!(
            edits.values,
            *json!({"layers": {"title": {"x": 3}}}).as_object().unwrap()
        );
        // a deploy that moves the default layer keeps the list edit and the value
        let mut deployed = d.clone();
        deployed["passes"][0]["layers"][0]["x"] = json!(2);
        assert_eq!(apply(&deployed, &edits, "clocks").doc, w);
        // a procedural scene keeps its program's lists
        assert!(diff(&d, &w, true).items.is_empty());
    }

    #[test]
    fn a_null_in_the_values_removes_the_key() {
        let mut d = clocks();
        d["passes"][0]["layers"] = json!([{"name": "title", "text": "hi", "font_size": 20,
            "glow": {"blur": 0.2}}]);
        let mut w = d.clone();
        let title = w["passes"][0]["layers"][0].as_object_mut().unwrap();
        title.remove("font_size");
        title.remove("glow");
        let edits = diff(&d, &w, false);
        assert_eq!(
            edits.values,
            *json!({"layers": {"title": {"font_size": null, "glow": null}}})
                .as_object()
                .unwrap()
        );
        assert_eq!(apply(&d, &edits, "clocks").doc, w);
    }

    #[test]
    fn an_asset_saves_whole_by_its_name_and_a_null_removes_one() {
        let mut d = clocks();
        d["assets"] = json!({"logo": {"file": "logo.png"}});
        let mut w = d.clone();
        w["assets"] = json!({"plaster": {"file": "plaster.png", "sampling": "linear"}});
        let edits = diff(&d, &w, false);
        assert_eq!(
            Value::Object(edits.values.clone()),
            json!({"assets": {"plaster": {"file": "plaster.png", "sampling": "linear"},
                              "logo": null}})
        );
        assert_eq!(apply(&d, &edits, "clocks").doc, w);
        assert!(diff(&d, &d, false).is_empty());
    }

    #[test]
    fn a_pass_s_sampling_saves_and_loads_back() {
        let with = |v: Option<&str>| {
            let mut s = clocks();
            if let Some(v) = v {
                s["passes"][0]["sampling"] = json!(v);
            }
            s
        };
        let saved = |v: &str| json!({"passes": {"main": {"sampling": v}}});
        // the default's sampling, the working copy's, and the saved values
        for (d, w, want) in [
            (None, Some("nearest"), saved("nearest")),
            (None, Some("trilinear"), saved("trilinear")),
            (Some("trilinear"), None, saved("linear")),
            (Some("nearest"), Some("trilinear"), saved("trilinear")),
            (None, Some("linear"), json!({})),
        ] {
            let edits = diff(&with(d), &with(w), false);
            assert_eq!(Value::Object(edits.values.clone()), want, "{d:?} {w:?}");
            let loaded = apply(&with(d), &edits, "clocks").doc;
            let got = loaded["passes"][0].get("sampling").and_then(Value::as_str);
            assert_eq!(
                got.unwrap_or("linear"),
                w.unwrap_or("linear"),
                "{d:?} {w:?}"
            );
        }
    }
}
