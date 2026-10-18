//! ghee ships named post-process passes, such as dither.
//!
//! A filter file (`<name>.filter.json`) holds one pass: a name, its fragment
//! and its params, channels and assets.
//! The pass reads iChannel0, the frame so far, and writes the result. ghee
//! appends a filter as a pass that reads the pass before it. The last filter
//! reaches the display.
//!
//! The build compiles the shipped filters in. A scan adds more from a directory.

use anyhow::{bail, Result};
use serde_json::{Map, Value};

use crate::scene::{inside, join, known_keys, normpath, strerror, Asset, Pass, RawLayout, Scene};

/// The shipped filters. A binary carries no package directory, so the build
/// compiles them in.
const BUILTIN: [(&str, &str); 3] = [
    (
        "dither.filter.json",
        include_str!("../ghee/filters/dither.filter.json"),
    ),
    (
        "frc.filter.json",
        include_str!("../ghee/filters/frc.filter.json"),
    ),
    (
        "frc256.filter.json",
        include_str!("../ghee/filters/frc256.filter.json"),
    ),
];

/// The files the shipped filters declare as assets, compiled in beside them.
const BUILTIN_ASSETS: [(&str, &[u8]); 1] = [(
    "bluenoise.png",
    include_bytes!("../ghee/filters/bluenoise.png"),
)];

/// The keys of a filter file. `description` is for its reader alone.
const FILTER_KEYS: [&str; 6] = [
    "name",
    "description",
    "fragment",
    "params",
    "channels",
    "assets",
];

/// The directory a scanned filter came from, kept on its definition so its
/// assets resolve beside it. A shipped filter has none.
const DIR_KEY: &str = "_dir";

/// Parse one filter file into `out`. The name it went in under, or None when
/// the file was skipped.
fn accept(out: &mut Map<String, Value>, path: &str, text: &str) -> Option<String> {
    let parsed: Result<Value, _> = serde_json::from_str(text);
    let fd = match parsed {
        Ok(v) => v,
        Err(e) => {
            eprintln!("filter {path}: skipped ({e})");
            return None;
        }
    };
    let named = fd.get("name").is_some_and(Value::is_string)
        && fd.get("fragment").is_some_and(Value::is_string);
    if !named {
        eprintln!("filter {path}: skipped (needs a string 'name' and 'fragment')");
        return None;
    }
    let name = fd["name"].as_str().unwrap_or_default().to_string();
    if let Some(o) = fd.as_object() {
        if let Err(e) = known_keys(o, &format!("filter '{name}'"), &FILTER_KEYS, &[]) {
            eprintln!("filter {path}: skipped ({e})");
            return None;
        }
    }
    out.insert(name.clone(), fd);
    Some(name)
}

/// name -> filter definition, one pass each. A later directory wins a name
/// clash. $GHEE_FILTERS adds directories after the shipped filters, separated
/// by colons.
pub fn load_filters(extra_dirs: &[&str]) -> Map<String, Value> {
    let env = std::env::var("GHEE_FILTERS").unwrap_or_default();
    let mut out = Map::new();
    for (name, text) in BUILTIN {
        accept(&mut out, name, text);
    }
    let dirs = env
        .split(':')
        .filter(|d| !d.is_empty())
        .chain(extra_dirs.iter().copied());
    for d in dirs {
        let Ok(entries) = std::fs::read_dir(d) else {
            continue;
        };
        let mut paths: Vec<String> = entries
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| !n.starts_with('.') && n.ends_with(".filter.json"))
            .map(|n| format!("{d}/{n}"))
            .collect();
        paths.sort();
        for p in paths {
            match std::fs::read_to_string(&p) {
                Ok(text) => {
                    if let Some(name) = accept(&mut out, &p, &text) {
                        out[&name][DIR_KEY] = Value::from(d.to_string());
                    }
                }
                Err(e) => eprintln!("filter {p}: skipped ({e})"),
            }
        }
    }
    out
}

/// The registry as the console's filter picker sees it, without the
/// directories the scan came from.
pub fn filter_defs() -> Map<String, Value> {
    let mut reg = load_filters(&[]);
    for fd in reg.values_mut() {
        if let Some(o) = fd.as_object_mut() {
            o.remove(DIR_KEY);
        }
    }
    reg
}

/// A filter that a scene ends with: the filter's name, whether its pass
/// starts shown, and the values that its params start at, by param name.
#[derive(Clone, Debug, PartialEq)]
pub struct Filter {
    pub name: String,
    pub visible: bool,
    pub values: Map<String, Value>,
}

impl Filter {
    /// The filter `name`, shown, at its defaults.
    pub fn named(name: &str) -> Filter {
        Filter {
            name: name.to_string(),
            visible: true,
            values: Map::new(),
        }
    }

    /// An entry of a scene's `filters`: a filter's name, or an object with
    /// its `name`, an optional `visible` and a value for any of its params,
    /// as in `{"name": "frc", "calm": 8}`.
    pub fn from_json(v: &Value) -> Result<Filter> {
        let entry = match v {
            Value::String(n) => Some(Filter::named(n)),
            Value::Object(o) => (|| {
                let visible = match o.get("visible") {
                    None => true,
                    Some(v) => v.as_bool()?,
                };
                let name = o.get("name")?.as_str()?.to_string();
                let values = o
                    .iter()
                    .filter(|(k, _)| *k != "name" && *k != "visible")
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                Some(Filter {
                    name,
                    visible,
                    values,
                })
            })(),
            _ => None,
        };
        entry.ok_or_else(|| {
            anyhow::anyhow!(
                "'filters' takes names, and objects with a 'name' and values for the \
                 filter's params, got {}",
                crate::scene::py_repr(v)
            )
        })
    }

    /// The entry as from_json() reads it: the bare name when the filter is
    /// shown at its defaults.
    pub fn to_json(&self) -> Value {
        if self.visible && self.values.is_empty() {
            return Value::from(self.name.as_str());
        }
        let mut m = Map::new();
        m.insert("name".to_string(), Value::from(self.name.as_str()));
        if !self.visible {
            m.insert("visible".to_string(), Value::from(false));
        }
        m.extend(self.values.clone());
        Value::Object(m)
    }

    /// The filters of `--filters`, separated by commas. Each is a name, then
    /// `:param=value` for each param that it sets, as in `frc:calm=8`. A
    /// value that reads as JSON is that value, and any other is a string.
    pub fn parse_list(text: &str) -> Result<Vec<Filter>> {
        let mut out = Vec::new();
        for item in text.split(',').map(str::trim).filter(|i| !i.is_empty()) {
            let mut parts = item.split(':');
            let mut entry = Map::new();
            entry.insert("name".to_string(), Value::from(parts.next().unwrap_or("")));
            for set in parts {
                let Some((k, v)) = set.split_once('=') else {
                    bail!("filter '{item}': '{set}' needs the form param=value");
                };
                let v = serde_json::from_str(v).unwrap_or_else(|_| Value::from(v));
                entry.insert(k.to_string(), v);
            }
            out.push(Filter::from_json(&Value::Object(entry))?);
        }
        Ok(out)
    }
}

/// The filters of `first`, then those of `later`. A filter of `later` that
/// `first` already holds keeps its place there, and its values replace those
/// of the filter in `first`.
pub fn chain(first: &[Filter], later: &[Filter]) -> Vec<Filter> {
    let mut out = first.to_vec();
    for f in later {
        match out.iter_mut().find(|o| o.name == f.name) {
            Some(o) => o.values.extend(f.values.clone()),
            None => out.push(f.clone()),
        }
    }
    out
}

/// A filter name that the registry lacks.
#[derive(Debug)]
pub struct Unknown(String);

impl std::fmt::Display for Unknown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Unknown {}

/// Fail with Unknown when `reg` lacks a filter of `filters`, naming each such
/// filter and every filter that `reg` holds.
pub fn check_filters(filters: &[Filter], reg: &Map<String, Value>) -> Result<()> {
    let unknown: Vec<&str> = filters
        .iter()
        .filter(|f| !reg.contains_key(f.name.as_str()))
        .map(|f| f.name.as_str())
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    Err(Unknown(format!(
        "unknown filter: {} (known: {})",
        unknown.join(", "),
        reg.keys().cloned().collect::<Vec<_>>().join(", ")
    ))
    .into())
}

/// Append each filter as a pass, in order. Each one reads the pass before
/// it, so the last filter reaches the display. The pass of a filter takes its
/// eye, so every filter of the scene's own list shows from then on. A name that
/// the registry lacks fails with Unknown.
pub fn apply_filters(
    scene: &mut Scene,
    filters: &[Filter],
    registry: Option<&Map<String, Value>>,
) -> Result<()> {
    if filters.is_empty() {
        return Ok(());
    }
    let loaded;
    let reg = match registry {
        Some(r) => r,
        None => {
            loaded = load_filters(&[]);
            &loaded
        }
    };
    check_filters(filters, reg)?;
    if scene.passes.is_empty() {
        return Ok(());
    }
    for f in filters {
        let name = &f.name;
        let fd = &reg[name];
        let Some(fname) = fd.get("name").and_then(Value::as_str) else {
            bail!("filter '{name}': needs a string 'name' and 'fragment'");
        };
        let own = unique(scene, fname, fd);
        let prev = scene.passes[scene.passes.len() - 1].name.clone();
        let mut p = filter_pass(name, fd, &own, &prev, &f.values, None)?;
        p.visible = f.visible;
        scene.passes.push(p);
        merge_assets(scene, name, fd)?;
    }
    for f in &mut scene.filters {
        f.visible = true;
    }
    // A scene param reaches the filter's pass too.
    match scene.name_clash() {
        Some(e) => bail!("{e}"),
        None => Ok(()),
    }
}

/// Build the filters of a posted scene. A filter pass the post carries is
/// rebuilt from its definition: the post gives its place, its name and its
/// param values, the filter gives its code, channels and assets. A filter the
/// scene's `filters` list names and no pass carries yet is appended, as a
/// load from disk does, so a scene file posted as it is gets its filters. An
/// unknown filter fails.
pub fn rebuild_filters(scene: &mut Scene, registry: Option<&Map<String, Value>>) -> Result<()> {
    scene.assets.retain(|a| a.from_filter.is_empty());
    let mut missing = scene.filters.clone();
    for p in &scene.passes {
        if let Some(i) = missing.iter().position(|f| f.name == p.filter) {
            missing.remove(i);
        }
    }
    if missing.is_empty() && scene.passes.iter().all(|p| p.filter.is_empty()) {
        return Ok(());
    }
    let loaded;
    let reg = match registry {
        Some(r) => r,
        None => {
            loaded = load_filters(&[]);
            &loaded
        }
    };
    for i in 0..scene.passes.len() {
        let name = scene.passes[i].filter.clone();
        if name.is_empty() {
            continue;
        }
        check_filters(&[Filter::named(&name)], reg)?;
        let fd = &reg[&name];
        if i == 0 {
            bail!("filter '{name}': a filter pass needs a pass before it");
        }
        let own = scene.passes[i].name.clone();
        let prev = scene.passes[i - 1].name.clone();
        let p = filter_pass(&name, fd, &own, &prev, &Map::new(), Some(&scene.passes[i]))?;
        scene.passes[i] = p;
        merge_assets(scene, &name, fd)?;
    }
    apply_filters(scene, &missing, Some(reg))?;
    match scene.name_clash() {
        Some(e) => bail!("{e}"),
        None => Ok(()),
    }
}

/// One filter pass named `own`, reading `prev`. Its params start at
/// `values`, and the param values of `was` replace those where the filter
/// still declares that param. The pass keeps the eye of `was`.
fn filter_pass(
    name: &str,
    fd: &Value,
    own: &str,
    prev: &str,
    values: &Map<String, Value>,
    was: Option<&Pass>,
) -> Result<Pass> {
    let Some(fragment) = fd.get("fragment").and_then(Value::as_str) else {
        bail!("filter '{name}': needs a string 'name' and 'fragment'");
    };
    let mut params = fd
        .get("params")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    if let Value::Object(defs) = &mut params {
        for (n, v) in values {
            let Some(Value::Object(def)) = defs.get_mut(n) else {
                bail!(
                    "filter '{name}': param '{n}' is not a param this filter declares ({})",
                    match defs.is_empty() {
                        true => "none".to_string(),
                        false => defs.keys().cloned().collect::<Vec<_>>().join(", "),
                    }
                );
            };
            def.insert("value".to_string(), v.clone());
            crate::scene::Param::from_dict(n, &Value::Object(def.clone()))
                .map_err(|e| anyhow::anyhow!("filter '{name}': {e}"))?;
        }
    }
    if let (Some(was), Value::Object(defs)) = (was, &mut params) {
        for (n, def) in defs.iter_mut() {
            if let (Some(p), Value::Object(def)) = (was.params.get(n), def) {
                def.insert("value".to_string(), p.value.to_json());
            }
        }
    }
    let mut d = Map::new();
    d.insert("name".to_string(), Value::from(own));
    d.insert("fragment".to_string(), Value::from(fragment));
    d.insert("params".to_string(), params);
    let mut channel = Map::new();
    channel.insert("buffer".to_string(), Value::from(prev));
    let mut channels = vec![Value::Object(channel)];
    // A filter def may declare more channels after the filtered frame.
    // "self" names the filter's own previous frame, so an accumulator
    // works on any scene.
    if let Some(extra) = fd.get("channels").and_then(Value::as_array) {
        for c in extra {
            let mut c = c.clone();
            if c.get("buffer").and_then(Value::as_str) == Some("self") {
                c["buffer"] = Value::from(own);
            }
            channels.push(c);
        }
    }
    d.insert("channels".to_string(), Value::Array(channels));
    let mut p = Pass::from_dict(&Value::Object(d), None)?;
    p.filter = name.to_string(); // The console renders a filter pass read-only.
    p.visible = was.is_none_or(|w| w.visible);
    Ok(p)
}

/// A filter's declared assets join the scene's, marked as the filter's and
/// named `<filter>_<asset>`, so none meets a scene name. Only the filter's own
/// pass declares them, under the names the filter wrote. A filter applied twice
/// shares one copy.
fn merge_assets(scene: &mut Scene, filter: &str, fd: &Value) -> Result<()> {
    let Some(Value::Object(assets)) = fd.get("assets") else {
        return Ok(());
    };
    let mut added = false;
    for (aname, ad) in assets {
        let full = format!("{filter}_{aname}");
        if scene.assets.iter().any(|a| a.name == full) {
            continue;
        }
        let mut asset =
            Asset::from_dict(aname, ad).map_err(|e| anyhow::anyhow!("filter '{filter}': {e}"))?;
        if asset.is_font() {
            anyhow::bail!(
                "filter '{filter}': asset '{aname}' is a font, a filter takes images only"
            );
        }
        asset.name = full;
        asset.from_filter = filter.to_string();
        resolve_asset(&mut asset, fd)?;
        scene.assets.push(asset);
        added = true;
    }
    if added {
        use std::mem::take;
        let root = take(&mut scene.root);
        *scene = Scene::new(
            take(&mut scene.passes),
            take(&mut scene.version),
            take(&mut scene.filters),
            take(&mut scene.params),
            take(&mut scene.common),
            take(&mut scene.assets),
        )
        .map_err(|e| anyhow::anyhow!("filter '{filter}': {e}"))?;
        scene.root = root;
    }
    Ok(())
}

/// Read and decode a filter's asset. A shipped filter's file comes from the
/// built-in table. A scanned filter's file resolves inside the directory the
/// filter came from, and never outside it.
fn resolve_asset(asset: &mut Asset, fd: &Value) -> Result<()> {
    let filter = &asset.from_filter;
    let name = asset.local_name().to_string();
    let Some(dir) = fd.get(DIR_KEY).and_then(Value::as_str) else {
        let Some((_, bytes)) = BUILTIN_ASSETS.iter().find(|(f, _)| *f == asset.file) else {
            bail!(
                "filter '{filter}' asset '{name}': {} is not a shipped file",
                asset.file
            );
        };
        let what = format!("shipped {}", asset.file);
        return decode_once(asset, &what, None, || Ok(bytes.to_vec()));
    };
    let root = normpath(dir);
    let path = normpath(&join(&root, &asset.file));
    if path == root || !inside(&path, &root) {
        bail!(
            "filter '{filter}' asset '{name}' file '{}' is outside the filter directory",
            asset.file
        );
    }
    let what = format!("filter '{filter}' asset '{name}' ({path})");
    let io = |e: std::io::Error| anyhow::anyhow!("{what}: {}", strerror(&e));
    let meta = std::fs::metadata(&path).map_err(io)?;
    let stamp = (meta.modified().ok(), meta.len());
    crate::sources::record(&path);
    decode_once(asset, &path, Some(stamp), || {
        std::fs::read(&path).map_err(io)
    })
}

type Stamp = (Option<std::time::SystemTime>, u64);

/// A decoded filter asset, by source.
struct Decoded {
    stamp: Option<Stamp>,
    raw: Option<RawLayout>,
    image: crate::assets::Image,
}

/// Every console post rebuilds its filters. The decoded pixels are kept, one
/// entry per source, so a file decodes once until its size or mtime changes.
static DECODED: std::sync::Mutex<Vec<(String, Decoded)>> = std::sync::Mutex::new(Vec::new());

fn decode_once(
    asset: &mut Asset,
    source: &str,
    stamp: Option<Stamp>,
    read: impl FnOnce() -> Result<Vec<u8>>,
) -> Result<()> {
    let mut cache = DECODED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, d)) = cache.iter().find(|(k, _)| k == source) {
        if d.stamp == stamp && d.raw == asset.raw {
            asset.image = Some(d.image.clone());
            return Ok(());
        }
    }
    asset.decode(&read()?, source)?;
    let image = asset.image.clone().expect("decoded");
    cache.retain(|(k, _)| k != source);
    cache.push((
        source.to_string(),
        Decoded {
            stamp,
            raw: asset.raw,
            image,
        },
    ));
    Ok(())
}

/// `base`, or `base` numbered from 1, so that the pass of filter `fd` takes
/// a name that neither the scene nor the filter's own params hold.
fn unique(scene: &Scene, base: &str, fd: &Value) -> String {
    let mut taken = scene.taken_names();
    if let Some(Value::Object(params)) = fd.get("params") {
        taken.extend(params.keys().map(String::as_str));
    }
    let mut name = base.to_string();
    let mut k = 1;
    while taken.contains(&name.as_str()) {
        name = format!("{base}{k}");
        k += 1;
    }
    name
}

/// $GHEE_FILTERS is process-wide. A test that sets it must not run beside a
/// test that reads the registry.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{ParamMap, ParamValue, Pass, Scene};
    use serde_json::json;

    const FLAT: &str = "void main(){ gl_FragColor = vec4(vec3(0.5), 1.0); }";
    const GOOD: &str = "void main(){ gl_FragColor = vec4(1.0); }";
    const COMMON: &str = "float sat(float x){ return clamp(x, 0.0, 1.0); }";

    fn names(sc: &Scene) -> Vec<String> {
        sc.passes.iter().map(|p| p.name.clone()).collect()
    }

    fn tmpdir(tag: &str) -> crate::testing::TempDir {
        crate::testing::TempDir::new(&format!("filters-{tag}"))
    }

    #[test]
    fn a_shipped_filter_brings_its_asset() {
        let _g = env_guard();
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "passes": [{"name": "m", "fragment": GOOD}]}),
            None,
            None,
        )
        .unwrap();
        apply_filters(&mut sc, &[crate::filters::Filter::named("frc256")], None).unwrap();
        let a = sc
            .assets
            .iter()
            .find(|a| a.name == "frc256_bluenoise")
            .expect("bluenoise asset");
        let img = a.image.as_ref().expect("decoded");
        assert_eq!((img.width, img.height), (64, 64));
        assert!(sc.passes.last().unwrap().name.starts_with("frc256"));
    }

    #[test]
    fn a_filter_takes_no_font_asset() {
        let _g = env_guard();
        let d = tmpdir("font-filter");
        std::fs::write(
            format!("{d}/lettered.filter.json"),
            json!({"name": "lettered", "fragment": GOOD,
                   "assets": {"face": {"file": "face.ttf"}}})
            .to_string(),
        )
        .unwrap();
        let reg = load_filters(&[&d]);
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "passes": [{"name": "m", "fragment": GOOD}]}),
            None,
            None,
        )
        .unwrap();
        let e = apply_filters(
            &mut sc,
            &[crate::filters::Filter::named("lettered")],
            Some(&reg),
        )
        .unwrap_err()
        .to_string();
        assert_eq!(
            e,
            "filter 'lettered': asset 'face' is a font, a filter takes images only"
        );
    }

    #[test]
    fn a_scanned_filter_resolves_its_asset_beside_itself() {
        let _g = env_guard();
        let d = tmpdir("asset-filter");
        std::fs::write(format!("{d}/m.raw"), [1u8, 2, 3, 4]).unwrap();
        std::fs::write(
            format!("{d}/masked.filter.json"),
            json!({"name": "masked", "fragment": GOOD,
                   "assets": {"m": {"file": "m.raw", "width": 2, "height": 2, "format": "u8"}}})
            .to_string(),
        )
        .unwrap();
        let reg = load_filters(&[&d]);
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "passes": [{"name": "m", "fragment": GOOD}]}),
            None,
            None,
        )
        .unwrap();
        apply_filters(
            &mut sc,
            &[crate::filters::Filter::named("masked")],
            Some(&reg),
        )
        .unwrap();
        let a = sc
            .assets
            .iter()
            .find(|a| a.name == "masked_m")
            .expect("asset m");
        assert_eq!(
            (
                a.image.as_ref().unwrap().width,
                a.image.as_ref().unwrap().height
            ),
            (2, 2)
        );
    }

    #[test]
    fn a_posted_filter_pass_is_rebuilt_from_its_filter() {
        // The console posts what GET /scene gave it. Only the filter's place,
        // name and param values count, and the code and the asset come from
        // the filter.
        let _g = env_guard();
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "passes": [{"name": "m", "fragment": GOOD}]}),
            None,
            None,
        )
        .unwrap();
        apply_filters(&mut sc, &[crate::filters::Filter::named("frc256")], None).unwrap();
        let mut doc = sc.to_dict();
        // A filter's asset stays out of the document, and comes back with its
        // filter.
        assert!(doc.get("assets").is_none(), "{doc}");
        doc["passes"][1]["fragment"] = json!("void main(){ gl_FragColor = vec4(0.0); }");
        let mut back = Scene::from_dict(&doc, None, None).unwrap();
        rebuild_filters(&mut back, None).unwrap();
        assert_eq!(back.passes[1].fragment, sc.passes[1].fragment);
        let img = back.assets[0].image.as_ref().expect("decoded");
        assert_eq!((img.width, img.height), (64, 64));
    }

    #[test]
    fn a_filter_asset_never_meets_a_scene_name() {
        // The scene has its own bluenoise and a param named like its size. The
        // filter still samples its own mask under the names it wrote.
        let _g = env_guard();
        let d = tmpdir("clash");
        std::fs::write(format!("{d}/bluenoise.raw"), [1u8, 2, 3, 4]).unwrap();
        let mut sc = Scene::from_dict(
            &json!({"version": 1,
                    "passes": [{"name": "m", "fragment": GOOD}],
                    "assets": {"bluenoise": {"file": "bluenoise.raw",
                                             "width": 2, "height": 2, "format": "u8"}}}),
            None,
            None,
        )
        .unwrap();
        sc.load_assets(&d).unwrap();
        apply_filters(&mut sc, &[crate::filters::Filter::named("frc256")], None).unwrap();
        let names: Vec<&str> = sc.assets.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["bluenoise", "frc256_bluenoise"]);
        let main = sc.passes[0].glsl(Some(&sc.params), None, &sc.assets, &[], false);
        assert!(main.contains("uniform sampler2D bluenoise_tex;"));
        assert!(!main.contains("frc256_bluenoise"));
        let filt = sc.passes[1].glsl(Some(&sc.params), None, &sc.assets, &[], false);
        assert!(filt.contains("uniform sampler2D frc256_bluenoise_tex;"));
        assert!(filt.contains("#define bluenoise_tex frc256_bluenoise_tex"));
        assert!(filt.contains("#define bluenoise_size frc256_bluenoise_size"));
        assert!(!filt.contains("uniform sampler2D bluenoise_tex;"));

        let mut sp = Scene::from_dict(
            &json!({"version": 1, "passes": [{"name": "m", "fragment": GOOD}],
                    "params": {"bluenoise_size": {"type": "float", "value": 0.5}}}),
            None,
            None,
        )
        .unwrap();
        apply_filters(&mut sp, &[crate::filters::Filter::named("frc256")], None).unwrap();
        let filt = sp.passes[1].glsl(Some(&sp.params), None, &sp.assets, &[], false);
        let (decl, def) = (
            filt.find("uniform float bluenoise_size;").unwrap(),
            filt.find("#define bluenoise_size ").unwrap(),
        );
        assert!(decl < def, "the define must follow every declaration");
    }

    #[test]
    fn a_changed_filter_asset_decodes_again() {
        let _g = env_guard();
        let d = tmpdir("redecode");
        let f = format!("{d}/m.raw");
        std::fs::write(&f, [1u8, 2, 3, 4]).unwrap();
        std::fs::write(
            format!("{d}/masked.filter.json"),
            json!({"name": "masked", "fragment": GOOD,
                   "assets": {"m": {"file": "m.raw", "width": 2, "height": 2, "format": "u8"}}})
            .to_string(),
        )
        .unwrap();
        let reg = load_filters(&[&d]);
        let first = |reg: &Map<String, Value>| {
            let mut sc = Scene::from_dict(
                &json!({"version": 1, "passes": [{"name": "m", "fragment": GOOD}]}),
                None,
                None,
            )
            .unwrap();
            apply_filters(
                &mut sc,
                &[crate::filters::Filter::named("masked")],
                Some(reg),
            )
            .unwrap();
            sc.assets[0].image.clone().unwrap().pixels
        };
        let a = first(&reg);
        assert_eq!(first(&reg), a);
        std::fs::write(&f, [9u8, 9, 9, 9]).unwrap();
        let later = std::time::SystemTime::now() + std::time::Duration::from_secs(60);
        std::fs::File::options()
            .write(true)
            .open(&f)
            .unwrap()
            .set_modified(later)
            .unwrap();
        assert_ne!(first(&reg), a);
    }

    #[test]
    fn a_page_built_filter_stub_gets_the_whole_filter() {
        // The console adds a filter as its name and place only.
        let _g = env_guard();
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "passes": [
                {"name": "m", "fragment": GOOD},
                {"name": "dither", "fragment": "", "filter": "dither",
                 "params": {"levels": {"type": "float", "value": 4.0}}}]}),
            None,
            None,
        )
        .unwrap();
        rebuild_filters(&mut sc, None).unwrap();
        let p = &sc.passes[1];
        assert!(p.fragment.contains("main"));
        assert_eq!(p.channels[0].buffer.as_deref(), Some("m"));
        assert_eq!(p.params.get("levels").unwrap().value.to_json(), json!(4.0));
        assert_eq!(p.params.get("amount").unwrap().value.to_json(), json!(1.0));
    }

    #[test]
    fn a_posted_filter_list_is_applied() {
        // A scene file posted as it is: its filters list and no filter pass.
        let _g = env_guard();
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "filters": ["frc256"],
                    "passes": [{"name": "m", "fragment": GOOD}]}),
            None,
            None,
        )
        .unwrap();
        rebuild_filters(&mut sc, None).unwrap();
        assert_eq!(names(&sc), ["m", "frc256"]);
        assert_eq!(sc.passes[1].filter, "frc256");
        assert_eq!(sc.assets[0].name, "frc256_bluenoise");
        // Posted again with its pass, as the console sends it: no second copy.
        let mut again = Scene::from_dict(&sc.to_dict(), None, None).unwrap();
        rebuild_filters(&mut again, None).unwrap();
        assert_eq!(names(&again), ["m", "frc256"]);
    }

    #[test]
    fn a_hidden_filter_pass_stays_hidden_when_a_post_rebuilds_it() {
        let _g = env_guard();
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "passes": [
                {"name": "m", "fragment": GOOD},
                {"name": "dither", "fragment": "", "filter": "dither", "visible": false}]}),
            None,
            None,
        )
        .unwrap();
        rebuild_filters(&mut sc, None).unwrap();
        assert!(sc.passes[1].fragment.contains("main"));
        assert!(!sc.passes[1].visible);
    }

    #[test]
    fn a_hidden_filters_entry_appends_its_pass_hidden() {
        let _g = env_guard();
        let hidden = json!({"name": "dither", "visible": false});
        // frc256 brings an asset, and its pass rebuilds the scene before the
        // hidden filter appends.
        for (filters, eyes, back) in [
            (
                json!(["frc256", hidden]),
                vec![("", true), ("frc256", true), ("dither", false)],
                json!(["frc256", "dither"]),
            ),
            (
                json!([hidden]),
                vec![("", true), ("dither", false)],
                json!(["dither"]),
            ),
        ] {
            let passes = json!([{"name": "m", "fragment": GOOD}]);
            let d = json!({"version": 1, "filters": filters, "passes": passes});
            let mut sc = Scene::from_dict(&d, None, None).unwrap();
            let names = sc.filters.clone();
            apply_filters(&mut sc, &names, None).unwrap();
            let got: Vec<_> = sc
                .passes
                .iter()
                .map(|p| (p.filter.as_str(), p.visible))
                .collect();
            assert_eq!(got, eyes, "{filters}");
            // Appended, a filter keeps its eye on its pass, and the list
            // holds none.
            assert_eq!(sc.to_dict()["filters"], back, "{filters}");
        }
    }

    #[test]
    fn a_listed_filter_the_post_leaves_out_stays_out() {
        // A plain pass that carries a filter's name stays plain when the
        // post's list leaves that filter out.
        let _g = env_guard();
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "filters": [],
                    "passes": [{"name": "m", "fragment": GOOD},
                               {"name": "frc256", "fragment": GOOD}]}),
            None,
            None,
        )
        .unwrap();
        rebuild_filters(&mut sc, None).unwrap();
        assert_eq!(names(&sc), ["m", "frc256"]);
        assert!(sc.passes[1].filter.is_empty());
    }

    #[test]
    fn an_unknown_listed_filter_fails() {
        let _g = env_guard();
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "filters": ["nope"],
                    "passes": [{"name": "m", "fragment": GOOD}]}),
            None,
            None,
        )
        .unwrap();
        let e = rebuild_filters(&mut sc, None).unwrap_err().to_string();
        assert!(e.starts_with("unknown filter: nope (known: "), "{e}");
    }

    #[test]
    fn an_unknown_posted_filter_fails() {
        let _g = env_guard();
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "passes": [
                {"name": "m", "fragment": GOOD},
                {"name": "x", "fragment": GOOD, "filter": "nope"}]}),
            None,
            None,
        )
        .unwrap();
        let e = rebuild_filters(&mut sc, None).unwrap_err().to_string();
        assert!(e.starts_with("unknown filter: nope (known: "), "{e}");
    }

    #[test]
    fn a_skipped_file_leaves_the_shipped_filter_alone() {
        let _g = env_guard();
        let d = tmpdir("skipped-shadow");
        std::fs::write(
            format!("{d}/frc256.filter.json"),
            json!({"name": "frc256"}).to_string(),
        )
        .unwrap();
        let reg = load_filters(&[&d]);
        assert!(reg["frc256"].get(DIR_KEY).is_none());
    }

    #[test]
    fn the_picker_sees_no_scan_directory() {
        let _g = env_guard();
        let d = tmpdir("picker");
        std::fs::write(
            format!("{d}/plain.filter.json"),
            json!({"name": "plain", "fragment": GOOD}).to_string(),
        )
        .unwrap();
        std::env::set_var("GHEE_FILTERS", &d);
        let defs = filter_defs();
        std::env::remove_var("GHEE_FILTERS");
        assert!(defs.contains_key("plain"));
        assert!(defs.values().all(|fd| fd.get(DIR_KEY).is_none()));
    }

    #[test]
    fn registry_ships_dither() {
        let _g = env_guard();
        let reg = load_filters(&[]);
        assert!(reg.contains_key("dither"));
        assert!(reg.contains_key("frc"));
        assert!(reg.contains_key("frc256"));
    }

    #[test]
    fn apply_appends_reading_prev_output() {
        let _g = env_guard();
        let mut sc = Scene::of(vec![Pass::of("main", FLAT).unwrap()]).unwrap();
        apply_filters(&mut sc, &[crate::filters::Filter::named("dither")], None).unwrap();
        assert_eq!(names(&sc), ["main", "dither"]);
        let d = &sc.passes[1];
        assert_eq!(d.filter, "dither");
        // iChannel0 is the frame so far.
        let buffers: Vec<_> = d.channels.iter().map(|c| c.buffer.clone()).collect();
        assert_eq!(buffers, [Some("main".to_string())]);
        let params: Vec<&String> = d.params.keys().collect();
        assert_eq!(params, ["amount", "levels", "color", "scale"]);
    }

    #[test]
    fn an_unknown_filter_fails_and_adds_no_pass() {
        let _g = env_guard();
        let mut sc = Scene::of(vec![Pass::of("main", FLAT).unwrap()]).unwrap();
        let wanted = [Filter::named("dither"), Filter::named("does-not-exist")];
        let e = apply_filters(&mut sc, &wanted, None).unwrap_err();
        assert!(e.is::<Unknown>());
        assert!(
            e.to_string()
                .starts_with("unknown filter: does-not-exist (known: "),
            "{e}"
        );
        assert_eq!(names(&sc), ["main"]);
    }

    #[test]
    fn a_filter_channel_names_itself_for_feedback() {
        let _g = env_guard();
        let mut reg = Map::new();
        let accum: Value = serde_json::from_str(
            r#"{"name":"accum","fragment":"void main(){gl_FragColor=mix(texture2D(iChannel1,gl_FragCoord.xy/iResolution.xy),texture2D(iChannel0,gl_FragCoord.xy/iResolution.xy),0.3);}","channels":[{"buffer":"self"}]}"#,
        )
        .unwrap();
        reg.insert("accum".to_string(), accum);
        let mut sc = Scene::of(vec![Pass::of("main", GOOD).unwrap()]).unwrap();
        apply_filters(
            &mut sc,
            &[crate::filters::Filter::named("accum")],
            Some(&reg),
        )
        .unwrap();
        let filt = &sc.passes[1];
        assert_eq!(filt.channels[0].buffer.as_deref(), Some("main"));
        assert_eq!(filt.channels[1].buffer.as_deref(), Some("accum"));
        Scene::of(sc.passes.clone()).unwrap();
    }

    #[test]
    fn a_filter_pass_takes_a_name_that_no_declaration_of_the_scene_holds() {
        let _g = env_guard();
        let f = json!({"type": "float", "value": 1.0});
        for doc in [
            json!({"passes": [{"name": "dither", "fragment": GOOD}]}),
            json!({"passes": [{"name": "main", "fragment": GOOD}],
                   "assets": {"dither": {"file": "d.png"}}}),
            json!({"passes": [{"name": "main", "fragment": GOOD}], "params": {"dither": f}}),
            json!({"passes": [{"name": "main", "fragment": GOOD, "params": {"dither": f}}]}),
            json!({"passes": [{"name": "main", "fragment": GOOD,
                               "arrays": [{"name": "dither", "size": 4}]}]}),
        ] {
            let mut sc = Scene::from_dict(&doc, None, None).unwrap();
            apply_filters(&mut sc, &[crate::filters::Filter::named("dither")], None).unwrap();
            assert_eq!(names(&sc)[1], "dither1", "{doc}");
        }
        // A filter's pass takes no name of the filter's own params.
        let reg = json!({"amount": {"name": "amount", "fragment": GOOD,
                                    "params": {"amount": f}}});
        let mut sc = Scene::of(vec![Pass::of("main", GOOD).unwrap()]).unwrap();
        apply_filters(
            &mut sc,
            &[crate::filters::Filter::named("amount")],
            reg.as_object(),
        )
        .unwrap();
        assert_eq!(names(&sc), ["main", "amount1"]);
    }

    #[test]
    fn a_filter_pass_never_sees_the_common() {
        let _g = env_guard();
        let d = json!({
            "version": 1,
            "passes": [{"name": "main", "fragment": GOOD}],
            "common": COMMON,
        });
        let mut sc = Scene::from_dict(&d, None, None).unwrap();
        apply_filters(&mut sc, &[crate::filters::Filter::named("dither")], None).unwrap();
        let filt = &sc.passes[1];
        assert_eq!(filt.filter, "dither");
        assert!(!filt
            .glsl(
                Some(&sc.params),
                sc.common.as_deref(),
                &sc.assets,
                &[],
                false
            )
            .contains(COMMON));
        assert!(sc.passes[0]
            .glsl(
                Some(&sc.params),
                sc.common.as_deref(),
                &sc.assets,
                &[],
                false
            )
            .contains(COMMON));
    }

    #[test]
    fn a_registry_can_be_passed_in() {
        let mut reg = Map::new();
        reg.insert(
            "invert".to_string(),
            json!({"name": "invert", "fragment": "void main(){}", "params": {}}),
        );
        let mut sc = Scene::of(vec![Pass::of("main", FLAT).unwrap()]).unwrap();
        apply_filters(
            &mut sc,
            &[crate::filters::Filter::named("invert")],
            Some(&reg),
        )
        .unwrap();
        assert_eq!(names(&sc), ["main", "invert"]);
    }

    #[test]
    fn extra_dirs_and_the_env_add_filters_and_skip_broken_ones() {
        let _g = env_guard();
        let dir = tmpdir("env");
        std::fs::write(format!("{dir}/broken.filter.json"), "{not json").unwrap();
        std::fs::write(
            format!("{dir}/nameless.filter.json"),
            json!({"fragment": "x"}).to_string(),
        )
        .unwrap();
        std::fs::write(
            format!("{dir}/typo.filter.json"),
            json!({"name": "typo", "fragment": GOOD, "parms": {}}).to_string(),
        )
        .unwrap();
        std::fs::write(
            format!("{dir}/invert.filter.json"),
            json!({"name": "invert", "fragment": GOOD}).to_string(),
        )
        .unwrap();

        let reg = load_filters(&[&dir]);
        assert!(reg.contains_key("dither") && reg.contains_key("invert"));
        assert!(!reg.contains_key("nameless") && !reg.contains_key("typo"));

        let prev = std::env::var("GHEE_FILTERS").ok();
        std::env::set_var("GHEE_FILTERS", &dir);
        let reg = load_filters(&[]);
        assert!(reg.contains_key("dither") && reg.contains_key("invert"));
        // A later directory wins a name clash.
        std::fs::write(
            format!("{dir}/dither.filter.json"),
            json!({"name": "dither", "fragment": "void main(){}", "params": {}}).to_string(),
        )
        .unwrap();
        assert_eq!(
            load_filters(&[])["dither"]["fragment"],
            json!("void main(){}")
        );
        // A shadowed name keeps its first insertion position.
        assert_eq!(load_filters(&[]).keys().next().unwrap(), "dither");
        match prev {
            Some(v) => std::env::set_var("GHEE_FILTERS", v),
            None => std::env::remove_var("GHEE_FILTERS"),
        }
    }

    #[test]
    fn a_filter_pass_carries_the_filter_params() {
        let _g = env_guard();
        let mut sc = Scene::of(vec![Pass::of("main", FLAT).unwrap()]).unwrap();
        apply_filters(&mut sc, &[crate::filters::Filter::named("dither")], None).unwrap();
        let amount = sc.passes[1].params.get("amount").unwrap();
        assert_eq!(amount.ty, "float");
        assert_eq!(amount.min, Some(json!(0.0)));
        assert!(sc.passes[1].params.get("levels").is_some());
        let _ = ParamMap::new();
    }

    #[test]
    fn repeated_filters_chain_and_keep_renaming() {
        let _g = env_guard();
        let mut sc = Scene::of(vec![Pass::of("dither", GOOD).unwrap()]).unwrap();
        let three = [
            Filter::named("dither"),
            Filter::named("dither"),
            Filter::named("dither"),
        ];
        apply_filters(&mut sc, &three, None).unwrap();
        assert_eq!(names(&sc), ["dither", "dither1", "dither2", "dither3"]);
        let chain: Vec<Option<String>> = sc
            .passes
            .iter()
            .flat_map(|p| p.channels.iter().map(|c| c.buffer.clone()))
            .collect();
        assert_eq!(
            chain,
            [
                Some("dither".to_string()),
                Some("dither1".to_string()),
                Some("dither2".to_string())
            ]
        );
        // The mark is the name asked for. The pass name can differ.
        let marks: Vec<&str> = sc.passes.iter().map(|p| p.filter.as_str()).collect();
        assert_eq!(marks, ["", "dither", "dither", "dither"]);
    }

    #[test]
    fn an_entry_starts_its_filter_at_its_values() {
        let _g = env_guard();
        let calm = |sc: &Scene| sc.passes[1].params.get("calm").unwrap().value.to_json();
        let entry = json!({"name": "frc", "calm": 8});
        let mut sc = Scene::of(vec![Pass::of("main", FLAT).unwrap()]).unwrap();
        apply_filters(&mut sc, &[Filter::from_json(&entry).unwrap()], None).unwrap();
        assert_eq!(calm(&sc), json!(8.0));
        // A post that carries the pass keeps the pass's value.
        let mut again = Scene::from_dict(&sc.to_dict(), None, None).unwrap();
        again.passes[1].params.get_mut("calm").unwrap().value = ParamValue::Float(10.0);
        let mut again = Scene::from_dict(&again.to_dict(), None, None).unwrap();
        rebuild_filters(&mut again, None).unwrap();
        assert_eq!(calm(&again), json!(10.0));
        for (entry, want) in [
            (
                json!({"name": "frc", "clam": 8}),
                "filter 'frc': param 'clam' is not a param this filter declares (coarse, calm)",
            ),
            (
                json!({"name": "frc", "calm": "lots"}),
                "filter 'frc': float param 'calm' needs a number, got 'lots'",
            ),
        ] {
            let mut sc = Scene::of(vec![Pass::of("main", FLAT).unwrap()]).unwrap();
            let f = Filter::from_json(&entry).unwrap();
            let e = apply_filters(&mut sc, &[f], None).unwrap_err();
            assert_eq!(e.to_string(), want);
        }
    }

    #[test]
    fn the_command_line_writes_a_filter_s_values_after_colons() {
        let got = Filter::parse_list("frc:calm=8:coarse=1, dither,,").unwrap();
        let want: Vec<Filter> = [
            json!({"name": "frc", "calm": 8, "coarse": 1}),
            json!("dither"),
        ]
        .iter()
        .map(|v| Filter::from_json(v).unwrap())
        .collect();
        assert_eq!(got, want);
        let e = Filter::parse_list("frc:calm").unwrap_err();
        assert_eq!(
            e.to_string(),
            "filter 'frc:calm': 'calm' needs the form param=value"
        );
        // A later filter of the same name sets its values in the earlier one's place.
        let first = Filter::parse_list("frc:calm=8,dither").unwrap();
        let later = Filter::parse_list("dither:levels=4,frc:coarse=1,frc256").unwrap();
        let both = Filter::parse_list("frc:calm=8:coarse=1,dither:levels=4,frc256").unwrap();
        assert_eq!(chain(&first, &later), both);
    }

    #[test]
    fn a_filter_without_params_appends_none() {
        let mut reg = Map::new();
        reg.insert(
            "invert".to_string(),
            json!({"name": "invert", "fragment": "void main(){}"}),
        );
        let mut sc = Scene::of(vec![Pass::of("main", FLAT).unwrap()]).unwrap();
        apply_filters(
            &mut sc,
            &[crate::filters::Filter::named("invert")],
            Some(&reg),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_string(&sc.passes[1].to_dict()).unwrap(),
            r#"{"name":"invert","fragment":"void main(){}","channels":[{"buffer":"main"}],"filter":"invert"}"#
        );
    }
}
