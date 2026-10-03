//! ghee ships named post-process passes, such as dither.
//!
//! A filter file (`<name>.filter.json`) holds one pass: a name, code and params.
//! The pass reads iChannel0, the frame so far, and writes the result. ghee
//! appends a filter as a pass that reads the pass before it. The last filter
//! reaches the display.
//!
//! The build compiles the shipped filters in. A scan adds more from a directory.

use anyhow::{bail, Result};
use serde_json::{Map, Value};

use crate::scene::{inside, join, normpath, strerror, Asset, Pass, RawLayout, Scene};

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
        && fd.get("code").is_some_and(Value::is_string);
    if !named {
        eprintln!("filter {path}: skipped (needs a string 'name' and 'code')");
        return None;
    }
    let name = fd["name"].as_str().unwrap_or_default().to_string();
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

/// Append each named filter as a pass, in order. Each one reads the pass before
/// it, so the last filter reaches the display.
pub fn apply_filters(
    scene: &mut Scene,
    names: &[String],
    registry: Option<&Map<String, Value>>,
) -> Result<()> {
    if names.is_empty() || scene.passes.is_empty() {
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
    for name in names {
        let Some(fd) = reg.get(name) else {
            eprintln!(
                "filter '{name}': unknown, skipped (known: {})",
                reg.keys().cloned().collect::<Vec<_>>().join(", ")
            );
            continue;
        };
        let Some(fname) = fd.get("name").and_then(Value::as_str) else {
            bail!("filter '{name}': needs a string 'name' and 'code'");
        };
        let own = unique(scene, fname);
        let prev = scene.passes[scene.passes.len() - 1].name.clone();
        let p = filter_pass(name, fd, &own, &prev, None)?;
        scene.passes.push(p);
        merge_assets(scene, name, fd)?;
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
        if let Some(i) = missing.iter().position(|n| *n == p.filter) {
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
    let unknown = |name: &str| {
        anyhow::anyhow!(
            "filter '{name}': unknown (known: {})",
            reg.keys().cloned().collect::<Vec<_>>().join(", ")
        )
    };
    for i in 0..scene.passes.len() {
        let name = scene.passes[i].filter.clone();
        if name.is_empty() {
            continue;
        }
        let fd = reg.get(&name).ok_or_else(|| unknown(&name))?;
        if i == 0 {
            bail!("filter '{name}': a filter pass needs a pass before it");
        }
        let own = scene.passes[i].name.clone();
        let prev = scene.passes[i - 1].name.clone();
        let p = filter_pass(&name, fd, &own, &prev, Some(&scene.passes[i]))?;
        scene.passes[i] = p;
        merge_assets(scene, &name, fd)?;
    }
    if let Some(name) = missing.iter().find(|n| !reg.contains_key(*n)) {
        return Err(unknown(name));
    }
    apply_filters(scene, &missing, Some(reg))?;
    match scene.name_clash() {
        Some(e) => bail!("{e}"),
        None => Ok(()),
    }
}

/// One filter pass named `own`, reading `prev`. The param values of `was`
/// replace the defaults where the filter still declares that param.
fn filter_pass(name: &str, fd: &Value, own: &str, prev: &str, was: Option<&Pass>) -> Result<Pass> {
    let Some(code) = fd.get("code").and_then(Value::as_str) else {
        bail!("filter '{name}': needs a string 'name' and 'code'");
    };
    let mut params = fd
        .get("params")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    if let (Some(was), Value::Object(defs)) = (was, &mut params) {
        for (n, def) in defs.iter_mut() {
            if let (Some(p), Value::Object(def)) = (was.params.get(n), def) {
                def.insert("value".to_string(), p.value.to_json());
            }
        }
    }
    let mut d = Map::new();
    d.insert("name".to_string(), Value::from(own));
    d.insert("code".to_string(), Value::from(code));
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
        let texts = take(&mut scene.texts);
        *scene = Scene::new(
            take(&mut scene.passes),
            take(&mut scene.version),
            take(&mut scene.filters),
            take(&mut scene.params),
            take(&mut scene.common),
            take(&mut scene.assets),
        )
        .and_then(|s| s.with_texts(texts))
        .map_err(|e| anyhow::anyhow!("filter '{filter}': {e}"))?;
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

fn unique(scene: &Scene, base: &str) -> String {
    let taken: Vec<&str> = scene.passes.iter().map(|p| p.name.as_str()).collect();
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
    use crate::scene::{ParamMap, Pass, Scene};
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
            &json!({"version": 1, "passes": [{"name": "m", "code": GOOD}]}),
            None,
            None,
        )
        .unwrap();
        apply_filters(&mut sc, &["frc256".to_string()], None).unwrap();
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
            json!({"name": "lettered", "code": GOOD,
                   "assets": {"face": {"file": "face.ttf"}}})
            .to_string(),
        )
        .unwrap();
        let reg = load_filters(&[&d]);
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "passes": [{"name": "m", "code": GOOD}]}),
            None,
            None,
        )
        .unwrap();
        let e = apply_filters(&mut sc, &["lettered".to_string()], Some(&reg))
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
            json!({"name": "masked", "code": GOOD,
                   "assets": {"m": {"file": "m.raw", "width": 2, "height": 2, "format": "u8"}}})
            .to_string(),
        )
        .unwrap();
        let reg = load_filters(&[&d]);
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "passes": [{"name": "m", "code": GOOD}]}),
            None,
            None,
        )
        .unwrap();
        apply_filters(&mut sc, &["masked".to_string()], Some(&reg)).unwrap();
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
        // name and param values count; the code and the asset come from the
        // filter.
        let _g = env_guard();
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "passes": [{"name": "m", "code": GOOD}]}),
            None,
            None,
        )
        .unwrap();
        apply_filters(&mut sc, &["frc256".to_string()], None).unwrap();
        let mut doc = sc.to_dict();
        // A filter's asset stays out of the document, and comes back with its
        // filter.
        assert!(doc.get("assets").is_none(), "{doc}");
        doc["passes"][1]["code"] = json!("void main(){ gl_FragColor = vec4(0.0); }");
        let mut back = Scene::from_dict(&doc, None, None).unwrap();
        rebuild_filters(&mut back, None).unwrap();
        assert_eq!(back.passes[1].code, sc.passes[1].code);
        let img = back.assets[0].image.as_ref().expect("decoded");
        assert_eq!((img.width, img.height), (64, 64));
    }

    #[test]
    fn a_filter_asset_never_meets_a_scene_name() {
        // The scene has its own bluenoise and a param of that name. The filter
        // still samples its own mask under the name it wrote.
        let _g = env_guard();
        let d = tmpdir("clash");
        std::fs::write(format!("{d}/bluenoise.raw"), [1u8, 2, 3, 4]).unwrap();
        let mut sc = Scene::from_dict(
            &json!({"version": 1,
                    "passes": [{"name": "m", "code": GOOD}],
                    "assets": {"bluenoise": {"file": "bluenoise.raw",
                                             "width": 2, "height": 2, "format": "u8"}}}),
            None,
            None,
        )
        .unwrap();
        sc.load_assets(&d).unwrap();
        apply_filters(&mut sc, &["frc256".to_string()], None).unwrap();
        let names: Vec<&str> = sc.assets.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["bluenoise", "frc256_bluenoise"]);
        let main = sc.passes[0].glsl(Some(&sc.params), None, &sc.assets, &[], false);
        assert!(main.contains("uniform sampler2D bluenoise;"));
        assert!(!main.contains("frc256_bluenoise"));
        let filt = sc.passes[1].glsl(Some(&sc.params), None, &sc.assets, &[], false);
        assert!(filt.contains("uniform sampler2D frc256_bluenoise;"));
        assert!(filt.contains("#define bluenoise frc256_bluenoise"));
        assert!(!filt.contains("uniform sampler2D bluenoise;"));

        let mut sp = Scene::from_dict(
            &json!({"version": 1, "passes": [{"name": "m", "code": GOOD}],
                    "params": {"bluenoise": {"type": "float", "value": 0.5}}}),
            None,
            None,
        )
        .unwrap();
        apply_filters(&mut sp, &["frc256".to_string()], None).unwrap();
        let filt = sp.passes[1].glsl(Some(&sp.params), None, &sp.assets, &[], false);
        let (decl, def) = (
            filt.find("uniform float bluenoise;").unwrap(),
            filt.find("#define bluenoise ").unwrap(),
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
            json!({"name": "masked", "code": GOOD,
                   "assets": {"m": {"file": "m.raw", "width": 2, "height": 2, "format": "u8"}}})
            .to_string(),
        )
        .unwrap();
        let reg = load_filters(&[&d]);
        let first = |reg: &Map<String, Value>| {
            let mut sc = Scene::from_dict(
                &json!({"version": 1, "passes": [{"name": "m", "code": GOOD}]}),
                None,
                None,
            )
            .unwrap();
            apply_filters(&mut sc, &["masked".to_string()], Some(reg)).unwrap();
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
                {"name": "m", "code": GOOD},
                {"name": "dither", "code": "", "filter": "dither",
                 "params": {"levels": {"type": "float", "value": 4.0}}}]}),
            None,
            None,
        )
        .unwrap();
        rebuild_filters(&mut sc, None).unwrap();
        let p = &sc.passes[1];
        assert!(p.code.contains("main"));
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
                    "passes": [{"name": "m", "code": GOOD}]}),
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
    fn a_listed_filter_the_post_leaves_out_stays_out() {
        // The console mutes a filter by posting it as a plain pass and leaving
        // its name out of the list.
        let _g = env_guard();
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "filters": [],
                    "passes": [{"name": "m", "code": GOOD},
                               {"name": "frc256", "code": GOOD}]}),
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
                    "passes": [{"name": "m", "code": GOOD}]}),
            None,
            None,
        )
        .unwrap();
        let e = rebuild_filters(&mut sc, None).unwrap_err().to_string();
        assert!(e.contains("'nope': unknown"), "{e}");
    }

    #[test]
    fn an_unknown_posted_filter_fails() {
        let _g = env_guard();
        let mut sc = Scene::from_dict(
            &json!({"version": 1, "passes": [
                {"name": "m", "code": GOOD},
                {"name": "x", "code": GOOD, "filter": "nope"}]}),
            None,
            None,
        )
        .unwrap();
        let e = rebuild_filters(&mut sc, None).unwrap_err().to_string();
        assert!(e.contains("'nope': unknown"), "{e}");
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
            json!({"name": "plain", "code": GOOD}).to_string(),
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
        apply_filters(&mut sc, &["dither".to_string()], None).unwrap();
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
    fn unknown_filter_skipped() {
        // Hosts share a filter list, so a missing filter still renders the scene.
        let _g = env_guard();
        let mut sc = Scene::of(vec![Pass::of("main", FLAT).unwrap()]).unwrap();
        apply_filters(&mut sc, &["does-not-exist".to_string()], None).unwrap();
        assert_eq!(names(&sc), ["main"]);
    }

    #[test]
    fn a_filter_channel_names_itself_for_feedback() {
        let _g = env_guard();
        let mut reg = Map::new();
        let accum: Value = serde_json::from_str(
            r#"{"name":"accum","code":"void main(){gl_FragColor=mix(texture2D(iChannel1,gl_FragCoord.xy/resolution),texture2D(iChannel0,gl_FragCoord.xy/resolution),0.3);}","channels":[{"buffer":"self"}]}"#,
        )
        .unwrap();
        reg.insert("accum".to_string(), accum);
        let mut sc = Scene::of(vec![Pass::of("main", GOOD).unwrap()]).unwrap();
        apply_filters(&mut sc, &["accum".to_string()], Some(&reg)).unwrap();
        let filt = &sc.passes[1];
        assert_eq!(filt.channels[0].buffer.as_deref(), Some("main"));
        assert_eq!(filt.channels[1].buffer.as_deref(), Some("accum"));
        Scene::of(sc.passes.clone()).unwrap();
    }

    #[test]
    fn filter_pass_never_collides_with_a_scene_pass() {
        let _g = env_guard();
        let mut sc = Scene::of(vec![Pass::of("dither", GOOD).unwrap()]).unwrap();
        apply_filters(&mut sc, &["dither".to_string()], None).unwrap();
        assert_eq!(names(&sc), ["dither", "dither1"]);
        Scene::of(sc.passes.clone()).unwrap();
    }

    #[test]
    fn a_filter_pass_never_sees_the_common() {
        let _g = env_guard();
        let d = json!({
            "version": 1,
            "passes": [{"name": "main", "code": GOOD}],
            "common": {"code": COMMON},
        });
        let mut sc = Scene::from_dict(&d, None, None).unwrap();
        apply_filters(&mut sc, &["dither".to_string()], None).unwrap();
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
            json!({"name": "invert", "code": "void main(){}", "params": {}}),
        );
        let mut sc = Scene::of(vec![Pass::of("main", FLAT).unwrap()]).unwrap();
        apply_filters(&mut sc, &["invert".to_string()], Some(&reg)).unwrap();
        assert_eq!(names(&sc), ["main", "invert"]);
    }

    #[test]
    fn extra_dirs_and_the_env_add_filters_and_skip_broken_ones() {
        let _g = env_guard();
        let dir = tmpdir("env");
        std::fs::write(format!("{dir}/broken.filter.json"), "{not json").unwrap();
        std::fs::write(
            format!("{dir}/nameless.filter.json"),
            json!({"code": "x"}).to_string(),
        )
        .unwrap();
        std::fs::write(
            format!("{dir}/invert.filter.json"),
            json!({"name": "invert", "code": GOOD}).to_string(),
        )
        .unwrap();

        let reg = load_filters(&[&dir]);
        assert!(reg.contains_key("dither") && reg.contains_key("invert"));
        assert!(!reg.contains_key("nameless"));

        let prev = std::env::var("GHEE_FILTERS").ok();
        std::env::set_var("GHEE_FILTERS", &dir);
        let reg = load_filters(&[]);
        assert!(reg.contains_key("dither") && reg.contains_key("invert"));
        // A later directory wins a name clash.
        std::fs::write(
            format!("{dir}/dither.filter.json"),
            json!({"name": "dither", "code": "void main(){}", "params": {}}).to_string(),
        )
        .unwrap();
        assert_eq!(load_filters(&[])["dither"]["code"], json!("void main(){}"));
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
        apply_filters(&mut sc, &["dither".to_string()], None).unwrap();
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
            "dither".to_string(),
            "dither".to_string(),
            "dither".to_string(),
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
    fn a_filter_without_params_appends_none() {
        let mut reg = Map::new();
        reg.insert(
            "invert".to_string(),
            json!({"name": "invert", "code": "void main(){}"}),
        );
        let mut sc = Scene::of(vec![Pass::of("main", FLAT).unwrap()]).unwrap();
        apply_filters(&mut sc, &["invert".to_string()], Some(&reg)).unwrap();
        assert_eq!(
            serde_json::to_string(&sc.passes[1].to_dict()).unwrap(),
            r#"{"name":"invert","code":"void main(){}","channels":[{"buffer":"main"}],"filter":"invert"}"#
        );
    }
}
