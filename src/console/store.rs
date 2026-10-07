//! A console shows, edits and saves these scenes.
//!
//! A store holds a directory of .scene.json and executable scenes. A save
//! writes into
//! modified/ beside them and never touches the original. A JSON scene saves a
//! modified copy that shadows it. A procedural scene saves a .values.json
//! sidecar of its param values, its filter list, each pass's `visible` and
//! each layer's settings and effects. A load applies that sidecar over the
//! scene that the code wrote, and the code keeps which layers exist and their
//! order.
//!
//! A host whose scenes live elsewhere implements names, resolve and save of its
//! own, and hands them to its ConsoleHost.

use std::fs;
use std::path::{Component, Path, PathBuf};

use serde_json::{Map, Value};

use crate::console::core::{ConsoleError, ConsoleResult};
use crate::filters::apply_filters;
use crate::layer::Layer;
use crate::procedural::process::{self, SceneProcess};
use crate::scene::{py_repr, Param, Reach, Scene};

const SCENE_SUFFIX: &str = ".scene.json";

/// Lay the saved settings of each layer in `saved` over the layer of that
/// name in `doc`, the scene document of procedural scene `name`. The layer
/// keeps its name and its words. A saved layer that the document no longer
/// declares, or whose settings the layer there refuses, is reported and
/// skipped.
fn overlay_layers(doc: &mut Value, name: &str, saved: &Map<String, Value>) {
    let mut found: Vec<String> = Vec::new();
    let layers = doc
        .get_mut("passes")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(|p| p.get_mut("layers")?.as_array_mut())
        .flatten();
    for l in layers {
        let Some(n) = l.get("name").and_then(Value::as_str).map(str::to_string) else {
            continue;
        };
        let Some(Value::Object(keys)) = saved.get(&n) else {
            continue;
        };
        found.push(n.clone());
        let mut over = keys.clone();
        over.insert("name".to_string(), Value::from(n.as_str()));
        if let Some(words) = l.get("text") {
            over.insert("text".to_string(), words.clone());
        }
        let over = Value::Object(over);
        match Layer::from_dict("a layer", &over) {
            Ok(_) => *l = over,
            Err(e) => eprintln!("values {name}: bad layer '{n}': {e}"),
        }
    }
    for n in saved.keys().filter(|n| !found.contains(n)) {
        eprintln!("values {name}: no layer '{n}'");
    }
}

/// Overlay a values sidecar onto the resolved scene: the params and the eye
/// of each pass. A filter pass takes the overlay too, because its params are
/// tweakable. The code decides which passes and params exist, so this
/// reports an unknown name and skips it, and a sidecar that would hide every
/// pass leaves the eyes alone.
pub fn apply_values(scene: &mut Scene, name: &str, vals: &Value) {
    let empty = Map::new();
    let saved_passes = vals
        .get("passes")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    let mut targets: Vec<(Option<usize>, String, &Map<String, Value>)> = vec![(
        None,
        "scene".to_string(),
        vals.get("scene")
            .and_then(Value::as_object)
            .unwrap_or(&empty),
    )];
    for (pn, params) in saved_passes {
        match scene.passes.iter().position(|p| p.name == *pn) {
            None => eprintln!("values {name}: no pass '{pn}'"),
            Some(i) => targets.push((Some(i), pn.clone(), params.as_object().unwrap_or(&empty))),
        }
    }
    for (index, label, saved) in targets {
        for (n, v) in saved {
            let existing = match index {
                None => &mut scene.params,
                Some(i) => &mut scene.passes[i].params,
            };
            let Some(p) = existing.get(n) else {
                eprintln!("values {name}: no param '{label}.{n}'");
                continue;
            };
            let numeric = p.ty == "float" || p.ty == "int";
            if (numeric && (v.is_boolean() || !v.is_number()))
                || (p.ty == "bool" && !v.is_boolean())
            {
                eprintln!("values {name}: bad '{label}.{n}': {}", py_repr(v));
                continue;
            }
            let built = Param::new(
                &p.name,
                &p.ty,
                v.clone(),
                p.min.clone(),
                p.max.clone(),
                p.step.clone(),
                p.options.clone().map(Value::from),
            );
            match built {
                // The words of a text layer stay the param that ghee declared.
                Ok(param) => existing.insert(
                    n.clone(),
                    Param {
                        of_text: p.of_text,
                        ..param
                    },
                ),
                Err(e) => eprintln!("values {name}: bad '{label}.{n}': {e}"),
            }
        }
    }
    let Some(Value::Object(eyes)) = vals.get("visible") else {
        return;
    };
    let mut visible: Vec<bool> = scene.passes.iter().map(|p| p.visible).collect();
    for (pn, v) in eyes {
        match (scene.passes.iter().position(|p| p.name == *pn), v) {
            (None, _) => eprintln!("values {name}: no pass '{pn}'"),
            (Some(i), Value::Bool(b)) => visible[i] = *b,
            (Some(_), v) => eprintln!("values {name}: bad '{pn}.visible': {}", py_repr(v)),
        }
    }
    if !visible.contains(&true) {
        eprintln!("values {name}: every pass hidden, so the passes keep their eyes");
        return;
    }
    for (p, v) in scene.passes.iter_mut().zip(visible) {
        p.visible = v;
    }
}

pub struct SceneStore {
    pub dir: String,
    pub read_only: bool,
    /// Scene args for the whole session. The store hands them to every scene it
    /// loads.
    pub args: Vec<String>,
    /// Each scene name and its path, in the order a scan found them.
    pub scenes: Vec<(String, String)>,
    /// Every scene in this store ends with these filters, after its own. A host
    /// whose panel needs one (a 1-bit panel needs the dither) sets them here
    /// instead of writing them into every scene.
    pub filters: Vec<String>,
}

impl SceneStore {
    pub fn new(scenes_dir: &str, read_only: bool, args: Vec<String>) -> SceneStore {
        let dir = abspath(scenes_dir);
        let scenes = scan(&dir);
        SceneStore {
            dir,
            read_only,
            args,
            scenes,
            filters: Vec::new(),
        }
    }

    /// A store over one scene: its own directory, narrowed to that scene.
    /// `render <scene> --serve` shows this. `serve <dir>` shows the whole list.
    pub fn one(path: &str, read_only: bool, args: Vec<String>) -> ConsoleResult<SceneStore> {
        let full = abspath(path);
        let (d, mut base) = match full.rsplit_once('/') {
            Some((d, b)) => (d.to_string(), b.to_string()),
            None => (String::new(), full.clone()),
        };
        base = crate::procedural::scene_name(&base);
        let mut store = SceneStore::new(&d, read_only, args);
        if !store.contains(&base) {
            return Err(ConsoleError::not_found(format!("no scene '{base}' in {d}")));
        }
        store.scenes.retain(|(n, _)| *n == base);
        Ok(store)
    }

    pub fn names(&self) -> Vec<String> {
        self.scenes.iter().map(|(n, _)| n.clone()).collect()
    }

    /// The names of the scenes that a program writes, as start() runs them.
    pub fn procedural_names(&self) -> Vec<String> {
        self.scenes
            .iter()
            .filter(|(n, _)| {
                self.path(n)
                    .is_ok_and(|p| process::executable(&p).is_some())
            })
            .map(|(n, _)| n.clone())
            .collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.scenes.iter().any(|(n, _)| n == name)
    }

    fn source(&self, name: &str) -> Option<&str> {
        self.scenes
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p.as_str())
    }

    fn modified_path(&self, name: &str) -> String {
        format!("{}/modified/{name}{SCENE_SUFFIX}", self.dir)
    }

    fn values_path(&self, name: &str) -> String {
        format!("{}/modified/{name}.values.json", self.dir)
    }

    /// The saved values sidecar for a procedural scene, or None.
    fn read_values(&self, name: &str) -> Option<Value> {
        let path = self.values_path(name);
        if !Path::new(&path).is_file() {
            return None;
        }
        match fs::read_to_string(&path).map(|t| serde_json::from_str::<Value>(&t)) {
            Ok(Ok(v)) => Some(v),
            Ok(Err(e)) => {
                eprintln!("values {name}: unreadable ({e})");
                None
            }
            Err(e) => {
                eprintln!("values {name}: unreadable ({e})");
                None
            }
        }
    }

    /// The path behind a scene load. A saved modified copy shadows the original,
    /// and nothing writes the original.
    pub fn path(&self, name: &str) -> ConsoleResult<String> {
        let p = self
            .source(name)
            .ok_or_else(|| ConsoleError::not_found("no scene"))?
            .to_string();
        let modified = self.modified_path(name);
        if p.ends_with(SCENE_SUFFIX) && Path::new(&modified).is_file() {
            return Ok(modified);
        }
        Ok(p)
    }

    /// A scene ready to render and edit: GLSL files as their text, declared
    /// filters expanded, saved values overlaid.
    ///
    /// The store hands the session's scene args to every scene it loads. A scene
    /// keeps what it declares and ignores the rest.
    ///
    /// Every read goes through sources::record, so a caller that wraps this in
    /// collect() gets the files behind the result. The values sidecar stays out
    /// of that set: /set and /save write one, and an applied param leaves the
    /// renderer up to date.
    pub fn resolve(&self, name: &str) -> ConsoleResult<Scene> {
        self.start(name).map(|(scene, _)| scene)
    }

    /// resolve(), plus the process when the scene is procedural. The process runs
    /// from here on. A caller that wants the document only drops it, which stops
    /// it.
    pub fn start(&self, name: &str) -> ConsoleResult<(Scene, Option<SceneProcess>)> {
        let path = self.path(name)?;
        // Anyone who reaches the console posts and saves the scenes that it
        // loads, so their files stay inside the store. A modified copy finds
        // its files beside its original.
        let vals = match self.source(name) {
            Some(p) if p.ends_with(SCENE_SUFFIX) => None,
            _ => self.read_values(name),
        };
        let layers = vals.as_ref().and_then(|v| v.get("layers"));
        let overlay = |doc: &mut Value| {
            if let Some(Value::Object(saved)) = layers {
                overlay_layers(doc, name, saved);
            }
        };
        let (mut scene, mut process) = match process::executable(&path) {
            // the console error state reaches a browser, so it carries the
            // stderr of a dead process
            Some(exe) => SceneProcess::start(&exe, &self.args, Reach::Inside, overlay)
                .map(|(scene, fed)| (scene, Some(fed)))
                .map_err(|e| ConsoleError::server_error(process::with_stderr(&e)))?,
            None => (
                Scene::load_in(&path, Some(&self.dir), Reach::Inside)
                    .map_err(|e| ConsoleError::server_error(e.to_string()))?,
                None,
            ),
        };
        if let Some(Value::Array(items)) = vals.as_ref().and_then(|v| v.get("filters")) {
            scene.filters = items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect();
        }
        // The scene's own filters run first, the store's follow, so a host
        // filter like the dither lands last where the panel needs it.
        let mut filters = scene.filters.clone();
        for name in &self.filters {
            if !filters.contains(name) {
                filters.push(name.clone());
            }
        }
        apply_filters(&mut scene, &filters, None)
            .map_err(|e| ConsoleError::server_error(e.to_string()))?;
        if let Some(vals) = vals {
            apply_values(&mut scene, name, &vals);
        }
        if let Some(fed) = process.as_mut() {
            fed.bind(&scene);
        }
        Ok((scene, process))
    }

    pub fn save(&self, name: &str, mut scene: Scene) -> ConsoleResult<()> {
        let source = self
            .source(name)
            .ok_or_else(|| ConsoleError::not_found("no scene"))?
            .to_string();
        // The working copy carries an applied filter as an expanded pass.
        // Recover the declared list from them, so a save keeps a filter the
        // client did not record.
        let applied: Vec<String> = scene
            .passes
            .iter()
            .filter(|p| !p.filter.is_empty())
            .map(|p| p.filter.clone())
            .collect();
        if !applied.is_empty() {
            scene.filters = applied;
        }
        if !source.ends_with(SCENE_SUFFIX) {
            // A procedural scene's shape lives in code. Save the param values,
            // the filter list, the eyes and the layers as a sidecar. The
            // loader applies that over the code's own scene.
            let mut vals = Map::new();
            vals.insert("filters".to_string(), Value::from(scene.filters.clone()));
            vals.insert("scene".to_string(), values_of(&scene.params));
            let (mut passes, mut visible, mut layers) = (Map::new(), Map::new(), Map::new());
            for ps in &scene.passes {
                passes.insert(ps.name.clone(), values_of(&ps.params));
                visible.insert(ps.name.clone(), Value::from(ps.visible));
                for l in &ps.layers {
                    layers.insert(l.name().to_string(), settings_of(l));
                }
            }
            vals.insert("passes".to_string(), Value::Object(passes));
            vals.insert("visible".to_string(), Value::Object(visible));
            vals.insert("layers".to_string(), Value::Object(layers));
            return write(&self.values_path(name), &Value::Object(vals));
        }
        // Store the scene unexpanded. A load rebuilds each filter pass from the
        // filters list, and a hidden filter's entry keeps its eye.
        scene.hidden_filters = scene
            .passes
            .iter()
            .filter(|p| !p.filter.is_empty() && !p.visible)
            .map(|p| p.filter.clone())
            .collect();
        scene.passes.retain(|p| p.filter.is_empty());
        scene.assets.retain(|a| a.from_filter.is_empty());
        write(&self.modified_path(name), &scene.to_dict())
    }
}

/// The settings and the effects of `layer` as a sidecar keeps them: the layer
/// as a scene declares it, less its name and its words, which the code and the
/// words param hold.
fn settings_of(layer: &Layer) -> Value {
    let mut d = layer.to_dict("");
    if let Value::Object(o) = &mut d {
        o.remove("name");
        o.remove("text");
    }
    d
}

fn values_of(params: &crate::scene::ParamMap) -> Value {
    let mut out = Map::new();
    for (n, p) in params.iter() {
        out.insert(n.clone(), p.value.to_json());
    }
    Value::Object(out)
}

fn write(path: &str, obj: &Value) -> ConsoleResult<()> {
    let io = |e: std::io::Error| ConsoleError::server_error(e.to_string());
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent).map_err(io)?;
    }
    let tmp = format!("{path}.tmp");
    let text =
        serde_json::to_string_pretty(obj).map_err(|e| ConsoleError::server_error(e.to_string()))?;
    fs::write(&tmp, text).map_err(io)?;
    fs::rename(&tmp, path).map_err(io)
}

/// The scenes in a directory, JSON first: every .scene.json, and every
/// executable file. An executable beside its .scene.json is one scene.
fn scan(dir: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut names: Vec<String> = match fs::read_dir(dir) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect(),
        Err(_) => return out,
    };
    names.sort();
    for n in &names {
        let Some(stem) = n.strip_suffix(SCENE_SUFFIX) else {
            continue;
        };
        out.push((stem.to_string(), format!("{dir}/{n}")));
    }
    for n in &names {
        if n.ends_with(SCENE_SUFFIX) {
            continue;
        }
        let path = format!("{dir}/{n}");
        // ghee runs a procedural scene as a process. A file without the
        // execute bit is a module or data the scenes beside it use, not a
        // scene.
        if !process::runnable(&path) {
            continue;
        }
        let stem = crate::procedural::scene_name(n);
        if out.iter().any(|(name, _)| *name == stem) {
            continue;
        }
        out.push((stem, path));
    }
    out
}

/// The path against the working directory. It resolves "." and ".." as text,
/// and it drops a trailing separator.
fn abspath(path: &str) -> String {
    let p = Path::new(path);
    let joined = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("/"))
            .join(p)
    };
    let mut parts: Vec<String> = Vec::new();
    for c in joined.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                parts.pop();
            }
            Component::Normal(s) => parts.push(s.to_string_lossy().into_owned()),
            Component::RootDir | Component::Prefix(_) => {}
        }
    }
    format!("/{}", parts.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{ParamMap, ParamValue, Pass};

    const SCENE: &str = r#"{"version": 1,
        "passes": [{"name": "main", "fragment": "void main(){ gl_FragColor = vec4(1.0); }"}]}"#;

    fn scene_json() -> Value {
        serde_json::from_str(SCENE).unwrap()
    }

    fn tmpdir(tag: &str) -> crate::testing::TempDir {
        crate::testing::TempDir::new(&format!("store-{tag}"))
    }

    fn write_text(path: &str, text: &str) {
        fs::write(path, text).unwrap();
    }

    /// Write a .py that the store takes for a scene. A scene process carries the
    /// execute bit.
    fn write_scene(path: &str, text: &str) {
        use std::os::unix::fs::PermissionsExt;
        fs::write(path, text).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    }

    fn read_json(path: &str) -> Value {
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
    }

    /// A host filter reaches a scene that declares none, and the scene's own
    /// filters keep their place in front of it.
    #[test]
    fn the_store_filters_every_scene_it_loads() {
        let d = tmpdir("host-filters");
        write_text(&format!("{d}/plain.scene.json"), SCENE);
        let mut store = SceneStore::new(&d, true, Vec::new());
        let (plain, _) = store.start("plain").unwrap();
        assert_eq!(plain.passes.len(), 1, "no filter yet");

        store.filters = vec!["dither".to_string()];
        let (dithered, _) = store.start("plain").unwrap();
        assert_eq!(dithered.passes.len(), 2);
        assert_eq!(dithered.passes[1].name, "dither");

        // a scene that already ends with the filter takes it once
        let own = r#"{"version": 1, "filters": ["dither"],
             "passes": [{"name": "main", "fragment": "void main(){ gl_FragColor = vec4(1.0); }"}]}"#;
        write_text(&format!("{d}/own.scene.json"), own);
        store.scenes = scan(&store.dir);
        let (once, _) = store.start("own").unwrap();
        assert_eq!(once.passes.len(), 2, "the dither is not applied twice");
    }

    /// A directory of two JSON scenes.
    fn scenes(tag: &str) -> (crate::testing::TempDir, SceneStore) {
        let d = tmpdir(tag);
        write_text(&format!("{d}/one{SCENE_SUFFIX}"), SCENE);
        write_text(&format!("{d}/two{SCENE_SUFFIX}"), SCENE);
        let store = SceneStore::new(&d, false, Vec::new());
        (d, store)
    }

    fn param(name: &str, ty: &str, value: Value) -> Param {
        Param::new(name, ty, value, None, None, None, None).unwrap()
    }

    #[test]
    fn the_store_scans_a_directory() {
        let (_d, store) = scenes("scan");
        assert_eq!(store.names(), vec!["one".to_string(), "two".to_string()]);
        assert!(store.contains("one") && !store.contains("three"));
    }

    #[test]
    fn the_store_names_the_scenes_that_a_program_writes() {
        let (d, _) = scenes("procedural-names");
        write_scene(&format!("{d}/proc.py"), "#!/bin/sh\nexit 0\n");
        write_text(&format!("{d}/helper.py"), "");
        let store = SceneStore::new(&d, false, Vec::new());
        assert_eq!(store.names(), ["one", "two", "proc"]);
        assert_eq!(store.procedural_names(), ["proc"]);
    }

    #[test]
    fn the_store_resolves_a_scene() {
        let (_d, store) = scenes("resolve");
        assert_eq!(store.resolve("one").unwrap().passes[0].name, "main");
    }

    #[test]
    fn the_store_refuses_an_unknown_name() {
        let (_d, store) = scenes("unknown");
        assert_eq!(store.path("three").unwrap_err().status, 404);
        let scene = Scene::from_dict(&scene_json(), None, None).unwrap();
        assert_eq!(store.save("three", scene).unwrap_err().status, 404);
    }

    #[test]
    fn a_save_shadows_the_original_without_touching_it() {
        let (d, store) = scenes("shadow");
        let before = fs::read_to_string(format!("{d}/one{SCENE_SUFFIX}")).unwrap();
        let mut edited = scene_json();
        edited["passes"][0]["fragment"] = Value::from("void main(){ discard; }");
        store
            .save("one", Scene::from_dict(&edited, None, None).unwrap())
            .unwrap();
        assert_eq!(
            fs::read_to_string(format!("{d}/one{SCENE_SUFFIX}")).unwrap(),
            before
        );
        assert!(Path::new(&format!("{d}/modified/one{SCENE_SUFFIX}")).is_file());
        assert!(store.resolve("one").unwrap().passes[0]
            .fragment
            .contains("discard"));
    }

    #[test]
    fn a_modified_copy_finds_its_files_beside_its_original_and_none_outside_the_store() {
        let root = tmpdir("files");
        let d = format!("{root}/scenes");
        fs::create_dir_all(&d).unwrap();
        fs::write(format!("{d}/d.raw"), [7u8; 4]).unwrap();
        fs::write(format!("{root}/out.raw"), [9u8; 4]).unwrap();
        let doc = |file: &str| {
            let mut doc = scene_json();
            doc["assets"] =
                serde_json::json!({"d": {"file": file, "width": 2, "height": 2, "format": "u8"}});
            doc.to_string()
        };
        write_text(&format!("{d}/one{SCENE_SUFFIX}"), &doc("d.raw"));
        write_text(&format!("{d}/two{SCENE_SUFFIX}"), &doc("../out.raw"));
        let store = SceneStore::new(&d, false, Vec::new());
        let mut edited = store.resolve("one").unwrap();
        edited.passes[0].fragment = "void main(){ discard; }".to_string();
        store.save("one", edited).unwrap();
        let copy = store.resolve("one").unwrap();
        assert!(copy.passes[0].fragment.contains("discard"));
        let img = copy.assets[0].image.as_ref().unwrap();
        assert_eq!(img.pixels, crate::assets::Pixels::U8(vec![7u8; 4]));
        let e = store.resolve("two").unwrap_err();
        assert!(
            e.message.contains("outside the scene directory"),
            "{}",
            e.message
        );
    }

    #[test]
    fn a_scene_json_scene_ignores_a_stray_sidecar() {
        // Only a procedural scene reads a sidecar. A JSON scene saves a whole
        // copy.
        let (d, store) = scenes("stray");
        fs::create_dir_all(format!("{d}/modified")).unwrap();
        write_text(
            &format!("{d}/modified/one.values.json"),
            r#"{"passes": {"main": {"nope": 1}}}"#,
        );
        assert_eq!(store.resolve("one").unwrap().passes[0].name, "main");
    }

    #[test]
    fn a_declared_filter_comes_back_as_a_pass() {
        let d = tmpdir("filter");
        let mut doc = scene_json();
        doc["filters"] = serde_json::json!(["dither"]);
        write_text(&format!("{d}/f{SCENE_SUFFIX}"), &doc.to_string());
        let store = SceneStore::new(&d, false, Vec::new());
        let scene = store.resolve("f").unwrap();
        assert_eq!(
            scene
                .passes
                .iter()
                .map(|p| p.filter.clone())
                .collect::<Vec<_>>(),
            vec!["".to_string(), "dither".to_string()]
        );
        store.save("f", scene).unwrap(); // the expanded pass must not be written back
        let written = read_json(&format!("{d}/modified/f{SCENE_SUFFIX}"));
        assert_eq!(written["filters"], serde_json::json!(["dither"]));
        assert_eq!(written["passes"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn a_hidden_filter_keeps_its_eye_through_a_save() {
        let d = tmpdir("filtereye");
        let mut doc = scene_json();
        doc["filters"] = serde_json::json!(["dither"]);
        write_text(&format!("{d}/f{SCENE_SUFFIX}"), &doc.to_string());
        let store = SceneStore::new(&d, false, Vec::new());
        let mut scene = store.resolve("f").unwrap();
        scene.passes[1].visible = false;
        store.save("f", scene).unwrap();
        let written = read_json(&format!("{d}/modified/f{SCENE_SUFFIX}"));
        assert_eq!(
            written["filters"],
            serde_json::json!([{"name": "dither", "visible": false}])
        );
        let again = store.resolve("f").unwrap();
        assert_eq!(
            (again.passes[1].filter.as_str(), again.passes[1].visible),
            ("dither", false)
        );
    }

    #[test]
    fn a_save_leaves_a_filter_asset_out() {
        let _g = crate::filters::env_guard();
        let d = tmpdir("filterasset");
        let mut doc = scene_json();
        doc["filters"] = serde_json::json!(["frc256"]);
        write_text(&format!("{d}/f{SCENE_SUFFIX}"), &doc.to_string());
        let store = SceneStore::new(&d, false, Vec::new());
        let scene = store.resolve("f").unwrap();
        assert_eq!(scene.assets[0].from_filter, "frc256");
        store.save("f", scene).unwrap();
        let written = read_json(&format!("{d}/modified/f{SCENE_SUFFIX}"));
        assert!(written.get("assets").is_none(), "{written}");
        assert_eq!(store.resolve("f").unwrap().assets.len(), 1);
    }

    #[test]
    fn a_save_writes_the_common_into_the_modified_copy() {
        let (d, store) = scenes("common");
        let mut scene = store.resolve("one").unwrap();
        scene.common = Some("float half(float x){ return x * 0.5; }".to_string());
        let want = scene.common.clone();
        store.save("one", scene).unwrap();
        let written = read_json(&format!("{d}/modified/one{SCENE_SUFFIX}"));
        assert_eq!(written["common"], serde_json::json!(want));
        assert_eq!(store.resolve("one").unwrap().common, want);
    }

    #[test]
    fn a_common_file_reference_saves_as_its_glsl_text() {
        let d = tmpdir("commonfile");
        write_text(&format!("{d}/shared.glsl"), "float k(){ return 1.0; }");
        let mut doc = scene_json();
        doc["common"] = serde_json::json!({"file": "shared.glsl"});
        write_text(&format!("{d}/c{SCENE_SUFFIX}"), &doc.to_string());
        let store = SceneStore::new(&d, false, Vec::new());
        let scene = store.resolve("c").unwrap();
        store.save("c", scene).unwrap();
        let written = read_json(&format!("{d}/modified/c{SCENE_SUFFIX}"));
        assert_eq!(
            written["common"],
            serde_json::json!("float k(){ return 1.0; }")
        );
    }

    #[test]
    fn the_store_reports_whether_it_may_be_written() {
        assert!(SceneStore::new(".", true, Vec::new()).read_only);
        assert!(!SceneStore::new(".", false, Vec::new()).read_only);
    }

    #[test]
    fn a_pass_keeps_its_params_through_a_round_trip() {
        let d = tmpdir("roundtrip");
        let mut params = ParamMap::new();
        params.insert("on".to_string(), param("on", "bool", Value::from(true)));
        let scene = Scene::of(vec![Pass::new(
            "main",
            "void main(){}",
            Vec::new(),
            Vec::new(),
            params,
            "",
            None,
        )
        .unwrap()])
        .unwrap();
        write_text(
            &format!("{d}/b{SCENE_SUFFIX}"),
            &scene.to_dict().to_string(),
        );
        let store = SceneStore::new(&d, false, Vec::new());
        let loaded = store.resolve("b").unwrap();
        assert_eq!(
            loaded.passes[0].params.get("on").unwrap().value.to_json(),
            Value::from(true)
        );
    }

    // a procedural scene: the values sidecar a save writes

    fn procedural_store(tag: &str) -> (crate::testing::TempDir, SceneStore) {
        let d = tmpdir(tag);
        write_scene(&format!("{d}/proc.py"), "#!/bin/sh\nexit 0\n");
        let store = SceneStore::new(&d, false, Vec::new());
        (d, store)
    }

    fn proc_scene() -> Scene {
        let mut params = ParamMap::new();
        params.insert(
            "amt".to_string(),
            Param::new(
                "amt",
                "float",
                Value::from(0.75),
                Some(Value::from(0.0)),
                Some(Value::from(1.0)),
                Some(Value::from(0.01)),
                None,
            )
            .unwrap(),
        );
        params.insert("on".to_string(), param("on", "bool", Value::from(true)));
        Scene::of(vec![Pass::new(
            "main",
            "void main(){ gl_FragColor = vec4(1.0); }",
            Vec::new(),
            Vec::new(),
            params,
            "",
            None,
        )
        .unwrap()])
        .unwrap()
    }

    #[test]
    fn a_procedural_scene_saves_a_values_sidecar() {
        let (d, store) = procedural_store("sidecar");
        store.save("proc", proc_scene()).unwrap();
        let vals = read_json(&format!("{d}/modified/proc.values.json"));
        assert_eq!(
            vals["passes"]["main"],
            serde_json::json!({"amt": 0.75, "on": true})
        );
        assert!(!Path::new(&format!("{d}/modified/proc{SCENE_SUFFIX}")).exists());
    }

    #[test]
    fn a_procedural_sidecar_carries_no_common() {
        // A procedural scene's shape lives in code. The sidecar holds param
        // values only.
        let (d, store) = procedural_store("sidecar-common");
        let mut scene = proc_scene();
        scene.common = Some("float k(){ return 1.0; }".to_string());
        store.save("proc", scene).unwrap();
        let vals = read_json(&format!("{d}/modified/proc.values.json"));
        assert!(vals.get("common").is_none());
    }

    #[test]
    fn a_scan_takes_the_json_scene_over_the_py_beside_it() {
        let d = tmpdir("pair");
        write_text(&format!("{d}/wob{SCENE_SUFFIX}"), SCENE);
        write_text(&format!("{d}/wob.py"), "# a scene process\n");
        let store = SceneStore::new(&d, false, Vec::new());
        assert_eq!(store.names(), vec!["wob".to_string()]);
        assert_eq!(store.path("wob").unwrap(), format!("{d}/wob{SCENE_SUFFIX}"));
    }

    /// A procedural scene imports its own modules from beside it. A module is
    /// not a scene, so the console does not list it.
    #[test]
    fn a_module_beside_the_scenes_is_not_a_scene() {
        let d = tmpdir("module");
        write_scene(&format!("{d}/feed.py"), "#!/bin/sh\nexit 0\n");
        write_text(&format!("{d}/look.py"), "\"\"\"a module\"\"\"\n");
        let store = SceneStore::new(&d, false, Vec::new());
        assert_eq!(store.names(), vec!["feed".to_string()]);
    }

    #[test]
    fn one_narrows_a_store_to_the_scene_it_names() {
        let (d, _) = scenes("one");
        let store = SceneStore::one(&format!("{d}/two{SCENE_SUFFIX}"), true, Vec::new()).unwrap();
        assert_eq!(store.names(), vec!["two".to_string()]);
        assert!(store.read_only);
        let missing = SceneStore::one(&format!("{d}/nope{SCENE_SUFFIX}"), true, Vec::new());
        assert_eq!(missing.err().map(|e| e.status), Some(404));
    }

    // the values overlay

    #[test]
    fn apply_values_overlays_and_keeps_shape() {
        let mut pass_params = ParamMap::new();
        pass_params.insert(
            "amt".to_string(),
            Param::new(
                "amt",
                "float",
                Value::from(0.5),
                Some(Value::from(0.0)),
                Some(Value::from(1.0)),
                Some(Value::from(0.01)),
                None,
            )
            .unwrap(),
        );
        let mut scene_params = ParamMap::new();
        scene_params.insert(
            "text".to_string(),
            param("text", "string", Value::from("GHEE")),
        );
        scene_params.insert(
            "font".to_string(),
            Param::new(
                "font",
                "string",
                Value::from("Verdana"),
                None,
                None,
                None,
                Some(serde_json::json!(["Verdana", "Impact"])),
            )
            .unwrap(),
        );
        let mut scene = Scene::new(
            vec![Pass::new(
                "m",
                "void main(){}",
                Vec::new(),
                Vec::new(),
                pass_params,
                "",
                None,
            )
            .unwrap()],
            Value::from(1),
            Vec::new(),
            scene_params,
            None,
            Vec::new(),
        )
        .unwrap();
        apply_values(
            &mut scene,
            "t",
            &serde_json::json!({
                "scene": {"text": "WOW", "font": "Impact", "gone": 1.0},
                "passes": {"m": {"amt": 0.9, "bad": "x"}, "nopass": {"a": 1}},
            }),
        );
        assert_eq!(
            scene.params.get("text").unwrap().value.to_json(),
            Value::from("WOW")
        );
        let font = scene.params.get("font").unwrap();
        assert_eq!(font.value.to_json(), Value::from("Impact"));
        assert_eq!(
            font.options.clone().unwrap(),
            vec!["Verdana".to_string(), "Impact".to_string()]
        ); // survives the overlay
        assert_eq!(
            scene.passes[0].params.get("amt").unwrap().value.to_json(),
            Value::from(0.9)
        );
    }

    #[test]
    fn apply_values_rejects_wrong_types() {
        let mut pass_params = ParamMap::new();
        pass_params.insert("amt".to_string(), param("amt", "float", Value::from(0.5)));
        let mut scene_params = ParamMap::new();
        scene_params.insert(
            "text".to_string(),
            param("text", "string", Value::from("GHEE")),
        );
        let mut scene = Scene::new(
            vec![Pass::new(
                "m",
                "void main(){}",
                Vec::new(),
                Vec::new(),
                pass_params,
                "",
                None,
            )
            .unwrap()],
            Value::from(1),
            Vec::new(),
            scene_params,
            None,
            Vec::new(),
        )
        .unwrap();
        apply_values(
            &mut scene,
            "t",
            &serde_json::json!({"scene": {"text": 5.0}, "passes": {"m": {"amt": "not-a-number"}}}),
        );
        assert_eq!(
            scene.params.get("text").unwrap().value.to_json(),
            Value::from("GHEE")
        ); // bad value skipped
        assert_eq!(
            scene.passes[0].params.get("amt").unwrap().value.to_json(),
            Value::from(0.5)
        );
    }

    #[test]
    fn a_sidecar_bool_written_as_a_number_is_skipped() {
        let mut params = ParamMap::new();
        params.insert("on".to_string(), param("on", "bool", Value::from(false)));
        let mut scene = Scene::of(vec![Pass::new(
            "main",
            "void main(){}",
            Vec::new(),
            Vec::new(),
            params,
            "",
            None,
        )
        .unwrap()])
        .unwrap();
        apply_values(
            &mut scene,
            "t",
            &serde_json::json!({"passes": {"main": {"on": 1}}}),
        );
        assert_eq!(
            scene.passes[0].params.get("on").unwrap().value.to_json(),
            Value::from(false)
        );
    }

    /// A procedural scene, proc, whose code writes passes a and b, the text
    /// layer title and the image layer icon over pic.png.
    fn layered_store(tag: &str) -> (crate::testing::TempDir, SceneStore) {
        let d = tmpdir(tag);
        let pic = crate::output::encode_png(crate::looper::Frame {
            data: &[255; 4],
            width: 1,
            height: 1,
            components: 4,
        })
        .unwrap();
        fs::write(format!("{d}/pic.png"), pic).unwrap();
        let doc = serde_json::json!({
            "version": 1,
            "assets": {"pic": {"file": "pic.png"}},
            "passes": [
                {"name": "a", "fragment": "void main(){ gl_FragColor = vec4(1.0); }",
                 "layers": [{"name": "title", "text": "hi", "x": 1, "karaoke": {}},
                            {"name": "icon", "image": "pic"}]},
                {"name": "b", "fragment": "void main(){ gl_FragColor = vec4(0.5); }"},
            ],
        });
        let body = format!("#!/bin/sh\nprintf '%s\\n' '{doc}'\nwhile read -r l; do :; done\n");
        crate::testing::script(&d, "proc", &body);
        let store = SceneStore::new(&d, false, Vec::new());
        (d, store)
    }

    /// The scene of proc as the store starts it.
    fn started(store: &SceneStore) -> Scene {
        let _spawn = crate::testing::SCRIPTS.read().unwrap();
        store.start("proc").unwrap().0
    }

    #[test]
    fn a_procedural_sidecar_keeps_each_eye_and_each_layer_over_the_layers_of_the_code() {
        let (d, store) = layered_store("layers");
        let code = started(&store);
        let mut scene = code.clone();
        scene.passes[0].visible = false;
        let at = |field| crate::layer::Address {
            layer: "title",
            effect: None,
            field,
        };
        scene.set_layer(&at("x"), &Value::from(10)).unwrap();
        scene
            .set_layer(&at("color"), &Value::from("#ff0000"))
            .unwrap();
        let sung = crate::layer::Address {
            effect: Some("karaoke"),
            ..at("sung")
        };
        scene.set_layer(&sung, &Value::from(2)).unwrap();
        if let Layer::Text(l) = &mut scene.passes[0].layers[0] {
            l.font_size = Some(30.0);
        }
        scene.params.get_mut("title").unwrap().value = ParamValue::Str("hey".to_string());
        store.save("proc", scene.clone()).unwrap();
        let path = format!("{d}/modified/proc.values.json");
        let mut vals = read_json(&path);
        assert_eq!(vals["visible"], serde_json::json!({"a": false, "b": true}));
        assert_eq!(
            vals["layers"]["title"],
            serde_json::json!({"x": 10.0, "color": "#ff0000", "font_size": 30.0,
                               "karaoke": {"sung": 2.0}})
        );
        // The code no longer declares a layer that the sidecar holds.
        vals["layers"]["gone"] = serde_json::json!({"x": 3});
        write_text(&path, &vals.to_string());
        let back = started(&store);
        assert_eq!(back.to_dict(), scene.to_dict());
        assert!(back.to_dict()["params"].get("title").is_none());
        // A sidecar that hides every pass leaves the code's eyes.
        vals["visible"] = serde_json::json!({"a": false, "b": false});
        write_text(&path, &vals.to_string());
        let eyes: Vec<bool> = started(&store).passes.iter().map(|p| p.visible).collect();
        assert_eq!(eyes, [true, true]);
        // A saved layer that the code's scene refuses leaves the code's layer.
        vals["visible"] = serde_json::json!({});
        vals["layers"]["title"]["font_family"] = serde_json::json!("nope");
        write_text(&path, &vals.to_string());
        assert_eq!(
            started(&store).to_dict(),
            {
                let mut want = code.clone();
                want.params.get_mut("title").unwrap().value = ParamValue::Str("hey".to_string());
                want
            }
            .to_dict()
        );
    }
}
