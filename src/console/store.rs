//! The scenes a console shows, edits and saves.
//!
//! A store holds a directory of .scene.json and .py scenes. A save writes into
//! modified/ beside them and never touches the original. A JSON scene saves a
//! modified copy that shadows it. A procedural scene saves a .values.json
//! sidecar of param values and its filter list. A load applies that sidecar over
//! the code's defaults.
//!
//! A host whose scenes live elsewhere implements names, resolve and save of its
//! own, and hands them to its ConsoleHost.

use std::fs;
use std::path::{Component, Path, PathBuf};

use serde_json::{Map, Value};

use crate::console::core::{ConsoleError, ConsoleResult};
use crate::filters::apply_filters;
use crate::procedural::load_scene;
use crate::procedural::process::{self, SceneProcess};
use crate::scene::{py_repr, Param, Scene};

const SCENE_SUFFIX: &str = ".scene.json";

/// Overlay a values sidecar onto the resolved scene. A filter pass takes the
/// overlay too, because its params are tweakable. The code's current shape wins,
/// so this reports an unknown name and skips it.
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
                Ok(param) => existing.insert(n.clone(), param),
                Err(e) => eprintln!("values {name}: bad '{label}.{n}': {e}"),
            }
        }
    }
}

pub struct SceneStore {
    pub dir: String,
    pub read_only: bool,
    /// Scene args for the whole session. Every scene the store loads gets them.
    pub args: Vec<String>,
    /// Each scene name and its path, in the order a scan found them.
    pub scenes: Vec<(String, String)>,
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
        for suffix in [SCENE_SUFFIX, ".py"] {
            if let Some(stem) = base.strip_suffix(suffix) {
                base = stem.to_string();
            }
        }
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

    /// The path a scene loads from. A saved modified copy shadows the original,
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

    /// A scene ready to render and edit: shader files inlined, declared filters
    /// expanded, saved values overlaid.
    ///
    /// Every scene the session loads gets the session's scene args. A scene keeps
    /// what it declares and ignores the rest.
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
        let (mut scene, mut process) = match process::executable(&path) {
            Some(exe) => SceneProcess::start(&exe, &self.args)
                .map(|(scene, fed)| (scene, Some(fed)))
                .map_err(|e| ConsoleError::server_error(format!("{e:#}")))?,
            None => (
                load_scene(&path, &self.args, false)
                    .map_err(|e| ConsoleError::server_error(e.to_string()))?,
                None,
            ),
        };
        let vals = match self.source(name) {
            Some(p) if p.ends_with(SCENE_SUFFIX) => None,
            _ => self.read_values(name),
        };
        if let Some(Value::Array(items)) = vals.as_ref().and_then(|v| v.get("filters")) {
            scene.filters = items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect();
        }
        let filters = scene.filters.clone();
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
            // A procedural scene's shape lives in code. Save the param values
            // and the filter list as a sidecar. The loader applies that over
            // the code's own defaults.
            let mut vals = Map::new();
            vals.insert("filters".to_string(), Value::from(scene.filters.clone()));
            vals.insert("scene".to_string(), values_of(&scene.params));
            let mut passes = Map::new();
            for ps in &scene.passes {
                passes.insert(ps.name.clone(), values_of(&ps.params));
            }
            vals.insert("passes".to_string(), Value::Object(passes));
            return write(&self.values_path(name), &Value::Object(vals));
        }
        // Store the scene unexpanded. A load rebuilds each filter pass from the
        // filters list.
        scene.passes.retain(|p| p.filter.is_empty());
        write(&self.modified_path(name), &scene.to_dict())
    }
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

/// The .scene.json and .py scenes in a directory, JSON first. A .py beside its
/// .scene.json is one scene, and load_scene finds both.
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
    for suffix in [SCENE_SUFFIX, ".py"] {
        for n in &names {
            let Some(stem) = n.strip_suffix(suffix) else {
                continue;
            };
            if out.iter().any(|(name, _)| name == stem) {
                continue;
            }
            let path = format!("{dir}/{n}");
            // ghee runs a procedural scene as a process. A .py without the
            // execute bit is a module the scenes beside it import, not a scene.
            if suffix == ".py" && process::executable(&path).is_none() {
                continue;
            }
            out.push((stem.to_string(), path));
        }
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
    use crate::scene::{ParamMap, Pass};

    const SCENE: &str = r#"{"version": 1,
        "passes": [{"name": "main", "code": "void main(){ gl_FragColor = vec4(1.0); }"}]}"#;

    fn scene_json() -> Value {
        serde_json::from_str(SCENE).unwrap()
    }

    fn tmpdir(tag: &str) -> String {
        let d = std::env::temp_dir().join(format!("ghee-store-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d.to_string_lossy().into_owned()
    }

    fn write_text(path: &str, text: &str) {
        fs::write(path, text).unwrap();
    }

    /// Write a .py the store takes for a scene. A scene process carries the
    /// execute bit.
    fn write_scene(path: &str, text: &str) {
        use std::os::unix::fs::PermissionsExt;
        fs::write(path, text).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    }

    fn read_json(path: &str) -> Value {
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
    }

    /// A directory of two JSON scenes.
    fn scenes(tag: &str) -> (String, SceneStore) {
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
        edited["passes"][0]["code"] = Value::from("void main(){ discard; }");
        store
            .save("one", Scene::from_dict(&edited, None, None).unwrap())
            .unwrap();
        assert_eq!(
            fs::read_to_string(format!("{d}/one{SCENE_SUFFIX}")).unwrap(),
            before
        );
        assert!(Path::new(&format!("{d}/modified/one{SCENE_SUFFIX}")).is_file());
        assert!(store.resolve("one").unwrap().passes[0]
            .code
            .contains("discard"));
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
    fn a_save_writes_the_common_into_the_modified_copy() {
        let (d, store) = scenes("common");
        let mut scene = store.resolve("one").unwrap();
        scene.common = Some("float half(float x){ return x * 0.5; }".to_string());
        let want = scene.common.clone();
        store.save("one", scene).unwrap();
        let written = read_json(&format!("{d}/modified/one{SCENE_SUFFIX}"));
        assert_eq!(written["common"], serde_json::json!({ "code": want }));
        assert_eq!(store.resolve("one").unwrap().common, want);
    }

    #[test]
    fn a_common_file_reference_saves_as_inline_code() {
        let d = tmpdir("commonfile");
        write_text(&format!("{d}/shared.glsl"), "float k(){ return 1.0; }");
        let mut doc = scene_json();
        doc["common"] = serde_json::json!({"shader": "shared.glsl"});
        write_text(&format!("{d}/c{SCENE_SUFFIX}"), &doc.to_string());
        let store = SceneStore::new(&d, false, Vec::new());
        let scene = store.resolve("c").unwrap();
        store.save("c", scene).unwrap();
        let written = read_json(&format!("{d}/modified/c{SCENE_SUFFIX}"));
        assert_eq!(
            written["common"],
            serde_json::json!({"code": "float k(){ return 1.0; }"})
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

    fn procedural_store(tag: &str) -> (String, SceneStore) {
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

    /// The shim ships beside the procedural scenes that import it. A module is
    /// not a scene, so the console does not list it.
    #[test]
    fn a_module_beside_the_scenes_is_not_a_scene() {
        let d = tmpdir("shim");
        write_scene(&format!("{d}/feed.py"), "#!/bin/sh\nexit 0\n");
        write_text(&format!("{d}/ghee.py"), "\"\"\"the shim\"\"\"\n");
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
}
