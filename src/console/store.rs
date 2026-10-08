//! The scenes that a console shows, edits and saves.
//!
//! A store holds a directory of scenes, each a .scene.json file or an
//! executable, and the state directory that keeps their edits. A load builds
//! a scene's working copy from its defaults, the scene that its file or its
//! program writes, and applies the edits of its state file
//! (crate::console::state). A save writes the working copy's edits to the
//! state file and leaves the scene's own files alone.
//!
//! A host whose scenes live outside one directory fills the store's `scenes`
//! with its own, by name and path, and hands the store to its ConsoleHost.

use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::console::core::{ConsoleError, ConsoleResult};
use crate::console::state::{self, Edits, Item};
use crate::filters::{apply_filters, rebuild_filters};
use crate::procedural::process::{self, SceneProcess};
use crate::scene::{Reach, Scene};

const SCENE_SUFFIX: &str = ".scene.json";

pub struct SceneStore {
    pub dir: String,
    /// The store holds the bundled demos, whose state files go by the
    /// scene's name.
    pub demo: bool,
    /// Where the state files live. None reads and writes none.
    pub state_dir: Option<PathBuf>,
    /// Scene args for the whole session. The store hands them to every scene it
    /// loads.
    pub args: Vec<String>,
    /// Each scene name and its path, in the order a scan found them. A path
    /// names a scene file, or a package that crate::procedural resolves.
    pub scenes: Vec<(String, String)>,
    /// Every scene in this store ends with these filters, after its own. A host
    /// whose panel needs one (a 1-bit panel needs the dither) sets them here
    /// instead of writing them into every scene.
    pub filters: Vec<String>,
}

/// What a load carries from the working copy that a scene had before: its
/// edits, the items set aside among them, and what its state file holds.
pub struct Carried {
    pub edits: Edits,
    pub saved: Option<Edits>,
}

/// A scene as a load builds it.
pub struct Opened {
    pub defaults: Scene,
    pub working: Scene,
    /// The items whose default changed since their edit.
    pub aside: Vec<Item>,
    /// What the scene's state file holds. None when it has no file.
    pub saved: Option<Edits>,
    pub procedural: bool,
}

impl SceneStore {
    pub fn new(scenes_dir: &str, args: Vec<String>) -> SceneStore {
        let dir = abspath(scenes_dir);
        let scenes = scan(&dir);
        SceneStore {
            dir,
            demo: false,
            state_dir: None,
            args,
            scenes,
            filters: Vec::new(),
        }
    }

    /// A store over one scene: its own directory, narrowed to that scene.
    /// `render <scene> --serve` shows this. `serve <dir>` shows the whole list.
    pub fn one(path: &str, args: Vec<String>) -> ConsoleResult<SceneStore> {
        let full = abspath(path);
        let (d, mut base) = match full.rsplit_once('/') {
            Some((d, b)) => (d.to_string(), b.to_string()),
            None => (String::new(), full.clone()),
        };
        base = crate::procedural::scene_name(&base);
        let mut store = SceneStore::new(&d, args);
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
                self.file(n)
                    .is_ok_and(|p| process::executable(&p).is_some())
            })
            .map(|(n, _)| n.clone())
            .collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.scenes.iter().any(|(n, _)| n == name)
    }

    /// The scene's file: its .scene.json, or the program that writes it.
    pub fn file(&self, name: &str) -> ConsoleResult<String> {
        let p = self
            .scenes
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p.as_str())
            .ok_or_else(|| ConsoleError::not_found("no scene"))?;
        if let Some(exe) = process::executable(p) {
            return Ok(exe);
        }
        if Path::new(p).is_file() {
            return Ok(p.to_string());
        }
        match crate::procedural::resolve(p) {
            Ok((Some(json), _)) => Ok(json),
            Ok(_) => Ok(p.to_string()),
            Err(e) => Err(ConsoleError::server_error(e.to_string())),
        }
    }

    /// The scene's state file, or None when the store keeps none.
    pub fn state_file(&self, name: &str) -> ConsoleResult<Option<PathBuf>> {
        let Some(dir) = &self.state_dir else {
            return Ok(None);
        };
        let file = self.file(name)?;
        Ok(Some(state::file_of(dir, &file, self.demo.then_some(name))))
    }

    /// The edits that the scene's state file holds. An unreadable file logs
    /// one line and holds none.
    fn read_state(&self, name: &str) -> Option<Edits> {
        let path = self.state_file(name).ok().flatten()?;
        match state::read(&path) {
            Ok(edits) => edits,
            Err(e) => {
                eprintln!("{name}: its state file is unreadable, so none of it applies: {e}");
                None
            }
        }
    }

    /// Write `edits` to the scene's state file, or remove the file when there
    /// are none.
    pub fn save(&self, name: &str, edits: &Edits) -> ConsoleResult<()> {
        let file = self.file(name)?;
        let path = self.state_file(name)?.ok_or_else(|| {
            ConsoleError::server_error(
                "no state directory: set XDG_STATE_HOME or HOME, or pass --state-dir",
            )
        })?;
        state::write(&path, &file, edits).map_err(ConsoleError::server_error)
    }

    /// The defaults of a scene: its files as they are, or the scene that its
    /// program writes at start, with the store's filters after its own. `size`
    /// is the size that the scene renders at, which a program reads before it
    /// writes its scene.
    ///
    /// The store hands the session's scene args to every scene it loads. A scene
    /// keeps what it declares and ignores the rest.
    ///
    /// Every read goes through sources::record, so a caller that wraps this in
    /// collect() gets the files behind the result.
    pub fn resolve(&self, name: &str, size: (i32, i32)) -> ConsoleResult<Scene> {
        self.start(name, size).map(|(scene, _)| scene)
    }

    /// resolve(), plus the process when the scene is procedural. The process runs
    /// from here on. A caller that wants the document only drops it, which stops
    /// it.
    pub fn start(
        &self,
        name: &str,
        size: (i32, i32),
    ) -> ConsoleResult<(Scene, Option<SceneProcess>)> {
        let path = self.file(name)?;
        // Anyone who reaches the console posts the scenes that it loads, so
        // their files stay inside the scene's directory.
        let (mut scene, mut process) = match process::executable(&path) {
            // the console error state reaches a browser, so it carries the
            // stderr of a dead process
            Some(exe) => SceneProcess::start(&exe, &self.args, Reach::Inside, size, |_| {})
                .map(|(scene, fed)| (scene, Some(fed)))
                .map_err(|e| ConsoleError::server_error(process::with_stderr(&e)))?,
            None => (
                Scene::load_in(&path, None, Reach::Inside)
                    .map_err(|e| ConsoleError::server_error(e.to_string()))?,
                None,
            ),
        };
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
        if let Some(fed) = process.as_mut() {
            fed.bind(&scene);
        }
        Ok((scene, process))
    }

    /// start() and the working copy: the edits that `carried` holds, or else
    /// those of the scene's state file, over the defaults. The process, when
    /// there is one, binds the working copy.
    pub fn open(
        &self,
        name: &str,
        carried: Option<Carried>,
        size: (i32, i32),
    ) -> ConsoleResult<(Opened, Option<SceneProcess>)> {
        let (defaults, mut process) = self.start(name, size)?;
        let (edits, saved) = match carried {
            Some(c) => (c.edits, c.saved),
            None => {
                let saved = self.read_state(name);
                (saved.clone().unwrap_or_default(), saved)
            }
        };
        let (working, aside) = working(name, &defaults, &edits);
        if let Some(fed) = process.as_mut() {
            fed.bind(&working);
        }
        let opened = Opened {
            procedural: process.is_some(),
            defaults,
            working,
            aside,
            saved,
        };
        Ok((opened, process))
    }
}

/// The working copy of scene `name`: `edits` over `defaults`, and the items
/// set aside. Each item set aside and each edit skipped logs one line.
pub fn working(name: &str, defaults: &Scene, edits: &Edits) -> (Scene, Vec<Item>) {
    if edits.is_empty() {
        return (defaults.clone(), Vec::new());
    }
    let applied = state::apply(&defaults.to_dict(), edits, name);
    for line in &applied.notes {
        eprintln!("{line}");
    }
    match scene_of(defaults, &applied.doc, name) {
        Ok(scene) => (scene, applied.aside),
        Err(e) => {
            eprintln!("{name}: its saved edits make no scene, so its defaults run: {e}");
            (defaults.clone(), applied.aside)
        }
    }
}

/// The scene of `doc`, a document of scene `name`, with its files where the
/// files of `defaults` resolve.
pub fn scene_of(defaults: &Scene, doc: &serde_json::Value, name: &str) -> anyhow::Result<Scene> {
    let (dir, reach) = match &defaults.root {
        Some(root) => (root.dir.clone(), root.reach),
        None => (String::new(), Reach::Inside),
    };
    let mut scene = match dir.is_empty() {
        true => Scene::from_dict(doc, None, Some(name))?,
        false => Scene::from_dict_in(doc, &dir, reach, Some(name))?,
    };
    rebuild_filters(&mut scene, None)?;
    if !dir.is_empty() {
        scene.load_assets_in(&dir, reach)?;
    }
    Ok(scene)
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
    use serde_json::{json, Value};

    const SCENE: &str = r#"{"version": 1,
        "passes": [{"name": "main", "fragment": "void main(){ gl_FragColor = vec4(1.0); }"}]}"#;

    /// The size that a test's scenes render at.
    const SIZE: (i32, i32) = (32, 24);

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

    /// A host filter reaches a scene that declares none, and the scene's own
    /// filters keep their place in front of it.
    #[test]
    fn the_store_filters_every_scene_it_loads() {
        let d = tmpdir("host-filters");
        write_text(&format!("{d}/plain.scene.json"), SCENE);
        let mut store = SceneStore::new(&d, Vec::new());
        let (plain, _) = store.start("plain", SIZE).unwrap();
        assert_eq!(plain.passes.len(), 1, "no filter yet");

        store.filters = vec!["dither".to_string()];
        let (dithered, _) = store.start("plain", SIZE).unwrap();
        assert_eq!(dithered.passes.len(), 2);
        assert_eq!(dithered.passes[1].name, "dither");

        // a scene that already ends with the filter takes it once
        let own = r#"{"version": 1, "filters": ["dither"],
             "passes": [{"name": "main", "fragment": "void main(){ gl_FragColor = vec4(1.0); }"}]}"#;
        write_text(&format!("{d}/own.scene.json"), own);
        store.scenes = scan(&store.dir);
        let (once, _) = store.start("own", SIZE).unwrap();
        assert_eq!(once.passes.len(), 2, "the dither is not applied twice");
    }

    /// A directory of two JSON scenes.
    fn scenes(tag: &str) -> (crate::testing::TempDir, SceneStore) {
        let d = tmpdir(tag);
        write_text(&format!("{d}/one{SCENE_SUFFIX}"), SCENE);
        write_text(&format!("{d}/two{SCENE_SUFFIX}"), SCENE);
        let store = SceneStore::new(&d, Vec::new());
        (d, store)
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
        let store = SceneStore::new(&d, Vec::new());
        assert_eq!(store.names(), ["one", "two", "proc"]);
        assert_eq!(store.procedural_names(), ["proc"]);
    }

    #[test]
    fn the_store_resolves_a_scene() {
        let (_d, store) = scenes("resolve");
        assert_eq!(store.resolve("one", SIZE).unwrap().passes[0].name, "main");
    }

    #[test]
    fn the_store_refuses_an_unknown_name() {
        let (_d, store) = scenes("unknown");
        assert_eq!(store.file("three").unwrap_err().status, 404);
        assert_eq!(
            store.save("three", &Edits::default()).unwrap_err().status,
            404
        );
    }

    /// A host that lists a package by its directory loads the scene file
    /// inside it, and the scene's files resolve beside that file.
    #[test]
    fn a_package_directory_loads_the_scene_file_inside_it() {
        let d = tmpdir("package");
        fs::create_dir_all(format!("{d}/clocks")).unwrap();
        write_text(
            &format!("{d}/clocks/clocks.glsl"),
            "void main(){ gl_FragColor = vec4(0.5); }",
        );
        write_text(
            &format!("{d}/clocks/clocks.scene.json"),
            r#"{"version": 1, "passes": [{"name": "main", "fragment": {"file": "clocks.glsl"}}]}"#,
        );
        let mut store = SceneStore::new(&d, Vec::new());
        store.scenes = vec![("clocks".to_string(), format!("{d}/clocks"))];
        assert_eq!(
            store.file("clocks").unwrap(),
            format!("{d}/clocks/clocks.scene.json")
        );
        let scene = store.resolve("clocks", SIZE).unwrap();
        assert!(scene.passes[0].fragment.contains("0.5"));
    }

    #[test]
    fn a_scene_reads_no_file_outside_its_directory() {
        let root = tmpdir("files");
        let d = format!("{root}/scenes");
        fs::create_dir_all(&d).unwrap();
        fs::write(format!("{root}/out.raw"), [9u8; 4]).unwrap();
        let mut doc: Value = serde_json::from_str(SCENE).unwrap();
        doc["assets"] =
            json!({"d": {"file": "../out.raw", "width": 2, "height": 2, "format": "u8"}});
        write_text(&format!("{d}/two{SCENE_SUFFIX}"), &doc.to_string());
        let store = SceneStore::new(&d, Vec::new());
        let e = store.resolve("two", SIZE).unwrap_err();
        assert!(
            e.message.contains("outside the scene directory"),
            "{}",
            e.message
        );
    }

    #[test]
    fn a_declared_filter_comes_back_as_a_pass() {
        let d = tmpdir("filter");
        let mut doc: Value = serde_json::from_str(SCENE).unwrap();
        doc["filters"] = json!(["dither"]);
        write_text(&format!("{d}/f{SCENE_SUFFIX}"), &doc.to_string());
        let store = SceneStore::new(&d, Vec::new());
        let scene = store.resolve("f", SIZE).unwrap();
        assert_eq!(
            scene
                .passes
                .iter()
                .map(|p| p.filter.clone())
                .collect::<Vec<_>>(),
            vec!["".to_string(), "dither".to_string()]
        );
    }

    #[test]
    fn a_scan_takes_the_json_scene_over_the_py_beside_it() {
        let d = tmpdir("pair");
        write_text(&format!("{d}/wob{SCENE_SUFFIX}"), SCENE);
        write_text(&format!("{d}/wob.py"), "# a scene process\n");
        let store = SceneStore::new(&d, Vec::new());
        assert_eq!(store.names(), vec!["wob".to_string()]);
        assert_eq!(store.file("wob").unwrap(), format!("{d}/wob{SCENE_SUFFIX}"));
    }

    /// A procedural scene imports its own modules from beside it. A module is
    /// not a scene, so the console does not list it.
    #[test]
    fn a_module_beside_the_scenes_is_not_a_scene() {
        let d = tmpdir("module");
        write_scene(&format!("{d}/feed.py"), "#!/bin/sh\nexit 0\n");
        write_text(&format!("{d}/look.py"), "\"\"\"a module\"\"\"\n");
        let store = SceneStore::new(&d, Vec::new());
        assert_eq!(store.names(), vec!["feed".to_string()]);
    }

    #[test]
    fn one_narrows_a_store_to_the_scene_it_names() {
        let (d, _) = scenes("one");
        let store = SceneStore::one(&format!("{d}/two{SCENE_SUFFIX}"), Vec::new()).unwrap();
        assert_eq!(store.names(), vec!["two".to_string()]);
        let missing = SceneStore::one(&format!("{d}/nope{SCENE_SUFFIX}"), Vec::new());
        assert_eq!(missing.err().map(|e| e.status), Some(404));
    }

    // the state file

    /// The store of `scenes(tag)` with its state files in a directory of
    /// their own.
    fn kept(tag: &str) -> (crate::testing::TempDir, crate::testing::TempDir, SceneStore) {
        let (d, mut store) = scenes(tag);
        let states = tmpdir(&format!("{tag}-state"));
        store.state_dir = Some(PathBuf::from(&*states));
        (d, states, store)
    }

    #[test]
    fn a_save_writes_the_state_file_and_leaves_the_scene_alone() {
        let (d, states, store) = kept("save");
        let before = fs::read_to_string(format!("{d}/one{SCENE_SUFFIX}")).unwrap();
        let mut edited = store.resolve("one", SIZE).unwrap().to_dict();
        edited["passes"][0]["fragment"] = json!("void main(){ discard; }");
        let edits = state::diff(
            &store.resolve("one", SIZE).unwrap().to_dict(),
            &edited,
            false,
        );
        store.save("one", &edits).unwrap();
        assert_eq!(
            fs::read_to_string(format!("{d}/one{SCENE_SUFFIX}")).unwrap(),
            before
        );
        let file = format!("{states}/scenes{d}/one.json");
        let written: Value = serde_json::from_str(&fs::read_to_string(&file).unwrap()).unwrap();
        assert_eq!(written["scene"], format!("{d}/one{SCENE_SUFFIX}"));
        let (opened, _) = store.open("one", None, SIZE).unwrap();
        assert!(opened.working.passes[0].fragment.contains("discard"));
        assert!(opened.defaults.passes[0].fragment.contains("vec4(1.0)"));
        // a save with no edit left removes the file
        store.save("one", &Edits::default()).unwrap();
        assert!(!Path::new(&file).exists());
    }

    #[test]
    fn a_demo_keeps_its_state_file_by_its_name() {
        let (_d, states, mut store) = kept("demo");
        store.demo = true;
        assert_eq!(
            store.state_file("two").unwrap(),
            Some(PathBuf::from(format!("{states}/demo/two.json")))
        );
    }
}
