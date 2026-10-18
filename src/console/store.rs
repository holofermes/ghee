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
//!
//! A store may also hold the scenes that its console makes, in a directory of
//! their own, one directory per scene: `name/name.scene.json`, the GLSL of
//! each pass in a file of its own, and the scene's files. A save writes the
//! edits of such a scene into its files, and no state file lays edits over
//! it. That directory lists scene files alone, so no program there ever runs.

use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use serde_json::{Map, Value};

use crate::console::core::{ConsoleError, ConsoleResult};
use crate::console::layout;
use crate::console::state::{self, Edits, Item};
use crate::filters::{apply_filters, rebuild_filters};
use crate::procedural::process::{self, Program, SceneProcess};
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
    /// instead of writing them into every scene. A filter that the scene lists
    /// too keeps the scene's place and takes the values set here.
    pub filters: Vec<crate::filters::Filter>,
    /// The directory of the scenes that this console makes. None when it
    /// makes none.
    pub mine: Option<String>,
    /// rescan() lists `dir` and `mine` again. A host that fills `scenes`
    /// itself leaves it false.
    pub scans: bool,
    /// What the last rescan() found, which then stands for `scenes`.
    found: Mutex<Option<Vec<(String, String)>>>,
    /// The program in this process that writes the store's one scene.
    here: Option<Arc<dyn Program>>,
}

/// What a load carries from the working copy that a scene had before: its
/// edits, the items set aside among them, and what its state file holds.
pub struct Carried {
    pub edits: Edits,
    pub saved: Option<Edits>,
    pub draft: Option<Edits>,
}

/// A saved edit that makes no scene, and why. It waits in the state file
/// until the console drops it, and each load tries it again.
#[derive(Clone, Debug, PartialEq)]
pub struct Broken {
    /// One edit, as Edits::units() splits them.
    pub edit: Edits,
    pub why: String,
}

/// A scene as a load builds it.
pub struct Opened {
    pub defaults: Scene,
    pub working: Scene,
    /// The items whose default changed since their edit.
    pub aside: Vec<Item>,
    /// The edits that make no scene, which the working copy leaves out.
    pub broken: Vec<Broken>,
    /// What the scene's state file holds. None when it has no file.
    pub saved: Option<Edits>,
    /// The edits that the draft beside the state file holds, which wait for
    /// the console to restore or drop them. None when no draft waits.
    pub draft: Option<Edits>,
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
            mine: None,
            scans: false,
            found: Mutex::new(None),
            here: None,
        }
    }

    /// A store over the scenes of `dir` and the scenes that the console makes
    /// in `mine`, which rescan() lists again. With `dir` the same directory as
    /// `mine`, the store holds the console's own scenes alone.
    pub fn with_mine(dir: &str, mine: Option<&str>, args: Vec<String>) -> SceneStore {
        let mut store = SceneStore::new(dir, args);
        store.mine = mine.map(abspath);
        store.scans = true;
        store.scenes = store.scan_all();
        store
    }

    /// The scenes of `dir`, then those of `mine` that no scene of `dir`
    /// takes the name of.
    fn scan_all(&self) -> Vec<(String, String)> {
        let mine = self.mine.as_deref();
        let mut out = match mine == Some(self.dir.as_str()) {
            true => Vec::new(),
            false => scan(&self.dir),
        };
        for (n, p) in mine.map(scan_mine).unwrap_or_default() {
            if !out.iter().any(|(o, _)| *o == n) {
                out.push((n, p));
            }
        }
        out
    }

    /// List the scenes again, when the store scans.
    pub fn rescan(&self) {
        if self.scans {
            *self.found() = Some(self.scan_all());
        }
    }

    fn found(&self) -> MutexGuard<'_, Option<Vec<(String, String)>>> {
        self.found.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Each scene name and its path, as the last scan found them.
    pub fn list(&self) -> Vec<(String, String)> {
        self.found().clone().unwrap_or_else(|| self.scenes.clone())
    }

    fn path_of(&self, name: &str) -> Option<String> {
        self.list()
            .into_iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p)
    }

    /// Whether `path` is the scene file of a scene in `mine`,
    /// `mine/name/name.scene.json`.
    fn in_mine(&self, path: &str) -> bool {
        let Some(rest) = self.mine.as_deref().and_then(|m| path.strip_prefix(m)) else {
            return false;
        };
        let parts: Vec<&str> = rest.split('/').collect();
        matches!(parts.as_slice(), ["", dir, file] if *file == format!("{dir}{SCENE_SUFFIX}"))
    }

    /// Whether scene `name` is one that this console made.
    pub fn is_mine(&self, name: &str) -> bool {
        self.path_of(name).is_some_and(|p| self.in_mine(&p))
    }

    /// The names of the scenes that this console made.
    pub fn mine_names(&self) -> Vec<String> {
        self.list()
            .into_iter()
            .filter(|(_, p)| self.in_mine(p))
            .map(|(n, _)| n)
            .collect()
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

    /// A store over one scene, the one that `program` writes in this
    /// process, named by `path`, the file of its program.
    pub fn here(path: &str, program: Arc<dyn Program>, args: Vec<String>) -> SceneStore {
        let full = abspath(path);
        let dir = Path::new(&full)
            .parent()
            .map_or_else(|| "/".to_string(), |d| d.to_string_lossy().into_owned());
        let mut store = SceneStore::new(&dir, args);
        store.scenes = vec![(crate::procedural::scene_name(&full), full)];
        store.here = Some(program);
        store
    }

    pub fn names(&self) -> Vec<String> {
        self.list().into_iter().map(|(n, _)| n).collect()
    }

    /// The names of the scenes that a program writes, as start() runs them.
    pub fn procedural_names(&self) -> Vec<String> {
        self.names()
            .into_iter()
            .filter(|n| self.is_procedural(n))
            .collect()
    }

    /// Whether a program writes scene `name`.
    pub fn is_procedural(&self, name: &str) -> bool {
        self.here.is_some() || self.file(name).is_ok_and(|p| self.program(&p).is_some())
    }

    /// The program that writes the scene of `path`, its file. A scene in
    /// `mine` has none.
    fn program(&self, path: &str) -> Option<String> {
        match self.in_mine(path) {
            true => None,
            false => process::executable(path),
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.path_of(name).is_some()
    }

    /// The scene's file: its .scene.json, or the program that writes it.
    pub fn file(&self, name: &str) -> ConsoleResult<String> {
        let p = self
            .path_of(name)
            .ok_or_else(|| ConsoleError::not_found("no scene"))?;
        let p = p.as_str();
        if self.in_mine(p) {
            return Ok(p.to_string());
        }
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

    /// The scene's state file, or None when the store keeps none. A scene
    /// that this console made has none.
    pub fn state_file(&self, name: &str) -> ConsoleResult<Option<PathBuf>> {
        let Some(dir) = &self.state_dir else {
            return Ok(None);
        };
        let file = self.file(name)?;
        if self.in_mine(&file) {
            return Ok(None);
        }
        Ok(Some(state::file_of(dir, &file, self.demo.then_some(name))))
    }

    /// The edits that the scene's state file holds. An unreadable file holds
    /// none: it moves aside before a save can replace it, and logs one line.
    fn read_state(&self, name: &str) -> Option<Edits> {
        let path = self.state_file(name).ok().flatten()?;
        match state::read(&path) {
            Ok(edits) => edits,
            Err(e) => {
                match state::move_unreadable(&path) {
                    Ok(to) => eprintln!(
                        "{name}: its state file is unreadable, so none of it applies: {e}. \
                         It moved to {}",
                        to.display()
                    ),
                    Err(why) => eprintln!(
                        "{name}: its state file is unreadable, so none of it applies: {e}. \
                         It could not move aside, so the next save replaces it: {why}"
                    ),
                }
                None
            }
        }
    }

    /// The draft beside the scene's state file, or None when the store keeps
    /// no state file for the scene.
    pub fn draft_file(&self, name: &str) -> Option<PathBuf> {
        let path = self.state_file(name).ok().flatten()?;
        Some(state::draft_of(&path))
    }

    /// The edits of the scene's draft, when they differ from `saved`, the
    /// edits of its state file. A draft that matches them is removed, and an
    /// unreadable one moves aside.
    fn read_draft(&self, name: &str, saved: Option<&Edits>) -> Option<Edits> {
        let path = self.draft_file(name)?;
        let draft = match state::read(&path) {
            Ok(draft) => draft?,
            Err(e) => {
                match state::move_unreadable(&path) {
                    Ok(to) => eprintln!(
                        "{name}: its unsaved edits are unreadable: {e}. They moved to {}",
                        to.display()
                    ),
                    Err(why) => eprintln!("{name}: its unsaved edits are unreadable: {e}: {why}"),
                }
                return None;
            }
        };
        if draft.same(saved.unwrap_or(&Edits::default())) {
            if let Err(e) = state::remove(&path) {
                eprintln!("{name}: {e}");
            }
            return None;
        }
        eprintln!("{name}: unsaved edits from before ghee stopped wait for the console");
        Some(draft)
    }

    /// Write `edits`, the working copy's edits, to the scene's draft.
    pub fn write_draft(&self, name: &str, edits: &Edits) -> Result<(), String> {
        let file = self.file(name).map_err(|e| e.to_string())?;
        match self.draft_file(name) {
            Some(path) => state::write_draft(&path, &file, edits),
            None => Ok(()),
        }
    }

    /// Remove the scene's draft.
    pub fn remove_draft(&self, name: &str) -> Result<(), String> {
        match self.draft_file(name) {
            Some(path) => state::remove(&path),
            None => Ok(()),
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

    /// The directory of scene `name`, one of the console's own, or why it has
    /// none.
    pub fn own_dir(&self, name: &str) -> ConsoleResult<String> {
        let file = self.file(name)?;
        match (self.in_mine(&file), file.rsplit_once('/')) {
            (true, Some((dir, _))) => Ok(dir.to_string()),
            _ => Err(ConsoleError::read_only(format!(
                "scene '{name}' is not in {}, the console's own scenes",
                self.mine
                    .as_deref()
                    .unwrap_or("the console's own directory")
            ))),
        }
    }

    /// Write `edits`, made over `defaults`, into the files of scene `name`,
    /// one of the console's own. The scene file keeps its layout and changes
    /// only where an edit lands, and each shader goes into the file that the
    /// scene names for it (into_files()).
    pub fn write(&self, name: &str, defaults: &Value, edits: &Edits) -> ConsoleResult<()> {
        let dir = self.own_dir(name)?;
        let file = self.file(name)?;
        let fail = |e: String| ConsoleError::server_error(format!("{name}: {e}"));
        let text = fs::read_to_string(&file).map_err(|e| fail(format!("{file}: {e}")))?;
        let raw: Value = serde_json::from_str(&text).map_err(|e| fail(format!("{file}: {e}")))?;
        let (doc, glsl) = into_files(&raw, defaults, edits, &dir).map_err(fail)?;
        let mut check = doc.clone();
        for g in &glsl {
            let at = match &g.pass {
                Some(p) => raw_pass_mut(&mut check, p).map(|p| &mut p["fragment"]),
                None => Some(&mut check["common"]),
            };
            if let Some(at) = at {
                *at = Value::from(g.text.as_str());
            }
        }
        Scene::from_dict_in(&check, &dir, Reach::Inside, Some(&file))
            .map_err(|e| fail(e.to_string()))?;
        for g in &glsl {
            let p = format!("{dir}/{}", g.file);
            if fs::read_to_string(&p).ok().as_deref() != Some(g.text.as_str()) {
                write_kept(Path::new(&p), g.text.as_bytes())
                    .map_err(|e| fail(format!("{p}: {e}")))?;
            }
        }
        if doc != raw {
            let out = layout::rewrite(&text, &doc).map_err(|e| fail(format!("{file}: {e}")))?;
            write_kept(Path::new(&file), out.as_bytes())
                .map_err(|e| fail(format!("{file}: {e}")))?;
        }
        Ok(())
    }

    /// Make scene `name` of the console's own from `doc`, with each of
    /// `files`, a path inside the directory `from`, copied beside it. The
    /// scene's directory appears whole, once the scene in it builds.
    pub fn create(
        &self,
        name: &str,
        doc: &Value,
        from: Option<&str>,
        files: &[String],
    ) -> ConsoleResult<()> {
        let mine = self.mine.as_deref().ok_or_else(keeps_none)?;
        if let Some(why) = name_fault(name) {
            return Err(ConsoleError::bad_request(why));
        }
        let dest = self.free(name)?;
        let tmp = format!("{mine}/.{name}.{}.new", std::process::id());
        let made = (|| -> Result<(), String> {
            fs::create_dir_all(mine).map_err(|e| format!("{mine}: {e}"))?;
            let _ = fs::remove_dir_all(&tmp);
            fs::create_dir(&tmp).map_err(|e| format!("{tmp}: {e}"))?;
            for f in files.iter().filter(|_| from.is_some()) {
                copy_into(from.unwrap_or_default(), f, &tmp)?;
            }
            write_scene(&tmp, name, doc)?;
            Scene::load_in(&format!("{tmp}/{name}{SCENE_SUFFIX}"), None, Reach::Inside)
                .map_err(|e| e.to_string())?;
            fs::rename(&tmp, &dest).map_err(|e| format!("{dest}: {e}"))
        })();
        if let Err(e) = made {
            let _ = fs::remove_dir_all(&tmp);
            return Err(ConsoleError::server_error(e));
        }
        self.rescan();
        Ok(())
    }

    /// The directory that scene `name` of the console's own would take, or
    /// the 409 when a scene or a file takes that name.
    fn free(&self, name: &str) -> ConsoleResult<String> {
        let mine = self.mine.as_deref().ok_or_else(keeps_none)?;
        self.rescan();
        if self.contains(name) {
            return Err(ConsoleError::conflict(format!(
                "a scene named '{name}' exists"
            )));
        }
        let dest = format!("{mine}/{name}");
        if fs::symlink_metadata(&dest).is_ok() {
            return Err(ConsoleError::conflict(format!("{dest} exists")));
        }
        Ok(dest)
    }

    /// Rename scene `name`, one of the console's own, to `to`: its directory
    /// and its scene file.
    pub fn rename(&self, name: &str, to: &str) -> ConsoleResult<()> {
        let dir = self.own_dir(name)?;
        if let Some(why) = name_fault(to) {
            return Err(ConsoleError::bad_request(why));
        }
        let dest = self.free(to)?;
        let (was, now) = (
            format!("{dir}/{name}{SCENE_SUFFIX}"),
            format!("{dir}/{to}{SCENE_SUFFIX}"),
        );
        fs::rename(&was, &now).map_err(|e| ConsoleError::server_error(format!("{now}: {e}")))?;
        if let Err(e) = fs::rename(&dir, &dest) {
            let _ = fs::rename(&now, &was);
            return Err(ConsoleError::server_error(format!("{dest}: {e}")));
        }
        self.rescan();
        Ok(())
    }

    /// Remove scene `name`, one of the console's own, with its directory.
    pub fn delete(&self, name: &str) -> ConsoleResult<()> {
        let dir = self.own_dir(name)?;
        fs::remove_dir_all(&dir).map_err(|e| ConsoleError::server_error(format!("{dir}: {e}")))?;
        self.rescan();
        Ok(())
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
        let started = match (&self.here, self.program(&path)) {
            (Some(program), _) => Some(SceneProcess::here(
                &path,
                Arc::clone(program),
                Reach::Inside,
                size,
                |_| {},
            )),
            (None, Some(exe)) => Some(SceneProcess::start(
                &exe,
                &self.args,
                Reach::Inside,
                size,
                |_| {},
            )),
            (None, None) => None,
        };
        let (mut scene, mut process) = match started {
            // the console error state reaches a browser, so it carries the
            // stderr of a dead process
            Some(started) => started
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
        let filters = crate::filters::chain(&scene.filters, &self.filters);
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
        let (edits, saved, draft) = match carried {
            Some(c) => (c.edits, c.saved, c.draft),
            None => {
                let saved = self.read_state(name);
                let draft = self.read_draft(name, saved.as_ref());
                (saved.clone().unwrap_or_default(), saved, draft)
            }
        };
        let (working, aside, broken) = working(name, &defaults, &edits);
        if let Some(fed) = process.as_mut() {
            fed.bind(&working);
        }
        let opened = Opened {
            procedural: process.is_some(),
            defaults,
            working,
            aside,
            broken,
            saved,
            draft,
        };
        Ok((opened, process))
    }
}

/// The working copy of scene `name`: `edits` over `defaults`, the items set
/// aside, and the edits that make no scene. When the edits together make no
/// scene, each edit is laid alone over those before it that make one. Each
/// item set aside, each edit skipped and each edit that makes no scene logs
/// one line.
pub fn working(name: &str, defaults: &Scene, edits: &Edits) -> (Scene, Vec<Item>, Vec<Broken>) {
    if edits.is_empty() {
        return (defaults.clone(), Vec::new(), Vec::new());
    }
    let base = defaults.to_dict();
    let applied = state::apply(&base, edits, name);
    for line in &applied.notes {
        eprintln!("{line}");
    }
    if let Ok(scene) = scene_of(defaults, &applied.doc, name) {
        return (scene, applied.aside, Vec::new());
    }
    let mut waiting = edits.clone();
    waiting.items.retain(|i| !applied.aside.contains(i));
    let (mut doc, mut broken) = (base.clone(), Vec::new());
    for edit in waiting.units() {
        let next = state::fold(&base, &doc, &edit);
        match scene_of(defaults, &next, name) {
            Ok(_) => doc = next,
            Err(e) => {
                let e = e.to_string();
                let why = e
                    .strip_prefix(&format!("{name}: "))
                    .unwrap_or(&e)
                    .to_string();
                eprintln!(
                    "{name}: its saved edit of {} makes no scene, so it waits in the state \
                     file: {why}",
                    edit.what()
                );
                broken.push(Broken { edit, why });
            }
        }
    }
    let scene = scene_of(defaults, &doc, name).unwrap_or_else(|_| defaults.clone());
    (scene, applied.aside, broken)
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

fn keeps_none() -> ConsoleError {
    ConsoleError::read_only("this console keeps no scenes of its own")
}

/// The scene that a new scene starts as: one pass that draws its uv.
pub fn blank() -> Value {
    serde_json::json!({"version": 1, "passes": [{"name": "main",
        "fragment": "void main(){\n  gl_FragColor = vec4(uv, 0.5, 1.0);\n}\n"}]})
}

/// Why `name` names no scene of the console's own, which is a directory and
/// a file name: it takes ASCII letters, digits, `-` and `_`, a letter or a
/// digit first, and 64 characters at most.
pub fn name_fault(name: &str) -> Option<String> {
    let ok = name.len() <= 64
        && name
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_alphanumeric())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    (!ok).then(|| {
        format!(
            "a scene name takes letters, digits, - and _, a letter or a digit first, and 64 \
             of them at most, got {}",
            crate::scene::py_repr(&Value::from(name))
        )
    })
}

/// The files that the scene document `doc` names beside it: each asset's
/// file, an image layer's image file and a text layer's font files.
pub fn files_of(doc: &Value) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut add = |f: Option<&str>| {
        if let Some(f) = f.filter(|f| !out.iter().any(|o| o == f)) {
            out.push(f.to_string());
        }
    };
    for a in doc
        .get("assets")
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(Map::values)
    {
        add(a.get("file").and_then(Value::as_str));
    }
    let layers = doc
        .get("passes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|p| p.get("layers")?.as_array())
        .flatten();
    for l in layers {
        let text = |k: &str| l.get(k).and_then(Value::as_str);
        add(text("image")
            .filter(|f| crate::layer::is_image_file(f) && !crate::layer::is_image_data(f)));
        let fonts = match l.get("font_family") {
            Some(Value::Array(names)) => names.iter().collect(),
            f => Vec::from_iter(f),
        };
        for f in fonts {
            add(f.as_str().filter(|f| crate::text::is_font_file(f)));
        }
    }
    out
}

/// Copy `rel`, a path inside `from`, to the same path inside `to`, readable
/// by all. A path that leaves `from`, or names no file in it, stays behind:
/// the copy then finds it where the original found it, through
/// `$GHEE_ASSETS`.
fn copy_into(from: &str, rel: &str, to: &str) -> Result<(), String> {
    let inside = Path::new(rel)
        .components()
        .all(|c| matches!(c, Component::Normal(_)));
    let src = Path::new(from).join(rel);
    if !inside || !src.is_file() {
        return Ok(());
    }
    let dst = Path::new(to).join(rel);
    let err = |e: std::io::Error| format!("{}: {e}", dst.display());
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).map_err(err)?;
    }
    fs::copy(&src, &dst).map_err(err)?;
    set_mode(&dst, 0o644).map_err(err)
}

pub fn set_mode(path: &Path, mode: u32) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
}

/// Write `doc`, a scene document as Scene::to_dict() writes it, as scene
/// `name` in the new directory `dir`: `name.scene.json`, the GLSL of each
/// pass in `<pass>.glsl` and the common block in `common.glsl`. A filter's
/// pass comes back from the scene's `filters`, so the file leaves it out.
fn write_scene(dir: &str, name: &str, doc: &Value) -> Result<(), String> {
    let file = format!("{dir}/{name}{SCENE_SUFFIX}");
    let (out, glsl) = split_glsl(doc);
    for (f, text) in &glsl {
        let p = format!("{dir}/{f}");
        state::write_file(Path::new(&p), text.as_bytes()).map_err(|e| format!("{p}: {e}"))?;
    }
    let text = serde_json::to_string_pretty(&out).map_err(|e| e.to_string())? + "\n";
    state::write_file(Path::new(&file), text.as_bytes()).map_err(|e| format!("{file}: {e}"))
}

/// `doc` with the GLSL of each pass and of the common block as a file, and
/// each such file's name and text. The scene's filter passes go.
fn split_glsl(doc: &Value) -> (Value, Vec<(String, String)>) {
    let mut out = doc.clone();
    let mut glsl = Vec::new();
    let own: Vec<Value> = doc
        .get("passes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|p| {
            p.get("filter")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
        })
        .cloned()
        .collect();
    let name = |p: &Value| {
        p.get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    let common = match own.iter().any(|p| name(p) == "common") {
        true => "common-block.glsl",
        false => "common.glsl",
    };
    let mut passes = Vec::new();
    for mut p in own {
        if let Some(Value::String(text)) = p.get("fragment").cloned() {
            let f = format!("{}.glsl", name(&p));
            p["fragment"] = serde_json::json!({ "file": f });
            glsl.push((f, text));
        }
        passes.push(p);
    }
    out["passes"] = Value::Array(passes);
    if let Some(Value::String(text)) = doc.get("common") {
        out["common"] = serde_json::json!({ "file": common });
        glsl.push((common.to_string(), text.clone()));
    }
    (out, glsl)
}

/// state::write_file(), with the old file's permissions kept.
fn write_kept(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mode = fs::metadata(path).ok().map(|m| m.permissions());
    state::write_file(path, bytes)?;
    match mode {
        Some(m) => fs::set_permissions(path, m),
        None => Ok(()),
    }
}

/// A shader that a save writes: the pass that it belongs to, None for the
/// common block, the file inside the scene's directory, and its text.
struct Glsl {
    pass: Option<String>,
    file: String,
    text: String,
}

/// `raw`, the scene document as its file holds it, with `edits`, made over
/// `defaults`, laid on, and the shaders to write. An edited shader goes into
/// the file that `raw` names for it, or stays in the scene file when `raw`
/// holds its text. A pass or a common block that names no file yet takes a
/// file of its own name in `dir`. A filter's values stay out, since `raw`
/// names a filter alone. Any other edit with no place in `raw` fails.
fn into_files(
    raw: &Value,
    defaults: &Value,
    edits: &Edits,
    dir: &str,
) -> Result<(Value, Vec<Glsl>), String> {
    let mut doc = raw.clone();
    let mut glsl: Vec<Glsl> = Vec::new();
    let is_filter = |pass: &str| {
        defaults["passes"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|p| {
                raw_name(p) == pass
                    && p.get("filter")
                        .and_then(Value::as_str)
                        .is_some_and(|f| !f.is_empty())
            })
    };
    let no_pass = |p: &str| format!("the scene file has no pass '{p}'");
    for item in &edits.items {
        let pass = item.pass.as_deref().unwrap_or_default();
        match item.item.as_str() {
            "passes" => {
                let mut passes = Vec::new();
                for entry in item.value.as_array().into_iter().flatten() {
                    match entry.as_str() {
                        Some(n) => {
                            passes.push(raw_pass(raw, n).cloned().ok_or_else(|| no_pass(n))?)
                        }
                        None => {
                            let mut p = entry.clone();
                            if let Some(Value::String(text)) = p.get("fragment").cloned() {
                                let file = free_glsl(dir, raw_name(&p), &glsl);
                                p["fragment"] = serde_json::json!({ "file": file });
                                glsl.push(Glsl {
                                    pass: Some(raw_name(&p).to_string()),
                                    file,
                                    text,
                                });
                            }
                            passes.push(p);
                        }
                    }
                }
                doc["passes"] = Value::Array(passes);
            }
            "filters" => match item.value.as_array().is_some_and(|f| f.is_empty()) {
                true => {
                    doc.as_object_mut().map(|d| d.shift_remove("filters"));
                }
                false => doc["filters"] = item.value.clone(),
            },
            "common" => match &item.value {
                Value::Null => {
                    doc.as_object_mut().map(|d| d.shift_remove("common"));
                }
                v => {
                    let text = v.as_str().unwrap_or_default();
                    put_glsl(&mut doc, None, text, dir, &mut glsl);
                }
            },
            "fragment" => {
                raw_pass(&doc, pass).ok_or_else(|| no_pass(pass))?;
                let text = item.value.as_str().unwrap_or_default();
                put_glsl(&mut doc, Some(pass), text, dir, &mut glsl);
            }
            "channels" => {
                let p = raw_pass_mut(&mut doc, pass).ok_or_else(|| no_pass(pass))?;
                set_or_drop(p, "channels", &item.value);
            }
            "layers" => {
                let mut layers = Vec::new();
                for entry in item.value.as_array().into_iter().flatten() {
                    match entry.as_str() {
                        Some(n) => layers.push(
                            raw_layer(raw, n)
                                .cloned()
                                .ok_or_else(|| format!("the scene file has no layer '{n}'"))?,
                        ),
                        None => layers.push(entry.clone()),
                    }
                }
                let p = raw_pass_mut(&mut doc, pass).ok_or_else(|| no_pass(pass))?;
                set_or_drop(p, "layers", &Value::Array(layers));
            }
            _ => {
                let param = item.param.as_deref().unwrap_or_default();
                let owner = match &item.pass {
                    Some(p) => raw_pass_mut(&mut doc, p).ok_or_else(|| no_pass(p))?,
                    None => &mut doc,
                };
                if !owner["params"].is_object() {
                    owner["params"] = Value::Object(Map::new());
                }
                owner["params"][param] = item.value.clone();
            }
        }
    }
    let values = |key: &str| {
        edits
            .values
            .get(key)
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
    };
    for (n, v) in values("params") {
        set_value(&mut doc, n, v).ok_or_else(|| format!("the scene file has no param '{n}'"))?;
    }
    for (pass, v) in values("passes") {
        if is_filter(pass) {
            continue;
        }
        let p = raw_pass_mut(&mut doc, pass).ok_or_else(|| no_pass(pass))?;
        for (n, pv) in v
            .get("params")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
        {
            set_value(p, n, pv)
                .ok_or_else(|| format!("pass '{pass}' in the scene file has no param '{n}'"))?;
        }
        for (k, absent) in [
            ("visible", Value::from(true)),
            ("sampling", Value::from("linear")),
            ("size", Value::Null),
            ("scale", Value::Null),
        ] {
            if let Some(kv) = v.get(k) {
                let kv = if *kv == absent { &Value::Null } else { kv };
                set_or_drop(p, k, kv);
            }
        }
    }
    for (n, patch) in values("layers") {
        let l = raw_layer_mut(&mut doc, n)
            .ok_or_else(|| format!("the scene file has no layer '{n}'"))?;
        state::merge(l, patch);
    }
    for (n, v) in values("assets") {
        if !doc["assets"].is_object() {
            doc["assets"] = Value::Object(Map::new());
        }
        match v.is_null() {
            true => {
                doc["assets"].as_object_mut().map(|a| a.shift_remove(n));
            }
            false => doc["assets"][n] = v.clone(),
        }
        if doc["assets"].as_object().is_some_and(Map::is_empty) {
            doc.as_object_mut().map(|d| d.shift_remove("assets"));
        }
    }
    Ok((doc, glsl))
}

/// Set the shader of pass `pass`, or the common block for None, in `doc`.
fn put_glsl(doc: &mut Value, pass: Option<&str>, text: &str, dir: &str, glsl: &mut Vec<Glsl>) {
    let (holder, key, own) = match pass {
        Some(p) => match raw_pass_mut(doc, p) {
            Some(h) => (h, "fragment", p.to_string()),
            None => return,
        },
        None => (doc, "common", "common".to_string()),
    };
    match holder.get(key) {
        Some(Value::String(_)) => holder[key] = Value::from(text),
        Some(Value::Object(o)) if o.get("file").and_then(Value::as_str).is_some() => {
            let file = o["file"].as_str().unwrap_or_default().to_string();
            glsl.retain(|g| g.file != file);
            glsl.push(Glsl {
                pass: pass.map(str::to_string),
                file,
                text: text.to_string(),
            });
        }
        _ => {
            let file = free_glsl(dir, &own, glsl);
            holder[key] = serde_json::json!({ "file": file });
            glsl.push(Glsl {
                pass: pass.map(str::to_string),
                file,
                text: text.to_string(),
            });
        }
    }
}

/// `<stem>.glsl`, or `<stem>-2.glsl` and on, the first that no file in
/// `dir` and no shader of `glsl` takes.
fn free_glsl(dir: &str, stem: &str, glsl: &[Glsl]) -> String {
    (1..)
        .map(|i| match i {
            1 => format!("{stem}.glsl"),
            i => format!("{stem}-{i}.glsl"),
        })
        .find(|f| !Path::new(&format!("{dir}/{f}")).exists() && !glsl.iter().any(|g| g.file == *f))
        .unwrap_or_default()
}

/// Set key `k` of `o` to `v`, or remove it for a null.
fn set_or_drop(o: &mut Value, k: &str, v: &Value) {
    match v.is_null() {
        true => {
            o.as_object_mut().map(|o| o.shift_remove(k));
        }
        false => o[k] = v.clone(),
    }
}

/// Set the value of param `name` of `owner`, a pass or the scene. None when
/// it has no such param.
fn set_value(owner: &mut Value, name: &str, v: &Value) -> Option<()> {
    let p = owner.get_mut("params")?.get_mut(name)?.as_object_mut()?;
    p.insert("value".to_string(), v.clone());
    Some(())
}

/// A pass's name as a scene file holds it: `main` when it names none.
fn raw_name(p: &Value) -> &str {
    p.get("name").and_then(Value::as_str).unwrap_or("main")
}

fn raw_pass<'a>(doc: &'a Value, name: &str) -> Option<&'a Value> {
    doc.get("passes")?
        .as_array()?
        .iter()
        .find(|p| raw_name(p) == name)
}

fn raw_pass_mut<'a>(doc: &'a mut Value, name: &str) -> Option<&'a mut Value> {
    doc.get_mut("passes")?
        .as_array_mut()?
        .iter_mut()
        .find(|p| raw_name(p) == name)
}

fn raw_layer<'a>(doc: &'a Value, name: &str) -> Option<&'a Value> {
    doc.get("passes")?
        .as_array()?
        .iter()
        .filter_map(|p| p.get("layers")?.as_array())
        .flatten()
        .find(|l| l.get("name").and_then(Value::as_str) == Some(name))
}

fn raw_layer_mut<'a>(doc: &'a mut Value, name: &str) -> Option<&'a mut Value> {
    doc.get_mut("passes")?
        .as_array_mut()?
        .iter_mut()
        .filter_map(|p| p.get_mut("layers")?.as_array_mut())
        .flatten()
        .find(|l| l.get("name").and_then(Value::as_str) == Some(name))
}

/// The scenes in a directory, as package() finds them, the documents first.
/// An executable beside its .scene.json is one scene.
fn scan(dir: &str) -> Vec<(String, String)> {
    let found: Vec<Package> = entries(dir)
        .iter()
        .filter_map(|n| package(dir, n))
        .collect();
    let mut out: Vec<(String, String)> = found
        .iter()
        .filter(|p| p.document)
        .map(|p| (p.name.clone(), p.path.clone()))
        .collect();
    out.sort();
    for p in found.into_iter().filter(|p| !p.document) {
        if !out.iter().any(|(name, _)| *name == p.name) {
            out.push((p.name, p.path));
        }
    }
    out
}

/// Every `name/name.scene.json` in `dir`, by name.
fn scan_mine(dir: &str) -> Vec<(String, String)> {
    entries(dir)
        .iter()
        .filter(|n| Path::new(&format!("{dir}/{n}")).is_dir())
        .filter_map(|n| package(dir, n))
        .filter(|p| p.document)
        .map(|p| (p.name, p.path))
        .collect()
}

/// A scene that an entry of a directory holds.
struct Package {
    name: String,
    /// The scene's document, else the program that writes it.
    path: String,
    document: bool,
}

/// The scene of entry `n` of `dir`, as procedural::resolve() finds a
/// package: a document, else an executable, a file or a member of a
/// directory alike. A directory whose members resolve() refuses is a
/// scene whose load says why. None for any other entry.
fn package(dir: &str, n: &str) -> Option<Package> {
    let path = format!("{dir}/{n}");
    let found = |name: String, path: String, document| {
        Some(Package {
            name,
            path,
            document,
        })
    };
    if Path::new(&path).is_dir() {
        if n.starts_with('.') {
            return None;
        }
        let found = |path, document| found(n.to_string(), path, document);
        return match crate::procedural::resolve(&path) {
            Ok((Some(json), _)) => found(json, true),
            Ok((None, Some(exe))) => found(exe, false),
            Ok((None, None)) => None,
            Err(_) => found(path, false),
        };
    }
    let name = crate::procedural::scene_name(n);
    if n.ends_with(SCENE_SUFFIX) {
        return found(name, path, true);
    }
    // A file without the execute bit is a module or data that the scenes
    // beside it use.
    match process::runnable(&path) {
        true => found(name, path, false),
        false => None,
    }
}

/// The names in a directory, sorted. None when it does not read.
fn entries(dir: &str) -> Vec<String> {
    let mut names: Vec<String> = match fs::read_dir(dir) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect(),
        Err(_) => Vec::new(),
    };
    names.sort();
    names
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

        store.filters = vec![crate::filters::Filter::named("dither")];
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

    #[test]
    fn a_scene_or_a_store_filter_that_no_filter_answers_fails_its_load() {
        let d = tmpdir("unknown-filter");
        let own = r#"{"version": 1, "filters": ["nope"],
             "passes": [{"name": "main", "fragment": "void main(){ gl_FragColor = vec4(1.0); }"}]}"#;
        write_text(&format!("{d}/own.scene.json"), own);
        write_text(&format!("{d}/plain.scene.json"), SCENE);
        let mut store = SceneStore::new(&d, Vec::new());
        store.filters = vec![crate::filters::Filter::named("dither")];
        let e = store.start("own", SIZE).err().unwrap();
        assert!(
            e.message.starts_with("unknown filter: nope (known: "),
            "{}",
            e.message
        );
        store.filters = vec![crate::filters::Filter::named("nope")];
        let e = store.start("plain", SIZE).err().unwrap();
        assert!(
            e.message.starts_with("unknown filter: nope (known: "),
            "{}",
            e.message
        );
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
    fn the_store_lists_a_directory_that_holds_only_its_program_as_ghee_render_runs_it() {
        let (d, _) = scenes("program-package");
        fs::create_dir(format!("{d}/pkg")).unwrap();
        write_scene(&format!("{d}/pkg/pkg.py"), "#!/bin/sh\nexit 0\n");
        fs::create_dir(format!("{d}/empty")).unwrap();
        write_text(&format!("{d}/empty/notes.txt"), "");
        let store = SceneStore::new(&d, Vec::new());
        assert_eq!(store.names(), ["one", "two", "pkg"]);
        assert_eq!(store.procedural_names(), ["pkg"]);
        assert_eq!(
            store.file("pkg").unwrap(),
            process::executable(&format!("{d}/pkg/")).unwrap()
        );
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
    fn a_filter_s_values_from_the_scene_file_outlast_an_edit_of_the_filter_list() {
        let d = tmpdir("filter-values");
        let mut doc: Value = serde_json::from_str(SCENE).unwrap();
        doc["filters"] = json!([{"name": "frc", "calm": 8}]);
        write_text(&format!("{d}/f{SCENE_SUFFIX}"), &doc.to_string());
        let mut store = SceneStore::new(&d, Vec::new());
        let calm = |sc: &Scene| sc.passes[1].params.get("calm").unwrap().value.to_json();
        let defaults = store.resolve("f", SIZE).unwrap();
        assert_eq!(calm(&defaults), json!(8.0));
        // The console adds dither after frc.
        let mut w = defaults.clone();
        w.filters.push(crate::filters::Filter::named("dither"));
        crate::filters::rebuild_filters(&mut w, None).unwrap();
        let edits = state::diff(&defaults.to_dict(), &w.to_dict(), false);
        let (back, aside, broken) = working("f", &defaults, &edits);
        assert!(aside.is_empty() && broken.is_empty());
        assert_eq!(calm(&back), json!(8.0));
        assert_eq!(back.passes[2].filter, "dither");
        // A host filter of the same name sets its values in the scene's place.
        let host = json!({"name": "frc", "calm": 10});
        store.filters = vec![crate::filters::Filter::from_json(&host).unwrap()];
        let (hosted, _) = store.start("f", SIZE).unwrap();
        assert_eq!(hosted.passes.len(), 2);
        assert_eq!(calm(&hosted), json!(10.0));
    }

    /// A .scene.json and the program beside it are the members of one
    /// package, and ghee runs the program.
    #[test]
    fn a_scene_file_and_the_program_beside_it_list_as_one_scene() {
        let d = tmpdir("pair");
        write_text(&format!("{d}/wob{SCENE_SUFFIX}"), SCENE);
        write_scene(&format!("{d}/wob.py"), "#!/bin/sh\nexit 0\n");
        let store = SceneStore::new(&d, Vec::new());
        assert_eq!(store.names(), vec!["wob".to_string()]);
        assert_eq!(store.file("wob").unwrap(), format!("{d}/wob.py"));
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

    /// A state file that does not read moves aside at the load, whole, so a
    /// save leaves it, and a second one takes the next free name.
    #[test]
    fn an_unreadable_state_file_moves_aside_and_outlives_the_next_save() {
        let (_d, _states, store) = kept("unreadable");
        let file = store.state_file("one").unwrap().unwrap();
        let aside = |n: &str| PathBuf::from(format!("{}.unreadable{n}", file.display()));
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, "{\"version\": 1, \"values\": ").unwrap();
        let (opened, _) = store.open("one", None, SIZE).unwrap();
        assert!(opened.saved.is_none());
        assert!(!file.exists());
        let mut edits = Edits::default();
        edits.values.insert("params".to_string(), json!({"x": 1}));
        store.save("one", &edits).unwrap();
        assert_eq!(
            fs::read_to_string(aside("")).unwrap(),
            "{\"version\": 1, \"values\": "
        );
        fs::write(&file, "{\"version\": 9}").unwrap();
        store.open("one", None, SIZE).unwrap();
        assert_eq!(fs::read_to_string(aside(".2")).unwrap(), "{\"version\": 9}");
        assert_eq!(
            fs::read_to_string(aside("")).unwrap(),
            "{\"version\": 1, \"values\": "
        );
    }

    /// A directory of scene folders, `name/name.scene.json`.
    #[test]
    fn the_scan_lists_a_scene_folder_one_level_down() {
        let d = tmpdir("folders");
        write_text(&format!("{d}/flat{SCENE_SUFFIX}"), SCENE);
        fs::create_dir_all(format!("{d}/clocks")).unwrap();
        write_text(&format!("{d}/clocks/clocks{SCENE_SUFFIX}"), SCENE);
        fs::create_dir_all(format!("{d}/fonts")).unwrap();
        write_text(&format!("{d}/fonts/other{SCENE_SUFFIX}"), SCENE);
        let store = SceneStore::new(&d, Vec::new());
        assert_eq!(store.names(), ["clocks", "flat"]);
        assert_eq!(
            store.file("clocks").unwrap(),
            format!("{d}/clocks/clocks{SCENE_SUFFIX}")
        );
    }

    /// `dir` of the store, and `mine`, the console's own scenes.
    fn mine(tag: &str) -> (crate::testing::TempDir, crate::testing::TempDir, SceneStore) {
        let d = tmpdir(tag);
        write_text(&format!("{d}/one{SCENE_SUFFIX}"), SCENE);
        let m = tmpdir(&format!("{tag}-mine"));
        fs::create_dir_all(format!("{m}/own")).unwrap();
        write_text(&format!("{m}/own/own{SCENE_SUFFIX}"), SCENE);
        let store = SceneStore::with_mine(&d, Some(&m), Vec::new());
        (d, m, store)
    }

    #[test]
    fn the_consoles_own_scenes_list_after_the_served_ones_and_a_rescan_finds_a_new_one() {
        let (d, m, store) = mine("own-list");
        assert_eq!(store.names(), ["one", "own"]);
        assert_eq!(store.mine_names(), ["own"]);
        assert!(store.is_mine("own") && !store.is_mine("one"));
        write_text(&format!("{d}/two{SCENE_SUFFIX}"), SCENE);
        fs::create_dir_all(format!("{m}/later")).unwrap();
        write_text(&format!("{m}/later/later{SCENE_SUFFIX}"), SCENE);
        assert_eq!(store.names(), ["one", "own"]);
        store.rescan();
        assert_eq!(store.names(), ["one", "two", "later", "own"]);
    }

    /// The console never runs a program in its own scenes: neither a scene
    /// file's executable beside it nor an executable of its own lists.
    #[test]
    fn no_program_among_the_consoles_own_scenes_runs() {
        let (_d, m, store) = mine("own-programs");
        write_scene(&format!("{m}/own/own.py"), "#!/bin/sh\nexit 1\n");
        write_scene(&format!("{m}/run.py"), "#!/bin/sh\nexit 1\n");
        fs::create_dir_all(format!("{m}/prog")).unwrap();
        write_scene(&format!("{m}/prog/prog.py"), "#!/bin/sh\nexit 1\n");
        store.rescan();
        assert_eq!(store.names(), ["one", "own"]);
        assert!(store.procedural_names().is_empty());
        assert_eq!(
            store.file("own").unwrap(),
            format!("{m}/own/own{SCENE_SUFFIX}")
        );
        assert!(store.start("own", SIZE).unwrap().1.is_none());
    }

    /// An own scene as a person writes it: shaders in files of their own
    /// names, and entries on one line each.
    const ART: &str = r#"{
  "version": 1,
  "common": {"file": "art-common.glsl"},
  "passes": [
    {"name": "main", "fragment": {"file": "art-main.glsl"},
     "params": {
       "gain":  {"type": "float", "value": 0.75, "min": 0.2, "max": 3.0},
       "grain": {"type": "float", "value": 0.03}
     },
     "layers": [{"name": "word", "visible": false, "text": "exhale"}]}
  ]
}
"#;

    #[test]
    fn a_save_of_an_own_scene_changes_only_what_changed_in_the_files_that_it_names() {
        let (_d, m, store) = mine("own-write");
        let dir = format!("{m}/own");
        write_text(&format!("{dir}/own{SCENE_SUFFIX}"), ART);
        write_text(
            &format!("{dir}/art-main.glsl"),
            "void main(){ gl_FragColor = vec4(k()); }",
        );
        write_text(
            &format!("{dir}/art-common.glsl"),
            "float k(){ return 1.0; }",
        );
        let scene = store.resolve("own", SIZE).unwrap();
        let defaults = scene.to_dict();
        let mut doc = defaults.clone();
        doc["passes"][0]["params"]["gain"]["value"] = json!(1.2);
        doc["passes"][0]["params"]["hot"] = json!({"type": "color", "value": "#ffd9a8"});
        doc["passes"][0]["fragment"] = json!("void main(){ gl_FragColor = vec4(k() * gain); }");
        doc["passes"][0]["layers"][0]["text"] = json!("inhale");
        doc["common"] = json!("float k(){ return 0.5; }");
        doc["filters"] = json!(["dither"]);
        let working = scene_of(&scene, &doc, "own").unwrap().to_dict();
        let edits = state::diff(&defaults, &working, false);
        store.write("own", &defaults, &edits).unwrap();
        assert_eq!(
            fs::read_to_string(format!("{dir}/own{SCENE_SUFFIX}")).unwrap(),
            ART.replace("\"value\": 0.75", "\"value\": 1.2")
                .replace(
                    "\"value\": 0.03}\n",
                    "\"value\": 0.03},\n       \"hot\": {\"type\": \"color\", \"value\": \"#ffd9a8\"}\n"
                )
                .replace("\"exhale\"", "\"inhale\"")
                .replace("  ]\n}", "  ],\n  \"filters\": [\"dither\"]\n}")
        );
        assert_eq!(
            fs::read_to_string(format!("{dir}/art-main.glsl")).unwrap(),
            "void main(){ gl_FragColor = vec4(k() * gain); }"
        );
        assert_eq!(
            fs::read_to_string(format!("{dir}/art-common.glsl")).unwrap(),
            "float k(){ return 0.5; }"
        );
        let mut files: Vec<String> = entries(&dir);
        files.sort();
        assert_eq!(
            files,
            ["art-common.glsl", "art-main.glsl", "own.scene.json"]
        );
        let back = store.resolve("own", SIZE).unwrap();
        assert_eq!(back.passes[1].filter, "dither");
        assert_eq!(store.state_file("own").unwrap(), None);
    }

    #[test]
    fn a_new_pass_of_an_own_scene_saves_its_shader_into_a_file_of_its_name() {
        let (_d, m, store) = mine("own-pass");
        let dir = format!("{m}/own");
        let defaults = store.resolve("own", SIZE).unwrap().to_dict();
        let mut doc = defaults.clone();
        let mut blur = doc["passes"][0].clone();
        blur["name"] = json!("blur");
        blur["fragment"] = json!("void main(){ gl_FragColor = vec4(0.5); }");
        doc["passes"].as_array_mut().unwrap().push(blur);
        write_text(&format!("{dir}/blur.glsl"), "taken");
        store
            .write("own", &defaults, &state::diff(&defaults, &doc, false))
            .unwrap();
        let written: Value =
            serde_json::from_str(&fs::read_to_string(format!("{dir}/own{SCENE_SUFFIX}")).unwrap())
                .unwrap();
        assert_eq!(
            written["passes"][1]["fragment"],
            json!({"file": "blur-2.glsl"})
        );
        assert_eq!(
            fs::read_to_string(format!("{dir}/blur.glsl")).unwrap(),
            "taken"
        );
        let back = store.resolve("own", SIZE).unwrap();
        assert_eq!(
            back.passes[1].fragment,
            "void main(){ gl_FragColor = vec4(0.5); }"
        );
        assert_eq!(
            back.passes[0].fragment,
            "void main(){ gl_FragColor = vec4(1.0); }"
        );
    }

    #[test]
    fn a_new_scene_takes_a_free_name_and_the_files_its_scene_names() {
        let (d, m, store) = mine("own-new");
        fs::write(format!("{d}/pic.png"), crate::testing::png(2, 2)).unwrap();
        let mut doc: Value = serde_json::from_str(SCENE).unwrap();
        doc["assets"] = json!({"pic": {"file": "pic.png"}});
        store
            .create("copy", &doc, Some(&d), &files_of(&doc))
            .unwrap();
        assert!(store.is_mine("copy"));
        assert!(Path::new(&format!("{m}/copy/pic.png")).is_file());
        assert_eq!(store.resolve("copy", SIZE).unwrap().assets[0].name, "pic");
        store.create("blank", &blank(), None, &[]).unwrap();
        let taken = store.create("one", &blank(), None, &[]).unwrap_err();
        assert_eq!(
            (taken.status, taken.message.as_str()),
            (409, "a scene named 'one' exists")
        );
        let bad = store.create("a b", &blank(), None, &[]).unwrap_err();
        assert_eq!(
            (bad.status, bad.message.as_str()),
            (
                400,
                "a scene name takes letters, digits, - and _, a letter or a digit first, and 64 \
                 of them at most, got 'a b'"
            )
        );
        let entries = fs::read_dir(&*m).unwrap().count();
        assert_eq!(entries, 3, "own, copy, blank and no temporary directory");
    }

    #[test]
    fn the_files_of_a_scene_hold_each_font_file_of_a_list() {
        let doc = json!({"passes": [{"name": "main", "layers": [
            {"name": "a", "text": "hi", "font_family": ["Pixel Operator 8", "a.ttf", "noto"]},
            {"name": "b", "text": "hi", "font_family": "b.otf"}]}]});
        assert_eq!(files_of(&doc), ["a.ttf", "b.otf"]);
    }

    /// A directory inside the console's own, served as a scenes
    /// directory, holds scenes of no one's own.
    #[test]
    fn a_scenes_directory_inside_the_consoles_own_is_not_its_own() {
        let m = tmpdir("own-inner");
        let d = format!("{m}/sub");
        fs::create_dir_all(&d).unwrap();
        write_text(&format!("{d}/one{SCENE_SUFFIX}"), SCENE);
        let store = SceneStore::with_mine(&d, Some(&m), Vec::new());
        assert_eq!(store.names(), ["one"]);
        assert!(!store.is_mine("one"));
        assert_eq!(store.delete("one").unwrap_err().status, 403);
    }

    #[test]
    fn the_consoles_own_scene_renames_and_goes_and_no_other_does() {
        let (d, m, store) = mine("own-rename");
        store.rename("own", "mine").unwrap();
        assert_eq!(store.names(), ["one", "mine"]);
        assert!(Path::new(&format!("{m}/mine/mine{SCENE_SUFFIX}")).is_file());
        assert!(!Path::new(&format!("{m}/own")).exists());
        assert_eq!(store.rename("one", "two").unwrap_err().status, 403);
        assert_eq!(store.delete("one").unwrap_err().status, 403);
        assert!(Path::new(&format!("{d}/one{SCENE_SUFFIX}")).is_file());
        store.delete("mine").unwrap();
        assert_eq!(store.names(), ["one"]);
        assert!(!Path::new(&format!("{m}/mine")).exists());
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
