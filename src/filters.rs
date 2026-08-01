//! Named post-process passes ghee ships, such as dither.
//!
//! A filter file (`<name>.filter.json`) holds one pass: a name, code and params.
//! The pass reads iChannel0, the frame so far, and writes the result. ghee
//! appends a filter as a pass that reads the pass before it. The last filter
//! reaches the display.
//!
//! The build compiles the shipped filters in. A scan adds more from a directory.

use anyhow::{bail, Result};
use serde_json::{Map, Value};

use crate::scene::{Pass, Scene};

/// The shipped filters. A binary carries no package directory, so the build
/// compiles them in.
const BUILTIN: [(&str, &str); 1] = [(
    "dither.filter.json",
    include_str!("../ghee/filters/dither.filter.json"),
)];

fn accept(out: &mut Map<String, Value>, path: &str, text: &str) {
    let parsed: Result<Value, _> = serde_json::from_str(text);
    let fd = match parsed {
        Ok(v) => v,
        Err(e) => {
            eprintln!("filter {path}: skipped ({e})");
            return;
        }
    };
    let named = fd.get("name").is_some_and(Value::is_string)
        && fd.get("code").is_some_and(Value::is_string);
    if !named {
        eprintln!("filter {path}: skipped (needs a string 'name' and 'code')");
        return;
    }
    let name = fd["name"].as_str().unwrap_or_default().to_string();
    out.insert(name, fd);
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
                Ok(text) => accept(&mut out, &p, &text),
                Err(e) => eprintln!("filter {p}: skipped ({e})"),
            }
        }
    }
    out
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
        let (Some(fname), Some(code)) = (
            fd.get("name").and_then(Value::as_str),
            fd.get("code").and_then(Value::as_str),
        ) else {
            bail!("filter '{name}': needs a string 'name' and 'code'");
        };
        let prev = scene.passes[scene.passes.len() - 1].name.clone();
        let mut d = Map::new();
        d.insert("name".to_string(), Value::from(unique(scene, fname)));
        d.insert("code".to_string(), Value::from(code));
        d.insert(
            "params".to_string(),
            fd.get("params")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new())),
        );
        let mut channel = Map::new();
        channel.insert("buffer".to_string(), Value::from(prev));
        d.insert(
            "channels".to_string(),
            Value::Array(vec![Value::Object(channel)]),
        );
        let mut p = Pass::from_dict(&Value::Object(d), None)?;
        p.filter = name.clone(); // The console renders a filter pass read-only.
        scene.passes.push(p);
    }
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

    fn tmpdir(tag: &str) -> String {
        let d = std::env::temp_dir().join(format!("ghee-filters-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d.to_string_lossy().into_owned()
    }

    #[test]
    fn registry_ships_dither() {
        let _g = env_guard();
        assert!(load_filters(&[]).contains_key("dither"));
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
            .glsl(Some(&sc.params), sc.common.as_deref(), false)
            .contains(COMMON));
        assert!(sc.passes[0]
            .glsl(Some(&sc.params), sc.common.as_deref(), false)
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
        std::fs::remove_dir_all(&dir).unwrap();
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
