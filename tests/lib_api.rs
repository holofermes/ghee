//! The lib's public surface, reached as a Rust consumer reaches it.

use std::sync::{Mutex, MutexGuard};

use serde_json::json;

use ghee::engine::{probe_dri_devices, Feed, Renderer};
use ghee::filters::apply_filters;
use ghee::scene::{ArrayInput, Channel, Param, ParamMap, Pass, PassBuilder, Scene, SceneBuilder};

const FLAT: &str = "void main(){ gl_FragColor = vec4(vec3(0.5), 1.0); }";

/// One Renderer at a time: the tests that render hold this.
static GPU: Mutex<()> = Mutex::new(());

/// A renderer on the first GL device that opens.
fn gpu() -> (MutexGuard<'static, ()>, Renderer) {
    let guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut errs = Vec::new();
    for dev in probe_dri_devices() {
        match Renderer::new(64, 48, 4, Some(&dev), false, false) {
            Ok(r) => return (guard, r),
            Err(e) => errs.push(format!("{dev}: {e}")),
        }
    }
    panic!(
        "no GL device opens ({}). A host without one runs the tests under \
         RUSTFLAGS=\"--cfg ghee_no_gpu\", which reports them as ignored",
        errs.join("; ")
    );
}

#[test]
fn the_sugar_builds_what_the_constructors_build() {
    let sweet = Pass::of("main", FLAT)
        .array("level", 8)
        .array("mask", (16, 4))
        .channel("main")
        .param(Param::float("gain", 1.0))
        .size(32, 24)
        .unwrap();

    let mut params = ParamMap::new();
    params.insert(
        "gain".to_string(),
        Param::new("gain", "float", json!(1.0), None, None, None, None).unwrap(),
    );
    let plain = Pass::new(
        "main",
        FLAT,
        vec![
            ArrayInput::new("level", json!(8)).unwrap(),
            ArrayInput::new("mask", json!([16, 4])).unwrap(),
        ],
        vec![Channel::new(Some("main"))],
        params,
        "",
        Some(json!([32, 24])),
    )
    .unwrap();

    assert_eq!(sweet, plain);
}

#[test]
fn a_size_refused_by_the_scene_model_stops_the_chain() {
    let e = Pass::of("main", FLAT)
        .array("level", 0)
        .param(Param::float("gain", 1.0))
        .unwrap_err();
    assert_eq!(
        e.to_string(),
        "array 'level' size needs an int in 1..16384, got 0"
    );
    let e = Pass::of("main", FLAT).size(0, 4).unwrap_err();
    assert_eq!(
        e.to_string(),
        "pass 'main' size needs two positive ints [w, h] up to 16384, got [0, 4]"
    );
}

#[test]
fn a_channel_binds_a_pass_buffer_or_nothing() {
    let p = Pass::of("main", FLAT)
        .channel(None)
        .channel("main")
        .unwrap();
    assert_eq!(p.channels, [Channel::new(None), Channel::new(Some("main"))]);
}

#[test]
fn each_param_constructor_carries_its_type_and_value() {
    assert_eq!(
        Param::float("gain", 0.5).to_dict(),
        json!({"type": "float", "value": 0.5})
    );
    assert_eq!(
        Param::int("steps", 4).to_dict(),
        json!({"type": "int", "value": 4})
    );
    assert_eq!(
        Param::bool("wrap", true).to_dict(),
        json!({"type": "bool", "value": true})
    );
    assert_eq!(
        Param::color("tint", [1.0, 0.0, 0.5]).to_dict(),
        json!({"type": "color", "value": [1.0, 0.0, 0.5]})
    );
    assert_eq!(
        Param::string("mode", "wave").to_dict(),
        json!({"type": "string", "value": "wave"})
    );
    for (values, ty) in [
        (vec![0.0, 1.0], "vec2"),
        (vec![0.0, 1.0, 2.0], "vec3"),
        (vec![0.0, 1.0, 2.0, 3.0], "vec4"),
    ] {
        assert_eq!(Param::vec("dir", &values).unwrap().ty, ty);
    }
    assert_eq!(
        Param::vec("dir", &[0.0]).unwrap_err().to_string(),
        "vec param 'dir' needs 2, 3 or 4 numbers, got 1"
    );
}

#[test]
fn the_json_path_and_the_sugar_build_the_same_param() {
    let from_json = |d| Param::from_dict("gain", &d).unwrap();
    assert_eq!(
        from_json(json!({"type": "float", "value": 0.5})),
        Param::float("gain", 0.5)
    );
    assert_eq!(
        from_json(json!({"type": "color", "value": [0.0, 0.25, 1.0]})),
        Param::color("gain", [0.0, 0.25, 1.0])
    );
    assert_eq!(
        from_json(json!({"type": "vec3", "value": [0.0, 0.25, 1.0]})),
        Param::vec("gain", &[0.0, 0.25, 1.0]).unwrap()
    );
}

#[test]
fn a_scene_param_cannot_shadow_a_pass_param() {
    let pass = Pass::of("main", FLAT)
        .param(Param::float("gain", 1.0))
        .unwrap();
    let e = Scene::of(vec![pass])
        .param(Param::float("gain", 0.5))
        .unwrap_err();
    assert_eq!(
        e.to_string(),
        "scene param 'gain' and param 'gain' in pass 'main' both declare 'gain'"
    );
}

#[test]
fn a_scene_filter_name_resolves_to_an_appended_pass() {
    let mut sc = Scene::of(vec![Pass::of("main", FLAT).unwrap()])
        .filters(&["dither"])
        .unwrap();
    assert_eq!(sc.filters, ["dither"]);
    let names = sc.filters.clone();
    apply_filters(&mut sc, &names, None).unwrap();
    let passes: Vec<&str> = sc.passes.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(passes, ["main", "dither"]);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_sugar_built_scene_renders_its_arrays_and_params() {
    let scene = Scene::of(vec![Pass::of(
        "main",
        "void main(){ gl_FragColor = vec4(level_at(0) * gain, 0.0, 0.0, 1.0); }",
    )
    .array("level", 4)
    .param(Param::float("gain", 1.0))
    .unwrap()])
    .unwrap();
    let (_g, mut r) = gpu();
    assert_eq!(r.setup(&scene).unwrap(), "");
    let frame = r
        .render(Some(&mut |_name, _size| Feed::Data(vec![1.0; 4])))
        .unwrap();
    assert_eq!(&frame[..4], [255, 0, 0, 255]);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_sugar_built_channel_reads_its_named_pass() {
    let scene = Scene::of(vec![
        Pass::of(
            "main",
            "void main(){ gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0); }",
        )
        .unwrap(),
        Pass::of(
            "out",
            "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }",
        )
        .channel("main")
        .unwrap(),
    ])
    .unwrap();
    let (_g, mut r) = gpu();
    assert_eq!(r.setup(&scene).unwrap(), "");
    let frame = r.render(None).unwrap();
    assert_eq!(&frame[..4], [255, 0, 255, 255]);
}
