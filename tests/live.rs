//! Live fields of layers: a value set on a running scene, with no rebuild,
//! against the same scene built with that value, on this host's GPU.
//!
//! The pictures that a built layer paints are checked against references
//! computed off the GPU in tests/text.rs and tests/image.rs, so a live set
//! that draws the built frame byte for byte draws the reference too.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use serde_json::{json, Value};

use ghee::engine::{probe_dri_devices, Given, Refused, Renderer};
use ghee::layer::Address;
use ghee::scene::Scene;

/// A Renderer opens a GL device, and load_assets reads $GHEE_ASSETS. A test
/// that does either holds this, so no variable changes under a device or a
/// lookup.
static LOCK: Mutex<()> = Mutex::new(());

/// A w x h RGBA renderer on the first GL device that opens. Its atlases cache
/// under the temp directory.
fn gpu(w: usize, h: usize) -> (MutexGuard<'static, ()>, Renderer) {
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var(
        "XDG_CACHE_HOME",
        std::env::temp_dir().join("ghee-live-cache"),
    );
    let mut errs = Vec::new();
    for dev in probe_dri_devices() {
        match Renderer::new(w as i32, h as i32, 4, Some(&dev), false, false) {
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

/// A directory of one test's own under the temp directory, which holds the
/// files of its scenes, pic.png and pic2.png among them. The drop removes it.
struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("ghee-live-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        write_pictures(&d);
        Dir(d)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    /// `doc` with its files here. The caller holds LOCK.
    fn load(&self, doc: &Value) -> Scene {
        let mut sc = Scene::from_dict(doc, None, None).unwrap();
        sc.load_assets(&self.0.to_string_lossy()).unwrap();
        sc
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Write `texels`, RGBA rows of `w` from the top, as `name` in `dir`.
fn write_png(dir: &Path, name: &str, w: u32, texels: &[[u8; 4]]) {
    let file = fs::File::create(dir.join(name)).unwrap();
    let h = texels.len() as u32 / w;
    let mut enc = png::Encoder::new(file, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut out = enc.write_header().unwrap();
    out.write_image_data(texels.as_flattened()).unwrap();
}

const BLACK: &str = "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }";

/// A scene of pass `main` over black with `layers`, and the image assets
/// `pic` and `pic2`.
fn scene(layers: Value) -> Value {
    json!({
        "version": 1,
        "assets": {"pic": {"file": "pic.png"}, "pic2": {"file": "pic2.png"}},
        "passes": [{"name": "main", "fragment": BLACK, "layers": layers}],
    })
}

/// Write pic.png, 2x2, and pic2.png, 3x1, into `dir`.
fn write_pictures(dir: &Path) {
    let pic = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255; 4],
    ];
    write_png(dir, "pic.png", 2, &pic);
    write_png(dir, "pic2.png", 3, &[[255, 255, 0, 255]; 3]);
}

fn render(r: &mut Renderer, sc: &Scene) -> Vec<u8> {
    assert_eq!(r.setup(sc).unwrap(), "");
    again(r)
}

/// The next frame of the running scene, at time 0.
fn again(r: &mut Renderer) -> Vec<u8> {
    r.set_time(Some(0.0));
    r.render(None).unwrap().to_vec()
}

/// The largest difference of a channel between two frames.
fn worst(a: &[u8], b: &[u8]) -> u8 {
    a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap()
}

/// `doc` with the field at `at` written as `value`, as a scene declares it.
fn written(doc: &Value, at: &Address, value: &Value) -> Value {
    let mut doc = doc.clone();
    let layers = doc["passes"][0]["layers"].as_array_mut().unwrap();
    let l = layers.iter_mut().find(|l| l["name"] == at.layer).unwrap();
    match at.effect {
        Some(e) => l[e][at.field] = value.clone(),
        None if value.is_null() => {
            l.as_object_mut().unwrap().remove(at.field);
        }
        None => l[at.field] = value.clone(),
    }
    doc
}

fn at<'a>(layer: &'a str, effect: Option<&'a str>, field: &'a str) -> Address<'a> {
    Address {
        layer,
        effect,
        field,
    }
}

/// A live value as a wire message gives it.
#[derive(Debug)]
enum Wire {
    Numbers(Vec<f32>),
    Text(&'static str),
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_live_field_draws_what_a_scene_built_with_its_value_draws() {
    // Each row sets a field as the scene format writes it, then as the wire's
    // numbers or text give it.
    let dir = Dir::new("fields");
    let doc = scene(json!([
        {"name": "t", "text": "Hi", "font_size": 40, "x": 10, "y": 8,
         "text_shadow": {}, "glow": {}, "text_stroke": {}, "karaoke": {"sung": 1}},
        {"name": "i", "image": "pic", "x": 110, "y": 10, "width": 8, "color": "#80ff80"},
    ]));
    let orange = 128.0 / 255.0;
    let rows = [
        (at("t", None, "x"), json!(30), Wire::Numbers(vec![30.0])),
        (at("t", None, "y"), json!(null), Wire::Numbers(vec![])),
        (
            at("t", None, "color"),
            json!("#ff0000"),
            Wire::Numbers(vec![1.0, 0.0, 0.0]),
        ),
        (
            at("t", None, "color"),
            json!({"from": "#ff0000", "to": [0, 0, 1], "angle": 90}),
            Wire::Numbers(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 90.0]),
        ),
        (
            at("t", None, "color"),
            json!("Transparent"),
            Wire::Text("Transparent"),
        ),
        (
            at("t", None, "color"),
            json!({"from": "red", "to": "transparent", "angle": 90}),
            Wire::Text(r#"{"from": "red", "to": "transparent", "angle": 90}"#),
        ),
        (
            at("t", None, "opacity"),
            json!(0.5),
            Wire::Numbers(vec![0.5]),
        ),
        (
            at("t", None, "visible"),
            json!(false),
            Wire::Numbers(vec![0.0]),
        ),
        (
            at("t", Some("text_shadow"), "visible"),
            json!(false),
            Wire::Numbers(vec![0.0]),
        ),
        (
            at("t", Some("text_shadow"), "color"),
            json!("#00ff00"),
            Wire::Numbers(vec![0.0, 1.0, 0.0]),
        ),
        (
            at("t", Some("text_shadow"), "strength"),
            json!(0.3),
            Wire::Numbers(vec![0.3]),
        ),
        (
            at("t", Some("glow"), "visible"),
            json!(false),
            Wire::Numbers(vec![0.0]),
        ),
        (
            at("t", Some("glow"), "color"),
            json!([0, 0, 1]),
            Wire::Numbers(vec![0.0, 0.0, 1.0]),
        ),
        (
            at("t", Some("glow"), "color"),
            json!("transparent"),
            Wire::Text("transparent"),
        ),
        (
            at("t", Some("glow"), "strength"),
            json!(0.2),
            Wire::Numbers(vec![0.2]),
        ),
        (
            at("t", Some("text_stroke"), "color"),
            json!("#ffffff"),
            Wire::Numbers(vec![1.0, 1.0, 1.0]),
        ),
        (
            at("t", Some("karaoke"), "sung"),
            json!(0.5),
            Wire::Numbers(vec![0.5]),
        ),
        (
            at("t", Some("karaoke"), "color"),
            json!("#00ffff"),
            Wire::Numbers(vec![0.0, 1.0, 1.0]),
        ),
        (at("i", None, "x"), json!(120), Wire::Numbers(vec![120.0])),
        (at("i", None, "y"), json!(null), Wire::Numbers(vec![])),
        (
            at("i", None, "color"),
            json!("#ff8000"),
            Wire::Numbers(vec![1.0, orange, 0.0]),
        ),
        (
            at("i", None, "color"),
            json!("DarkOrange"),
            Wire::Text("DarkOrange"),
        ),
        (at("i", None, "color"), json!(null), Wire::Numbers(vec![])),
        (
            at("i", None, "opacity"),
            json!(0.5),
            Wire::Numbers(vec![0.5]),
        ),
        (
            at("i", None, "visible"),
            json!(false),
            Wire::Numbers(vec![0.0]),
        ),
        (at("i", None, "image"), json!("pic2"), Wire::Text("pic2")),
    ];
    let (_g, mut r) = gpu(160, 64);
    for (at, value, wire) in rows {
        let row = format!("{at:?} = {value}");
        let before = render(&mut r, &dir.load(&doc));
        r.set_layer(&at, Given::Json(&value)).unwrap();
        let live = again(&mut r);
        let built = render(&mut r, &dir.load(&written(&doc, &at, &value)));
        assert_ne!(live, before, "{row}: the frame changes");
        assert_eq!(worst(&live, &built), 0, "{row}");
        render(&mut r, &dir.load(&doc));
        let given = match &wire {
            Wire::Numbers(numbers) => Given::Numbers(numbers),
            Wire::Text(text) => Given::Text(text),
        };
        r.set_layer(&at, given).unwrap();
        assert_eq!(worst(&again(&mut r), &built), 0, "{row} from {wire:?}");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_hidden_layer_shown_live_builds_its_atlas_and_draws_its_words() {
    let dir = Dir::new("shown");
    let shown = scene(json!([{"name": "t", "text": "Hi", "font_size": 40, "x": 10, "y": 8}]));
    let mut hidden = shown.clone();
    hidden["passes"][0]["layers"][0]["visible"] = json!(false);
    let mut none = shown.clone();
    none["passes"][0]["layers"] = json!([]);
    let (_g, mut r) = gpu(160, 64);
    let empty = render(&mut r, &dir.load(&none));
    let want = render(&mut r, &dir.load(&shown));
    assert_eq!(render(&mut r, &dir.load(&hidden)), empty);
    r.set_layer(&at("t", None, "visible"), Given::Json(&json!(true)))
        .unwrap();
    // A pinned time waits for the glyphs, so the next frame draws them.
    assert_eq!(worst(&again(&mut r), &want), 0);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_colour_glyph_drawn_after_a_live_set_takes_the_live_values() {
    // The colour glyphs of a layer compile at its first words that hold one,
    // here after the layer moved and faded live.
    let dir = Dir::new("emoji");
    fs::copy(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tools/fixtures/fonts/NotoColorEmoji-Smiles.ttf"
        ),
        dir.path().join(ghee::emoji::FILE),
    )
    .unwrap();
    let doc = scene(json!([{"name": "t", "text": "Hi", "font_size": 30, "x": 10, "y": 8}]));
    let words = "Hi \u{1f600}";
    let mut want = doc.clone();
    want["passes"][0]["layers"][0] = json!(
        {"name": "t", "text": words, "font_size": 30, "x": 50, "y": 8, "opacity": 0.5}
    );
    let mut plain = want.clone();
    plain["passes"][0]["layers"][0]["text"] = json!("Hi ");
    let (_g, mut r) = gpu(160, 64);
    let built = render(&mut r, &dir.load(&want));
    let letters = render(&mut r, &dir.load(&plain));
    render(&mut r, &dir.load(&doc));
    r.set_layer(&at("t", None, "x"), Given::Json(&json!(50)))
        .unwrap();
    r.set_layer(&at("t", None, "opacity"), Given::Json(&json!(0.5)))
        .unwrap();
    r.set_layer(&at("t", None, "text"), Given::Text(words))
        .unwrap();
    let live = again(&mut r);
    assert_ne!(live, letters, "the emoji draws");
    assert_eq!(worst(&live, &built), 0);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_live_set_that_the_scene_refuses_says_why_and_changes_nothing() {
    let dir = Dir::new("refused");
    let doc = scene(json!([
        {"name": "t", "text": "Hi", "font_size": 40, "x": 10, "y": 8},
        {"name": "i", "image": "pic", "x": 110, "y": 10},
    ]));
    let (one, two, white, file) = (json!(1), json!(2), json!("#ffffff"), json!("nope.png"));
    let rows = [
        (at("nope", None, "x"), Given::Json(&one), "has no layer"),
        (
            at("t", None, "font_size"),
            Given::Json(&two),
            "changes with a rebuild",
        ),
        (
            at("t", Some("glow"), "color"),
            Given::Json(&white),
            "has no effect",
        ),
        (at("t", None, "opacity"), Given::Json(&two), "from 0 to 1"),
        (
            at("t", None, "x"),
            Given::Numbers(&[1.0, 2.0]),
            "takes 1 number",
        ),
        (at("t", None, "x"), Given::Text("1"), "got text"),
        (
            at("i", None, "image"),
            Given::Json(&file),
            "No such file or directory",
        ),
    ];
    let (_g, mut r) = gpu(160, 64);
    let before = render(&mut r, &dir.load(&doc));
    for (at, given, why) in rows {
        let e = r.set_layer(&at, given).unwrap_err();
        let refused = e.downcast_ref::<Refused>();
        assert!(refused.is_some_and(|r| r.0.contains(why)), "{at:?}: {e}");
        assert_eq!(again(&mut r), before, "{at:?}");
    }
}

/// The left edge of the first lit column of `frame`, w pixels wide.
fn left(frame: &[u8], w: usize) -> Option<usize> {
    (0..w).find(|x| {
        frame
            .as_chunks::<4>()
            .0
            .iter()
            .skip(*x)
            .step_by(w)
            .any(|p| p[0] > 0)
    })
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_new_speed_carries_the_motion_on_from_where_it_is() {
    // A scroll moves speed em a second, and each render advances it by the
    // speed of the moment times the time since the render before. Every
    // number here is exact in binary, so the block lands on the pixel that
    // the arithmetic gives.
    let dir = Dir::new("speed");
    let (w, h, ppem, dt) = (160, 48, 16.0, 0.0625);
    let doc = scene(
        json!([{"name": "t", "text": "Hi", "font_size": ppem, "y": 8,
                            "scroll": {"speed": 2}}]),
    );
    let still =
        |x: f64| scene(json!([{"name": "t", "text": "Hi", "font_size": ppem, "x": x, "y": 8}]));
    let speed = at("t", Some("scroll"), "speed");
    let (_g, mut r) = gpu(w, h);
    // Each frame: the pinned time, else the time step, and the speed set
    // before it from frame 10 on.
    let run = |r: &mut Renderer, pinned: bool| -> Vec<Vec<u8>> {
        assert_eq!(r.setup(&dir.load(&doc)).unwrap(), "");
        r.set_dt(dt);
        (0..16)
            .map(|k| {
                if k == 10 {
                    r.set_layer(&speed, Given::Json(&json!(6))).unwrap();
                }
                r.set_time(pinned.then_some(f64::from(k) * dt));
                r.render(None).unwrap().to_vec()
            })
            .collect()
    };
    let pinned = run(&mut r, true);
    assert_eq!(run(&mut r, true), pinned, "a pinned run repeats");
    let width = {
        let sc = dir.load(&still(0.0));
        let l = sc.text_layer("t").unwrap();
        l.place("Hi").unwrap().built.unwrap().size[0] * ppem
    };
    let mut phase = 0.0;
    for (k, frame) in pinned.iter().enumerate() {
        if k > 0 {
            phase += if k < 10 { 2.0 } else { 6.0 } * dt;
        }
        let x = (w as f64 - (phase * ppem).rem_euclid(w as f64 + width)).floor();
        let want = render(&mut r, &dir.load(&still(x)));
        assert!(worst(frame, &want) <= 1, "frame {k} at x {x}");
    }
    assert_eq!(left(&pinned[10], w), left(&pinned[9], w).map(|x| x - 6));
    // Running free, each render advances the motion by the time step: the
    // first one too, so the frames run one step ahead of the pinned ones.
    let free = run(&mut r, false);
    for k in 0..9 {
        assert_eq!(free[k], pinned[k + 1], "free frame {k}");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_shader_reads_a_hidden_layers_words_as_a_shown_ones_and_its_new_words_at_once() {
    // main draws t's words through t_cover and never reads the pass that
    // draws t itself.
    let dir = Dir::new("cover");
    let doc = |visible: bool, words: &str| {
        json!({"version": 1, "passes": [
            {"name": "words", "fragment": BLACK, "layers": [
                {"name": "t", "text": words, "font_size": 16, "visible": visible}]},
            {"name": "main", "fragment": "void main(){ float fs = 16.0; \
                vec2 px = vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y); \
                vec2 p = (px - 0.5 * (iResolution.xy - t_size * fs)) / fs; \
                gl_FragColor = vec4(vec3(t_cover(p, fs)), 1.0); }"},
        ]})
    };
    let (_g, mut r) = gpu(64, 32);
    let frame = |r: &mut Renderer, visible: bool, words: &str| {
        assert_eq!(r.setup(&dir.load(&doc(visible, words))).unwrap(), "");
        r.set_time(Some(0.0));
        r.render(None).unwrap().to_vec()
    };
    let shown = frame(&mut r, true, "Hi");
    assert!(shown.chunks(4).any(|p| p[0] > 128), "no words drawn");
    let lo = frame(&mut r, false, "Lo");
    assert_eq!(frame(&mut r, false, "Hi"), shown);
    r.set_layer(&at("t", None, "text"), Given::Text("Lo"))
        .unwrap();
    r.set_time(Some(0.0));
    assert_eq!(r.render(None).unwrap(), lo);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_hidden_layer_that_moves_draws_no_frame_and_shows_where_its_motion_went() {
    // A hidden scroll moves on with nothing to draw, so each of its frames
    // repeats the one before, and once shown it stands where it would stand
    // had it shown all along.
    let dir = Dir::new("hidden-motion");
    let doc = |visible: bool| {
        scene(json!([{"name": "t", "text": "Hi", "font_size": 16, "y": 8,
                      "visible": visible, "scroll": {"speed": 2}}]))
    };
    let dt = 0.0625;
    let (_g, mut r) = gpu(160, 48);
    assert_eq!(r.setup(&dir.load(&doc(true))).unwrap(), "");
    let mut shown = Vec::new();
    for k in 0..8 {
        r.set_time(Some(f64::from(k) * dt));
        shown = r.render(None).unwrap().to_vec();
    }
    assert_eq!(r.setup(&dir.load(&doc(false))).unwrap(), "");
    r.set_time(Some(0.0));
    let hidden = r.render(None).unwrap().to_vec();
    let draws = r.draws();
    for k in 1..7 {
        r.set_time(Some(f64::from(k) * dt));
        assert_eq!(r.render(None).unwrap(), hidden, "frame {k}");
    }
    assert_eq!(r.draws(), draws);
    r.set_layer(&at("t", None, "visible"), Given::Json(&json!(true)))
        .unwrap();
    r.set_time(Some(7.0 * dt));
    assert_eq!(r.render(None).unwrap(), shown);
    assert_eq!(r.draws(), draws + 1);
}
