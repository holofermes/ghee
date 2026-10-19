//! A running scene that grows: a layer added to it, with no rebuild, or
//! built again alone in its place, against the same scene built with that
//! layer in its place, on this host's GPU.
//!
//! The pictures that a built layer paints are checked against references
//! computed off the GPU in tests/text.rs, tests/image.rs and tests/shape.rs,
//! so an added layer that draws the built frame byte for byte draws the
//! reference too.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use serde_json::{json, Value};

use ghee::engine::{probe_dri_devices, Feed, Given, Refused, Renderer};
use ghee::layer::Address;
use ghee::scene::{ArraySize, Scene};

/// A Renderer opens a GL device. A test that opens one holds this.
static LOCK: Mutex<()> = Mutex::new(());

/// A w x h RGBA renderer on the first GL device that opens. Its atlases cache
/// under the temp directory.
fn gpu(w: usize, h: usize) -> (MutexGuard<'static, ()>, Renderer) {
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var(
        "XDG_CACHE_HOME",
        std::env::temp_dir().join("ghee-grow-cache"),
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

/// A directory of one test's own, which holds the fixture font as sans.ttf,
/// and pic.png and dot.png. The drop removes it.
struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("ghee-grow-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        fs::copy(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tools/fixtures/fonts/DejaVuSans-ASCII.ttf"
            ),
            d.join("sans.ttf"),
        )
        .unwrap();
        write_png(
            &d,
            "pic.png",
            2,
            &[
                [255, 0, 0, 255],
                [0, 255, 0, 255],
                [0, 0, 255, 255],
                [255; 4],
            ],
        );
        write_png(&d, "dot.png", 1, &[[255, 255, 0, 255]]);
        Dir(d)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    /// `doc` with its files here.
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

const GREY: &str = "void main(){ gl_FragColor = vec4(0.25, 0.25, 0.25, 1.0); }";

/// A scene of two passes, `small`, sized, and `main`, which shows `small`,
/// each with its layers: a text layer, an image layer and a run of two
/// shapes in `main`, and the font asset sans and the image asset pic.
fn scene() -> Value {
    json!({
        "version": 1,
        "assets": {"sans": {"file": "sans.ttf"}, "pic": {"file": "pic.png"}},
        "passes": [
            {"name": "small", "size": [64, 48], "fragment": GREY,
             "layers": [{"name": "bar", "shape": "rect", "x": 4, "y": 4, "width": 20,
                         "height": 6, "color": "orange"}]},
            {"name": "main", "channels": [{"buffer": "small"}],
             "fragment": "void main(){ gl_FragColor = texture2D(iChannel0, uv) * 0.5; }",
             "layers": [
                {"name": "title", "text": "Hi", "font_family": "sans", "font_size": 20,
                 "x": 4, "y": 4, "glow": {}},
                {"name": "a", "shape": "rect", "x": 100, "y": 10, "width": 30, "height": 20,
                 "color": "navy"},
                {"name": "b", "shape": "line", "points": [[100, 50], [140, 70]],
                 "stroke_width": 3},
                {"name": "icon", "image": "pic", "x": 10, "y": 60, "width": 16},
             ]},
        ],
    })
}

/// `doc` with `layer` at `index` of pass `pass`.
fn with(doc: &Value, pass: &str, index: usize, layer: &Value) -> Value {
    let mut doc = doc.clone();
    for p in doc["passes"].as_array_mut().unwrap() {
        if p["name"] == pass {
            if p.get("layers").is_none() {
                p["layers"] = json!([]);
            }
            p["layers"]
                .as_array_mut()
                .unwrap()
                .insert(index, layer.clone());
        }
    }
    doc
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

/// The bytes of two frames that differ.
fn differ(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_added_to_a_running_scene_draws_as_a_scene_built_with_it() {
    // Each row adds a layer at an index of a pass, and the frame matches the
    // scene built with the layer there, byte for byte. A row's count is the
    // programs that the add links: a shape or a layer like one that the
    // scene holds compiles nothing.
    let dir = Dir::new("add");
    let doc = scene();
    let rows = [
        // A shape joins the run beside it of its blend, makes its own, or
        // splits a run of another blend.
        (
            "main",
            1,
            json!({"name": "s", "shape": "rect", "x": 60, "y": 60, "width": 8, "height": 8}),
            0,
        ),
        (
            "main",
            2,
            json!({"name": "s", "shape": "rect", "x": 110, "y": 52, "width": 20, "height": 16,
                   "color": "lime", "blend": "add"}),
            0,
        ),
        (
            "main",
            4,
            json!({"name": "s", "shape": "line", "points": [[0, 90], [160, 80]],
                   "stroke_width": 2, "color": "red"}),
            0,
        ),
        (
            "main",
            0,
            json!({"name": "s", "shape": "rect", "color": "transparent",
                   "border": {"width": 2, "color": "white"}}),
            0,
        ),
        (
            "small",
            1,
            json!({"name": "s", "shape": "rect", "x": 10, "y": 20, "width": 30,
                   "height": 4, "border_radius": 2}),
            0,
        ),
        // An image splits the run that it lands in, and loads its file.
        (
            "main",
            2,
            json!({"name": "s", "image": "dot.png", "x": 115, "y": 52, "width": 10}),
            1,
        ),
        (
            "main",
            4,
            json!({"name": "s", "image": "pic", "x": 40, "y": 70}),
            1,
        ),
        (
            "small",
            0,
            json!({"name": "s", "image": "pic", "x": 40, "y": 30, "width": 16}),
            0,
        ),
        // A text layer builds its own shaders, glyphs and error boxes, and
        // its atlas, or shares the shaders of a layer with its settings, its
        // size aside. Error boxes take no setting but a text's motion, so
        // these share the title's.
        (
            "main",
            4,
            json!({"name": "s", "text": "Ok", "font_family": "sans", "font_size": 30,
                   "x": 60, "y": 40, "color": "yellow"}),
            1,
        ),
        (
            "main",
            1,
            json!({"name": "s", "text": "Ho", "font_family": "sans", "font_size": 20,
                   "x": 60, "y": 40, "glow": {"color": "red"}}),
            0,
        ),
        (
            "main",
            2,
            json!({"name": "s", "text": "Mm", "font_family": "sans", "font_size": 20,
                   "x": 104, "y": 46, "glow": {}}),
            0,
        ),
        (
            "main",
            0,
            json!({"name": "s", "text": "Ab", "font_family": "sans", "font_size": 24,
                   "x": 60, "y": 70, "render": "slug", "text_shadow": {}}),
            1,
        ),
        (
            "small",
            1,
            json!({"name": "s", "text": "go", "font_family": "sans", "font_size": 14,
                   "x": 30, "y": 2}),
            1,
        ),
    ];
    let (_g, mut r) = gpu(160, 96);
    for (pass, index, layer, programs) in rows {
        let row = format!("{pass} {index} {layer}");
        let built = render(&mut r, &dir.load(&with(&doc, pass, index, &layer)));
        render(&mut r, &dir.load(&doc));
        let compiled = r.compiled();
        r.add_layer(pass, index, &layer).unwrap();
        assert_eq!(r.compiled(), compiled + programs, "{row}: programs");
        assert_eq!(differ(&again(&mut r), &built), 0, "{row}");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn an_added_layer_leaves_the_running_state_of_the_scene_alone() {
    // A scrolling layer runs from time 0, a pass reads an array fed once,
    // and a shape is added at 0.5 s, away from both: every frame matches
    // the frame of the same scene that holds the shape from the start,
    // which a setup at 0.5 s would start over.
    let dir = Dir::new("state");
    let doc = json!({
        "version": 1,
        "assets": {"sans": {"file": "sans.ttf"}},
        "passes": [{"name": "main", "arrays": [{"name": "level", "size": 1}],
                    "fragment": "void main(){ gl_FragColor = vec4(level_at(0), 0.0, 0.0, 1.0); }",
                    "layers": [{"name": "news", "text": "scrolling words", "font_family": "sans",
                                "font_size": 16, "y": 4, "scroll": {"speed": 3}}]}],
    });
    let shape = json!({"name": "s", "shape": "rect", "x": 10, "y": 60, "width": 20, "height": 10});
    let (_g, mut r) = gpu(160, 96);
    let feed = |r: &mut Renderer, t: f64, fed: bool| {
        r.set_time(Some(t));
        let mut provider = |_: &str, _: ArraySize| match fed {
            true => Feed::Data(vec![0.5]),
            false => Feed::Unchanged,
        };
        r.render(Some(&mut provider)).unwrap().to_vec()
    };
    let times = [0.0, 0.25, 0.5, 0.75, 1.0];
    assert_eq!(
        r.setup(&dir.load(&with(&doc, "main", 1, &shape))).unwrap(),
        ""
    );
    let want: Vec<Vec<u8>> = times
        .iter()
        .enumerate()
        .map(|(i, t)| feed(&mut r, *t, i == 0))
        .collect();
    assert_eq!(r.setup(&dir.load(&doc)).unwrap(), "");
    for (i, t) in times.iter().enumerate() {
        if *t == 0.5 {
            let compiled = r.compiled();
            r.add_layer("main", 1, &shape).unwrap();
            assert_eq!(r.compiled(), compiled, "no program compiles");
        }
        let got = feed(&mut r, *t, i == 0);
        if *t >= 0.5 {
            assert_eq!(differ(&got, &want[i]), 0, "at {t} s");
        }
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn layers_whose_shaders_come_out_the_same_share_one_program() {
    // Two text layers and two image layers with the same settings, each
    // pair with its own words, place and colour, compile the programs of
    // one layer of each.
    let dir = Dir::new("share");
    let text = |name: &str, x: i32, words: &str| {
        json!({"name": name, "text": words, "font_family": "sans", "font_size": 20,
               "x": x, "y": 4, "glow": {}})
    };
    let image =
        |name: &str, y: i32| json!({"name": name, "image": "pic", "x": 4, "y": y, "width": 8});
    let doc = |layers: Value| {
        json!({"version": 1, "assets": {"sans": {"file": "sans.ttf"}, "pic": {"file": "pic.png"}},
               "passes": [{"name": "main", "fragment": GREY, "layers": layers}]})
    };
    let (_g, mut r) = gpu(160, 96);
    render(&mut r, &dir.load(&doc(json!([]))));
    let empty = r.compiled();
    render(
        &mut r,
        &dir.load(&doc(json!([text("a", 4, "Hi"), image("i", 50)]))),
    );
    let one = r.compiled() - empty;
    render(&mut r, &dir.load(&doc(json!([]))));
    let empty = r.compiled();
    let both = json!([
        text("a", 4, "Hi"),
        image("i", 50),
        text("b", 60, "Ho"),
        image("j", 70)
    ]);
    render(&mut r, &dir.load(&doc(both)));
    assert_eq!(r.compiled() - empty, one);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_live_image_takes_a_file_and_keeps_it_once_loaded() {
    // pic.png's first texel is red, and dot.png is yellow. The file stays
    // with the renderer after it leaves the disk.
    let dir = Dir::new("file");
    let doc = json!({
        "version": 1,
        "assets": {"pic": {"file": "pic.png"}},
        "passes": [{"name": "main", "fragment": GREY,
                    "layers": [{"name": "icon", "image": "pic", "x": 0, "y": 0, "width": 8}]}],
    });
    let at = Address {
        layer: "icon",
        effect: None,
        field: "image",
    };
    let (_g, mut r) = gpu(32, 32);
    let pic = render(&mut r, &dir.load(&doc));
    let mut dotted = doc.clone();
    dotted["passes"][0]["layers"][0]["image"] = json!("dot.png");
    let dot = render(&mut r, &dir.load(&dotted));
    assert_ne!(dot, pic);
    render(&mut r, &dir.load(&doc));
    r.set_layer(&at, Given::Json(&json!("dot.png"))).unwrap();
    assert_eq!(differ(&again(&mut r), &dot), 0);
    r.set_layer(&at, Given::Text("pic")).unwrap();
    assert_eq!(differ(&again(&mut r), &pic), 0);
    fs::remove_file(dir.path().join("dot.png")).unwrap();
    r.set_layer(&at, Given::Text("dot.png")).unwrap();
    assert_eq!(differ(&again(&mut r), &dot), 0);
    let e = r.set_layer(&at, Given::Text("nope.png")).unwrap_err();
    assert!(e.downcast_ref::<Refused>().is_some(), "{e}");
    assert_eq!(
        e.to_string(),
        format!(
            "layer 'icon' image 'nope.png' ({}/nope.png): No such file or directory, and \
             $GHEE_ASSETS names no directory",
            dir.path().display()
        )
    );
    assert_eq!(differ(&again(&mut r), &dot), 0);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_that_the_scene_refuses_is_refused_whole_and_changes_nothing() {
    let dir = Dir::new("refused");
    let doc = scene();
    let rows = [
        (
            "nope",
            0,
            json!({"name": "s", "shape": "rect"}),
            "the scene has no pass 'nope'",
        ),
        (
            "main",
            5,
            json!({"name": "s", "shape": "rect"}),
            "pass 'main' holds 4 layers, so a layer goes in at 0 to 4, got 5",
        ),
        (
            "main",
            0,
            json!({"shape": "rect"}),
            "the layer added at 0 to pass 'main' has no 'name'",
        ),
        (
            "main",
            0,
            json!({"name": "small", "shape": "rect"}),
            "pass 'small' and layer 'small' share a name",
        ),
        (
            "main",
            0,
            json!({"name": "s", "shape": "rect", "font_size": 3}),
            "layer 's' takes no 'font_size', which sits on a text layer",
        ),
        (
            "main",
            0,
            json!({"name": "s", "image": "nope"}),
            "layer 's' image 'nope' is no asset of the scene and no .png or .jpg file",
        ),
    ];
    let (_g, mut r) = gpu(160, 96);
    let before = render(&mut r, &dir.load(&doc));
    for (pass, index, layer, want) in rows {
        let e = r.add_layer(pass, index, &layer).unwrap_err();
        assert!(e.downcast_ref::<Refused>().is_some(), "{e}");
        assert_eq!(e.to_string(), want);
        assert_eq!(differ(&again(&mut r), &before), 0, "{want}");
    }
    // The scene takes the layer that it refused once it fits.
    r.add_layer(
        "main",
        0,
        &json!({"name": "s", "shape": "rect", "width": 4, "height": 4}),
    )
    .unwrap();
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_added_to_a_hidden_pass_joins_it_and_draws_nothing() {
    let dir = Dir::new("hidden");
    let mut doc = scene();
    doc["passes"][0]["visible"] = json!(false);
    let layer = json!({"name": "s", "shape": "rect", "width": 8, "height": 8});
    let (_g, mut r) = gpu(160, 96);
    let before = render(&mut r, &dir.load(&doc));
    r.add_layer("small", 0, &layer).unwrap();
    assert_eq!(differ(&again(&mut r), &before), 0);
    // The layer is the scene's: a live set reaches it.
    let at = Address {
        layer: "s",
        effect: None,
        field: "x",
    };
    r.set_layer(&at, Given::Json(&json!(3))).unwrap();
    assert_eq!(differ(&again(&mut r), &before), 0);
}

/// `doc` with layer `name` in its pass replaced by `layer`.
fn replaced(doc: &Value, name: &str, layer: &Value) -> Value {
    let mut doc = doc.clone();
    for p in doc["passes"].as_array_mut().unwrap() {
        for l in p["layers"].as_array_mut().into_iter().flatten() {
            if l["name"] == name {
                *l = layer.clone();
            }
        }
    }
    doc
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_built_again_in_its_place_draws_as_a_scene_built_with_it() {
    // Each row builds a layer of the running scene again from a new layer
    // of its name, of its kind or of another, and the frame matches the
    // scene built with the new layer there, byte for byte. Building the
    // first layer back then draws the first frame again.
    let dir = Dir::new("replace");
    let doc = scene();
    let rows = [
        json!({"name": "title", "text": "Ho", "font_family": "sans", "font_size": 28,
               "x": 4, "y": 4, "text_stroke": {}}),
        json!({"name": "title", "shape": "rect", "x": 4, "y": 4, "width": 30, "height": 10}),
        json!({"name": "a", "shape": "rect", "x": 100, "y": 10, "width": 30, "height": 20,
               "color": "navy", "blend": "add"}),
        json!({"name": "a", "text": "Ok", "font_family": "sans", "font_size": 20,
               "x": 100, "y": 10}),
        json!({"name": "b", "image": "dot.png", "x": 100, "y": 50, "width": 10}),
        json!({"name": "icon", "text": "Mm", "font_family": "sans", "font_size": 16,
               "x": 10, "y": 60, "render": "slug"}),
        json!({"name": "bar", "shape": "line", "points": [[4, 4], [40, 30]],
               "stroke_width": 2}),
    ];
    let (_g, mut r) = gpu(160, 96);
    let first = render(&mut r, &dir.load(&doc));
    for layer in rows {
        let name = layer["name"].as_str().unwrap();
        let built = render(&mut r, &dir.load(&replaced(&doc, name, &layer)));
        let was = render(&mut r, &dir.load(&doc));
        assert_eq!(differ(&was, &first), 0, "{layer}");
        r.replace_layer(&layer).unwrap();
        assert_eq!(differ(&again(&mut r), &built), 0, "{layer}");
        let old = doc["passes"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|p| p["layers"].as_array().unwrap())
            .find(|l| l["name"] == name)
            .unwrap()
            .clone();
        r.replace_layer(&old).unwrap();
        assert_eq!(differ(&again(&mut r), &first), 0, "{layer} back");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_built_again_carries_its_motion_on_and_leaves_the_rest_running() {
    // Two scrolling layers run from time 0 and a pass reads an array fed
    // once. At 0.5 s one layer takes a new size and a new colour, built
    // again: every frame from then matches the scene that holds the new
    // layer from the start, where a setup at 0.5 s would start over.
    let dir = Dir::new("motion");
    let news = |size: i32, color: &str| {
        json!({"name": "news", "text": "scrolling words", "font_family": "sans",
               "font_size": size, "y": 4, "color": color, "scroll": {"speed": 3}})
    };
    let doc = |news: Value| {
        json!({
            "version": 1,
            "assets": {"sans": {"file": "sans.ttf"}},
            "passes": [{"name": "main", "arrays": [{"name": "level", "size": 1}],
                        "fragment": "void main(){ gl_FragColor = vec4(level_at(0), 0.0, 0.0, 1.0); }",
                        "layers": [news,
                                   {"name": "other", "text": "more words", "font_family": "sans",
                                    "font_size": 14, "y": 60, "scroll": {"speed": 5}}]}],
        })
    };
    let (_g, mut r) = gpu(160, 96);
    let feed = |r: &mut Renderer, t: f64, fed: bool| {
        r.set_time(Some(t));
        let mut provider = |_: &str, _: ArraySize| match fed {
            true => Feed::Data(vec![0.5]),
            false => Feed::Unchanged,
        };
        r.render(Some(&mut provider)).unwrap().to_vec()
    };
    let times = [0.0, 0.25, 0.5, 0.75, 1.0];
    assert_eq!(r.setup(&dir.load(&doc(news(22, "lime")))).unwrap(), "");
    let want: Vec<Vec<u8>> = times
        .iter()
        .enumerate()
        .map(|(i, t)| feed(&mut r, *t, i == 0))
        .collect();
    assert_eq!(r.setup(&dir.load(&doc(news(16, "white")))).unwrap(), "");
    for (i, t) in times.iter().enumerate() {
        if *t == 0.5 {
            r.replace_layer(&news(22, "lime")).unwrap();
        }
        let got = feed(&mut r, *t, i == 0);
        if *t >= 0.5 {
            assert_eq!(differ(&got, &want[i]), 0, "at {t} s");
        }
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn an_update_sets_live_fields_live_and_builds_the_layer_again_for_the_rest() {
    // Each row gives a layer a new look, and the frame matches the scene
    // built with the look that the layer then has: an effect or a border
    // that the look leaves out stays, bypassed. A live row links no
    // program.
    let dir = Dir::new("update");
    let doc = scene();
    let rows = [
        (
            json!({"name": "title", "text": "Hey", "font_family": "sans", "font_size": 20,
                   "x": 10, "y": 8, "color": "orange", "glow": {"color": "red"}}),
            json!({"name": "title", "text": "Hey", "font_family": "sans", "font_size": 20,
                   "x": 10, "y": 8, "color": "orange", "glow": {"color": "red"}}),
            true,
        ),
        (
            json!({"name": "title", "text": "Hi", "font_family": "sans", "font_size": 20,
                   "x": 4, "y": 4}),
            json!({"name": "title", "text": "Hi", "font_family": "sans", "font_size": 20,
                   "x": 4, "y": 4, "glow": {"visible": false}}),
            true,
        ),
        (
            json!({"name": "title", "text": "Hi", "font_family": "sans", "font_size": 30,
                   "x": 4, "y": 4, "glow": {}}),
            json!({"name": "title", "text": "Hi", "font_family": "sans", "font_size": 30,
                   "x": 4, "y": 4, "glow": {}}),
            false,
        ),
        (
            json!({"name": "a", "shape": "rect", "x": 90, "y": 12, "width": 40,
                   "height": 16, "border": {"width": 2, "color": "white"}}),
            json!({"name": "a", "shape": "rect", "x": 90, "y": 12, "width": 40,
                   "height": 16, "border": {"width": 2, "color": "white"}}),
            true,
        ),
        (
            json!({"name": "a", "shape": "line", "points": [[90, 12], [130, 30]]}),
            json!({"name": "a", "shape": "line", "points": [[90, 12], [130, 30]]}),
            false,
        ),
        (
            json!({"name": "icon", "image": "dot.png", "x": 10, "y": 60, "width": 16}),
            json!({"name": "icon", "image": "dot.png", "x": 10, "y": 60, "width": 16}),
            true,
        ),
        (
            json!({"name": "icon", "image": "pic", "x": 10, "y": 60, "width": 16,
                   "blend": "screen"}),
            json!({"name": "icon", "image": "pic", "x": 10, "y": 60, "width": 16,
                   "blend": "screen"}),
            false,
        ),
    ];
    let (_g, mut r) = gpu(160, 96);
    for (look, has, live) in rows {
        let name = look["name"].as_str().unwrap();
        let built = render(&mut r, &dir.load(&replaced(&doc, name, &has)));
        render(&mut r, &dir.load(&doc));
        let compiled = r.compiled();
        assert!(r.update_layer(&look).unwrap(), "{look}");
        if live {
            assert_eq!(r.compiled(), compiled, "{look}: programs");
        }
        assert_eq!(differ(&again(&mut r), &built), 0, "{look}");
        // The same look again changes nothing.
        assert!(!r.update_layer(&look).unwrap(), "{look} again");
        assert_eq!(differ(&again(&mut r), &built), 0, "{look} again");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_look_that_the_scene_refuses_is_refused_whole_and_changes_nothing() {
    let dir = Dir::new("refused-look");
    let rows = [
        (
            json!({"name": "nope", "shape": "rect"}),
            "the scene has no layer 'nope'",
        ),
        (
            json!({"name": "a", "shape": "rect", "color": "navyy"}),
            "layer 'a' color needs a hex colour, a CSS colour name or three numbers [r, g, b], \
             got 'navyy'",
        ),
        (
            json!({"name": "icon", "image": "nope", "x": 10, "y": 60, "width": 16}),
            "layer 'icon' image 'nope' is no asset of the scene and no .png or .jpg file",
        ),
        (
            json!({"name": "title", "text": "Hi", "font_family": "serif"}),
            "layer 'title' font_family 'serif' is no asset of the scene, no .ttf or .otf file \
             and no font that ghee carries, 'DejaVu Sans' or 'Pixel Operator 8'",
        ),
    ];
    let (_g, mut r) = gpu(160, 96);
    let before = render(&mut r, &dir.load(&scene()));
    for (look, want) in rows {
        for e in [
            r.update_layer(&look).unwrap_err(),
            r.replace_layer(&look).unwrap_err(),
        ] {
            assert!(e.downcast_ref::<Refused>().is_some(), "{e}");
            assert_eq!(e.to_string(), want);
        }
        assert_eq!(differ(&again(&mut r), &before), 0, "{want}");
    }
}
