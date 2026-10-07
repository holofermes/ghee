//! A drawing's calls handed to a running scene with no look written, against
//! the same looks written out and handed to update_layer(), and against the
//! live sets that a host and the console make, on this host's GPU.

use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use serde_json::{json, Map, Value};

use ghee::draw::{Arg, Call, CallKind};
use ghee::engine::{probe_dri_devices, Given, Renderer};
use ghee::layer::Address;
use ghee::scene::Scene;

/// A Renderer opens a GL device. A test that opens one holds this.
static LOCK: Mutex<()> = Mutex::new(());

/// A w x h RGBA renderer on the first GL device that opens. Its atlases cache
/// under the temp directory.
fn gpu(w: usize, h: usize) -> (MutexGuard<'static, ()>, Renderer) {
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var(
        "XDG_CACHE_HOME",
        std::env::temp_dir().join("ghee-draw-cache"),
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

/// A directory of one test's own that holds the fixture font as sans.ttf
/// and a 1x1 yellow dot.png. The drop removes it.
struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("ghee-draw-{}-{tag}", std::process::id()));
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
        let file = fs::File::create(d.join("dot.png")).unwrap();
        let mut enc = png::Encoder::new(file, 1, 1);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut out = enc.write_header().unwrap();
        out.write_image_data(&[255, 255, 0, 255]).unwrap();
        drop(out);
        Dir(d)
    }

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

/// One pass whose layers a drawing drew: a pixel-font text, a DejaVu text
/// with a glow, a rect with a border, a line and an image file.
fn scene() -> Value {
    json!({
        "version": 1,
        "assets": {"sans": {"file": "sans.ttf"}},
        "passes": [{"name": "main",
                    "fragment": "void main(){ gl_FragColor = vec4(0.1, 0.1, 0.3, 1.0); }",
                    "layers": [
            {"name": "clock", "text": "12:00", "font_family": "pixel", "font_size": 8,
             "x": 2, "y": 2},
            {"name": "title", "text": "Hi", "font_family": "sans", "font_size": 18, "x": 4,
             "y": 14, "karaoke": {"sung": 1}, "glow": {"strength": 0.5}},
            {"name": "box", "shape": "rect", "x": 60, "y": 4, "width": 20, "height": 10,
             "color": "navy", "border": {"width": 1, "color": "white"}, "border_radius": 2},
            {"name": "bar", "shape": "rect", "x": 60, "y": 20, "width": 8, "height": 4},
            {"name": "trace", "shape": "line", "points": [[0, 40], [30, 30], [60, 44]],
             "stroke_width": 2, "color": "lime"},
            {"name": "dot", "image": "dot.png", "x": 90, "y": 30, "width": 6}
        ]}],
    })
}

fn n(f: f64) -> Arg {
    Arg::Num(f)
}

fn s(t: &str) -> Arg {
    Arg::Str(t.to_string())
}

fn list(items: &[Arg]) -> Arg {
    Arg::List(items.to_vec())
}

fn dict(pairs: &[(&str, Arg)]) -> Arg {
    Arg::Map(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    )
}

/// A call of `kind` on layer `layer`, its keywords from `given`, by name,
/// the last of a name standing, and None for the rest.
fn call(kind: CallKind, layer: &str, given: &[(&str, Arg)]) -> Call {
    let args = kind
        .keys()
        .iter()
        .map(|k| {
            given
                .iter()
                .rev()
                .find(|(g, _)| g == k)
                .map_or(Arg::None, |(_, v)| v.clone())
        })
        .collect();
    Call {
        kind,
        layer: layer.to_string(),
        args,
    }
}

fn json_of(a: &Arg) -> Value {
    match a {
        Arg::None => Value::Null,
        Arg::Bool(b) => Value::from(*b),
        Arg::Num(f) => Value::from(*f),
        Arg::Str(t) => Value::from(t.clone()),
        Arg::List(items) => Value::from(items.iter().map(json_of).collect::<Vec<_>>()),
        Arg::Map(pairs) => {
            let m: Map<String, Value> =
                pairs.iter().map(|(k, v)| (k.clone(), json_of(v))).collect();
            Value::Object(m)
        }
    }
}

/// The look of `c` written as a scene declares it, as a host writes it for
/// update_layer(): each keyword that the call states, and its shape.
fn look_of(c: &Call) -> Value {
    let mut look = Map::new();
    look.insert("name".to_string(), Value::from(c.layer.clone()));
    match c.kind {
        CallKind::Rect => look.insert("shape".to_string(), json!("rect")),
        CallKind::Line => look.insert("shape".to_string(), json!("line")),
        _ => None,
    };
    for (k, a) in c.kind.keys().iter().zip(&c.args) {
        if *a != Arg::None {
            look.insert(k.to_string(), json_of(a));
        }
    }
    Value::Object(look)
}

/// The next frame of the running scene at time 0, and the scene as it
/// stands.
fn state(r: &mut Renderer) -> (Vec<u8>, Value) {
    r.set_time(Some(0.0));
    let frame = r.render(None).unwrap().to_vec();
    (frame, r.with_scene(Scene::to_dict).unwrap())
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_call_gives_its_layer_the_look_that_update_layer_gives_its_look_written_out() {
    // Each row hands the running scene its calls in turn, and the frame and
    // the scene match those of a scene handed the same looks written out
    // through update_layer(). The last of a row's calls takes its layer
    // with no look written when `typed` says so, and is left for
    // update_layer() otherwise: a call that builds its layer again, gives
    // an image another file, draws another kind of layer, names no layer,
    // or holds a value that the scene refuses.
    use CallKind::{Image, Line, Rect, Text};
    let dir = Dir::new("calls");
    let clock = |more: &[(&str, Arg)]| {
        let mut given = vec![
            ("text", s("12:00")),
            ("font_family", s("pixel")),
            ("font_size", n(8.0)),
            ("x", n(2.0)),
            ("y", n(2.0)),
        ];
        given.extend_from_slice(more);
        call(Text, "clock", &given)
    };
    let title = |more: &[(&str, Arg)]| {
        let mut given = vec![
            ("text", s("Hi")),
            ("font_family", s("sans")),
            ("font_size", n(18.0)),
            ("x", n(4.0)),
            ("y", n(14.0)),
        ];
        given.extend_from_slice(more);
        call(Text, "title", &given)
    };
    let gradient = dict(&[("from", s("red")), ("to", list(&[n(0.0), n(0.5), n(1.0)]))]);
    let rows: Vec<(Vec<Call>, bool)> = vec![
        (vec![clock(&[("text", s("12:01"))])], true),
        (vec![clock(&[("x", Arg::None), ("opacity", n(0.5))])], true),
        (vec![clock(&[("color", s("Orange"))])], true),
        (
            vec![clock(&[
                ("color", s("#20c040")),
                ("visible", Arg::Bool(false)),
            ])],
            true,
        ),
        (vec![clock(&[("color", gradient.clone())])], true),
        (vec![clock(&[("blink", dict(&[("rate", n(2.0))]))])], false),
        (vec![clock(&[("font_size", n(16.0))])], false),
        (vec![clock(&[("opacity", n(1.5))])], false),
        (vec![clock(&[("text_align", s("middle"))])], false),
        (vec![clock(&[("visible", n(1.0))])], false),
        (
            vec![title(&[
                (
                    "karaoke",
                    dict(&[("sung", n(2.0)), ("color", list(&[n(1.0), n(0.0), n(0.0)]))]),
                ),
                ("glow", dict(&[("strength", n(0.9)), ("color", s("red"))])),
            ])],
            true,
        ),
        (
            vec![title(&[("glow", dict(&[("strength", n(0.5))]))])],
            true,
        ),
        (vec![title(&[("glow", dict(&[("blur", n(0.5))]))])], false),
        (vec![title(&[("glow", dict(&[("sung", n(1.0))]))])], false),
        (
            vec![
                title(&[("karaoke", dict(&[("sung", n(2.0))]))]),
                title(&[("text", s("Hey")), ("y", Arg::None)]),
            ],
            true,
        ),
        (
            vec![call(
                Rect,
                "box",
                &[
                    ("x", n(50.0)),
                    ("y", n(4.0)),
                    ("width", n(30.0)),
                    ("height", n(12.0)),
                    ("border_radius", n(6.0)),
                ],
            )],
            true,
        ),
        (
            vec![call(
                Rect,
                "box",
                &[
                    ("width", n(20.0)),
                    ("height", n(10.0)),
                    ("color", s("transparent")),
                    ("border", dict(&[("width", n(2.0)), ("color", s("red"))])),
                ],
            )],
            true,
        ),
        (
            vec![call(
                Rect,
                "box",
                &[("border", dict(&[("radius", n(2.0))]))],
            )],
            false,
        ),
        (vec![call(Rect, "box", &[("blend", s("add"))])], false),
        (
            vec![
                call(
                    Rect,
                    "bar",
                    &[
                        ("x", n(60.0)),
                        ("y", n(20.0)),
                        ("width", n(12.0)),
                        ("height", n(4.0)),
                    ],
                ),
                call(
                    Rect,
                    "bar",
                    &[
                        ("x", n(61.0)),
                        ("y", n(20.0)),
                        ("height", n(4.0)),
                        ("opacity", n(0.25)),
                    ],
                ),
            ],
            true,
        ),
        (
            vec![call(
                Line,
                "bar",
                &[(
                    "points",
                    list(&[list(&[n(0.0), n(0.0)]), list(&[n(5.0), n(5.0)])]),
                )],
            )],
            false,
        ),
        (
            vec![call(
                Line,
                "trace",
                &[
                    (
                        "points",
                        list(&[list(&[n(0.0), n(40.0)]), list(&[n(40.0), n(20.0)])]),
                    ),
                    ("color", list(&[n(1.0), n(0.0), n(1.0)])),
                    ("x", n(3.0)),
                ],
            )],
            true,
        ),
        (
            vec![call(
                Line,
                "trace",
                &[("points", list(&[list(&[n(0.0), n(40.0)])]))],
            )],
            false,
        ),
        (
            vec![call(
                Image,
                "dot",
                &[
                    ("image", s("dot.png")),
                    ("x", n(80.0)),
                    ("y", n(30.0)),
                    ("width", n(6.0)),
                    ("color", s("red")),
                    ("opacity", n(0.5)),
                ],
            )],
            true,
        ),
        (
            vec![call(
                Image,
                "dot",
                &[("image", s("dot.png")), ("width", n(8.0))],
            )],
            false,
        ),
        (
            vec![call(
                Image,
                "dot",
                &[("image", s("sans")), ("width", n(6.0))],
            )],
            false,
        ),
        (vec![call(Rect, "clock", &[("x", n(1.0))])], false),
        (vec![call(Rect, "nothing", &[("x", n(1.0))])], false),
    ];
    let (_g, mut r) = gpu(128, 48);
    let sc = dir.load(&scene());
    for (calls, typed) in rows {
        assert_eq!(r.setup(&sc).unwrap(), "");
        let written: Vec<_> = calls
            .iter()
            .map(|c| r.update_layer(&look_of(c)).map_err(|e| e.to_string()))
            .collect();
        let want = state(&mut r);
        assert_eq!(r.setup(&sc).unwrap(), "");
        for (k, c) in calls.iter().enumerate() {
            let taken = r.update_calls(std::slice::from_ref(c)).unwrap();
            if k + 1 == calls.len() {
                assert_eq!(taken.left.is_empty(), typed, "{c:?}");
            }
            if !taken.left.is_empty() {
                let got = r.update_layer(&look_of(c)).map_err(|e| e.to_string());
                assert_eq!(got, written[k], "{c:?}");
            }
        }
        let got = state(&mut r);
        assert!(got.0 == want.0, "{calls:?}: frames differ");
        assert_eq!(got.1, want.1, "{calls:?}");
    }
}

/// A live set: the layer, the effect, the field and its value.
type Set = (&'static str, Option<&'static str>, &'static str, Value);

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_call_that_leaves_a_keyword_out_gives_the_default_of_a_live_set() {
    // Each row gives a layer a field, then takes it back to its default,
    // once through a live set, as a host's layers and the console set it,
    // and once through a call that leaves the keyword out. The frames and
    // the scenes match.
    use CallKind::{Image, Line, Rect, Text};
    let dir = Dir::new("defaults");
    let text = |more: &[(&str, Arg)]| {
        let mut given = vec![
            ("text", s("Hi")),
            ("font_family", s("sans")),
            ("font_size", n(18.0)),
            ("glow", dict(&[("strength", n(0.5))])),
        ];
        given.extend_from_slice(more);
        call(Text, "title", &given)
    };
    let rect = |more: &[(&str, Arg)]| {
        let mut given = vec![("width", n(20.0)), ("height", n(10.0))];
        given.extend_from_slice(more);
        call(Rect, "box", &given)
    };
    // (a call that gives the field, the live set back, the call without it)
    let rows: Vec<(Call, Set, Call)> = vec![
        (
            text(&[("x", n(4.0)), ("y", n(9.0))]),
            ("title", None, "x", json!(null)),
            text(&[("y", n(9.0))]),
        ),
        (
            text(&[("color", s("red"))]),
            ("title", None, "color", json!(null)),
            text(&[]),
        ),
        (
            text(&[("opacity", n(0.5))]),
            ("title", None, "opacity", json!(1.0)),
            text(&[]),
        ),
        (
            text(&[("karaoke", dict(&[("sung", n(2.0))]))]),
            ("title", Some("karaoke"), "sung", json!(0.0)),
            text(&[("karaoke", dict(&[]))]),
        ),
        (
            text(&[("glow", dict(&[("strength", n(0.9))]))]),
            ("title", Some("glow"), "strength", json!(0.8)),
            text(&[("glow", dict(&[]))]),
        ),
        (
            rect(&[("x", n(4.0))]),
            ("box", None, "x", json!(null)),
            rect(&[]),
        ),
        (
            rect(&[("color", s("red"))]),
            ("box", None, "color", json!(null)),
            rect(&[]),
        ),
        (
            rect(&[("border_radius", n(3.0))]),
            ("box", None, "border_radius", json!(0.0)),
            rect(&[]),
        ),
        (
            call(Rect, "box", &[("height", n(10.0)), ("width", n(30.0))]),
            ("box", None, "width", json!(null)),
            call(Rect, "box", &[("height", n(10.0))]),
        ),
        (
            rect(&[("border", dict(&[("width", n(3.0)), ("color", s("red"))]))]),
            ("box", Some("border"), "width", json!(1.0)),
            rect(&[("border", dict(&[("color", s("red"))]))]),
        ),
        (
            call(
                Line,
                "trace",
                &[
                    (
                        "points",
                        list(&[list(&[n(0.0), n(40.0)]), list(&[n(9.0), n(9.0)])]),
                    ),
                    ("stroke_width", n(3.0)),
                ],
            ),
            ("trace", None, "stroke_width", json!(1.0)),
            call(
                Line,
                "trace",
                &[(
                    "points",
                    list(&[list(&[n(0.0), n(40.0)]), list(&[n(9.0), n(9.0)])]),
                )],
            ),
        ),
        (
            call(
                Image,
                "dot",
                &[
                    ("image", s("dot.png")),
                    ("width", n(6.0)),
                    ("color", s("red")),
                ],
            ),
            ("dot", None, "color", json!(null)),
            call(Image, "dot", &[("image", s("dot.png")), ("width", n(6.0))]),
        ),
    ];
    let (_g, mut r) = gpu(128, 48);
    let sc = dir.load(&scene());
    for (given, (layer, effect, field, value), without) in rows {
        assert_eq!(r.setup(&sc).unwrap(), "");
        assert!(r
            .update_calls(std::slice::from_ref(&given))
            .unwrap()
            .left
            .is_empty());
        let at = Address {
            layer,
            effect,
            field,
        };
        r.set_layer(&at, Given::Json(&value)).unwrap();
        let set = state(&mut r);
        assert_eq!(r.setup(&sc).unwrap(), "");
        assert!(r
            .update_calls(std::slice::from_ref(&given))
            .unwrap()
            .left
            .is_empty());
        let taken = r.update_calls(std::slice::from_ref(&without)).unwrap();
        assert!(taken.left.is_empty(), "{without:?}");
        let called = state(&mut r);
        assert!(called.0 == set.0, "{without:?}: frames differ");
        assert_eq!(called.1, set.1, "{without:?}");
    }
}
