//! Image layers: the scene model, and the pictures that they paint on this
//! host's GPU.
//!
//! The reference below paints each layer's picture on the CPU, from the
//! layer's keys and the texels of a fixture image, over the pass's colour.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use serde_json::{json, Value};

use ghee::engine::{probe_dri_devices, Given, Renderer};
use ghee::layer::Address;
use ghee::scene::Scene;

/// A Renderer opens a GL device, and load_assets reads $GHEE_ASSETS. A test
/// that does either holds this, so no variable changes under a device or a
/// lookup.
static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A w x h RGBA renderer on the first GL device that opens.
fn gpu(w: usize, h: usize) -> (MutexGuard<'static, ()>, Renderer) {
    let guard = lock();
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

/// A directory of one test's own under the temp directory. The drop removes
/// it, with everything that the test wrote in it.
struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("ghee-image-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        Dir(d)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The fixture image, 2x2 RGBA with straight alpha, rows from the top: red
/// then green, blue then white at half alpha.
const PICTURE: [[u8; 4]; 4] = [
    [255, 0, 0, 255],
    [0, 255, 0, 255],
    [0, 0, 255, 255],
    [255, 255, 255, 128],
];

/// The colour of the pass under the layers.
const UNDER: [f64; 3] = [0.25, 0.25, 0.25];

const GREY: &str = "void main(){ gl_FragColor = vec4(0.25, 0.25, 0.25, 1.0); }";

/// Write PICTURE to `dir` as pic.png, through the png crate's encoder.
fn write_picture(dir: &Path) {
    let file = fs::File::create(dir.join("pic.png")).unwrap();
    let mut enc = png::Encoder::new(file, 2, 2);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().unwrap();
    w.write_image_data(PICTURE.as_flattened()).unwrap();
}

/// A scene of the image asset `pic`, pic.png, and pass `main` in UNDER with
/// `layers`.
fn image_scene(layers: Value) -> Value {
    json!({
        "version": 1,
        "assets": {"pic": {"file": "pic.png"}},
        "passes": [{"name": "main", "code": GREY, "layers": layers}],
    })
}

fn scene_err(doc: &Value) -> String {
    Scene::from_dict(doc, None, None).err().unwrap().to_string()
}

/// `doc` loaded from `dir`. The caller holds LOCK.
fn loaded(doc: &Value, dir: &Path) -> Scene {
    let mut sc = Scene::from_dict(doc, None, None).unwrap();
    sc.load_assets(&dir.to_string_lossy()).unwrap();
    sc
}

fn render(r: &mut Renderer, sc: &Scene) -> Vec<u8> {
    assert_eq!(r.setup(sc).unwrap(), "");
    r.set_time(Some(0.0));
    r.render(None).unwrap().to_vec()
}

#[test]
fn image_layers_round_trip_and_leave_their_defaults_out() {
    let doc = image_scene(json!([
        {"name": "a", "image": "pic"},
        {"name": "b", "visible": false, "x": 4.0, "y": -2.5, "color": "#ff8000",
         "opacity": 0.5, "blend": "multiply", "image": "pic.png", "width": 8.0, "height": 3.0},
        {"name": "c", "color": [1.0, 1.0, 1.0], "image": "pic", "height": 2.0},
    ]));
    let back = Scene::from_dict(&doc, None, None).unwrap().to_dict();
    assert_eq!(back["passes"][0]["layers"], doc["passes"][0]["layers"]);
    let defaults = image_scene(json!([{"name": "a", "visible": true, "x": null,
        "color": null, "opacity": 1, "blend": "normal", "image": "pic", "width": null}]));
    let back = Scene::from_dict(&defaults, None, None).unwrap().to_dict();
    assert_eq!(
        back["passes"][0]["layers"],
        json!([{"name": "a", "image": "pic"}])
    );
}

#[test]
fn an_image_layer_declaration_names_what_it_takes() {
    let i = |keys: Value| {
        let mut d = json!({"name": "i", "image": "pic"});
        d.as_object_mut()
            .unwrap()
            .extend(keys.as_object().unwrap().clone());
        let mut doc = image_scene(json!([d]));
        doc["assets"]["sans"] = json!({"file": "sans.ttf"});
        scene_err(&doc)
    };
    let image = "image needs the name of an image asset or a .png or .jpg file";
    let cases = [
        (i(json!({"image": 3})), format!("layer 'i' {image}, got 3")),
        (
            i(json!({"image": ""})),
            format!("layer 'i' {image}, got ''"),
        ),
        (
            i(json!({"image": "wifi"})),
            "layer 'i' image 'wifi' is no asset of the scene and no .png or .jpg file".to_string(),
        ),
        (
            i(json!({"image": "sans"})),
            "layer 'i' image 'sans' is the font 'sans.ttf', an image is a .png or .jpg file"
                .to_string(),
        ),
        (
            i(json!({"width": 0})),
            "layer 'i' width needs a number of pixels above 0, got 0".to_string(),
        ),
        (
            i(json!({"height": "4"})),
            "layer 'i' height needs a number of pixels above 0, got '4'".to_string(),
        ),
        (
            i(json!({"color": {"from": "#000000", "to": "#ffffff"}})),
            "layer 'i' color takes no gradient on an image layer, only '#rrggbb' or [r, g, b]"
                .to_string(),
        ),
        (
            i(json!({"color": "red"})),
            "layer 'i' color needs '#rrggbb' or [r, g, b], got 'red'".to_string(),
        ),
        (
            i(json!({"glow": {}})),
            "layer 'i' takes no 'glow', which sits on a text layer".to_string(),
        ),
        (
            i(json!({"font_size": 12})),
            "layer 'i' takes no 'font_size', which sits on a text layer".to_string(),
        ),
        (
            scene_err(&image_scene(
                json!([{"name": "t", "text": "I", "width": 4}]),
            )),
            "layer 't' takes no 'width', which sits on an image layer".to_string(),
        ),
    ];
    for (got, want) in cases {
        assert_eq!(got, format!("scene: {want}"));
    }
}

#[test]
fn an_image_layer_takes_a_scene_wide_name_and_gives_a_pass_none() {
    // The asset `pic` and a layer `pic` share the scene's set of names.
    let clash = image_scene(json!([{"name": "pic", "image": "pic"}]));
    let e = scene_err(&clash);
    assert!(
        e.ends_with("layer 'pic' and asset 'pic' share a name"),
        "{e}"
    );
    // A pass's own names may end the way a text layer's GLSL names do,
    // since an image layer declares none.
    let mut doc = image_scene(json!([{"name": "icon", "image": "pic"}]));
    doc["passes"][0]["params"] = json!({
        "icon_size": {"type": "float", "value": 1.0},
        "icon_cover": {"type": "float", "value": 1.0},
    });
    assert!(Scene::from_dict(&doc, None, None).is_ok());
}

/// The frame that PICTURE paints over UNDER on a w x h pass, spanning
/// `rect`, [x, y, width, height] in pass pixels from the top left, tinted by
/// `tint` at `opacity`, as RGB in 0 to 1, rows from the top. None paints
/// nothing.
fn painted(
    w: usize,
    h: usize,
    rect: Option<[f64; 4]>,
    tint: [f64; 3],
    opacity: f64,
) -> Vec<[f64; 3]> {
    let mut out = vec![UNDER; w * h];
    let Some([rx, ry, rw, rh]) = rect else {
        return out;
    };
    for y in 0..h {
        for x in 0..w {
            // The texel under the pixel centre, nearest.
            let (u, v) = ((x as f64 + 0.5 - rx) / rw, (y as f64 + 0.5 - ry) / rh);
            if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                continue;
            }
            let t = PICTURE[(v * 2.0) as usize * 2 + (u * 2.0) as usize];
            let a = f64::from(t[3]) / 255.0 * opacity;
            for c in 0..3 {
                out[y * w + x][c] = UNDER[c] * (1.0 - a) + f64::from(t[c]) / 255.0 * tint[c] * a;
            }
        }
    }
    out
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn an_image_layer_paints_its_picture_where_its_keys_place_it() {
    // Every rect scales the 2x2 picture by whole pixels, so no pixel centre
    // falls on the edge between two texels.
    let (w, h) = (24, 16);
    let dir = Dir::new("paint");
    write_picture(dir.path());
    let plain = [1.0; 3];
    let orange = [1.0, 128.0 / 255.0, 0.0];
    let rows = [
        (
            json!({"image": "pic", "x": 2, "y": 3}),
            Some([2.0, 3.0, 2.0, 2.0]),
            plain,
            1.0,
        ),
        (
            json!({"image": "pic", "x": 1, "y": 1, "width": 6}),
            Some([1.0, 1.0, 6.0, 6.0]),
            plain,
            1.0,
        ),
        (
            json!({"image": "pic", "x": 1, "y": 2, "height": 4}),
            Some([1.0, 2.0, 4.0, 4.0]),
            plain,
            1.0,
        ),
        (
            json!({"image": "pic", "x": 1, "y": 1, "width": 8, "height": 4}),
            Some([1.0, 1.0, 8.0, 4.0]),
            plain,
            1.0,
        ),
        (
            json!({"image": "pic", "width": 4}),
            Some([10.0, 6.0, 4.0, 4.0]),
            plain,
            1.0,
        ),
        (
            json!({"image": "pic", "y": 2, "width": 4}),
            Some([10.0, 2.0, 4.0, 4.0]),
            plain,
            1.0,
        ),
        (
            json!({"image": "pic", "x": -1, "width": 4}),
            Some([-1.0, 6.0, 4.0, 4.0]),
            plain,
            1.0,
        ),
        (
            json!({"image": "pic", "x": 3, "y": 2, "width": 6, "color": "#ff8000",
                   "opacity": 0.5}),
            Some([3.0, 2.0, 6.0, 6.0]),
            orange,
            0.5,
        ),
        (
            json!({"image": "pic.png", "x": 3, "y": 2, "width": 6}),
            Some([3.0, 2.0, 6.0, 6.0]),
            plain,
            1.0,
        ),
        (
            json!({"image": "pic", "x": 3, "y": 2, "visible": false}),
            None,
            plain,
            1.0,
        ),
    ];
    let (_g, mut r) = gpu(w, h);
    for (keys, rect, tint, opacity) in rows {
        let mut layer = json!({"name": "icon"});
        layer
            .as_object_mut()
            .unwrap()
            .extend(keys.as_object().unwrap().clone());
        let doc = image_scene(json!([layer]));
        let got = render(&mut r, &loaded(&doc, dir.path()));
        assert_painted(&got, &painted(w, h, rect, tint, opacity), w, &keys);
    }
}

/// Assert that `got`, an RGBA frame w pixels wide, holds `want` within 2
/// levels. `what` names the case.
fn assert_painted(got: &[u8], want: &[[f64; 3]], w: usize, what: &Value) {
    for (i, px) in want.iter().enumerate() {
        for c in 0..3 {
            let level = f64::from(got[i * 4 + c]);
            assert!(
                (level - px[c] * 255.0).abs() <= 2.0,
                "{what}: pixel ({}, {}) channel {c} is {level}, the reference {:.1}",
                i % w,
                i / w,
                px[c] * 255.0
            );
        }
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn an_image_layer_takes_another_image_asset_live_at_that_images_size() {
    // big.png is PICTURE at twice its size, each texel 2x2, so it paints
    // what PICTURE paints over a rect twice the size.
    let (w, h) = (24, 16);
    let dir = Dir::new("live");
    write_picture(dir.path());
    let big: Vec<[u8; 4]> = (0..16)
        .map(|i| PICTURE[(i / 8) * 2 + (i % 4) / 2])
        .collect();
    let file = fs::File::create(dir.path().join("big.png")).unwrap();
    let mut enc = png::Encoder::new(file, 4, 4);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut out = enc.write_header().unwrap();
    out.write_image_data(big.as_flattened()).unwrap();
    out.finish().unwrap();
    let orange = [1.0, 128.0 / 255.0, 0.0];
    let rows = [
        (
            json!({"image": "pic", "x": 2, "y": 3, "color": "#ff8000", "opacity": 0.5}),
            [2.0, 3.0, 4.0, 4.0],
            orange,
            0.5,
        ),
        (
            json!({"image": "pic.png", "x": 2, "y": 3}),
            [2.0, 3.0, 4.0, 4.0],
            [1.0; 3],
            1.0,
        ),
        (
            json!({"image": "pic", "x": 1, "y": 1, "width": 6}),
            [1.0, 1.0, 6.0, 6.0],
            [1.0; 3],
            1.0,
        ),
    ];
    let (_g, mut r) = gpu(w, h);
    let at = Address {
        layer: "icon",
        effect: None,
        field: "image",
    };
    for (keys, rect, tint, opacity) in rows {
        let mut layer = json!({"name": "icon"});
        layer
            .as_object_mut()
            .unwrap()
            .extend(keys.as_object().unwrap().clone());
        let mut doc = image_scene(json!([layer]));
        doc["assets"]["big"] = json!({"file": "big.png"});
        render(&mut r, &loaded(&doc, dir.path()));
        r.set_layer(&at, Given::Json(&json!("big"))).unwrap();
        r.set_time(Some(0.0));
        let got = r.render(None).unwrap().to_vec();
        assert_painted(&got, &painted(w, h, Some(rect), tint, opacity), w, &keys);
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn an_image_layer_past_the_texture_units_fails_naming_the_layer() {
    // main's channels and array take every unit, and its image layer needs
    // one more for the picture.
    let dir = Dir::new("units");
    write_picture(dir.path());
    let (_g, mut r) = gpu(16, 8);
    let plain = image_scene(json!([{"name": "icon", "image": "pic"}]));
    let before = render(&mut r, &loaded(&plain, dir.path()));
    let units = r.max_texture_units() as usize;
    let channels: Vec<Value> = (0..units - 1).map(|_| json!({"buffer": "a"})).collect();
    let mut doc = plain.clone();
    doc["passes"] = json!([
        {"name": "a", "code": GREY},
        {"name": "main", "channels": channels, "arrays": [{"name": "v", "size": 4}],
         "code": "void main(){ gl_FragColor = vec4(v_at(0)); }",
         "layers": [{"name": "icon", "image": "pic"}]},
    ]);
    let past = match unsafe { ghee::ffi::ghee_gl_texture_units(r.handle()) } > 16 {
        true => "a pass binds at most",
        false => "this GPU has",
    };
    assert_eq!(
        r.setup(&loaded(&doc, dir.path())).unwrap(),
        format!(
            "layer 'icon' needs {} texture units, {past} {units}",
            units + 1
        )
    );
    r.set_time(Some(0.0));
    assert!(r.render(None).unwrap() == before.as_slice());
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn an_image_layer_of_a_scene_built_without_a_directory_names_its_file() {
    let doc = json!({"version": 1, "passes": [{"name": "main", "code": GREY,
        "layers": [{"name": "icon", "image": "pic.png"}]}]});
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    let (_g, mut r) = gpu(8, 8);
    assert_eq!(
        r.setup(&sc).unwrap(),
        "layer 'icon': 'pic.png' was never loaded, the scene was built without a directory"
    );
}
