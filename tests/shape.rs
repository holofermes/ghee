//! Shape layers: the scene model, and the rects, lines and circles that they
//! paint on this host's GPU.
//!
//! The references below classify each pixel from the shape's own geometry,
//! as the distance from the pixel's centre to the outline: inside, outside,
//! or in the band of 1.5 px along an edge, which the antialiasing owns.

use std::sync::{Mutex, MutexGuard};

use serde_json::{json, Value};

use ghee::engine::{probe_dri_devices, Given, Renderer};
use ghee::layer::Address;
use ghee::scene::Scene;

/// A Renderer opens a GL device. A test that opens one holds this.
static LOCK: Mutex<()> = Mutex::new(());

/// A w x h RGBA renderer on the first GL device that opens.
fn gpu(w: usize, h: usize) -> (MutexGuard<'static, ()>, Renderer) {
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

/// The grey of the pass under the shapes.
const UNDER: f64 = 0.25;

const GREY: &str = "void main(){ gl_FragColor = vec4(0.25, 0.25, 0.25, 1.0); }";

/// A scene of pass `main` in UNDER with `layers`.
fn shape_scene(layers: Value) -> Value {
    json!({
        "version": 1,
        "passes": [{"name": "main", "fragment": GREY, "layers": layers}],
    })
}

fn loaded(doc: &Value) -> Scene {
    Scene::from_dict(doc, None, None).unwrap()
}

fn render(r: &mut Renderer, sc: &Scene) -> Vec<u8> {
    r.setup(sc).unwrap();
    again(r)
}

/// The next frame of the running scene.
fn again(r: &mut Renderer) -> Vec<u8> {
    r.set_time(Some(0.0));
    r.render(None).unwrap().to_vec()
}

/// The bytes of two frames that differ.
fn differ(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

fn scene_err(doc: &Value) -> String {
    Scene::from_dict(doc, None, None).err().unwrap().to_string()
}

#[test]
fn shape_layers_round_trip_and_leave_their_defaults_out() {
    let doc = shape_scene(json!([
        {"name": "a", "shape": "rect"},
        {"name": "b", "visible": false, "x": 4.0, "y": -2.5, "color": "transparent",
         "opacity": 0.5, "blend": "add", "shape": "rect", "width": 8.0, "height": 0.0,
         "border": {"width": 2.0, "color": "Lime", "visible": false}, "border_radius": 3.0},
        {"name": "c", "shape": "rect", "border": {}},
        {"name": "d", "color": [1.0, 0.0, 0.25], "shape": "line",
         "points": [[0.0, 16.0], [8.5, 4.0]], "stroke_width": 0.0},
        {"name": "e", "x": 2.0, "shape": "line", "points": [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]},
        {"name": "f", "y": 3.0, "opacity": 0.25, "shape": "circle", "radius": 5.5,
         "border": {"width": 2.0, "color": "navy"}},
        {"name": "g", "shape": "circle"},
    ]));
    let sc = loaded(&doc);
    assert_eq!(sc.to_dict(), doc);
    assert_eq!(loaded(&sc.to_dict()), sc);
    // Each key at its default.
    let defaults = shape_scene(json!([
        {"name": "a", "visible": true, "x": null, "color": "White", "opacity": 1,
         "blend": "normal", "shape": "rect", "width": null, "border_radius": 0,
         "border": {"width": 1, "color": "#ffffff", "visible": true}},
        {"name": "b", "shape": "line", "points": [[0, 0], [1, 1]], "stroke_width": 1},
        {"name": "c", "shape": "circle", "radius": null,
         "border": {"width": 1, "color": "white", "visible": true}},
    ]));
    let back = loaded(&defaults).to_dict();
    assert_eq!(
        back["passes"][0]["layers"],
        json!([{"name": "a", "shape": "rect", "border": {}},
               {"name": "b", "shape": "line", "points": [[0.0, 0.0], [1.0, 1.0]]},
               {"name": "c", "shape": "circle", "border": {}}])
    );
}

#[test]
fn a_shape_layer_declaration_names_what_is_wrong() {
    let s = |keys: Value| {
        let mut d = json!({"name": "s"});
        d.as_object_mut()
            .unwrap()
            .extend(keys.as_object().unwrap().clone());
        scene_err(&shape_scene(json!([d])))
    };
    let colour = "a hex colour, a CSS colour name or three numbers [r, g, b]";
    let pixels = "a number of pixels, 0 or more";
    let points = "needs two points or more, each [x, y]";
    let rows = [
        (
            s(json!({"shape": "oval"})),
            "layer 's' shape needs 'rect', 'line' or 'circle', got 'oval'".to_string(),
        ),
        (
            s(json!({"shape": 3})),
            "layer 's' shape needs 'rect', 'line' or 'circle', got 3".to_string(),
        ),
        (
            s(json!({"shape": "rect", "radius": 4})),
            "layer 's' takes no 'radius', which sits on a circle".to_string(),
        ),
        (
            s(json!({"shape": "circle", "width": 4})),
            "layer 's' takes no 'width', which sits on a rect".to_string(),
        ),
        (
            s(json!({"shape": "circle", "radius": -1})),
            format!("layer 's' radius needs {pixels}, got -1"),
        ),
        (
            s(json!({"shape": "rect", "points": [[0, 0], [1, 1]]})),
            "layer 's' takes no 'points', which sits on a line".to_string(),
        ),
        (
            s(json!({"shape": "line", "points": [[0, 0], [1, 1]], "border": {}})),
            "layer 's' takes no 'border', which sits on a rect".to_string(),
        ),
        (
            s(json!({"shape": "rect", "text": "I"})),
            "layer 's' takes no 'text', which sits on a text layer".to_string(),
        ),
        (
            s(json!({"shape": "rect", "image": "pic"})),
            "layer 's' takes no 'image', which sits on an image layer".to_string(),
        ),
        (
            s(json!({"shape": "rect", "font_size": 4})),
            "layer 's' takes no 'font_size', which sits on a text layer".to_string(),
        ),
        (
            s(json!({"shape": "line", "points": [[0, 0], [1, 1]], "stroke": 2})),
            "layer 's' takes no 'stroke', the closest key is 'stroke_width'".to_string(),
        ),
        (
            s(json!({"shape": "rect", "width": -1})),
            format!("layer 's' width needs {pixels}, got -1"),
        ),
        (
            s(json!({"shape": "rect", "border_radius": "3"})),
            format!("layer 's' border_radius needs {pixels}, got '3'"),
        ),
        (
            s(json!({"shape": "rect", "color": {"from": "red", "to": "blue"}})),
            format!("layer 's' color takes no gradient on a shape layer, only {colour}"),
        ),
        (
            s(json!({"shape": "rect", "color": "rde"})),
            format!("layer 's' color needs {colour}, got 'rde'"),
        ),
        (
            s(json!({"shape": "rect", "border": 1})),
            "layer 's' border needs an object, got 1".to_string(),
        ),
        (
            s(json!({"shape": "rect", "border": {"style": "solid"}})),
            "layer 's' border takes no 'style', only width, color and visible".to_string(),
        ),
        (
            s(json!({"shape": "rect", "border": {"width": -2}})),
            format!("layer 's' border width needs {pixels}, got -2"),
        ),
        (
            s(json!({"shape": "rect", "border": {"color": "rde"}})),
            format!("layer 's' border color needs {colour}, got 'rde'"),
        ),
        (
            s(json!({"shape": "rect", "border": {"visible": 0}})),
            "layer 's' border visible needs true or false, got 0".to_string(),
        ),
        (
            s(json!({"shape": "line"})),
            "layer 's' has no 'points' (the line's points, each [x, y])".to_string(),
        ),
        (
            s(json!({"shape": "line", "points": [[0, 0]]})),
            format!("layer 's' points {points}, got [[0, 0]]"),
        ),
        (
            s(json!({"shape": "line", "points": [[0, 0], [1]]})),
            format!("layer 's' points {points}, got [[0, 0], [1]]"),
        ),
        (
            s(json!({"shape": "line", "points": [[0, 0], [1, 1]], "stroke_width": -1})),
            format!("layer 's' stroke_width needs {pixels}, got -1"),
        ),
    ];
    for (got, want) in rows {
        assert_eq!(got, format!("scene: {want}"));
    }
}

/// Assert that pixel channel `got` lies within `tol` levels of `want`, 0 to
/// 1, else push the pixel to `off`.
fn check(
    off: &mut Vec<(usize, usize, usize, u8, f64)>,
    xy: (usize, usize),
    frame: &[u8],
    w: usize,
    want: [f64; 3],
    tol: f64,
) {
    for (k, c) in want.iter().enumerate() {
        let got = frame[(xy.1 * w + xy.0) * 4 + k];
        if (f64::from(got) - 255.0 * c).abs() > tol {
            off.push((xy.0, xy.1, k, got, (255.0 * c).round()));
        }
    }
}

/// The distance from `p` to a box of half size `h` around `c`, its corners
/// rounded by `r`, below 0 inside.
fn rounded_box(p: [f64; 2], c: [f64; 2], h: [f64; 2], r: f64) -> f64 {
    let q = [0, 1].map(|a| (p[a] - c[a]).abs() - h[a] + r);
    q[0].max(0.0).hypot(q[1].max(0.0)) + q[0].max(q[1]).min(0.0) - r
}

/// The distance from `p` to the segment from `a` to `b`.
fn segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (pa, ba) = ([p[0] - a[0], p[1] - a[1]], [b[0] - a[0], b[1] - a[1]]);
    let len2 = ba[0] * ba[0] + ba[1] * ba[1];
    let t = match len2 > 0.0 {
        true => ((pa[0] * ba[0] + pa[1] * ba[1]) / len2).clamp(0.0, 1.0),
        false => 0.0,
    };
    (pa[0] - ba[0] * t).hypot(pa[1] - ba[1] * t)
}

/// `top`, straight, at `alpha`, over `under`.
fn over(top: [f64; 3], alpha: f64, under: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|k| top[k] * alpha + under[k] * (1.0 - alpha))
}

/// The antialiasing owns the band of EDGE px either side of an edge.
const EDGE: f64 = 1.5;

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_rect_fills_its_box_and_draws_its_border_inside_its_edge() {
    // A 96x64 pass. Each row: the rect's keys, its box as x, y, width and
    // height, its corner radius, its border's width, and the fill and the
    // border, each straight with its alpha. CSS Color 4 gives orange as
    // #ffa500 and navy as #000080.
    let (w, h) = (96, 64);
    let orange = [1.0, 165.0 / 255.0, 0.0];
    let navy = [0.0, 0.0, 128.0 / 255.0];
    let white = [1.0; 3];
    type Row = (Value, [f64; 4], f64, f64, ([f64; 3], f64), ([f64; 3], f64));
    let rows: Vec<Row> = vec![
        (
            json!({"x": 10, "y": 8, "width": 40, "height": 30, "color": "orange"}),
            [10.0, 8.0, 40.0, 30.0],
            0.0,
            0.0,
            (orange, 1.0),
            (white, 0.0),
        ),
        (
            json!({"x": 10.25, "y": 8.5, "width": 40.5, "height": 30, "opacity": 0.5}),
            [10.25, 8.5, 40.5, 30.0],
            0.0,
            0.0,
            (white, 0.5),
            (white, 0.0),
        ),
        // A missing size is the pass's, and a missing axis centres.
        (
            json!({"y": 8, "height": 20, "color": "navy"}),
            [0.0, 8.0, 96.0, 20.0],
            0.0,
            0.0,
            (navy, 1.0),
            (white, 0.0),
        ),
        (
            json!({"width": 30, "height": 20}),
            [33.0, 22.0, 30.0, 20.0],
            0.0,
            0.0,
            (white, 1.0),
            (white, 0.0),
        ),
        (
            json!({"x": 10, "y": 8, "width": 40, "height": 30, "color": "navy",
                   "border": {"width": 6, "color": "orange"}}),
            [10.0, 8.0, 40.0, 30.0],
            0.0,
            6.0,
            (navy, 1.0),
            (orange, 1.0),
        ),
        // Under a transparent border the fill shows, as CSS paints a
        // background under its border.
        (
            json!({"x": 10, "y": 8, "width": 40, "height": 30, "color": "navy",
                   "border": {"width": 6, "color": "transparent"}}),
            [10.0, 8.0, 40.0, 30.0],
            0.0,
            6.0,
            (navy, 1.0),
            (white, 0.0),
        ),
        (
            json!({"x": 10, "y": 8, "width": 40, "height": 30, "color": "transparent",
                   "border": {"width": 5}}),
            [10.0, 8.0, 40.0, 30.0],
            0.0,
            5.0,
            (white, 0.0),
            (white, 1.0),
        ),
        (
            json!({"x": 10, "y": 8, "width": 40, "height": 30, "color": "navy",
                   "border": {"width": 6, "color": "orange", "visible": false}}),
            [10.0, 8.0, 40.0, 30.0],
            0.0,
            0.0,
            (navy, 1.0),
            (white, 0.0),
        ),
        (
            json!({"x": 10, "y": 8, "width": 50, "height": 40, "border_radius": 12,
                   "color": "orange", "border": {"width": 5, "color": "navy"}}),
            [10.0, 8.0, 50.0, 40.0],
            12.0,
            5.0,
            (orange, 1.0),
            (navy, 1.0),
        ),
        // A radius past half the shorter side makes round ends.
        (
            json!({"x": 20, "y": 10, "width": 40, "height": 40, "border_radius": 100}),
            [20.0, 10.0, 40.0, 40.0],
            20.0,
            0.0,
            (white, 1.0),
            (white, 0.0),
        ),
    ];
    let (_g, mut r) = gpu(w, h);
    for (keys, rect, radius, border, fill, ink) in rows {
        let mut layer = json!({"name": "box", "shape": "rect"});
        layer
            .as_object_mut()
            .unwrap()
            .extend(keys.as_object().unwrap().clone());
        let frame = render(&mut r, &loaded(&shape_scene(json!([layer]))));
        let half = [rect[2] / 2.0, rect[3] / 2.0];
        let c = [rect[0] + half[0], rect[1] + half[1]];
        let inner = [half[0] - border, half[1] - border];
        let (mut seen, mut off) = ([0; 3], Vec::new());
        for y in 0..h {
            for x in 0..w {
                let p = [x as f64 + 0.5, y as f64 + 0.5];
                let d = rounded_box(p, c, half, radius);
                let di = rounded_box(p, c, inner, (radius - border).max(0.0));
                if d.abs() < EDGE || (border > 0.0 && di.abs() < EDGE) {
                    continue;
                }
                let filled = over(fill.0, fill.1, [UNDER; 3]);
                let (want, at) = match (d < 0.0, border > 0.0 && di > 0.0) {
                    (false, _) => ([UNDER; 3], 0),
                    (true, false) => (filled, 1),
                    (true, true) => (over(ink.0, ink.1, filled), 2),
                };
                seen[at] += 1;
                check(&mut off, (x, y), &frame, w, want, 2.0);
            }
        }
        assert!(seen[0] > 100 && seen[1] > 100, "{keys}: {seen:?}");
        assert!(border == 0.0 || seen[2] > 100, "{keys}: {seen:?}");
        assert!(
            off.is_empty(),
            "{keys}: {} off, (x, y, channel, got, want) {:?}",
            off.len(),
            &off[..off.len().min(8)]
        );
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_circle_draws_as_the_square_rect_of_its_box_rounded_by_its_radius() {
    // A 96x64 pass. Each row: a circle's keys, and the keys of the rect of
    // side 2 * radius around the circle's x, y, its centre as SVG's cx, cy,
    // whose border_radius is the radius. A circle with no radius is the
    // largest that fits the pass.
    let rows = [
        (
            json!({"x": 30, "y": 28, "radius": 20, "color": "orange"}),
            json!({"x": 10, "y": 8, "width": 40, "height": 40, "border_radius": 20,
                   "color": "orange"}),
        ),
        (
            json!({"x": 23, "y": 21.25, "radius": 12.75, "opacity": 0.5,
                   "border": {"width": 3, "color": "navy"}}),
            json!({"x": 10.25, "y": 8.5, "width": 25.5, "height": 25.5, "border_radius": 12.75,
                   "opacity": 0.5, "border": {"width": 3, "color": "navy"}}),
        ),
        (
            json!({"y": 19, "radius": 15, "color": "navy", "blend": "add"}),
            json!({"y": 4, "width": 30, "height": 30, "border_radius": 15, "color": "navy",
                   "blend": "add"}),
        ),
        (
            json!({"color": "transparent", "border": {"width": 2}}),
            json!({"width": 64, "height": 64, "border_radius": 32, "color": "transparent",
                   "border": {"width": 2}}),
        ),
    ];
    let (_g, mut r) = gpu(96, 64);
    for (circle, rect) in rows {
        let layer = |shape: &str, keys: &Value| {
            let mut l = json!({"name": "s", "shape": shape});
            l.as_object_mut()
                .unwrap()
                .extend(keys.as_object().unwrap().clone());
            loaded(&shape_scene(json!([l])))
        };
        let got = render(&mut r, &layer("circle", &circle));
        let want = render(&mut r, &layer("rect", &rect));
        assert_eq!(differ(&got, &want), 0, "{circle}");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_hex_colours_alpha_draws_as_the_opacity_of_its_colour() {
    let (_g, mut r) = gpu(32, 16);
    let rect = |keys: Value| {
        let mut l = json!({"name": "s", "shape": "rect", "width": 20, "height": 10});
        l.as_object_mut()
            .unwrap()
            .extend(keys.as_object().unwrap().clone());
        loaded(&shape_scene(json!([l])))
    };
    for (hex, rgb, alpha) in [
        ("#f008", "#f00", 0x88),
        ("#ff000080", "#ff0000", 0x80),
        ("#0f0f", "lime", 0xff),
    ] {
        let opacity = f64::from(alpha) / 255.0;
        let got = render(&mut r, &rect(json!({"color": hex})));
        let want = render(&mut r, &rect(json!({"color": rgb, "opacity": opacity})));
        assert_eq!(differ(&got, &want), 0, "{hex}");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_line_runs_through_its_points_and_draws_each_joint_once() {
    // Each row: the line's keys and its points as pixel centres. Away from
    // the edges every pixel takes the line's colour at its opacity, the
    // joints too, or the grey.
    let (w, h) = (96, 64);
    let red = [1.0, 0.0, 0.0];
    let rows = [
        (
            json!({"points": [[8, 8], [80, 50]], "stroke_width": 8, "color": "red"}),
            vec![[8.5, 8.5], [80.5, 50.5]],
            1.0,
        ),
        (
            json!({"points": [[8, 50], [30, 8], [50, 50], [88, 30]], "stroke_width": 9,
                   "color": "red", "opacity": 0.5}),
            vec![[8.5, 50.5], [30.5, 8.5], [50.5, 50.5], [88.5, 30.5]],
            0.5,
        ),
        // x and y shift every point, and a point may repeat.
        (
            json!({"points": [[0, 0], [30, 20], [30, 20], [60, 0]], "x": 10, "y": 20,
                   "stroke_width": 7, "color": "red", "opacity": 0.5}),
            vec![[10.5, 20.5], [40.5, 40.5], [40.5, 40.5], [70.5, 20.5]],
            0.5,
        ),
    ];
    let (_g, mut r) = gpu(w, h);
    for (keys, pts, opacity) in rows {
        let mut layer = json!({"name": "l", "shape": "line"});
        layer
            .as_object_mut()
            .unwrap()
            .extend(keys.as_object().unwrap().clone());
        let half = keys["stroke_width"].as_f64().unwrap() / 2.0;
        let frame = render(&mut r, &loaded(&shape_scene(json!([layer]))));
        let (mut seen, mut off) = ([0; 2], Vec::new());
        for y in 0..h {
            for x in 0..w {
                let p = [x as f64 + 0.5, y as f64 + 0.5];
                let d = pts
                    .windows(2)
                    .map(|s| segment(p, s[0], s[1]))
                    .fold(f64::MAX, f64::min)
                    - half;
                if d.abs() < EDGE {
                    continue;
                }
                let (want, at) = match d < 0.0 {
                    true => (over(red, opacity, [UNDER; 3]), 1),
                    false => ([UNDER; 3], 0),
                };
                seen[at] += 1;
                check(&mut off, (x, y), &frame, w, want, 2.0);
            }
        }
        assert!(seen[0] > 100 && seen[1] > 100, "{keys}: {seen:?}");
        assert!(
            off.is_empty(),
            "{keys}: {} off, (x, y, channel, got, want) {:?}",
            off.len(),
            &off[..off.len().min(8)]
        );
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn shapes_at_whole_pixels_light_whole_pixels_as_pillow_does() {
    // A point is a pixel, so a line of stroke_width 1 lights the pixels from
    // its first point to its last, and a rect at whole pixels with a border
    // of 1 lights its edge pixels. Every other pixel stays grey.
    let (w, h) = (32, 24);
    let doc = shape_scene(json!([
        {"name": "row", "shape": "line", "points": [[2, 3], [20, 3]]},
        {"name": "column", "shape": "line", "points": [[26, 2], [26, 21]], "color": "lime"},
        {"name": "box", "shape": "rect", "x": 2, "y": 8, "width": 12, "height": 10,
         "color": "blue", "border": {"color": "red"}},
    ]));
    let (_g, mut r) = gpu(w, h);
    let frame = render(&mut r, &loaded(&doc));
    let mut off = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let want = if y == 3 && (2..=20).contains(&x) {
                [1.0; 3]
            } else if x == 26 && (2..=21).contains(&y) {
                [0.0, 1.0, 0.0]
            } else if (2..14).contains(&x) && (8..18).contains(&y) {
                match x == 2 || x == 13 || y == 8 || y == 17 {
                    true => [1.0, 0.0, 0.0],
                    false => [0.0, 0.0, 1.0],
                }
            } else {
                [UNDER; 3]
            };
            check(&mut off, (x, y), &frame, w, want, 1.0);
        }
    }
    assert!(
        off.is_empty(),
        "{} off, (x, y, channel, got, want) {:?}",
        off.len(),
        &off[..off.len().min(8)]
    );
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn shape_layers_paint_in_the_order_of_the_layers() {
    // Four rects overlap at one pixel, each half opaque, in two runs split
    // by a blend and an image layer between them: the layers' order sets
    // the colour there, which `over` composes from the grey up.
    let (w, h) = (16, 16);
    let dir = std::env::temp_dir().join(format!("ghee-shape-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = std::fs::File::create(dir.join("dot.png")).unwrap();
    let mut enc = png::Encoder::new(file, 1, 1);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().unwrap();
    wr.write_image_data(&[0, 255, 0, 255]).unwrap();
    drop(wr);
    let rect = |name: &str, color: &str, blend: &str| {
        json!({"name": name, "shape": "rect", "x": 4, "y": 4, "width": 8, "height": 8,
               "color": color, "opacity": 0.5, "blend": blend})
    };
    let doc = shape_scene(json!([
        rect("a", "red", "normal"),
        rect("b", "blue", "normal"),
        {"name": "dot", "image": "dot.png", "x": 4, "y": 4, "width": 8, "opacity": 0.5},
        rect("c", "white", "normal"),
        rect("d", "red", "add"),
        rect("e", "blue", "normal"),
    ]));
    let mut sc = loaded(&doc);
    sc.load_assets(&dir.to_string_lossy()).unwrap();
    let (_g, mut r) = gpu(w, h);
    let frame = render(&mut r, &sc);
    std::fs::remove_dir_all(&dir).unwrap();
    let mut want = [UNDER; 3];
    for c in [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ] {
        want = over(c, 0.5, want);
    }
    want[0] = (want[0] + 0.5).min(1.0);
    want = over([0.0, 0.0, 1.0], 0.5, want);
    let mut off = Vec::new();
    check(&mut off, (8, 8), &frame, w, want, 2.0);
    assert!(off.is_empty(), "{off:?}");
}

/// `doc` with the field at `at` written as `value`, as a scene declares it.
fn written(doc: &Value, at: &Address, value: &Value) -> Value {
    let mut doc = doc.clone();
    for pass in doc["passes"].as_array_mut().unwrap() {
        for l in pass["layers"].as_array_mut().unwrap() {
            if l["name"] != at.layer {
                continue;
            }
            match at.effect {
                Some(e) => l[e][at.field] = value.clone(),
                None if value.is_null() => {
                    l.as_object_mut().unwrap().remove(at.field);
                }
                None => l[at.field] = value.clone(),
            }
        }
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

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn every_field_of_a_shape_changes_live_and_compiles_nothing() {
    // Two passes of different sizes, each with shapes, which one program
    // draws. Each row sets a field live, and the frame matches the scene
    // built with that value, byte for byte, with no program compiled.
    let layers = json!([
        {"name": "r", "shape": "rect", "x": 4, "y": 4, "width": 30, "height": 20,
         "color": "orange", "border": {"width": 3, "color": "navy"}, "border_radius": 4},
        {"name": "l", "shape": "line", "points": [[40, 4], [60, 30], [70, 10]],
         "stroke_width": 3, "color": "lime"},
        {"name": "c", "shape": "circle", "x": 66, "y": 36, "radius": 10, "color": "red",
         "border": {"width": 2}},
    ]);
    let doc = json!({
        "version": 1,
        "passes": [
            {"name": "small", "size": [48, 32], "fragment": GREY,
             "layers": [{"name": "dot", "shape": "rect", "x": 2, "y": 2, "width": 6,
                         "height": 6}]},
            {"name": "main", "fragment": "void main(){ gl_FragColor = texture2D(iChannel0, \
              uv); }", "channels": [{"buffer": "small"}], "layers": layers},
        ],
    });
    let rows = [
        (at("r", None, "x"), json!(10)),
        (at("r", None, "y"), json!(null)),
        (at("r", None, "width"), json!(null)),
        (at("r", None, "height"), json!(12.5)),
        (at("r", None, "color"), json!("transparent")),
        (at("r", None, "opacity"), json!(0.5)),
        (at("r", None, "visible"), json!(false)),
        (at("r", None, "border_radius"), json!(10)),
        (at("r", Some("border"), "width"), json!(6)),
        (at("r", Some("border"), "color"), json!("Red")),
        (at("r", Some("border"), "visible"), json!(false)),
        (at("l", None, "points"), json!([[0, 30], [90, 2]])),
        (at("l", None, "stroke_width"), json!(6)),
        (at("l", None, "x"), json!(5)),
        (at("l", None, "color"), json!([0, 0, 1])),
        (at("c", None, "radius"), json!(14.5)),
        (at("c", None, "radius"), json!(null)),
        (at("c", Some("border"), "width"), json!(4)),
        (at("dot", Some("border"), "color"), json!("blue")),
    ];
    let (_g, mut r) = gpu(96, 64);
    let base = loaded(&doc);
    let one = render(&mut r, &base);
    let compiled = r.compiled();
    // Each pass compiles its own program again, and the shapes share the one
    // that the last setup compiled.
    assert_eq!(differ(&render(&mut r, &base), &one), 0);
    assert_eq!(r.compiled(), compiled + 2);
    for (at, value) in rows {
        let row = format!("{at:?} = {value}");
        let before = render(&mut r, &base);
        let compiled = r.compiled();
        r.set_layer(&at, Given::Json(&value)).unwrap();
        let live = again(&mut r);
        assert_eq!(r.compiled(), compiled, "{row}: no program compiles");
        let built = loaded(&written(&doc, &at, &value));
        assert_ne!(differ(&live, &before), 0, "{row}: the frame changes");
        assert_eq!(differ(&live, &render(&mut r, &built)), 0, "{row}");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn the_shapes_of_a_renderer_compile_one_program_once() {
    // The same scene with and without its shapes, set up after a scene that
    // compiled no shape: the shapes add one program, however many runs and
    // passes they span, and the renderer compiles it once.
    let pass =
        |name: &str, layers: Value| json!({"name": name, "fragment": GREY, "layers": layers});
    let shapes = json!([
        {"name": "a", "shape": "rect", "width": 4, "height": 4},
        {"name": "b", "shape": "rect", "width": 8, "height": 8, "blend": "add"},
        {"name": "c", "shape": "line", "points": [[0, 0], [9, 9]]},
    ]);
    let more = json!([{"name": "d", "shape": "rect", "width": 2, "height": 2}]);
    let without = json!({"version": 1, "passes": [pass("one", json!([])), pass("two", json!([]))]});
    let with = json!({"version": 1, "passes": [pass("one", shapes), pass("two", more)]});
    let (_g, mut r) = gpu(32, 32);
    render(&mut r, &loaded(&without));
    let before = r.compiled();
    render(&mut r, &loaded(&with));
    assert_eq!(r.compiled(), before + 2 + 1);
    // The renderer keeps the program through a scene with no shapes.
    render(&mut r, &loaded(&without));
    let before = r.compiled();
    render(&mut r, &loaded(&with));
    assert_eq!(r.compiled(), before + 2);
}
