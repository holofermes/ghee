//! Shape layers: a rect or a line that a pass paints over its buffer, in the
//! order of its `layers`.
//!
//! Every shape draws through one program, [`program`], which takes no GLSL
//! from the layer: a shape is quads whose vertices carry all of it, and a
//! change to a live field of a shape uploads new quads and compiles nothing.
//! Consecutive shape layers of a pass with one blend draw as one batch, a
//! run, in the order of the layers, and `crate::run` takes the slug texts
//! beside them into the run too.
//!
//! A rect is a box of `width` by `height` pass pixels at `x`, `y`, its
//! corners rounded by `border_radius`, filled with `color`, and its `border`
//! lies inside the box. A line runs through its `points`, each a pixel as
//! luma and Pillow count them, so a point's centre is the pixel's centre, and
//! `x`, `y` shift all of them. The fragment shader antialiases every edge
//! from the distance to the shape.

use anyhow::{bail, Result};
use serde_json::{Map, Value};

use crate::draw::{self, Arg};
use crate::engine::Blend;
use crate::layer::{
    every_dict, given, number, object, spoken, Every, Form, Live, Rgb, COLOUR_FORMS, IMAGE_KIND,
    TEXT_KIND,
};
use crate::scene::{known_keys, py_repr};

/// The keys of a rect past those of every layer.
pub(crate) const RECT_KEYS: [&str; 5] = ["shape", "width", "height", "border", "border_radius"];

/// The keys of a line past those of every layer.
pub(crate) const LINE_KEYS: [&str; 3] = ["shape", "points", "stroke_width"];

/// A rect's keys, and the kind of layer in an error.
pub(crate) const RECT_KIND: (&[&str], &str) = (&RECT_KEYS, "a rect");

/// A line's keys, and the kind of layer in an error.
pub(crate) const LINE_KIND: (&[&str], &str) = (&LINE_KEYS, "a line");

/// The keys of a border.
const BORDER_KEYS: [&str; 3] = ["width", "color", "visible"];

/// A border's width when it names none, in pixels.
const BORDER_WIDTH: f64 = 1.0;

/// A line's thickness when it names none, in pixels.
pub(crate) const STROKE_WIDTH: f64 = 1.0;

/// A rect's corner radius when it names none, in pixels.
pub(crate) const BORDER_RADIUS: f64 = 0.0;

/// The vec4 attributes of one vertex of a shape's quad.
pub(crate) const ATTRIBUTES: usize = 4;

/// The floats of one quad: four vertices of ATTRIBUTES vec4s.
const QUAD: usize = 4 * 4 * ATTRIBUTES;

/// Positions and sizes go to the GPU on a grid of 1/GRID pixel, which the
/// fragment shader finds again after the interpolation. Two quads that meet
/// then compute one distance, bit for bit.
const GRID: f64 = 64.0;

/// The border of a rect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Border {
    /// Pixels inside the box's edge.
    pub width: f64,
    pub color: Rgb,
    /// False bypasses the border, which keeps its values.
    pub visible: bool,
}

/// What a shape layer draws.
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Rect {
        /// Pass pixels. None is the pass's own.
        width: Option<f64>,
        height: Option<f64>,
        border: Option<Border>,
        border_radius: f64,
    },
    Line {
        /// Pixels, each [x, y], two or more.
        points: Vec<[f64; 2]>,
        stroke_width: f64,
    },
}

/// A shape layer of a pass.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeLayer {
    pub name: String,
    /// False hides the layer.
    pub visible: bool,
    /// A rect's top left, centred on a missing axis, or a line's shift, 0 on
    /// a missing axis.
    pub x: Option<f64>,
    pub y: Option<f64>,
    /// The fill of a rect, the colour of a line.
    pub color: Rgb,
    pub opacity: f64,
    pub blend: Blend,
    pub shape: Shape,
}

/// A number of pixels, 0 or more.
fn pixels(f: f64) -> bool {
    f >= 0.0
}

const PIXELS: &str = "a number of pixels, 0 or more";

impl ShapeLayer {
    /// `at` names the layer by its place in an error, as "the 2nd layer of
    /// pass 'main'", until its name is known.
    pub fn from_dict(at: &str, d: &Value) -> Result<ShapeLayer> {
        let o = object(at, d)?;
        let line = match given(o, "shape") {
            Some(Value::String(s)) if s == "rect" => false,
            Some(Value::String(s)) if s == "line" => true,
            v => {
                let what = match o.get("name") {
                    Some(Value::String(n)) => format!("layer '{n}'"),
                    _ => at.to_string(),
                };
                bail!(
                    "{what} shape needs 'rect' or 'line', got {}",
                    py_repr(v.unwrap_or(&Value::Null))
                );
            }
        };
        let e = match line {
            true => Every::from_dict(at, o, &LINE_KEYS, &[RECT_KIND, TEXT_KIND, IMAGE_KIND])?,
            false => Every::from_dict(at, o, &RECT_KEYS, &[LINE_KIND, TEXT_KIND, IMAGE_KIND])?,
        };
        let what = e.what.as_str();
        let color = match given(o, "color") {
            None => Rgb::WHITE,
            Some(v) => match Rgb::from_json(v) {
                Some(c) => c,
                None if v.is_object() => {
                    bail!("{what} color takes no gradient on a shape layer, only {COLOUR_FORMS}")
                }
                None => bail!("{what} color needs {COLOUR_FORMS}, got {}", py_repr(v)),
            },
        };
        let px = |k: &str| number(o, what, k, pixels, PIXELS);
        let shape = match line {
            true => Shape::Line {
                points: points(what, given(o, "points"))?,
                stroke_width: px("stroke_width")?.unwrap_or(STROKE_WIDTH),
            },
            false => Shape::Rect {
                width: px("width")?,
                height: px("height")?,
                border: match given(o, "border") {
                    None => None,
                    Some(b) => Some(border(&format!("{what} border"), b)?),
                },
                border_radius: px("border_radius")?.unwrap_or(BORDER_RADIUS),
            },
        };
        Ok(ShapeLayer {
            name: e.name,
            visible: e.visible,
            x: e.x,
            y: e.y,
            color,
            opacity: e.opacity,
            blend: e.blend,
            shape,
        })
    }

    /// The layer as a scene holds it, a key at its default left out.
    pub fn to_dict(&self) -> Value {
        let white = self.color.alpha == 1.0 && self.color.rgb == [1.0; 3];
        let mut d = every_dict(
            &self.name,
            self.visible,
            self.x,
            self.y,
            (!white).then(|| self.color.to_json()),
            self.opacity,
            self.blend,
        );
        match &self.shape {
            Shape::Rect {
                width,
                height,
                border,
                border_radius,
            } => {
                d.insert("shape".to_string(), Value::from("rect"));
                for (k, v) in [("width", width), ("height", height)] {
                    if let Some(v) = v {
                        d.insert(k.to_string(), Value::from(*v));
                    }
                }
                if let Some(b) = border {
                    d.insert("border".to_string(), border_dict(b));
                }
                if *border_radius != 0.0 {
                    d.insert("border_radius".to_string(), Value::from(*border_radius));
                }
            }
            Shape::Line {
                points,
                stroke_width,
            } => {
                d.insert("shape".to_string(), Value::from("line"));
                let pts: Vec<Value> = points.iter().map(|p| Value::from(p.to_vec())).collect();
                d.insert("points".to_string(), Value::from(pts));
                if *stroke_width != STROKE_WIDTH {
                    d.insert("stroke_width".to_string(), Value::from(*stroke_width));
                }
            }
        }
        Value::Object(d)
    }

    /// The layer named `name` that a drawing's call with `args` gives
    /// `old`: a line's in the order of draw::LINE_ARGS, a rect's in the order
    /// of draw::RECT_ARGS, as draw::layer_of() makes it. None for another
    /// shape than `old`'s.
    pub(crate) fn of_args(
        name: &str,
        line: bool,
        args: &[Arg],
        old: &ShapeLayer,
    ) -> Option<ShapeLayer> {
        let px = |a: &Arg| draw::number(a, pixels);
        let colour = |a: &Arg| match a {
            Arg::None => Some(Rgb::WHITE),
            c => draw::rgb(c),
        };
        let (e, color, shape) = match (line, &old.shape) {
            (true, Shape::Line { .. }) => {
                let [points, x, y, color, opacity, blend, visible, stroke_width] = args else {
                    return None;
                };
                let Arg::List(items) = points else {
                    return None;
                };
                let pair = |p: &Arg| match p {
                    Arg::List(xy) => match xy.as_slice() {
                        [Arg::Num(x), Arg::Num(y)] if x.is_finite() && y.is_finite() => {
                            Some([*x, *y])
                        }
                        _ => None,
                    },
                    _ => None,
                };
                if items.len() < 2 {
                    return None;
                }
                let shape = Shape::Line {
                    points: items.iter().map(pair).collect::<Option<Vec<_>>>()?,
                    stroke_width: px(stroke_width)?.unwrap_or(STROKE_WIDTH),
                };
                let e = draw::Every::of_args(visible, x, y, opacity, blend)?;
                (e, colour(color)?, shape)
            }
            (false, Shape::Rect { border: kept, .. }) => {
                let [x, y, width, height, color, opacity, blend, visible, border, border_radius] =
                    args
                else {
                    return None;
                };
                let border = match border {
                    Arg::None => kept.map(|b| Border {
                        visible: false,
                        ..b
                    }),
                    b => {
                        let m = draw::map(b, &BORDER_KEYS)?;
                        Some(Border {
                            width: px(draw::get(m, "width"))?.unwrap_or(BORDER_WIDTH),
                            color: colour(draw::get(m, "color"))?,
                            visible: draw::flag(draw::get(m, "visible"), true)?,
                        })
                    }
                };
                let shape = Shape::Rect {
                    width: px(width)?,
                    height: px(height)?,
                    border,
                    border_radius: px(border_radius)?.unwrap_or(BORDER_RADIUS),
                };
                let e = draw::Every::of_args(visible, x, y, opacity, blend)?;
                (e, colour(color)?, shape)
            }
            _ => return None,
        };
        Some(ShapeLayer {
            name: name.to_string(),
            visible: e.visible,
            x: e.x,
            y: e.y,
            color,
            opacity: e.opacity,
            blend: e.blend,
            shape,
        })
    }

    /// True when `other` is the same shape, a rect or a line.
    pub(crate) fn same_shape(&self, other: &ShapeLayer) -> bool {
        self.kind() == other.kind()
    }

    /// The name of the shape, as the scene writes it.
    fn kind(&self) -> &'static str {
        match self.shape {
            Shape::Rect { .. } => "rect",
            Shape::Line { .. } => "line",
        }
    }

    /// The live field `field` of the layer, or of its border for `effect`
    /// border, or why the layer takes no live set there.
    pub fn live(&self, effect: Option<&str>, field: &str) -> Result<Live> {
        let layer = format!("layer '{}'", self.name);
        let kind = self.kind();
        if let Some(key) = effect {
            match (key, &self.shape) {
                ("border", Shape::Rect { .. }) => {}
                (_, Shape::Rect { .. }) => {
                    bail!("{layer} has no effect '{key}', and a rect takes border alone")
                }
                (_, Shape::Line { .. }) => {
                    bail!("{layer} has no effect '{key}', and a line takes no effects")
                }
            }
            let form = match field {
                "width" => Form::Number { none: false },
                "color" => Form::Colour {
                    gradient: false,
                    none: false,
                },
                "visible" => Form::Flag,
                _ => bail!(
                    "{layer} border has no field '{field}', and the live fields of border are \
                     width, color and visible"
                ),
            };
            return Ok(Live {
                what: format!("{layer} border {field}"),
                form,
            });
        }
        let line = matches!(self.shape, Shape::Line { .. });
        let live: &[&str] = match line {
            true => &[
                "x",
                "y",
                "color",
                "opacity",
                "visible",
                "points",
                "stroke_width",
            ],
            false => &[
                "x",
                "y",
                "color",
                "opacity",
                "visible",
                "width",
                "height",
                "border_radius",
            ],
        };
        let form = match field {
            "x" | "y" => Form::Number { none: true },
            "color" => Form::Colour {
                gradient: false,
                none: true,
            },
            "opacity" => Form::Number { none: false },
            "visible" => Form::Flag,
            "width" | "height" if !line => Form::Number { none: true },
            "border_radius" if !line => Form::Number { none: false },
            "points" if line => Form::Points,
            "stroke_width" if line => Form::Number { none: false },
            "shape" | "blend" => bail!(
                "{layer} {field} changes with a rebuild, and the live fields of a {kind} are {}",
                spoken(live)
            ),
            _ => bail!(
                "{layer} has no field '{field}', and the live fields of a {kind} are {}",
                spoken(live)
            ),
        };
        Ok(Live {
            what: format!("{layer} {field}"),
            form,
        })
    }

    /// Set the live field `field`, or the field of the border for `effect`
    /// border, to `value`, written as the scene writes the key, through the
    /// checks of a layer that a scene declares. A rect with no border takes
    /// one, at the defaults of its other fields.
    pub fn set_live(&mut self, effect: Option<&str>, field: &str, value: &Value) -> Result<()> {
        let mut d = self.to_dict();
        match effect {
            Some(key) => {
                let o = d.as_object_mut().expect("a layer is an object");
                let border = o
                    .entry(key.to_string())
                    .or_insert_with(|| Value::Object(Map::new()));
                border[field] = value.clone();
            }
            None => d[field] = value.clone(),
        }
        *self = ShapeLayer::from_dict(&format!("layer '{}'", self.name), &d)?;
        Ok(())
    }

    /// Append the quads of the layer, as its run's batch draws them, in a
    /// pass of `pass` pixels. A hidden layer, an empty rect and a line of no
    /// thickness have none.
    pub fn quads(&self, pass: [f64; 2], out: &mut Vec<f32>) {
        if !self.visible {
            return;
        }
        let fill = premultiplied(self.color, self.opacity);
        match &self.shape {
            Shape::Rect {
                width,
                height,
                border,
                border_radius,
            } => {
                let size = [width.unwrap_or(pass[0]), height.unwrap_or(pass[1])];
                if size[0] <= 0.0 || size[1] <= 0.0 {
                    return;
                }
                let at = [self.x, self.y];
                let corner = [0, 1].map(|a| at[a].unwrap_or((pass[a] - size[a]) / 2.0));
                let (b, ink) = match border {
                    Some(b) if b.visible && b.width > 0.0 => {
                        (b.width, premultiplied(b.color, self.opacity))
                    }
                    _ => (0.0, [0.0; 4]),
                };
                let geometry = [
                    corner[0] + size[0] / 2.0,
                    corner[1] + size[1] / 2.0,
                    size[0] / 2.0,
                    size[1] / 2.0,
                ];
                quad(out, 0.0, geometry, [*border_radius, b], ink, fill);
            }
            Shape::Line {
                points,
                stroke_width,
            } => {
                if *stroke_width <= 0.0 {
                    return;
                }
                let shift = [self.x.unwrap_or(0.0) + 0.5, self.y.unwrap_or(0.0) + 0.5];
                // A point that repeats the one before it adds no segment, so
                // every segment's neighbours are the segments that touch it.
                let mut p: Vec<[f64; 2]> = Vec::with_capacity(points.len());
                for q in points {
                    let q = [q[0] + shift[0], q[1] + shift[1]];
                    if p.last() != Some(&q) {
                        p.push(q);
                    }
                }
                if p.len() == 1 {
                    p.push(p[0]);
                }
                for i in 0..p.len() - 1 {
                    let prev = i.checked_sub(1).map(|k| p[k]);
                    let next = p.get(i + 2).copied();
                    let code = 4.0
                        + if prev.is_some() { 8.0 } else { 0.0 }
                        + if next.is_some() { 16.0 } else { 0.0 };
                    let near = [
                        prev.unwrap_or_default()[0],
                        prev.unwrap_or_default()[1],
                        next.unwrap_or_default()[0],
                        next.unwrap_or_default()[1],
                    ];
                    let geometry = [p[i][0], p[i][1], p[i + 1][0], p[i + 1][1]];
                    quad(
                        out,
                        code,
                        geometry,
                        [stroke_width / 2.0, 0.0],
                        on_grid(near),
                        fill,
                    );
                }
            }
        }
    }

    /// The pixels that the layer's quads cover in a pass of `pass` pixels,
    /// shown or hidden, each reaching a pixel past its shape.
    pub(crate) fn area(&self, pass: [f64; 2]) -> f64 {
        let shown = ShapeLayer {
            visible: true,
            ..self.clone()
        };
        let mut quads = Vec::new();
        shown.quads(pass, &mut quads);
        quads
            .chunks_exact(QUAD)
            .map(|q| {
                let [a, b, c, d] = [q[0], q[1], q[2], q[3]].map(f64::from);
                let segment = (f64::from(q[4]) / 4.0).floor() % 2.0 == 1.0;
                match segment {
                    false => 4.0 * (c + 1.0) * (d + 1.0),
                    true => {
                        let reach = f64::from(q[5]) + 1.0;
                        ((c - a).hypot(d - b) + 2.0 * reach) * 2.0 * reach
                    }
                }
            })
            .sum()
    }
}

/// The points of a line, `v`, each [x, y], two or more. `what` names the
/// layer in an error.
fn points(what: &str, v: Option<&Value>) -> Result<Vec<[f64; 2]>> {
    let needs = "needs two points or more, each [x, y]";
    let Some(v) = v else {
        bail!("{what} has no 'points' (the line's points, each [x, y])");
    };
    let pair = |p: &Value| match p.as_array().map(Vec::as_slice) {
        Some([x, y]) => match (x.as_f64(), y.as_f64()) {
            (Some(x), Some(y)) if x.is_finite() && y.is_finite() => Some([x, y]),
            _ => None,
        },
        _ => None,
    };
    match v.as_array() {
        Some(a) if a.len() >= 2 => a
            .iter()
            .map(pair)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| anyhow::anyhow!("{what} points {needs}, got {}", py_repr(v))),
        _ => bail!("{what} points {needs}, got {}", py_repr(v)),
    }
}

/// The border `d`, `what` in an error.
fn border(what: &str, d: &Value) -> Result<Border> {
    let Value::Object(o) = d else {
        bail!("{what} needs an object, got {}", py_repr(d));
    };
    known_keys(o, what, &BORDER_KEYS, &[])?;
    let width = number(o, what, "width", pixels, PIXELS)?.unwrap_or(BORDER_WIDTH);
    let color = match given(o, "color") {
        None => Rgb::WHITE,
        Some(v) => match Rgb::from_json(v) {
            Some(c) => c,
            None => bail!("{what} color needs {COLOUR_FORMS}, got {}", py_repr(v)),
        },
    };
    let visible = match given(o, "visible") {
        None => true,
        Some(Value::Bool(b)) => *b,
        Some(v) => bail!("{what} visible needs true or false, got {}", py_repr(v)),
    };
    Ok(Border {
        width,
        color,
        visible,
    })
}

/// The border as a scene holds it, a field at its default left out.
fn border_dict(b: &Border) -> Value {
    let mut d = Map::new();
    if b.width != BORDER_WIDTH {
        d.insert("width".to_string(), Value::from(b.width));
    }
    if !(b.color.alpha == 1.0 && b.color.rgb == [1.0; 3]) {
        d.insert("color".to_string(), b.color.to_json());
    }
    if !b.visible {
        d.insert("visible".to_string(), Value::from(false));
    }
    Value::Object(d)
}

/// `c` at `opacity`, premultiplied by its alpha: r, g, b and a.
fn premultiplied(c: Rgb, opacity: f64) -> [f64; 4] {
    let a = c.alpha * opacity;
    [c.rgb[0] * a, c.rgb[1] * a, c.rgb[2] * a, a]
}

/// `v` on the grid of 1/GRID pixel.
fn on_grid<const N: usize>(v: [f64; N]) -> [f64; N] {
    v.map(|x| (x * GRID).round() / GRID)
}

/// Append one quad of the shape program: its corners in triangle-strip
/// order, each with the corner and the kind in `code`, the `geometry`, two
/// sizes `sizes`, the vec4 `third` and the colour `fill`. A rect's code is
/// 0, a line segment's 4, plus 8 for a segment before it and 16 for one
/// after it.
fn quad(
    out: &mut Vec<f32>,
    code: f64,
    geometry: [f64; 4],
    sizes: [f64; 2],
    third: [f64; 4],
    fill: [f64; 4],
) {
    let geometry = on_grid(geometry);
    let sizes = on_grid(sizes);
    out.reserve(QUAD);
    for corner in 0..4 {
        let v = [
            geometry,
            [code + f64::from(corner), sizes[0], sizes[1], 0.0],
            third,
            fill,
        ];
        out.extend(v.as_flattened().iter().map(|f| *f as f32));
    }
}

/// The declarations of the vertex shader of every shape. ghee_a0 holds a
/// rect's centre and half size or a segment's two ends, ghee_a1 the code of
/// `quad()` and two sizes, ghee_a2 a rect's border colour or a segment's
/// neighbours, ghee_a3 the fill.
const VERTEX_DECLS: &str = r#"attribute vec4 ghee_a0;
attribute vec4 ghee_a1;
attribute vec4 ghee_a2;
attribute vec4 ghee_a3;
uniform vec3 iResolution;
uniform float ghee_vflip;
varying vec2 ghee_px;
varying vec4 ghee_v0;
varying vec4 ghee_v1;
varying vec4 ghee_v2;
varying vec4 ghee_v3;
"#;

/// The main of the vertex shader of every shape, which sets gl_Position and
/// the varyings ghee_px and ghee_v0 to ghee_v3. Each quad reaches a pixel
/// past its shape for the antialiasing.
pub(crate) const VERTEX_MAIN: &str = r#"    float code = ghee_a1.x;
    float corner = mod(code, 4.0);
    vec2 s = vec2(mod(corner, 2.0), floor(corner / 2.0)) * 2.0 - 1.0;
    float segment = mod(floor(code / 4.0), 2.0);
    vec2 at;
    if (segment < 0.5) {
        at = ghee_a0.xy + s * (ghee_a0.zw + 1.0);
    } else {
        vec2 d = ghee_a0.zw - ghee_a0.xy;
        float len = length(d);
        vec2 u = len > 0.0 ? d / len : vec2(1.0, 0.0);
        float reach = ghee_a1.y + 1.0;
        at = (ghee_a0.xy + ghee_a0.zw) * 0.5 + u * s.x * (0.5 * len + reach)
           + vec2(-u.y, u.x) * s.y * reach;
    }
    ghee_px = at;
    ghee_v0 = ghee_a0;
    // x: a segment before it, plus 2 for one after it.
    ghee_v1 = vec4(floor(code / 8.0), ghee_a1.yz, segment);
    ghee_v2 = ghee_a2;
    ghee_v3 = ghee_a3;
    vec2 ghee_tc = at / iResolution.xy;
    gl_Position = vec4((ghee_tc * 2.0 - 1.0) * vec2(1.0, ghee_vflip), 0.0, 1.0);
"#;

/// The varyings of the fragment shader of every shape.
const FRAGMENT_DECLS: &str = r#"varying vec2 ghee_px;
varying vec4 ghee_v0;
varying vec4 ghee_v1;
varying vec4 ghee_v2;
varying vec4 ghee_v3;
"#;

/// The functions of the fragment shader of every shape.
const FRAGMENT_FUNCTIONS: &str = r#"// The distance from p to a box of half size h around 0, its corners rounded
// by r, below 0 inside.
float ghee_box(vec2 p, vec2 h, float r) {
    vec2 q = abs(p) - h + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}
// The distance from p to the segment from a to b.
float ghee_segment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1.0e-12), 0.0, 1.0);
    return length(pa - ba * h);
}
// v back on the grid of GRID_INV pixel that ghee puts positions and sizes on.
vec4 ghee_grid(vec4 v) {
    return floor(v * GRID + 0.5) / GRID;
}
"#;

/// The main of the fragment shader of every shape, which sets gl_FragColor.
/// A rect takes its fill over its box and its border over the band inside
/// its edge. A line segment draws a pixel when it lies nearest of the
/// segment and its neighbours, the earlier on a tie, so a joint draws once.
/// Every edge takes the coverage of half a pixel either side of it.
pub(crate) const FRAGMENT_MAIN: &str = r#"    // VC4 lands a vertex up to 1/16 px off the position set by the vertex
    // shader, and the varyings follow the vertex, so the pixel's centre
    // comes from the floor.
    vec2 p = floor(ghee_px) + 0.5;
    vec4 g = ghee_grid(ghee_v0);
    vec4 sizes = ghee_grid(ghee_v1);
    vec4 o = vec4(0.0);
    if (ghee_v1.w < 0.5) {
        vec2 q = p - g.xy;
        float r = min(sizes.y, min(g.z, g.w));
        float b = sizes.z;
        float outside = clamp(0.5 - ghee_box(q, g.zw, r), 0.0, 1.0);
        float inside = 0.0;
        vec2 hi = g.zw - b;
        if (min(hi.x, hi.y) > 0.0) inside = clamp(0.5 - ghee_box(q, hi, max(r - b, 0.0)), 0.0, 1.0);
        float ring = outside - inside;
        o = ghee_v2 * ring + ghee_v3 * outside * (1.0 - ghee_v2.a * ring);
    } else {
        vec4 n = ghee_grid(ghee_v2);
        float w = sizes.y;
        float d = ghee_segment(p, g.xy, g.zw) - w;
        float dp = mod(ghee_v1.x, 2.0) > 0.5 ? ghee_segment(p, n.xy, g.xy) - w : 1.0e9;
        float dn = ghee_v1.x > 1.5 ? ghee_segment(p, g.zw, n.zw) - w : 1.0e9;
        if (d < dp && d <= dn) o = ghee_v3 * clamp(0.5 - d, 0.0, 1.0);
    }
    gl_FragColor = o;
"#;

/// The functions of the fragment shader of every shape, with the grid
/// written in.
pub(crate) fn fragment_functions() -> String {
    FRAGMENT_FUNCTIONS
        .replace("GRID_INV", &format!("1/{GRID}"))
        .replace("GRID", &format!("{GRID:?}"))
}

/// The two shaders of every shape, with the grid written in.
pub fn program() -> (String, String) {
    let precision = crate::scene::PRECISION.join("\n");
    let vertex = format!("{precision}\n{VERTEX_DECLS}void main() {{\n{VERTEX_MAIN}}}");
    let fragment = format!(
        "{precision}\n{FRAGMENT_DECLS}{}void main() {{\n{FRAGMENT_MAIN}}}",
        fragment_functions()
    );
    (vertex, fragment)
}
