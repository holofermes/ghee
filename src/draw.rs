//! The calls of a drawing. A host's drawing states the whole look of each
//! layer that it draws, every frame, as the keywords of one call: a text, a
//! rect, an image, a line or a circle. A Call holds the values of those
//! keywords in their order, and layer_of() makes the layer that the call
//! gives the layer of its name, the layer that Scene::look() makes from the
//! same look written as a scene declares it.
//!
//! A value that the scene format refuses, and a value of a kind that a Call
//! holds no form for, make no layer here: the host then writes the look and
//! hands it to Renderer::update_layer(), which says what it refuses.
//!
//! A scene process sends the calls of a drawing as looks, look_of() writes
//! one, and call_of() reads one back.

use serde_json::{Map, Number, Value};

use crate::engine::Blend;
use crate::layer::{ImageLayer, Layer, Rgb, TextLayer};
use crate::shape::ShapeLayer;

/// The value of one keyword of a call, as the scene format reads it. None
/// stands for a keyword left out, and for a number that JSON holds no form
/// for, such as NaN, which a look written as JSON leaves out the same way.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    None,
    Bool(bool),
    Num(f64),
    Str(String),
    List(Vec<Arg>),
    /// A dict's keys and values, in its order.
    Map(Vec<(String, Arg)>),
}

impl Arg {
    /// The number `f`, None when it is not finite.
    pub fn num(f: f64) -> Arg {
        match f.is_finite() {
            true => Arg::Num(f),
            false => Arg::None,
        }
    }

    /// `v`, a value of a look written as JSON, as an arg.
    pub fn of_json(v: &Value) -> Arg {
        match v {
            Value::Null => Arg::None,
            Value::Bool(b) => Arg::Bool(*b),
            Value::Number(n) => n.as_f64().map_or(Arg::None, Arg::num),
            Value::String(s) => Arg::Str(s.clone()),
            Value::Array(items) => Arg::List(items.iter().map(Arg::of_json).collect()),
            Value::Object(m) => Arg::Map(
                m.iter()
                    .map(|(k, v)| (k.clone(), Arg::of_json(v)))
                    .collect(),
            ),
        }
    }

    /// The arg as a look written as JSON holds it.
    pub fn to_json(&self) -> Value {
        match self {
            Arg::None => Value::Null,
            Arg::Bool(b) => Value::Bool(*b),
            Arg::Num(f) => Number::from_f64(*f).map_or(Value::Null, Value::Number),
            Arg::Str(s) => Value::String(s.clone()),
            Arg::List(items) => Value::Array(items.iter().map(Arg::to_json).collect()),
            Arg::Map(m) => Value::Object(m.iter().map(|(k, a)| (k.clone(), a.to_json())).collect()),
        }
    }
}

/// The kind of a call, which names the layer that it draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallKind {
    Text,
    Rect,
    Image,
    Line,
    Circle,
}

/// The keywords of a text call, in the order of its args. The words come
/// first.
pub const TEXT_ARGS: [&str; 18] = [
    "text",
    "x",
    "y",
    "color",
    "opacity",
    "blend",
    "visible",
    "font_size",
    "font_family",
    "text_align",
    "render",
    "text_shadow",
    "glow",
    "text_stroke",
    "karaoke",
    "wave",
    "scroll",
    "blink",
];

/// The keywords of a rect call, in the order of its args.
pub const RECT_ARGS: [&str; 10] = [
    "x",
    "y",
    "width",
    "height",
    "color",
    "opacity",
    "blend",
    "visible",
    "border",
    "border_radius",
];

/// The keywords of an image call, in the order of its args. The image
/// comes first.
pub const IMAGE_ARGS: [&str; 9] = [
    "image", "x", "y", "width", "height", "color", "opacity", "blend", "visible",
];

/// The keywords of a line call, in the order of its args. The points come
/// first.
pub const LINE_ARGS: [&str; 8] = [
    "points",
    "x",
    "y",
    "color",
    "opacity",
    "blend",
    "visible",
    "stroke_width",
];

/// The keywords of a circle call, in the order of its args.
pub const CIRCLE_ARGS: [&str; 8] = [
    "x", "y", "radius", "color", "opacity", "blend", "visible", "border",
];

impl CallKind {
    /// The keywords that a call of the kind takes, in the order of its args.
    pub fn keys(self) -> &'static [&'static str] {
        match self {
            CallKind::Text => &TEXT_ARGS,
            CallKind::Rect => &RECT_ARGS,
            CallKind::Image => &IMAGE_ARGS,
            CallKind::Line => &LINE_ARGS,
            CallKind::Circle => &CIRCLE_ARGS,
        }
    }
}

/// One call of a drawing: the layer that it draws, by name, and its args,
/// one for each keyword of its kind.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub kind: CallKind,
    pub layer: String,
    pub args: Vec<Arg>,
}

/// The layer that `call` gives `old`, the layer of its name, as
/// Scene::look() makes it: every keyword left out takes its default, and an
/// effect of `old` that the call leaves out stays, bypassed, with its
/// values, and so does a rect's or a circle's border. A text keeps the
/// fonts of `old`, and an image keeps its pixels while its image stays. None
/// for a call of another kind than `old`, and for a value that the scene
/// format refuses.
pub fn layer_of(call: &Call, old: &Layer) -> Option<Layer> {
    if call.args.len() != call.kind.keys().len() {
        return None;
    }
    let name = &call.layer;
    match (call.kind, old) {
        (CallKind::Text, Layer::Text(o)) => {
            TextLayer::of_args(name, &call.args, o).map(Layer::Text)
        }
        (CallKind::Image, Layer::Image(o)) => {
            ImageLayer::of_args(name, &call.args, o).map(Layer::Image)
        }
        (CallKind::Rect | CallKind::Line | CallKind::Circle, Layer::Shape(o)) => {
            ShapeLayer::of_args(name, call.kind, &call.args, o).map(Layer::Shape)
        }
        _ => None,
    }
}

/// The layer that `call` gives `old`, the layer of its name, when the two
/// differ in live fields alone and an image layer keeps its image, None in
/// the place of the layer when `old` has that look already. None for a call
/// that layer_of() makes no layer of, and for a look that builds the layer
/// again or gives an image layer another image: Scene::update_layer() takes
/// those as looks.
pub fn live_of(call: &Call, old: &Layer) -> Option<Option<Layer>> {
    let new = layer_of(call, old)?;
    if new == *old {
        return Some(None);
    }
    let other_image =
        matches!((old, &new), (Layer::Image(o), Layer::Image(n)) if o.image != n.image);
    (old.builds_as(&new) && !other_image).then_some(Some(new))
}

/// The look that `call` gives its layer, written as a scene declares the
/// layer: the name, a shape's `shape`, and each keyword that the call gives.
pub fn look_of(call: &Call) -> Value {
    let mut look = Map::new();
    look.insert("name".to_string(), Value::from(call.layer.as_str()));
    match call.kind {
        CallKind::Rect => look.insert("shape".to_string(), Value::from("rect")),
        CallKind::Line => look.insert("shape".to_string(), Value::from("line")),
        CallKind::Circle => look.insert("shape".to_string(), Value::from("circle")),
        CallKind::Text | CallKind::Image => None,
    };
    for (k, a) in call.kind.keys().iter().zip(&call.args) {
        if *a != Arg::None {
            look.insert(k.to_string(), a.to_json());
        }
    }
    Value::Object(look)
}

/// The call that gives the layer named in `look` that look, a layer as a
/// scene declares it. A `shape` makes a rect, a line or a circle, an `image`
/// an image, and any other look a text. None for a look with no name, and
/// for a look that holds a key that no call of its kind takes.
pub fn call_of(look: &Value) -> Option<Call> {
    let m = look.as_object()?;
    let layer = m.get("name")?.as_str()?.to_string();
    let kind = match (m.get("shape").map(Value::as_str), m.contains_key("image")) {
        (Some(Some("rect")), _) => CallKind::Rect,
        (Some(Some("line")), _) => CallKind::Line,
        (Some(Some("circle")), _) => CallKind::Circle,
        (Some(_), _) => return None,
        (None, true) => CallKind::Image,
        (None, false) => CallKind::Text,
    };
    let keys = kind.keys();
    if m.keys()
        .any(|k| k != "name" && k != "shape" && !keys.contains(&k.as_str()))
    {
        return None;
    }
    let args = keys
        .iter()
        .map(|k| m.get(*k).map_or(Arg::None, Arg::of_json))
        .collect();
    Some(Call { kind, layer, args })
}

/// `a` as a flag, `default` for None. None for any other value.
pub(crate) fn flag(a: &Arg, default: bool) -> Option<bool> {
    match a {
        Arg::None => Some(default),
        Arg::Bool(b) => Some(*b),
        _ => None,
    }
}

/// `a` as a number that `ok` takes, Some(None) for None. None for any
/// other value.
pub(crate) fn number(a: &Arg, ok: fn(f64) -> bool) -> Option<Option<f64>> {
    match a {
        Arg::None => Some(None),
        Arg::Num(f) if f.is_finite() && ok(*f) => Some(Some(*f)),
        _ => None,
    }
}

/// `a` as a colour, as Rgb::from_json() reads one: `#rrggbb`, a CSS colour
/// name or three numbers. None for any other value.
pub(crate) fn rgb(a: &Arg) -> Option<Rgb> {
    match a {
        Arg::Str(s) => Rgb::from_text(s),
        Arg::List(items) => match items.as_slice() {
            [Arg::Num(r), Arg::Num(g), Arg::Num(b)] if [r, g, b].iter().all(|c| c.is_finite()) => {
                Some(Rgb::numbers([*r, *g, *b]))
            }
            _ => None,
        },
        _ => None,
    }
}

/// The keys and the values of `a`, a dict whose every key is one of
/// `keys`. None for any other value.
pub(crate) fn map<'a>(a: &'a Arg, keys: &[&str]) -> Option<&'a [(String, Arg)]> {
    match a {
        Arg::Map(m) if m.iter().all(|(k, _)| keys.contains(&k.as_str())) => Some(m),
        _ => None,
    }
}

/// The value of key `k` of `m`, None when it is missing.
pub(crate) fn get<'a>(m: &'a [(String, Arg)], k: &str) -> &'a Arg {
    m.iter()
        .find(|(key, _)| key == k)
        .map_or(&Arg::None, |(_, v)| v)
}

/// The settings that every layer takes, from the args `visible`, `x`, `y`,
/// `opacity` and `blend`, as Every::from_dict() reads them.
pub(crate) struct Every {
    pub visible: bool,
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub opacity: f64,
    pub blend: Blend,
}

impl Every {
    pub(crate) fn of_args(
        visible: &Arg,
        x: &Arg,
        y: &Arg,
        opacity: &Arg,
        blend: &Arg,
    ) -> Option<Every> {
        let blend = match blend {
            Arg::None => Blend::Normal,
            Arg::Str(s) => Blend::NAMES.iter().find(|(_, n)| n == s)?.0,
            _ => return None,
        };
        Some(Every {
            visible: flag(visible, true)?,
            x: number(x, |_| true)?,
            y: number(y, |_| true)?,
            opacity: number(opacity, |f| (0.0..=1.0).contains(&f))?.unwrap_or(1.0),
            blend,
        })
    }
}
