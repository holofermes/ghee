//! The calls of a drawing. A host's drawing states the whole look of each
//! layer that it draws, every frame, as the keywords of one call: a text, a
//! rect, an image or a line. A Call holds the values of those keywords in
//! their order, and layer_of() makes the layer that the call gives the
//! layer of its name, the layer that Scene::look() makes from the same look
//! written as a scene declares it.
//!
//! A value that the scene format refuses, and a value of a kind that a Call
//! holds no form for, make no layer here: the host then writes the look and
//! hands it to Renderer::update_layer(), which says what it refuses.

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
}

/// The kind of a call, which names the layer that it draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallKind {
    Text,
    Rect,
    Image,
    Line,
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

impl CallKind {
    /// The keywords that a call of the kind takes, in the order of its args.
    pub fn keys(self) -> &'static [&'static str] {
        match self {
            CallKind::Text => &TEXT_ARGS,
            CallKind::Rect => &RECT_ARGS,
            CallKind::Image => &IMAGE_ARGS,
            CallKind::Line => &LINE_ARGS,
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
/// values, and so does a rect's border. A text keeps the fonts of `old`, and
/// an image keeps its pixels while its image stays. None for a call of
/// another kind than `old`, and for a value that the scene format refuses.
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
        (CallKind::Rect | CallKind::Line, Layer::Shape(o)) => {
            ShapeLayer::of_args(name, call.kind == CallKind::Line, &call.args, o).map(Layer::Shape)
        }
        _ => None,
    }
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
