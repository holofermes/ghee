//! Layers: words, a picture or a shape that a pass paints over its buffer
//! after its own shader, in the order of its `layers`. `crate::shape`
//! describes shape layers.
//!
//! A text layer owns its words, its font and its look. Its `text` holds the
//! words, live, and every pass of the scene except a filter pass reads them
//! with `<name>_cover` and `<name>_size`, whether the layer shows or not. The
//! layer draws one quad over each inked glyph through shaders that ghee
//! writes from its settings, [`TextLayer::shaders`], and a slug layer with
//! no effect draws in one batch with the shapes and the slug layers beside
//! it, as `crate::run` describes.
//!
//! An image layer draws an image asset of the scene, or an image file of its
//! own, as one quad through [`ImageLayer::shaders`]. Its shader samples the
//! image as `ghee_0_image_tex`, which the engine binds to the image for the
//! layer's batch alone, so the layer gives the passes no name.
//!
//! A layer's own shaders call the layer GLSL_NAME, a name that no layer can
//! take, so every name that they hold meets no name that ghee gives a pass.
//! A setting that changes with no rebuild reaches them as a uniform,
//! `ghee_0_<token>`, as [`Layer::uniforms`] lists them, which the engine
//! sets on each of the layer's batches alone. Layers with the same settings
//! then write the same shaders, which one compiled program draws for all of
//! them. A wave, a scroll and a blink move by a phase, a uniform of the
//! layer's batches too, which the engine advances each frame by the
//! effect's speed or rate times the frame's time step, [`TextLayer::paces`],
//! so a new speed carries the motion on from where it is.
//!
//! The block of the words or the picture sits at `x`, `y`, the pass pixel of
//! its top left, y down, and a missing axis centres the block on it. With no
//! `font_size`, a text block takes the largest size at which it fits its
//! room: the pass less FIT_MARGIN of its shorter side on each edge, and on a
//! given axis the pass past the given pixel less the margin at the far edge.
//! The shaders work the size out from the block's size in em,
//! `ghee_0_size`, which the engine sets as it sets `<name>_size` for the
//! passes, so it follows the words.

use std::borrow::Cow;
use std::sync::Arc;

use anyhow::{bail, Result};
use serde_json::{Map, Value};

use crate::atlas;
use crate::colour;
use crate::css;
use crate::draw::{self, Arg};
use crate::emoji::{self, EmojiFont};
use crate::engine::Blend;
use crate::scene::{hex_colour, keys_on, known_keys, number_list, py_float, py_repr, PASS_KEYS};
use crate::shape::{self, ShapeLayer};
use crate::text::{self, Align, Font, Layout};

/// The name of every layer in its own shaders. A layer's name starts with a
/// letter, so no layer takes it.
pub const GLSL_NAME: &str = "0";

/// The keys that every layer takes.
const EVERY_KEYS: [&str; 7] = ["name", "visible", "x", "y", "color", "opacity", "blend"];

/// The keys of a text layer past EVERY_KEYS.
const TEXT_KEYS: [&str; 12] = [
    "text",
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

/// The keys of an image layer past EVERY_KEYS.
const IMAGE_KEYS: [&str; 3] = ["image", "width", "height"];

/// Every key that a layer takes, of any kind.
pub(crate) fn layer_keys() -> Vec<&'static str> {
    [
        &EVERY_KEYS[..],
        &TEXT_KEYS,
        &IMAGE_KEYS,
        &shape::RECT_KEYS,
        &shape::LINE_KEYS,
    ]
    .concat()
}

/// A text layer's keys, and the kind of layer in an error.
pub(crate) const TEXT_KIND: (&[&str], &str) = (&TEXT_KEYS, "a text layer");

/// An image layer's keys, and the kind of layer in an error.
pub(crate) const IMAGE_KIND: (&[&str], &str) = (&IMAGE_KEYS, "an image layer");

/// The keys of every layer that change with no rebuild, as live fields.
const LIVE_KEYS: [&str; 5] = ["x", "y", "color", "opacity", "visible"];

/// The keys of a gradient.
const GRADIENT_KEYS: [&str; 3] = ["from", "to", "angle"];

/// The angle of a gradient that names none, as CSS's: top to bottom.
const GRADIENT_ANGLE: f64 = 180.0;

/// The margin on every edge of the room that fits a block with no
/// `font_size`, as a fraction of the pass's shorter side.
pub const FIT_MARGIN: f64 = 0.05;

/// The texels per em of an MTSDF atlas, smallest first. A layer takes the
/// first step at or above its pixels per em at build, so layers of near sizes
/// share an atlas, and the last step at most. Text under 32 px takes 32, since
/// an atlas at a small text's own size strays further from its outline. On a
/// Pi Zero 2 W the first render with an empty cache, which generates the
/// preload, takes 7 s at 128 texels per em and 14 s at 192.
pub const ATLAS_STEPS: [u32; 5] = [32, 48, 64, 96, 128];

/// A colour, each channel 0 to 1, its alpha, and the form that the scene
/// wrote it in, so a save writes it back in that form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgb {
    pub rgb: [f64; 3],
    /// 1, or 0 for `transparent`.
    pub alpha: f64,
    pub written: Written,
}

/// The form of a colour in a scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Written {
    /// `[r, g, b]`.
    Numbers,
    /// `"#rrggbb"`.
    Hex,
    /// A CSS colour name, in the case that the scene wrote it in.
    Name(css::Name),
}

/// What a colour takes, as an error says it.
pub const COLOUR_FORMS: &str = "'#rrggbb', a CSS colour name or three numbers [r, g, b]";

impl Rgb {
    pub const WHITE: Rgb = Rgb::numbers([1.0; 3]);

    /// The opaque colour `[r, g, b]`, written as three numbers.
    pub const fn numbers(rgb: [f64; 3]) -> Rgb {
        Rgb {
            rgb,
            alpha: 1.0,
            written: Written::Numbers,
        }
    }

    /// `"#rrggbb"`, a CSS colour name or `[r, g, b]`, None for any other
    /// value.
    pub fn from_json(v: &Value) -> Option<Rgb> {
        match v {
            Value::String(s) => Rgb::from_text(s),
            v => number_list(v, 3).map(|n| Rgb::numbers([n[0], n[1], n[2]])),
        }
    }

    /// `#rrggbb` or a CSS colour name, None for any other text.
    pub fn from_text(s: &str) -> Option<Rgb> {
        if let Some(rgb) = hex_colour(s) {
            return Some(Rgb {
                written: Written::Hex,
                ..Rgb::numbers(rgb)
            });
        }
        let (rgb, alpha, name) = css::named(s)?;
        Some(Rgb {
            rgb: rgb.map(|c| f64::from(c) / 255.0),
            alpha,
            written: Written::Name(name),
        })
    }

    pub fn to_json(self) -> Value {
        match self.written {
            Written::Hex => {
                let [r, g, b] = self
                    .rgb
                    .map(|c| (c * 255.0).round().clamp(0.0, 255.0) as u8);
                Value::from(format!("#{r:02x}{g:02x}{b:02x}"))
            }
            Written::Name(name) => Value::from(name.as_str()),
            Written::Numbers => Value::from(self.rgb.to_vec()),
        }
    }

    /// The same colour, whatever form it came in.
    fn same(self, other: Rgb) -> bool {
        self.rgb == other.rgb && self.alpha == other.alpha
    }
}

/// A text layer's `color`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Color {
    Solid(Rgb),
    /// CSS's linear gradient over the block, `from` at its start and `to` at
    /// its end along `angle`, in degrees clockwise from up. It moves with the
    /// block.
    Gradient {
        from: Rgb,
        to: Rgb,
        angle: f64,
    },
}

impl Color {
    /// `what` names the layer in an error.
    fn from_json(what: &str, v: &Value) -> Result<Color> {
        if let Some(rgb) = Rgb::from_json(v) {
            return Ok(Color::Solid(rgb));
        }
        let Value::Object(o) = v else {
            bail!(
                "{what} color needs '#rrggbb', a CSS colour name, [r, g, b] or a gradient \
                 {{\"from\", \"to\", \"angle\"}}, got {}",
                py_repr(v)
            );
        };
        let at = format!("{what} color");
        known_keys(o, &at, &GRADIENT_KEYS, &[])?;
        let end = |k: &str| match o.get(k).filter(|v| !v.is_null()) {
            None => bail!("{at} has no '{k}'"),
            Some(v) => Rgb::from_json(v).ok_or_else(|| {
                anyhow::anyhow!("{at} {k} needs {COLOUR_FORMS}, got {}", py_repr(v))
            }),
        };
        let (from, to) = (end("from")?, end("to")?);
        let angle = match o.get("angle").filter(|v| !v.is_null()) {
            None => GRADIENT_ANGLE,
            Some(v) => match v.as_f64() {
                Some(a) if a.is_finite() => a,
                _ => bail!("{at} angle needs a number of degrees, got {}", py_repr(v)),
            },
        };
        Ok(Color::Gradient { from, to, angle })
    }

    fn to_json(self) -> Value {
        match self {
            Color::Solid(c) => c.to_json(),
            Color::Gradient { from, to, angle } => {
                let mut d = Map::new();
                d.insert("from".to_string(), from.to_json());
                d.insert("to".to_string(), to.to_json());
                if angle != GRADIENT_ANGLE {
                    d.insert("angle".to_string(), Value::from(angle));
                }
                Value::Object(d)
            }
        }
    }

    /// The colours at the gradient's start and end, and its angle. A solid
    /// colour is a gradient of one colour.
    pub(crate) fn ends(self) -> (Rgb, Rgb, f64) {
        match self {
            Color::Solid(c) => (c, c, GRADIENT_ANGLE),
            Color::Gradient { from, to, angle } => (from, to, angle),
        }
    }
}

/// How a layer finds the coverage of each pixel of a glyph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Render {
    /// One read of the glyph's MTSDF in an atlas a pixel, with the distance
    /// to the outline.
    Mtsdf,
    /// Slug's exact coverage from the glyph's curves, as `<name>_cover`
    /// reads it, sharp at any size and with no atlas.
    Slug,
}

impl Render {
    /// The render of a text layer in `font_family` that names none: slug
    /// for the pixel font, whose square edges it draws exactly, and mtsdf
    /// for any other.
    pub fn default_for(font_family: Option<&str>) -> Render {
        match font_family {
            Some(text::PIXEL) => Render::Slug,
            _ => Render::Mtsdf,
        }
    }

    /// The render as a scene writes it.
    pub fn as_str(self) -> &'static str {
        match self {
            Render::Mtsdf => "mtsdf",
            Render::Slug => "slug",
        }
    }
}

/// The texels per em of an MTSDF atlas, and the range of its field in
/// texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Detail {
    pub px_em: u32,
    pub range: u32,
}

impl Detail {
    /// How far the field reaches past the outline on each side, in em.
    fn reach(self) -> f64 {
        f64::from(self.range) / f64::from(self.px_em) / 2.0
    }
}

/// What an effect field takes.
#[derive(Clone, Copy)]
enum Takes {
    /// A number of em, 0 or more.
    Em,
    /// Any number, in the unit named.
    Number(&'static str),
    /// A number from 0 to 1.
    Unit,
    /// A number a second, 0 or more, of the thing named.
    Rate(&'static str),
    /// A colour, as Rgb::from_json() takes it.
    Colour,
}

/// How a change to an effect field reaches the layer's drawing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Change {
    /// The layer builds again.
    Rebuild,
    /// A uniform of the layer's own takes the value.
    Uniform,
    /// The field is the pace of the effect's phase, which ghee advances by
    /// the pace times each frame's time step.
    Phase,
}

/// A field of an effect: its key, its default, what it takes, and how a
/// change to it reaches the layer's drawing.
struct Field {
    name: &'static str,
    default: FieldValue,
    takes: Takes,
    change: Change,
}

/// The value of an effect field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldValue {
    Num(f64),
    Rgb(Rgb),
}

const fn num(name: &'static str, default: f64, takes: Takes, change: Change) -> Field {
    Field {
        name,
        default: FieldValue::Num(default),
        takes,
        change,
    }
}

/// A colour field, a uniform, its default `"#rrggbb"` written as its bytes.
const fn colour(default: [u8; 3]) -> Field {
    let [r, g, b] = default;
    Field {
        name: "color",
        default: FieldValue::Rgb(Rgb {
            written: Written::Hex,
            ..Rgb::numbers([r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0])
        }),
        takes: Takes::Colour,
        change: Change::Uniform,
    }
}

const EM: &str = "a number of em";

const SHADOW: [Field; 5] = [
    num("x", 0.06, Takes::Number(EM), Change::Rebuild),
    num("y", 0.08, Takes::Number(EM), Change::Rebuild),
    num("blur", 0.15, Takes::Em, Change::Rebuild),
    colour([0, 0, 0]),
    num("strength", 0.7, Takes::Unit, Change::Uniform),
];
const GLOW: [Field; 3] = [
    num("blur", 0.3, Takes::Em, Change::Rebuild),
    colour([0xff, 0xa5, 0x00]),
    num("strength", 0.8, Takes::Unit, Change::Uniform),
];
const STROKE: [Field; 2] = [
    num("width", 0.06, Takes::Em, Change::Rebuild),
    colour([0, 0, 0]),
];
const KARAOKE: [Field; 2] = [
    num(
        "sung",
        0.0,
        Takes::Number("a number of characters"),
        Change::Uniform,
    ),
    colour([0xff, 0xad, 0x2e]),
];
const WAVE: [Field; 2] = [
    num("height", 0.1, Takes::Number(EM), Change::Rebuild),
    num(
        "speed",
        0.6,
        Takes::Number("a number of waves a second"),
        Change::Phase,
    ),
];
const SCROLL: [Field; 1] = [num(
    "speed",
    2.0,
    Takes::Number("a number of em a second"),
    Change::Phase,
)];
const BLINK: [Field; 1] = [num("rate", 1.0, Takes::Rate("blinks"), Change::Phase)];

/// An effect that ghee draws on a text layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    TextShadow,
    Glow,
    TextStroke,
    Karaoke,
    Wave,
    Scroll,
    Blink,
}

impl Kind {
    /// Every effect, in the order a layer holds them. The first three draw
    /// in this order, under the letters.
    pub const ALL: [Kind; 7] = [
        Kind::TextShadow,
        Kind::Glow,
        Kind::TextStroke,
        Kind::Karaoke,
        Kind::Wave,
        Kind::Scroll,
        Kind::Blink,
    ];

    /// The effect's key on a layer.
    pub fn key(self) -> &'static str {
        match self {
            Kind::TextShadow => "text_shadow",
            Kind::Glow => "glow",
            Kind::TextStroke => "text_stroke",
            Kind::Karaoke => "karaoke",
            Kind::Wave => "wave",
            Kind::Scroll => "scroll",
            Kind::Blink => "blink",
        }
    }

    /// The effect in the tokens of its uniforms.
    fn token(self) -> &'static str {
        match self {
            Kind::TextShadow => "shadow",
            Kind::TextStroke => "stroke",
            k => k.key(),
        }
    }

    fn fields(self) -> &'static [Field] {
        match self {
            Kind::TextShadow => &SHADOW,
            Kind::Glow => &GLOW,
            Kind::TextStroke => &STROKE,
            Kind::Karaoke => &KARAOKE,
            Kind::Wave => &WAVE,
            Kind::Scroll => &SCROLL,
            Kind::Blink => &BLINK,
        }
    }

    /// True for an effect that reads the distance to the letters, which an
    /// MTSDF atlas holds and the slug render lacks.
    pub fn reads_distance(self) -> bool {
        matches!(self, Kind::TextShadow | Kind::Glow | Kind::TextStroke)
    }

    /// The field that paces the effect's phase, for an effect that moves by
    /// one: wave, scroll and blink.
    fn pace(self) -> Option<&'static str> {
        self.fields()
            .iter()
            .find(|f| f.change == Change::Phase)
            .map(|f| f.name)
    }
}

/// An effect of a text layer, with a value for each of its fields.
#[derive(Clone, Debug, PartialEq)]
pub struct Effect {
    pub kind: Kind,
    /// False bypasses the effect, which keeps its values.
    pub visible: bool,
    /// The value of each field, in the order of the kind's fields.
    values: Vec<FieldValue>,
}

impl Effect {
    /// `what` names the effect in an error, as "layer 't' glow".
    fn from_dict(what: &str, kind: Kind, d: &Value) -> Result<Effect> {
        let Value::Object(o) = d else {
            bail!("{what} needs an object, got {}", py_repr(d));
        };
        let fields = kind.fields();
        let keys: Vec<&str> = fields.iter().map(|f| f.name).chain(["visible"]).collect();
        known_keys(o, what, &keys, &[])?;
        let visible = match o.get("visible").filter(|v| !v.is_null()) {
            None => true,
            Some(Value::Bool(b)) => *b,
            Some(v) => bail!("{what} visible needs true or false, got {}", py_repr(v)),
        };
        let mut values = Vec::new();
        for f in fields {
            let Some(v) = o.get(f.name).filter(|v| !v.is_null()) else {
                values.push(f.default);
                continue;
            };
            let got = |needs: &str| {
                anyhow::anyhow!("{what} {} needs {needs}, got {}", f.name, py_repr(v))
            };
            let n = v.as_f64().filter(|n| n.is_finite());
            values.push(match f.takes {
                Takes::Colour => {
                    FieldValue::Rgb(Rgb::from_json(v).ok_or_else(|| got(COLOUR_FORMS))?)
                }
                Takes::Em => match n {
                    Some(n) if n >= 0.0 => FieldValue::Num(n),
                    _ => return Err(got("a number of em, 0 or more")),
                },
                Takes::Number(unit) => FieldValue::Num(n.ok_or_else(|| got(unit))?),
                Takes::Unit => match n {
                    Some(n) if (0.0..=1.0).contains(&n) => FieldValue::Num(n),
                    _ => return Err(got("a number from 0 to 1")),
                },
                Takes::Rate(of) => match n {
                    Some(n) if n >= 0.0 => FieldValue::Num(n),
                    _ => return Err(got(&format!("a number of {of} a second, 0 or more"))),
                },
            });
        }
        Ok(Effect {
            kind,
            visible,
            values,
        })
    }

    /// The effect of a drawing's call, `a`, as from_dict() reads the same
    /// dict. None for a value that from_dict() refuses.
    fn of_arg(kind: Kind, a: &Arg) -> Option<Effect> {
        let fields = kind.fields();
        let Arg::Map(m) = a else { return None };
        let known = |k: &str| k == "visible" || fields.iter().any(|f| f.name == k);
        if !m.iter().all(|(k, _)| known(k)) {
            return None;
        }
        let mut values = Vec::with_capacity(fields.len());
        for f in fields {
            values.push(Effect::field_of(f, draw::get(m, f.name))?);
        }
        Some(Effect {
            kind,
            visible: draw::flag(draw::get(m, "visible"), true)?,
            values,
        })
    }

    /// The value of field `f` that a drawing's call gives as `v`, its
    /// default for None, as from_dict() reads the same value. None for a
    /// value that from_dict() refuses.
    fn field_of(f: &Field, v: &Arg) -> Option<FieldValue> {
        if *v == Arg::None {
            return Some(f.default);
        }
        Some(match f.takes {
            Takes::Colour => FieldValue::Rgb(draw::rgb(v)?),
            Takes::Em | Takes::Rate(_) => FieldValue::Num(draw::number(v, |n| n >= 0.0)??),
            Takes::Number(_) => FieldValue::Num(draw::number(v, |_| true)??),
            Takes::Unit => FieldValue::Num(draw::number(v, |n| (0.0..=1.0).contains(&n))??),
        })
    }

    /// Set field `field` to `v`, read as field_of() reads it, or the
    /// effect's `visible`. False for a field that changes with a rebuild or
    /// moves a phase, and for a value that field_of() refuses, which leave
    /// the effect as it was.
    fn set_uniform(&mut self, field: &str, v: &Arg) -> bool {
        if field == "visible" {
            return match draw::flag(v, true) {
                Some(b) => {
                    self.visible = b;
                    true
                }
                None => false,
            };
        }
        let fields = self.kind.fields();
        let Some(k) = fields.iter().position(|f| f.name == field) else {
            return false;
        };
        match (fields[k].change, Effect::field_of(&fields[k], v)) {
            (Change::Uniform, Some(value)) => {
                self.values[k] = value;
                true
            }
            _ => false,
        }
    }

    /// The effect as a layer holds it, a field at its default left out.
    fn to_dict(&self) -> Value {
        let mut d = Map::new();
        for (f, v) in self.kind.fields().iter().zip(&self.values) {
            let at_default = match (v, f.default) {
                (FieldValue::Num(a), FieldValue::Num(b)) => *a == b,
                (FieldValue::Rgb(a), FieldValue::Rgb(b)) => a.same(b),
                _ => false,
            };
            if !at_default {
                let json = match v {
                    FieldValue::Num(n) => Value::from(*n),
                    FieldValue::Rgb(c) => c.to_json(),
                };
                d.insert(f.name.to_string(), json);
            }
        }
        if !self.visible {
            d.insert("visible".to_string(), Value::from(false));
        }
        Value::Object(d)
    }

    fn value(&self, field: &str) -> FieldValue {
        let at = self.kind.fields().iter().position(|f| f.name == field);
        self.values[at.unwrap_or_else(|| panic!("{} has no field {field}", self.kind.key()))]
    }

    /// The number in field `field`.
    pub fn num(&self, field: &str) -> f64 {
        match self.value(field) {
            FieldValue::Num(n) => n,
            FieldValue::Rgb(_) => panic!("{} {field} is a colour", self.kind.key()),
        }
    }

    /// The colour in field `field`.
    pub fn rgb(&self, field: &str) -> [f64; 3] {
        match self.value(field) {
            FieldValue::Rgb(c) => c.rgb,
            FieldValue::Num(_) => panic!("{} {field} is a number", self.kind.key()),
        }
    }
}

/// The settings that every layer takes, and `what`, the layer in an error.
pub(crate) struct Every {
    pub(crate) name: String,
    pub(crate) what: String,
    pub(crate) visible: bool,
    pub(crate) x: Option<f64>,
    pub(crate) y: Option<f64>,
    pub(crate) opacity: f64,
    pub(crate) blend: Blend,
}

impl Every {
    /// Check the keys of layer `o`, which takes EVERY_KEYS and `own`, and
    /// read the settings of every layer. `at` names the layer by its place
    /// in an error until its name is known, and `others` holds the keys of
    /// the other kinds of layer, each with the kind that takes it.
    pub(crate) fn from_dict(
        at: &str,
        o: &Map<String, Value>,
        own: &[&str],
        others: &[(&[&str], &str)],
    ) -> Result<Every> {
        let what = match o.get("name") {
            Some(Value::String(n)) => format!("layer '{n}'"),
            _ => at.to_string(),
        };
        let takes = [&EVERY_KEYS[..], own].concat();
        let mut elsewhere: Vec<(&str, String)> = Vec::new();
        for (keys, kind) in others {
            elsewhere.extend(keys_on(&takes, keys, kind));
        }
        elsewhere.extend(keys_on(&takes, &PASS_KEYS, "its pass"));
        known_keys(o, &what, &takes, &elsewhere)?;
        let name = match given(o, "name") {
            None => bail!("{at} has no 'name'"),
            Some(Value::String(n)) => n.clone(),
            Some(v) => bail!("{at} name needs a string, got {}", py_repr(v)),
        };
        let visible = match given(o, "visible") {
            None => true,
            Some(Value::Bool(b)) => *b,
            Some(v) => bail!("{what} visible needs true or false, got {}", py_repr(v)),
        };
        let x = number(o, &what, "x", |_| true, "a number of pixels")?;
        let y = number(o, &what, "y", |_| true, "a number of pixels")?;
        let opacity = number(
            o,
            &what,
            "opacity",
            |f| (0.0..=1.0).contains(&f),
            "a number from 0 to 1",
        )?
        .unwrap_or(1.0);
        let blend = match given(o, "blend") {
            None => Blend::Normal,
            Some(v) => match Blend::NAMES.iter().find(|(_, n)| v.as_str() == Some(n)) {
                Some((b, _)) => *b,
                None => bail!(
                    "{what} blend needs 'normal', 'add', 'multiply' or 'screen', got {}",
                    py_repr(v)
                ),
            },
        };
        Ok(Every {
            name,
            what,
            visible,
            x,
            y,
            opacity,
            blend,
        })
    }
}

/// The keys of every layer that leave their default, as a scene holds them,
/// `name` first, then `color` when it is Some.
pub(crate) fn every_dict(
    name: &str,
    visible: bool,
    x: Option<f64>,
    y: Option<f64>,
    color: Option<Value>,
    opacity: f64,
    blend: Blend,
) -> Map<String, Value> {
    let mut d = Map::new();
    d.insert("name".to_string(), Value::from(name));
    if !visible {
        d.insert("visible".to_string(), Value::from(false));
    }
    for (k, v) in [("x", x), ("y", y)] {
        if let Some(v) = v {
            d.insert(k.to_string(), Value::from(v));
        }
    }
    if let Some(c) = color {
        d.insert("color".to_string(), c);
    }
    if opacity != 1.0 {
        d.insert("opacity".to_string(), Value::from(opacity));
    }
    if blend != Blend::Normal {
        d.insert("blend".to_string(), Value::from(blend.name()));
    }
    d
}

/// Layer `d` as an object. `at` names it in the error.
pub(crate) fn object<'a>(at: &str, d: &'a Value) -> Result<&'a Map<String, Value>> {
    match d {
        Value::Object(o) => Ok(o),
        _ => bail!("{at} needs an object, got {}", py_repr(d)),
    }
}

/// Key `k` of `o`, None when it is missing or null.
pub(crate) fn given<'a>(o: &'a Map<String, Value>, k: &str) -> Option<&'a Value> {
    o.get(k).filter(|v| !v.is_null())
}

/// Key `k` of layer `o`, a finite number that `ok` takes, or None when it
/// is missing. `needs` says what it takes in the error.
pub(crate) fn number(
    o: &Map<String, Value>,
    what: &str,
    k: &str,
    ok: fn(f64) -> bool,
    needs: &str,
) -> Result<Option<f64>> {
    match given(o, k) {
        None => Ok(None),
        Some(v) => match v.as_f64() {
            Some(f) if f.is_finite() && ok(f) => Ok(Some(f)),
            _ => bail!("{what} {k} needs {needs}, got {}", py_repr(v)),
        },
    }
}

/// 1 for true and 0 for false, as a uniform takes a flag.
fn flag(b: bool) -> Value {
    Value::from(if b { 1.0 } else { 0.0 })
}

/// The uniforms that place a layer and show it, at their values.
fn placing(
    x: Option<f64>,
    y: Option<f64>,
    visible: bool,
    opacity: f64,
) -> Vec<(String, &'static str, Value)> {
    let name = GLSL_NAME;
    vec![
        (uniform(name, "x"), "float", Value::from(x.unwrap_or(0.0))),
        (uniform(name, "y"), "float", Value::from(y.unwrap_or(0.0))),
        (
            uniform(name, "centre"),
            "vec2",
            Value::from(vec![flag(x.is_none()), flag(y.is_none())]),
        ),
        (uniform(name, "visible"), "float", flag(visible)),
        (uniform(name, "opacity"), "float", Value::from(opacity)),
    ]
}

/// A text layer of a pass.
#[derive(Clone, Debug, PartialEq)]
pub struct TextLayer {
    pub name: String,
    /// False hides the layer. Its words still read through `<name>_cover`.
    pub visible: bool,
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub color: Color,
    pub opacity: f64,
    pub blend: Blend,
    /// The words, a live field.
    pub text: String,
    /// The pixels per em. None fits the block to its room.
    pub font_size: Option<f64>,
    /// A font asset of the scene, or a .ttf or .otf file resolved as an
    /// asset's file is. None draws from the built-in font.
    pub font_family: Option<String>,
    pub text_align: Align,
    pub render: Render,
    /// The layer's effects, in the order of Kind::ALL.
    pub effects: Vec<Effect>,
    /// The bytes of the layer's fonts, or why they failed to load, in the
    /// order that a character looks for its glyph: the font of
    /// `font_family`, or the built-in font. load_assets fills them, and
    /// from_dict for the built-in font.
    pub data: Option<Vec<Font>>,
    /// The font of the emoji that the layer's fonts lack, as load_assets
    /// found it. Words with a character that the layer's fonts lack, or one
    /// that U+FE0F follows, draw from it after the layer's fonts.
    pub emoji: Option<Arc<EmojiFont>>,
}

/// The vertex and the fragment shader of one batch of a layer.
pub type Program = (String, String);

/// The shaders of a layer's three batches.
pub struct Shaders {
    /// The glyphs.
    pub glyphs: Program,
    /// The error boxes, yellow and black hazard stripes in place of the
    /// glyphs that ghee cannot draw.
    pub boxes: Program,
    /// The colour glyphs, each from its bitmap.
    pub colour: Program,
}

impl TextLayer {
    /// `at` names the layer by its place in an error, as "the 2nd layer of
    /// pass 'main'", until its name is known.
    pub fn from_dict(at: &str, d: &Value) -> Result<TextLayer> {
        let o = object(at, d)?;
        let e = Every::from_dict(
            at,
            o,
            &TEXT_KEYS,
            &[IMAGE_KIND, shape::RECT_KIND, shape::LINE_KIND],
        )?;
        let what = e.what.as_str();
        let given = |k: &str| given(o, k);
        let number = |k: &str, ok: fn(f64) -> bool, needs: &str| number(o, what, k, ok, needs);
        let color = match given("color") {
            None => Color::Solid(Rgb::WHITE),
            Some(v) => Color::from_json(what, v)?,
        };
        let text = match given("text") {
            None => bail!(
                "{what} has no 'text' (the words that it draws) and no 'image' (an image asset \
                 or a file)"
            ),
            Some(Value::String(s)) => s.clone(),
            Some(v) => bail!("{what} text needs words in a string, got {}", py_repr(v)),
        };
        let font_size = number("font_size", |f| f > 0.0, "a number of pixels above 0")?;
        let font_family = match given("font_family") {
            None => None,
            Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
            Some(v) => bail!(
                "{what} font_family needs the name of a font asset or a .ttf or .otf file, got {}",
                py_repr(v)
            ),
        };
        let text_align = match given("text_align") {
            None => Align::Left,
            Some(v) => match Align::ALL.iter().find(|a| v.as_str() == Some(a.as_str())) {
                Some(a) => *a,
                None => bail!(
                    "{what} text_align needs 'left', 'center' or 'right', got {}",
                    py_repr(v)
                ),
            },
        };
        let render = match given("render").map(Value::as_str) {
            None => Render::default_for(font_family.as_deref()),
            Some(Some("mtsdf")) => Render::Mtsdf,
            Some(Some("slug")) => Render::Slug,
            Some(_) => bail!(
                "{what} render needs 'mtsdf' or 'slug', got {}",
                py_repr(&o["render"])
            ),
        };
        let mut effects = Vec::new();
        for kind in Kind::ALL {
            if let Some(d) = given(kind.key()) {
                effects.push(Effect::from_dict(
                    &format!("{what} {}", kind.key()),
                    kind,
                    d,
                )?);
            }
        }
        Ok(TextLayer {
            name: e.name,
            visible: e.visible,
            x: e.x,
            y: e.y,
            color,
            opacity: e.opacity,
            blend: e.blend,
            text,
            font_size,
            data: text::built_in(font_family.as_deref()).map(|f| vec![f]),
            emoji: None,
            font_family,
            text_align,
            render,
            effects,
        })
    }

    /// The layer named `name` that a drawing's text call with `args`, in
    /// the order of draw::TEXT_ARGS, gives `old`, as draw::layer_of() makes
    /// it.
    pub(crate) fn of_args(name: &str, args: &[Arg], old: &TextLayer) -> Option<TextLayer> {
        let [words, x, y, color, opacity, blend, visible, rest @ ..] = args else {
            return None;
        };
        let [font_size, font_family, text_align, render, effects @ ..] = rest else {
            return None;
        };
        let e = draw::Every::of_args(visible, x, y, opacity, blend)?;
        let Arg::Str(words) = words else { return None };
        let color = TextLayer::color_of(color)?;
        let font_family = match font_family {
            Arg::None => None,
            Arg::Str(s) if !s.is_empty() => Some(s.clone()),
            _ => return None,
        };
        let text_align = match text_align {
            Arg::None => Align::Left,
            Arg::Str(s) => *Align::ALL.iter().find(|a| a.as_str() == s)?,
            _ => return None,
        };
        let render = match render {
            Arg::None => Render::default_for(font_family.as_deref()),
            Arg::Str(s) if s == "mtsdf" => Render::Mtsdf,
            Arg::Str(s) if s == "slug" => Render::Slug,
            _ => return None,
        };
        let mut kept = Vec::new();
        for (kind, a) in Kind::ALL.into_iter().zip(effects) {
            match a {
                Arg::None => {
                    if let Some(e) = old.effect(kind) {
                        kept.push(Effect {
                            visible: false,
                            ..e.clone()
                        });
                    }
                }
                a => kept.push(Effect::of_arg(kind, a)?),
            }
        }
        Some(TextLayer {
            name: name.to_string(),
            visible: e.visible,
            x: e.x,
            y: e.y,
            color,
            opacity: e.opacity,
            blend: e.blend,
            text: words.clone(),
            font_size: draw::number(font_size, |f| f > 0.0)?,
            font_family,
            text_align,
            render,
            effects: kept,
            data: old.data.clone(),
            emoji: old.emoji.clone(),
        })
    }

    /// The colour that a drawing's call gives a text as `color`, white for
    /// None. None for a value that from_dict() refuses.
    fn color_of(color: &Arg) -> Option<Color> {
        Some(match color {
            Arg::None => Color::Solid(Rgb::WHITE),
            Arg::Map(_) => {
                let m = draw::map(color, &GRADIENT_KEYS)?;
                let end = |k: &str| match draw::get(m, k) {
                    Arg::None => None,
                    v => draw::rgb(v),
                };
                Color::Gradient {
                    from: end("from")?,
                    to: end("to")?,
                    angle: draw::number(draw::get(m, "angle"), |_| true)?.unwrap_or(GRADIENT_ANGLE),
                }
            }
            c => Color::Solid(draw::rgb(c)?),
        })
    }

    /// The layer as a scene holds it, a key at its default left out.
    pub fn to_dict(&self) -> Value {
        let white = matches!(self.color, Color::Solid(c) if c.same(Rgb::WHITE));
        let mut d = every_dict(
            &self.name,
            self.visible,
            self.x,
            self.y,
            (!white).then(|| self.color.to_json()),
            self.opacity,
            self.blend,
        );
        d.insert("text".to_string(), Value::from(self.text.clone()));
        if let Some(px) = self.font_size {
            d.insert("font_size".to_string(), Value::from(px));
        }
        if let Some(f) = &self.font_family {
            d.insert("font_family".to_string(), Value::from(f.clone()));
        }
        if self.text_align != Align::Left {
            d.insert(
                "text_align".to_string(),
                Value::from(self.text_align.as_str()),
            );
        }
        if self.render != Render::default_for(self.font_family.as_deref()) {
            d.insert("render".to_string(), Value::from(self.render.as_str()));
        }
        for e in &self.effects {
            d.insert(e.kind.key().to_string(), e.to_dict());
        }
        Value::Object(d)
    }

    /// Lay out `words` and pack them into a texture of curves. Fails when the
    /// fonts were never loaded.
    pub fn build(&self, words: &str) -> Result<Layout> {
        self.lay(words, true)
    }

    /// Lay out `words` with no texture of curves. Fails when the fonts were
    /// never loaded.
    pub fn place(&self, words: &str) -> Result<Layout> {
        self.lay(words, false)
    }

    /// Lay out `words` into the texture of curves `curves`, with the
    /// outlines and the shared texture of `glyphs`. Fails when the fonts
    /// were never loaded.
    pub fn lay_out(
        &self,
        words: &str,
        curves: text::Curves,
        glyphs: &mut text::Glyphs,
    ) -> Result<Layout> {
        let fonts = self.fonts(self.loaded()?, words);
        Ok(glyphs.lay(&fonts, words, self.text_align, curves))
    }

    fn lay(&self, words: &str, curves: bool) -> Result<Layout> {
        let fonts = self.fonts(self.loaded()?, words);
        Ok(match curves {
            true => text::build(&fonts, words, self.text_align),
            false => text::place(&fonts, words, self.text_align),
        })
    }

    /// The layer's own fonts, or why a layout has none.
    fn loaded(&self) -> Result<&[Font]> {
        match &self.data {
            Some(own) => Ok(own),
            None => bail!(
                "layer '{}': {} was never loaded, the scene was built without a directory",
                self.name,
                self.font_name(0)
            ),
        }
    }

    /// The fonts that draw `words`: the layer's own fonts `own`, then the
    /// emoji font for words with a character that `own` lacks or that U+FE0F
    /// follows. The emoji font loads at the first such words. A layer whose
    /// own fonts all failed to load takes no emoji font, and draws an error
    /// box for each character.
    fn fonts<'a>(&self, own: &'a [Font], words: &str) -> Cow<'a, [Font]> {
        let loaded = own.iter().any(Result::is_ok);
        match self.emoji.as_ref() {
            Some(emoji) if loaded && text::may_fall_back(own, words) => {
                Cow::Owned([own, &[emoji.font()]].concat())
            }
            _ => Cow::Borrowed(own),
        }
    }

    /// True while the emoji font that `words` need is read on a thread of
    /// its own, which the first such call starts.
    pub fn emoji_pending(&self, words: &str) -> bool {
        let Some(own) = self.data.as_deref() else {
            return false;
        };
        match self.emoji.as_ref() {
            Some(emoji) if own.iter().any(Result::is_ok) && text::may_fall_back(own, words) => {
                emoji.ready().is_none()
            }
            _ => false,
        }
    }

    /// The font at index `i` of the fonts that a layout of the layer draws
    /// from: the layer's own, then the emoji font. None for a font that
    /// failed to load.
    pub fn font(&self, i: usize) -> Option<Arc<Vec<u8>>> {
        let own = self.data.as_deref().unwrap_or_default();
        match own.get(i) {
            Some(f) => f.clone().ok(),
            None if i == own.len() => self.emoji.as_ref()?.font().ok(),
            None => None,
        }
    }

    /// The font at index `i` of the fonts that a layout of the layer draws
    /// from, in an error: "font 'sans'", "the built-in font", or "Noto Color
    /// Emoji".
    pub fn font_name(&self, i: usize) -> String {
        match (i, &self.font_family) {
            (0, Some(f)) => format!("font '{f}'"),
            (0, None) => "the built-in font".to_string(),
            _ => emoji::NAME.to_string(),
        }
    }

    /// True when a font of the layer that loaded holds outlines.
    pub fn has_outlines(&self) -> bool {
        self.data.as_deref().is_some_and(text::any_outlines)
    }

    /// ghee's name for the uniform `token` of this layer, in its shaders.
    pub fn uniform(&self, token: &str) -> String {
        uniform(GLSL_NAME, token)
    }

    /// The effect `kind` of the layer, present or bypassed.
    pub fn effect(&self, kind: Kind) -> Option<&Effect> {
        self.effects.iter().find(|e| e.kind == kind)
    }

    /// The effects that the layer draws, present or bypassed: under slug,
    /// none that reads the distance to the letters.
    fn drawn(&self) -> impl Iterator<Item = &Effect> {
        let slug = self.render == Render::Slug;
        self.effects
            .iter()
            .filter(move |e| !(slug && e.kind.reads_distance()))
    }

    /// The drawn effect `kind`.
    fn draws(&self, kind: Kind) -> Option<&Effect> {
        self.drawn().find(|e| e.kind == kind)
    }

    /// True when the layer draws an effect that reads the distance to the
    /// letters: its glyphs then draw twice, the effects of every glyph, then
    /// every glyph's letters.
    pub fn doubled(&self) -> bool {
        self.drawn().any(|e| e.kind.reads_distance())
    }

    /// How far past the letters the field of the atlas must reach for the
    /// drawn effects, in em: the shadow's blur, the glow's blur, the stroke's
    /// width. A bypassed effect counts, so its eye needs no new atlas.
    pub fn reach(&self) -> f64 {
        let mut reach: f64 = 0.0;
        for e in self.drawn() {
            reach = reach.max(match e.kind {
                Kind::TextShadow | Kind::Glow => e.num("blur"),
                Kind::TextStroke => e.num("width"),
                _ => 0.0,
            });
        }
        reach
    }

    /// How far past a glyph's box its quad reaches for the drawn effects, in
    /// em: the shadow's offset and blur, the glow's blur, the stroke's width.
    /// A bypassed effect counts.
    pub fn pad(&self) -> f64 {
        let mut pad: f64 = 0.0;
        for e in self.drawn() {
            pad = pad.max(match e.kind {
                Kind::TextShadow => e.num("x").abs().max(e.num("y").abs()) + e.num("blur"),
                Kind::Glow => e.num("blur"),
                Kind::TextStroke => e.num("width"),
                _ => 0.0,
            });
        }
        pad
    }

    /// The pixels per em of the layer when ghee builds it in a pass of `pass`
    /// pixels, its words a block of `block` em: its `font_size`, or the size
    /// that fits the block to its room, as the vertex stage works it out.
    pub fn font_px(&self, pass: [f64; 2], block: [f64; 2]) -> f64 {
        if let Some(px) = self.font_size {
            return px;
        }
        let m = FIT_MARGIN * pass[0].min(pass[1]);
        let room = |axis: usize, at: Option<f64>| match at {
            Some(at) => pass[axis] - m - at,
            None => pass[axis] - 2.0 * m,
        };
        let mut fit = [room(0, self.x), room(1, self.y)];
        // A scrolling block fits its room on y alone.
        if self.draws(Kind::Scroll).is_some_and(|e| e.visible) {
            fit[0] = f64::INFINITY;
        }
        let s = [0, 1].map(|a| fit[a] / block[a].max(1.0e-6));
        let px = s[0].min(s[1]).max(0.0);
        match self.grid() {
            Some(g) => ((px / g).floor() * g).max(g),
            None => px,
        }
    }

    /// The pixels per em of the grid that the layer's font draws on, for
    /// the pixel font. A fitted size then rounds down to a whole multiple of
    /// it, one grid at the least, and the block, each pen and each baseline
    /// land on whole pixels.
    pub(crate) fn grid(&self) -> Option<f64> {
        (self.font_family.as_deref() == Some(text::PIXEL)).then_some(text::PIXEL_EM)
    }

    /// The texels per em of the layer's atlas at `font_px` pixels per em: the
    /// first of ATLAS_STEPS at or above it, the last at most.
    pub fn atlas_px_em(font_px: f64) -> u32 {
        let last = ATLAS_STEPS[ATLAS_STEPS.len() - 1];
        ATLAS_STEPS
            .into_iter()
            .find(|s| f64::from(*s) >= font_px)
            .unwrap_or(last)
    }

    /// The atlas detail of the layer at `px_em` texels per em: a field that
    /// reaches the drawn effects and a texel more, and atlas::RANGE texels at
    /// least.
    pub fn detail(&self, px_em: u32) -> Detail {
        let range = (2.0 * (self.reach() * f64::from(px_em) + 1.0))
            .ceil()
            .min(f64::from(atlas::MAX_SIZE)) as u32;
        Detail {
            px_em,
            range: range.max(atlas::RANGE),
        }
    }

    /// `quads`, the glyphs of a layout, as the layer's glyph batch draws them:
    /// twice when the layer is doubled(), every effect first, then every
    /// letter, which the copy marks with 1 in the w of ghee_a3.
    pub fn copies(&self, quads: &[f32]) -> Vec<f32> {
        if !self.doubled() {
            return quads.to_vec();
        }
        let mut out = quads.to_vec();
        let at = out.len();
        out.extend_from_slice(quads);
        for v in out[at..].chunks_exact_mut(text::QUAD_FLOATS) {
            v[text::QUAD_FLOATS - 1] = 1.0;
        }
        out
    }

    /// Each uniform of the layer's own at its value from the scene, as its
    /// name, its type as a param takes it, and its value.
    pub fn uniforms(&self) -> Vec<(String, &'static str, Value)> {
        let mut out = placing(self.x, self.y, self.visible, self.opacity);
        self.colour_uniforms(&mut out);
        for e in self.drawn() {
            self.effect_uniforms(e, &mut out);
        }
        out
    }

    /// The uniforms of the layer's colour, into `out`.
    fn colour_uniforms(&self, out: &mut Vec<(String, &'static str, Value)>) {
        let (from, to, angle) = self.color.ends();
        out.extend([
            (
                self.uniform("color"),
                "vec3",
                Value::from(from.rgb.to_vec()),
            ),
            (
                self.uniform("colorto"),
                "vec3",
                Value::from(to.rgb.to_vec()),
            ),
            (self.uniform("angle"), "float", Value::from(angle)),
            (self.uniform("alpha"), "float", Value::from(from.alpha)),
            (self.uniform("alphato"), "float", Value::from(to.alpha)),
        ]);
    }

    /// The uniforms of `e`, an effect that the layer draws, into `out`.
    fn effect_uniforms(&self, e: &Effect, out: &mut Vec<(String, &'static str, Value)>) {
        let token = e.kind.token();
        out.push((
            self.uniform(&format!("{token}visible")),
            "float",
            flag(e.visible),
        ));
        for (f, v) in e.kind.fields().iter().zip(&e.values) {
            let name = self.uniform(&format!("{token}{}", f.name));
            match (f.change, v) {
                (Change::Rebuild | Change::Phase, _) => {}
                (Change::Uniform, FieldValue::Num(n)) => {
                    out.push((name, "float", Value::from(*n)));
                }
                (Change::Uniform, FieldValue::Rgb(c)) => {
                    out.push((name, "vec3", Value::from(c.rgb.to_vec())));
                    let alpha = self.uniform(&format!("{token}alpha"));
                    out.push((alpha, "float", Value::from(c.alpha)));
                }
            }
        }
    }

    /// Each effect that the layer draws and that moves by a phase, with the
    /// phase's uniform and its pace, what the phase gains a second.
    pub fn paces(&self) -> Vec<(Kind, String, f64)> {
        self.drawn()
            .filter_map(|e| {
                let pace = e.num(e.kind.pace()?);
                let token = format!("{}phase", e.kind.token());
                Some((e.kind, self.uniform(&token), pace))
            })
            .collect()
    }

    /// The phase at which effect `kind` draws as it does at 0, its words a
    /// block of `block` em in a pass of `pass` pixels: one wave, one blink,
    /// or one trip of the scroll, the pass's width and the block's in em.
    pub fn cycle(&self, kind: Kind, pass: [f64; 2], block: Option<[f64; 2]>) -> Option<f64> {
        match kind {
            Kind::Wave | Kind::Blink => Some(1.0),
            Kind::Scroll => {
                let block = block?;
                let trip = pass[0] / self.font_px(pass, block) + block[0];
                (trip.is_finite() && trip > 0.0).then_some(trip)
            }
            _ => None,
        }
    }

    /// The shaders of the layer's batches, for an MTSDF layer at `detail`.
    pub fn shaders(&self, detail: Detail) -> Shaders {
        let glyphs = match self.render {
            Render::Mtsdf => (
                self.vertex(Batch::Glyphs, detail),
                self.glsl(&self.mtsdf_fragment(detail)),
            ),
            Render::Slug => (self.vertex(Batch::Glyphs, detail), self.slug_fragment()),
        };
        let boxes = [PRECISION.join("\n"), text::box_fragment().join("\n")].join("\n");
        Shaders {
            glyphs,
            boxes: (self.vertex(Batch::Boxes, detail), boxes),
            colour: (
                self.vertex(Batch::Colour, detail),
                self.glsl(colour::FRAGMENT),
            ),
        }
    }

    /// The vertex shader of `batch`: the placement of the block, the motion
    /// of each glyph, and the batch's own varyings.
    fn vertex(&self, batch: Batch, detail: Detail) -> String {
        let mut decls: Vec<String> = Vec::new();
        let mut main: Vec<String> = Vec::new();
        let wave = self.draws(Kind::Wave);
        let scroll = self.draws(Kind::Scroll);
        let blink = self.draws(Kind::Blink);
        let karaoke = self.draws(Kind::Karaoke);
        // The motion effects' numbers and phases reach the vertex stage, and
        // every colour the fragment stage.
        for e in self.drawn().filter(|e| !e.kind.reads_distance()) {
            let token = e.kind.token();
            decls.push(format!("uniform float ghee_NAME_{token}visible;"));
            for f in e.kind.fields() {
                match (f.change, f.takes) {
                    (Change::Uniform, Takes::Colour) | (Change::Rebuild, _) => {}
                    (Change::Uniform, _) => {
                        decls.push(format!("uniform float ghee_NAME_{token}{};", f.name));
                    }
                    (Change::Phase, _) => {
                        decls.push(format!("uniform float ghee_NAME_{token}phase;"));
                    }
                }
            }
        }
        match self.font_size {
            Some(px) => main.push(format!("    float ghee_fs = {};", py_float(px))),
            None => {
                main.push(FIT.replace("FIT_MARGIN", &py_float(FIT_MARGIN)));
                // A scrolling block fits its room on y alone.
                if scroll.is_some() {
                    main.push(
                        "    ghee_s.x = mix(ghee_s.x, 1.0e6, ghee_NAME_scrollvisible);".to_string(),
                    );
                }
                main.push(format!("    float ghee_fs = {};", fitted(self.grid())));
            }
        }
        main.push(PLACE.to_string());
        if self.grid().is_some() {
            main.push(SNAP.to_string());
        }
        if scroll.is_some() {
            main.push(SCROLL_LINES.to_string());
        }
        let grow = match (batch, self.render) {
            (Batch::Glyphs, Render::Mtsdf) if self.doubled() => format!(
                "mix(min(0.625 / ghee_ppe, {:?}), {:?} + 1.5 / ghee_ppe, 1.0 - ghee_a3.w)",
                detail.reach(),
                self.pad()
            ),
            (Batch::Glyphs, Render::Mtsdf) => {
                format!("min(0.625 / ghee_ppe, {:?})", detail.reach())
            }
            (Batch::Colour, _) => {
                format!("min(0.625 / ghee_ppe, {:?} / ghee_a2.z)", colour::REACH)
            }
            _ => "0.625 / ghee_ppe".to_string(),
        };
        main.push(CORNER.replace("GROW", &grow));
        if let Some(w) = wave {
            main.push(format!(
                "    // Each letter trails the one before it by a tenth of a wave.\n    \
                 ghee_mv.y += {:?} * sin(6.2831853 * (ghee_NAME_wavephase - 0.1 * ghee_a0.z))\n    \
                 \x20         * ghee_NAME_wavevisible;",
                w.num("height")
            ));
        }
        if karaoke.is_some() {
            main.push(KARAOKE_LINES.to_string());
        }
        let (own, lines) = match (batch, self.render) {
            (Batch::Glyphs, Render::Mtsdf) => self.mtsdf_varyings(detail),
            (Batch::Glyphs, Render::Slug) => slug_varyings(true, karaoke.is_some()),
            (Batch::Boxes, _) => slug_varyings(false, false),
            (Batch::Colour, _) => colour_varyings(),
        };
        decls.push(own);
        main.push(lines);
        if self.grid().is_some() {
            main.push(pen(self.data.as_deref().unwrap_or_default()));
        }
        let mut hide = "ghee_NAME_visible < 0.5".to_string();
        if blink.is_some() {
            hide.push_str(
                " || (ghee_NAME_blinkvisible > 0.5 && fract(ghee_NAME_blinkphase) >= 0.5)",
            );
        }
        main.push(POSITION.replace("HIDE", &hide));
        let v = VERTEX
            .replace("DECLS", &decls.join("\n"))
            .replace("MAIN", &main.join("\n"));
        self.glsl(&v)
    }

    /// The varyings of an MTSDF layer's glyphs, and the lines that declare
    /// them: the atlas coordinate, the field's range in screen pixels, which
    /// msdfgen holds at 1 or more, the pixels per em, the point on the
    /// gradient, the karaoke fill, which copy the quad is, and its page. A doubled
    /// layer's quads also carry the box of the glyph's cell, the atlas
    /// coordinate of the shadow, and the pixels to each edge of the quad.
    fn mtsdf_varyings(&self, detail: Detail) -> (String, String) {
        let range_em = f64::from(detail.range) / f64::from(detail.px_em);
        let mut decls = vec![
            "uniform float ghee_NAME_angle;",
            "uniform vec2 ghee_NAME_atlas_size;",
            "uniform vec2 ghee_NAME_atlas2_size;",
            "varying vec4 ghee_v0;",
            "varying vec4 ghee_v1;",
        ];
        let kf = match self.draws(Kind::Karaoke) {
            Some(_) => "ghee_kf",
            None => "0.0",
        };
        let page = f64::from(atlas::MAX_SIZE);
        let mut lines = format!(
            "    // A glyph on the second page carries its texel x past the first\n    \
             // page's last texel.\n    \
             float ghee_pg = step({page:?}, ghee_a2.x);\n    \
             vec2 ghee_at = mix(ghee_NAME_atlas_size, ghee_NAME_atlas2_size, ghee_pg);\n    \
             vec2 ghee_ap = vec2(ghee_a2.x - {page:?} * ghee_pg, ghee_a2.y);\n    \
             vec2 ghee_tc = ghee_ap + (ghee_e - ghee_a1.xy) * ghee_a2.zw;\n\
             {GRADIENT}    ghee_v0 = vec4(ghee_tc / ghee_at, \
             max({range_em:?} * ghee_fs, 1.0), ghee_fs);\n    \
             ghee_v1 = vec4(ghee_t, {kf}, ghee_a3.w, ghee_pg);\n"
        );
        if self.doubled() {
            decls.extend([
                "varying vec4 ghee_v2;",
                "varying vec4 ghee_v3;",
                "varying vec4 ghee_v4;",
            ]);
            let half = f64::from(detail.range) / 2.0 + 0.5;
            let (sx, sy) = match self.draws(Kind::TextShadow) {
                Some(e) => (e.num("x"), e.num("y")),
                None => (0.0, 0.0),
            };
            lines.push_str(&format!(
                "    // A read stays inside the glyph's cell, which holds the field half\n    \
                 // the range past the ink and a texel more.\n    \
                 vec2 ghee_t1 = ghee_ap + (ghee_a1.zw - ghee_a1.xy) * ghee_a2.zw;\n    \
                 ghee_v2 = vec4(min(ghee_ap, ghee_t1) - {half:?}, max(ghee_ap, ghee_t1) + {half:?})\n    \
                 \x20         / ghee_at.xyxy;\n    \
                 // The shadow shows the letters' shape at the point less its offset.\n    \
                 ghee_v3 = vec4((ghee_tc - vec2({sx:?}, {sy:?}) * ghee_a2.zw) / ghee_at, \
                 0.0, 0.0);\n    \
                 vec2 ghee_q = (ghee_a1.zw - ghee_a1.xy + 2.0 * ghee_g) * ghee_fs;\n    \
                 ghee_v4 = vec4(ghee_a0.xy * ghee_q, (1.0 - ghee_a0.xy) * ghee_q);\n"
            ));
        }
        (decls.join("\n"), lines)
    }

    /// The fragment shader of an MTSDF layer's glyphs. A doubled layer draws
    /// its effects for a quad whose copy is 0, and its letters for 1.
    fn mtsdf_fragment(&self, detail: Detail) -> String {
        let range_em = f64::from(detail.range) / f64::from(detail.px_em);
        let mut decls: Vec<String> = vec![
            "uniform sampler2D ghee_NAME_atlas_tex;".to_string(),
            "uniform sampler2D ghee_NAME_atlas2_tex;".to_string(),
            "uniform vec3 ghee_NAME_color;".to_string(),
            "uniform vec3 ghee_NAME_colorto;".to_string(),
            "uniform float ghee_NAME_alpha;".to_string(),
            "uniform float ghee_NAME_alphato;".to_string(),
            "uniform float ghee_NAME_opacity;".to_string(),
            "varying vec4 ghee_v0;".to_string(),
            "varying vec4 ghee_v1;".to_string(),
        ];
        for e in self.drawn() {
            let token = e.kind.token();
            if !matches!(
                e.kind,
                Kind::TextShadow | Kind::Glow | Kind::TextStroke | Kind::Karaoke
            ) {
                continue;
            }
            decls.push(format!("uniform float ghee_NAME_{token}visible;"));
            decls.push(format!("uniform vec3 ghee_NAME_{token}color;"));
            decls.push(format!("uniform float ghee_NAME_{token}alpha;"));
            if matches!(e.kind, Kind::TextShadow | Kind::Glow) {
                decls.push(format!("uniform float ghee_NAME_{token}strength;"));
            }
        }
        let letters = format!(
            "        // The median of red, green and blue is the distance to the outline,\n        \
             // 0.5 on it and growing inward over the range.\n        \
             float ghee_d = max(min(ghee_t.r, ghee_t.g), min(max(ghee_t.r, ghee_t.g), ghee_t.b)) - 0.5;\n        \
             float ghee_c = clamp(ghee_v0.z * ghee_d + 0.5, 0.0, 1.0);\n{}        \
             ghee_o = vec4(ghee_rgb, ghee_al) * ghee_c;",
            self.letter_colour("        ", "ghee_v1.x", "ghee_v1.y")
        );
        let mut main = vec![
            "    vec4 ghee_t = vec4(0.0);".to_string(),
            "    vec4 ghee_o = vec4(0.0);".to_string(),
        ];
        if !self.doubled() {
            main.push(GUARDED_READ.replace("UV", "ghee_v0.xy"));
            main.push("    {".to_string());
            main.push(letters);
            main.push("    }".to_string());
        } else {
            decls.extend(
                [
                    "varying vec4 ghee_v2;",
                    "varying vec4 ghee_v3;",
                    "varying vec4 ghee_v4;",
                ]
                .map(str::to_string),
            );
            main.push(GUARDED_READ.replace("UV", "clamp(ghee_v0.xy, ghee_v2.xy, ghee_v2.zw)"));
            let shadow = self.draws(Kind::TextShadow);
            if shadow.is_some() {
                main.push("    vec4 ghee_ts = vec4(0.0);".to_string());
                main.push(
                    GUARDED_READ
                        .replace("ghee_t =", "ghee_ts =")
                        .replace("UV", "clamp(ghee_v3.xy, ghee_v2.xy, ghee_v2.zw)"),
                );
            }
            main.push("    if (ghee_v1.z > 0.5) {".to_string());
            main.push(letters);
            main.push("    } else {".to_string());
            main.push(format!(
                "        // The true distance into the letters in em, from the field's alpha,\n        \
                 // and the em of a pixel.\n        \
                 float ghee_in = (ghee_t.a - 0.5) * {range_em:?};\n        \
                 float ghee_px = 1.0 / max(ghee_v0.w, 1.0 / 16384.0);"
            ));
            if let Some(e) = shadow {
                main.push(format!(
                    "        // The shadow covers the shifted letters, softened over its blur on\n        \
                     // each side of their outline.\n        \
                     float ghee_sb = max({:?}, 0.5 * ghee_px);\n        \
                     float ghee_a = smoothstep(-ghee_sb, ghee_sb, (ghee_ts.a - 0.5) * {range_em:?})\n        \
                     \x20             * ghee_NAME_shadowstrength * ghee_NAME_shadowvisible\n        \
                     \x20             * ghee_NAME_shadowalpha;\n        \
                     ghee_o = vec4(ghee_NAME_shadowcolor, 1.0) * ghee_a;",
                    e.num("blur")
                ));
            }
            if let Some(e) = self.draws(Kind::Glow) {
                main.push(format!(
                    "        // The glow falls from the outline to nothing at its blur.\n        \
                     float ghee_f = clamp(1.0 - max(-ghee_in, 0.0) / {:?}, 0.0, 1.0);\n        \
                     float ghee_ga = ghee_f * ghee_f * ghee_NAME_glowstrength * ghee_NAME_glowvisible\n        \
                     \x20               * ghee_NAME_glowalpha;\n        \
                     ghee_o = vec4(ghee_NAME_glowcolor, 1.0) * ghee_ga + ghee_o * (1.0 - ghee_ga);",
                    e.num("blur").max(1.0e-6)
                ));
            }
            if let Some(e) = self.draws(Kind::TextStroke) {
                main.push(format!(
                    "        // The stroke grows the letters by its width, its edge antialiased\n        \
                     // over a pixel.\n        \
                     float ghee_sa = clamp((ghee_in + {:?}) * ghee_v0.w + 0.5, 0.0, 1.0)\n        \
                     \x20              * ghee_NAME_strokevisible * ghee_NAME_strokealpha;\n        \
                     ghee_o = vec4(ghee_NAME_strokecolor, 1.0) * ghee_sa + ghee_o * (1.0 - ghee_sa);",
                    e.num("width")
                ));
            }
            main.push(
                "        // Every effect fades out over the last pixel of its quad.\n        \
                 float ghee_edge = min(min(ghee_v4.x, ghee_v4.z), min(ghee_v4.y, ghee_v4.w));\n        \
                 ghee_o *= clamp(ghee_edge - 0.5, 0.0, 1.0);\n    }"
                    .to_string(),
            );
        }
        main.push("    gl_FragColor = ghee_o * ghee_NAME_opacity;".to_string());
        format!(
            "{}\n{}\nvoid main() {{\n{GUARD}{}\n}}",
            PRECISION.join("\n"),
            decls.join("\n"),
            main.join("\n")
        )
    }

    /// The lines that set ghee_rgb and ghee_al, the letters' colour and
    /// alpha at a pixel, after `indent`: the colour, or the point `t` on its
    /// gradient, and the karaoke colour where the fill `fill` reaches. A
    /// colour of alpha 0 holds black, so the mix of the colours is the mix
    /// of their premultiplied values.
    fn letter_colour(&self, indent: &str, t: &str, fill: &str) -> String {
        let mut l = gradient_colour(indent, t);
        if self.draws(Kind::Karaoke).is_some() {
            l.push_str(&format!(
                "{indent}float ghee_kt = clamp({fill}, 0.0, 1.0) * ghee_NAME_karaokevisible;\n\
                 {indent}ghee_rgb = mix(ghee_rgb, ghee_NAME_karaokecolor, ghee_kt);\n\
                 {indent}ghee_al += (ghee_NAME_karaokealpha - ghee_al) * ghee_kt;\n"
            ));
        }
        l
    }

    /// The fragment shader of a slug layer: the readers of the layer's curves,
    /// then ghee's main.
    fn slug_fragment(&self) -> String {
        let mut decls = vec![
            "uniform vec3 iResolution;",
            "uniform vec3 ghee_NAME_color;",
            "uniform vec3 ghee_NAME_colorto;",
            "uniform float ghee_NAME_alpha;",
            "uniform float ghee_NAME_alphato;",
            "uniform float ghee_NAME_opacity;",
            "varying vec2 uv;",
            "varying vec4 ghee_v0;",
            "varying vec4 ghee_v1;",
            "varying vec4 ghee_v4;",
        ];
        let karaoke = self.draws(Kind::Karaoke).is_some();
        if karaoke {
            decls.extend([
                "uniform float ghee_NAME_karaokevisible;",
                "uniform vec3 ghee_NAME_karaokecolor;",
                "uniform float ghee_NAME_karaokealpha;",
                "varying vec4 ghee_v2;",
            ]);
        }
        let body = SLUG_BODY.replace(
            "COLOUR\n",
            &self.letter_colour("    ", "ghee_v1.w", "ghee_v2.x"),
        );
        let main = format!("void main() {{\n{body}{SLUG_GUARD}}}");
        let mut f = vec![PRECISION.join("\n")];
        f.extend(text::curve_reader_lines(GLSL_NAME));
        f.push(self.glsl(&format!("{}\n{main}", decls.join("\n"))));
        f.join("\n")
    }

    /// `glsl` with the layer's name for NAME and the lattice for LATTICE.
    fn glsl(&self, glsl: &str) -> String {
        named(glsl)
    }
}

/// `glsl` with a layer's name in its own shaders for NAME and the lattice
/// for LATTICE.
fn named(glsl: &str) -> String {
    glsl.replace("LATTICE", &format!("{:?}", text::LATTICE))
        .replace("NAME", GLSL_NAME)
}

/// The main of a glyph of a slug text with no effects in each stage of a
/// run's program, which draws the glyph as the text's own program does:
/// the vertex stage's lines, with the font size in `fs`, 0 to fit the
/// block to its room, and `grid` true for a text on the pixel grid, and the
/// fragment stage's lines, which set gl_FragColor. Each uniform that the
/// text's own program reads is a local of the same name, which the caller
/// declares, and so are uv and ghee_v4.
pub(crate) fn run_glyph(fs: &str, grid: &str) -> (String, String) {
    let pixel = text::built_in(Some(text::PIXEL)).expect("ghee carries the pixel font");
    let vertex = [
        format!("    float ghee_fs = {fs};"),
        "    if (ghee_fs <= 0.0) {".to_string(),
        FIT.replace("FIT_MARGIN", &py_float(FIT_MARGIN)),
        format!(
            "    if ({grid}) ghee_fs = {};",
            fitted(Some(text::PIXEL_EM))
        ),
        format!("    else ghee_fs = {};", fitted(None)),
        "    }".to_string(),
        PLACE.to_string(),
        format!("    if ({grid}) {}", SNAP.trim_start()),
        CORNER.replace("GROW", "0.625 / ghee_ppe"),
        slug_varyings(true, false).1.trim_end().to_string(),
        format!("    if ({grid}) {{\n{}\n    }}", pen(&[pixel])),
        POSITION.replace("HIDE", "ghee_NAME_visible < 0.5"),
    ];
    let fragment = SLUG_BODY.replace("COLOUR\n", &gradient_colour("    ", "ghee_v1.w"));
    (named(&vertex.join("\n")), named(&fragment))
}

/// The last lines of the main of a fragment stage that draws slug glyphs,
/// which read the texture of curves at `at`.
pub(crate) fn run_guard(at: &str) -> String {
    named(&SLUG_GUARD.replace("_curves_tex, uv)", &format!("_curves_tex, {at})")))
}

/// An image layer of a pass.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageLayer {
    pub name: String,
    /// False hides the layer.
    pub visible: bool,
    pub x: Option<f64>,
    pub y: Option<f64>,
    /// The tint that multiplies the picture. None leaves it as it is.
    pub color: Option<Rgb>,
    pub opacity: f64,
    pub blend: Blend,
    /// An image asset of the scene, or a .png or .jpg file resolved as an
    /// asset's file is.
    pub image: String,
    /// The pass pixels that the picture spans. One of them missing keeps the
    /// image's aspect, and both missing take its own size.
    pub width: Option<f64>,
    pub height: Option<f64>,
    /// The pixels of a file `image`. load_assets fills them.
    pub data: Option<crate::assets::Image>,
}

impl ImageLayer {
    /// `at` names the layer by its place in an error, as "the 2nd layer of
    /// pass 'main'", until its name is known.
    pub fn from_dict(at: &str, d: &Value) -> Result<ImageLayer> {
        let o = object(at, d)?;
        let e = Every::from_dict(
            at,
            o,
            &IMAGE_KEYS,
            &[TEXT_KIND, shape::RECT_KIND, shape::LINE_KIND],
        )?;
        let what = e.what.as_str();
        let color = match given(o, "color") {
            None => None,
            Some(v) => match Rgb::from_json(v) {
                Some(c) => Some(c),
                None if v.is_object() => {
                    bail!("{what} color takes no gradient on an image layer, only {COLOUR_FORMS}")
                }
                None => bail!("{what} color needs {COLOUR_FORMS}, got {}", py_repr(v)),
            },
        };
        let image = match given(o, "image") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            v => bail!(
                "{what} image needs the name of an image asset or a .png or .jpg file, got {}",
                py_repr(v.unwrap_or(&Value::Null))
            ),
        };
        let px = |k: &str| number(o, what, k, |f| f > 0.0, "a number of pixels above 0");
        Ok(ImageLayer {
            name: e.name,
            visible: e.visible,
            x: e.x,
            y: e.y,
            color,
            opacity: e.opacity,
            blend: e.blend,
            image,
            width: px("width")?,
            height: px("height")?,
            data: None,
        })
    }

    /// The layer named `name` that a drawing's image call with `args`, in
    /// the order of draw::IMAGE_ARGS, gives `old`, as draw::layer_of() makes
    /// it.
    pub(crate) fn of_args(name: &str, args: &[Arg], old: &ImageLayer) -> Option<ImageLayer> {
        let [image, x, y, width, height, color, opacity, blend, visible] = args else {
            return None;
        };
        let e = draw::Every::of_args(visible, x, y, opacity, blend)?;
        let image = match image {
            Arg::Str(s) if !s.is_empty() => s.clone(),
            _ => return None,
        };
        let color = match color {
            Arg::None => None,
            c => Some(draw::rgb(c)?),
        };
        let data = match image == old.image {
            true => old.data.clone(),
            false => None,
        };
        Some(ImageLayer {
            name: name.to_string(),
            visible: e.visible,
            x: e.x,
            y: e.y,
            color,
            opacity: e.opacity,
            blend: e.blend,
            image,
            width: draw::number(width, |f| f > 0.0)?,
            height: draw::number(height, |f| f > 0.0)?,
            data,
        })
    }

    /// The layer as a scene holds it, a key at its default left out.
    pub fn to_dict(&self) -> Value {
        let mut d = every_dict(
            &self.name,
            self.visible,
            self.x,
            self.y,
            self.color.map(Rgb::to_json),
            self.opacity,
            self.blend,
        );
        d.insert("image".to_string(), Value::from(self.image.clone()));
        for (k, v) in [("width", self.width), ("height", self.height)] {
            if let Some(v) = v {
                d.insert(k.to_string(), Value::from(v));
            }
        }
        Value::Object(d)
    }

    /// True when `image` names a file of the layer's own, not an asset.
    pub fn is_file(&self) -> bool {
        is_image_file(&self.image)
    }

    /// Each uniform of the layer's own at its value from the scene, as
    /// TextLayer::uniforms() lists them. No tint is white.
    pub fn uniforms(&self) -> Vec<(String, &'static str, Value)> {
        let tint = self.color.unwrap_or(Rgb::WHITE);
        let mut out = placing(self.x, self.y, self.visible, self.opacity);
        out.push((
            uniform(GLSL_NAME, "color"),
            "vec3",
            Value::from(tint.rgb.to_vec()),
        ));
        out.push((
            uniform(GLSL_NAME, "alpha"),
            "float",
            Value::from(tint.alpha),
        ));
        out
    }

    /// The vertex and the fragment shader of the layer's quad.
    pub fn shaders(&self) -> Program {
        let size = match (self.width, self.height) {
            (Some(w), Some(h)) => format!("vec2({w:?}, {h:?})"),
            (Some(w), None) => format!(
                "vec2({w:?}, {w:?} * ghee_NAME_image_size.y / max(ghee_NAME_image_size.x, 1.0))"
            ),
            (None, Some(h)) => format!(
                "vec2({h:?} * ghee_NAME_image_size.x / max(ghee_NAME_image_size.y, 1.0), {h:?})"
            ),
            (None, None) => "ghee_NAME_image_size".to_string(),
        };
        let vertex = IMAGE_VERTEX.replace("SIZE", &size);
        let fragment = IMAGE_FRAGMENT.replace("GUARD", GUARD);
        (
            vertex.replace("NAME", GLSL_NAME),
            fragment.replace("NAME", GLSL_NAME),
        )
    }
}

/// True when `image` names a .png or a .jpg file, where an asset's name
/// holds no `.`.
pub fn is_image_file(image: &str) -> bool {
    let f = image.to_ascii_lowercase();
    [".png", ".jpg", ".jpeg"].iter().any(|e| f.ends_with(e))
}

/// A layer of a pass, which paints its words, its picture or its shape over
/// the pass's buffer.
#[derive(Clone, Debug, PartialEq)]
pub enum Layer {
    Text(TextLayer),
    Image(ImageLayer),
    Shape(ShapeLayer),
}

impl Layer {
    /// A shape layer when `d` holds a `shape`, else an image layer when it
    /// holds an `image`, else a text layer. `at` names the layer by its
    /// place in an error, as "the 2nd layer of pass 'main'", until its name
    /// is known.
    pub fn from_dict(at: &str, d: &Value) -> Result<Layer> {
        let o = object(at, d)?;
        if given(o, "shape").is_some() {
            return ShapeLayer::from_dict(at, d).map(Layer::Shape);
        }
        match given(o, "image") {
            Some(_) => ImageLayer::from_dict(at, d).map(Layer::Image),
            None => TextLayer::from_dict(at, d).map(Layer::Text),
        }
    }

    /// The layer as a scene holds it.
    pub fn to_dict(&self) -> Value {
        match self {
            Layer::Text(l) => l.to_dict(),
            Layer::Image(l) => l.to_dict(),
            Layer::Shape(l) => l.to_dict(),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Layer::Text(l) => &l.name,
            Layer::Image(l) => &l.name,
            Layer::Shape(l) => &l.name,
        }
    }

    pub fn blend(&self) -> Blend {
        match self {
            Layer::Text(l) => l.blend,
            Layer::Image(l) => l.blend,
            Layer::Shape(l) => l.blend,
        }
    }

    /// Each uniform of the layer's own at its value from the scene, as its
    /// name, its type as a param takes it, and its value. A shape layer has
    /// none.
    pub fn uniforms(&self) -> Vec<(String, &'static str, Value)> {
        match self {
            Layer::Text(l) => l.uniforms(),
            Layer::Image(l) => l.uniforms(),
            Layer::Shape(_) => Vec::new(),
        }
    }

    /// Each uniform of `new`, a new look of this layer, whose value differs
    /// from this layer's, as uniforms() lists them.
    pub fn uniforms_changed(&self, new: &Layer) -> Vec<(String, &'static str, Value)> {
        let differ = |before: Vec<(String, &'static str, Value)>,
                      after: Vec<(String, &'static str, Value)>| {
            after
                .into_iter()
                .filter(|(n, _, v)| !before.iter().any(|(m, _, w)| m == n && w == v))
                .collect::<Vec<_>>()
        };
        // A text whose effects keep their kinds compares its colour and each
        // effect apart.
        let mut parts = Vec::new();
        let placed_alone = match (self, new) {
            (Layer::Text(a), Layer::Text(b))
                if a.render == b.render
                    && a.effects.len() == b.effects.len()
                    && a.effects
                        .iter()
                        .zip(&b.effects)
                        .all(|(x, y)| x.kind == y.kind) =>
            {
                if a.color != b.color {
                    let (mut was, mut will) = (Vec::new(), Vec::new());
                    a.colour_uniforms(&mut was);
                    b.colour_uniforms(&mut will);
                    parts.extend(differ(was, will));
                }
                for (x, y) in a.drawn().zip(b.drawn()).filter(|(x, y)| x != y) {
                    let (mut was, mut will) = (Vec::new(), Vec::new());
                    a.effect_uniforms(x, &mut was);
                    b.effect_uniforms(y, &mut will);
                    parts.extend(differ(was, will));
                }
                true
            }
            (Layer::Image(a), Layer::Image(b)) => a.color == b.color,
            (Layer::Shape(_), Layer::Shape(_)) => return Vec::new(),
            _ => false,
        };
        if !placed_alone {
            return differ(self.uniforms(), new.uniforms());
        }
        let placed = |l: &Layer| match l {
            Layer::Text(l) => (l.x, l.y, l.visible, l.opacity),
            Layer::Image(l) => (l.x, l.y, l.visible, l.opacity),
            Layer::Shape(l) => (l.x, l.y, l.visible, l.opacity),
        };
        let ((x, y, visible, opacity), (nx, ny, nvisible, nopacity)) = (placed(self), placed(new));
        let at = |v: Option<f64>| v.unwrap_or(0.0);
        let mut out = Vec::new();
        if at(x) != at(nx) {
            out.push((uniform(GLSL_NAME, "x"), "float", Value::from(at(nx))));
        }
        if at(y) != at(ny) {
            out.push((uniform(GLSL_NAME, "y"), "float", Value::from(at(ny))));
        }
        if (x.is_none(), y.is_none()) != (nx.is_none(), ny.is_none()) {
            let centre = vec![flag(nx.is_none()), flag(ny.is_none())];
            out.push((uniform(GLSL_NAME, "centre"), "vec2", Value::from(centre)));
        }
        if visible != nvisible {
            out.push((uniform(GLSL_NAME, "visible"), "float", flag(nvisible)));
        }
        if opacity != nopacity {
            out.push((
                uniform(GLSL_NAME, "opacity"),
                "float",
                Value::from(nopacity),
            ));
        }
        out.extend(parts);
        out
    }

    /// True when `other`, a new look of this layer, differs from it in live
    /// fields alone, as Scene::look() tells a live change from one that
    /// builds the layer again: the words and an image layer's image are
    /// live.
    pub(crate) fn builds_as(&self, other: &Layer) -> bool {
        if self.blend() != other.blend() {
            return false;
        }
        let rebuilds = |a: &Effect, b: &Effect| {
            let fields = a.kind.fields().iter().zip(a.values.iter().zip(&b.values));
            a.kind != b.kind
                || fields
                    .filter(|(f, _)| f.change == Change::Rebuild)
                    .any(|(_, (x, y))| x != y)
        };
        match (self, other) {
            (Layer::Text(a), Layer::Text(b)) => {
                a.font_size == b.font_size
                    && a.font_family == b.font_family
                    && a.text_align == b.text_align
                    && a.render == b.render
                    && a.effects.len() == b.effects.len()
                    && !a
                        .effects
                        .iter()
                        .zip(&b.effects)
                        .any(|(x, y)| rebuilds(x, y))
            }
            (Layer::Image(a), Layer::Image(b)) => a.width == b.width && a.height == b.height,
            (Layer::Shape(a), Layer::Shape(b)) => a.same_shape(b),
            _ => false,
        }
    }

    pub fn as_text(&self) -> Option<&TextLayer> {
        match self {
            Layer::Text(l) => Some(l),
            _ => None,
        }
    }

    pub fn as_text_mut(&mut self) -> Option<&mut TextLayer> {
        match self {
            Layer::Text(l) => Some(l),
            _ => None,
        }
    }

    pub fn as_image(&self) -> Option<&ImageLayer> {
        match self {
            Layer::Image(l) => Some(l),
            _ => None,
        }
    }

    pub fn as_shape(&self) -> Option<&ShapeLayer> {
        match self {
            Layer::Shape(l) => Some(l),
            _ => None,
        }
    }
}

/// The address of a live field: the layer's name, the effect for a field of
/// one of its effects, and the field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Address<'a> {
    pub layer: &'a str,
    pub effect: Option<&'a str>,
    pub field: &'a str,
}

impl<'a> Address<'a> {
    /// `layer.field` or `layer.effect.field`, as the wire names a live field,
    /// or None for a name of another shape.
    pub fn dotted(name: &'a str) -> Option<Address<'a>> {
        let parts: Vec<&str> = name.split('.').collect();
        match parts[..] {
            [layer, field] => Some(Address {
                layer,
                effect: None,
                field,
            }),
            [layer, effect, field] => Some(Address {
                layer,
                effect: Some(effect),
                field,
            }),
            _ => None,
        }
    }
}

/// The form of the value that a live field takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// A number. With `none`, none takes the field back to its default.
    Number { none: bool },
    /// true or false.
    Flag,
    /// A colour, as Rgb::from_json() takes it, a gradient object too with
    /// `gradient`. With `none`, none takes the field back to its default.
    Colour { gradient: bool, none: bool },
    /// The name of an image asset of the scene.
    Image,
    /// A line's points, each [x, y], two or more.
    Points,
    /// A text layer's words.
    Words,
}

impl Form {
    /// True for a field that none takes back to its default.
    fn takes_none(self) -> bool {
        matches!(
            self,
            Form::Number { none: true } | Form::Colour { none: true, .. }
        )
    }

    /// What a value of the form is, as an error says it.
    fn needs(self) -> &'static str {
        match self {
            Form::Number { .. } => "a number",
            Form::Flag => "true or false",
            Form::Colour { .. } => COLOUR_FORMS,
            Form::Image => "the name of an image asset",
            Form::Points => "two points or more, each [x, y]",
            Form::Words => "words in a string",
        }
    }
}

/// A live field of a layer: `what` names it in an error, as "layer 't'
/// karaoke sung", and `form` is the form of its value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Live {
    pub what: String,
    pub form: Form,
}

impl Live {
    /// The value of the numbers `values` that the wire gives the field: one
    /// for a number, 0 or 1 for a flag, three for a colour, seven for a
    /// gradient, from, to and angle, and no value for none.
    pub fn of_numbers(&self, values: &[f32]) -> Result<Value> {
        let n: Vec<f64> = values.iter().map(|v| f64::from(*v)).collect();
        let what = &self.what;
        Ok(match (self.form, &n[..]) {
            (Form::Number { none: true } | Form::Colour { none: true, .. }, []) => Value::Null,
            (Form::Number { .. }, [v]) => Value::from(*v),
            (Form::Flag, [v]) => Value::from(*v != 0.0),
            (Form::Colour { .. }, [r, g, b]) => Value::from(vec![*r, *g, *b]),
            (Form::Colour { gradient: true, .. }, [r, g, b, r2, g2, b2, angle]) => {
                let mut d = Map::new();
                d.insert("from".to_string(), Value::from(vec![*r, *g, *b]));
                d.insert("to".to_string(), Value::from(vec![*r2, *g2, *b2]));
                d.insert("angle".to_string(), Value::from(*angle));
                Value::Object(d)
            }
            (Form::Image, _) => bail!("{what} takes the name of an image asset, got numbers"),
            (Form::Words, _) => bail!("{what} takes text, got numbers"),
            (Form::Points, n) if n.len() >= 4 && n.len() % 2 == 0 => {
                Value::from(n.chunks_exact(2).map(<[f64]>::to_vec).collect::<Vec<_>>())
            }
            (Form::Points, _) => bail!(
                "{what} takes x and y in turn for two points or more, got {} numbers",
                n.len()
            ),
            (Form::Number { none }, _) => {
                let or = if none { " or 0 for its default" } else { "" };
                bail!("{what} takes 1 number{or}, got {}", n.len())
            }
            (Form::Flag, _) => bail!("{what} takes 1 number, 0 or 1, got {}", n.len()),
            (Form::Colour { gradient, none }, _) => {
                let more = match (gradient, none) {
                    (true, true) => ", 7 for a gradient or 0 for its default",
                    (true, false) => " or 7 for a gradient",
                    (false, true) => " or 0 for its default",
                    (false, false) => "",
                };
                bail!("{what} takes 3 numbers{more}, got {}", n.len())
            }
        })
    }

    /// The value of the text `text` that the wire gives the field: the name
    /// of an image asset, a text layer's words, or a colour, `#rrggbb` or a
    /// CSS colour name, or a text layer's gradient as its JSON object.
    pub fn of_text(&self, text: &str) -> Result<Value> {
        match self.form {
            Form::Colour { gradient: true, .. } if text.trim_start().starts_with('{') => {
                Ok(serde_json::from_str(text).unwrap_or_else(|_| Value::from(text)))
            }
            Form::Image | Form::Colour { .. } | Form::Words => Ok(Value::from(text)),
            _ => bail!("{} takes numbers, got text", self.what),
        }
    }
}

/// The layer format that the console edits, for GET /format: the fields of
/// each effect in order, with their defaults and whether a /set changes them
/// with no rebuild, the same for a rect's border, the live fields of each kind
/// of layer, what an empty field of each kind takes, the keys of each shape,
/// the built-in fonts, each name the console shows with the `font_family` that
/// it writes, and the render that a text layer takes in each font.
pub fn format() -> Value {
    let mut effects = Map::new();
    for kind in Kind::ALL {
        let fields: Vec<Value> = kind
            .fields()
            .iter()
            .map(|f| {
                let default = match f.default {
                    FieldValue::Num(n) => Value::from(n),
                    FieldValue::Rgb(c) => c.to_json(),
                };
                serde_json::json!({"key": f.name, "default": default,
                                   "live": f.change != Change::Rebuild})
            })
            .collect();
        effects.insert(kind.key().to_string(), Value::from(fields));
    }
    let samples = [
        ("text", serde_json::json!({"name": "a", "text": "a"})),
        ("image", serde_json::json!({"name": "a", "image": "a.png"})),
        ("rect", serde_json::json!({"name": "a", "shape": "rect"})),
        (
            "line",
            serde_json::json!({"name": "a", "shape": "line", "points": [[0, 0], [1, 1]]}),
        ),
    ];
    let mut live = Map::new();
    for (kind, d) in samples {
        let layer = Layer::from_dict("a layer", &d).expect("a sample layer parses");
        let mut keys: Vec<&str> = Vec::new();
        for key in layer_keys() {
            if key != "name" && !keys.contains(&key) && layer.live(None, key).is_ok() {
                keys.push(key);
            }
        }
        live.insert(kind.to_string(), Value::from(keys));
    }
    let centre = serde_json::json!({"x": "centre", "y": "centre"});
    let with = |more: Value| {
        let mut d = centre.clone();
        if let (Value::Object(d), Value::Object(m)) = (&mut d, more) {
            d.extend(m);
        }
        d
    };
    let empty = serde_json::json!({
        "text": with(serde_json::json!({"font_size": "fit", "render": "auto"})),
        "image": with(serde_json::json!({"width": "auto", "height": "auto"})),
        "rect": with(serde_json::json!({"width": "pass", "height": "pass",
                                        "border_radius": shape::BORDER_RADIUS})),
        "line": {"x": 0, "y": 0, "stroke_width": shape::STROKE_WIDTH},
    });
    let rect = Layer::from_dict(
        "a layer",
        &serde_json::json!({"name": "a", "shape": "rect"}),
    )
    .expect("a sample rect parses");
    let color = Rgb {
        written: Written::Hex,
        ..shape::BORDER_COLOR
    };
    let border: Vec<Value> = [
        ("width", Value::from(shape::BORDER_WIDTH)),
        ("color", color.to_json()),
    ]
    .into_iter()
    .map(|(key, default)| {
        let live = rect.live(Some("border"), key).is_ok();
        serde_json::json!({"key": key, "default": default, "live": live})
    })
    .collect();
    let own = |keys: &[&str]| -> Vec<String> {
        keys.iter()
            .filter(|k| **k != "shape")
            .map(|k| (*k).to_string())
            .collect()
    };
    let shapes =
        serde_json::json!({"rect": own(&shape::RECT_KEYS), "line": own(&shape::LINE_KEYS)});
    let mut fonts = Map::new();
    fonts.insert("built-in".to_string(), Value::Null);
    fonts.insert(text::PIXEL.to_string(), Value::from(text::PIXEL));
    let plain = Render::default_for(None);
    let mut renders = Map::new();
    let pixel = Render::default_for(Some(text::PIXEL));
    if pixel != plain {
        renders.insert(text::PIXEL.to_string(), Value::from(pixel.as_str()));
    }
    let render = serde_json::json!({"default": plain.as_str(), "fonts": renders});
    serde_json::json!({"effects": effects, "border": border, "live": live, "empty": empty,
                       "shapes": shapes, "fonts": fonts, "render": render})
}

/// `items` as a sentence names them: "a, b and c".
pub(crate) fn spoken(items: &[&str]) -> String {
    match items.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
        _ => items.join(""),
    }
}

impl Layer {
    /// The live field `field` of the layer, or of its effect `effect`, or
    /// why the layer takes no live set there: it has no such effect or
    /// field, or the field changes with a rebuild.
    pub fn live(&self, effect: Option<&str>, field: &str) -> Result<Live> {
        let (name, kind, own) = match self {
            Layer::Text(l) => (&l.name, "a text layer", &TEXT_KEYS[..]),
            Layer::Image(l) => (&l.name, "an image layer", &IMAGE_KEYS[..]),
            Layer::Shape(l) => return l.live(effect, field),
        };
        let layer = format!("layer '{name}'");
        if let Some(key) = effect {
            let e = self
                .as_text()
                .and_then(|l| l.effects.iter().find(|e| e.kind.key() == key));
            let Some(e) = e else {
                let effects: Vec<&str> = match self {
                    Layer::Text(l) => l.effects.iter().map(|e| e.kind.key()).collect(),
                    Layer::Shape(_) => unreachable!("a shape layer answers for itself"),
                    Layer::Image(_) => {
                        bail!("{layer} has no effect '{key}', and an image layer takes no effects")
                    }
                };
                match effects[..] {
                    [] => bail!("{layer} has no effect '{key}' and no other effect"),
                    [one] => bail!("{layer} has no effect '{key}', and its effect is {one}"),
                    _ => bail!(
                        "{layer} has no effect '{key}', and its effects are {}",
                        spoken(&effects)
                    ),
                }
            };
            let fields = e.kind.fields();
            let live: Vec<&str> = fields
                .iter()
                .filter(|f| f.change != Change::Rebuild)
                .map(|f| f.name)
                .chain(["visible"])
                .collect();
            let form = match fields.iter().find(|f| f.name == field) {
                _ if field == "visible" => Form::Flag,
                None => bail!(
                    "{layer} {key} has no field '{field}', and the live fields of {key} are {}",
                    spoken(&live)
                ),
                Some(f) if f.change == Change::Rebuild => bail!(
                    "{layer} {key} {field} changes with a rebuild, and the live fields of {key} \
                     are {}",
                    spoken(&live)
                ),
                Some(f) => match f.takes {
                    Takes::Colour => Form::Colour {
                        gradient: false,
                        none: false,
                    },
                    _ => Form::Number { none: false },
                },
            };
            return Ok(Live {
                what: format!("{layer} {key} {field}"),
                form,
            });
        }
        let image = self.as_image().is_some();
        let mut live = LIVE_KEYS.to_vec();
        live.push(if image { "image" } else { "text" });
        let form = match field {
            "x" | "y" => Form::Number { none: true },
            "opacity" => Form::Number { none: false },
            "visible" => Form::Flag,
            "color" => Form::Colour {
                gradient: !image,
                none: true,
            },
            "image" if image => Form::Image,
            "text" if !image => Form::Words,
            _ if EVERY_KEYS.contains(&field) || own.contains(&field) => bail!(
                "{layer} {field} changes with a rebuild, and the live fields of {kind} are {}",
                spoken(&live)
            ),
            _ => bail!(
                "{layer} has no field '{field}', and the live fields of {kind} are {}",
                spoken(&live)
            ),
        };
        Ok(Live {
            what: format!("{layer} {field}"),
            form,
        })
    }

    /// Set the live field `field` of the layer, or of its effect `effect`,
    /// to `value`, written as the scene format writes the key. Null takes a
    /// field that takes none back to its default. The value goes through the
    /// checks of a layer that a scene declares, and the layer keeps its
    /// fonts, and the pixels of its file while its image stays.
    pub fn set_live(&mut self, effect: Option<&str>, field: &str, value: &Value) -> Result<()> {
        let live = self.live(effect, field)?;
        if value.is_null() && !live.form.takes_none() {
            bail!("{} needs {}, got None", live.what, live.form.needs());
        }
        if self.set_uniform(effect, field, &Arg::of_json(value)) {
            return Ok(());
        }
        let at = format!("layer '{}'", self.name());
        let mut d = match self {
            Layer::Text(l) => l.to_dict(),
            Layer::Image(l) => l.to_dict(),
            Layer::Shape(l) => return l.set_live(effect, field, value),
        };
        match effect {
            Some(key) => d[key][field] = value.clone(),
            None => d[field] = value.clone(),
        }
        match self {
            Layer::Text(l) => {
                let mut new = TextLayer::from_dict(&at, &d)?;
                new.data = l.data.take();
                new.emoji = l.emoji.take();
                *l = new;
            }
            Layer::Image(l) => {
                let mut new = ImageLayer::from_dict(&at, &d)?;
                if new.image == l.image {
                    new.data = l.data.take();
                }
                *l = new;
            }
            Layer::Shape(_) => unreachable!("a shape layer sets its own fields"),
        }
        Ok(())
    }

    /// Set a field of a text or an image layer that sets a uniform alone, or
    /// a live field of a text's effect, to `v`, read as a drawing's call
    /// reads its keyword. False for any other field, and for a value that
    /// the call refuses, which leave the layer as it was: from_dict() then
    /// takes the value, or says why it refuses it.
    fn set_uniform(&mut self, effect: Option<&str>, field: &str, v: &Arg) -> bool {
        let (visible, x, y, opacity) = match self {
            Layer::Text(l) => (&mut l.visible, &mut l.x, &mut l.y, &mut l.opacity),
            Layer::Image(l) => (&mut l.visible, &mut l.x, &mut l.y, &mut l.opacity),
            Layer::Shape(_) => return false,
        };
        let set = |slot: &mut Option<f64>| match draw::number(v, |_| true) {
            Some(n) => {
                *slot = n;
                true
            }
            None => false,
        };
        match (effect, field) {
            (Some(key), _) => match self {
                Layer::Text(l) => l
                    .effects
                    .iter_mut()
                    .find(|e| e.kind.key() == key)
                    .is_some_and(|e| e.set_uniform(field, v)),
                _ => false,
            },
            (None, "x") => set(x),
            (None, "y") => set(y),
            (None, "visible") => match draw::flag(v, true) {
                Some(b) => {
                    *visible = b;
                    true
                }
                None => false,
            },
            (None, "opacity") => match draw::number(v, |f| (0.0..=1.0).contains(&f)) {
                Some(n) => {
                    *opacity = n.unwrap_or(1.0);
                    true
                }
                None => false,
            },
            (None, "color") => match self {
                Layer::Text(l) => match TextLayer::color_of(v) {
                    Some(c) => {
                        l.color = c;
                        true
                    }
                    None => false,
                },
                Layer::Image(l) => match v {
                    Arg::None => {
                        l.color = None;
                        true
                    }
                    c => match draw::rgb(c) {
                        Some(c) => {
                            l.color = Some(c);
                            true
                        }
                        None => false,
                    },
                },
                Layer::Shape(_) => false,
            },
            _ => false,
        }
    }

    /// The layer with no fonts and no pixels, which compares its uniforms
    /// as the layer does.
    pub(crate) fn lean(&self) -> Layer {
        match self {
            Layer::Text(l) => Layer::Text(TextLayer {
                name: l.name.clone(),
                visible: l.visible,
                x: l.x,
                y: l.y,
                color: l.color,
                opacity: l.opacity,
                blend: l.blend,
                text: String::new(),
                font_size: l.font_size,
                font_family: None,
                text_align: l.text_align,
                render: l.render,
                effects: l.effects.clone(),
                data: None,
                emoji: None,
            }),
            Layer::Image(l) => Layer::Image(ImageLayer {
                name: l.name.clone(),
                visible: l.visible,
                x: l.x,
                y: l.y,
                color: l.color,
                opacity: l.opacity,
                blend: l.blend,
                image: l.image.clone(),
                width: l.width,
                height: l.height,
                data: None,
            }),
            Layer::Shape(l) => Layer::Shape(l.clone()),
        }
    }
}

/// ghee's name for the uniform `token` of layer `layer`.
pub fn uniform(layer: &str, token: &str) -> String {
    format!("{}{layer}_{token}", crate::scene::PREFIX)
}

/// The batches of a layer.
#[derive(Clone, Copy)]
enum Batch {
    Glyphs,
    Boxes,
    Colour,
}

/// The precision of every shader of a layer, a pass's own.
const PRECISION: [&str; 6] = crate::scene::PRECISION;

/// The vertex shader of every batch of a layer. DECLS declares the uniforms
/// and the varyings of the batch and its effects, and MAIN places each
/// vertex.
const VERTEX: &str = r#"#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
attribute vec4 ghee_a0;
attribute vec4 ghee_a1;
attribute vec4 ghee_a2;
attribute vec4 ghee_a3;
uniform vec3 iResolution;
uniform float ghee_vflip;
uniform vec2 ghee_NAME_size;
uniform float ghee_NAME_x;
uniform float ghee_NAME_y;
uniform vec2 ghee_NAME_centre;
uniform float ghee_NAME_visible;
varying vec2 uv;
DECLS
void main() {
MAIN
}"#;

/// The lines that fit the block to its room, up to ghee_s, the pixels per
/// em that fit each axis.
const FIT: &str = r#"    float ghee_m = FIT_MARGIN * min(iResolution.x, iResolution.y);
    vec2 ghee_room = mix(iResolution.xy - ghee_m - vec2(ghee_NAME_x, ghee_NAME_y),
                         iResolution.xy - 2.0 * ghee_m, ghee_NAME_centre);
    vec2 ghee_s = ghee_room / max(ghee_NAME_size, vec2(1.0e-6));"#;

/// The lines that set ghee_ppe, the pixels per em held above 0 so that a
/// block fitted to nothing keeps every position finite, and ghee_org, the
/// pass pixel of the block's top left.
const PLACE: &str = r#"    float ghee_ppe = max(ghee_fs, 1.0 / 16384.0);
    vec2 ghee_org = mix(vec2(ghee_NAME_x, ghee_NAME_y), (iResolution.xy - ghee_NAME_size * ghee_fs) * 0.5,
                        ghee_NAME_centre);"#;

/// The lines that scroll the block from right to left by whole pixels: it
/// enters at the pass's right edge, and starts there again once it has left
/// at the left.
const SCROLL_LINES: &str = r#"    float ghee_trip = iResolution.x + ghee_NAME_size.x * ghee_fs;
    float ghee_sx = floor(iResolution.x - mod(ghee_NAME_scrollphase * ghee_fs, ghee_trip));
    ghee_org.x = mix(ghee_org.x, ghee_sx, ghee_NAME_scrollvisible);"#;

/// The lines of karaoke: ghee_kf, the fill of the letters up to the sung
/// point, which reaches the end of a character's advance as `sung` passes
/// it, in pixels right of this vertex, and the hop of the character being
/// sung, up to 0.15 em as the point passes its middle.
const KARAOKE_LINES: &str = r#"    float ghee_ks = ghee_NAME_karaokesung - ghee_a0.z;
    float ghee_kx = ghee_ks >= 1.0 ? 1.0e4
                  : (ghee_ks <= 0.0 ? -1.0e4 : mix(ghee_a3.y, ghee_a3.z, ghee_ks));
    float ghee_kf = (ghee_kx - ghee_e.x) * ghee_fs + 0.5;
    float ghee_kh = clamp(ghee_ks, 0.0, 1.0);
    ghee_mv.y -= 0.6 * ghee_kh * (1.0 - ghee_kh) * ghee_NAME_karaokevisible;"#;

/// The lines that place the point on the colour gradient into ghee_t: 0 at
/// its start, 1 at its end, along CSS's gradient line through the block's
/// centre.
const GRADIENT: &str = r#"    float ghee_r = radians(ghee_NAME_angle);
    vec2 ghee_dir = vec2(sin(ghee_r), -cos(ghee_r));
    float ghee_len = abs(ghee_NAME_size.x * ghee_dir.x) + abs(ghee_NAME_size.y * ghee_dir.y);
    float ghee_t = dot(ghee_e - 0.5 * ghee_NAME_size, ghee_dir) / max(ghee_len, 1.0e-6) + 0.5;
"#;

/// The line that puts the block's top left on a whole pixel.
const SNAP: &str = "    ghee_org = floor(ghee_org + 0.5);";

/// The lines that set ghee_e, the em from the block's top left of the
/// corner of a glyph's quad, and ghee_mv, its motion. GROW is the em that
/// the quad reaches past the glyph's box.
const CORNER: &str = r#"    // A quad reaches half a pixel past its glyph's box for the
    // antialiasing, and 1/8 px more, since VC4 lands a vertex up to 1/16
    // px off the position set by this shader.
    float ghee_g = GROW;
    vec2 ghee_corner = mix(ghee_a1.xy, ghee_a1.zw, ghee_a0.xy);
    vec2 ghee_e = ghee_corner + (ghee_a0.xy * 2.0 - 1.0) * ghee_g;
    vec2 ghee_mv = vec2(0.0);"#;

/// The lines that place a vertex, and HIDE, the condition that takes its
/// quad out of the pass.
const POSITION: &str = r#"    uv = (ghee_org + (ghee_e + ghee_mv) * ghee_fs) / iResolution.xy;
    gl_Position = vec4((uv * 2.0 - 1.0) * vec2(1.0, ghee_vflip), 0.0, 1.0);
    // A hidden glyph's quad lies outside the pass.
    if (HIDE) gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);"#;

/// The pixels per em that fit the block to its room, from ghee_s, a whole
/// multiple of `grid` pixels per em, one at the least, when given.
fn fitted(grid: Option<f64>) -> String {
    match grid {
        Some(g) => format!("max(floor(min(ghee_s.x, ghee_s.y) / {g:?}) * {g:?}, {g:?})"),
        None => "max(min(ghee_s.x, ghee_s.y), 0.0)".to_string(),
    }
}

/// The lines that put each glyph's pen and baseline on whole pixels, in a
/// text of `fonts` on the pixel grid.
fn pen(fonts: &[text::Font]) -> String {
    let (ascent, line) = text::line_metrics(fonts);
    format!(
        "    // Each glyph's pen and baseline, after its motion, land on whole pixels.\n    \
         vec2 ghee_pen = (vec2(ghee_a3.y, {ascent:?} + ghee_a3.x * {line:?}) + ghee_mv) * ghee_fs;\n    \
         ghee_org += floor(ghee_pen + 0.5) - ghee_pen;"
    )
}

/// The lines after `indent` that set ghee_rgb and ghee_al, the letters'
/// colour and alpha at the point `t` on the colour's gradient.
fn gradient_colour(indent: &str, t: &str) -> String {
    format!(
        "{indent}vec3 ghee_rgb = mix(ghee_NAME_color, ghee_NAME_colorto, clamp({t}, 0.0, 1.0));\n\
         {indent}float ghee_al = ghee_NAME_alpha\n{indent}              \
         + (ghee_NAME_alphato - ghee_NAME_alpha) * clamp({t}, 0.0, 1.0);\n"
    )
}

/// The comment over the reads of a fragment shader.
const GUARD: &str = r#"    // On VC4 a short shader whose last texture read sits outside an if runs
    // in two threads, whose switch loses the condition flags. Every read sits
    // in an if that the compiler cannot drop, so the shader stays on one
    // thread.
"#;

/// The vertex shader of an image layer. SIZE is the pass pixels that the
/// picture spans.
const IMAGE_VERTEX: &str = r#"#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
attribute vec4 ghee_a0;
uniform vec3 iResolution;
uniform float ghee_vflip;
uniform vec2 ghee_NAME_image_size;
uniform float ghee_NAME_x;
uniform float ghee_NAME_y;
uniform vec2 ghee_NAME_centre;
uniform float ghee_NAME_visible;
varying vec2 uv;
varying vec4 ghee_v0;
void main() {
    vec2 ghee_wh = SIZE;
    vec2 ghee_org = mix(vec2(ghee_NAME_x, ghee_NAME_y), (iResolution.xy - ghee_wh) * 0.5,
                        ghee_NAME_centre);
    uv = (ghee_org + ghee_a0.xy * ghee_wh) / iResolution.xy;
    // The point on the image, and its step per pass pixel.
    ghee_v0 = vec4(ghee_a0.xy, 1.0 / max(ghee_wh, vec2(1.0e-6)));
    gl_Position = vec4((uv * 2.0 - 1.0) * vec2(1.0, ghee_vflip), 0.0, 1.0);
    // A hidden layer's quad lies outside the pass.
    if (ghee_NAME_visible < 0.5) gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
}"#;

/// The fragment shader of an image layer. The image holds straight alpha,
/// and the blend takes premultiplied colour.
const IMAGE_FRAGMENT: &str = r#"#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform vec3 iResolution;
uniform sampler2D ghee_NAME_image_tex;
uniform vec3 ghee_NAME_color;
uniform float ghee_NAME_alpha;
uniform float ghee_NAME_opacity;
varying vec2 uv;
varying vec4 ghee_v0;
void main() {
    // VC4 lands a vertex up to 1/16 px off the position set by the vertex
    // shader, and the varyings follow the vertex, so the sample moves to the
    // pixel centre given by uv.
    vec2 ghee_px = uv * iResolution.xy;
    vec2 ghee_uv = ghee_v0.xy + (floor(ghee_px) + 0.5 - ghee_px) * ghee_v0.zw;
    vec4 ghee_t = vec4(0.0);
GUARD    if (gl_FragCoord.x >= 0.0) ghee_t = texture2D(ghee_NAME_image_tex, ghee_uv);
    gl_FragColor = vec4(ghee_t.rgb * ghee_NAME_color * ghee_t.a, ghee_t.a) * ghee_NAME_opacity
                 * ghee_NAME_alpha;
}"#;

/// A read of the layer's atlas at UV, from the page of the quad.
const GUARDED_READ: &str = "    if (ghee_v1.w < 0.5) ghee_t = texture2D(ghee_NAME_atlas_tex, UV); \
     else ghee_t = texture2D(ghee_NAME_atlas2_tex, UV);";

/// The varyings of a slug layer's glyphs, with the point on the gradient and
/// the karaoke fill, or of its error boxes, without, and the lines that
/// declare them: the sample on the glyph's lattice, y up, and the pixels
/// spanned by a lattice step, the texel of the glyph's band entries and its
/// bands, and the lattice steps spanned by a pixel along x and along y of the
/// pass. The fragment shader declares its own.
fn slug_varyings(gradient: bool, karaoke: bool) -> (String, String) {
    let mut decls = vec![
        "varying vec4 ghee_v0;",
        "varying vec4 ghee_v1;",
        "varying vec4 ghee_v4;",
    ];
    let t = match gradient {
        true => {
            decls.push("uniform float ghee_NAME_angle;");
            GRADIENT
        }
        false => "    float ghee_t = 0.0;\n",
    };
    let mut lines = format!(
        "    vec2 ghee_sz = ghee_a1.zw - ghee_a1.xy;\n    \
         ghee_v0 = vec4(vec2(ghee_e.x - ghee_a1.x, ghee_a1.w - ghee_e.y) / ghee_sz * LATTICE,\n    \
         \x20              ghee_sz * ghee_ppe / LATTICE);\n    \
         vec2 ghee_k = LATTICE / (ghee_sz * ghee_ppe);\n    \
         ghee_v4 = vec4(ghee_k.x, 0.0, 0.0, -ghee_k.y);\n\
         {t}    ghee_v1 = vec4(ghee_a2.xyz, ghee_t);\n"
    );
    if karaoke {
        decls.push("varying vec4 ghee_v2;");
        lines.push_str("    ghee_v2 = vec4(ghee_kf, 0.0, 0.0, 0.0);\n");
    }
    (decls.join("\n"), lines)
}

/// The varyings of a colour glyph, and the lines that declare them: the atlas
/// coordinate, and the bias that holds GL's mip level at colour::MAX_LEVEL
/// at most. GL picks the level from the atlas texels spanned by a pixel.
fn colour_varyings() -> (String, String) {
    let decls = "uniform vec2 ghee_NAME_emoji_size;\nvarying vec4 ghee_v0;".to_string();
    let lines = format!(
        "    vec2 ghee_tc = ghee_a2.xy + (ghee_e - ghee_a1.xy) * ghee_a2.zw;\n    \
         float ghee_lod = log2(max(ghee_a2.z / ghee_ppe, 1.0 / 65536.0));\n    \
         ghee_v0 = vec4(ghee_tc / ghee_NAME_emoji_size, min(ghee_lod, {:?}) - ghee_lod, 0.0);\n",
        colour::MAX_LEVEL
    );
    (decls, lines)
}

/// The main of a slug layer's glyphs, after the readers of its curves, up
/// to SLUG_GUARD. COLOUR sets ghee_rgb and ghee_al.
const SLUG_BODY: &str = r#"    // VC4 lands a vertex up to 1/16 px off the position set by the vertex
    // shader, and the varyings follow the vertex, so the sample moves to the
    // pixel centre given by uv.
    vec2 ghee_px = uv * iResolution.xy;
    vec2 ghee_d = floor(ghee_px) + 0.5 - ghee_px;
    vec2 ghee_u = ghee_v0.xy + vec2(dot(ghee_v4.xy, ghee_d), dot(ghee_v4.zw, ghee_d));
    float ghee_c = ghee_NAME_outline(float(int(ghee_v1.x + 0.5)), vec2(ivec2(ghee_v1.yz + 0.5)),
                                     ghee_u, ghee_v0.zw);
COLOUR
    gl_FragColor = vec4(ghee_rgb, ghee_al) * (ghee_c * ghee_NAME_opacity);
"#;

/// The last lines of the main of a slug layer's glyphs.
const SLUG_GUARD: &str = r#"    // On VC4 a shader whose last texture read sits outside an if runs in two
    // threads, whose switch loses the condition flags. A read after ghee's
    // own code that never runs, in an if that the compiler cannot drop,
    // keeps the shader on one thread.
    if (gl_FragCoord.x < 0.0) gl_FragColor = texture2D(ghee_NAME_curves_tex, uv);
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The identifiers of GLSL `t`, outside its comments.
    fn words(t: &str) -> Vec<String> {
        let ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
        t.lines()
            .flat_map(|l| {
                l.split("//")
                    .next()
                    .unwrap_or("")
                    .split(|c: char| !ident(c))
            })
            .filter(|w| !w.is_empty())
            .map(str::to_string)
            .collect()
    }

    /// Every name with ghee's prefix that ghee gives the passes for layer
    /// `name`: the readers of its words, and the engine's assets of the
    /// layer, each with the sampler and the size that a program reads.
    fn written(name: &str) -> Vec<String> {
        let mut out: Vec<String> = text::template_names(name);
        for suffix in ["_curves", "_atlas", "_atlas2", "_emoji", "_file"] {
            let asset = text::internal(name, suffix);
            out.extend([format!("{asset}_tex"), format!("{asset}_size"), asset]);
        }
        out.retain(|w| w.starts_with(crate::scene::PREFIX));
        out.sort();
        out.dedup();
        out
    }

    /// Every name with ghee's prefix in the shaders and the uniforms of the
    /// layers' own batches: text layers with every effect in either render,
    /// and an image layer.
    fn own() -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for render in ["mtsdf", "slug"] {
            let mut d = json!({"name": "t", "text": "", "render": render});
            for kind in Kind::ALL {
                d[kind.key()] = json!({});
            }
            let l = TextLayer::from_dict("a layer", &d).unwrap();
            let sh = l.shaders(Detail {
                px_em: 32,
                range: 6,
            });
            for glsl in [sh.glyphs, sh.boxes, sh.colour]
                .into_iter()
                .flat_map(|(v, f)| [v, f])
            {
                out.extend(words(&glsl));
            }
            out.extend(l.uniforms().into_iter().map(|u| u.0));
        }
        let l = ImageLayer::from_dict("a layer", &json!({"name": "i", "image": "pic"})).unwrap();
        let (v, f) = l.shaders();
        out.extend(words(&v));
        out.extend(words(&f));
        out.extend(l.uniforms().into_iter().map(|u| u.0));
        out.retain(|w| w.starts_with(crate::scene::PREFIX));
        out.sort();
        out.dedup();
        out
    }

    #[test]
    fn no_name_that_ghee_writes_for_a_layer_meets_another() {
        // Names that end the way another layer's names do, and the names of
        // ghee's own GLSL.
        let names = [
            "a",
            "a_x",
            "a_y",
            "a_color",
            "a_colorto",
            "a_centre",
            "a_visible",
            "a_opacity",
            "a_angle",
            "a_tex",
            "a_size",
            "a_curves",
            "a_curves_tex",
            "a_atlas",
            "a_atlas_size",
            "a_atlas2",
            "a_atlas2_size",
            "a_emoji",
            "a_image",
            "a_file",
            "a_cover",
            "a_outline",
            "a_tw",
            "fs",
            "v0",
            "e",
            "t",
            "d",
        ];
        let made: Vec<Vec<String>> = names.iter().map(|n| written(n)).collect();
        // A uniform that ghee sets on a pass reaches the batches of the
        // pass too, so no name in a layer's own shaders is one of them.
        let own = own();
        for (x, mx) in names.iter().zip(&made) {
            let met: Vec<&String> = own.iter().filter(|n| mx.contains(n)).collect();
            assert!(met.is_empty(), "the batches and {x} meet in {met:?}");
        }
        for (i, (x, mx)) in names.iter().zip(&made).enumerate() {
            let own = format!("{}{x}_", crate::scene::PREFIX);
            assert!(mx.iter().any(|n| n.starts_with(&own)), "{x}");
            for (y, my) in names[i + 1..].iter().zip(&made[i + 1..]) {
                let met: Vec<&String> = mx.iter().filter(|n| my.contains(n)).collect();
                assert!(met.is_empty(), "{x} and {y} meet in {met:?}");
            }
        }
    }

    fn layer(keys: serde_json::Value) -> TextLayer {
        let mut d = json!({"name": "t", "text": "I"});
        d.as_object_mut()
            .unwrap()
            .extend(keys.as_object().unwrap().clone());
        TextLayer::from_dict("a layer", &d).unwrap()
    }

    #[test]
    fn the_reach_and_the_pad_count_every_effect_that_reads_the_distance_bypassed_or_not() {
        let off = json!({"visible": false});
        // Each row: the layer's keys, its field reach and its quad's pad in
        // em, as the effects' fields give them.
        for (keys, reach, pad) in [
            (json!({}), 0.0, 0.0),
            (
                json!({"wave": {}, "karaoke": {}, "scroll": {}, "blink": {}}),
                0.0,
                0.0,
            ),
            (json!({"text_shadow": {}}), 0.15, 0.08 + 0.15),
            (
                json!({"text_shadow": {"x": -0.5, "y": 0.25, "blur": 0.0}}),
                0.0,
                0.5,
            ),
            (json!({"glow": {}}), 0.3, 0.3),
            (json!({"glow": off}), 0.3, 0.3),
            (
                json!({"text_stroke": {"width": 0.2, "visible": false}}),
                0.2,
                0.2,
            ),
            (json!({"text_stroke": {}, "glow": {"blur": 0.1}}), 0.1, 0.1),
            // The slug render draws none of them.
            (json!({"glow": {}, "render": "slug"}), 0.0, 0.0),
        ] {
            let l = layer(keys.clone());
            assert_eq!((l.reach(), l.pad()), (reach, pad), "{keys}");
            assert_eq!(
                l.doubled(),
                pad > 0.0 || keys.get("text_stroke").is_some(),
                "{keys}"
            );
        }
    }

    #[test]
    fn the_field_reaches_the_effects_and_a_texel_more_at_the_least_range() {
        // A range spans both sides of the outline, so it takes twice the
        // reach in texels, and a texel more on each side.
        for (keys, px_em, range) in [
            (json!({}), 32, atlas::RANGE),
            (json!({"glow": {"blur": 0.5}}), 32, 34),
            (json!({"glow": {"blur": 0.5}}), 64, 66),
            (json!({"text_stroke": {"width": 0.01}}), 32, atlas::RANGE),
            (json!({"glow": {"blur": 1000.0}}), 32, atlas::MAX_SIZE),
        ] {
            let d = layer(keys.clone()).detail(px_em);
            assert_eq!((d.px_em, d.range), (px_em, range), "{keys}");
        }
    }

    #[test]
    fn a_layer_without_font_size_takes_the_size_that_fits_its_words_to_its_room() {
        // A pass of 400x200 has a margin of 10 px, 5% of its shorter side.
        // Each row: the layer's keys, its block in em, and its pixels per em.
        let off = json!({"visible": false});
        for (keys, block, px) in [
            (json!({"font_size": 30}), [4.0, 1.0], 30.0),
            // Centred: 380x180 of room.
            (json!({}), [4.0, 1.0], 95.0),
            (json!({}), [1.0, 1.0], 180.0),
            // At x 100, 290 px to the margin at the right edge.
            (json!({"x": 100}), [4.0, 1.0], 72.5),
            // At y 150, 40 px to the margin at the bottom.
            (json!({"y": 150}), [4.0, 1.0], 40.0),
            (json!({"x": 500}), [4.0, 1.0], 0.0),
            // A scrolling block fits on y alone.
            (json!({"scroll": {}}), [40.0, 1.0], 180.0),
            (json!({"scroll": off}), [40.0, 1.0], 9.5),
            // The pixel font rounds a fit down to a whole multiple of 8 px,
            // 8 at the least, and takes a font_size as given.
            (json!({"font_family": "pixel"}), [4.0, 1.0], 88.0),
            (json!({"font_family": "pixel", "y": 150}), [4.0, 1.0], 40.0),
            (json!({"font_family": "pixel", "x": 500}), [4.0, 1.0], 8.0),
            (
                json!({"font_family": "pixel", "font_size": 30}),
                [4.0, 1.0],
                30.0,
            ),
        ] {
            assert_eq!(
                layer(keys.clone()).font_px([400.0, 200.0], block),
                px,
                "{keys}"
            );
        }
    }

    #[test]
    fn a_layer_takes_the_first_atlas_step_at_or_above_its_size_and_the_last_at_most() {
        assert_eq!(TextLayer::atlas_px_em(0.0), ATLAS_STEPS[0]);
        let mut below = 0.0;
        for step in ATLAS_STEPS {
            let at = f64::from(step);
            for px in [below + 0.01, (below + at) / 2.0, at] {
                assert_eq!(TextLayer::atlas_px_em(px), step, "{px} px/em");
            }
            below = at;
        }
        for px in [below + 0.01, 4.0 * below] {
            assert_eq!(TextLayer::atlas_px_em(px), below as u32, "{px} px/em");
        }
    }

    #[test]
    fn a_doubled_layer_draws_every_effect_then_every_letter() {
        let quads: Vec<f32> = (0..2 * 4 * text::QUAD_FLOATS).map(|i| i as f32).collect();
        assert_eq!(layer(json!({"wave": {}})).copies(&quads), quads);
        let both = layer(json!({"glow": {}})).copies(&quads);
        assert_eq!(both.len(), 2 * quads.len());
        let (effects, letters) = both.split_at(quads.len());
        assert_eq!(effects, quads);
        for (v, w) in letters
            .chunks_exact(text::QUAD_FLOATS)
            .zip(quads.chunks_exact(text::QUAD_FLOATS))
        {
            let last = text::QUAD_FLOATS - 1;
            assert_eq!((&v[..last], v[last]), (&w[..last], 1.0));
        }
    }

    /// A text layer of `words` in the fixture font `own`, the built-in font
    /// for None, that falls back to the fixture `emoji` copied into its own
    /// directory of `dir` as the emoji font's file.
    fn falling_back(
        dir: &crate::testing::TempDir,
        own: Option<&str>,
        emoji: Option<&str>,
        words: &str,
    ) -> TextLayer {
        let fixture = |f: &str| format!("{}/tools/fixtures/fonts/{f}", env!("CARGO_MANIFEST_DIR"));
        let mut l = TextLayer::from_dict("a layer", &json!({"name": "t", "text": words})).unwrap();
        match own {
            Some("no font") => l.data = Some(vec![text::font(b"no font")]),
            Some(f) => l.data = Some(vec![text::font(&std::fs::read(fixture(f)).unwrap())]),
            None => {}
        }
        l.emoji = emoji.map(|f| {
            let at = dir.join(f);
            std::fs::create_dir_all(&at).unwrap();
            let path = at.join(emoji::FILE);
            match f {
                "no font" => std::fs::write(&path, b"no font").unwrap(),
                _ => {
                    std::fs::copy(fixture(f), &path).unwrap();
                }
            }
            emoji::at(path.to_str().unwrap())
        });
        l
    }

    #[test]
    fn a_character_that_the_font_lacks_draws_a_colour_glyph_of_the_emoji_font_or_the_box() {
        // DejaVuSans-Heart holds an outline heart, U+2764, and the two
        // selectors, NotoColorEmoji-Heart a colour heart, and
        // NotoColorEmoji-Smiles U+1F600 and U+1F642 in colour. Each row is the
        // layer's font, the emoji font, the words, and the characters drawn
        // as glyphs and as colour glyphs, with the index of their font: the
        // layer's own, then the emoji font.
        let dir = crate::testing::TempDir::new("layer-emoji");
        let smiles = Some("NotoColorEmoji-Smiles.ttf");
        let heart = Some("DejaVuSans-Heart.ttf");
        let colour_heart = Some("NotoColorEmoji-Heart.ttf");
        type Row<'a> = (
            Option<&'a str>,
            Option<&'a str>,
            &'a str,
            &'a [(char, usize)],
            &'a [(char, usize)],
        );
        let rows: [Row; 5] = [
            (None, smiles, "A\u{1f600}", &[('A', 0)], &[('\u{1f600}', 1)]),
            // With no emoji font, the emoji draws glyph 0 of the layer's
            // font, its missing-glyph box.
            (None, None, "A\u{1f600}", &[('A', 0), ('\u{1f600}', 0)], &[]),
            // U+FE0F asks for the colour glyph of a character that the font
            // holds, and the heart without it keeps its outline.
            (
                heart,
                colour_heart,
                "\u{2764}\u{fe0f}\u{2764}",
                &[('\u{2764}', 0)],
                &[('\u{2764}', 1)],
            ),
            (heart, smiles, "\u{2764}\u{fe0f}", &[('\u{2764}', 0)], &[]),
            (None, smiles, "A", &[('A', 0)], &[]),
        ];
        for (own, emoji, words, glyphs, colour) in rows {
            let l = falling_back(&dir, own, emoji, words);
            let laid = l.build(words).unwrap();
            assert_eq!(laid.faults, [], "{words:?}");
            let built = laid.built.unwrap();
            let of = |ps: &[text::Placement]| -> Vec<(char, usize)> {
                ps.iter().map(|p| (p.ch, p.font)).collect()
            };
            assert_eq!(of(&built.glyphs), glyphs, "{words:?} with {emoji:?}");
            assert_eq!(of(&built.colour), colour, "{words:?} with {emoji:?}");
            if emoji.is_none() {
                assert_eq!(built.glyphs[1].gid, 0);
            }
        }
        // A layer whose font failed to load draws an error box for each
        // character, its emoji too.
        let words = "A\u{1f600}";
        let laid = falling_back(&dir, Some("no font"), smiles, words)
            .build(words)
            .unwrap();
        assert!(
            matches!(laid.faults[..], [text::Fault::Font(_)]),
            "{:?}",
            laid.faults
        );
        let built = laid.built.unwrap();
        assert!(built.glyphs.is_empty() && built.colour.is_empty());
        assert_eq!(built.boxes.count, 2);
    }

    #[test]
    fn an_emoji_font_that_does_not_load_names_itself_once_words_need_it() {
        let dir = crate::testing::TempDir::new("layer-no-emoji");
        let path = dir.join("no font").join(emoji::FILE);
        let why = ttf_parser::Face::parse(b"no font", 0).unwrap_err();
        let l = falling_back(&dir, None, Some("no font"), "A");
        assert_eq!(l.build("A").unwrap().faults, []);
        let laid = l.build("A\u{1f600}").unwrap();
        assert_eq!(
            laid.faults,
            [text::Fault::Font(format!(
                "Noto Color Emoji ({}): unreadable font: {why}",
                path.display()
            ))]
        );
        assert_eq!(laid.built.unwrap().glyphs[1].gid, 0);
    }

    fn text_and_image() -> (Layer, Layer) {
        let d = json!({"name": "t", "text": "I", "glow": {}, "wave": {}});
        let text = Layer::from_dict("a layer", &d).unwrap();
        let image = Layer::from_dict("a layer", &json!({"name": "i", "image": "pic"})).unwrap();
        (text, image)
    }

    #[test]
    fn a_live_set_that_a_layer_cannot_take_is_refused_naming_what_it_takes() {
        let (text, image) = text_and_image();
        let glow = Layer::from_dict("a layer", &json!({"name": "g", "text": "I", "glow": {}}));
        let glow = glow.unwrap();
        let bare = Layer::from_dict("a layer", &json!({"name": "b", "text": "I"})).unwrap();
        let rebuild = "changes with a rebuild, and the live fields";
        let text_live =
            "and the live fields of a text layer are x, y, color, opacity, visible and text";
        let rect = Layer::from_dict("a layer", &json!({"name": "r", "shape": "rect"})).unwrap();
        let line = json!({"name": "l", "shape": "line", "points": [[0, 0], [1, 1]]});
        let line = Layer::from_dict("a layer", &line).unwrap();
        let rect_live = "the live fields of a rect are x, y, color, opacity, visible, width, \
                         height and border_radius";
        let line_live = "the live fields of a line are x, y, color, opacity, visible, points \
                         and stroke_width";
        let rows = [
            (
                &rect,
                None,
                "points",
                format!("layer 'r' has no field 'points', and {rect_live}"),
            ),
            (
                &line,
                None,
                "width",
                format!("layer 'l' has no field 'width', and {line_live}"),
            ),
            (
                &rect,
                None,
                "blend",
                format!("layer 'r' blend changes with a rebuild, and {rect_live}"),
            ),
            (
                &line,
                None,
                "shape",
                format!("layer 'l' shape changes with a rebuild, and {line_live}"),
            ),
            (
                &rect,
                Some("border"),
                "style",
                "layer 'r' border has no field 'style', and the live fields of border are width, \
                 color and visible"
                    .to_string(),
            ),
            (
                &rect,
                Some("glow"),
                "color",
                "layer 'r' has no effect 'glow', and a rect takes border alone".to_string(),
            ),
            (
                &line,
                Some("border"),
                "color",
                "layer 'l' has no effect 'border', and a line takes no effects".to_string(),
            ),
            (
                &text,
                None,
                "font_size",
                format!(
                    "layer 't' font_size {rebuild} of a text layer are x, y, color, opacity, \
                     visible and text"
                ),
            ),
            (
                &image,
                None,
                "width",
                format!(
                    "layer 'i' width {rebuild} of an image layer are x, y, color, opacity, visible \
                     and image"
                ),
            ),
            (
                &text,
                Some("glow"),
                "blur",
                format!("layer 't' glow blur {rebuild} of glow are color, strength and visible"),
            ),
            (
                &text,
                Some("wave"),
                "height",
                format!("layer 't' wave height {rebuild} of wave are speed and visible"),
            ),
            (
                &text,
                None,
                "colour",
                format!("layer 't' has no field 'colour', {text_live}"),
            ),
            (
                &text,
                None,
                "image",
                format!("layer 't' has no field 'image', {text_live}"),
            ),
            (
                &image,
                None,
                "text",
                "layer 'i' has no field 'text', and the live fields of an image layer are x, y, \
                 color, opacity, visible and image"
                    .to_string(),
            ),
            (
                &text,
                Some("glow"),
                "size",
                "layer 't' glow has no field 'size', and the live fields of glow are color, \
                 strength and visible"
                    .to_string(),
            ),
            (
                &text,
                Some("karaoke"),
                "sung",
                "layer 't' has no effect 'karaoke', and its effects are glow and wave".to_string(),
            ),
            (
                &glow,
                Some("karaoke"),
                "sung",
                "layer 'g' has no effect 'karaoke', and its effect is glow".to_string(),
            ),
            (
                &bare,
                Some("karaoke"),
                "sung",
                "layer 'b' has no effect 'karaoke' and no other effect".to_string(),
            ),
            (
                &image,
                Some("glow"),
                "color",
                "layer 'i' has no effect 'glow', and an image layer takes no effects".to_string(),
            ),
        ];
        for (l, effect, field, want) in rows {
            assert_eq!(l.live(effect, field).unwrap_err().to_string(), want);
        }
    }

    #[test]
    fn a_live_value_takes_the_form_of_its_field_on_the_wire() {
        let (text, image) = text_and_image();
        let rect = Layer::from_dict("a layer", &json!({"name": "r", "shape": "rect"})).unwrap();
        let line = json!({"name": "l", "shape": "line", "points": [[0, 0], [1, 1]]});
        let line = Layer::from_dict("a layer", &line).unwrap();
        let gradient = json!({"from": [1.0, 0.0, 0.0], "to": [0.0, 0.0, 1.0], "angle": 90.0});
        let ok = [
            (&text, None, "x", vec![5.0], json!(5.0)),
            (&text, None, "x", vec![], json!(null)),
            (&text, None, "visible", vec![0.0], json!(false)),
            (&text, None, "visible", vec![1.0], json!(true)),
            (
                &text,
                None,
                "color",
                vec![1.0, 0.0, 0.0],
                json!([1.0, 0.0, 0.0]),
            ),
            (
                &text,
                None,
                "color",
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 90.0],
                gradient,
            ),
            (&text, None, "color", vec![], json!(null)),
            (&image, None, "color", vec![], json!(null)),
            (&text, Some("glow"), "strength", vec![0.5], json!(0.5)),
            (&text, Some("wave"), "visible", vec![0.0], json!(false)),
            (
                &line,
                None,
                "points",
                vec![0.0, 16.0, 8.0, 4.5],
                json!([[0.0, 16.0], [8.0, 4.5]]),
            ),
            (&rect, Some("border"), "width", vec![2.0], json!(2.0)),
        ];
        for (l, effect, field, numbers, want) in ok {
            let live = l.live(effect, field).unwrap();
            assert_eq!(
                live.of_numbers(&numbers).unwrap(),
                want,
                "{field} {numbers:?}"
            );
        }
        let refused = [
            (
                &text,
                None,
                "x",
                vec![1.0, 2.0],
                "layer 't' x takes 1 number or 0 for its default, got 2",
            ),
            (
                &text,
                None,
                "opacity",
                vec![],
                "layer 't' opacity takes 1 number, got 0",
            ),
            (
                &text,
                None,
                "visible",
                vec![],
                "layer 't' visible takes 1 number, 0 or 1, got 0",
            ),
            (
                &text,
                None,
                "color",
                vec![1.0, 0.0],
                "layer 't' color takes 3 numbers, 7 for a gradient or 0 for its default, got 2",
            ),
            (
                &image,
                None,
                "color",
                vec![0.0; 7],
                "layer 'i' color takes 3 numbers or 0 for its default, got 7",
            ),
            (
                &text,
                Some("glow"),
                "color",
                vec![],
                "layer 't' glow color takes 3 numbers, got 0",
            ),
            (
                &image,
                None,
                "image",
                vec![1.0],
                "layer 'i' image takes the name of an image asset, got numbers",
            ),
            (
                &text,
                None,
                "text",
                vec![1.0],
                "layer 't' text takes text, got numbers",
            ),
            (
                &line,
                None,
                "points",
                vec![0.0, 16.0, 8.0],
                "layer 'l' points takes x and y in turn for two points or more, got 3 numbers",
            ),
            (
                &line,
                None,
                "points",
                vec![0.0, 16.0],
                "layer 'l' points takes x and y in turn for two points or more, got 2 numbers",
            ),
        ];
        for (l, effect, field, numbers, want) in refused {
            let live = l.live(effect, field).unwrap();
            assert_eq!(live.of_numbers(&numbers).unwrap_err().to_string(), want);
        }
        let live = image.live(None, "image").unwrap();
        assert_eq!(live.of_text("wifi").unwrap(), json!("wifi"));
        let live = text.live(None, "text").unwrap();
        assert_eq!(live.of_text(" la la ").unwrap(), json!(" la la "));
        let live = text.live(None, "color").unwrap();
        assert_eq!(live.of_text("Navy").unwrap(), json!("Navy"));
        let gradient = r#"{"from": "red", "to": "transparent"}"#;
        assert_eq!(
            live.of_text(gradient).unwrap(),
            json!({"from": "red", "to": "transparent"})
        );
        let live = text.live(None, "x").unwrap();
        assert_eq!(
            live.of_text("4").unwrap_err().to_string(),
            "layer 't' x takes numbers, got text"
        );
    }

    #[test]
    fn a_live_value_changes_the_layer_as_the_scene_would_declare_it() {
        let (mut text, mut image) = text_and_image();
        let gradient = json!({"from": "#ff0000", "to": "#0000ff", "angle": 90.0});
        let rows = [
            ("t", None, "x", json!(5.0), json!({"x": 5.0})),
            ("t", None, "x", json!(null), json!({})),
            (
                "t",
                None,
                "color",
                gradient.clone(),
                json!({"color": gradient}),
            ),
            (
                "t",
                None,
                "visible",
                json!(false),
                json!({"visible": false}),
            ),
            (
                "t",
                Some("glow"),
                "visible",
                json!(false),
                json!({"glow": {"visible": false}}),
            ),
            (
                "t",
                Some("wave"),
                "speed",
                json!(2.0),
                json!({"wave": {"speed": 2.0}}),
            ),
            ("t", None, "y", json!(3), json!({"y": 3.0})),
            ("t", None, "opacity", json!(0.5), json!({"opacity": 0.5})),
            ("t", None, "color", json!("navy"), json!({"color": "navy"})),
            ("t", None, "color", json!(null), json!({})),
            ("i", None, "x", json!(-2.5), json!({"x": -2.5})),
            ("i", None, "opacity", json!(0.25), json!({"opacity": 0.25})),
            (
                "i",
                None,
                "visible",
                json!(false),
                json!({"visible": false}),
            ),
            ("i", None, "image", json!("pic2"), json!({"image": "pic2"})),
            (
                "i",
                None,
                "color",
                json!("#ff8000"),
                json!({"color": "#ff8000"}),
            ),
            ("i", None, "color", json!(null), json!({})),
        ];
        for (name, effect, field, value, keys) in rows {
            let l = if name == "t" { &mut text } else { &mut image };
            let mut want = l.to_dict();
            match effect {
                Some(e) => want[e] = json!({}),
                None => {
                    want.as_object_mut().unwrap().remove(field);
                }
            }
            want.as_object_mut()
                .unwrap()
                .extend(keys.as_object().unwrap().clone());
            l.set_live(effect, field, &value).unwrap();
            let got = l.to_dict();
            assert_eq!(got, want, "{field} = {value}");
        }
        text.set_live(None, "text", &json!(" la la ")).unwrap();
        assert_eq!(text.to_dict()["text"], " la la ");
        let glow = |look: Value| {
            let d = json!({"name": "g", "text": "I", "glow": look});
            Layer::from_dict("a layer", &d).unwrap()
        };
        let mut lit = glow(json!({}));
        lit.set_live(Some("glow"), "color", &json!([0.0, 1.0, 0.5]))
            .unwrap();
        assert_eq!(lit, glow(json!({"color": [0.0, 1.0, 0.5]})));
        let err = |l: &mut Layer, field: &str, value: Value| {
            l.set_live(None, field, &value).unwrap_err().to_string()
        };
        assert_eq!(
            err(&mut text, "opacity", json!(null)),
            "layer 't' opacity needs a number, got None"
        );
        assert_eq!(
            err(&mut text, "opacity", json!(2)),
            "layer 't' opacity needs a number from 0 to 1, got 2"
        );
        assert_eq!(
            err(&mut text, "text", json!(null)),
            "layer 't' text needs words in a string, got None"
        );
    }
}
