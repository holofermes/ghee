//! Text layers: words that a pass paints over its buffer after its own
//! shader, in the order of its `layers`.
//!
//! A text layer owns its words, its font and its look. ghee declares a scene
//! string param under the layer's name, which holds the words, live, and
//! every pass of the scene except a filter pass reads them with
//! `<name>_cover` and `<name>_size`, whether the layer shows or not. The
//! layer draws one quad over each inked glyph through shaders that ghee
//! writes from its settings, [`Layer::shaders`].
//!
//! A setting that changes with no rebuild reaches those shaders as a uniform
//! of the layer's own, `ghee_<name>_<token>`, whose token holds no `_`, as
//! [`Layer::uniforms`] lists them. A layer name never ends in `_`, so the
//! name is all that comes before the last `_`, and two layers never share a
//! uniform.
//!
//! The block of the words sits at `x`, `y`, the pass pixel of its top left, y
//! down, and a missing axis centres the block on it. With no `font_size`, the
//! block takes the largest size at which it fits its room: the pass less
//! FIT_MARGIN of its shorter side on each edge, and on a given axis the pass
//! past the given pixel less the margin at the far edge. The shaders work the
//! size out from `<name>_size`, so it follows the words.

use anyhow::{bail, Result};
use serde_json::{Map, Value};

use crate::atlas;
use crate::colour;
use crate::engine::Blend;
use crate::scene::{hex_colour, keys_on, known_keys, number_list, py_float, py_repr, PASS_KEYS};
use crate::text::{self, Align, Font, Layout};

/// The keys of a layer.
pub(crate) const LAYER_KEYS: [&str; 19] = [
    "name",
    "visible",
    "x",
    "y",
    "color",
    "opacity",
    "blend",
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

/// The keys of a gradient.
const GRADIENT_KEYS: [&str; 3] = ["from", "to", "angle"];

/// The angle of a gradient that names none, as CSS's: top to bottom.
const GRADIENT_ANGLE: f64 = 180.0;

/// The margin on every edge of the room that fits a block with no
/// `font_size`, as a fraction of the pass's shorter side.
pub const FIT_MARGIN: f64 = 0.05;

/// A colour, each channel 0 to 1, and whether the scene wrote it as
/// `"#rrggbb"`, so a save writes it back in the form it came in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgb {
    pub rgb: [f64; 3],
    pub hex: bool,
}

impl Rgb {
    pub const WHITE: Rgb = Rgb {
        rgb: [1.0; 3],
        hex: false,
    };

    /// `"#rrggbb"` or `[r, g, b]`, None for any other value.
    pub fn from_json(v: &Value) -> Option<Rgb> {
        match v {
            Value::String(s) => hex_colour(s).map(|rgb| Rgb { rgb, hex: true }),
            v => number_list(v, 3).map(|n| Rgb {
                rgb: [n[0], n[1], n[2]],
                hex: false,
            }),
        }
    }

    pub fn to_json(self) -> Value {
        match self.hex {
            true => {
                let [r, g, b] = self
                    .rgb
                    .map(|c| (c * 255.0).round().clamp(0.0, 255.0) as u8);
                Value::from(format!("#{r:02x}{g:02x}{b:02x}"))
            }
            false => Value::from(self.rgb.to_vec()),
        }
    }

    /// The same colour, whatever form it came in.
    fn same(self, other: Rgb) -> bool {
        self.rgb == other.rgb
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
                "{what} color needs '#rrggbb', [r, g, b] or a gradient {{\"from\", \"to\", \
                 \"angle\"}}, got {}",
                py_repr(v)
            );
        };
        let at = format!("{what} color");
        known_keys(o, &at, &GRADIENT_KEYS, &[])?;
        let end = |k: &str| match o.get(k).filter(|v| !v.is_null()) {
            None => bail!("{at} has no '{k}'"),
            Some(v) => Rgb::from_json(v).ok_or_else(|| {
                anyhow::anyhow!(
                    "{at} {k} needs '#rrggbb' or three numbers [r, g, b], got {}",
                    py_repr(v)
                )
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
    fn ends(self) -> (Rgb, Rgb, f64) {
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
    /// A colour, `"#rrggbb"` or `[r, g, b]`.
    Colour,
}

/// A field of an effect: its key, its default, what it takes, and whether a
/// change to it reaches the layer's shaders with no rebuild.
struct Field {
    name: &'static str,
    default: FieldValue,
    takes: Takes,
    live: bool,
}

/// The value of an effect field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldValue {
    Num(f64),
    Rgb(Rgb),
}

const fn num(name: &'static str, default: f64, takes: Takes, live: bool) -> Field {
    Field {
        name,
        default: FieldValue::Num(default),
        takes,
        live,
    }
}

/// A colour field, live, its default `"#rrggbb"` written as its bytes.
const fn colour(default: [u8; 3]) -> Field {
    let [r, g, b] = default;
    Field {
        name: "color",
        default: FieldValue::Rgb(Rgb {
            rgb: [r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0],
            hex: true,
        }),
        takes: Takes::Colour,
        live: true,
    }
}

const EM: &str = "a number of em";

const SHADOW: [Field; 5] = [
    num("x", 0.06, Takes::Number(EM), false),
    num("y", 0.08, Takes::Number(EM), false),
    num("blur", 0.15, Takes::Em, false),
    colour([0, 0, 0]),
    num("strength", 0.7, Takes::Unit, true),
];
const GLOW: [Field; 3] = [
    num("blur", 0.3, Takes::Em, false),
    colour([0xff, 0xa5, 0x00]),
    num("strength", 0.8, Takes::Unit, true),
];
const STROKE: [Field; 2] = [num("width", 0.06, Takes::Em, false), colour([0, 0, 0])];
const KARAOKE: [Field; 2] = [
    num("sung", 0.0, Takes::Number("a number of characters"), true),
    colour([0xff, 0xad, 0x2e]),
];
const WAVE: [Field; 2] = [
    num("height", 0.1, Takes::Number(EM), false),
    num(
        "speed",
        0.6,
        Takes::Number("a number of waves a second"),
        true,
    ),
];
const SCROLL: [Field; 1] = [num(
    "speed",
    2.0,
    Takes::Number("a number of em a second"),
    true,
)];
const BLINK: [Field; 1] = [num("rate", 1.0, Takes::Rate("blinks"), true)];

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
                Takes::Colour => FieldValue::Rgb(
                    Rgb::from_json(v).ok_or_else(|| got("'#rrggbb' or three numbers [r, g, b]"))?,
                ),
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

/// A text layer of a pass.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub name: String,
    /// False hides the layer. Its words still read through `<name>_cover`.
    pub visible: bool,
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub color: Color,
    pub opacity: f64,
    pub blend: Blend,
    /// The words that the layer starts with. The scene string param of the
    /// layer's name holds them from then on, live.
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

impl Layer {
    /// `at` names the layer by its place in an error, as "the 2nd layer of
    /// pass 'main'", until its name is known.
    pub fn from_dict(at: &str, d: &Value) -> Result<Layer> {
        let Value::Object(o) = d else {
            bail!("{at} needs an object, got {}", py_repr(d));
        };
        let what = match o.get("name") {
            Some(Value::String(n)) => format!("layer '{n}'"),
            _ => at.to_string(),
        };
        let elsewhere = keys_on(&LAYER_KEYS, &PASS_KEYS, "its pass");
        known_keys(o, &what, &LAYER_KEYS, &elsewhere)?;
        let name = match o.get("name").filter(|v| !v.is_null()) {
            None => bail!("{at} has no 'name'"),
            Some(Value::String(n)) => n.clone(),
            Some(v) => bail!("{at} name needs a string, got {}", py_repr(v)),
        };
        let given = |k: &str| o.get(k).filter(|v| !v.is_null());
        let number = |k: &str, ok: fn(f64) -> bool, needs: &str| match given(k) {
            None => Ok(None),
            Some(v) => match v.as_f64() {
                Some(f) if f.is_finite() && ok(f) => Ok(Some(f)),
                _ => bail!("{what} {k} needs {needs}, got {}", py_repr(v)),
            },
        };
        let visible = match given("visible") {
            None => true,
            Some(Value::Bool(b)) => *b,
            Some(v) => bail!("{what} visible needs true or false, got {}", py_repr(v)),
        };
        let x = number("x", |_| true, "a number of pixels")?;
        let y = number("y", |_| true, "a number of pixels")?;
        let color = match given("color") {
            None => Color::Solid(Rgb::WHITE),
            Some(v) => Color::from_json(&what, v)?,
        };
        let opacity = number(
            "opacity",
            |f| (0.0..=1.0).contains(&f),
            "a number from 0 to 1",
        )?
        .unwrap_or(1.0);
        let blend = match given("blend") {
            None => Blend::Normal,
            Some(v) => match Blend::NAMES.iter().find(|(_, n)| v.as_str() == Some(n)) {
                Some((b, _)) => *b,
                None => bail!(
                    "{what} blend needs 'normal', 'add', 'multiply' or 'screen', got {}",
                    py_repr(v)
                ),
            },
        };
        let text = match given("text") {
            None => bail!("{what} has no 'text', the words it draws"),
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
            None => Render::Mtsdf,
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
        Ok(Layer {
            name,
            visible,
            x,
            y,
            color,
            opacity,
            blend,
            text,
            font_size,
            data: font_family.is_none().then(|| vec![text::built_in_font()]),
            font_family,
            text_align,
            render,
            effects,
        })
    }

    /// The layer as a scene holds it, a key at its default left out, with
    /// `words` in `text`.
    pub fn to_dict(&self, words: &str) -> Value {
        let mut d = Map::new();
        d.insert("name".to_string(), Value::from(self.name.clone()));
        if !self.visible {
            d.insert("visible".to_string(), Value::from(false));
        }
        for (k, v) in [("x", self.x), ("y", self.y)] {
            if let Some(v) = v {
                d.insert(k.to_string(), Value::from(v));
            }
        }
        if !matches!(self.color, Color::Solid(c) if c.same(Rgb::WHITE)) {
            d.insert("color".to_string(), self.color.to_json());
        }
        if self.opacity != 1.0 {
            d.insert("opacity".to_string(), Value::from(self.opacity));
        }
        if self.blend != Blend::Normal {
            d.insert("blend".to_string(), Value::from(self.blend.name()));
        }
        d.insert("text".to_string(), Value::from(words));
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
        if self.render == Render::Slug {
            d.insert("render".to_string(), Value::from("slug"));
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

    fn lay(&self, words: &str, curves: bool) -> Result<Layout> {
        let Some(fonts) = &self.data else {
            bail!(
                "layer '{}': {} was never loaded, the scene was built without a directory",
                self.name,
                self.font_name(0)
            );
        };
        Ok(match curves {
            true => text::build(fonts, words, self.text_align),
            false => text::place(fonts, words, self.text_align),
        })
    }

    /// The font at index `i` of the layer's fonts in an error: "font
    /// 'sans'", or "the built-in font".
    pub fn font_name(&self, i: usize) -> String {
        match (i, &self.font_family) {
            (0, Some(f)) => format!("font '{f}'"),
            _ => "the built-in font".to_string(),
        }
    }

    /// True when a font of the layer that loaded holds colour bitmaps.
    pub fn has_colour(&self) -> bool {
        self.data.as_deref().is_some_and(text::any_colour)
    }

    /// True when a font of the layer that loaded holds outlines.
    pub fn has_outlines(&self) -> bool {
        self.data.as_deref().is_some_and(text::any_outlines)
    }

    /// ghee's name for the uniform `token` of this layer.
    pub fn uniform(&self, token: &str) -> String {
        uniform(&self.name, token)
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
        let (from, to, angle) = self.color.ends();
        let flag = |b: bool| Value::from(if b { 1.0 } else { 0.0 });
        let mut out = vec![
            (
                self.uniform("x"),
                "float",
                Value::from(self.x.unwrap_or(0.0)),
            ),
            (
                self.uniform("y"),
                "float",
                Value::from(self.y.unwrap_or(0.0)),
            ),
            (
                self.uniform("centre"),
                "vec2",
                Value::from(vec![flag(self.x.is_none()), flag(self.y.is_none())]),
            ),
            (self.uniform("visible"), "float", flag(self.visible)),
            (self.uniform("opacity"), "float", Value::from(self.opacity)),
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
        ];
        for e in self.drawn() {
            let token = e.kind.token();
            out.push((
                self.uniform(&format!("{token}visible")),
                "float",
                flag(e.visible),
            ));
            for (f, v) in e.kind.fields().iter().zip(&e.values) {
                let name = self.uniform(&format!("{token}{}", f.name));
                match (f.live, v) {
                    (false, _) => {}
                    (true, FieldValue::Num(n)) => out.push((name, "float", Value::from(*n))),
                    (true, FieldValue::Rgb(c)) => {
                        out.push((name, "vec3", Value::from(c.rgb.to_vec())));
                    }
                }
            }
        }
        out
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
        if wave.is_some() || scroll.is_some() || blink.is_some() {
            decls.push("uniform float time;".to_string());
        }
        // The motion effects' numbers reach the vertex stage, and every
        // colour the fragment stage.
        for e in self.drawn().filter(|e| !e.kind.reads_distance()) {
            let token = e.kind.token();
            decls.push(format!("uniform float ghee_NAME_{token}visible;"));
            for f in e.kind.fields().iter().filter(|f| f.live) {
                if !matches!(f.takes, Takes::Colour) {
                    decls.push(format!("uniform float ghee_NAME_{token}{};", f.name));
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
                main.push("    float ghee_fs = max(min(ghee_s.x, ghee_s.y), 0.0);".to_string());
            }
        }
        main.push(PLACE.to_string());
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
        main.push(format!(
            "    // A quad reaches half a pixel past its glyph's box for the\n    \
             // antialiasing, and 1/8 px more, since VC4 lands a vertex up to 1/16\n    \
             // px off the position set by this shader.\n    \
             float ghee_g = {grow};\n    \
             vec2 ghee_corner = mix(ghee_a1.xy, ghee_a1.zw, ghee_a0.xy);\n    \
             vec2 ghee_e = ghee_corner + (ghee_a0.xy * 2.0 - 1.0) * ghee_g;\n    \
             vec2 ghee_mv = vec2(0.0);"
        ));
        if let Some(w) = wave {
            main.push(format!(
                "    // Each letter trails the one before it by a tenth of a wave.\n    \
                 ghee_mv.y += {:?} * sin(6.2831853 * (time * ghee_NAME_wavespeed - 0.1 * ghee_a0.z))\n    \
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
        let mut hide = "ghee_NAME_visible < 0.5".to_string();
        if blink.is_some() {
            hide.push_str(
                " || (ghee_NAME_blinkvisible > 0.5 && fract(time * ghee_NAME_blinkrate) >= 0.5)",
            );
        }
        main.push(format!(
            "    v_texCoord = (ghee_org + (ghee_e + ghee_mv) * ghee_fs) / resolution;\n    \
             gl_Position = vec4((v_texCoord * 2.0 - 1.0) * vec2(1.0, ghee_vflip), 0.0, 1.0);\n    \
             // A hidden glyph's quad lies outside the pass.\n    \
             if ({hide}) gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);"
        ));
        let v = VERTEX
            .replace("DECLS", &decls.join("\n"))
            .replace("MAIN", &main.join("\n"));
        self.glsl(&v)
    }

    /// The varyings of an MTSDF layer's glyphs, and the lines that declare
    /// them: the atlas coordinate, the field's range in screen pixels, which
    /// msdfgen holds at 1 or more, the pixels per em, the point on the
    /// gradient, the karaoke fill, and which copy the quad is. A doubled
    /// layer's quads also carry the box of the glyph's cell, the atlas
    /// coordinate of the shadow, and the pixels to each edge of the quad.
    fn mtsdf_varyings(&self, detail: Detail) -> (String, String) {
        let range_em = f64::from(detail.range) / f64::from(detail.px_em);
        let mut decls = vec![
            "uniform float ghee_NAME_angle;",
            "uniform vec2 ghee_NAME_atlas_size;",
            "varying vec4 ghee_v0;",
            "varying vec4 ghee_v1;",
        ];
        let kf = match self.draws(Kind::Karaoke) {
            Some(_) => "ghee_kf",
            None => "0.0",
        };
        let mut lines = format!(
            "    vec2 ghee_tc = ghee_a2.xy + (ghee_e - ghee_a1.xy) * ghee_a2.zw;\n\
             {GRADIENT}    ghee_v0 = vec4(ghee_tc / ghee_NAME_atlas_size, \
             max({range_em:?} * ghee_fs, 1.0), ghee_fs);\n    \
             ghee_v1 = vec4(ghee_t, {kf}, ghee_a3.w, 0.0);\n"
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
                 vec2 ghee_t1 = ghee_a2.xy + (ghee_a1.zw - ghee_a1.xy) * ghee_a2.zw;\n    \
                 ghee_v2 = vec4(min(ghee_a2.xy, ghee_t1) - {half:?}, max(ghee_a2.xy, ghee_t1) + {half:?})\n    \
                 \x20         / ghee_NAME_atlas_size.xyxy;\n    \
                 // The shadow shows the letters' shape at the point less its offset.\n    \
                 ghee_v3 = vec4((ghee_tc - vec2({sx:?}, {sy:?}) * ghee_a2.zw) / ghee_NAME_atlas_size, \
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
            "uniform vec3 ghee_NAME_color;".to_string(),
            "uniform vec3 ghee_NAME_colorto;".to_string(),
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
            if matches!(e.kind, Kind::TextShadow | Kind::Glow) {
                decls.push(format!("uniform float ghee_NAME_{token}strength;"));
            }
        }
        let letters = format!(
            "        // The median of red, green and blue is the distance to the outline,\n        \
             // 0.5 on it and growing inward over the range.\n        \
             float ghee_d = max(min(ghee_t.r, ghee_t.g), min(max(ghee_t.r, ghee_t.g), ghee_t.b)) - 0.5;\n        \
             float ghee_c = clamp(ghee_v0.z * ghee_d + 0.5, 0.0, 1.0);\n{}        \
             ghee_o = vec4(ghee_rgb, 1.0) * ghee_c;",
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
                     \x20             * ghee_NAME_shadowstrength * ghee_NAME_shadowvisible;\n        \
                     ghee_o = vec4(ghee_NAME_shadowcolor, 1.0) * ghee_a;",
                    e.num("blur")
                ));
            }
            if let Some(e) = self.draws(Kind::Glow) {
                main.push(format!(
                    "        // The glow falls from the outline to nothing at its blur.\n        \
                     float ghee_f = clamp(1.0 - max(-ghee_in, 0.0) / {:?}, 0.0, 1.0);\n        \
                     float ghee_ga = ghee_f * ghee_f * ghee_NAME_glowstrength * ghee_NAME_glowvisible;\n        \
                     ghee_o = vec4(ghee_NAME_glowcolor, 1.0) * ghee_ga + ghee_o * (1.0 - ghee_ga);",
                    e.num("blur").max(1.0e-6)
                ));
            }
            if let Some(e) = self.draws(Kind::TextStroke) {
                main.push(format!(
                    "        // The stroke grows the letters by its width, its edge antialiased\n        \
                     // over a pixel.\n        \
                     float ghee_sa = clamp((ghee_in + {:?}) * ghee_v0.w + 0.5, 0.0, 1.0)\n        \
                     \x20              * ghee_NAME_strokevisible;\n        \
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

    /// The lines that set ghee_rgb, the letters' colour at a pixel, after
    /// `indent`: the colour, or the point `t` on its gradient, and the
    /// karaoke colour where the fill `fill` reaches.
    fn letter_colour(&self, indent: &str, t: &str, fill: &str) -> String {
        let mut l = format!(
            "{indent}vec3 ghee_rgb = mix(ghee_NAME_color, ghee_NAME_colorto, clamp({t}, 0.0, 1.0));\n"
        );
        if self.draws(Kind::Karaoke).is_some() {
            l.push_str(&format!(
                "{indent}ghee_rgb = mix(ghee_rgb, ghee_NAME_karaokecolor,\n{indent}               \
                 clamp({fill}, 0.0, 1.0) * ghee_NAME_karaokevisible);\n"
            ));
        }
        l
    }

    /// The fragment shader of a slug layer: the readers of the layer's curves,
    /// then ghee's main.
    fn slug_fragment(&self) -> String {
        let mut decls = vec![
            "uniform vec2 resolution;",
            "uniform vec3 ghee_NAME_color;",
            "uniform vec3 ghee_NAME_colorto;",
            "uniform float ghee_NAME_opacity;",
            "varying vec2 v_texCoord;",
            "varying vec4 ghee_v0;",
            "varying vec4 ghee_v1;",
            "varying vec4 ghee_v4;",
        ];
        let karaoke = self.draws(Kind::Karaoke).is_some();
        if karaoke {
            decls.extend([
                "uniform float ghee_NAME_karaokevisible;",
                "uniform vec3 ghee_NAME_karaokecolor;",
                "varying vec4 ghee_v2;",
            ]);
        }
        let main = SLUG_MAIN.replace(
            "COLOUR\n",
            &self.letter_colour("    ", "ghee_v1.w", "ghee_v2.x"),
        );
        let mut f = vec![PRECISION.join("\n")];
        f.extend(text::curve_reader_lines(&self.name));
        f.push(self.glsl(&format!("{}\n{main}", decls.join("\n"))));
        f.join("\n")
    }

    /// `glsl` with the layer's name for NAME and the lattice for LATTICE.
    fn glsl(&self, glsl: &str) -> String {
        glsl.replace("LATTICE", &format!("{:?}", text::LATTICE))
            .replace("NAME", &self.name)
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
uniform vec2 resolution;
uniform float ghee_vflip;
uniform vec2 NAME_size;
uniform float ghee_NAME_x;
uniform float ghee_NAME_y;
uniform vec2 ghee_NAME_centre;
uniform float ghee_NAME_visible;
varying vec2 v_texCoord;
DECLS
void main() {
MAIN
}"#;

/// The lines that fit the block to its room, up to ghee_s, the pixels per
/// em that fit each axis.
const FIT: &str = r#"    float ghee_m = FIT_MARGIN * min(resolution.x, resolution.y);
    vec2 ghee_room = mix(resolution - ghee_m - vec2(ghee_NAME_x, ghee_NAME_y),
                         resolution - 2.0 * ghee_m, ghee_NAME_centre);
    vec2 ghee_s = ghee_room / max(NAME_size, vec2(1.0e-6));"#;

/// The lines that set ghee_ppe, the pixels per em held above 0 so that a
/// block fitted to nothing keeps every position finite, and ghee_org, the
/// pass pixel of the block's top left.
const PLACE: &str = r#"    float ghee_ppe = max(ghee_fs, 1.0 / 16384.0);
    vec2 ghee_org = mix(vec2(ghee_NAME_x, ghee_NAME_y), (resolution - NAME_size * ghee_fs) * 0.5,
                        ghee_NAME_centre);"#;

/// The lines that scroll the block from right to left by whole pixels: it
/// enters at the pass's right edge, and starts there again once it has left
/// at the left.
const SCROLL_LINES: &str = r#"    float ghee_trip = resolution.x + NAME_size.x * ghee_fs;
    float ghee_sx = floor(resolution.x - mod(time * ghee_NAME_scrollspeed * ghee_fs, ghee_trip));
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
    float ghee_len = abs(NAME_size.x * ghee_dir.x) + abs(NAME_size.y * ghee_dir.y);
    float ghee_t = dot(ghee_e - 0.5 * NAME_size, ghee_dir) / max(ghee_len, 1.0e-6) + 0.5;
"#;

/// The comment over the reads of a fragment shader.
const GUARD: &str = r#"    // On VC4 a short shader whose last texture read sits outside an if runs
    // in two threads, whose switch loses the condition flags. Every read sits
    // in an if that the compiler cannot drop, so the shader stays on one
    // thread.
"#;

/// A read of the layer's atlas at UV.
const GUARDED_READ: &str =
    "    if (gl_FragCoord.x >= 0.0) ghee_t = texture2D(ghee_NAME_atlas_tex, UV);";

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

/// The main of a slug layer's glyphs, after the readers of its curves.
/// COLOUR sets ghee_rgb.
const SLUG_MAIN: &str = r#"void main() {
    // VC4 lands a vertex up to 1/16 px off the position set by the vertex
    // shader, and the varyings follow the vertex, so the sample moves to the
    // pixel centre given by v_texCoord.
    vec2 ghee_px = v_texCoord * resolution;
    vec2 ghee_d = floor(ghee_px) + 0.5 - ghee_px;
    vec2 ghee_u = ghee_v0.xy + vec2(dot(ghee_v4.xy, ghee_d), dot(ghee_v4.zw, ghee_d));
    float ghee_c = ghee_NAME_outline(float(int(ghee_v1.x + 0.5)), vec2(ivec2(ghee_v1.yz + 0.5)),
                                     ghee_u, ghee_v0.zw);
COLOUR
    gl_FragColor = vec4(ghee_rgb, 1.0) * (ghee_c * ghee_NAME_opacity);
    // On VC4 a shader whose last texture read sits outside an if runs in two
    // threads, whose switch loses the condition flags. A read after ghee's
    // own code that never runs, in an if that the compiler cannot drop,
    // keeps the shader on one thread.
    if (gl_FragCoord.x < 0.0) gl_FragColor = texture2D(ghee_NAME_curves_tex, v_texCoord);
}"#;

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

    /// Every name with ghee's prefix in the shaders and the uniforms of a
    /// layer `name` in either render, and in the readers of its words.
    fn written(name: &str) -> Vec<String> {
        let mut out: Vec<String> = text::template_names(name);
        for render in ["mtsdf", "slug"] {
            let mut d = json!({"name": name, "text": "", "render": render});
            for kind in Kind::ALL {
                d[kind.key()] = json!({});
            }
            let l = Layer::from_dict("a layer", &d).unwrap();
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
        out.retain(|w| w.starts_with(crate::scene::PREFIX));
        out.sort();
        out.dedup();
        out
    }

    #[test]
    fn two_layers_never_meet_in_a_name_that_ghee_writes_for_them() {
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
            "a_emoji",
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
        for (i, (x, mx)) in names.iter().zip(&made).enumerate() {
            let own = format!("{}{x}_", crate::scene::PREFIX);
            assert!(mx.iter().any(|n| n.starts_with(&own)), "{x}");
            for (y, my) in names[i + 1..].iter().zip(&made[i + 1..]) {
                // ghee's names for one layer are its own, and the rest of
                // the names that its shaders hold are the same in every
                // layer's.
                let mine = |n: &&String, of: &str| {
                    n.starts_with(&format!("{}{of}_", crate::scene::PREFIX))
                };
                let met: Vec<&String> = mx
                    .iter()
                    .filter(|n| my.contains(n) && (mine(n, x) || mine(n, y)))
                    .collect();
                assert!(met.is_empty(), "{x} and {y} meet in {met:?}");
            }
        }
    }

    fn layer(keys: serde_json::Value) -> Layer {
        let mut d = json!({"name": "t", "text": "I"});
        d.as_object_mut()
            .unwrap()
            .extend(keys.as_object().unwrap().clone());
        Layer::from_dict("a layer", &d).unwrap()
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
}
