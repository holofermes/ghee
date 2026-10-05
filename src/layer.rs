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

use crate::colour;
use crate::engine::Blend;
use crate::scene::{hex_colour, keys_on, known_keys, number_list, py_float, py_repr, PASS_KEYS};
use crate::text::{self, Align, Font, Layout};

/// The keys of a layer.
pub(crate) const LAYER_KEYS: [&str; 12] = [
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

    /// Each uniform of the layer's own at its value from the scene, as its
    /// name, its type as a param takes it, and its value.
    pub fn uniforms(&self) -> Vec<(String, &'static str, Value)> {
        let (from, to, angle) = self.color.ends();
        let flag = |b: bool| Value::from(if b { 1.0 } else { 0.0 });
        vec![
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
        ]
    }

    /// The shaders of the layer's batches. `detail` is the atlas of an
    /// MTSDF layer.
    pub fn shaders(&self, detail: Detail) -> Shaders {
        let glyphs = match self.render {
            Render::Mtsdf => (
                self.vertex(mtsdf_varyings(detail), Grow::Reach(detail.reach())),
                self.glsl(MTSDF_FRAGMENT),
            ),
            Render::Slug => (
                self.vertex(slug_varyings(true), Grow::Pixel),
                self.slug_fragment(),
            ),
        };
        let mut boxes = PRECISION.join("\n");
        boxes.push('\n');
        boxes.push_str(&text::box_fragment().join("\n"));
        Shaders {
            glyphs,
            boxes: (self.vertex(slug_varyings(false), Grow::Pixel), boxes),
            colour: (
                self.vertex(colour_varyings(), Grow::Bitmap),
                self.glsl(colour::FRAGMENT),
            ),
        }
    }

    /// The vertex shader of a batch: the placement of the block, then the
    /// lines that write the batch's own varyings, after the lines that
    /// declare them, with each quad grown past its glyph by `grow`.
    fn vertex(&self, (decls, varyings): (&str, String), grow: Grow) -> String {
        let fit = match self.font_size {
            Some(px) => format!("    float ghee_fs = {};", py_float(px)),
            None => FIT.replace("FIT_MARGIN", &py_float(FIT_MARGIN)),
        };
        let grow = match grow {
            Grow::Pixel => "0.625 / ghee_ppe".to_string(),
            Grow::Reach(r) => format!("min(0.625 / ghee_ppe, {r:?})"),
            Grow::Bitmap => format!("min(0.625 / ghee_ppe, {:?} / ghee_a2.z)", colour::REACH),
        };
        let main = VERTEX
            .replace("FIT\n", &format!("{fit}\n"))
            .replace("GROW", &grow)
            .replace("DECLS", decls)
            .replace("VARYINGS\n", &varyings);
        self.glsl(&main)
    }

    /// The fragment shader of a slug layer: the readers of the layer's curves,
    /// then ghee's main.
    fn slug_fragment(&self) -> String {
        let mut f = vec![PRECISION.join("\n")];
        f.extend(text::curve_reader_lines(&self.name));
        f.push(self.glsl(SLUG_FRAGMENT));
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

/// How far a batch's quad reaches past its glyph's box.
#[derive(Clone, Copy)]
enum Grow {
    /// Half a pixel of antialiasing, and 1/8 px more, since VC4 lands a
    /// vertex up to 1/16 px off the position set by the shader.
    Pixel,
    /// As Pixel, and never past the field's reach in em, since a read past it
    /// would take a neighbouring cell's texels.
    Reach(f64),
    /// As Pixel, and never past colour::REACH texels of the bitmap.
    Bitmap,
}

/// The precision of every shader of a layer, a pass's own.
const PRECISION: [&str; 6] = crate::scene::PRECISION;

/// The vertex shader of every batch of a layer. DECLS declares the batch's
/// own uniforms and varyings, FIT sets ghee_fs, the pixels per em, GROW the
/// em that a quad reaches past its glyph's box, and VARYINGS writes the
/// batch's varyings from ghee_e, the vertex's point in em from the block's
/// top left, y down.
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
FIT
    // The pixels per em, held above 0 so that a block fitted to nothing
    // keeps every position finite, and the pass pixel of the block's top
    // left.
    float ghee_ppe = max(ghee_fs, 1.0 / 16384.0);
    vec2 ghee_org = mix(vec2(ghee_NAME_x, ghee_NAME_y), (resolution - NAME_size * ghee_fs) * 0.5,
                        ghee_NAME_centre);
    vec2 ghee_corner = mix(ghee_a1.xy, ghee_a1.zw, ghee_a0.xy);
    vec2 ghee_e = ghee_corner + (ghee_a0.xy * 2.0 - 1.0) * GROW;
VARYINGS
    v_texCoord = (ghee_org + ghee_e * ghee_fs) / resolution;
    gl_Position = vec4((v_texCoord * 2.0 - 1.0) * vec2(1.0, ghee_vflip), 0.0, 1.0);
    // A hidden layer's quads lie outside the pass.
    if (ghee_NAME_visible < 0.5) gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
}"#;

/// The lines that fit the block to its room.
const FIT: &str = r#"    float ghee_m = FIT_MARGIN * min(resolution.x, resolution.y);
    vec2 ghee_room = mix(resolution - ghee_m - vec2(ghee_NAME_x, ghee_NAME_y),
                         resolution - 2.0 * ghee_m, ghee_NAME_centre);
    vec2 ghee_s = ghee_room / max(NAME_size, vec2(1.0e-6));
    float ghee_fs = max(min(ghee_s.x, ghee_s.y), 0.0);"#;

/// The lines that place the point on the colour gradient into the x of
/// ghee_v1: 0 at its start, 1 at its end, along CSS's gradient line through
/// the block's centre.
const GRADIENT: &str = r#"    float ghee_r = radians(ghee_NAME_angle);
    vec2 ghee_dir = vec2(sin(ghee_r), -cos(ghee_r));
    float ghee_len = abs(NAME_size.x * ghee_dir.x) + abs(NAME_size.y * ghee_dir.y);
    float ghee_t = dot(ghee_e - 0.5 * NAME_size, ghee_dir) / max(ghee_len, 1.0e-6) + 0.5;
"#;

/// The varyings of an MTSDF layer's glyphs: the atlas coordinate, the field's
/// range in screen pixels, which msdfgen holds at 1 or more, the pixels per
/// em, and the point on the gradient.
fn mtsdf_varyings(detail: Detail) -> (&'static str, String) {
    let range_em = f64::from(detail.range) / f64::from(detail.px_em);
    let decls = "uniform float ghee_NAME_angle;\n\
                 uniform vec2 ghee_NAME_atlas_size;\n\
                 varying vec4 ghee_v0;\n\
                 varying vec4 ghee_v1;";
    let lines = format!(
        "    vec2 ghee_tc = ghee_a2.xy + (ghee_e - ghee_a1.xy) * ghee_a2.zw;\n\
         {GRADIENT}    ghee_v0 = vec4(ghee_tc / ghee_NAME_atlas_size, \
         max({range_em:?} * ghee_fs, 1.0), ghee_fs);\n\
         \x20   ghee_v1 = vec4(ghee_t, 0.0, 0.0, 0.0);\n"
    );
    (decls, lines)
}

/// The varyings of a slug layer's glyphs, with the point on the gradient, or
/// of its error boxes, without: the sample on the glyph's lattice, y up, and
/// the pixels spanned by a lattice step, the texel of the glyph's band
/// entries and its bands, and the lattice steps spanned by a pixel along x
/// and along y of the pass.
fn slug_varyings(gradient: bool) -> (&'static str, String) {
    let (decls, t) = match gradient {
        true => (
            "uniform float ghee_NAME_angle;\n\
             varying vec4 ghee_v0;\n\
             varying vec4 ghee_v1;\n\
             varying vec4 ghee_v4;",
            GRADIENT,
        ),
        false => (
            "varying vec4 ghee_v0;\n\
             varying vec4 ghee_v1;\n\
             varying vec4 ghee_v4;",
            "    float ghee_t = 0.0;\n",
        ),
    };
    let lines = format!(
        "    vec2 ghee_sz = ghee_a1.zw - ghee_a1.xy;\n\
         \x20   ghee_v0 = vec4(vec2(ghee_e.x - ghee_a1.x, ghee_a1.w - ghee_e.y) / ghee_sz * LATTICE,\n\
         \x20                  ghee_sz * ghee_ppe / LATTICE);\n\
         \x20   vec2 ghee_k = LATTICE / (ghee_sz * ghee_ppe);\n\
         \x20   ghee_v4 = vec4(ghee_k.x, 0.0, 0.0, -ghee_k.y);\n\
         {t}    ghee_v1 = vec4(ghee_a2.xyz, ghee_t);\n"
    );
    (decls, lines)
}

/// The varyings of a colour glyph: the atlas coordinate, and the bias that
/// holds GL's mip level at colour::MAX_LEVEL at most. GL picks the level from
/// the atlas texels spanned by a pixel.
fn colour_varyings() -> (&'static str, String) {
    let decls = "uniform vec2 ghee_NAME_emoji_size;\n\
                 varying vec4 ghee_v0;";
    let lines = format!(
        "    vec2 ghee_tc = ghee_a2.xy + (ghee_e - ghee_a1.xy) * ghee_a2.zw;\n\
         \x20   float ghee_lod = log2(max(ghee_a2.z / ghee_ppe, 1.0 / 65536.0));\n\
         \x20   ghee_v0 = vec4(ghee_tc / ghee_NAME_emoji_size, min(ghee_lod, {:?}) - ghee_lod, \
         0.0);\n",
        colour::MAX_LEVEL
    );
    (decls, lines)
}

/// The fragment shader of an MTSDF layer's glyphs.
const MTSDF_FRAGMENT: &str = r#"#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D ghee_NAME_atlas_tex;
uniform vec3 ghee_NAME_color;
uniform vec3 ghee_NAME_colorto;
uniform float ghee_NAME_opacity;
varying vec4 ghee_v0;
varying vec4 ghee_v1;
void main() {
    // On VC4 a short shader whose last texture read sits outside an if runs
    // in two threads, whose switch loses the condition flags. The read sits
    // in an if that the compiler cannot drop, so the shader stays on one
    // thread.
    vec4 ghee_t = vec4(0.0);
    if (gl_FragCoord.x >= 0.0) ghee_t = texture2D(ghee_NAME_atlas_tex, ghee_v0.xy);
    // The median of red, green and blue is the distance to the outline, 0.5
    // on it and growing inward over the range.
    float ghee_d = max(min(ghee_t.r, ghee_t.g), min(max(ghee_t.r, ghee_t.g), ghee_t.b)) - 0.5;
    float ghee_c = clamp(ghee_v0.z * ghee_d + 0.5, 0.0, 1.0);
    vec3 ghee_rgb = mix(ghee_NAME_color, ghee_NAME_colorto, clamp(ghee_v1.x, 0.0, 1.0));
    gl_FragColor = vec4(ghee_rgb, 1.0) * (ghee_c * ghee_NAME_opacity);
}"#;

/// The main of a slug layer's glyphs, after the readers of its curves.
const SLUG_FRAGMENT: &str = r#"uniform vec2 resolution;
uniform vec3 ghee_NAME_color;
uniform vec3 ghee_NAME_colorto;
uniform float ghee_NAME_opacity;
varying vec2 v_texCoord;
varying vec4 ghee_v0;
varying vec4 ghee_v1;
varying vec4 ghee_v4;
void main() {
    // VC4 lands a vertex up to 1/16 px off the position set by the vertex
    // shader, and the varyings follow the vertex, so the sample moves to the
    // pixel centre given by v_texCoord.
    vec2 ghee_px = v_texCoord * resolution;
    vec2 ghee_d = floor(ghee_px) + 0.5 - ghee_px;
    vec2 ghee_u = ghee_v0.xy + vec2(dot(ghee_v4.xy, ghee_d), dot(ghee_v4.zw, ghee_d));
    float ghee_c = ghee_NAME_outline(float(int(ghee_v1.x + 0.5)), vec2(ivec2(ghee_v1.yz + 0.5)),
                                     ghee_u, ghee_v0.zw);
    vec3 ghee_rgb = mix(ghee_NAME_color, ghee_NAME_colorto, clamp(ghee_v1.w, 0.0, 1.0));
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
            let d = json!({"name": name, "text": "", "render": render});
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
}
