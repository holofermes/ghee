//! Runs: consecutive layers of a pass that one batch draws in one draw.
//!
//! The members of a run are shape layers and slug text layers with no
//! effect, consecutive in their pass, with one blend. A run of shapes alone
//! draws through the shape program of `crate::shape`, a run of one text
//! through the text's own program, and a run of a text and any other member
//! through [`program`], which draws a shape or a glyph by the kind of each
//! vertex.
//!
//! [`program`] holds the code of the shape program and of a slug text's own
//! program, and reads from each glyph's vertex the settings that a text's
//! own batch takes as uniforms. A vertex of such a run holds ATTRIBUTES
//! vec4s. A shape's holds its own four of `crate::shape`, then kind 0 and
//! zeros. A glyph's holds its quad's vertex of `crate::text`, with the
//! settings in the lanes that a slug text with no effect leaves unread:
//!
//! - ghee_a0: the corner x and y, the gradient's angle, alpha at its end
//! - ghee_a1: the ink box in em
//! - ghee_a2: the texel of the glyph's band entries, its bands, opacity
//! - ghee_a3: the line index, the advance's start x, the first font's
//!   ascender in its units, its units per em
//! - ghee_a4: the kind, x, y, visible
//! - ghee_a5: centred on x and on y, the block's size in em
//! - ghee_a6: the font size, 0 to fit, the colour
//! - ghee_a7: the colour at the gradient's end, alpha
//!
//! A glyph's kind is 1, or 2 on the pixel grid. The program reads the font
//! size from the vertex where the text's own program holds it as a
//! constant, and a glyph's edge pixels can land one level apart.
//!
//! A pixel of [`program`] costs more than a pixel of a member's own program,
//! so a member whose quads cover more than its kind's cutoff stays out of a
//! run with a text: a text draws alone, a shape in a run of shapes.

use std::ops::Range;

use crate::engine::Blend;
use crate::layer::{run_glyph, run_guard, TextLayer, GLSL_NAME};
use crate::scene::PRECISION;
use crate::text;

/// The vec4 attributes of a vertex of a run that holds a text and another
/// member.
pub const ATTRIBUTES: usize = 8;

/// The floats of such a vertex.
const VERTEX: usize = 4 * ATTRIBUTES;

/// The lanes of a glyph's vertex that hold its settings, each the index of
/// its first float.
const ANGLE: usize = 2;
const ALPHATO: usize = 3;
const OPACITY: usize = 11;
const ASCENDER: usize = 14;
const UPEM: usize = 15;
const KIND: usize = 16;
const X: usize = 17;
const Y: usize = 18;
const VISIBLE: usize = 19;
const CENTRE: usize = 20;
const SIZE: usize = 22;
const FONT_SIZE: usize = 24;
const COLOR: usize = 25;
const COLORTO: usize = 28;
const ALPHA: usize = 31;

/// The most pixels that the quads of a text's glyphs cover in a run with
/// other members. On a Pi Zero 2 W's VC4 a text costs [`program`] about
/// 6 us of GPU time more than its own program, and 28 ns more for each of
/// those pixels, while a batch of its own costs about 29 us of CPU time a
/// frame.
pub(crate) const GLYPH_PIXELS: f64 = 800.0;

/// The most pixels that the quad of a shape covers in a run with a text. On
/// a Pi Zero 2 W's VC4 a shape costs [`program`] about 3.5 us of GPU time
/// more than the shape program, and 5.6 ns more for each of those pixels,
/// while a batch of shapes costs about 23 us of CPU time a frame.
pub(crate) const SHAPE_PIXELS: f64 = 3400.0;

/// How a member takes its place in a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    /// It joins any run.
    Joins,
    /// A shape past SHAPE_PIXELS, which joins no run that holds a text.
    Large,
    /// A text that draws alone: past GLYPH_PIXELS, or its glyphs read a
    /// texture of their own.
    Alone,
}

/// A layer of a pass that a run takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Candidate {
    pub text: bool,
    pub class: Class,
    pub blend: Blend,
}

/// The program that a run's batch draws with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// The shape program, for a run of shapes alone.
    Shapes,
    /// The text's own program, for a run of one text.
    Alone,
    /// [`program`], for a run of a text and another member.
    Mixed,
}

impl Mode {
    /// The mode of a run of `members` members, `texts` of them texts.
    pub(crate) fn of(members: usize, texts: usize) -> Mode {
        match (members, texts) {
            (_, 0) => Mode::Shapes,
            (1, _) => Mode::Alone,
            _ => Mode::Mixed,
        }
    }
}

/// A run as it takes layers: its blend, whether it holds a text and a Large
/// shape, and whether it takes no more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Open {
    blend: Blend,
    text: bool,
    big: bool,
    closed: bool,
}

impl Open {
    /// The run that starts at `c`.
    pub(crate) fn at(c: &Candidate) -> Open {
        Open {
            blend: c.blend,
            text: c.text,
            big: c.class == Class::Large,
            closed: c.class == Class::Alone,
        }
    }

    /// True when `c` joins the run: one blend, no text and Large shape
    /// together, and a text of class Alone by itself.
    pub(crate) fn takes(&self, c: &Candidate) -> bool {
        !self.closed
            && c.class != Class::Alone
            && self.blend == c.blend
            && !(c.text && self.big)
            && !(c.class == Class::Large && self.text)
    }

    /// The run with `c`, which joins it.
    pub(crate) fn take(&mut self, c: &Candidate) {
        self.text |= c.text;
        self.big |= c.class == Class::Large;
    }
}

/// The runs of the layers of a pass, each a span of `layers`, where None
/// stands for a layer that draws on its own. A run takes the layers from
/// its first on while each joins it.
pub(crate) fn partition(layers: &[Option<Candidate>]) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::new();
    // The run that the layer before joined.
    let mut open: Option<Open> = None;
    for (i, c) in layers.iter().enumerate() {
        let Some(c) = c else {
            open = None;
            continue;
        };
        match (open.as_mut().filter(|run| run.takes(c)), out.last_mut()) {
            (Some(run), Some(span)) => {
                run.take(c);
                span.end = i + 1;
            }
            _ => {
                open = Some(Open::at(c));
                out.push(i..i + 1);
            }
        }
    }
    out
}

/// The pixels that the quads `quads` of a text's glyphs cover at `fs`
/// pixels per em, each reaching 0.625 px past its glyph's box on each side.
pub(crate) fn glyph_pixels(quads: &[f32], fs: f64) -> f64 {
    quads
        .as_chunks::<{ 4 * text::QUAD_FLOATS }>()
        .0
        .iter()
        .map(|q| {
            let [x0, y0, x1, y1] = [q[4], q[5], q[6], q[7]].map(f64::from);
            ((x1 - x0) * fs + 1.25) * ((y1 - y0) * fs + 1.25)
        })
        .sum()
}

/// Append `quads`, a shape's quads, as a vertex of [`program`] holds them.
pub(crate) fn shape_vertices(quads: &[f32], out: &mut Vec<f32>) {
    for v in quads.as_chunks::<{ 4 * crate::shape::ATTRIBUTES }>().0 {
        out.extend_from_slice(v);
        out.extend_from_slice(&[0.0; VERTEX - 4 * crate::shape::ATTRIBUTES]);
    }
}

/// Append `quads`, the glyph quads of text layer `l`, whose block is `size`
/// em, as a vertex of [`program`] holds them, with `l`'s settings.
pub(crate) fn glyph_vertices(l: &TextLayer, quads: &[f32], size: [f64; 2], out: &mut Vec<f32>) {
    let (from, to, angle) = l.color.ends();
    let flag = |b: bool| if b { 1.0 } else { 0.0 };
    let mut v = [0.0f32; VERTEX];
    let mut put = |at: usize, values: &[f64]| {
        for (k, x) in values.iter().enumerate() {
            v[at + k] = *x as f32;
        }
    };
    put(KIND, &[1.0 + flag(l.grid().is_some())]);
    put(X, &[l.x.unwrap_or(0.0), l.y.unwrap_or(0.0)]);
    put(VISIBLE, &[flag(l.visible)]);
    put(CENTRE, &[flag(l.x.is_none()), flag(l.y.is_none())]);
    put(SIZE, &size);
    put(FONT_SIZE, &[l.font_size.unwrap_or(0.0)]);
    put(COLOR, &from.rgb);
    put(COLORTO, &to.rgb);
    put(ALPHA, &[from.alpha]);
    let (units, upem) = text::ascender(l.data.as_deref().unwrap_or_default());
    let lanes = [
        (ANGLE, angle as f32),
        (ALPHATO, to.alpha as f32),
        (OPACITY, l.opacity as f32),
        (ASCENDER, units as f32),
        (UPEM, upem as f32),
    ];
    for q in quads.as_chunks::<{ text::QUAD_FLOATS }>().0 {
        v[..text::QUAD_FLOATS].copy_from_slice(q);
        for (at, x) in lanes {
            v[at] = x;
        }
        out.extend_from_slice(&v);
    }
}

/// The GLSL of the `n` lanes from `at` of a vertex.
fn lane(at: usize, n: usize) -> String {
    let c = at % 4;
    format!("ghee_a{}.{}", at / 4, &"xyzw"[c..c + n])
}

/// The two shaders of every run that holds a text and another member.
pub fn program() -> (String, String) {
    let precision = PRECISION.join("\n");
    let kind = lane(KIND, 1);
    let (glyph_vs, glyph_fs) = run_glyph(
        &lane(FONT_SIZE, 1),
        &format!("{kind} > 1.5"),
        &lane(ASCENDER, 1),
        &lane(UPEM, 1),
    );
    let local = |ty: &str, token: &str, at: String| {
        format!(
            "    {ty} {} = {at};",
            crate::layer::uniform(GLSL_NAME, token)
        )
    };
    let mut vs = vec![precision.clone()];
    vs.extend((0..ATTRIBUTES).map(|k| format!("attribute vec4 ghee_a{k};")));
    vs.extend([
        "uniform vec3 iResolution;".to_string(),
        "uniform float ghee_vflip;".to_string(),
        VARYINGS.trim_end().to_string(),
        "void main() {".to_string(),
        format!("if ({kind} < 0.5) {{"),
        "    vec2 ghee_px;".to_string(),
        crate::shape::VERTEX_MAIN.trim_end().to_string(),
        "    ghee_pk = vec4(ghee_px, 0.0, 0.0);".to_string(),
        "    ghee_v5 = vec4(0.0);".to_string(),
        "} else {".to_string(),
        local("float", "x", lane(X, 1)),
        local("float", "y", lane(Y, 1)),
        local("float", "visible", lane(VISIBLE, 1)),
        local("vec2", "centre", lane(CENTRE, 2)),
        local("vec2", "size", lane(SIZE, 2)),
        local("float", "angle", lane(ANGLE, 1)),
        "    vec2 uv;".to_string(),
        "    vec4 ghee_v4;".to_string(),
        glyph_vs,
        format!("    ghee_pk = vec4(uv, {kind}, {});", lane(OPACITY, 1)),
        "    ghee_v2 = ghee_v4;".to_string(),
        format!(
            "    ghee_v3 = vec4({}, {});",
            lane(COLOR, 3),
            lane(ALPHA, 1)
        ),
        format!(
            "    ghee_v5 = vec4({}, {});",
            lane(COLORTO, 3),
            lane(ALPHATO, 1)
        ),
        "}".to_string(),
        "}".to_string(),
    ]);
    let mut fs = vec![precision];
    fs.extend(text::curve_reader_lines(GLSL_NAME));
    fs.extend([
        "uniform vec3 iResolution;".to_string(),
        VARYINGS.trim_end().to_string(),
        crate::shape::fragment_functions().trim_end().to_string(),
        "void main() {".to_string(),
        "if (ghee_pk.z < 0.5) {".to_string(),
        "    vec2 ghee_px = ghee_pk.xy;".to_string(),
        crate::shape::FRAGMENT_MAIN.trim_end().to_string(),
        "} else {".to_string(),
        "    vec2 uv = ghee_pk.xy;".to_string(),
        "    vec4 ghee_v4 = ghee_v2;".to_string(),
        local("vec3", "color", "ghee_v3.rgb".to_string()),
        local("float", "alpha", "ghee_v3.a".to_string()),
        local("vec3", "colorto", "ghee_v5.rgb".to_string()),
        local("float", "alphato", "ghee_v5.a".to_string()),
        local("float", "opacity", "ghee_pk.w".to_string()),
        glyph_fs.trim_end().to_string(),
        "}".to_string(),
        run_guard("ghee_pk.xy").trim_end().to_string(),
        "}".to_string(),
    ]);
    (vs.join("\n"), fs.join("\n"))
}

/// The varyings of [`program`]: ghee_pk holds a shape's pixel or a glyph's
/// point in the pass, then the kind and a glyph's opacity, and ghee_v0 to
/// ghee_v3 hold a shape's own, or a glyph's own with ghee_v4 in ghee_v2 and
/// its colour and alpha in ghee_v3, and in ghee_v5 those at the gradient's
/// end.
const VARYINGS: &str = "varying vec4 ghee_pk;
varying vec4 ghee_v0;
varying vec4 ghee_v1;
varying vec4 ghee_v2;
varying vec4 ghee_v3;
varying vec4 ghee_v5;
";

#[cfg(test)]
mod tests {
    use super::*;

    const fn c(text: bool, class: Class) -> Option<Candidate> {
        Some(Candidate {
            text,
            class,
            blend: Blend::Normal,
        })
    }

    #[test]
    fn a_run_takes_its_layers_while_each_joins_it() {
        // s a shape, S a Large shape, t a text, T a text alone, a a shape
        // of another blend, and - a layer that draws on its own.
        let s = c(false, Class::Joins);
        let big = c(false, Class::Large);
        let t = c(true, Class::Joins);
        let alone = c(true, Class::Alone);
        let add = Some(Candidate {
            text: false,
            class: Class::Joins,
            blend: Blend::Add,
        });
        type Row<'a> = (&'a [Option<Candidate>], &'a [(usize, usize)]);
        let rows: [Row; 9] = [
            (&[s, t, s, t], &[(0, 4)]),
            (&[s, big, s], &[(0, 3)]),
            (&[t, s, big, s, t], &[(0, 2), (2, 4), (4, 5)]),
            (&[big, s, t], &[(0, 2), (2, 3)]),
            (&[s, alone, s], &[(0, 1), (1, 2), (2, 3)]),
            (&[t, None, t], &[(0, 1), (2, 3)]),
            (&[s, add, s], &[(0, 1), (1, 2), (2, 3)]),
            (&[None, s, s, None], &[(1, 3)]),
            (&[t, alone, alone, t], &[(0, 1), (1, 2), (2, 3), (3, 4)]),
        ];
        for (layers, want) in rows {
            let want: Vec<Range<usize>> = want.iter().map(|(a, b)| *a..*b).collect();
            assert_eq!(partition(layers), want, "{layers:?}");
        }
    }
}
