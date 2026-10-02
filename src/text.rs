//! Text drawn on the GPU from the outlines of a font.
//!
//! A scene declares a text under the name of one of its string params, and the
//! font asset to draw it with. Every pass of the scene then reads the coverage of
//! those words at any point with `<name>_cover(p, ppem)`. The words change
//! with the param, live, with no recompile.
//!
//! Coverage follows the Slug algorithm by Eric Lengyel
//! (https://jcgt.org/published/0006/02/02/, https://github.com/EricLengyel/Slug).
//! A glyph is a set of quadratic Bézier curves, sorted into horizontal and
//! vertical bands. A pixel casts a ray along x and a ray along y through its
//! band, and the curves the rays cross give its coverage, antialiased at any
//! scale.
//!
//! The data rides in one RGBA8 texture read with NEAREST, so a GL without
//! float or integer textures, the VC4, holds it. A 16-bit value spans two
//! bytes of a texel, a 24-bit offset three, and a curve two texels.

use std::collections::{hash_map::Entry, HashMap};
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use serde_json::{Map, Value};

/// Texture width in texels. A power of two keeps every texel centre an exact
/// binary fraction, so a NEAREST read lands on the texel the index names.
const TEX_W: usize = 1024;
/// VC4's GL_MAX_TEXTURE_SIZE.
const MAX_TEX_H: usize = 2048;
/// Glyphs one grid cell lists. The shader loops over them.
const CELL_K: usize = 4;
/// Texels a grid cell spends on one glyph. A cell's CELL_K * SLOT texels divide
/// TEX_W, so a cell sits in one texture row, and the shader steps through it
/// with both numbers written in.
const SLOT: usize = 2;
/// The shader's loop bound over the curves of one band.
const BAND_CURVES: usize = 64;
/// Bands across a glyph's box on each axis. Words whose glyphs outgrow the
/// texture take half as many, down to one.
const BANDS: usize = 64;
/// Ink past a glyph's box that a cell still answers for, in em: the half
/// pixel of antialiasing at 8 pixels per em.
const INK_MARGIN: f64 = 1.0 / 16.0;
/// Bands overlap by this much, in em, so a sample on a band edge sees every
/// curve near it.
const BAND_EPS: f64 = 1.0 / 1024.0;
/// A cubic splits into quadratics until each strays less than this, in em.
const CUBIC_TOL: f64 = 1.0 / 4096.0;
/// Control points sit on a lattice of this many steps across the glyph's box,
/// 11 bits a coordinate. A point lands within half a step of the outline on each
/// axis, 1/4094 of the box: 0.17 px for a 1 em box at 700 px per em.
const LATTICE: f64 = 2047.0;
/// p2 rides as its offset from p3 in 10 bits, so a curve whose p2 lies further
/// from p3 splits in two.
const OFFSET_MAX: i32 = 511;
/// A glyph box size is 16 bits at 1/8192 em, and so is its corner, offset by
/// 7 em, from the corner of a grid cell that lists the glyph.
const BOX_SCALE: f64 = 8192.0;
const CORNER_OFFSET: f64 = 7.0;
/// Glyph origins and the grid corner sit on a 1/4096 em lattice.
const ORIGIN_SCALE: f64 = 4096.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Align {
    Left,
    Center,
    Right,
}

impl Align {
    fn as_str(self) -> &'static str {
        match self {
            Align::Left => "left",
            Align::Center => "center",
            Align::Right => "right",
        }
    }
}

/// A text a scene declares. Its words are the scene string param of the same
/// name.
#[derive(Clone, Debug, PartialEq)]
pub struct Text {
    pub name: String,
    /// The name of the scene's font asset.
    pub font: String,
    pub align: Align,
    /// The font asset's bytes. load_assets fills them.
    pub data: Option<Arc<Vec<u8>>>,
}

impl Text {
    pub fn from_dict(name: &str, d: &Value) -> Result<Text> {
        let Value::Object(o) = d else {
            bail!("text '{name}' needs an object, got {d}");
        };
        let font = match o.get("font") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            Some(v) => bail!("text '{name}' 'font' needs the name of a font asset, got {v}"),
            None => bail!("text '{name}' has no 'font'"),
        };
        let align = match o.get("align").filter(|v| !v.is_null()) {
            None => Align::Left,
            Some(Value::String(s)) if s == "left" => Align::Left,
            Some(Value::String(s)) if s == "center" => Align::Center,
            Some(Value::String(s)) if s == "right" => Align::Right,
            Some(v) => bail!("text '{name}' align needs 'left', 'center' or 'right', got {v}"),
        };
        Ok(Text {
            name: name.to_string(),
            font,
            align,
            data: None,
        })
    }

    pub fn to_dict(&self) -> Value {
        let mut d = Map::new();
        d.insert("font".to_string(), Value::from(self.font.clone()));
        if self.align != Align::Left {
            d.insert("align".to_string(), Value::from(self.align.as_str()));
        }
        Value::Object(d)
    }

    /// Lay out `words` and pack them. Fails when the font was never loaded.
    pub fn build(&self, words: &str) -> Result<Built> {
        let data = self.data.as_ref().ok_or_else(|| {
            anyhow!(
                "text '{}': font '{}' was never loaded, the scene was built without a directory",
                self.name,
                self.font
            )
        })?;
        build(data, words, self.align).map_err(|e| anyhow!("text '{}': {e}", self.name))
    }
}

/// The texture and the uniforms of one laid-out text.
pub struct Built {
    pub width: u32,
    pub height: u32,
    /// RGBA8, row-major.
    pub texels: Vec<u8>,
    /// Uniform suffixes after the text's name, each a float.
    pub uniforms: Vec<(&'static str, f64)>,
    /// One quad per inked glyph, for a draw. Its glyph data is this texture's.
    pub quads: Quads,
}

/// Floats a quad vertex holds: the attributes a0 to a3, a vec4 each.
pub const QUAD_FLOATS: usize = 16;

/// The quads of a text's inked glyphs, 4 vertices a quad at the corners
/// (0, 0), (1, 0), (0, 1), (1, 1) of the glyph's box from its top left, so the
/// triangles 0 1 2 and 2 1 3 cover it. A vertex holds QUAD_FLOATS floats:
///
/// - a0: the corner x and y, the character index, the word index
/// - a1: the ink box in em from the block's top left, y down: x0, y0, x1, y1
/// - a2: the texel of the glyph's band entries, its horizontal bands, its
///   vertical bands, 0
/// - a3: the line index, the advance's start and end x in em, 0
///
/// Every index counts from 0.
pub struct Quads {
    pub floats: Vec<f32>,
    pub per_vertex: usize,
    pub count: usize,
}

/// One quadratic curve, three control points in em.
type Quad = [[f64; 2]; 3];

/// Collects a glyph's contours as quadratics, in em.
struct Outline {
    scale: f64,
    start: [f64; 2],
    at: [f64; 2],
    quads: Vec<Quad>,
}

impl Outline {
    fn point(&self, x: f32, y: f32) -> [f64; 2] {
        [f64::from(x) * self.scale, f64::from(y) * self.scale]
    }

    fn push(&mut self, q: Quad) {
        if q[0] != q[1] || q[0] != q[2] {
            self.quads.push(q);
        }
    }

    /// A line is the quadratic {p1, p2, p2}.
    fn line(&mut self, p: [f64; 2]) {
        let at = self.at;
        self.push([at, p, p]);
        self.at = p;
    }
}

fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

impl ttf_parser::OutlineBuilder for Outline {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.point(x, y);
        self.start = p;
        self.at = p;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.point(x, y);
        self.line(p);
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let (c, p) = (self.point(x1, y1), self.point(x, y));
        let at = self.at;
        self.push([at, c, p]);
        self.at = p;
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let (p0, c1, c2, p3) = (
            self.at,
            self.point(x1, y1),
            self.point(x2, y2),
            self.point(x, y),
        );
        // One quadratic strays from a cubic by sqrt(3)/36 |p3 - 3c2 + 3c1 - p0|,
        // and n pieces cut that by n^3.
        let d = [
            p3[0] - 3.0 * c2[0] + 3.0 * c1[0] - p0[0],
            p3[1] - 3.0 * c2[1] + 3.0 * c1[1] - p0[1],
        ];
        let err = 3f64.sqrt() / 36.0 * d[0].hypot(d[1]);
        let n = ((err / CUBIC_TOL).cbrt().ceil() as usize).clamp(1, 32);
        let at = |t: f64| {
            let (a, b, c) = (lerp(p0, c1, t), lerp(c1, c2, t), lerp(c2, p3, t));
            let (d, e) = (lerp(a, b, t), lerp(b, c, t));
            lerp(d, e, t)
        };
        let tangent = |t: f64| {
            let u = 1.0 - t;
            let k = [3.0 * u * u, 6.0 * u * t, 3.0 * t * t];
            [
                k[0] * (c1[0] - p0[0]) + k[1] * (c2[0] - c1[0]) + k[2] * (p3[0] - c2[0]),
                k[0] * (c1[1] - p0[1]) + k[1] * (c2[1] - c1[1]) + k[2] * (p3[1] - c2[1]),
            ]
        };
        // Neighbouring pieces share an endpoint computed once, so the contour
        // stays closed to the bit.
        let mut q0 = p0;
        for i in 0..n {
            let (t0, t1) = (i as f64 / n as f64, (i + 1) as f64 / n as f64);
            let q3 = if i + 1 == n { p3 } else { at(t1) };
            let (d0, d1) = (tangent(t0), tangent(t1));
            let h = (t1 - t0) / 3.0;
            let q1 = [q0[0] + d0[0] * h, q0[1] + d0[1] * h];
            let q2 = [q3[0] - d1[0] * h, q3[1] - d1[1] * h];
            let c = [
                (3.0 * (q1[0] + q2[0]) - q0[0] - q3[0]) / 4.0,
                (3.0 * (q1[1] + q2[1]) - q0[1] - q3[1]) / 4.0,
            ];
            self.push([q0, c, q3]);
            q0 = q3;
        }
        self.at = p3;
    }

    fn close(&mut self) {
        if self.at != self.start {
            let s = self.start;
            self.line(s);
        }
        self.at = self.start;
    }
}

/// A curve's control points on its glyph's lattice, LATTICE steps across the
/// box.
type Curve = [[u16; 2]; 3];

/// A glyph's outline on its lattice.
struct Glyph {
    /// The box, quantized as the shader decodes it.
    min: [f64; 2],
    size: [f64; 2],
    /// p2 lies within OFFSET_MAX steps of p3 on both axes.
    curves: Vec<Curve>,
}

fn glyph(face: &ttf_parser::Face, gid: ttf_parser::GlyphId) -> Result<Option<Glyph>> {
    let mut o = Outline {
        scale: 1.0 / f64::from(face.units_per_em()),
        start: [0.0; 2],
        at: [0.0; 2],
        quads: Vec::new(),
    };
    if face.outline_glyph(gid, &mut o).is_none() || o.quads.is_empty() {
        return Ok(None);
    }
    let mut lo = [f64::MAX; 2];
    let mut hi = [f64::MIN; 2];
    for q in &o.quads {
        for p in q {
            for a in 0..2 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
    }
    let mut min = [0.0; 2];
    let mut size = [0.0; 2];
    for a in 0..2 {
        min[a] = (lo[a] * BOX_SCALE).floor() / BOX_SCALE;
        let qsize = ((hi[a] - min[a]) * BOX_SCALE).ceil().max(1.0);
        size[a] = qsize / BOX_SCALE;
        if qsize > 65535.0 {
            bail!("glyph {} spans more than 8 em", gid.0);
        }
    }
    let snap = |p: [f64; 2]| {
        [0, 1].map(|a| {
            ((p[a] - min[a]) / size[a] * LATTICE)
                .round()
                .clamp(0.0, LATTICE) as u16
        })
    };
    let mut curves: Vec<Curve> = Vec::new();
    let mut todo: Vec<Quad> = o.quads.iter().rev().copied().collect();
    while let Some(q) = todo.pop() {
        let c = q.map(snap);
        let far = (0..2).any(|a| (i32::from(c[1][a]) - i32::from(c[2][a])).abs() > OFFSET_MAX);
        if far {
            // The halves share their middle point, computed once.
            let (l, r) = (lerp(q[0], q[1], 0.5), lerp(q[1], q[2], 0.5));
            let m = lerp(l, r, 0.5);
            todo.push([m, r, q[2]]);
            todo.push([q[0], l, m]);
        } else if c[0] != c[1] || c[0] != c[2] {
            curves.push(c);
        }
    }
    Ok(Some(Glyph { min, size, curves }))
}

/// The lowest and highest lattice step of a curve along `axis`.
fn span(c: &Curve, axis: usize) -> (u16, u16) {
    let v = c.map(|p| p[axis]);
    (v[0].min(v[1]).min(v[2]), v[0].max(v[1]).max(v[2]))
}

/// The curve as the opposite ray along `axis` sees it: mirrored across the box.
fn mirror(c: &Curve, axis: usize) -> Curve {
    c.map(|mut p| {
        p[axis] = LATTICE as u16 - p[axis];
        p
    })
}

/// `n` bands across `axis` (1: horizontal bands, split along y), each listing
/// the curves it crosses. None when a band holds more curves than the shader's
/// loop reads.
fn bands(curves: &[Curve], axis: usize, n: usize, eps: f64) -> Option<Vec<Vec<usize>>> {
    let mut out = vec![Vec::new(); n];
    for (i, c) in curves.iter().enumerate() {
        // A line along the band never crosses the band's ray.
        if c[0][axis] == c[1][axis] && c[1][axis] == c[2][axis] {
            continue;
        }
        let (lo, hi) = span(c, axis);
        let (lo, hi) = (f64::from(lo) / LATTICE, f64::from(hi) / LATTICE);
        for (j, band) in out.iter_mut().enumerate() {
            let (b0, b1) = (j as f64 / n as f64 - eps, (j + 1) as f64 / n as f64 + eps);
            if hi >= b0 && lo <= b1 {
                band.push(i);
            }
        }
    }
    out.iter().all(|b| b.len() <= BAND_CURVES).then_some(out)
}

/// A glyph placed in the block, its origin in em.
struct Placed {
    gid: u16,
    x: f64,
    y: f64,
    /// The character it draws, counted over the words.
    index: usize,
    /// The run of non-whitespace it belongs to, counted from 0. Whitespace
    /// takes the run before it.
    word: usize,
    line: usize,
    /// Its advance from start to end x, in em.
    span: [f64; 2],
}

fn kerning(face: &ttf_parser::Face, l: ttf_parser::GlyphId, r: ttf_parser::GlyphId) -> f64 {
    let Some(kern) = face.tables().kern else {
        return 0.0;
    };
    kern.subtables
        .into_iter()
        .filter(|s| s.horizontal && !s.variable && !s.has_cross_stream)
        .filter_map(|s| s.glyphs_kerning(l, r))
        .map(f64::from)
        .sum()
}

/// The block holds the lines top to bottom. Glyph origins are in em from the
/// block's bottom left, y up like the font's own units, and the shader flips
/// the y down of its callers. Returns the placed glyphs and the block's width
/// and height, in em.
fn layout(face: &ttf_parser::Face, words: &str, align: Align) -> (Vec<Placed>, f64, f64) {
    let upem = f64::from(face.units_per_em());
    let ascent = f64::from(face.ascender()) / upem;
    let line = (f64::from(face.ascender()) - f64::from(face.descender())
        + f64::from(face.line_gap()))
        / upem;
    let lines: Vec<&str> = words.split('\n').collect();
    let height = line * lines.len() as f64;
    let mut placed = Vec::new();
    let mut widths = Vec::new();
    let mut index = 0;
    let (mut word, mut in_word, mut any_word) = (0, false, false);
    for (k, text) in lines.iter().enumerate() {
        let y = height - ascent - line * k as f64;
        let first = placed.len();
        let mut x = 0.0;
        let mut prev: Option<ttf_parser::GlyphId> = None;
        // A draw's glyph_index counts every character of the words, the
        // newline that ends a line too.
        for (i, ch) in text.chars().enumerate() {
            let at = index + i;
            if ch.is_whitespace() {
                in_word = false;
            } else {
                if !in_word && any_word {
                    word += 1;
                }
                (in_word, any_word) = (true, true);
            }
            let ch = if ch == '\t' { ' ' } else { ch };
            if ch.is_control() {
                continue;
            }
            let gid = face.glyph_index(ch).unwrap_or(ttf_parser::GlyphId(0));
            if let Some(p) = prev {
                x += kerning(face, p, gid) / upem;
            }
            let advance = f64::from(face.glyph_hor_advance(gid).unwrap_or(0)) / upem;
            placed.push(Placed {
                gid: gid.0,
                x,
                y,
                index: at,
                word,
                line: k,
                span: [x, x + advance],
            });
            x += advance;
            prev = Some(gid);
        }
        index += text.chars().count() + 1;
        in_word = false;
        widths.push((first, placed.len(), x));
    }
    let width = widths.iter().map(|w| w.2).fold(0.0, f64::max);
    for (from, to, w) in widths {
        let shift = match align {
            Align::Left => 0.0,
            Align::Center => (width - w) / 2.0,
            Align::Right => width - w,
        };
        for p in &mut placed[from..to] {
            p.x += shift;
            p.span = p.span.map(|v| v + shift);
        }
    }
    (placed, width, height)
}

fn u16x2(a: u16, b: u16) -> [u8; 4] {
    [(a >> 8) as u8, a as u8, (b >> 8) as u8, b as u8]
}

/// A 24-bit value in rgb, a byte in alpha.
fn u24u8(v: usize, n: u8) -> [u8; 4] {
    [(v >> 16) as u8, (v >> 8) as u8, v as u8, n]
}

fn fixed(v: f64, offset: f64, scale: f64) -> Option<u16> {
    let q = ((v + offset) * scale).round();
    (0.0..=65535.0).contains(&q).then_some(q as u16)
}

/// A lattice point in 11 bits a coordinate and an offset in 10, as the bytes
/// x >> 3, y >> 3, (x & 7) << 5 | d >> 5, (y & 7) << 5 | d & 31.
fn lattice_texel(p: [u16; 2], d: u16) -> [u8; 4] {
    [
        (p[0] >> 3) as u8,
        (p[1] >> 3) as u8,
        ((p[0] & 7) << 5 | d >> 5) as u8,
        ((p[1] & 7) << 5 | (d & 31)) as u8,
    ]
}

/// A curve in two texels: p1 with the x offset of p2 from p3, then p3 with the
/// y offset. An offset rides biased by OFFSET_MAX + 1.
fn curve_texels(c: &Curve) -> [[u8; 4]; 2] {
    let d = |a: usize| (i32::from(c[1][a]) - i32::from(c[2][a]) + OFFSET_MAX + 1) as u16;
    [lattice_texel(c[0], d(0)), lattice_texel(c[2], d(1))]
}

/// Where a glyph's data starts, and its box: the corner in em from the block's
/// bottom left, and the size.
type Packed = (usize, [f64; 2], [f64; 2]);

/// Lay out `words` in the font `data` and pack glyphs and grid into texels.
pub fn build(data: &[u8], words: &str, align: Align) -> Result<Built> {
    let face = ttf_parser::Face::parse(data, 0).map_err(|e| anyhow!("unreadable font: {e}"))?;
    let (placed, width, height) = layout(&face, words, align);
    // None for a glyph with no outline.
    let mut glyphs: HashMap<u16, Option<Glyph>> = HashMap::new();
    for p in &placed {
        if let Entry::Vacant(e) = glyphs.entry(p.gid) {
            e.insert(glyph(&face, ttf_parser::GlyphId(p.gid))?);
        }
    }
    let mut need = None;
    let mut nb = BANDS;
    while nb >= 1 {
        match pack(&placed, &glyphs, nb, height) {
            Ok(Ok(mut built)) => {
                built.uniforms.extend([("_w", width), ("_h", height)]);
                return Ok(built);
            }
            Ok(Err(tex_h)) => need = Some(tex_h),
            // A band past BAND_CURVES at fewer bands leaves the texture's
            // size as the error.
            Err(e) if need.is_none() => return Err(e),
            Err(_) => break,
        }
        nb /= 2;
    }
    bail!(
        "the words need a {TEX_W}x{} texture, past the {TEX_W}x{MAX_TEX_H} a VC4 holds",
        need.unwrap_or(0)
    )
}

/// The texture rows that hold `texels`.
fn tex_rows(texels: usize) -> usize {
    texels
        .div_ceil(TEX_W)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX)
}

/// Pack the glyphs, `nb` bands to an axis, and the grid into texels, and lay
/// the quads over a block `height` em tall. Err holds the texture height the
/// words need when it passes MAX_TEX_H.
fn pack(
    placed: &[Placed],
    glyphs: &HashMap<u16, Option<Glyph>>,
    nb: usize,
    height: f64,
) -> Result<std::result::Result<Built, usize>> {
    // Texel 0 stays empty, so a slot that names a glyph is never all zero.
    let mut tex: Vec<[u8; 4]> = vec![[0; 4]];
    // Where each glyph's data starts. None for a glyph with no outline.
    let mut starts: HashMap<u16, Option<usize>> = HashMap::new();
    for p in placed {
        if starts.contains_key(&p.gid) {
            continue;
        }
        let Some(g) = &glyphs[&p.gid] else {
            starts.insert(p.gid, None);
            continue;
        };
        let too_many = || {
            anyhow!(
                "glyph {} holds too many curves for {BAND_CURVES} a band",
                p.gid
            )
        };
        let hbands = bands(&g.curves, 1, nb, BAND_EPS / g.size[1]).ok_or_else(too_many)?;
        let vbands = bands(&g.curves, 0, nb, BAND_EPS / g.size[0]).ok_or_else(too_many)?;
        let at = tex.len();
        starts.insert(p.gid, Some(at));
        // The box size opens the glyph's data, then the band entries.
        tex.resize(at + 1 + 2 * nb, [0; 4]);
        let sz = |a: usize| fixed(g.size[a], 0.0, BOX_SCALE).unwrap_or(0);
        tex[at] = u16x2(sz(0), sz(1));
        // A band that lists the same curves as an earlier one reads its lists.
        let mut lists: HashMap<(usize, &[usize]), usize> = HashMap::new();
        let all = hbands
            .iter()
            .map(|b| (0, b))
            .chain(vbands.iter().map(|b| (1, b)));
        for (j, (along, band)) in all.enumerate() {
            let loc = *lists.entry((along, band.as_slice())).or_insert_with(|| {
                // Both lists of a band sit in one texture row, so the shader
                // steps from curve to curve along x.
                if tex.len() % TEX_W + band.len() * 4 > TEX_W {
                    tex.resize(tex.len().next_multiple_of(TEX_W), [0; 4]);
                }
                let loc = tex.len();
                // The ray toward + along reads the first list, the ray toward -
                // the second, mirrored. Each sorts by the far end of its curves,
                // largest first.
                for flip in [false, true] {
                    let mut cs: Vec<Curve> = band
                        .iter()
                        .map(|&i| match flip {
                            false => g.curves[i],
                            true => mirror(&g.curves[i], along),
                        })
                        .collect();
                    cs.sort_by_key(|c| std::cmp::Reverse(span(c, along).1));
                    for c in &cs {
                        tex.extend(curve_texels(c));
                    }
                }
                loc
            });
            tex[at + 1 + j] = u24u8(loc, band.len() as u8);
        }
    }

    // A grid over the ink of every glyph. Each cell lists the glyphs whose box,
    // grown by INK_MARGIN, reaches into it.
    // Origins and the grid corner sit on the 1/ORIGIN_SCALE em lattice and box
    // corners on the 1/BOX_SCALE em one, so a glyph that spans cells decodes to
    // the same box in each of them.
    let snap = |v: f64| (v * ORIGIN_SCALE).round() / ORIGIN_SCALE;
    let inked: Vec<(&Placed, Packed)> = placed
        .iter()
        .filter_map(|p| {
            let at = starts[&p.gid]?;
            let g = glyphs[&p.gid].as_ref()?;
            Some((
                p,
                (at, [snap(p.x) + g.min[0], snap(p.y) + g.min[1]], g.size),
            ))
        })
        .collect();
    let boxes: Vec<(Packed, [f64; 4])> = inked
        .iter()
        .map(|&(_, (at, lo, size))| {
            (
                (at, lo, size),
                [
                    lo[0] - INK_MARGIN,
                    lo[1] - INK_MARGIN,
                    lo[0] + size[0] + INK_MARGIN,
                    lo[1] + size[1] + INK_MARGIN,
                ],
            )
        })
        .collect();
    let (mut gx, mut gy, mut gx1, mut gy1) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (_, b) in &boxes {
        gx = gx.min(b[0]);
        gy = gy.min(b[1]);
        gx1 = gx1.max(b[2]);
        gy1 = gy1.max(b[3]);
    }
    let (gx, gy) = (
        (gx * ORIGIN_SCALE).floor() / ORIGIN_SCALE,
        (gy * ORIGIN_SCALE).floor() / ORIGIN_SCALE,
    );
    let mut cell = 0.5;
    let (cols, rows, grid) = loop {
        let cols = ((gx1 - gx) / cell).ceil().max(0.0) as usize;
        let rows = ((gy1 - gy) / cell).ceil().max(0.0) as usize;
        // A grid grows with the block's area, so words past the texture fail
        // before their grid takes memory.
        let need = tex_rows(
            tex.len().next_multiple_of(CELL_K * SLOT)
                + cols.saturating_mul(rows).saturating_mul(CELL_K * SLOT),
        );
        if need > MAX_TEX_H {
            return Ok(Err(need));
        }
        let mut grid: Vec<Vec<Packed>> = vec![Vec::new(); cols * rows];
        for (g, b) in &boxes {
            let c0 = (((b[0] - gx) / cell).floor().max(0.0) as usize).min(cols.saturating_sub(1));
            let c1 = (((b[2] - gx) / cell).floor().max(0.0) as usize).min(cols.saturating_sub(1));
            let r0 = (((b[1] - gy) / cell).floor().max(0.0) as usize).min(rows.saturating_sub(1));
            let r1 = (((b[3] - gy) / cell).floor().max(0.0) as usize).min(rows.saturating_sub(1));
            for r in r0..=r1 {
                for c in c0..=c1 {
                    grid[r * cols + c].push(*g);
                }
            }
        }
        let fullest = grid.iter().map(Vec::len).max().unwrap_or(0);
        if fullest <= CELL_K {
            break (cols, rows, grid);
        }
        if cell <= 1.0 / 16.0 {
            bail!("more than {CELL_K} glyphs overlap at one spot");
        }
        cell /= 2.0;
    };
    tex.resize(tex.len().next_multiple_of(CELL_K * SLOT), [0; 4]);
    let cells = tex.len();
    tex.resize(cells + cols * rows * CELL_K * SLOT, [0; 4]);
    for r in 0..rows {
        for c in 0..cols {
            let corner = [gx + c as f64 * cell, gy + r as f64 * cell];
            let at = cells + (r * cols + c) * CELL_K * SLOT;
            let list = &grid[r * cols + c];
            for (k, (h, lo, _)) in list.iter().enumerate() {
                let rel = |a: usize| fixed(lo[a] - corner[a], CORNER_OFFSET, BOX_SCALE);
                let (Some(bx), Some(by)) = (rel(0), rel(1)) else {
                    bail!("a glyph box sits past {CORNER_OFFSET} em from its grid cell");
                };
                // The top bit of a slot's location marks the cell's last glyph.
                let last = if k + 1 == list.len() { 1 << 23 } else { 0 };
                tex[at + k * SLOT] = u24u8(h | last, 0);
                tex[at + k * SLOT + 1] = u16x2(bx, by);
            }
        }
    }

    let tex_h = tex_rows(tex.len());
    let mut texels = vec![0u8; TEX_W * tex_h * 4];
    for (i, t) in tex.iter().enumerate() {
        texels[i * 4..i * 4 + 4].copy_from_slice(t);
    }
    let mut floats = Vec::with_capacity(inked.len() * 4 * QUAD_FLOATS);
    for (p, (at, lo, size)) in &inked {
        // The box flips to y down, the way a draw's caller counts.
        let a1 = [
            lo[0],
            height - lo[1] - size[1],
            lo[0] + size[0],
            height - lo[1],
        ];
        let a2 = [(at + 1) as f64, nb as f64, nb as f64, 0.0];
        let a3 = [p.line as f64, p.span[0], p.span[1], 0.0];
        for corner in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] {
            let a0 = [corner[0], corner[1], p.index as f64, p.word as f64];
            floats.extend([a0, a1, a2, a3].iter().flatten().map(|v| *v as f32));
        }
    }
    Ok(Ok(Built {
        width: TEX_W as u32,
        height: tex_h as u32,
        quads: Quads {
            floats,
            per_vertex: QUAD_FLOATS,
            count: inked.len(),
        },
        texels,
        uniforms: vec![
            ("_tw", TEX_W as f64),
            ("_itw", 1.0 / TEX_W as f64),
            ("_ith", 1.0 / tex_h as f64),
            ("_gx", gx),
            ("_gy", gy),
            ("_cols", cols as f64),
            ("_rows", rows as f64),
            ("_cell", cell),
            ("_cells", cells as f64),
            ("_nb", nb as f64),
        ],
    }))
}

/// The preamble lines of one text: its uniforms, its sampler, and the readers.
///
/// The coverage math ports SlugPixelShader.hlsl, Copyright 2017 Eric Lengyel,
/// MIT License (https://github.com/EricLengyel/Slug). GLSL ES 1.00 has no
/// integer textures, no bit operations and no derivatives, so the port reads
/// bytes, takes the root eligibility from comparisons, and takes the pixels
/// per em from its caller. The roots come from the stable form of the
/// quadratic formula, which also covers a curve that is a line.
pub fn reader_lines(name: &str) -> Vec<String> {
    lines(GLSL, name)
}

/// The lines of a draw's vertex shader ahead of the author's
/// `vec2 glyph_place(vec2 em)`: the quad attributes, the text's size, the
/// varyings, and the glyph values glyph_place reads. The caller's preamble
/// declares `resolution` and `vflip`, with the precision the fragment stage
/// gives them.
pub fn draw_vertex_head(name: &str) -> Vec<String> {
    lines(DRAW_VERTEX_HEAD, name)
}

/// The vertex main of an outline draw, after glyph_place. Each box edge moves
/// out 5/8 of a screen pixel, or `pad` em where that reaches further.
pub fn outline_vertex_main(pad: f64) -> Vec<String> {
    lines(OUTLINE_VERTEX_MAIN, "")
        .into_iter()
        .map(|l| l.replace("PAD", &format!("{pad:?}")))
        .collect()
}

/// The lines of an outline draw's fragment shader between the text's reader
/// lines and the author's code.
pub fn outline_fragment_head(name: &str) -> Vec<String> {
    lines(OUTLINE_FRAGMENT_HEAD, name)
}

/// The lines of an outline draw's fragment shader after the author's code.
pub fn outline_fragment_tail(name: &str) -> Vec<String> {
    lines(OUTLINE_FRAGMENT_TAIL, name)
}

fn lines(glsl: &str, name: &str) -> Vec<String> {
    glsl.replace("NAME", name)
        .lines()
        .map(str::to_string)
        .collect()
}

const DRAW_VERTEX_HEAD: &str = r#"attribute vec4 a0;
attribute vec4 a1;
attribute vec4 a2;
attribute vec4 a3;
uniform float NAME_w;
uniform float NAME_h;
#define NAME_size vec2(NAME_w, NAME_h)
varying vec2 v_texCoord;
varying vec4 glyph_v0;
varying vec4 glyph_v1;
varying vec4 glyph_v2;
varying vec4 glyph_v3;
varying vec4 glyph_v4;
float glyph_index;
float glyph_word;
float glyph_line;
vec2 glyph_span;
vec4 glyph_box;"#;

const OUTLINE_VERTEX_MAIN: &str = r#"void main() {
    glyph_index = a0.z;
    glyph_word = a0.w;
    glyph_line = a3.x;
    glyph_span = a3.yz;
    glyph_box = a1;
    vec2 em = mix(a1.xy, a1.zw, a0.xy);
    vec2 p = glyph_place(em);
    // The pixels glyph_place moves per em along each em axis, exact for an
    // affine glyph_place.
    vec2 jx = (glyph_place(em + vec2(0.0625, 0.0)) - p) * 16.0;
    vec2 jy = (glyph_place(em + vec2(0.0, 0.0625)) - p) * 16.0;
    float det = abs(jx.x * jy.y - jx.y * jy.x);
    // Half a pixel of antialiasing, and 1/8 px more, since VC4 lands a vertex
    // up to 1/16 px off where this shader puts it.
    vec2 grow = max(0.625 * vec2(length(jy), length(jx)) / det, vec2(PAD));
    vec2 e = em + (a0.xy * 2.0 - 1.0) * grow;
    vec2 px = glyph_place(e);
    // The em a screen pixel spans along each em axis, as fwidth would.
    vec2 fw = vec2(abs(jy.x) + abs(jy.y), abs(jx.x) + abs(jx.y)) / det;
    vec2 sz = a1.zw - a1.xy;
    // The sample on the glyph's lattice, y up, and the pixels a lattice step
    // spans.
    glyph_v0 = vec4(vec2(e.x - a1.x, a1.w - e.y) / sz * 2047.0, sz / (fw * 2047.0));
    // glyph_ppem: the pixels per em of a placement that scales both axes
    // alike.
    glyph_v1 = vec4(a2.xyz, sqrt(det));
    glyph_v2 = vec4(e, a0.zw);
    glyph_v3 = vec4(a3.xyz, 0.0);
    // The lattice steps a pixel spans along x and along y of the pass.
    vec2 k = 2047.0 / sz;
    glyph_v4 = vec4(k.x * jy.y, -k.x * jy.x, k.y * jx.y, -k.y * jx.x)
             / (jx.x * jy.y - jx.y * jy.x);
    v_texCoord = px / resolution;
    gl_Position = vec4((v_texCoord * 2.0 - 1.0) * vec2(1.0, vflip), 0.0, 1.0);
}"#;

const OUTLINE_FRAGMENT_HEAD: &str = r#"varying vec4 glyph_v0;
varying vec4 glyph_v1;
varying vec4 glyph_v2;
varying vec4 glyph_v3;
varying vec4 glyph_v4;
// A varying that holds one value at every vertex may still interpolate a hair
// off it, so an index rounds.
#define glyph_em glyph_v2.xy
#define glyph_index float(int(glyph_v2.z + 0.5))
#define glyph_word float(int(glyph_v2.w + 0.5))
#define glyph_line float(int(glyph_v3.x + 0.5))
#define glyph_span glyph_v3.yz
#define glyph_ppem glyph_v1.w
#define glyph_dist glyph_dist_needs_fill_mtsdf
#define glyph_soft glyph_soft_needs_fill_mtsdf
float glyph_c = 0.0;
float glyph_cover() { return glyph_c; }
// The fill reads its texture ahead of the author's main: on VC4 a value
// computed ahead of a texture read and used after it comes back wrong.
#define main glyph_main"#;

const OUTLINE_FRAGMENT_TAIL: &str = r#"#undef main
void main() {
    // VC4 lands a vertex up to 1/16 px off where the vertex shader put it, and
    // the varyings follow the vertex, so the sample moves to the pixel centre
    // as v_texCoord tells it.
    vec2 px = v_texCoord * resolution;
    vec2 d = floor(px) + 0.5 - px;
    vec2 u = glyph_v0.xy + vec2(dot(glyph_v4.xy, d), dot(glyph_v4.zw, d));
    glyph_c = NAME_outline(float(int(glyph_v1.x + 0.5)), vec2(ivec2(glyph_v1.yz + 0.5)),
                           u, glyph_v0.zw);
    glyph_main();
}"#;

const GLSL: &str = r#"uniform sampler2D NAME_tex;
uniform float NAME_tw;
uniform float NAME_itw;
uniform float NAME_ith;
uniform float NAME_w;
uniform float NAME_h;
uniform float NAME_gx;
uniform float NAME_gy;
uniform float NAME_cols;
uniform float NAME_rows;
uniform float NAME_cell;
uniform float NAME_cells;
uniform float NAME_nb;
#define NAME_size vec2(NAME_w, NAME_h)
// The Pi 5 reads a texel back too coarse to take bits from the raw values, so
// every byte goes through int() first: a read of byte b lies near b / 255, and
// int(t * 255.5) floors to b. On VC4, float(int(x)) takes 2 instructions and
// floor(x) 4, and every floor here takes x >= 0.
vec4 NAME_texel(float i) {
    float row = float(int(i * NAME_itw));
    vec2 at = vec2(i - row * NAME_tw + 0.5, row + 0.5);
    return vec4(ivec4(texture2D(NAME_tex, at * vec2(NAME_itw, NAME_ith)) * 255.5));
}
// The texture coordinate of a band's list: the first list for the ray toward
// +, the second for the ray toward -.
vec2 NAME_list(vec4 band, float dir) {
    float i = (band.r * 256.0 + band.g) * 256.0 + band.b + (dir < 0.0 ? band.a * 2.0 : 0.0);
    float row = float(int(i * NAME_itw));
    return vec2(i - row * NAME_tw + 0.5, row + 0.5) * vec2(NAME_itw, NAME_ith);
}
// One texel of a curve: a lattice point, and the offset of p2 from p3 on one
// axis.
vec3 NAME_lattice(vec4 t) {
    vec4 b = vec4(ivec4(t * 255.5));
    vec2 hi = vec2(ivec2(b.ba * (1.0 / 32.0)));
    return vec3(b.rg * 8.0 + hi, (b.b - hi.x * 32.0) * 32.0 + b.a - hi.y * 32.0 - 512.0);
}
// Where the curve crosses y = 0, as x. The roots of a t^2 - 2b t + c, each
// from the form that cancels no digits.
vec2 NAME_cross(vec2 p1, vec2 p2, vec2 p3) {
    vec2 a = p1 - p2 * 2.0 + p3;
    vec2 b = p1 - p2;
    float disc = b.y * b.y - a.y * p1.y;
    float d = sqrt(max(disc, 0.0));
    float q = b.y >= 0.0 ? b.y + d : b.y - d;
    float ra = 1.0 / a.y;
    float cq = q != 0.0 ? p1.y / q : 0.0;
    float qa = q * ra;
    float t1 = b.y >= 0.0 ? cq : qa;
    float t2 = b.y >= 0.0 ? qa : cq;
    // A curve that misses the ray has both roots at its extreme, where they
    // cancel.
    float tm = b.y * ra;
    t1 = disc > 0.0 ? t1 : tm;
    t2 = disc > 0.0 ? t2 : tm;
    return vec2((a.x * t1 - b.x * 2.0) * t1 + p1.x, (a.x * t2 - b.x * 2.0) * t2 + p1.x);
}
// The coverage of a glyph at u on its lattice, y up, drawn at ppu pixels a
// lattice step. Its band entries start at texel h: nb.x horizontal bands, then
// nb.y vertical ones.
float NAME_outline(float h, vec2 nb, vec2 u, vec2 ppu) {
    vec2 m = 0.5 / ppu;
    float cov = 0.0;
    if (u.x > -m.x && u.y > -m.y && u.x < 2047.0 + m.x && u.y < 2047.0 + m.y) {
        vec2 band = min(vec2(ivec2(max(u.yx, 0.0) * (nb / 2047.0))), nb - 1.0);
        vec4 hb = NAME_texel(h + band.x);
        // Each ray runs toward the nearer edge of the box. A ray toward -
        // reads its band mirrored, so its coverage flips sign.
        float dx = u.x < 1023.5 ? -1.0 : 1.0;
        vec2 uh = vec2(dx > 0.0 ? u.x : 2047.0 - u.x, u.y);
        float xcov = 0.0;
        float xwgt = 0.0;
        vec2 at = NAME_list(hb, dx);
        for (int i = 0; i < 64; i++) {
            if (float(i) >= hb.a) break;
            vec3 c1 = NAME_lattice(texture2D(NAME_tex, at));
            vec3 c3 = NAME_lattice(texture2D(NAME_tex, at + vec2(NAME_itw, 0.0)));
            at.x += NAME_itw * 2.0;
            vec2 p1 = c1.xy - uh;
            vec2 p3 = c3.xy - uh;
            vec2 p2 = p3 + vec2(c1.z, c3.z);
            // Sorted by their far end: a curve behind the pixel crosses
            // nothing the pixel sees, and neither does any after it.
            if (max(max(p1.x, p2.x), p3.x) * ppu.x < -0.5) break;
            // Which roots count, from the signs of the control points: the
            // table 0x2E74.
            bool n1 = p1.y < 0.0;
            bool n2 = p2.y < 0.0;
            bool n3 = p3.y < 0.0;
            bool e1 = n3 ? !(n1 && n2) : n2 && !n1;
            bool e2 = n3 ? n1 && !n2 : n1 || n2;
            if (e1 || e2) {
                vec2 r = NAME_cross(p1, p2, p3) * ppu.x;
                xcov += e1 ? clamp(r.x + 0.5, 0.0, 1.0) : 0.0;
                xwgt = max(xwgt, e1 ? clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0) : 0.0);
                xcov -= e2 ? clamp(r.y + 0.5, 0.0, 1.0) : 0.0;
                xwgt = max(xwgt, e2 ? clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0) : 0.0);
            }
        }
        xcov *= dx;
        vec4 vb = NAME_texel(h + nb.x + band.y);
        float dy = u.y < 1023.5 ? -1.0 : 1.0;
        vec2 uv = vec2(u.x, dy > 0.0 ? u.y : 2047.0 - u.y);
        float ycov = 0.0;
        float ywgt = 0.0;
        at = NAME_list(vb, dy);
        for (int i = 0; i < 64; i++) {
            if (float(i) >= vb.a) break;
            vec3 c1 = NAME_lattice(texture2D(NAME_tex, at));
            vec3 c3 = NAME_lattice(texture2D(NAME_tex, at + vec2(NAME_itw, 0.0)));
            at.x += NAME_itw * 2.0;
            vec2 p1 = (c1.xy - uv).yx;
            vec2 p3 = (c3.xy - uv).yx;
            vec2 p2 = p3 + vec2(c3.z, c1.z);
            if (max(max(p1.x, p2.x), p3.x) * ppu.y < -0.5) break;
            // Which roots count, from the signs of the control points: the
            // table 0x2E74.
            bool n1 = p1.y < 0.0;
            bool n2 = p2.y < 0.0;
            bool n3 = p3.y < 0.0;
            bool e1 = n3 ? !(n1 && n2) : n2 && !n1;
            bool e2 = n3 ? n1 && !n2 : n1 || n2;
            if (e1 || e2) {
                vec2 r = NAME_cross(p1, p2, p3) * ppu.y;
                ycov -= e1 ? clamp(r.x + 0.5, 0.0, 1.0) : 0.0;
                ywgt = max(ywgt, e1 ? clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0) : 0.0);
                ycov += e2 ? clamp(r.y + 0.5, 0.0, 1.0) : 0.0;
                ywgt = max(ywgt, e2 ? clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0) : 0.0);
            }
        }
        ycov *= dy;
        cov = clamp(max(abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0),
                        min(abs(xcov), abs(ycov))), 0.0, 1.0);
    }
    return cov;
}
// The coverage of the glyph whose data starts at texel h, at q in em from its
// box's corner. Its data holds its box size, then its band entries.
float NAME_glyph(float h, vec2 q, vec2 sz, float ppem) {
    return NAME_outline(h + 1.0, vec2(NAME_nb), q / sz * 2047.0, sz * (ppem / 2047.0));
}
// The coverage of the words at at, in em from the top left of the block, y
// down like v_texCoord, drawn at ppem pixels per em.
float NAME_cover(vec2 at, float ppem) {
    float cov = 0.0;
    vec2 p = vec2(at.x, NAME_h - at.y);
    vec2 g = (p - vec2(NAME_gx, NAME_gy)) / NAME_cell;
    if (g.x >= 0.0 && g.y >= 0.0 && g.x < NAME_cols && g.y < NAME_rows) {
        vec2 c = vec2(ivec2(g));
        float cell = NAME_cells + (c.y * NAME_cols + c.x) * 8.0;
        float row = float(int(cell * NAME_itw));
        vec2 slot = vec2(cell - row * NAME_tw + 0.5, row + 0.5) * vec2(NAME_itw, NAME_ith);
        vec2 pc = p - vec2(NAME_gx, NAME_gy) - c * NAME_cell;
        // An empty slot, or the one marked last, ends the cell.
        for (int k = 0; k < 64; k++) {
            vec4 a = vec4(ivec4(texture2D(NAME_tex, slot) * 255.5));
            if (a.r + a.g + a.b < 0.5) break;
            vec4 b = vec4(ivec4(texture2D(NAME_tex, slot + vec2(NAME_itw, 0.0)) * 255.5));
            slot.x += NAME_itw * 2.0;
            float last = a.r >= 128.0 ? 1.0 : 0.0;
            float h = ((a.r - last * 128.0) * 256.0 + a.g) * 256.0 + a.b;
            vec2 lo = (b.rb * 256.0 + b.ga) * (1.0 / 8192.0) - 7.0;
            vec4 s = NAME_texel(h);
            vec2 sz = (s.rb * 256.0 + s.ga) * (1.0 / 8192.0);
            cov = max(cov, NAME_glyph(h, pc - lo, sz, ppem));
            if (last > 0.5) break;
        }
    }
    return cov;
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn laid(words: &str, align: Align) -> (Vec<Placed>, f64) {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tools/fixtures/fonts/DejaVuSans-ASCII.ttf"
        );
        let data = std::fs::read(path).unwrap();
        let face = ttf_parser::Face::parse(&data, 0).unwrap();
        let (placed, width, _) = layout(&face, words, align);
        (placed, width)
    }

    fn fields(p: &[Placed]) -> Vec<(usize, usize, usize)> {
        p.iter().map(|p| (p.index, p.word, p.line)).collect()
    }

    #[test]
    fn every_character_counts_and_whitespace_keeps_the_word_before_it() {
        let (p, _) = laid("ab c\nd", Align::Left);
        assert_eq!(
            fields(&p),
            [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 1, 0), (5, 2, 1)]
        );
    }

    #[test]
    fn leading_whitespace_is_word_0_and_a_control_character_counts_unplaced() {
        let (p, _) = laid(" \tab\u{7}  c\n\nd", Align::Left);
        assert_eq!(
            fields(&p),
            [
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (3, 0, 0),
                (5, 0, 0),
                (6, 0, 0),
                (7, 1, 0),
                (10, 2, 2)
            ]
        );
    }

    #[test]
    fn a_span_runs_from_the_origin_one_advance_and_kerns_into_the_next() {
        let (p, _) = laid("AVA", Align::Left);
        for g in &p {
            assert_eq!(g.span[0], g.x);
            assert!(g.span[1] > g.span[0]);
        }
        // The font kerns A and V together.
        assert!(p[1].span[0] < p[0].span[1]);
        assert!(p[2].span[0] < p[1].span[1]);
    }

    #[test]
    fn a_span_shifts_with_its_line() {
        let (left, _) = laid("AV To\nghee", Align::Left);
        let (centre, width) = laid("AV To\nghee", Align::Center);
        let (right, _) = laid("AV To\nghee", Align::Right);
        let end = left[8].span[1];
        for (l, (c, r)) in left.iter().zip(centre.iter().zip(&right)) {
            let shift = if l.line == 1 { width - end } else { 0.0 };
            assert!((c.span[0] - l.span[0] - shift / 2.0).abs() < 1e-12);
            assert!((r.span[1] - l.span[1] - shift).abs() < 1e-12);
            assert_eq!((c.x, r.x), (c.span[0], r.span[0]));
        }
    }
}
