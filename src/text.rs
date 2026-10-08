//! Text drawn on the GPU from the outlines of a font.
//!
//! A text layer, `crate::layer`, holds words and the fonts to draw them
//! with, in order: each character draws from the first font whose cmap holds
//! it. This module lays the words out into a block and packs what the GPU
//! reads. Every pass of the scene except a filter pass reads the coverage of
//! a layer's words at any point with `<name>_cover(p, font_size)`.
//!
//! A layer draws its words as one quad over each inked glyph. Under the MTSDF
//! render, a quad takes one bilinear sample of the glyph's field in an atlas,
//! which `crate::atlas` builds, and under the slug render, it computes the
//! coverage from the curves as `<name>_cover` does. A colour glyph, a bitmap
//! with no outline, draws through the layer's colour batch, which
//! `crate::colour` describes.
//!
//! Coverage follows the Slug algorithm by Eric Lengyel
//! (https://jcgt.org/published/0006/02/02/, https://github.com/EricLengyel/Slug).
//! A glyph is a set of quadratic Bézier curves. A pixel casts a ray along x
//! and a ray along y, and the curves that a ray crosses count how often the
//! outline winds around the pixel. A crossing within half a pixel of the
//! ray's start counts in part, which antialiases the edge at any scale.
//!
//! The curves of a layer's words lie in one RGBA8 texture of the layer's own,
//! read with NEAREST, so a GL without float or integer textures, the VC4,
//! holds them. A 16-bit value spans two bytes of a texel, a 24-bit texel
//! index three, and a curve two texels. pack() writes three tables into it,
//! and `<name>_cover(p, font_size)` walks them to find the curves near `p`:
//!
//! 1. The grid. Square cells cover the ink of the words, 0.5 em a side,
//!    halved until no cell lists more than CELL_K glyphs. The cell under `p`
//!    lists each glyph whose box, grown by INK_MARGIN, reaches into it: the
//!    texel where the glyph's data starts, and the corner of its box.
//! 2. The bands. A glyph's data holds its box size, then its bands: the box
//!    cut into as many as BANDS horizontal strips and as many vertical ones,
//!    each listing the curves that cross it. A glyph that the words use
//!    twice stores its data once. The ray along x reads the horizontal band
//!    that holds `p`, and the ray along y the vertical one.
//! 3. The curves. A band holds two lists of its curves, the second mirrored,
//!    one for a ray that runs up its axis and one for a ray that runs down
//!    it, and a ray runs toward the nearer edge of the box. A list sorts its
//!    curves by their far end, largest first, so the loop over it stops at
//!    the first curve wholly behind `p`.
//!
//! The coverage at `p` is the largest among the glyphs of its cell. A slug
//! layer skips the grid: each of its quads carries the texel where its
//! glyph's data starts. So the slug layers whose `<name>_cover` no pass
//! reads share one texture, a Glyphs, which takes each glyph's data at the
//! first words that draw it and keeps it there, and new words upload only
//! the glyphs new to it. Words that a texture of their own would not hold
//! at BANDS bands, and glyphs past the room of the shared one, take a
//! texture of their own.
//!
//! A text problem leaves the scene running. A character whose glyph has no
//! outline and no colour bitmap, or one past a limit, draws an error box:
//! hazard stripes, laid out as an outline, so `<name>_cover` reads them as
//! coverage, and a layer's error-box batch paints them yellow and black. A
//! font that failed to load drops out of the layer's fonts, and when none is
//! left, every character draws an error box. Words that fit no texture leave
//! the last words that did.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};

use anyhow::{anyhow, bail, Result};
use sha2::{Digest, Sha256};
use ttf_parser::gpos::{PairAdjustment, PositioningSubtable};
use ttf_parser::opentype_layout::Lookup;
use ttf_parser::{Face, GlyphId, Tag};

use crate::atlas::{self, Cell};
use crate::colour;

/// Texture width in texels. A power of two keeps every texel centre an exact
/// binary fraction, so a NEAREST read lands on the texel named by the index.
const TEX_W: usize = 1024;
/// VC4's GL_MAX_TEXTURE_SIZE.
const MAX_TEX_H: usize = 2048;
/// The most glyphs listed by one grid cell. The shader loops over them.
const CELL_K: usize = 4;
/// The texels of one glyph's slot in a grid cell. CELL_K * SLOT divides TEX_W,
/// so a cell sits in one texture row.
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
pub(crate) const LATTICE: f64 = 2047.0;
/// A curve stores p2 as its offset from p3, in 10 bits. A curve whose p2 lies
/// more than this many lattice steps from p3 on an axis splits in two.
const OFFSET_MAX: i32 = 511;
/// A glyph box size is 16 bits at 1/8192 em, and so is its corner, offset by
/// 7 em, from the corner of a grid cell that lists the glyph.
const BOX_SCALE: f64 = 8192.0;
const CORNER_OFFSET: f64 = 7.0;
/// Glyph origins and the grid corner sit on a 1/4096 em lattice.
const ORIGIN_SCALE: f64 = 4096.0;
/// The top bit of a slot's 24-bit location marks the cell's last glyph.
const LAST_GLYPH: usize = 1 << 23;
/// An error box in em, sized like a sans font's capitals. It stands on the
/// baseline, centred in its character's advance. A font that failed to load
/// takes ERROR_ADVANCE a character, its first baseline ERROR_ASCENT below the
/// top of the block, and ERROR_LINE a line.
pub const ERROR_BOX: [f64; 2] = [0.625, 0.75];
const ERROR_ADVANCE: f64 = 0.75;
const ERROR_ASCENT: f64 = 1.0;
const ERROR_LINE: f64 = 1.25;
/// The width along x of one hazard stripe of an error box, in em. The
/// stripes run at 45 degrees, up to the right.
pub const STRIPE: f64 = 0.125;
/// Unicode's Default_Ignorable_Code_Point ranges: format characters, joiners,
/// variation selectors and fillers.
const IGNORABLE: [(char, char); 17] = [
    ('\u{ad}', '\u{ad}'),
    ('\u{34f}', '\u{34f}'),
    ('\u{61c}', '\u{61c}'),
    ('\u{115f}', '\u{1160}'),
    ('\u{17b4}', '\u{17b5}'),
    ('\u{180b}', '\u{180f}'),
    ('\u{200b}', '\u{200f}'),
    ('\u{202a}', '\u{202e}'),
    ('\u{2060}', '\u{206f}'),
    ('\u{3164}', '\u{3164}'),
    ('\u{fe00}', '\u{fe0f}'),
    ('\u{feff}', '\u{feff}'),
    ('\u{ffa0}', '\u{ffa0}'),
    ('\u{fff0}', '\u{fff8}'),
    ('\u{1bca0}', '\u{1bca3}'),
    ('\u{1d173}', '\u{1d17a}'),
    ('\u{e0000}', '\u{e0fff}'),
];

fn ignorable(ch: char) -> bool {
    IGNORABLE.iter().any(|(lo, hi)| (*lo..=*hi).contains(&ch))
}

/// Whitespace, control characters and default ignorables draw no ink, so
/// their glyphs have no outline to miss.
fn needs_ink(ch: char) -> bool {
    !(ch.is_whitespace() || ch.is_control() || ignorable(ch))
}

/// A font file's bytes, or the reason they failed to load.
pub type Font = std::result::Result<Arc<Vec<u8>>, String>;

/// DejaVu Sans cut to Latin: the font of a text that names none.
pub const BUILT_IN_FONT: &[u8] = include_bytes!("../ghee/examples/fonts/DejaVuSans-Latin.ttf");

/// The built-in font's bytes, one copy for every text that takes it.
pub fn built_in_font() -> Font {
    static BYTES: std::sync::OnceLock<Arc<Vec<u8>>> = std::sync::OnceLock::new();
    Ok(BYTES
        .get_or_init(|| Arc::new(BUILT_IN_FONT.to_vec()))
        .clone())
}

/// The font_family that names the built-in pixel font, PIXEL_FONT.
pub const PIXEL: &str = "pixel";

/// Pixel Operator 8, drawn on a grid of PIXEL_EM pixels to the em, so at
/// `font_size` 8 every edge falls on a pixel edge.
pub const PIXEL_FONT: &[u8] = include_bytes!("../ghee/examples/fonts/PixelOperator8.ttf");

/// The pixels per em of PIXEL_FONT's grid.
pub const PIXEL_EM: f64 = 8.0;

/// The font of a text that names `family`, when ghee carries it: the
/// built-in font for none, and the pixel font for PIXEL.
pub fn built_in(family: Option<&str>) -> Option<Font> {
    static PIXEL_BYTES: std::sync::OnceLock<Arc<Vec<u8>>> = std::sync::OnceLock::new();
    match family {
        None => Some(built_in_font()),
        Some(PIXEL) => Some(Ok(PIXEL_BYTES
            .get_or_init(|| Arc::new(PIXEL_FONT.to_vec()))
            .clone())),
        Some(_) => None,
    }
}

/// The bytes of a font file, or why ttf-parser reads no font in them.
pub fn font(bytes: &[u8]) -> Font {
    match Face::parse(bytes, 0) {
        Ok(_) => Ok(Arc::new(bytes.to_vec())),
        Err(e) => Err(e.to_string()),
    }
}

/// True for the name of a .ttf or .otf file. A text's font that names no such
/// file names a font asset.
pub fn is_font_file(name: &str) -> bool {
    let f = name.to_ascii_lowercase();
    f.ends_with(".ttf") || f.ends_with(".otf")
}

/// A problem met by a layout. The text still draws.
#[derive(Clone, Debug, PartialEq)]
pub enum Fault {
    /// A font failed to load, for the reason held, and the text draws from
    /// its other fonts. With none left, every character takes an error box.
    Font(String),
    /// These characters need ink, the font at this index in the text's
    /// fonts gives their glyphs no outline and no colour bitmap, and each
    /// takes an error box.
    NoOutline(usize, Vec<char>),
    /// The glyph of this character is past the limit named, and it takes an
    /// error box.
    Glyph(char, String),
    /// The words are past the limit named, and the text keeps its last
    /// words.
    Layout(String),
}

/// New words, laid out.
#[derive(Clone, PartialEq)]
pub struct Layout {
    /// None when the words fit no texture.
    pub built: Option<Built>,
    pub faults: Vec<Fault>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Align {
    Left,
    Center,
    Right,
}

impl Align {
    pub const ALL: [Align; 3] = [Align::Left, Align::Center, Align::Right];

    pub fn as_str(self) -> &'static str {
        match self {
            Align::Left => "left",
            Align::Center => "center",
            Align::Right => "right",
        }
    }
}

/// True when a font of `fonts` that loaded holds outlines.
pub fn any_outlines(fonts: &[Font]) -> bool {
    fonts
        .iter()
        .flatten()
        .any(|d| Face::parse(d, 0).is_ok_and(|f| outlined(&f)))
}

/// The texture and the uniforms of one laid-out text.
#[derive(Clone, PartialEq)]
pub struct Built {
    pub width: u32,
    pub height: u32,
    /// RGBA8, row-major.
    pub texels: Vec<u8>,
    /// The block's width and height in em, `<name>_size`.
    pub size: [f64; 2],
    /// Float uniforms, each named by internal() from the text's name and its
    /// suffix.
    pub uniforms: Vec<(&'static str, f64)>,
    /// One quad per glyph drawn from the font, for a layer of the slug fill.
    /// Its glyph data is this texture's.
    pub quads: Quads,
    /// One quad per error box, for a layer's error-box batch.
    pub boxes: Quads,
    /// The glyphs drawn from the fonts' outlines, for the quads of the MTSDF
    /// fill.
    pub glyphs: Vec<Placement>,
    /// The colour glyphs, for a layer's colour batch.
    pub colour: Vec<Placement>,
    /// True when the glyph data that the quads name lies in the texture of
    /// a Glyphs, and this Built holds no texels.
    pub shared: bool,
}

/// A glyph of a font, placed in the block.
#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    /// The index of its font in the text's fonts.
    pub font: usize,
    pub gid: u16,
    pub ch: char,
    /// The glyph's origin in em from the block's top left, y down.
    pub origin: [f64; 2],
    /// The index of its character, counted over the words.
    pub index: usize,
    /// The index of its word, counted from 0.
    pub word: usize,
    pub line: usize,
    /// Its advance from start to end x, in em.
    pub span: [f64; 2],
}

/// The floats of one quad vertex: the attributes ghee_a0 to ghee_a3, a vec4
/// each.
pub const QUAD_FLOATS: usize = 16;

/// The quads of a text's inked glyphs, 4 vertices a quad at the corners
/// (0, 0), (1, 0), (0, 1), (1, 1) of the glyph's box from its top left, so the
/// triangles 0 1 2 and 2 1 3 cover it. A vertex holds QUAD_FLOATS floats:
///
/// - ghee_a0: the corner x and y, the character index, the word index
/// - ghee_a1: the ink box in em from the block's top left, y down: x0, y0,
///   x1, y1
/// - ghee_a2, slug fill: the texel of the glyph's band entries, its
///   horizontal bands, its vertical bands, 0
/// - ghee_a2, MTSDF fill and colour glyphs: the atlas texel coordinate at
///   (x0, y0), and the texels per em along x and along y
/// - ghee_a3: the line index, the advance's start and end x in em, 0
///
/// Every index counts from 0.
#[derive(Clone, PartialEq)]
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
    contours: Vec<Vec<Quad>>,
}

impl Outline {
    fn point(&self, x: f32, y: f32) -> [f64; 2] {
        [f64::from(x) * self.scale, f64::from(y) * self.scale]
    }

    fn push(&mut self, q: Quad) {
        if q[0] != q[1] || q[0] != q[2] {
            match self.contours.last_mut() {
                Some(c) => c.push(q),
                None => self.contours.push(vec![q]),
            }
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
        self.contours.push(Vec::new());
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
#[derive(Clone)]
struct Glyph {
    /// The box, quantized as the shader decodes it.
    min: [f64; 2],
    size: [f64; 2],
    /// p2 lies within OFFSET_MAX steps of p3 on both axes.
    curves: Vec<Curve>,
}

/// The closed contours of glyph `gid` in em, y up, none for a glyph with no
/// outline.
fn contours(face: &Face, gid: GlyphId) -> Vec<Vec<Quad>> {
    let mut o = Outline {
        scale: 1.0 / f64::from(face.units_per_em()),
        start: [0.0; 2],
        at: [0.0; 2],
        contours: Vec::new(),
    };
    let drawn = face.outline_glyph(gid, &mut o).is_some();
    // Coverage and the MTSDF need closed contours, and ttf-parser leaves the
    // last contour of a CFF2 glyph open.
    ttf_parser::OutlineBuilder::close(&mut o);
    if !drawn {
        return Vec::new();
    }
    o.contours.retain(|c| !c.is_empty());
    o.contours
}

/// The outline of glyph `gid`. None for a glyph with no outline. An error
/// names the limit of a glyph past it.
fn glyph(face: &Face, gid: GlyphId) -> std::result::Result<Option<Glyph>, String> {
    let quads = contours(face, gid).concat();
    if quads.is_empty() {
        return Ok(None);
    }
    let mut lo = [f64::MAX; 2];
    let mut hi = [f64::MIN; 2];
    for q in &quads {
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
        size[a] = ((hi[a] - min[a]) * BOX_SCALE).ceil().max(1.0) / BOX_SCALE;
        // A grid cell holds a glyph's corner at most CORNER_OFFSET em from
        // its own, and a cell lists the glyph up to INK_MARGIN past the box.
        if size[a] + INK_MARGIN > CORNER_OFFSET {
            return Err(format!("spans more than {} em", CORNER_OFFSET - INK_MARGIN));
        }
    }
    let curves = on_lattice(&quads, min, size);
    Ok(Some(Glyph { min, size, curves }))
}

/// The glyph that a layout draws for `ch`, found in its font or not, None
/// when it draws nothing for it. A character that no font has draws glyph 0,
/// the missing-glyph box, unless it needs no ink.
fn ink_glyph(found: Option<GlyphId>, ch: char) -> Option<u16> {
    (found.is_some() || needs_ink(ch)).then(|| found.map_or(0, |g| g.0))
}

/// A text's fonts, parsed, in order. None stands for a font that failed to
/// load or to parse.
type Faces<'a> = [Option<Face<'a>>];

fn faces(fonts: &[Font]) -> Vec<Option<Face<'_>>> {
    fonts
        .iter()
        .map(|f| f.as_ref().ok().and_then(|d| Face::parse(d, 0).ok()))
        .collect()
}

/// The font of `faces` that draws `ch`, its index, and its glyph for `ch`:
/// the first font whose cmap holds `ch`, else the first font, with no glyph,
/// which then draws its missing-glyph box. None when no font loaded.
fn pick<'f, 'a>(faces: &'f Faces<'a>, ch: char) -> Option<(usize, &'f Face<'a>, Option<GlyphId>)> {
    let mut first = None;
    for (i, face) in faces.iter().enumerate() {
        let Some(face) = face else { continue };
        if let Some(g) = face.glyph_index(ch) {
            return Some((i, face, Some(g)));
        }
        first.get_or_insert((i, face, None));
    }
    first
}

/// U+FE0F, the emoji variation selector: the character before it asks for
/// its emoji.
const EMOJI_SELECTOR: char = '\u{fe0f}';

/// The font of `faces` that draws `ch`, its index, and its glyph for `ch`,
/// as pick() finds them. A character followed by U+FE0F takes the first font
/// with a colour glyph for it instead, when one has.
fn pick_before<'f, 'a>(
    faces: &'f Faces<'a>,
    ch: char,
    next: Option<char>,
) -> Option<(usize, &'f Face<'a>, Option<GlyphId>)> {
    if next == Some(EMOJI_SELECTOR) {
        for (i, face) in faces.iter().enumerate() {
            let Some(face) = face else { continue };
            if let Some(g) = face.glyph_index(ch).filter(|g| is_colour(face, *g)) {
                return Some((i, face, Some(g)));
            }
        }
    }
    pick(faces, ch)
}

/// True when a font after `fonts` may draw a character of `words`: a
/// character that needs ink and that no font of `fonts` holds, or one that
/// U+FE0F follows.
pub fn may_fall_back(fonts: &[Font], words: &str) -> bool {
    let faces = faces(fonts);
    let chars: Vec<char> = words.chars().collect();
    chars.iter().enumerate().any(|(i, &ch)| {
        chars.get(i + 1) == Some(&EMOJI_SELECTOR)
            || (needs_ink(ch) && pick(&faces, ch).is_none_or(|(_, _, g)| g.is_none()))
    })
}

/// True when `face` holds outlines, TrueType or CFF.
fn outlined(face: &Face) -> bool {
    let t = face.tables();
    t.glyf.is_some() || t.cff.is_some() || t.cff2.is_some()
}

/// True when glyph `gid` of `face` is a colour glyph: a PNG bitmap and no
/// outline.
fn is_colour(face: &Face, gid: GlyphId) -> bool {
    colour::bitmap(face, gid).is_some() && contours(face, gid).is_empty()
}

/// The atlas id of glyph `gid` of the font at index `font` in a text's fonts.
pub fn atlas_id(font: usize, gid: u16) -> u32 {
    (font as u32) << 16 | u32::from(gid)
}

/// A text's fonts, for its MTSDF atlas. Glyph `gid` of the font at index `i`
/// takes the atlas id `i << 16 | gid`.
pub struct AtlasFont(pub Vec<Font>);

impl atlas::Font for AtlasFont {
    /// A text whose one font with outlines comes first keys its atlas with
    /// that font's sha256, so it shares the atlas of a text of that font
    /// alone. Any other text keys it with the sha256 of each such font's
    /// index, as 4 bytes little-endian, and sha256, in order. A font with no
    /// outline gives the atlas no glyph, and it stays out of the key.
    fn key(&self) -> [u8; 32] {
        let outlined: Vec<(usize, &[u8])> = self
            .0
            .iter()
            .enumerate()
            .filter_map(|(i, f)| {
                let d = f.as_ref().ok()?;
                outlined(&Face::parse(d, 0).ok()?).then_some((i, &d[..]))
            })
            .collect();
        if let [(0, d)] = outlined[..] {
            return Sha256::digest(d).into();
        }
        let mut h = Sha256::new();
        for (i, d) in outlined {
            h.update((i as u32).to_le_bytes());
            h.update(Sha256::digest(d));
        }
        h.finalize().into()
    }

    /// A colour glyph draws from the colour atlas, and takes no id here, nor
    /// does a glyph with no outline for a character that needs no ink, such
    /// as a space, which draws nothing. A character that U+FE0F turns to a
    /// colour glyph keeps the id of its glyph alone, so the atlas may hold a
    /// field that no layer reads.
    fn glyph(&self, ch: char) -> Option<u32> {
        let ch = if ch == '\t' { ' ' } else { ch };
        if ch.is_control() {
            return None;
        }
        let faces = faces(&self.0);
        let (i, face, found) = pick(&faces, ch)?;
        let gid = ink_glyph(found, ch)?;
        let blank = !needs_ink(ch) && contours(face, GlyphId(gid)).is_empty();
        (!blank && !is_colour(face, GlyphId(gid))).then(|| atlas_id(i, gid))
    }

    fn outline(&self, id: u32) -> Result<Vec<Vec<atlas::Curve>>> {
        let font = self
            .0
            .get((id >> 16) as usize)
            .and_then(|f| f.as_ref().ok());
        let font = font.ok_or_else(|| anyhow!("no font holds atlas id {id}"))?;
        let face = Face::parse(font, 0).map_err(|e| anyhow!("unreadable font: {e}"))?;
        Ok(contours(&face, GlyphId(id as u16)))
    }
}

/// `quads` on the lattice of a box at `min`, `size` em.
fn on_lattice(quads: &[Quad], min: [f64; 2], size: [f64; 2]) -> Vec<Curve> {
    let snap = |p: [f64; 2]| {
        [0, 1].map(|a| {
            ((p[a] - min[a]) / size[a] * LATTICE)
                .round()
                .clamp(0.0, LATTICE) as u16
        })
    };
    let mut curves: Vec<Curve> = Vec::new();
    let mut todo: Vec<Quad> = quads.iter().rev().copied().collect();
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
    curves
}

/// The part of the convex polygon `poly` where `f` is 0 or more, for an
/// `f` linear in the point.
fn clip(poly: &[[f64; 2]], f: impl Fn([f64; 2]) -> f64) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    for (i, &a) in poly.iter().enumerate() {
        let b = poly[(i + 1) % poly.len()];
        let (fa, fb) = (f(a), f(b));
        if fa >= 0.0 {
            out.push(a);
        }
        if (fa >= 0.0) != (fb >= 0.0) {
            out.push(lerp(a, b, fa / (fa - fb)));
        }
    }
    out
}

/// The error box as a glyph: its yellow stripes, clipped to the box, as the
/// outline. Stripe k covers x - y from 2k to 2k + 1 STRIPE, in em from the
/// box's bottom left, y up.
fn error_box() -> Glyph {
    let [w, h] = ERROR_BOX;
    let mut quads: Vec<Quad> = Vec::new();
    let first = (-h / (2.0 * STRIPE)).floor() as i32;
    let last = (w / (2.0 * STRIPE)).ceil() as i32;
    for k in first..=last {
        let (c0, c1) = (f64::from(2 * k) * STRIPE, f64::from(2 * k + 1) * STRIPE);
        let rect = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
        let stripe = clip(&clip(&rect, |p| p[0] - p[1] - c0), |p| c1 - p[0] + p[1]);
        let area: f64 = (0..stripe.len())
            .map(|i| {
                let (a, b) = (stripe[i], stripe[(i + 1) % stripe.len()]);
                a[0] * b[1] - b[0] * a[1]
            })
            .sum();
        if area.abs() < 1e-9 {
            continue;
        }
        for (i, &a) in stripe.iter().enumerate() {
            let b = stripe[(i + 1) % stripe.len()];
            quads.push([a, b, b]);
        }
    }
    Glyph {
        min: [0.0; 2],
        size: ERROR_BOX,
        curves: on_lattice(&quads, [0.0; 2], ERROR_BOX),
    }
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

/// `n` bands across `axis`, each listing the curves that cross it. Axis 1
/// splits the box along y into horizontal bands, axis 0 along x into vertical
/// ones. None when a band holds more curves than the shader's loop reads.
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

/// What a placed character draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Ink {
    /// The outline of a glyph: the index of its font in the text's fonts,
    /// and its glyph id.
    Glyph(usize, u16),
    /// The bitmap of a colour glyph, which only a layer's colour batch draws.
    Colour(usize, u16),
    /// An error box.
    Box,
}

impl Ink {
    /// The ink drawn from the text's texture of curves.
    fn curves(self) -> Option<Ink> {
        (!matches!(self, Ink::Colour(..))).then_some(self)
    }
}

/// A character placed in the block, its origin in em. An error box's origin
/// is the box's bottom left corner.
struct Placed {
    /// None draws nothing.
    ink: Option<Ink>,
    ch: char,
    x: f64,
    y: f64,
    /// The index of its character, counted over the words.
    index: usize,
    /// The index of its word, a run of non-whitespace, counted from 0.
    /// Whitespace takes the word before it.
    word: usize,
    line: usize,
    /// Its advance from start to end x, in em.
    span: [f64; 2],
}

/// The GPOS feature that holds a font's pair kerning.
const KERN: Tag = Tag::from_bytes(b"kern");

/// The kerning of the pair `l` then `r`, in font units, as HarfBuzz gives
/// it. A font whose GPOS lists the `kern` feature kerns through its pair
/// adjustments alone, every other font through its `kern` table.
fn kerning(face: &Face, l: GlyphId, r: GlyphId) -> f64 {
    let tables = face.tables();
    if let Some(gpos) = tables.gpos.filter(|g| g.features.find(KERN).is_some()) {
        let mut lookups: Vec<u16> = gpos
            .features
            .into_iter()
            .filter(|f| f.tag == KERN)
            .flat_map(|f| f.lookup_indices)
            .collect();
        lookups.sort_unstable();
        lookups.dedup();
        return lookups
            .into_iter()
            .filter_map(|i| gpos.lookups.get(i))
            .filter_map(|lookup| pair_advance(lookup, l, r))
            .map(f64::from)
            .sum();
    }
    let Some(kern) = tables.kern else {
        return 0.0;
    };
    kern.subtables
        .into_iter()
        .filter(|s| s.horizontal && !s.variable && !s.has_cross_stream)
        .filter_map(|s| s.glyphs_kerning(l, r))
        .map(f64::from)
        .sum()
}

/// The x advance that a GPOS lookup adds to `l` before `r`, in font units.
/// The first pair adjustment that covers `l` and holds a value for the pair
/// applies, and the lookup's later subtables are skipped. A class pair
/// adjustment holds a value for every second glyph, class 0 included.
fn pair_advance(lookup: Lookup, l: GlyphId, r: GlyphId) -> Option<i16> {
    lookup
        .subtables
        .into_iter::<PositioningSubtable>()
        .find_map(|s| {
            let PositioningSubtable::Pair(pair) = s else {
                return None;
            };
            let i = pair.coverage().get(l)?;
            let (first, _) = match pair {
                PairAdjustment::Format1 { sets, .. } => sets.get(i)?.get(r)?,
                PairAdjustment::Format2 {
                    classes, matrix, ..
                } => matrix.get((classes.0.get(l), classes.1.get(r)))?,
            };
            Some(first.x_advance)
        })
}

/// The block holds the lines top to bottom. Glyph origins are in em from the
/// block's bottom left, y up like the font's own units, and the shader turns
/// the y-down points of its callers y up. Returns the placed glyphs and the
/// block's width and height, in em.
///
/// Line height and baseline come from the first font of `faces`, and each
/// glyph's advance and kerning from its own font. Two glyphs of two fonts
/// do not kern. With no font in `faces`, every character that needs ink takes
/// an error box, ERROR_ADVANCE wide.
fn layout(faces: &Faces, words: &str, align: Align) -> (Vec<Placed>, f64, f64) {
    let (ascent, line) = face_lines(faces);
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
        let mut prev: Option<(usize, GlyphId)> = None;
        // A layer's glyph_index counts every character of the words, the
        // newline that ends a line too.
        let chars: Vec<char> = text.chars().collect();
        for (i, &ch) in chars.iter().enumerate() {
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
            let (ink, origin, advance) = match pick_before(faces, ch, chars.get(i + 1).copied()) {
                // A default ignorable that no font has takes no room.
                Some((_, _, None)) if ignorable(ch) => (None, x, 0.0),
                Some((f, face, found)) => {
                    // A character that no font has draws glyph 0 of the
                    // first, the missing-glyph box, and whitespace leaves the
                    // box out.
                    let upem = f64::from(face.units_per_em());
                    let gid = found.unwrap_or(GlyphId(0));
                    if let Some((_, p)) = prev.filter(|(pf, _)| *pf == f) {
                        x += kerning(face, p, gid) / upem;
                    }
                    prev = Some((f, gid));
                    let advance = f64::from(face.glyph_hor_advance(gid).unwrap_or(0)) / upem;
                    (ink_glyph(found, ch).map(|g| Ink::Glyph(f, g)), x, advance)
                }
                None if ignorable(ch) => (None, x, 0.0),
                None if needs_ink(ch) => (
                    Some(Ink::Box),
                    x + (ERROR_ADVANCE - ERROR_BOX[0]) / 2.0,
                    ERROR_ADVANCE,
                ),
                None => (None, x, ERROR_ADVANCE),
            };
            placed.push(Placed {
                ink,
                ch,
                x: origin,
                y,
                index: at,
                word,
                line: k,
                span: [x, x + advance],
            });
            x += advance;
        }
        index += chars.len() + 1;
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

/// The first baseline below the top of a block and the height of a line,
/// in em, from the first font of `faces` that parsed: its ascent, and its
/// ascent less its descent plus its line gap.
fn face_lines(faces: &Faces) -> (f64, f64) {
    match faces.iter().flatten().next() {
        Some(face) => {
            let upem = f64::from(face.units_per_em());
            let line = f64::from(face.ascender()) - f64::from(face.descender())
                + f64::from(face.line_gap());
            (f64::from(face.ascender()) / upem, line / upem)
        }
        None => (ERROR_ASCENT, ERROR_LINE),
    }
}

/// The first baseline below the top of a block and the height of a line,
/// in em, of a text in `fonts`, as its layout places them.
pub fn line_metrics(fonts: &[Font]) -> (f64, f64) {
    face_lines(&faces(fonts))
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

/// The texture of curves that a layout packs its glyphs into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Curves {
    /// None: the layout of a text that the MTSDF fill alone draws. Its Built
    /// holds no texels, no uniforms and no glyph quads.
    None,
    /// A texture of the layout's own, with the grid that `<name>_cover`
    /// reads.
    Own,
    /// The texture of a Glyphs, which needs no grid, or a texture of the
    /// layout's own when the glyphs do not fit it.
    Shared,
}

/// Lay out `words` in `fonts`, in order, and pack glyphs and grid into
/// texels. A font that failed to load or to parse drops out with a fault.
pub fn build(fonts: &[Font], words: &str, align: Align) -> Layout {
    lay(fonts, words, align, Curves::Own, None)
}

/// Lay out `words` as build() does, with no texture of curves: the layout
/// of a text that the MTSDF fill alone draws. Its Built holds no texels, no
/// uniforms and no glyph quads.
pub fn place(fonts: &[Font], words: &str, align: Align) -> Layout {
    lay(fonts, words, align, Curves::None, None)
}

/// The outline of a glyph, or why it draws none.
type Read = std::result::Result<Option<Glyph>, String>;

/// Lay out `words` into the texture `curves`, reading each outline, and
/// each glyph's data at BANDS bands, through `glyphs` when given.
fn lay(
    fonts: &[Font],
    words: &str,
    align: Align,
    curves: Curves,
    mut glyphs: Option<&mut Glyphs>,
) -> Layout {
    let faces = faces(fonts);
    let (mut placed, width, height) = layout(&faces, words, align);
    let keys: Vec<Option<usize>> = match glyphs.as_deref_mut() {
        Some(g) => fonts.iter().map(|f| g.font_key(f)).collect(),
        None => vec![None; fonts.len()],
    };
    let read = |f: usize, gid: u16| -> Read {
        faces[f]
            .as_ref()
            .map_or(Ok(None), |face| glyph(face, GlyphId(gid)))
    };
    let (inks, mut faults) = settle(
        &mut placed,
        curves != Curves::None,
        |f, gid| match (glyphs.as_deref_mut(), keys[f]) {
            (Some(g), Some(k)) => match g.outlines.get(&(k, gid)) {
                Some(got) => got.clone(),
                None => g.keep_outline((k, gid), read(f, gid)),
            },
            _ => read(f, gid),
        },
        |f, gid| {
            faces[f]
                .as_ref()
                .is_some_and(|face| colour::bitmap(face, GlyphId(gid)).is_some())
        },
    );
    let failed = fonts
        .iter()
        .zip(&faces)
        .filter_map(|(f, face)| match (f, face) {
            (Err(why), _) => Some(why.clone()),
            (Ok(d), None) => Some(format!("unreadable font: {}", Face::parse(d, 0).err()?)),
            _ => None,
        });
    let failed: Vec<Fault> = failed.map(Fault::Font).collect();
    faults.splice(0..0, failed);
    // A glyph's key in `glyphs`: the index of its font's content and its
    // glyph id, None for the error box.
    let key = |ink: Ink| match ink {
        Ink::Glyph(f, gid) => keys[f].map(|k| Some((k, gid))),
        _ => Some(None),
    };
    let packed = match curves {
        Curves::None => Ok(boxes_alone(&placed, &inks, height)),
        Curves::Own | Curves::Shared => {
            let mut blocks: HashMap<Ink, Arc<Block>> = HashMap::new();
            for (ink, g) in &inks {
                let made = match (glyphs.as_deref_mut(), key(*ink)) {
                    (Some(store), Some(k)) => store.block(k, g),
                    _ => Block::of(g, BANDS).map(Arc::new),
                };
                if let Some(b) = made {
                    blocks.insert(*ink, b);
                }
            }
            // The texture of the Glyphs takes the words that a texture of
            // their own packs at BANDS bands, so they draw the same.
            let shared = match (curves, glyphs) {
                (Curves::Shared, Some(store)) if fits(&placed, &inks, &blocks) => {
                    store.share(&placed, &inks, &blocks, key, height)
                }
                _ => None,
            };
            match shared {
                Some(built) => Ok(built),
                None => pack_fitting(&placed, &inks, &blocks, height),
            }
        }
    };
    finish(&placed, packed, faults, width, height)
}

/// Settle what each placed character draws. `outline` reads the outline of
/// a glyph, and `bitmap` says whether the glyph has a colour bitmap, each
/// given the index of the glyph's font and its glyph id. A glyph with no
/// outline draws nothing for a character that needs no ink, its bitmap where
/// it has one, and an error box otherwise. A glyph past a limit draws an
/// error box, and BAND_CURVES limits a glyph only when `curves` packs it into
/// a texture of curves. Returns the glyph of each ink, and the faults met.
fn settle(
    placed: &mut [Placed],
    curves: bool,
    mut outline: impl FnMut(usize, u16) -> std::result::Result<Option<Glyph>, String>,
    mut bitmap: impl FnMut(usize, u16) -> bool,
) -> (HashMap<Ink, Glyph>, Vec<Fault>) {
    let mut read: HashMap<(usize, u16), Read> = HashMap::new();
    let mut coloured: HashMap<(usize, u16), bool> = HashMap::new();
    let mut missing: Vec<(usize, Vec<char>)> = Vec::new();
    let mut past: Vec<(char, String)> = Vec::new();
    for p in placed.iter_mut() {
        let Some(Ink::Glyph(f, gid)) = p.ink else {
            continue;
        };
        let got = read.entry((f, gid)).or_insert_with(|| {
            let g = outline(f, gid)?;
            // Every band of a glyph is read by one loop of BAND_CURVES, and a
            // glyph of no more curves fits any band.
            match &g {
                Some(g)
                    if curves
                        && g.curves.len() > BAND_CURVES
                        && (bands(&g.curves, 1, BANDS, BAND_EPS / g.size[1]).is_none()
                            || bands(&g.curves, 0, BANDS, BAND_EPS / g.size[0]).is_none()) =>
                {
                    Err(format!("has more than {BAND_CURVES} curves in one band"))
                }
                _ => Ok(g),
            }
        });
        match got {
            Ok(Some(_)) => continue,
            Ok(None) if !needs_ink(p.ch) => {
                p.ink = None;
                continue;
            }
            Ok(None) if *coloured.entry((f, gid)).or_insert_with(|| bitmap(f, gid)) => {
                p.ink = Some(Ink::Colour(f, gid));
                continue;
            }
            Ok(None) => {
                let at = match missing.iter().position(|m| m.0 == f) {
                    Some(at) => at,
                    None => {
                        missing.push((f, Vec::new()));
                        missing.len() - 1
                    }
                };
                if !missing[at].1.contains(&p.ch) {
                    missing[at].1.push(p.ch);
                }
            }
            Err(why) => {
                if !past.iter().any(|(c, _)| *c == p.ch) {
                    past.push((p.ch, why.clone()));
                }
            }
        }
        p.ink = Some(Ink::Box);
        p.x = (p.span[0] + p.span[1] - ERROR_BOX[0]) / 2.0;
    }
    let mut glyphs: HashMap<Ink, Glyph> = read
        .into_iter()
        .filter_map(|((f, gid), g)| Some((Ink::Glyph(f, gid), g.ok()??)))
        .collect();
    if placed.iter().any(|p| p.ink == Some(Ink::Box)) {
        glyphs.insert(Ink::Box, error_box());
    }
    let mut faults: Vec<Fault> = missing
        .into_iter()
        .map(|(f, chars)| Fault::NoOutline(f, chars))
        .collect();
    faults.extend(past.into_iter().map(|(ch, why)| Fault::Glyph(ch, why)));
    (glyphs, faults)
}

/// The layout of the placed glyphs in a block `width` x `height` em, from
/// `packed`, their texture of curves and their quads, or why the words fit
/// no texture.
fn finish(
    placed: &[Placed],
    packed: Result<Built>,
    mut faults: Vec<Fault>,
    width: f64,
    height: f64,
) -> Layout {
    let built = match packed {
        Ok(mut built) => {
            built.size = [width, height];
            let placement = |p: &Placed, font: usize, gid: u16| Placement {
                font,
                gid,
                ch: p.ch,
                origin: [p.x, height - p.y],
                index: p.index,
                word: p.word,
                line: p.line,
                span: p.span,
            };
            for p in placed {
                match p.ink {
                    Some(Ink::Glyph(f, gid)) => built.glyphs.push(placement(p, f, gid)),
                    Some(Ink::Colour(f, gid)) => built.colour.push(placement(p, f, gid)),
                    _ => {}
                }
            }
            Some(built)
        }
        Err(e) => {
            faults.push(Fault::Layout(e.to_string()));
            None
        }
    };
    Layout { built, faults }
}

/// The Built of a layout with no texture of curves: the error boxes alone.
fn boxes_alone(placed: &[Placed], glyphs: &HashMap<Ink, Glyph>, height: f64) -> Built {
    let inked = inked(placed, glyphs, |_| 0);
    Built {
        width: 0,
        height: 0,
        texels: Vec::new(),
        size: [0.0; 2],
        uniforms: Vec::new(),
        quads: glyph_quads(&inked, false, 0, height),
        boxes: glyph_quads(&inked, true, 0, height),
        glyphs: Vec::new(),
        colour: Vec::new(),
        shared: false,
    }
}

/// The glyphs that a renderer's texts draw: the outline of each glyph and
/// its data at BANDS bands, each made once, and one texture of curves that
/// the slug layers share. A glyph's data goes into the texture at the first
/// layout that draws it, and stays at that texel. The texture only grows.
#[derive(Default)]
pub struct Glyphs {
    /// One font of each content that a layout drew from. A glyph's key names
    /// its font by the index here.
    fonts: Vec<Arc<Vec<u8>>>,
    /// Each font that a layout drew from, by its address, with the index of
    /// its content in `fonts`. The Weak keeps a later font off the address.
    seen: Vec<(Weak<Vec<u8>>, usize)>,
    /// The outline of each glyph, by the index of its font and its glyph id.
    outlines: HashMap<(usize, u16), Read>,
    /// The data of each glyph at BANDS bands, by its key: the index of its
    /// font and its glyph id, or None for the error box. None for a glyph
    /// with a band past BAND_CURVES curves.
    blocks: HashMap<Option<(usize, u16)>, Option<Arc<Block>>>,
    /// The texture, TEX_W texels to a row.
    tex: Vec<[u8; 4]>,
    /// The texel where the data of each glyph in `tex` starts, by its key.
    starts: HashMap<Option<(usize, u16)>, usize>,
    /// The texels and the rows of `tex` that the renderer took.
    sent: usize,
    rows: usize,
    /// Each layout that lay() made, by the content of its fonts, its align
    /// and its texture of curves, then by its words. A layout reads the
    /// texture at the texels where its glyphs went in, which stay. LAID_KEEP
    /// layouts at most: the next one starts the store again.
    laid: HashMap<(Vec<usize>, Align, Curves), HashMap<String, Layout>>,
    laid_count: usize,
    /// The outlines read from the fonts, for a test to count.
    #[cfg(test)]
    pub(crate) reads: usize,
    /// The layouts that lay() made, for a test to count.
    #[cfg(test)]
    pub(crate) lays: usize,
}

/// The layouts that a Glyphs keeps.
const LAID_KEEP: usize = 512;

/// The rows of the texture of a Glyphs that changed since the renderer last
/// took them.
pub struct Changed {
    /// The texture's width and height. A height past the last one taken sets
    /// the texture again, whole, from row 0.
    pub width: u32,
    pub height: u32,
    pub first: u32,
    /// RGBA8, the rows from `first` on.
    pub texels: Vec<u8>,
}

impl Glyphs {
    /// Lay `words` out in `fonts`, into the texture of curves `curves`, as
    /// build() and place() do. Words laid out before in fonts of the same
    /// content, with the same align and texture, take that layout again.
    pub fn lay(&mut self, fonts: &[Font], words: &str, align: Align, curves: Curves) -> Layout {
        // A font that failed to load names its reason in the layout's
        // faults, so such a layout is made each time.
        let keys: Option<Vec<usize>> = fonts.iter().map(|f| self.font_key(f)).collect();
        let Some(keys) = keys else {
            return self.lay_now(fonts, words, align, curves);
        };
        let key = (keys, align, curves);
        if let Some(laid) = self.laid.get(&key).and_then(|w| w.get(words)) {
            return laid.clone();
        }
        let laid = self.lay_now(fonts, words, align, curves);
        if laid.built.is_some() {
            if self.laid_count >= LAID_KEEP {
                self.laid.clear();
                self.laid_count = 0;
            }
            self.laid
                .entry(key)
                .or_default()
                .insert(words.to_string(), laid.clone());
            self.laid_count += 1;
        }
        laid
    }

    fn lay_now(&mut self, fonts: &[Font], words: &str, align: Align, curves: Curves) -> Layout {
        #[cfg(test)]
        {
            self.lays += 1;
        }
        lay(fonts, words, align, curves, Some(self))
    }

    /// The index of the content of `font` among the fonts, None for a font
    /// that failed to load.
    fn font_key(&mut self, font: &Font) -> Option<usize> {
        let font = font.as_ref().ok()?;
        self.seen.retain(|(w, _)| w.strong_count() > 0);
        if let Some((_, k)) = self
            .seen
            .iter()
            .find(|(w, _)| w.as_ptr() == Arc::as_ptr(font))
        {
            return Some(*k);
        }
        let k = match self.fonts.iter().position(|f| f[..] == font[..]) {
            Some(k) => k,
            None => {
                self.fonts.push(font.clone());
                self.fonts.len() - 1
            }
        };
        self.seen.push((Arc::downgrade(font), k));
        Some(k)
    }

    /// Hold `got`, the outline of the glyph of key `k`, and give it.
    fn keep_outline(&mut self, k: (usize, u16), got: Read) -> Read {
        #[cfg(test)]
        {
            self.reads += 1;
        }
        self.outlines.insert(k, got.clone());
        got
    }

    /// The data of glyph `g`, of key `k`, at BANDS bands.
    fn block(&mut self, k: Option<(usize, u16)>, g: &Glyph) -> Option<Arc<Block>> {
        self.blocks
            .entry(k)
            .or_insert_with(|| Block::of(g, BANDS).map(Arc::new))
            .clone()
    }

    /// The Built of `placed` whose quads read each glyph's data from the
    /// texture, with `blocks` the data of each ink and `key` its key. A glyph
    /// new to the texture goes in. None when the texture would grow past
    /// MAX_TEX_H rows, and it then keeps what it held.
    fn share(
        &mut self,
        placed: &[Placed],
        glyphs: &HashMap<Ink, Glyph>,
        blocks: &HashMap<Ink, Arc<Block>>,
        key: impl Fn(Ink) -> Option<Option<(usize, u16)>>,
        height: f64,
    ) -> Option<Built> {
        let (len, mut new) = (self.tex.len(), Vec::new());
        let mut fits = true;
        for p in placed {
            let Some(ink) = p.ink.and_then(Ink::curves) else {
                continue;
            };
            let (Some(k), Some(b)) = (key(ink), blocks.get(&ink)) else {
                fits = false;
                break;
            };
            if !self.starts.contains_key(&k) {
                self.starts.insert(k, b.put(&mut self.tex));
                new.push(k);
            }
        }
        if !fits || tex_rows(self.tex.len()) > MAX_TEX_H {
            self.tex.truncate(len);
            for k in new {
                self.starts.remove(&k);
            }
            return None;
        }
        let inked = inked(placed, glyphs, |ink| {
            self.starts[&key(ink).expect("every ink has its key")]
        });
        Some(Built {
            width: 0,
            height: 0,
            texels: Vec::new(),
            size: [0.0; 2],
            uniforms: Vec::new(),
            quads: glyph_quads(&inked, false, BANDS, height),
            boxes: glyph_quads(&inked, true, BANDS, height),
            glyphs: Vec::new(),
            colour: Vec::new(),
            shared: true,
        })
    }

    /// The rows that changed since took(), None when none did.
    pub fn changed(&self) -> Option<Changed> {
        if self.sent == self.tex.len() {
            return None;
        }
        let height = tex_rows(self.tex.len());
        let (first, end) = match height > self.rows {
            true => (0, height * TEX_W),
            false => (self.sent / TEX_W, self.tex.len().next_multiple_of(TEX_W)),
        };
        let mut texels = vec![0u8; (end - first * TEX_W) * 4];
        let have = self.tex[first * TEX_W..].as_flattened();
        texels[..have.len()].copy_from_slice(have);
        Some(Changed {
            width: TEX_W as u32,
            height: height as u32,
            first: first as u32,
            texels,
        })
    }

    /// Mark the rows that changed() gave as taken.
    pub fn took(&mut self) {
        self.sent = self.tex.len();
        self.rows = self.rows.max(tex_rows(self.tex.len()));
    }

    /// The height of the texture that the renderer took, 0 before the first.
    pub fn rows(&self) -> u32 {
        self.rows as u32
    }

    /// The float uniforms of the texture at `rows` rows that the reader of a
    /// glyph's curves takes, each named by internal() from a layer's name
    /// and its suffix.
    pub fn uniforms(rows: u32) -> [(&'static str, f64); 3] {
        [
            ("_tw", TEX_W as f64),
            ("_itw", 1.0 / TEX_W as f64),
            ("_ith", 1.0 / f64::from(rows)),
        ]
    }
}

/// Each placed glyph drawn from the curves, with its box: where its data
/// starts, as `start` gives it for its ink, and its corner on the lattices
/// that the cells decode.
fn inked<'a>(
    placed: &'a [Placed],
    glyphs: &HashMap<Ink, Glyph>,
    start: impl Fn(Ink) -> usize,
) -> Vec<(&'a Placed, Packed)> {
    // Origins sit on the 1/ORIGIN_SCALE em lattice and box corners on the
    // 1/BOX_SCALE em one, so a glyph that spans cells decodes to the same box
    // in each of them.
    let snap = |v: f64| (v * ORIGIN_SCALE).round() / ORIGIN_SCALE;
    placed
        .iter()
        .filter_map(|p| {
            let ink = p.ink.and_then(Ink::curves)?;
            let g = &glyphs[&ink];
            Some((
                p,
                (
                    start(ink),
                    [snap(p.x) + g.min[0], snap(p.y) + g.min[1]],
                    g.size,
                ),
            ))
        })
        .collect()
}

/// The quads over the glyphs of `inked`, the error boxes when `boxed`, the
/// glyphs drawn from outlines otherwise, `nb` bands to an axis, laid over a
/// block `height` em tall.
fn glyph_quads(inked: &[(&Placed, Packed)], boxed: bool, nb: usize, height: f64) -> Quads {
    let mut floats = Vec::new();
    let mut count = 0;
    for (p, (at, lo, size)) in inked {
        if (p.ink == Some(Ink::Box)) != boxed {
            continue;
        }
        // The box flips to y down, the way a layer's caller counts.
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
        count += 1;
    }
    Quads {
        floats,
        per_vertex: QUAD_FLOATS,
        count,
    }
}

/// A glyph's data at some number of bands, as a texture of curves holds
/// it: the box size, an entry for each band, horizontal bands first, and
/// the lists of the bands. A band whose curves match an earlier band's
/// shares that band's lists.
struct Block {
    size: [u8; 4],
    /// Each band's index in `lists`, and its count of curves.
    entries: Vec<(usize, u8)>,
    /// The texels of each band's two lists. The ray toward + along the band
    /// reads the first list, the ray toward - the second, mirrored. Each
    /// sorts by the far end of its curves, largest first.
    lists: Vec<Vec<[u8; 4]>>,
}

impl Block {
    /// The data of glyph `g`, `nb` bands to an axis. None when a band holds
    /// more than BAND_CURVES curves.
    fn of(g: &Glyph, nb: usize) -> Option<Block> {
        let hbands = bands(&g.curves, 1, nb, BAND_EPS / g.size[1])?;
        let vbands = bands(&g.curves, 0, nb, BAND_EPS / g.size[0])?;
        let sz = |a: usize| fixed(g.size[a], 0.0, BOX_SCALE).unwrap_or(0);
        let mut index: HashMap<(usize, &[usize]), usize> = HashMap::new();
        let mut lists: Vec<Vec<[u8; 4]>> = Vec::new();
        let mut entries = Vec::with_capacity(2 * nb);
        let all = hbands
            .iter()
            .map(|b| (0, b))
            .chain(vbands.iter().map(|b| (1, b)));
        for (along, band) in all {
            let k = *index.entry((along, band.as_slice())).or_insert_with(|| {
                let mut texels = Vec::with_capacity(band.len() * 4);
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
                        texels.extend(curve_texels(c));
                    }
                }
                lists.push(texels);
                lists.len() - 1
            });
            entries.push((k, band.len() as u8));
        }
        Some(Block {
            size: u16x2(sz(0), sz(1)),
            entries,
            lists,
        })
    }

    /// Append the data to `tex`, and give the texel where it starts.
    fn put(&self, tex: &mut Vec<[u8; 4]>) -> usize {
        let at = tex.len();
        tex.resize(at + 1 + self.entries.len(), [0; 4]);
        tex[at] = self.size;
        let mut locs = Vec::with_capacity(self.lists.len());
        for list in &self.lists {
            // Both lists of a band sit in one texture row, so the shader
            // steps from curve to curve along x.
            if tex.len() % TEX_W + list.len() > TEX_W {
                tex.resize(tex.len().next_multiple_of(TEX_W), [0; 4]);
            }
            locs.push(tex.len());
            tex.extend_from_slice(list);
        }
        for (j, (k, n)) in self.entries.iter().enumerate() {
            tex[at + 1 + j] = u24u8(locs[*k], *n);
        }
        at
    }

    /// The length that a texture of `len` texels reaches with the data
    /// appended, as put() appends it.
    fn end(&self, len: usize) -> usize {
        let mut n = len + 1 + self.entries.len();
        for list in &self.lists {
            if n % TEX_W + list.len() > TEX_W {
                n = n.next_multiple_of(TEX_W);
            }
            n += list.len();
        }
        n
    }
}

/// pack() at the most bands to an axis, up to BANDS, that fit the texture,
/// `blocks` the data of each glyph at BANDS bands.
fn pack_fitting(
    placed: &[Placed],
    glyphs: &HashMap<Ink, Glyph>,
    blocks: &HashMap<Ink, Arc<Block>>,
    height: f64,
) -> Result<Built> {
    let mut need = None;
    let mut nb = BANDS;
    while nb >= 1 {
        let fewer: HashMap<Ink, Arc<Block>>;
        let at = match nb {
            BANDS => blocks,
            _ => {
                fewer = glyphs
                    .iter()
                    .filter_map(|(ink, g)| Some((*ink, Arc::new(Block::of(g, nb)?))))
                    .collect();
                &fewer
            }
        };
        match pack(placed, glyphs, at, nb, height) {
            Ok(built) => return Ok(built),
            Err(Unpacked::TooTall(tex_h)) => need = Some(tex_h),
            // Fewer bands hold more curves each. Once the words outgrow the
            // texture, a band past BAND_CURVES at fewer bands means that no
            // band count fits them.
            Err(Unpacked::Band(_)) if need.is_some() => break,
            Err(Unpacked::Band(ch)) => {
                bail!("{ch:?} has more than {BAND_CURVES} curves in one band")
            }
            Err(Unpacked::Fail(e)) => return Err(e),
        }
        nb /= 2;
    }
    bail!(
        "the words need a {TEX_W}x{} texture, and a VC4 holds {TEX_W}x{MAX_TEX_H} at most",
        need.unwrap_or(0)
    )
}

/// True when pack() packs the placed glyphs at BANDS bands into a texture of
/// their own, `blocks` the data of each glyph at BANDS bands.
fn fits(
    placed: &[Placed],
    glyphs: &HashMap<Ink, Glyph>,
    blocks: &HashMap<Ink, Arc<Block>>,
) -> bool {
    let mut len = 1;
    let mut seen: HashSet<Ink> = HashSet::new();
    for p in placed {
        let Some(ink) = p.ink.and_then(Ink::curves) else {
            continue;
        };
        if seen.insert(ink) {
            match blocks.get(&ink) {
                Some(b) => len = b.end(len),
                None => return false,
            }
        }
    }
    grid(&inked(placed, glyphs, |_| 0), len).is_ok()
}

/// Why pack() refused the words.
enum Unpacked {
    /// The texture rows needed by the words, past MAX_TEX_H.
    TooTall(usize),
    /// The glyph of this character holds more than BAND_CURVES curves in a
    /// band.
    Band(char),
    Fail(anyhow::Error),
}

/// The texture rows that hold `texels`.
fn tex_rows(texels: usize) -> usize {
    texels
        .div_ceil(TEX_W)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX)
}

/// The grid over the ink of every glyph of `inked`, in a texture whose
/// glyph data ends at texel `len`.
struct Grid {
    /// Its corner in em, on the 1/ORIGIN_SCALE em lattice, as the origins
    /// are.
    gx: f64,
    gy: f64,
    cols: usize,
    rows: usize,
    /// The side of a cell in em.
    cell: f64,
    /// The slots of each cell, row by row: for each glyph that the cell
    /// lists, the texel where its data starts, the top bit marking the
    /// cell's last, and the corner of its box from the cell's.
    slots: Vec<Vec<[[u8; 4]; SLOT]>>,
}

/// The grid over the glyphs of `inked` in a texture whose glyph data ends at
/// texel `len`. Each cell lists the glyphs whose box, grown by INK_MARGIN,
/// reaches into it.
fn grid(inked: &[(&Placed, Packed)], len: usize) -> std::result::Result<Grid, Unpacked> {
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
            len.next_multiple_of(CELL_K * SLOT)
                + cols.saturating_mul(rows).saturating_mul(CELL_K * SLOT),
        );
        if need > MAX_TEX_H {
            return Err(Unpacked::TooTall(need));
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
            return Err(Unpacked::Fail(anyhow!(
                "more than {CELL_K} glyphs overlap at one spot"
            )));
        }
        cell /= 2.0;
    };
    let mut slots = Vec::with_capacity(cols * rows);
    for r in 0..rows {
        for c in 0..cols {
            let corner = [gx + c as f64 * cell, gy + r as f64 * cell];
            let list = &grid[r * cols + c];
            let mut cell_slots = Vec::with_capacity(list.len());
            for (k, (h, lo, _)) in list.iter().enumerate() {
                let rel = |a: usize| fixed(lo[a] - corner[a], CORNER_OFFSET, BOX_SCALE);
                let (Some(bx), Some(by)) = (rel(0), rel(1)) else {
                    return Err(Unpacked::Fail(anyhow!(
                        "a glyph box sits past {CORNER_OFFSET} em from its grid cell"
                    )));
                };
                let last = if k + 1 == list.len() { LAST_GLYPH } else { 0 };
                cell_slots.push([u24u8(h | last, 0), u16x2(bx, by)]);
            }
            slots.push(cell_slots);
        }
    }
    Ok(Grid {
        gx,
        gy,
        cols,
        rows,
        cell,
        slots,
    })
}

/// Pack the glyphs, `nb` bands to an axis, `blocks` the data of each, and
/// the grid into texels, and lay the quads over a block `height` em tall.
fn pack(
    placed: &[Placed],
    glyphs: &HashMap<Ink, Glyph>,
    blocks: &HashMap<Ink, Arc<Block>>,
    nb: usize,
    height: f64,
) -> std::result::Result<Built, Unpacked> {
    // Texel 0 stays empty, so a slot that names a glyph is never all zero.
    let mut tex: Vec<[u8; 4]> = vec![[0; 4]];
    // Where the data of each ink's glyph starts.
    let mut starts: HashMap<Ink, usize> = HashMap::new();
    for p in placed {
        let Some(ink) = p.ink.and_then(Ink::curves) else {
            continue;
        };
        if starts.contains_key(&ink) {
            continue;
        }
        let block = blocks.get(&ink).ok_or(Unpacked::Band(p.ch))?;
        starts.insert(ink, block.put(&mut tex));
    }
    let inked = inked(placed, glyphs, |ink| starts[&ink]);
    let Grid {
        gx,
        gy,
        cols,
        rows,
        cell,
        slots,
    } = grid(&inked, tex.len())?;
    tex.resize(tex.len().next_multiple_of(CELL_K * SLOT), [0; 4]);
    let cells = tex.len();
    tex.resize(cells + cols * rows * CELL_K * SLOT, [0; 4]);
    for (i, list) in slots.iter().enumerate() {
        let at = cells + i * CELL_K * SLOT;
        for (k, slot) in list.iter().enumerate() {
            tex[at + k * SLOT..at + (k + 1) * SLOT].copy_from_slice(slot);
        }
    }

    let tex_h = tex_rows(tex.len());
    let mut texels = vec![0u8; TEX_W * tex_h * 4];
    texels[..tex.len() * 4].copy_from_slice(tex.as_flattened());
    Ok(Built {
        width: TEX_W as u32,
        height: tex_h as u32,
        quads: glyph_quads(&inked, false, nb, height),
        boxes: glyph_quads(&inked, true, nb, height),
        glyphs: Vec::new(),
        colour: Vec::new(),
        shared: false,
        texels,
        size: [0.0; 2],
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
    })
}

/// The quads over `glyphs` that read an atlas, the MTSDF atlas or the colour
/// atlas, each from its glyph's cell and the cell's texels per em, as `cell`
/// gives them, and an error box for each glyph that has none. Returns the
/// glyph quads, the error boxes, and the glyphs with no cell.
pub fn atlas_quads(
    glyphs: &[Placement],
    mut cell: impl FnMut(&Placement) -> Option<(Cell, f64)>,
) -> (Quads, Quads, Vec<&Placement>) {
    let (mut drawn, mut boxes) = (Vec::new(), Vec::new());
    let mut missing = Vec::new();
    for p in glyphs {
        let [ox, oy] = p.origin;
        let (a1, a2, out) = match cell(p) {
            Some((c, s)) => {
                let [x0, y0, x1, y1] = c.ink;
                // The texel coordinate of the box's top left: y runs up the
                // atlas and down the block.
                let t = [
                    f64::from(c.rect[0]) + c.pad,
                    f64::from(c.rect[1]) + c.pad + (y1 - y0) * s,
                ];
                (
                    [ox + x0, oy - y1, ox + x1, oy - y0],
                    [t[0], t[1], s, -s],
                    &mut drawn,
                )
            }
            None => {
                missing.push(p);
                let x = (p.span[0] + p.span[1] - ERROR_BOX[0]) / 2.0;
                (
                    [x, oy - ERROR_BOX[1], x + ERROR_BOX[0], oy],
                    [0.0; 4],
                    &mut boxes,
                )
            }
        };
        let a3 = [p.line as f64, p.span[0], p.span[1], 0.0];
        for corner in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] {
            let a0 = [corner[0], corner[1], p.index as f64, p.word as f64];
            out.extend([a0, a1, a2, a3].iter().flatten().map(|v| *v as f32));
        }
    }
    let quads = |floats: Vec<f32>| Quads {
        count: floats.len() / (4 * QUAD_FLOATS),
        floats,
        per_vertex: QUAD_FLOATS,
    };
    (quads(drawn), quads(boxes), missing)
}

/// The lines of the coverage reader of layer `name`, `<name>_cover`, with
/// the uniforms that it reads. A pass's preamble declares `<name>_size`
/// apart.
///
/// The coverage math ports SlugPixelShader.hlsl, Copyright 2017 Eric Lengyel,
/// MIT License (https://github.com/EricLengyel/Slug). GLSL ES 1.00 has no
/// integer textures, no bit operations and no derivatives, so the port reads
/// bytes, takes the root eligibility from comparisons, and takes the pixels
/// per em from its caller. The roots come from the stable form of the
/// quadratic formula, which also covers a curve that is a line.
pub fn reader_lines(name: &str) -> Vec<String> {
    let mut l = lines(CURVES, name);
    l.extend(lines(COVER, name));
    l
}

/// The lines of the reader of one glyph's curves, `ghee_<name>_outline`, with
/// the texture and the uniforms that it reads: what a slug layer's shader
/// takes of reader_lines().
pub fn curve_reader_lines(name: &str) -> Vec<String> {
    lines(CURVES, name)
}

/// The fragment shader of a layer's error boxes after its precision lines:
/// yellow and black hazard stripes over each box, opaque, as error_box()
/// lays them out.
pub fn box_fragment() -> Vec<String> {
    lines(BOX_FRAGMENT, "")
}

/// The GLSL names that layer `name` declares in every scene pass for an
/// author to read, `<name>_cover` and `<name>_size`. ghee's own names for the
/// layer start with ghee_, as internal() makes them.
pub fn names(name: &str) -> Vec<String> {
    let mut out = vec![format!("{name}_size")];
    for n in template_names(name) {
        if !n.starts_with(crate::scene::PREFIX) && !out.contains(&n) {
            out.push(n);
        }
    }
    out
}

/// ghee's name for the engine asset or the uniform `suffix` of layer `name`,
/// such as its texture of curves, `_curves`.
pub fn internal(name: &str, suffix: &str) -> String {
    format!("{}{name}{suffix}", crate::scene::PREFIX)
}

/// Every name made from NAME in the templates of the readers of layer
/// `name`, with the `<asset>_size` set by the engine beside each sampler
/// `<asset>_tex`.
pub(crate) fn template_names(name: &str) -> Vec<String> {
    let ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut out: Vec<String> = Vec::new();
    for t in [CURVES, COVER] {
        for (i, _) in t.match_indices("NAME_") {
            let (head, prefixed) = match t[..i].strip_suffix(crate::scene::PREFIX) {
                Some(head) => (head, true),
                None => (&t[..i], false),
            };
            if head.ends_with(ident) {
                continue;
            }
            let rest = &t[i + "NAME".len()..];
            let suffix = &rest[..rest.find(|c| !ident(c)).unwrap_or(rest.len())];
            let made = match prefixed {
                true => internal(name, suffix),
                false => format!("{name}{suffix}"),
            };
            if let Some(asset) = made
                .strip_suffix("_tex")
                .filter(|_| head.ends_with("uniform sampler2D "))
            {
                out.push(format!("{asset}_size"));
            }
            out.push(made);
        }
    }
    out
}

/// The lines of a GLSL template, with the constants of this module written in
/// for their tokens, then `name` for NAME.
fn lines(glsl: &str, name: &str) -> Vec<String> {
    let f = |v: f64| format!("{v:?}");
    // A token comes ahead of any shorter token inside it.
    let consts = [
        ("LATTICE_MID", f(LATTICE / 2.0)),
        ("LATTICE", f(LATTICE)),
        ("BAND_CURVES", BAND_CURVES.to_string()),
        ("CELL_TEXELS", f((CELL_K * SLOT) as f64)),
        ("CELL_K", CELL_K.to_string()),
        ("SLOT", f(SLOT as f64)),
        ("BOX_SCALE", f(BOX_SCALE)),
        ("CORNER_OFFSET", f(CORNER_OFFSET)),
        ("OFFSET_BIAS", f(f64::from(OFFSET_MAX + 1))),
        ("LAST_BIT", f((LAST_GLYPH >> 16) as f64)),
        ("ERROR_BOX_W", f(ERROR_BOX[0])),
        ("ERROR_BOX_H", f(ERROR_BOX[1])),
        ("STRIPE", f(STRIPE)),
    ];
    let mut glsl = glsl.to_string();
    for (token, value) in &consts {
        glsl = glsl.replace(token, value);
    }
    glsl.replace("NAME", name)
        .lines()
        .map(str::to_string)
        .collect()
}

const BOX_FRAGMENT: &str = r#"varying vec2 uv;
uniform vec3 iResolution;
varying vec4 ghee_v0;
varying vec4 ghee_v4;
void main() {
    // The sample moves to the pixel centre, as the slug fill's does.
    vec2 px = uv * iResolution.xy;
    vec2 d = floor(px) + 0.5 - px;
    vec2 u = ghee_v0.xy + vec2(dot(ghee_v4.xy, d), dot(ghee_v4.zw, d));
    // The box's coverage, antialiased over a pixel at each edge.
    vec2 edge = clamp(min(u, LATTICE - u) * ghee_v0.zw + 0.5, 0.0, 1.0);
    float a = edge.x * edge.y;
    // Yellow where (x - y) / the stripe width has an even floor, x and y in
    // em from the box's bottom left.
    vec2 box = vec2(ERROR_BOX_W, ERROR_BOX_H);
    vec2 em = u * box / LATTICE;
    float s = (em.x - em.y) / STRIPE - 0.5;
    float t = abs(fract(s * 0.5) * 2.0 - 1.0) - 0.5;
    // The em spanned by a pixel along each axis of the box.
    vec2 fw = box / (LATTICE * ghee_v0.zw);
    float yellow = clamp(t * STRIPE / (fw.x + fw.y) + 0.5, 0.0, 1.0);
    gl_FragColor = vec4(vec3(1.0, 0.8, 0.0) * (yellow * a), a);
}"#;

/// The reader of one glyph's curves.
const CURVES: &str = r#"uniform sampler2D ghee_NAME_curves_tex;
uniform float ghee_NAME_tw;
uniform float ghee_NAME_itw;
uniform float ghee_NAME_ith;
// A texture read on the Pi 5 returns a byte too coarse to take bits from as it
// is, so every byte goes through int() first: a read of byte b lies near
// b / 255, and int(t * 255.5) floors to b. On VC4, float(int(x)) takes 2
// instructions and floor(x) 4, and every floor here takes x >= 0.
vec4 ghee_NAME_texel(float i) {
    float row = float(int(i * ghee_NAME_itw));
    vec2 at = vec2(i - row * ghee_NAME_tw + 0.5, row + 0.5);
    return vec4(ivec4(texture2D(ghee_NAME_curves_tex, at * vec2(ghee_NAME_itw, ghee_NAME_ith)) * 255.5));
}
// The texture coordinate of a band's list: the first list for the ray toward
// +, the second for the ray toward -.
vec2 ghee_NAME_list(vec4 band, float dir) {
    float i = (band.r * 256.0 + band.g) * 256.0 + band.b + (dir < 0.0 ? band.a * 2.0 : 0.0);
    float row = float(int(i * ghee_NAME_itw));
    return vec2(i - row * ghee_NAME_tw + 0.5, row + 0.5) * vec2(ghee_NAME_itw, ghee_NAME_ith);
}
// One texel of a curve: a lattice point, and the offset of p2 from p3 on one
// axis.
vec3 ghee_NAME_lattice(vec4 t) {
    vec4 b = vec4(ivec4(t * 255.5));
    vec2 hi = vec2(ivec2(b.ba * (1.0 / 32.0)));
    return vec3(b.rg * 8.0 + hi, (b.b - hi.x * 32.0) * 32.0 + b.a - hi.y * 32.0 - OFFSET_BIAS);
}
// Where the curve crosses y = 0, as x. The roots of a t^2 - 2b t + c, each
// from the form of the quadratic formula that subtracts no near-equal values.
vec2 ghee_NAME_cross(vec2 p1, vec2 p2, vec2 p3) {
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
float ghee_NAME_outline(float h, vec2 nb, vec2 u, vec2 ppu) {
    vec2 m = 0.5 / ppu;
    float cov = 0.0;
    if (u.x > -m.x && u.y > -m.y && u.x < LATTICE + m.x && u.y < LATTICE + m.y) {
        vec2 band = min(vec2(ivec2(max(u.yx, 0.0) * (nb / LATTICE))), nb - 1.0);
        vec4 hb = ghee_NAME_texel(h + band.x);
        // Each ray runs toward the nearer edge of the box. A ray toward -
        // reads its band mirrored, so its coverage flips sign.
        float dx = u.x < LATTICE_MID ? -1.0 : 1.0;
        vec2 uh = vec2(dx > 0.0 ? u.x : LATTICE - u.x, u.y);
        float xcov = 0.0;
        float xwgt = 0.0;
        vec2 at = ghee_NAME_list(hb, dx);
        for (int i = 0; i < BAND_CURVES; i++) {
            if (float(i) >= hb.a) break;
            vec3 c1 = ghee_NAME_lattice(texture2D(ghee_NAME_curves_tex, at));
            vec3 c3 = ghee_NAME_lattice(texture2D(ghee_NAME_curves_tex, at + vec2(ghee_NAME_itw, 0.0)));
            at.x += ghee_NAME_itw * 2.0;
            vec2 p1 = c1.xy - uh;
            vec2 p3 = c3.xy - uh;
            vec2 p2 = p3 + vec2(c1.z, c3.z);
            // Sorted by their far end, largest first: a curve wholly behind
            // the pixel adds nothing, and neither does any after it.
            if (max(max(p1.x, p2.x), p3.x) * ppu.x < -0.5) break;
            // Which roots count, by the signs of p1.y, p2.y and p3.y: Slug's
            // lookup table 0x2E74, written out as logic.
            bool n1 = p1.y < 0.0;
            bool n2 = p2.y < 0.0;
            bool n3 = p3.y < 0.0;
            bool e1 = n3 ? !(n1 && n2) : n2 && !n1;
            bool e2 = n3 ? n1 && !n2 : n1 || n2;
            if (e1 || e2) {
                vec2 r = ghee_NAME_cross(p1, p2, p3) * ppu.x;
                xcov += e1 ? clamp(r.x + 0.5, 0.0, 1.0) : 0.0;
                xwgt = max(xwgt, e1 ? clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0) : 0.0);
                xcov -= e2 ? clamp(r.y + 0.5, 0.0, 1.0) : 0.0;
                xwgt = max(xwgt, e2 ? clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0) : 0.0);
            }
        }
        xcov *= dx;
        vec4 vb = ghee_NAME_texel(h + nb.x + band.y);
        float dy = u.y < LATTICE_MID ? -1.0 : 1.0;
        vec2 uf = vec2(u.x, dy > 0.0 ? u.y : LATTICE - u.y);
        float ycov = 0.0;
        float ywgt = 0.0;
        at = ghee_NAME_list(vb, dy);
        for (int i = 0; i < BAND_CURVES; i++) {
            if (float(i) >= vb.a) break;
            vec3 c1 = ghee_NAME_lattice(texture2D(ghee_NAME_curves_tex, at));
            vec3 c3 = ghee_NAME_lattice(texture2D(ghee_NAME_curves_tex, at + vec2(ghee_NAME_itw, 0.0)));
            at.x += ghee_NAME_itw * 2.0;
            vec2 p1 = (c1.xy - uf).yx;
            vec2 p3 = (c3.xy - uf).yx;
            vec2 p2 = p3 + vec2(c3.z, c1.z);
            if (max(max(p1.x, p2.x), p3.x) * ppu.y < -0.5) break;
            // Which roots count, by the signs of p1.y, p2.y and p3.y: Slug's
            // lookup table 0x2E74, written out as logic.
            bool n1 = p1.y < 0.0;
            bool n2 = p2.y < 0.0;
            bool n3 = p3.y < 0.0;
            bool e1 = n3 ? !(n1 && n2) : n2 && !n1;
            bool e2 = n3 ? n1 && !n2 : n1 || n2;
            if (e1 || e2) {
                vec2 r = ghee_NAME_cross(p1, p2, p3) * ppu.y;
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
}"#;

/// The reader of the words' coverage at any point, through the grid, after
/// CURVES.
const COVER: &str = r#"uniform float ghee_NAME_gx;
uniform float ghee_NAME_gy;
uniform float ghee_NAME_cols;
uniform float ghee_NAME_rows;
uniform float ghee_NAME_cell;
uniform float ghee_NAME_cells;
uniform float ghee_NAME_nb;
// The coverage of the glyph whose data starts at texel h, at q in em from its
// box's corner. Its data holds its box size, then its band entries.
float ghee_NAME_glyph(float h, vec2 q, vec2 sz, float font_size) {
    return ghee_NAME_outline(h + 1.0, vec2(ghee_NAME_nb), q / sz * LATTICE, sz * (font_size / LATTICE));
}
// The coverage of the words at p, in em from the top left of the block, y
// down like uv, drawn at font_size pixels per em.
float NAME_cover(vec2 p, float font_size) {
    float cov = 0.0;
    vec2 at = vec2(p.x, NAME_size.y - p.y);
    vec2 g = (at - vec2(ghee_NAME_gx, ghee_NAME_gy)) / ghee_NAME_cell;
    if (g.x >= 0.0 && g.y >= 0.0 && g.x < ghee_NAME_cols && g.y < ghee_NAME_rows) {
        vec2 c = vec2(ivec2(g));
        float cell = ghee_NAME_cells + (c.y * ghee_NAME_cols + c.x) * CELL_TEXELS;
        float row = float(int(cell * ghee_NAME_itw));
        vec2 slot = vec2(cell - row * ghee_NAME_tw + 0.5, row + 0.5) * vec2(ghee_NAME_itw, ghee_NAME_ith);
        vec2 pc = at - vec2(ghee_NAME_gx, ghee_NAME_gy) - c * ghee_NAME_cell;
        // An empty slot, or the one marked last, ends the cell.
        for (int k = 0; k < CELL_K; k++) {
            vec4 a = vec4(ivec4(texture2D(ghee_NAME_curves_tex, slot) * 255.5));
            if (a.r + a.g + a.b < 0.5) break;
            vec4 b = vec4(ivec4(texture2D(ghee_NAME_curves_tex, slot + vec2(ghee_NAME_itw, 0.0)) * 255.5));
            slot.x += ghee_NAME_itw * SLOT;
            float last = a.r >= LAST_BIT ? 1.0 : 0.0;
            float h = ((a.r - last * LAST_BIT) * 256.0 + a.g) * 256.0 + a.b;
            vec2 lo = (b.rb * 256.0 + b.ga) * (1.0 / BOX_SCALE) - CORNER_OFFSET;
            vec4 s = ghee_NAME_texel(h);
            vec2 sz = (s.rb * 256.0 + s.ga) * (1.0 / BOX_SCALE);
            cov = max(cov, ghee_NAME_glyph(h, pc - lo, sz, font_size));
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
        let faces = [Some(Face::parse(&data, 0).unwrap())];
        let (placed, width, _) = layout(&faces, words, align);
        (placed, width)
    }

    fn fields(p: &[Placed]) -> Vec<(usize, usize, usize)> {
        p.iter().map(|p| (p.index, p.word, p.line)).collect()
    }

    /// A glyph of `curves` vertical lines, each across the whole box.
    fn lines_glyph(curves: usize) -> Glyph {
        let step = LATTICE as usize / (curves + 1);
        Glyph {
            min: [0.0, 0.0],
            size: [0.5, 0.5],
            curves: (1..=curves)
                .map(|i| {
                    let x = (i * step) as u16;
                    [[x, 0], [x, 1023], [x, LATTICE as u16]]
                })
                .collect(),
        }
    }

    /// The data of each of `glyphs` at BANDS bands, where it has any.
    fn blocks(glyphs: &HashMap<Ink, Glyph>) -> HashMap<Ink, Arc<Block>> {
        glyphs
            .iter()
            .filter_map(|(ink, g)| Some((*ink, Arc::new(Block::of(g, BANDS)?))))
            .collect()
    }

    fn at_origin(gid: u16, ch: char) -> Placed {
        Placed {
            ink: Some(Ink::Glyph(0, gid)),
            ch,
            x: 0.0,
            y: 0.0,
            index: 0,
            word: 0,
            line: 0,
            span: [0.0, 0.5],
        }
    }

    #[test]
    fn a_band_past_its_loop_bound_names_the_character() {
        let glyphs = HashMap::from([(Ink::Glyph(0, 7), lines_glyph(BAND_CURVES + 1))]);
        let got = pack(&[at_origin(7, 'x')], &glyphs, &blocks(&glyphs), BANDS, 1.0);
        assert!(matches!(got, Err(Unpacked::Band('x'))));
        let e = pack_fitting(&[at_origin(7, 'x')], &glyphs, &blocks(&glyphs), 1.0)
            .err()
            .unwrap();
        assert_eq!(e.to_string(), "'x' has more than 64 curves in one band");
    }

    #[test]
    fn a_glyph_past_the_band_bound_takes_an_error_box_and_the_rest_draw() {
        let mut placed = [at_origin(7, 'x'), at_origin(8, 'y'), at_origin(7, 'x')];
        placed[1].span = [0.5, 1.0];
        placed[2].span = [1.0, 1.5];
        let (glyphs, faults) = settle(
            &mut placed,
            true,
            |_, gid| {
                Ok(Some(lines_glyph(if gid == 7 {
                    BAND_CURVES + 1
                } else {
                    1
                })))
            },
            |_, _| false,
        );
        assert_eq!(
            faults,
            [Fault::Glyph(
                'x',
                "has more than 64 curves in one band".to_string()
            )]
        );
        let inks: Vec<Option<Ink>> = placed.iter().map(|p| p.ink).collect();
        assert_eq!(
            inks,
            [Some(Ink::Box), Some(Ink::Glyph(0, 8)), Some(Ink::Box)]
        );
        // A box stands centred in its advance.
        assert_eq!(placed[2].x, 1.25 - ERROR_BOX[0] / 2.0);
        let built = pack_fitting(&placed, &glyphs, &blocks(&glyphs), 1.0).unwrap();
        assert_eq!((built.quads.count, built.boxes.count), (1, 2));
    }

    #[test]
    fn a_glyph_with_no_outline_takes_a_box_only_where_its_character_needs_ink() {
        let chars = [
            'a',
            ' ',
            '\u{a0}',
            '\u{200d}',
            '\u{fe0f}',
            '\u{e0101}',
            '😀',
        ];
        let mut placed: Vec<Placed> = chars.iter().map(|c| at_origin(3, *c)).collect();
        let (glyphs, faults) = settle(&mut placed, true, |_, _| Ok(None), |_, _| false);
        let boxed: Vec<char> = placed
            .iter()
            .filter(|p| p.ink == Some(Ink::Box))
            .map(|p| p.ch)
            .collect();
        assert_eq!(boxed, ['a', '😀']);
        assert!(placed.iter().all(|p| p.ink != Some(Ink::Glyph(0, 3))));
        assert_eq!(faults, [Fault::NoOutline(0, vec!['a', '😀'])]);
        assert_eq!(glyphs.keys().collect::<Vec<_>>(), [&Ink::Box]);
    }

    #[test]
    fn a_font_that_failed_to_load_takes_a_box_a_character() {
        let (placed, width, height) = layout(&[None], "ab c\u{200b}d\ne", Align::Left);
        let inks: Vec<(char, bool)> = placed
            .iter()
            .map(|p| (p.ch, p.ink == Some(Ink::Box)))
            .collect();
        assert_eq!(
            inks,
            [
                ('a', true),
                ('b', true),
                (' ', false),
                ('c', true),
                ('\u{200b}', false),
                ('d', true),
                ('e', true)
            ]
        );
        assert_eq!((width, height), (5.0 * ERROR_ADVANCE, 2.0 * ERROR_LINE));
        // The zero-width space takes no room.
        assert_eq!(placed[5].span, [4.0 * ERROR_ADVANCE, 5.0 * ERROR_ADVANCE]);
        assert_eq!(placed[6].y, ERROR_LINE - ERROR_ASCENT);
    }

    #[test]
    fn the_error_box_fills_its_box_with_stripes_one_loop_reads() {
        let g = error_box();
        assert_eq!((g.min, g.size), ([0.0; 2], ERROR_BOX));
        // Stripes reach every edge of the box.
        let top = LATTICE as u16;
        for (axis, at) in [(0, 0), (0, top), (1, 0), (1, top)] {
            assert!(
                g.curves.iter().flatten().any(|p| p[axis] == at),
                "axis {axis} at {at}"
            );
        }
        for axis in [0, 1] {
            assert!(bands(&g.curves, axis, BANDS, BAND_EPS).is_some());
        }
    }

    #[test]
    fn an_overlap_at_fewer_bands_reports_the_overlap() {
        // The glyphs outgrow the texture at 64 bands and fit it at 32, all
        // at one spot.
        let n = 20_000;
        let glyphs: HashMap<Ink, Glyph> =
            (0..n).map(|g| (Ink::Glyph(0, g), lines_glyph(1))).collect();
        let placed: Vec<Placed> = (0..n).map(|g| at_origin(g, 'x')).collect();
        let at = blocks(&glyphs);
        assert!(matches!(
            pack(&placed, &glyphs, &at, BANDS, 1.0),
            Err(Unpacked::TooTall(_))
        ));
        let e = pack_fitting(&placed, &glyphs, &at, 1.0).err().unwrap();
        assert_eq!(e.to_string(), "more than 4 glyphs overlap at one spot");
    }

    #[test]
    fn the_shared_texture_takes_the_words_that_a_texture_of_their_own_takes_at_every_band() {
        // Each row places glyphs at one spot or in a row, and fits() agrees
        // with a pack at BANDS bands.
        let in_row = |n: u16| -> Vec<Placed> {
            (0..n)
                .map(|g| {
                    let mut p = at_origin(g % 50, 'x');
                    p.x = f64::from(g) * 0.6;
                    p
                })
                .collect()
        };
        let rows: [(&str, Vec<Placed>); 4] = [
            ("one glyph", vec![at_origin(0, 'x')]),
            (
                "four at one spot",
                (0..4).map(|g| at_origin(g, 'x')).collect(),
            ),
            (
                "five at one spot",
                (0..5).map(|g| at_origin(g, 'x')).collect(),
            ),
            ("past the texture", in_row(60_000)),
        ];
        for (what, placed) in rows {
            let glyphs: HashMap<Ink, Glyph> = placed
                .iter()
                .filter_map(|p| p.ink)
                .map(|ink| (ink, lines_glyph(1)))
                .collect();
            let at = blocks(&glyphs);
            let packs = pack(&placed, &glyphs, &at, BANDS, 1.0).is_ok();
            assert_eq!(fits(&placed, &glyphs, &at), packs, "{what}");
        }
    }

    #[test]
    fn a_font_read_again_takes_the_glyphs_that_its_first_read_packed() {
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tools/fixtures/fonts/DejaVuSans-ASCII.ttf"
        ))
        .unwrap();
        let mut g = Glyphs::default();
        let first: Vec<Font> = vec![Ok(Arc::new(bytes.clone()))];
        let again: Vec<Font> = vec![Ok(Arc::new(bytes))];
        let shared = |laid: Layout| laid.built.is_some_and(|b| b.shared);
        assert!(shared(g.lay(&first, "AV", Align::Left, Curves::Shared)));
        let (len, reads) = (g.tex.len(), g.reads);
        assert!(shared(g.lay(&again, "VA", Align::Left, Curves::Shared)));
        assert_eq!((g.tex.len(), g.reads), (len, reads));
    }

    #[test]
    fn words_laid_out_before_take_that_layout_in_fonts_of_its_content_alone() {
        let sans = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tools/fixtures/fonts/DejaVuSans-ASCII.ttf"
        ))
        .unwrap();
        let font = |bytes: &[u8]| -> Vec<Font> { vec![Ok(Arc::new(bytes.to_vec()))] };
        let (a, b) = (font(&sans), font(PIXEL_FONT));
        let mut g = Glyphs::default();
        // (fonts, align, texture, true when the layout is made again)
        let rows = [
            (&a, Align::Left, Curves::Shared, true),
            (&font(&sans), Align::Left, Curves::Shared, false),
            (&a, Align::Center, Curves::Shared, true),
            (&b, Align::Left, Curves::Shared, true),
            (&a, Align::Left, Curves::Own, true),
            (&a, Align::Left, Curves::None, true),
            (&a, Align::Center, Curves::Shared, false),
        ];
        for (k, (fonts, align, curves, made)) in rows.into_iter().enumerate() {
            let lays = g.lays;
            let laid = g.lay(fonts, "AV\nTo", align, curves);
            assert_eq!(g.lays - lays, usize::from(made), "row {k}");
            assert!(laid == g.lay_now(fonts, "AV\nTo", align, curves), "row {k}");
        }
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
