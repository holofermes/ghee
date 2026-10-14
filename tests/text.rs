//! Text layers: the layout, the scene model, the font assets, and the
//! coverage that passes and layers render on this host's GPU.
//!
//! The reference below rebuilds the promised layout and outline from the font
//! with ttf-parser, so every check holds against any packing of the texture.

use std::fs;
use std::sync::{Arc, Mutex, MutexGuard};

use serde_json::{json, Value};
use ttf_parser::{Face, GlyphId, OutlineBuilder};

use ghee::engine::{probe_dri_devices, Given, Renderer};
use ghee::filters::apply_filters;
use ghee::layer::{Address, Detail, Layer, TextLayer};
use ghee::scene::{Param, ParamValue, Pass, Scene, SceneBuilder};
use ghee::text::{self, Align};

const FONTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts");
/// DejaVu Sans cut to printable ASCII, TrueType outlines, a kern table.
const SANS: &str = "DejaVuSans-ASCII.ttf";
/// Inter 4.001 cut by fontTools' subset to A, V, T, o, 7 and 5 and the GPOS
/// `kern` feature: a class pair adjustment in an extension lookup, then a
/// lookup of a glyph pair adjustment and a class pair adjustment. Inter has
/// no `kern` table.
const GPOS_KERN: &str = "Inter-AVTo75.ttf";
/// Noto Sans CJK JP cut to I and O, CFF outlines.
const CFF: &str = "NotoSansCJKjp-IO.otf";
/// CFF converted to CFF2 by fontTools' cffLib.CFFToCFF2, with no variation
/// store.
const CFF2: &str = "NotoSansCJKjp-IO-CFF2.otf";
/// Noto Color Emoji cut by fontTools' subset to space, U+200D, U+1F600 and
/// U+1F642: colour bitmaps, no outlines.
const EMOJI: &str = "NotoColorEmoji-Smiles.ttf";
/// EMOJI rebuilt by fontTools' FontBuilder with its two bitmaps in an sbix
/// table at 109 px/em, beside a strike at 20 px/em of flat green, and empty
/// TrueType outlines.
const EMOJI_SBIX: &str = "NotoColorEmoji-Smiles-sbix.ttf";
/// DejaVu Sans cut to Latin, the font of a text that names none, as
/// ghee/examples/fonts holds it.
const LATIN: &str = "../../../ghee/examples/fonts/DejaVuSans-Latin.ttf";
/// Pixel Operator 8, the pixel font that ghee carries, as ghee/examples/fonts
/// holds it.
const PIXEL: &str = "../../../ghee/examples/fonts/PixelOperator8.ttf";

fn font(file: &str) -> Vec<u8> {
    fs::read(format!("{FONTS}/{file}")).unwrap()
}

/// The fonts of a text of `bytes` alone.
fn alone(bytes: &[u8]) -> Vec<text::Font> {
    vec![Ok(Arc::new(bytes.to_vec()))]
}

/// A Renderer opens a GL device, and load_assets reads $GHEE_ASSETS. A test
/// that does either holds this, so no variable changes under a device or a
/// lookup.
static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A w x h RGBA renderer on the first GL device that opens. Its atlases cache
/// under the temp directory.
fn gpu(w: usize, h: usize) -> (MutexGuard<'static, ()>, Renderer) {
    let guard = lock();
    std::env::set_var(
        "XDG_CACHE_HOME",
        std::env::temp_dir().join("ghee-text-cache"),
    );
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

fn tmpdir(tag: &str) -> String {
    let d = std::env::temp_dir().join(format!("ghee-text-{}-{tag}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d.to_string_lossy().into_owned()
}

// The reference layout.

/// The layout a text promises: lines top to bottom one line height apart
/// (ascent - descent + line gap), the first baseline one ascent below the top,
/// glyphs advanced and kerned by the kern table, each line shifted inside the
/// widest by its alignment. A character the font lacks draws glyph 0, a tab
/// draws a space, any other control character draws nothing.
struct Laid {
    width: f64,
    height: f64,
    glyphs: Vec<Outline>,
}

/// One glyph's outline as straight segments, their box, and the glyph's
/// pen on its baseline, in em from the block's top left, y down.
struct Outline {
    origin: [f64; 2],
    lo: [f64; 2],
    hi: [f64; 2],
    segs: Vec<[[f64; 2]; 2]>,
}

/// Steps per curve. At 100 pixels per em a step strays well under 1/10 px.
const STEPS: usize = 16;

struct Flat {
    origin: [f64; 2],
    upem: f64,
    at: [f64; 2],
    start: [f64; 2],
    segs: Vec<[[f64; 2]; 2]>,
}

impl Flat {
    fn point(&self, x: f32, y: f32) -> [f64; 2] {
        [
            self.origin[0] + f64::from(x) / self.upem,
            self.origin[1] - f64::from(y) / self.upem,
        ]
    }

    fn to(&mut self, p: [f64; 2]) {
        self.segs.push([self.at, p]);
        self.at = p;
    }

    fn curve(&mut self, f: impl Fn(f64) -> [f64; 2]) {
        for i in 1..=STEPS {
            self.to(f(i as f64 / STEPS as f64));
        }
    }
}

impl OutlineBuilder for Flat {
    fn move_to(&mut self, x: f32, y: f32) {
        self.at = self.point(x, y);
        self.start = self.at;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.point(x, y);
        self.to(p);
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let (p0, p1, p2) = (self.at, self.point(x1, y1), self.point(x, y));
        self.curve(|t| {
            let u = 1.0 - t;
            let b = [u * u, 2.0 * u * t, t * t];
            [0, 1].map(|a| b[0] * p0[a] + b[1] * p1[a] + b[2] * p2[a])
        });
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let (p0, p1, p2, p3) = (
            self.at,
            self.point(x1, y1),
            self.point(x2, y2),
            self.point(x, y),
        );
        self.curve(|t| {
            let u = 1.0 - t;
            let b = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
            [0, 1].map(|a| b[0] * p0[a] + b[1] * p1[a] + b[2] * p2[a] + b[3] * p3[a])
        });
    }

    fn close(&mut self) {
        if self.at != self.start {
            let s = self.start;
            self.to(s);
        }
    }
}

fn kerning(face: &Face, l: GlyphId, r: GlyphId) -> f64 {
    face.tables().kern.map_or(0.0, |k| {
        k.subtables
            .into_iter()
            .filter(|s| s.horizontal && !s.variable && !s.has_cross_stream)
            .filter_map(|s| s.glyphs_kerning(l, r))
            .map(f64::from)
            .sum()
    })
}

fn lay(file: &str, words: &str, align: Align) -> Laid {
    let data = font(file);
    let face = Face::parse(&data, 0).unwrap();
    let upem = f64::from(face.units_per_em());
    let ascent = f64::from(face.ascender()) / upem;
    let line = (f64::from(face.ascender()) - f64::from(face.descender())
        + f64::from(face.line_gap()))
        / upem;
    let mut rows = Vec::new();
    for text in words.split('\n') {
        let (mut x, mut prev, mut glyphs) = (0.0, None, Vec::new());
        for ch in text.chars().map(|c| if c == '\t' { ' ' } else { c }) {
            if ch.is_control() {
                continue;
            }
            let g = face.glyph_index(ch).unwrap_or(GlyphId(0));
            if let Some(p) = prev {
                x += kerning(&face, p, g) / upem;
            }
            glyphs.push((g, x));
            x += f64::from(face.glyph_hor_advance(g).unwrap()) / upem;
            prev = Some(g);
        }
        rows.push((glyphs, x));
    }
    let width = rows.iter().map(|r| r.1).fold(0.0, f64::max);
    let mut outlines = Vec::new();
    for (k, (glyphs, w)) in rows.iter().enumerate() {
        let shift = match align {
            Align::Left => 0.0,
            Align::Center => (width - w) / 2.0,
            Align::Right => width - w,
        };
        for (g, x) in glyphs {
            let mut f = Flat {
                origin: [x + shift, ascent + line * k as f64],
                upem,
                at: [0.0; 2],
                start: [0.0; 2],
                segs: Vec::new(),
            };
            face.outline_glyph(*g, &mut f);
            let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
            for q in f.segs.iter().flatten() {
                for a in 0..2 {
                    lo[a] = lo[a].min(q[a]);
                    hi[a] = hi[a].max(q[a]);
                }
            }
            outlines.push(Outline {
                origin: f.origin,
                lo,
                hi,
                segs: f.segs,
            });
        }
    }
    Laid {
        width,
        height: line * rows.len() as f64,
        glyphs: outlines,
    }
}

impl Laid {
    /// The distance from `p` to the outline in em, below 0 inside it.
    fn distance(&self, p: [f64; 2]) -> f64 {
        let mut d = f64::MAX;
        for [a, b] in self.glyphs.iter().flat_map(|g| &g.segs) {
            let v = [b[0] - a[0], b[1] - a[1]];
            let len2 = v[0] * v[0] + v[1] * v[1];
            let t = if len2 > 0.0 {
                (((p[0] - a[0]) * v[0] + (p[1] - a[1]) * v[1]) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            d = d.min((p[0] - a[0] - t * v[0]).hypot(p[1] - a[1] - t * v[1]));
        }
        match self.inside(p, 0.0) {
            Some(true) => -d,
            _ => d,
        }
    }

    /// Whether `p` lies inside the outline, by nonzero winding, the fill rule
    /// of TrueType and CFF alike. None within `edge` em of an edge.
    fn inside(&self, p: [f64; 2], edge: f64) -> Option<bool> {
        let mut winding = 0;
        for g in &self.glyphs {
            // A point outside a closed outline's box winds zero times round it.
            if (0..2).any(|a| p[a] < g.lo[a] - edge || p[a] > g.hi[a] + edge) {
                continue;
            }
            for [a, b] in &g.segs {
                let d = [b[0] - a[0], b[1] - a[1]];
                let len2 = d[0] * d[0] + d[1] * d[1];
                let t = if len2 > 0.0 {
                    (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                if (p[0] - a[0] - t * d[0]).hypot(p[1] - a[1] - t * d[1]) <= edge {
                    return None;
                }
                if (a[1] <= p[1]) != (b[1] <= p[1]) {
                    let t = (p[1] - a[1]) / (b[1] - a[1]);
                    if a[0] + t * (b[0] - a[0]) > p[0] {
                        winding += if b[1] > a[1] { 1 } else { -1 };
                    }
                }
            }
        }
        Some(winding != 0)
    }
}

impl Laid {
    /// The layout in pixels at `px` pixels per em, its top left at the pixel
    /// `top`, each glyph's pen and baseline moved to the nearest whole pixel.
    fn on_pixels(&self, top: [f64; 2], px: f64) -> Laid {
        let glyphs = self
            .glyphs
            .iter()
            .map(|g| {
                let pen = [0, 1].map(|a| top[a] + (g.origin[a] * px + 0.5).floor());
                let at = |p: [f64; 2]| [0, 1].map(|a| pen[a] + (p[a] - g.origin[a]) * px);
                Outline {
                    origin: pen,
                    lo: at(g.lo),
                    hi: at(g.hi),
                    segs: g.segs.iter().map(|s| s.map(at)).collect(),
                }
            })
            .collect();
        Laid {
            width: self.width * px,
            height: self.height * px,
            glyphs,
        }
    }
}

/// Where the block sits in the frame, in pixels: its top left corner and the
/// pixels per em. Every value is a multiple of 1/16, so the GLSL literal holds
/// it exactly.
#[derive(Clone, Copy)]
struct Fit {
    x: f64,
    y: f64,
    ppem: f64,
}

/// Frame margin, in pixels.
const MARGIN: f64 = 8.0;

impl Fit {
    /// A width x height em block centred in a w x h frame, as large as fits.
    fn of(width: f64, height: f64, w: usize, h: usize) -> Fit {
        let snap = |v: f64| (v * 16.0).floor() / 16.0;
        let (w, h) = (w as f64, h as f64);
        let ppem = snap(((w - 2.0 * MARGIN) / width).min((h - 2.0 * MARGIN) / height));
        Fit {
            x: snap((w - width * ppem) / 2.0),
            y: snap((h - height * ppem) / 2.0),
            ppem,
        }
    }

    /// The GLSL that takes `name`'s coverage at this pixel.
    fn cover(&self, name: &str) -> String {
        format!(
            "{name}_cover((uv * iResolution.xy - vec2({:.4}, {:.4})) / {:.4}, {:.4})",
            self.x, self.y, self.ppem, self.ppem
        )
    }

    /// The em point at the centre of pixel (x, y), y counted from the top.
    fn em(&self, x: usize, y: usize) -> [f64; 2] {
        [
            (x as f64 + 0.5 - self.x) / self.ppem,
            (y as f64 + 0.5 - self.y) / self.ppem,
        ]
    }
}

/// The pixels that a layer in the font `file` at `px` pixels per em draws
/// its first baseline below the em of its layout's ascent: the ascender as
/// FreeType's size metrics round it for Pillow's default anchor, less the
/// exact one. Pillow requests the size in 26.6 pixels, cut from a float;
/// FreeType takes the scale with FT_DivFix, the ascender with FT_MulFix,
/// then rounds it up to a whole pixel. A font that does not load lays out
/// with an ascender of one em.
fn rounding(file: &str, px: f64) -> f64 {
    let (a, u) = match fs::read(format!("{FONTS}/{file}")) {
        Ok(data) => {
            let face = Face::parse(&data, 0).unwrap();
            (i64::from(face.ascender()), i64::from(face.units_per_em()))
        }
        Err(_) => (1, 1),
    };
    let size = (px as f32 * 64.0) as i64;
    let scale = ((size << 16) + (u >> 1)) / u;
    let ascender = (a * scale + 0x8000) >> 16;
    ((ascender + 63) >> 6) as f64 - a as f64 / u as f64 * px
}

/// A pixel centre further than this from every edge, in pixels, takes full
/// coverage or none. The antialiasing owns the band along the edges.
const EDGE: f64 = 1.5;
/// Off by at most this, of 255.
const TOL: u8 = 4;

/// Channel `c` of an RGBA `frame` w pixels wide against the reference: full
/// inside the outline, none outside it, away from the edges.
fn assert_draws(frame: &[u8], w: usize, c: usize, laid: &Laid, fit: Fit, what: &str) {
    assert_draws_at(
        frame,
        w,
        c,
        laid,
        |x, y| fit.em(x, y),
        EDGE / fit.ppem,
        what,
    );
}

/// assert_draws where `em` maps pixel (x, y) into the block and `edge` is
/// EDGE in em.
fn assert_draws_at(
    frame: &[u8],
    w: usize,
    c: usize,
    laid: &Laid,
    em: impl Fn(usize, usize) -> [f64; 2],
    edge: f64,
    what: &str,
) {
    let h = frame.len() / 4 / w;
    let (mut inside, mut outside, mut off) = (0, 0, Vec::new());
    for y in 0..h {
        for x in 0..w {
            let Some(is_in) = laid.inside(em(x, y), edge) else {
                continue;
            };
            let want = if is_in {
                inside += 1;
                255
            } else {
                outside += 1;
                0
            };
            let got = frame[(y * w + x) * 4 + c];
            if got.abs_diff(want) > TOL {
                off.push((x, y, got, want));
            }
        }
    }
    assert!(
        inside > 0 && outside > 0,
        "{what}: {inside} pixels inside, {outside} outside"
    );
    assert!(
        off.is_empty(),
        "{what}: {} of {} pixels off, (x, y, got, want) {:?}",
        off.len(),
        inside + outside,
        &off[..off.len().min(8)]
    );
}

fn blank(frame: &[u8], c: usize) -> bool {
    frame.iter().skip(c).step_by(4).all(|v| *v == 0)
}

fn align_name(a: Align) -> &'static str {
    match a {
        Align::Left => "left",
        Align::Center => "center",
        Align::Right => "right",
    }
}

/// One pass that draws the coverage of the hidden text layer `label`, in the
/// font asset `face`, the fixture `file`, to red, green and blue.
fn label_scene(file: &str, words: &str, align: Align, fit: Fit) -> Value {
    json!({
        "version": 1,
        "assets": {"face": {"file": file}},
        "passes": [{"name": "main", "fragment": format!(
            "void main(){{ float c = {}; gl_FragColor = vec4(c, c, c, 1.0); }}",
            fit.cover("label")),
            "layers": [{"name": "label", "text": words, "font_family": "face",
                        "text_align": align_name(align), "visible": false}]}],
    })
}

/// The layer of `doc` named `name`.
fn layer_of<'a>(doc: &'a mut Value, name: &str) -> &'a mut Value {
    doc["passes"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .flat_map(|p| p["layers"].as_array_mut().into_iter().flatten())
        .find(|l| l["name"] == name)
        .unwrap()
}

/// The scene with its fonts read from the fixtures. The caller holds LOCK.
fn loaded(doc: &Value) -> Scene {
    let mut sc = Scene::from_dict(doc, None, None).unwrap();
    sc.load_assets(FONTS).unwrap();
    sc
}

/// The live `text` of layer `layer`.
fn text_of(layer: &str) -> Address<'_> {
    Address {
        layer,
        effect: None,
        field: "text",
    }
}

/// Give text layer `layer` of `r` the words `words` through its live `text`.
fn set_words(r: &Renderer, layer: &str, words: &str) {
    r.set_layer(&text_of(layer), Given::Text(words)).unwrap();
}

fn render(r: &mut Renderer, sc: &Scene) -> Vec<u8> {
    assert_eq!(r.setup(sc).unwrap(), "");
    r.set_time(Some(0.0));
    r.render(None).unwrap().to_vec()
}

// text::build

/// The block size of a built text, in em.
fn block(built: &text::Built) -> (f64, f64) {
    (built.size[0], built.size[1])
}

/// The layout of `words` in a fixture font that draws every one of them.
fn drawn(file: &str, words: &str, align: Align) -> text::Built {
    let laid = text::build(&alone(&font(file)), words, align);
    assert_eq!(laid.faults, [], "{words:?}");
    laid.built.unwrap()
}

fn built(file: &str, words: &str, align: Align) -> (f64, f64) {
    block(&drawn(file, words, align))
}

fn near(got: (f64, f64), want: (f64, f64)) -> bool {
    (got.0 - want.0).abs() < 1e-9 && (got.1 - want.1).abs() < 1e-9
}

#[test]
fn a_block_spans_the_widest_line_and_one_line_height_per_line() {
    for align in [Align::Left, Align::Center, Align::Right] {
        for words in ["I", "AV To\nghee", "ghee\nI", "I\n"] {
            let want = lay(SANS, words, align);
            let got = built(SANS, words, align);
            assert!(near(got, (want.width, want.height)), "{words:?}: {got:?}");
        }
    }
    let one = built(SANS, "ghee", Align::Left);
    let two = built(SANS, "ghee\nI", Align::Right);
    assert_eq!(one.0, two.0); // The narrower line widens nothing.
    assert!((two.1 - 2.0 * one.1).abs() < 1e-9);
}

#[test]
fn a_kerned_pair_draws_closer_than_its_advances() {
    let data = font(SANS);
    let face = Face::parse(&data, 0).unwrap();
    let upem = f64::from(face.units_per_em());
    let (a, v) = (
        face.glyph_index('A').unwrap(),
        face.glyph_index('V').unwrap(),
    );
    let kern = kerning(&face, a, v);
    assert!(kern < 0.0); // DejaVu Sans kerns AV.
    let advances = f64::from(face.glyph_hor_advance(a).unwrap())
        + f64::from(face.glyph_hor_advance(v).unwrap());
    let (w, _) = built(SANS, "AV", Align::Left);
    assert!((w - (advances + kern) / upem).abs() < 1e-9, "{w}");
}

#[test]
fn a_font_that_kerns_through_gpos_alone_kerns_each_pair_as_harfbuzz_does() {
    let data = font(GPOS_KERN);
    let face = Face::parse(&data, 0).unwrap();
    assert!(face.tables().kern.is_none());
    let upem = f64::from(face.units_per_em());
    // HarfBuzz's kerning of each pair, in font units. AV and To kern through
    // the extension lookup, 75 through the glyph pair adjustment, and 77,
    // which the glyph pair adjustment lacks, through the class pair
    // adjustment after it.
    for (pair, kern) in [("AV", -140.0), ("To", -160.0), ("75", -20.0), ("77", 40.0)] {
        let advances: f64 = pair
            .chars()
            .map(|c| {
                f64::from(
                    face.glyph_hor_advance(face.glyph_index(c).unwrap())
                        .unwrap(),
                )
            })
            .sum();
        let (w, _) = built(GPOS_KERN, pair, Align::Left);
        assert!((w - (advances + kern) / upem).abs() < 1e-9, "{pair}: {w}");
    }
}

#[test]
fn a_character_missing_from_the_font_takes_glyph_0() {
    let data = font(SANS);
    let face = Face::parse(&data, 0).unwrap();
    assert!(face.glyph_index('字').is_none());
    let notdef = f64::from(face.glyph_hor_advance(GlyphId(0)).unwrap());
    let (w, _) = built(SANS, "字", Align::Left);
    assert!((w - notdef / f64::from(face.units_per_em())).abs() < 1e-9);
}

#[test]
fn a_tab_is_a_space_and_other_control_characters_take_no_room() {
    let (tab, space) = (
        built(SANS, "a\tb", Align::Left),
        built(SANS, "a b", Align::Left),
    );
    assert!(near(tab, space));
    assert!(near(
        built(SANS, "a\u{7}\rb", Align::Left),
        built(SANS, "ab", Align::Left)
    ));
}

#[test]
fn empty_words_are_one_empty_line() {
    let line = lay(SANS, "I", Align::Left).height;
    assert!(near(built(SANS, "", Align::Left), (0.0, line)));
    assert!(near(built(SANS, "\n", Align::Left), (0.0, 2.0 * line)));
}

#[test]
fn a_cff_font_lays_out_like_a_truetype_one() {
    let want = lay(CFF, "IO\nO", Align::Center);
    assert!(near(
        built(CFF, "IO\nO", Align::Center),
        (want.width, want.height)
    ));
}

#[test]
fn bytes_that_are_no_font_draw_a_box_a_character() {
    let laid = text::build(&alone(b"not a font"), "I O", Align::Left);
    assert!(
        matches!(&laid.faults[..], [text::Fault::Font(why)] if why.starts_with("unreadable font")),
        "{:?}",
        laid.faults
    );
    let built = laid.built.unwrap();
    assert_eq!((built.quads.count, built.boxes.count), (0, 2));
    let mut l = face_layer();
    l.data = Some(vec![Err("asset 'face' (face.ttf): gone".to_string())]);
    let laid = l.build("I O").unwrap();
    assert_eq!(
        laid.faults,
        [text::Fault::Font(
            "asset 'face' (face.ttf): gone".to_string()
        )]
    );
    assert_eq!(laid.built.unwrap().boxes.count, 2);
}

/// A layer `label` whose font_family names the font asset `face`, before any
/// font loads.
fn face_layer() -> TextLayer {
    let d = json!({"name": "label", "text": "", "font_family": "face"});
    TextLayer::from_dict("a layer", &d).unwrap()
}

#[test]
fn a_layer_built_before_its_font_loads_names_the_font() {
    let e = face_layer().build("I").err().unwrap().to_string();
    assert_eq!(
        e,
        "layer 'label': font 'face' was never loaded, the scene was built without a directory"
    );
}

#[test]
fn a_layer_with_no_font_family_lays_out_in_dejavu_sans_latin() {
    let doc = layers_scene(json!([{"name": "label", "text": "I"}]));
    // DejaVu Sans needs no directory, and a scene's fonts read from one
    // leave it in place.
    let bare = Scene::from_dict(&doc, None, None).unwrap();
    let _g = lock();
    for sc in [bare, loaded(&doc)] {
        // é and ß lie past ASCII, and AV kerns.
        let words = "Hé ßAV\nI";
        let laid = sc.passes[0].layers[0]
            .as_text()
            .unwrap()
            .build(words)
            .unwrap();
        assert_eq!(laid.faults, []);
        let want = lay(LATIN, words, Align::Left);
        assert!(near(
            block(laid.built.as_ref().unwrap()),
            (want.width, want.height)
        ));
        assert_eq!(
            sc.to_dict()["passes"][0]["layers"],
            json!([{"name": "label", "text": "I"}])
        );
    }
}

/// The one fault of words that fit no texture.
fn refused(words: &str) -> String {
    let laid = text::build(&alone(&font(SANS)), words, Align::Left);
    assert!(laid.built.is_none());
    match &laid.faults[..] {
        [text::Fault::Layout(why)] => why.clone(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn words_past_a_vc4_texture_lay_out_nothing_naming_the_limit() {
    for words in [
        // One line of glyphs, far past what one texture holds.
        "I".repeat(400_000),
        // Few glyphs over a vast block: its grid alone would take gigabytes.
        "I".repeat(30_000) + &"\n".repeat(30_000),
    ] {
        let e = refused(&words);
        let rows = e
            .strip_prefix("the words need a 1024x")
            .and_then(|r| r.strip_suffix(" texture, and a VC4 holds 1024x2048 at most"))
            .and_then(|r| r.parse::<usize>().ok());
        assert!(rows.is_some_and(|r| r > 2048), "{e}");
    }
}

// The scene model

/// The fixture fonts as the scene assets `sans` and `cff`.
fn font_assets() -> Value {
    json!({"sans": {"file": SANS}, "cff": {"file": CFF}})
}

/// A scene of the fixture fonts and one pass, `main`, white, with `layers`.
fn layers_scene(layers: Value) -> Value {
    json!({
        "version": 1,
        "assets": font_assets(),
        "passes": [{"name": "main", "fragment": "void main(){ gl_FragColor = vec4(1.0); }",
                    "layers": layers}],
    })
}

fn scene_err(doc: &Value) -> String {
    Scene::from_dict(doc, None, None).err().unwrap().to_string()
}

#[test]
fn layers_round_trip_and_leave_their_defaults_out() {
    let doc = layers_scene(json!([
        {"name": "a", "text": "I"},
        {"name": "b", "visible": false, "x": 4.0, "y": -2.5, "color": "#ff8000",
         "opacity": 0.5, "blend": "add", "text": "AV\nI", "font_size": 48.0,
         "font_family": "sans", "text_align": "center", "render": "slug"},
        {"name": "c", "x": 3.0, "color": [1.0, 0.0, 0.25], "text": ""},
        {"name": "d", "y": 7.0,
         "color": {"from": "#000000", "to": [0.0, 1.0, 0.0], "angle": 90.0}, "text": "I"},
        {"name": "e", "color": {"from": "#ffffff", "to": "#000000"}, "text": "I"},
        {"name": "f", "text": "I", "text_shadow": {}, "glow": {}, "text_stroke": {},
         "karaoke": {}, "wave": {}, "scroll": {}, "blink": {}},
        {"name": "g", "text": "I",
         "text_shadow": {"x": -0.1, "y": 0.0, "blur": 0.0, "color": [0.5, 0.5, 0.5],
                         "strength": 1.0, "visible": false},
         "glow": {"blur": 0.5, "color": "#00ff00", "strength": 0.25},
         "text_stroke": {"width": 0.0, "color": "#ffffff", "visible": false},
         "karaoke": {"sung": 2.5, "color": "#0000ff"}, "wave": {"height": -0.2, "speed": 1.5},
         "scroll": {"speed": -1.0}, "blink": {"rate": 0.0, "visible": false}},
        {"name": "h", "color": "NavY", "text": "I", "text_shadow": {"color": "transparent"},
         "glow": {"color": "Gold"}, "text_stroke": {"color": "white"},
         "karaoke": {"color": "TRANSPARENT"}},
        {"name": "i", "color": {"from": "lime", "to": "Transparent", "angle": 90.0},
         "text": "I"},
        {"name": "j", "text": "I", "font_family": "Pixel Operator 8", "render": "mtsdf"},
        {"name": "k", "text": "I", "render": "hinted"},
        {"name": "l", "text": "I", "render": "mono"},
        {"name": "m", "text": "I", "render": "mtsdf"},
    ]));
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    assert_eq!(sc.to_dict(), doc);
    assert_eq!(Scene::from_dict(&sc.to_dict(), None, None).unwrap(), sc);
    // Each key at its default.
    let defaults = layers_scene(json!([{
        "name": "a", "visible": true, "x": null, "y": null, "color": "#ffffff",
        "opacity": 1, "blend": "normal", "text": "I", "font_size": null,
        "font_family": null, "text_align": "left", "render": "auto",
    }, {
        "name": "b", "color": {"from": [1, 0, 0], "to": [0, 0, 1], "angle": 180},
        "text": "I",
    }, {
        "name": "c", "text": "I",
        "text_shadow": {"x": 0.06, "y": 0.08, "blur": 0.15, "color": "#000000",
                        "strength": 0.7, "visible": true},
        "glow": {"blur": 0.3, "color": "#ffa500", "strength": 0.8},
        "text_stroke": {"width": 0.06, "color": [0, 0, 0]},
        "karaoke": {"sung": 0, "color": "#ffad2e"}, "wave": {"height": 0.1, "speed": 0.6},
        "scroll": {"speed": 2}, "blink": {"rate": 1, "visible": null},
    }, {
        "name": "d", "color": "White", "text": "I",
        "text_shadow": {"color": "black"}, "glow": {"color": "ORANGE"},
    }, {
        "name": "e", "text": "I", "font_family": "Pixel Operator 8", "render": "hinted",
    }]));
    let back = Scene::from_dict(&defaults, None, None).unwrap().to_dict();
    assert_eq!(
        back["passes"][0]["layers"],
        json!([{"name": "a", "text": "I"},
               {"name": "b", "color": {"from": [1.0, 0.0, 0.0], "to": [0.0, 0.0, 1.0]},
                "text": "I"},
               {"name": "c", "text": "I", "text_shadow": {}, "glow": {}, "text_stroke": {},
                "karaoke": {}, "wave": {}, "scroll": {}, "blink": {}},
               {"name": "d", "text": "I", "text_shadow": {}, "glow": {}},
               {"name": "e", "text": "I", "font_family": "Pixel Operator 8"}])
    );
}

#[test]
fn a_layer_declaration_names_what_it_lacks() {
    let one = |d: Value| scene_err(&layers_scene(json!([d])));
    let t = |keys: Value| {
        let mut d = json!({"name": "t", "text": "I"});
        d.as_object_mut()
            .unwrap()
            .extend(keys.as_object().unwrap().clone());
        one(d)
    };
    let colour =
        "needs '#rrggbb', a CSS colour name, [r, g, b] or a gradient {\"from\", \"to\", \"angle\"}";
    let forms = "needs '#rrggbb', a CSS colour name or three numbers [r, g, b]";
    let family = "font_family needs the name of a font asset or a .ttf or .otf file";
    let cases = [
        (
            scene_err(&layers_scene(json!(3))),
            "pass 'main': 'layers' needs a list, got 3".to_string(),
        ),
        (
            one(json!(3)),
            "the 1st layer of pass 'main' needs an object, got 3".to_string(),
        ),
        (
            one(json!({"text": "I"})),
            "the 1st layer of pass 'main' has no 'name'".to_string(),
        ),
        (
            one(json!({"name": 3, "text": "I"})),
            "the 1st layer of pass 'main' name needs a string, got 3".to_string(),
        ),
        (
            one(json!({"name": "t"})),
            "layer 't' has no 'text' (the words that it draws) and no 'image' (an image asset \
             or a file)"
                .to_string(),
        ),
        (
            t(json!({"text": 3})),
            "layer 't' text needs words in a string, got 3".to_string(),
        ),
        (
            t(json!({"visible": "no"})),
            "layer 't' visible needs true or false, got 'no'".to_string(),
        ),
        (
            t(json!({"x": "4"})),
            "layer 't' x needs a number of pixels, got '4'".to_string(),
        ),
        (
            t(json!({"y": true})),
            "layer 't' y needs a number of pixels, got True".to_string(),
        ),
        (
            t(json!({"color": "rde"})),
            format!("layer 't' color {colour}, got 'rde'"),
        ),
        (
            t(json!({"color": [1, 0]})),
            format!("layer 't' color {colour}, got [1, 0]"),
        ),
        (
            t(json!({"color": {"from": "#000000"}})),
            "layer 't' color has no 'to'".to_string(),
        ),
        (
            t(json!({"color": {"from": "#00", "to": "#000000"}})),
            format!("layer 't' color from {forms}, got '#00'"),
        ),
        (
            t(json!({"color": {"from": "#000000", "to": "#000000", "angle": "up"}})),
            "layer 't' color angle needs a number of degrees, got 'up'".to_string(),
        ),
        (
            t(json!({"color": {"from": "#000000", "to": "#000000", "turn": 1}})),
            "layer 't' color takes no 'turn', only from, to and angle".to_string(),
        ),
        (
            t(json!({"opacity": 2})),
            "layer 't' opacity needs a number from 0 to 1, got 2".to_string(),
        ),
        (
            t(json!({"blend": "max"})),
            "layer 't' blend needs 'normal', 'add', 'multiply' or 'screen', got 'max'".to_string(),
        ),
        (
            t(json!({"font_size": 0})),
            "layer 't' font_size needs a number of pixels above 0, got 0".to_string(),
        ),
        (
            t(json!({"font_size": "32"})),
            "layer 't' font_size needs a number of pixels above 0, got '32'".to_string(),
        ),
        (
            t(json!({"font_family": 3})),
            format!("layer 't' {family}, got 3"),
        ),
        (
            t(json!({"font_family": ""})),
            format!("layer 't' {family}, got ''"),
        ),
        (
            t(json!({"font_family": "serif"})),
            "layer 't' font_family 'serif' is no asset of the scene, no .ttf or .otf file and \
             no font that ghee carries, 'DejaVu Sans' or 'Pixel Operator 8'"
                .to_string(),
        ),
        (
            t(json!({"text_align": "middle"})),
            "layer 't' text_align needs 'left', 'center' or 'right', got 'middle'".to_string(),
        ),
        (
            t(json!({"render": "msdf"})),
            "layer 't' render needs 'auto', 'hinted', 'mono', 'mtsdf' or 'slug', got 'msdf'"
                .to_string(),
        ),
        (
            t(json!({"glow": true})),
            "layer 't' glow needs an object, got True".to_string(),
        ),
        (
            t(json!({"glow": {"radius": 1}})),
            "layer 't' glow takes no 'radius', only blur, color, strength and visible".to_string(),
        ),
        (
            t(json!({"text_stroke": {"visible": 1}})),
            "layer 't' text_stroke visible needs true or false, got 1".to_string(),
        ),
        (
            t(json!({"glow": {"blur": -1}})),
            "layer 't' glow blur needs a number of em, 0 or more, got -1".to_string(),
        ),
        (
            t(json!({"text_stroke": {"width": "1"}})),
            "layer 't' text_stroke width needs a number of em, 0 or more, got '1'".to_string(),
        ),
        (
            t(json!({"text_shadow": {"x": "left"}})),
            "layer 't' text_shadow x needs a number of em, got 'left'".to_string(),
        ),
        (
            t(json!({"text_shadow": {"color": "blak"}})),
            format!("layer 't' text_shadow color {forms}, got 'blak'"),
        ),
        (
            t(json!({"glow": {"strength": 1.5}})),
            "layer 't' glow strength needs a number from 0 to 1, got 1.5".to_string(),
        ),
        (
            t(json!({"karaoke": {"sung": null, "color": 3}})),
            format!("layer 't' karaoke color {forms}, got 3"),
        ),
        (
            t(json!({"karaoke": {"sung": "4"}})),
            "layer 't' karaoke sung needs a number of characters, got '4'".to_string(),
        ),
        (
            t(json!({"wave": {"speed": [1]}})),
            "layer 't' wave speed needs a number of waves a second, got [1]".to_string(),
        ),
        (
            t(json!({"scroll": {"speed": "fast"}})),
            "layer 't' scroll speed needs a number of em a second, got 'fast'".to_string(),
        ),
        (
            t(json!({"blink": {"rate": -1}})),
            "layer 't' blink rate needs a number of blinks a second, 0 or more, got -1".to_string(),
        ),
    ];
    for (got, want) in cases {
        assert_eq!(got, format!("scene: {want}"));
    }
    let mut image = layers_scene(json!([{"name": "t", "text": "I", "font_family": "logo"}]));
    image["assets"]["logo"] = json!({"file": "logo.png"});
    assert_eq!(
        scene_err(&image),
        "scene: layer 't' font_family 'logo' is the image 'logo.png', a font is a .ttf or .otf \
         file"
    );
    let mut doc = layers_scene(json!([]));
    doc["passes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name": "d", "fragment": BLACK, "filter": "dither",
            "layers": [{"name": "t", "text": "I"}]}));
    assert_eq!(
        scene_err(&doc),
        "scene: pass 'd' is filter 'dither', and a filter pass takes no layers"
    );
}

#[test]
fn a_font_asset_takes_no_image_keys_and_no_pass_sees_it() {
    let label = || layers_scene(json!([{"name": "label", "text": "I", "font_family": "sans"}]));
    for (k, v) in [
        ("sampling", json!("linear")),
        ("width", json!(4)),
        ("components", json!(1)),
    ] {
        let mut doc = label();
        doc["assets"]["sans"][k] = v;
        assert_eq!(
            scene_err(&doc),
            format!("scene: asset 'sans' takes no '{k}', a .ttf file is a font")
        );
    }
    let sc = Scene::from_dict(&label(), None, None).unwrap();
    let g = sc.passes[0].glsl(Some(&sc.params), None, &sc.assets, &sc.text_names(), false);
    assert!(!g.contains("sans") && !g.contains("cff"), "{g}");
}

#[test]
fn a_layer_declares_its_cover_and_its_size_for_an_author() {
    let mut names = text::names("label");
    names.sort();
    assert_eq!(names, ["label_cover", "label_size"]);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_name_beside_the_names_of_a_layer_compiles() {
    // ghee's own names for a layer start with ghee_, so label_w, label_atlas,
    // label_tex, label_curves and label_emoji are free, and two layers never
    // meet in them. A pass name never reaches GLSL.
    let float = json!({"type": "float", "value": 1.0});
    let mut doc = layers_scene(json!([
        {"name": "label", "text": "\u{1f600}", "font_family": "emoji",
         "x": 0, "y": 0, "font_size": 24},
        {"name": "label_atlas", "text": "I", "font_family": "sans",
         "x": 0, "y": 40, "font_size": 24},
        {"name": "labels", "text": "I", "font_family": "sans", "render": "slug"},
    ]));
    doc["assets"]["emoji"] = json!({"file": EMOJI});
    doc["params"] = json!({"label_w": float, "label_sizes": float});
    doc["passes"][0]["name"] = json!("label_tex");
    doc["passes"][0]["fragment"] = json!(BLACK);
    doc["passes"][0]["params"] =
        json!({"label_emoji_size": float, "label_emoji": float, "label_covers": float});
    doc["passes"][0]["arrays"] = json!([{"name": "label_texels", "size": 4}]);
    let mut image = doc.clone();
    image["assets"]["label_curves"] = json!({"file": "label.png"});
    assert!(Scene::from_dict(&image, None, None).is_ok());
    let (_g, mut r) = gpu(W, H);
    let f = render(&mut r, &loaded(&doc));
    assert!(f.contains(&255), "the layers paint nothing");
}

/// The names declared after a type in GLSL `src`, the locals and the function
/// parameters, outside the comments and the file-scope declarations, and
/// except ghee's own names.
fn locals(src: &str) -> Vec<String> {
    let types = [
        "bool", "int", "float", "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4",
    ];
    let scope = ["uniform", "varying", "attribute", "const", "precision"];
    let ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let opens = |s: &str| -> i32 {
        s.chars()
            .map(|c| match c {
                '{' | '(' => 1,
                '}' | ')' => -1,
                _ => 0,
            })
            .sum()
    };
    let mut out: Vec<String> = Vec::new();
    // The braces and parentheses open before a word: a name declared with
    // none open is a file-scope declaration.
    let mut depth = 0;
    for line in src.lines() {
        let code = line.split("//").next().unwrap_or("");
        let first = code.split_whitespace().next().unwrap_or("");
        if first.starts_with('#') || scope.contains(&first) {
            continue;
        }
        // Each word, the text up to the next word, and the depth at the word.
        let mut words: Vec<(&str, &str, i32)> = Vec::new();
        let mut rest = code;
        while let Some(i) = rest.find(ident) {
            depth += opens(&rest[..i]);
            let tail = &rest[i..];
            let n = tail.find(|c| !ident(c)).unwrap_or(tail.len());
            let gap = &tail[n..];
            let next = gap.find(ident).unwrap_or(gap.len());
            words.push((&tail[..n], &gap[..next], depth));
            rest = gap;
        }
        depth += opens(rest);
        for pair in words.windows(2) {
            let ((ty, gap, _), (name, after, at)) = (pair[0], pair[1]);
            let declared = types.contains(&ty)
                && at > 0
                && !gap.is_empty()
                && gap.trim().is_empty()
                && !after.trim_start().starts_with('(');
            if declared && !name.starts_with("ghee_") && !out.iter().any(|o| o == name) {
                out.push(name.to_string());
            }
        }
    }
    out
}

/// The atlas detail of a test that writes a layer's shaders itself.
const DETAIL: Detail = Detail {
    px_em: 32,
    range: 6,
};

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_vec2_param_named_like_a_local_of_ghees_glsl_changes_no_pixel() {
    let mut doc = layers_scene(json!([
        {"name": "label", "text": "\u{1f600}", "font_family": "emoji",
         "x": 8, "y": 8, "font_size": 48},
        {"name": "glyphs", "text": "I", "font_family": "sans", "x": 8, "y": 8, "font_size": 48},
        {"name": "title", "text": "I", "font_family": "sans", "render": "slug",
         "x": 60, "y": 8, "font_size": 48},
    ]));
    doc["assets"]["emoji"] = json!({"file": EMOJI});
    doc["passes"][0]["fragment"] =
        json!("void main(){ gl_FragColor = vec4(vec3(title_cover(uv * title_size, 32.0)), 1.0); }");
    doc["passes"][0]["arrays"] =
        json!([{"name": "amps", "size": 4}, {"name": "mask", "size": [4, 4]}]);
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    let p = &sc.passes[0];
    let mut glsl: Vec<String> = Vec::new();
    for hw_linear in [false, true] {
        glsl.push(p.glsl(
            Some(&sc.params),
            None,
            &sc.assets,
            &sc.text_names(),
            hw_linear,
        ));
    }
    for l in p.layers.iter().filter_map(Layer::as_text) {
        let sh = l.shaders(DETAIL);
        for (v, f) in [sh.glyphs, sh.boxes, sh.colour] {
            glsl.extend([v, f]);
        }
    }
    let mut names: Vec<String> = Vec::new();
    for n in glsl.iter().flat_map(|g| locals(g)) {
        if !names.contains(&n) {
            names.push(n);
        }
    }
    assert!(names.iter().any(|n| n == "at"), "{names:?}");
    let mut named = doc.clone();
    for n in &names {
        named["params"][n] = json!({"type": "vec2", "value": [0.0, 0.0]});
    }
    let (_g, mut r) = gpu(W, H);
    let plain = render(&mut r, &loaded(&doc));
    assert!(plain.contains(&255), "the scene draws nothing");
    // The pass and the layers paint grey, and the emoji paints colour.
    let coloured = |f: &[u8]| f.as_chunks::<4>().0.iter().any(|p| p[0] != p[2]);
    assert!(coloured(&plain), "the scene draws no emoji");
    assert!(render(&mut r, &loaded(&named)) == plain, "{names:?}");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_vec_or_color_param_named_like_a_swizzle_reads_as_any_uniform() {
    let red = json!({"type": "color", "value": [1.0, 0.0, 0.0]});
    let code = |read: &str| {
        format!(
            "void main(){{\n    vec4 v = vec4(uv, 0.25, 1.0);\n    \
             gl_FragColor = vec4({read}, 0.0, 1.0);\n}}"
        )
    };
    let scene = |params: Value, code: &str| {
        let mut doc = layers_scene(json!([]));
        doc["params"] = params;
        doc["passes"][0]["fragment"] = json!(code);
        doc
    };
    let (_g, mut r) = gpu(W, H);
    // The pass selects .bg and reads the color bg, which multiplies by 1.
    let plain = render(&mut r, &loaded(&scene(json!({}), &code("v.bg"))));
    let bg = scene(json!({"bg": red}), &code("v.bg * bg.r"));
    assert!(render(&mut r, &loaded(&bg)) == plain);
    // A filter's code selects .rgb beside a scene param rgb.
    let rgb = json!({"rgb": red});
    let mut sc = loaded(&scene(rgb, &code("v.rg")));
    apply_filters(&mut sc, &["dither".to_string()], None).unwrap();
    let mut plain = loaded(&scene(json!({}), &code("v.rg")));
    apply_filters(&mut plain, &["dither".to_string()], None).unwrap();
    assert!(render(&mut r, &sc) == render(&mut r, &plain));
}

#[test]
fn every_scene_pass_reads_the_layers_and_a_filter_pass_none() {
    let reads = "float c = label_cover(uv, 8.0) + title_cover(uv, 8.0);";
    let doc = json!({
        "version": 1,
        "assets": font_assets(),
        "passes": [
            {"name": "a", "fragment": format!("void main(){{ {reads} gl_FragColor = vec4(c); }}"),
             "layers": [{"name": "label", "text": "I", "font_family": "sans"},
                        {"name": "title", "text": "O", "font_family": "cff"}]},
            {"name": "main", "channels": [{"buffer": "a"}],
             "fragment": format!("void main(){{ {reads} \
                 gl_FragColor = texture2D(iChannel0, uv) * c; }}")},
        ],
    });
    let mut sc = Scene::from_dict(&doc, None, None).unwrap();
    apply_filters(&mut sc, &["dither".to_string()], None).unwrap();
    assert_eq!(sc.passes.len(), 3);
    let glsl = |p: &Pass| p.glsl(Some(&sc.params), None, &sc.assets, &sc.text_names(), false);
    for p in &sc.passes[..2] {
        let g = glsl(p);
        for name in ["label", "title"] {
            assert!(
                g.contains(&format!("float {name}_cover(vec2 ")),
                "{}",
                p.name
            );
            assert!(g.contains(&format!("{name}_size")), "{}", p.name);
        }
    }
    let filter = glsl(&sc.passes[2]);
    assert!(!filter.contains("label_") && !filter.contains("title_"));
}

#[test]
fn layers_survive_a_filter_asset_and_a_built_param() {
    let doc = layers_scene(json!([{"name": "label", "text": "I", "font_family": "sans",
                                   "text_align": "right"}]));
    let mut sc = Scene::from_dict(&doc, None, None).unwrap();
    sc.passes[0].layers[0].as_text_mut().unwrap().data = Some(alone(&font(SANS)));
    let layers = sc.passes[0].layers.clone();
    apply_filters(&mut sc, &["frc256".to_string()], None).unwrap();
    assert!(sc.assets.iter().any(|a| a.name == "frc256_bluenoise"));
    assert_eq!(sc.passes[0].layers, layers);
    let sc = Ok::<_, anyhow::Error>(sc)
        .param(Param::float("gain", 1.0))
        .unwrap();
    assert_eq!(sc.passes[0].layers, layers);
}

// Measuring words

#[test]
fn a_measure_is_the_block_in_pixels_at_its_size_or_at_the_fit() {
    // The block of the reference layout times the pixels per em: the
    // font_size, or the size that fits the block to the pass less a
    // twentieth of its shorter side on each edge, which the pixel font
    // rounds down to a whole multiple of 8 px, 8 at the least.
    let _g = lock();
    let sc = loaded(&json!({"version": 1, "assets": {"face": {"file": SANS}},
                            "passes": [{"name": "main", "fragment": "void main(){}"}]}));
    let rows = [
        (
            Some("face"),
            SANS,
            "AV To\nghee",
            Some(20.0),
            [160.0_f64, 96.0],
        ),
        (Some("face"), SANS, "AV To\nghee", None, [160.0, 96.0]),
        (Some("face"), SANS, "Hi", None, [128.0, 32.0]),
        (None, LATIN, "ghee", None, [128.0, 32.0]),
        (Some(text::DEJAVU), LATIN, "ghee", None, [128.0, 32.0]),
        (Some(text::PIXEL), PIXEL, "42%", None, [128.0, 32.0]),
        (Some(text::PIXEL), PIXEL, "load 42%", None, [64.0, 16.0]),
        (Some(text::PIXEL), PIXEL, "42%", Some(8.0), [128.0, 32.0]),
    ];
    for (family, file, words, size, pass) in rows {
        let laid = lay(file, words, Align::Left);
        let px = size.unwrap_or_else(|| {
            let m = 0.05 * pass[0].min(pass[1]);
            let fit = ((pass[0] - 2.0 * m) / laid.width).min((pass[1] - 2.0 * m) / laid.height);
            match family {
                Some(text::PIXEL) => ((fit / 8.0).floor() * 8.0).max(8.0),
                _ => fit,
            }
        });
        let got = sc.measure(pass, words, size, family).unwrap();
        let want = [laid.width * px, laid.height * px];
        assert!(
            (got[0] - want[0]).abs() < 1e-6 && (got[1] - want[1]).abs() < 1e-6,
            "{family:?} {words:?} {size:?}: {got:?}, want {want:?}"
        );
    }
}

#[test]
fn a_measure_names_a_size_or_a_font_that_the_scene_cannot_take() {
    let _g = lock();
    let sc = loaded(&json!({"version": 1, "assets": {"face": {"file": SANS}},
                            "passes": [{"name": "main", "fragment": "void main(){}"}]}));
    let rows = [
        (
            Some(0.0),
            None,
            "font_size needs a number of pixels above 0, got 0.0".to_string(),
        ),
        (
            None,
            Some("serif"),
            "font_family 'serif' is no asset of the scene, no .ttf or .otf file and no font \
             that ghee carries, 'DejaVu Sans' or 'Pixel Operator 8'"
                .to_string(),
        ),
        (
            None,
            Some("gone.ttf"),
            format!(
                "font_family 'gone.ttf' ({FONTS}/gone.ttf): No such file or directory, and \
                 $GHEE_ASSETS names no directory"
            ),
        ),
    ];
    for (size, family, want) in rows {
        let e = sc.measure([128.0, 32.0], "hi", size, family).unwrap_err();
        assert_eq!(e.to_string(), want);
    }
}

// Font assets

/// Run `f` with $GHEE_ASSETS set, then put it back. The caller holds LOCK.
fn with_assets_env<T>(assets: &str, f: impl FnOnce() -> T) -> T {
    let saved = std::env::var("GHEE_ASSETS").ok();
    std::env::set_var("GHEE_ASSETS", assets);
    let out = f();
    match saved {
        Some(v) => std::env::set_var("GHEE_ASSETS", v),
        None => std::env::remove_var("GHEE_ASSETS"),
    }
    out
}

fn font_data(sc: &Scene, name: &str) -> Vec<u8> {
    let l = sc.text_layer(name).unwrap();
    l.data.clone().unwrap()[0].clone().unwrap().to_vec()
}

/// A scene whose layer `label` draws in the font asset `face`, the file
/// `file`.
fn face_scene(file: &str) -> Scene {
    let mut doc = layers_scene(json!([{"name": "label", "text": "I", "font_family": "face"}]));
    doc["assets"] = json!({"face": {"file": file}});
    Scene::from_dict(&doc, None, None).unwrap()
}

#[test]
fn a_font_resolves_beside_the_scene_then_through_ghee_assets() {
    let _g = lock();
    let root = tmpdir("order");
    let (dir, store) = (format!("{root}/scene"), format!("{root}/store"));
    for d in [&dir, &store] {
        fs::create_dir_all(d).unwrap();
    }
    let (sans, cff) = (font(SANS), font(CFF));
    fs::write(format!("{dir}/here.ttf"), &sans).unwrap();
    fs::write(format!("{store}/store.ttf"), &sans).unwrap();
    // The scene directory wins over the store.
    fs::write(format!("{dir}/both.ttf"), &sans).unwrap();
    fs::write(format!("{store}/both.ttf"), &cff).unwrap();
    let names = ["here", "store", "both"];
    let mut doc = layers_scene(Value::Array(
        names
            .iter()
            .map(|n| json!({"name": n, "text": "I", "font_family": format!("{n}_font")}))
            .collect(),
    ));
    doc["assets"] = Value::Object(
        names
            .iter()
            .map(|n| (format!("{n}_font"), json!({"file": format!("{n}.ttf")})))
            .collect(),
    );
    let mut sc = Scene::from_dict(&doc, None, None).unwrap();
    with_assets_env(&store, || sc.load_assets(&dir)).unwrap();
    for name in names {
        assert_eq!(font_data(&sc, name), sans, "{name}");
    }
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn layers_in_one_font_share_its_bytes() {
    let _g = lock();
    let mut sc = Scene::from_dict(
        &layers_scene(
            json!([{"name": "label", "text": "I", "font_family": "sans"},
                             {"name": "title", "text": "I", "font_family": "sans"}]),
        ),
        None,
        None,
    )
    .unwrap();
    sc.load_assets(FONTS).unwrap();
    let data = |n: &str| {
        sc.text_layer(n).unwrap().data.clone().unwrap()[0]
            .clone()
            .unwrap()
    };
    assert!(Arc::ptr_eq(&data("label"), &data("title")));
}

/// Why the font of `sc`'s layer `label` failed to load.
fn font_failure(sc: &Scene) -> String {
    let l = sc.text_layer("label").unwrap();
    l.data.clone().unwrap()[0].clone().err().unwrap()
}

#[test]
fn a_missing_font_names_its_path_and_the_directories_of_ghee_assets() {
    let _g = lock();
    let root = tmpdir("missing");
    let (dir, store) = (format!("{root}/scene"), format!("{root}/store"));
    for d in [&dir, &store] {
        fs::create_dir_all(d).unwrap();
    }
    let file = format!("ghee-text-missing-{}.ttf", std::process::id());
    for (assets, more) in [
        (store.as_str(), format!(", nor in $GHEE_ASSETS ({store})")),
        ("", ", and $GHEE_ASSETS names no directory".to_string()),
    ] {
        let mut sc = face_scene(&file);
        with_assets_env(assets, || sc.load_assets(&dir)).unwrap();
        assert_eq!(
            font_failure(&sc),
            format!("asset 'face' ({dir}/{file}): No such file or directory{more}")
        );
    }
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_missing_image_fails_the_load() {
    let _g = lock();
    let dir = tmpdir("noimage");
    let mut doc = layers_scene(json!([{"name": "label", "text": "I", "font_family": "sans"}]));
    doc["assets"]["logo"] = json!({"file": "logo.png"});
    let mut sc = Scene::from_dict(&doc, None, None).unwrap();
    fs::write(format!("{dir}/{SANS}"), font(SANS)).unwrap();
    fs::write(format!("{dir}/{CFF}"), font(CFF)).unwrap();
    let e = with_assets_env("", || sc.load_assets(&dir))
        .err()
        .unwrap()
        .to_string();
    assert!(
        e.starts_with(&format!("asset 'logo' ({dir}/logo.png): No such file")),
        "{e}"
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_font_file_that_is_no_font_loads_as_its_failure() {
    let _g = lock();
    let dir = tmpdir("nofont");
    fs::write(format!("{dir}/bad.ttf"), b"not a font").unwrap();
    let mut sc = face_scene("bad.ttf");
    with_assets_env("", || sc.load_assets(&dir)).unwrap();
    let e = font_failure(&sc);
    assert!(
        e.starts_with(&format!("asset 'face' ({dir}/bad.ttf): unreadable font")),
        "{e}"
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_scene_file_loads_its_fonts() {
    let _g = lock();
    let dir = tmpdir("load");
    fs::write(format!("{dir}/{SANS}"), font(SANS)).unwrap();
    fs::write(format!("{dir}/{CFF}"), font(CFF)).unwrap();
    let path = format!("{dir}/x.scene.json");
    let doc = layers_scene(json!([{"name": "label", "text": "I", "font_family": "sans"}]));
    fs::write(&path, doc.to_string()).unwrap();
    let sc = Scene::load(&path).unwrap();
    assert_eq!(font_data(&sc, "label"), font(SANS));
    fs::remove_dir_all(&dir).unwrap();
}

// Rendering

const W: usize = 192;
const H: usize = 128;

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_fitted_capital_i_covers_its_stem_and_nothing_beside_it() {
    let laid = lay(SANS, "I", Align::Left);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let (_g, mut r) = gpu(W, H);
    let f = render(&mut r, &loaded(&label_scene(SANS, "I", Align::Left, fit)));
    assert_draws(&f, W, 0, &laid, fit, "I");

    // DejaVu Sans' I is one rectangle, its bounding box.
    let data = font(SANS);
    let face = Face::parse(&data, 0).unwrap();
    let upem = f64::from(face.units_per_em());
    let ascent = f64::from(face.ascender()) / upem;
    let b = face
        .glyph_bounding_box(face.glyph_index('I').unwrap())
        .unwrap();
    assert!(laid
        .glyphs
        .iter()
        .flat_map(|g| g.segs.iter().flatten())
        .all(|p| {
            [b.x_min, b.x_max]
                .iter()
                .any(|x| (p[0] - f64::from(*x) / upem).abs() < 1e-12)
                && [b.y_min, b.y_max]
                    .iter()
                    .any(|y| (p[1] - (ascent - f64::from(*y) / upem)).abs() < 1e-12)
        }));
    let px = |em: f64, origin: f64| origin + em * fit.ppem;
    let (x0, x1) = (
        px(f64::from(b.x_min) / upem, fit.x),
        px(f64::from(b.x_max) / upem, fit.x),
    );
    let (y0, y1) = (
        px(ascent - f64::from(b.y_max) / upem, fit.y),
        px(ascent - f64::from(b.y_min) / upem, fit.y),
    );
    assert!(x1 - x0 > 8.0 && y1 - y0 > 60.0); // The stem spans pixels to read.
    let (mut stem, mut beside) = (0, 0);
    for y in 0..H {
        for x in 0..W {
            let (cx, cy) = (x as f64 + 0.5, y as f64 + 0.5);
            let v = f[(y * W + x) * 4];
            let inner = cx > x0 + EDGE && cx < x1 - EDGE && cy > y0 + EDGE && cy < y1 - EDGE;
            let outer = cx < x0 - EDGE || cx > x1 + EDGE || cy < y0 - EDGE || cy > y1 + EDGE;
            if inner {
                assert!(v >= 255 - TOL, "({x}, {y}) inside the stem reads {v}");
                stem += 1;
            } else if outer {
                assert!(v <= TOL, "({x}, {y}) beside the stem reads {v}");
                beside += 1;
            }
        }
    }
    assert!(stem > 300 && beside > 10_000, "{stem} {beside}");
}

/// A layer in the pixel font that names no render lights the font's own
/// pixels. The reference lays the words out from the font with ttf-parser,
/// at the size that the layer gives, or the size that fits rounded down to a
/// whole multiple of 8 px and 8 at the least, with the block's top left,
/// each pen and each baseline rounded to whole pixels. At a multiple of 8 px
/// the font's squares cover whole pixels, so every pixel reads 0 or 255 as
/// the outline says, up to TOL: slug's curves sit on a lattice of 2047 steps
/// across each glyph's box, which moves an edge by up to 1/400 px at 8 px.
/// At 28 px the edges antialias, and the pixels away from them read full or
/// none.
#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn the_pixel_font_lights_its_own_pixels_on_whole_pixels() {
    // Each row: the pass's size, the words, their alignment, the layer's
    // keys, and the pixels per em that it draws at.
    let rows = [
        // Room for 9.2 px, two lines 8.72 px apart, the second 13.5 px in.
        (
            (80, 24),
            "Hg 42%\nxy",
            Align::Center,
            json!({"x": 3.4, "y": 2.7}),
            8.0,
        ),
        // Room for 7.98 px.
        ((80, 12), "Hg", Align::Left, json!({"x": 0, "y": 2.7}), 8.0),
        // Room for 19.8 px, the block centred 23.5 px across, 6.56 px down.
        ((97, 48), "Hg 4\n2%,", Align::Left, json!({}), 16.0),
        (
            (64, 40),
            "Ag",
            Align::Left,
            json!({"x": 2, "y": 1, "font_size": 8}),
            8.0,
        ),
        (
            (64, 40),
            "Ag",
            Align::Left,
            json!({"x": 1, "y": 2, "font_size": 28}),
            28.0,
        ),
    ];
    for ((w, h), words, align, keys, px) in rows {
        let laid = lay(PIXEL, words, align);
        let corner = |at: &str, room: usize, block: f64| {
            let centred = (room as f64 - block * px) / 2.0;
            (keys[at].as_f64().unwrap_or(centred) + 0.5).floor()
        };
        let top = [corner("x", w, laid.width), corner("y", h, laid.height)];
        let want = laid.on_pixels(top, px);
        let mut layer = json!({"name": "t", "text": words, "font_family": "Pixel Operator 8",
                               "text_align": align_name(align)});
        layer
            .as_object_mut()
            .unwrap()
            .extend(keys.as_object().unwrap().clone());
        let doc = json!({"version": 1, "passes": [{"name": "main",
            "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
            "layers": [layer]}]});
        let (_g, mut r) = gpu(w, h);
        let f = render(&mut r, &Scene::from_dict(&doc, None, None).unwrap());
        let what = format!("{words:?} in {w}x{h}");
        let centre = |x: usize, y: usize| [x as f64 + 0.5, y as f64 + 0.5];
        if px % 8.0 != 0.0 {
            assert_draws_at(&f, w, 0, &want, centre, EDGE, &what);
            continue;
        }
        let (mut lit, mut off) = (0, Vec::new());
        for (x, y) in (0..h).flat_map(|y| (0..w).map(move |x| (x, y))) {
            let inside = want.inside(centre(x, y), 0.0);
            lit += usize::from(inside == Some(true));
            let got = f[(y * w + x) * 4];
            let want = if inside == Some(true) { 255 } else { 0 };
            if inside.is_none() || got.abs_diff(want) > TOL {
                off.push((x, y, got, inside));
            }
        }
        assert!(lit > 20, "{what}: {lit} pixels lit");
        assert!(
            off.is_empty(),
            "{what}: {} pixels off, (x, y, got, inside) {:?}",
            off.len(),
            &off[..off.len().min(8)]
        );
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn two_lines_take_their_alignment() {
    let words = "AV To\nghee";
    let (_g, mut r) = gpu(W, H);
    let mut frames = Vec::new();
    for align in [Align::Left, Align::Center, Align::Right] {
        let laid = lay(SANS, words, align);
        let fit = Fit::of(laid.width, laid.height, W, H);
        let f = render(&mut r, &loaded(&label_scene(SANS, words, align, fit)));
        assert_draws(&f, W, 0, &laid, fit, align_name(align));
        frames.push(f);
    }
    assert!(frames[0] != frames[1] && frames[1] != frames[2]);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn cubic_outlines_draw() {
    let laid = lay(CFF, "IO", Align::Left);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let (_g, mut r) = gpu(W, H);
    let f = render(&mut r, &loaded(&label_scene(CFF, "IO", Align::Left, fit)));
    assert_draws(&f, W, 0, &laid, fit, "IO");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_missing_character_draws_glyph_0() {
    let laid = lay(SANS, "I字", Align::Left);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let (_g, mut r) = gpu(W, H);
    let f = render(&mut r, &loaded(&label_scene(SANS, "I字", Align::Left, fit)));
    assert_draws(&f, W, 0, &laid, fit, "I字");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn empty_words_cover_nothing() {
    let fit = Fit {
        x: 8.0,
        y: 8.0,
        ppem: 64.0,
    };
    let (_g, mut r) = gpu(W, H);
    for words in ["", "\n", " "] {
        let f = render(&mut r, &loaded(&label_scene(SANS, words, Align::Left, fit)));
        assert!(blank(&f, 0), "{words:?}");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn size_reads_the_block_in_em() {
    let words = "AV To\nghee";
    let laid = lay(SANS, words, Align::Center);
    let doc = json!({
        "version": 1,
        "assets": font_assets(),
        "passes": [{"name": "main", "fragment": format!(
            "void main(){{ gl_FragColor = vec4(\
             abs(label_size.x - {:.6}) < 1e-4 ? 1.0 : 0.0, \
             abs(label_size.y - {:.6}) < 1e-4 ? 1.0 : 0.0, 0.0, 1.0); }}",
            laid.width, laid.height),
            "layers": [{"name": "label", "text": words, "font_family": "sans",
                        "text_align": "center", "visible": false}]}],
    });
    let (_g, mut r) = gpu(W, H);
    let f = render(&mut r, &loaded(&doc));
    assert_eq!(&f[..4], [255, 255, 0, 255]);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_string_param_relays_the_text_live() {
    let (i, o) = (lay(SANS, "I", Align::Left), lay(SANS, "O", Align::Left));
    let fit = Fit::of(o.width, o.height, W, H);
    let mut doc = label_scene(SANS, "I", Align::Left, fit);
    doc["params"]["gain"] = json!({"type": "float", "value": 0.0});
    doc["passes"][0]["fragment"] = json!(format!(
        "void main(){{ gl_FragColor = vec4({}, gain, 0.0, 1.0); }}",
        fit.cover("label")
    ));
    let (_g, mut r) = gpu(W, H);
    let f = render(&mut r, &loaded(&doc));
    assert_draws(&f, W, 0, &i, fit, "I");
    // gain takes a value the scene does not hold, so a rebuild from the scene
    // shows in green.
    assert!(r
        .set_scene_param("gain", "float", &ParamValue::Float(1.0))
        .unwrap());
    set_words(&r, "label", "O");
    let f = r.render(None).unwrap().to_vec();
    assert_draws(&f, W, 0, &o, fit, "O");
    assert!(f.iter().skip(1).step_by(4).all(|g| *g == 255));
    set_words(&r, "label", "");
    assert!(blank(r.render(None).unwrap(), 0));
    assert_eq!(r.text_errors(), [] as [String; 0]);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_text_reaches_every_pass() {
    let laid = lay(SANS, "AV", Align::Left);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let doc = json!({
        "version": 1,
        "assets": font_assets(),
        "passes": [
            {"name": "a", "fragment": format!(
                "void main(){{ gl_FragColor = vec4({}, 0.0, 0.0, 1.0); }}", fit.cover("label")),
             "layers": [{"name": "label", "text": "AV", "font_family": "sans", "visible": false}]},
            {"name": "main", "channels": [{"buffer": "a"}], "fragment": format!(
                "void main(){{ gl_FragColor = vec4(texture2D(iChannel0, uv).r, {}, \
                 0.0, 1.0); }}", fit.cover("label"))},
        ],
    });
    let (_g, mut r) = gpu(W, H);
    let f = render(&mut r, &loaded(&doc));
    assert_draws(&f, W, 0, &laid, fit, "pass a");
    assert_draws(&f, W, 1, &laid, fit, "pass main");
    set_words(&r, "label", "To");
    let to = lay(SANS, "To", Align::Left);
    let f = r.render(None).unwrap().to_vec();
    assert_draws(&f, W, 0, &to, fit, "pass a, live");
    assert_draws(&f, W, 1, &to, fit, "pass main, live");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn two_layers_draw_their_own_words() {
    let (label, title) = (lay(SANS, "AV", Align::Left), lay(CFF, "O", Align::Left));
    let fit = Fit::of(
        label.width.max(title.width),
        label.height.max(title.height),
        W,
        H,
    );
    let doc = json!({
        "version": 1,
        "assets": font_assets(),
        "passes": [{"name": "main", "fragment": format!(
            "void main(){{ gl_FragColor = vec4({}, {}, 0.0, 1.0); }}",
            fit.cover("label"), fit.cover("title")),
            "layers": [{"name": "label", "text": "AV", "font_family": "sans", "visible": false},
                       {"name": "title", "text": "O", "font_family": "cff", "visible": false}]}],
    });
    let (_g, mut r) = gpu(W, H);
    let f = render(&mut r, &loaded(&doc));
    assert_draws(&f, W, 0, &label, fit, "label");
    assert_draws(&f, W, 1, &title, fit, "title");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_later_setup_draws_its_own_text() {
    let words = ["I", "O", "AV"];
    let laid: Vec<Laid> = words.iter().map(|w| lay(SANS, w, Align::Left)).collect();
    let fit = Fit::of(
        laid.iter().map(|l| l.width).fold(0.0, f64::max),
        laid[0].height,
        W,
        H,
    );
    let (_g, mut r) = gpu(W, H);
    for (w, l) in words.iter().zip(&laid) {
        let f = render(&mut r, &loaded(&label_scene(SANS, w, Align::Left, fit)));
        assert_draws(&f, W, 0, l, fit, w);
    }
    // A scene whose layer has another name.
    let title = json!({
        "version": 1,
        "assets": font_assets(),
        "passes": [{"name": "main", "fragment": format!(
            "void main(){{ gl_FragColor = vec4(vec3({}), 1.0); }}", fit.cover("title")),
            "layers": [{"name": "title", "text": "I", "font_family": "sans", "visible": false}]}],
    });
    let f = render(&mut r, &loaded(&title));
    assert_draws(&f, W, 0, &laid[0], fit, "title");
    assert!(r.set_layer(&text_of("label"), Given::Text("O")).is_err());
    // A scene with no layer, then a layer again.
    let flat = Scene::of(vec![Pass::of(
        "main",
        "void main(){ gl_FragColor = vec4(0.5, 0.0, 0.0, 1.0); }",
    )
    .unwrap()])
    .unwrap();
    let f = render(&mut r, &flat);
    assert!((i32::from(f[0]) - 128).abs() <= 1);
    let f = render(&mut r, &loaded(&label_scene(SANS, "O", Align::Left, fit)));
    assert_draws(&f, W, 0, &laid[1], fit, "O again");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_scene_whose_font_never_loaded_fails_its_setup_naming_the_font() {
    let fit = Fit {
        x: 8.0,
        y: 8.0,
        ppem: 64.0,
    };
    let sc = Scene::from_dict(&label_scene(SANS, "I", Align::Left, fit), None, None).unwrap();
    let (_g, mut r) = gpu(W, H);
    assert_eq!(
        r.setup(&sc).unwrap(),
        "layer 'label': font 'face' was never loaded, the scene was built without a directory"
    );
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_filter_after_a_layer_compiles() {
    let laid = lay(SANS, "I", Align::Left);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let (_g, mut r) = gpu(W, H);
    let mut sc = loaded(&label_scene(SANS, "I", Align::Left, fit));
    apply_filters(&mut sc, &["dither".to_string()], None).unwrap();
    assert_eq!(r.setup(&sc).unwrap(), "");
}

// Text problems

/// The error boxes of a layout: x0, y0, x1, y1 in em from the block's top
/// left, y down.
fn box_rects(laid: &text::Layout) -> Vec<[f64; 4]> {
    let b = &laid.built.as_ref().unwrap().boxes;
    (0..b.count)
        .map(|i| vertex(b, i, 0)[1].map(f64::from))
        .collect()
}

/// What a w x h frame shows at each pixel more than EDGE pixels from every
/// edge of the error boxes `rects` and of their stripes, as `fit` places the
/// block: Some(true) on a yellow stripe, Some(false) on a black one, None
/// outside every box. Returns the pixels seen of each kind, yellow, black
/// and outside.
fn each_box_pixel(
    w: usize,
    h: usize,
    fit: Fit,
    rects: &[[f64; 4]],
    mut f: impl FnMut(usize, usize, Option<bool>),
) -> [usize; 3] {
    let edge = EDGE / fit.ppem;
    let mut seen = [0; 3];
    for y in 0..h {
        for x in 0..w {
            let p = fit.em(x, y);
            let near = rects.iter().any(|r| {
                let out = (p[0] - r[0])
                    .min(r[2] - p[0])
                    .min(p[1] - r[1])
                    .min(r[3] - p[1]);
                out.abs() <= edge
            });
            if near {
                continue;
            }
            let Some(r) = rects
                .iter()
                .find(|r| p[0] > r[0] && p[0] < r[2] && p[1] > r[1] && p[1] < r[3])
            else {
                seen[2] += 1;
                f(x, y, None);
                continue;
            };
            // x - y from the box's bottom left, y up, in stripe widths.
            let c = ((p[0] - r[0]) - (r[3] - p[1])) / text::STRIPE;
            if (c - c.round()).abs() * text::STRIPE / std::f64::consts::SQRT_2 <= edge {
                continue;
            }
            let yellow = c.floor().rem_euclid(2.0) == 0.0;
            seen[usize::from(!yellow)] += 1;
            f(x, y, Some(yellow));
        }
    }
    seen
}

/// Channel 0 of an RGBA frame shows full coverage on the yellow stripes of
/// the boxes and none on the black ones or outside them.
fn assert_striped(frame: &[u8], w: usize, fit: Fit, rects: &[[f64; 4]], what: &str) {
    let h = frame.len() / 4 / w;
    let mut off = Vec::new();
    let seen = each_box_pixel(w, h, fit, rects, |x, y, yellow| {
        let want = if yellow == Some(true) { 255 } else { 0 };
        let got = frame[(y * w + x) * 4];
        if got.abs_diff(want) > TOL {
            off.push((x, y, got, want));
        }
    });
    assert!(seen.iter().all(|n| *n > 50), "{what}: {seen:?}");
    assert!(
        off.is_empty(),
        "{what}: (x, y, got, want) {:?}",
        &off[..off.len().min(8)]
    );
}

/// label_scene with the words drawn by a white layer over a blue pass.
fn hazard_scene(file: &str, words: &str, fit: Fit) -> Value {
    let mut doc = label_scene(file, words, Align::Left, fit);
    doc["passes"][0] = json!({"name": "main",
        "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.5, 1.0); }",
        "layers": [placed(file, words, fit)]});
    doc
}

/// An RGBA frame shows the error boxes `rects` as hazard stripes, yellow
/// and black, and the blue pass outside them.
fn assert_hazard(frame: &[u8], w: usize, fit: Fit, rects: &[[f64; 4]], what: &str) {
    let h = frame.len() / 4 / w;
    let mut off = Vec::new();
    let seen = each_box_pixel(w, h, fit, rects, |x, y, yellow| {
        let want = match yellow {
            Some(true) => [255, 204, 0],
            Some(false) => [0, 0, 0],
            None => [0, 0, 128],
        };
        let got = &frame[(y * w + x) * 4..(y * w + x) * 4 + 3];
        if got.iter().zip(want).any(|(g, w)| g.abs_diff(w) > TOL) {
            off.push((x, y, got.to_vec()));
        }
    });
    assert!(seen.iter().all(|n| *n > 50), "{what}: {seen:?}");
    assert!(
        off.is_empty(),
        "{what}: (x, y, got) {:?}",
        &off[..off.len().min(8)]
    );
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_missing_font_draws_a_box_a_character_and_the_scene_runs() {
    let words = "Ag j\nI";
    let mut l = face_layer();
    l.data = Some(vec![Err("gone".to_string())]);
    let laid = l.build(words).unwrap();
    let rects = box_rects(&laid);
    assert_eq!(rects.len(), 4);
    let (bw, bh) = block(laid.built.as_ref().unwrap());
    let fit = Fit::of(bw, bh, W, H);
    let (_g, mut r) = gpu(W, H);
    let cover = render(
        &mut r,
        &loaded(&label_scene("Missing.ttf", words, Align::Left, fit)),
    );
    assert_striped(&cover, W, fit, &rects, "_cover");
    let errors = r.text_errors();
    let want = format!("asset 'face' ({FONTS}/Missing.ttf): No such file or directory");
    assert!(
        errors.len() == 1 && errors[0].starts_with(&want),
        "{errors:?}"
    );
    for how in RENDERS {
        let doc = rendered(hazard_scene("Missing.ttf", words, fit), how);
        let layered = render(&mut r, &loaded(&doc));
        assert_hazard(&layered, W, fit, &rects, &format!("{how} layer"));
        // New words lay out in boxes too.
        set_words(&r, "label", "I");
        r.set_time(Some(0.0));
        let one = r.render(None).unwrap().to_vec();
        assert_hazard(&one, W, fit, &rects[..1], &format!("{how} one box"));
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn words_that_take_their_first_error_box_after_a_frame_draw_it() {
    let words = "Ag j\nI";
    let mut l = face_layer();
    l.data = Some(vec![Err("gone".to_string())]);
    let laid = l.build(words).unwrap();
    let rects = box_rects(&laid);
    let (bw, bh) = block(laid.built.as_ref().unwrap());
    let fit = Fit::of(bw, bh, W, H);
    let (_g, mut r) = gpu(W, H);
    for how in RENDERS {
        let doc = rendered(hazard_scene("Missing.ttf", "", fit), how);
        render(&mut r, &loaded(&doc));
        set_words(&r, "label", words);
        r.set_time(Some(0.0));
        let boxed = r.render(None).unwrap().to_vec();
        assert_hazard(&boxed, W, fit, &rects, how);
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_font_family_names_a_font_asset_or_a_file() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let (_g, mut r) = gpu(W, H);
    let asset = hazard_scene(SANS, "Ag", fit);
    let want = render(&mut r, &loaded(&asset));
    // The file itself.
    let mut doc = asset.clone();
    layer_of(&mut doc, "label")["font_family"] = json!(SANS);
    let sc = loaded(&doc);
    assert_eq!(sc.to_dict()["passes"][0]["layers"][0]["font_family"], SANS);
    assert!(render(&mut r, &sc) == want);
    assert_eq!(r.text_errors(), [] as [String; 0]);
    // A missing file leaves the layer running with error boxes, and names
    // the file.
    layer_of(&mut doc, "label")["font_family"] = json!("Missing.ttf");
    render(&mut r, &loaded(&doc));
    let errors = r.text_errors();
    let missing =
        format!("layer 'label' font_family 'Missing.ttf' ({FONTS}/Missing.ttf): No such file");
    assert!(errors.iter().any(|e| e.starts_with(&missing)), "{errors:?}");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn an_unreadable_font_draws_a_box_a_character() {
    let words = "Ag";
    let laid = text::build(&alone(b"not a font"), words, Align::Left);
    let rects = box_rects(&laid);
    let (bw, bh) = block(laid.built.as_ref().unwrap());
    let fit = Fit::of(bw, bh, W, H);
    let dir = tmpdir("unreadable");
    fs::write(format!("{dir}/bad.ttf"), b"not a font").unwrap();
    let (_g, mut r) = gpu(W, H);
    let mut sc =
        Scene::from_dict(&label_scene("bad.ttf", words, Align::Left, fit), None, None).unwrap();
    with_assets_env("", || sc.load_assets(&dir)).unwrap();
    let cover = render(&mut r, &sc);
    assert_striped(&cover, W, fit, &rects, "unreadable");
    let errors = r.text_errors();
    let want = format!("asset 'face' ({dir}/bad.ttf): unreadable font");
    assert!(
        errors.len() == 1 && errors[0].starts_with(&want),
        "{errors:?}"
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_colour_font_alone_draws_its_emoji_in_layers_and_nothing_through_cover() {
    // A space, a zero-width joiner the font maps, and a variation selector
    // it lacks need no ink.
    let words = "\u{1f600} \u{200d}\u{1f642}\u{fe0f}";
    let fonts = alone(&font(EMOJI));
    let laid = text::build(&fonts, words, Align::Left);
    assert_eq!(laid.faults, []);
    let built = laid.built.as_ref().unwrap();
    assert_eq!((built.quads.count, built.boxes.count), (0, 0));
    assert!(built.glyphs.is_empty());
    let chars: Vec<char> = built.colour.iter().map(|p| p.ch).collect();
    assert_eq!(chars, ['\u{1f600}', '\u{1f642}']);
    // The space and the two emoji take an advance each.
    let data = font(EMOJI);
    let face = Face::parse(&data, 0).unwrap();
    let advance =
        f64::from(face.glyph_hor_advance(GlyphId(3)).unwrap()) / f64::from(face.units_per_em());
    let (bw, _) = block(built);
    assert!((bw - 3.0 * advance).abs() < 1e-9, "{bw}");
    let (w, h) = (160, 150);
    let (_g, mut r) = gpu(w, h);
    for (k, ch) in ['\u{1f600}', '\u{1f642}'].into_iter().enumerate() {
        let b = bitmap(EMOJI, ch);
        let (fit, at) = bitmap_fit(built, k, &b, 1.0, [8.0, 6.0]);
        let cover = render(
            &mut r,
            &loaded(&label_scene(EMOJI, words, Align::Left, fit)),
        );
        assert!(blank(&cover, 0), "_cover drew a colour glyph");
        assert_eq!(r.text_errors(), [] as [String; 0]);
        for how in RENDERS {
            let doc = rendered(hazard_scene(EMOJI, words, fit), how);
            let layered = render(&mut r, &loaded(&doc));
            assert_bitmap(&layered, w, &b, at, 1, [0, 0, 128], &format!("{how} {ch}"));
            assert_eq!(r.text_errors(), [] as [String; 0]);
        }
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_cff2_font_without_a_variation_store_draws_like_its_cff_source() {
    let words = "IO\nOI";
    assert!(near(
        built(CFF2, words, Align::Center),
        built(CFF, words, Align::Center)
    ));
    let laid = lay(CFF, words, Align::Center);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let (_g, mut r) = gpu(W, H);
    let cff = render(
        &mut r,
        &loaded(&label_scene(CFF, words, Align::Center, fit)),
    );
    let cff2 = render(
        &mut r,
        &loaded(&label_scene(CFF2, words, Align::Center, fit)),
    );
    assert_draws(&cff2, W, 0, &laid, fit, "CFF2");
    let (n, m) = apart(&cff2, &cff);
    assert!(n == 0 && m <= 1, "{n} pixels apart, at most {m}");
    let cff = render(
        &mut r,
        &loaded(&layer_scene(CFF, words, Align::Center, fit)),
    );
    let cff2 = render(
        &mut r,
        &loaded(&layer_scene(CFF2, words, Align::Center, fit)),
    );
    let (n, m) = apart(&cff2, &cff);
    assert!(n == 0 && m <= 1, "layers: {n} pixels apart, at most {m}");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn words_that_fit_no_texture_draw_nothing_at_setup_keep_the_last_words_later_and_say_why() {
    let fit = Fit {
        x: 8.0,
        y: 8.0,
        ppem: 64.0,
    };
    let past = "I".repeat(400_000);
    let (_g, mut r) = gpu(W, H);
    let why = |r: &Renderer| {
        let errors = r.text_errors();
        assert!(
            errors.len() == 1 && errors[0].starts_with("layer 'label': the words need a "),
            "{errors:?}"
        );
    };
    for (doc, what) in [
        (label_scene(SANS, &past, Align::Left, fit), "cover"),
        (layer_scene(SANS, &past, Align::Left, fit), "slug"),
    ] {
        assert!(blank(&render(&mut r, &loaded(&doc)), 0), "{what}");
        why(&r);
        set_words(&r, "label", "I");
        r.set_time(Some(0.0));
        let last = r.render(None).unwrap().to_vec();
        assert!(!blank(&last, 0), "{what}");
        assert_eq!(r.text_errors(), [] as [String; 0], "{what}");
        set_words(&r, "label", &past);
        r.set_time(Some(0.0));
        assert!(r.render(None).unwrap() == last.as_slice(), "{what}");
        why(&r);
    }
}

/// Two slug layers in the pixel font, `a` over `b`, on a black pass.
fn pixel_pair(a: &str, b: &str) -> Scene {
    let layer = |name: &str, words: &str, y: i32| {
        json!({"name": name, "text": words, "font_family": "Pixel Operator 8", "font_size": 8,
               "x": 0, "y": y})
    };
    let doc = json!({"version": 1, "passes": [{"name": "main",
        "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
        "layers": [layer("a", a, 0), layer("b", b, 12)]}]});
    Scene::from_dict(&doc, None, None).unwrap()
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn slug_words_changed_many_times_draw_what_a_scene_built_with_them_draws() {
    // The slug layers of a renderer share one texture of their glyphs,
    // which grows as words bring new glyphs. Layer b's words walk the
    // printable ASCII and back, so the texture takes new glyphs within its
    // rows and grows several times under layer a, whose words stay. A
    // second renderer builds the scene afresh at each step.
    let printable: Vec<char> = (' '..='~').collect();
    let mut steps: Vec<String> = printable.chunks(5).map(String::from_iter).collect();
    steps.extend(steps.clone().into_iter().rev().step_by(3));
    let (_g, mut r) = gpu(256, 24);
    let mut fresh = probe_dri_devices()
        .iter()
        .find_map(|d| Renderer::new(256, 24, 4, Some(d), false, false).ok())
        .unwrap();
    render(&mut r, &pixel_pair("AV 42", "x"));
    for words in &steps {
        set_words(&r, "b", words);
        r.set_time(Some(0.0));
        let changed = r.render(None).unwrap().to_vec();
        let built = render(&mut fresh, &pixel_pair("AV 42", words));
        assert!(!blank(&built, 0), "{words:?}");
        assert!(changed == built, "{words:?}");
    }
}

// Layers

/// Quad `i`'s vertex `v`: glyph_a0 to glyph_a3.
fn vertex(q: &text::Quads, i: usize, v: usize) -> [[f32; 4]; 4] {
    let at = (i * 4 + v) * q.per_vertex;
    let f = &q.floats[at..at + 16];
    [0, 1, 2, 3].map(|k| [f[k * 4], f[k * 4 + 1], f[k * 4 + 2], f[k * 4 + 3]])
}

#[test]
fn a_quad_spans_each_inked_glyph_with_its_indices() {
    let words = "AV To\nghee";
    let built = drawn(SANS, words, Align::Center);
    let q = &built.quads;
    // A0 V1 T3 o4 g6 h7 e8 e9: every character counts, the space and the
    // newline too, and neither takes a quad.
    let want = [
        (0, 0, 0),
        (1, 0, 0),
        (3, 1, 0),
        (4, 1, 0),
        (6, 2, 1),
        (7, 2, 1),
        (8, 2, 1),
        (9, 2, 1),
    ];
    assert_eq!((q.count, q.per_vertex), (want.len(), text::QUAD_FLOATS));
    assert_eq!(q.floats.len(), q.count * 4 * q.per_vertex);
    let laid = lay(SANS, words, Align::Center);
    let inked: Vec<&Outline> = laid.glyphs.iter().filter(|g| !g.segs.is_empty()).collect();
    let texels = (built.width * built.height) as f32;
    let data = font(SANS);
    let face = Face::parse(&data, 0).unwrap();
    let upem = f64::from(face.units_per_em());
    for (i, (&(index, word, line), g)) in want.iter().zip(&inked).enumerate() {
        let a = vertex(q, i, 0);
        for (v, corner) in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
            .iter()
            .enumerate()
        {
            let b = vertex(q, i, v);
            assert_eq!(&b[0][..2], corner, "quad {i} vertex {v}");
            assert_eq!(
                (&b[0][2..], &b[1..]),
                (&a[0][2..], &a[1..]),
                "quad {i} vertex {v}"
            );
        }
        assert_eq!(a[0][2..], [index as f32, word as f32], "quad {i}");
        // The box holds the ink, quantized outward by less than 1/64 em.
        let [x0, y0, x1, y1] = a[1].map(f64::from);
        for (got, ink, out) in [
            (x0, g.lo[0], -1.0),
            (y0, g.lo[1], -1.0),
            (x1, g.hi[0], 1.0),
            (y1, g.hi[1], 1.0),
        ] {
            let past = (got - ink) * out;
            assert!(
                (-1e-6..1.0 / 64.0).contains(&past),
                "quad {i}: box {:?} ink {:?} {:?}",
                a[1],
                g.lo,
                g.hi
            );
        }
        assert!(a[2][0] >= 1.0 && a[2][0] < texels, "quad {i}: {:?}", a[2]);
        assert!(a[2][1] == a[2][2] && [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0].contains(&a[2][1]));
        assert_eq!(
            (a[2][3], a[3][0], a[3][3]),
            (0.0, line as f32, 0.0),
            "quad {i}"
        );
        // The advance runs one advance width from the origin, which sits the
        // left side bearing left of the ink.
        let gid = face.glyph_index(words.chars().nth(index).unwrap()).unwrap();
        let lsb = f64::from(face.glyph_bounding_box(gid).unwrap().x_min) / upem;
        let advance = f64::from(face.glyph_hor_advance(gid).unwrap()) / upem;
        let (x0, x1) = (f64::from(a[3][1]), f64::from(a[3][2]));
        assert!(
            (x0 - (g.lo[0] - lsb)).abs() < 1e-4 && (x1 - x0 - advance).abs() < 1e-4,
            "quad {i}: span {:?}, ink from {}, bearing {lsb}, advance {advance}",
            a[3],
            g.lo[0]
        );
    }
}

/// The layer `label` in the font asset `face`, the fixture `file`, its
/// block where `fit` puts it: its `y` above the fit's top by rounding().
fn placed(file: &str, words: &str, fit: Fit) -> Value {
    json!({"name": "label", "text": words, "font_family": "face",
           "x": fit.x, "y": fit.y - rounding(file, fit.ppem), "font_size": fit.ppem})
}

const BLACK: &str = "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }";

/// label_scene with the words drawn white by layer `label` over a black
/// pass, in the slug render.
fn layer_scene(file: &str, words: &str, align: Align, fit: Fit) -> Value {
    let mut doc = label_scene(file, words, align, fit);
    let mut l = placed(file, words, fit);
    l["text_align"] = json!(align_name(align));
    l["render"] = json!("slug");
    doc["passes"][0] = json!({"name": "main", "fragment": BLACK, "layers": [l]});
    doc
}

/// `doc` with its layer `label` in `render`.
fn rendered(mut doc: Value, render: &str) -> Value {
    layer_of(&mut doc, "label")["render"] = json!(render);
    doc
}

const RENDERS: [&str; 2] = ["mtsdf", "slug"];

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_hidden_pass_draws_none_of_its_layers() {
    // under is gray, main draws white words over black, and post shows main.
    let fit = Fit {
        x: 8.0,
        y: 8.0,
        ppem: 48.0,
    };
    let (_g, mut r) = gpu(W, H);
    let mut doc = layer_scene(SANS, "Ag", Align::Left, fit);
    let main = doc["passes"][0].clone();
    doc["passes"] = json!([
        {"name": "under", "fragment": "void main(){ gl_FragColor = vec4(0.25, 0.25, 0.25, 1.0); }"},
        main,
        {"name": "post", "channels": [{"buffer": "main"}],
         "fragment": "void main(){ gl_FragColor = texture2D(iChannel0, uv); }"},
    ]);
    let shown = render(&mut r, &loaded(&doc));
    assert!(shown.contains(&255), "the layer paints nothing");
    doc["passes"][1]["visible"] = json!(false);
    let hidden = render(&mut r, &loaded(&doc));
    assert!(
        hidden
            .chunks(4)
            .all(|p| p[..3].iter().all(|v| v.abs_diff(64) <= 1)),
        "a hidden pass's layer drew"
    );
}

/// The red channels of two RGBA frames: the pixels more than 1 level apart,
/// and the largest difference.
fn apart(a: &[u8], b: &[u8]) -> (usize, u8) {
    a.iter()
        .zip(b)
        .step_by(4)
        .map(|(x, y)| x.abs_diff(*y))
        .fold((0, 0), |(n, m), d| (n + usize::from(d > 1), m.max(d)))
}

/// Slug's coverage jumps at a glyph's corner, where the last bits of the
/// sample decide it, and a layer reaches its sample by another sum than
/// `<name>_cover` does. Every other pixel lands within 1 level.
const CORNERS: f64 = 0.005;

fn assert_like_cover(layered: &[u8], cover: &[u8], what: &str) {
    let inked = cover.iter().step_by(4).filter(|v| **v > 0).count();
    let (n, m) = apart(layered, cover);
    assert!(inked > 200, "{what}: {inked} pixels inked");
    assert!(
        (n as f64) <= CORNERS * inked as f64,
        "{what}: {n} of {inked} inked pixels more than 1 level apart, at most {m}"
    );
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_covers_as_the_text_does() {
    let (_g, mut r) = gpu(320, 120);
    let cases = [
        (SANS, "AV To\nghee", 13.0),
        (SANS, "Wave fjord {}@", 29.5625),
        (SANS, "ghee", 61.0),
        (CFF, "IO OI", 37.25),
    ];
    for (file, words, ppem) in cases {
        for (dx, dy) in [(0.0, 0.0), (0.3125, 0.6875)] {
            let fit = Fit {
                x: 6.0 + dx,
                y: 4.0 + dy,
                ppem,
            };
            let what = format!("{file} {words:?} at {ppem} px/em, +({dx}, {dy})");
            let cover = render(&mut r, &loaded(&label_scene(file, words, Align::Left, fit)));
            let layered = render(&mut r, &loaded(&layer_scene(file, words, Align::Left, fit)));
            assert_like_cover(&layered, &cover, &what);
        }
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_follows_its_words_live() {
    let (w, h) = (256, 64);
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 40.0,
    };
    let (_g, mut r) = gpu(w, h);
    let want = render(
        &mut r,
        &loaded(&label_scene(SANS, "ghee", Align::Left, fit)),
    );
    let before = render(&mut r, &loaded(&layer_scene(SANS, "I", Align::Left, fit)));
    set_words(&r, "label", "ghee");
    r.set_time(Some(0.0));
    let after = r.render(None).unwrap().to_vec();
    assert!(after != before, "the layer kept the old words");
    assert_like_cover(&after, &want, "ghee after I");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn new_words_whose_glyphs_the_atlas_holds_draw_in_the_frame_they_arrive() {
    let (w, h) = (256, 64);
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 40.0,
    };
    let (_g, mut r) = gpu(w, h);
    let cache = tmpdir("held");
    std::env::set_var("XDG_CACHE_HOME", &cache);
    // The setup waits for "I", and for the preload, which holds every
    // letter of "AV To", and its space draws nothing.
    render(
        &mut r,
        &loaded(&rendered(layer_scene(SANS, "I", Align::Left, fit), "mtsdf")),
    );
    // Time runs free, so the render waits for no glyph.
    r.set_time(None);
    set_words(&r, "label", "AV To");
    let frame = r.render(None).unwrap().to_vec();
    assert_draws(
        &frame,
        w,
        0,
        &lay(SANS, "AV To", Align::Left),
        fit,
        "the first frame of AV To",
    );
    fs::remove_dir_all(&cache).unwrap();
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn layers_land_in_their_order() {
    let (w, h) = (96, 96);
    let fit = Fit {
        x: 32.0,
        y: 8.0,
        ppem: 64.0,
    };
    let laid = lay(SANS, "I", Align::Left);
    let g = &laid.glyphs[0];
    let stem = (
        (fit.x + (g.lo[0] + g.hi[0]) / 2.0 * fit.ppem) as usize,
        (fit.y + (g.lo[1] + g.hi[1]) / 2.0 * fit.ppem) as usize,
    );
    let (_g, mut r) = gpu(w, h);
    for (first, last) in [
        ([0.0, 1.0, 0.0], [255, 0, 0]),
        ([1.0, 0.0, 0.0], [0, 255, 0]),
    ] {
        let mut under = placed(SANS, "I", fit);
        under["name"] = json!("under");
        under["color"] = json!(first);
        let mut over = placed(SANS, "I", fit);
        over["color"] = json!(last.map(|c| f64::from(c) / 255.0));
        let mut doc = label_scene(SANS, "I", Align::Left, fit);
        doc["passes"][0] = json!({"name": "main", "fragment": BLACK, "layers": [under, over]});
        let frame = render(&mut r, &loaded(&doc));
        let at = (stem.1 * w + stem.0) * 4;
        assert_eq!(frame[at..at + 3], last, "{first:?} under");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_paints_its_color_at_its_opacity_by_its_blend() {
    // A stem pixel of the I takes the layer's colour at full coverage, over
    // a pass of grey.
    let (w, h) = (96, 96);
    let fit = Fit {
        x: 32.0,
        y: 8.0,
        ppem: 64.0,
    };
    let laid = lay(SANS, "I", Align::Left);
    let g = &laid.glyphs[0];
    let at = {
        let (x, y) = (
            (fit.x + (g.lo[0] + g.hi[0]) / 2.0 * fit.ppem) as usize,
            (fit.y + (g.lo[1] + g.hi[1]) / 2.0 * fit.ppem) as usize,
        );
        (y * w + x) * 4
    };
    let grey: f64 = 0.25;
    let c: [f64; 3] = [1.0, 0.5, 0.25];
    let (_g, mut r) = gpu(w, h);
    for (how, opacity, blend) in [
        ("slug", 1.0, "normal"),
        ("mtsdf", 0.5, "normal"),
        ("slug", 0.5, "add"),
        ("mtsdf", 1.0, "multiply"),
        ("slug", 1.0, "screen"),
    ] {
        let mut l = placed(SANS, "I", fit);
        l["render"] = json!(how);
        l["color"] = json!(c);
        l["opacity"] = json!(opacity);
        l["blend"] = json!(blend);
        let mut doc = label_scene(SANS, "I", Align::Left, fit);
        doc["passes"][0] = json!({"name": "main", "layers": [l],
            "fragment": "void main(){ gl_FragColor = vec4(vec3(0.25), 1.0); }"});
        let frame = render(&mut r, &loaded(&doc));
        for (k, c) in c.iter().enumerate() {
            let s = c * opacity;
            let want = match blend {
                "normal" => s + grey * (1.0 - opacity),
                "add" => s + grey,
                "multiply" => s * grey + grey * (1.0 - opacity),
                _ => s + grey * (1.0 - s),
            };
            let got = f64::from(frame[at + k]);
            assert!(
                (got - 255.0 * want.min(1.0)).abs() <= 1.5,
                "{how} {opacity} {blend}: channel {k} {got}, want {}",
                255.0 * want
            );
        }
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_gradient_spans_the_block_along_its_angle_as_css_does() {
    // CSS's gradient line runs through the block's centre at the angle, 0
    // up and 90 right, and reaches the block's far corners at its ends.
    let words = "IIII\nIIII";
    let laid = lay(SANS, words, Align::Left);
    let (w, h) = (320, 240);
    let fit = Fit::of(laid.width, laid.height, w, h);
    let (from, to) = ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let (_g, mut r) = gpu(w, h);
    for angle in [90.0, 180.0, 30.0] {
        let mut l = placed(SANS, words, fit);
        l["color"] = json!({"from": "#ff0000", "to": [0.0, 0.0, 1.0], "angle": angle});
        let mut doc = label_scene(SANS, words, Align::Left, fit);
        doc["passes"][0] = json!({"name": "main", "fragment": BLACK, "layers": [l]});
        let frame = render(&mut r, &loaded(&doc));
        let a = f64::to_radians(angle);
        let dir = [a.sin(), -a.cos()];
        let len = (laid.width * dir[0]).abs() + (laid.height * dir[1]).abs();
        let (mut seen, mut off) = (0, Vec::new());
        for y in 0..h {
            for x in 0..w {
                let p = fit.em(x, y);
                if laid.inside(p, EDGE / fit.ppem) != Some(true) {
                    continue;
                }
                let d = [p[0] - laid.width / 2.0, p[1] - laid.height / 2.0];
                let t = ((d[0] * dir[0] + d[1] * dir[1]) / len + 0.5).clamp(0.0, 1.0);
                seen += 1;
                for k in 0..3 {
                    let want = 255.0 * (from[k] + (to[k] - from[k]) * t);
                    let got = f64::from(frame[(y * w + x) * 4 + k]);
                    if (got - want).abs() > 2.0 {
                        off.push((x, y, k, got, want));
                    }
                }
            }
        }
        assert!(seen > 500, "{angle}: {seen}");
        assert!(off.is_empty(), "{angle}: {:?}", &off[..off.len().min(8)]);
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_colour_name_paints_its_value_from_css_and_transparent_paints_nothing() {
    // An I at 64 px/em over a grey pass. CSS Color 4 gives lime as #00ff00,
    // blue as #0000ff and red as #ff0000. Each row: the layer's keys, its
    // renders, the stroke's width, and the reference's straight colour and
    // alpha at a pixel, from the distance to the outline, d, in em and below
    // 0 inside, and the point t on the gradient line. A gradient to
    // transparent fades its colour's alpha, as CSS mixes premultiplied
    // colours.
    let words = "I";
    let laid = lay(SANS, words, Align::Left);
    let fit = Fit {
        x: 48.0,
        y: 16.0,
        ppem: 64.0,
    };
    let (w, h) = (160, 128);
    let px = 1.0 / fit.ppem;
    let grey = 0.25;
    let letters =
        |rgb: [f64; 3], alpha: f64| move |d: f64| (rgb, if d < 0.0 { alpha } else { 0.0 });
    type Want = Box<dyn Fn(f64, f64) -> ([f64; 3], f64)>;
    let rows: Vec<(Value, &[&str], f64, Want)> = vec![
        (
            json!({"color": "Lime"}),
            &RENDERS,
            0.0,
            Box::new(move |d, _| letters([0.0, 1.0, 0.0], 1.0)(d)),
        ),
        (
            json!({"color": "transparent"}),
            &RENDERS,
            0.0,
            Box::new(|_, _| ([0.0; 3], 0.0)),
        ),
        (
            json!({"color": {"from": "red", "to": "Transparent", "angle": 90}}),
            &RENDERS,
            0.0,
            Box::new(move |d, t| letters([1.0, 0.0, 0.0], 1.0 - t)(d)),
        ),
        (
            json!({"color": "Lime", "karaoke": {"sung": 1, "color": "transparent"}}),
            &RENDERS,
            0.0,
            Box::new(|_, _| ([0.0; 3], 0.0)),
        ),
        (
            json!({"color": "TRANSPARENT", "text_stroke": {"width": 0.1, "color": "blue"}}),
            &["mtsdf"],
            0.1,
            Box::new(move |d, _| ([0.0, 0.0, 1.0], ((0.1 - d) / px + 0.5).clamp(0.0, 1.0))),
        ),
        (
            json!({"color": "lime", "glow": {"color": "transparent", "strength": 1.0},
                   "text_shadow": {"color": "Transparent", "strength": 1.0}}),
            &["mtsdf"],
            0.0,
            Box::new(move |d, _| letters([0.0, 1.0, 0.0], 1.0)(d)),
        ),
    ];
    let (_g, mut r) = gpu(w, h);
    for (keys, renders, width, want) in rows {
        for render_as in renders {
            let mut doc = effect_scene(words, fit, keys.clone());
            doc["passes"][0]["fragment"] =
                json!("void main(){ gl_FragColor = vec4(vec3(0.25), 1.0); }");
            layer_of(&mut doc, "label")["render"] = json!(render_as);
            let frame = render(&mut r, &loaded(&doc));
            let len = laid.width;
            let (mut seen, mut off) = (0, Vec::new());
            for y in 0..h {
                for x in 0..w {
                    let p = fit.em(x, y);
                    let d = laid.distance(p);
                    if d.abs() < EDGE * px || (width > 0.0 && (d - width).abs() < EDGE * px) {
                        continue;
                    }
                    // At 90 degrees the gradient line runs left to right
                    // across the block.
                    let t = (p[0] / len).clamp(0.0, 1.0);
                    // The field gives the distance within half a pixel.
                    let q = 0.5 * px;
                    let ends = [-q, q].map(|e| {
                        let (rgb, a) = want(d + e, t);
                        rgb.map(|c| 255.0 * (c * a + grey * (1.0 - a)))
                    });
                    seen += 1;
                    for k in 0..3 {
                        let got = f64::from(frame[(y * w + x) * 4 + k]);
                        let (lo, hi) = (ends[0][k].min(ends[1][k]), ends[0][k].max(ends[1][k]));
                        if got < lo - 3.0 || got > hi + 3.0 {
                            off.push((x, y, k, got, lo.round(), hi.round()));
                        }
                    }
                }
            }
            assert!(seen > 10_000, "{keys} {render_as}: {seen}");
            assert!(
                off.is_empty(),
                "{keys} {render_as}: {} off, (x, y, channel, got, least, most) {:?}",
                off.len(),
                &off[..off.len().min(8)]
            );
        }
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_fits_its_room_and_centres_on_each_axis_left_out() {
    let (w, h) = (320, 150);
    let words = "AV To\nghee";
    let laid = lay(SANS, words, Align::Left);
    let (fw, fh) = (w as f64, h as f64);
    // A block with no font_size fits its room less a margin of 1/20 of the
    // pass's shorter side: the whole pass on an axis left out, else the pass
    // past the given pixel.
    let m = 0.05 * fw.min(fh);
    let fits = |rw: f64, rh: f64| (rw / laid.width).min(rh / laid.height);
    let centre = |ppem: f64| {
        (
            (fw - laid.width * ppem) / 2.0,
            (fh - laid.height * ppem) / 2.0,
        )
    };
    let at = |x: Option<f64>, y: Option<f64>, ppem: f64| Fit {
        x: x.unwrap_or(centre(ppem).0),
        y: y.unwrap_or(centre(ppem).1),
        ppem,
    };
    let cases = [
        (json!({}), at(None, None, fits(fw - 2.0 * m, fh - 2.0 * m))),
        (json!({"font_size": 40}), at(None, None, 40.0)),
        (
            json!({"x": 14, "y": 7.5}),
            at(Some(14.0), Some(7.5), fits(fw - m - 14.0, fh - m - 7.5)),
        ),
        (
            json!({"font_size": 40, "x": -6, "y": 30}),
            at(Some(-6.0), Some(30.0), 40.0),
        ),
        (
            json!({"x": 14}),
            at(Some(14.0), None, fits(fw - m - 14.0, fh - 2.0 * m)),
        ),
        (
            json!({"y": 50}),
            at(None, Some(50.0), fits(fw - 2.0 * m, fh - m - 50.0)),
        ),
        (
            json!({"font_size": 40, "x": 10}),
            at(Some(10.0), None, 40.0),
        ),
    ];
    let (_g, mut r) = gpu(w, h);
    for (keys, fit) in cases {
        let mut doc = layer_scene(SANS, words, Align::Left, fit);
        let mut layer = json!({"name": "label", "text": words, "font_family": "face",
                               "render": "slug"});
        layer
            .as_object_mut()
            .unwrap()
            .extend(keys.as_object().unwrap().clone());
        doc["passes"][0]["layers"] = json!([layer]);
        let frame = render(&mut r, &loaded(&doc));
        assert_draws(&frame, w, 0, &laid, fit, &keys.to_string());
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_hidden_layer_draws_nothing_and_its_words_read_through_cover() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 64.0,
    };
    let (_g, mut r) = gpu(W, H);
    for how in RENDERS {
        let shown = rendered(layer_scene(SANS, "Ag", Align::Left, fit), how);
        let mut hidden = shown.clone();
        layer_of(&mut hidden, "label")["visible"] = json!(false);
        let mut none = shown.clone();
        none["passes"][0]["layers"] = json!([]);
        assert!(render(&mut r, &loaded(&shown)).contains(&255), "{how}");
        let want = render(&mut r, &loaded(&none));
        assert!(render(&mut r, &loaded(&hidden)) == want, "{how}");
    }
    let laid = lay(SANS, "Ag", Align::Left);
    let f = render(&mut r, &loaded(&label_scene(SANS, "Ag", Align::Left, fit)));
    assert_draws(&f, W, 0, &laid, fit, "a hidden layer's words");
}

#[test]
fn a_pass_of_layers_alone_round_trips_with_no_code_and_a_pass_of_neither_is_refused() {
    let mut doc = layers_scene(json!([]));
    doc["passes"] = json!([
        {"name": "words", "layers": [{"name": "label", "text": "I", "font_family": "sans"}]},
        {"name": "main", "fragment": BLACK},
    ]);
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    assert!(!sc.passes[0].has_fragment());
    let back = sc.to_dict();
    assert_eq!(back["passes"][0], doc["passes"][0]);
    assert_eq!(Scene::from_dict(&back, None, None).unwrap(), sc);
    for words in [
        json!({"name": "words"}),
        json!({"name": "words", "fragment": " "}),
    ] {
        doc["passes"][0] = words;
        assert_eq!(
            scene_err(&doc),
            "scene: pass 'words' has no 'fragment' and no 'layers'"
        );
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_on_a_sized_pass_covers_as_the_text_does_there() {
    // Pass a renders at half the frame on each axis, and main shows it.
    let (w, h) = (320, 120);
    let words = "AV To\nghee";
    let laid = lay(SANS, words, Align::Left);
    let fit = Fit::of(laid.width, laid.height, w / 2, h / 2);
    let show = "void main(){ gl_FragColor = texture2D(iChannel0, uv); }";
    let sized = |pass: Value| {
        let mut doc = label_scene(SANS, words, Align::Left, fit);
        let mut a = pass;
        a["name"] = json!("a");
        a["size"] = json!([w / 2, h / 2]);
        doc["passes"] =
            json!([a, {"name": "main", "channels": [{"buffer": "a"}], "fragment": show}]);
        loaded(&doc)
    };
    let cover = sized(label_scene(SANS, words, Align::Left, fit)["passes"][0].clone());
    let layered = sized(layer_scene(SANS, words, Align::Left, fit)["passes"][0].clone());
    let (_g, mut r) = gpu(w, h);
    let want = render(&mut r, &cover);
    let got = render(&mut r, &layered);
    assert_like_cover(&got, &want, "a layer on a 160x60 pass");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_past_the_texture_units_fails_naming_the_layer() {
    // main's channels and array take every unit, and its layer needs one
    // more for the text.
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let (_g, mut r) = gpu(160, 48);
    let before = render(&mut r, &loaded(&layer_scene(SANS, "I", Align::Left, fit)));
    let units = r.max_texture_units() as usize;
    let mut doc = layer_scene(SANS, "I", Align::Left, fit);
    let channels: Vec<Value> = (0..units - 1).map(|_| json!({"buffer": "a"})).collect();
    let mut main = doc["passes"][0].clone();
    main["channels"] = json!(channels);
    main["arrays"] = json!([{"name": "v", "size": 4}]);
    main["fragment"] = json!("void main(){ gl_FragColor = vec4(v_at(0)); }");
    doc["passes"] = json!([{"name": "a", "fragment": BLACK}, main]);
    let past = match unsafe { ghee::ffi::ghee_gl_texture_units(r.handle()) } > 16 {
        true => "a pass binds at most",
        false => "this GPU has",
    };
    assert_eq!(
        r.setup(&loaded(&doc)).unwrap(),
        format!(
            "layer 'label' needs {} texture units, {past} {units}",
            units + 1
        )
    );
    r.set_time(Some(0.0));
    assert!(r.render(None).unwrap() == before.as_slice());
}

/// The alpha channel of an RGBA frame.
fn alphas(frame: &[u8]) -> Vec<u8> {
    frame.iter().skip(3).step_by(4).copied().collect()
}

/// One channel of an RGBA frame.
fn plane(frame: &[u8], c: usize) -> Vec<u8> {
    frame.iter().skip(c).step_by(4).copied().collect()
}

const I_FIT: Fit = Fit {
    x: 4.0,
    y: 4.0,
    ppem: 40.0,
};

/// layer_scene with its layer in a pass of layers alone, `words`, and that pass
/// the output.
fn words_only() -> Value {
    let mut doc = layer_scene(SANS, "I", Align::Left, I_FIT);
    let layers = doc["passes"][0]["layers"].clone();
    doc["passes"] = json!([{"name": "words", "layers": layers}]);
    doc
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_over_a_transparent_pass_leaves_its_coverage_in_alpha() {
    let (_g, mut r) = gpu(64, 56);
    let opaque = render(&mut r, &loaded(&words_only()));
    assert!(alphas(&opaque).iter().all(|a| *a == 255));
    r.set_alpha(true);
    let kept = render(&mut r, &loaded(&words_only()));
    // White coverage, premultiplied: each of r, g, b and a is the coverage.
    let cover = plane(&kept, 0);
    assert_eq!(alphas(&kept), cover);
    assert_eq!(plane(&opaque, 0), cover);
    assert!(cover.contains(&255) && cover.contains(&0));
    assert!(cover.iter().any(|c| (1..255).contains(c)), "no edge pixel");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_pass_of_layers_alone_starts_transparent_and_a_later_pass_composites_it() {
    let (_g, mut r) = gpu(64, 56);
    // The same layer over black, and the words pass composited over blue.
    let over_black = render(&mut r, &loaded(&layer_scene(SANS, "I", Align::Left, I_FIT)));
    let mut doc = words_only();
    doc["passes"].as_array_mut().unwrap().push(json!({
        "name": "main",
        "channels": [{"buffer": "words"}],
        "fragment": "void main(){ vec4 t = texture2D(iChannel0, uv); \
                 gl_FragColor = vec4(vec3(0.0, 0.0, 1.0) * (1.0 - t.a) + t.rgb, 1.0); }",
    }));
    let composed = render(&mut r, &loaded(&doc));
    assert!(plane(&over_black, 0).contains(&255));
    for c in [0, 1] {
        let worst = plane(&composed, c)
            .iter()
            .zip(plane(&over_black, c))
            .map(|(a, b)| a.abs_diff(b))
            .max();
        assert!(worst <= Some(1), "channel {c}: {worst:?}");
    }
    assert!(plane(&composed, 2).iter().all(|b| *b >= 254));
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_blends_the_pass_alpha_as_its_colour() {
    let (_g, mut r) = gpu(64, 56);
    r.set_alpha(true);
    for how in RENDERS {
        let mut doc = rendered(layer_scene(SANS, "I", Align::Left, I_FIT), how);
        doc["passes"][0]["fragment"] =
            json!("void main(){ gl_FragColor = vec4(0.0, 0.0, 0.25, 0.5); }");
        let frame = render(&mut r, &loaded(&doc));
        // White over the half-transparent pass: opaque in the stem, the pass's
        // alpha beside it.
        assert!(alphas(&frame).contains(&255), "{how}");
        assert!(alphas(&frame)[0].abs_diff(128) <= 1, "{how}");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_text_set_after_a_reset_reaches_no_layer_of_the_dropped_scene() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let (_g, mut r) = gpu(64, 48);
    render(&mut r, &loaded(&layer_scene(SANS, "I", Align::Left, fit)));
    r.reset();
    set_words(&r, "label", "O");
}

// Effects

/// The frame of `doc` at time `t`.
fn render_at(r: &mut Renderer, sc: &Scene, t: f64) -> Vec<u8> {
    assert_eq!(r.setup(sc).unwrap(), "");
    r.set_time(Some(t));
    r.render(None).unwrap().to_vec()
}

/// A scene of layer `label`, in the fixture font SANS, its block where `fit`
/// puts it, over a black pass, with `keys` merged in.
fn effect_scene(words: &str, fit: Fit, keys: Value) -> Value {
    let mut doc = layer_scene(SANS, words, Align::Left, fit);
    let l = layer_of(&mut doc, "label");
    l["render"] = json!("mtsdf");
    l.as_object_mut()
        .unwrap()
        .extend(keys.as_object().unwrap().clone());
    doc
}

/// `top` over `under`, each a colour and an alpha, as the premultiplied
/// colour and the alpha of the two.
fn over(top: ([f64; 3], f64), under: ([f64; 3], f64)) -> ([f64; 3], f64) {
    let (c, a) = top;
    let (u, b) = under;
    (
        [0, 1, 2].map(|k| c[k] * a + u[k] * (1.0 - a)),
        a + b * (1.0 - a),
    )
}

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn the_effects_draw_by_the_distance_to_the_letters_in_their_order() {
    // An I, one rectangle, at 64 px/em over black. Each row draws effects
    // under red letters, and the reference composes them from the true
    // distance to the outline, d, and to the outline moved by the shadow's
    // offset, ds, both in em and below 0 inside: shadow, glow, stroke, then
    // the letters, each over the last.
    let words = "I";
    let laid = lay(SANS, words, Align::Left);
    let fit = Fit {
        x: 48.0,
        y: 16.0,
        ppem: 64.0,
    };
    let (w, h) = (160, 128);
    let px = 1.0 / fit.ppem;
    let shadow = move |d: f64, blur: f64, rgb: [f64; 3], strength: f64| {
        let b = blur.max(0.5 * px);
        (rgb, smoothstep(-b, b, -d) * strength)
    };
    let glow = |d: f64, blur: f64, rgb: [f64; 3], strength: f64| {
        let f = (1.0 - d.max(0.0) / blur).clamp(0.0, 1.0);
        (rgb, f * f * strength)
    };
    let stroke =
        move |d: f64, width: f64, rgb: [f64; 3]| (rgb, ((width - d) / px + 0.5).clamp(0.0, 1.0));
    // Each row: the effects, the shadow's offset, the stroke's width, where
    // the stroke's outer edge takes its antialiasing, and the reference.
    type Want = Box<dyn Fn(f64, f64) -> [f64; 3]>;
    let rows: Vec<(Value, [f64; 2], f64, Want)> = vec![
        (
            json!({"text_stroke": {"width": 0.1, "color": [0.0, 1.0, 0.0]}}),
            [0.0, 0.0],
            0.1,
            Box::new(move |d, _| {
                stroke(d, 0.1, [0.0, 1.0, 0.0])
                    .0
                    .map(|c| c * stroke(d, 0.1, [0.0; 3]).1)
            }),
        ),
        (
            json!({"glow": {"blur": 0.25, "color": [0.0, 1.0, 0.0], "strength": 1.0}}),
            [0.0, 0.0],
            0.0,
            Box::new(move |d, _| {
                let (c, a) = glow(d, 0.25, [0.0, 1.0, 0.0], 1.0);
                c.map(|c| c * a)
            }),
        ),
        (
            json!({"text_shadow": {"x": 0.25, "y": 0.125, "blur": 0.0,
                                   "color": [0.0, 0.0, 1.0], "strength": 1.0}}),
            [0.25, 0.125],
            0.0,
            Box::new(move |_, ds| {
                let (c, a) = shadow(ds, 0.0, [0.0, 0.0, 1.0], 1.0);
                c.map(|c| c * a)
            }),
        ),
        (
            json!({"text_shadow": {"x": 0.2, "y": 0.0, "blur": 0.15,
                                   "color": [0.0, 0.0, 1.0], "strength": 0.8},
                   "glow": {"blur": 0.3, "color": [0.0, 1.0, 0.0], "strength": 0.5},
                   "text_stroke": {"width": 0.05, "color": [1.0, 1.0, 1.0]}}),
            [0.2, 0.0],
            0.05,
            Box::new(move |d, ds| {
                // Each layer over the last, its colour premultiplied and so
                // over black.
                let s = over(shadow(ds, 0.15, [0.0, 0.0, 1.0], 0.8), ([0.0; 3], 0.0));
                let g = over(glow(d, 0.3, [0.0, 1.0, 0.0], 0.5), s);
                let k = over(stroke(d, 0.05, [1.0; 3]), g);
                k.0
            }),
        ),
    ];
    let (_g, mut r) = gpu(w, h);
    for (keys, off, width, want) in rows {
        let mut keys = keys;
        keys["color"] = json!([1.0, 0.0, 0.0]);
        let frame = render(&mut r, &loaded(&effect_scene(words, fit, keys.clone())));
        let (mut seen, mut off_px) = (0, Vec::new());
        for y in 0..h {
            for x in 0..w {
                let p = fit.em(x, y);
                let d = laid.distance(p);
                let ds = laid.distance([p[0] - off[0], p[1] - off[1]]);
                // The letters' own edge, and the edges of the effects, take a
                // pixel and a half of antialiasing.
                if d.abs() < EDGE * px
                    || ds.abs() < EDGE * px
                    || (width > 0.0 && (d - width).abs() < EDGE * px)
                {
                    continue;
                }
                // The field gives the distance within half a pixel, so each
                // channel lies between the references half a pixel nearer and
                // further.
                let q = 0.5 * px;
                let at = |e: f64| match d < 0.0 {
                    true => [1.0, 0.0, 0.0],
                    false => want(d + e, ds + e),
                };
                let (a, b) = (at(-q), at(q));
                seen += 1;
                for k in 0..3 {
                    let got = f64::from(frame[(y * w + x) * 4 + k]);
                    let (lo, hi) = (255.0 * a[k].min(b[k]), 255.0 * a[k].max(b[k]));
                    if got < lo - 3.0 || got > hi + 3.0 {
                        off_px.push((x, y, k, got, lo.round(), hi.round()));
                    }
                }
            }
        }
        assert!(seen > 10_000, "{keys}: {seen}");
        assert!(
            off_px.is_empty(),
            "{keys}: {} off, (x, y, channel, got, least, most) {:?}",
            off_px.len(),
            &off_px[..off_px.len().min(8)]
        );
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_text_layer_below_opacity_1_fades_its_letters_and_effects_as_one() {
    // Red letters over a green glow and a blue shadow, an I at 64 px/em, at
    // opacity 0.5 over grey. The reference composes the layer at opacity 1
    // from the true distance to the outline, as the effects test above does,
    // and lays it over the grey at half, so inside the letters the effects
    // take no part.
    let words = "I";
    let laid = lay(SANS, words, Align::Left);
    let fit = Fit {
        x: 48.0,
        y: 16.0,
        ppem: 64.0,
    };
    let (w, h) = (160, 128);
    let px = 1.0 / fit.ppem;
    let (off, opacity, grey) = ([0.2, 0.0], 0.5, 0.5);
    let mut doc = effect_scene(
        words,
        fit,
        json!({"color": [1.0, 0.0, 0.0], "opacity": opacity,
               "text_shadow": {"x": off[0], "y": off[1], "blur": 0.15,
                               "color": [0.0, 0.0, 1.0], "strength": 0.8},
               "glow": {"blur": 0.3, "color": [0.0, 1.0, 0.0], "strength": 0.5}}),
    );
    doc["passes"][0]["fragment"] = json!("void main(){ gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0); }");
    let want = |d: f64, ds: f64| -> [f64; 3] {
        let (c, a) = match d < 0.0 {
            true => ([1.0, 0.0, 0.0], 1.0),
            false => {
                let b = 0.15_f64.max(0.5 * px);
                let s = ([0.0, 0.0, 1.0], smoothstep(-b, b, -ds) * 0.8);
                let f = (1.0 - d / 0.3).clamp(0.0, 1.0);
                over(([0.0, 1.0, 0.0], f * f * 0.5), over(s, ([0.0; 3], 0.0)))
            }
        };
        c.map(|c| opacity * c + (1.0 - opacity * a) * grey)
    };
    let (_g, mut r) = gpu(w, h);
    let frame = render(&mut r, &loaded(&doc));
    let (mut inside, mut seen, mut off_px) = (0, 0, Vec::new());
    for y in 0..h {
        for x in 0..w {
            let p = fit.em(x, y);
            let d = laid.distance(p);
            let ds = laid.distance([p[0] - off[0], p[1] - off[1]]);
            if d.abs() < EDGE * px || ds.abs() < EDGE * px {
                continue;
            }
            let q = 0.5 * px;
            let (a, b) = (want(d - q, ds - q), want(d + q, ds + q));
            seen += 1;
            inside += usize::from(d < 0.0);
            for k in 0..3 {
                let got = f64::from(frame[(y * w + x) * 4 + k]);
                let (lo, hi) = (255.0 * a[k].min(b[k]), 255.0 * a[k].max(b[k]));
                if got < lo - 3.0 || got > hi + 3.0 {
                    off_px.push((x, y, k, got, lo.round(), hi.round()));
                }
            }
        }
    }
    assert!(
        seen > 10_000 && inside > 100,
        "{seen} seen, {inside} inside"
    );
    assert!(
        off_px.is_empty(),
        "{} off, (x, y, channel, got, least, most) {:?}",
        off_px.len(),
        &off_px[..off_px.len().min(8)]
    );
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn the_slug_render_draws_no_effect_that_reads_the_distance() {
    let fit = Fit {
        x: 8.0,
        y: 8.0,
        ppem: 48.0,
    };
    let (_g, mut r) = gpu(W, H);
    let plain = render(&mut r, &loaded(&layer_scene(SANS, "Ag", Align::Left, fit)));
    let mut doc = layer_scene(SANS, "Ag", Align::Left, fit);
    let l = layer_of(&mut doc, "label");
    for (k, v) in [
        ("text_shadow", json!({"strength": 1.0})),
        ("glow", json!({"strength": 1.0})),
        ("text_stroke", json!({"color": [0.0, 1.0, 0.0]})),
    ] {
        l[k] = v;
    }
    let sc = loaded(&doc);
    assert!(render(&mut r, &sc) == plain);
    assert_eq!(r.text_errors(), [] as [String; 0]);
    // The effects stay in the scene, and an MTSDF render draws them.
    assert_eq!(
        sc.to_dict()["passes"][0]["layers"][0]["glow"],
        json!({"strength": 1.0})
    );
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_bypassed_effect_draws_nothing_and_leaves_the_letters_alone() {
    let fit = Fit {
        x: 8.0,
        y: 8.0,
        ppem: 48.0,
    };
    let (_g, mut r) = gpu(W, H);
    let frame = |r: &mut Renderer, keys: Value| {
        let mut keys = keys;
        keys["color"] = json!([1.0, 0.5, 0.25]);
        render(r, &loaded(&effect_scene("Ag", fit, keys)))
    };
    // A bypassed glow or shadow, which keeps its reach, draws as one of no
    // strength.
    for k in ["glow", "text_shadow"] {
        let bypassed = frame(
            &mut r,
            json!({k: {"visible": false, "color": [0.0, 1.0, 0.0]}}),
        );
        let none = frame(&mut r, json!({k: {"strength": 0.0}}));
        assert!(bypassed == none, "{k}");
    }
    // A stroke within the least range draws the letters as no effect does:
    // the effects' copy of each glyph leaves the letters' edge alone. The
    // least range, 6 texels, holds a stroke of 2 texels of the layer's atlas.
    let plain = TextLayer::from_dict("a layer", &json!({"name": "t", "text": ""})).unwrap();
    let width = 1.5 / f64::from(plain.atlas_px_em(fit.ppem));
    let plain = frame(&mut r, json!({}));
    for keys in [
        json!({"text_stroke": {"visible": false, "width": width, "color": [0.0, 1.0, 0.0]}}),
        json!({"text_stroke": {"width": 0.0}}),
    ] {
        let got = frame(&mut r, keys.clone());
        let worst = got.iter().zip(&plain).map(|(a, b)| a.abs_diff(*b)).max();
        assert!(worst <= Some(1), "{keys}: {worst:?}");
    }
}

/// `laid` with each glyph's outline moved down by `dy(i)` em, glyph i
/// counted over the inked glyphs.
fn moved(laid: &Laid, dy: impl Fn(usize) -> f64) -> Laid {
    Laid {
        width: laid.width,
        height: laid.height,
        glyphs: laid
            .glyphs
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let d = dy(i);
                Outline {
                    origin: [g.origin[0], g.origin[1] + d],
                    lo: [g.lo[0], g.lo[1] + d],
                    hi: [g.hi[0], g.hi[1] + d],
                    segs: g.segs.iter().map(|s| s.map(|p| [p[0], p[1] + d])).collect(),
                }
            })
            .collect(),
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_wave_moves_each_letter_and_its_effects_a_tenth_of_a_wave_behind_the_one_before() {
    // At a quarter of a wave, the first I sits a whole height down, and the
    // second one height times the sine of 0.15 of a turn.
    let words = "II";
    let laid = lay(SANS, words, Align::Left);
    let fit = Fit {
        x: 16.0,
        y: 24.0,
        ppem: 64.0,
    };
    let (w, h) = (160, 128);
    let height = 0.25;
    let dy = |i: usize| height * (std::f64::consts::TAU * (0.25 - 0.1 * i as f64)).sin();
    let want = moved(&laid, dy);
    let keys = json!({"wave": {"height": height, "speed": 1.0},
                      "text_stroke": {"width": 0.1, "color": [0.0, 1.0, 0.0]},
                      "color": [1.0, 0.0, 0.0]});
    let (_g, mut r) = gpu(w, h);
    let frame = render_at(&mut r, &loaded(&effect_scene(words, fit, keys)), 0.25);
    assert_draws(&frame, w, 0, &want, fit, "the letters");
    // The stroke rings the moved letters.
    let px = 1.0 / fit.ppem;
    let (mut ring, mut off) = (0, Vec::new());
    for y in 0..h {
        for x in 0..w {
            let d = want.distance(fit.em(x, y));
            if d > EDGE * px && d < 0.1 - EDGE * px {
                ring += 1;
                let g = frame[(y * w + x) * 4 + 1];
                if g < 255 - TOL {
                    off.push((x, y, g));
                }
            }
        }
    }
    assert!(
        ring > 300 && off.is_empty(),
        "{ring} {:?}",
        &off[..off.len().min(8)]
    );
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn karaoke_fills_the_letters_up_to_the_sung_point_and_lifts_the_letter_being_sung() {
    // Half of the H is sung: it rises 0.15 em, and its fill reaches half its
    // advance, between its stems. The I waits on the line, unsung.
    let words = "HI";
    let laid = lay(SANS, words, Align::Left);
    let fit = Fit {
        x: 16.0,
        y: 24.0,
        ppem: 64.0,
    };
    let (w, h) = (160, 128);
    let built = drawn(SANS, words, Align::Left);
    let span = built.glyphs[0].span;
    let edge = span[0] + 0.5 * (span[1] - span[0]);
    let want = moved(&laid, |i| if i == 0 { -0.15 } else { 0.0 });
    for how in RENDERS {
        let keys = json!({"render": how, "color": [0.0, 0.0, 1.0],
                          "karaoke": {"sung": 0.5, "color": [1.0, 1.0, 1.0]}});
        let (_g, mut r) = gpu(w, h);
        let frame = render(&mut r, &loaded(&effect_scene(words, fit, keys)));
        assert_draws(&frame, w, 2, &want, fit, &format!("{how}: the letters"));
        let (mut seen, mut off) = ([0; 2], Vec::new());
        for y in 0..h {
            for x in 0..w {
                let p = fit.em(x, y);
                if want.inside(p, EDGE / fit.ppem) != Some(true)
                    || (p[0] - edge).abs() < EDGE / fit.ppem
                {
                    continue;
                }
                let sung = p[0] < edge;
                seen[usize::from(sung)] += 1;
                let got = &frame[(y * w + x) * 4..][..3];
                let rg = if sung { 255 } else { 0 };
                if got[0].abs_diff(rg) > TOL || got[1].abs_diff(rg) > TOL {
                    off.push((x, y, got.to_vec()));
                }
            }
        }
        assert!(seen.iter().all(|n| *n > 100), "{how}: {seen:?}");
        assert!(off.is_empty(), "{how}: {:?}", &off[..off.len().min(8)]);
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_scroll_moves_the_block_right_to_left_by_whole_pixels_and_fits_it_on_y() {
    // At each time, the scrolling block draws as a still one at the whole
    // pixel that the scroll gives: right of the pass at the start, then
    // speed em a second to the left, around again once it has left, and
    // still on its pixel days in.
    let words = "AV To ghee";
    let laid = lay(SANS, words, Align::Left);
    let (w, h) = (192, 64);
    let (fw, fh) = (w as f64, h as f64);
    let m = 0.05 * fw.min(fh);
    let speed = 3.0;
    let (_g, mut r) = gpu(w, h);
    // A block with no font_size takes the height of the pass alone, less the
    // margin, and sits centred on y.
    let fitted = (fh - 2.0 * m) / laid.height;
    for (keys, ppem, y) in [
        (json!({"font_size": 40, "y": 8, "x": 50}), 40.0, 8.0),
        (json!({}), fitted, (fh - laid.height * fitted) / 2.0),
    ] {
        for t in [0.0, 0.51, 2.03, 1.0e6 + 0.51] {
            let trip = fw + laid.width * ppem;
            let x = (fw - (t * speed * ppem).rem_euclid(trip)).floor();
            let fit = Fit {
                x,
                y: y + rounding(SANS, ppem),
                ppem,
            };
            let still = render(&mut r, &loaded(&layer_scene(SANS, words, Align::Left, fit)));
            let mut keys = keys.clone();
            keys["scroll"] = json!({"speed": speed});
            keys["render"] = json!("slug");
            let mut doc = layer_scene(SANS, words, Align::Left, fit);
            let l = layer_of(&mut doc, "label");
            for k in ["x", "y", "font_size"] {
                l.as_object_mut().unwrap().remove(k);
            }
            l.as_object_mut()
                .unwrap()
                .extend(keys.as_object().unwrap().clone());
            let frame = render_at(&mut r, &loaded(&doc), t);
            let worst = frame.iter().zip(&still).map(|(a, b)| a.abs_diff(*b)).max();
            assert!(worst <= Some(1), "{keys} at {t}: {worst:?}");
        }
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_blink_shows_the_layer_for_the_first_half_of_each_blink() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 40.0,
    };
    let (_g, mut r) = gpu(W, H);
    // The letters, and a colour glyph, which blinks with them.
    for (font, words) in [(SANS, "Ag"), (EMOJI, "\u{1f600}")] {
        let mut doc = layer_scene(font, words, Align::Left, fit);
        let still = render(&mut r, &loaded(&doc));
        let mut none = doc.clone();
        none["passes"][0]["layers"] = json!([]);
        let empty = render(&mut r, &loaded(&none));
        layer_of(&mut doc, "label")["blink"] = json!({"rate": 2.0});
        let sc = loaded(&doc);
        assert!(render_at(&mut r, &sc, 0.1) == still, "{font} shown");
        assert!(render_at(&mut r, &sc, 0.3) == empty, "{font} hidden");
        assert!(render_at(&mut r, &sc, 0.6) == still, "{font} shown again");
    }
}

// The MTSDF render

/// The threads of this process named `name`.
fn threads_named(name: &str) -> usize {
    fs::read_dir("/proc/self/task")
        .unwrap()
        .flatten()
        .filter(|t| fs::read_to_string(t.path().join("comm")).is_ok_and(|c| c.trim_end() == name))
        .count()
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn only_an_mtsdf_layer_that_shows_starts_an_atlas() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let (_g, mut r) = gpu(64, 48);
    let cache = tmpdir("no-atlas");
    std::env::set_var("XDG_CACHE_HOME", &cache);
    // A slug layer, and a hidden MTSDF layer read through _cover.
    let slug = layer_scene(SANS, "I", Align::Left, fit);
    let hidden = label_scene(SANS, "I", Align::Left, fit);
    for doc in [&slug, &hidden] {
        render(&mut r, &loaded(doc));
        assert_eq!(threads_named("ghee-atlas"), 0);
        assert_eq!(fs::read_dir(&cache).unwrap().count(), 0);
    }
    render(&mut r, &loaded(&rendered(slug, "mtsdf")));
    // The worker's glyph threads inherit its name, so more than one can list.
    assert!(threads_named("ghee-atlas") >= 1);
    // The worker writes the cache after the glyphs are in.
    let file = std::path::Path::new(&cache).join("ghee/atlas");
    let t0 = std::time::Instant::now();
    while fs::read_dir(&file).map_or(0, |d| d.count()) == 0 {
        assert!(t0.elapsed().as_secs() < 20, "no cache file");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    drop(r);
    // A joined thread can stay listed under /proc for a moment after the join.
    let t0 = std::time::Instant::now();
    while threads_named("ghee-atlas") > 0 {
        assert!(
            t0.elapsed().as_secs() < 5,
            "the atlas worker outlived its renderer"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    fs::remove_dir_all(&cache).unwrap();
}

/// The lines of the main of a fragment shader: its last `void main()` to the
/// end.
fn own_main(fs: &str) -> Vec<&str> {
    let lines: Vec<&str> = fs.lines().collect();
    let at = lines
        .iter()
        .rposition(|l| l.starts_with("void main()"))
        .expect("a main");
    lines[at..].to_vec()
}

/// A line that reads a texture in an if the compiler cannot drop.
fn guarded(line: &str) -> bool {
    let l = line.trim_start();
    let read = l.contains("texture2D(");
    read && (l.starts_with("if (gl_FragCoord.x < 0.0) ")
        || l.starts_with("if (gl_FragCoord.x >= 0.0) ")
        || l.starts_with("if (ghee_v1.w < 0.5) "))
}

#[test]
fn every_layer_shader_that_ghee_writes_runs_on_one_thread_on_vc4() {
    // VC4 runs a short fragment shader whose last texture read sits outside
    // an if in two threads, and the switch loses the condition flags. Each
    // row is one layer, and every fragment shader of each of its batches that
    // reads a texture reads it last in such an if, as does the program of a
    // run of a text and other layers.
    let effects = [
        json!({}),
        json!({"text_shadow": {}}),
        json!({"glow": {}}),
        json!({"text_stroke": {}}),
        json!({"karaoke": {}}),
        json!({"wave": {}}),
        json!({"scroll": {}}),
        json!({"blink": {}}),
        json!({"text_shadow": {}, "glow": {}, "text_stroke": {}}),
        json!({"text_shadow": {}, "glow": {"visible": false}, "text_stroke": {},
               "karaoke": {}, "wave": {}, "scroll": {}, "blink": {}}),
    ];
    let mut rows = Vec::new();
    for e in effects {
        for render in ["mtsdf", "slug", "hinted", "mono"] {
            let mut keys = json!({"name": "t", "text": "A", "render": render});
            keys.as_object_mut()
                .unwrap()
                .extend(e.as_object().unwrap().clone());
            rows.push(keys);
        }
    }
    rows.extend([
        json!({"name": "i", "image": "pic"}),
        json!({"name": "i", "image": "pic.png", "color": "#ff8000", "width": 8}),
        json!({"name": "i", "image": "pic", "height": 8, "visible": false}),
        json!({"name": "r", "shape": "rect"}),
        json!({"name": "l", "shape": "line", "points": [[0, 0], [4, 4]]}),
    ]);
    let mut all = vec![(json!("a run"), vec![("runs", ghee::run::program().1)])];
    for keys in rows {
        let programs = match Layer::from_dict("a layer", &keys).unwrap() {
            Layer::Text(l) => {
                let sh = l.shaders(DETAIL);
                vec![
                    ("glyphs", sh.glyphs.1),
                    ("colour glyphs", sh.colour.1),
                    ("error boxes", sh.boxes.1),
                ]
            }
            Layer::Image(l) => vec![("image", l.shaders().1)],
            // Every shape draws through one program, which reads no texture.
            Layer::Shape(_) => vec![("shapes", ghee::shape::program().1)],
        };
        all.push((keys, programs));
    }
    for (keys, programs) in all {
        for (what, fs) in programs {
            let main = own_main(&fs);
            let reads: Vec<usize> = (0..main.len())
                .filter(|i| main[*i].contains("texture2D("))
                .collect();
            assert!(
                reads.iter().all(|i| guarded(main[*i])),
                "{keys} {what}: {main:#?}"
            );
            if fs.contains("texture2D(") {
                let lines: Vec<&str> = fs.lines().collect();
                let last = lines
                    .iter()
                    .rposition(|l| l.contains("texture2D("))
                    .unwrap();
                assert!(guarded(lines[last]), "{keys} {what}");
            }
        }
    }
}

#[test]
fn every_filter_that_ghee_ships_runs_on_one_thread_on_vc4() {
    // Each filter that ghee ships reads its last texture in an if that the
    // compiler cannot drop, so VC4 runs it on one thread.
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/ghee/filters");
    let mut files: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".filter.json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no filter in {dir}");
    for path in files {
        let fd: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let fragment = fd["fragment"].as_str().unwrap();
        if let Some(last) = fragment.lines().rfind(|l| l.contains("texture2D(")) {
            assert!(guarded(last), "{}: {last}", path.display());
        }
    }
}

// Colour glyphs

/// A colour bitmap of a fixture font, at its largest strike.
struct Bitmap {
    /// The bottom left corner in pixels from the glyph's origin, y up.
    x: f64,
    y: f64,
    w: usize,
    h: usize,
    /// Pixels per em.
    ppem: f64,
    /// RGBA premultiplied by alpha, rows from the top.
    px: Vec<[u8; 4]>,
}

fn bitmap(file: &str, ch: char) -> Bitmap {
    let data = font(file);
    let face = Face::parse(&data, 0).unwrap();
    let img = face
        .glyph_raster_image(face.glyph_index(ch).unwrap(), u16::MAX)
        .unwrap();
    let mut d = png::Decoder::new(img.data);
    d.set_transformations(png::Transformations::EXPAND);
    let mut reader = d.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let n = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        c => panic!("{c:?}"),
    };
    let px = buf[..info.buffer_size()]
        .chunks_exact(n)
        .map(|p| {
            let a = if n == 4 { p[3] } else { 255 };
            let m = |c: u8| ((u32::from(c) * u32::from(a) + 127) / 255) as u8;
            [m(p[0]), m(p[1]), m(p[2]), a]
        })
        .collect();
    Bitmap {
        x: f64::from(img.x),
        y: f64::from(img.y),
        w: info.width as usize,
        h: info.height as usize,
        ppem: f64::from(img.pixels_per_em),
        px,
    }
}

/// The fit that draws colour glyph `k` of `built` at 1 / `shrink` of its
/// bitmap's pixels per em, the top left of the bitmap on the pixel corner
/// `corner`, and that pixel.
fn bitmap_fit(
    built: &text::Built,
    k: usize,
    b: &Bitmap,
    shrink: f64,
    corner: [f64; 2],
) -> (Fit, [usize; 2]) {
    let o = built.colour[k].origin;
    let em = [o[0] + b.x / b.ppem, o[1] - (b.y + b.h as f64) / b.ppem];
    let ppem = b.ppem / shrink;
    let fit = Fit {
        x: corner[0] - em[0] * ppem,
        y: corner[1] - em[1] * ppem,
        ppem,
    };
    (fit, corner.map(|v| v as usize))
}

/// The RGBA `frame`, w pixels wide, shows bitmap `b` with its top left at
/// pixel `at`, shrunk by `shrink` as each `shrink` x `shrink` block of its
/// pixels averaged, over the opaque colour `under`.
fn assert_bitmap(
    frame: &[u8],
    w: usize,
    b: &Bitmap,
    at: [usize; 2],
    shrink: usize,
    under: [u8; 3],
    what: &str,
) {
    let tol = if shrink == 1 { 2 } else { 4 };
    let (mut worst, mut off) = (0, Vec::new());
    for j in 0..b.h / shrink {
        for i in 0..b.w / shrink {
            let mut sum = [0u32; 4];
            for dy in 0..shrink {
                for dx in 0..shrink {
                    let p = b.px[(j * shrink + dy) * b.w + i * shrink + dx];
                    for c in 0..4 {
                        sum[c] += u32::from(p[c]);
                    }
                }
            }
            let n = (shrink * shrink) as f64;
            let a = f64::from(sum[3]) / n;
            let got = &frame[((at[1] + j) * w + at[0] + i) * 4..][..3];
            for c in 0..3 {
                let want = f64::from(sum[c]) / n + f64::from(under[c]) * (1.0 - a / 255.0);
                let d = (f64::from(got[c]) - want).abs();
                worst = worst.max(d.round() as u32);
                if d > f64::from(tol) {
                    off.push((i, j, c, got[c], want.round()));
                }
            }
        }
    }
    assert!(
        off.is_empty(),
        "{what}: {} channels off, worst {worst}, (x, y, c, got, want) {:?}",
        off.len(),
        &off[..off.len().min(8)]
    );
}

/// A scene of layer `label` in the colour font `file`, the asset `face`,
/// placed by `fit` over a black pass.
fn colour_scene(file: &str, words: &str, fit: Fit) -> Value {
    let mut doc = label_scene(file, words, Align::Left, fit);
    doc["passes"][0] =
        json!({"name": "main", "fragment": BLACK, "layers": [placed(file, words, fit)]});
    doc
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_paints_its_colour_glyphs_in_their_own_colours() {
    let words = "\u{1f600}";
    let (w, h) = (220, 150);
    let (_g, mut r) = gpu(w, h);
    // The sbix font carries a strike of flat green beside the bitmaps, and
    // the largest strike draws.
    for file in [EMOJI, EMOJI_SBIX] {
        let laid = text::build(&alone(&font(file)), words, Align::Left);
        let built = laid.built.as_ref().unwrap();
        let b = bitmap(file, '\u{1f600}');
        assert_eq!(b.ppem, 109.0);
        let (fit, at) = bitmap_fit(built, 0, &b, 1.0, [8.0, 6.0]);
        for how in RENDERS {
            let what = format!("{file} {how}");
            let doc = rendered(colour_scene(file, words, fit), how);
            let frame = render(&mut r, &loaded(&doc));
            assert_bitmap(&frame, w, &b, at, 1, [0, 0, 0], &what);
            assert_eq!(r.text_errors(), [] as [String; 0], "{what}");
        }
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_colour_glyph_shrinks_through_its_mipmaps() {
    // At a quarter of its strike, each pixel of the bitmap's box shows the
    // average of a 4 x 4 block of the bitmap: mip level 2.
    let words = "\u{1f600}";
    let laid = text::build(&alone(&font(EMOJI)), words, Align::Left);
    let b = bitmap(EMOJI, '\u{1f600}');
    let (fit, at) = bitmap_fit(laid.built.as_ref().unwrap(), 0, &b, 4.0, [5.0, 3.0]);
    let (_g, mut r) = gpu(48, 40);
    let frame = render(&mut r, &loaded(&colour_scene(EMOJI, words, fit)));
    assert_bitmap(&frame, 48, &b, at, 4, [0, 0, 0], "a quarter");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_colour_glyph_takes_its_layers_place_and_opacity() {
    let words = "\u{1f600}";
    let fit = Fit {
        x: 4.0,
        y: 2.0,
        ppem: 40.0,
    };
    let (w, h) = (128, 56);
    let (_g, mut r) = gpu(w, h);
    let still = render(&mut r, &loaded(&colour_scene(EMOJI, words, fit)));
    let moved = Fit {
        x: fit.x + 32.0,
        ..fit
    };
    let mut doc = colour_scene(EMOJI, words, moved);
    let moved = render(&mut r, &loaded(&doc));
    layer_of(&mut doc, "label")["opacity"] = json!(0.5);
    let faded = render(&mut r, &loaded(&doc));
    let px = |f: &[u8], x: usize, y: usize| -> [u8; 3] {
        let at = (y * w + x) * 4;
        [f[at], f[at + 1], f[at + 2]]
    };
    let (mut n, mut off) = (0, 0);
    for y in 0..h {
        for x in 0..w - 32 {
            let a = px(&still, x, y);
            assert!(px(&moved, x, y) == [0; 3] || x >= 32, "({x}, {y})");
            if a == [0; 3] {
                continue;
            }
            n += 1;
            off += usize::from(px(&moved, x + 32, y) != a);
            let half = px(&faded, x + 32, y);
            for k in 0..3 {
                off += usize::from((f64::from(half[k]) - f64::from(a[k]) / 2.0).abs() > 1.0);
            }
        }
    }
    assert!(n > 500 && off == 0, "{n} coloured pixels, {off} off");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_colour_glyph_blends_the_pass_alpha_as_its_colour() {
    let words = "\u{1f600}";
    let fit = Fit {
        x: 2.0,
        y: 2.0,
        ppem: 40.0,
    };
    let (_g, mut r) = gpu(56, 56);
    r.set_alpha(true);
    let mut doc = colour_scene(EMOJI, words, fit);
    doc["passes"][0]["fragment"] =
        json!("void main(){ gl_FragColor = vec4(0.0, 0.0, 0.25, 0.5); }");
    let frame = render(&mut r, &loaded(&doc));
    assert!(plane(&frame, 0).iter().any(|v| *v > 200), "no emoji drawn");
    assert!(alphas(&frame).contains(&255));
    assert!(alphas(&frame)[0].abs_diff(128) <= 1);
}

// The emoji fallback

/// A directory that holds EMOJI under the file name of the emoji font, for
/// the fallback to find through $GHEE_ASSETS. The drop removes it.
struct EmojiDir(String);

impl EmojiDir {
    fn new(tag: &str) -> EmojiDir {
        let d = tmpdir(tag);
        fs::copy(
            format!("{FONTS}/{EMOJI}"),
            format!("{d}/{}", ghee::emoji::FILE),
        )
        .unwrap();
        EmojiDir(d)
    }

    /// `doc` loaded as loaded() loads it, with its emoji font found here.
    /// The caller holds LOCK.
    fn load(&self, doc: &Value) -> Scene {
        with_assets_env(&self.0, || loaded(doc))
    }
}

impl Drop for EmojiDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A scene of one red text layer `label` in DejaVu Sans, which lacks
/// every emoji, of `words` placed by `fit` over a black pass, as placed()
/// places a layer, with `keys` merged in.
fn fallback_scene(words: &str, fit: Fit, keys: Value) -> Value {
    let mut l = json!({"name": "label", "text": words, "x": fit.x,
                       "y": fit.y - rounding(LATIN, fit.ppem), "font_size": fit.ppem,
                       "color": "#ff0000"});
    l.as_object_mut()
        .unwrap()
        .extend(keys.as_object().unwrap().clone());
    json!({"version": 1, "passes": [{"name": "main", "fragment": BLACK, "layers": [l]}]})
}

/// The layout of `words` in DejaVu Sans, then EMOJI.
fn fallen_back(words: &str) -> text::Built {
    let fonts = vec![Ok(Arc::new(font(LATIN))), Ok(Arc::new(font(EMOJI)))];
    text::build(&fonts, words, Align::Left).built.unwrap()
}

/// The largest difference of a channel between two frames.
fn worst(a: &[u8], b: &[u8]) -> u8 {
    a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap()
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn an_emoji_that_the_font_lacks_keeps_its_colours_and_moves_as_the_letters_do() {
    let words = "\u{1f600}";
    let b = bitmap(EMOJI, '\u{1f600}');
    let built = fallen_back(words);
    assert_eq!(built.colour[0].font, 1);
    let (fit, at) = bitmap_fit(&built, 0, &b, 1.0, [40.0, 30.0]);
    let (w, h) = (260, 200);
    let emoji = EmojiDir::new("fallback-motion");
    let (_g, mut r) = gpu(w, h);
    let still = render(&mut r, &emoji.load(&fallback_scene(words, fit, json!({}))));
    assert_bitmap(&still, w, &b, at, 1, [0, 0, 0], "the red layer's emoji");
    let mut none = fallback_scene(words, fit, json!({}));
    none["passes"][0]["layers"] = json!([]);
    let empty = render(&mut r, &loaded(&none));
    // Karaoke lifts the character being sung by 0.6 k (1 - k) em, k the part
    // of it that is sung: 15 pixels here.
    let k = (1.0 - (1.0 - 4.0 * 15.0 / (0.6 * fit.ppem)).sqrt()) / 2.0;
    // A scroll puts the block's left edge on a whole pixel, speed em a second
    // to the left of the pass's right edge.
    let advance = built.size[0];
    let (t, speed) = (1.5, 1.0);
    let trip = w as f64 + advance * fit.ppem;
    let scrolled = (w as f64 - (t * speed * fit.ppem).rem_euclid(trip)).floor();
    // The keys, the time, and where the emoji sits then, as the pixels that
    // its layer moved by, or nowhere.
    let rows = [
        (
            json!({"x": fit.x + 12.0, "y": fit.y - 9.0 - rounding(LATIN, fit.ppem)}),
            0.0,
            Some([12.0, -9.0]),
        ),
        (
            json!({"wave": {"height": 10.0 / fit.ppem, "speed": 1.0}}),
            0.25,
            Some([0.0, 10.0]),
        ),
        (json!({"karaoke": {"sung": k}}), 0.0, Some([0.0, -15.0])),
        (
            json!({"scroll": {"speed": speed}}),
            t,
            Some([scrolled - fit.x, 0.0]),
        ),
        (json!({"blink": {"rate": 2.0}}), 0.1, Some([0.0, 0.0])),
        (json!({"blink": {"rate": 2.0}}), 0.3, None),
        (json!({"visible": false}), 0.0, None),
    ];
    for (keys, t, moved) in rows {
        let frame = render_at(
            &mut r,
            &emoji.load(&fallback_scene(words, fit, keys.clone())),
            t,
        );
        let want = match moved {
            None => empty.clone(),
            Some([dx, dy]) => {
                let there = Fit {
                    x: fit.x + dx,
                    y: fit.y + dy,
                    ..fit
                };
                render(
                    &mut r,
                    &emoji.load(&fallback_scene(words, there, json!({}))),
                )
            }
        };
        assert!(
            worst(&frame, &want) <= 1,
            "{keys} at {t}: {}",
            worst(&frame, &want)
        );
    }
    let faded = render(
        &mut r,
        &emoji.load(&fallback_scene(words, fit, json!({"opacity": 0.5}))),
    );
    let half: Vec<u8> = still
        .iter()
        .map(|v| (f64::from(*v) / 2.0).round() as u8)
        .collect();
    let rgb = |f: &[u8]| -> Vec<u8> {
        f.as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| p[..3].to_vec())
            .collect()
    };
    assert!(worst(&rgb(&faded), &rgb(&half)) <= 1);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn an_emoji_takes_no_shadow_glow_or_stroke_beside_letters_that_do() {
    // The emoji, then an I two spaces to its right, out of reach of the
    // effects of the I.
    let words = "\u{1f600}  I";
    let b = bitmap(EMOJI, '\u{1f600}');
    let (fit, at) = bitmap_fit(&fallen_back(words), 0, &b, 1.0, [8.0, 8.0]);
    let (w, h) = (300, 150);
    let emoji = EmojiDir::new("fallback-effects");
    let (_g, mut r) = gpu(w, h);
    let plain = render(&mut r, &emoji.load(&fallback_scene(words, fit, json!({}))));
    let shadow = json!({"x": 0.05, "y": 0.05, "blur": 0.02, "color": "#00ff00"});
    let glow = json!({"blur": 0.1, "color": "#00ff00"});
    let stroke = json!({"width": 0.03, "color": "#00ff00"});
    for keys in [
        json!({"text_shadow": shadow}),
        json!({"glow": glow}),
        json!({"text_stroke": stroke}),
        json!({"text_shadow": shadow, "glow": glow, "text_stroke": stroke}),
    ] {
        let frame = render(
            &mut r,
            &emoji.load(&fallback_scene(words, fit, keys.clone())),
        );
        assert_bitmap(&frame, w, &b, at, 1, [0, 0, 0], &keys.to_string());
        assert!(
            frame.as_chunks::<4>().0.iter().any(|p| p[1] > 100),
            "{keys}: no effect drawn"
        );
        // Outside the I and its effects, the frame is the plain one.
        let edge = (at[0] + b.w + 8) * 4;
        let left = |f: &[u8]| -> Vec<u8> {
            f.chunks_exact(w * 4)
                .flat_map(|row| row[..edge].to_vec())
                .collect()
        };
        assert!(worst(&left(&frame), &left(&plain)) <= 1, "{keys}");
    }
}

// The hinted renders

/// The PNG `file` of tools/fixtures/hinted, one grey byte a pixel, and its
/// width.
fn pillow_png(file: &str) -> (usize, Vec<u8>) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/hinted");
    let reader = png::Decoder::new(fs::File::open(format!("{dir}/{file}")).unwrap());
    let mut reader = reader.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!(info.color_type, png::ColorType::Grayscale, "{file}");
    buf.truncate(info.buffer_size());
    (info.width as usize, buf)
}

/// One black pass of a text layer in DejaVu Sans: the words of
/// tools/fixtures/hinted/pillow.py at `size` px at (1, `y`), and `keys` on
/// top.
fn hinted_scene(size: f64, y: f64, keys: Value) -> Scene {
    let mut layer = json!({"name": "t", "text": "cpu 42% ping 12ms", "font_size": size,
                           "x": 1.0, "y": y});
    layer
        .as_object_mut()
        .unwrap()
        .extend(keys.as_object().unwrap().clone());
    loaded(&json!({"version": 1, "passes": [{"name": "main",
        "fragment": "void main() { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
        "layers": [layer]}]}))
}

/// A hinted layer of DejaVu Sans at (x, y) lights the pixels that
/// Pillow lights with the same font at the same (x, y) and its default
/// anchor: FreeType's grey coverage under `hinted`, its 1-bit render under
/// `mono`, each glyph at Pillow's pen and on its baseline.
/// tools/fixtures/hinted/pillow.py drew the references with Pillow 12.3.0,
/// whose FreeType is ghee's, 2.14.3. A black shadow under the letters leaves
/// their pixels as they are, so a layer with effects draws its letters from
/// the same bitmaps.
#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_hinted_layer_lights_the_pixels_that_pillow_lights() {
    let (_g, mut r) = gpu(192, 24);
    let shadow = json!({"text_shadow": {"x": 0, "y": 0, "blur": 0, "color": "black",
                                        "strength": 1}});
    for (size, y, kind, file) in [
        (8.0, 10.0, "hinted", "dejavu-8-grey.png"),
        (12.0, 6.0, "hinted", "dejavu-12-grey.png"),
        (10.0, 8.0, "mono", "dejavu-10-mono.png"),
        (16.0, 3.0, "mono", "dejavu-16-mono.png"),
    ] {
        let (w, want) = pillow_png(file);
        assert_eq!(w, 192);
        let mut shadowed = shadow.clone();
        shadowed["render"] = json!(kind);
        for keys in [json!({"render": kind}), shadowed] {
            let frame = render(&mut r, &hinted_scene(size, y, keys.clone()));
            let got: Vec<u8> = frame.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
            let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert_eq!(
                differ, 0,
                "{file} {keys}: {differ} pixels differ from Pillow's"
            );
        }
    }
}

/// `auto`, the default, draws a text below HINTED_BELOW px as `hinted` does,
/// and from it as `mtsdf` does.
#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn auto_draws_hinted_below_the_switch_and_mtsdf_from_it() {
    let (_g, mut r) = gpu(192, 24);
    let below = ghee::layer::HINTED_BELOW - 1.0;
    let at = ghee::layer::HINTED_BELOW;
    for (size, same) in [(below, "hinted"), (at, "mtsdf")] {
        let y = 18.0 - size;
        let auto = render(&mut r, &hinted_scene(size, y, json!({})));
        let named = render(&mut r, &hinted_scene(size, y, json!({"render": same})));
        assert!(auto == named, "auto at {size} px draws unlike {same}");
        assert!(
            auto.as_chunks::<4>().0.iter().any(|p| p[0] > 0),
            "{size} px draws nothing"
        );
    }
}
