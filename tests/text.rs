//! Scene texts: the layout, the scene model, the font assets, and the
//! coverage that passes and layers render on this host's GPU.
//!
//! The reference below rebuilds the promised layout and outline from the font
//! with ttf-parser, so every check holds against any packing of the texture.

use std::fs;
use std::sync::{Arc, Mutex, MutexGuard};

use serde_json::{json, Map, Value};
use ttf_parser::{Face, GlyphId, OutlineBuilder};

use ghee::engine::{probe_dri_devices, ErrorLayer, ErrorPass, LayerPart, Renderer};
use ghee::filters::apply_filters;
use ghee::scene::{Param, ParamValue, Pass, Reach, Scene, SceneBuilder};
use ghee::text::{self, Align, Fill, Text};

const FONTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts");
/// DejaVu Sans cut to printable ASCII, TrueType outlines, a kern table.
const SANS: &str = "DejaVuSans-ASCII.ttf";
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
/// DejaVu Sans cut by fontTools' subset to space, U+2764, U+FE0E and U+FE0F:
/// an outline heart, and two selectors with no advance and no ink.
const HEART: &str = "DejaVuSans-Heart.ttf";
/// Noto Color Emoji cut by fontTools' subset to space and U+2764: a colour
/// heart.
const EMOJI_HEART: &str = "NotoColorEmoji-Heart.ttf";

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

/// One glyph's outline as straight segments, and their box, in em from the
/// block's top left, y down.
struct Outline {
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
            "{name}_cover((v_texCoord * resolution - vec2({:.4}, {:.4})) / {:.4}, {:.4})",
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

/// One pass that draws the coverage of text `label` in the font asset `face`,
/// the fixture `file`, to red, green and blue.
fn label_scene(file: &str, words: &str, align: Align, fit: Fit) -> Value {
    json!({
        "version": 1,
        "assets": {"face": {"file": file}},
        "texts": {"label": {"value": words, "font": "face", "align": align_name(align)}},
        "passes": [{"name": "main", "code": format!(
            "void main(){{ float c = {}; gl_FragColor = vec4(c, c, c, 1.0); }}",
            fit.cover("label"))}],
    })
}

/// The scene with its fonts read from the fixtures. The caller holds LOCK.
fn loaded(doc: &Value) -> Scene {
    let mut sc = Scene::from_dict(doc, None, None).unwrap();
    sc.load_assets(FONTS).unwrap();
    sc
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
    let mut t = Text::from_dict("label", &json!({"font": "face"})).unwrap();
    t.data = Some(vec![Err("asset 'face' (face.ttf): gone".to_string())]);
    let laid = t.build("I O").unwrap();
    assert_eq!(
        laid.faults,
        [text::Fault::Font(
            "asset 'face' (face.ttf): gone".to_string()
        )]
    );
    assert_eq!(laid.built.unwrap().boxes.count, 2);
}

#[test]
fn a_text_built_before_its_font_loads_names_the_font() {
    let t = Text::from_dict("label", &json!({"font": "face"})).unwrap();
    let e = t.build("I").err().unwrap().to_string();
    assert_eq!(
        e,
        "text 'label': font 'face' was never loaded, the scene was built without a directory"
    );
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
    // One line of glyphs, far past what one texture holds.
    let e = refused(&"I".repeat(400_000));
    assert!(e.contains("VC4"), "{e}");
}

#[test]
fn a_block_too_wide_and_too_tall_for_the_texture_lays_out_nothing_naming_the_limit() {
    // Few glyphs over a vast block: its grid alone would take gigabytes.
    let e = refused(&("I".repeat(30_000) + &"\n".repeat(30_000)));
    assert!(e.contains("VC4"), "{e}");
}

// The scene model

/// The fixture fonts as the scene assets `sans` and `cff`.
fn font_assets() -> Value {
    json!({"sans": {"file": SANS}, "cff": {"file": CFF}})
}

/// A scene that declares `texts`, each with the words "I" ahead of its keys,
/// and the fixture fonts.
fn texts_scene(texts: Value) -> Value {
    let mut words = Map::new();
    for (name, t) in texts.as_object().unwrap() {
        let Some(keys) = t.as_object() else {
            words.insert(name.clone(), t.clone());
            continue;
        };
        let mut d = Map::new();
        d.insert("value".to_string(), json!("I"));
        d.extend(keys.clone());
        words.insert(name.clone(), Value::Object(d));
    }
    json!({
        "version": 1,
        "assets": font_assets(),
        "texts": words,
        "passes": [{"name": "main", "code": "void main(){ gl_FragColor = vec4(1.0); }"}],
    })
}

fn scene_err(doc: &Value) -> String {
    Scene::from_dict(doc, None, None).err().unwrap().to_string()
}

#[test]
fn texts_round_trip_and_left_is_the_default() {
    let doc = texts_scene(json!({
        "label": {"font": "sans", "align": "center"},
        "title": {"font": "cff"},
        "tail": {"font": "sans", "align": "right"},
    }));
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    assert_eq!(sc.to_dict(), doc);
    let names: Vec<&str> = sc.texts.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["label", "title", "tail"]);
    assert_eq!(sc.texts[0].align, Align::Center);
    assert_eq!(sc.texts[1].align, Align::Left);
    for left in [json!("left"), Value::Null] {
        let sc = Scene::from_dict(
            &texts_scene(json!({"label": {"font": "sans", "align": left}})),
            None,
            None,
        )
        .unwrap();
        assert_eq!(sc.texts[0].align, Align::Left);
        assert_eq!(
            sc.to_dict()["texts"],
            json!({"label": {"value": "I", "font": "sans"}})
        );
    }
    // MTSDF is the default fill, at 32 px/em and a range of 6 with no layer,
    // and a save leaves the defaults out.
    assert_eq!(
        (
            sc.texts[0].fill,
            sc.texts[0].px_per_em,
            sc.atlas_range("label")
        ),
        (Fill::Mtsdf, 32, 6)
    );
    let doc = texts_scene(json!({
        "label": {"font": "sans", "fill": "mtsdf", "atlas": {"px_per_em": 32}},
        "title": {"font": "sans", "fill": "slug"},
        "tail": {"font": "sans", "atlas": {"px_per_em": 48}},
    }));
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    assert_eq!(
        sc.to_dict()["texts"],
        json!({
            "label": {"value": "I", "font": "sans"},
            "title": {"value": "I", "font": "sans", "fill": "slug"},
            "tail": {"value": "I", "font": "sans", "atlas": {"px_per_em": 48}},
        })
    );
    assert_eq!(sc.texts[1].fill, Fill::Slug);
    assert_eq!(sc.texts[2].px_per_em, 48);
    assert_eq!(
        Scene::from_dict(&sc.to_dict(), None, None).unwrap(),
        sc,
        "a save reloads the same texts"
    );
}

#[test]
fn a_text_declares_the_string_param_of_its_name_and_refuses_another() {
    let doc = json!({
        "version": 1, "assets": font_assets(),
        "texts": {"label": {"value": "hi", "font": "sans"}, "title": {"font": "sans"}},
        "params": {"gain": {"type": "float", "value": 1.0}},
        "passes": [{"name": "main", "code": "void main(){}"}],
    });
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    let words = |sc: &Scene, n: &str| sc.params.get(n).map(|p| p.value.clone());
    let str = |w: &str| Some(ParamValue::Str(w.to_string()));
    assert_eq!(words(&sc, "label"), str("hi"));
    assert_eq!(words(&sc, "title"), str(""));
    // A save leaves the param out and writes the words as they stand into the
    // text, and a scene built again keeps them.
    let mut sc = Ok(sc).param(Param::float("k", 0.5)).unwrap();
    sc.params.get_mut("label").unwrap().value = ParamValue::Str("ho".to_string());
    let sc = Ok(sc).param(Param::float("j", 0.5)).unwrap();
    let back = sc.to_dict();
    assert_eq!(back["params"].as_object().unwrap().len(), 3);
    assert_eq!(
        back["texts"]["label"],
        json!({"value": "ho", "font": "sans"})
    );
    assert_eq!(back["texts"]["title"], json!({"value": "", "font": "sans"}));
    assert_eq!(Scene::from_dict(&back, None, None).unwrap(), sc);
    let none = Scene::from_dict(&texts_scene(json!({"label": {"font": "sans"}})), None, None);
    assert!(none.unwrap().to_dict().get("params").is_none());
    // A param under the text's name is refused, of any type and in any pass.
    let scene = "scene param 'label' has the name of text 'label', which holds its words in \
                 its 'value'";
    let pass = "param 'label' in pass 'main' has the name of text 'label', which holds its \
                words in its 'value'";
    for (at, ty, want) in [
        ("scene", "string", scene),
        ("scene", "float", scene),
        ("pass", "string", pass),
    ] {
        let mut doc = texts_scene(json!({"label": {"font": "sans"}}));
        let p = json!({"label": {"type": ty, "value": if ty == "string" { json!("I") } else { json!(1.0) }}});
        match at {
            "scene" => doc["params"] = p,
            _ => doc["passes"][0]["params"] = p,
        }
        assert_eq!(scene_err(&doc), format!("scene: {want}"), "{at} {ty}");
    }
    let mut doc = texts_scene(json!({"label": {"font": "sans"}}));
    doc["texts"]["label"]["value"] = json!(3);
    assert_eq!(
        scene_err(&doc),
        "scene: text 'label' value needs words in a string, got 3"
    );
}

#[test]
fn a_text_declaration_names_what_it_lacks() {
    for (td, want) in [
        (json!("x.ttf"), "text 'label' needs an object"),
        (json!({}), "text 'label' has no 'font'"),
        (
            json!({"font": ""}),
            "text 'label' 'font' needs the name of a font asset",
        ),
        (
            json!({"font": 3}),
            "text 'label' 'font' needs the name of a font asset",
        ),
        (
            json!({"font": "serif"}),
            "text 'label' font 'serif' is no asset of the scene and no .ttf or .otf file",
        ),
        (
            json!({"font": "sans", "align": "middle"}),
            "text 'label' align needs 'left', 'center' or 'right', got 'middle'",
        ),
        (
            json!({"font": "sans", "fill": "msdf"}),
            "text 'label' fill needs 'mtsdf', fast and with glyph_dist and glyph_soft, \
             or 'slug', exact edges at any size, got 'msdf'",
        ),
        (
            json!({"font": "sans", "fill": 3}),
            "text 'label' fill needs 'mtsdf', fast and with glyph_dist and glyph_soft, \
             or 'slug', exact edges at any size, got 3",
        ),
        (
            json!({"font": "sans", "fill": "outline"}),
            "text 'label' fill needs 'mtsdf', fast and with glyph_dist and glyph_soft, \
             or 'slug', exact edges at any size, got 'outline'",
        ),
        (
            json!({"font": "sans", "fill": "slug", "atlas": {"px_per_em": 32}}),
            "text 'label' takes no 'atlas' with \"fill\": \"slug\", which reads the curves",
        ),
        (
            json!({"font": "sans", "atlas": 32}),
            "text 'label' atlas needs an object with 'px_per_em', got 32",
        ),
        (
            json!({"font": "sans", "atlas": {"px_em": 32}}),
            "text 'label' atlas takes 'px_per_em', got 'px_em'",
        ),
        (
            json!({"font": "sans", "atlas": {"range": 6}}),
            "text 'label' atlas takes no 'range': the field reaches the largest 'pad' of \
             the text's layers",
        ),
        (
            json!({"font": "sans", "atlas": {"px_per_em": 0}}),
            "text 'label' atlas px_per_em needs a whole number from 1 to 2048, got 0",
        ),
        (
            json!({"font": "sans", "atlas": {"px_per_em": 6.5}}),
            "text 'label' atlas px_per_em needs a whole number from 1 to 2048, got 6.5",
        ),
        (
            json!({"font": "sans", "atlas": {"px_per_em": 4096}}),
            "text 'label' atlas px_per_em needs a whole number from 1 to 2048, got 4096",
        ),
    ] {
        let e = scene_err(&texts_scene(json!({ "label": td })));
        assert!(e.starts_with(&format!("scene: {want}")), "{td}: {e}");
    }
    let mut image = texts_scene(json!({"label": {"font": "logo"}}));
    image["assets"]["logo"] = json!({"file": "logo.png"});
    assert_eq!(
        scene_err(&image),
        "scene: text 'label' font 'logo' is the image 'logo.png', a font is a .ttf or .otf file"
    );
}

#[test]
fn a_font_asset_takes_no_image_keys_and_no_pass_sees_it() {
    for (k, v) in [
        ("sampling", json!("linear")),
        ("width", json!(4)),
        ("components", json!(1)),
    ] {
        let mut doc = texts_scene(json!({"label": {"font": "sans"}}));
        doc["assets"]["sans"][k] = v;
        assert_eq!(
            scene_err(&doc),
            format!("scene: asset 'sans' takes no '{k}', a .ttf file is a font")
        );
    }
    let sc =
        Scene::from_dict(&texts_scene(json!({"label": {"font": "sans"}})), None, None).unwrap();
    let g = sc.passes[0].glsl(Some(&sc.params), None, &sc.assets, &sc.texts, false);
    assert!(!g.contains("sans") && !g.contains("cff"), "{g}");
}

#[test]
fn a_text_name_is_a_glsl_identifier_declared_once() {
    let e = scene_err(&texts_scene(json!({"2d": {"font": "sans"}})));
    assert!(e.contains("identifier"), "{e}");
    let sc =
        Scene::from_dict(&texts_scene(json!({"label": {"font": "sans"}})), None, None).unwrap();
    let t = sc.texts[0].clone();
    let e = sc.with_texts(vec![t.clone(), t]).err().unwrap().to_string();
    assert_eq!(e, "text 'label' is declared twice");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_text_named_like_a_layer_helper_is_refused() {
    // A text named glyph declares glyph_cover, which a layer declares too.
    assert_eq!(
        scene_err(&texts_scene(json!({"glyph": {"font": "sans"}}))),
        "scene: text 'glyph' and the layers both declare 'glyph_cover'"
    );
    // glyph_label and glyphs declare names of their own, beside ghee's.
    for name in ["glyph_label", "glyphs"] {
        let mut doc = texts_scene(json!({name: {"font": "sans"}}));
        doc["passes"][0]["layers"] = json!([{"text": name, "fragment": WHITE,
            "vertex": "vec2 glyph_place(vec2 em) { return em * 40.0; }"}]);
        let (_g, mut r) = gpu(W, H);
        assert_eq!(r.setup(&loaded(&doc)).unwrap(), "", "{name}");
    }
}

#[test]
fn a_text_name_collides_with_nothing() {
    let mut vs_array = texts_scene(json!({"label": {"font": "sans"}}));
    vs_array["passes"][0]["arrays"] = json!([{"name": "label", "size": 4}]);
    assert_eq!(
        scene_err(&vs_array),
        "scene: text 'label' and array 'label' in pass 'main' both declare 'label_size'"
    );
    let mut vs_asset = texts_scene(json!({"label": {"font": "sans"}}));
    vs_asset["assets"]["label"] = json!({"file": "label.png"});
    assert_eq!(
        scene_err(&vs_asset),
        "scene: asset 'label' and text 'label' both declare 'label_size'"
    );
}

#[test]
fn a_text_declares_its_cover_and_its_size_for_an_author() {
    let mut names = text::names("label");
    names.sort();
    assert_eq!(names, ["label_cover", "label_size"]);
}

#[test]
fn a_name_declared_by_a_text_collides_with_nothing() {
    let float = json!({"type": "float", "value": 1.0});
    let label = || texts_scene(json!({"label": {"font": "sans"}}));
    let mut scene_param = label();
    scene_param["params"]["label_size"] = float.clone();
    let mut pass_param = label();
    pass_param["passes"][0]["params"] = json!({"label_cover": float});
    let mut asset = label();
    asset["assets"]["label_cover"] = json!({"file": "label.png"});
    for (doc, want) in [
        (
            scene_param,
            "scene param 'label_size' and text 'label' both declare 'label_size'",
        ),
        (
            pass_param,
            "text 'label' and param 'label_cover' in pass 'main' both declare 'label_cover'",
        ),
        (
            asset,
            "asset 'label_cover' and text 'label' both declare 'label_cover'",
        ),
    ] {
        assert_eq!(scene_err(&doc), format!("scene: {want}"));
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_name_beside_the_names_of_a_text_compiles() {
    // ghee's own names for a text start with ghee_, so label_w, label_atlas,
    // label_tex and label_emoji are free, and two texts never meet in them.
    // A pass name never reaches GLSL.
    let float = json!({"type": "float", "value": 1.0});
    let mut doc = texts_scene(json!({
        "label": {"font": ["sans", "emoji"]},
        "label_atlas": {"font": "sans"},
        "labels": {"font": "sans", "fill": "slug"},
    }));
    doc["assets"]["emoji"] = json!({"file": EMOJI});
    doc["texts"]["label"]["value"] = json!("I\u{1f600}");
    doc["params"]["label_w"] = float.clone();
    doc["params"]["label_sizes"] = float.clone();
    doc["passes"][0]["name"] = json!("label_tex");
    doc["passes"][0]["params"] =
        json!({"label_emoji_size": float, "label_emoji": float, "label_covers": float});
    doc["passes"][0]["arrays"] = json!([{"name": "label_texels", "size": 4}]);
    doc["passes"][0]["layers"] = json!([
        {"text": "label", "fragment": WHITE, "vertex": "vec2 glyph_place(vec2 em) { return em * 24.0; }"},
        {"text": "label_atlas", "fragment": WHITE,
         "vertex": "vec2 glyph_place(vec2 em) { return vec2(0.0, 40.0) + em * 24.0; }"},
    ]);
    let mut image = doc.clone();
    image["assets"]["label_tex"] = json!({"file": "label.png"});
    assert!(Scene::from_dict(&image, None, None).is_ok());
    let (_g, mut r) = gpu(W, H);
    assert_eq!(r.setup(&loaded(&doc)).unwrap(), "");
    let f = render(&mut r, &loaded(&doc));
    assert!(f.contains(&255), "the layers paint nothing");
}

/// The names declared after a type in GLSL `src`, the locals and the function
/// parameters, outside the comments and the file-scope declarations, and
/// except ghee's own names and the glyph_ names of a layer.
fn locals(src: &str) -> Vec<String> {
    let types = [
        "bool", "int", "float", "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4",
    ];
    let scope = ["uniform", "varying", "attribute", "const", "precision"];
    let ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut out: Vec<String> = Vec::new();
    for line in src.lines() {
        let code = line.split("//").next().unwrap_or("");
        let first = code.split_whitespace().next().unwrap_or("");
        if first.starts_with('#') || scope.contains(&first) {
            continue;
        }
        // Each word, and the text up to the next word.
        let mut words: Vec<(&str, &str)> = Vec::new();
        let mut rest = code;
        while let Some(i) = rest.find(ident) {
            let tail = &rest[i..];
            let n = tail.find(|c| !ident(c)).unwrap_or(tail.len());
            let gap = &tail[n..];
            let next = gap.find(ident).unwrap_or(gap.len());
            words.push((&tail[..n], &gap[..next]));
            rest = gap;
        }
        for pair in words.windows(2) {
            let ((ty, gap), (name, after)) = (pair[0], pair[1]);
            let declared = types.contains(&ty)
                && !gap.is_empty()
                && gap.trim().is_empty()
                && !after.trim_start().starts_with('(');
            let ghees = name.starts_with("glyph_") || name.starts_with("ghee_");
            if declared && !ghees && !out.iter().any(|o| o == name) {
                out.push(name.to_string());
            }
        }
    }
    out
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_vec2_param_named_like_a_local_of_ghees_glsl_changes_no_pixel() {
    let place = "vec2 glyph_place(vec2 glyph_q) { return glyph_q * 48.0 + 8.0; }";
    let ink = "void main(){ gl_FragColor = vec4(vec3(glyph_cover()), 1.0); }";
    let mut doc = texts_scene(json!({
        "label": {"font": ["sans", "emoji"]},
        "title": {"font": "sans", "fill": "slug"},
    }));
    doc["assets"]["emoji"] = json!({"file": EMOJI});
    doc["texts"]["label"]["value"] = json!("I\u{1f600}");
    doc["passes"][0]["code"] = json!(
        "void main(){ gl_FragColor = vec4(vec3(title_cover(v_texCoord * title_size, 32.0)), 1.0); }"
    );
    doc["passes"][0]["arrays"] =
        json!([{"name": "amps", "size": 4}, {"name": "mask", "size": [4, 4]}]);
    doc["passes"][0]["layers"] = json!([
        {"text": "label", "fragment": ink, "vertex": place},
        {"text": "title", "fragment": ink, "vertex": place},
    ]);
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    let p = &sc.passes[0];
    let mut glsl: Vec<String> = Vec::new();
    for hw_linear in [false, true] {
        glsl.push(p.glsl(Some(&sc.params), None, &sc.assets, &sc.texts, hw_linear));
        for k in 0..p.layers.len() {
            let (v, f) = p.layer_glsl(k, &sc, hw_linear);
            let (bv, bf) = p.layer_box_glsl(k, &sc, hw_linear);
            let (cv, cf) = p.layer_colour_glsl(k, &sc, hw_linear);
            glsl.extend([v, f, bv, bf, cv, cf]);
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
    // The passes and the layers paint grey, and the emoji paints colour.
    let coloured = |f: &[u8]| f.chunks_exact(4).any(|p| p[0] != p[2]);
    assert!(coloured(&plain), "the scene draws no emoji");
    assert!(render(&mut r, &loaded(&named)) == plain, "{names:?}");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_vec_or_color_param_named_like_a_swizzle_reads_as_any_uniform() {
    let red = json!({"type": "color", "value": [1.0, 0.0, 0.0]});
    let code = |read: &str| {
        format!(
            "void main(){{\n    vec4 v = vec4(v_texCoord, 0.25, 1.0);\n    \
             gl_FragColor = vec4({read}, 0.0, 1.0);\n}}"
        )
    };
    let place = "vec2 glyph_place(vec2 em) { return em * 24.0; }";
    let scene = |params: Value, code: &str| {
        let mut doc = texts_scene(json!({"label": {"font": "sans"}}));
        doc["params"] = params;
        doc["passes"][0]["code"] = json!(code);
        doc
    };
    let (_g, mut r) = gpu(W, H);
    // The pass selects .bg and reads the color bg, which multiplies by 1.
    let plain = render(&mut r, &loaded(&scene(json!({}), &code("v.bg"))));
    let bg = scene(json!({"bg": red}), &code("v.bg * bg.r"));
    assert!(render(&mut r, &loaded(&bg)) == plain);
    // A layer's code selects .bg under a pass param bg, and ghee's GLSL of
    // the layer selects .xy beside a scene param xy.
    let ink =
        "void main(){\n    gl_FragColor = vec4(vec3(glyph_cover()).bgr, 1.0) * glyph_cover();\n}";
    let mut drawn = scene(json!({}), &code("v.rg"));
    drawn["passes"][0]["layers"] = json!([{"text": "label", "vertex": place, "fragment": ink}]);
    let want = render(&mut r, &loaded(&drawn));
    assert!(want.contains(&255), "the layer paints nothing");
    let mut named = drawn.clone();
    named["passes"][0]["params"] = json!({"bg": red});
    named["params"]["xy"] = json!({"type": "vec2", "value": [0.0, 0.0]});
    named["passes"][0]["layers"][0]["fragment"] =
        json!(ink.replace("glyph_cover())", "glyph_cover() * bg.r)"));
    assert!(render(&mut r, &loaded(&named)) == want);
    // A filter's code selects .rgb beside a scene param rgb.
    let rgb = json!({"rgb": red});
    let mut sc = loaded(&scene(rgb, &code("v.rg")));
    apply_filters(&mut sc, &["dither".to_string()], None).unwrap();
    let mut plain = loaded(&scene(json!({}), &code("v.rg")));
    apply_filters(&mut plain, &["dither".to_string()], None).unwrap();
    assert!(render(&mut r, &sc) == render(&mut r, &plain));
}

#[test]
fn every_scene_pass_reads_the_texts_and_a_filter_pass_none() {
    let doc = json!({
        "version": 1,
        "assets": font_assets(),
        "texts": {"label": {"value": "I", "font": "sans"}, "title": {"value": "O", "font": "cff"}},
        "passes": [
            {"name": "a", "code": "void main(){ gl_FragColor = vec4(1.0); }"},
            {"name": "main", "channels": [{"buffer": "a"}],
             "code": "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }"},
        ],
    });
    let mut sc = Scene::from_dict(&doc, None, None).unwrap();
    apply_filters(&mut sc, &["dither".to_string()], None).unwrap();
    assert_eq!(sc.passes.len(), 3);
    let glsl = |p: &Pass| p.glsl(Some(&sc.params), None, &sc.assets, &sc.texts, false);
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
fn texts_survive_a_filter_asset_and_a_built_param() {
    let mut sc = Scene::from_dict(
        &texts_scene(json!({"label": {"font": "sans", "align": "right"}})),
        None,
        None,
    )
    .unwrap();
    sc.texts[0].data = Some(alone(&font(SANS)));
    let texts = sc.texts.clone();
    apply_filters(&mut sc, &["frc256".to_string()], None).unwrap();
    assert!(sc.assets.iter().any(|a| a.name == "frc256_bluenoise"));
    assert_eq!(sc.texts, texts);
    let sc = Ok::<_, anyhow::Error>(sc)
        .param(Param::float("gain", 1.0))
        .unwrap();
    assert_eq!(sc.texts, texts);
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
    let t = sc.texts.iter().find(|t| t.name == name).unwrap();
    t.data.clone().unwrap()[0].clone().unwrap().to_vec()
}

/// A scene whose text `label` draws in the font asset `face`, the file
/// `file`.
fn face_scene(file: &str) -> Scene {
    let mut doc = texts_scene(json!({"label": {"font": "face"}}));
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
    let mut doc = texts_scene(Value::Object(
        names
            .iter()
            .map(|n| (n.to_string(), json!({"font": format!("{n}_font")})))
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
fn texts_in_one_font_share_its_bytes() {
    let _g = lock();
    let mut sc = Scene::from_dict(
        &texts_scene(json!({"label": {"font": "sans"}, "title": {"font": "sans"}})),
        None,
        None,
    )
    .unwrap();
    sc.load_assets(FONTS).unwrap();
    let (a, b) = (sc.texts[0].data.as_ref(), sc.texts[1].data.as_ref());
    let (a, b) = (
        a.unwrap()[0].as_ref().unwrap(),
        b.unwrap()[0].as_ref().unwrap(),
    );
    assert!(Arc::ptr_eq(a, b));
}

/// Why the font of `sc`'s text `label` failed to load.
fn font_failure(sc: &Scene) -> String {
    let t = sc.texts.iter().find(|t| t.name == "label").unwrap();
    t.data.clone().unwrap()[0].clone().err().unwrap()
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
fn a_missing_image_still_fails_the_load() {
    let _g = lock();
    let dir = tmpdir("noimage");
    let mut doc = texts_scene(json!({"label": {"font": "sans"}}));
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
fn the_console_refuses_a_font_outside_the_scene_directory() {
    let _g = lock();
    let root = tmpdir("outside");
    let dir = format!("{root}/scene");
    fs::create_dir_all(&dir).unwrap();
    fs::write(format!("{root}/up.ttf"), font(SANS)).unwrap();
    let mut sc = face_scene("../up.ttf");
    with_assets_env("", || sc.load_assets(&dir)).unwrap();
    assert_eq!(font_data(&sc, "label"), font(SANS));
    let mut sc = face_scene("../up.ttf");
    let e = with_assets_env("", || sc.load_assets_in(&dir, Reach::Inside))
        .err()
        .unwrap()
        .to_string();
    assert_eq!(
        e,
        "asset 'face' file '../up.ttf' is outside the scene directory"
    );
    fs::remove_dir_all(&root).unwrap();
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
    fs::write(
        &path,
        texts_scene(json!({"label": {"font": "sans"}})).to_string(),
    )
    .unwrap();
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
        "texts": {"label": {"value": words, "font": "sans", "align": "center"}},
        "passes": [{"name": "main", "code": format!(
            "void main(){{ gl_FragColor = vec4(\
             abs(label_size.x - {:.6}) < 1e-4 ? 1.0 : 0.0, \
             abs(label_size.y - {:.6}) < 1e-4 ? 1.0 : 0.0, 0.0, 1.0); }}",
            laid.width, laid.height)}],
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
    doc["passes"][0]["code"] = json!(format!(
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
    assert!(r
        .set_scene_param("label", "string", &ParamValue::Str("O".to_string()))
        .unwrap());
    let f = r.render(None).unwrap().to_vec();
    assert_draws(&f, W, 0, &o, fit, "O");
    assert!(f.iter().skip(1).step_by(4).all(|g| *g == 255));
    // Words the texture cannot hold leave the last words, and the error
    // stands until words lay out again.
    assert!(r.set_text("label", &"I".repeat(400_000)).unwrap());
    let f = r.render(None).unwrap().to_vec();
    assert_draws(&f, W, 0, &o, fit, "O kept");
    let errors = r.text_errors();
    assert!(
        errors.len() == 1 && errors[0].starts_with("text 'label': ") && errors[0].contains("VC4"),
        "{errors:?}"
    );
    assert!(r.set_text("label", "").unwrap());
    assert!(blank(r.render(None).unwrap(), 0));
    assert_eq!(r.text_errors(), [] as [String; 0]);
    // A string param of no text's name draws nothing.
    assert!(!r
        .set_scene_param("ghost", "string", &ParamValue::Str("O".to_string()))
        .unwrap());
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_text_reaches_every_pass() {
    let laid = lay(SANS, "AV", Align::Left);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let doc = json!({
        "version": 1,
        "assets": font_assets(),
        "texts": {"label": {"value": "AV", "font": "sans"}},
        "passes": [
            {"name": "a", "code": format!(
                "void main(){{ gl_FragColor = vec4({}, 0.0, 0.0, 1.0); }}", fit.cover("label"))},
            {"name": "main", "channels": [{"buffer": "a"}], "code": format!(
                "void main(){{ gl_FragColor = vec4(texture2D(iChannel0, v_texCoord).r, {}, \
                 0.0, 1.0); }}", fit.cover("label"))},
        ],
    });
    let (_g, mut r) = gpu(W, H);
    let f = render(&mut r, &loaded(&doc));
    assert_draws(&f, W, 0, &laid, fit, "pass a");
    assert_draws(&f, W, 1, &laid, fit, "pass main");
    assert!(r
        .set_scene_param("label", "string", &ParamValue::Str("To".to_string()))
        .unwrap());
    let to = lay(SANS, "To", Align::Left);
    let f = r.render(None).unwrap().to_vec();
    assert_draws(&f, W, 0, &to, fit, "pass a, live");
    assert_draws(&f, W, 1, &to, fit, "pass main, live");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn two_texts_draw_their_own_words() {
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
        "texts": {"label": {"value": "AV", "font": "sans"}, "title": {"value": "O", "font": "cff"}},
        "passes": [{"name": "main", "code": format!(
            "void main(){{ gl_FragColor = vec4({}, {}, 0.0, 1.0); }}",
            fit.cover("label"), fit.cover("title"))}],
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
    // A scene whose text has another name.
    let title = json!({
        "version": 1,
        "assets": font_assets(),
        "texts": {"title": {"value": "I", "font": "sans"}},
        "passes": [{"name": "main", "code": format!(
            "void main(){{ gl_FragColor = vec4(vec3({}), 1.0); }}", fit.cover("title"))}],
    });
    let f = render(&mut r, &loaded(&title));
    assert_draws(&f, W, 0, &laid[0], fit, "title");
    assert!(!r.set_text("label", "O").unwrap());
    // A scene with no text, then a text again.
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
        "text 'label': font 'face' was never loaded, the scene was built without a directory"
    );
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_setup_frees_the_asset_slots_of_old_texts() {
    // The engine binds 16 assets at a time, so each setup has to hand back
    // the slots of the texts that its scene no longer declares.
    let laid = lay(SANS, "I", Align::Left);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let (_g, mut r) = gpu(W, H);
    for i in 0..20 {
        let name = format!("t{i}");
        let doc = json!({
            "version": 1,
            "assets": font_assets(),
            "texts": {name.clone(): {"value": "I", "font": "sans"}},
            "passes": [{"name": "main", "code": format!(
                "void main(){{ float c = {}; gl_FragColor = vec4(c, c, c, 1.0); }}",
                fit.cover(&name))}],
        });
        let frame = render(&mut r, &loaded(&doc));
        assert_draws(&frame, W, 0, &laid, fit, &name);
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_filter_after_a_text_compiles() {
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
        "code": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.5, 1.0); }",
        "layers": [{"text": "label", "fragment": WHITE, "vertex": fit_place(fit)}]});
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
    let mut t = Text::from_dict("label", &json!({"font": "face"})).unwrap();
    t.data = Some(vec![Err("gone".to_string())]);
    let laid = t.build(words).unwrap();
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
    for fill in FILLS {
        let doc = filled(hazard_scene("Missing.ttf", words, fit), fill);
        let layered = render(&mut r, &loaded(&doc));
        assert_hazard(&layered, W, fit, &rects, &format!("{fill} layer"));
        // New words lay out in boxes too.
        assert!(r.set_text("label", "I").unwrap());
        r.set_time(Some(0.0));
        let one = r.render(None).unwrap().to_vec();
        assert_hazard(&one, W, fit, &rects[..1], &format!("{fill} one box"));
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_font_names_an_asset_or_a_file_and_a_list_mixes_both() {
    let words = "I\u{1f600}";
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let (_g, mut r) = gpu(W, H);
    let assets = colour_scene(&[SANS, EMOJI], words, "mtsdf", &fit_place(fit));
    let want = render(&mut r, &loaded(&assets));
    // The files themselves, and an asset beside a file.
    for fonts in [json!([SANS, EMOJI]), json!(["f0", EMOJI])] {
        let mut doc = assets.clone();
        doc["texts"]["label"]["font"] = fonts.clone();
        let sc = loaded(&doc);
        assert_eq!(sc.to_dict()["texts"]["label"]["font"], fonts);
        assert!(render(&mut r, &sc) == want, "{fonts}");
        assert_eq!(r.text_errors(), [] as [String; 0], "{fonts}");
    }
    // A missing file leaves the text running with an error box, and names
    // the file. A file outside the scene directory is refused.
    let mut doc = assets.clone();
    doc["texts"]["label"]["font"] = json!(["Missing.ttf", EMOJI]);
    render(&mut r, &loaded(&doc));
    let errors = r.text_errors();
    let missing = format!("text 'label' font 'Missing.ttf' ({FONTS}/Missing.ttf): No such file");
    assert!(errors.iter().any(|e| e.starts_with(&missing)), "{errors:?}");
    doc["texts"]["label"]["font"] = json!(format!("../{SANS}"));
    let mut sc = Scene::from_dict(&doc, None, None).unwrap();
    assert_eq!(
        sc.load_assets_in(FONTS, Reach::Inside)
            .unwrap_err()
            .to_string(),
        format!("text 'label' font '../{SANS}' is outside the scene directory")
    );
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
        for fill in FILLS {
            let mut doc = filled(hazard_scene(EMOJI, words, fit), fill);
            doc["passes"][0]["layers"][0]["vertex"] = json!(exact_place(fit));
            let layered = render(&mut r, &loaded(&doc));
            assert_bitmap(&layered, w, &b, at, 1, [0, 0, 128], &format!("{fill} {ch}"));
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
fn words_that_fit_no_texture_at_setup_draw_nothing_and_the_scene_runs() {
    let fit = Fit {
        x: 8.0,
        y: 8.0,
        ppem: 64.0,
    };
    let words = "I".repeat(400_000);
    let (_g, mut r) = gpu(W, H);
    for doc in [
        label_scene(SANS, &words, Align::Left, fit),
        layer_scene(SANS, &words, Align::Left, fit),
    ] {
        let f = render(&mut r, &loaded(&doc));
        assert!(blank(&f, 0));
        let errors = r.text_errors();
        assert!(
            errors.len() == 1 && errors[0].starts_with("text 'label': the words need a "),
            "{errors:?}"
        );
        assert!(r.set_text("label", "I").unwrap());
        r.set_time(Some(0.0));
        assert!(!blank(r.render(None).unwrap(), 0));
        assert_eq!(r.text_errors(), [] as [String; 0]);
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_slug_layer_keeps_its_last_words_and_their_boxes_past_words_that_fit_no_texture() {
    let fit = Fit {
        x: 8.0,
        y: 8.0,
        ppem: 48.0,
    };
    let (_g, mut r) = gpu(W, H);
    let doc = filled(hazard_scene("Missing.ttf", "x", fit), "slug");
    let before = render(&mut r, &loaded(&doc));
    assert!(r.set_text("label", &"x".repeat(400_000)).unwrap());
    r.set_time(Some(0.0));
    assert!(r.render(None).unwrap() == before.as_slice());
    // The box stays, and so does the problem behind it.
    let errors = r.text_errors();
    assert_eq!(errors.len(), 2, "{errors:?}");
    assert!(errors[0].contains("Missing.ttf"), "{errors:?}");
    assert!(errors[1].contains("VC4"), "{errors:?}");
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

// Layers

/// A layer's place that puts the block where `fit` does.
fn fit_place(fit: Fit) -> String {
    format!(
        "vec2 glyph_place(vec2 em) {{ return vec2({:.4}, {:.4}) + em * {:.4}; }}",
        fit.x, fit.y, fit.ppem
    )
}

const BLACK: &str = "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }";
/// White coverage, premultiplied.
const WHITE: &str = "void main(){ float c = glyph_cover(); gl_FragColor = vec4(c); }";

/// label_scene with the words drawn by one layer over a black pass, in the
/// slug fill.
fn layer_scene(file: &str, words: &str, align: Align, fit: Fit) -> Value {
    let mut doc = label_scene(file, words, align, fit);
    doc["passes"][0] = json!({"name": "main", "code": BLACK,
        "layers": [{"text": "label", "fragment": WHITE, "vertex": fit_place(fit)}]});
    filled(doc, "slug")
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_vertex_reads_the_glyph_values_the_frame_uniforms_the_params_its_text_size_and_the_common_block(
) {
    let fit = Fit {
        x: 8.0,
        y: 8.0,
        ppem: 48.0,
    };
    let (_g, mut r) = gpu(W, H);
    let reads = "glyph_index + glyph_word + glyph_line + glyph_span.x + glyph_box.x \
                 + resolution.x + time + DT + frame + shift + k + label_size.x + twice(1.0)";
    for fill in FILLS {
        let plain = render(
            &mut r,
            &loaded(&filled(layer_scene(SANS, "Ag", Align::Left, fit), fill)),
        );
        let mut doc = filled(layer_scene(SANS, "Ag", Align::Left, fit), fill);
        doc["params"]["shift"] = json!({"type": "float", "value": 0.0});
        doc["passes"][0]["params"] = json!({"k": {"type": "float", "value": 0.0}});
        // The common block compiles in the vertex stage as it does in a pass,
        // where it reads the pass's arrays, channels and texts.
        doc["passes"][0]["arrays"] = json!([{"name": "amps", "size": 4}]);
        doc["passes"][0]["channels"] = json!([{"buffer": "main"}]);
        doc["common"] = json!({"code": "float twice(float x) { return 2.0 * x; }\n\
            float shade(vec2 p) {\n    return label_cover(p, 48.0) + amps_at(0) \
            + texture2D(iChannel0, p).r;\n}"});
        doc["passes"][0]["layers"][0]["vertex"] = json!(format!(
            "vec2 glyph_place(vec2 em) {{ return vec2(8.0) + em * 48.0 + ({reads}) * 0.0; }}"
        ));
        assert!(render(&mut r, &loaded(&doc)) == plain, "{fill}");
        // The vertex GLSL reads the pass's text and arrays itself too.
        doc["common"] = json!({"code": "float twice(float x) { return 2.0 * x; }"});
        let own = format!("{reads} + label_cover(em, 48.0) + amps_at(0)");
        doc["passes"][0]["layers"][0]["vertex"] = json!(format!(
            "vec2 glyph_place(vec2 em) {{ return vec2(8.0) + em * 48.0 + ({own}) * 0.0; }}"
        ));
        assert!(render(&mut r, &loaded(&doc)) == plain, "{fill}");
    }
}

/// `doc` with its text `label` in `fill`.
fn filled(mut doc: Value, fill: &str) -> Value {
    doc["texts"]["label"]["fill"] = json!(fill);
    doc
}

const FILLS: [&str; 2] = ["mtsdf", "slug"];

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
    assert!(r
        .set_scene_param("label", "string", &ParamValue::Str("ghee".into()))
        .unwrap());
    r.set_time(Some(0.0));
    let after = r.render(None).unwrap().to_vec();
    assert!(after != before, "the layer kept the old words");
    assert_like_cover(&after, &want, "ghee after I");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_rotated_and_scaled_layer_lands_its_outline() {
    let (w, h) = (240, 160);
    let words = "ghee";
    let laid = lay(SANS, words, Align::Left);
    let (a, s): (f64, f64) = (0.4, 52.0);
    let (cx, cy) = (120.0, 80.0);
    let place = format!(
        "vec2 glyph_place(vec2 em) {{
    vec2 d = (em - 0.5 * label_size) * {s:.4};
    return vec2({cx:.4}, {cy:.4}) + mat2({c:.6}, {sn:.6}, {nsn:.6}, {c:.6}) * d;
}}",
        c = a.cos(),
        sn = a.sin(),
        nsn = -a.sin()
    );
    let mut doc = label_scene(SANS, words, Align::Left, Fit::of(1.0, 1.0, w, h));
    doc["passes"][0] = json!({"name": "main", "code": BLACK,
        "layers": [{"text": "label", "fragment": WHITE, "vertex": place}]});
    let (_g, mut r) = gpu(w, h);
    let frame = render(&mut r, &loaded(&doc));
    // The pixel centre back through the rotation into the block.
    let em = |x: usize, y: usize| {
        let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
        let (u, v) = (a.cos() * dx + a.sin() * dy, -a.sin() * dx + a.cos() * dy);
        [u / s + 0.5 * laid.width, v / s + 0.5 * laid.height]
    };
    assert_draws_at(&frame, w, 0, &laid, em, EDGE / s, "rotated ghee");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn layers_land_in_their_order_and_pad_past_the_ink() {
    let (w, h) = (96, 96);
    let fit = Fit {
        x: 32.0,
        y: 8.0,
        ppem: 64.0,
    };
    // The slug fill takes any pad.
    let mut doc = filled(label_scene(SANS, "I", Align::Left, fit), "slug");
    let place = fit_place(fit);
    doc["passes"][0] = json!({"name": "main", "code": BLACK, "layers": [
        {"text": "label", "vertex": place, "pad": 0.25, "blend": "add",
         "fragment": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.5, 0.5); }"},
        {"text": "label", "vertex": place,
         "fragment": "void main(){ float c = glyph_cover(); gl_FragColor = vec4(c, 0.0, 0.0, c); }"},
    ]});
    let laid = lay(SANS, "I", Align::Left);
    let g = &laid.glyphs[0];
    let px = |em: [f64; 2]| {
        let (x, y) = (fit.x + em[0] * fit.ppem, fit.y + em[1] * fit.ppem);
        (x as usize, y as usize)
    };
    let mid = (g.lo[1] + g.hi[1]) / 2.0;
    let (_g, mut r) = gpu(w, h);
    let frame = render(&mut r, &loaded(&doc));
    let at = |(x, y): (usize, usize)| &frame[(y * w + x) * 4..(y * w + x) * 4 + 3];
    // The red fill lands over the pad: a pad added after it would leave blue
    // on the stem.
    assert_eq!(at(px([(g.lo[0] + g.hi[0]) / 2.0, mid])), [255, 0, 0]);
    // The pad reaches 1/4 em past the ink box.
    assert_eq!(at(px([g.lo[0] - 0.15, mid])), [0, 0, 128]);
    assert_eq!(at(px([g.hi[0] + 0.15, mid])), [0, 0, 128]);
    assert_eq!(at(px([g.hi[0] + 0.4, mid])), [0, 0, 0]);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_reads_int_params_in_both_stages() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    // The place moves the block n pixels right and m down, and the code scales
    // the coverage by n - m, so the layer matches the text 3 right and 2 down
    // only when both stages read both params.
    let mut doc = layer_scene(SANS, "ghee", Align::Left, fit);
    doc["params"]["n"] = json!({"type": "int", "value": 3});
    doc["passes"][0]["params"] = json!({"m": {"type": "int", "value": 2}});
    doc["passes"][0]["layers"][0]["vertex"] =
        json!(fit_place(fit).replace("return ", "return vec2(float(n), float(m)) + "));
    doc["passes"][0]["layers"][0]["fragment"] =
        json!(WHITE.replace("vec4(c)", "vec4(c * float(n - m))"));
    let (_g, mut r) = gpu(160, 48);
    let moved = Fit {
        x: fit.x + 3.0,
        y: fit.y + 2.0,
        ppem: fit.ppem,
    };
    let want = render(
        &mut r,
        &loaded(&label_scene(SANS, "ghee", Align::Left, moved)),
    );
    let got = render(&mut r, &loaded(&doc));
    assert_like_cover(&got, &want, "int params");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_compile_error_names_the_pass_the_layer_and_its_line() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let good = layer_scene(SANS, "I", Align::Left, fit);
    let (_g, mut r) = gpu(64, 48);
    let first = render(&mut r, &loaded(&good));
    let mut bad = good.clone();
    bad["passes"][0]["layers"][0]["fragment"] =
        json!("void main(){\n    float c = glyph_cover();\n    nope(c);\n}");
    bad["passes"][0]["layers"][0]["name"] = json!("ink");
    let e = r.setup(&loaded(&bad)).unwrap();
    assert!(
        e.starts_with("pass 'main' layer 'ink', line 3 of its fragment: "),
        "{e}"
    );
    let at = |index, part| Some(ErrorLayer { index, part });
    assert_eq!(r.error_pass, Some(ErrorPass::Index(0)));
    assert_eq!(r.error_layer, at(0, LayerPart::Fragment));
    // /stats names the layer's key that holds the error.
    let part = |r: &Renderer| r.error_layer.map(ErrorLayer::to_json);
    assert_eq!(part(&r), Some(json!({"index": 0, "part": "fragment"})));
    assert_eq!(r.error_line, Some(3));
    let mut bad = good.clone();
    bad["passes"][0]["layers"][0]["vertex"] =
        json!("vec2 glyph_place(vec2 em) {\n    return nope;\n}");
    let e = r.setup(&loaded(&bad)).unwrap();
    assert!(
        e.starts_with("pass 'main' layer 'label', line 2 of its vertex: "),
        "{e}"
    );
    assert_eq!(r.error_layer, at(0, LayerPart::Vertex));
    assert_eq!(part(&r), Some(json!({"index": 0, "part": "vertex"})));
    assert_eq!(r.error_line, Some(2));
    // An error on ghee's own lines carries no line, and a missing function
    // is named.
    let mut bad = good.clone();
    bad["passes"][0]["layers"][0]["fragment"] =
        json!("float shade(){\n    return glyph_cover();\n}\nfloat x = 0.0;");
    let e = r.setup(&loaded(&bad)).unwrap();
    assert_eq!(
        e,
        "pass 'main' layer 'label': its fragment needs a void main()"
    );
    assert_eq!(r.error_layer, at(0, LayerPart::Fragment));
    assert_eq!(r.error_line, None);
    let mut bad = good.clone();
    bad["passes"][0]["layers"][0]["vertex"] = json!("vec2 place(vec2 em) {\n    return em;\n}");
    let e = r.setup(&loaded(&bad)).unwrap();
    assert_eq!(
        e,
        "pass 'main' layer 'label': its vertex needs vec2 glyph_place(vec2 em)"
    );
    assert_eq!(r.error_layer, at(0, LayerPart::Vertex));
    // A line of the common block counts in the block, as a pass's would.
    let mut bad = good.clone();
    bad["passes"][0] = json!({"name": "main", "layers": bad["passes"][0]["layers"]});
    bad["common"] = json!({"code": "float twice(float x) {\n    return nope;\n}"});
    let e = r.setup(&loaded(&bad)).unwrap();
    assert!(
        e.starts_with("pass 'main' layer 'label', line 2 of the common block: "),
        "{e}"
    );
    assert_eq!(r.error_pass, Some(ErrorPass::Common));
    assert_eq!(r.error_layer, None);
    assert_eq!(r.error_line, Some(2));
    // The running scene stays.
    r.set_time(Some(0.0));
    assert!(r.render(None).unwrap() == first.as_slice());
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_with_no_vertex_centres_its_words_and_with_no_fragment_draws_them_white() {
    let (w, h) = (160, 64);
    let laid = lay(SANS, "ghee", Align::Left);
    let ppem = 32.0;
    let fit = Fit {
        x: (w as f64 - laid.width * ppem) / 2.0,
        y: (h as f64 - laid.height * ppem) / 2.0,
        ppem,
    };
    let (_g, mut r) = gpu(w, h);
    for fill in FILLS {
        let scene = |layer: Value| {
            let mut doc = filled(label_scene(SANS, "ghee", Align::Left, fit), fill);
            doc["passes"][0] = json!({"name": "main", "code": BLACK, "layers": [layer]});
            loaded(&doc)
        };
        let plain = render(&mut r, &scene(json!({"text": "label", "px_per_em": ppem})));
        let centred = "vec2 glyph_place(vec2 em) { return (resolution - label_size * 32.0) \
                       * 0.5 + em * 32.0; }";
        let written = json!({"text": "label", "vertex": centred,
            "fragment": "void main() { gl_FragColor = vec4(glyph_cover()); }"});
        assert!(render(&mut r, &scene(written)) == plain, "{fill}");
        // The slug fill covers as the text does.
        if fill == "slug" {
            let cover = render(
                &mut r,
                &loaded(&label_scene(SANS, "ghee", Align::Left, fit)),
            );
            assert_like_cover(&plain, &cover, fill);
        }
    }
}

#[test]
fn layers_round_trip_and_over_and_no_pad_are_the_defaults() {
    let mut doc = texts_scene(json!({"label": {"font": "sans"}}));
    doc["passes"][0]["layers"] = json!([
        {"text": "label", "fragment": WHITE, "vertex": "vec2 glyph_place(vec2 em) { return em; }"},
        {"name": "glow", "text": "label", "fragment": WHITE,
         "vertex": "vec2 glyph_place(vec2 em) { return em; }", "blend": "add", "pad": 0.125},
        {"text": "label", "fragment": WHITE, "vertex": "vec2 glyph_place(vec2 em) { return em; }",
         "blend": "over", "pad": 0},
        {"text": "label", "fragment": WHITE, "vertex": "vec2 glyph_place(vec2 em) { return em; }",
         "blend": "replace"},
        {"text": "label", "px_per_em": 48.0},
    ]);
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    let back = sc.to_dict();
    assert_eq!(
        back["passes"][0]["layers"][4],
        doc["passes"][0]["layers"][4]
    );
    assert_eq!(
        back["passes"][0]["layers"][0],
        doc["passes"][0]["layers"][0]
    );
    assert_eq!(
        back["passes"][0]["layers"][1],
        doc["passes"][0]["layers"][1]
    );
    assert_eq!(
        back["passes"][0]["layers"][2], doc["passes"][0]["layers"][0],
        "over and pad 0 are left out"
    );
    assert_eq!(
        back["passes"][0]["layers"][3],
        doc["passes"][0]["layers"][3]
    );
    assert_eq!(Scene::from_dict(&back, None, None).unwrap(), sc);
}

#[test]
fn a_pass_of_layers_alone_round_trips_with_no_code_and_a_pass_of_neither_is_refused() {
    let place = "vec2 glyph_place(vec2 em) { return em; }";
    let mut doc = texts_scene(json!({"label": {"font": "sans"}}));
    doc["passes"] = json!([
        {"name": "words", "layers": [{"text": "label", "fragment": WHITE, "vertex": place}]},
        {"name": "main", "code": BLACK},
    ]);
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    assert!(!sc.passes[0].has_code());
    let back = sc.to_dict();
    assert_eq!(back["passes"][0], doc["passes"][0]);
    assert_eq!(Scene::from_dict(&back, None, None).unwrap(), sc);
    for words in [
        json!({"name": "words"}),
        json!({"name": "words", "code": " "}),
    ] {
        doc["passes"][0] = words;
        assert_eq!(
            scene_err(&doc),
            "scene: pass 'words' has no 'code' (inline GLSL), no 'shader' (a file next to \
             the scene) and no 'layers'"
        );
    }
}

#[test]
fn a_layer_goes_by_its_name_else_by_its_text_counted_among_the_layers_with_no_name() {
    let place = "vec2 glyph_place(vec2 em) { return em; }";
    let err = |layers: Value| {
        let mut doc = texts_scene(json!({"label": {"font": "sans"}}));
        doc["passes"][0]["layers"] = layers;
        scene_err(&doc)
    };
    let d = |keys: Value| {
        let mut d = json!({"text": "label", "vertex": place});
        d.as_object_mut()
            .unwrap()
            .extend(keys.as_object().unwrap().clone());
        d
    };
    let name = "name needs one or more ASCII letters, digits, '_' or '-'";
    let cases = [
        (
            json!([
                d(json!({})),
                d(json!({"name": "glow"})),
                d(json!({"pad": -1}))
            ]),
            "scene: pass 'main' layer 'label 2' pad needs a number of em, 0 or more, got -1"
                .to_string(),
        ),
        (
            json!([d(json!({"name": "glow", "blend": "max"}))]),
            "scene: pass 'main' layer 'glow' blend needs 'over', 'add' or 'replace', got 'max'"
                .to_string(),
        ),
        (
            json!([d(json!({"name": "my glow"}))]),
            format!("scene: pass 'main' layer 'label' {name}, got 'my glow'"),
        ),
        (
            json!([d(json!({"name": ""}))]),
            format!("scene: pass 'main' layer 'label' {name}, got ''"),
        ),
        (
            json!([d(json!({"name": 3}))]),
            format!("scene: pass 'main' layer 'label' {name}, got 3"),
        ),
        (
            json!([d(json!({"name": "glow"})), d(json!({"name": "glow"}))]),
            "scene: pass 'main' has two layers that go by 'glow'".to_string(),
        ),
        (
            json!([d(json!({})), d(json!({"name": "label"}))]),
            "scene: pass 'main' has two layers that go by 'label'".to_string(),
        ),
        (
            json!([d(json!({})), json!(3)]),
            "scene: the 2nd layer of pass 'main' needs an object, got 3".to_string(),
        ),
    ];
    for (layers, want) in cases {
        assert_eq!(err(layers), want);
    }
}

#[test]
fn a_layer_declaration_names_what_it_lacks() {
    let place = "vec2 glyph_place(vec2 em) { return em; }";
    let with = |layers: Value| {
        let mut doc = texts_scene(json!({"label": {"font": "sans"}}));
        doc["passes"][0]["layers"] = layers;
        scene_err(&doc)
    };
    let one = |d: Value| with(json!([d]));
    let cases = [
        (with(json!(3)), "pass 'main': 'layers' needs a list, got 3"),
        (
            one(json!(3)),
            "the 1st layer of pass 'main' needs an object, got 3",
        ),
        (
            one(json!({"fragment": WHITE, "vertex": place})),
            "the 1st layer of pass 'main' has no 'text', the name of a scene text",
        ),
        (
            one(json!({"text": "label", "fragment": WHITE})),
            "pass 'main' layer 'label' needs 'px_per_em', the pixels per em of the words centred \
             in the pass, or 'vertex', GLSL that defines vec2 glyph_place(vec2 em)",
        ),
        (
            one(json!({"text": "label", "vertex": place, "px_per_em": 32})),
            "pass 'main' layer 'label' takes 'vertex' or 'px_per_em', got both",
        ),
        (
            one(json!({"text": "label", "px_per_em": 0})),
            "pass 'main' layer 'label' px_per_em needs a number of pixels above 0, got 0",
        ),
        (
            one(json!({"text": "label", "px_per_em": "32"})),
            "pass 'main' layer 'label' px_per_em needs a number of pixels above 0, got '32'",
        ),
        (
            one(json!({"text": "label", "vertex": "", "px_per_em": null})),
            "pass 'main' layer 'label' 'vertex' needs vertex GLSL that defines vec2 \
             glyph_place(vec2 em), got ''",
        ),
        (
            one(json!({"text": "label", "fragment": " ", "vertex": place})),
            "pass 'main' layer 'label' 'fragment' needs fragment GLSL with its own void main(), \
             got ' '",
        ),
        (
            one(json!({"text": "label", "fragment": WHITE, "place": place})),
            "pass 'main' layer 'label' takes its vertex GLSL in 'vertex', got 'place'",
        ),
        (
            one(json!({"text": "label", "code": WHITE, "vertex": place})),
            "pass 'main' layer 'label' takes its fragment GLSL in 'fragment', got 'code'",
        ),
        (
            one(json!({"text": "label", "fragment": WHITE, "vertex": place, "blend": "max"})),
            "pass 'main' layer 'label' blend needs 'over', 'add' or 'replace', got 'max'",
        ),
        (
            one(json!({"text": "label", "fragment": WHITE, "vertex": place, "pad": -1})),
            "pass 'main' layer 'label' pad needs a number of em, 0 or more, got -1",
        ),
        (
            one(json!({"text": "label", "fragment": WHITE, "vertex": place, "alpha": "keep"})),
            "pass 'main' layer 'label' takes no 'alpha': its alpha blends by 'blend', as its \
             colour does",
        ),
        (
            one(json!({"text": "label", "fragment": WHITE, "vertex": place, "emoji": "yes"})),
            "pass 'main' layer 'label' emoji needs true or false, got 'yes'",
        ),
        (
            one(json!({"text": "words", "fragment": WHITE, "vertex": place})),
            "pass 'main' layer 'words' names text 'words', which the scene does not declare",
        ),
    ];
    for (got, want) in cases {
        assert!(got.ends_with(want), "{got}\n  wants {want}");
    }
    let mut doc = texts_scene(json!({"label": {"font": "sans"}}));
    doc["passes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name": "d", "code": BLACK, "filter": "dither",
            "layers": [{"text": "label", "fragment": WHITE, "vertex": place}]}));
    let got = scene_err(&doc);
    assert!(
        got.ends_with("pass 'd' is filter 'dither', and a filter pass takes no layers"),
        "{got}"
    );
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_on_a_sized_pass_covers_as_the_text_does_there() {
    // Pass a renders at half the frame on each axis, and main shows it.
    let (w, h) = (320, 120);
    let words = "AV To\nghee";
    let laid = lay(SANS, words, Align::Left);
    let fit = Fit::of(laid.width, laid.height, w / 2, h / 2);
    let show = "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }";
    let sized = |pass: Value| {
        let mut doc = filled(label_scene(SANS, words, Align::Left, fit), "slug");
        let mut a = pass;
        a["name"] = json!("a");
        a["size"] = json!([w / 2, h / 2]);
        doc["passes"] = json!([a, {"name": "main", "channels": [{"buffer": "a"}], "code": show}]);
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
fn one_text_drawn_by_two_passes_follows_its_words_in_both() {
    // Pass a draws the words in red. main shows a and adds them in green.
    let (w, h) = (256, 64);
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 40.0,
    };
    let place = fit_place(fit);
    let mut doc = filled(label_scene(SANS, "AV", Align::Left, fit), "slug");
    doc["passes"] = json!([
        {"name": "a", "code": BLACK, "layers": [{"text": "label", "vertex": place,
            "fragment": "void main(){ float c = glyph_cover(); gl_FragColor = vec4(c, 0.0, 0.0, c); }"}]},
        {"name": "main", "channels": [{"buffer": "a"}],
         "code": "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }",
         "layers": [{"text": "label", "vertex": place, "blend": "add",
            "fragment": "void main(){ float c = glyph_cover(); gl_FragColor = vec4(0.0, c, 0.0, c); }"}]},
    ]);
    let (_g, mut r) = gpu(w, h);
    let words = ["AV", "ghee"];
    let wants: Vec<Vec<u8>> = words
        .iter()
        .map(|w| render(&mut r, &loaded(&label_scene(SANS, w, Align::Left, fit))))
        .collect();
    for (words, want) in words.iter().zip(&wants) {
        let got = if *words == "AV" {
            render(&mut r, &loaded(&doc))
        } else {
            assert!(r
                .set_scene_param("label", "string", &ParamValue::Str(words.to_string()))
                .unwrap());
            r.render(None).unwrap().to_vec()
        };
        assert_like_cover(&got, want, &format!("{words} in red"));
        assert_like_cover(&got[1..], &want[1..], &format!("{words} in green"));
        assert!(blank(&got, 2), "{words}: blue");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_param_takes_a_name_of_ghees_glsl_without_its_prefix() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let (_g, mut r) = gpu(160, 48);
    let want = render(
        &mut r,
        &loaded(&label_scene(SANS, "ghee", Align::Left, fit)),
    );
    // ghee's attributes, its vertex flip and its locals take the ghee_ prefix.
    for name in ["a1", "vflip", "glyph_a1", "glyph_corner", "glyph_main"] {
        let mut doc = layer_scene(SANS, "ghee", Align::Left, fit);
        doc["params"][name] = json!({"type": "float", "value": 1.0});
        let code = WHITE.replace("vec4(c)", &format!("vec4(c * {name})"));
        doc["passes"][0]["layers"][0]["fragment"] = json!(code);
        let got = render(&mut r, &loaded(&doc));
        assert_like_cover(&got, &want, &format!("a param named {name}"));
    }
    // A layer's own names and the prefix are ghee's in any scene, with a layer
    // or without one.
    for (name, why) in [
        (
            "glyph_index",
            "scene param 'glyph_index' and the layers both declare 'glyph_index'",
        ),
        (
            "ghee_a1",
            "scene param 'ghee_a1': a name that starts with ghee_ belongs to ghee",
        ),
    ] {
        for doc in [
            layer_scene(SANS, "ghee", Align::Left, fit),
            label_scene(SANS, "ghee", Align::Left, fit),
        ] {
            let mut doc = doc;
            doc["params"][name] = json!({"type": "float", "value": 1.0});
            assert_eq!(scene_err(&doc), format!("scene: {why}"));
        }
    }
    let mut doc = label_scene(SANS, "ghee", Align::Left, fit);
    doc["passes"][0]["params"] = json!({"glyph_cover": {"type": "float", "value": 1.0}});
    assert_eq!(
        scene_err(&doc),
        "scene: the layers and param 'glyph_cover' in pass 'main' both declare 'glyph_cover'"
    );
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
    main["code"] = json!("void main(){ gl_FragColor = vec4(v_at(0)); }");
    doc["passes"] = json!([{"name": "a", "code": BLACK}, main]);
    let past = match unsafe { ghee::ffi::ghee_gl_texture_units(r.handle()) } > 16 {
        true => "a pass binds at most",
        false => "this GPU has",
    };
    assert_eq!(
        r.setup(&loaded(&doc)).unwrap(),
        format!(
            "pass 'main' layer 'label' needs {} texture units, {past} {units}",
            units + 1
        )
    );
    r.set_time(Some(0.0));
    assert!(r.render(None).unwrap() == before.as_slice());
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_that_samples_past_the_texture_units_fails_naming_the_layer() {
    // main reads one channel, and its layer samples the label's texture and an
    // image in every other unit.
    let (_g, mut r) = gpu(64, 48);
    let units = r.max_texture_units() as usize;
    let dir = tmpdir("units");
    fs::copy(format!("{FONTS}/{SANS}"), format!("{dir}/{SANS}")).unwrap();
    let names: Vec<String> = (0..units - 1).map(|i| format!("img{i}")).collect();
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let mut doc = layer_scene(SANS, "I", Align::Left, fit);
    for n in &names {
        fs::write(format!("{dir}/{n}.raw"), [128u8]).unwrap();
        doc["assets"][n] =
            json!({"file": format!("{n}.raw"), "width": 1, "height": 1, "format": "u8"});
    }
    let sum: Vec<String> = names
        .iter()
        .map(|n| format!("texture2D({n}, vec2(0.5)).r"))
        .collect();
    doc["passes"][0]["layers"][0]["fragment"] = json!(format!(
        "void main(){{ gl_FragColor = vec4(glyph_cover() * ({})); }}",
        sum.join(" + ")
    ));
    let scene = |doc: &Value| {
        let mut sc = Scene::from_dict(doc, None, None).unwrap();
        sc.load_assets(&dir).unwrap();
        sc
    };
    assert_eq!(r.setup(&scene(&doc)).unwrap(), "");
    let main = doc["passes"][0].clone();
    doc["passes"] = json!([{"name": "a", "code": BLACK}, main]);
    doc["passes"][1]["channels"] = json!([{"buffer": "a"}]);
    let past = match unsafe { ghee::ffi::ghee_gl_texture_units(r.handle()) } > 16 {
        true => "a pass binds at most",
        false => "this GPU has",
    };
    assert_eq!(
        r.setup(&scene(&doc)).unwrap(),
        format!(
            "pass 'main' layer 'label' needs {} texture units, {past} {units}",
            units + 1
        )
    );
    fs::remove_dir_all(&dir).unwrap();
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
        "code": "void main(){ vec4 t = texture2D(iChannel0, v_texCoord); \
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
    for fill in FILLS {
        let mut doc = filled(layer_scene(SANS, "I", Align::Left, I_FIT), fill);
        doc["passes"][0]["code"] =
            json!("void main(){ gl_FragColor = vec4(0.0, 0.0, 0.25, 0.5); }");
        let frame = render(&mut r, &loaded(&doc));
        // White over the half-transparent pass: opaque in the stem, the pass's
        // alpha beside it.
        assert!(alphas(&frame).contains(&255), "{fill}");
        assert!(alphas(&frame)[0].abs_diff(128) <= 1, "{fill}");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_reset_drops_the_layers_with_the_scene() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let (_g, mut r) = gpu(64, 48);
    render(&mut r, &loaded(&layer_scene(SANS, "I", Align::Left, fit)));
    r.reset();
    assert!(r.set_text("label", "O").unwrap());
}

// The MTSDF fill

/// The signed distance in pixels from the centre of pixel (x, y) to the
/// outline of `laid` placed by `fit`, below 0 inside.
fn px_distance(laid: &Laid, fit: Fit, x: usize, y: usize) -> f64 {
    laid.distance(fit.em(x, y)) * fit.ppem
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn glyph_dist_and_glyph_soft_draw_an_outline_and_a_glow_by_the_distance() {
    // One glyph a frame, so no quad overlaps another: a bowl, and corners.
    for words in ["e", "k"] {
        let laid = lay(SANS, words, Align::Left);
        let g = &laid.glyphs[0];
        let fit = Fit {
            x: 16.0,
            y: 8.0,
            ppem: 64.0,
        };
        let (w, h) = (
            (laid.width * fit.ppem + 32.0).ceil() as usize,
            (laid.height * fit.ppem + 16.0).ceil() as usize,
        );
        // Red is the fill, green a 3 px ring outside it in a quad padded 4 px,
        // and blue a glow that halves every 1.4 px, in a quad padded 8 px.
        let mut doc = label_scene(SANS, words, Align::Left, fit);
        doc["passes"][0] = json!({"name": "main", "code": BLACK, "layers": [
            {"text": "label", "vertex": fit_place(fit), "blend": "add", "pad": 0.125,
             "fragment": "void main(){ float g = exp(-max(-glyph_soft(), 0.0) / 2.0); \
                      gl_FragColor = vec4(0.0, 0.0, g, 0.0); }"},
            {"text": "label", "vertex": fit_place(fit), "blend": "add", "pad": 0.0625,
             "fragment": "void main(){ float fill = clamp(glyph_dist() + 0.5, 0.0, 1.0); \
                      float ring = clamp(glyph_soft() + 3.5, 0.0, 1.0); \
                      gl_FragColor = vec4(fill, ring - fill, 0.0, 0.0); }"},
        ]});
        let (_g, mut r) = gpu(w, h);
        let frame = render(&mut r, &loaded(&doc));
        let at = |x: usize, y: usize, c: usize| frame[(y * w + x) * 4 + c];
        // The field holds the true distance up to its reach, the glow's pad,
        // and the glow's quad spans the ink box grown by that much.
        let reach = 0.125 * fit.ppem;
        let (mut seen, mut off) = ([0usize; 6], Vec::new());
        for y in 0..h {
            for x in 0..w {
                let d = px_distance(&laid, fit, x, y);
                let p = fit.em(x, y);
                let past = (0..2)
                    .map(|a| (g.lo[a] - p[a]).max(p[a] - g.hi[a]) * fit.ppem)
                    .fold(f64::MIN, f64::max);
                let mut check = |k: usize, c: usize, want: f64, tol: f64| {
                    seen[k] += 1;
                    let got = f64::from(at(x, y, c));
                    if (got - want).abs() > tol {
                        off.push((x, y, d, c, got, want));
                    }
                };
                if d < -1.5 {
                    check(0, 0, 255.0, 4.0);
                    check(0, 1, 0.0, 4.0);
                    check(0, 2, 255.0, 4.0);
                }
                if (1.5..2.25).contains(&d) {
                    check(1, 0, 0.0, 4.0);
                    check(1, 1, 255.0, 4.0);
                }
                if d > 4.5 {
                    check(2, 0, 0.0, 4.0);
                    check(2, 1, 0.0, 4.0);
                }
                if d > reach + 1.0 && past < reach - 1.0 {
                    check(4, 2, 255.0 * (-reach / 2.0).exp(), 4.0);
                }
                if past > reach + 1.0 {
                    check(5, 2, 0.0, 0.0);
                }
                // The glow read back as a distance, within 1/2 px of the
                // outline's. Inside the box, a counter's walls face each other,
                // and a bilinear read across the ridge between them falls
                // short.
                if (0.75..4.5).contains(&d) && (0.5..reach - 1.0).contains(&past) {
                    seen[3] += 1;
                    let got = -2.0 * (f64::from(at(x, y, 2)) / 255.0).ln();
                    if (got - d).abs() > 0.5 {
                        off.push((x, y, d, 2, got, d));
                    }
                }
            }
        }
        assert!(seen.iter().all(|n| *n > 50), "{words}: {seen:?}");
        assert!(
            off.is_empty(),
            "{words}: {} pixels off, (x, y, d, channel, got, want) {:?}",
            off.len(),
            &off[..off.len().min(8)]
        );
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn an_mtsdf_layer_pads_its_quads_by_its_pad() {
    let fit = Fit {
        x: 24.0,
        y: 8.0,
        ppem: 64.0,
    };
    let (w, h) = (96, 96);
    let laid = lay(SANS, "I", Align::Left);
    let g = &laid.glyphs[0];
    let (_g, mut r) = gpu(w, h);
    // 0.25 em reaches past the field of the least range, 6 texels at 32 px/em.
    for pad in [0.25, 0.05] {
        let mut doc = label_scene(SANS, "I", Align::Left, fit);
        doc["passes"][0] = json!({"name": "main", "code": BLACK, "layers": [
            {"text": "label", "vertex": fit_place(fit), "pad": pad,
             "fragment": "void main(){ gl_FragColor = vec4(1.0); }"}]});
        let frame = render(&mut r, &loaded(&doc));
        // The pixels from the ink box along x, at the middle of the stem.
        let y = (fit.y + (g.lo[1] + g.hi[1]) / 2.0 * fit.ppem) as usize;
        let lit = |x: f64| frame[(y * w + x as usize) * 4] == 255;
        let edge = (
            fit.x + (g.hi[0] + pad) * fit.ppem,
            fit.x + (g.lo[0] - pad) * fit.ppem,
        );
        assert!(lit(edge.0 - 1.0) && !lit(edge.0 + 1.0), "pad {pad}: right");
        assert!(lit(edge.1 + 1.0) && !lit(edge.1 - 1.0), "pad {pad}: left");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn an_mtsdf_field_reaches_the_largest_pad_of_its_texts_layers() {
    let laid = lay(SANS, "I", Align::Left);
    let g = &laid.glyphs[0];
    let fit = Fit {
        x: 40.0,
        y: 40.0,
        ppem: 64.0,
    };
    let (w, h) = (
        (laid.width * fit.ppem + 80.0).ceil() as usize,
        (laid.height * fit.ppem + 80.0).ceil() as usize,
    );
    // The fill pads nothing, and a glow in a second pass pads 0.5 em, 32 px,
    // which takes a range of 32 texels at 32 px/em. Blue holds the true
    // distance outside the glyph, 255 at 32 px.
    let pad = 0.5;
    let mut doc = label_scene(SANS, "I", Align::Left, fit);
    let glow =
        "void main(){ gl_FragColor = vec4(0.0, 0.0, clamp(-glyph_soft() / 32.0, 0.0, 1.0), 0.0); }";
    let fill = "void main(){ float c = glyph_cover(); gl_FragColor = vec4(c, 0.0, 0.0, c); }";
    doc["passes"] = json!([
        {"name": "fill", "code": BLACK, "layers": [
            {"text": "label", "vertex": fit_place(fit), "fragment": fill}]},
        {"name": "main", "channels": [{"buffer": "fill"}],
         "code": "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }",
         "layers": [{"text": "label", "vertex": fit_place(fit), "pad": pad, "blend": "add",
                    "fragment": glow}]},
    ]);
    let sc = loaded(&doc);
    assert_eq!(sc.atlas_range("label"), 32);
    let (_g, mut r) = gpu(w, h);
    let frame = render(&mut r, &sc);
    let (mut seen, mut off) = (0, Vec::new());
    for y in 0..h {
        for x in 0..w {
            let d = px_distance(&laid, fit, x, y);
            let p = fit.em(x, y);
            let past = (0..2)
                .map(|a| (g.lo[a] - p[a]).max(p[a] - g.hi[a]) * fit.ppem)
                .fold(f64::MIN, f64::max);
            // Inside the quad, out to 1 px short of the pad, where the least
            // range reaches 3 px.
            if (1.0..pad * fit.ppem - 1.0).contains(&d) && past < pad * fit.ppem - 1.0 {
                seen += usize::from(d > 8.0);
                let got = f64::from(frame[(y * w + x) * 4 + 2]) / 255.0 * 32.0;
                if (got - d).abs() > 1.0 {
                    off.push((x, y, d, got));
                }
            }
        }
    }
    assert!(seen > 500, "{seen}");
    assert!(
        off.is_empty(),
        "{} pixels off, (x, y, d, got) {:?}",
        off.len(),
        &off[..off.len().min(8)]
    );
}

#[test]
fn an_mtsdf_atlas_takes_its_range_from_the_layers_of_its_text() {
    let place = "vec2 glyph_place(vec2 em) { return em; }";
    let layer = |pad: f64| json!({"text": "label", "vertex": place, "fragment": WHITE, "pad": pad});
    let mut doc = texts_scene(json!({
        "label": {"font": "sans"},
        "title": {"font": "sans", "atlas": {"px_per_em": 48}},
    }));
    doc["passes"] = json!([
        {"name": "a", "layers": [layer(0.1), layer(0.3)]},
        {"name": "main", "code": BLACK, "layers": [layer(0.2),
            {"text": "title", "vertex": place, "fragment": WHITE, "pad": 0.25}]},
    ]);
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    // ceil(2 * 0.3 * 32) and 2 * 0.25 * 48
    assert_eq!((sc.atlas_range("label"), sc.atlas_range("title")), (20, 24));
    // A small pad takes the least range.
    doc["passes"][0]["layers"] = json!([layer(0.05)]);
    doc["passes"][1]["layers"] = json!([layer(0.0)]);
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    assert_eq!(sc.atlas_range("label"), 6);
    // A pad past the side of an atlas refuses the scene, and the slug fill
    // takes any pad.
    doc["passes"][1]["layers"] = json!([layer(33.0)]);
    assert_eq!(
        scene_err(&doc),
        "scene: pass 'main' layer 'label' pad 33.0 needs an atlas range of 2112 texels at the 32 \
         px_per_em of text 'label', over the cap of 2048"
    );
    doc["texts"]["label"]["fill"] = json!("slug");
    assert!(Scene::from_dict(&doc, None, None).is_ok());
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn the_slug_fill_names_the_mtsdf_fill_for_a_distance() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let (_g, mut r) = gpu(64, 48);
    for read in ["glyph_dist", "glyph_soft"] {
        let mut doc = layer_scene(SANS, "I", Align::Left, fit);
        doc["passes"][0]["layers"][0]["fragment"] = json!(format!(
            "void main(){{\n    float d = {read}();\n    gl_FragColor = vec4(d);\n}}"
        ));
        let e = r.setup(&loaded(&doc)).unwrap();
        assert_eq!(
            e,
            format!(
                "pass 'main' layer 'label', line 2 of its fragment: {read} needs \"fill\": \"mtsdf\" on \
                 text 'label'"
            )
        );
        assert_eq!(r.error_line, Some(2));
        assert_eq!(r.setup(&loaded(&filled(doc, "mtsdf"))).unwrap(), "");
    }
}

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
fn only_an_mtsdf_layer_starts_an_atlas() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let (_g, mut r) = gpu(64, 48);
    let cache = tmpdir("no-atlas");
    std::env::set_var("XDG_CACHE_HOME", &cache);
    // A slug layer, and an MTSDF text that no layer names.
    let slug = layer_scene(SANS, "I", Align::Left, fit);
    let undrawn = label_scene(SANS, "I", Align::Left, fit);
    for doc in [&slug, &undrawn] {
        render(&mut r, &loaded(doc));
        assert_eq!(threads_named("ghee-atlas"), 0);
        assert_eq!(fs::read_dir(&cache).unwrap().count(), 0);
    }
    render(&mut r, &loaded(&filled(slug, "mtsdf")));
    assert_eq!(threads_named("ghee-atlas"), 1);
    // The worker writes the cache after the glyphs are in.
    let file = std::path::Path::new(&cache).join("ghee/atlas");
    let t0 = std::time::Instant::now();
    while fs::read_dir(&file).map_or(0, |d| d.count()) == 0 {
        assert!(t0.elapsed().as_secs() < 20, "no cache file");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    drop(r);
    assert_eq!(threads_named("ghee-atlas"), 0);
    fs::remove_dir_all(&cache).unwrap();
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_glyph_the_atlas_cannot_hold_draws_an_error_box_and_says_why() {
    // A pad of 32 em takes a range of 2048 texels, which pads every cell past
    // the atlas. The advances of m and w are wider than a box.
    let words = "mw";
    let laid = text::build(&alone(&font(SANS)), words, Align::Left);
    let built = laid.built.as_ref().unwrap();
    let (bw, bh) = block(built);
    let fit = Fit::of(bw, bh, W, H);
    let mut doc = hazard_scene(SANS, words, fit);
    doc["passes"][0]["layers"][0]["pad"] = json!(32.0);
    let (_g, mut r) = gpu(W, H);
    let frame = render(&mut r, &loaded(&doc));
    // Each box stands centred in its character's advance, on the baseline.
    let q = &built.quads;
    let rects: Vec<[f64; 4]> = (0..q.count)
        .map(|i| {
            let v = vertex(q, i, 0);
            let mid = f64::from(v[3][1] + v[3][2]) / 2.0;
            let base = laid_baseline(&laid);
            [
                mid - text::ERROR_BOX[0] / 2.0,
                base - text::ERROR_BOX[1],
                mid + text::ERROR_BOX[0] / 2.0,
                base,
            ]
        })
        .collect();
    assert_hazard(&frame, W, fit, &rects, "atlas boxes");
    let errors = r.text_errors();
    assert_eq!(errors.len(), 2, "{errors:?}");
    for (e, ch) in errors.iter().zip(['m', 'w']) {
        let want = format!("text 'label': '{ch}' has no field in the atlas: its ");
        assert!(
            e.starts_with(&want) && e.ends_with("cell is larger than the 2048x2048 atlas"),
            "{e}"
        );
    }
}

/// The baseline of a one-line layout, in em from the block's top.
fn laid_baseline(laid: &text::Layout) -> f64 {
    let g = &laid.built.as_ref().unwrap().glyphs[0];
    g.origin[1]
}

#[test]
fn the_slug_fill_assembles_as_before_the_mtsdf_fill() {
    // A layer of the slug fill against the shaders pinned in the fixtures,
    // whose fragment shader leaves out the lines that name the MTSDF fill and
    // the read that keeps VC4 on one thread.
    let doc = json!({
        "version": 1,
        "params": {
            "sung": {"type": "float", "value": 2.5},
            "n": {"type": "int", "value": 3},
            "ink": {"type": "color", "value": [0.9, 0.9, 1.0]},
        },
        "assets": {"sans": {"file": "DejaVuSans-ASCII.ttf"}},
        "texts": {"t": {"value": "ghee", "font": "sans", "align": "center", "fill": "slug"}},
        "common": {"code": "float twice(float x) { return 2.0 * x; }"},
        "passes": [{
            "name": "main",
            "code": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }",
            "params": {"gain": {"type": "float", "value": 1.0}},
            "arrays": [{"name": "times", "size": 64}],
            "layers": [{
                "text": "t",
                "pad": 0.05,
                "vertex": "vec2 glyph_place(vec2 em) {\n    vec2 off = (resolution - t_size * 46.0) * 0.5;\n    return off + em * 46.0;\n}",
                "fragment": "void main() {\n    float c = glyph_cover() * gain;\n    float k = clamp(sung - glyph_index, 0.0, 1.0);\n    gl_FragColor = vec4(ink * c * twice(k), c);\n}"
            }]
        }]
    });
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    let p = &sc.passes[0];
    let (vs, fs) = p.layer_glsl(0, &sc, false);
    let (bvs, bfs) = p.layer_box_glsl(0, &sc, false);
    let fixture = |f: &str| {
        fs::read_to_string(format!(
            "{}/tools/fixtures/glsl/{f}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    };
    let named = [
        "// Distances come from the MTSDF fill's field, and a compile error names it.",
        "#define glyph_dist glyph_dist_needs_fill_mtsdf",
        "#define glyph_soft glyph_soft_needs_fill_mtsdf",
        "    // On VC4 a shader whose last texture read sits outside an if runs in two",
        "    // threads, whose switch loses the condition flags. A read after the",
        "    // author's code that never runs, in an if that the compiler cannot drop,",
        "    // keeps the shader on one thread whatever textures the author's code",
        "    // reads.",
        "    if (gl_FragCoord.x < 0.0) gl_FragColor = texture2D(ghee_t_tex, v_texCoord);",
    ];
    assert!(named.iter().all(|l| fs.lines().any(|f| f == *l)));
    let without: Vec<&str> = fs.lines().filter(|l| !named.contains(l)).collect();
    assert_eq!(vs, fixture("layer-slug.vert"));
    assert_eq!(without.join("\n"), fixture("layer-slug.frag"));
    assert_eq!(bvs, fixture("layer-box.vert"));
    assert_eq!(bfs, fixture("layer-box.frag"));
}

// Colour glyphs

/// The fonts of a text, in order, from the fixtures.
fn listed(files: &[&str]) -> Vec<text::Font> {
    files.iter().map(|f| Ok(Arc::new(font(f)))).collect()
}

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

/// The layer's place for `fit`, at full precision.
fn exact_place(fit: Fit) -> String {
    format!(
        "vec2 glyph_place(vec2 em) {{ return vec2({:?}, {:?}) + em * {:?}; }}",
        fit.x, fit.y, fit.ppem
    )
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

/// A scene of text `label` in the fonts `files`, assets `f0`, `f1` and on,
/// drawn white over a black pass by a layer of `fill` placed by `place`.
fn colour_scene(files: &[&str], words: &str, fill: &str, place: &str) -> Value {
    let assets: Map<String, Value> = files
        .iter()
        .enumerate()
        .map(|(i, f)| (format!("f{i}"), json!({"file": f})))
        .collect();
    let fonts: Vec<String> = (0..files.len()).map(|i| format!("f{i}")).collect();
    json!({
        "version": 1,
        "assets": assets,
        "texts": {"label": {"value": words, "font": fonts, "fill": fill}},
        "passes": [{"name": "main", "code": BLACK,
            "layers": [{"text": "label", "fragment": WHITE, "vertex": place}]}],
    })
}

#[test]
fn a_text_takes_each_character_from_the_first_font_that_has_it() {
    let em = |file: &str, ch: char| {
        let data = font(file);
        let face = Face::parse(&data, 0).unwrap();
        let g = face.glyph_index(ch).unwrap_or(GlyphId(0));
        f64::from(face.glyph_hor_advance(g).unwrap()) / f64::from(face.units_per_em())
    };
    let line = |file: &str| {
        let data = font(file);
        let f = Face::parse(&data, 0).unwrap();
        f64::from(f.ascender() - f.descender() + f.line_gap()) / f64::from(f.units_per_em())
    };
    // Letters from the text font, emoji from the emoji font, and a character
    // that neither has from glyph 0 of the first, the missing-glyph box.
    let words = "A\u{1f600}b\u{3042}";
    let laid = text::build(&listed(&[SANS, EMOJI]), words, Align::Left);
    assert_eq!(laid.faults, []);
    let built = laid.built.as_ref().unwrap();
    let glyphs: Vec<(char, usize)> = built.glyphs.iter().map(|p| (p.ch, p.font)).collect();
    assert_eq!(glyphs, [('A', 0), ('b', 0), ('\u{3042}', 0)]);
    assert_eq!(built.glyphs[2].gid, 0);
    let colour: Vec<(char, usize)> = built.colour.iter().map(|p| (p.ch, p.font)).collect();
    assert_eq!(colour, [('\u{1f600}', 1)]);
    // Each glyph advances by its own font's advance, and the line height
    // and the baseline are the first font's.
    let spans: Vec<f64> = built
        .glyphs
        .iter()
        .chain(&built.colour)
        .map(|p| p.span[1] - p.span[0])
        .collect();
    let want = [
        em(SANS, 'A'),
        em(SANS, 'b'),
        em(SANS, '\u{3042}'),
        em(EMOJI, '\u{1f600}'),
    ];
    for (got, want) in spans.iter().zip(want) {
        assert!((got - want).abs() < 1e-12, "{spans:?}");
    }
    assert_eq!(built.colour[0].span[0], built.glyphs[0].span[1]);
    assert_eq!(block(built).1, line(SANS));
    // The other way round, the letters fall through to the text font, and
    // the emoji font's glyph 0 has no ink.
    let laid = text::build(&listed(&[EMOJI, SANS]), words, Align::Left);
    assert_eq!(laid.faults, [text::Fault::NoOutline(0, vec!['\u{3042}'])]);
    let built = laid.built.as_ref().unwrap();
    let glyphs: Vec<(char, usize)> = built.glyphs.iter().map(|p| (p.ch, p.font)).collect();
    assert_eq!(glyphs, [('A', 1), ('b', 1)]);
    assert_eq!(built.colour[0].font, 0);
    assert_eq!(built.boxes.count, 1);
    assert_eq!(block(built).1, line(EMOJI));
}

#[test]
fn an_emoji_selector_takes_the_first_colour_glyph_and_a_text_selector_keeps_the_first_font() {
    let hearts = |laid: &text::Layout| {
        let built = laid.built.as_ref().unwrap();
        let of = |ps: &[text::Placement]| -> Vec<(usize, usize)> {
            ps.iter()
                .filter(|p| p.ch == '\u{2764}')
                .map(|p| (p.index, p.font))
                .collect()
        };
        (of(&built.glyphs), of(&built.colour))
    };
    let words = "\u{2764}\u{fe0f}\u{2764}\u{fe0e}\u{2764}";
    let laid = text::build(&listed(&[HEART, EMOJI_HEART]), words, Align::Left);
    assert_eq!(laid.faults, []);
    assert_eq!(hearts(&laid), (vec![(2, 0), (4, 0)], vec![(0, 1)]));
    // A colour font ahead in the list draws every heart in colour.
    let laid = text::build(&listed(&[EMOJI_HEART, HEART]), words, Align::Left);
    assert_eq!(hearts(&laid), (vec![], vec![(0, 0), (2, 0), (4, 0)]));
    // With no colour glyph for the character, U+FE0F leaves it to the first
    // font that holds it.
    let laid = text::build(&listed(&[HEART, EMOJI]), words, Align::Left);
    assert_eq!(hearts(&laid), (vec![(0, 0), (2, 0), (4, 0)], vec![]));
    // The heart after the selector starts where the colour heart ends.
    let laid = text::build(&listed(&[HEART, EMOJI_HEART]), words, Align::Left);
    let built = laid.built.as_ref().unwrap();
    let after = built.glyphs.iter().find(|p| p.index == 2).unwrap();
    assert_eq!(after.span[0], built.colour[0].span[1]);
}

#[test]
fn glyphs_of_two_fonts_do_not_kern() {
    // A pair that the text font kerns, its second character taken from a
    // font ahead of it.
    let data = font(SANS);
    let face = Face::parse(&data, 0).unwrap();
    let o = face.glyph_index('O').unwrap();
    let kern = |l: GlyphId| {
        let t = face.tables().kern.unwrap();
        t.subtables
            .into_iter()
            .filter_map(|s| s.glyphs_kerning(l, o))
            .sum::<i16>()
    };
    let l = ('!'..='~')
        .find(|c| kern(face.glyph_index(*c).unwrap()) != 0)
        .unwrap();
    let words = format!("{l}O");
    let alone = text::build(&listed(&[SANS]), &words, Align::Left);
    let g = &alone.built.as_ref().unwrap().glyphs;
    assert!(g[1].span[0] != g[0].span[1], "{l}O kerns in {SANS}");
    let laid = text::build(&listed(&[CFF, SANS]), &words, Align::Left);
    let g = &laid.built.as_ref().unwrap().glyphs;
    assert_eq!((g[0].font, g[1].font), (1, 0));
    assert_eq!(g[1].span[0], g[0].span[1]);
}

#[test]
fn default_ignorables_take_no_room_between_colour_glyphs() {
    // Neither font maps U+FE0F, and the emoji font maps U+200D with no
    // advance. A sequence draws its parts one by one.
    let laid = text::build(
        &listed(&[SANS, EMOJI]),
        "\u{1f600}\u{fe0f}\u{200d}\u{1f642}\u{fe0f}",
        Align::Left,
    );
    assert_eq!(laid.faults, []);
    let built = laid.built.as_ref().unwrap();
    let c = &built.colour;
    assert_eq!(c.len(), 2);
    assert_eq!(c[1].span[0], c[0].span[1]);
    assert_eq!(c[1].index, 3);
    assert!((block(built).0 - c[1].span[1]).abs() < 1e-12);
}

#[test]
fn a_text_round_trips_its_fonts_and_names_a_bad_list() {
    let doc = texts_scene(json!({
        "label": {"font": ["sans", "cff"]},
        "title": {"font": ["cff"]},
    }));
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    assert_eq!(sc.texts[0].fonts, ["sans", "cff"]);
    // A list of one saves as the name.
    assert_eq!(
        sc.to_dict()["texts"],
        json!({
            "label": {"value": "I", "font": ["sans", "cff"]},
            "title": {"value": "I", "font": "cff"},
        })
    );
    for (font, want) in [
        (
            json!([]),
            "text 'label' 'font' needs the name of a font asset or a .ttf or .otf file, or a \
             list of them, got []",
        ),
        (
            json!(["sans", 3]),
            "text 'label' 'font' needs the name of a font asset or a .ttf or .otf file, or a \
             list of them, got ['sans', 3]",
        ),
        (
            json!(["sans", "nope"]),
            "text 'label' font 'nope' is no asset of the scene and no .ttf or .otf file",
        ),
    ] {
        let e = scene_err(&texts_scene(json!({"label": {"font": font}})));
        assert!(e.ends_with(want), "{e}");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_paints_its_colour_glyphs_in_their_own_colours() {
    let words = "\u{1f600}I";
    let (w, h) = (220, 150);
    let (_g, mut r) = gpu(w, h);
    // The sbix font carries a strike of flat green beside the bitmaps, and
    // the largest strike draws.
    for file in [EMOJI, EMOJI_SBIX] {
        let laid = text::build(&listed(&[SANS, file]), words, Align::Left);
        let built = laid.built.as_ref().unwrap();
        let b = bitmap(file, '\u{1f600}');
        assert_eq!(b.ppem, 109.0);
        let (fit, at) = bitmap_fit(built, 0, &b, 1.0, [8.0, 6.0]);
        for fill in FILLS {
            let what = format!("{file} {fill}");
            let doc = colour_scene(&[SANS, file], words, fill, &exact_place(fit));
            let frame = render(&mut r, &loaded(&doc));
            assert_bitmap(&frame, w, &b, at, 1, [0, 0, 0], &what);
            assert_eq!(r.text_errors(), [] as [String; 0], "{what}");
            // The layer's own code paints the I white.
            let i = &built.glyphs[0];
            let x = fit.x + (i.span[0] + i.span[1]) / 2.0 * fit.ppem;
            let y = fit.y + (i.origin[1] - 0.3) * fit.ppem;
            let px = &frame[((y as usize) * w + x as usize) * 4..][..3];
            assert_eq!(px, [255, 255, 255], "{what}");
        }
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_layer_paints_the_colour_glyph_that_an_emoji_selector_takes() {
    let words = "\u{2764}\u{fe0f}";
    let (w, h) = (220, 150);
    let (_g, mut r) = gpu(w, h);
    let laid = text::build(&listed(&[HEART, EMOJI_HEART]), words, Align::Left);
    let built = laid.built.as_ref().unwrap();
    let b = bitmap(EMOJI_HEART, '\u{2764}');
    let (fit, at) = bitmap_fit(built, 0, &b, 1.0, [8.0, 6.0]);
    for fill in FILLS {
        let doc = colour_scene(&[HEART, EMOJI_HEART], words, fill, &exact_place(fit));
        let frame = render(&mut r, &loaded(&doc));
        assert_bitmap(&frame, w, &b, at, 1, [0, 0, 0], fill);
        assert_eq!(r.text_errors(), [] as [String; 0], "{fill}");
    }
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_colour_glyph_shrinks_through_its_mipmaps() {
    // At a quarter of its strike, each pixel of the bitmap's box shows the
    // average of a 4 x 4 block of the bitmap: mip level 2.
    let words = "\u{1f600}";
    let laid = text::build(&listed(&[SANS, EMOJI]), words, Align::Left);
    let b = bitmap(EMOJI, '\u{1f600}');
    let (fit, at) = bitmap_fit(laid.built.as_ref().unwrap(), 0, &b, 4.0, [5.0, 3.0]);
    let (_g, mut r) = gpu(48, 40);
    let doc = colour_scene(&[SANS, EMOJI], words, "mtsdf", &exact_place(fit));
    let frame = render(&mut r, &loaded(&doc));
    assert_bitmap(&frame, 48, &b, at, 4, [0, 0, 0], "a quarter");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_colour_glyph_moves_with_its_letter() {
    let words = "I\u{1f600}";
    let fit = Fit {
        x: 4.0,
        y: 2.0,
        ppem: 40.0,
    };
    let (w, h) = (128, 56);
    let (_g, mut r) = gpu(w, h);
    let still = render(
        &mut r,
        &loaded(&colour_scene(
            &[SANS, EMOJI],
            words,
            "mtsdf",
            &fit_place(fit),
        )),
    );
    // glyph_place moves the emoji alone, 32 pixels to the right.
    let moving = format!(
        "vec2 glyph_place(vec2 em) {{ return vec2({:?}, {:?}) + em * {:?} \
         + vec2(glyph_index > 0.5 ? 32.0 : 0.0, 0.0); }}",
        fit.x, fit.y, fit.ppem
    );
    let moved = render(
        &mut r,
        &loaded(&colour_scene(&[SANS, EMOJI], words, "mtsdf", &moving)),
    );
    // The emoji is the only colour in either frame.
    let coloured = |f: &[u8], x: usize, y: usize| {
        let p = &f[(y * w + x) * 4..][..3];
        p[0] != p[2]
    };
    let (mut n, mut off) = (0, 0);
    for y in 0..h {
        for x in 0..w - 32 {
            if coloured(&still, x, y) {
                n += 1;
                let (a, b) = (
                    &still[(y * w + x) * 4..][..3],
                    &moved[(y * w + x + 32) * 4..][..3],
                );
                off += usize::from(a != b);
            }
            assert!(!coloured(&moved, x, y) || x >= 32 && coloured(&still, x - 32, y));
        }
    }
    assert!(n > 500 && off == 0, "{n} coloured pixels, {off} off");
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn cover_reads_a_colour_glyph_as_no_ink_and_says_nothing() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 40.0,
    };
    let (_g, mut r) = gpu(128, 56);
    let want = render(&mut r, &loaded(&label_scene(SANS, "I", Align::Left, fit)));
    let mut doc = label_scene(SANS, "I\u{1f600}", Align::Left, fit);
    doc["assets"]["emoji"] = json!({"file": EMOJI});
    doc["texts"]["label"]["font"] = json!(["face", "emoji"]);
    let got = render(&mut r, &loaded(&doc));
    assert!(got == want, "the emoji changed the coverage");
    assert_eq!(r.text_errors(), [] as [String; 0]);
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
    let mut doc = colour_scene(&[SANS, EMOJI], words, "mtsdf", &fit_place(fit));
    doc["passes"][0]["code"] = json!("void main(){ gl_FragColor = vec4(0.0, 0.0, 0.25, 0.5); }");
    let frame = render(&mut r, &loaded(&doc));
    assert!(plane(&frame, 0).iter().any(|v| *v > 200), "no emoji drawn");
    assert!(alphas(&frame).contains(&255));
    assert!(alphas(&frame)[0].abs_diff(128) <= 1);
}

#[test]
fn the_last_layer_of_a_text_in_a_pass_paints_its_emoji_and_emoji_says_otherwise() {
    let place = "vec2 glyph_place(vec2 em) { return em; }";
    let layer = |text: &str, emoji: Option<bool>| {
        let mut d = json!({"text": text, "fragment": WHITE, "vertex": place});
        if let Some(e) = emoji {
            d["emoji"] = json!(e);
        }
        d
    };
    let mut doc = texts_scene(json!({"label": {"font": "sans"}, "title": {"font": "sans"}}));
    doc["passes"][0]["layers"] = json!([
        layer("label", Some(false)),
        layer("label", Some(true)),
        layer("title", None),
        layer("label", Some(true)),
    ]);
    doc["passes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name": "top", "code": BLACK, "layers": [layer("label", Some(false))]}));
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    let paints = |sc: &Scene| -> Vec<Vec<bool>> {
        sc.passes
            .iter()
            .map(|p| (0..p.layers.len()).map(|k| p.paints_emoji(k)).collect())
            .collect()
    };
    let want = vec![vec![false, true, true, true], vec![false]];
    assert_eq!(paints(&sc), want);
    // The dict holds emoji where it differs from the default of its place.
    let back = sc.to_dict();
    let layers = &back["passes"][0]["layers"];
    assert_eq!(layers[0], layer("label", None));
    assert_eq!(layers[1], layer("label", Some(true)));
    assert_eq!(layers[2], layer("title", None));
    assert_eq!(layers[3], layer("label", None));
    assert_eq!(back["passes"][1]["layers"][0], layer("label", Some(false)));
    assert_eq!(paints(&Scene::from_dict(&back, None, None).unwrap()), want);
}

#[test]
#[cfg_attr(ghee_no_gpu, ignore = "needs a GL device")]
fn a_shadow_and_a_glow_paint_no_emoji_unless_emoji_says_so() {
    // Three layers of one text, a shadow, a glow and the fill, each 40 pixels
    // right of the one before.
    let (w, h) = (128, 40);
    let place = |k: usize| {
        format!(
            "vec2 glyph_place(vec2 em) {{ return vec2({:?}, 4.0) + em * 24.0; }}",
            4.0 + 40.0 * k as f64
        )
    };
    let doc = |emoji: [Option<bool>; 3]| {
        let mut doc = colour_scene(&[SANS, EMOJI], "\u{1f600}", "mtsdf", &place(0));
        let layers: Vec<Value> = (0..3)
            .map(|k| {
                let mut d = json!({"text": "label", "fragment": WHITE, "vertex": place(k)});
                if let Some(e) = emoji[k] {
                    d["emoji"] = json!(e);
                }
                d
            })
            .collect();
        doc["passes"][0]["layers"] = json!(layers);
        doc
    };
    let (_g, mut r) = gpu(w, h);
    // The layers whose 40 pixels hold an emoji, over the black pass.
    let mut painted = |doc: &Value| -> Vec<bool> {
        let frame = render(&mut r, &loaded(doc));
        assert_eq!(r.text_errors(), [] as [String; 0]);
        (0..3)
            .map(|k| {
                (0..h).any(|y| {
                    (k * 40..k * 40 + 40).any(|x| frame[(y * w + x) * 4..][..3] != [0, 0, 0])
                })
            })
            .collect()
    };
    assert_eq!(painted(&doc([None; 3])), [false, false, true]);
    assert_eq!(painted(&doc([Some(true), None, None])), [true, false, true]);
    assert_eq!(painted(&doc([None, None, Some(false)])), [false; 3]);
}
