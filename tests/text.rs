//! Scene texts: the layout, the scene model, the font assets, and the
//! coverage that passes and draws render on this host's GPU.
//!
//! The reference below rebuilds the promised layout and outline from the font
//! with ttf-parser, so every check holds against any packing of the texture.

use std::fs;
use std::sync::{Arc, Mutex, MutexGuard};

use serde_json::{json, Map, Value};
use ttf_parser::{Face, GlyphId, OutlineBuilder};

use ghee::engine::{probe_dri_devices, ErrorPass, Renderer};
use ghee::filters::apply_filters;
use ghee::scene::{Param, ParamValue, Pass, Scene, SceneBuilder};
use ghee::text::{self, Align, Text};

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

fn font(file: &str) -> Vec<u8> {
    fs::read(format!("{FONTS}/{file}")).unwrap()
}

/// A Renderer opens a GL device, and load_assets reads $GHEE_ASSETS. A test
/// that does either holds this, so no variable changes under a device or a
/// lookup.
static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A w x h RGBA renderer on the first GL device that opens. None = no device,
/// and the test reports the skip.
fn gpu(w: usize, h: usize) -> Option<(MutexGuard<'static, ()>, Renderer)> {
    let guard = lock();
    let mut errs = Vec::new();
    for dev in probe_dri_devices() {
        match Renderer::new(w as i32, h as i32, 4, Some(&dev), false, false) {
            Ok(r) => return Some((guard, r)),
            Err(e) => errs.push(format!("{dev}: {e}")),
        }
    }
    eprintln!("no GL device for rendering: {}", errs.join("; "));
    None
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
        "params": {"label": {"type": "string", "value": words}},
        "assets": {"face": {"file": file}},
        "texts": {"label": {"font": "face", "align": align_name(align)}},
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
/// Built carries it only as the uniforms behind `<name>_size`.
fn block(built: &text::Built) -> (f64, f64) {
    let get = |suffix: &str| {
        built
            .uniforms
            .iter()
            .find(|(s, _)| *s == suffix)
            .unwrap_or_else(|| panic!("no {suffix} uniform"))
            .1
    };
    (get("_w"), get("_h"))
}

/// The layout of `words` in a fixture font that draws every one of them.
fn drawn(file: &str, words: &str, align: Align) -> text::Built {
    let laid = text::build(&font(file), words, align);
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
    let laid = text::build(b"not a font", "I O", Align::Left);
    assert!(
        matches!(&laid.faults[..], [text::Fault::Font(why)] if why.starts_with("unreadable font")),
        "{:?}",
        laid.faults
    );
    let built = laid.built.unwrap();
    assert_eq!((built.quads.count, built.boxes.count), (0, 2));
    let mut t = Text::from_dict("label", &json!({"font": "face"})).unwrap();
    t.data = Some(Err("asset 'face' (face.ttf): gone".to_string()));
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
    let laid = text::build(&font(SANS), words, Align::Left);
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

/// A scene that declares `texts`, a string param for each, and the fixture
/// fonts.
fn texts_scene(texts: Value) -> Value {
    let mut params = Map::new();
    for name in texts.as_object().unwrap().keys() {
        params.insert(name.clone(), json!({"type": "string", "value": "I"}));
    }
    json!({
        "version": 1,
        "params": params,
        "assets": font_assets(),
        "texts": texts,
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
        assert_eq!(sc.to_dict()["texts"], json!({"label": {"font": "sans"}}));
    }
    // The outline fill is the one fill, so a save leaves it out.
    let sc = Scene::from_dict(
        &texts_scene(json!({"label": {"font": "sans", "fill": "outline"}})),
        None,
        None,
    )
    .unwrap();
    assert_eq!(sc.to_dict()["texts"], json!({"label": {"font": "sans"}}));
}

#[test]
fn a_text_needs_a_scene_string_param_of_its_name() {
    let want = "scene: text 'label' needs a scene string param 'label' that holds its words";
    let text = json!({"label": {"font": "sans"}});
    let none = json!({
        "version": 1, "assets": font_assets(), "texts": text,
        "passes": [{"name": "main", "code": "void main(){}"}],
    });
    assert_eq!(scene_err(&none), want);
    let float = json!({
        "version": 1, "assets": font_assets(), "texts": text,
        "params": {"label": {"type": "float", "value": 1.0}},
        "passes": [{"name": "main", "code": "void main(){}"}],
    });
    assert_eq!(scene_err(&float), want);
    let in_a_pass = json!({
        "version": 1, "assets": font_assets(), "texts": text,
        "passes": [{"name": "main", "code": "void main(){}",
                    "params": {"label": {"type": "string", "value": "I"}}}],
    });
    assert_eq!(scene_err(&in_a_pass), want);
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
            json!({"font": SANS}),
            "text 'label' font 'DejaVuSans-ASCII.ttf' is no asset of the scene",
        ),
        (
            json!({"font": "sans", "align": "middle"}),
            "text 'label' align needs 'left', 'center' or 'right', got 'middle'",
        ),
        (
            json!({"font": "sans", "fill": "mtsdf"}),
            "text 'label' fill needs 'outline', got 'mtsdf'",
        ),
        (
            json!({"font": "sans", "fill": 3}),
            "text 'label' fill needs 'outline', got 3",
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
        ("filter", json!("linear")),
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
fn a_text_name_collides_with_nothing() {
    let one_scope = |doc: &Value| scene_err(doc).contains("a name belongs to one scope only");
    let mut vs_array = texts_scene(json!({"label": {"font": "sans"}}));
    vs_array["passes"][0]["arrays"] = json!([{"name": "label", "size": 4}]);
    assert_eq!(
        scene_err(&vs_array),
        "scene: 'label' is a text and a name in pass 'main', a name belongs to one scope only"
    );
    let mut vs_asset = texts_scene(json!({"label": {"font": "sans"}}));
    vs_asset["assets"]["label"] = json!({"file": "label.png"});
    assert!(one_scope(&vs_asset));
}

#[test]
fn image_assets_and_texts_share_the_engine_cap() {
    // The two font assets beside them take no slot.
    let with_images = |n: usize| {
        let mut doc = texts_scene(json!({"label": {"font": "sans"}}));
        for i in 0..n {
            doc["assets"][format!("a{i}")] = json!({"file": format!("a{i}.png")});
        }
        doc
    };
    assert!(Scene::from_dict(&with_images(15), None, None).is_ok());
    let e = scene_err(&with_images(16));
    assert!(
        e.ends_with("17 image assets and texts, over the cap of 16"),
        "{e}"
    );
}

#[test]
fn every_scene_pass_reads_the_texts_and_a_filter_pass_none() {
    let doc = json!({
        "version": 1,
        "params": {"label": {"type": "string", "value": "I"},
                   "title": {"type": "string", "value": "O"}},
        "assets": font_assets(),
        "texts": {"label": {"font": "sans"}, "title": {"font": "cff"}},
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
    sc.texts[0].data = Some(Ok(Arc::new(font(SANS))));
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
    t.data.clone().unwrap().unwrap().to_vec()
}

/// A scene whose text `label` draws in the font asset `face`, the file
/// `file`.
fn face_scene(file: &str) -> Scene {
    let mut doc = texts_scene(json!({"label": {"font": "face"}}));
    doc["assets"] = json!({"face": {"file": file}});
    Scene::from_dict(&doc, None, None).unwrap()
}

#[test]
fn a_font_resolves_beside_the_scene_then_one_up_then_through_ghee_assets() {
    let _g = lock();
    let root = tmpdir("order");
    let (dir, store) = (format!("{root}/scene"), format!("{root}/store"));
    for d in [&dir, &store] {
        fs::create_dir_all(d).unwrap();
    }
    let (sans, cff) = (font(SANS), font(CFF));
    fs::write(format!("{dir}/here.ttf"), &sans).unwrap();
    fs::write(format!("{root}/up.ttf"), &sans).unwrap();
    fs::write(format!("{store}/store.ttf"), &sans).unwrap();
    // The scene directory wins over the one above it, and that one over the
    // store.
    fs::write(format!("{dir}/both.ttf"), &sans).unwrap();
    fs::write(format!("{root}/both.ttf"), &cff).unwrap();
    fs::write(format!("{root}/above.ttf"), &sans).unwrap();
    fs::write(format!("{store}/above.ttf"), &cff).unwrap();
    let names = ["here", "up", "store", "both", "above"];
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
    let (a, b) = (a.unwrap().as_ref().unwrap(), b.unwrap().as_ref().unwrap());
    assert!(Arc::ptr_eq(a, b));
}

/// Why the font of `sc`'s text `label` failed to load.
fn font_failure(sc: &Scene) -> String {
    let t = sc.texts.iter().find(|t| t.name == "label").unwrap();
    t.data.clone().unwrap().err().unwrap()
}

#[test]
fn a_missing_font_names_the_first_path_and_every_other_one_tried() {
    let _g = lock();
    let root = tmpdir("missing");
    let (dir, store) = (format!("{root}/scene"), format!("{root}/store"));
    for d in [&dir, &store] {
        fs::create_dir_all(d).unwrap();
    }
    let file = format!("ghee-text-missing-{}.ttf", std::process::id());
    let mut sc = face_scene(&file);
    with_assets_env(&store, || sc.load_assets(&dir)).unwrap();
    assert_eq!(
        font_failure(&sc),
        format!(
            "asset 'face' ({dir}/{file}): No such file or directory \
             (also tried: {root}/{file}, {store}/{file})"
        )
    );
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
fn a_font_outside_the_scene_directory_is_refused() {
    let _g = lock();
    let root = tmpdir("outside");
    let dir = format!("{root}/scene");
    fs::create_dir_all(&dir).unwrap();
    fs::write(format!("{root}/up.ttf"), font(SANS)).unwrap();
    let mut sc = face_scene("../up.ttf");
    let e = with_assets_env("", || sc.load_assets(&dir))
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
fn a_fitted_capital_i_covers_its_stem_and_nothing_beside_it() {
    let laid = lay(SANS, "I", Align::Left);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let Some((_g, mut r)) = gpu(W, H) else { return };
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
fn two_lines_take_their_alignment() {
    let words = "AV To\nghee";
    let Some((_g, mut r)) = gpu(W, H) else { return };
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
fn cubic_outlines_draw() {
    let laid = lay(CFF, "IO", Align::Left);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let Some((_g, mut r)) = gpu(W, H) else { return };
    let f = render(&mut r, &loaded(&label_scene(CFF, "IO", Align::Left, fit)));
    assert_draws(&f, W, 0, &laid, fit, "IO");
}

#[test]
fn a_missing_character_draws_glyph_0() {
    let laid = lay(SANS, "I字", Align::Left);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let Some((_g, mut r)) = gpu(W, H) else { return };
    let f = render(&mut r, &loaded(&label_scene(SANS, "I字", Align::Left, fit)));
    assert_draws(&f, W, 0, &laid, fit, "I字");
}

#[test]
fn empty_words_cover_nothing() {
    let fit = Fit {
        x: 8.0,
        y: 8.0,
        ppem: 64.0,
    };
    let Some((_g, mut r)) = gpu(W, H) else { return };
    for words in ["", "\n", " "] {
        let f = render(&mut r, &loaded(&label_scene(SANS, words, Align::Left, fit)));
        assert!(blank(&f, 0), "{words:?}");
    }
}

#[test]
fn size_reads_the_block_in_em() {
    let words = "AV To\nghee";
    let laid = lay(SANS, words, Align::Center);
    let doc = json!({
        "version": 1,
        "params": {"label": {"type": "string", "value": words}},
        "assets": font_assets(),
        "texts": {"label": {"font": "sans", "align": "center"}},
        "passes": [{"name": "main", "code": format!(
            "void main(){{ gl_FragColor = vec4(\
             abs(label_size.x - {:.6}) < 1e-4 ? 1.0 : 0.0, \
             abs(label_size.y - {:.6}) < 1e-4 ? 1.0 : 0.0, 0.0, 1.0); }}",
            laid.width, laid.height)}],
    });
    let Some((_g, mut r)) = gpu(W, H) else { return };
    let f = render(&mut r, &loaded(&doc));
    assert_eq!(&f[..4], [255, 255, 0, 255]);
}

#[test]
fn a_string_param_relays_the_text_live() {
    let (i, o) = (lay(SANS, "I", Align::Left), lay(SANS, "O", Align::Left));
    let fit = Fit::of(o.width, o.height, W, H);
    let mut doc = label_scene(SANS, "I", Align::Left, fit);
    doc["params"]["gain"] = json!({"type": "float", "value": 0.0});
    doc["passes"][0]["code"] = json!(format!(
        "void main(){{ gl_FragColor = vec4({}, gain, 0.0, 1.0); }}",
        fit.cover("label")
    ));
    let Some((_g, mut r)) = gpu(W, H) else { return };
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
fn a_text_reaches_every_pass() {
    let laid = lay(SANS, "AV", Align::Left);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let doc = json!({
        "version": 1,
        "params": {"label": {"type": "string", "value": "AV"}},
        "assets": font_assets(),
        "texts": {"label": {"font": "sans"}},
        "passes": [
            {"name": "a", "code": format!(
                "void main(){{ gl_FragColor = vec4({}, 0.0, 0.0, 1.0); }}", fit.cover("label"))},
            {"name": "main", "channels": [{"buffer": "a"}], "code": format!(
                "void main(){{ gl_FragColor = vec4(texture2D(iChannel0, v_texCoord).r, {}, \
                 0.0, 1.0); }}", fit.cover("label"))},
        ],
    });
    let Some((_g, mut r)) = gpu(W, H) else { return };
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
        "params": {"label": {"type": "string", "value": "AV"},
                   "title": {"type": "string", "value": "O"}},
        "assets": font_assets(),
        "texts": {"label": {"font": "sans"}, "title": {"font": "cff"}},
        "passes": [{"name": "main", "code": format!(
            "void main(){{ gl_FragColor = vec4({}, {}, 0.0, 1.0); }}",
            fit.cover("label"), fit.cover("title"))}],
    });
    let Some((_g, mut r)) = gpu(W, H) else { return };
    let f = render(&mut r, &loaded(&doc));
    assert_draws(&f, W, 0, &label, fit, "label");
    assert_draws(&f, W, 1, &title, fit, "title");
}

#[test]
fn a_later_setup_draws_its_own_text() {
    let words = ["I", "O", "AV"];
    let laid: Vec<Laid> = words.iter().map(|w| lay(SANS, w, Align::Left)).collect();
    let fit = Fit::of(
        laid.iter().map(|l| l.width).fold(0.0, f64::max),
        laid[0].height,
        W,
        H,
    );
    let Some((_g, mut r)) = gpu(W, H) else { return };
    for (w, l) in words.iter().zip(&laid) {
        let f = render(&mut r, &loaded(&label_scene(SANS, w, Align::Left, fit)));
        assert_draws(&f, W, 0, l, fit, w);
    }
    // A scene whose text has another name.
    let title = json!({
        "version": 1,
        "params": {"title": {"type": "string", "value": "I"}},
        "assets": font_assets(),
        "texts": {"title": {"font": "sans"}},
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
fn a_scene_whose_font_never_loaded_fails_its_setup_naming_the_font() {
    let fit = Fit {
        x: 8.0,
        y: 8.0,
        ppem: 64.0,
    };
    let sc = Scene::from_dict(&label_scene(SANS, "I", Align::Left, fit), None, None).unwrap();
    let Some((_g, mut r)) = gpu(W, H) else { return };
    assert_eq!(
        r.setup(&sc).unwrap(),
        "text 'label': font 'face' was never loaded, the scene was built without a directory"
    );
}

#[test]
fn a_setup_frees_the_asset_slots_of_old_texts() {
    // The engine binds 16 assets at a time, so each setup has to hand back
    // the slots of the texts that its scene no longer declares.
    let laid = lay(SANS, "I", Align::Left);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let Some((_g, mut r)) = gpu(W, H) else { return };
    for i in 0..20 {
        let name = format!("t{i}");
        let doc = json!({
            "version": 1,
            "params": {name.clone(): {"type": "string", "value": "I"}},
            "assets": font_assets(),
            "texts": {name.clone(): {"font": "sans"}},
            "passes": [{"name": "main", "code": format!(
                "void main(){{ float c = {}; gl_FragColor = vec4(c, c, c, 1.0); }}",
                fit.cover(&name))}],
        });
        let frame = render(&mut r, &loaded(&doc));
        assert_draws(&frame, W, 0, &laid, fit, &name);
    }
}

#[test]
fn a_filter_after_a_text_compiles() {
    let laid = lay(SANS, "I", Align::Left);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let Some((_g, mut r)) = gpu(W, H) else { return };
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

/// label_scene with the words drawn by a white draw over a blue pass.
fn hazard_scene(file: &str, words: &str, fit: Fit) -> Value {
    let mut doc = label_scene(file, words, Align::Left, fit);
    doc["passes"][0] = json!({"name": "main",
        "code": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.5, 1.0); }",
        "draws": [{"text": "label", "code": WHITE, "place": fit_place(fit)}]});
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
fn a_missing_font_draws_a_box_a_character_and_the_scene_runs() {
    let words = "Ag j\nI";
    let mut t = Text::from_dict("label", &json!({"font": "face"})).unwrap();
    t.data = Some(Err("gone".to_string()));
    let laid = t.build(words).unwrap();
    let rects = box_rects(&laid);
    assert_eq!(rects.len(), 4);
    let (bw, bh) = block(laid.built.as_ref().unwrap());
    let fit = Fit::of(bw, bh, W, H);
    let Some((_g, mut r)) = gpu(W, H) else { return };
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
    let draw = render(&mut r, &loaded(&hazard_scene("Missing.ttf", words, fit)));
    assert_hazard(&draw, W, fit, &rects, "draw");
    // New words lay out in boxes too.
    assert!(r.set_text("label", "I").unwrap());
    r.set_time(Some(0.0));
    let one = r.render(None).unwrap().to_vec();
    assert_hazard(&one, W, fit, &rects[..1], "one box");
}

#[test]
fn an_unreadable_font_draws_a_box_a_character() {
    let words = "Ag";
    let laid = text::build(b"not a font", words, Align::Left);
    let rects = box_rects(&laid);
    let (bw, bh) = block(laid.built.as_ref().unwrap());
    let fit = Fit::of(bw, bh, W, H);
    let dir = tmpdir("unreadable");
    fs::write(format!("{dir}/bad.ttf"), b"not a font").unwrap();
    let Some((_g, mut r)) = gpu(W, H) else { return };
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
fn a_bitmap_font_draws_boxes_where_ink_should_be_and_nothing_else() {
    // A space, a zero-width joiner the font maps, and a variation selector
    // it lacks need no ink.
    let words = "\u{1f600} \u{200d}\u{1f642}\u{fe0f}";
    let laid = text::build(&font(EMOJI), words, Align::Left);
    assert_eq!(
        laid.faults,
        [text::Fault::NoOutline(vec!['\u{1f600}', '\u{1f642}'])]
    );
    let built = laid.built.as_ref().unwrap();
    assert_eq!((built.quads.count, built.boxes.count), (0, 2));
    // Each box stands centred in its character's advance, three advances
    // across the block.
    let data = font(EMOJI);
    let face = Face::parse(&data, 0).unwrap();
    let advance =
        f64::from(face.glyph_hor_advance(GlyphId(3)).unwrap()) / f64::from(face.units_per_em());
    let (bw, bh) = block(built);
    assert!((bw - 3.0 * advance).abs() < 1e-9, "{bw}");
    let rects = box_rects(&laid);
    for (i, (r, slot)) in rects.iter().zip([0.0, 2.0]).enumerate() {
        let mid = (slot + 0.5) * advance;
        assert!(((r[0] + r[2]) / 2.0 - mid).abs() < 1e-4, "box {i}: {r:?}");
        assert!(
            (r[2] - r[0] - text::ERROR_BOX[0]).abs() < 1e-4,
            "box {i}: {r:?}"
        );
    }
    let fit = Fit::of(bw, bh, W, H);
    let Some((_g, mut r)) = gpu(W, H) else { return };
    let cover = render(
        &mut r,
        &loaded(&label_scene(EMOJI, words, Align::Left, fit)),
    );
    assert_striped(&cover, W, fit, &rects, "emoji");
    assert_eq!(
        r.text_errors(),
        ["text 'label': font 'face' has no outline for '\u{1f600}', '\u{1f642}'"]
    );
    let draw = render(&mut r, &loaded(&hazard_scene(EMOJI, words, fit)));
    assert_hazard(&draw, W, fit, &rects, "emoji draw");
}

#[test]
fn a_cff2_font_without_a_variation_store_draws_like_its_cff_source() {
    let words = "IO\nOI";
    assert!(near(
        built(CFF2, words, Align::Center),
        built(CFF, words, Align::Center)
    ));
    let laid = lay(CFF, words, Align::Center);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let Some((_g, mut r)) = gpu(W, H) else { return };
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
    let cff = render(&mut r, &loaded(&draw_scene(CFF, words, Align::Center, fit)));
    let cff2 = render(
        &mut r,
        &loaded(&draw_scene(CFF2, words, Align::Center, fit)),
    );
    let (n, m) = apart(&cff2, &cff);
    assert!(n == 0 && m <= 1, "draws: {n} pixels apart, at most {m}");
}

#[test]
fn words_that_fit_no_texture_at_setup_draw_nothing_and_the_scene_runs() {
    let fit = Fit {
        x: 8.0,
        y: 8.0,
        ppem: 64.0,
    };
    let words = "I".repeat(400_000);
    let Some((_g, mut r)) = gpu(W, H) else { return };
    for doc in [
        label_scene(SANS, &words, Align::Left, fit),
        draw_scene(SANS, &words, Align::Left, fit),
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
fn a_draw_keeps_its_last_words_and_their_boxes_past_words_that_fit_no_texture() {
    let fit = Fit {
        x: 8.0,
        y: 8.0,
        ppem: 48.0,
    };
    let Some((_g, mut r)) = gpu(W, H) else { return };
    let before = render(&mut r, &loaded(&hazard_scene(EMOJI, "\u{1f600}", fit)));
    assert!(r.set_text("label", &"\u{1f600}".repeat(400_000)).unwrap());
    r.set_time(Some(0.0));
    assert!(r.render(None).unwrap() == before.as_slice());
    // The box stays, and so does the problem behind it.
    let errors = r.text_errors();
    assert_eq!(errors.len(), 2, "{errors:?}");
    assert!(
        errors[0].contains("has no outline for '\u{1f600}'"),
        "{errors:?}"
    );
    assert!(errors[1].contains("VC4"), "{errors:?}");
}

// Draws

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

// Draws

/// A draw's place that puts the block where `fit` does.
fn fit_place(fit: Fit) -> String {
    format!(
        "vec2 glyph_place(vec2 em) {{ return vec2({:.4}, {:.4}) + em * {:.4}; }}",
        fit.x, fit.y, fit.ppem
    )
}

const BLACK: &str = "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }";
/// White coverage, premultiplied.
const WHITE: &str = "void main(){ float c = glyph_cover(); gl_FragColor = vec4(c); }";

/// label_scene with the words drawn by one draw over a black pass.
fn draw_scene(file: &str, words: &str, align: Align, fit: Fit) -> Value {
    let mut doc = label_scene(file, words, align, fit);
    doc["passes"][0] = json!({"name": "main", "code": BLACK,
        "draws": [{"text": "label", "code": WHITE, "place": fit_place(fit)}]});
    doc
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
/// sample decide it, and a draw reaches its sample by another sum than
/// `<name>_cover` does. Every other pixel lands within 1 level.
const CORNERS: f64 = 0.005;

fn assert_like_cover(draw: &[u8], cover: &[u8], what: &str) {
    let inked = cover.iter().step_by(4).filter(|v| **v > 0).count();
    let (n, m) = apart(draw, cover);
    assert!(inked > 200, "{what}: {inked} pixels inked");
    assert!(
        (n as f64) <= CORNERS * inked as f64,
        "{what}: {n} of {inked} inked pixels more than 1 level apart, at most {m}"
    );
}

#[test]
fn a_draw_covers_as_the_text_does() {
    let Some((_g, mut r)) = gpu(320, 120) else {
        return;
    };
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
            let draw = render(&mut r, &loaded(&draw_scene(file, words, Align::Left, fit)));
            assert_like_cover(&draw, &cover, &what);
        }
    }
}

#[test]
fn a_draw_follows_its_words_live() {
    let (w, h) = (256, 64);
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 40.0,
    };
    let Some((_g, mut r)) = gpu(w, h) else { return };
    let want = render(
        &mut r,
        &loaded(&label_scene(SANS, "ghee", Align::Left, fit)),
    );
    let before = render(&mut r, &loaded(&draw_scene(SANS, "I", Align::Left, fit)));
    assert!(r
        .set_scene_param("label", "string", &ParamValue::Str("ghee".into()))
        .unwrap());
    r.set_time(Some(0.0));
    let after = r.render(None).unwrap().to_vec();
    assert!(after != before, "the draw kept the old words");
    assert_like_cover(&after, &want, "ghee after I");
}

#[test]
fn a_rotated_and_scaled_draw_lands_its_outline() {
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
        "draws": [{"text": "label", "code": WHITE, "place": place}]});
    let Some((_g, mut r)) = gpu(w, h) else { return };
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
fn draws_land_in_their_order_and_pad_past_the_ink() {
    let (w, h) = (96, 96);
    let fit = Fit {
        x: 32.0,
        y: 8.0,
        ppem: 64.0,
    };
    let mut doc = label_scene(SANS, "I", Align::Left, fit);
    let place = fit_place(fit);
    doc["passes"][0] = json!({"name": "main", "code": BLACK, "draws": [
        {"text": "label", "place": place, "pad": 0.25, "blend": "add",
         "code": "void main(){ gl_FragColor = vec4(0.0, 0.0, 0.5, 0.5); }"},
        {"text": "label", "place": place,
         "code": "void main(){ float c = glyph_cover(); gl_FragColor = vec4(c, 0.0, 0.0, c); }"},
    ]});
    let laid = lay(SANS, "I", Align::Left);
    let g = &laid.glyphs[0];
    let px = |em: [f64; 2]| {
        let (x, y) = (fit.x + em[0] * fit.ppem, fit.y + em[1] * fit.ppem);
        (x as usize, y as usize)
    };
    let mid = (g.lo[1] + g.hi[1]) / 2.0;
    let Some((_g, mut r)) = gpu(w, h) else { return };
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
fn a_draw_reads_int_params_in_both_stages() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    // The place moves the block n pixels right and m down, and the code scales
    // the coverage by n - m, so the draw matches the text 3 right and 2 down
    // only when both stages read both params.
    let mut doc = draw_scene(SANS, "ghee", Align::Left, fit);
    doc["params"]["n"] = json!({"type": "int", "value": 3});
    doc["passes"][0]["params"] = json!({"m": {"type": "int", "value": 2}});
    doc["passes"][0]["draws"][0]["place"] =
        json!(fit_place(fit).replace("return ", "return vec2(float(n), float(m)) + "));
    doc["passes"][0]["draws"][0]["code"] =
        json!(WHITE.replace("vec4(c)", "vec4(c * float(n - m))"));
    let Some((_g, mut r)) = gpu(160, 48) else {
        return;
    };
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
fn a_draw_compile_error_names_the_pass_the_draw_and_its_line() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let good = draw_scene(SANS, "I", Align::Left, fit);
    let Some((_g, mut r)) = gpu(64, 48) else {
        return;
    };
    let first = render(&mut r, &loaded(&good));
    let mut bad = good.clone();
    bad["passes"][0]["draws"][0]["code"] =
        json!("void main(){\n    float c = glyph_cover();\n    nope(c);\n}");
    let e = r.setup(&loaded(&bad)).unwrap();
    assert!(
        e.starts_with("pass 'main' draw 0, line 3 of its code: "),
        "{e}"
    );
    assert_eq!(r.error_pass, Some(ErrorPass::Index(0)));
    // The console marks error_line in the pass's own code, which holds no draw.
    assert_eq!(r.error_line, None);
    let mut bad = good.clone();
    bad["passes"][0]["draws"][0]["place"] =
        json!("vec2 glyph_place(vec2 em) {\n    return nope;\n}");
    let e = r.setup(&loaded(&bad)).unwrap();
    assert!(
        e.starts_with("pass 'main' draw 0, line 2 of its place: "),
        "{e}"
    );
    // An error on ghee's own lines carries no line, and a missing function
    // is named.
    let mut bad = good.clone();
    bad["passes"][0]["draws"][0]["code"] =
        json!("float shade(){\n    return glyph_cover();\n}\nfloat x = 0.0;");
    let e = r.setup(&loaded(&bad)).unwrap();
    assert_eq!(e, "pass 'main' draw 0: its code needs a void main()");
    let mut bad = good.clone();
    bad["passes"][0]["draws"][0]["place"] = json!("vec2 place(vec2 em) {\n    return em;\n}");
    let e = r.setup(&loaded(&bad)).unwrap();
    assert_eq!(
        e,
        "pass 'main' draw 0: its place needs vec2 glyph_place(vec2 em)"
    );
    // The running scene stays.
    r.set_time(Some(0.0));
    assert!(r.render(None).unwrap() == first.as_slice());
}

#[test]
fn draws_round_trip_and_over_and_no_pad_are_the_defaults() {
    let mut doc = texts_scene(json!({"label": {"font": "sans"}}));
    doc["passes"][0]["draws"] = json!([
        {"text": "label", "code": WHITE, "place": "vec2 glyph_place(vec2 em) { return em; }"},
        {"text": "label", "code": WHITE, "place": "vec2 glyph_place(vec2 em) { return em; }",
         "blend": "add", "pad": 0.125},
        {"text": "label", "code": WHITE, "place": "vec2 glyph_place(vec2 em) { return em; }",
         "blend": "over", "pad": 0, "alpha": "blend"},
        {"text": "label", "code": WHITE, "place": "vec2 glyph_place(vec2 em) { return em; }",
         "alpha": "keep"},
    ]);
    let sc = Scene::from_dict(&doc, None, None).unwrap();
    let back = sc.to_dict();
    assert_eq!(back["passes"][0]["draws"][0], doc["passes"][0]["draws"][0]);
    assert_eq!(back["passes"][0]["draws"][1], doc["passes"][0]["draws"][1]);
    assert_eq!(
        back["passes"][0]["draws"][2], doc["passes"][0]["draws"][0],
        "over, pad 0 and alpha blend are left out"
    );
    assert_eq!(back["passes"][0]["draws"][3], doc["passes"][0]["draws"][3]);
    assert_eq!(Scene::from_dict(&back, None, None).unwrap(), sc);
}

#[test]
fn a_pass_of_draws_alone_round_trips_with_no_code_and_a_pass_of_neither_is_refused() {
    let place = "vec2 glyph_place(vec2 em) { return em; }";
    let mut doc = texts_scene(json!({"label": {"font": "sans"}}));
    doc["passes"] = json!([
        {"name": "words", "draws": [{"text": "label", "code": WHITE, "place": place}]},
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
             the scene) and no 'draws'"
        );
    }
}

#[test]
fn a_draw_declaration_names_what_it_lacks() {
    let place = "vec2 glyph_place(vec2 em) { return em; }";
    let with = |draws: Value| {
        let mut doc = texts_scene(json!({"label": {"font": "sans"}}));
        doc["passes"][0]["draws"] = draws;
        scene_err(&doc)
    };
    let one = |d: Value| with(json!([d]));
    let cases = [
        (with(json!(3)), "pass 'main': 'draws' needs a list, got 3"),
        (one(json!(3)), "pass 'main' draw 0 needs an object, got 3"),
        (
            one(json!({"code": WHITE, "place": place})),
            "pass 'main' draw 0 has no 'text', the name of a scene text",
        ),
        (
            one(json!({"text": "label", "place": place})),
            "pass 'main' draw 0 has no 'code', fragment GLSL with its own void main()",
        ),
        (
            one(json!({"text": "label", "code": WHITE})),
            "pass 'main' draw 0 has no 'place', vertex GLSL that defines vec2 glyph_place(vec2 em)",
        ),
        (
            one(json!({"text": "label", "code": " ", "place": place})),
            "pass 'main' draw 0 'code' needs fragment GLSL with its own void main(), got ' '",
        ),
        (
            one(json!({"text": "label", "code": WHITE, "place": place, "blend": "max"})),
            "pass 'main' draw 0 blend needs 'over', 'add' or 'replace', got 'max'",
        ),
        (
            one(json!({"text": "label", "code": WHITE, "place": place, "pad": -1})),
            "pass 'main' draw 0 pad needs a number of em, 0 or more, got -1",
        ),
        (
            one(json!({"text": "label", "code": WHITE, "place": place, "alpha": "add"})),
            "pass 'main' draw 0 alpha needs 'blend' or 'keep', got 'add'",
        ),
        (
            one(json!({"text": "words", "code": WHITE, "place": place})),
            "pass 'main' draw 0 names text 'words', which the scene does not declare",
        ),
        (
            with(Value::from(vec![
                json!({"text": "label", "code": WHITE, "place": place});
                17
            ])),
            "17 draws, over the cap of 16",
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
            "draws": [{"text": "label", "code": WHITE, "place": place}]}));
    let got = scene_err(&doc);
    assert!(
        got.ends_with("pass 'd' is filter 'dither', and a filter pass takes no draws"),
        "{got}"
    );
}

#[test]
fn a_draw_on_a_sized_pass_covers_as_the_text_does_there() {
    // Pass a renders at half the frame on each axis, and main shows it.
    let (w, h) = (320, 120);
    let words = "AV To\nghee";
    let laid = lay(SANS, words, Align::Left);
    let fit = Fit::of(laid.width, laid.height, w / 2, h / 2);
    let show = "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }";
    let sized = |pass: Value| {
        let mut doc = label_scene(SANS, words, Align::Left, fit);
        let mut a = pass;
        a["name"] = json!("a");
        a["size"] = json!([w / 2, h / 2]);
        doc["passes"] = json!([a, {"name": "main", "channels": [{"buffer": "a"}], "code": show}]);
        loaded(&doc)
    };
    let cover = sized(label_scene(SANS, words, Align::Left, fit)["passes"][0].clone());
    let draw = sized(draw_scene(SANS, words, Align::Left, fit)["passes"][0].clone());
    let Some((_g, mut r)) = gpu(w, h) else { return };
    let want = render(&mut r, &cover);
    let got = render(&mut r, &draw);
    assert_like_cover(&got, &want, "a draw on a 160x60 pass");
}

#[test]
fn one_text_drawn_by_two_passes_follows_its_words_in_both() {
    // Pass a draws the words in red. main shows a and adds them in green.
    let (w, h) = (256, 64);
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 40.0,
    };
    let place = fit_place(fit);
    let mut doc = label_scene(SANS, "AV", Align::Left, fit);
    doc["passes"] = json!([
        {"name": "a", "code": BLACK, "draws": [{"text": "label", "place": place,
            "code": "void main(){ float c = glyph_cover(); gl_FragColor = vec4(c, 0.0, 0.0, c); }"}]},
        {"name": "main", "channels": [{"buffer": "a"}],
         "code": "void main(){ gl_FragColor = texture2D(iChannel0, v_texCoord); }",
         "draws": [{"text": "label", "place": place, "blend": "add",
            "code": "void main(){ float c = glyph_cover(); gl_FragColor = vec4(0.0, c, 0.0, c); }"}]},
    ]);
    let Some((_g, mut r)) = gpu(w, h) else { return };
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
fn a_param_takes_an_old_attribute_name_and_ghee_keeps_vflip_and_glyph_names() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let mut doc = draw_scene(SANS, "ghee", Align::Left, fit);
    doc["params"]["a1"] = json!({"type": "float", "value": 1.0});
    doc["passes"][0]["draws"][0]["code"] = json!(WHITE.replace("vec4(c)", "vec4(c * a1)"));
    let Some((_g, mut r)) = gpu(160, 48) else {
        return;
    };
    let want = render(
        &mut r,
        &loaded(&label_scene(SANS, "ghee", Align::Left, fit)),
    );
    let got = render(&mut r, &loaded(&doc));
    assert_like_cover(&got, &want, "a param named a1");
    // A scene with no draw takes the names.
    let mut doc = label_scene(SANS, "ghee", Align::Left, fit);
    doc["params"]["vflip"] = json!({"type": "float", "value": 1.0});
    doc["passes"][0]["params"] = json!({"glyph_index": {"type": "float", "value": 1.0}});
    assert!(Scene::from_dict(&doc, None, None).is_ok());
    let kept = "vflip and every name that starts with glyph_ belong to ghee";
    for name in ["vflip", "glyph_a1", "glyph_index"] {
        let mut doc = draw_scene(SANS, "ghee", Align::Left, fit);
        doc["params"][name] = json!({"type": "float", "value": 1.0});
        assert_eq!(scene_err(&doc), format!("scene: param '{name}': {kept}"));
        let mut doc = draw_scene(SANS, "ghee", Align::Left, fit);
        doc["passes"][0]["params"] = json!({name: {"type": "float", "value": 1.0}});
        assert_eq!(
            scene_err(&doc),
            format!("scene: param '{name}' in pass 'main': {kept}")
        );
    }
}

#[test]
fn a_draw_past_the_texture_units_fails_naming_the_draw() {
    // main's channels and array take every unit, and its draw needs one
    // more for the text.
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let Some((_g, mut r)) = gpu(160, 48) else {
        return;
    };
    let before = render(&mut r, &loaded(&draw_scene(SANS, "I", Align::Left, fit)));
    let units = r.max_texture_units() as usize;
    let mut doc = draw_scene(SANS, "I", Align::Left, fit);
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
            "pass 'main' draw 0 needs {} texture units, {past} {units}",
            units + 1
        )
    );
    r.set_time(Some(0.0));
    assert!(r.render(None).unwrap() == before.as_slice());
}

#[test]
fn a_draw_that_samples_past_the_texture_units_fails_naming_the_draw() {
    // main reads one channel, and its draw samples the label's texture and an
    // image in every other unit.
    let Some((_g, mut r)) = gpu(64, 48) else {
        return;
    };
    let units = r.max_texture_units() as usize;
    let dir = tmpdir("units");
    fs::copy(format!("{FONTS}/{SANS}"), format!("{dir}/{SANS}")).unwrap();
    let names: Vec<String> = (0..units - 1).map(|i| format!("img{i}")).collect();
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let mut doc = draw_scene(SANS, "I", Align::Left, fit);
    for n in &names {
        fs::write(format!("{dir}/{n}.raw"), [128u8]).unwrap();
        doc["assets"][n] =
            json!({"file": format!("{n}.raw"), "width": 1, "height": 1, "format": "u8"});
    }
    let sum: Vec<String> = names
        .iter()
        .map(|n| format!("texture2D({n}, vec2(0.5)).r"))
        .collect();
    doc["passes"][0]["draws"][0]["code"] = json!(format!(
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
            "pass 'main' draw 0 needs {} texture units, {past} {units}",
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

/// draw_scene with its draw in a pass of draws alone, `words`, and that pass
/// the output.
fn words_only() -> Value {
    let mut doc = draw_scene(SANS, "I", Align::Left, I_FIT);
    let draws = doc["passes"][0]["draws"].clone();
    doc["passes"] = json!([{"name": "words", "draws": draws}]);
    doc
}

#[test]
fn a_draw_over_a_transparent_pass_leaves_its_coverage_in_alpha() {
    let Some((_g, mut r)) = gpu(64, 56) else {
        return;
    };
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
fn a_pass_of_draws_alone_starts_transparent_and_a_later_pass_composites_it() {
    let Some((_g, mut r)) = gpu(64, 56) else {
        return;
    };
    // The same draw over black, and the words pass composited over blue.
    let over_black = render(&mut r, &loaded(&draw_scene(SANS, "I", Align::Left, I_FIT)));
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
fn a_draw_that_keeps_alpha_leaves_the_pass_alpha() {
    let Some((_g, mut r)) = gpu(64, 56) else {
        return;
    };
    r.set_alpha(true);
    let mut frames = Vec::new();
    for alpha in ["blend", "keep"] {
        let mut doc = draw_scene(SANS, "I", Align::Left, I_FIT);
        doc["passes"][0]["code"] =
            json!("void main(){ gl_FragColor = vec4(0.0, 0.0, 0.25, 0.5); }");
        doc["passes"][0]["draws"][0]["alpha"] = json!(alpha);
        frames.push(render(&mut r, &loaded(&doc)));
    }
    let (blend, keep) = (&frames[0], &frames[1]);
    for c in 0..3 {
        assert_eq!(plane(blend, c), plane(keep, c), "channel {c}");
    }
    let half = alphas(keep)[0];
    assert!(half.abs_diff(128) <= 1, "{half}");
    assert!(alphas(keep).iter().all(|a| *a == half));
    assert!(alphas(blend).contains(&255));
}

#[test]
fn a_reset_drops_the_draws_with_the_scene() {
    let fit = Fit {
        x: 4.0,
        y: 4.0,
        ppem: 30.0,
    };
    let Some((_g, mut r)) = gpu(64, 48) else {
        return;
    };
    render(&mut r, &loaded(&draw_scene(SANS, "I", Align::Left, fit)));
    r.reset();
    assert!(r.set_text("label", "O").unwrap());
}
