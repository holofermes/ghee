//! Scene texts: the layout, the scene model, the font assets, and the
//! coverage a pass reads back on this host's GPU.
//!
//! The reference below rebuilds the promised layout and outline from the font
//! with ttf-parser, so every check holds against any packing of the texture.

use std::fs;
use std::sync::{Arc, Mutex, MutexGuard};

use serde_json::{json, Map, Value};
use ttf_parser::{Face, GlyphId, OutlineBuilder};

use ghee::engine::{probe_dri_devices, Renderer};
use ghee::filters::apply_filters;
use ghee::scene::{Param, ParamValue, Pass, Scene, SceneBuilder};
use ghee::text::{self, Align, Text};

const FONTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts");
/// DejaVu Sans cut to printable ASCII, TrueType outlines, a kern table.
const SANS: &str = "DejaVuSans-ASCII.ttf";
/// Noto Sans CJK JP cut to I and O, CFF outlines.
const CFF: &str = "NotoSansCJKjp-IO.otf";

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
    let h = frame.len() / 4 / w;
    let (mut inside, mut outside, mut off) = (0, 0, Vec::new());
    for y in 0..h {
        for x in 0..w {
            let Some(is_in) = laid.inside(fit.em(x, y), EDGE / fit.ppem) else {
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

fn built(file: &str, words: &str, align: Align) -> (f64, f64) {
    block(&text::build(&font(file), words, align).unwrap())
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
fn bytes_that_are_no_font_fail_as_an_unreadable_font() {
    let e = text::build(b"not a font", "I", Align::Left)
        .err()
        .unwrap()
        .to_string();
    assert!(e.starts_with("unreadable font"), "{e}");
    let mut t = Text::from_dict("label", &json!({"font": "face"})).unwrap();
    t.data = Some(Arc::new(b"not a font".to_vec()));
    let e = t.build("I").err().unwrap().to_string();
    assert!(e.starts_with("text 'label': unreadable font"), "{e}");
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

#[test]
fn words_past_a_vc4_texture_fail_naming_the_limit() {
    // One line of glyphs, far past what one texture holds.
    let words = "I".repeat(400_000);
    let e = text::build(&font(SANS), &words, Align::Left)
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("VC4"), "{e}");
}

#[test]
fn a_block_too_wide_and_too_tall_for_the_texture_fails_naming_the_limit() {
    // Few glyphs over a vast block: its grid alone would take gigabytes.
    let words = "I".repeat(30_000) + &"\n".repeat(30_000);
    let e = text::build(&font(SANS), &words, Align::Left)
        .err()
        .unwrap()
        .to_string();
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
            "text 'label' align needs 'left', 'center' or 'right'",
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
    assert!(Scene::from_dict(&with_images(7), None, None).is_ok());
    let e = scene_err(&with_images(8));
    assert!(
        e.ends_with("9 image assets and texts, over the 8 the engine binds"),
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
    sc.texts[0].data = Some(Arc::new(font(SANS)));
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
    t.data.as_ref().unwrap().to_vec()
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
    assert!(Arc::ptr_eq(a.unwrap(), b.unwrap()));
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
    let e = with_assets_env(&store, || sc.load_assets(&dir))
        .err()
        .unwrap()
        .to_string();
    assert_eq!(
        e,
        format!(
            "asset 'face' ({dir}/{file}): No such file or directory \
             (also tried: {root}/{file}, {store}/{file})"
        )
    );
    fs::remove_dir_all(&root).unwrap();
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
fn a_font_file_that_is_no_font_fails_its_load() {
    let _g = lock();
    let dir = tmpdir("nofont");
    fs::write(format!("{dir}/bad.ttf"), b"not a font").unwrap();
    let mut sc = face_scene("bad.ttf");
    let e = with_assets_env("", || sc.load_assets(&dir))
        .err()
        .unwrap()
        .to_string();
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
    // Words the texture cannot hold fail, and the last words stay.
    let e = r
        .set_text("label", &"I".repeat(400_000))
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("VC4"), "{e}");
    let f = r.render(None).unwrap().to_vec();
    assert_draws(&f, W, 0, &o, fit, "O kept");
    assert!(r.set_text("label", "").unwrap());
    assert!(blank(r.render(None).unwrap(), 0));
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
fn every_text_name_a_renderer_meets_draws() {
    // The engine binds 8 assets at a time, so each setup has to hand back
    // the slots of the texts its scene no longer declares.
    let laid = lay(SANS, "I", Align::Left);
    let fit = Fit::of(laid.width, laid.height, W, H);
    let Some((_g, mut r)) = gpu(W, H) else { return };
    for i in 0..12 {
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
