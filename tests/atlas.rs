//! The MTSDF atlas against exact coverage: fields from `ghee::atlas`, packed in
//! an atlas and read back the way the fill shader reads them, against what
//! `<name>_cover` draws on this host's GPU.

use std::collections::HashSet;
use std::fs;

use serde_json::json;
use ttf_parser::{Face, GlyphId, OutlineBuilder};

use ghee::atlas::{self, Atlas, Cell, Curve};
use ghee::engine::{probe_dri_devices, Renderer};
use ghee::scene::Scene;

const FONTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tools/fixtures/fonts");
/// DejaVu Sans cut to printable ASCII, TrueType outlines, a kern table.
const SANS: &str = "DejaVuSans-ASCII.ttf";
const LYRICS: &str = "the radio hums a borrowed tune\n\
                      we drove until the map ran out\n\
                      and every streetlight knew our names\n\
                      so sing it back the way it was\n\
                      a little louder, little slower\n\
                      the night is long and so are we\n\
                      hold the note until it breaks";
const W: usize = 1280;
const H: usize = 800;
const PPEM: f64 = 46.0;

/// A W x H RGBA renderer on the first GL device that opens. None = no device,
/// and the test reports the skip.
fn gpu() -> Option<Renderer> {
    let mut errs = Vec::new();
    for dev in probe_dri_devices() {
        match Renderer::new(W as i32, H as i32, 4, Some(&dev), false, false) {
            Ok(r) => return Some(r),
            Err(e) => errs.push(format!("{dev}: {e}")),
        }
    }
    eprintln!("no GL device for rendering: {}", errs.join("; "));
    None
}

/// A TrueType glyph's contours as curves in em.
struct Contours {
    scale: f64,
    start: [f64; 2],
    at: [f64; 2],
    out: Vec<Vec<Curve>>,
}

impl Contours {
    fn point(&self, x: f32, y: f32) -> [f64; 2] {
        [f64::from(x) * self.scale, f64::from(y) * self.scale]
    }

    fn push(&mut self, c: Curve) {
        self.out.last_mut().unwrap().push(c);
        self.at = c[2];
    }
}

impl OutlineBuilder for Contours {
    fn move_to(&mut self, x: f32, y: f32) {
        self.start = self.point(x, y);
        self.at = self.start;
        self.out.push(Vec::new());
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.point(x, y);
        self.push([self.at, p, p]);
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let (c, p) = (self.point(x1, y1), self.point(x, y));
        self.push([self.at, c, p]);
    }

    fn curve_to(&mut self, _: f32, _: f32, _: f32, _: f32, _: f32, _: f32) {
        panic!("the fixture font draws quadratics");
    }

    fn close(&mut self) {
        if self.at != self.start {
            let s = self.start;
            self.push([self.at, s, s]);
        }
    }
}

fn contours(face: &Face, gid: u16) -> Vec<Vec<Curve>> {
    let mut b = Contours {
        scale: 1.0 / f64::from(face.units_per_em()),
        start: [0.0; 2],
        at: [0.0; 2],
        out: Vec::new(),
    };
    face.outline_glyph(GlyphId(gid), &mut b);
    b.out
}

fn kerning(face: &Face, l: GlyphId, r: GlyphId) -> f64 {
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

/// The centred layout a text draws: each glyph's id and origin in em from the
/// block's bottom left, y up, on the 1/4096 em lattice, then the block's width
/// and height.
fn layout(face: &Face, words: &str) -> (Vec<(u16, f64, f64)>, f64, f64) {
    let upem = f64::from(face.units_per_em());
    let ascent = f64::from(face.ascender()) / upem;
    let line = (f64::from(face.ascender()) - f64::from(face.descender())
        + f64::from(face.line_gap()))
        / upem;
    let lines: Vec<&str> = words.split('\n').collect();
    let height = line * lines.len() as f64;
    let mut placed = Vec::new();
    let mut widths = Vec::new();
    for (k, text) in lines.iter().enumerate() {
        let y = height - ascent - line * k as f64;
        let first = placed.len();
        let mut x = 0.0;
        let mut prev = None;
        for ch in text.chars() {
            let g = face.glyph_index(ch).unwrap_or(GlyphId(0));
            if let Some(p) = prev {
                x += kerning(face, p, g) / upem;
            }
            placed.push((g.0, x, y));
            x += f64::from(face.glyph_hor_advance(g).unwrap_or(0)) / upem;
            prev = Some(g);
        }
        widths.push((first, placed.len(), x));
    }
    let width = widths.iter().map(|w| w.2).fold(0.0, f64::max);
    for (from, to, w) in widths {
        for p in &mut placed[from..to] {
            p.1 += (width - w) / 2.0;
        }
    }
    let snap = |v: f64| (v * 4096.0).round() / 4096.0;
    let placed = placed.into_iter().map(|(g, x, y)| (g, snap(x), snap(y)));
    (placed.collect(), width, height)
}

/// A GL_LINEAR read of `a` at texel coordinate `t` (texel i spans i to i + 1),
/// clamped to the edge. Panics when a tap leaves `cell`.
fn bilinear(a: &Atlas, cell: &Cell, t: [f64; 2]) -> [f64; 4] {
    let (x0, y0) = (t[0] - 0.5, t[1] - 0.5);
    let (ix, iy) = (x0.floor(), y0.floor());
    let (fx, fy) = (x0 - ix, y0 - iy);
    let [cx, cy, cw, ch] = cell.rect.map(f64::from);
    assert!(
        ix >= cx && ix + 1.0 < cx + cw && iy >= cy && iy + 1.0 < cy + ch,
        "a tap at {t:?} leaves the cell {:?}",
        cell.rect
    );
    let get = |x: f64, y: f64, k: usize| {
        let xi = (x as i64).clamp(0, i64::from(a.width) - 1) as usize;
        let yi = (y as i64).clamp(0, i64::from(a.height) - 1) as usize;
        f64::from(a.texels[(yi * a.width as usize + xi) * 4 + k]) / 255.0
    };
    [0, 1, 2, 3].map(|k| {
        let lo = get(ix, iy, k) * (1.0 - fx) + get(ix + 1.0, iy, k) * fx;
        let hi = get(ix, iy + 1.0, k) * (1.0 - fx) + get(ix + 1.0, iy + 1.0, k) * fx;
        lo * (1.0 - fy) + hi * fy
    })
}

#[test]
fn the_mtsdf_fill_at_46_px_em_matches_exact_coverage() {
    let Some(mut r) = gpu() else { return };
    let code = "void main() {\n\
                vec2 px = floor(v_texCoord * resolution) + 0.5;\n\
                vec2 off = (resolution - t_size * 46.0) * 0.5;\n\
                float c = t_cover((px - off) / 46.0, 46.0);\n\
                gl_FragColor = vec4(vec3(c), 1.0);\n\
                }";
    let doc = json!({
        "version": 1,
        "params": {"t": {"type": "string", "value": LYRICS}},
        "assets": {"font": {"file": SANS}},
        "texts": {"t": {"font": "font", "align": "center"}},
        "passes": [{"name": "main", "code": code}],
    });
    let mut sc = Scene::from_dict(&doc, None, None).unwrap();
    sc.load_assets(FONTS).unwrap();
    assert_eq!(r.setup(&sc).unwrap(), "");
    r.set_time(Some(0.0));
    let cover = r.render(None).unwrap().to_vec();

    let data = fs::read(format!("{FONTS}/{SANS}")).unwrap();
    let face = Face::parse(&data, 0).unwrap();
    let (placed, width, height) = layout(&face, LYRICS);
    let mut a = Atlas::default();
    let mut seen = HashSet::new();
    for &(gid, ..) in &placed {
        if seen.insert(gid) {
            let f = atlas::field(&contours(&face, gid), atlas::PX_EM, atlas::RANGE).unwrap();
            if let Some(f) = f {
                a.insert(gid, &f).unwrap();
            }
        }
    }

    // Each glyph's rectangle spans its ink box and 1 px around it. A fragment
    // takes its point's texel coordinate as Cell states it, and its coverage
    // from the median at the screen's px range, like glyph_cover().
    let s = f64::from(atlas::PX_EM);
    let px_range = f64::from(atlas::RANGE) * PPEM / s;
    let off = [
        (W as f64 - width * PPEM) * 0.5,
        (H as f64 - height * PPEM) * 0.5,
    ];
    let mut fill = vec![0.0f64; W * H];
    let mut drawn = 0;
    for &(gid, ox, oy) in &placed {
        let Some(c) = a.cells.get(&gid) else { continue };
        drawn += 1;
        let m = 1.0 / PPEM;
        let col = |x: f64| off[0] + x * PPEM;
        let row = |y: f64| off[1] + (height - y) * PPEM;
        let c0 = col(ox + c.ink[0] - m).floor().max(0.0) as usize;
        let c1 = col(ox + c.ink[2] + m).ceil().min(W as f64) as usize;
        let r0 = row(oy + c.ink[3] + m).floor().max(0.0) as usize;
        let r1 = row(oy + c.ink[1] - m).ceil().min(H as f64) as usize;
        for y in r0..r1 {
            for x in c0..c1 {
                let p = [
                    (x as f64 + 0.5 - off[0]) / PPEM - ox,
                    height - (y as f64 + 0.5 - off[1]) / PPEM - oy,
                ];
                let t = [
                    f64::from(c.rect[0]) + c.pad + (p[0] - c.ink[0]) * s,
                    f64::from(c.rect[1]) + c.pad + (p[1] - c.ink[1]) * s,
                ];
                let v = bilinear(&a, c, t);
                let med = v[0].min(v[1]).max(v[0].max(v[1]).min(v[2]));
                let k = (px_range * (med - 0.5) + 0.5).clamp(0.0, 1.0);
                let i = y * W + x;
                fill[i] = 1.0 - (1.0 - fill[i]) * (1.0 - k);
            }
        }
    }
    assert!(drawn > 150, "{drawn} glyphs drawn");

    let mut errs = Vec::new();
    for (i, f) in fill.iter().enumerate() {
        let want = f64::from(cover[i * 4]) / 255.0;
        if want > 0.0 || *f > 0.0 {
            errs.push((f - want).abs() * 255.0);
        }
    }
    let mean = errs.iter().sum::<f64>() / errs.len() as f64;
    let max = errs.iter().copied().fold(0.0, f64::max);
    eprintln!(
        "MTSDF fill against t_cover at 46 px/em: {} px, mean {mean:.2}/255, max {max:.1}/255",
        errs.len()
    );
    assert!(errs.len() > 50_000, "{} pixels compared", errs.len());
    assert!(mean < 4.0, "mean {mean:.2}/255");
}
