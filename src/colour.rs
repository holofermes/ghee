//! Colour glyphs: the bitmaps of colour emoji.
//!
//! A glyph with a PNG bitmap in its font's CBDT or sbix table, and no outline,
//! is a colour glyph. A layer of a text puts the text's colour glyphs into its
//! pass through a colour batch, after the layers of the pass. The batch places
//! each glyph with the layer's `glyph_place`, and samples its bitmap from the
//! text's colour atlas, an RGBA8 texture with mipmaps. The layer's own code
//! never runs for a colour glyph, and `<name>_cover` reads one as no ink.

use anyhow::{bail, Result};
use ttf_parser::{Face, GlyphId, RasterGlyphImage, RasterImageFormat};

use crate::assets::{decode_png, Pixels};
use crate::atlas::Field;

/// The pixels per em asked of a font's bitmaps, which picks its largest
/// strike. Mipmaps shrink a bitmap to any smaller size.
const STRIKE: u16 = u16::MAX;

/// A cell holds its bitmap MIP_PAD transparent texels in from each edge, and
/// its sides are multiples of MIP_PAD, so every cell starts on a grid of
/// MIP_PAD texels. Down to mip level MAX_LEVEL, log2(MIP_PAD), a texel then
/// holds one cell alone, and a read up to REACH texels past a bitmap meets
/// no other bitmap. The batch reads no level past MAX_LEVEL.
const MIP_PAD: u32 = 16;
const MAX_LEVEL: f64 = 4.0;
const REACH: f64 = 8.0;

/// The colour bitmap of glyph `gid` of `face`, from the font's largest
/// strike. None when the glyph has no PNG bitmap there.
pub fn bitmap<'f>(face: &'f Face, gid: GlyphId) -> Option<RasterGlyphImage<'f>> {
    face.glyph_raster_image(gid, STRIKE)
        .filter(|i| i.format == RasterImageFormat::PNG)
}

/// The cell of colour glyph `gid` of `face`, and the texels per em of its
/// bitmap. The cell holds the bitmap premultiplied by alpha, rows from the
/// bottom up, MIP_PAD texels in from its left and bottom edges. Fails with
/// the reason for a bitmap that does not decode.
pub fn field(face: &Face, gid: u16) -> Result<(Field, f64)> {
    let Some(img) = bitmap(face, GlyphId(gid)) else {
        bail!("has no PNG bitmap");
    };
    let s = f64::from(img.pixels_per_em);
    if s == 0.0 {
        bail!("its bitmap's strike has 0 pixels per em");
    }
    let png =
        decode_png(img.data).map_err(|e| anyhow::anyhow!("its bitmap does not decode: {e}"))?;
    let Pixels::U8(px) = &png.pixels else {
        bail!("its bitmap decodes to floats");
    };
    let (w, h, n) = (png.width, png.height, png.components as usize);
    let cw = (w + 2 * MIP_PAD).next_multiple_of(MIP_PAD);
    let ch = (h + 2 * MIP_PAD).next_multiple_of(MIP_PAD);
    let mut texels = vec![0u8; cw as usize * ch as usize * 4];
    for row in 0..h as usize {
        let to = ((MIP_PAD as usize + h as usize - 1 - row) * cw as usize + MIP_PAD as usize) * 4;
        let from = &px[row * w as usize * n..(row + 1) * w as usize * n];
        for (i, p) in from.chunks_exact(n).enumerate() {
            let (rgb, a) = match n {
                1 => ([p[0]; 3], 255),
                3 => ([p[0], p[1], p[2]], 255),
                _ => ([p[0], p[1], p[2]], p[3]),
            };
            let at = to + i * 4;
            for (c, v) in rgb.iter().enumerate() {
                texels[at + c] = ((u32::from(*v) * u32::from(a) + 127) / 255) as u8;
            }
            texels[at + 3] = a;
        }
    }
    let (x, y) = (f64::from(img.x), f64::from(img.y));
    Ok((
        Field {
            width: cw,
            height: ch,
            texels,
            ink: [x / s, y / s, (x + f64::from(w)) / s, (y + f64::from(h)) / s],
            pad: f64::from(MIP_PAD),
        },
        s,
    ))
}

/// The mip levels of a colour atlas below level 0, the atlas's own texels,
/// built on the CPU. A texel of level k + 1 is the box average of the 2 x 2
/// texels above it in level k, premultiplied RGBA8, row 0 at v = 0. A new
/// cell changes the texels over it alone at every level, so an update
/// uploads those rects in place of whole levels.
#[derive(Default)]
pub struct Mips {
    /// Level k + 1 at index k: its width, its height and its texels.
    levels: Vec<(u32, u32, Vec<u8>)>,
}

/// A rect of a level: x, y, width and height in texels.
pub type Rect = [u32; 4];

impl Mips {
    /// The levels below level 0 of a `width` x `height` atlas of transparent
    /// texels, down to 1 x 1. Each side is a power of two.
    pub fn new(width: u32, height: u32) -> Mips {
        let mut levels = Vec::new();
        let (mut w, mut h) = (width, height);
        while w > 1 || h > 1 {
            (w, h) = ((w / 2).max(1), (h / 2).max(1));
            levels.push((w, h, vec![0; w as usize * h as usize * 4]));
        }
        Mips { levels }
    }

    /// Level `level`, 1 or deeper: its width, its height and its texels.
    pub fn level(&self, level: usize) -> (u32, u32, &[u8]) {
        let (w, h, t) = &self.levels[level - 1];
        (*w, *h, t)
    }

    /// The number of levels below level 0.
    pub fn count(&self) -> usize {
        self.levels.len()
    }

    /// Average again the texels under `rect` of level 0, the `width` x
    /// `height` atlas `texels`, into every level below it. Returns the rect
    /// that changed in each level, from level 1 down.
    pub fn refresh(&mut self, texels: &[u8], width: u32, height: u32, rect: Rect) -> Vec<Rect> {
        let [x, y, w, h] = rect;
        let (mut lo, mut hi) = ([x, y], [x + w, y + h]);
        let mut changed = Vec::with_capacity(self.levels.len());
        for k in 0..self.levels.len() {
            let (above, below) = self.levels.split_at_mut(k);
            let (sw, sh, src) = match above.last() {
                Some((w, h, t)) => (*w, *h, &t[..]),
                None => (width, height, texels),
            };
            let (dw, dh, dst) = &mut below[0];
            lo = [lo[0] / 2, lo[1] / 2];
            hi = [hi[0].div_ceil(2).min(*dw), hi[1].div_ceil(2).min(*dh)];
            let (row, span) = (sw as usize * 4, lo[0] as usize..hi[0] as usize);
            for ty in lo[1] as usize..hi[1] as usize {
                // A side of 1 texel pairs its texel with itself.
                let y1 = (2 * ty + 1).min(sh as usize - 1);
                let (r0, r1) = (&src[2 * ty * row..][..row], &src[y1 * row..][..row]);
                let out = &mut dst[ty * *dw as usize * 4..][..*dw as usize * 4];
                if sw == 1 {
                    let px = average(texel(r0), texel(r0), texel(r1), texel(r1));
                    out.copy_from_slice(&px.to_le_bytes());
                    continue;
                }
                let a = r0[8 * span.start..8 * span.end].chunks_exact(8);
                let b = r1[8 * span.start..8 * span.end].chunks_exact(8);
                let o = out[4 * span.start..4 * span.end].chunks_exact_mut(4);
                for ((a, b), o) in a.zip(b).zip(o) {
                    let px = average(texel(a), texel(&a[4..]), texel(b), texel(&b[4..]));
                    o.copy_from_slice(&px.to_le_bytes());
                }
            }
            changed.push([lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]]);
        }
        changed
    }

    /// Average the new cells `cells` of the `width` x `height` atlas
    /// `texels` into every level below level 0, and return the rects that
    /// changed, as (level, rect), level 0 included. Down to MAX_LEVEL each
    /// texel holds one cell, and cells side by side on a shelf join in one
    /// rect. Past it a texel spans several cells, and each level changes one
    /// rect around every changed texel.
    pub fn refresh_cells(
        &mut self,
        texels: &[u8],
        width: u32,
        height: u32,
        cells: &[Rect],
    ) -> Vec<(usize, Rect)> {
        let mut changed: Vec<Vec<Rect>> = vec![Vec::new(); self.levels.len() + 1];
        for &cell in cells {
            changed[0].push(cell);
            let below = self.refresh(texels, width, height, cell);
            for (rects, rect) in changed[1..].iter_mut().zip(below) {
                rects.push(rect);
            }
        }
        let mut out = Vec::new();
        for (level, mut rects) in changed.into_iter().enumerate() {
            if rects.is_empty() {
                continue;
            }
            if level as f64 > MAX_LEVEL {
                let lo = [0, 1].map(|i| rects.iter().map(|r| r[i]).min().unwrap_or(0));
                let hi = [0, 1].map(|i| rects.iter().map(|r| r[i] + r[i + 2]).max().unwrap_or(0));
                out.push((level, [lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]]));
                continue;
            }
            rects.sort_unstable_by_key(|r| (r[1], r[3], r[0]));
            let mut joined: Vec<Rect> = Vec::new();
            for r in rects {
                match joined.last_mut() {
                    Some(j) if j[1] == r[1] && j[3] == r[3] && j[0] + j[2] == r[0] => j[2] += r[2],
                    _ => joined.push(r),
                }
            }
            out.extend(joined.into_iter().map(|r| (level, r)));
        }
        out
    }
}

/// The RGBA8 texel at the start of `t`.
fn texel(t: &[u8]) -> u32 {
    u32::from_le_bytes([t[0], t[1], t[2], t[3]])
}

/// The rounded average of four RGBA8 texels, channel by channel. Each pair of
/// channels sums in a 16-bit lane of its own.
fn average(p: u32, q: u32, r: u32, s: u32) -> u32 {
    const M: u32 = 0x00ff_00ff;
    let lo = (p & M) + (q & M) + (r & M) + (s & M) + 0x0002_0002;
    let hi = ((p >> 8) & M) + ((q >> 8) & M) + ((r >> 8) & M) + ((s >> 8) & M) + 0x0002_0002;
    ((lo >> 2) & M) | (((hi >> 2) & M) << 8)
}

/// The texels of `rect` of the RGBA8 `texels`, rows `width` texels wide, as
/// rows of the rect's own width.
pub fn crop(texels: &[u8], width: u32, rect: Rect) -> Vec<u8> {
    let [x, y, w, h] = rect.map(|v| v as usize);
    let mut out = Vec::with_capacity(w * h * 4);
    for row in y..y + h {
        let at = (row * width as usize + x) * 4;
        out.extend_from_slice(&texels[at..at + w * 4]);
    }
    out
}

/// The vertex main of the colour batch of a layer of text `name`, after
/// glyph_place. Each box edge moves out 5/8 of a screen pixel, and never
/// past REACH texels of the bitmap.
pub fn vertex_main(name: &str) -> Vec<String> {
    let f = |v: f64| format!("{v:?}");
    VERTEX_MAIN
        .replace("NAME", name)
        .replace("MAX_LEVEL", &f(MAX_LEVEL))
        .replace("REACH", &f(REACH))
        .lines()
        .map(str::to_string)
        .collect()
}

/// The fragment shader of the colour batch of a layer of text `name`, after
/// its precision lines.
pub fn fragment(name: &str) -> Vec<String> {
    FRAGMENT
        .replace("NAME", name)
        .lines()
        .map(str::to_string)
        .collect()
}

pub(crate) const VERTEX_MAIN: &str = r#"uniform vec2 ghee_NAME_emoji_size;
void main() {
    glyph_index = ghee_a0.z;
    glyph_word = ghee_a0.w;
    glyph_line = ghee_a3.x;
    glyph_span = ghee_a3.yz;
    glyph_box = ghee_a1;
    vec2 ghee_corner = mix(ghee_a1.xy, ghee_a1.zw, ghee_a0.xy);
    vec2 ghee_p = glyph_place(ghee_corner);
    // The pixels moved by glyph_place per em along each em axis, exact for an
    // affine glyph_place.
    vec2 ghee_jx = (glyph_place(ghee_corner + vec2(0.0625, 0.0)) - ghee_p) * 16.0;
    vec2 ghee_jy = (glyph_place(ghee_corner + vec2(0.0, 0.0625)) - ghee_p) * 16.0;
    // ghee_det stays above 0, so a glyph scaled to nothing keeps every
    // position finite.
    float ghee_det = max(abs(ghee_jx.x * ghee_jy.y - ghee_jx.y * ghee_jy.x),
                         1.0 / 16384.0);
    // Half a pixel of antialiasing, and 1/8 px more, since VC4 lands a vertex
    // up to 1/16 px off the position set by this shader. A read further than
    // REACH texels past the bitmap could take a neighbouring cell's.
    vec2 ghee_grow = min(0.625 * vec2(length(ghee_jy), length(ghee_jx)) / ghee_det,
                         vec2(REACH / ghee_a2.z));
    vec2 ghee_e = ghee_corner + (ghee_a0.xy * 2.0 - 1.0) * ghee_grow;
    vec2 ghee_px = glyph_place(ghee_e);
    vec2 ghee_tc = ghee_a2.xy + (ghee_e - ghee_a1.xy) * ghee_a2.zw;
    // GL picks the mip level from the atlas texels spanned by a pixel along x
    // or along y of the pass, whichever is more. A bias in the read holds the
    // level at MAX_LEVEL at most.
    float ghee_rho = ghee_a2.z * max(length(vec2(ghee_jy.y, ghee_jx.y)),
                                      length(vec2(ghee_jy.x, ghee_jx.x))) / ghee_det;
    float ghee_lod = log2(max(ghee_rho, 1.0 / 65536.0));
    ghee_v0 = vec4(ghee_tc / ghee_NAME_emoji_size, min(ghee_lod, MAX_LEVEL) - ghee_lod, 0.0);
    v_texCoord = ghee_px / resolution;
    gl_Position = vec4((v_texCoord * 2.0 - 1.0) * vec2(1.0, ghee_vflip), 0.0, 1.0);
}"#;

pub(crate) const FRAGMENT: &str = r#"uniform sampler2D ghee_NAME_emoji;
varying vec4 ghee_v0;
void main() {
    // On VC4 a short shader whose texture read sits outside an if runs in two
    // threads, whose switch loses the condition flags. The read sits in an if
    // that the compiler cannot drop, so the shader stays on one thread.
    vec4 c = vec4(0.0);
    if (gl_FragCoord.x >= 0.0) c = texture2D(ghee_NAME_emoji, ghee_v0.xy, ghee_v0.z);
    gl_FragColor = c;
}"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::Atlas;

    fn fixture(file: &str) -> Vec<u8> {
        let path = format!("{}/tools/fixtures/fonts/{file}", env!("CARGO_MANIFEST_DIR"));
        std::fs::read(path).unwrap()
    }

    #[test]
    fn a_cell_pads_its_bitmap_on_the_mip_grid() {
        for file in [
            "NotoColorEmoji-Smiles.ttf",
            "NotoColorEmoji-Smiles-sbix.ttf",
        ] {
            let data = fixture(file);
            let face = Face::parse(&data, 0).unwrap();
            let gid = face.glyph_index('\u{1f600}').unwrap().0;
            let (f, s) = field(&face, gid).unwrap();
            // The largest strike: 136 x 128 pixels at 109 px/em, its bottom
            // 27 pixels under the baseline.
            assert_eq!(s, 109.0, "{file}");
            assert_eq!((f.width, f.height, f.pad), (176, 160, 16.0), "{file}");
            assert_eq!(f.ink, [0.0, -27.0 / 109.0, 136.0 / 109.0, 101.0 / 109.0]);
            // Every texel outside the bitmap is transparent black.
            let inside = |x: u32, y: u32| (16..152).contains(&x) && (16..144).contains(&y);
            for y in 0..f.height {
                for x in 0..f.width {
                    let at = ((y * f.width + x) * 4) as usize;
                    if !inside(x, y) {
                        assert_eq!(&f.texels[at..at + 4], &[0; 4], "{file} ({x}, {y})");
                    }
                    // Premultiplied: no channel above alpha.
                    let t = &f.texels[at..at + 4];
                    assert!(t[..3].iter().all(|c| *c <= t[3]), "{file} ({x}, {y}) {t:?}");
                }
            }
            assert!(f.texels.chunks_exact(4).any(|t| t[3] == 255));
            // Cells pack on the grid of MIP_PAD texels.
            let mut a = Atlas::default();
            for id in 0..9 {
                let c = a.insert(id, &f).unwrap();
                assert!(c.rect.iter().all(|v| v % MIP_PAD == 0), "{:?}", c.rect);
            }
        }
    }

    /// A w x h cell of texels that vary with `seed`, its bitmap MIP_PAD
    /// texels in from each edge, the rest transparent.
    fn cell(seed: u32, w: u32, h: u32) -> Field {
        let mut texels = vec![0u8; (w * h * 4) as usize];
        for y in MIP_PAD..h - MIP_PAD {
            for x in MIP_PAD..w - MIP_PAD {
                let at = ((y * w + x) * 4) as usize;
                let v = (x * 37 + y * 11 + seed * 101).wrapping_mul(2_654_435_761);
                let a = (v >> 24) as u8;
                for c in 0..3 {
                    texels[at + c] = ((v >> (8 * c)) as u8).min(a);
                }
                texels[at + 3] = a;
            }
        }
        Field {
            width: w,
            height: h,
            texels,
            ink: [0.0; 4],
            pad: f64::from(MIP_PAD),
        }
    }

    #[test]
    fn a_level_averages_the_four_texels_above_it_down_to_1x1() {
        let mut a = Atlas::default();
        a.insert(0, &cell(1, 64, 48)).unwrap();
        let mut m = Mips::new(a.width, a.height);
        m.refresh(&a.texels, a.width, a.height, [0, 0, a.width, a.height]);
        let sizes: Vec<(u32, u32)> = (1..=m.count())
            .map(|k| (m.level(k).0, m.level(k).1))
            .collect();
        assert_eq!(
            sizes,
            [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]
        );
        let (w1, _, l1) = m.level(1);
        for (x, y) in [(8, 8), (20, 13), (31, 23)] {
            let at = |x: u32, y: u32| ((y * a.width + x) * 4) as usize;
            for c in 0..4 {
                let sum: u32 = [(0, 0), (1, 0), (0, 1), (1, 1)]
                    .iter()
                    .map(|(dx, dy)| u32::from(a.texels[at(2 * x + dx, 2 * y + dy) + c]))
                    .sum();
                assert_eq!(
                    u32::from(l1[((y * w1 + x) * 4) as usize + c]),
                    (sum + 2) / 4
                );
            }
        }
    }

    #[test]
    fn a_new_cell_changes_its_own_rect_of_each_level_as_a_whole_build_would() {
        let mut a = Atlas::default();
        for (id, (w, h)) in [(48, 64), (80, 32), (32, 48)].into_iter().enumerate() {
            a.insert(id as u32, &cell(id as u32, w, h)).unwrap();
        }
        a.take_change();
        let mut m = Mips::new(a.width, a.height);
        assert!(a.top() < a.height);
        m.refresh(&a.texels, a.width, a.height, [0, 0, a.width, a.top()]);
        let size = (a.width, a.height);
        let rect = a.insert(9, &cell(9, 64, 48)).unwrap().rect;
        assert_eq!(a.take_change(), Some(false));
        assert_eq!((a.width, a.height), size);
        let changed = m.refresh(&a.texels, a.width, a.height, rect);
        // Down to MAX_LEVEL the rect of a level holds the new cell alone.
        for k in 1..=MAX_LEVEL as usize {
            assert_eq!(changed[k - 1], rect.map(|v| v >> k), "level {k}");
        }
        let mut whole = Mips::new(a.width, a.height);
        whole.refresh(&a.texels, a.width, a.height, [0, 0, a.width, a.height]);
        for k in 1..=whole.count() {
            assert!(m.level(k) == whole.level(k), "level {k}");
        }
        // The crop of a rect holds its rows.
        let [x, y, w, h] = rect;
        let cut = crop(&a.texels, a.width, rect);
        assert_eq!(cut.len(), (w * h * 4) as usize);
        let at = (((y + h - 1) * a.width + x) * 4) as usize;
        assert_eq!(
            &cut[cut.len() - (w * 4) as usize..],
            &a.texels[at..at + (w * 4) as usize]
        );
    }

    #[test]
    fn cells_side_by_side_change_one_rect_a_level() {
        let mut a = Atlas::default();
        a.insert(0, &cell(0, 48, 48)).unwrap();
        a.take_change();
        let mut m = Mips::new(a.width, a.height);
        m.refresh(&a.texels, a.width, a.height, [0, 0, a.width, a.top()]);
        // Two cells on the first shelf, after the first cell.
        let more = [1, 2].map(|id| a.insert(id, &cell(id, 32, 32)).unwrap().rect);
        assert_eq!(a.take_change(), Some(false));
        assert_eq!(more, [[48, 0, 32, 32], [80, 0, 32, 32]]);
        let rects = m.refresh_cells(&a.texels, a.width, a.height, &more);
        let levels: Vec<usize> = rects.iter().map(|(l, _)| *l).collect();
        assert_eq!(levels, (0..=m.count()).collect::<Vec<_>>());
        for (level, rect) in &rects {
            if *level <= MAX_LEVEL as usize {
                assert_eq!(*rect, [48, 0, 64, 32].map(|v| v >> level), "level {level}");
            }
        }
        let mut whole = Mips::new(a.width, a.height);
        whole.refresh(&a.texels, a.width, a.height, [0, 0, a.width, a.height]);
        for k in 1..=whole.count() {
            assert!(m.level(k) == whole.level(k), "level {k}");
        }
    }

    #[test]
    fn a_glyph_without_a_png_bitmap_has_no_cell() {
        let data = fixture("NotoColorEmoji-Smiles.ttf");
        let face = Face::parse(&data, 0).unwrap();
        let space = face.glyph_index(' ').unwrap();
        assert!(bitmap(&face, space).is_none());
        assert_eq!(
            field(&face, space.0).unwrap_err().to_string(),
            "has no PNG bitmap"
        );
    }
}
