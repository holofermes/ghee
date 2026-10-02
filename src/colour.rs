//! Colour glyphs: the bitmaps of colour emoji.
//!
//! A glyph with a PNG bitmap in its font's CBDT or sbix table, and no outline,
//! is a colour glyph. A draw of a text puts the text's colour glyphs into its
//! pass through a colour layer, after the draws of the pass. The layer places
//! each glyph with the draw's `glyph_place`, and samples its bitmap from the
//! text's colour atlas, an RGBA8 texture with mipmaps. The draw's own code
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
/// no other bitmap. The layer reads no level past MAX_LEVEL.
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

/// The vertex main of the colour layer of a draw of text `name`, after
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

/// The fragment shader of the colour layer of a draw of text `name`, after
/// its precision lines.
pub fn fragment(name: &str) -> Vec<String> {
    FRAGMENT
        .replace("NAME", name)
        .lines()
        .map(str::to_string)
        .collect()
}

pub(crate) const VERTEX_MAIN: &str = r#"uniform vec2 NAME_emoji_size;
void main() {
    glyph_index = glyph_a0.z;
    glyph_word = glyph_a0.w;
    glyph_line = glyph_a3.x;
    glyph_span = glyph_a3.yz;
    glyph_box = glyph_a1;
    vec2 glyph_corner = mix(glyph_a1.xy, glyph_a1.zw, glyph_a0.xy);
    vec2 glyph_p = glyph_place(glyph_corner);
    // The pixels moved by glyph_place per em along each em axis, exact for an
    // affine glyph_place.
    vec2 glyph_jx = (glyph_place(glyph_corner + vec2(0.0625, 0.0)) - glyph_p) * 16.0;
    vec2 glyph_jy = (glyph_place(glyph_corner + vec2(0.0, 0.0625)) - glyph_p) * 16.0;
    // glyph_det stays above 0, so a glyph scaled to nothing keeps every
    // position finite.
    float glyph_det = max(abs(glyph_jx.x * glyph_jy.y - glyph_jx.y * glyph_jy.x),
                          1.0 / 16384.0);
    // Half a pixel of antialiasing, and 1/8 px more, since VC4 lands a vertex
    // up to 1/16 px off the position set by this shader. A read further than
    // REACH texels past the bitmap could take a neighbouring cell's.
    vec2 glyph_grow = min(0.625 * vec2(length(glyph_jy), length(glyph_jx)) / glyph_det,
                          vec2(REACH / glyph_a2.z));
    vec2 glyph_e = glyph_corner + (glyph_a0.xy * 2.0 - 1.0) * glyph_grow;
    vec2 glyph_px = glyph_place(glyph_e);
    vec2 glyph_tc = glyph_a2.xy + (glyph_e - glyph_a1.xy) * glyph_a2.zw;
    // GL picks the mip level from the atlas texels spanned by a pixel along x
    // or along y of the pass, whichever is more. A bias in the read holds the
    // level at MAX_LEVEL at most.
    float glyph_rho = glyph_a2.z * max(length(vec2(glyph_jy.y, glyph_jx.y)),
                                       length(vec2(glyph_jy.x, glyph_jx.x))) / glyph_det;
    float glyph_lod = log2(max(glyph_rho, 1.0 / 65536.0));
    glyph_v0 = vec4(glyph_tc / NAME_emoji_size, min(glyph_lod, MAX_LEVEL) - glyph_lod, 0.0);
    v_texCoord = glyph_px / resolution;
    gl_Position = vec4((v_texCoord * 2.0 - 1.0) * vec2(1.0, vflip), 0.0, 1.0);
}"#;

pub(crate) const FRAGMENT: &str = r#"uniform sampler2D NAME_emoji;
varying vec4 glyph_v0;
void main() {
    // On VC4 a short shader whose texture read sits outside an if runs in two
    // threads, whose switch loses the condition flags. The read sits in an if
    // that the compiler cannot drop, so the shader stays on one thread.
    vec4 c = vec4(0.0);
    if (gl_FragCoord.x >= 0.0) c = texture2D(NAME_emoji, glyph_v0.xy, glyph_v0.z);
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
