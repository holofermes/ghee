//! Hinted text: each glyph hinted at the text's own pixel size and drawn there
//! as a bitmap, the way Pillow draws text through FreeType.
//!
//! FreeType 2.14.3, in vendor/freetype, hints and rasterizes each glyph: its
//! TrueType instructions, the PostScript hints of a CFF font, or FreeType's
//! autohinter for a font with neither. The `hinted` render takes FreeType's
//! normal target and its grey rasterizer, as Pillow draws on an "L" image,
//! and `mono` its mono target and 1-bit rasterizer, as Pillow draws on a
//! "1" image. ghee/_freetype/glyphs.c holds the calls into FreeType.
//!
//! A text layer of either render keeps its glyphs' bitmaps in an atlas of its
//! own, a [`Bitmaps`], one cell a glyph at the layer's size, and draws one quad
//! a glyph over its bitmap, one pixel a texel. Each glyph's pen lands on a
//! whole pixel, as Pillow's does: the pen runs along the line in 26.6 fixed
//! point, through the unhinted advances and the kerning as HarfBuzz scales
//! them, and the shader rounds it to the nearest pixel after the layer's
//! place and the glyph's motion.

use std::collections::HashMap;
use std::os::raw::{c_int, c_long, c_longlong, c_uchar, c_uint};
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};

use crate::atlas::{self, Atlas, Field};
use crate::text::{self, Font, Pen, Placement, Quads, QUAD_FLOATS};

#[repr(C)]
struct GheeFtFace {
    _private: [u8; 0],
}

#[repr(C)]
struct GheeFtBitmap {
    left: c_int,
    top: c_int,
    width: c_int,
    rows: c_int,
    buffer: *const c_uchar,
}

extern "C" {
    fn ghee_ft_open(data: *const c_uchar, len: c_long) -> *mut GheeFtFace;
    fn ghee_ft_close(f: *mut GheeFtFace);
    fn ghee_ft_size(f: *mut GheeFtFace, ppem: c_long) -> c_int;
    fn ghee_ft_x_scale(f: *mut GheeFtFace) -> c_longlong;
    fn ghee_ft_advance(f: *mut GheeFtFace, gid: c_uint) -> c_longlong;
    fn ghee_ft_render(
        f: *mut GheeFtFace,
        gid: c_uint,
        mono: c_int,
        out: *mut GheeFtBitmap,
    ) -> c_int;
}

/// A font open in FreeType at one size.
pub struct Face {
    raw: NonNull<GheeFtFace>,
    /// FreeType reads the font from these bytes while the face lives.
    _data: Arc<Vec<u8>>,
    upem: i64,
    /// The size set last, in 26.6 pixels per em, 0 before the first.
    ppem: i64,
}

// FreeType holds a face and its library in this one value alone, and
// glyphs.c gives each face a library of its own.
unsafe impl Send for Face {}

impl Drop for Face {
    fn drop(&mut self) {
        unsafe { ghee_ft_close(self.raw.as_ptr()) };
    }
}

/// A glyph's bitmap, rows top first, 0 to 255 a pixel, and its top left
/// corner from the pen in pixels, x right and y up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bitmap {
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub rows: u32,
    pub pixels: Vec<u8>,
}

impl Face {
    /// The font in `data`, None when FreeType refuses it.
    pub fn open(data: Arc<Vec<u8>>) -> Option<Face> {
        let len = c_long::try_from(data.len()).ok()?;
        let raw = NonNull::new(unsafe { ghee_ft_open(data.as_ptr(), len) })?;
        let upem = ttf_parser::Face::parse(&data, 0).map_or(0, |f| i64::from(f.units_per_em()));
        Some(Face {
            raw,
            _data: data,
            upem,
            ppem: 0,
        })
    }

    /// Set the size to `px` pixels per em, rounded to 1/64 as FreeType takes
    /// it. False when FreeType refuses the size.
    pub fn set_size(&mut self, px: f64) -> bool {
        let ppem = (px * 64.0).round() as i64;
        if ppem == self.ppem {
            return true;
        }
        if ppem <= 0 || unsafe { ghee_ft_size(self.raw.as_ptr(), ppem as c_long) } != 0 {
            self.ppem = 0;
            return false;
        }
        self.ppem = ppem;
        true
    }

    /// The unhinted advance of glyph `gid` at the size, in 26.6 pixels,
    /// rounded as HarfBuzz rounds FreeType's: (16.16 + 512) >> 10.
    pub fn advance(&self, gid: u16) -> Option<i64> {
        let v = unsafe { ghee_ft_advance(self.raw.as_ptr(), c_uint::from(gid)) };
        (v >= 0).then_some((v + 512) >> 10)
    }

    /// `units` of the font's kerning at the size, in 26.6 pixels, scaled as
    /// HarfBuzz scales a font's values: its scale from FreeType's x scale,
    /// then a 16.16 multiplier per font unit.
    pub fn kerning(&self, units: i32) -> i64 {
        if self.upem <= 0 || units == 0 {
            return 0;
        }
        let x_scale = unsafe { ghee_ft_x_scale(self.raw.as_ptr()) };
        let hb_scale = (x_scale * self.upem + (1 << 15)) >> 16;
        let mult = (hb_scale << 16) / self.upem;
        (i64::from(units) * mult + 32768) >> 16
    }

    /// Glyph `gid` hinted at the size and rendered, by the 1-bit rasterizer
    /// when `mono`, each pixel then 0 or 255. None when FreeType fails it.
    pub fn bitmap(&self, gid: u16, mono: bool) -> Option<Bitmap> {
        let mut b = GheeFtBitmap {
            left: 0,
            top: 0,
            width: 0,
            rows: 0,
            buffer: std::ptr::null(),
        };
        let rc = unsafe {
            ghee_ft_render(
                self.raw.as_ptr(),
                c_uint::from(gid),
                c_int::from(mono),
                &mut b,
            )
        };
        if rc != 0 || b.width < 0 || b.rows < 0 {
            return None;
        }
        let (w, h) = (b.width as usize, b.rows as usize);
        let pixels = match w * h {
            0 => Vec::new(),
            n if b.buffer.is_null() => vec![0; n],
            n => unsafe { std::slice::from_raw_parts(b.buffer, n) }.to_vec(),
        };
        Some(Bitmap {
            left: b.left,
            top: b.top,
            width: w as u32,
            rows: h as u32,
            pixels,
        })
    }
}

/// The pen of each of `pens` at `px` pixels per em, in 26.6 pixels from the
/// block's left, where its glyph starts: its line's start, then each advance
/// and kerning before it along the line. `face` gives the face of a font of
/// the text by its index, set to the size. A glyph whose font has no face
/// moves the pen by its advance in em.
pub fn pens<'f>(pens: &[Pen], px: f64, face: impl Fn(usize) -> Option<&'f Face>) -> Vec<i64> {
    let fixed = |em: f64| (em * px * 64.0).round() as i64;
    let mut out = Vec::with_capacity(pens.len());
    let mut at = 0;
    let mut line = None;
    for p in pens {
        if line != Some(p.line) {
            line = Some(p.line);
            at = fixed(p.start);
        }
        let face = p.glyph.and_then(|(f, gid)| Some((face(f)?, gid)));
        let advance = match face {
            Some((face, gid)) => {
                at += face.kerning(p.kern);
                face.advance(gid).unwrap_or_else(|| fixed(p.advance))
            }
            None => fixed(p.advance),
        };
        out.push(at);
        at += advance;
    }
    out
}

/// The texels between two cells of a Bitmaps, so a read at a quad's edge
/// meets no other glyph.
const PAD: u32 = 1;

/// The glyph bitmaps of one text layer at one size, in an atlas of RGBA8
/// cells, each texel the bitmap's value in all four channels, its top row
/// first.
pub struct Bitmaps {
    pub atlas: Atlas,
    /// The faces of the layer's fonts, by index, None for a font that
    /// FreeType refuses or that failed to load.
    faces: Vec<Option<Face>>,
    /// True for each face that took the size of the cells.
    sized: Vec<bool>,
    /// The fonts that `faces` open.
    fonts: Vec<Font>,
    /// The size of every cell, in 26.6 pixels per em, and the rasterizer.
    ppem: i64,
    mono: bool,
    /// Each glyph's bitmap corner from its pen, and its size, by atlas id.
    metrics: HashMap<u32, [i32; 4]>,
    /// Why each glyph that failed took no cell, by atlas id.
    pub failed: HashMap<u32, String>,
}

impl Bitmaps {
    /// An empty atlas for a layer whose fonts are `fonts`.
    pub fn new(fonts: &[Font]) -> Bitmaps {
        Bitmaps {
            atlas: Atlas::default(),
            faces: fonts
                .iter()
                .map(|f| f.as_ref().ok().and_then(|d| Face::open(d.clone())))
                .collect(),
            fonts: fonts.to_vec(),
            sized: vec![false; fonts.len()],
            ppem: 0,
            mono: false,
            metrics: HashMap::new(),
            failed: HashMap::new(),
        }
    }

    /// True when the atlas was made for `fonts`.
    pub fn holds(&self, fonts: &[Font]) -> bool {
        self.fonts.len() == fonts.len()
            && self.fonts.iter().zip(fonts).all(|(a, b)| match (a, b) {
                (Ok(a), Ok(b)) => Arc::ptr_eq(a, b),
                (Err(a), Err(b)) => a == b,
                _ => false,
            })
    }

    /// The quads of `glyphs` at `px` pixels per em, by the 1-bit rasterizer
    /// when `mono`, with `pens` the pens of their layout. A glyph new to the
    /// atlas, or every glyph at a new size, takes a cell first. Returns the
    /// quads, and each glyph's pen in pixels from the block's left, None for a
    /// glyph that FreeType failed, which `failed` names.
    ///
    /// A quad's vertex holds, beside the corner, the index and the word of
    /// ghee_a0 and the line and the span of ghee_a3:
    ///
    /// - ghee_a1: the pen in pixels from the block's left, and the bitmap's
    ///   left and top corner from the pen, x right and y up, then 0
    /// - ghee_a2: the texel of the bitmap's top left, its width and its rows
    pub fn quads(
        &mut self,
        glyphs: &[Placement],
        pens: &[Pen],
        px: f64,
        mono: bool,
    ) -> (Quads, Vec<Option<f64>>) {
        let ppem = (px * 64.0).round() as i64;
        if ppem != self.ppem || mono != self.mono {
            self.atlas = Atlas::default();
            self.metrics.clear();
            self.failed.clear();
            self.ppem = ppem;
            self.mono = mono;
        }
        self.sized = self
            .faces
            .iter_mut()
            .map(|f| f.as_mut().is_some_and(|f| f.set_size(px)))
            .collect();
        let at = self::pens(pens, px, |f| self.face(f));
        let mut floats = Vec::new();
        let mut pen_px = Vec::with_capacity(glyphs.len());
        for p in glyphs {
            let id = text::atlas_id(p.font, p.gid);
            if !self.atlas.cells.contains_key(&id) && !self.failed.contains_key(&id) {
                if let Err(e) = self.insert(id, p.font, p.gid) {
                    self.failed.insert(id, format!("{e:#}"));
                }
            }
            let (Some(cell), Some(m)) = (self.atlas.cells.get(&id), self.metrics.get(&id)) else {
                pen_px.push(None);
                continue;
            };
            let pen = at[p.pen] as f64 / 64.0;
            pen_px.push(Some(pen));
            let [left, top, w, h] = *m;
            if w == 0 || h == 0 {
                continue;
            }
            let a1 = [pen, f64::from(left), f64::from(top), 0.0];
            let a2 = [
                f64::from(cell.rect[0] + PAD),
                f64::from(cell.rect[1] + PAD),
                f64::from(w),
                f64::from(h),
            ];
            let a3 = [p.line as f64, p.span[0], p.span[1], 0.0];
            for corner in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] {
                let a0 = [corner[0], corner[1], p.index as f64, p.word as f64];
                floats.extend([a0, a1, a2, a3].iter().flatten().map(|v| *v as f32));
            }
        }
        let count = floats.len() / (4 * QUAD_FLOATS);
        (
            Quads {
                floats,
                per_vertex: QUAD_FLOATS,
                count,
            },
            pen_px,
        )
    }

    /// The face of font `font` at the size of the cells, None when it has
    /// none.
    fn face(&self, font: usize) -> Option<&Face> {
        match self.sized.get(font) {
            Some(true) => self.faces[font].as_ref(),
            _ => None,
        }
    }

    /// Hint and render glyph `gid` of font `font` and give it a cell as `id`.
    fn insert(&mut self, id: u32, font: usize, gid: u16) -> Result<()> {
        let Some(face) = self.face(font) else {
            bail!("FreeType cannot read its font at this size");
        };
        let b = face
            .bitmap(gid, self.mono)
            .ok_or_else(|| anyhow!("FreeType fails to render it"))?;
        let (w, h) = (b.width + 2 * PAD, b.rows + 2 * PAD);
        if w > atlas::MAX_SIZE || h > atlas::MAX_SIZE {
            bail!("its {w}x{h} bitmap is larger than the atlas");
        }
        let mut texels = vec![0u8; w as usize * h as usize * 4];
        for row in 0..b.rows as usize {
            for col in 0..b.width as usize {
                let v = b.pixels[row * b.width as usize + col];
                let at = ((row + PAD as usize) * w as usize + col + PAD as usize) * 4;
                texels[at..at + 4].fill(v);
            }
        }
        let field = Field {
            width: w,
            height: h,
            texels,
            ink: [0.0; 4],
            pad: f64::from(PAD),
        };
        self.atlas.insert(id, &field)?;
        self.metrics
            .insert(id, [b.left, b.top, b.width as i32, b.rows as i32]);
        Ok(())
    }
}
