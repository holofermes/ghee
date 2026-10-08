/* Hinted glyph bitmaps through FreeType, for src/hinted.rs. */
#ifndef GHEE_GLYPHS_H
#define GHEE_GLYPHS_H

typedef struct GheeFtFace GheeFtFace;

/* A glyph's bitmap: `rows` rows of `width` bytes, top row first, 0 to 255
 * each, valid until the next call on the face. `left` and `top` place its top
 * left corner from the pen, x right and y up, in pixels. */
typedef struct {
    int left, top, width, rows;
    const unsigned char *buffer;
} GheeFtBitmap;

/* The face in `data`, which must outlive it. NULL when FreeType refuses it. */
GheeFtFace *ghee_ft_open(const unsigned char *data, long len);
void ghee_ft_close(GheeFtFace *f);
/* Set the size to `ppem` 26.6 pixels per em, as Pillow asks for it. 0 when
 * FreeType takes it. */
int ghee_ft_size(GheeFtFace *f, long ppem);
/* The size's horizontal scale, font units to 26.6 pixels, in 16.16. */
long long ghee_ft_x_scale(GheeFtFace *f);
/* The unhinted advance of glyph `gid` at the size, 16.16 pixels, as
 * HarfBuzz's FreeType font functions read it. -1 on an error. */
long long ghee_ft_advance(GheeFtFace *f, unsigned gid);
/* Hint glyph `gid` at the size and render it: FreeType's normal target and
 * grey rasterizer, or with `mono` its mono target and 1-bit rasterizer, each
 * pixel then 0 or 255. 0 when FreeType renders it. */
int ghee_ft_render(GheeFtFace *f, unsigned gid, int mono, GheeFtBitmap *out);

#endif
