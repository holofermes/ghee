#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_ADVANCES_H
#include <stdlib.h>
#include <string.h>

#include "glyphs.h"

/* A face of its own library: FreeType shares nothing between libraries, so
 * two faces never contend. `mono` holds the last 1-bit bitmap as bytes. */
struct GheeFtFace {
    FT_Library lib;
    FT_Face face;
    unsigned char *mono;
    size_t mono_len;
};

GheeFtFace *ghee_ft_open(const unsigned char *data, long len) {
    GheeFtFace *f = calloc(1, sizeof *f);
    if (!f) return NULL;
    if (FT_Init_FreeType(&f->lib)) {
        free(f);
        return NULL;
    }
    if (FT_New_Memory_Face(f->lib, data, len, 0, &f->face)) {
        FT_Done_FreeType(f->lib);
        free(f);
        return NULL;
    }
    return f;
}

void ghee_ft_close(GheeFtFace *f) {
    if (!f) return;
    FT_Done_Face(f->face);
    FT_Done_FreeType(f->lib);
    free(f->mono);
    free(f);
}

int ghee_ft_size(GheeFtFace *f, long ppem) {
    FT_Size_RequestRec req = {FT_SIZE_REQUEST_TYPE_NOMINAL, ppem, ppem, 0, 0};
    return FT_Request_Size(f->face, &req) ? -1 : 0;
}

long long ghee_ft_x_scale(GheeFtFace *f) {
    return f->face->size->metrics.x_scale;
}

long long ghee_ft_advance(GheeFtFace *f, unsigned gid) {
    FT_Fixed v = 0;
    if (FT_Get_Advance(f->face, gid, FT_LOAD_DEFAULT | FT_LOAD_NO_HINTING, &v)) return -1;
    return v < 0 ? -v : v;
}

int ghee_ft_render(GheeFtFace *f, unsigned gid, int mono, GheeFtBitmap *out) {
    FT_Int32 flags = FT_LOAD_DEFAULT | FT_LOAD_RENDER | (mono ? FT_LOAD_TARGET_MONO : 0);
    if (FT_Load_Glyph(f->face, gid, flags)) return -1;
    FT_GlyphSlot s = f->face->glyph;
    FT_Bitmap *b = &s->bitmap;
    out->left = s->bitmap_left;
    out->top = s->bitmap_top;
    out->width = (int)b->width;
    out->rows = (int)b->rows;
    out->buffer = b->buffer;
    if (b->rows == 0 || b->width == 0) return 0;
    if (b->pixel_mode == FT_PIXEL_MODE_GRAY && b->pitch == (int)b->width) return 0;
    size_t n = (size_t)b->width * b->rows;
    if (n > f->mono_len) {
        unsigned char *m = realloc(f->mono, n);
        if (!m) return -1;
        f->mono = m;
        f->mono_len = n;
    }
    for (unsigned y = 0; y < b->rows; y++) {
        const unsigned char *row = b->buffer + (long)y * b->pitch;
        for (unsigned x = 0; x < b->width; x++) {
            unsigned char v;
            if (b->pixel_mode == FT_PIXEL_MODE_MONO)
                v = ((row[x / 8] >> (7 - x % 8)) & 1) ? 255 : 0;
            else if (b->pixel_mode == FT_PIXEL_MODE_GRAY)
                v = row[x];
            else
                return -1;
            f->mono[(size_t)y * b->width + x] = v;
        }
    }
    out->buffer = f->mono;
    return 0;
}
