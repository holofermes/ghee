// Internal seams between engine.c and encoder.c. Not part of the public ABI
// (engine.h), and encoder.c is the only intended consumer.
#ifndef GHEE_ENGINE_INTERNAL_H
#define GHEE_ENGINE_INTERNAL_H

#include "engine.h"

void *ghee_i_egl_display(GheeScene *g);    // EGLDisplay
void  ghee_i_make_current(GheeScene *g);
void  ghee_i_error(GheeScene *g, const char *msg);

// Blit the final pass into an external FBO at (w,h). A whole multiple of the
// frame size repeats each pixel as a square block. Call it once ghee_mp_render
// or ghee_mp_present has left the frame in place.
void  ghee_i_blit_last(GheeScene *g, unsigned fbo, int w, int h, float flipY);
// While nonzero, ghee_mp_present keeps the final pass in its FBO (no direct
// scanout) so an encoder can sample it after the flip. delta = +1 / -1.
void  ghee_i_keep_last(GheeScene *g, int delta);

#endif
