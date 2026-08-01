// Internal seams between engine.c and encoder.c. Not part of the public ABI
// (engine.h); encoder.c is the only intended consumer.
#ifndef GHEE_ENGINE_INTERNAL_H
#define GHEE_ENGINE_INTERNAL_H

#include "engine.h"

void *ghee_i_egl_display(GheeScene *g);    // EGLDisplay
void  ghee_i_make_current(GheeScene *g);
void  ghee_i_error(GheeScene *g, const char *msg);

// Render the pass chain into the FBOs, no output stage, no ping-pong swap.
void  ghee_i_render_frame(GheeScene *g);
// Swap the ping-pong buffers (call after all output blits for the frame).
void  ghee_i_swap(GheeScene *g);
// Blit the final pass into an external FBO at (w,h). post_swap picks which
// ping-pong buffer holds the frame: 0 = just rendered (before ghee_i_swap),
// 1 = after a completed ghee_mp_render/ghee_mp_present call.
void  ghee_i_blit_last(GheeScene *g, unsigned fbo, int w, int h, float flipY,
                       int post_swap);
// While nonzero, ghee_mp_present keeps the final pass in its FBO (no direct
// scanout) so an encoder can sample it after the flip. delta = +1 / -1.
void  ghee_i_keep_last(GheeScene *g, int delta);

#endif
