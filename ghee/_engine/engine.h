#ifndef GHEE_ENGINE_H
#define GHEE_ENGINE_H
#include <stdint.h>
#include <stdbool.h>
#include <GLES2/gl2.h>

// set_param splits a vector into one float uniform per component, so every
// type below is a scalar.
typedef enum {
    PARAM_TYPE_FLOAT,
    PARAM_TYPE_INT
} param_type_t;

typedef struct { float value; } float_param_t;
typedef struct { int value; } int_param_t;

typedef struct {
    char name[256];
    param_type_t type;
    union {
        float_param_t *float_param;
        int_param_t *int_param;
    } value;
} parameter_t;

// Opaque renderer instance. All state lives here, so multiple
// independent renderers can coexist in one process.
typedef struct GheeScene GheeScene;

// Create a headless GLES2 renderer on the given DRI device (NULL -> "/dev/dri/card0").
// Returns NULL on failure. A non-NULL err receives a static error string.
GheeScene *ghee_create(int width, int height, int components, const char *dri_device, char **err);

// Like ghee_create, but renders straight to a connected KMS display (zero-copy
// scanout instead of readback). width/height should match the connector mode.
// Needs DRM master. Present frames with ghee_mp_present.
GheeScene *ghee_display_create(int width, int height, const char *dri_device, int rgb565, char **err);

void ghee_set_verbosity(int level);

// The GL this renderer runs on: GL_RENDERER and GL_VERSION in one string.
const char *ghee_renderer(GheeScene *g);

// Last error message for this instance ("" if none). Cleared on read.
const char *ghee_last_error(GheeScene *g);

void ghee_destroy(GheeScene *g);

// Building a scene is staged: add_pass (and set_pass_size) compile and allocate
// the PENDING passes, commit makes them the scene, abort drops them. The running
// scene keeps rendering until the commit, so a compile error costs nothing.
// `name` stands for the pass in its errors. A NULL glsl_source makes a pass
// with no quad: each frame it clears to transparent black and draws its layers
// alone. `alpha` gives the buffer an alpha channel, else the buffer stores RGB
// and every read of its alpha gives 1.
int  ghee_mp_add_pass(GheeScene *g, const char *name, const char *glsl_source, int alpha);
// Set a pending pass's buffer size (default = renderer W,H). Call after
// ghee_mp_add_pass, before ghee_mp_commit. The last pass must keep the renderer
// size (it feeds readback/scanout), consumers sample any pass through normalized
// UVs.
bool ghee_mp_set_pass_size(GheeScene *g, int pass, int w, int h);
int  ghee_mp_commit(GheeScene *g);   // -> pass count, -1 = nothing pending
void ghee_mp_abort(GheeScene *g);
bool ghee_mp_set_channel(GheeScene *g, int pass, int channel, int src_pass);
bool ghee_mp_set_uniform(GheeScene *g, int pass, const parameter_t *param);
// Unused: superseded by ghee_mp_set_array (uniform arrays hit the VC4 size cap).
void ghee_mp_set_float_array(GheeScene *g, int pass, const char *name, const float *data, int count);
// Data bound as a count x rows luminance texture sampler (<name>_tex): rows=1 is
// the 1xN array case (computed-coord fetch, no VC4 uniform-array cap), rows>1 is
// a 2D input sampled with normalized uv. One texture per NAME: declare it
// on every pass that reads it, upload it once a frame.
// -> array index, -1 = refused.
int  ghee_mp_declare_array(GheeScene *g, int pass, const char *name);
void ghee_mp_set_array(GheeScene *g, const char *name, const float *data, int count, int rows);
// The same upload from bytes, where the value is data[i] / 255. An 8-bit
// LUMINANCE texture takes the bytes as they lie, so a large mask costs no
// conversion at all.
void ghee_mp_set_array_u8(GheeScene *g, const char *name, const uint8_t *data, int count, int rows);
// An image asset: width x height pixels the host hands over once, bound as a
// sampler uniform <name> on every pass whose program reads it, with a vec2
// <name>_size of (width, height). components 1, 3 or 4 = LUMINANCE, RGB,
// RGBA. is_f32 uploads floats and needs a GL with float textures. filter
// GHEE_FILTER_LINEAR samples with GL_LINEAR, f32 pixels fall back to nearest
// without ghee_data_texture_linear. GHEE_FILTER_MIPMAP samples 8-bit pixels
// with GL_LINEAR_MIPMAP_LINEAR and needs a power-of-two size: `data` fills
// mip level 0, and ghee_mp_set_asset_rect fills every level below it, each
// half the size of the one above, down to 1x1. An asset outlives the scene:
// the same name again replaces its pixels, ghee_mp_drop_asset or
// ghee_destroy frees it. Returns false with ghee_last_error set when the
// engine refuses the asset.
#define MP_MAX_ASSETS 16
enum { GHEE_FILTER_NEAREST, GHEE_FILTER_LINEAR, GHEE_FILTER_MIPMAP };
bool ghee_mp_set_asset(GheeScene *g, const char *name, const void *data,
                       int width, int height, int components, int is_f32, int filter);
// Replace rows y to y + rows of a set asset with `data`, rows of its full
// width in the layout of its last ghee_mp_set_asset. Returns false with
// ghee_last_error set for an asset that is not set or rows past its height.
bool ghee_mp_set_asset_rows(GheeScene *g, const char *name, const void *data, int y, int rows);
// Replace the w x h rect at (x, y) of mip level `level` of a set asset with
// `data`, rows of w pixels in the layout of its last ghee_mp_set_asset. An
// asset set without GHEE_FILTER_MIPMAP has level 0 alone. Returns false with
// ghee_last_error set for an asset that is not set, a level that it lacks,
// or a rect past the level's edges.
bool ghee_mp_set_asset_rect(GheeScene *g, const char *name, int level,
                            int x, int y, int w, int h, const void *data);
void ghee_mp_drop_asset(GheeScene *g, const char *name);
// A layer: quads drawn by a pass into its own buffer after its full-screen
// quad, with a vertex and a fragment shader of their own, in the order of the
// ghee_mp_add_layer calls. A vertex holds 1 to 4 vec4 attributes, glyph_a0 to
// glyph_a3. A quad's 4 vertices are its corners in triangle-strip order, drawn
// as the triangles (0, 1, 2) and (2, 1, 3). A layer reads its pass's uniforms,
// channels, arrays and assets under the same names, and `vflip` as the shared
// quad vertex shader does. Blending composes premultiplied colour: over is
// src + dst * (1 - src.a), add is src + dst, replace is src. The alpha channel
// blends the same way, and keep_alpha leaves the buffer's alpha as it was.
enum { GHEE_BLEND_OVER, GHEE_BLEND_ADD, GHEE_BLEND_REPLACE };
// Staged like ghee_mp_add_pass: `pass` is a pending pass, the commit keeps the
// returned index, and an abort drops the layer. `name` stands for the layer in
// its errors. -1 = refused.
int  ghee_mp_add_layer(GheeScene *g, int pass, const char *name, const char *vertex_source,
                       const char *fragment_source, int blend, int keep_alpha);
// The quads of a committed layer: quads * 4 vertices of nattr vec4s each,
// which replace the last set. false with ghee_last_error set on refusal.
bool ghee_mp_set_layer_quads(GheeScene *g, int layer, const float *data, int quads, int nattr);
// Render the passes and swap. buffer_rgb NULL = no readback at all, pw/ph > 0 =
// GPU-scale the final pass to pw x ph and read back only that, otherwise the
// full frame lands in buffer_rgb. A whole multiple of the frame size repeats
// each pixel as a square block. A 4-component readback reads alpha 255 unless
// ghee_keep_alpha is on.
void ghee_mp_render(GheeScene *g, uint8_t *buffer_rgb, int pw, int ph);
// Render the passes and scan the final buffer out to the display (display mode
// only). buffer_rgb non-NULL also reads the frame back (for a console mirror).
void ghee_mp_present(GheeScene *g, uint8_t *buffer_rgb, int pw, int ph);
void ghee_mp_reset(GheeScene *g);
// Keep the final pass's alpha in 4-component readbacks, the full frame and the
// downscaled one. Off by default. The scanout and the encoder input take alpha
// 1 either way.
void ghee_keep_alpha(GheeScene *g, int keep);
// The texture units one program of a pass binds at most: its pass's channels and
// arrays, then the assets it samples. min(16, GL_MAX_TEXTURE_IMAGE_UNITS): VC4
// has 16, so a pass that fits one Pi fits them all.
int  ghee_max_texture_units(GheeScene *g);
int  ghee_gl_texture_units(GheeScene *g);    // GL_MAX_TEXTURE_IMAGE_UNITS itself
bool ghee_data_texture_linear(GheeScene *g); // GPU linear-filters data textures

void ghee_set_profiling(GheeScene *g, bool on);
void ghee_set_time(GheeScene *g, float t);
void ghee_clear_time(GheeScene *g);   // back to the monotonic clock
void ghee_set_dt(GheeScene *g, float dt);  // the DT uniform on every pass, 1/60 until set
void ghee_get_timings(GheeScene *g, double *draw_ms, double *read_ms);
int  ghee_display_vrefresh(GheeScene *g);  // mode refresh Hz, 0 = not display mode
// Size of the mode that the connected display drives, without creating a scene
// or touching GL. Needs read access to the card node only, not DRM master.
// 0 = width/height set, -1 = no device or nothing connected.
int  ghee_display_probe(const char *dri_device, int *width, int *height);

// Whether GL on this card runs on hardware. A display-only card (a panel
// under a kernel display driver) still gets hardware GL when Mesa pairs it
// with the render node, so 0 means no usable render node exists. Quiet, and
// it touches no GL state of a live renderer.
int ghee_display_renders(const char *dri_device);

// Blank the display, or light it again (display mode only). Blanking stops
// the CRTC driving the connector, which runs the kernel driver's own disable
// path: a panel turns its display off and drops its backlight. A blanked
// display draws nothing, so ghee_mp_present returns at once and the caller
// paces its own loop. 0, or -1 with the reason in ghee_last_error.
int ghee_display_blank(GheeScene *g, int blank);
int ghee_display_blanked(GheeScene *g);

// Zero-copy hardware H264 (encoder.c): the final pass is blitted into a linear
// dmabuf and fed to the V4L2 M2M encoder (bcm2835-codec), with no CPU pixel copy.
// create fails cleanly (NULL + err) on boards without /dev/video11.
typedef struct GheeEncoder GheeEncoder;
GheeEncoder *ghee_encoder_create(GheeScene *g, int w, int h, int fps, int bitrate, char **err);
// Encode the frame left by the last ghee_mp_render/ghee_mp_present (display
// mirror: one render feeds screen and stream). 0 = *nal/*len/*is_key set
// (valid until the next call), 1 = encoder produced nothing yet, <0 = error.
int  ghee_encoder_encode_last(GheeScene *g, GheeEncoder *e, const uint8_t **nal, int *len, int *is_key);
void ghee_encoder_force_keyframe(GheeEncoder *e);   // instant join for a new viewer
// End of stream: sends the encoder STOP once, then hands back the trailing
// access units one call at a time. 0 = *nal/*len/*is_key set, 1 = drained,
// <0 = error. A capture that stops without this ends mid-GOP.
int  ghee_encoder_flush(GheeScene *g, GheeEncoder *e, const uint8_t **nal, int *len, int *is_key);
void ghee_encoder_destroy(GheeScene *g, GheeEncoder *e);

void ghee_pack_frame(int width, int height, int components, int bw,
                    const uint8_t *rgb, uint8_t *packed);

#endif // GHEE_ENGINE_H
