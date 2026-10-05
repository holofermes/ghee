// The C engine's interface: a GLES2 renderer of multipass scenes on a GPU
// opened through DRM and GBM, with no window system.
//
// A GheeScene is one renderer: a GL context, the scene that it draws, and the
// textures that the scene reads. ghee_create opens one that renders headless,
// ghee_display_create one that scans its frames out to a KMS display. The
// ghee_mp_* calls build and draw the scene, mp for multipass: add the passes
// and their batches, which compile beside the running scene, commit them,
// hand over the uniforms, arrays and assets, then render a frame at a time.
// The ghee_encoder_* calls feed the frames to the hardware H264 codec.
//
// Every call makes its renderer's GL context current first, so several
// renderers live in one process, each used from the thread that created it.

#ifndef GHEE_ENGINE_H
#define GHEE_ENGINE_H
#include <stdint.h>
#include <stdbool.h>
#include <GLES2/gl2.h>

// A VEC2 to VEC4 param sets a vec2 to vec4 uniform from the first 2 to 4
// values of its vec_param.
typedef enum {
    PARAM_TYPE_FLOAT,
    PARAM_TYPE_INT,
    PARAM_TYPE_VEC2,
    PARAM_TYPE_VEC3,
    PARAM_TYPE_VEC4
} param_type_t;

typedef struct { float value; } float_param_t;
typedef struct { int value; } int_param_t;
typedef struct { float value[4]; } vec_param_t;

typedef struct {
    const char *name;
    param_type_t type;
    union {
        float_param_t *float_param;
        int_param_t *int_param;
        vec_param_t *vec_param;
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
// with no quad: each frame it clears to transparent black and draws its batches
// alone. `alpha` gives the buffer an alpha channel, else the buffer stores RGB
// and every read of its alpha gives 1.
int  ghee_mp_add_pass(GheeScene *g, const char *name, const char *glsl_source, int alpha);
// Set a pending pass's buffer size (default = renderer W,H). Call after
// ghee_mp_add_pass, before ghee_mp_commit. The last pass must keep the renderer
// size (it feeds readback/scanout), consumers sample any pass through normalized
// UVs.
bool ghee_mp_set_pass_size(GheeScene *g, int pass, int w, int h);
// Give a pending pass a mip chain: after the pass renders each frame, the
// engine builds the levels below its buffer, and every read of the buffer
// samples with GL_LINEAR_MIPMAP_LINEAR. A GPU without ghee_mipmaps_npot
// takes a power-of-two size alone. false with ghee_last_error set on refusal.
bool ghee_mp_set_pass_mipmaps(GheeScene *g, int pass);
int  ghee_mp_commit(GheeScene *g);   // -> pass count, -1 = nothing pending
void ghee_mp_abort(GheeScene *g);
bool ghee_mp_set_channel(GheeScene *g, int pass, int channel, int src_pass);
bool ghee_mp_set_uniform(GheeScene *g, int pass, const parameter_t *param);
// Set the uniform float array `name` of a committed pass. ghee binds no
// uniform array of its own: VC4 caps the size of a uniform array that a
// shader indexes with a variable, so a scene's arrays are textures, filled by
// ghee_mp_set_array.
void ghee_mp_set_float_array(GheeScene *g, int pass, const char *name, const float *data, int count);
// An array: a luminance texture of count x rows values, sampled as
// <name>_tex. With rows = 1 a shader reads it at a computed index, and with
// more rows it samples it as a 2D input at a normalized uv. The scene holds one
// texture per name: declare it on every pass that reads it, and upload it at
// most once a frame. Returns the array's index, or -1 with ghee_last_error set
// on refusal.
int  ghee_mp_declare_array(GheeScene *g, int pass, const char *name);
void ghee_mp_set_array(GheeScene *g, const char *name, const float *data, int count, int rows);
// The same upload from bytes, where the value is data[i] / 255. An 8-bit
// LUMINANCE texture takes the bytes as they lie, so a large mask costs no
// conversion at all.
void ghee_mp_set_array_u8(GheeScene *g, const char *name, const uint8_t *data, int count, int rows);
// An image asset: width x height pixels the host hands over once, bound as a
// sampler uniform <name>_tex on every pass whose program reads it, with a vec2
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
// Whether a program of the committed scene, a pass's or a batch's, reads the
// uniform `name`. The compiler drops a uniform that no code it keeps reads.
bool ghee_mp_reads(GheeScene *g, const char *name);
// A batch: quads drawn by a pass into its own buffer after its full-screen
// quad, with a vertex and a fragment shader of their own, in the order of the
// ghee_mp_add_batch calls. A vertex holds 1 to 4 vec4 attributes, ghee_a0 to
// ghee_a3. A quad's 4 vertices are its corners in triangle-strip order, drawn
// as the triangles (0, 1, 2) and (2, 1, 3). A batch reads its pass's uniforms,
// channels, arrays and assets under the same names, and `ghee_vflip` as the shared
// quad vertex shader does. Blending composes premultiplied colour: normal is
// src + dst * (1 - src.a), add is src + dst, multiply is
// src * dst + dst * (1 - src.a), and screen is src + dst * (1 - src). The
// alpha channel blends by the same factors. multiply and screen hold their
// formulas over an opaque buffer, and over a transparent one multiply draws
// nothing.
enum { GHEE_BLEND_NORMAL, GHEE_BLEND_ADD, GHEE_BLEND_MULTIPLY, GHEE_BLEND_SCREEN };
// Staged like ghee_mp_add_pass: `pass` is a pending pass, the commit keeps the
// returned index, and an abort drops the batch. `name` stands for the batch in
// the errors of later calls. An error of this call leaves the name out, as a
// compile error does, and its caller names the batch. -1 = refused.
int  ghee_mp_add_batch(GheeScene *g, int pass, const char *name, const char *vertex_source,
                       const char *fragment_source, int blend);
// The quads of a committed batch: quads * 4 vertices of nattr vec4s each,
// which replace the last set. false with ghee_last_error set on refusal.
bool ghee_mp_set_batch_quads(GheeScene *g, int batch, const float *data, int quads, int nattr);
// Bind the set asset `asset` to committed batch `batch` under `name`: the
// batch's program samples it as <name>_tex and reads its size as <name>_size,
// in place of the asset bound there before, and pays a texture unit for it as
// for an asset sampled under its own name. A drop of the asset unbinds it.
// false with ghee_last_error set for a batch or an asset that is not set.
bool ghee_mp_bind_asset(GheeScene *g, int batch, const char *name, const char *asset);
// Render the passes and swap. buffer_rgb NULL = no readback at all, pw/ph > 0 =
// GPU-scale the final pass to pw x ph and read back only that, otherwise the
// full frame lands in buffer_rgb. A whole multiple of the frame size repeats
// each pixel as a square block. A 4-component readback reads alpha 255 unless
// ghee_keep_alpha is on.
void ghee_mp_render(GheeScene *g, uint8_t *buffer_rgb, int pw, int ph);
// Render the passes and scan the final buffer out to the display (display mode
// only). buffer_rgb non-NULL also reads the frame back (for a console mirror).
void ghee_mp_present(GheeScene *g, uint8_t *buffer_rgb, int pw, int ph);
// The scanout buffers after a present of `bo`. `prev` holds the buffer of the
// most recent flip, pending or on the glass, and `shown` the one before it,
// on the glass until that flip completes. An accepted flip displaces `shown`.
// A flip that never happened, still waiting on the last one or refused,
// leaves both on the glass. Returns the buffer free to reuse.
struct gbm_bo;
struct gbm_bo *ghee_flip_rotate(struct gbm_bo **prev, struct gbm_bo **shown,
                                struct gbm_bo *bo, bool flipped);
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
bool ghee_mipmaps_npot(GheeScene *g);        // GPU mipmaps a texture of any size

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
