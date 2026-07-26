#ifndef GHEE_ENGINE_H
#define GHEE_ENGINE_H
#include <stdint.h>
#include <stdbool.h>
#include <GLES2/gl2.h>

typedef enum {
    PARAM_TYPE_FLOAT,
    PARAM_TYPE_INT,
    PARAM_TYPE_VEC2,
    PARAM_TYPE_VEC3,
    PARAM_TYPE_VEC4,
    PARAM_TYPE_TEXTURE
} param_type_t;

typedef struct { float value; } float_param_t;
typedef struct { int value; } int_param_t;
typedef struct { float values[2]; } vec2_param_t;
typedef struct { float values[3]; } vec3_param_t;
typedef struct { float values[4]; } vec4_param_t;
typedef struct { char value[256]; } texture_param_t;

typedef struct {
    char name[256];
    param_type_t type;
    union {
        float_param_t *float_param;
        int_param_t *int_param;
        vec2_param_t *vec2_param;
        vec3_param_t *vec3_param;
        vec4_param_t *vec4_param;
        texture_param_t *texture_param;
    } value;
} parameter_t;

// Opaque renderer instance. All state lives here, so multiple
// independent renderers can coexist in one process.
typedef struct GheeScene GheeScene;

// Create a headless GLES2 renderer on the given DRI device (NULL -> "/dev/dri/card0").
// Returns NULL on failure; if err != NULL it receives a static error string.
GheeScene *ghee_create(int width, int height, int components, const char *dri_device, char **err);

// Like ghee_create, but renders straight to a connected KMS display (zero-copy
// scanout instead of readback). width/height should match the connector mode.
// Needs DRM master, so run as root. Present frames with ghee_mp_present.
GheeScene *ghee_display_create(int width, int height, const char *dri_device, int rgb565, char **err);

// Last error message for this instance ("" if none). Cleared on read.
const char *ghee_last_error(GheeScene *g);

void ghee_destroy(GheeScene *g);

// Load a JPEG file into a GL texture on the current context. 0 on failure.
// TODO: bind from Python (image files as texture channels).
GLuint ghee_load_jpeg_texture(const char *filename);

int  ghee_mp_add_pass(GheeScene *g, const char *glsl_source);
// Set a pass's buffer size (default = renderer W,H). Call after ghee_mp_add_pass,
// before the first render. The last pass must keep the renderer size (it feeds
// readback/scanout); consumers sample any pass through normalized UVs.
void ghee_mp_set_pass_size(GheeScene *g, int pass, int w, int h);
void ghee_mp_set_channel(GheeScene *g, int pass, int channel, int src_pass);
bool ghee_mp_set_uniform(GheeScene *g, int pass, const parameter_t *param);
// Unused: superseded by ghee_mp_set_float_texture (uniform arrays hit the VC4 size cap).
void ghee_mp_set_float_array(GheeScene *g, int pass, const char *name, const float *data, int count);
// Data bound as a count x rows luminance texture sampler (<name>_tex): rows=1 is
// the 1xN array case (computed-coord fetch, no VC4 uniform-array cap), rows>1 is
// a 2D input (masks, host-rasterized text) sampled with normalized uv.
void ghee_mp_set_float_texture(GheeScene *g, int pass, const char *name, const float *data, int count, int rows);
void ghee_mp_render(GheeScene *g, uint8_t *buffer_rgb);
// Render the passes and scan the final buffer out to the display (display mode
// only). buffer_rgb non-NULL also reads the frame back (for a console mirror).
void ghee_mp_present(GheeScene *g, uint8_t *buffer_rgb, int pw, int ph);
void ghee_mp_reset(GheeScene *g);

void ghee_set_profiling(GheeScene *g, bool on);
void ghee_set_time(GheeScene *g, float t);
void ghee_get_timings(GheeScene *g, double *draw_ms, double *read_ms);
int  ghee_display_vrefresh(GheeScene *g);  // mode refresh Hz, 0 = not display mode
// Size of the mode the connected display is driving, without creating a scene
// or touching GL. Needs read access to the card node only, not DRM master.
// 0 = width/height set, -1 = no device or nothing connected.
int  ghee_display_probe(const char *dri_device, int *width, int *height);

// Zero-copy hardware H264 (encoder.c): the final pass is blitted into a linear
// dmabuf and fed to the V4L2 M2M encoder (bcm2835-codec); no CPU pixel copy.
// create fails cleanly (NULL + err) on boards without /dev/video11.
typedef struct GheeEncoder GheeEncoder;
GheeEncoder *ghee_encoder_create(GheeScene *g, int w, int h, int fps, int bitrate, char **err);
// Render + encode one frame (no readback). 0 = *nal/*len/*is_key set (valid
// until the next call), 1 = encoder produced nothing yet, <0 = error.
int  ghee_mp_render_encoded(GheeScene *g, GheeEncoder *e, const uint8_t **nal, int *len, int *is_key);
// Encode the frame the last ghee_mp_render/ghee_mp_present produced (display
// mirror: one render feeds screen and stream). Same returns.
int  ghee_encoder_encode_last(GheeScene *g, GheeEncoder *e, const uint8_t **nal, int *len, int *is_key);
void ghee_encoder_force_keyframe(GheeEncoder *e);   // instant join for a new viewer
void ghee_encoder_destroy(GheeScene *g, GheeEncoder *e);

void ghee_pack_frame(int width, int height, int components, int bw,
                    const uint8_t *rgb, uint8_t *packed);

#endif // GHEE_ENGINE_H
