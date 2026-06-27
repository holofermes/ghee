#ifndef GL_VISUALIZER_H
#define GL_VISUALIZER_H
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

// Opaque renderer instance. All state lives here — no globals, so multiple
// independent renderers can coexist in one process.
typedef struct GLScene GLScene;

// Create a headless GLES2 renderer on the given DRI device (NULL -> "/dev/dri/card0").
// Returns NULL on failure; if err != NULL it receives a static error string.
GLScene *gls_create(int width, int height, int components, const char *dri_device, char **err);

// Last error message for this instance ("" if none). Cleared on read.
const char *gls_last_error(GLScene *g);

void gls_destroy(GLScene *g);

// Load a JPEG file into a GL texture on the current context. 0 on failure.
GLuint gls_load_jpeg_texture(const char *filename);

// ---- multipass: a scene as a list of passes (last -> screen). Channels route
// between passes (reading the source pass's previous frame; self = feedback). ----
// The engine is data-agnostic: arrays are named (the host names and fills them),
// params carry their own name/type, channels are routed by index. It has no notion
// of audio, "amplitudes", orientation, or display format; the host owns all of that.
int  gls_mp_add_pass(GLScene *g, const char *glsl_source);                 // -> pass index, or -1
bool gls_mp_set_shader(GLScene *g, int pass, const char *glsl_source);     // recompile a pass live
void gls_mp_set_channel(GLScene *g, int pass, int channel, int src_pass); // iChannel<channel> <- src_pass
bool gls_mp_set_uniform(GLScene *g, int pass, const parameter_t *param);
void gls_mp_set_float_array(GLScene *g, int pass, const char *name, const float *data, int count);
void gls_mp_render(GLScene *g, uint8_t *buffer_rgb);                       // render all passes; final -> buffer_rgb
void gls_mp_reset(GLScene *g);                                            // tear down all passes (scene switch)

// Optional host-side helper: pack an RGB(A) frame into a physical-display format,
// 1bpp column-major (bw) or RGB565 big-endian. Not used by rendering; the host
// calls it if its panel needs it.
void gls_pack_frame(int width, int height, int components, int bw,
                    const uint8_t *rgb, uint8_t *packed);

#endif // GL_VISUALIZER_H
