// The engine behind engine.h. It opens a GL context on a DRM device through
// GBM and EGL, and draws a scene of passes with it.
//
// Each pass owns a program and two buffers, and draws into one of them while
// the other holds its frame before. A frame draws the passes in order, then
// swaps the two buffers of every pass. A channel that reads an earlier pass
// reads the buffer that it just drew, and a channel that reads the pass
// itself or a later one reads the frame before. After its full-screen quad a
// pass draws its batches, the quads of the scene's layers, blended onto its
// buffer. The last pass holds the frame: ghee_mp_render reads it back, and in
// display mode ghee_mp_present scans it out to a KMS connector.
//
// A scene builds in stages. The passes and the batches compile into a pending
// scene beside the running one, and ghee_mp_commit swaps them in at once, so a
// compile error leaves the running scene drawing.

#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl3.h>
#include <unistd.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>
#include <gbm.h>
#include <drm_fourcc.h>
#include <fcntl.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <poll.h>
#include <pthread.h>
#include <errno.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/ioctl.h>
#include <dirent.h>
#include <linux/fb.h>
#include "engine.h"
#include "engine_internal.h"

#define ERROR_BUFFER_SIZE 512
#define DEFAULT_DRI_DEVICE "/dev/dri/card0"
#define MP_MAX_PASSES 16
// The texture units one program binds at most, on any GPU. VC4 has 16.
#define MP_MAX_UNITS 16
// Each channel and each array of a pass takes a texture unit of its own.
#define MP_MAX_CHANNELS MP_MAX_UNITS
#define MP_MAX_DATATEX MP_MAX_UNITS
// Each pass declares MP_MAX_DATATEX array names at most.
#define MP_MAX_ARRAYS (MP_MAX_PASSES * MP_MAX_DATATEX)
#define FLIP_STUCK_MS 5000
// One buffer on the screen, one waiting on a page flip, one waiting for the
// flip thread, and one to draw into. Without a flip thread, three.
#define LENT_BUFFERS 4
// The latest flips and draws that time a panel's next frame.
#define TIMING_RING 16
struct timing_ring { double v[TIMING_RING]; int n, at; };
// The time that a frame drawn just in time waits for the bus, for a draw a
// little slower than the ones before it.
#define JIT_MARGIN 300e-6
// GLES2 guarantees 8 vertex attributes, and VC4 holds no more.
#define MP_BATCH_ATTRIBS 8
// GLES2 guarantees 16-bit indices alone, and a quad takes 4 vertices, so a
// batch of more quads than this draws in chunks.
#define QUAD_CHUNK 16383
// The float of a layer's own batch program, as src/layer.rs writes it, that
// sends every vertex off the pass while it is below 0.5.
#define VISIBLE_UNIFORM "ghee_0_visible"

// An asset that a program reads, by its index in the engine's assets, and
// the locations of its sampler and of its size, each -1 when the program
// does not read it.
struct mp_use {
    int asset;
    GLint loc, size;
};

// The uniform locations of one committed program, a pass's or a batch's,
// resolved once: a link fixes them, and the name lookup is the render loop's
// own cost otherwise.
struct mp_locs {
    GLint res;                           // iResolution, which a draw sets: a batch's program may serve passes of two sizes
    GLint pixel;                         // ghee_pixel, 1 / iResolution.xy
    GLint yflip;                         // ghee_yflip, scale and offset from gl_FragCoord.y to the picture's rows, y up
    GLint time, dt, frame;
    GLint chan_res;                      // iChannelResolution, the size of each channel's source, which a draw sets
    GLint vflip;                         // ghee_vflip of vertex_shader_source, -1 to draw into a lent scanout buffer
    GLint visible;                       // VISIBLE_UNIFORM
    GLint chan[MP_MAX_CHANNELS];
    GLint data[MP_MAX_DATATEX];
    struct mp_use *uses;                 // in asset order, which is the order of their units
    int nuses, uses_cap;
};

// The dma-bufs that one asset draws from, at most.
#define MP_DMABUFS 8

// One image asset: the host hands the pixels over once, they bind as a
// sampler uniform <name>_tex, with <name>_size a vec2 of (width, height).
struct mp_asset {
    GLuint tex;
    char *name;                          // the asset's name, before the suffixes
    int w, h;
    GLenum fmt, type;
    bool mip;                            // every mip level, down to 1x1
    // The dma-bufs that ghee_mp_set_asset_dmabuf imported, each once, by
    // the descriptor that the host gave.
    struct { int fd; EGLImageKHR img; GLuint tex; } dma[MP_DMABUFS];
    int ndma;
    GLuint fbo;                          // draws a dma-buf into `tex`
};

static void mp_forget_dmabufs(GheeScene *g, struct mp_asset *as);
static bool fade_ready(GheeScene *g);

// A batch's program, compiled for the first batch whose two sources ask for
// it and shared by every batch after it that asks for the same two. The last
// batch to let go of it deletes it.
struct mp_prog {
    GLuint program;
    char *vertex, *fragment;
    int holders;
    int loaded;                          // the id of the batch whose values it holds, 0 for none
    // The batch, target and bind_epoch of the last full mp_bind, whose sizes
    // and texture units the program still holds. bound_batch 0 for none.
    int bound_batch;
    int bound_into;
    unsigned bound_epoch;
};

// A uniform of one batch's own, which the batch sets on its program before
// it draws, so batches that share a program each draw with their own values.
struct mp_value {
    char *name;
    param_type_t type;
    float v[4];
    int i;
    GLuint prog;                         // the program that `loc` belongs to
    GLint loc;
};

// One batch of ghee_mp_add_batch: quads that a pass draws into its buffer
// after its full-screen quad, with a program and a blend.
struct mp_batch {
    int pass;
    char name[96];                       // the batch's name in an error
    int id;                              // the renderer's own, from 1, never reused
    GLuint program;
    int blend;                           // GHEE_BLEND_*
    GLuint vbo, vao;
    int quads, nattr;                    // nattr vec4 attributes a vertex
    // A batch whose quads change in part draws from two buffers in turn:
    // vc4 maps a buffer for a write only once the GPU is done reading it, so
    // the next frame's change goes into `spare`, which the frame before the
    // last drew from. `copy` holds every quad, and the quads that `spare`
    // lacks start at stale_first.
    GLuint spare;
    size_t spare_bytes;                  // the bytes that `spare` has room for
    float *copy;
    int stale_first, stale_quads;
    struct mp_locs loc;
    struct mp_value *values;
    int nvalues, values_cap;
    bool dirty;                          // a value changed since the batch last set them
    float opacity;                       // below 1 the batch draws through the fade buffer
    bool hidden;                         // its program sends every vertex off the pass
    GLuint pointed;                      // the buffer that the VAO's attribute pointers read, 0 for none
    int pointed_nattr;
};

struct GheeScene {
    // The scene renders at width x height. The readback, the scanout and the
    // encoder take the frame at out_w x out_h, turned `turn` quarter turns
    // clockwise, so at an odd turn the two sizes swap.
    int width, height, components;
    int out_w, out_h, turn;

    int drm_fd;
    struct gbm_device *gbm;
    struct gbm_surface *gbm_surface;
    EGLDisplay display;
    EGLSurface surface;
    EGLContext context;

    GLuint VBO, VAO;

    struct timespec start_time;
    bool restart;                        // the next frame drawn sets start_time
    bool has_time_override;              // when set, render uses time_override instead of the wall clock
    float time_override;
    float dt;                            // how many seconds the last frame took, fed to every pass as iTimeDelta
    int frame;                           // the frame being drawn, 0 first, fed to every pass as iFrame, wrapped at 2^20 so that float(iFrame) stays exact

    // The running scene's passes, in draw order. Each pass has a program and
    // two buffers, and mp_write names the one that it draws into this frame. A
    // channel reads the buffer drawn this frame from an earlier pass, and the
    // other buffer, the frame before, from the pass itself or a later one. The
    // last pass holds the frame that the readback and the scanout take.
    int mp_count;
    GLuint mp_program[MP_MAX_PASSES];    // 0 = a pass of batches alone, with no quad
    char mp_name[MP_MAX_PASSES][96];     // the pass's name in an error
    bool mp_alpha[MP_MAX_PASSES];        // the buffer stores alpha, else RGB
    GLuint mp_fbo[MP_MAX_PASSES][2];
    GLuint mp_tex[MP_MAX_PASSES][2];
    int mp_w[MP_MAX_PASSES], mp_h[MP_MAX_PASSES];   // per-pass buffer size, W,H until set
    int mp_write[MP_MAX_PASSES];
    int mp_filter[MP_MAX_PASSES];        // GHEE_FILTER_*
    int mp_chan_src[MP_MAX_PASSES][MP_MAX_CHANNELS];  // channel -> source pass index, -1 = none
    int mp_nchan[MP_MAX_PASSES];
    struct mp_locs mp_loc[MP_MAX_PASSES];
    GLint gl_units;                      // GL_MAX_TEXTURE_IMAGE_UNITS, 16 on VC4
    GLint max_units;                     // min(MP_MAX_UNITS, gl_units)
    GLint max_tex;                       // GL_MAX_TEXTURE_SIZE, 2048 on VC4

    // pending scene: passes compiled and allocated beside the running ones and
    // made the scene by ghee_mp_commit, so a compile error leaves the running
    // scene rendering.
    int stage_count;
    GLuint stage_program[MP_MAX_PASSES];
    GLuint stage_fbo[MP_MAX_PASSES][2];
    GLuint stage_tex[MP_MAX_PASSES][2];
    int stage_w[MP_MAX_PASSES], stage_h[MP_MAX_PASSES];
    char stage_name[MP_MAX_PASSES][96];
    bool stage_alpha[MP_MAX_PASSES];
    int stage_filter[MP_MAX_PASSES];
    int stage_from[MP_MAX_PASSES];

    // Batches, staged with the passes and committed with them.
    int mp_nbatch, stage_nbatch;
    // The programs of the batches, running and staged, each held by one or
    // more of them.
    int mp_nprog, mp_prog_cap;
    struct mp_prog *mp_prog;
    int batch_ids;                       // the id of the last batch made
    // The running batches in draw order, each pass drawing its own in this
    // order. A commit draws them in the order of their indices, and
    // ghee_mp_insert_batch puts a batch anywhere among its pass's.
    int *mp_order;
    int mp_order_cap;
    int mp_batch_cap, stage_batch_cap;
    struct mp_batch *mp_batch, *stage_batch;
    GLuint quad_ibo;                     // QUAD_CHUNK quads of indices, shared by every batch

    // GLES2 promises RGBA readback and nothing else. vc4 reads RGB too, v3d
    // hands back nothing for it, so an RGB frame there would come back all
    // black. Where RGB is missing the readback takes RGBA into scratch and
    // drops the alpha, which costs nothing on the boards that read RGB.
    bool reads_rgb;
    uint8_t *rgba_scratch;
    size_t rgba_scratch_size;
    bool has_float_tex, has_float_linear;   // OES_texture_float[_linear] on this GL
    bool has_npot;                       // mipmaps a texture of any size: GLES 3 or OES_texture_npot
    bool gles3;                          // the GL is OpenGL ES 3, which compiles GLSL ES 3.00
    // The arrays: each is a texture of count x rows values, sampled as
    // <name>_tex, which a shader reads at a computed index in one fetch. VC4
    // caps the size of a uniform array that a shader indexes with a variable,
    // and a texture has no such cap. The scene holds one texture per array
    // name, uploaded at most once a frame and bound in every pass that
    // declares the array.
    int mp_narray;
    GLuint mp_array_tex[MP_MAX_ARRAYS];
    char *mp_array_name[MP_MAX_ARRAYS];           // sampler uniform name
    int mp_array_size[MP_MAX_ARRAYS];
    int mp_ndatatex[MP_MAX_PASSES];
    int mp_datatex[MP_MAX_PASSES][MP_MAX_DATATEX];   // texture unit -> array index
    // Engine scope: a scene switch keeps the assets, ghee_mp_drop_asset and
    // ghee_destroy free them.
    int mp_nasset, mp_asset_cap;
    struct mp_asset *mp_asset;
    GLuint black_tex;                    // bound to a channel with no source pass

    bool keep_alpha;                     // readbacks carry the final pass's alpha

    // display mode: scan the final buffer straight out to a KMS connector
    // (zero CPU copy), instead of reading it back. Set up by ghee_display_create.
    bool display_mode;
    uint32_t disp_conn, disp_crtc;
    drmModeModeInfo disp_mode;
    struct gbm_bo *disp_prev;    // buffer of the most recent flip (pending or on the screen)
    struct gbm_bo *disp_shown;   // buffer of the flip before that (still on the screen until
                                 // the pending one completes, released after)
    bool disp_first;
    // A display driver that reads each frame with the CPU (ssd130x, gud, udl)
    // maps its own dumb buffers cached, and nothing keeps that cache coherent
    // with a GPU writing into them. So when the card is a display-only device
    // apart from the GPU, the GPU allocates the scanout buffers on its render
    // node and lends them to the card, which imports them. vc4 keeps Mesa's
    // path either way: as the GPU, Mesa already lends vc4's buffers to the
    // display, and as the display, vc4's dumb buffers are uncached.
    int gpu_fd;                  // the render node holding the lent buffers, -1 otherwise
    int lent_count;              // 0 when the window surface's buffers scan out
    struct { struct gbm_bo *bo; EGLImageKHR image; GLuint tex, fbo; } lent[LENT_BUFFERS];
    GLuint scan_fbo;             // the scanout target of this present: 0 or a lent buffer's FBO
    // A buffer smaller than the mode scans out through the display's own
    // scaler: the plane takes a source rectangle the size of the buffer and a
    // destination rectangle the size of the mode, and the hardware stretches
    // it. The GPU shades the smaller count, and the stretch costs nothing.
    // Page flip carries no source rectangle, so this path sets the plane every
    // frame and waits a vblank of its own for pacing.
    uint32_t disp_plane;         // the primary plane, 0 when the render fills the mode
    bool disp_blanked;           // the CRTC drives nothing: ghee_mp_present draws nothing
    bool flip_pending;           // a submitted page flip whose event hasn't been reaped
    bool flip_failed;            // a refused flip, reported once instead of every frame
    // On a host with more than one CPU, the flip thread submits each page
    // flip and reaps its event, and present queues a frame for it and
    // returns. A kernel can run a flip's work on the CPU that submitted it,
    // so the work then waits behind no drawing, and the next frame draws
    // while the last one waits on the GPU. The thread makes no GL call.
    // flip_mu guards disp_prev, disp_shown, flip_pending, flip_failed and
    // everything below while the thread runs.
    bool flipper;                // the flip thread runs
    pthread_t flip_thread;
    pthread_mutex_t flip_mu;
    pthread_cond_t flip_cv;
    struct gbm_bo *flip_queued;  // a drawn frame that waits for its flip, NULL for none
    struct gbm_bo *flip_sending; // the frame inside the flip ioctl, NULL for none
    uint32_t flip_queued_fb;
    bool flip_stop;
    struct gbm_bo *flip_freed[LENT_BUFFERS];   // off the glass, for present to release
    int flip_nfreed;
    char flip_error[ERROR_BUFFER_SIZE];        // the thread's error, for present to report
    // A display with no vblank, a panel on a bus, takes the next frame the
    // moment the transfer before it ends. present then returns once its frame
    // is on the bus, as late as lets the next draw end when the bus frees.
    bool jit;
    bool drop;                   // a frame that waits for its flip gives way to the next
    double flip_sent;            // CLOCK_MONOTONIC seconds of the latest flip's ioctl
    double flip_landed;          // the timestamp of the latest flip's event
    struct timing_ring flip_busy;   // each flip's ioctl to its event
    struct timing_ring draw_time;   // each return of present to its next frame, drawn
    double presented;            // the latest return of present, 0 before the first
    uint64_t frames;             // the frames that present has handed to the display
    uint64_t flip_queued_frame;  // flip_queued's number among them
    uint64_t taken;              // the number of the latest frame that the display took
    GLuint blit_prog;
    GLint blit_sampler, blit_opaque, blit_turn, blit_shift;
    GLuint preview_fbo, preview_tex;   // small FBO for a cheap downscaled readback
    int preview_w, preview_h;
    // A batch below opacity 1 draws into the fade buffer, as large as the
    // largest pass that needed it, and fade_prog lays the buffer on its pass.
    GLuint fade_fbo, fade_tex;
    int fade_w, fade_h;
    GLuint fade_prog;
    GLint fade_sampler, fade_scale, fade_opacity, fade_vflip;
    bool fade_failed;                  // a fade buffer that GL refused, reported once
    int keep_last;                     // >0: an encoder samples the final pass, never draw it direct
    // A headless readback of the whole frame can draw the last pass straight
    // into a linear buffer and copy that out with the CPU. glReadPixels runs
    // a blit on the GPU and detiles on the CPU, which costs VC4 and V3D more.
    int read_linear;                   // 1 wherever the buffer can be made, -1 on VC4 and V3D, 0 never
    struct gbm_bo *lin_bo;
    EGLImageKHR lin_image;
    GLuint lin_tex, lin_fbo;
    bool lin_tried;
    bool lin_last;                     // the last frame left its last pass in lin_fbo alone

    // `changed` is set by every call that changes what the running scene
    // draws, and a drawn frame clears it. `kept` holds the readback of the
    // last frame drawn, as asked at kept_w x kept_h, while nothing that draws
    // reads the time.
    bool changed;
    unsigned bind_epoch;               // counts the calls that change what mp_bind sets on a batch's program
    uint64_t draws;                    // render_passes runs
    uint64_t draws_seen;               // `draws` at the last ghee_mp_drew
    bool drawn;
    uint8_t *kept;
    size_t kept_len, kept_cap;
    int kept_w, kept_h;
    bool kept_valid;

    char renderer[160];                  // GL_RENDERER and GL_VERSION, captured at create
    int compiled;                        // programs linked since create
    char *last_error;                    // NULL when there is none
    char *read_error;                    // what ghee_last_error last returned
    bool error_is_compile;               // last_error is a GLSL compile or link log
};

// Where a pass draws: its own buffer, the window surface, or a lent scanout
// buffer.
enum { INTO_BUFFER, INTO_WINDOW, INTO_SCANOUT };

// The vertex shader of every pass's full-screen quad, and of the present
// blit. A pass's buffer stores the picture's bottom row first, GL's row
// order. A lent scanout buffer is an FBO that the card reads top row first,
// so a pass that draws straight into it takes ghee_vflip = -1, which mirrors
// the quad in y, and its preamble turns gl_FragCoord.y back by ghee_yflip.
#define QUAD_MAIN \
    "uniform float ghee_vflip;" \
    "void main() {" \
    "   gl_Position = vec4(position.x, position.y * ghee_vflip, position.z, position.w);" \
    "   uv = texCoord;" \
    "}"
static const char *vertex_shader_source =
    "#version 100\n"
    "attribute vec4 position;"
    "attribute vec2 texCoord;"
    "varying vec2 uv;"
    QUAD_MAIN;

// GLSL ES links a program of shaders of one version alone, so a GLSL ES 3.00
// pass takes the quad in GLSL ES 3.00.
static const char *vertex_shader_es3_source =
    "#version 300 es\n"
    "in vec4 position;"
    "in vec2 texCoord;"
    "out vec2 uv;"
    QUAD_MAIN;

// Blit: sample the final pass texture onto a target. `turn` and `shift` map
// the target's uv onto the texture's, and opaque 1 writes alpha 1.
static const char *blit_fragment_source =
    "#version 100\n"
    "precision mediump float;"
    "varying vec2 uv;"
    "uniform sampler2D tex;"
    "uniform float opaque;"
    "uniform mat2 turn;"
    "uniform vec2 shift;"
    "void main() { vec4 c = texture2D(tex, turn * uv + shift);"
    " gl_FragColor = vec4(c.rgb, max(c.a, opaque)); }";

// Lay the fade buffer on a pass at `opacity`: `scale` maps the pass's uv
// into the corner of the buffer that the batch drew in.
static const char *fade_fragment_source =
    "#version 100\n"
    "precision mediump float;"
    "varying vec2 uv;"
    "uniform sampler2D tex;"
    "uniform vec2 scale;"
    "uniform float opacity;"
    "void main() { gl_FragColor = texture2D(tex, uv * scale) * opacity; }";

// Each quarter turn clockwise as a matrix, column-major, then its shift: the
// target's uv o samples the frame at turn * o + shift, both with y down from
// the top row.
static const GLfloat TURNS[4][6] = {
    { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f },
    { 0.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f },
    { -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f },
    { 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f },
};

static int say_level = 1;
void ghee_set_verbosity(int level) { say_level = level; }
#define SAY(at, ...) do { if (say_level >= (at)) fprintf(stderr, __VA_ARGS__); } while (0)

// Defined further down, beside the present path that they serve.
static void settle_flips(GheeScene *g);
static void stop_flipper(GheeScene *g);
static bool start_flipper(GheeScene *g, char *eb, size_t n);
static uint32_t primary_plane(int fd, uint32_t crtc_id);
static bool display_takes(int fd, uint32_t format);
static int pick_display_mode(int fd, uint32_t *conn_out, uint32_t *crtc_out,
                             drmModeModeInfo *mode_out);
static bool plane_scales(GheeScene *g, int width, int height);
static bool ensure_blit(GheeScene *g);

// A GL's compile log runs to any length, so the error is sized to its text.
static void keep_error(GheeScene *g, const char *fmt, va_list ap) {
    va_list again;
    va_copy(again, ap);
    const int n = vsnprintf(NULL, 0, fmt, ap);
    char *s = n < 0 ? NULL : malloc((size_t)n + 1);
    if (s) vsnprintf(s, (size_t)n + 1, fmt, again);
    va_end(again);
    free(g->last_error);
    g->last_error = s;
}

static const char *error_text(GheeScene *g) { return g->last_error ? g->last_error : ""; }

static void set_error(GheeScene *g, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    keep_error(g, fmt, ap);
    va_end(ap);
    g->error_is_compile = false;
    fprintf(stderr, "ghee: %s\n", error_text(g));
}

// A compile or link log goes to the caller unprinted: the caller names the
// pass and the author's line in it, and reports it once. A GL log ends with
// a newline, which the error drops.
static void set_shader_error(GheeScene *g, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    keep_error(g, fmt, ap);
    va_end(ap);
    size_t n = g->last_error ? strlen(g->last_error) : 0;
    while (n > 0 && strchr(" \t\r\n", g->last_error[n - 1])) g->last_error[--n] = '\0';
    g->error_is_compile = true;
}

// Make this instance's GL context current. Required at the top of every public
// op so multiple GheeScene instances can be used interleaved in one process.
static void make_current(GheeScene *g) {
    if (!eglMakeCurrent(g->display, g->surface, g->surface, g->context))
        fprintf(stderr, "ghee: eglMakeCurrent failed: 0x%X (disp=%p ctx=%p)\n",
                eglGetError(), (void *)g->display, (void *)g->context);
}

const char *ghee_last_error(GheeScene *g) {
    if (!g) return "null instance";
    free(g->read_error);
    g->read_error = g->last_error;
    g->last_error = NULL;
    return g->read_error ? g->read_error : "";
}

bool ghee_last_error_is_compile(GheeScene *g) { return g && g->error_is_compile; }

// ---- shader compilation ---------------------------------------------------

static GLuint create_shader(GheeScene *g, GLenum type, const char *source) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        char *info = calloc(len > 0 ? (size_t)len : 1, 1);
        if (info && len > 0) glGetShaderInfoLog(shader, len, NULL, info);
        set_shader_error(g, "%s%s", type == GL_VERTEX_SHADER ? "vertex shader: " : "",
                         info ? info : "");
        free(info);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// The attributes bind to locations 0, 1, ... in the order of `attribs`, so a
// vertex layout set once holds for every program that reads it.
static GLuint create_program(GheeScene *g, const char *vertex_source, const char *fragment_source,
                             const char *const *attribs, int nattribs) {
    const GLuint vs = create_shader(g, GL_VERTEX_SHADER, vertex_source);
    if (!vs) return 0;
    const GLuint fs = create_shader(g, GL_FRAGMENT_SHADER, fragment_source);
    if (!fs) { glDeleteShader(vs); return 0; }

    const GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    for (int i = 0; i < nattribs; i++) glBindAttribLocation(program, (GLuint)i, attribs[i]);
    glLinkProgram(program);
    GLint ok;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
        char *info = calloc(len > 0 ? (size_t)len : 1, 1);
        if (info && len > 0) glGetProgramInfoLog(program, len, NULL, info);
        set_shader_error(g, "%s", info ? info : "");
        free(info);
        glDeleteShader(vs);
        glDeleteShader(fs);
        glDeleteProgram(program);
        return 0;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    g->compiled++;
    return program;
}

static const char *const quad_attribs[] = { "position", "texCoord" };
static const char *const batch_attribs[MP_BATCH_ATTRIBS] = {
    "ghee_a0", "ghee_a1", "ghee_a2", "ghee_a3", "ghee_a4", "ghee_a5", "ghee_a6", "ghee_a7"
};

// A program on the shared full-screen quad vertex shader.
static GLuint create_shader_program(GheeScene *g, const char *fragment_shader_source) {
    const bool es3 = strncmp(fragment_shader_source, "#version 300 es", 15) == 0;
    return create_program(g, es3 ? vertex_shader_es3_source : vertex_shader_source,
                          fragment_shader_source, quad_attribs, 2);
}

static char *suffixed(GheeScene *g, const char *name, const char *suffix);

// The batch program of `vertex_source` and `fragment_source`, with one more
// holder: the program that a batch holds already, else a new one. 0 with
// the error set on a compile error or when memory runs out.
static GLuint program_take(GheeScene *g, const char *vertex_source, const char *fragment_source) {
    for (int i = 0; i < g->mp_nprog; i++) {
        struct mp_prog *p = &g->mp_prog[i];
        if (strcmp(p->fragment, fragment_source) == 0 && strcmp(p->vertex, vertex_source) == 0) {
            p->holders++;
            return p->program;
        }
    }
    if (g->mp_nprog == g->mp_prog_cap) {
        const int cap = g->mp_prog_cap ? 2 * g->mp_prog_cap : 16;
        struct mp_prog *grown = realloc(g->mp_prog, (size_t)cap * sizeof *grown);
        if (!grown) { set_error(g, "no memory for program %d", g->mp_nprog); return 0; }
        g->mp_prog = grown;
        g->mp_prog_cap = cap;
    }
    char *vertex = suffixed(g, vertex_source, "");
    char *fragment = suffixed(g, fragment_source, "");
    const GLuint prog = vertex && fragment
        ? create_program(g, vertex_source, fragment_source, batch_attribs, MP_BATCH_ATTRIBS) : 0;
    if (!prog) {
        free(vertex);
        free(fragment);
        return 0;
    }
    g->mp_prog[g->mp_nprog++] = (struct mp_prog){ .program = prog, .vertex = vertex,
                                                  .fragment = fragment, .holders = 1 };
    return prog;
}

// One holder of batch program `prog` lets go of it, and the last deletes it.
static void program_drop(GheeScene *g, GLuint prog) {
    for (int i = 0; prog && i < g->mp_nprog; i++) {
        struct mp_prog *p = &g->mp_prog[i];
        if (p->program != prog) continue;
        if (--p->holders > 0) return;
        glDeleteProgram(prog);
        free(p->vertex);
        free(p->fragment);
        g->mp_prog[i] = g->mp_prog[--g->mp_nprog];
        return;
    }
}

// ---- lifecycle ------------------------------------------------------------

// Send stderr to /dev/null while GBM and EGL come up, and return the saved
// stderr for unhush_stderr, or -1 when the redirect failed and unhush_stderr
// has nothing to restore. Mesa's loader prints a burst of "failed to retrieve
// device information" while it brings up the GPU that a display-only card
// borrows, which the reader can do nothing about, and ghee reads its own
// failures back through eglGetError.
static int hush_stderr(void) {
    const int null = open("/dev/null", O_WRONLY | O_CLOEXEC);
    if (null < 0) return -1;
    const int saved = dup(STDERR_FILENO);
    if (saved < 0) { close(null); return -1; }
    fflush(stderr);
    dup2(null, STDERR_FILENO);
    close(null);
    return saved;
}

static void unhush_stderr(int saved) {
    if (saved < 0) return;
    fflush(stderr);
    dup2(saved, STDERR_FILENO);
    close(saved);
}

static bool driver_is(int fd, const char *name) {
    drmVersionPtr v = drmGetVersion(fd);
    const bool is = v && v->name && strcmp(v->name, name) == 0;
    drmFreeVersion(v);
    return is;
}

// The sysfs device behind a DRM node. A card and its own render node resolve
// to the same device. out holds PATH_MAX bytes.
static bool node_device(int fd, char *out) {
    struct stat st;
    if (fstat(fd, &st) || !S_ISCHR(st.st_mode)) return false;
    char link[64];
    snprintf(link, sizeof link, "/sys/dev/char/%u:%u/device", major(st.st_rdev), minor(st.st_rdev));
    return realpath(link, out) != NULL;
}

static void gem_close(int fd, uint32_t handle) {
    struct drm_gem_close c = { .handle = handle };
    drmIoctl(fd, DRM_IOCTL_GEM_CLOSE, &c);
}

// Import a GPU buffer into the display card and wrap it in a framebuffer
// there. The handle and the framebuffer belong to the card. On failure errno
// holds the card's refusal.
static bool card_import(int card, struct gbm_bo *bo, uint32_t *handle, uint32_t *fb) {
    const int fd = gbm_bo_get_fd(bo);
    if (fd < 0) return false;
    const int refused = drmPrimeFDToHandle(card, fd, handle);
    const int e = errno;
    close(fd);
    if (refused) { errno = e; return false; }
    const uint32_t handles[4] = { *handle }, strides[4] = { gbm_bo_get_stride(bo) }, offsets[4] = { 0 };
    if (drmModeAddFB2(card, gbm_bo_get_width(bo), gbm_bo_get_height(bo), GBM_FORMAT_XRGB8888,
                      handles, strides, offsets, fb, 0)) {
        const int e2 = errno;
        gem_close(card, *handle);
        errno = e2;
        return false;
    }
    return true;
}

// A GBM device on the render node of the GPU that Mesa pairs with the card,
// when the card takes lent buffers: a display-only device apart from the GPU,
// neither driver vc4, and a test buffer from the GPU imports. NULL keeps the
// window surface.
static struct gbm_device *lending_gpu(GheeScene *g, int width, int height) {
    if (driver_is(g->drm_fd, "vc4")) return NULL;
    PFNEGLQUERYDISPLAYATTRIBEXTPROC query_display =
        (PFNEGLQUERYDISPLAYATTRIBEXTPROC)eglGetProcAddress("eglQueryDisplayAttribEXT");
    PFNEGLQUERYDEVICESTRINGEXTPROC query_device =
        (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress("eglQueryDeviceStringEXT");
    EGLAttrib dev = 0;
    if (!query_display || !query_device || !query_display(g->display, EGL_DEVICE_EXT, &dev)) return NULL;
    const char *egl_node = query_device((EGLDeviceEXT)dev, EGL_DRM_RENDER_NODE_FILE_EXT);
    if (!egl_node) return NULL;
    char node[PATH_MAX];
    snprintf(node, sizeof node, "%s", egl_node);
    const int fd = open(node, O_RDWR | O_CLOEXEC);
    if (fd < 0) return NULL;
    char card_dev[PATH_MAX], gpu_dev[PATH_MAX];
    if (!node_device(g->drm_fd, card_dev) || !node_device(fd, gpu_dev) ||
        strcmp(card_dev, gpu_dev) == 0 || driver_is(fd, "vc4")) {
        close(fd);
        return NULL;
    }
    struct gbm_device *gbm = gbm_create_device(fd);
    struct gbm_bo *bo = gbm ? gbm_bo_create(gbm, width, height, GBM_FORMAT_XRGB8888,
                                            GBM_BO_USE_RENDERING | GBM_BO_USE_LINEAR) : NULL;
    uint32_t handle = 0, fb = 0;
    const bool takes = bo && card_import(g->drm_fd, bo, &handle, &fb);
    if (takes) {
        drmModeRmFB(g->drm_fd, fb);
        gem_close(g->drm_fd, handle);
    } else {
        SAY(2, "  scanout  the display cannot import buffers from %s: %s\n",
                node, bo ? strerror(errno) : "no linear buffer");
    }
    if (bo) gbm_bo_destroy(bo);
    if (!takes) {
        if (gbm) gbm_device_destroy(gbm);
        close(fd);
        return NULL;
    }
    SAY(1, "  scanout  buffers from %s, imported by the display\n", node);
    return gbm;
}

static GheeScene *create_impl(int width, int height, int components, const char *dri_device, int card, int scanout, int rgb565, char **err) {
    static char early_err[ERROR_BUFFER_SIZE];
    early_err[0] = '\0';

    GheeScene *g = calloc(1, sizeof(GheeScene));
    if (!g) { if (card >= 0) close(card); if (err) *err = "out of memory"; return NULL; }
    g->width = g->out_w = width;
    g->height = g->out_h = height;
    g->components = components;
    g->drm_fd = -1;
    g->gpu_fd = -1;
    g->display = EGL_NO_DISPLAY;

    const char *dev = dri_device ? dri_device : DEFAULT_DRI_DEVICE;
    g->drm_fd = card >= 0 ? card : open(dev, O_RDWR | O_CLOEXEC);
    if (g->drm_fd < 0) {
        snprintf(early_err, sizeof(early_err), "cannot open DRI device %s", dev);
        goto fail;
    }
    const int hushed = hush_stderr();
    g->gbm = gbm_create_device(g->drm_fd);
    if (!g->gbm) {
        unhush_stderr(hushed);
        snprintf(early_err, sizeof(early_err), "gbm_create_device failed");
        goto fail;
    }

    g->display = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, g->gbm, NULL);
    if (g->display == EGL_NO_DISPLAY) {
        unhush_stderr(hushed);
        snprintf(early_err, sizeof(early_err), "eglGetPlatformDisplay failed");
        goto fail;
    }

    int major, minor;
    const EGLBoolean ready = eglInitialize(g->display, &major, &minor);
    unhush_stderr(hushed);
    if (ready == EGL_FALSE) {
        snprintf(early_err, sizeof(early_err), "eglInitialize failed: 0x%X", eglGetError());
        goto fail;
    }
    SAY(2, "      egl  %d.%d on %s\n", major, minor, dev);

    struct gbm_device *lender = scanout ? lending_gpu(g, width, height) : NULL;
    if (lender) {
        eglTerminate(g->display);
        gbm_device_destroy(g->gbm);
        g->gbm = lender;
        g->gpu_fd = gbm_device_get_fd(lender);
        const int hushed_gpu = hush_stderr();
        g->display = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, g->gbm, NULL);
        const EGLBoolean gpu_ready = g->display != EGL_NO_DISPLAY &&
                                     eglInitialize(g->display, &major, &minor);
        unhush_stderr(hushed_gpu);
        if (!gpu_ready) {
            snprintf(early_err, sizeof(early_err), "eglInitialize on the GPU's render node failed: 0x%X", eglGetError());
            goto fail;
        }
    }

    const GLint alphaSize = (components == 4) ? 8 : 0;
    const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, alphaSize,
        EGL_DEPTH_SIZE, 24,
        EGL_STENCIL_SIZE, 8,
        // no MSAA: a pass and a batch antialias in their own shaders, so
        // multisampling adds nothing visible and (on VC4) quadruples the
        // per-pixel tile work
        EGL_SAMPLE_BUFFERS, 0,
        EGL_SAMPLES, 0,
        EGL_NONE
    };
    // Optional (rgb565): display scanout renders RGB565, halving the bytes
    // through the GPU's tile-store path (~300MB/s effective on VC4-class
    // boards, the frame rate ceiling) and halving scanout traffic. The HVS
    // expands 565 to the wire's 888 during scanout, so the panel sees a
    // normal HDMI signal, and the only cost is color depth. GL dithering (on by
    // default) hides most banding. Intermediate pass FBOs stay RGBA8888.
    const EGLint configAttribs565[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 5,
        EGL_GREEN_SIZE, 6,
        EGL_BLUE_SIZE, 5,
        EGL_ALPHA_SIZE, 0,
        EGL_DEPTH_SIZE, 24,
        EGL_STENCIL_SIZE, 8,
        // no MSAA on the scanout surface: 4x samples quadruple the per-pixel
        // tile work, and the shaders antialias their own edges
        EGL_SAMPLE_BUFFERS, 0,
        EGL_SAMPLES, 0,
        EGL_NONE
    };
    if (scanout && rgb565 && !display_takes(g->drm_fd, DRM_FORMAT_RGB565)) {
        snprintf(early_err, sizeof(early_err), "this display takes no RGB565 frames, so render without rgb565");
        goto fail;
    }
    EGLConfig configs[64];
    EGLConfig config = NULL;
    EGLint numConfigs = 0;
    uint32_t surf_fmt = GBM_FORMAT_XRGB8888;
    if (scanout && rgb565 && !lender &&
        eglChooseConfig(g->display, configAttribs565, configs, 64, &numConfigs) &&
        numConfigs > 0) {
        // the chosen config's native visual must equal the GBM surface format,
        // so filter for a true 565 visual rather than trusting sort order
        for (int i = 0; i < numConfigs; i++) {
            EGLint vid = 0;
            if (eglGetConfigAttrib(g->display, configs[i], EGL_NATIVE_VISUAL_ID, &vid) &&
                (uint32_t)vid == GBM_FORMAT_RGB565) {
                config = configs[i];
                surf_fmt = GBM_FORMAT_RGB565;
                break;
            }
        }
    }
    if (!config) {   // offscreen, or no 565 visual on this driver: 888 as before
        if (!eglChooseConfig(g->display, configAttribs, configs, 1, &numConfigs) || numConfigs < 1) {
            snprintf(early_err, sizeof(early_err), "eglChooseConfig failed: 0x%X", eglGetError());
            goto fail;
        }
        config = configs[0];
    }
    if (scanout)
        SAY(2, "   format  scanout %s\n",
                surf_fmt == GBM_FORMAT_RGB565 ? "RGB565" : "XRGB8888");

    if (!lender) {
        g->gbm_surface = gbm_surface_create(g->gbm, width, height, surf_fmt,
            GBM_BO_USE_RENDERING | (scanout ? GBM_BO_USE_SCANOUT : 0));
        if (!g->gbm_surface) { snprintf(early_err, sizeof(early_err), "gbm_surface_create failed"); goto fail; }
        g->surface = eglCreatePlatformWindowSurface(g->display, config, g->gbm_surface, NULL);
    }
    if (g->surface == EGL_NO_SURFACE) {
        // Desktop Mesa rejects the GBM window surface (EGL_BAD_MATCH) that the Pi's
        // VC4 accepts, and lent scanout buffers carry FBOs of their own. Rendering
        // goes to FBOs, so a surfaceless context serves.
        const char *exts = eglQueryString(g->display, EGL_EXTENSIONS);
        if (!exts || !strstr(exts, "EGL_KHR_surfaceless_context")) {
            snprintf(early_err, sizeof(early_err), "eglCreatePlatformWindowSurface failed: 0x%X", eglGetError());
            goto fail;
        }
    }
    const EGLint contextAttribs[] = { EGL_CONTEXT_MAJOR_VERSION_KHR, 2, EGL_CONTEXT_MINOR_VERSION_KHR, 0, EGL_NONE };
    g->context = eglCreateContext(g->display, config, EGL_NO_CONTEXT, contextAttribs);
    if (g->context == EGL_NO_CONTEXT) {
        snprintf(early_err, sizeof(early_err), "eglCreateContext failed: 0x%X", eglGetError());
        goto fail;
    }
    if (!eglMakeCurrent(g->display, g->surface, g->surface, g->context)) {
        snprintf(early_err, sizeof(early_err), "eglMakeCurrent failed: 0x%X", eglGetError());
        goto fail;
    }

    const GLfloat vertices[] = {
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    glGenBuffers(1, &g->VBO);
    glGenVertexArrays(1, &g->VAO);
    glBindVertexArray(g->VAO);
    glBindBuffer(GL_ARRAY_BUFFER, g->VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid *)(2 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    // Both default to 4. Rows of an 8-bit data texture are packed at its width,
    // so any width that is not a multiple of 4 would otherwise be read with a
    // padded stride: each row shifted from the one above, and the tail read past
    // the end of the buffer.
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &g->gl_units);
    if (g->gl_units < 1) g->gl_units = 8;   // the GLES2 minimum
    g->max_units = g->gl_units < MP_MAX_UNITS ? g->gl_units : MP_MAX_UNITS;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &g->max_tex);
    if (g->max_tex < 64) g->max_tex = 64;   // the GLES2 minimum

    // GL_IMPLEMENTATION_COLOR_READ_FORMAT answers RGBA on hardware that reads
    // RGB perfectly well, so the probe reads a known pixel instead: a driver
    // without RGB hands back nothing rather than raising an error.
    uint8_t probe[3] = { 0, 0, 0 };
    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    while (glGetError() != GL_NO_ERROR) { }
    glReadPixels(0, 0, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, probe);
    g->reads_rgb = glGetError() == GL_NO_ERROR && probe[0] > 200 && probe[1] < 60;
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    snprintf(g->renderer, sizeof g->renderer, "%s, %s",
             (const char *)glGetString(GL_RENDERER), (const char *)glGetString(GL_VERSION));

    const char *glexts = (const char *)glGetString(GL_EXTENSIONS);
    g->has_float_tex = glexts && strstr(glexts, "GL_OES_texture_float") != NULL;
    g->has_float_linear = glexts && strstr(glexts, "GL_OES_texture_float_linear") != NULL;
    const char *version = (const char *)glGetString(GL_VERSION);
    g->gles3 = version && strncmp(version, "OpenGL ES 3", 11) == 0;
    g->has_npot = (glexts && strstr(glexts, "GL_OES_texture_npot") != NULL) || g->gles3;
    SAY(2, "   format  data texture %s%s\n",
            g->has_float_tex ? "float" : "8-bit LUMINANCE",
            g->has_float_tex && g->has_float_linear ? " (linear)" : " (nearest)");

    clock_gettime(CLOCK_MONOTONIC, &g->start_time);
    g->dt = 1.0f / 60.0f;
    g->read_linear = -1;
    return g;

fail:
    if (err) *err = early_err;
    ghee_destroy(g);
    return NULL;
}

GheeScene *ghee_create(int width, int height, int components, const char *dri_device, char **err) {
    return create_impl(width, height, components, dri_device, -1, 0, 0, err);
}

// One DRM framebuffer per GBM bo, cached in the bo's user data (added once).
// A lent bo also holds the card's handle from its import, 0 otherwise.
struct bo_fb { int drm_fd; uint32_t fb_id; uint32_t handle; };
static void bo_destroy_fb(struct gbm_bo *bo, void *data) {
    (void)bo;
    struct bo_fb *f = data;
    if (!f) return;
    if (f->fb_id) drmModeRmFB(f->drm_fd, f->fb_id);
    if (f->handle) gem_close(f->drm_fd, f->handle);
    free(f);
}
static uint32_t fb_for(GheeScene *g, struct gbm_bo *bo) {
    struct bo_fb *cached = gbm_bo_get_user_data(bo);
    if (cached) return cached->fb_id;
    uint32_t handles[4] = { gbm_bo_get_handle(bo).u32, 0, 0, 0 };
    uint32_t strides[4] = { gbm_bo_get_stride(bo), 0, 0, 0 };
    uint32_t offsets[4] = { 0, 0, 0, 0 };
    uint32_t fb = 0;
    if (drmModeAddFB2(g->drm_fd, gbm_bo_get_width(bo), gbm_bo_get_height(bo),
                      gbm_bo_get_format(bo), handles, strides, offsets, &fb, 0)) {
        set_error(g, "drmModeAddFB2 failed: %s", strerror(errno));
        return 0;
    }
    struct bo_fb *f = calloc(1, sizeof *f);
    f->drm_fd = g->drm_fd; f->fb_id = fb;
    gbm_bo_set_user_data(bo, f, bo_destroy_fb);
    return fb;
}

// Allocate the lent buffers on the GPU, import each into the card with its
// framebuffer, and wrap each in an FBO for the GPU to draw into. The context
// must be current. false leaves the reason in eb.
static bool lend_buffers(GheeScene *g, char *eb, size_t n) {
    PFNEGLCREATEIMAGEKHRPROC create_image = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
    PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target =
        (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
    if (!create_image || !image_target) { snprintf(eb, n, "EGL on the GPU cannot import a dma-buf"); return false; }
    const int count = g->flipper ? LENT_BUFFERS : LENT_BUFFERS - 1;
    for (int i = g->lent_count; i < count; i++) {
        // The card's driver reads a buffer row by row and knows nothing of the
        // GPU's tiling, so every lent buffer is linear.
        struct gbm_bo *bo = gbm_bo_create(g->gbm, g->out_w, g->out_h, GBM_FORMAT_XRGB8888,
                                          GBM_BO_USE_RENDERING | GBM_BO_USE_LINEAR);
        if (!bo) { snprintf(eb, n, "the GPU cannot allocate a linear %dx%d scanout buffer", g->out_w, g->out_h); return false; }
        g->lent[i].bo = bo;
        g->lent_count = i + 1;
        uint32_t handle = 0, fb = 0;
        if (!card_import(g->drm_fd, bo, &handle, &fb)) {
            snprintf(eb, n, "the display refused a scanout buffer from the GPU: %s", strerror(errno));
            return false;
        }
        struct bo_fb *f = calloc(1, sizeof *f);
        if (!f) { drmModeRmFB(g->drm_fd, fb); gem_close(g->drm_fd, handle); snprintf(eb, n, "out of memory"); return false; }
        f->drm_fd = g->drm_fd; f->fb_id = fb; f->handle = handle;
        gbm_bo_set_user_data(bo, f, bo_destroy_fb);
        // v3d reads a dma-buf that arrives without a modifier as tiled.
        uint64_t mod = gbm_bo_get_modifier(bo);
        if (mod == DRM_FORMAT_MOD_INVALID) mod = DRM_FORMAT_MOD_LINEAR;
        const int fd = gbm_bo_get_fd(bo);
        if (fd < 0) { snprintf(eb, n, "cannot export a scanout buffer from the GPU"); return false; }
        const EGLint attr[] = {
            EGL_WIDTH, g->out_w,
            EGL_HEIGHT, g->out_h,
            EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_XRGB8888,
            EGL_DMA_BUF_PLANE0_FD_EXT, fd,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
            EGL_DMA_BUF_PLANE0_PITCH_EXT, (EGLint)gbm_bo_get_stride(bo),
            EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, (EGLint)(mod & 0xffffffff),
            EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, (EGLint)(mod >> 32),
            EGL_NONE };
        g->lent[i].image = create_image(g->display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, attr);
        close(fd);
        if (g->lent[i].image == EGL_NO_IMAGE_KHR) {
            snprintf(eb, n, "EGL cannot import its own scanout buffer: 0x%X", eglGetError());
            return false;
        }
        glGenTextures(1, &g->lent[i].tex);
        glBindTexture(GL_TEXTURE_2D, g->lent[i].tex);
        image_target(GL_TEXTURE_2D, g->lent[i].image);
        glGenFramebuffers(1, &g->lent[i].fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, g->lent[i].fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g->lent[i].tex, 0);
        const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            snprintf(eb, n, "a scanout buffer from the GPU is incomplete as a framebuffer: 0x%X", status);
            return false;
        }
    }
    return true;
}

// Hand a buffer back once the display lets go of it. A lent buffer is free as
// soon as it is neither disp_prev nor disp_shown.
static void release_bo(GheeScene *g, struct gbm_bo *bo) {
    if (bo && !g->lent_count) gbm_surface_release_buffer(g->gbm_surface, bo);
}

// The connector if a display with modes is on it, else NULL.
// drmModeGetConnectorCurrent reads the kernel's cached connector state.
// drmModeGetConnector probes the connector first, and on HDMI the probe
// rereads the EDID over the live cable. A connector caches no modes until
// its first probe.
static drmModeConnector *connected_display(int fd, uint32_t id) {
    for (int probe = 0; probe < 2; probe++) {
        drmModeConnector *c = probe ? drmModeGetConnector(fd, id) : drmModeGetConnectorCurrent(fd, id);
        if (c && c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) return c;
        if (c) drmModeFreeConnector(c);
    }
    return NULL;
}

// Pick the display to drive: first connected connector with modes, its
// encoder's CRTC (else the first), and the mode that CRTC already drives
// (honours a video= override or whatever the console negotiated, falls back
// to the connector's first). Shared by create and probe, so probe reports
// exactly the mode that create will drive. -1 = nothing connected.
static int pick_display_mode(int fd, uint32_t *conn_out, uint32_t *crtc_out,
                             drmModeModeInfo *mode_out) {
    drmModeRes *res = drmModeGetResources(fd);
    if (!res) return -1;
    drmModeConnector *conn = NULL;
    for (int i = 0; i < res->count_connectors && !conn; i++)
        conn = connected_display(fd, res->connectors[i]);
    if (!conn) { drmModeFreeResources(res); return -1; }
    drmModeEncoder *enc = drmModeGetEncoder(fd, conn->encoder_id);
    uint32_t crtc = (enc && enc->crtc_id) ? enc->crtc_id : res->crtcs[0];
    if (enc) drmModeFreeEncoder(enc);
    drmModeCrtc *cur = drmModeGetCrtc(fd, crtc);
    *mode_out = (cur && cur->mode_valid) ? cur->mode : conn->modes[0];
    if (cur) drmModeFreeCrtc(cur);
    *conn_out = conn->connector_id;
    *crtc_out = crtc;
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    return 0;
}

int ghee_display_open(const char *dri_device, int *width, int *height) {
    const char *dev = dri_device ? dri_device : DEFAULT_DRI_DEVICE;
    const int fd = open(dev, O_RDWR | O_CLOEXEC);
    if (fd < 0) return -1;
    uint32_t conn, crtc;
    drmModeModeInfo mode;
    if (pick_display_mode(fd, &conn, &crtc, &mode)) {
        close(fd);
        return -1;
    }
    if (width) *width = mode.hdisplay;
    if (height) *height = mode.vdisplay;
    return fd;
}

int ghee_display_probe(const char *dri_device, int *width, int *height) {
    const int fd = ghee_display_open(dri_device, width, height);
    if (fd < 0) return -1;
    close(fd);
    return 0;
}

GheeScene *ghee_display_create(int width, int height, const char *dri_device, int rgb565, char **err) {
    return ghee_display_create_on(-1, width, height, dri_device, rgb565, err);
}

// Render to a KMS connector instead of reading back. width/height should match
// the connector mode (ghee_display_probe reports it). Needs DRM master.
// rgb565 renders/scans 16-bit color: half the tile-store + scanout bytes.
GheeScene *ghee_display_create_on(int card, int width, int height, const char *dri_device, int rgb565, char **err) {
    static char eb[ERROR_BUFFER_SIZE];
    GheeScene *g = create_impl(width, height, 4, dri_device, card, 1, rgb565, err);
    if (!g) return NULL;
    // Without master every page flip is refused, which used to show up as a
    // per-frame error and a display that never updates.
    if (drmSetMaster(g->drm_fd)) {
        snprintf(eb, sizeof(eb), "cannot take DRM master on %s: %s",
                 dri_device ? dri_device : DEFAULT_DRI_DEVICE, strerror(errno));
        if (err) *err = eb;
        ghee_destroy(g);
        return NULL;
    }

    if (pick_display_mode(g->drm_fd, &g->disp_conn, &g->disp_crtc, &g->disp_mode)) {
        if (err) *err = "no connected display";
        ghee_destroy(g);
        return NULL;
    }
    SAY(1, "  display  %s %ux%u@%u on connector %u\n",
            g->disp_mode.name, g->disp_mode.hdisplay, g->disp_mode.vdisplay,
            g->disp_mode.vrefresh, g->disp_conn);

    if (width != g->disp_mode.hdisplay || height != g->disp_mode.vdisplay) {
        // Every KMS driver has a primary plane, so its presence says nothing
        // about scaling. The only answer worth having is the kernel's, on this
        // card, at this size: put a buffer on the plane and see.
        g->disp_plane = primary_plane(g->drm_fd, g->disp_crtc);
        if (!g->disp_plane || !plane_scales(g, width, height)) {
            snprintf(eb, sizeof(eb),
                     "this display is fixed at %ux%u with no scaler, so %dx%d does not fit. "
                     "Render at %ux%u",
                     g->disp_mode.hdisplay, g->disp_mode.vdisplay, width, height,
                     g->disp_mode.hdisplay, g->disp_mode.vdisplay);
            if (err) *err = eb;
            ghee_destroy(g);
            return NULL;
        }
        SAY(1, "  scanout  scales %dx%d to %ux%u on plane %u\n",
                width, height, g->disp_mode.hdisplay, g->disp_mode.vdisplay, g->disp_plane);
    }

    // The scaling plane paces on the vblank, with no page flip. On one CPU
    // a flip's work runs on the drawing CPU wherever it is submitted.
    g->flipper = !g->disp_plane && sysconf(_SC_NPROCESSORS_ONLN) > 1;
    uint64_t seq, ns;
    g->jit = !g->disp_plane && drmCrtcGetSequence(g->drm_fd, g->disp_crtc, &seq, &ns) &&
             errno == EOPNOTSUPP;
    make_current(g);
    if (g->gpu_fd >= 0 && !lend_buffers(g, eb, sizeof eb)) { if (err) *err = eb; ghee_destroy(g); return NULL; }
    if (!ensure_blit(g)) { if (err) *err = "blit shader failed"; ghee_destroy(g); return NULL; }
    if (g->flipper && !start_flipper(g, eb, sizeof eb)) { if (err) *err = eb; ghee_destroy(g); return NULL; }
    g->display_mode = true;
    g->disp_first = true;
    return g;
}

// The kernel's fbdev emulation of a card, open, else -1.
static int open_fbdev(int drm_fd) {
    char dir[PATH_MAX + 16];
    if (!node_device(drm_fd, dir)) return -1;
    strcat(dir, "/graphics");
    DIR *d = opendir(dir);
    if (!d) return -1;
    int fd = -1;
    for (struct dirent *e; fd < 0 && (e = readdir(d));) {
        if (strncmp(e->d_name, "fb", 2)) continue;
        char node[300];
        snprintf(node, sizeof node, "/dev/%s", e->d_name);
        fd = open(node, O_RDWR | O_CLOEXEC);
        if (fd < 0) SAY(1, "  display  cannot open %s: %s\n", node, strerror(errno));
    }
    closedir(d);
    return fd;
}

static void zero_fbdev(int fb) {
    struct fb_fix_screeninfo fix;
    if (ioctl(fb, FBIOGET_FSCREENINFO, &fix)) return;
    static const char zeros[4096];
    for (size_t at = 0; at < fix.smem_len;) {
        size_t n = fix.smem_len - at < sizeof zeros ? fix.smem_len - at : sizeof zeros;
        ssize_t put = pwrite(fb, zeros, n, (off_t)at);
        if (put <= 0) return;
        at += (size_t)put;
    }
}

void ghee_destroy(GheeScene *g) {
    if (!g) return;
    // every GL delete below needs THIS instance's context current: with two live
    // renderers the wrong context would destroy the other one's objects
    const bool have_gl = g->display != EGL_NO_DISPLAY && g->context != EGL_NO_CONTEXT;
    if (have_gl) make_current(g);
    // The display ends dark. Removing its framebuffers turns its CRTC off. The
    // kernel lights an fbdev again at the card's last close, so the fbdev is
    // cleared before that close and blanked after it.
    int fbdev = -1;
    if (g->display_mode) {
        stop_flipper(g);   // reap any in-flight flip before tearing the CRTC down
        fbdev = open_fbdev(g->drm_fd);
        if (fbdev >= 0) zero_fbdev(fbdev);
        release_bo(g, g->disp_shown);
        release_bo(g, g->disp_prev);
        drmDropMaster(g->drm_fd);
    }
    if (have_gl) {
        PFNEGLDESTROYIMAGEKHRPROC destroy_image =
            (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
        for (int i = 0; i < g->lent_count; i++) {
            if (g->lent[i].fbo) glDeleteFramebuffers(1, &g->lent[i].fbo);
            if (g->lent[i].tex) glDeleteTextures(1, &g->lent[i].tex);
            if (g->lent[i].image != EGL_NO_IMAGE_KHR && destroy_image) destroy_image(g->display, g->lent[i].image);
        }
        if (g->lin_fbo) glDeleteFramebuffers(1, &g->lin_fbo);
        if (g->lin_tex) glDeleteTextures(1, &g->lin_tex);
        if (g->lin_image != EGL_NO_IMAGE_KHR && destroy_image) destroy_image(g->display, g->lin_image);
        if (g->preview_fbo) { glDeleteFramebuffers(1, &g->preview_fbo); glDeleteTextures(1, &g->preview_tex); }
        if (g->blit_prog) glDeleteProgram(g->blit_prog);
        if (g->fade_fbo) { glDeleteFramebuffers(1, &g->fade_fbo); glDeleteTextures(1, &g->fade_tex); }
        if (g->fade_prog) glDeleteProgram(g->fade_prog);
        if (g->VBO) glDeleteBuffers(1, &g->VBO);
        if (g->VAO) glDeleteVertexArrays(1, &g->VAO);
        ghee_mp_reset(g);   // all pass programs/buffers + the pending scene
        for (int i = 0; i < g->mp_nprog; i++) glDeleteProgram(g->mp_prog[i].program);
        if (g->quad_ibo) glDeleteBuffers(1, &g->quad_ibo);
        for (int a = 0; a < g->mp_nasset; a++) {
            mp_forget_dmabufs(g, &g->mp_asset[a]);
            glDeleteTextures(1, &g->mp_asset[a].tex);
        }
        eglMakeCurrent(g->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }
    if (g->display != EGL_NO_DISPLAY) {
        if (g->context != EGL_NO_CONTEXT) eglDestroyContext(g->display, g->context);
        if (g->surface != EGL_NO_SURFACE) eglDestroySurface(g->display, g->surface);
        eglTerminate(g->display);
    }
    free(g->rgba_scratch);
    free(g->kept);
    for (int a = 0; a < g->mp_nasset; a++) free(g->mp_asset[a].name);
    free(g->mp_asset);
    free(g->mp_batch);
    free(g->stage_batch);
    for (int i = 0; i < g->mp_nprog; i++) {
        free(g->mp_prog[i].vertex);
        free(g->mp_prog[i].fragment);
    }
    free(g->mp_prog);
    free(g->mp_order);
    for (int i = 0; i < g->lent_count; i++) gbm_bo_destroy(g->lent[i].bo);
    if (g->lin_bo) gbm_bo_destroy(g->lin_bo);
    if (g->gbm_surface) gbm_surface_destroy(g->gbm_surface);
    if (g->gbm) gbm_device_destroy(g->gbm);
    if (g->gpu_fd >= 0) close(g->gpu_fd);
    if (g->drm_fd >= 0) close(g->drm_fd);
    if (fbdev >= 0) {
        ioctl(fbdev, FBIOBLANK, FB_BLANK_POWERDOWN);
        close(fbdev);
    }
    free(g->last_error);
    free(g->read_error);
    free(g);
}

// ---- multipass -------------------------------------------------------------
// Build, commit and draw the passes of a scene, and their batches, arrays and
// assets.

// A tiler's driver (Mesa's v3d and vc4) keeps the draws into a framebuffer as
// one job, and submits the job when a read of its texture or a flush needs it.
// The job holds every buffer it draws with until then. A texture deleted under
// a job that nothing will read keeps that job, and its buffers, until a flush.
// A run with no display has no swap to flush them, so the flush comes first.
static void mp_free_buffers(GLuint fbo[2], GLuint tex[2]) {
    glFlush();
    glDeleteFramebuffers(2, fbo);
    glDeleteTextures(2, tex);
    fbo[0] = fbo[1] = tex[0] = tex[1] = 0;
}

// A pass with a quad starts opaque black, a pass of batches alone transparent.
static bool mp_make_buffers(GheeScene *g, GLuint fbo[2], GLuint tex[2], int w, int h,
                            bool alpha, bool quad) {
    const GLenum fmt = alpha ? GL_RGBA : GL_RGB;
    while (glGetError() != GL_NO_ERROR) { }
    glGenFramebuffers(2, fbo);
    glGenTextures(2, tex);
    for (int i = 0; i < 2; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo[i]);
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, fmt, w, h, 0, fmt, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex[i], 0);
        const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        const GLenum err = glGetError();
        if (status != GL_FRAMEBUFFER_COMPLETE || err != GL_NO_ERROR) {
            set_error(g, "%dx%d pass buffer failed: framebuffer 0x%X, GL 0x%X", w, h, status, err);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            mp_free_buffers(fbo, tex);
            return false;
        }
        glClearColor(0.0f, 0.0f, 0.0f, quad ? 1.0f : 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return true;
}

// Bound to a channel with no source pass, instead of texture unit 0, which
// holds whatever the last bind left there.
static GLuint make_black_texture(void) {
    const uint8_t px[4] = { 0, 0, 0, 255 };
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

// Sample both buffers of a pass with `filter`, a GHEE_FILTER_*. A mip chain
// fills its levels below 0 from what the buffers hold, so a read before the
// pass first renders finds a complete texture.
static void mp_sample_buffers(GLuint tex[2], int filter) {
    const GLint mag = filter == GHEE_FILTER_NEAREST ? GL_NEAREST : GL_LINEAR;
    const GLint min = filter == GHEE_FILTER_MIPMAP ? GL_LINEAR_MIPMAP_LINEAR : mag;
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag);
        if (filter == GHEE_FILTER_MIPMAP) glGenerateMipmap(GL_TEXTURE_2D);
    }
}

static void mp_free_uses(struct mp_locs *loc) {
    free(loc->uses);
    loc->uses = NULL;
    loc->nuses = loc->uses_cap = 0;
}

static void mp_free_values(struct mp_batch *l) {
    for (int k = 0; k < l->nvalues; k++) free(l->values[k].name);
    free(l->values);
    l->values = NULL;
    l->nvalues = l->values_cap = 0;
}

static void mp_free_batches(GheeScene *g, struct mp_batch *ls, int *n) {
    for (int i = 0; i < *n; i++) {
        program_drop(g, ls[i].program);
        mp_free_values(&ls[i]);
        if (ls[i].vbo) glDeleteBuffers(1, &ls[i].vbo);
        if (ls[i].spare) glDeleteBuffers(1, &ls[i].spare);
        free(ls[i].copy);
        if (ls[i].vao) glDeleteVertexArrays(1, &ls[i].vao);
        mp_free_uses(&ls[i].loc);
    }
    *n = 0;
}

// Free the running scene's passes, their batches and the array textures they share.
static void mp_free_passes(GheeScene *g) {
    mp_free_batches(g, g->mp_batch, &g->mp_nbatch);
    for (int p = 0; p < g->mp_count; p++) {
        if (g->mp_program[p]) glDeleteProgram(g->mp_program[p]);
        mp_free_buffers(g->mp_fbo[p], g->mp_tex[p]);
        mp_free_uses(&g->mp_loc[p]);
        g->mp_ndatatex[p] = 0;
    }
    if (g->mp_narray > 0) glDeleteTextures(g->mp_narray, g->mp_array_tex);
    for (int i = 0; i < g->mp_narray; i++) {
        free(g->mp_array_name[i]);
        g->mp_array_name[i] = NULL;
    }
    g->mp_narray = 0;
    g->mp_count = 0;
    if (g->black_tex) { glDeleteTextures(1, &g->black_tex); g->black_tex = 0; }
}

// Compile a pass into the PENDING scene: program and buffers are allocated
// beside the running scene, which keeps rendering until ghee_mp_commit. Returns
// the pass index (the commit keeps it) or -1.
int ghee_mp_add_pass(GheeScene *g, const char *name, const char *glsl_source, int alpha) {
    make_current(g);
    if (g->stage_count >= MP_MAX_PASSES) {
        set_error(g, "%s is pass %d, over the cap of %d passes", name ? name : "a pass",
                  g->stage_count + 1, MP_MAX_PASSES);
        return -1;
    }
    GLuint prog = 0;
    if (glsl_source && !(prog = create_shader_program(g, glsl_source))) return -1;
    const int idx = g->stage_count;
    if (!mp_make_buffers(g, g->stage_fbo[idx], g->stage_tex[idx], g->width, g->height,
                         alpha, prog != 0)) {
        if (prog) glDeleteProgram(prog);
        return -1;
    }
    snprintf(g->stage_name[idx], sizeof g->stage_name[idx], "%s", name ? name : "a pass");
    g->stage_alpha[idx] = alpha;
    g->stage_filter[idx] = GHEE_FILTER_LINEAR;
    g->stage_from[idx] = -1;
    g->stage_program[idx] = prog;
    g->stage_w[idx] = g->width;
    g->stage_h[idx] = g->height;
    g->stage_count++;
    return idx;
}

// Whether pending pass `pass` can hold a mip chain at w x h, with the error
// set when it cannot.
static bool mips_fit(GheeScene *g, int pass, int w, int h) {
    if (g->has_npot || (!(w & (w - 1)) && !(h & (h - 1)))) return true;
    set_error(g, "%s samples trilinearly at %dx%d, and this GPU builds a mip chain at a "
              "power-of-two size alone", g->stage_name[pass], w, h);
    return false;
}

// Give a pending pass a buffer size of its own, in place of the renderer's.
// Call it after ghee_mp_add_pass and before ghee_mp_commit. The last pass
// keeps the renderer's size, since the readback and the scanout take its
// buffer, and Renderer::setup refuses a sized last pass before it gets here.
bool ghee_mp_set_pass_size(GheeScene *g, int pass, int w, int h) {
    if (pass < 0 || pass >= g->stage_count || w <= 0 || h <= 0) {
        set_error(g, "pass %d is not pending, or %dx%d is not a positive size", pass, w, h);
        return false;
    }
    if (g->stage_filter[pass] == GHEE_FILTER_MIPMAP && !mips_fit(g, pass, w, h)) return false;
    make_current(g);
    GLuint fbo[2], tex[2];
    if (!mp_make_buffers(g, fbo, tex, w, h, g->stage_alpha[pass], g->stage_program[pass] != 0))
        return false;
    mp_free_buffers(g->stage_fbo[pass], g->stage_tex[pass]);
    memcpy(g->stage_fbo[pass], fbo, sizeof fbo);
    memcpy(g->stage_tex[pass], tex, sizeof tex);
    g->stage_w[pass] = w;
    g->stage_h[pass] = h;
    mp_sample_buffers(tex, g->stage_filter[pass]);
    return true;
}

bool ghee_mp_set_pass_sampling(GheeScene *g, int pass, int filter) {
    if (pass < 0 || pass >= g->stage_count || filter < GHEE_FILTER_NEAREST
        || filter > GHEE_FILTER_MIPMAP) {
        set_error(g, "pass %d is not pending, or %d is not a GHEE_FILTER", pass, filter);
        return false;
    }
    if (filter == GHEE_FILTER_MIPMAP && !mips_fit(g, pass, g->stage_w[pass], g->stage_h[pass]))
        return false;
    make_current(g);
    mp_sample_buffers(g->stage_tex[pass], filter);
    g->stage_filter[pass] = filter;
    return true;
}

bool ghee_mp_carry(GheeScene *g, int pass, int from) {
    if (pass < 0 || pass >= g->stage_count || from < 0 || from >= g->mp_count) {
        set_error(g, "pass %d is not pending, or pass %d is not running", pass, from);
        return false;
    }
    g->stage_from[pass] = from;
    return true;
}

int ghee_mp_add_batch(GheeScene *g, int pass, const char *name, const char *vertex_source,
                      const char *fragment_source, int blend) {
    if (pass < 0 || pass >= g->stage_count) { set_error(g, "pass %d is not pending", pass); return -1; }
    if (blend < GHEE_BLEND_NORMAL || blend > GHEE_BLEND_SCREEN) { set_error(g, "no blend mode %d", blend); return -1; }
    if (g->stage_nbatch == g->stage_batch_cap) {
        const int cap = g->stage_batch_cap ? 2 * g->stage_batch_cap : 16;
        struct mp_batch *grown = realloc(g->stage_batch, (size_t)cap * sizeof *grown);
        if (!grown) { set_error(g, "no memory for batch %d", g->stage_nbatch); return -1; }
        g->stage_batch = grown;
        g->stage_batch_cap = cap;
    }
    make_current(g);
    GLuint prog = 0;
    if (vertex_source || fragment_source) {
        prog = program_take(g, vertex_source, fragment_source);
        if (!prog) return -1;
    }
    struct mp_batch *l = &g->stage_batch[g->stage_nbatch];
    memset(l, 0, sizeof *l);
    l->id = ++g->batch_ids;
    l->pass = pass;
    snprintf(l->name, sizeof l->name, "%s", name);
    l->program = prog;
    l->blend = blend;
    l->opacity = 1.0f;
    glGenBuffers(1, &l->vbo);
    glGenVertexArrays(1, &l->vao);
    return g->stage_nbatch++;
}

// Drop the pending passes (a scene that failed to compile).
void ghee_mp_abort(GheeScene *g) {
    if (g->stage_count <= 0) return;
    make_current(g);
    mp_free_batches(g, g->stage_batch, &g->stage_nbatch);
    for (int p = 0; p < g->stage_count; p++) {
        if (g->stage_program[p]) glDeleteProgram(g->stage_program[p]);
        mp_free_buffers(g->stage_fbo[p], g->stage_tex[p]);
    }
    g->stage_count = 0;
}

// "<name><suffix>", allocated, or NULL with the error set.
static char *suffixed(GheeScene *g, const char *name, const char *suffix) {
    const size_t n = strlen(name) + strlen(suffix) + 1;
    char *out = malloc(n);
    if (!out) { set_error(g, "no memory for the name %s%s", name, suffix); return NULL; }
    snprintf(out, n, "%s%s", name, suffix);
    return out;
}

// Append `use` to the uses of a program. false with the error set when memory
// runs out.
static bool mp_add_use(GheeScene *g, struct mp_locs *loc, struct mp_use use) {
    if (loc->nuses == loc->uses_cap) {
        const int cap = loc->uses_cap ? 2 * loc->uses_cap : 4;
        struct mp_use *grown = realloc(loc->uses, (size_t)cap * sizeof *grown);
        if (!grown) { set_error(g, "no memory for the assets of a program"); return false; }
        loc->uses = grown;
        loc->uses_cap = cap;
    }
    loc->uses[loc->nuses++] = use;
    return true;
}

// Add asset `a` to the uses of program `prog` when the program reads its
// sampler or its size. false with the error set when memory runs out.
static bool mp_use_asset(GheeScene *g, GLuint prog, int a, struct mp_locs *loc) {
    if (!prog) return true;
    char *tex_name = suffixed(g, g->mp_asset[a].name, "_tex");
    char *size_name = suffixed(g, g->mp_asset[a].name, "_size");
    if (!tex_name || !size_name) {
        free(tex_name);
        free(size_name);
        return false;
    }
    const GLint at = glGetUniformLocation(prog, tex_name);
    const GLint size = glGetUniformLocation(prog, size_name);
    free(tex_name);
    free(size_name);
    if (at < 0 && size < 0) return true;
    return mp_add_use(g, loc, (struct mp_use){ a, at, size });
}

// Drop asset `a` from the uses of a program, and renumber the assets after
// it, which move down one.
static void mp_forget_asset(struct mp_locs *loc, int a) {
    int k = 0;
    for (int i = 0; i < loc->nuses; i++) {
        if (loc->uses[i].asset == a) continue;
        loc->uses[k] = loc->uses[i];
        if (loc->uses[k].asset > a) loc->uses[k].asset--;
        k++;
    }
    loc->nuses = k;
}

// Resolve the locations of a committed program that draws into a w x h buffer,
// and set its iResolution, its ghee_pixel, its iOutputResolution, and its
// ghee_vflip and ghee_yflip for a pass's own buffer.
// Program 0, a pass with no quad, has no uniforms.
static void mp_resolve(GheeScene *g, GLuint prog, int w, int h, struct mp_locs *loc) {
    loc->res = loc->pixel = loc->yflip = loc->time = loc->dt = loc->frame = loc->chan_res =
        loc->vflip = loc->visible = -1;
    for (int c = 0; c < MP_MAX_CHANNELS; c++) loc->chan[c] = -1;
    for (int i = 0; i < MP_MAX_DATATEX; i++) loc->data[i] = -1;
    loc->uses = NULL;
    loc->nuses = loc->uses_cap = 0;
    if (!prog) return;
    glUseProgram(prog);
    loc->res = glGetUniformLocation(prog, "iResolution");
    if (loc->res >= 0) glUniform3f(loc->res, (float)w, (float)h, 1.0f);
    loc->pixel = glGetUniformLocation(prog, "ghee_pixel");
    if (loc->pixel >= 0) glUniform2f(loc->pixel, 1.0f / (float)w, 1.0f / (float)h);
    const GLint out_res = glGetUniformLocation(prog, "iOutputResolution");
    if (out_res >= 0) glUniform3f(out_res, (float)g->width, (float)g->height, 1.0f);
    loc->time = glGetUniformLocation(prog, "iTime");
    loc->dt = glGetUniformLocation(prog, "iTimeDelta");
    loc->frame = glGetUniformLocation(prog, "iFrame");
    loc->chan_res = glGetUniformLocation(prog, "iChannelResolution");
    for (int c = 0; c < MP_MAX_CHANNELS; c++) {
        char nm[16];
        snprintf(nm, sizeof(nm), "iChannel%d", c);
        loc->chan[c] = glGetUniformLocation(prog, nm);
    }
    for (int a = 0; a < g->mp_nasset; a++) mp_use_asset(g, prog, a, loc);
    loc->vflip = glGetUniformLocation(prog, "ghee_vflip");
    if (loc->vflip >= 0) glUniform1f(loc->vflip, 1.0f);
    loc->yflip = glGetUniformLocation(prog, "ghee_yflip");
    if (loc->yflip >= 0) glUniform2f(loc->yflip, 1.0f, 0.25f);
    loc->visible = glGetUniformLocation(prog, VISIBLE_UNIFORM);
    g->bind_epoch++;
}

static void lin_restore(GheeScene *g);

// Make the pending passes the scene: the running one is freed and replaced in a
// single step, all but the buffers that ghee_mp_carry hands on. Returns the
// pass count, -1 if nothing was pending (in which case the running scene is
// left alone).
int ghee_mp_commit(GheeScene *g) {
    if (g->stage_count <= 0) { set_error(g, "no pending passes to commit"); return -1; }
    make_current(g);
    lin_restore(g);
    g->changed = true;
    int write[MP_MAX_PASSES] = { 0 };
    for (int p = 0; p < g->stage_count; p++) {
        const int q = g->stage_from[p];
        if (q < 0 || q >= g->mp_count || !g->mp_tex[q][0] || g->mp_w[q] != g->stage_w[p]
            || g->mp_h[q] != g->stage_h[p] || g->mp_alpha[q] != g->stage_alpha[p])
            continue;
        mp_free_buffers(g->stage_fbo[p], g->stage_tex[p]);
        memcpy(g->stage_fbo[p], g->mp_fbo[q], sizeof g->stage_fbo[p]);
        memcpy(g->stage_tex[p], g->mp_tex[q], sizeof g->stage_tex[p]);
        if (g->mp_filter[q] != g->stage_filter[p])
            mp_sample_buffers(g->stage_tex[p], g->stage_filter[p]);
        memset(g->mp_fbo[q], 0, sizeof g->mp_fbo[q]);
        memset(g->mp_tex[q], 0, sizeof g->mp_tex[q]);
        write[p] = g->mp_write[q];
    }
    mp_free_passes(g);
    g->black_tex = make_black_texture();
    for (int p = 0; p < g->stage_count; p++) {
        const GLuint prog = g->stage_program[p];
        g->mp_program[p] = prog;
        memcpy(g->mp_fbo[p], g->stage_fbo[p], sizeof g->mp_fbo[p]);
        memcpy(g->mp_tex[p], g->stage_tex[p], sizeof g->mp_tex[p]);
        g->mp_w[p] = g->stage_w[p];
        g->mp_h[p] = g->stage_h[p];
        memcpy(g->mp_name[p], g->stage_name[p], sizeof g->mp_name[p]);
        g->mp_alpha[p] = g->stage_alpha[p];
        g->mp_filter[p] = g->stage_filter[p];
        g->mp_write[p] = write[p];
        g->mp_nchan[p] = 0;
        g->mp_ndatatex[p] = 0;
        for (int c = 0; c < MP_MAX_CHANNELS; c++) g->mp_chan_src[p][c] = -1;
        mp_resolve(g, prog, g->mp_w[p], g->mp_h[p], &g->mp_loc[p]);
    }
    g->mp_count = g->stage_count;
    g->stage_count = 0;
    // The staged batches become the scene's, and the old table takes the next
    // staged ones.
    struct mp_batch *was = g->mp_batch;
    const int was_cap = g->mp_batch_cap;
    g->mp_batch = g->stage_batch;
    g->mp_batch_cap = g->stage_batch_cap;
    g->stage_batch = was;
    g->stage_batch_cap = was_cap;
    g->mp_nbatch = g->stage_nbatch;
    g->stage_nbatch = 0;
    if (g->mp_order_cap < g->mp_batch_cap) {
        int *grown = realloc(g->mp_order, (size_t)g->mp_batch_cap * sizeof *grown);
        if (!grown) {
            set_error(g, "no memory for the order of %d batches", g->mp_nbatch);
            mp_free_batches(g, g->mp_batch, &g->mp_nbatch);
        } else {
            g->mp_order = grown;
            g->mp_order_cap = g->mp_batch_cap;
        }
    }
    for (int i = 0; i < g->mp_nbatch; i++) {
        struct mp_batch *l = &g->mp_batch[i];
        mp_resolve(g, l->program, g->mp_w[l->pass], g->mp_h[l->pass], &l->loc);
        l->quads = 0;
        g->mp_order[i] = i;
    }
    return g->mp_count;
}

static bool make_quad_indices(GheeScene *g) {
    GLushort *ix = malloc((size_t)QUAD_CHUNK * 6 * sizeof *ix);
    if (!ix) { set_error(g, "no memory for the quad indices"); return false; }
    for (int q = 0; q < QUAD_CHUNK; q++) {
        const GLushort v = (GLushort)(q * 4);
        GLushort *o = ix + (size_t)q * 6;
        o[0] = v; o[1] = v + 1; o[2] = v + 2;
        o[3] = v + 2; o[4] = v + 1; o[5] = v + 3;
    }
    glGenBuffers(1, &g->quad_ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g->quad_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)QUAD_CHUNK * 6 * sizeof *ix, ix, GL_STATIC_DRAW);
    free(ix);
    return true;
}

// Whether batch l lays any pixel on its pass.
static bool batch_draws(const struct mp_batch *l) {
    return l->pass >= 0 && l->quads > 0 && l->program && l->opacity > 0.0f && !l->hidden;
}

// Committed batch `batch`, or NULL with the error set for an index that
// holds none: out of range, or a batch that ghee_mp_remove_batch took out.
static struct mp_batch *running_batch(GheeScene *g, int batch) {
    if (batch < 0 || batch >= g->mp_nbatch || g->mp_batch[batch].pass < 0) {
        set_error(g, "no batch %d in the scene", batch);
        return NULL;
    }
    return &g->mp_batch[batch];
}

bool ghee_mp_set_batch_quads(GheeScene *g, int batch, const float *data, int quads, int nattr) {
    struct mp_batch *l = running_batch(g, batch);
    if (!l) return false;
    if (quads < 0 || nattr < 1 || nattr > MP_BATCH_ATTRIBS) {
        set_error(g, "%s: %d quads of %d attributes a vertex, where a vertex holds 1 to %d",
                  l->name, quads, nattr, MP_BATCH_ATTRIBS);
        return false;
    }
    // A size_t product wraps on a 32-bit target.
    const size_t quad_bytes = 4 * (size_t)nattr * 4 * sizeof(GLfloat);
    if ((size_t)quads > (size_t)PTRDIFF_MAX / quad_bytes) {
        set_error(g, "%s: %d quads overflow a buffer", l->name, quads);
        return false;
    }
    const size_t bytes = (size_t)quads * quad_bytes;
    const bool drew = batch_draws(l);
    float *copy = bytes ? realloc(l->copy, bytes) : NULL;
    if (bytes && !copy) {
        set_error(g, "%s: no memory for a copy of %d quads", l->name, quads);
        return false;
    }
    if (!bytes) free(l->copy);
    l->copy = copy;
    if (bytes) memcpy(copy, data, bytes);
    l->stale_first = 0;
    l->stale_quads = quads;
    make_current(g);
    glBindVertexArray(l->vao);
    if (!g->quad_ibo && !make_quad_indices(g)) { glBindVertexArray(g->VAO); return false; }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g->quad_ibo);
    for (int k = 0; k < MP_BATCH_ATTRIBS; k++) {
        if (k < nattr) glEnableVertexAttribArray(k);
        else glDisableVertexAttribArray(k);
    }
    glBindBuffer(GL_ARRAY_BUFFER, l->vbo);
    while (glGetError() != GL_NO_ERROR) { }
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)bytes, data, GL_DYNAMIC_DRAW);
    const GLenum err = glGetError();
    glBindVertexArray(g->VAO);
    glBindBuffer(GL_ARRAY_BUFFER, g->VBO);
    if (err != GL_NO_ERROR) {
        l->quads = 0;
        g->changed = true;
        set_error(g, "%s: the upload of %d quads failed: GL 0x%X", l->name, quads, err);
        return false;
    }
    l->quads = quads;
    l->nattr = nattr;
    if (drew || batch_draws(l)) g->changed = true;
    return true;
}

int ghee_mp_batch_quads(GheeScene *g, int batch, int nattr) {
    struct mp_batch *l = running_batch(g, batch);
    if (!l || l->nattr != nattr) return -1;
    return l->quads;
}

bool ghee_mp_set_batch_quads_at(GheeScene *g, int batch, int first, const float *data, int quads) {
    struct mp_batch *l = running_batch(g, batch);
    if (!l) return false;
    if (first < 0 || quads < 1 || first > l->quads - quads) {
        set_error(g, "%s: quads %d to %d, where the batch holds %d", l->name, first,
                  first + quads, l->quads);
        return false;
    }
    const size_t quad_floats = 4 * (size_t)l->nattr * 4;
    const size_t quad_bytes = quad_floats * sizeof(GLfloat);
    memcpy(l->copy + (size_t)first * quad_floats, data, (size_t)quads * quad_bytes);
    make_current(g);
    while (glGetError() != GL_NO_ERROR) { }
    const bool whole = first == 0 && quads == l->quads;
    const size_t bytes = (size_t)l->quads * quad_bytes;
    if (whole) {
        // A write of the whole buffer takes a fresh one from the driver.
        glBindBuffer(GL_ARRAY_BUFFER, l->vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)bytes, l->copy);
    } else {
        if (!l->spare) glGenBuffers(1, &l->spare);
        glBindBuffer(GL_ARRAY_BUFFER, l->spare);
        if (l->spare_bytes != bytes) {
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)bytes, l->copy, GL_DYNAMIC_DRAW);
        } else {
            int lo = first, hi = first + quads;
            if (l->stale_quads > 0) {
                if (l->stale_first < lo) lo = l->stale_first;
                if (l->stale_first + l->stale_quads > hi) hi = l->stale_first + l->stale_quads;
            }
            glBufferSubData(GL_ARRAY_BUFFER, (GLintptr)((size_t)lo * quad_bytes),
                            (GLsizeiptr)((size_t)(hi - lo) * quad_bytes), l->copy + (size_t)lo * quad_floats);
        }
    }
    const GLenum err = glGetError();
    glBindBuffer(GL_ARRAY_BUFFER, g->VBO);
    if (err != GL_NO_ERROR) {
        // The batch draws from `vbo` as it was, and the next write in part
        // gives `spare` all of `copy`.
        l->spare_bytes = 0;
        set_error(g, "%s: the upload of %d quads failed: GL 0x%X", l->name, quads, err);
        return false;
    }
    if (whole) {
        l->stale_first = 0;
        l->stale_quads = quads;
    } else {
        const GLuint drawn = l->vbo;
        l->vbo = l->spare;
        l->spare = drawn;
        l->spare_bytes = bytes;
        l->stale_first = first;
        l->stale_quads = quads;
    }
    if (batch_draws(l)) g->changed = true;
    return true;
}

// assets sampled by one program, one texture unit each
static int mp_assets(const struct mp_locs *loc) {
    int n = 0;
    for (int i = 0; i < loc->nuses; i++)
        if (loc->uses[i].loc >= 0) n++;
    return n;
}

// Whether the program named `name` fits in `want` texture units, with the
// error set when it does not.
static bool units_fit(GheeScene *g, const char *name, int want) {
    if (want <= g->max_units) return true;
    if (g->gl_units > MP_MAX_UNITS)
        set_error(g, "%s needs %d texture units, a pass binds at most %d", name, want, (int)g->max_units);
    else
        set_error(g, "%s needs %d texture units, this GPU has %d", name, want, (int)g->max_units);
    return false;
}

// Whether pass p fits its texture units with nchan channels and ndata arrays:
// its own program and each of its batches, every one with the assets it
// samples. A batch shares the pass's channel and array units.
static bool mp_pass_fits(GheeScene *g, int p, int nchan, int ndata) {
    if (!units_fit(g, g->mp_name[p], nchan + ndata + mp_assets(&g->mp_loc[p]))) return false;
    for (int l = 0; l < g->mp_nbatch; l++) {
        const struct mp_batch *ly = &g->mp_batch[l];
        if (ly->pass == p && !units_fit(g, ly->name, nchan + ndata + mp_assets(&ly->loc)))
            return false;
    }
    return true;
}

bool ghee_mp_set_channel(GheeScene *g, int pass, int channel, int src_pass) {
    if (pass < 0 || pass >= g->mp_count || channel < 0 || channel >= MP_MAX_CHANNELS) {
        set_error(g, "no channel %d on pass %d", channel, pass);
        return false;
    }
    if (src_pass < -1 || src_pass >= g->mp_count) {   // -1 = unbound channel
        set_error(g, "channel %d on pass %d: no source pass %d", channel, pass, src_pass);
        return false;
    }
    const int nchan = (channel + 1 > g->mp_nchan[pass]) ? channel + 1 : g->mp_nchan[pass];
    if (!mp_pass_fits(g, pass, nchan, g->mp_ndatatex[pass])) return false;
    lin_restore(g);
    g->mp_chan_src[pass][channel] = src_pass;
    g->mp_nchan[pass] = nchan;
    g->changed = true;
    g->bind_epoch++;
    return true;
}

// Set `param` on `program`, which is in use after. false when the program
// does not read it. A value other than the one that the program holds marks
// the scene changed.
static bool program_uniform(GheeScene *g, GLuint program, const parameter_t *param) {
    if (!program) return false;
    glUseProgram(program);
    const GLint loc = glGetUniformLocation(program, param->name);
    if (loc < 0) return false;
    // A location reads back a mat4 at most.
    union { GLfloat f[16]; GLint i[16]; } held;
    memset(&held, 0xff, sizeof held);
    if (param->type == PARAM_TYPE_INT) {
        const GLint v = param->value.int_param->value;
        glGetUniformiv(program, loc, held.i);
        if (held.i[0] != v) g->changed = true;
        glUniform1i(loc, v);
        return true;
    }
    int n;
    const float *v;
    switch (param->type) {
        case PARAM_TYPE_FLOAT: n = 1; v = &param->value.float_param->value; break;
        case PARAM_TYPE_VEC2: n = 2; v = param->value.vec_param->value; break;
        case PARAM_TYPE_VEC3: n = 3; v = param->value.vec_param->value; break;
        case PARAM_TYPE_VEC4: n = 4; v = param->value.vec_param->value; break;
        default: return false;
    }
    glGetUniformfv(program, loc, held.f);
    if (memcmp(held.f, v, (size_t)n * sizeof *v) != 0) g->changed = true;
    switch (n) {
        case 1: glUniform1f(loc, v[0]); break;
        case 2: glUniform2f(loc, v[0], v[1]); break;
        case 3: glUniform3f(loc, v[0], v[1], v[2]); break;
        default: glUniform4f(loc, v[0], v[1], v[2], v[3]); break;
    }
    return true;
}

// The value reaches the pass's program and every batch of the pass.
bool ghee_mp_set_uniform(GheeScene *g, int pass, const parameter_t *param) {
    if (pass < 0 || pass >= g->mp_count) return false;
    make_current(g);
    bool any = program_uniform(g, g->mp_program[pass], param);
    for (int l = 0; l < g->mp_nbatch; l++)
        if (g->mp_batch[l].pass == pass) any = program_uniform(g, g->mp_batch[l].program, param) || any;
    return any;
}

bool ghee_mp_set_pass_uniform(GheeScene *g, int pass, const parameter_t *param) {
    if (pass < 0 || pass >= g->mp_count) return false;
    make_current(g);
    return program_uniform(g, g->mp_program[pass], param);
}

// Find a declared array by its "<name>_tex" sampler name. -1 = not declared.
static int mp_array_index(GheeScene *g, const char *uni) {
    for (int i = 0; i < g->mp_narray; i++)
        if (strcmp(g->mp_array_name[i], uni) == 0) return i;
    return -1;
}

// The index of the array whose sampler is named "<name>_tex", -1 when no pass
// declares it or memory runs out, with the error set either way.
static int mp_array_named(GheeScene *g, const char *name) {
    char *uni = suffixed(g, name, "_tex");
    if (!uni) return -1;
    const int idx = mp_array_index(g, uni);
    free(uni);
    if (idx < 0) set_error(g, "array '%s' is declared by no pass", name);
    return idx;
}

// Declare an array NAME on a pass. One texture per name at scene scope, bound
// into this pass's next free unit (data textures follow the channels), so the
// data uploads once a frame however many passes read it. Returns the array
// index, -1 on refusal. Call after ghee_mp_commit, before the first render.
int ghee_mp_declare_array(GheeScene *g, int pass, const char *name) {
    if (pass < 0 || pass >= g->mp_count) { set_error(g, "no pass %d to declare '%s' on", pass, name); return -1; }
    char *uni = suffixed(g, name, "_tex");
    if (!uni) return -1;
    int idx = mp_array_index(g, uni);
    for (int i = 0; idx >= 0 && i < g->mp_ndatatex[pass]; i++)
        if (g->mp_datatex[pass][i] == idx) { free(uni); return idx; }
    if (!mp_pass_fits(g, pass, g->mp_nchan[pass], g->mp_ndatatex[pass] + 1)) {
        free(uni);
        return -1;
    }
    // The texture units run out before either table does: a pass binds
    // MP_MAX_UNITS textures at most, and a scene holds MP_MAX_PASSES passes.
    if (g->mp_ndatatex[pass] >= MP_MAX_DATATEX) {
        set_error(g, "%s declares more than %d arrays", g->mp_name[pass], MP_MAX_DATATEX);
        free(uni);
        return -1;
    }
    if (idx < 0 && g->mp_narray >= MP_MAX_ARRAYS) {
        set_error(g, "more than %d arrays in the scene", MP_MAX_ARRAYS);
        free(uni);
        return -1;
    }
    if (idx < 0) {
        make_current(g);
        idx = g->mp_narray++;
        g->mp_array_name[idx] = uni;
        glGenTextures(1, &g->mp_array_tex[idx]);
        g->mp_array_size[idx] = 0;
    } else {
        free(uni);
    }
    if (g->mp_program[pass]) {
        glUseProgram(g->mp_program[pass]);
        g->mp_loc[pass].data[g->mp_ndatatex[pass]] =
            glGetUniformLocation(g->mp_program[pass], g->mp_array_name[idx]);
    }
    for (int l = 0; l < g->mp_nbatch; l++)
        if (g->mp_batch[l].pass == pass && g->mp_batch[l].program)
            g->mp_batch[l].loc.data[g->mp_ndatatex[pass]] =
                glGetUniformLocation(g->mp_batch[l].program, g->mp_array_name[idx]);
    g->mp_datatex[pass][g->mp_ndatatex[pass]++] = idx;
    g->changed = true;
    g->bind_epoch++;
    return idx;
}

// Bind one declared array's texture, and set the filters on a resize.
// -1 = no pass declares the name. *fresh means the texture needs a full
// glTexImage2D, and not a glTexSubImage2D.
static int mp_array_bind(GheeScene *g, const char *name, int count, int rows, bool *fresh) {
    const int idx = mp_array_named(g, name);
    if (idx < 0) return -1;
    make_current(g);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g->mp_array_tex[idx]);
    *fresh = (g->mp_array_size[idx] != count * rows);
    if (*fresh) {
        const GLint filt = ghee_data_texture_linear(g) ? GL_LINEAR : GL_NEAREST;
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    return idx;
}

// GLES2 pins a texture's texel type at the allocation, so every upload of one
// array carries the type that has_float_tex picked.
static void mp_array_upload(bool fresh, int count, int rows, GLenum type, const void *px) {
    if (fresh)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, count, rows, 0, GL_LUMINANCE, type, px);
    else
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, count, rows, GL_LUMINANCE, type, px);
}

// Upload one declared array as a count x rows texture named "<name>_tex", once
// for every pass that declared it. Float texels if the GL supports them, else
// 8-bit LUMINANCE (clamped to [0,1]). The units are bound in render_passes.
void ghee_mp_set_array(GheeScene *g, const char *name, const float *data, int count, int rows) {
    if (count <= 0 || rows <= 0) return;
    bool fresh = false;
    const int idx = mp_array_bind(g, name, count, rows, &fresh);
    if (idx < 0) return;
    g->changed = true;
    if (g->has_float_tex) {
        mp_array_upload(fresh, count, rows, GL_FLOAT, data);
    } else {
        uint8_t *b = malloc((size_t)count * rows);
        if (!b) return;
        for (int i = 0; i < count * rows; i++) {
            float v = data[i];
            b[i] = (uint8_t)(v <= 0.0f ? 0 : v >= 1.0f ? 255 : v * 255.0f + 0.5f);
        }
        mp_array_upload(fresh, count, rows, GL_UNSIGNED_BYTE, b);
        free(b);
    }
    g->mp_array_size[idx] = count * rows;
}

// ghee_mp_set_array from bytes, where the value is data[i] / 255. The 8-bit
// texture takes the bytes as they lie. A float texture takes one conversion.
void ghee_mp_set_array_u8(GheeScene *g, const char *name, const uint8_t *data, int count, int rows) {
    if (count <= 0 || rows <= 0) return;
    bool fresh = false;
    const int idx = mp_array_bind(g, name, count, rows, &fresh);
    if (idx < 0) return;
    g->changed = true;
    if (g->has_float_tex) {
        float *b = malloc((size_t)count * rows * sizeof(float));
        if (!b) return;
        for (int i = 0; i < count * rows; i++) b[i] = data[i] / 255.0f;
        mp_array_upload(fresh, count, rows, GL_FLOAT, b);
        free(b);
    } else {
        mp_array_upload(fresh, count, rows, GL_UNSIGNED_BYTE, data);
    }
    g->mp_array_size[idx] = count * rows;
}

// Find a set asset by its name. -1 = not set.
static int mp_asset_index(GheeScene *g, const char *name) {
    for (int i = 0; i < g->mp_nasset; i++)
        if (strcmp(g->mp_asset[i].name, name) == 0) return i;
    return -1;
}

// Drop asset `a` from every committed program, and renumber the assets after
// it in each.
static void mp_forget_everywhere(GheeScene *g, int a) {
    for (int p = 0; p < g->mp_count; p++) mp_forget_asset(&g->mp_loc[p], a);
    for (int l = 0; l < g->mp_nbatch; l++) mp_forget_asset(&g->mp_batch[l].loc, a);
}

bool ghee_mp_reads(GheeScene *g, const char *name) {
    make_current(g);
    for (int p = 0; p < g->mp_count; p++)
        if (g->mp_program[p] && glGetUniformLocation(g->mp_program[p], name) >= 0) return true;
    for (int l = 0; l < g->mp_nbatch; l++)
        if (g->mp_batch[l].program && glGetUniformLocation(g->mp_batch[l].program, name) >= 0)
            return true;
    return false;
}

// Bind a set asset to a committed batch under `name`, in place of the asset
// bound there before.
bool ghee_mp_bind_asset(GheeScene *g, int batch, const char *name, const char *asset) {
    struct mp_batch *l = running_batch(g, batch);
    if (!l) return false;
    if (!l->program) { set_error(g, "%s has no program", l->name); return false; }
    const int a = mp_asset_index(g, asset);
    if (a < 0) { set_error(g, "%s: asset '%s' is not set", l->name, asset); return false; }
    make_current(g);
    char *tex_name = suffixed(g, name, "_tex");
    char *size_name = suffixed(g, name, "_size");
    if (!tex_name || !size_name) {
        free(tex_name);
        free(size_name);
        return false;
    }
    const GLint at = glGetUniformLocation(l->program, tex_name);
    const GLint size = glGetUniformLocation(l->program, size_name);
    free(tex_name);
    free(size_name);
    if (at < 0 && size < 0) return true;
    struct mp_locs *loc = &l->loc;
    for (int i = 0; i < loc->nuses; i++) {
        struct mp_use *u = &loc->uses[i];
        if ((at >= 0 && u->loc == at) || (size >= 0 && u->size == size)) {
            *u = (struct mp_use){ a, at, size };
            g->changed = true;
            g->bind_epoch++;
            return true;
        }
    }
    g->changed = true;
    g->bind_epoch++;
    return mp_add_use(g, loc, (struct mp_use){ a, at, size });
}

// Whether batch l draws nothing: its program reads VISIBLE_UNIFORM, and the
// batch holds it below 0.5.
static bool batch_hidden(const struct mp_batch *l) {
    if (l->loc.visible < 0) return false;
    for (int k = 0; k < l->nvalues; k++)
        if (strcmp(l->values[k].name, VISIBLE_UNIFORM) == 0)
            return l->values[k].type == PARAM_TYPE_FLOAT && l->values[k].v[0] < 0.5f;
    return false;
}

bool ghee_mp_set_batch_program(GheeScene *g, int batch, const char *vertex_source,
                               const char *fragment_source) {
    struct mp_batch *l = running_batch(g, batch);
    if (!l) return false;
    const int p = l->pass;
    make_current(g);
    const GLuint prog = program_take(g, vertex_source, fragment_source);
    if (!prog) return false;
    const GLuint was = l->program;
    struct mp_locs old = l->loc;
    l->program = prog;
    mp_resolve(g, prog, g->mp_w[p], g->mp_h[p], &l->loc);
    for (int i = 0; i < g->mp_ndatatex[p]; i++)
        l->loc.data[i] = glGetUniformLocation(prog, g->mp_array_name[g->mp_datatex[p][i]]);
    if (!mp_pass_fits(g, p, g->mp_nchan[p], g->mp_ndatatex[p])) {
        mp_free_uses(&l->loc);
        program_drop(g, prog);
        l->program = was;
        l->loc = old;
        return false;
    }
    mp_free_uses(&old);
    program_drop(g, was);
    l->dirty = true;
    l->hidden = batch_hidden(l);
    g->changed = true;
    return true;
}

bool ghee_mp_keep_program(GheeScene *g, const char *vertex_source, const char *fragment_source) {
    make_current(g);
    return program_take(g, vertex_source, fragment_source) != 0;
}

void ghee_mp_let_go_program(GheeScene *g, const char *vertex_source, const char *fragment_source) {
    for (int i = 0; i < g->mp_nprog; i++) {
        const struct mp_prog *p = &g->mp_prog[i];
        if (strcmp(p->fragment, fragment_source) == 0 && strcmp(p->vertex, vertex_source) == 0) {
            make_current(g);
            program_drop(g, p->program);
            return;
        }
    }
}

bool ghee_mp_set_batch_uniform(GheeScene *g, int batch, const parameter_t *param) {
    struct mp_batch *l = running_batch(g, batch);
    if (!l) return false;
    struct mp_value *v = NULL;
    for (int k = 0; k < l->nvalues && !v; k++)
        if (strcmp(l->values[k].name, param->name) == 0) v = &l->values[k];
    const bool fresh = !v;
    if (!v) {
        if (l->nvalues == l->values_cap) {
            const int cap = l->values_cap ? 2 * l->values_cap : 16;
            struct mp_value *grown = realloc(l->values, (size_t)cap * sizeof *grown);
            if (!grown) { set_error(g, "%s: no memory for its uniform %s", l->name, param->name); return false; }
            l->values = grown;
            l->values_cap = cap;
        }
        char *name = suffixed(g, param->name, "");
        if (!name) return false;
        v = &l->values[l->nvalues++];
        memset(v, 0, sizeof *v);
        v->name = name;
    }
    const struct mp_value was = *v;
    v->type = param->type;
    switch (param->type) {
        case PARAM_TYPE_FLOAT: v->v[0] = param->value.float_param->value; break;
        case PARAM_TYPE_INT:   v->i = param->value.int_param->value; break;
        default: memcpy(v->v, param->value.vec_param->value, sizeof v->v); break;
    }
    if (!fresh && was.type == v->type && was.i == v->i && memcmp(was.v, v->v, sizeof v->v) == 0)
        return true;
    const bool drew = batch_draws(l);
    if (strcmp(v->name, VISIBLE_UNIFORM) == 0) l->hidden = batch_hidden(l);
    l->dirty = true;
    if (drew || batch_draws(l)) g->changed = true;
    return true;
}

bool ghee_mp_set_batch_opacity(GheeScene *g, int batch, float opacity) {
    struct mp_batch *l = running_batch(g, batch);
    if (!l) return false;
    const float o = opacity < 0.0f ? 0.0f : opacity > 1.0f ? 1.0f : opacity;
    const bool drew = batch_draws(l);
    const bool same = memcmp(&o, &l->opacity, sizeof o) == 0;
    l->opacity = o;
    if (!same && (drew || batch_draws(l))) g->changed = true;
    return true;
}

int ghee_mp_insert_batch(GheeScene *g, int pass, int after, const char *name, const char *vertex_source,
                         const char *fragment_source, int blend) {
    if (pass < 0 || pass >= g->mp_count) { set_error(g, "no pass %d in the scene", pass); return -1; }
    if (after >= g->mp_nbatch || (after >= 0 && g->mp_batch[after].pass != pass)) {
        set_error(g, "batch %d is not a batch of pass %d", after, pass);
        return -1;
    }
    if (blend < GHEE_BLEND_NORMAL || blend > GHEE_BLEND_SCREEN) { set_error(g, "no blend mode %d", blend); return -1; }
    // The slot of a batch that ghee_mp_remove_batch took out goes to the
    // next batch, so a scene that removes and inserts batches holds no more
    // slots than batches at once.
    int idx = -1;
    for (int i = 0; i < g->mp_nbatch && idx < 0; i++)
        if (g->mp_batch[i].pass < 0) idx = i;
    const bool fresh = idx < 0;
    if (fresh && g->mp_nbatch == g->mp_batch_cap) {
        const int cap = g->mp_batch_cap ? 2 * g->mp_batch_cap : 16;
        struct mp_batch *grown = realloc(g->mp_batch, (size_t)cap * sizeof *grown);
        if (!grown) { set_error(g, "no memory for batch %d", g->mp_nbatch); return -1; }
        g->mp_batch = grown;
        g->mp_batch_cap = cap;
    }
    if (g->mp_order_cap < g->mp_batch_cap) {
        int *grown = realloc(g->mp_order, (size_t)g->mp_batch_cap * sizeof *grown);
        if (!grown) { set_error(g, "no memory for the order of batch %d", g->mp_nbatch); return -1; }
        g->mp_order = grown;
        g->mp_order_cap = g->mp_batch_cap;
    }
    make_current(g);
    GLuint prog = 0;
    if (vertex_source || fragment_source) {
        prog = program_take(g, vertex_source, fragment_source);
        if (!prog) return -1;
    }
    if (fresh) idx = g->mp_nbatch++;
    struct mp_batch *l = &g->mp_batch[idx];
    memset(l, 0, sizeof *l);
    l->id = ++g->batch_ids;
    l->pass = pass;
    snprintf(l->name, sizeof l->name, "%s", name);
    l->program = prog;
    l->blend = blend;
    l->opacity = 1.0f;
    mp_resolve(g, prog, g->mp_w[pass], g->mp_h[pass], &l->loc);
    for (int i = 0; prog && i < g->mp_ndatatex[pass]; i++)
        l->loc.data[i] = glGetUniformLocation(prog, g->mp_array_name[g->mp_datatex[pass][i]]);
    if (!mp_pass_fits(g, pass, g->mp_nchan[pass], g->mp_ndatatex[pass])) {
        mp_free_uses(&l->loc);
        program_drop(g, prog);
        memset(l, 0, sizeof *l);
        l->pass = -1;
        if (fresh) g->mp_nbatch--;
        return -1;
    }
    glGenBuffers(1, &l->vbo);
    glGenVertexArrays(1, &l->vao);
    // The order holds every slot. A reused slot leaves its old place, and
    // the batch then draws right after `after`, or ahead of every batch of
    // its pass.
    const int others = g->mp_nbatch - 1;
    if (!fresh) {
        int k = 0;
        while (g->mp_order[k] != idx) k++;
        memmove(&g->mp_order[k], &g->mp_order[k + 1], (size_t)(others - k) * sizeof *g->mp_order);
    }
    int at = others;
    for (int k = 0; k < others; k++) {
        const int b = g->mp_order[k];
        if (after >= 0 ? b == after : g->mp_batch[b].pass == pass) {
            at = after >= 0 ? k + 1 : k;
            break;
        }
    }
    memmove(&g->mp_order[at + 1], &g->mp_order[at], (size_t)(others - at) * sizeof *g->mp_order);
    g->mp_order[at] = idx;
    g->changed = true;
    return idx;
}

bool ghee_mp_remove_batch(GheeScene *g, int batch) {
    struct mp_batch *l = running_batch(g, batch);
    if (!l) return false;
    make_current(g);
    program_drop(g, l->program);
    mp_free_values(l);
    if (l->vbo) glDeleteBuffers(1, &l->vbo);
    if (l->spare) glDeleteBuffers(1, &l->spare);
    free(l->copy);
    if (l->vao) glDeleteVertexArrays(1, &l->vao);
    mp_free_uses(&l->loc);
    memset(l, 0, sizeof *l);
    l->pass = -1;
    g->changed = true;
    return true;
}

// Upload one named image asset. Every pass whose program reads it gets it as
// a sampler uniform <name>_tex and a vec2 <name>_size of (width, height),
// bound after the channels and the arrays. An asset outlives the scene: a
// commit rebinds it, the same name again replaces it, ghee_mp_drop_asset or
// ghee_destroy frees it.
bool ghee_mp_set_asset(GheeScene *g, const char *name, const void *data,
                       int width, int height, int components, int is_f32, int filter) {
    if (width <= 0 || height <= 0) {
        set_error(g, "asset '%s' is %dx%d", name, width, height);
        return false;
    }
    const bool mip = filter == GHEE_FILTER_MIPMAP;
    // GLES2 mipmaps a power-of-two texture alone.
    if (mip && ((width & (width - 1)) || (height & (height - 1)))) {
        set_error(g, "asset '%s' is %dx%d, and mipmaps need a power-of-two size", name, width, height);
        return false;
    }
    if (mip && is_f32) {
        set_error(g, "asset '%s': mipmaps need 8-bit pixels", name);
        return false;
    }
    GLenum fmt;
    switch (components) {
        case 1: fmt = GL_LUMINANCE; break;
        case 3: fmt = GL_RGB; break;
        case 4: fmt = GL_RGBA; break;
        default:
            set_error(g, "asset '%s': %d components is not 1, 3 or 4", name, components);
            return false;
    }
    if (is_f32 && !g->has_float_tex) {
        set_error(g, "asset '%s': this GL has no float textures", name);
        return false;
    }
    make_current(g);
    g->changed = true;
    g->bind_epoch++;
    int idx = mp_asset_index(g, name);
    const bool fresh = idx < 0;
    if (fresh) {
        if (g->mp_nasset == g->mp_asset_cap) {
            const int cap = g->mp_asset_cap ? 2 * g->mp_asset_cap : 16;
            struct mp_asset *grown = realloc(g->mp_asset, (size_t)cap * sizeof *grown);
            if (!grown) { set_error(g, "asset '%s': no memory for it", name); return false; }
            g->mp_asset = grown;
            g->mp_asset_cap = cap;
        }
        char *held = suffixed(g, name, "");
        if (!held) return false;
        // assets set before the commit resolve there, later ones resolve here
        idx = g->mp_nasset++;
        memset(&g->mp_asset[idx], 0, sizeof g->mp_asset[idx]);
        g->mp_asset[idx].name = held;
        bool used = true;
        for (int p = 0; used && p < g->mp_count; p++)
            used = mp_use_asset(g, g->mp_program[p], idx, &g->mp_loc[p]);
        for (int l = 0; used && l < g->mp_nbatch; l++)
            used = mp_use_asset(g, g->mp_batch[l].program, idx, &g->mp_batch[l].loc);
        if (!used) {
            mp_forget_everywhere(g, idx);
            free(held);
            g->mp_nasset--;
            return false;
        }
    }
    // A commit resolves the assets already held into its programs, so an asset
    // set again counts too.
    for (int p = 0; p < g->mp_count; p++) {
        if (!mp_pass_fits(g, p, g->mp_nchan[p], g->mp_ndatatex[p])) {
            if (fresh) {
                mp_forget_everywhere(g, idx);
                free(g->mp_asset[idx].name);
                g->mp_nasset--;
            }
            return false;
        }
    }
    struct mp_asset *as = &g->mp_asset[idx];
    if (fresh) glGenTextures(1, &as->tex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, as->tex);
    const GLint filt = filter != GHEE_FILTER_NEAREST && !(is_f32 && !g->has_float_linear)
                     ? GL_LINEAR : GL_NEAREST;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mip ? GL_LINEAR_MIPMAP_LINEAR : filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    const GLenum type = is_f32 ? GL_FLOAT : GL_UNSIGNED_BYTE;
    glTexImage2D(GL_TEXTURE_2D, 0, fmt, width, height, 0, fmt, type, data);
    // The levels below 0 take their pixels from ghee_mp_set_asset_rect.
    for (int l = 1, w = width, h = height; mip && (w > 1 || h > 1); l++) {
        w = w > 1 ? w / 2 : 1;
        h = h > 1 ? h / 2 : 1;
        glTexImage2D(GL_TEXTURE_2D, l, fmt, w, h, 0, fmt, type, NULL);
    }
    as->w = width;
    as->h = height;
    as->fmt = fmt;
    as->type = type;
    as->mip = mip;
    return true;
}

// Replace rows y to y + rows of a set asset, each its full width, in the
// layout of its last ghee_mp_set_asset.
bool ghee_mp_set_asset_rows(GheeScene *g, const char *name, const void *data, int y, int rows) {
    const int idx = mp_asset_index(g, name);
    if (idx < 0) {
        set_error(g, "asset '%s' is not set", name);
        return false;
    }
    const struct mp_asset *as = &g->mp_asset[idx];
    if (y < 0 || rows <= 0 || y > as->h - rows) {
        set_error(g, "asset '%s' is %d rows high, and %d rows from row %d do not fit it",
                  name, as->h, rows, y);
        return false;
    }
    make_current(g);
    g->changed = true;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, as->tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, y, as->w, rows, as->fmt, as->type, data);
    return true;
}

// Replace a w x h rect at (x, y) of one mip level of a set asset.
bool ghee_mp_set_asset_rect(GheeScene *g, const char *name, int level,
                            int x, int y, int w, int h, const void *data) {
    const int idx = mp_asset_index(g, name);
    if (idx < 0) {
        set_error(g, "asset '%s' is not set", name);
        return false;
    }
    const struct mp_asset *as = &g->mp_asset[idx];
    int lw = as->w, lh = as->h, last = 0;
    if (as->mip) {
        for (int sw = lw, sh = lh; sw > 1 || sh > 1; last++) {
            sw = sw > 1 ? sw / 2 : 1;
            sh = sh > 1 ? sh / 2 : 1;
        }
    }
    if (level < 0 || level > last) {
        set_error(g, "asset '%s' has mip levels 0 to %d, and no level %d", name, last, level);
        return false;
    }
    for (int l = 0; l < level; l++) {
        lw = lw > 1 ? lw / 2 : 1;
        lh = lh > 1 ? lh / 2 : 1;
    }
    if (x < 0 || y < 0 || w <= 0 || h <= 0 || x > lw - w || y > lh - h) {
        set_error(g, "asset '%s' level %d is %dx%d, and a %dx%d rect at (%d, %d) does not fit it",
                  name, level, lw, lh, w, h, x, y);
        return false;
    }
    make_current(g);
    g->changed = true;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, as->tex);
    glTexSubImage2D(GL_TEXTURE_2D, level, x, y, w, h, as->fmt, as->type, data);
    return true;
}

// The EGL calls that import a dma-buf, looked up once.
static PFNEGLCREATEIMAGEKHRPROC mp_create_image;
static PFNEGLDESTROYIMAGEKHRPROC mp_destroy_image;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC mp_image_texture;

// Let go of every dma-buf that asset `as` imported.
static void mp_forget_dmabufs(GheeScene *g, struct mp_asset *as) {
    for (int i = 0; i < as->ndma; i++) {
        glDeleteTextures(1, &as->dma[i].tex);
        if (mp_destroy_image) mp_destroy_image(g->display, as->dma[i].img);
    }
    as->ndma = 0;
    if (as->fbo) glDeleteFramebuffers(1, &as->fbo);
    as->fbo = 0;
}

bool ghee_mp_set_asset_dmabuf(GheeScene *g, const char *name, int fd, int width, int height,
                              int pitch) {
    if (width <= 0 || height <= 0 || pitch < width * 4) {
        set_error(g, "asset '%s': a %dx%d dma-buf with rows of %d bytes", name, width, height, pitch);
        return false;
    }
    make_current(g);
    if (!mp_create_image) {
        mp_create_image = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
        mp_destroy_image = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
        mp_image_texture =
            (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
    }
    if (!mp_create_image || !mp_destroy_image || !mp_image_texture) {
        set_error(g, "asset '%s': this EGL imports no dma-buf", name);
        return false;
    }
    int idx = mp_asset_index(g, name);
    if (idx < 0) {
        if (!ghee_mp_set_asset(g, name, NULL, width, height, 4, 0, GHEE_FILTER_LINEAR)) return false;
        idx = mp_asset_index(g, name);
    }
    struct mp_asset *as = &g->mp_asset[idx];
    int k = 0;
    while (k < as->ndma && as->dma[k].fd != fd) k++;
    if (k == as->ndma) {
        if (as->ndma == MP_DMABUFS) {
            set_error(g, "asset '%s' holds %d dma-bufs already", name, MP_DMABUFS);
            return false;
        }
        const EGLint attr[] = {
            EGL_WIDTH, width,
            EGL_HEIGHT, height,
            EGL_LINUX_DRM_FOURCC_EXT, (EGLint)DRM_FORMAT_ABGR8888,
            EGL_DMA_BUF_PLANE0_FD_EXT, fd,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
            EGL_DMA_BUF_PLANE0_PITCH_EXT, pitch,
            EGL_NONE};
        EGLImageKHR img = mp_create_image(g->display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, attr);
        if (img == EGL_NO_IMAGE_KHR) {
            set_error(g, "asset '%s': EGL refused the %dx%d dma-buf, error 0x%X", name, width, height,
                      eglGetError());
            return false;
        }
        GLuint tex;
        glGenTextures(1, &tex);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        while (glGetError() != GL_NO_ERROR) { }
        mp_image_texture(GL_TEXTURE_2D, img);
        const GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            glDeleteTextures(1, &tex);
            mp_destroy_image(g->display, img);
            set_error(g, "asset '%s': GL refused the dma-buf as a texture, error 0x%X", name, err);
            return false;
        }
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        as->dma[k].fd = fd;
        as->dma[k].img = img;
        as->dma[k].tex = tex;
        as->ndma++;
    }
    // The passes sample the asset's texture, which takes the buffer once a
    // call: VC4 samples no linear texture, and Mesa copies an imported one
    // into a tiled shadow at each draw that samples it.
    if (!fade_ready(g)) return false;
    if (as->w != width || as->h != height || as->fmt != GL_RGBA || as->type != GL_UNSIGNED_BYTE ||
        !as->fbo) {
        glBindTexture(GL_TEXTURE_2D, as->tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        if (!as->fbo) glGenFramebuffers(1, &as->fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, as->fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, as->tex, 0);
        const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            set_error(g, "asset '%s': a %dx%d buffer to draw dma-bufs into failed: framebuffer 0x%X",
                      name, width, height, status);
            return false;
        }
        as->w = width;
        as->h = height;
        as->fmt = GL_RGBA;
        as->type = GL_UNSIGNED_BYTE;
        g->bind_epoch++;
    }
    GLint viewport[4], framebuffer;
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, as->fbo);
    glViewport(0, 0, width, height);
    glDisable(GL_BLEND);
    glUseProgram(g->fade_prog);
    glUniform1f(g->fade_vflip, 1.0f);
    glUniform2f(g->fade_scale, 1.0f, 1.0f);
    glUniform1f(g->fade_opacity, 1.0f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, as->dma[k].tex);
    glUniform1i(g->fade_sampler, 0);
    glBindVertexArray(g->VAO);
    glBindBuffer(GL_ARRAY_BUFFER, g->VBO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)framebuffer);
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    g->changed = true;
    return true;
}

void ghee_mp_forget_dmabufs(GheeScene *g, const char *name) {
    const int idx = mp_asset_index(g, name);
    if (idx < 0) return;
    make_current(g);
    mp_forget_dmabufs(g, &g->mp_asset[idx]);
}

// Free one named asset and its slot. An unknown name is a no-op.
void ghee_mp_drop_asset(GheeScene *g, const char *name) {
    const int idx = mp_asset_index(g, name);
    if (idx < 0) return;
    make_current(g);
    g->changed = true;
    g->bind_epoch++;
    mp_forget_dmabufs(g, &g->mp_asset[idx]);
    glDeleteTextures(1, &g->mp_asset[idx].tex);
    free(g->mp_asset[idx].name);
    memmove(&g->mp_asset[idx], &g->mp_asset[idx + 1],
            (size_t)(g->mp_nasset - idx - 1) * sizeof g->mp_asset[idx]);
    mp_forget_everywhere(g, idx);
    g->mp_nasset--;
}

static float elapsed_seconds(GheeScene *g) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (g->restart) {
        g->start_time = now;
        g->restart = false;
    }
    return g->has_time_override ? g->time_override
        : (now.tv_sec - g->start_time.tv_sec) + (now.tv_nsec - g->start_time.tv_nsec) / 1e9f;
}

// Set the per-frame uniforms of pass p's program or of one of its batches, in
// the program in use, and point its samplers at the units: the pass's channels,
// then its arrays, then the assets the program samples. `bind` also binds the
// pass's channel and array textures, which then stay bound for its batches.
static void mp_bind(GheeScene *g, int p, const struct mp_locs *loc, int into,
                    float elapsed, bool bind) {
    if (loc->res >= 0) glUniform3f(loc->res, (float)g->mp_w[p], (float)g->mp_h[p], 1.0f);
    if (loc->pixel >= 0)
        glUniform2f(loc->pixel, 1.0f / (float)g->mp_w[p], 1.0f / (float)g->mp_h[p]);
    if (loc->vflip >= 0) glUniform1f(loc->vflip, into == INTO_SCANOUT ? -1.0f : 1.0f);
    if (loc->yflip >= 0) {
        // VC4 puts gl_FragCoord on a corner of the pixel, the low one in a
        // buffer and the high one in the window surface. Each offset puts
        // either corner, and the centre, in the pixel whose floor it takes.
        if (into == INTO_SCANOUT) glUniform2f(loc->yflip, -1.0f, (float)g->mp_h[p] - 0.25f);
        else if (into == INTO_WINDOW) glUniform2f(loc->yflip, 1.0f, -0.25f);
        else glUniform2f(loc->yflip, 1.0f, 0.25f);
    }
    if (loc->time >= 0) glUniform1f(loc->time, elapsed);
    if (loc->dt >= 0) glUniform1f(loc->dt, g->dt);
    if (loc->frame >= 0) glUniform1i(loc->frame, g->frame);
    if (loc->chan_res >= 0) {
        // An unbound channel reads (0, 0, 0), as on Shadertoy. GL drops the
        // values past the array's length.
        float res[MP_MAX_CHANNELS][3] = {{0.0f}};
        for (int c = 0; c < g->mp_nchan[p]; c++) {
            const int src = g->mp_chan_src[p][c];
            if (src < 0) continue;
            res[c][0] = (float)g->mp_w[src];
            res[c][1] = (float)g->mp_h[src];
            res[c][2] = 1.0f;
        }
        glUniform3fv(loc->chan_res, MP_MAX_CHANNELS, &res[0][0]);
    }
    for (int c = 0; c < g->mp_nchan[p]; c++) {
        if (bind) {
            const int src = g->mp_chan_src[p][c];
            glActiveTexture(GL_TEXTURE0 + c);
            if (src < 0) {
                glBindTexture(GL_TEXTURE_2D, g->black_tex);
            } else {
                const int rd = (src < p) ? g->mp_write[src] : 1 - g->mp_write[src];
                glBindTexture(GL_TEXTURE_2D, g->mp_tex[src][rd]);
            }
        }
        if (loc->chan[c] >= 0) glUniform1i(loc->chan[c], c);
    }
    for (int i = 0; i < g->mp_ndatatex[p]; i++) {
        // a channel declared after this array shifts the unit, so the
        // unit is the render's to set and the location alone is cached
        const int unit = g->mp_nchan[p] + i;
        if (bind) {
            glActiveTexture(GL_TEXTURE0 + unit);
            glBindTexture(GL_TEXTURE_2D, g->mp_array_tex[g->mp_datatex[p][i]]);
        }
        if (loc->data[i] >= 0) glUniform1i(loc->data[i], unit);
    }
    // assets follow the channels and the arrays, and only a program that
    // samples the name pays a unit for it
    int unit = g->mp_nchan[p] + g->mp_ndatatex[p];
    for (int i = 0; i < loc->nuses; i++) {
        const struct mp_use *u = &loc->uses[i];
        const struct mp_asset *as = &g->mp_asset[u->asset];
        if (u->size >= 0) glUniform2f(u->size, (float)as->w, (float)as->h);
        if (u->loc < 0) continue;
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, as->tex);
        glUniform1i(u->loc, unit);
        unit++;
    }
}

// mp_bind() of batch `l` of pass p, its program in use. A program that this
// batch bound last at this flip still holds its sizes and its texture units,
// and takes the time and the asset textures alone.
static void mp_bind_batch(GheeScene *g, int p, const struct mp_batch *l, int into, float elapsed) {
    const struct mp_locs *loc = &l->loc;
    struct mp_prog *pr = NULL;
    for (int i = 0; i < g->mp_nprog && !pr; i++)
        if (g->mp_prog[i].program == l->program) pr = &g->mp_prog[i];
    if (!pr || pr->bound_batch != l->id || pr->bound_into != into ||
        pr->bound_epoch != g->bind_epoch) {
        mp_bind(g, p, loc, into, elapsed, false);
        if (pr) {
            pr->bound_batch = l->id;
            pr->bound_into = into;
            pr->bound_epoch = g->bind_epoch;
        }
        return;
    }
    if (loc->time >= 0) glUniform1f(loc->time, elapsed);
    if (loc->dt >= 0) glUniform1f(loc->dt, g->dt);
    if (loc->frame >= 0) glUniform1i(loc->frame, g->frame);
    int unit = g->mp_nchan[p] + g->mp_ndatatex[p];
    for (int i = 0; i < loc->nuses; i++) {
        const struct mp_use *u = &loc->uses[i];
        if (u->loc < 0) continue;
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, g->mp_asset[u->asset].tex);
        unit++;
    }
}

// Set the values of batch `l` on its program, in use, unless the program
// holds them already.
static void mp_load_values(GheeScene *g, struct mp_batch *l) {
    struct mp_prog *held = NULL;
    for (int i = 0; i < g->mp_nprog && !held; i++)
        if (g->mp_prog[i].program == l->program) held = &g->mp_prog[i];
    if (held && held->loaded == l->id && !l->dirty) return;
    for (int k = 0; k < l->nvalues; k++) {
        struct mp_value *v = &l->values[k];
        if (v->prog != l->program) {
            v->loc = glGetUniformLocation(l->program, v->name);
            v->prog = l->program;
        }
        if (v->loc < 0) continue;
        switch (v->type) {
            case PARAM_TYPE_FLOAT: glUniform1f(v->loc, v->v[0]); break;
            case PARAM_TYPE_INT:   glUniform1i(v->loc, v->i); break;
            case PARAM_TYPE_VEC2:  glUniform2f(v->loc, v->v[0], v->v[1]); break;
            case PARAM_TYPE_VEC3:  glUniform3f(v->loc, v->v[0], v->v[1], v->v[2]); break;
            case PARAM_TYPE_VEC4:  glUniform4f(v->loc, v->v[0], v->v[1], v->v[2], v->v[3]); break;
        }
    }
    if (held) held->loaded = l->id;
    l->dirty = false;
}

static void mp_blend(int blend) {
    switch (blend) {
    case GHEE_BLEND_ADD: glBlendFunc(GL_ONE, GL_ONE); break;
    case GHEE_BLEND_MULTIPLY: glBlendFunc(GL_DST_COLOR, GL_ONE_MINUS_SRC_ALPHA); break;
    case GHEE_BLEND_SCREEN: glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR); break;
    default: glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA); break;
    }
}

static void mp_draw_quads(struct mp_batch *l) {
    glBindVertexArray(l->vao);
    const GLsizei stride = l->nattr * 4 * (GLsizei)sizeof(GLfloat);
    if (l->quads <= QUAD_CHUNK) {
        if (l->pointed != l->vbo || l->pointed_nattr != l->nattr) {
            glBindBuffer(GL_ARRAY_BUFFER, l->vbo);
            for (int k = 0; k < l->nattr; k++)
                glVertexAttribPointer(k, 4, GL_FLOAT, GL_FALSE, stride,
                                      (const GLvoid *)((size_t)k * 4 * sizeof(GLfloat)));
            l->pointed = l->vbo;
            l->pointed_nattr = l->nattr;
        }
        glDrawElements(GL_TRIANGLES, l->quads * 6, GL_UNSIGNED_SHORT, (const GLvoid *)0);
        return;
    }
    l->pointed = 0;
    glBindBuffer(GL_ARRAY_BUFFER, l->vbo);
    for (int at = 0; at < l->quads; at += QUAD_CHUNK) {
        const int n = l->quads - at < QUAD_CHUNK ? l->quads - at : QUAD_CHUNK;
        const size_t base = (size_t)at * 4 * (size_t)stride;
        for (int k = 0; k < l->nattr; k++)
            glVertexAttribPointer(k, 4, GL_FLOAT, GL_FALSE, stride,
                                  (const GLvoid *)(base + (size_t)k * 4 * sizeof(GLfloat)));
        glDrawElements(GL_TRIANGLES, n * 6, GL_UNSIGNED_SHORT, (const GLvoid *)0);
    }
}

// Make fade_prog ready. false with the error set when GL refuses it.
static bool fade_ready(GheeScene *g) {
    if (g->fade_prog) return true;
    g->fade_prog = create_shader_program(g, fade_fragment_source);
    if (!g->fade_prog) return false;
    g->fade_sampler = glGetUniformLocation(g->fade_prog, "tex");
    g->fade_scale = glGetUniformLocation(g->fade_prog, "scale");
    g->fade_opacity = glGetUniformLocation(g->fade_prog, "opacity");
    g->fade_vflip = glGetUniformLocation(g->fade_prog, "ghee_vflip");
    return true;
}

// Make the fade buffer hold a w x h pass, and fade_prog ready. false with
// the error set when GL refuses either.
static bool ensure_fade(GheeScene *g, int w, int h) {
    if (!fade_ready(g)) return false;
    if (g->fade_fbo && g->fade_w >= w && g->fade_h >= h) return true;
    const int fw = w > g->fade_w ? w : g->fade_w, fh = h > g->fade_h ? h : g->fade_h;
    if (g->fade_fbo) {
        glDeleteFramebuffers(1, &g->fade_fbo);
        glDeleteTextures(1, &g->fade_tex);
        g->fade_fbo = g->fade_tex = 0;
        g->fade_w = g->fade_h = 0;
    }
    while (glGetError() != GL_NO_ERROR) { }
    glGenFramebuffers(1, &g->fade_fbo);
    glGenTextures(1, &g->fade_tex);
    glBindTexture(GL_TEXTURE_2D, g->fade_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fw, fh, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindFramebuffer(GL_FRAMEBUFFER, g->fade_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g->fade_tex, 0);
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    const GLenum err = glGetError();
    if (status != GL_FRAMEBUFFER_COMPLETE || err != GL_NO_ERROR) {
        set_error(g, "%dx%d fade buffer failed: framebuffer 0x%X, GL 0x%X", fw, fh, status, err);
        glDeleteFramebuffers(1, &g->fade_fbo);
        glDeleteTextures(1, &g->fade_tex);
        g->fade_fbo = g->fade_tex = 0;
        return false;
    }
    g->fade_w = fw;
    g->fade_h = fh;
    return true;
}

// Draw the batches of pass p into `target`, its bound buffer, after the
// pass's quad. A batch below opacity 1 draws into the fade buffer, each
// quad over the ones before it, and the buffer then lands on the target at
// the batch's opacity by its blend.
static void mp_draw_batches(GheeScene *g, int p, GLuint target, int into, float elapsed) {
    bool drew = false;
    for (int k = 0; k < g->mp_nbatch; k++) {
        struct mp_batch *l = &g->mp_batch[g->mp_order[k]];
        if (l->pass != p || !batch_draws(l)) continue;
        // Texture unit 0 holds what the pass bound there for its batches,
        // which the fade buffer takes for a while.
        GLint unit0 = 0;
        bool alone = false;
        if (l->opacity < 1.0f) {
            glActiveTexture(GL_TEXTURE0);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &unit0);
            alone = ensure_fade(g, g->mp_w[p], g->mp_h[p]);
            glBindTexture(GL_TEXTURE_2D, (GLuint)unit0);
            if (!alone && !g->fade_failed) {
                fprintf(stderr, "ghee: %s draws at full opacity: %s\n", l->name, error_text(g));
                g->fade_failed = true;
            }
        }
        glUseProgram(l->program);
        mp_load_values(g, l);
        glEnable(GL_BLEND);
        if (alone) {
            glBindFramebuffer(GL_FRAMEBUFFER, g->fade_fbo);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            // The fade buffer keeps GL's row order, as a pass's buffer does.
            mp_bind_batch(g, p, l, INTO_BUFFER, elapsed);
            mp_blend(GHEE_BLEND_NORMAL);
        } else {
            mp_bind_batch(g, p, l, into, elapsed);
            mp_blend(l->blend);
        }
        mp_draw_quads(l);
        if (alone) {
            glBindFramebuffer(GL_FRAMEBUFFER, target);
            glUseProgram(g->fade_prog);
            glUniform1f(g->fade_vflip, into == INTO_SCANOUT ? -1.0f : 1.0f);
            glUniform2f(g->fade_scale, (float)g->mp_w[p] / (float)g->fade_w,
                        (float)g->mp_h[p] / (float)g->fade_h);
            glUniform1f(g->fade_opacity, l->opacity);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, g->fade_tex);
            glUniform1i(g->fade_sampler, 0);
            mp_blend(l->blend);
            glBindVertexArray(g->VAO);
            glBindBuffer(GL_ARRAY_BUFFER, g->VBO);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glBindTexture(GL_TEXTURE_2D, (GLuint)unit0);
        }
        drew = true;
    }
    if (drew) {
        glDisable(GL_BLEND);
        glBindVertexArray(g->VAO);
        glBindBuffer(GL_ARRAY_BUFFER, g->VBO);
    }
}

// Render every pass into its buffer, for the readback and for the present.
// direct_last draws the last pass straight into scan_fbo, in place of its own
// buffer, mirrored when that is a lent scanout buffer. The caller sets it
// only when nothing reads the last pass this frame, neither a channel nor a
// preview.
static void render_passes(GheeScene *g, float elapsed, bool direct_last) {
    for (int p = 0; p < g->mp_count; p++) {
        const bool direct = direct_last && p == g->mp_count - 1;
        const GLuint target = direct ? g->scan_fbo : g->mp_fbo[p][g->mp_write[p]];
        int into = INTO_BUFFER;
        if (direct && !target) into = INTO_WINDOW;
        else if (direct && g->display_mode) into = INTO_SCANOUT;
        glBindFramebuffer(GL_FRAMEBUFFER, target);
        // The per-pass buffer size. The output pass is always full-size (the caller
        // enforces that), so the direct scanout draw still gets the display
        // viewport.
        glViewport(0, 0, g->mp_w[p], g->mp_h[p]);
        if (g->mp_program[p]) {
            // The fullscreen quad overwrites every pixel, so the clear is invisible, and
            // on tiled GPUs (VC4) it flags the target as not-loaded, skipping a
            // full-frame tile read from memory before the draw.
            glClear(GL_COLOR_BUFFER_BIT);
            glUseProgram(g->mp_program[p]);
            mp_bind(g, p, &g->mp_loc[p], into, elapsed, true);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        } else {
            // A pass of batches alone draws them over transparent black. The
            // bind leaves the channels and the arrays bound for the batches.
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mp_bind(g, p, &g->mp_loc[p], into, elapsed, true);
        }
        mp_draw_batches(g, p, target, into, elapsed);
        if (g->mp_filter[p] == GHEE_FILTER_MIPMAP && !direct) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, g->mp_tex[p][g->mp_write[p]]);
            glGenerateMipmap(GL_TEXTURE_2D);
        }
    }
    g->frame += 1;
    if (g->frame >= 1048576) g->frame = 0;
    g->draws++;
}

// Read a rectangle back as `components` bytes a pixel. A GL that cannot hand
// back RGB reads RGBA into scratch, and the copy drops every fourth byte.
// `out` takes the top row first, and the bound buffer holds it last when
// bottom_first.
static void read_pixels(GheeScene *g, int w, int h, int components, bool bottom_first,
                        uint8_t *out) {
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    const bool direct = components != 3 || g->reads_rgb;
    const size_t want = direct ? (bottom_first ? (size_t)w * components : 0) : (size_t)w * h * 4;
    if (g->rgba_scratch_size < want) {
        uint8_t *grown = realloc(g->rgba_scratch, want);
        if (!grown) { set_error(g, "cannot allocate a %dx%d readback buffer", w, h); return; }
        g->rgba_scratch = grown;
        g->rgba_scratch_size = want;
    }
    if (direct) {
        glReadPixels(0, 0, w, h, components == 3 ? GL_RGB : GL_RGBA, GL_UNSIGNED_BYTE, out);
        if (!bottom_first) return;
        const size_t row = (size_t)w * components;
        for (int y = 0; y < h / 2; y++) {
            uint8_t *a = out + (size_t)y * row, *b = out + (size_t)(h - 1 - y) * row;
            memcpy(g->rgba_scratch, a, row);
            memcpy(a, b, row);
            memcpy(b, g->rgba_scratch, row);
        }
        return;
    }
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, g->rgba_scratch);
    for (int y = 0; y < h; y++) {
        const uint8_t *src = g->rgba_scratch + (size_t)(bottom_first ? h - 1 - y : y) * w * 4;
        uint8_t *dst = out + (size_t)y * w * 3;
        for (int x = 0; x < w; x++, src += 4, dst += 3) {
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
        }
    }
}

// The CRTC's primary plane. ghee drives it when the render is scaled, and
// only the primary one covers the whole screen. The universal-planes cap is
// what makes the kernel list it at all: without it a client sees the overlay
// planes alone. 0 = none, and the caller then refuses the size.
static uint32_t primary_plane(int fd, uint32_t crtc_id) {
    drmModeRes *res = drmModeGetResources(fd);
    if (!res) return 0;
    int index = -1;
    for (int i = 0; i < res->count_crtcs; i++)
        if (res->crtcs[i] == crtc_id) index = i;
    drmModeFreeResources(res);
    if (index < 0) return 0;

    if (drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1)) return 0;
    drmModePlaneRes *pr = drmModeGetPlaneResources(fd);
    if (!pr) return 0;
    uint32_t found = 0;
    for (uint32_t i = 0; i < pr->count_planes && !found; i++) {
        drmModePlane *p = drmModeGetPlane(fd, pr->planes[i]);
        if (!p) continue;
        if (p->possible_crtcs & (1u << index)) {
            drmModeObjectProperties *props =
                drmModeObjectGetProperties(fd, p->plane_id, DRM_MODE_OBJECT_PLANE);
            for (uint32_t k = 0; props && k < props->count_props; k++) {
                drmModePropertyRes *pp = drmModeGetProperty(fd, props->props[k]);
                if (pp && strcmp(pp->name, "type") == 0 &&
                    props->prop_values[k] == DRM_PLANE_TYPE_PRIMARY)
                    found = p->plane_id;
                if (pp) drmModeFreeProperty(pp);
            }
            if (props) drmModeFreeObjectProperties(props);
        }
        drmModeFreePlane(p);
    }
    drmModeFreePlaneResources(pr);
    return found;
}

// Whether the primary plane of the display that create drives takes frames
// of `format`. A card with no connected display answers true, and the open
// then fails on the display.
static bool display_takes(int fd, uint32_t format) {
    uint32_t conn, crtc;
    drmModeModeInfo mode;
    if (pick_display_mode(fd, &conn, &crtc, &mode)) return true;
    const uint32_t plane = primary_plane(fd, crtc);
    drmModePlane *p = plane ? drmModeGetPlane(fd, plane) : NULL;
    bool takes = false;
    for (uint32_t i = 0; p && i < p->count_formats; i++)
        takes = takes || p->formats[i] == format;
    if (p) drmModeFreePlane(p);
    return takes;
}

// Whether this card will scan a width x height buffer across the whole mode.
// A dumb buffer of that size goes on the plane and straight back off, so the
// answer comes from the kernel rather than from a guess at what the driver
// supports. A card that refuses either step keeps its old plane.
static bool plane_scales(GheeScene *g, int width, int height) {
    struct drm_mode_create_dumb creq = { .width = width, .height = height, .bpp = 32 };
    if (drmIoctl(g->drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &creq)) return false;
    uint32_t handles[4] = { creq.handle }, strides[4] = { creq.pitch }, offsets[4] = { 0 };
    uint32_t fb = 0;
    bool ok = false;
    if (!drmModeAddFB2(g->drm_fd, width, height, GBM_FORMAT_XRGB8888,
                       handles, strides, offsets, &fb, 0)) {
        drmModePlane *was = drmModeGetPlane(g->drm_fd, g->disp_plane);
        ok = drmModeSetPlane(g->drm_fd, g->disp_plane, g->disp_crtc, fb, 0,
                             0, 0, g->disp_mode.hdisplay, g->disp_mode.vdisplay,
                             0, 0, (uint32_t)width << 16, (uint32_t)height << 16) == 0;
        if (was) {
            if (ok && was->fb_id)
                drmModeSetPlane(g->drm_fd, g->disp_plane, g->disp_crtc, was->fb_id, 0,
                                0, 0, g->disp_mode.hdisplay, g->disp_mode.vdisplay,
                                0, 0, (uint32_t)g->disp_mode.hdisplay << 16,
                                (uint32_t)g->disp_mode.vdisplay << 16);
            drmModeFreePlane(was);
        }
        drmModeRmFB(g->drm_fd, fb);
    }
    struct drm_mode_destroy_dumb dreq = { .handle = creq.handle };
    drmIoctl(g->drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &dreq);
    return ok;
}

// Put the buffer on the scaling plane and wait out the frame it lands on.
// drmModeSetPlane sends no completion event, so the vblank wait is what paces
// the loop, and after it the previous buffer is off the glass.
static bool set_scaled_plane(GheeScene *g, uint32_t fb, int w, int h) {
    if (drmModeSetPlane(g->drm_fd, g->disp_plane, g->disp_crtc, fb, 0,
                        0, 0, g->disp_mode.hdisplay, g->disp_mode.vdisplay,
                        0, 0, (uint32_t)w << 16, (uint32_t)h << 16)) {
        if (!g->flip_failed) {
            g->flip_failed = true;
            set_error(g, "drmModeSetPlane failed: %s%s", strerror(errno),
                      errno == EACCES ? ". DRM master went to another process" : "");
        }
        return false;
    }
    g->flip_failed = false;
    drmVBlank vb = { .request = { .type = DRM_VBLANK_RELATIVE, .sequence = 1 } };
    drmWaitVBlank(g->drm_fd, &vb);
    return true;
}

struct gbm_bo *ghee_flip_rotate(struct gbm_bo **prev, struct gbm_bo **shown,
                                struct gbm_bo *bo, bool flipped) {
    if (!flipped) return bo;
    struct gbm_bo *freed = *shown;
    *shown = *prev;
    *prev = bo;
    return freed;
}

static double mono_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void sleep_until(double t) {
    struct timespec ts = { .tv_sec = (time_t)t, .tv_nsec = (long)((t - (time_t)t) * 1e9) };
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {}
}

static void ring_push(struct timing_ring *r, double x) {
    r->v[r->at] = x;
    r->at = (r->at + 1) % TIMING_RING;
    if (r->n < TIMING_RING) r->n++;
}

// The ring's timings into out, oldest first. Returns their count.
static int ring_read(const struct timing_ring *r, double *out) {
    for (int i = 0; i < r->n; i++) out[i] = r->v[(r->at - r->n + i + TIMING_RING) % TIMING_RING];
    return r->n;
}

// The longest of n timings while they are few, else the second longest, so
// that one stray timing moves nothing.
static double usual_longest(const double *v, int n) {
    double first = 0, second = 0;
    for (int i = 0; i < n; i++) {
        if (v[i] > first) {
            second = first;
            first = v[i];
        } else if (v[i] > second) {
            second = v[i];
        }
    }
    return n < TIMING_RING / 2 ? first : second;
}

double ghee_jit_start(const double *draw, int ndraw, const double *busy, int nbusy) {
    if (nbusy <= 0) return 0;
    double fastest = busy[0];
    for (int i = 1; i < nbusy; i++)
        if (busy[i] < fastest) fastest = busy[i];
    // A draw as long as a transfer, from a loop that sleeps or a stall, ends
    // the draws that time the next one.
    int from = ndraw;
    while (from > 0 && draw[from - 1] < fastest) from--;
    if (from == ndraw) return 0;
    // A commit slower than the fastest waited on the GPU's fence.
    const double lead = usual_longest(draw + from, ndraw - from) +
                        usual_longest(busy, nbusy) - fastest + JIT_MARGIN;
    return fastest - lead;
}

// A flip's user data is where its event's timestamp goes, in seconds.
static void on_flip_event(int fd, unsigned int seq, unsigned int sec, unsigned int usec, void *at) {
    (void)fd;
    (void)seq;
    if (at) *(double *)at = sec + usec * 1e-6;
}

// Wait for the event of the page flip in flight on `fd`, and reap it. false
// leaves the reason in err.
static bool await_flip(int fd, char *err, size_t n) {
    drmEventContext ev = { .version = 2, .page_flip_handler = on_flip_event };
    struct pollfd pfd = { .fd = fd, .events = POLLIN };
    for (;;) {
        const int r = poll(&pfd, 1, FLIP_STUCK_MS);
        if (r > 0) {
            drmHandleEvent(fd, &ev);
            return true;
        }
        if (r == 0) { snprintf(err, n, "page flip still not complete after %d ms", FLIP_STUCK_MS); return false; }
        if (errno == EINTR || errno == EAGAIN) continue;
        snprintf(err, n, "page flip poll failed: %s", strerror(errno));
        return false;
    }
}

static bool wait_flip(GheeScene *g) {
    if (!g->flip_pending) return true;
    char err[ERROR_BUFFER_SIZE];
    if (!await_flip(g->drm_fd, err, sizeof err)) {
        set_error(g, "%s", err);
        return false;
    }
    g->flip_pending = false;
    if (g->flip_landed > g->flip_sent) ring_push(&g->flip_busy, g->flip_landed - g->flip_sent);
    return true;
}

static void flip_lock(GheeScene *g) { if (g->flipper) pthread_mutex_lock(&g->flip_mu); }
static void flip_unlock(GheeScene *g) { if (g->flipper) pthread_mutex_unlock(&g->flip_mu); }

uint64_t ghee_display_taken(GheeScene *g) {
    flip_lock(g);
    const uint64_t n = g->taken;
    flip_unlock(g);
    return n;
}

// Note the time since present last returned: the draw of the frame at hand.
static void note_draw(GheeScene *g) {
    if (g->presented > 0) ring_push(&g->draw_time, mono_seconds() - g->presented);
}

// The seconds after the latest flip's ioctl at which present returns.
static double jit_start(GheeScene *g) {
    double draw[TIMING_RING], busy[TIMING_RING];
    const int nd = ring_read(&g->draw_time, draw);
    const int nb = ring_read(&g->flip_busy, busy);
    return ghee_jit_start(draw, nd, busy, nb);
}

// Release the buffers that the flip thread took off the glass, and report its
// error. A window surface's buffers go back on the thread that draws into
// them. flip_mu held.
static void take_freed(GheeScene *g) {
    for (int i = 0; i < g->flip_nfreed; i++) release_bo(g, g->flip_freed[i]);
    g->flip_nfreed = 0;
    if (g->flip_error[0]) {
        set_error(g, "%s", g->flip_error);
        g->flip_error[0] = '\0';
    }
}

static void *flip_main(void *arg) {
    GheeScene *g = arg;
    char err[ERROR_BUFFER_SIZE];
    pthread_mutex_lock(&g->flip_mu);
    for (;;) {
        while (!g->flip_queued && !g->flip_pending && !g->flip_stop)
            pthread_cond_wait(&g->flip_cv, &g->flip_mu);
        if (g->flip_pending) {
            pthread_mutex_unlock(&g->flip_mu);
            const bool reaped = await_flip(g->drm_fd, err, sizeof err);
            pthread_mutex_lock(&g->flip_mu);
            if (reaped && g->flip_landed > g->flip_sent)
                ring_push(&g->flip_busy, g->flip_landed - g->flip_sent);
            // The event puts disp_prev on the glass, and the frame before it
            // is off.
            if (reaped && g->disp_shown && g->flip_nfreed < LENT_BUFFERS) {
                g->flip_freed[g->flip_nfreed++] = g->disp_shown;
                g->disp_shown = NULL;
            }
            g->flip_pending = false;
            if (!reaped) snprintf(g->flip_error, sizeof g->flip_error, "%s", err);
            pthread_cond_broadcast(&g->flip_cv);
            continue;
        }
        if (!g->flip_queued) break;
        struct gbm_bo *bo = g->flip_queued;
        const uint32_t fb = g->flip_queued_fb;
        const uint64_t frame = g->flip_queued_frame;
        g->flip_queued = NULL;
        g->flip_sending = bo;
        pthread_cond_broadcast(&g->flip_cv);
        pthread_mutex_unlock(&g->flip_mu);
        const double sent = mono_seconds();
        const bool flipped =
            !drmModePageFlip(g->drm_fd, g->disp_crtc, fb, DRM_MODE_PAGE_FLIP_EVENT, &g->flip_landed);
        const int e = errno;
        pthread_mutex_lock(&g->flip_mu);
        g->flip_sending = NULL;
        struct gbm_bo *freed = ghee_flip_rotate(&g->disp_prev, &g->disp_shown, bo, flipped);
        if (freed && g->flip_nfreed < LENT_BUFFERS) g->flip_freed[g->flip_nfreed++] = freed;
        g->flip_pending = flipped;
        if (flipped) {
            g->flip_failed = false;
            g->flip_sent = sent;
            g->taken = frame;
        } else if (!g->flip_failed) {   // a refused flip leaves disp_prev on the screen
            g->flip_failed = true;
            snprintf(g->flip_error, sizeof g->flip_error, "drmModePageFlip failed: %s%s", strerror(e),
                     e == EACCES ? ". DRM master went to another process" : "");
        }
        pthread_cond_broadcast(&g->flip_cv);
    }
    pthread_mutex_unlock(&g->flip_mu);
    return NULL;
}

static bool start_flipper(GheeScene *g, char *eb, size_t n) {
    pthread_mutex_init(&g->flip_mu, NULL);
    pthread_cond_init(&g->flip_cv, NULL);
    const int e = pthread_create(&g->flip_thread, NULL, flip_main, g);
    if (e) {
        pthread_cond_destroy(&g->flip_cv);
        pthread_mutex_destroy(&g->flip_mu);
        g->flipper = false;
        snprintf(eb, n, "cannot start the flip thread: %s", strerror(e));
        return false;
    }
    return true;
}

// Wait until no frame waits for its page flip and no flip is in flight.
static void settle_flips(GheeScene *g) {
    if (!g->flipper) {
        wait_flip(g);
        return;
    }
    pthread_mutex_lock(&g->flip_mu);
    while (g->flip_queued || g->flip_sending || g->flip_pending)
        pthread_cond_wait(&g->flip_cv, &g->flip_mu);
    take_freed(g);
    pthread_mutex_unlock(&g->flip_mu);
}

int ghee_display_set_drop(GheeScene *g, int drop) {
    if (!g->display_mode) {
        set_error(g, "drop: this renderer drives no display");
        return -1;
    }
    if (drop && g->disp_plane) {
        set_error(g, "drop takes the render at the display's own size, %ux%u",
                  g->disp_mode.hdisplay, g->disp_mode.vdisplay);
        return -1;
    }
    if (drop && !g->flipper) {
        // A frame that waits for the display goes on when the bus frees,
        // with no present to put it there, so one CPU runs the thread too.
        char eb[ERROR_BUFFER_SIZE];
        settle_flips(g);
        g->flipper = true;
        make_current(g);
        if ((g->lent_count && !lend_buffers(g, eb, sizeof eb)) || !start_flipper(g, eb, sizeof eb)) {
            g->flipper = false;
            set_error(g, "%s", eb);
            return -1;
        }
    }
    flip_lock(g);
    g->drop = drop;
    flip_unlock(g);
    return 0;
}

// Flip the queued frame, reap the last flip and end the flip thread.
static void stop_flipper(GheeScene *g) {
    if (!g->flipper) {
        wait_flip(g);
        return;
    }
    pthread_mutex_lock(&g->flip_mu);
    g->flip_stop = true;
    pthread_cond_broadcast(&g->flip_cv);
    pthread_mutex_unlock(&g->flip_mu);
    pthread_join(g->flip_thread, NULL);
    take_freed(g);
    g->flipper = false;
    pthread_cond_destroy(&g->flip_cv);
    pthread_mutex_destroy(&g->flip_mu);
}

// The blit program (present, preview readback, encoder), built on first use.
// display mode builds it at create.
static bool ensure_blit(GheeScene *g) {
    if (g->blit_prog) return true;
    g->blit_prog = create_shader_program(g, blit_fragment_source);
    if (!g->blit_prog) {
        fprintf(stderr, "ghee: the blit program: %s\n", error_text(g));
        return false;
    }
    g->blit_sampler = glGetUniformLocation(g->blit_prog, "tex");
    g->blit_opaque = glGetUniformLocation(g->blit_prog, "opaque");
    g->blit_turn = glGetUniformLocation(g->blit_prog, "turn");
    g->blit_shift = glGetUniformLocation(g->blit_prog, "shift");
    glUseProgram(g->blit_prog);
    const GLint bvf = glGetUniformLocation(g->blit_prog, "ghee_vflip");
    if (bvf >= 0) glUniform1f(bvf, 1.0f);
    return true;
}

// Set the blit program up to sample `tex`, the frame, turned as the output
// takes it, into a target that stores its top row first when top_first, or
// its bottom row first as GL does. opaque 1 writes alpha 1.
static void bind_blit(GheeScene *g, GLuint tex, bool top_first, float opaque) {
    // The target's uv, turned y down, goes through TURNS onto the frame, y
    // down, and then onto the texture, which holds the frame's bottom row
    // first.
    const GLfloat *t = TURNS[g->turn];
    const GLfloat dy = top_first ? 1.0f : -1.0f, oy = top_first ? 0.0f : 1.0f;
    const GLfloat turn[4] = { t[0], -t[1], t[2] * dy, -t[3] * dy };
    const GLfloat shift[2] = { t[2] * oy + t[4], 1.0f - (t[3] * oy + t[5]) };
    glUseProgram(g->blit_prog);
    glUniform1f(g->blit_opaque, opaque);
    glUniformMatrix2fv(g->blit_turn, 1, GL_FALSE, turn);
    glUniform2fv(g->blit_shift, 1, shift);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(g->blit_sampler, 0);
    glBindVertexArray(g->VAO);
    glBindBuffer(GL_ARRAY_BUFFER, g->VBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid *)(2 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
}

// Draw the bound blit quad into a w x h target. A target at a whole multiple of
// the output's size samples nearest, so each frame pixel lands as an exact
// square block, and a turned frame lands pixel for pixel.
static void draw_blit(GheeScene *g, int w, int h) {
    const int ow = g->out_w, oh = g->out_h;
    const bool blocks = ow > 0 && oh > 0 && w >= ow && w % ow == 0 && h % oh == 0 &&
                        w / ow == h / oh;
    GLint mag = GL_LINEAR;
    if (blocks) {
        glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, &mag);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    if (blocks) glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag);
}

// Scale the final pass into a small FBO on the GPU. The FBO is kept until the
// requested size changes.
static void read_preview(GheeScene *g, uint8_t *buffer_rgb, int pw, int ph) {
    if (!ensure_blit(g)) return;
    const int last = g->mp_count - 1;
    const GLenum fmt = (g->components == 3) ? GL_RGB : GL_RGBA;
    if (!g->preview_fbo || g->preview_w != pw || g->preview_h != ph) {
        if (g->preview_fbo) { glDeleteFramebuffers(1, &g->preview_fbo); glDeleteTextures(1, &g->preview_tex); }
        glGenTextures(1, &g->preview_tex);
        glBindTexture(GL_TEXTURE_2D, g->preview_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, fmt, pw, ph, 0, fmt, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glGenFramebuffers(1, &g->preview_fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, g->preview_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g->preview_tex, 0);
        const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            set_error(g, "%dx%d preview buffer is incomplete: framebuffer 0x%X", pw, ph, status);
            glDeleteFramebuffers(1, &g->preview_fbo);
            glDeleteTextures(1, &g->preview_tex);
            g->preview_fbo = 0;
            g->preview_tex = 0;
            g->preview_w = 0;
            g->preview_h = 0;
            return;
        }
        g->preview_w = pw;
        g->preview_h = ph;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, g->preview_fbo);
    glViewport(0, 0, pw, ph);
    bind_blit(g, g->mp_tex[last][g->mp_write[last]], true, g->keep_alpha ? 0.0f : 1.0f);
    draw_blit(g, pw, ph);
    read_pixels(g, pw, ph, g->components, false, buffer_rgb);
}

// Blank the display, or light it again. Blanking stops the CRTC driving the
// connector, which runs the kernel driver's own disable path: a panel turns
// its display off and drops its backlight. After a relight, the next present
// sets the mode again. A blanked display draws nothing at all, so the caller
// paces its own loop while the display is dark.
int ghee_display_blank(GheeScene *g, int blank) {
    if (!g || !g->display_mode) return -1;
    if (!blank == !g->disp_blanked) return 0;
    if (!blank) {
        g->disp_blanked = false;
        g->disp_first = true;
        return 0;
    }
    settle_flips(g);
    if (drmModeSetCrtc(g->drm_fd, g->disp_crtc, 0, 0, 0, NULL, 0, NULL)) {
        set_error(g, "cannot blank the display: %s", strerror(errno));
        return -1;
    }
    // Nothing scans these out now, and the next light starts from a fresh
    // front buffer.
    release_bo(g, g->disp_shown);
    g->disp_shown = NULL;
    release_bo(g, g->disp_prev);
    g->disp_prev = NULL;
    g->disp_blanked = true;
    return 0;
}

int ghee_display_blanked(GheeScene *g) { return g && g->disp_blanked; }

// Render the passes and scan the last one out to the connector. A non-NULL
// buffer_rgb also reads the frame back at pw x ph, for the console's preview.
//
// Two choices keep the memory traffic down on a VC4:
//  * When nothing reads the last pass this frame, neither a channel nor a
//    preview, the last pass draws straight into the buffer that scans out.
//    That skips storing a full-size texture and copying it over, two thirds
//    of the frame's traffic.
//  * The page flip goes in now, and the next present collects its
//    completion, so the next frame draws while this one scans out, instead
//    of waiting for the vertical blank.
void ghee_mp_present(GheeScene *g, uint8_t *buffer_rgb, int pw, int ph) {
    if (!g->display_mode || g->disp_blanked) return;
    if (g->mp_count <= 0) { set_error(g, "present: the scene has no passes"); return; }
    make_current(g);
    const int last = g->mp_count - 1;
    bool direct = !(buffer_rgb && pw > 0 && ph > 0) && !g->keep_last && !g->turn;
    for (int p = 0; p < g->mp_count && direct; p++)
        for (int c = 0; c < g->mp_nchan[p]; c++)
            if (g->mp_chan_src[p][c] == last) { direct = false; break; }
    struct gbm_bo *bo = NULL;
    flip_lock(g);
    take_freed(g);
    for (int i = 0; i < g->lent_count && !bo; i++)
        if (g->lent[i].bo != g->disp_prev && g->lent[i].bo != g->disp_shown &&
            g->lent[i].bo != g->flip_queued && g->lent[i].bo != g->flip_sending) {
            bo = g->lent[i].bo;
            g->scan_fbo = g->lent[i].fbo;
        }
    flip_unlock(g);
    if (g->lent_count && !bo) { set_error(g, "present: every lent scanout buffer is still on the display"); return; }
    render_passes(g, elapsed_seconds(g), direct);

    if (!direct) {
        glBindFramebuffer(GL_FRAMEBUFFER, g->scan_fbo);
        glViewport(0, 0, g->out_w, g->out_h);
        glClear(GL_COLOR_BUFFER_BIT);   // skip the tile load (see render_passes)
        bind_blit(g, g->mp_tex[last][g->mp_write[last]], g->lent_count > 0, 1.0f);
        draw_blit(g, g->out_w, g->out_h);
    }
    if (g->lent_count) {
        // The page flip waits on the fence of the GPU's writes, so a flush
        // is enough to hand the buffer on.
        glFlush();
    } else {
        eglSwapBuffers(g->display, g->surface);
        bo = gbm_surface_lock_front_buffer(g->gbm_surface);
    }
    uint32_t fb = fb_for(g, bo);
    double until = 0;
    if (g->disp_plane) {
        // The mode is already set: the plane carries each frame from here, and
        // the wait inside means the buffer before this one is off the glass.
        g->disp_first = false;
        g->frames++;
        if (set_scaled_plane(g, fb, g->out_w, g->out_h)) {
            release_bo(g, g->disp_shown);
            g->disp_shown = g->disp_prev;
            g->disp_prev = bo;
            g->taken = g->frames;
        } else {
            release_bo(g, bo);
        }
    } else if (g->disp_first) {
        g->frames++;
        if (drmModeSetCrtc(g->drm_fd, g->disp_crtc, fb, 0, 0, &g->disp_conn, 1, &g->disp_mode))
            set_error(g, "drmModeSetCrtc failed: %s", strerror(errno));
        else
            g->taken = g->frames;
        g->disp_first = false;
        g->disp_prev = bo;               // on the screen now (SetCrtc is immediate)
    } else if (g->flipper) {
        // The frame before waits for the flip in flight, and this one waits
        // for it, or with drop takes its place.
        pthread_mutex_lock(&g->flip_mu);
        note_draw(g);
        if (g->drop && g->flip_queued)
            release_bo(g, g->flip_queued);
        else
            while (g->flip_queued) pthread_cond_wait(&g->flip_cv, &g->flip_mu);
        g->flip_queued = bo;
        g->flip_queued_fb = fb;
        g->flip_queued_frame = ++g->frames;
        pthread_cond_broadcast(&g->flip_cv);
        if (g->jit && !g->drop) {
            while (g->flip_queued == bo || g->flip_sending == bo)
                pthread_cond_wait(&g->flip_cv, &g->flip_mu);
            const double start = g->disp_prev == bo ? jit_start(g) : 0;
            if (start > 0) until = g->flip_sent + start;
        }
        pthread_mutex_unlock(&g->flip_mu);
    } else {
        note_draw(g);
        const bool waited = wait_flip(g);
        const double sent = mono_seconds();
        const bool flipped = waited && !drmModePageFlip(g->drm_fd, g->disp_crtc, fb,
                                                        DRM_MODE_PAGE_FLIP_EVENT, &g->flip_landed);
        const int err = errno;
        g->frames++;
        release_bo(g, ghee_flip_rotate(&g->disp_prev, &g->disp_shown, bo, flipped));
        if (flipped) {
            g->flip_pending = true;
            g->flip_failed = false;
            g->flip_sent = sent;
            g->taken = g->frames;
            const double start = g->jit ? jit_start(g) : 0;
            if (start > 0) until = sent + start;
            // The kernel queues the flip's work on this CPU, the only one,
            // where it waits until this thread blocks, and a host busy with
            // its next frame holds the CPU for milliseconds. A yield starts
            // that work now.
            sched_yield();
        } else if (waited && !g->flip_failed) {  // a refused flip leaves disp_prev on the screen
            g->flip_failed = true;
            set_error(g, "drmModePageFlip failed: %s%s", strerror(err),
                      err == EACCES ? ". DRM master went to another process" : "");
        }
    }
    if (buffer_rgb && pw > 0 && ph > 0) read_preview(g, buffer_rgb, pw, ph);
    for (int p = 0; p < g->mp_count; p++) g->mp_write[p] ^= 1;
    if (until > 0) sleep_until(until);
    g->presented = mono_seconds();
}

// Make the linear buffer of a headless readback, once. false when the GPU
// makes no linear buffer that it renders into and the CPU maps.
static bool ensure_linear(GheeScene *g) {
    if (g->lin_fbo) return true;
    if (g->lin_tried) return false;
    g->lin_tried = true;
    PFNEGLCREATEIMAGEKHRPROC create_image = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
    PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target =
        (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
    PFNEGLDESTROYIMAGEKHRPROC destroy_image =
        (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
    if (!create_image || !image_target || !destroy_image || !g->gbm) return false;
    struct gbm_bo *bo = gbm_bo_create(g->gbm, g->out_w, g->out_h, GBM_FORMAT_ABGR8888,
                                      GBM_BO_USE_RENDERING | GBM_BO_USE_LINEAR);
    if (!bo) return false;
    uint32_t stride = 0;
    void *map = NULL;
    if (!gbm_bo_map(bo, 0, 0, g->out_w, g->out_h, GBM_BO_TRANSFER_READ, &stride, &map)) {
        gbm_bo_destroy(bo);
        return false;
    }
    gbm_bo_unmap(bo, map);
    // v3d reads a dma-buf that arrives without a modifier as tiled.
    uint64_t mod = gbm_bo_get_modifier(bo);
    if (mod == DRM_FORMAT_MOD_INVALID) mod = DRM_FORMAT_MOD_LINEAR;
    const int fd = gbm_bo_get_fd(bo);
    if (fd < 0) { gbm_bo_destroy(bo); return false; }
    const EGLint attr[] = {
        EGL_WIDTH, g->out_w,
        EGL_HEIGHT, g->out_h,
        EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_ABGR8888,
        EGL_DMA_BUF_PLANE0_FD_EXT, fd,
        EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
        EGL_DMA_BUF_PLANE0_PITCH_EXT, (EGLint)gbm_bo_get_stride(bo),
        EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, (EGLint)(mod & 0xffffffff),
        EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, (EGLint)(mod >> 32),
        EGL_NONE };
    const EGLImageKHR image = create_image(g->display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, attr);
    close(fd);
    if (image == EGL_NO_IMAGE_KHR) { gbm_bo_destroy(bo); return false; }
    while (glGetError() != GL_NO_ERROR) { }
    GLuint tex = 0, fbo = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    image_target(GL_TEXTURE_2D, image);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (status != GL_FRAMEBUFFER_COMPLETE || glGetError() != GL_NO_ERROR) {
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &tex);
        destroy_image(g->display, image);
        gbm_bo_destroy(bo);
        return false;
    }
    g->lin_bo = bo;
    g->lin_image = image;
    g->lin_tex = tex;
    g->lin_fbo = fbo;
    return true;
}

// Whether this frame's whole readback draws the last pass into the linear
// buffer alone: nothing else samples the last pass's own buffer, neither a
// pass, nor a turn, nor an encoder.
static bool reads_linear(GheeScene *g) {
    if (g->display_mode || g->turn || g->keep_last || !g->read_linear) return false;
    if (g->read_linear < 0 && !strstr(g->renderer, "VC4") && !strstr(g->renderer, "V3D")) return false;
    const int last = g->mp_count - 1;
    for (int p = 0; p < g->mp_count; p++)
        for (int c = 0; c < g->mp_nchan[p]; c++)
            if (g->mp_chan_src[p][c] == last) return false;
    return ensure_linear(g);
}

// Copy the frame out of the linear buffer, which holds it bottom row first,
// as read_pixels reads the last pass's buffer. false with the error set when
// the CPU cannot map it.
static bool read_linear(GheeScene *g, uint8_t *out) {
    uint32_t stride = 0;
    void *map = NULL;
    glFinish();
    const uint8_t *src = gbm_bo_map(g->lin_bo, 0, 0, g->out_w, g->out_h, GBM_BO_TRANSFER_READ,
                                    &stride, &map);
    if (!src) {
        set_error(g, "cannot map the %dx%d readback buffer: %s", g->out_w, g->out_h, strerror(errno));
        return false;
    }
    const int w = g->out_w, h = g->out_h, c = g->components;
    // A last pass that stores no alpha reads back 255 from its RGB buffer.
    const bool opaque = !(g->mp_alpha[g->mp_count - 1] && g->keep_alpha);
    for (int y = 0; y < h; y++) {
        const uint8_t *row = src + (size_t)(h - 1 - y) * stride;
        uint8_t *dst = out + (size_t)y * w * c;
        if (c == 4) {
            memcpy(dst, row, (size_t)w * 4);
            if (opaque)
                for (int x = 0; x < w; x++) dst[4 * x + 3] = 255;
        } else {
            for (int x = 0; x < w; x++) {
                dst[3 * x] = row[4 * x];
                dst[3 * x + 1] = row[4 * x + 1];
                dst[3 * x + 2] = row[4 * x + 2];
            }
        }
    }
    gbm_bo_unmap(g->lin_bo, map);
    return true;
}

// Copy a frame that left its last pass in the linear buffer alone into the
// last pass's own buffer, for a reader that the frame did not have.
static void lin_restore(GheeScene *g) {
    if (!g->lin_last) return;
    g->lin_last = false;
    make_current(g);
    if (g->mp_count <= 0 || !ensure_blit(g)) return;
    const int last = g->mp_count - 1;
    glBindFramebuffer(GL_FRAMEBUFFER, g->mp_fbo[last][1 - g->mp_write[last]]);
    glViewport(0, 0, g->out_w, g->out_h);
    glDisable(GL_BLEND);
    bind_blit(g, g->lin_tex, false, 0.0f);
    draw_blit(g, g->out_w, g->out_h);
}

bool ghee_read_linear(GheeScene *g, int mode) {
    g->read_linear = mode;
    if (!mode || g->display_mode) return false;
    make_current(g);
    return ensure_linear(g);
}

// Render every pass and swap. buffer_rgb NULL renders nothing back to the CPU,
// pw/ph > 0 read back a GPU-downscaled preview instead of the full frame.
void ghee_mp_render(GheeScene *g, uint8_t *buffer_rgb, int pw, int ph) {
    if (g->mp_count <= 0) { set_error(g, "render: the scene has no passes"); return; }
    make_current(g);
    if (buffer_rgb && !(pw > 0 && ph > 0) && reads_linear(g)) {
        g->scan_fbo = g->lin_fbo;
        render_passes(g, elapsed_seconds(g), true);
        g->lin_last = true;
        if (!read_linear(g, buffer_rgb)) g->read_linear = 0;
        for (int p = 0; p < g->mp_count; p++) g->mp_write[p] ^= 1;
        return;
    }
    g->lin_last = false;
    render_passes(g, elapsed_seconds(g), false);

    const int last = g->mp_count - 1;
    if (buffer_rgb && pw > 0 && ph > 0) {
        read_preview(g, buffer_rgb, pw, ph);
    } else if (buffer_rgb && g->turn) {
        read_preview(g, buffer_rgb, g->out_w, g->out_h);
    } else if (buffer_rgb) {
        glBindFramebuffer(GL_FRAMEBUFFER, g->mp_fbo[last][g->mp_write[last]]);
        read_pixels(g, g->out_w, g->out_h, g->components, true, buffer_rgb);
        // An RGB buffer reads back alpha 255 by itself. A final pass that
        // stores its alpha takes 255 here.
        if (g->components == 4 && g->mp_alpha[last] && !g->keep_alpha) {
            uint8_t *a = buffer_rgb + 3;
            for (size_t i = 0, n = (size_t)g->out_w * g->out_h; i < n; i++, a += 4) *a = 255;
        }
    }

    for (int p = 0; p < g->mp_count; p++) g->mp_write[p] ^= 1;  // swap all
}

static bool reads_time(const struct mp_locs *loc) {
    return loc->time >= 0 || loc->dt >= 0 || loc->frame >= 0;
}

// True while a frame drawn now may differ from the last one with no call in
// between: a program that draws reads iTime, iTimeDelta or iFrame, or a pass
// reads a buffer of the frame before.
static bool moves(GheeScene *g) {
    for (int p = 0; p < g->mp_count; p++) {
        if (g->mp_program[p] && reads_time(&g->mp_loc[p])) return true;
        for (int c = 0; c < g->mp_nchan[p]; c++)
            if (g->mp_chan_src[p][c] >= p) return true;
    }
    for (int k = 0; k < g->mp_nbatch; k++) {
        const struct mp_batch *l = &g->mp_batch[k];
        if (batch_draws(l) && reads_time(&l->loc)) return true;
    }
    return false;
}

bool ghee_mp_repeat(GheeScene *g, uint8_t *out, int pw, int ph) {
    if (!(pw > 0 && ph > 0)) pw = ph = 0;
    if (!g->drawn || g->changed || g->mp_count <= 0 || moves(g)) return false;
    if (out && !(g->kept_valid && g->kept_w == pw && g->kept_h == ph)) return false;
    // A repeat on a display returns when a present of a frame drawn in no
    // time would: at once with drop, else a frame's time after the last
    // return, the transfer on a display with no vblank, else the refresh.
    double period = 0;
    if (g->display_mode) {
        if (g->disp_blanked || g->disp_first) return false;
        flip_lock(g);
        const bool failed = g->flip_failed, drop = g->drop;
        double busy[TIMING_RING];
        const int nb = g->jit ? ring_read(&g->flip_busy, busy) : 0;
        flip_unlock(g);
        if (failed) return false;
        for (int i = 0; i < nb; i++)
            if (period <= 0 || busy[i] < period) period = busy[i];
        if (period <= 0 && g->disp_mode.vrefresh) period = 1.0 / g->disp_mode.vrefresh;
        if (period <= 0) period = 1.0 / 60;
        if (drop) period = 0;
    }
    if (out) memcpy(out, g->kept, g->kept_len);
    elapsed_seconds(g);
    g->frame += 1;
    if (g->frame >= 1048576) g->frame = 0;
    if (g->display_mode) {
        if (period > 0 && g->presented + period > mono_seconds()) sleep_until(g->presented + period);
        g->presented = mono_seconds();
    }
    return true;
}

void ghee_mp_drew(GheeScene *g, const uint8_t *out, int pw, int ph) {
    if (g->draws == g->draws_seen) return;
    g->draws_seen = g->draws;
    g->changed = false;
    g->drawn = true;
    g->kept_valid = false;
    if (!out || moves(g)) return;
    if (!(pw > 0 && ph > 0)) pw = ph = 0;
    const size_t n = (size_t)(pw ? pw : g->out_w) * (size_t)(ph ? ph : g->out_h) * (size_t)g->components;
    if (n > g->kept_cap) {
        uint8_t *grown = realloc(g->kept, n);
        if (!grown) return;
        g->kept = grown;
        g->kept_cap = n;
    }
    memcpy(g->kept, out, n);
    g->kept_len = n;
    g->kept_w = pw;
    g->kept_h = ph;
    g->kept_valid = true;
}

// The GL this renderer runs on: GL_RENDERER and GL_VERSION in one string.
const char *ghee_renderer(GheeScene *g) { return g ? g->renderer : ""; }

void ghee_set_time(GheeScene *g, float t) { g->time_override = t; g->has_time_override = true; }

// Back to the monotonic clock after a ghee_set_time.
void ghee_clear_time(GheeScene *g) { g->has_time_override = false; }

void ghee_restart_clock(GheeScene *g) {
    g->frame = 0;
    g->restart = true;
}

void ghee_set_dt(GheeScene *g, float dt) { g->dt = dt; }

void ghee_keep_alpha(GheeScene *g, int keep) {
    if (!keep != !g->keep_alpha) g->changed = true;
    g->keep_alpha = keep;
}

bool ghee_set_rotate(GheeScene *g, int quarter_turns) {
    if (quarter_turns < 0 || quarter_turns > 3) {
        set_error(g, "%d quarter turns is not 0 to 3", quarter_turns);
        return false;
    }
    if (g->mp_count > 0 || g->stage_count > 0) {
        set_error(g, "rotate needs a renderer with no scene yet");
        return false;
    }
    g->turn = quarter_turns;
    g->width = quarter_turns % 2 ? g->out_h : g->out_w;
    g->height = quarter_turns % 2 ? g->out_w : g->out_h;
    return true;
}

int ghee_max_texture_units(GheeScene *g) { return g ? (int)g->max_units : 0; }

int ghee_gl_texture_units(GheeScene *g) { return g ? (int)g->gl_units : 0; }

int ghee_max_texture_size(GheeScene *g) { return g ? (int)g->max_tex : 0; }

int ghee_gl_fd(GheeScene *g) {
    if (!g) return -1;
    return g->gpu_fd >= 0 ? g->gpu_fd : g->drm_fd;
}

// Whether this GL linear-filters a data texture, which ghee_mp_set_array
// filters by the same answer. False on V3D, which has float textures but no
// OES_texture_float_linear, and on VC4, whose linear filter mixes a
// neighbour into a read at a texel centre that is no binary fraction of the
// side.
bool ghee_data_texture_linear(GheeScene *g) { return g && g->has_float_tex && g->has_float_linear; }

bool ghee_mipmaps_npot(GheeScene *g) { return g && g->has_npot; }

bool ghee_gles3(GheeScene *g) { return g && g->gles3; }

bool ghee_compiles(GheeScene *g, const char *fragment) {
    if (!g || !fragment) return false;
    make_current(g);
    const GLuint shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(shader, 1, &fragment, NULL);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    glDeleteShader(shader);
    return ok != 0;
}

int ghee_compiled(GheeScene *g) { return g ? g->compiled : 0; }

uint64_t ghee_draws(GheeScene *g) { return g ? g->draws : 0; }

// The display mode's refresh rate in Hz, 0 outside display mode.
int ghee_display_vrefresh(GheeScene *g) { return g ? (int)g->disp_mode.vrefresh : 0; }

// ---- internal seams for encoder.c (engine_internal.h) ---------------------

void *ghee_i_egl_display(GheeScene *g) { return g->display; }
void  ghee_i_make_current(GheeScene *g) { make_current(g); }
void  ghee_i_error(GheeScene *g, const char *msg) { set_error(g, "%s", msg); }

void ghee_i_blit_last(GheeScene *g, unsigned fbo, int w, int h) {
    if (g->mp_count <= 0) { set_error(g, "blit: the scene has no passes"); return; }
    if (!ensure_blit(g)) return;
    const int last = g->mp_count - 1;
    const int buf = 1 - g->mp_write[last];
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, w, h);
    glClear(GL_COLOR_BUFFER_BIT);
    bind_blit(g, g->mp_tex[last][buf], true, 1.0f);
    draw_blit(g, w, h);
}

void ghee_i_keep_last(GheeScene *g, int delta) {
    lin_restore(g);
    g->keep_last += delta;
    if (g->keep_last < 0) g->keep_last = 0;
    g->changed = true;
}

// Tear down the running scene and anything pending (for switching scenes).
// Keeps the GL context.
void ghee_mp_reset(GheeScene *g) {
    make_current(g);
    g->changed = true;
    ghee_mp_abort(g);
    mp_free_passes(g);
}

// Whether GL on this card runs on hardware. A display-only card (a panel
// under a kernel display driver) still gets hardware GL when Mesa pairs it
// with the render node, so software here means no usable render node exists.
// A probe, so it stays quiet. Bring-up trouble reports 1: the display path
// then runs as before and reports its own error.
int ghee_display_renders(const char *dri_device) {
    const char *dev = dri_device ? dri_device : DEFAULT_DRI_DEVICE;
    const int fd = open(dev, O_RDWR | O_CLOEXEC);
    if (fd < 0) return 1;
    const int hw = ghee_display_renders_on(fd);
    close(fd);
    return hw;
}

int ghee_display_renders_on(int card) {
    int hw = 1;
    struct gbm_device *gbm = NULL;
    EGLDisplay dpy = EGL_NO_DISPLAY;
    EGLContext ctx = EGL_NO_CONTEXT;
    const int hushed = hush_stderr();
    gbm = gbm_create_device(card);
    if (!gbm) goto out;
    dpy = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, gbm, NULL);
    if (dpy == EGL_NO_DISPLAY || eglInitialize(dpy, NULL, NULL) == EGL_FALSE) {
        dpy = EGL_NO_DISPLAY;
        goto out;
    }
    const EGLint attribs[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_NONE };
    const EGLint ctx_attribs[] = { EGL_CONTEXT_MAJOR_VERSION_KHR, 2, EGL_NONE };
    EGLConfig config;
    EGLint n = 0;
    if (!eglChooseConfig(dpy, attribs, &config, 1, &n) || n < 1) goto out;
    ctx = eglCreateContext(dpy, config, EGL_NO_CONTEXT, ctx_attribs);
    if (ctx == EGL_NO_CONTEXT) goto out;
    if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) goto out;
    const char *renderer = (const char *)glGetString(GL_RENDERER);
    if (renderer && (strstr(renderer, "llvmpipe") || strstr(renderer, "softpipe") ||
                     strstr(renderer, "swrast")))
        hw = 0;
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
out:
    if (ctx != EGL_NO_CONTEXT) eglDestroyContext(dpy, ctx);
    if (dpy != EGL_NO_DISPLAY) eglTerminate(dpy);
    if (gbm) gbm_device_destroy(gbm);
    unhush_stderr(hushed);
    return hw;
}
