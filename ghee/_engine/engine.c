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
#include <errno.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
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
// One buffer on the screen, one waiting on a page flip, and one to draw into.
#define LENT_BUFFERS 3
#define MP_BATCH_ATTRIBS 4
// GLES2 guarantees 16-bit indices alone, and a quad takes 4 vertices, so a
// batch of more quads than this draws in chunks.
#define QUAD_CHUNK 16383

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
    GLint time, dt, frame;
    GLint vflip;                         // ghee_vflip of vertex_shader_source, -1 to draw into the window surface
    GLint chan[MP_MAX_CHANNELS];
    GLint data[MP_MAX_DATATEX];
    struct mp_use *uses;                 // in asset order, which is the order of their units
    int nuses, uses_cap;
};

// One image asset: the host hands the pixels over once, they bind as a
// sampler uniform <name>_tex, with <name>_size a vec2 of (width, height).
struct mp_asset {
    GLuint tex;
    char *name;                          // the asset's name, before the suffixes
    int w, h;
    GLenum fmt, type;
    bool mip;                            // every mip level, down to 1x1
};

// One batch of ghee_mp_add_batch: quads that a pass draws into its buffer
// after its full-screen quad, with a program of their own and a blend.
struct mp_batch {
    int pass;
    char name[96];                       // the batch's name in an error
    GLuint program;
    int blend;                           // GHEE_BLEND_*
    GLuint vbo, vao;
    int quads, nattr;                    // nattr vec4 attributes a vertex
    struct mp_locs loc;
};

struct GheeScene {
    int width, height, components;

    int drm_fd;
    struct gbm_device *gbm;
    struct gbm_surface *gbm_surface;
    EGLDisplay display;
    EGLSurface surface;
    EGLContext context;

    GLuint VBO, VAO;

    struct timespec start_time;
    bool has_time_override;              // when set, render uses time_override instead of the wall clock
    float time_override;
    float dt;                            // how many seconds the last frame took, fed to every pass as DT
    float frame;                         // frames rendered, fed to every pass as `frame`, wrapped to stay float-exact

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
    bool mp_mip[MP_MAX_PASSES];          // the buffer holds a mip chain, built after each render
    int mp_chan_src[MP_MAX_PASSES][MP_MAX_CHANNELS];  // channel -> source pass index, -1 = none
    int mp_nchan[MP_MAX_PASSES];
    struct mp_locs mp_loc[MP_MAX_PASSES];
    GLint gl_units;                      // GL_MAX_TEXTURE_IMAGE_UNITS, 16 on VC4
    GLint max_units;                     // min(MP_MAX_UNITS, gl_units)

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
    bool stage_mip[MP_MAX_PASSES];

    // Batches, staged with the passes and committed with them.
    int mp_nbatch, stage_nbatch;
    int mp_batch_cap, stage_batch_cap;
    struct mp_batch *mp_batch, *stage_batch;
    GLuint quad_ibo;                     // QUAD_CHUNK quads of indices, shared by every batch

    // GLES2 promises RGBA readback and nothing else. vc4 reads RGB too, v3d
    // hands back nothing for it, so a panel there would pack an all-black
    // frame. Where RGB is missing the readback takes RGBA into scratch and
    // drops the alpha, which costs nothing on the boards that read RGB.
    bool reads_rgb;
    uint8_t *rgba_scratch;
    size_t rgba_scratch_size;
    bool has_float_tex, has_float_linear;   // OES_texture_float[_linear] on this GL
    bool has_npot;                       // mipmaps a texture of any size: GLES 3 or OES_texture_npot
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
    bool profile;                        // when on, ghee_mp_render splits draw vs readback (adds a glFinish)
    double prof_draw_ms, prof_read_ms;   // last frame's split (profiling mode)

    // display mode: scan the final buffer straight out to a KMS connector
    // (zero CPU copy), instead of reading it back. Set up by ghee_display_create.
    bool display_mode;
    uint32_t disp_conn, disp_crtc;
    drmModeModeInfo disp_mode;
    drmModeCrtc *disp_saved;
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
    GLuint blit_prog;
    GLint blit_sampler, blit_flip, blit_opaque;
    GLuint preview_fbo, preview_tex;   // small FBO for a cheap downscaled readback
    int preview_w, preview_h;
    int keep_last;                     // >0: an encoder samples the final pass, never draw it direct

    char renderer[160];                  // GL_RENDERER and GL_VERSION, captured at create
    char last_error[ERROR_BUFFER_SIZE];
};

static double ms_between(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1000.0 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

// The vertex shader of every pass's full-screen quad, and of the present
// blit. ghee_vflip = -1 mirrors the quad in y, for a pass that draws straight
// into the window surface, which Mesa stores top row first. A lent scanout
// buffer is an FBO and keeps GL's row order, the order of the readback, so a
// pass draws into it unmirrored. v_texCoord stays as it is either way, so a
// pass's own GLSL reads the same coordinates.
static const char *vertex_shader_source =
    "#version 100\n"
    "attribute vec4 position;"
    "attribute vec2 texCoord;"
    "uniform float ghee_vflip;"
    "varying vec2 v_texCoord;"
    "void main() {"
    "   gl_Position = vec4(position.x, position.y * ghee_vflip, position.z, position.w);"
    "   v_texCoord = texCoord;"
    "}";

// Present blit: sample the final pass texture onto the scanout target. flipY
// mirrors it for the window surface, which Mesa stores top row first, and
// opaque 1 writes alpha 1.
static const char *blit_fragment_source =
    "#version 100\n"
    "precision mediump float;"
    "varying vec2 v_texCoord;"
    "uniform sampler2D tex;"
    "uniform float flipY;"
    "uniform float opaque;"
    "void main() { vec2 uv = vec2(v_texCoord.x, mix(v_texCoord.y, 1.0 - v_texCoord.y, flipY));"
    " vec4 c = texture2D(tex, uv);"
    " gl_FragColor = vec4(c.rgb, max(c.a, opaque)); }";

static int say_level = 1;
void ghee_set_verbosity(int level) { say_level = level; }
#define SAY(at, ...) do { if (say_level >= (at)) fprintf(stderr, __VA_ARGS__); } while (0)

// Defined further down, beside the present path that they serve.
static bool wait_flip(GheeScene *g);
static uint32_t primary_plane(int fd, uint32_t crtc_id);
static bool plane_scales(GheeScene *g, int width, int height);
static bool ensure_blit(GheeScene *g);

static void keep_error(GheeScene *g, const char *fmt, va_list ap) {
    vsnprintf(g->last_error, ERROR_BUFFER_SIZE, fmt, ap);
    g->last_error[ERROR_BUFFER_SIZE - 1] = '\0';
}

static void set_error(GheeScene *g, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    keep_error(g, fmt, ap);
    va_end(ap);
    fprintf(stderr, "ghee: %s\n", g->last_error);
}

// A compile or link log goes to the caller unprinted: the caller names the
// pass and the author's line in it, and reports it once.
static void set_shader_error(GheeScene *g, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    keep_error(g, fmt, ap);
    va_end(ap);
}

// Make this instance's GL context current. Required at the top of every public
// op so multiple GheeScene instances can be used interleaved in one process.
static void make_current(GheeScene *g) {
    if (!eglMakeCurrent(g->display, g->surface, g->surface, g->context))
        fprintf(stderr, "ghee: eglMakeCurrent failed: 0x%X (disp=%p ctx=%p)\n",
                eglGetError(), (void *)g->display, (void *)g->context);
}

const char *ghee_last_error(GheeScene *g) {
    static char err[ERROR_BUFFER_SIZE];
    if (!g) return "null instance";
    strncpy(err, g->last_error, ERROR_BUFFER_SIZE - 1);
    err[ERROR_BUFFER_SIZE - 1] = '\0';
    g->last_error[0] = '\0';
    return err;
}

// ---- shader compilation ---------------------------------------------------

static GLuint create_shader(GheeScene *g, GLenum type, const char *source) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char info[ERROR_BUFFER_SIZE];
        glGetShaderInfoLog(shader, ERROR_BUFFER_SIZE, NULL, info);
        set_shader_error(g, "%s%s", type == GL_VERTEX_SHADER ? "vertex shader: " : "", info);
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
        char info[ERROR_BUFFER_SIZE];
        glGetProgramInfoLog(program, ERROR_BUFFER_SIZE, NULL, info);
        set_shader_error(g, "%s", info);
        glDeleteShader(vs);
        glDeleteShader(fs);
        glDeleteProgram(program);
        return 0;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

static const char *const quad_attribs[] = { "position", "texCoord" };
static const char *const batch_attribs[MP_BATCH_ATTRIBS] = {
    "ghee_a0", "ghee_a1", "ghee_a2", "ghee_a3"
};

// A program on the shared full-screen quad vertex shader.
static GLuint create_shader_program(GheeScene *g, const char *fragment_shader_source) {
    return create_program(g, vertex_shader_source, fragment_shader_source, quad_attribs, 2);
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

static GheeScene *create_impl(int width, int height, int components, const char *dri_device, int scanout, int rgb565, char **err) {
    static char early_err[ERROR_BUFFER_SIZE];
    early_err[0] = '\0';

    GheeScene *g = calloc(1, sizeof(GheeScene));
    if (!g) { if (err) *err = "out of memory"; return NULL; }
    g->width = width;
    g->height = height;
    g->components = components;
    g->drm_fd = -1;
    g->gpu_fd = -1;
    g->display = EGL_NO_DISPLAY;

    const char *dev = dri_device ? dri_device : DEFAULT_DRI_DEVICE;
    g->drm_fd = open(dev, O_RDWR | O_CLOEXEC);
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
    g->has_npot = (glexts && strstr(glexts, "GL_OES_texture_npot") != NULL) ||
                  (version && strncmp(version, "OpenGL ES 3", 11) == 0);
    SAY(2, "   format  data texture %s%s\n",
            g->has_float_tex ? "float" : "8-bit LUMINANCE",
            g->has_float_tex ? (g->has_float_linear ? " (linear)" : " (nearest)") : " (linear)");

    clock_gettime(CLOCK_MONOTONIC, &g->start_time);
    g->dt = 1.0f / 60.0f;
    return g;

fail:
    if (err) *err = early_err;
    ghee_destroy(g);
    return NULL;
}

GheeScene *ghee_create(int width, int height, int components, const char *dri_device, char **err) {
    return create_impl(width, height, components, dri_device, 0, 0, err);
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
    for (int i = 0; i < LENT_BUFFERS; i++) {
        // The card's driver reads a buffer row by row and knows nothing of the
        // GPU's tiling, so every lent buffer is linear.
        struct gbm_bo *bo = gbm_bo_create(g->gbm, g->width, g->height, GBM_FORMAT_XRGB8888,
                                          GBM_BO_USE_RENDERING | GBM_BO_USE_LINEAR);
        if (!bo) { snprintf(eb, n, "the GPU cannot allocate a linear %dx%d scanout buffer", g->width, g->height); return false; }
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
            EGL_WIDTH, g->width,
            EGL_HEIGHT, g->height,
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

int ghee_display_probe(const char *dri_device, int *width, int *height) {
    const char *dev = dri_device ? dri_device : DEFAULT_DRI_DEVICE;
    int fd = open(dev, O_RDWR | O_CLOEXEC);
    if (fd < 0) return -1;
    uint32_t conn, crtc;
    drmModeModeInfo mode;
    int rc = pick_display_mode(fd, &conn, &crtc, &mode);
    close(fd);
    if (rc) return -1;
    if (width) *width = mode.hdisplay;
    if (height) *height = mode.vdisplay;
    return 0;
}

// Render to a KMS connector instead of reading back. width/height should match
// the connector mode (ghee_display_probe reports it). Needs DRM master.
// rgb565 renders/scans 16-bit color: half the tile-store + scanout bytes.
GheeScene *ghee_display_create(int width, int height, const char *dri_device, int rgb565, char **err) {
    static char eb[ERROR_BUFFER_SIZE];
    GheeScene *g = create_impl(width, height, 4, dri_device, 1, rgb565, err);
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

    make_current(g);
    if (g->gpu_fd >= 0 && !lend_buffers(g, eb, sizeof eb)) { if (err) *err = eb; ghee_destroy(g); return NULL; }
    if (!ensure_blit(g)) { if (err) *err = "blit shader failed"; ghee_destroy(g); return NULL; }
    g->display_mode = true;
    g->disp_first = true;
    return g;
}

void ghee_destroy(GheeScene *g) {
    if (!g) return;
    // every GL delete below needs THIS instance's context current: with two live
    // renderers the wrong context would destroy the other one's objects
    const bool have_gl = g->display != EGL_NO_DISPLAY && g->context != EGL_NO_CONTEXT;
    if (have_gl) make_current(g);
    if (g->display_mode) {
        wait_flip(g);   // reap any in-flight flip before tearing the CRTC down
        if (g->disp_plane && g->disp_saved) {
            // The scaled path set no mode, so destroy puts none back. The plane
            // still points at a buffer about to be freed, so hand it the old
            // framebuffer at full size and leave the CRTC alone.
            drmModeSetPlane(g->drm_fd, g->disp_plane, g->disp_crtc,
                            g->disp_saved->buffer_id, 0,
                            0, 0, g->disp_mode.hdisplay, g->disp_mode.vdisplay,
                            0, 0, (uint32_t)g->disp_mode.hdisplay << 16,
                            (uint32_t)g->disp_mode.vdisplay << 16);
        } else if (g->disp_saved) {
            drmModeSetCrtc(g->drm_fd, g->disp_saved->crtc_id, g->disp_saved->buffer_id,
                           g->disp_saved->x, g->disp_saved->y, &g->disp_conn, 1, &g->disp_saved->mode);
        }
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
        if (g->preview_fbo) { glDeleteFramebuffers(1, &g->preview_fbo); glDeleteTextures(1, &g->preview_tex); }
        if (g->blit_prog) glDeleteProgram(g->blit_prog);
        if (g->VBO) glDeleteBuffers(1, &g->VBO);
        if (g->VAO) glDeleteVertexArrays(1, &g->VAO);
        ghee_mp_reset(g);   // all pass programs/buffers + the pending scene
        if (g->quad_ibo) glDeleteBuffers(1, &g->quad_ibo);
        for (int a = 0; a < g->mp_nasset; a++) glDeleteTextures(1, &g->mp_asset[a].tex);
        eglMakeCurrent(g->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }
    if (g->display != EGL_NO_DISPLAY) {
        if (g->context != EGL_NO_CONTEXT) eglDestroyContext(g->display, g->context);
        if (g->surface != EGL_NO_SURFACE) eglDestroySurface(g->display, g->surface);
        eglTerminate(g->display);
    }
    free(g->rgba_scratch);
    for (int a = 0; a < g->mp_nasset; a++) free(g->mp_asset[a].name);
    free(g->mp_asset);
    free(g->mp_batch);
    free(g->stage_batch);
    for (int i = 0; i < g->lent_count; i++) gbm_bo_destroy(g->lent[i].bo);
    if (g->gbm_surface) gbm_surface_destroy(g->gbm_surface);
    if (g->gbm) gbm_device_destroy(g->gbm);
    if (g->gpu_fd >= 0) close(g->gpu_fd);
    if (g->drm_fd >= 0) close(g->drm_fd);
    free(g);
}

// ---- multipass -------------------------------------------------------------
// Build, commit and draw the passes of a scene, and their batches, arrays and
// assets.

static void mp_free_buffers(GLuint fbo[2], GLuint tex[2]) {
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

// Sample both buffers of a pass trilinearly, and fill their levels below 0
// from what they hold, so a read before the pass first renders finds a
// complete texture.
static void mp_mip_buffers(GLuint tex[2]) {
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
}

static void mp_free_uses(struct mp_locs *loc) {
    free(loc->uses);
    loc->uses = NULL;
    loc->nuses = loc->uses_cap = 0;
}

static void mp_free_batches(struct mp_batch *ls, int *n) {
    for (int i = 0; i < *n; i++) {
        if (ls[i].program) glDeleteProgram(ls[i].program);
        if (ls[i].vbo) glDeleteBuffers(1, &ls[i].vbo);
        if (ls[i].vao) glDeleteVertexArrays(1, &ls[i].vao);
        mp_free_uses(&ls[i].loc);
    }
    *n = 0;
}

// Free the running scene's passes, their batches and the array textures they share.
static void mp_free_passes(GheeScene *g) {
    mp_free_batches(g->mp_batch, &g->mp_nbatch);
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
    g->stage_mip[idx] = false;
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
    set_error(g, "%s takes mipmaps at %dx%d, and this GPU mipmaps a power-of-two size alone",
              g->stage_name[pass], w, h);
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
    if (g->stage_mip[pass] && !mips_fit(g, pass, w, h)) return false;
    make_current(g);
    GLuint fbo[2], tex[2];
    if (!mp_make_buffers(g, fbo, tex, w, h, g->stage_alpha[pass], g->stage_program[pass] != 0))
        return false;
    mp_free_buffers(g->stage_fbo[pass], g->stage_tex[pass]);
    memcpy(g->stage_fbo[pass], fbo, sizeof fbo);
    memcpy(g->stage_tex[pass], tex, sizeof tex);
    g->stage_w[pass] = w;
    g->stage_h[pass] = h;
    if (g->stage_mip[pass]) mp_mip_buffers(tex);
    return true;
}

bool ghee_mp_set_pass_mipmaps(GheeScene *g, int pass) {
    if (pass < 0 || pass >= g->stage_count) {
        set_error(g, "pass %d is not pending", pass);
        return false;
    }
    if (!mips_fit(g, pass, g->stage_w[pass], g->stage_h[pass])) return false;
    make_current(g);
    mp_mip_buffers(g->stage_tex[pass]);
    g->stage_mip[pass] = true;
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
    const GLuint prog = create_program(g, vertex_source, fragment_source, batch_attribs, MP_BATCH_ATTRIBS);
    if (!prog) return -1;
    struct mp_batch *l = &g->stage_batch[g->stage_nbatch];
    memset(l, 0, sizeof *l);
    l->pass = pass;
    snprintf(l->name, sizeof l->name, "%s", name);
    l->program = prog;
    l->blend = blend;
    glGenBuffers(1, &l->vbo);
    glGenVertexArrays(1, &l->vao);
    return g->stage_nbatch++;
}

// Drop the pending passes (a scene that failed to compile).
void ghee_mp_abort(GheeScene *g) {
    if (g->stage_count <= 0) return;
    make_current(g);
    mp_free_batches(g->stage_batch, &g->stage_nbatch);
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
// and set its resolution and an unflipped vflip. Program 0, a pass with no
// quad, has no uniforms.
static void mp_resolve(GheeScene *g, GLuint prog, int w, int h, struct mp_locs *loc) {
    loc->time = loc->dt = loc->frame = loc->vflip = -1;
    for (int c = 0; c < MP_MAX_CHANNELS; c++) loc->chan[c] = -1;
    for (int i = 0; i < MP_MAX_DATATEX; i++) loc->data[i] = -1;
    loc->uses = NULL;
    loc->nuses = loc->uses_cap = 0;
    if (!prog) return;
    glUseProgram(prog);
    const GLint rloc = glGetUniformLocation(prog, "resolution");
    if (rloc >= 0) glUniform2f(rloc, (float)w, (float)h);
    loc->time = glGetUniformLocation(prog, "time");
    loc->dt = glGetUniformLocation(prog, "DT");
    loc->frame = glGetUniformLocation(prog, "frame");
    for (int c = 0; c < MP_MAX_CHANNELS; c++) {
        char nm[16];
        snprintf(nm, sizeof(nm), "iChannel%d", c);
        loc->chan[c] = glGetUniformLocation(prog, nm);
    }
    for (int a = 0; a < g->mp_nasset; a++) mp_use_asset(g, prog, a, loc);
    loc->vflip = glGetUniformLocation(prog, "ghee_vflip");
    if (loc->vflip >= 0) glUniform1f(loc->vflip, 1.0f);
}

// Make the pending passes the scene: the running one is freed and replaced in a
// single step. Returns the pass count, -1 if nothing was pending (in which case
// the running scene is left alone).
int ghee_mp_commit(GheeScene *g) {
    if (g->stage_count <= 0) { set_error(g, "no pending passes to commit"); return -1; }
    make_current(g);
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
        g->mp_mip[p] = g->stage_mip[p];
        g->mp_write[p] = 0;
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
    for (int i = 0; i < g->mp_nbatch; i++) {
        struct mp_batch *l = &g->mp_batch[i];
        mp_resolve(g, l->program, g->mp_w[l->pass], g->mp_h[l->pass], &l->loc);
        l->quads = 0;
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

bool ghee_mp_set_batch_quads(GheeScene *g, int batch, const float *data, int quads, int nattr) {
    if (batch < 0 || batch >= g->mp_nbatch) { set_error(g, "no batch %d in the scene", batch); return false; }
    struct mp_batch *l = &g->mp_batch[batch];
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
        set_error(g, "%s: the upload of %d quads failed: GL 0x%X", l->name, quads, err);
        return false;
    }
    l->quads = quads;
    l->nattr = nattr;
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
    g->mp_chan_src[pass][channel] = src_pass;
    g->mp_nchan[pass] = nchan;
    return true;
}

static bool program_uniform(GLuint program, const parameter_t *param) {
    if (!program) return false;
    glUseProgram(program);
    const GLint loc = glGetUniformLocation(program, param->name);
    if (loc < 0) return false;
    switch (param->type) {
        case PARAM_TYPE_FLOAT: glUniform1f(loc, param->value.float_param->value); break;
        case PARAM_TYPE_INT:   glUniform1i(loc, param->value.int_param->value); break;
        case PARAM_TYPE_VEC2: { const float *v = param->value.vec_param->value; glUniform2f(loc, v[0], v[1]); break; }
        case PARAM_TYPE_VEC3: { const float *v = param->value.vec_param->value; glUniform3f(loc, v[0], v[1], v[2]); break; }
        case PARAM_TYPE_VEC4: { const float *v = param->value.vec_param->value; glUniform4f(loc, v[0], v[1], v[2], v[3]); break; }
        default: return false;
    }
    return true;
}

// The value reaches the pass's program and every batch of the pass.
bool ghee_mp_set_uniform(GheeScene *g, int pass, const parameter_t *param) {
    if (pass < 0 || pass >= g->mp_count) return false;
    make_current(g);
    bool any = program_uniform(g->mp_program[pass], param);
    for (int l = 0; l < g->mp_nbatch; l++)
        if (g->mp_batch[l].pass == pass) any = program_uniform(g->mp_batch[l].program, param) || any;
    return any;
}

// Set the uniform float array `name` of a committed pass. A scene's arrays
// are textures, filled by ghee_mp_set_array, since VC4 caps the size of a
// uniform array that a shader indexes with a variable.
void ghee_mp_set_float_array(GheeScene *g, int pass, const char *name, const float *data, int count) {
    if (pass < 0 || pass >= g->mp_count || !g->mp_program[pass]) return;
    make_current(g);
    glUseProgram(g->mp_program[pass]);
    const GLint loc = glGetUniformLocation(g->mp_program[pass], name);
    if (loc >= 0) glUniform1fv(loc, count, data);
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
        if (g->mp_batch[l].pass == pass)
            g->mp_batch[l].loc.data[g->mp_ndatatex[pass]] =
                glGetUniformLocation(g->mp_batch[l].program, g->mp_array_name[idx]);
    g->mp_datatex[pass][g->mp_ndatatex[pass]++] = idx;
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
        const GLint filt = g->has_float_tex && !g->has_float_linear ? GL_NEAREST : GL_LINEAR;
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
        if (glGetUniformLocation(g->mp_batch[l].program, name) >= 0) return true;
    return false;
}

// Bind a set asset to a committed batch under `name`, in place of the asset
// bound there before.
bool ghee_mp_bind_asset(GheeScene *g, int batch, const char *name, const char *asset) {
    if (batch < 0 || batch >= g->mp_nbatch) { set_error(g, "no batch %d in the scene", batch); return false; }
    struct mp_batch *l = &g->mp_batch[batch];
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
            return true;
        }
    }
    return mp_add_use(g, loc, (struct mp_use){ a, at, size });
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
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, as->tex);
    glTexSubImage2D(GL_TEXTURE_2D, level, x, y, w, h, as->fmt, as->type, data);
    return true;
}

// Free one named asset and its slot. An unknown name is a no-op.
void ghee_mp_drop_asset(GheeScene *g, const char *name) {
    const int idx = mp_asset_index(g, name);
    if (idx < 0) return;
    make_current(g);
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
    return g->has_time_override ? g->time_override
        : (now.tv_sec - g->start_time.tv_sec) + (now.tv_nsec - g->start_time.tv_nsec) / 1e9f;
}

// Set the per-frame uniforms of pass p's program or of one of its batches, in
// the program in use, and point its samplers at the units: the pass's channels,
// then its arrays, then the assets the program samples. `bind` also binds the
// pass's channel and array textures, which then stay bound for its batches.
static void mp_bind(GheeScene *g, int p, const struct mp_locs *loc, float vflip,
                    float elapsed, bool bind) {
    if (loc->vflip >= 0) glUniform1f(loc->vflip, vflip);
    if (loc->time >= 0) glUniform1f(loc->time, elapsed);
    if (loc->dt >= 0) glUniform1f(loc->dt, g->dt);
    if (loc->frame >= 0) glUniform1f(loc->frame, g->frame);
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

// Draw the batches of pass p into the bound target, after the pass's quad.
static void mp_draw_batches(GheeScene *g, int p, float vflip, float elapsed) {
    bool drew = false;
    for (int i = 0; i < g->mp_nbatch; i++) {
        const struct mp_batch *l = &g->mp_batch[i];
        if (l->pass != p || l->quads <= 0) continue;
        glUseProgram(l->program);
        mp_bind(g, p, &l->loc, vflip, elapsed, false);
        glEnable(GL_BLEND);
        switch (l->blend) {
        case GHEE_BLEND_ADD: glBlendFunc(GL_ONE, GL_ONE); break;
        case GHEE_BLEND_MULTIPLY: glBlendFunc(GL_DST_COLOR, GL_ONE_MINUS_SRC_ALPHA); break;
        case GHEE_BLEND_SCREEN: glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR); break;
        default: glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA); break;
        }
        glBindVertexArray(l->vao);
        glBindBuffer(GL_ARRAY_BUFFER, l->vbo);
        const GLsizei stride = l->nattr * 4 * (GLsizei)sizeof(GLfloat);
        for (int at = 0; at < l->quads; at += QUAD_CHUNK) {
            const int n = l->quads - at < QUAD_CHUNK ? l->quads - at : QUAD_CHUNK;
            const size_t base = (size_t)at * 4 * (size_t)stride;
            for (int k = 0; k < l->nattr; k++)
                glVertexAttribPointer(k, 4, GL_FLOAT, GL_FALSE, stride,
                                      (const GLvoid *)(base + (size_t)k * 4 * sizeof(GLfloat)));
            glDrawElements(GL_TRIANGLES, n * 6, GL_UNSIGNED_SHORT, (const GLvoid *)0);
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
// direct_last draws the last pass straight into the buffer that scans out,
// in place of its own buffer, mirrored by ghee_vflip when that buffer is the
// window surface. The caller sets it only when nothing reads the last pass
// this frame, neither a channel nor a preview.
static void render_passes(GheeScene *g, float elapsed, bool direct_last) {
    // wrap frame at a power of two
    g->frame += 1.0f;
    if (g->frame >= 1048576.0f) g->frame = 0.0f;
    for (int p = 0; p < g->mp_count; p++) {
        const bool direct = direct_last && p == g->mp_count - 1;
        glBindFramebuffer(GL_FRAMEBUFFER, direct ? g->scan_fbo : g->mp_fbo[p][g->mp_write[p]]);
        // The per-pass buffer size. The output pass is always full-size (the caller
        // enforces that), so the direct scanout draw still gets the display
        // viewport.
        glViewport(0, 0, g->mp_w[p], g->mp_h[p]);
        const float vflip = direct && !g->lent_count ? -1.0f : 1.0f;
        if (g->mp_program[p]) {
            // The fullscreen quad overwrites every pixel, so the clear is invisible, and
            // on tiled GPUs (VC4) it flags the target as not-loaded, skipping a
            // full-frame tile read from memory before the draw.
            glClear(GL_COLOR_BUFFER_BIT);
            glUseProgram(g->mp_program[p]);
            mp_bind(g, p, &g->mp_loc[p], vflip, elapsed, true);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        } else {
            // A pass of batches alone draws them over transparent black. The
            // bind leaves the channels and the arrays bound for the batches.
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            mp_bind(g, p, &g->mp_loc[p], vflip, elapsed, true);
        }
        mp_draw_batches(g, p, vflip, elapsed);
        if (g->mp_mip[p] && !direct) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, g->mp_tex[p][g->mp_write[p]]);
            glGenerateMipmap(GL_TEXTURE_2D);
        }
    }
}

// Read a rectangle back as `components` bytes a pixel. A GL that cannot hand
// back RGB reads RGBA into scratch, and the copy drops every fourth byte.
static void read_pixels(GheeScene *g, int w, int h, int components, uint8_t *out) {
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    if (components != 3 || g->reads_rgb) {
        glReadPixels(0, 0, w, h, components == 3 ? GL_RGB : GL_RGBA, GL_UNSIGNED_BYTE, out);
        return;
    }
    const size_t want = (size_t)w * h * 4;
    if (g->rgba_scratch_size < want) {
        uint8_t *grown = realloc(g->rgba_scratch, want);
        if (!grown) { set_error(g, "cannot allocate a %dx%d readback buffer", w, h); return; }
        g->rgba_scratch = grown;
        g->rgba_scratch_size = want;
    }
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, g->rgba_scratch);
    const uint8_t *src = g->rgba_scratch;
    for (size_t i = 0, n = (size_t)w * h; i < n; i++, src += 4, out += 3) {
        out[0] = src[0]; out[1] = src[1]; out[2] = src[2];
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

static bool wait_flip(GheeScene *g) {
    if (!g->flip_pending) return true;
    drmEventContext ev = { .version = 2 };
    struct pollfd pfd = { .fd = g->drm_fd, .events = POLLIN };
    for (;;) {
        const int r = poll(&pfd, 1, FLIP_STUCK_MS);
        if (r > 0) {
            drmHandleEvent(g->drm_fd, &ev);
            g->flip_pending = false;
            return true;
        }
        if (r == 0) { set_error(g, "page flip still not complete after %d ms", FLIP_STUCK_MS); return false; }
        if (errno == EINTR || errno == EAGAIN) continue;
        set_error(g, "page flip poll failed: %s", strerror(errno));
        return false;
    }
}

// The blit program (present, preview readback, encoder), built on first use.
// display mode builds it at create.
static bool ensure_blit(GheeScene *g) {
    if (g->blit_prog) return true;
    g->blit_prog = create_shader_program(g, blit_fragment_source);
    if (!g->blit_prog) {
        fprintf(stderr, "ghee: the blit program: %s\n", g->last_error);
        return false;
    }
    g->blit_sampler = glGetUniformLocation(g->blit_prog, "tex");
    g->blit_flip = glGetUniformLocation(g->blit_prog, "flipY");
    g->blit_opaque = glGetUniformLocation(g->blit_prog, "opaque");
    glUseProgram(g->blit_prog);   // ghee_vflip stays 1: the blit mirrors with flipY in its fragment shader
    const GLint bvf = glGetUniformLocation(g->blit_prog, "ghee_vflip");
    if (bvf >= 0) glUniform1f(bvf, 1.0f);
    return true;
}

// Draw the bound blit quad into a w x h target. A target at a whole multiple of
// the frame samples nearest, so each frame pixel lands as an exact square block.
static void draw_blit(GheeScene *g, int w, int h) {
    const bool blocks = g->width > 0 && g->height > 0 && w > g->width &&
                        w % g->width == 0 && h % g->height == 0 &&
                        w / g->width == h / g->height;
    if (blocks) glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    if (blocks) glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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
    glUseProgram(g->blit_prog);
    glUniform1f(g->blit_flip, 0.0f);   // preview matches glReadPixels orientation
    glUniform1f(g->blit_opaque, g->keep_alpha ? 0.0f : 1.0f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g->mp_tex[last][g->mp_write[last]]);
    glUniform1i(g->blit_sampler, 0);
    glBindVertexArray(g->VAO);
    glBindBuffer(GL_ARRAY_BUFFER, g->VBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid *)(2 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    draw_blit(g, pw, ph);
    read_pixels(g, pw, ph, g->components, buffer_rgb);
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
    wait_flip(g);
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
    bool direct = !(buffer_rgb && pw > 0 && ph > 0) && !g->keep_last;
    for (int p = 0; p < g->mp_count && direct; p++)
        for (int c = 0; c < g->mp_nchan[p]; c++)
            if (g->mp_chan_src[p][c] == last) { direct = false; break; }
    struct gbm_bo *bo = NULL;
    for (int i = 0; i < g->lent_count && !bo; i++)
        if (g->lent[i].bo != g->disp_prev && g->lent[i].bo != g->disp_shown) {
            bo = g->lent[i].bo;
            g->scan_fbo = g->lent[i].fbo;
        }
    if (g->lent_count && !bo) { set_error(g, "present: every lent scanout buffer is still on the display"); return; }
    render_passes(g, elapsed_seconds(g), direct);

    if (!direct) {
        glBindFramebuffer(GL_FRAMEBUFFER, g->scan_fbo);
        glViewport(0, 0, g->width, g->height);
        glClear(GL_COLOR_BUFFER_BIT);   // skip the tile load (see render_passes)
        glUseProgram(g->blit_prog);
        glUniform1f(g->blit_flip, g->lent_count ? 0.0f : 1.0f);
        glUniform1f(g->blit_opaque, 1.0f);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g->mp_tex[last][g->mp_write[last]]);
        glUniform1i(g->blit_sampler, 0);
        glBindVertexArray(g->VAO);
        glBindBuffer(GL_ARRAY_BUFFER, g->VBO);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid *)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid *)(2 * sizeof(GLfloat)));
        glEnableVertexAttribArray(1);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
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
    if (g->disp_plane) {
        // The mode is already set: the plane carries each frame from here, and
        // the wait inside means the buffer before this one is off the glass.
        if (g->disp_first) {
            // ghee_display_blank sets disp_first again on a relight, with the
            // CRTC already disabled. Only the first capture holds what
            // ghee_destroy restores.
            if (!g->disp_saved) g->disp_saved = drmModeGetCrtc(g->drm_fd, g->disp_crtc);
            g->disp_first = false;
        }
        if (set_scaled_plane(g, fb, g->width, g->height)) {
            release_bo(g, g->disp_shown);
            g->disp_shown = g->disp_prev;
            g->disp_prev = bo;
        } else {
            release_bo(g, bo);
        }
    } else if (g->disp_first) {
        if (!g->disp_saved) g->disp_saved = drmModeGetCrtc(g->drm_fd, g->disp_crtc);
        if (drmModeSetCrtc(g->drm_fd, g->disp_crtc, fb, 0, 0, &g->disp_conn, 1, &g->disp_mode))
            set_error(g, "drmModeSetCrtc failed: %s", strerror(errno));
        g->disp_first = false;
        g->disp_prev = bo;               // on the screen now (SetCrtc is immediate)
    } else {
        const bool waited = wait_flip(g);
        const bool flipped =
            waited && !drmModePageFlip(g->drm_fd, g->disp_crtc, fb, DRM_MODE_PAGE_FLIP_EVENT, NULL);
        const int err = errno;
        release_bo(g, ghee_flip_rotate(&g->disp_prev, &g->disp_shown, bo, flipped));
        if (flipped) {
            g->flip_pending = true;
            g->flip_failed = false;
        } else if (waited && !g->flip_failed) {  // a refused flip leaves disp_prev on the screen
            g->flip_failed = true;
            set_error(g, "drmModePageFlip failed: %s%s", strerror(err),
                      err == EACCES ? ". DRM master went to another process" : "");
        }
    }
    if (buffer_rgb && pw > 0 && ph > 0) read_preview(g, buffer_rgb, pw, ph);
    for (int p = 0; p < g->mp_count; p++) g->mp_write[p] ^= 1;
}

// Render every pass and swap. buffer_rgb NULL renders nothing back to the CPU,
// pw/ph > 0 read back a GPU-downscaled preview instead of the full frame.
void ghee_mp_render(GheeScene *g, uint8_t *buffer_rgb, int pw, int ph) {
    if (g->mp_count <= 0) { set_error(g, "render: the scene has no passes"); return; }
    make_current(g);
    struct timespec now, t_draw, t_read;
    clock_gettime(CLOCK_MONOTONIC, &now);
    render_passes(g, elapsed_seconds(g), false);

    // Profiling: force the GPU to finish the passes so the readback timing below
    // is pure readback.
    if (g->profile) { glFinish(); clock_gettime(CLOCK_MONOTONIC, &t_draw); }

    const int last = g->mp_count - 1;
    if (buffer_rgb && pw > 0 && ph > 0) {
        read_preview(g, buffer_rgb, pw, ph);
    } else if (buffer_rgb) {
        glBindFramebuffer(GL_FRAMEBUFFER, g->mp_fbo[last][g->mp_write[last]]);
        read_pixels(g, g->width, g->height, g->components, buffer_rgb);
        // An RGB buffer reads back alpha 255 by itself. A final pass that
        // stores its alpha takes 255 here.
        if (g->components == 4 && g->mp_alpha[last] && !g->keep_alpha) {
            uint8_t *a = buffer_rgb + 3;
            for (size_t i = 0, n = (size_t)g->width * g->height; i < n; i++, a += 4) *a = 255;
        }
    }

    if (g->profile) {
        clock_gettime(CLOCK_MONOTONIC, &t_read);
        g->prof_draw_ms = ms_between(now, t_draw);   // GPU render (passes), fenced
        g->prof_read_ms = ms_between(t_draw, t_read); // pure glReadPixels
    }

    for (int p = 0; p < g->mp_count; p++) g->mp_write[p] ^= 1;  // swap all
}

// The GL this renderer runs on: GL_RENDERER and GL_VERSION in one string.
const char *ghee_renderer(GheeScene *g) { return g ? g->renderer : ""; }

void ghee_set_profiling(GheeScene *g, bool on) { g->profile = on; }

void ghee_set_time(GheeScene *g, float t) { g->time_override = t; g->has_time_override = true; }

// Back to the monotonic clock after a ghee_set_time.
void ghee_clear_time(GheeScene *g) { g->has_time_override = false; }

void ghee_set_dt(GheeScene *g, float dt) { g->dt = dt; }

void ghee_keep_alpha(GheeScene *g, int keep) { g->keep_alpha = keep; }

int ghee_max_texture_units(GheeScene *g) { return g ? (int)g->max_units : 0; }

int ghee_gl_texture_units(GheeScene *g) { return g ? (int)g->gl_units : 0; }

// Whether this GL linear-filters a data texture. ghee_mp_set_array picks its
// filter from the same condition. False on V3D, which has float textures but
// no OES_texture_float_linear, so a filtered fetch is nearest.
bool ghee_data_texture_linear(GheeScene *g) { return g && !(g->has_float_tex && !g->has_float_linear); }

bool ghee_mipmaps_npot(GheeScene *g) { return g && g->has_npot; }

// The display mode's refresh rate in Hz, 0 outside display mode.
int ghee_display_vrefresh(GheeScene *g) { return g ? (int)g->disp_mode.vrefresh : 0; }

// ---- internal seams for encoder.c (engine_internal.h) ---------------------

void *ghee_i_egl_display(GheeScene *g) { return g->display; }
void  ghee_i_make_current(GheeScene *g) { make_current(g); }
void  ghee_i_error(GheeScene *g, const char *msg) { set_error(g, "%s", msg); }

void ghee_i_blit_last(GheeScene *g, unsigned fbo, int w, int h, float flipY) {
    if (g->mp_count <= 0) { set_error(g, "blit: the scene has no passes"); return; }
    if (!ensure_blit(g)) return;
    const int last = g->mp_count - 1;
    const int buf = 1 - g->mp_write[last];
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, w, h);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(g->blit_prog);
    glUniform1f(g->blit_flip, flipY);
    glUniform1f(g->blit_opaque, 1.0f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g->mp_tex[last][buf]);
    glUniform1i(g->blit_sampler, 0);
    glBindVertexArray(g->VAO);
    glBindBuffer(GL_ARRAY_BUFFER, g->VBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid *)(2 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    draw_blit(g, w, h);
}

void ghee_i_keep_last(GheeScene *g, int delta) {
    g->keep_last += delta;
    if (g->keep_last < 0) g->keep_last = 0;
}

void ghee_get_timings(GheeScene *g, double *draw_ms, double *read_ms) {
    if (draw_ms) *draw_ms = g->prof_draw_ms;
    if (read_ms) *read_ms = g->prof_read_ms;
}

// Tear down the running scene and anything pending (for switching scenes).
// Keeps the GL context.
void ghee_mp_reset(GheeScene *g) {
    make_current(g);
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
    int hw = 1;
    struct gbm_device *gbm = NULL;
    EGLDisplay dpy = EGL_NO_DISPLAY;
    EGLContext ctx = EGL_NO_CONTEXT;
    const int fd = open(dev, O_RDWR | O_CLOEXEC);
    if (fd < 0) return 1;
    gbm = gbm_create_device(fd);
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
    close(fd);
    return hw;
}

// Pack an RGB(A) frame into a compact output format: 1-bit column-major (bw)
// or RGB565 big-endian (color). Standalone so the host can pack the output
// buffer after a multipass render.
void ghee_pack_frame(int width, int height, int components, int bw,
                    const uint8_t *rgb, uint8_t *packed) {
    if (bw) {
        for (size_t row = 0; row < (size_t)height; row++) {
            const size_t row_offset = (row >> 3) * (size_t)width;
            const uint8_t bit_mask = 1 << (row & 7);
            for (size_t col = 0; col < (size_t)width; col++) {
                const size_t pi = (row * width + col) * components;
                const uint8_t gray = (rgb[pi] * 30 + rgb[pi + 1] * 59 + rgb[pi + 2] * 11) / 100;
                if (gray > 128) packed[row_offset + col] |= bit_mask;
            }
        }
    } else {
        for (size_t i = 0; i < (size_t)width * height; i++) {
            const uint16_t r = rgb[i * components + 0];
            const uint16_t gg = rgb[i * components + 1];
            const uint16_t b = rgb[i * components + 2];
            const uint16_t v = ((r & 0xF8) << 8) | ((gg & 0xFC) << 3) | (b >> 3);
            packed[i * 2 + 0] = (v >> 8) & 0xFF;
            packed[i * 2 + 1] = v & 0xFF;
        }
    }
}
