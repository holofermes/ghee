#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
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
#include <fcntl.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <poll.h>
#include <errno.h>
#include <sys/mman.h>
#include "engine.h"
#include "engine_internal.h"

#define ERROR_BUFFER_SIZE 512
#define DEFAULT_DRI_DEVICE "/dev/dri/card0"
#define MP_MAX_PASSES 8
#define MP_MAX_CHANNELS 8
#define MP_MAX_DATATEX 4
#define MP_MAX_ARRAYS 16
#define FLIP_STUCK_MS 5000

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
    float dt;                            // seconds the last frame took; fed to every pass as DT

    // multipass: each pass has its own program + ping-pong buffer; channels route
    // between passes (reading a pass's PREVIOUS-frame output). A pass whose channel
    // points at itself = feedback. Last pass -> screen.
    int mp_count;
    GLuint mp_program[MP_MAX_PASSES];
    GLuint mp_fbo[MP_MAX_PASSES][2];
    GLuint mp_tex[MP_MAX_PASSES][2];
    int mp_w[MP_MAX_PASSES], mp_h[MP_MAX_PASSES];   // per-pass buffer size; defaults to W,H
    int mp_write[MP_MAX_PASSES];
    int mp_chan_src[MP_MAX_PASSES][MP_MAX_CHANNELS];  // channel -> source pass index, -1 = none
    int mp_nchan[MP_MAX_PASSES];
    GLint mp_timeLoc[MP_MAX_PASSES];
    GLint mp_dtLoc[MP_MAX_PASSES];
    GLint mp_vflipLoc[MP_MAX_PASSES];    // shared-VS y-flip (direct scanout draws)
    GLint max_units;                     // GL_MAX_TEXTURE_IMAGE_UNITS (8 on VC4)

    // pending scene: passes compiled and allocated beside the running ones and
    // made the scene by ghee_mp_commit, so a compile error leaves the running
    // scene rendering.
    int stage_count;
    GLuint stage_program[MP_MAX_PASSES];
    GLuint stage_fbo[MP_MAX_PASSES][2];
    GLuint stage_tex[MP_MAX_PASSES][2];
    int stage_w[MP_MAX_PASSES], stage_h[MP_MAX_PASSES];

    // data textures: a float array bound as a count x rows sampler (<name>_tex) so a
    // computed index samples it in one fetch, dodging the VC4 uniform-array unroll cap.
    // One texture per array NAME, at scene scope: uploaded once a frame and bound
    // into every pass that declared it.
    bool has_float_tex, has_float_linear;   // OES_texture_float[_linear] on this GL
    int mp_narray;
    GLuint mp_array_tex[MP_MAX_ARRAYS];
    char mp_array_name[MP_MAX_ARRAYS][64];        // sampler uniform name
    int mp_array_size[MP_MAX_ARRAYS];
    int mp_ndatatex[MP_MAX_PASSES];
    int mp_datatex[MP_MAX_PASSES][MP_MAX_DATATEX];   // texture unit -> array index
    GLuint black_tex;                    // what a channel with no source samples

    bool profile;                        // when on, ghee_mp_render splits draw vs readback (adds a glFinish)
    double prof_draw_ms, prof_read_ms;   // last frame's split (profiling mode)

    // display mode: scan the final buffer straight out to a KMS connector
    // (zero CPU copy), instead of reading it back. Set up by ghee_display_create.
    bool display_mode;
    uint32_t disp_conn, disp_crtc;
    drmModeModeInfo disp_mode;
    drmModeCrtc *disp_saved;
    struct gbm_bo *disp_prev;    // buffer of the most recent flip (pending or on glass)
    struct gbm_bo *disp_shown;   // buffer of the flip before that (still on glass until
                                 // the pending one completes; released after)
    bool disp_first;
    bool disp_blanked;           // the CRTC drives nothing: ghee_mp_present draws nothing
    bool flip_pending;           // a submitted page flip whose event hasn't been reaped
    bool flip_failed;            // a refused flip, reported once instead of every frame
    GLuint blit_prog;
    GLint blit_sampler, blit_flip;
    GLuint preview_fbo, preview_tex;   // small FBO for a cheap downscaled readback
    int preview_w, preview_h;
    int keep_last;                     // >0: an encoder samples the final pass, never draw it direct

    char last_error[ERROR_BUFFER_SIZE];
};

static double ms_between(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1000.0 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

// vflip = -1 renders geometry y-mirrored: used when a pass draws straight into
// the scanout buffer (GL is bottom-origin, the display scans top-first). The
// fragment's v_texCoord is untouched, so user-shader semantics don't change.
static const char *vertex_shader_source =
    "#version 100\n"
    "attribute vec4 position;"
    "attribute vec2 texCoord;"
    "uniform float vflip;"
    "varying vec2 v_texCoord;"
    "void main() {"
    "   gl_Position = vec4(position.x, position.y * vflip, position.z, position.w);"
    "   v_texCoord = texCoord;"
    "}";

// Present blit: sample the final pass texture onto the window surface. Flips Y
// because GL framebuffers are bottom-origin and the display scans out top-first.
static const char *blit_fragment_source =
    "#version 100\n"
    "precision mediump float;"
    "varying vec2 v_texCoord;"
    "uniform sampler2D tex;"
    "uniform float flipY;"
    "void main() { vec2 uv = vec2(v_texCoord.x, mix(v_texCoord.y, 1.0 - v_texCoord.y, flipY));"
    " gl_FragColor = texture2D(tex, uv); }";

static bool wait_flip(GheeScene *g);        // both defined with ghee_mp_present below
static bool ensure_blit(GheeScene *g);

static void set_error(GheeScene *g, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g->last_error, ERROR_BUFFER_SIZE, fmt, ap);
    va_end(ap);
    g->last_error[ERROR_BUFFER_SIZE - 1] = '\0';
    fprintf(stderr, "ghee: %s\n", g->last_error);
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
        set_error(g, "%s", info);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint create_shader_program(GheeScene *g, const char *fragment_shader_source) {
    const GLuint vs = create_shader(g, GL_VERTEX_SHADER, vertex_shader_source);
    if (!vs) return 0;
    const GLuint fs = create_shader(g, GL_FRAGMENT_SHADER, fragment_shader_source);
    if (!fs) { glDeleteShader(vs); return 0; }

    const GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    GLint ok;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char info[ERROR_BUFFER_SIZE];
        glGetProgramInfoLog(program, ERROR_BUFFER_SIZE, NULL, info);
        set_error(g, "%s", info);
        glDeleteShader(vs);
        glDeleteShader(fs);
        glDeleteProgram(program);
        return 0;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

// ---- lifecycle ------------------------------------------------------------

static GheeScene *create_impl(int width, int height, int components, const char *dri_device, int scanout, int rgb565, char **err) {
    static char early_err[ERROR_BUFFER_SIZE];
    early_err[0] = '\0';

    GheeScene *g = calloc(1, sizeof(GheeScene));
    if (!g) { if (err) *err = "out of memory"; return NULL; }
    g->width = width;
    g->height = height;
    g->components = components;
    g->drm_fd = -1;
    g->display = EGL_NO_DISPLAY;

    const char *dev = dri_device ? dri_device : DEFAULT_DRI_DEVICE;
    g->drm_fd = open(dev, O_RDWR | O_CLOEXEC);
    if (g->drm_fd < 0) {
        snprintf(early_err, sizeof(early_err), "cannot open DRI device %s", dev);
        goto fail;
    }
    g->gbm = gbm_create_device(g->drm_fd);
    if (!g->gbm) { snprintf(early_err, sizeof(early_err), "gbm_create_device failed"); goto fail; }

    g->display = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, g->gbm, NULL);
    if (g->display == EGL_NO_DISPLAY) { snprintf(early_err, sizeof(early_err), "eglGetPlatformDisplay failed"); goto fail; }

    int major, minor;
    if (eglInitialize(g->display, &major, &minor) == EGL_FALSE) {
        snprintf(early_err, sizeof(early_err), "eglInitialize failed: 0x%X", eglGetError());
        goto fail;
    }
    fprintf(stderr, "ghee: EGL %d.%d on %s\n", major, minor, dev);

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
        // no MSAA: ghee only draws fullscreen quads, so multisampling has zero
        // visual effect and (on VC4) quadruples the per-pixel tile work
        EGL_SAMPLE_BUFFERS, 0,
        EGL_SAMPLES, 0,
        EGL_NONE
    };
    // Optional (rgb565): display scanout renders RGB565, halving the bytes
    // through the GPU's tile-store path (~300MB/s effective on VC4-class
    // boards, the frame rate ceiling) and halving scanout traffic. The HVS
    // expands 565 to the wire's 888 during scanout, so the panel sees a
    // normal HDMI signal; the only cost is color depth. GL dithering (on by
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
        // tile work for zero visual gain on fullscreen-quad scenes
        EGL_SAMPLE_BUFFERS, 0,
        EGL_SAMPLES, 0,
        EGL_NONE
    };
    EGLConfig configs[64];
    EGLConfig config = NULL;
    EGLint numConfigs = 0;
    uint32_t surf_fmt = GBM_FORMAT_XRGB8888;
    if (scanout && rgb565 &&
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
        fprintf(stderr, "ghee: scanout format %s\n",
                surf_fmt == GBM_FORMAT_RGB565 ? "RGB565" : "XRGB8888");

    g->gbm_surface = gbm_surface_create(g->gbm, width, height, surf_fmt,
        GBM_BO_USE_RENDERING | (scanout ? GBM_BO_USE_SCANOUT : 0));
    if (!g->gbm_surface) { snprintf(early_err, sizeof(early_err), "gbm_surface_create failed"); goto fail; }

    g->surface = eglCreatePlatformWindowSurface(g->display, config, g->gbm_surface, NULL);
    if (g->surface == EGL_NO_SURFACE) {
        // Desktop Mesa rejects the GBM window surface (EGL_BAD_MATCH) that the Pi's
        // VC4 accepts. Rendering goes to FBOs, so fall back to a surfaceless context.
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

    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &g->max_units);
    if (g->max_units < 1) g->max_units = 8;

    const char *glexts = (const char *)glGetString(GL_EXTENSIONS);
    g->has_float_tex = glexts && strstr(glexts, "GL_OES_texture_float") != NULL;
    g->has_float_linear = glexts && strstr(glexts, "GL_OES_texture_float_linear") != NULL;
    fprintf(stderr, "ghee: data-texture format = %s%s\n",
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
struct bo_fb { int drm_fd; uint32_t fb_id; };
static void bo_destroy_fb(struct gbm_bo *bo, void *data) {
    (void)bo;
    struct bo_fb *f = data;
    if (f) { if (f->fb_id) drmModeRmFB(f->drm_fd, f->fb_id); free(f); }
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

// Pick the display to drive: first connected connector with modes, its
// encoder's CRTC (else the first), and the mode that CRTC already drives
// (honours a video= override or whatever the console negotiated; falls back
// to the connector's first). Shared by create and probe so the size the probe
// reports is exactly the mode create ends up driving. -1 = nothing connected.
static int pick_display_mode(int fd, uint32_t *conn_out, uint32_t *crtc_out,
                             drmModeModeInfo *mode_out) {
    drmModeRes *res = drmModeGetResources(fd);
    if (!res) return -1;
    drmModeConnector *conn = NULL;
    for (int i = 0; i < res->count_connectors; i++) {
        drmModeConnector *c = drmModeGetConnector(fd, res->connectors[i]);
        if (c && c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) { conn = c; break; }
        if (c) drmModeFreeConnector(c);
    }
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
    fprintf(stderr, "ghee: display %s %ux%u@%u on connector %u\n",
            g->disp_mode.name, g->disp_mode.hdisplay, g->disp_mode.vdisplay,
            g->disp_mode.vrefresh, g->disp_conn);

    make_current(g);
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
        if (g->disp_saved)
            drmModeSetCrtc(g->drm_fd, g->disp_saved->crtc_id, g->disp_saved->buffer_id,
                           g->disp_saved->x, g->disp_saved->y, &g->disp_conn, 1, &g->disp_saved->mode);
        if (g->disp_shown) gbm_surface_release_buffer(g->gbm_surface, g->disp_shown);
        if (g->disp_prev) gbm_surface_release_buffer(g->gbm_surface, g->disp_prev);
        drmDropMaster(g->drm_fd);
    }
    if (have_gl) {
        if (g->preview_fbo) { glDeleteFramebuffers(1, &g->preview_fbo); glDeleteTextures(1, &g->preview_tex); }
        if (g->blit_prog) glDeleteProgram(g->blit_prog);
        if (g->VBO) glDeleteBuffers(1, &g->VBO);
        if (g->VAO) glDeleteVertexArrays(1, &g->VAO);
        ghee_mp_reset(g);   // all pass programs/buffers + the pending scene
        eglMakeCurrent(g->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }
    if (g->display != EGL_NO_DISPLAY) {
        if (g->context != EGL_NO_CONTEXT) eglDestroyContext(g->display, g->context);
        if (g->surface != EGL_NO_SURFACE) eglDestroySurface(g->display, g->surface);
        eglTerminate(g->display);
    }
    if (g->gbm_surface) gbm_surface_destroy(g->gbm_surface);
    if (g->gbm) gbm_device_destroy(g->gbm);
    if (g->drm_fd >= 0) close(g->drm_fd);
    free(g);
}

// ---- multipass -------------------------------------------------------------
// A scene as a list of passes. Each pass: own program + ping-pong buffer. Channels
// route between passes: a forward ref (earlier pass) reads that pass's CURRENT
// frame (same-frame chaining); a self-reference or later-pass ref reads the
// PREVIOUS frame (= feedback). Last pass -> screen.

static void mp_free_buffers(GLuint fbo[2], GLuint tex[2]) {
    glDeleteFramebuffers(2, fbo);
    glDeleteTextures(2, tex);
    fbo[0] = fbo[1] = tex[0] = tex[1] = 0;
}

static bool mp_make_buffers(GheeScene *g, GLuint fbo[2], GLuint tex[2], int w, int h) {
    while (glGetError() != GL_NO_ERROR) { }
    glGenFramebuffers(2, fbo);
    glGenTextures(2, tex);
    for (int i = 0; i < 2; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo[i]);
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
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
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return true;
}

// A channel with no source samples this instead of texture unit 0, which holds
// whatever the last bind left there.
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

// Free the running scene's passes and the array textures they share.
static void mp_free_passes(GheeScene *g) {
    for (int p = 0; p < g->mp_count; p++) {
        if (g->mp_program[p]) glDeleteProgram(g->mp_program[p]);
        mp_free_buffers(g->mp_fbo[p], g->mp_tex[p]);
        g->mp_ndatatex[p] = 0;
    }
    if (g->mp_narray > 0) glDeleteTextures(g->mp_narray, g->mp_array_tex);
    g->mp_narray = 0;
    g->mp_count = 0;
    if (g->black_tex) { glDeleteTextures(1, &g->black_tex); g->black_tex = 0; }
}

// Compile a pass into the PENDING scene: program and buffers are allocated
// beside the running scene, which keeps rendering until ghee_mp_commit. Returns
// the pass index (the same index the committed scene uses) or -1.
int ghee_mp_add_pass(GheeScene *g, const char *glsl_source) {
    make_current(g);
    if (g->stage_count >= MP_MAX_PASSES) { set_error(g, "too many passes"); return -1; }
    const GLuint prog = create_shader_program(g, glsl_source);
    if (!prog) return -1;
    const int idx = g->stage_count;
    if (!mp_make_buffers(g, g->stage_fbo[idx], g->stage_tex[idx], g->width, g->height)) {
        glDeleteProgram(prog);
        return -1;
    }
    g->stage_program[idx] = prog;
    g->stage_w[idx] = g->width;
    g->stage_h[idx] = g->height;
    g->stage_count++;
    return idx;
}

// Give a pending pass its own buffer size (default = renderer W,H). Call after
// ghee_mp_add_pass, before ghee_mp_commit. The last (output) pass must keep the
// renderer size (its buffer feeds readback/scanout); Renderer::setup rejects a
// sized output pass before it reaches here.
bool ghee_mp_set_pass_size(GheeScene *g, int pass, int w, int h) {
    if (pass < 0 || pass >= g->stage_count || w <= 0 || h <= 0) {
        set_error(g, "cannot size pass %d at %dx%d", pass, w, h);
        return false;
    }
    make_current(g);
    GLuint fbo[2], tex[2];
    if (!mp_make_buffers(g, fbo, tex, w, h)) return false;
    mp_free_buffers(g->stage_fbo[pass], g->stage_tex[pass]);
    memcpy(g->stage_fbo[pass], fbo, sizeof fbo);
    memcpy(g->stage_tex[pass], tex, sizeof tex);
    g->stage_w[pass] = w;
    g->stage_h[pass] = h;
    return true;
}

// Drop the pending passes (a scene that failed to compile).
void ghee_mp_abort(GheeScene *g) {
    if (g->stage_count <= 0) return;
    make_current(g);
    for (int p = 0; p < g->stage_count; p++) {
        glDeleteProgram(g->stage_program[p]);
        mp_free_buffers(g->stage_fbo[p], g->stage_tex[p]);
    }
    g->stage_count = 0;
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
        g->mp_write[p] = 0;
        g->mp_nchan[p] = 0;
        g->mp_ndatatex[p] = 0;
        for (int c = 0; c < MP_MAX_CHANNELS; c++) g->mp_chan_src[p][c] = -1;
        glUseProgram(prog);
        const GLint rloc = glGetUniformLocation(prog, "resolution");
        if (rloc >= 0) glUniform2f(rloc, (float)g->mp_w[p], (float)g->mp_h[p]);
        g->mp_timeLoc[p] = glGetUniformLocation(prog, "time");
        g->mp_dtLoc[p] = glGetUniformLocation(prog, "DT");
        g->mp_vflipLoc[p] = glGetUniformLocation(prog, "vflip");
        if (g->mp_vflipLoc[p] >= 0) glUniform1f(g->mp_vflipLoc[p], 1.0f);
    }
    g->mp_count = g->stage_count;
    g->stage_count = 0;
    return g->mp_count;
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
    if (nchan + g->mp_ndatatex[pass] > g->max_units) {
        set_error(g, "pass %d wants %d texture units, this GL has %d",
                  pass, nchan + g->mp_ndatatex[pass], (int)g->max_units);
        return false;
    }
    g->mp_chan_src[pass][channel] = src_pass;
    g->mp_nchan[pass] = nchan;
    return true;
}

bool ghee_mp_set_uniform(GheeScene *g, int pass, const parameter_t *param) {
    if (pass < 0 || pass >= g->mp_count) return false;
    make_current(g);
    glUseProgram(g->mp_program[pass]);
    const GLint loc = glGetUniformLocation(g->mp_program[pass], param->name);
    if (loc < 0) return false;
    switch (param->type) {
        case PARAM_TYPE_FLOAT: glUniform1f(loc, param->value.float_param->value); break;
        case PARAM_TYPE_INT:   glUniform1i(loc, param->value.int_param->value); break;
        case PARAM_TYPE_VEC2:  glUniform2fv(loc, 1, param->value.vec2_param->values); break;
        case PARAM_TYPE_VEC3:  glUniform3fv(loc, 1, param->value.vec3_param->values); break;
        case PARAM_TYPE_VEC4:  glUniform4fv(loc, 1, param->value.vec4_param->values); break;
        default: return false;  // textures are channels, not params
    }
    return true;
}

// Unused: superseded by ghee_mp_set_array.
void ghee_mp_set_float_array(GheeScene *g, int pass, const char *name, const float *data, int count) {
    if (pass < 0 || pass >= g->mp_count) return;
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

// Declare an array NAME on a pass. One texture per name at scene scope, bound
// into this pass's next free unit (data textures follow the channels), so the
// data uploads once a frame however many passes read it. Returns the array
// index, -1 on refusal. Call after ghee_mp_commit, before the first render.
int ghee_mp_declare_array(GheeScene *g, int pass, const char *name) {
    if (pass < 0 || pass >= g->mp_count) { set_error(g, "no pass %d to declare '%s' on", pass, name); return -1; }
    char uni[64];
    snprintf(uni, sizeof(uni), "%s_tex", name);
    int idx = mp_array_index(g, uni);
    for (int i = 0; idx >= 0 && i < g->mp_ndatatex[pass]; i++)
        if (g->mp_datatex[pass][i] == idx) return idx;
    if (g->mp_ndatatex[pass] >= MP_MAX_DATATEX) {
        set_error(g, "pass %d declares more than %d arrays", pass, MP_MAX_DATATEX);
        return -1;
    }
    if (g->mp_nchan[pass] + g->mp_ndatatex[pass] + 1 > g->max_units) {
        set_error(g, "pass %d wants %d texture units, this GL has %d",
                  pass, g->mp_nchan[pass] + g->mp_ndatatex[pass] + 1, (int)g->max_units);
        return -1;
    }
    if (idx < 0) {
        if (g->mp_narray >= MP_MAX_ARRAYS) { set_error(g, "more than %d arrays in the scene", MP_MAX_ARRAYS); return -1; }
        make_current(g);
        idx = g->mp_narray++;
        snprintf(g->mp_array_name[idx], 64, "%s", uni);
        glGenTextures(1, &g->mp_array_tex[idx]);
        g->mp_array_size[idx] = 0;
    }
    g->mp_datatex[pass][g->mp_ndatatex[pass]++] = idx;
    return idx;
}

// Bind one declared array's texture, and set the filters on a resize.
// -1 = no pass declares the name. *fresh means the texture needs a full
// glTexImage2D, and not a glTexSubImage2D.
static int mp_array_bind(GheeScene *g, const char *name, int count, int rows, bool *fresh) {
    char uni[64];
    snprintf(uni, sizeof(uni), "%s_tex", name);
    const int idx = mp_array_index(g, uni);
    if (idx < 0) { set_error(g, "array '%s' is declared by no pass", name); return -1; }
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
// array carries the type has_float_tex picked.
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

static float elapsed_seconds(GheeScene *g) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return g->has_time_override ? g->time_override
        : (now.tv_sec - g->start_time.tv_sec) + (now.tv_nsec - g->start_time.tv_nsec) / 1e9f;
}

// Render every pass into its FBO (shared by readback and display present).
// direct_last: draw the FINAL pass straight into the current backbuffer
// (display scanout) instead of its FBO texture, y-flipped via the shared VS.
// Only valid when nothing samples the last pass's output (no channel routes to
// it and no preview readback this frame) -- the caller guarantees that.
static void render_passes(GheeScene *g, float elapsed, bool direct_last) {
    for (int p = 0; p < g->mp_count; p++) {
        const bool direct = direct_last && p == g->mp_count - 1;
        glBindFramebuffer(GL_FRAMEBUFFER, direct ? 0 : g->mp_fbo[p][g->mp_write[p]]);
        // per-pass buffer size; the output pass is always full-size (the caller
        // enforces that), so the direct scanout draw still gets the display
        // viewport.
        glViewport(0, 0, g->mp_w[p], g->mp_h[p]);
        // The fullscreen quad overwrites every pixel, so the clear is invisible;
        // on tiled GPUs (VC4) it flags the target as not-loaded, skipping a
        // full-frame tile read from memory before the draw.
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(g->mp_program[p]);
        if (g->mp_vflipLoc[p] >= 0) glUniform1f(g->mp_vflipLoc[p], direct ? -1.0f : 1.0f);
        if (g->mp_timeLoc[p] >= 0) glUniform1f(g->mp_timeLoc[p], elapsed);
        if (g->mp_dtLoc[p] >= 0) glUniform1f(g->mp_dtLoc[p], g->dt);
        for (int c = 0; c < g->mp_nchan[p]; c++) {
            const int src = g->mp_chan_src[p][c];
            glActiveTexture(GL_TEXTURE0 + c);
            if (src < 0) {
                glBindTexture(GL_TEXTURE_2D, g->black_tex);
            } else {
                const int rd = (src < p) ? g->mp_write[src] : 1 - g->mp_write[src];
                glBindTexture(GL_TEXTURE_2D, g->mp_tex[src][rd]);
            }
            char nm[16];
            snprintf(nm, sizeof(nm), "iChannel%d", c);
            const GLint loc = glGetUniformLocation(g->mp_program[p], nm);
            if (loc >= 0) glUniform1i(loc, c);
        }
        for (int i = 0; i < g->mp_ndatatex[p]; i++) {
            const int unit = g->mp_nchan[p] + i;
            const int a = g->mp_datatex[p][i];
            glActiveTexture(GL_TEXTURE0 + unit);
            glBindTexture(GL_TEXTURE_2D, g->mp_array_tex[a]);
            const GLint loc = glGetUniformLocation(g->mp_program[p], g->mp_array_name[a]);
            if (loc >= 0) glUniform1i(loc, unit);
        }
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
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

// The blit program (present, preview readback, encoder), built on first use;
// display mode builds it at create.
static bool ensure_blit(GheeScene *g) {
    if (g->blit_prog) return true;
    g->blit_prog = create_shader_program(g, blit_fragment_source);
    if (!g->blit_prog) return false;
    g->blit_sampler = glGetUniformLocation(g->blit_prog, "tex");
    g->blit_flip = glGetUniformLocation(g->blit_prog, "flipY");
    glUseProgram(g->blit_prog);   // shared VS: neutral vflip (blit flips in the fragment)
    const GLint bvf = glGetUniformLocation(g->blit_prog, "vflip");
    if (bvf >= 0) glUniform1f(bvf, 1.0f);
    return true;
}

// Downscale the final pass into a small FBO on the GPU. The FBO is kept until
// the requested size changes.
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
            set_error(g, "%dx%d preview buffer failed: framebuffer 0x%X", pw, ph, status);
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
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, pw, ph, fmt, GL_UNSIGNED_BYTE, buffer_rgb);
}

// Render + scan the final buffer straight out to the connector. If buffer_rgb
// is non-NULL, also read the frame back (for a console MJPEG mirror).
//
// Two throughput choices matter on VC4-class memory:
//  * direct scanout: when nothing samples the last pass (no channel routes to
//    it, no preview this frame), it draws straight into the backbuffer,
//    skipping a full-res texture store + blit (2/3 of the frame's traffic).
//  * pipelined flip: the flip is submitted and reaped on the NEXT present, so
//    feed/render overlap scanout instead of serializing with the vblank.
// Blank the display, or light it again. Blanking stops the CRTC driving the
// connector, which runs the kernel driver's own disable path: a panel turns
// its display off and drops its backlight. The next present after a light
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
    if (g->disp_shown) { gbm_surface_release_buffer(g->gbm_surface, g->disp_shown); g->disp_shown = NULL; }
    if (g->disp_prev) { gbm_surface_release_buffer(g->gbm_surface, g->disp_prev); g->disp_prev = NULL; }
    g->disp_blanked = true;
    return 0;
}

int ghee_display_blanked(GheeScene *g) { return g && g->disp_blanked; }

void ghee_mp_present(GheeScene *g, uint8_t *buffer_rgb, int pw, int ph) {
    if (!g->display_mode || g->disp_blanked) return;
    if (g->mp_count <= 0) { set_error(g, "present: the scene has no passes"); return; }
    make_current(g);
    const int last = g->mp_count - 1;
    bool direct = !(buffer_rgb && pw > 0 && ph > 0) && !g->keep_last;
    for (int p = 0; p < g->mp_count && direct; p++)
        for (int c = 0; c < g->mp_nchan[p]; c++)
            if (g->mp_chan_src[p][c] == last) { direct = false; break; }
    render_passes(g, elapsed_seconds(g), direct);

    if (!direct) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, g->width, g->height);
        glClear(GL_COLOR_BUFFER_BIT);   // skip the tile load (see render_passes)
        glUseProgram(g->blit_prog);
        glUniform1f(g->blit_flip, 1.0f);   // scanout: GL bottom-origin -> top-first display
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
    eglSwapBuffers(g->display, g->surface);

    struct gbm_bo *bo = gbm_surface_lock_front_buffer(g->gbm_surface);
    uint32_t fb = fb_for(g, bo);
    if (g->disp_first) {
        g->disp_saved = drmModeGetCrtc(g->drm_fd, g->disp_crtc);
        if (drmModeSetCrtc(g->drm_fd, g->disp_crtc, fb, 0, 0, &g->disp_conn, 1, &g->disp_mode))
            set_error(g, "drmModeSetCrtc failed: %s", strerror(errno));
        g->disp_first = false;
        g->disp_prev = bo;               // on glass now (SetCrtc is immediate)
    } else if (!wait_flip(g)) {
        gbm_surface_release_buffer(g->gbm_surface, bo);
    } else {                             // the previous flip's buffer is on glass
        if (g->disp_shown) gbm_surface_release_buffer(g->gbm_surface, g->disp_shown);
        g->disp_shown = g->disp_prev;    // displaced once the flip below completes
        if (!drmModePageFlip(g->drm_fd, g->disp_crtc, fb, DRM_MODE_PAGE_FLIP_EVENT, NULL)) {
            g->flip_pending = true;
            g->flip_failed = false;
        } else if (!g->flip_failed) {
            g->flip_failed = true;
            set_error(g, "drmModePageFlip failed: %s%s", strerror(errno),
                      errno == EACCES ? ". DRM master went to another process" : "");
        }
        g->disp_prev = bo;
    }
    if (buffer_rgb && pw > 0 && ph > 0) read_preview(g, buffer_rgb, pw, ph);
    for (int p = 0; p < g->mp_count; p++) g->mp_write[p] ^= 1;
}

// Render every pass and swap. buffer_rgb NULL renders nothing back to the CPU;
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
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, g->width, g->height, (g->components == 3) ? GL_RGB : GL_RGBA, GL_UNSIGNED_BYTE, buffer_rgb);
    }

    if (g->profile) {
        clock_gettime(CLOCK_MONOTONIC, &t_read);
        g->prof_draw_ms = ms_between(now, t_draw);   // GPU render (passes), fenced
        g->prof_read_ms = ms_between(t_draw, t_read); // pure glReadPixels
    }

    for (int p = 0; p < g->mp_count; p++) g->mp_write[p] ^= 1;  // swap all
}

void ghee_set_profiling(GheeScene *g, bool on) { g->profile = on; }

void ghee_set_time(GheeScene *g, float t) { g->time_override = t; g->has_time_override = true; }

// Back to the monotonic clock after a ghee_set_time.
void ghee_clear_time(GheeScene *g) { g->has_time_override = false; }

void ghee_set_dt(GheeScene *g, float dt) { g->dt = dt; }

// Fragment texture units this GL exposes: a pass's channels plus its data
// textures have to fit.
int ghee_max_texture_units(GheeScene *g) { return g ? (int)g->max_units : 0; }

// Whether a data texture is linear-filtered by this GL, the same condition
// ghee_mp_set_array picks its filter from. False on V3D, which has float
// textures but no OES_texture_float_linear, so a filtered fetch is nearest.
bool ghee_data_texture_linear(GheeScene *g) { return g && !(g->has_float_tex && !g->has_float_linear); }

// The display mode's refresh rate in Hz; 0 when not in display mode.
int ghee_display_vrefresh(GheeScene *g) { return g ? (int)g->disp_mode.vrefresh : 0; }

// ---- internal seams for encoder.c (engine_internal.h) ---------------------

void *ghee_i_egl_display(GheeScene *g) { return g->display; }
void  ghee_i_make_current(GheeScene *g) { make_current(g); }
void  ghee_i_error(GheeScene *g, const char *msg) { set_error(g, "%s", msg); }

void ghee_i_render_frame(GheeScene *g) {
    render_passes(g, elapsed_seconds(g), false);
}

void ghee_i_swap(GheeScene *g) {
    for (int p = 0; p < g->mp_count; p++) g->mp_write[p] ^= 1;
}

void ghee_i_blit_last(GheeScene *g, unsigned fbo, int w, int h, float flipY,
                      int post_swap) {
    if (g->mp_count <= 0) { set_error(g, "blit: the scene has no passes"); return; }
    if (!ensure_blit(g)) return;
    const int last = g->mp_count - 1;
    const int buf = post_swap ? 1 - g->mp_write[last] : g->mp_write[last];
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, w, h);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(g->blit_prog);
    glUniform1f(g->blit_flip, flipY);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g->mp_tex[last][buf]);
    glUniform1i(g->blit_sampler, 0);
    glBindVertexArray(g->VAO);
    glBindBuffer(GL_ARRAY_BUFFER, g->VBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid *)(2 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
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
