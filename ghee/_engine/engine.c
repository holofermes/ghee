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
#include "engine.h"

#define ERROR_BUFFER_SIZE 512
#define DEFAULT_DRI_DEVICE "/dev/dri/card0"
#define MP_MAX_PASSES 8
#define MP_MAX_CHANNELS 8
#define MP_MAX_DATATEX 4

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

    // multipass: each pass has its own program + ping-pong buffer; channels route
    // between passes (reading a pass's PREVIOUS-frame output). A pass whose channel
    // points at itself = feedback. Last pass -> screen.
    int mp_count;
    GLuint mp_program[MP_MAX_PASSES];
    GLuint mp_fbo[MP_MAX_PASSES][2];
    GLuint mp_tex[MP_MAX_PASSES][2];
    int mp_write[MP_MAX_PASSES];
    int mp_chan_src[MP_MAX_PASSES][MP_MAX_CHANNELS];  // channel -> source pass index, -1 = none
    int mp_nchan[MP_MAX_PASSES];
    GLint mp_timeLoc[MP_MAX_PASSES];

    // data textures: a float array bound as a 1xN sampler (<name>_tex) so a computed
    // index samples it in one fetch, dodging the VC4 uniform-array unroll cap.
    bool has_float_tex, has_float_linear;   // OES_texture_float[_linear] on this GL
    int mp_ndatatex[MP_MAX_PASSES];
    GLuint mp_datatex[MP_MAX_PASSES][MP_MAX_DATATEX];
    char mp_datatex_name[MP_MAX_PASSES][MP_MAX_DATATEX][64];   // sampler uniform name
    int mp_datatex_size[MP_MAX_PASSES][MP_MAX_DATATEX];

    bool profile;                        // when on, ghee_mp_render splits draw vs readback (adds a glFinish)
    double prof_draw_ms, prof_read_ms;   // last frame's split (profiling mode)

    // display mode: scan the final buffer straight out to a KMS connector
    // (zero CPU copy), instead of reading it back. Set up by ghee_display_create.
    bool display_mode;
    uint32_t disp_conn, disp_crtc;
    drmModeModeInfo disp_mode;
    drmModeCrtc *disp_saved;
    struct gbm_bo *disp_prev;
    bool disp_first;
    GLuint blit_prog;
    GLint blit_sampler, blit_flip;
    GLuint preview_fbo, preview_tex;   // small FBO for a cheap downscaled readback
    int preview_w, preview_h;

    char last_error[ERROR_BUFFER_SIZE];
};

static double ms_between(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1000.0 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

static const char *vertex_shader_source =
    "#version 100\n"
    "attribute vec4 position;"
    "attribute vec2 texCoord;"
    "varying vec2 v_texCoord;"
    "void main() {"
    "   gl_Position = position;"
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

static GheeScene *create_impl(int width, int height, int components, const char *dri_device, int scanout, char **err) {
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

    g->gbm_surface = gbm_surface_create(g->gbm, width, height, GBM_FORMAT_XRGB8888,
        GBM_BO_USE_RENDERING | (scanout ? GBM_BO_USE_SCANOUT : 0));
    if (!g->gbm_surface) { snprintf(early_err, sizeof(early_err), "gbm_surface_create failed"); goto fail; }

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
        EGL_SAMPLE_BUFFERS, 1,
        EGL_SAMPLES, 4,
        EGL_NONE
    };
    EGLConfig config;
    EGLint numConfigs;
    if (!eglChooseConfig(g->display, configAttribs, &config, 1, &numConfigs) || numConfigs < 1) {
        snprintf(early_err, sizeof(early_err), "eglChooseConfig failed: 0x%X", eglGetError());
        goto fail;
    }
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
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    const char *glexts = (const char *)glGetString(GL_EXTENSIONS);
    g->has_float_tex = glexts && strstr(glexts, "GL_OES_texture_float") != NULL;
    g->has_float_linear = glexts && strstr(glexts, "GL_OES_texture_float_linear") != NULL;
    fprintf(stderr, "ghee: data-texture format = %s%s\n",
            g->has_float_tex ? "float" : "8-bit LUMINANCE",
            g->has_float_tex ? (g->has_float_linear ? " (linear)" : " (nearest)") : " (linear)");

    clock_gettime(CLOCK_MONOTONIC, &g->start_time);
    return g;

fail:
    if (err) *err = early_err;
    ghee_destroy(g);
    return NULL;
}

GheeScene *ghee_create(int width, int height, int components, const char *dri_device, char **err) {
    return create_impl(width, height, components, dri_device, 0, err);
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

// Render to a KMS connector instead of reading back. width/height should match
// the connector mode (query it first). Must run as DRM master (root).
GheeScene *ghee_display_create(int width, int height, const char *dri_device, char **err) {
    GheeScene *g = create_impl(width, height, 4, dri_device, 1, err);
    if (!g) return NULL;
    if (drmSetMaster(g->drm_fd))
        fprintf(stderr, "ghee: drmSetMaster failed: %s (run as root)\n", strerror(errno));

    drmModeRes *res = drmModeGetResources(g->drm_fd);
    if (!res) { if (err) *err = "drmModeGetResources failed"; ghee_destroy(g); return NULL; }
    drmModeConnector *conn = NULL;
    for (int i = 0; i < res->count_connectors; i++) {
        drmModeConnector *c = drmModeGetConnector(g->drm_fd, res->connectors[i]);
        if (c && c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) { conn = c; break; }
        if (c) drmModeFreeConnector(c);
    }
    if (!conn) { if (err) *err = "no connected display"; drmModeFreeResources(res); ghee_destroy(g); return NULL; }
    g->disp_conn = conn->connector_id;
    drmModeEncoder *enc = drmModeGetEncoder(g->drm_fd, conn->encoder_id);
    g->disp_crtc = (enc && enc->crtc_id) ? enc->crtc_id : res->crtcs[0];
    if (enc) drmModeFreeEncoder(enc);
    // Prefer the mode the display is already driving (honours a video= override
    // or whatever the console negotiated); fall back to the connector's first.
    drmModeCrtc *cur = drmModeGetCrtc(g->drm_fd, g->disp_crtc);
    g->disp_mode = (cur && cur->mode_valid) ? cur->mode : conn->modes[0];
    if (cur) drmModeFreeCrtc(cur);
    fprintf(stderr, "ghee: display %s %ux%u@%u on connector %u\n",
            g->disp_mode.name, g->disp_mode.hdisplay, g->disp_mode.vdisplay,
            g->disp_mode.vrefresh, g->disp_conn);
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);

    make_current(g);
    g->blit_prog = create_shader_program(g, blit_fragment_source);
    if (!g->blit_prog) { if (err) *err = "blit shader failed"; ghee_destroy(g); return NULL; }
    g->blit_sampler = glGetUniformLocation(g->blit_prog, "tex");
    g->blit_flip = glGetUniformLocation(g->blit_prog, "flipY");
    g->display_mode = true;
    g->disp_first = true;
    return g;
}

void ghee_destroy(GheeScene *g) {
    if (!g) return;
    if (g->display_mode) {
        if (g->preview_fbo) { glDeleteFramebuffers(1, &g->preview_fbo); glDeleteTextures(1, &g->preview_tex); }
        if (g->disp_saved)
            drmModeSetCrtc(g->drm_fd, g->disp_saved->crtc_id, g->disp_saved->buffer_id,
                           g->disp_saved->x, g->disp_saved->y, &g->disp_conn, 1, &g->disp_saved->mode);
        if (g->disp_prev) gbm_surface_release_buffer(g->gbm_surface, g->disp_prev);
        drmDropMaster(g->drm_fd);
    }
    if (g->display != EGL_NO_DISPLAY) {
        ghee_mp_reset(g);   // make_current + free all pass programs/buffers
        eglMakeCurrent(g->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }
    if (g->VBO) glDeleteBuffers(1, &g->VBO);
    if (g->VAO) glDeleteVertexArrays(1, &g->VAO);
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

static void mp_make_buffers(GheeScene *g, GLuint fbo[2], GLuint tex[2]) {
    glGenFramebuffers(2, fbo);
    glGenTextures(2, tex);
    for (int i = 0; i < 2; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo[i]);
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, g->width, g->height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex[i], 0);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

int ghee_mp_add_pass(GheeScene *g, const char *glsl_source) {
    make_current(g);
    if (g->mp_count >= MP_MAX_PASSES) { set_error(g, "too many passes"); return -1; }
    GLuint prog = create_shader_program(g, glsl_source);
    if (!prog) return -1;
    int idx = g->mp_count++;
    g->mp_program[idx] = prog;
    mp_make_buffers(g, g->mp_fbo[idx], g->mp_tex[idx]);
    g->mp_write[idx] = 0;
    g->mp_nchan[idx] = 0;
    g->mp_ndatatex[idx] = 0;
    for (int c = 0; c < MP_MAX_CHANNELS; c++) g->mp_chan_src[idx][c] = -1;
    glUseProgram(prog);
    GLint rloc = glGetUniformLocation(prog, "resolution");
    if (rloc >= 0) glUniform2f(rloc, (float)g->width, (float)g->height);
    g->mp_timeLoc[idx] = glGetUniformLocation(prog, "time");
    return idx;
}

void ghee_mp_set_channel(GheeScene *g, int pass, int channel, int src_pass) {
    if (pass < 0 || pass >= g->mp_count || channel < 0 || channel >= MP_MAX_CHANNELS) return;
    g->mp_chan_src[pass][channel] = src_pass;
    if (channel + 1 > g->mp_nchan[pass]) g->mp_nchan[pass] = channel + 1;
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

// Unused: superseded by ghee_mp_set_float_texture (uniform arrays hit the VC4 size cap).
void ghee_mp_set_float_array(GheeScene *g, int pass, const char *name, const float *data, int count) {
    if (pass < 0 || pass >= g->mp_count) return;
    make_current(g);
    glUseProgram(g->mp_program[pass]);
    const GLint loc = glGetUniformLocation(g->mp_program[pass], name);
    if (loc >= 0) glUniform1fv(loc, count, data);
}

// Upload the float array as a 1xN texture named "<name>_tex". Created lazily on
// first use, updated in place after. Float texels if the GL supports them, else
// 8-bit LUMINANCE (clamped to [0,1]). The unit is bound in ghee_mp_render.
void ghee_mp_set_float_texture(GheeScene *g, int pass, const char *name, const float *data, int count, int rows) {
    if (pass < 0 || pass >= g->mp_count || count <= 0 || rows <= 0) return;
    make_current(g);
    char uni[64];
    snprintf(uni, sizeof(uni), "%s_tex", name);
    int idx = -1;
    for (int i = 0; i < g->mp_ndatatex[pass]; i++)
        if (strcmp(g->mp_datatex_name[pass][i], uni) == 0) { idx = i; break; }
    if (idx < 0) {
        if (g->mp_ndatatex[pass] >= MP_MAX_DATATEX) return;
        idx = g->mp_ndatatex[pass]++;
        snprintf(g->mp_datatex_name[pass][idx], 64, "%s", uni);
        glGenTextures(1, &g->mp_datatex[pass][idx]);
        g->mp_datatex_size[pass][idx] = 0;
    }
    const int unit = g->mp_nchan[pass] + idx;   // data textures follow the channels
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, g->mp_datatex[pass][idx]);
    const bool realloc = (g->mp_datatex_size[pass][idx] != count * rows);
    if (realloc) {
        const GLint filt = g->has_float_tex && !g->has_float_linear ? GL_NEAREST : GL_LINEAR;
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    if (g->has_float_tex) {
        if (realloc)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, count, rows, 0, GL_LUMINANCE, GL_FLOAT, data);
        else
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, count, rows, GL_LUMINANCE, GL_FLOAT, data);
    } else {
        uint8_t *b = malloc((size_t)count * rows);
        if (!b) return;
        for (int i = 0; i < count * rows; i++) {
            float v = data[i];
            b[i] = (uint8_t)(v <= 0.0f ? 0 : v >= 1.0f ? 255 : v * 255.0f + 0.5f);
        }
        if (realloc)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, count, rows, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, b);
        else
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, count, rows, GL_LUMINANCE, GL_UNSIGNED_BYTE, b);
        free(b);
    }
    g->mp_datatex_size[pass][idx] = count * rows;
}

static float elapsed_seconds(GheeScene *g) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return g->has_time_override ? g->time_override
        : (now.tv_sec - g->start_time.tv_sec) + (now.tv_nsec - g->start_time.tv_nsec) / 1e9f;
}

// Render every pass into its FBO (shared by readback and display present).
static void render_passes(GheeScene *g, float elapsed) {
    for (int p = 0; p < g->mp_count; p++) {
        glBindFramebuffer(GL_FRAMEBUFFER, g->mp_fbo[p][g->mp_write[p]]);
        glViewport(0, 0, g->width, g->height);
        glUseProgram(g->mp_program[p]);
        if (g->mp_timeLoc[p] >= 0) glUniform1f(g->mp_timeLoc[p], elapsed);
        for (int c = 0; c < g->mp_nchan[p]; c++) {
            const int src = g->mp_chan_src[p][c];
            if (src < 0) continue;
            const int rd = (src < p) ? g->mp_write[src] : 1 - g->mp_write[src];
            glActiveTexture(GL_TEXTURE0 + c);
            glBindTexture(GL_TEXTURE_2D, g->mp_tex[src][rd]);
            char nm[16];
            snprintf(nm, sizeof(nm), "iChannel%d", c);
            const GLint loc = glGetUniformLocation(g->mp_program[p], nm);
            if (loc >= 0) glUniform1i(loc, c);
        }
        for (int i = 0; i < g->mp_ndatatex[p]; i++) {
            const int unit = g->mp_nchan[p] + i;
            glActiveTexture(GL_TEXTURE0 + unit);
            glBindTexture(GL_TEXTURE_2D, g->mp_datatex[p][i]);
            const GLint loc = glGetUniformLocation(g->mp_program[p], g->mp_datatex_name[p][i]);
            if (loc >= 0) glUniform1i(loc, unit);
        }
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
}

// Render + scan the final buffer straight out to the connector. If buffer_rgb
// is non-NULL, also read the frame back (for a console MJPEG mirror).
void ghee_mp_present(GheeScene *g, uint8_t *buffer_rgb, int pw, int ph) {
    if (!g->display_mode) return;
    make_current(g);
    render_passes(g, elapsed_seconds(g));
    const int last = g->mp_count - 1;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, g->width, g->height);
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
    eglSwapBuffers(g->display, g->surface);

    struct gbm_bo *bo = gbm_surface_lock_front_buffer(g->gbm_surface);
    uint32_t fb = fb_for(g, bo);
    if (g->disp_first) {
        g->disp_saved = drmModeGetCrtc(g->drm_fd, g->disp_crtc);
        if (drmModeSetCrtc(g->drm_fd, g->disp_crtc, fb, 0, 0, &g->disp_conn, 1, &g->disp_mode))
            set_error(g, "drmModeSetCrtc failed: %s", strerror(errno));
        g->disp_first = false;
    } else {
        int waiting = 1;
        if (!drmModePageFlip(g->drm_fd, g->disp_crtc, fb, DRM_MODE_PAGE_FLIP_EVENT, &waiting)) {
            drmEventContext ev = { .version = 2 };
            struct pollfd pfd = { .fd = g->drm_fd, .events = POLLIN };
            while (waiting) {
                if (poll(&pfd, 1, 100) <= 0) break;
                drmHandleEvent(g->drm_fd, &ev);
                waiting = 0;  // any flip event completes the wait
            }
        }
    }
    if (g->disp_prev) gbm_surface_release_buffer(g->gbm_surface, g->disp_prev);
    g->disp_prev = bo;
    if (buffer_rgb && pw > 0 && ph > 0) {
        // downscale the final frame into a small FBO on the GPU, then read that
        // back (a few hundred KB instead of the full frame) so the flip cadence
        // isn't stalled by a megapixel glReadPixels.
        if (g->preview_w != pw || g->preview_h != ph) {
            if (g->preview_fbo) { glDeleteFramebuffers(1, &g->preview_fbo); glDeleteTextures(1, &g->preview_tex); }
            glGenTextures(1, &g->preview_tex);
            glBindTexture(GL_TEXTURE_2D, g->preview_tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pw, ph, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glGenFramebuffers(1, &g->preview_fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, g->preview_fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g->preview_tex, 0);
            g->preview_w = pw; g->preview_h = ph;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, g->preview_fbo);
        glViewport(0, 0, pw, ph);
        glUseProgram(g->blit_prog);
        glUniform1f(g->blit_flip, 0.0f);   // preview matches glReadPixels orientation
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g->mp_tex[last][g->mp_write[last]]);
        glUniform1i(g->blit_sampler, 0);
        glBindBuffer(GL_ARRAY_BUFFER, g->VBO);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid *)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid *)(2 * sizeof(GLfloat)));
        glEnableVertexAttribArray(1);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, pw, ph, GL_RGBA, GL_UNSIGNED_BYTE, buffer_rgb);
    }
    for (int p = 0; p < g->mp_count; p++) g->mp_write[p] ^= 1;
}

void ghee_mp_render(GheeScene *g, uint8_t *buffer_rgb) {
    make_current(g);
    struct timespec now, t_draw, t_read;
    clock_gettime(CLOCK_MONOTONIC, &now);
    render_passes(g, elapsed_seconds(g));

    // Profiling: force the GPU to finish the passes so the readback timing below
    // is pure readback.
    if (g->profile) { glFinish(); clock_gettime(CLOCK_MONOTONIC, &t_draw); }

    const int last = g->mp_count - 1;
    glBindFramebuffer(GL_FRAMEBUFFER, g->mp_fbo[last][g->mp_write[last]]);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, g->width, g->height, (g->components == 3) ? GL_RGB : GL_RGBA, GL_UNSIGNED_BYTE, buffer_rgb);

    if (g->profile) {
        clock_gettime(CLOCK_MONOTONIC, &t_read);
        g->prof_draw_ms = ms_between(now, t_draw);   // GPU render (passes), fenced
        g->prof_read_ms = ms_between(t_draw, t_read); // pure glReadPixels
    }

    for (int p = 0; p < g->mp_count; p++) g->mp_write[p] ^= 1;  // swap all
}

void ghee_set_profiling(GheeScene *g, bool on) { g->profile = on; }

void ghee_set_time(GheeScene *g, float t) { g->time_override = t; g->has_time_override = true; }

void ghee_get_timings(GheeScene *g, double *draw_ms, double *read_ms) {
    if (draw_ms) *draw_ms = g->prof_draw_ms;
    if (read_ms) *read_ms = g->prof_read_ms;
}

// Tear down all passes (for switching scenes). Keeps the GL context.
void ghee_mp_reset(GheeScene *g) {
    make_current(g);
    for (int p = 0; p < g->mp_count; p++) {
        if (g->mp_program[p]) glDeleteProgram(g->mp_program[p]);
        glDeleteFramebuffers(2, g->mp_fbo[p]);
        glDeleteTextures(2, g->mp_tex[p]);
        if (g->mp_ndatatex[p] > 0) glDeleteTextures(g->mp_ndatatex[p], g->mp_datatex[p]);
        g->mp_ndatatex[p] = 0;
    }
    g->mp_count = 0;
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
