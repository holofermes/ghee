// encoder.c — zero-copy hardware H264 output (V4L2 M2M, bcm2835-codec).
//
// The final pass is blitted into a LINEAR dmabuf-backed FBO (GBM bo) and the
// dmabuf fd is queued straight to the encoder's OUTPUT queue: pixels go
// GPU -> VPU with no glReadPixels and no CPU copy. The encoder ingests AB24
// (RGBA) directly and converts to YUV in hardware. Compressed Annex-B access
// units come back off the CAPTURE queue.
//
// This file is the only V4L2 user in the engine; boards without a hardware
// encoder (Pi 5, desktops) fail ghee_encoder_create cleanly at open() and the
// Python layer falls back to a software encoder.

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <gbm.h>

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <poll.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "engine_internal.h"

#define ENC_DEVICE "/dev/video11"
#define NBUF 3          // GL-rendered dmabuf input pool == V4L2 OUTPUT queue depth
#define NCAP 3          // compressed capture buffers
#define ENC_POLL_MS 500

struct inbuf {
    struct gbm_bo *bo;
    EGLImageKHR img;
    GLuint tex, fbo;
    int dmafd;
    uint32_t stride;
    size_t size;
};

struct GheeEncoder {
    int w, h, fps, bitrate;
    int vfd;
    struct inbuf in[NBUF];
    void *cap_ptr[NCAP];
    size_t cap_len[NCAP];
    int freeidx[NBUF], nfree;
    uint8_t *au;            // returned access unit, grown as needed
    size_t au_cap;
};

static PFNEGLCREATEIMAGEKHRPROC p_eglCreateImageKHR;
static PFNEGLDESTROYIMAGEKHRPROC p_eglDestroyImageKHR;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC p_glEGLImageTargetTexture2DOES;

static int enc_ioctl(int fd, unsigned long req, void *arg) {
    int r;
    do { r = ioctl(fd, req, arg); } while (r < 0 && errno == EINTR);
    return r;
}

static void enc_ctrl(int fd, uint32_t id, int32_t val) {
    struct v4l2_control c = { .id = id, .value = val };
    if (enc_ioctl(fd, VIDIOC_S_CTRL, &c) < 0)
        fprintf(stderr, "ghee: encoder ctrl 0x%X failed: %s\n", id, strerror(errno));
}

static int make_inbuf(GheeScene *g, GheeEncoder *e, struct inbuf *ib, char **err) {
    struct gbm_device *gbm = ghee_i_gbm(g);
    EGLDisplay dpy = ghee_i_egl_display(g);
    ib->bo = gbm_bo_create(gbm, e->w, e->h, GBM_FORMAT_ABGR8888,
                           GBM_BO_USE_LINEAR | GBM_BO_USE_RENDERING);
    if (!ib->bo) { if (err) *err = "gbm_bo_create(linear) failed"; return -1; }
    ib->stride = gbm_bo_get_stride(ib->bo);
    ib->dmafd = gbm_bo_get_fd(ib->bo);
    off_t sz = lseek(ib->dmafd, 0, SEEK_END);
    lseek(ib->dmafd, 0, SEEK_SET);
    ib->size = (sz > 0) ? (size_t)sz : (size_t)ib->stride * e->h;
    ib->img = p_eglCreateImageKHR(dpy, EGL_NO_CONTEXT, EGL_NATIVE_PIXMAP_KHR,
                                  (EGLClientBuffer)ib->bo, NULL);
    if (ib->img == EGL_NO_IMAGE_KHR) { if (err) *err = "eglCreateImageKHR failed"; return -1; }
    glGenTextures(1, &ib->tex);
    glBindTexture(GL_TEXTURE_2D, ib->tex);
    p_glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, ib->img);
    glGenFramebuffers(1, &ib->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, ib->fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ib->tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        if (err) *err = "dmabuf FBO incomplete";
        return -1;
    }
    return 0;
}

GheeEncoder *ghee_encoder_create(GheeScene *g, int w, int h, int fps, int bitrate,
                                 char **err) {
    static char eb[256];
    GheeEncoder *e = calloc(1, sizeof(GheeEncoder));
    if (!e) { if (err) *err = "out of memory"; return NULL; }
    e->w = w; e->h = h; e->fps = fps > 0 ? fps : 30; e->bitrate = bitrate;
    e->vfd = -1;

    // Non-blocking: reclaim() drains consumed buffers with DQBUF until EAGAIN;
    // the waits that matter are explicit poll()s.
    e->vfd = open(ENC_DEVICE, O_RDWR | O_CLOEXEC | O_NONBLOCK);
    if (e->vfd < 0) {
        snprintf(eb, sizeof(eb), "no hardware encoder (%s: %s)", ENC_DEVICE, strerror(errno));
        if (err) *err = eb;
        free(e);
        return NULL;
    }
    ghee_i_keep_last(g, +1);   // paired with the vfd>=0 decrement in destroy

    ghee_i_make_current(g);
    if (!p_eglCreateImageKHR) {
        p_eglCreateImageKHR = (void *)eglGetProcAddress("eglCreateImageKHR");
        p_eglDestroyImageKHR = (void *)eglGetProcAddress("eglDestroyImageKHR");
        p_glEGLImageTargetTexture2DOES = (void *)eglGetProcAddress("glEGLImageTargetTexture2DOES");
    }
    if (!p_eglCreateImageKHR || !p_glEGLImageTargetTexture2DOES) {
        if (err) *err = "EGLImage extensions missing";
        goto fail;
    }
    for (int i = 0; i < NBUF; i++)
        if (make_inbuf(g, e, &e->in[i], err)) goto fail;

    struct v4l2_format ofmt = {0};
    ofmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    ofmt.fmt.pix_mp.width = w;
    ofmt.fmt.pix_mp.height = h;
    ofmt.fmt.pix_mp.pixelformat = v4l2_fourcc('A', 'B', '2', '4');   // RGBA, matches the dmabuf
    ofmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
    ofmt.fmt.pix_mp.num_planes = 1;
    ofmt.fmt.pix_mp.plane_fmt[0].bytesperline = e->in[0].stride;
    ofmt.fmt.pix_mp.plane_fmt[0].sizeimage = e->in[0].stride * h;
    ofmt.fmt.pix_mp.colorspace = V4L2_COLORSPACE_REC709;
    if (enc_ioctl(e->vfd, VIDIOC_S_FMT, &ofmt) < 0) {
        snprintf(eb, sizeof(eb), "encoder rejects AB24 input: %s", strerror(errno));
        if (err) *err = eb;
        goto fail;
    }

    struct v4l2_format cfmt = {0};
    cfmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    cfmt.fmt.pix_mp.width = w;
    cfmt.fmt.pix_mp.height = h;
    cfmt.fmt.pix_mp.pixelformat = v4l2_fourcc('H', '2', '6', '4');
    cfmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
    cfmt.fmt.pix_mp.num_planes = 1;
    cfmt.fmt.pix_mp.plane_fmt[0].sizeimage = 512 * 1024;
    if (enc_ioctl(e->vfd, VIDIOC_S_FMT, &cfmt) < 0) {
        snprintf(eb, sizeof(eb), "encoder rejects H264 output: %s", strerror(errno));
        if (err) *err = eb;
        goto fail;
    }

    struct v4l2_streamparm parm = {0};   // frame rate, for rate control
    parm.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    parm.parm.output.timeperframe.numerator = 1;
    parm.parm.output.timeperframe.denominator = e->fps;
    enc_ioctl(e->vfd, VIDIOC_S_PARM, &parm);   // best effort

    enc_ctrl(e->vfd, V4L2_CID_MPEG_VIDEO_BITRATE, bitrate);
    enc_ctrl(e->vfd, V4L2_CID_MPEG_VIDEO_REPEAT_SEQ_HEADER, 1);   // SPS/PPS on every IDR
    enc_ctrl(e->vfd, V4L2_CID_MPEG_VIDEO_H264_I_PERIOD, e->fps * 2);

    struct v4l2_requestbuffers orq = {
        .count = NBUF, .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, .memory = V4L2_MEMORY_DMABUF};
    if (enc_ioctl(e->vfd, VIDIOC_REQBUFS, &orq) < 0 || orq.count < 1) {
        if (err) *err = "REQBUFS OUTPUT DMABUF failed";
        goto fail;
    }
    struct v4l2_requestbuffers crq = {
        .count = NCAP, .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, .memory = V4L2_MEMORY_MMAP};
    if (enc_ioctl(e->vfd, VIDIOC_REQBUFS, &crq) < 0 || crq.count < 1) {
        if (err) *err = "REQBUFS CAPTURE MMAP failed";
        goto fail;
    }
    for (int i = 0; i < (int)crq.count && i < NCAP; i++) {
        struct v4l2_plane pl = {0};
        struct v4l2_buffer b = {0};
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        b.memory = V4L2_MEMORY_MMAP;
        b.index = i;
        b.length = 1;
        b.m.planes = &pl;
        if (enc_ioctl(e->vfd, VIDIOC_QUERYBUF, &b) < 0) { if (err) *err = "QUERYBUF failed"; goto fail; }
        e->cap_len[i] = pl.length;
        e->cap_ptr[i] = mmap(NULL, pl.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                             e->vfd, pl.m.mem_offset);
        if (e->cap_ptr[i] == MAP_FAILED) { if (err) *err = "mmap capture failed"; goto fail; }
        struct v4l2_plane pl2 = {0};
        struct v4l2_buffer qb = {0};
        qb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        qb.memory = V4L2_MEMORY_MMAP;
        qb.index = i;
        qb.length = 1;
        qb.m.planes = &pl2;
        if (enc_ioctl(e->vfd, VIDIOC_QBUF, &qb) < 0) { if (err) *err = "QBUF capture failed"; goto fail; }
    }

    int t1 = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, t2 = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (enc_ioctl(e->vfd, VIDIOC_STREAMON, &t1) < 0 ||
        enc_ioctl(e->vfd, VIDIOC_STREAMON, &t2) < 0) {
        if (err) *err = "STREAMON failed";
        goto fail;
    }

    e->nfree = NBUF;
    for (int i = 0; i < NBUF; i++) e->freeidx[i] = i;
    return e;

fail:
    ghee_encoder_destroy(g, e);
    return NULL;
}

void ghee_encoder_force_keyframe(GheeEncoder *e) {
    if (e) enc_ctrl(e->vfd, V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME, 1);
}

// Reclaim any consumed OUTPUT (input-side) buffers back onto the free list.
static void reclaim(GheeEncoder *e) {
    for (;;) {
        struct v4l2_plane pl = {0};
        struct v4l2_buffer b = {0};
        b.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        b.memory = V4L2_MEMORY_DMABUF;
        b.length = 1;
        b.m.planes = &pl;
        if (enc_ioctl(e->vfd, VIDIOC_DQBUF, &b) < 0) return;
        e->freeidx[e->nfree++] = b.index;
    }
}

// Blit the current final pass into a free dmabuf, queue it, wait for one
// compressed access unit. Returns 0 with *nal/*len/*is_key set, 1 if the
// encoder produced nothing yet (pipeline priming), <0 on error.
static int encode_common(GheeScene *g, GheeEncoder *e, int post_swap,
                         const uint8_t **nal, int *len, int *is_key) {
    reclaim(e);
    if (e->nfree == 0) {
        struct pollfd pfd = { .fd = e->vfd, .events = POLLOUT };
        if (poll(&pfd, 1, ENC_POLL_MS) <= 0) { ghee_i_error(g, "encoder stalled (no free input)"); return -1; }
        reclaim(e);
        if (e->nfree == 0) { ghee_i_error(g, "encoder stalled (no free input)"); return -1; }
    }
    const int idx = e->freeidx[--e->nfree];

    // flipY 0.0: an FBO's memory rows come out in glReadPixels order, which is
    // what the codec consumes top-first (same as the preview readback, opposite
    // of the scanout blit).
    ghee_i_blit_last(g, e->in[idx].fbo, e->w, e->h, 0.0f, post_swap);
    glFinish();   // GPU done before the codec reads this dmabuf

    struct v4l2_plane pl = {0};
    struct v4l2_buffer b = {0};
    b.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    b.memory = V4L2_MEMORY_DMABUF;
    b.index = idx;
    b.length = 1;
    b.m.planes = &pl;
    b.field = V4L2_FIELD_NONE;
    pl.m.fd = e->in[idx].dmafd;
    pl.bytesused = e->in[idx].stride * e->h;
    pl.length = e->in[idx].size;
    if (enc_ioctl(e->vfd, VIDIOC_QBUF, &b) < 0) {
        ghee_i_error(g, "encoder QBUF failed");
        e->freeidx[e->nfree++] = idx;
        return -1;
    }

    struct pollfd pfd = { .fd = e->vfd, .events = POLLIN };
    if (poll(&pfd, 1, ENC_POLL_MS) <= 0) return 1;   // nothing compressed yet

    struct v4l2_plane cpl = {0};
    struct v4l2_buffer cb = {0};
    cb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    cb.memory = V4L2_MEMORY_MMAP;
    cb.length = 1;
    cb.m.planes = &cpl;
    if (enc_ioctl(e->vfd, VIDIOC_DQBUF, &cb) < 0) return 1;

    const size_t n = cpl.bytesused;
    if (n > e->au_cap) {
        uint8_t *p = realloc(e->au, n);
        if (!p) { ghee_i_error(g, "encoder OOM"); return -1; }
        e->au = p;
        e->au_cap = n;
    }
    memcpy(e->au, e->cap_ptr[cb.index], n);
    if (is_key) *is_key = (cb.flags & V4L2_BUF_FLAG_KEYFRAME) ? 1 : 0;

    struct v4l2_plane rpl = {0};
    struct v4l2_buffer rb = {0};
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    rb.memory = V4L2_MEMORY_MMAP;
    rb.index = cb.index;
    rb.length = 1;
    rb.m.planes = &rpl;
    enc_ioctl(e->vfd, VIDIOC_QBUF, &rb);

    *nal = e->au;
    *len = (int)n;
    return 0;
}

// Full frame: render the pass chain, encode it, no readback anywhere.
int ghee_mp_render_encoded(GheeScene *g, GheeEncoder *e, const uint8_t **nal,
                           int *len, int *is_key) {
    ghee_i_make_current(g);
    ghee_i_render_frame(g);
    const int r = encode_common(g, e, 0, nal, len, is_key);
    ghee_i_swap(g);
    return r;
}

// Encode the frame a completed ghee_mp_render/ghee_mp_present call produced
// (display mirror: one render feeds the screen and the stream).
int ghee_encoder_encode_last(GheeScene *g, GheeEncoder *e, const uint8_t **nal,
                             int *len, int *is_key) {
    ghee_i_make_current(g);
    return encode_common(g, e, 1, nal, len, is_key);
}

void ghee_encoder_destroy(GheeScene *g, GheeEncoder *e) {
    if (!e) return;
    if (e->vfd >= 0) {
        int t1 = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, t2 = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        enc_ioctl(e->vfd, VIDIOC_STREAMOFF, &t1);
        enc_ioctl(e->vfd, VIDIOC_STREAMOFF, &t2);
    }
    for (int i = 0; i < NCAP; i++)
        if (e->cap_ptr[i] && e->cap_ptr[i] != MAP_FAILED) munmap(e->cap_ptr[i], e->cap_len[i]);
    ghee_i_make_current(g);
    EGLDisplay dpy = ghee_i_egl_display(g);
    for (int i = 0; i < NBUF; i++) {
        struct inbuf *ib = &e->in[i];
        if (ib->fbo) glDeleteFramebuffers(1, &ib->fbo);
        if (ib->tex) glDeleteTextures(1, &ib->tex);
        if (ib->img && p_eglDestroyImageKHR) p_eglDestroyImageKHR(dpy, ib->img);
        if (ib->dmafd > 0) close(ib->dmafd);
        if (ib->bo) gbm_bo_destroy(ib->bo);
    }
    if (e->vfd >= 0) {
        close(e->vfd);
        ghee_i_keep_last(g, -1);   // only counted after a successful open
    }
    free(e->au);
    free(e);
}
