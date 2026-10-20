// video.c: V4L2 capture for video layers and video assets.
//
// A capture device streams MJPEG or YUYV frames into buffers that the driver
// maps into ghee's memory. ghee_capture_next() hands out the newest frame
// that the driver holds and gives the older ones straight back, so a reader
// slower than the camera always gets the latest picture. The caller decodes
// it and gives it back with ghee_capture_release().
//
// Nothing here touches GL: the capture runs on a thread of its own.

#define _GNU_SOURCE
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sched.h>
#include <sys/syscall.h>
#include <poll.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "engine.h"

#define CAP_BUFS 4

struct GheeCapture {
    int fd;
    uint32_t fourcc;
    int w, h, bytesperline;
    void *ptr[CAP_BUFS];
    size_t len[CAP_BUFS];
    int nbuf;
    int held;                    // the index that next() handed out, -1 for none
};

static int cap_ioctl(int fd, unsigned long req, void *arg) {
    int r;
    do { r = ioctl(fd, req, arg); } while (r < 0 && errno == EINTR);
    return r;
}

// `what`, then the reason of errno `e` after a colon. A NULL `what` gives the
// reason alone, and a 0 `e` `what` alone. A failed call says its reason alone:
// the call's name means nothing to the reader.
static void cap_err(char *err, int n, const char *what, int e) {
    if (!err || n <= 0) return;
    if (!what) snprintf(err, (size_t)n, "%s", strerror(e));
    else snprintf(err, (size_t)n, "%s%s%s", what, e ? ": " : "", e ? strerror(e) : "");
}

// The capabilities of the node itself, as a device with several nodes gives
// each its own.
static uint32_t node_caps(const struct v4l2_capability *c) {
    return (c->capabilities & V4L2_CAP_DEVICE_CAPS) ? c->device_caps : c->capabilities;
}

static int takes(uint32_t fourcc) {
    return fourcc == V4L2_PIX_FMT_MJPEG || fourcc == V4L2_PIX_FMT_YUYV;
}

int ghee_capture_probe(const char *path, char *name, int name_len) {
    int fd = open(path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return 0;
    struct v4l2_capability c = {0};
    int ok = cap_ioctl(fd, VIDIOC_QUERYCAP, &c) == 0;
    const uint32_t caps = node_caps(&c);
    ok = ok && (caps & V4L2_CAP_VIDEO_CAPTURE) && (caps & V4L2_CAP_STREAMING) &&
         !(caps & (V4L2_CAP_VIDEO_M2M | V4L2_CAP_VIDEO_M2M_MPLANE));
    int found = 0;
    struct v4l2_fmtdesc f = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE};
    for (f.index = 0; ok && !found && cap_ioctl(fd, VIDIOC_ENUM_FMT, &f) == 0; f.index++)
        found = takes(f.pixelformat);
    close(fd);
    if (!found) return 0;
    if (name && name_len > 0) snprintf(name, (size_t)name_len, "%s", (const char *)c.card);
    return 1;
}

GheeCapture *ghee_capture_open(const char *path, char *err, int err_len) {
    int fd = open(path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) {
        cap_err(err, err_len, NULL, errno);
        return NULL;
    }
    struct v4l2_capability c = {0};
    if (cap_ioctl(fd, VIDIOC_QUERYCAP, &c) < 0) {
        cap_err(err, err_len, NULL, errno);
        close(fd);
        return NULL;
    }
    const uint32_t caps = node_caps(&c);
    if (!(caps & V4L2_CAP_VIDEO_CAPTURE) || !(caps & V4L2_CAP_STREAMING) ||
        (caps & (V4L2_CAP_VIDEO_M2M | V4L2_CAP_VIDEO_M2M_MPLANE))) {
        cap_err(err, err_len, "not a capture device", 0);
        close(fd);
        return NULL;
    }
    GheeCapture *cap = calloc(1, sizeof *cap);
    if (!cap) {
        cap_err(err, err_len, "out of memory", 0);
        close(fd);
        return NULL;
    }
    cap->fd = fd;
    cap->held = -1;
    return cap;
}

int ghee_capture_modes(GheeCapture *cap, GheeCaptureMode *out, int max) {
    int n = 0;
    struct v4l2_fmtdesc f = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE};
    for (f.index = 0; cap_ioctl(cap->fd, VIDIOC_ENUM_FMT, &f) == 0; f.index++) {
        if (!takes(f.pixelformat)) continue;
        struct v4l2_frmsizeenum s = {.pixel_format = f.pixelformat};
        for (s.index = 0; n < max && cap_ioctl(cap->fd, VIDIOC_ENUM_FRAMESIZES, &s) == 0; s.index++) {
            uint32_t w, h;
            if (s.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
                w = s.discrete.width;
                h = s.discrete.height;
            } else {
                w = s.stepwise.max_width;
                h = s.stepwise.max_height;
            }
            // The shortest frame interval is the highest rate.
            uint32_t num = 0, den = 0;
            struct v4l2_frmivalenum iv = {.pixel_format = f.pixelformat, .width = w, .height = h};
            for (iv.index = 0; cap_ioctl(cap->fd, VIDIOC_ENUM_FRAMEINTERVALS, &iv) == 0; iv.index++) {
                const struct v4l2_fract *fr = iv.type == V4L2_FRMIVAL_TYPE_DISCRETE ? &iv.discrete
                                                                                    : &iv.stepwise.min;
                if (fr->numerator && fr->denominator &&
                    (!den || (uint64_t)fr->numerator * den < (uint64_t)num * fr->denominator)) {
                    num = fr->numerator;
                    den = fr->denominator;
                }
                if (iv.type != V4L2_FRMIVAL_TYPE_DISCRETE) break;
            }
            out[n].fourcc = f.pixelformat;
            out[n].width = (int)w;
            out[n].height = (int)h;
            out[n].fps = den && num ? (double)den / num : 0.0;
            n++;
            if (s.type != V4L2_FRMSIZE_TYPE_DISCRETE) break;
        }
    }
    return n;
}

int ghee_capture_start(GheeCapture *cap, uint32_t fourcc, int w, int h, double fps,
                       char *err, int err_len) {
    struct v4l2_format f = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE};
    f.fmt.pix.width = (uint32_t)w;
    f.fmt.pix.height = (uint32_t)h;
    f.fmt.pix.pixelformat = fourcc;
    f.fmt.pix.field = V4L2_FIELD_NONE;
    if (cap_ioctl(cap->fd, VIDIOC_S_FMT, &f) < 0) {
        cap_err(err, err_len, NULL, errno);
        return -1;
    }
    if (f.fmt.pix.pixelformat != fourcc) {
        cap_err(err, err_len, "the device changed the format it was asked for", 0);
        return -1;
    }
    cap->fourcc = fourcc;
    cap->w = (int)f.fmt.pix.width;
    cap->h = (int)f.fmt.pix.height;
    cap->bytesperline = (int)f.fmt.pix.bytesperline;
    if (fps > 0) {
        struct v4l2_streamparm p = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE};
        p.parm.capture.timeperframe.numerator = 1000;
        p.parm.capture.timeperframe.denominator = (uint32_t)(fps * 1000.0 + 0.5);
        cap_ioctl(cap->fd, VIDIOC_S_PARM, &p);   // a device that keeps its rate still streams
    }
    struct v4l2_requestbuffers rq = {.count = CAP_BUFS, .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
                                     .memory = V4L2_MEMORY_MMAP};
    if (cap_ioctl(cap->fd, VIDIOC_REQBUFS, &rq) < 0) {
        cap_err(err, err_len, NULL, errno);
        return -1;
    }
    if (rq.count < 2) {
        cap_err(err, err_len, NULL, ENOMEM);
        return -1;
    }
    cap->nbuf = rq.count < CAP_BUFS ? (int)rq.count : CAP_BUFS;
    for (int i = 0; i < cap->nbuf; i++) {
        struct v4l2_buffer b = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory = V4L2_MEMORY_MMAP,
                                .index = (uint32_t)i};
        if (cap_ioctl(cap->fd, VIDIOC_QUERYBUF, &b) < 0) {
            cap_err(err, err_len, NULL, errno);
            return -1;
        }
        cap->len[i] = b.length;
        cap->ptr[i] = mmap(NULL, b.length, PROT_READ, MAP_SHARED, cap->fd, b.m.offset);
        if (cap->ptr[i] == MAP_FAILED) {
            cap->ptr[i] = NULL;
            cap_err(err, err_len, NULL, errno);
            return -1;
        }
        if (cap_ioctl(cap->fd, VIDIOC_QBUF, &b) < 0) {
            cap_err(err, err_len, NULL, errno);
            return -1;
        }
    }
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (cap_ioctl(cap->fd, VIDIOC_STREAMON, &type) < 0) {
        cap_err(err, err_len, NULL, errno);
        return -1;
    }
    return 0;
}

void ghee_capture_format(GheeCapture *cap, uint32_t *fourcc, int *w, int *h, int *bytesperline) {
    *fourcc = cap->fourcc;
    *w = cap->w;
    *h = cap->h;
    *bytesperline = cap->bytesperline;
}

static void requeue(GheeCapture *cap, int i) {
    struct v4l2_buffer b = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory = V4L2_MEMORY_MMAP,
                            .index = (uint32_t)i};
    cap_ioctl(cap->fd, VIDIOC_QBUF, &b);
}

void ghee_capture_release(GheeCapture *cap) {
    if (cap->held >= 0) requeue(cap, cap->held);
    cap->held = -1;
}

int ghee_capture_next(GheeCapture *cap, int timeout_ms, const uint8_t **data, size_t *len,
                      char *err, int err_len) {
    ghee_capture_release(cap);
    struct pollfd p = {.fd = cap->fd, .events = POLLIN};
    int r;
    do { r = poll(&p, 1, timeout_ms); } while (r < 0 && errno == EINTR);
    if (r < 0) {
        cap_err(err, err_len, NULL, errno);
        return -1;
    }
    if (r == 0) return 0;
    // A device that went away wakes the poll with an error, and its DQBUF
    // fails with ENODEV.
    int newest = -1;
    size_t used = 0;
    for (;;) {
        struct v4l2_buffer b = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory = V4L2_MEMORY_MMAP};
        if (cap_ioctl(cap->fd, VIDIOC_DQBUF, &b) < 0) {
            if (errno == EAGAIN) break;
            if (newest >= 0) requeue(cap, newest);
            cap_err(err, err_len, NULL, errno);
            return -1;
        }
        if (b.index >= (uint32_t)cap->nbuf) continue;
        if (b.flags & V4L2_BUF_FLAG_ERROR || b.bytesused == 0) {
            requeue(cap, (int)b.index);
            continue;
        }
        if (newest >= 0) requeue(cap, newest);
        newest = (int)b.index;
        used = b.bytesused;
    }
    if (newest < 0) {
        if (p.revents & (POLLERR | POLLHUP)) {
            cap_err(err, err_len, NULL, ENODEV);
            return -1;
        }
        return 0;
    }
    cap->held = newest;
    *data = cap->ptr[newest];
    *len = used < cap->len[newest] ? used : cap->len[newest];
    return 1;
}

void ghee_capture_close(GheeCapture *cap) {
    if (!cap) return;
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (cap->nbuf) cap_ioctl(cap->fd, VIDIOC_STREAMOFF, &type);
    for (int i = 0; i < cap->nbuf; i++)
        if (cap->ptr[i]) munmap(cap->ptr[i], cap->len[i]);
    close(cap->fd);
    free(cap);
}

int ghee_thread_idle(void) {
    const struct sched_param p = {.sched_priority = 0};
    return sched_setscheduler((pid_t)syscall(SYS_gettid), SCHED_IDLE, &p);
}

// ---- the hardware MJPEG decoder of a Pi 0 to 4 ----

#define DEC_IN 2         // compressed buffers
#define DEC_OUT 3        // decoded YU12 buffers, each the ISP's input in turn
#define ISP_OUT 5        // scaled buffers: one filling, one waiting, the rest held by GL

struct plane_buf { void *p; size_t len; int fd; };

struct GheeHwdec {
    int dec, isp;
    struct plane_buf in[DEC_IN], mid[DEC_OUT], out[ISP_OUT];
    int nin, nmid, nout;
    int free_in[DEC_IN], nfree_in;
    bool out_queued[ISP_OUT];
    uint32_t mid_size;          // the bytes of one decoded frame
    int ow, oh, opitch;
    int queued;                 // compressed frames queued since the start
    struct timespec progress;   // the open, or the last frame out of the decoder
};

// The codec node whose card is `card`, opened, or -1.
static int codec_node(const char *card) {
    for (int i = 0; i < 64; i++) {
        char path[32];
        snprintf(path, sizeof path, "/dev/video%d", i);
        const int fd = open(path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (fd < 0) continue;
        struct v4l2_capability c = {0};
        if (cap_ioctl(fd, VIDIOC_QUERYCAP, &c) == 0 && !strcmp((const char *)c.card, card)) return fd;
        close(fd);
    }
    return -1;
}

static int mp_fmt(int fd, uint32_t type, uint32_t fourcc, int w, int h, uint32_t bpl, uint32_t size,
                  struct v4l2_format *got) {
    memset(got, 0, sizeof *got);
    got->type = type;
    got->fmt.pix_mp.width = (uint32_t)w;
    got->fmt.pix_mp.height = (uint32_t)h;
    got->fmt.pix_mp.pixelformat = fourcc;
    got->fmt.pix_mp.field = V4L2_FIELD_NONE;
    got->fmt.pix_mp.num_planes = 1;
    got->fmt.pix_mp.plane_fmt[0].bytesperline = bpl;
    got->fmt.pix_mp.plane_fmt[0].sizeimage = size;
    if (cap_ioctl(fd, VIDIOC_S_FMT, got) < 0) return -1;
    if (got->fmt.pix_mp.pixelformat == fourcc) return 0;
    errno = EINVAL;
    return -1;
}

// Request `count` buffers of `memory` on queue `type`, and map and export
// the MMAP ones. The count that the driver gave, or -1.
static int mp_bufs(int fd, uint32_t type, uint32_t memory, int count, struct plane_buf *b) {
    struct v4l2_requestbuffers rq = {.count = (uint32_t)count, .type = type, .memory = memory};
    if (cap_ioctl(fd, VIDIOC_REQBUFS, &rq) < 0 || rq.count < 1) return -1;
    const int n = rq.count < (uint32_t)count ? (int)rq.count : count;
    for (int i = 0; memory == V4L2_MEMORY_MMAP && i < n; i++) {
        struct v4l2_plane pl = {0};
        struct v4l2_buffer q = {.type = type, .memory = memory, .index = (uint32_t)i, .length = 1,
                                .m.planes = &pl};
        if (cap_ioctl(fd, VIDIOC_QUERYBUF, &q) < 0) return -1;
        b[i].len = pl.length;
        b[i].p = mmap(NULL, pl.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, pl.m.mem_offset);
        if (b[i].p == MAP_FAILED) { b[i].p = NULL; return -1; }
        struct v4l2_exportbuffer ex = {.type = type, .index = (uint32_t)i, .flags = O_RDWR | O_CLOEXEC};
        if (cap_ioctl(fd, VIDIOC_EXPBUF, &ex) < 0) return -1;
        b[i].fd = ex.fd;
    }
    return n;
}

static int mp_qbuf(int fd, uint32_t type, uint32_t memory, int index, int dmafd, uint32_t used,
                   size_t len) {
    struct v4l2_plane pl = {.bytesused = used, .length = (uint32_t)len};
    if (memory == V4L2_MEMORY_DMABUF) pl.m.fd = dmafd;
    struct v4l2_buffer q = {.type = type, .memory = memory, .index = (uint32_t)index, .length = 1,
                            .m.planes = &pl, .field = V4L2_FIELD_NONE};
    return cap_ioctl(fd, VIDIOC_QBUF, &q);
}

// The index of a buffer done on queue `type`, its flags in *flags, or -1.
static int mp_dqbuf(int fd, uint32_t type, uint32_t memory, uint32_t *flags) {
    struct v4l2_plane pl = {0};
    struct v4l2_buffer q = {.type = type, .memory = memory, .length = 1, .m.planes = &pl};
    if (cap_ioctl(fd, VIDIOC_DQBUF, &q) < 0) return -1;
    if (flags) *flags = q.flags;
    return (int)q.index;
}

static int mp_stream(int fd, bool on) {
    int out = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, in = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    const unsigned long r = on ? VIDIOC_STREAMON : VIDIOC_STREAMOFF;
    return cap_ioctl(fd, r, &out) | cap_ioctl(fd, r, &in);
}

// Wait up to timeout_ms for `events` on `fd`. 1 when they came.
static int mp_wait(int fd, short events, int timeout_ms) {
    struct pollfd p = {.fd = fd, .events = events};
    int r;
    do { r = poll(&p, 1, timeout_ms); } while (r < 0 && errno == EINTR);
    return r > 0 && (p.revents & events);
}

GheeHwdec *ghee_hwdec_open(int w, int h, int ow, int oh, char *err, int err_len) {
    GheeHwdec *d = calloc(1, sizeof *d);
    if (!d) {
        cap_err(err, err_len, "out of memory", 0);
        return NULL;
    }
    struct plane_buf *all[] = {d->in, d->mid, d->out};
    const int sizes[] = {DEC_IN, DEC_OUT, ISP_OUT};
    for (int k = 0; k < 3; k++)
        for (int i = 0; i < sizes[k]; i++) all[k][i].fd = -1;
    d->dec = codec_node("bcm2835-codec-decode");
    d->isp = codec_node("bcm2835-codec-isp");
    const uint32_t OUT = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, CAP = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    struct v4l2_format f;
    const char *why = NULL;
    int e = 0;
#define FAIL(msg) do { why = msg; e = errno; } while (0)
    if (d->dec < 0 || d->isp < 0) why = "no hardware MJPEG decoder";
    // room for a compressed frame of a byte a pixel
    else if (mp_fmt(d->dec, OUT, V4L2_PIX_FMT_MJPEG, w, h, 0, (uint32_t)(w * h), &f))
        FAIL("the hardware decoder takes no MJPEG of this size");
    else if (mp_fmt(d->dec, CAP, V4L2_PIX_FMT_YUV420, w, h, 0, 0, &f))
        FAIL("the hardware decoder gives no YU12");
    if (!why) {
        const uint32_t mw = f.fmt.pix_mp.width, mh = f.fmt.pix_mp.height;
        const uint32_t mbpl = f.fmt.pix_mp.plane_fmt[0].bytesperline;
        d->mid_size = f.fmt.pix_mp.plane_fmt[0].sizeimage;
        struct v4l2_selection crop = {.type = V4L2_BUF_TYPE_VIDEO_OUTPUT, .target = V4L2_SEL_TGT_CROP,
                                      .r = {0, 0, (uint32_t)w, (uint32_t)h}};
        struct v4l2_control flip = {.id = V4L2_CID_VFLIP, .value = 1};
        if (mp_fmt(d->isp, OUT, V4L2_PIX_FMT_YUV420, (int)mw, (int)mh, mbpl, d->mid_size, &f))
            FAIL("the ISP takes no YU12 of this size");
        else if (cap_ioctl(d->isp, VIDIOC_S_SELECTION, &crop) < 0)
            FAIL("the ISP takes no crop");
        else if (mp_fmt(d->isp, CAP, V4L2_PIX_FMT_RGBA32, ow, oh, 0, 0, &f))
            FAIL("the ISP gives no RGBA of this size");
        else if (cap_ioctl(d->isp, VIDIOC_S_CTRL, &flip) < 0)
            FAIL("the ISP flips no frame");
        d->ow = (int)f.fmt.pix_mp.width;
        d->oh = (int)f.fmt.pix_mp.height;
        d->opitch = (int)f.fmt.pix_mp.plane_fmt[0].bytesperline;
    }
    if (!why) {
        d->nin = mp_bufs(d->dec, OUT, V4L2_MEMORY_MMAP, DEC_IN, d->in);
        d->nmid = mp_bufs(d->dec, CAP, V4L2_MEMORY_MMAP, DEC_OUT, d->mid);
        const int nmid_in = mp_bufs(d->isp, OUT, V4L2_MEMORY_DMABUF, DEC_OUT, NULL);
        d->nout = mp_bufs(d->isp, CAP, V4L2_MEMORY_MMAP, ISP_OUT, d->out);
        if (d->nin < 1 || d->nmid < 1 || nmid_in < d->nmid || d->nout < 2)
            FAIL("the codec gave no buffers");
    }
    for (int i = 0; !why && i < d->nmid; i++)
        if (mp_qbuf(d->dec, CAP, V4L2_MEMORY_MMAP, i, -1, 0, d->mid[i].len) < 0)
            FAIL("the hardware decoder took no buffer");
    for (int i = 0; !why && i < d->nout; i++) {
        if (mp_qbuf(d->isp, CAP, V4L2_MEMORY_MMAP, i, -1, 0, d->out[i].len) < 0)
            FAIL("the ISP took no buffer");
        d->out_queued[i] = true;
    }
    if (!why && (mp_stream(d->dec, true) < 0 || mp_stream(d->isp, true) < 0))
        FAIL("the codec does not stream");
#undef FAIL
    if (why) {
        cap_err(err, err_len, why, e);
        ghee_hwdec_close(d);
        return NULL;
    }
    d->nfree_in = d->nin;
    for (int i = 0; i < d->nin; i++) d->free_in[i] = i;
    clock_gettime(CLOCK_MONOTONIC, &d->progress);
    return d;
}
// Milliseconds since `t`.
static double ms_since(const struct timespec *t) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec - t->tv_sec) * 1e3 + (double)(now.tv_nsec - t->tv_nsec) / 1e6;
}

int ghee_hwdec_push(GheeHwdec *d, const uint8_t *jpg, size_t len, int scale, int frame_ms,
                    int stall_ms, int *index, char *err, int err_len) {
    const uint32_t OUT = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, CAP = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    for (int i; (i = mp_dqbuf(d->dec, OUT, V4L2_MEMORY_MMAP, NULL)) >= 0;)
        if (d->nfree_in < DEC_IN) d->free_in[d->nfree_in++] = i;
    if (d->nfree_in == 0) {
        if (!mp_wait(d->dec, POLLOUT, stall_ms)) {
            cap_err(err, err_len, "the hardware decoder kept its input", 0);
            return -1;
        }
        for (int i; (i = mp_dqbuf(d->dec, OUT, V4L2_MEMORY_MMAP, NULL)) >= 0;)
            if (d->nfree_in < DEC_IN) d->free_in[d->nfree_in++] = i;
        if (d->nfree_in == 0) {
            cap_err(err, err_len, "the hardware decoder kept its input", 0);
            return -1;
        }
    }
    const int k = d->free_in[d->nfree_in - 1];
    if (len > d->in[k].len) return 0;
    memcpy(d->in[k].p, jpg, len);
    if (mp_qbuf(d->dec, OUT, V4L2_MEMORY_MMAP, k, -1, (uint32_t)len, d->in[k].len) < 0) {
        cap_err(err, err_len, NULL, errno);
        return -1;
    }
    d->nfree_in--;
    if (++d->queued < 2) return 0;
    // A frame that the decoder drops gives nothing, so a frame may not come;
    // none for stall_ms is a failure.
    uint32_t flags = 0;
    int m = -1;
    if (mp_wait(d->dec, POLLIN, frame_ms)) {
        uint32_t f;
        for (int got; (got = mp_dqbuf(d->dec, CAP, V4L2_MEMORY_MMAP, &f)) >= 0;) {
            if (m >= 0) mp_qbuf(d->dec, CAP, V4L2_MEMORY_MMAP, m, -1, 0, d->mid[m].len);
            m = got;
            flags = f;
        }
    }
    if (m < 0) {
        if (ms_since(&d->progress) < stall_ms) return 0;
        if (err && err_len > 0)
            snprintf(err, (size_t)err_len, "the hardware decoder gave no frame for %d ms", stall_ms);
        return -1;
    }
    clock_gettime(CLOCK_MONOTONIC, &d->progress);
    int o = 0;
    while (o < d->nout && !d->out_queued[o]) o++;
    if (!scale || (flags & V4L2_BUF_FLAG_ERROR) || o == d->nout) {
        mp_qbuf(d->dec, CAP, V4L2_MEMORY_MMAP, m, -1, 0, d->mid[m].len);
        return 0;
    }
    if (mp_qbuf(d->isp, OUT, V4L2_MEMORY_DMABUF, m, d->mid[m].fd, d->mid_size, d->mid[m].len) < 0) {
        cap_err(err, err_len, NULL, errno);
        mp_qbuf(d->dec, CAP, V4L2_MEMORY_MMAP, m, -1, 0, d->mid[m].len);
        return -1;
    }
    int got = -1;
    while (got < 0) {
        if (!mp_wait(d->isp, POLLIN, stall_ms)) {
            cap_err(err, err_len, "the ISP gave no frame", 0);
            return -1;
        }
        got = mp_dqbuf(d->isp, CAP, V4L2_MEMORY_MMAP, NULL);
    }
    // The ISP is done with the decoder's frame once it gave its own.
    while (mp_dqbuf(d->isp, OUT, V4L2_MEMORY_DMABUF, NULL) < 0)
        if (!mp_wait(d->isp, POLLOUT, stall_ms)) {
            cap_err(err, err_len, "the ISP kept its input", 0);
            return -1;
        }
    mp_qbuf(d->dec, CAP, V4L2_MEMORY_MMAP, m, -1, 0, d->mid[m].len);
    d->out_queued[got] = false;
    *index = got;
    return 1;
}

void ghee_hwdec_out(GheeHwdec *d, int index, int *fd, int *w, int *h, int *pitch) {
    *fd = d->out[index].fd;
    *w = d->ow;
    *h = d->oh;
    *pitch = d->opitch;
}

void ghee_hwdec_give(GheeHwdec *d, int index) {
    if (index < 0 || index >= d->nout || d->out_queued[index]) return;
    if (mp_qbuf(d->isp, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, V4L2_MEMORY_MMAP, index, -1, 0,
                d->out[index].len) == 0)
        d->out_queued[index] = true;
}

static void release_bufs(struct plane_buf *b, int n) {
    for (int i = 0; i < n; i++) {
        if (b[i].p) munmap(b[i].p, b[i].len);
        if (b[i].fd >= 0) close(b[i].fd);
    }
}

void ghee_hwdec_close(GheeHwdec *d) {
    if (!d) return;
    if (d->isp >= 0) {
        mp_stream(d->isp, false);
        close(d->isp);
    }
    if (d->dec >= 0) {
        mp_stream(d->dec, false);
        close(d->dec);
    }
    release_bufs(d->in, d->nin > 0 ? d->nin : 0);
    release_bufs(d->mid, d->nmid > 0 ? d->nmid : 0);
    release_bufs(d->out, d->nout > 0 ? d->nout : 0);
    free(d);
}
