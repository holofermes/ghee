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
