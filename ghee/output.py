"""Output sinks for rendered frames.

A sink takes an RGB(A) frame (h, w, components) uint8 and puts it somewhere: image
files on disk, a hardware panel (see ghee.drivers), a byte stream. The renderer never
knows which sink it feeds.
"""

import abc
import queue
import re
import threading

import numpy as np

_RUN = re.compile(r"#+")


def expand_pattern(pattern, index):
    """frame.####.jpg + 7 -> frame.0007.jpg. A printf field (frame.%04d.jpg) also
    works. A plain filename is returned as is (single frame, or overwrite)."""
    m = _RUN.search(pattern)
    if m:
        width = m.end() - m.start()
        return pattern[: m.start()] + str(index).zfill(width) + pattern[m.end() :]
    if "%" in pattern:
        return pattern % index
    return pattern


class AdaptivePacer:
    """Pace a render loop off the sink's own backpressure. The sink write time
    is the felt speed of the display bus; keeping the frame period at
    factor * write time holds the bus at a 1/factor duty cycle, so a slow bus
    gets few frames and a fast one gets many, with no configured rate. max_fps
    bounds the fast end (image files write in microseconds)."""

    def __init__(self, factor=2.0, max_fps=60):
        self.factor = factor
        self.min_period = 1.0 / max_fps
        self.write_ema = None

    def sleep_for(self, write_s, frame_s):
        """Seconds to sleep after a frame: write_s = this frame's sink write,
        frame_s = the whole frame so far (render + write)."""
        self.write_ema = (
            write_s if self.write_ema is None else self.write_ema * 0.9 + write_s * 0.1
        )
        period = max(self.write_ema * self.factor, self.min_period)
        return max(0.0, period - frame_s)

    def idle_period(self):
        """Current frame period without a new measurement: what a skipped
        (dirty=False) frame sleeps before re-polling."""
        if self.write_ema is None:
            return self.min_period
        return max(self.write_ema * self.factor, self.min_period)


class FixedPacer:
    """Hold a display loop at an explicit frame rate."""

    def __init__(self, fps):
        self.period = 1.0 / fps

    def sleep_for(self, write_s, frame_s):
        return max(0.0, self.period - frame_s)

    def idle_period(self):
        return self.period


class Output(abc.ABC):
    @abc.abstractmethod
    def write(self, frame):
        pass

    def close(self):  # noqa: B027  optional hook, sinks override only if they hold resources
        pass


class ImageSequence(Output):
    """Write each frame to its own numbered file. Encoding and disk I/O run on a
    single worker thread so the render loop never waits on PIL. save_opts is
    passed verbatim to PIL's Image.save (e.g. quality, lossless).

    The pool path takes no per-frame copy: acquire() hands out a preallocated
    buffer, the renderer reads the GPU straight into it (render(out=buf)), and
    submit() queues it; the worker encodes, writes, and recycles the buffer.
    The pool grows lazily to depth+1 buffers, so acquire() blocks only when the
    writer is `depth` frames behind. depth is the grace for a slow writer; the
    depth+1 cap is the memory bound on a small board. Set depth to the frame
    count and the render never waits, the writer drains at the end."""

    def __init__(self, pattern, start=0, save_opts=None, depth=16):
        try:
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                "image output needs Pillow: pip install ghee[console]"
            ) from e
        self._Image = Image
        self.pattern = pattern
        self.index = start
        self.save_opts = save_opts or {}
        self._cap = depth + 1
        self._allocated = 0
        self._q = queue.Queue()
        self._free = queue.Queue()
        self._err = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            index, buf = item
            # The buffer is the worker's alone between submit and recycle, so PIL
            # reads it in place. On failure keep draining and recycling so a
            # blocked acquire() never starves; the error surfaces on the next
            # acquire()/submit() or on close().
            if self._err is None:
                try:
                    mode = "RGBA" if buf.shape[2] == 4 else "RGB"
                    path = expand_pattern(self.pattern, index)
                    self._Image.fromarray(buf, mode).save(path, **self.save_opts)
                except Exception as e:
                    self._err = e
            self._free.put(buf)

    def acquire(self, shape):
        """A buffer of `shape` to render into. Reuses a recycled one, else grows
        the pool to its cap, else blocks until the writer recycles one (the
        render's only wait). Single-producer: only the render loop calls this."""
        if self._err is not None:
            raise self._err
        try:
            return self._free.get_nowait()
        except queue.Empty:
            pass
        if self._allocated < self._cap:
            self._allocated += 1
            return np.empty(shape, dtype=np.uint8)
        return self._free.get()

    def submit(self, buf):
        """Queue a filled buffer for encoding under the next numbered name."""
        if self._err is not None:
            self._free.put(buf)
            raise self._err
        self._q.put((self.index, buf))
        self.index += 1

    def write(self, frame):
        """Copy a caller-owned frame into a pooled buffer and submit it."""
        buf = self.acquire(frame.shape)
        np.copyto(buf, frame)
        self.submit(buf)

    def close(self):
        self._q.put(None)
        self._worker.join()
        if self._err is not None:
            raise self._err


class RawStream(Output):
    """Write each frame's exact bytes to a binary stream, no header, one frame
    after another (h*w*components, row-major). For piping raw video to ffmpeg,
    mpv, or a recorder."""

    def __init__(self, stream):
        self.stream = stream

    def write(self, frame):
        self.stream.write(frame.tobytes())
        self.stream.flush()
