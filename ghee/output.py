"""Output sinks for rendered frames.

A sink takes an RGB(A) frame (h, w, components) uint8 and puts it somewhere: image
files on disk, a hardware panel (see ghee.drivers), a byte stream. The renderer never
knows which sink it feeds.
"""

import abc
import re

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
            write_s
            if self.write_ema is None
            else self.write_ema * 0.9 + write_s * 0.1
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
    """Write each frame to its own numbered file."""

    def __init__(self, pattern, start=0):
        try:
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                "image output needs Pillow: pip install ghee[console]"
            ) from e
        self._Image = Image
        self.pattern = pattern
        self.index = start

    def write(self, frame):
        mode = "RGBA" if frame.shape[2] == 4 else "RGB"
        path = expand_pattern(self.pattern, self.index)
        self._Image.fromarray(frame, mode).save(path)
        self.index += 1
