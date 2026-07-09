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
