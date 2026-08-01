"""Published rates, and the names for where frames go.

One definition of fps for everything that publishes one: frames completed over
real elapsed seconds, never a mean of reciprocals. One vocabulary for
destinations: a sink carries its own in Output.dest, and a host adds the ones
only it knows about (a scanout, a live stream).
"""

import time

# /stats "out" entries.
DEST_KMS = "kms"
DEST_PANEL = "panel"
DEST_FILE = "file"
DEST_PIPE = "pipe"
DEST_H264 = "h264"
DEST_MJPEG = "mjpeg"


def rate(count, elapsed):
    """Completed work per real elapsed second."""
    return count / elapsed if elapsed > 0 else 0.0


class FrameRate:
    """Frames completed per real elapsed second, over a trailing window.

    tick() belongs to the thread doing the work. value() only reads the clock,
    so any thread may call it.
    """

    def __init__(self, window=1.0):
        self.window = window
        self.reset()

    def reset(self, now=None):
        self._start = time.monotonic() if now is None else now
        self._last = self._start
        self._count = 0
        self._rate = None  # no window has closed yet

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        self._count += 1
        self._last = now
        elapsed = now - self._start
        if elapsed >= self.window:
            self._rate = rate(self._count, elapsed)
            self._start = now
            self._count = 0

    def value(self, now=None):
        """The last closed window's rate. Nothing completed for a whole window
        reads zero, so a counter that stopped mid-window does not decay towards
        it forever holding the frames it happened to have."""
        now = time.monotonic() if now is None else now
        if now - self._last >= self.window:
            return 0.0
        elapsed = now - self._start
        if self._rate is None or elapsed >= self.window:
            return rate(self._count, elapsed)
        return self._rate
