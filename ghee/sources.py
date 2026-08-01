"""The files a scene was built from, and whether they have moved on.

Loading a scene reads more than the file it names: a pass or a common block can
point at a .glsl beside it, and a .py scene runs its own module plus the
sibling modules it imports. Each of those reads calls record(); a collect()
around the load receives the set.

    with collect() as paths:
        scene, controller = store.resolve(name)
    snap = Snapshot(paths)
    ...
    snap.stale()    # disk holds code this scene was not built from

Copying files (an rsync, a redeploy) rewrites mtimes without changing a byte,
so an mtime only triggers the check and the content hash decides it.
"""

import contextlib
import hashlib
import os
import threading
import time

STALE_INTERVAL = 1.0  # seconds between disk checks: stale() is polled

_collecting = threading.local()


def record(path):
    """Note a file a scene load has just read."""
    into = getattr(_collecting, "into", None)
    if into is not None:
        into.add(os.path.abspath(path))


@contextlib.contextmanager
def collect():
    """Gather what record() sees on this thread while the block runs."""
    prev = getattr(_collecting, "into", None)
    paths = set()
    _collecting.into = paths
    try:
        yield paths
    finally:
        _collecting.into = prev


def _mtime(path):
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return -1


def _digest(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return ""


class Snapshot:
    """A compiled scene's files as they were when it compiled."""

    def __init__(self, paths, interval=STALE_INTERVAL, clock=time.monotonic):
        self.interval = interval
        self.clock = clock
        self.files = {p: (_mtime(p), _digest(p)) for p in sorted(paths)}
        self._stale = False
        self._checked = clock()

    def stale(self):
        """Whether the files now hold content this scene was not built from.
        Answers from the last check inside `interval`."""
        now = self.clock()
        if now - self._checked < self.interval:
            return self._stale
        self._checked = now
        self._stale = self._diverged()
        return self._stale

    def _diverged(self):
        out = False
        for path, (mtime, digest) in list(self.files.items()):
            fresh = _mtime(path)
            if fresh == mtime:
                continue
            if _digest(path) != digest:
                out = True
            else:
                self.files[path] = (fresh, digest)  # copied, not edited
        return out
