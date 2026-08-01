"""The render loop.

One loop for everything ghee renders: image files, a pipe, a panel, KMS
scanout, H264, the browser console, or several of those at once. It draws a
frame and hands it to its sinks; what each sink needs from a frame the sink
declares itself (ghee.output.Output).

    from ghee.loop import Loop
    from ghee.output import ImageSequence

    r = Renderer(320, 240, components=4)
    r.setup(scene)
    Loop(r, [ImageSequence("f.####.png")], frames=60).run()

GL is thread-affine, so the renderer belongs to the thread that runs the loop:
a host running the loop in a worker creates it there.

A host drives the loop through callables instead of subclassing it: `poll`
drains its command queue, `idle` says how long a skipped frame waits, `stopping`
ends the run, `on_frame`/`on_error` publish. `controller` is read every frame,
so assigning it switches scenes mid-run.
"""

import time

import numpy as np

from ghee.stats import FrameRate


def should_draw(forced, controller, frame, t):
    """A frame draws when something demands it: a queued command, the very
    first frame (the sink must show something once), a scene with no
    controller (free-running), or a controller reporting dirty."""
    if forced or frame == 0 or controller is None:
        return True
    return bool(controller.dirty(frame, t))


class Loop:
    """Draw frames into a list of sinks until the frame budget runs out or the
    host stops it.

    frames=0 runs until `stopping`; anything else is a finite render. The loop
    turns its sinks on and off around the run but never closes them: their
    lifetime belongs to whoever built them.
    """

    def __init__(
        self,
        renderer,
        sinks=(),
        *,
        controller=None,
        provider=None,
        frames=0,
        timestep=None,
        pacer=None,
        rate=None,
        rest=0.0,
        poll=None,
        idle=None,
        stopping=None,
        on_frame=None,
        on_error=None,
    ):
        self.renderer = renderer
        self.sinks = list(sinks)
        self.controller = controller
        self.provider = provider  # (name, size, frame) -> floats, or None
        self.frames = frames
        self.timestep = timestep  # pin time and DT to frame * timestep
        self.pacer = pacer
        self.rate = rate if rate is not None else FrameRate()
        self.rest = rest  # seconds yielded after a frame, for a host's threads
        self.poll = poll  # -> True to force a draw this iteration
        self.stopping = stopping
        self.on_frame = on_frame
        self.on_error = on_error  # -> True to keep the loop alive past an error
        self._idle = idle
        self.frame = 0  # iterations, skipped ones included
        self.drawn = 0
        self.elapsed = 0.0
        self.period = None  # measured frame period, the DT a scene is fed
        self._prev = None
        self._buf = None

    @property
    def out(self):
        """The destination names the frames reach (ghee.stats.DEST_*)."""
        return [s.dest for s in self.sinks if s.dest]

    def idle(self):
        """How long a skipped (clean) frame sleeps before re-polling."""
        if self._idle is not None:
            return self._idle(self.period)
        if self.pacer is not None:
            return self.pacer.idle_period()
        return self.period or 1.0 / 60.0

    def run(self):
        r = self.renderer
        scanout = any(s.scanout for s in self.sinks)
        # Only a run that never ends, into sinks that all keep showing their
        # last frame, may skip a frame a controller reports clean. A file or a
        # pipe would lose it.
        skippable = (
            self.frames == 0
            and bool(self.sinks)
            and all(s.persists for s in self.sinks)
        )
        lit = False
        started = time.monotonic()
        try:
            while (self.frames == 0 or self.frame < self.frames) and not self._done():
                try:
                    t0 = time.monotonic()
                    forced = bool(self.poll()) if self.poll is not None else False
                    # the t a controller is handed; unpinned it assumes a 60Hz
                    # cadence, so a slower panel gets a proportionally scaled t
                    t = (
                        self.frame * self.timestep
                        if self.timestep is not None
                        else self.frame / 60.0
                    )
                    if skippable and not should_draw(
                        forced, self.controller, self.frame, t
                    ):
                        self.frame += 1
                        time.sleep(self.idle())
                        continue
                    if self._prev is not None:
                        self.period = t0 - self._prev
                    self._prev = t0
                    # The one place DT is fed: the real frame period, or the
                    # pinned step when the render has to be reproducible. Until
                    # a period has been measured the engine's default stands.
                    if self.timestep is not None:
                        r.set_dt(self.timestep)
                    elif self.period is not None:
                        r.set_dt(self.period)
                    if self.controller is not None:
                        self.controller.update(r, self.frame, t)
                    if self.timestep is not None:
                        r.set_time(t)
                    feed = self._feed(self.frame)
                    readers = [s for s in self.sinks if s.wants(self.frame)]
                    size = self._readback(readers)
                    pool = self._pool(readers, size)
                    buf = self._target(readers, size, pool)
                    if scanout:
                        # present() reads back through a downscale blit or not
                        # at all; there is no direct full-frame path here.
                        r.present(
                            feed,
                            out=buf,
                            out_size=None
                            if buf is None
                            else (size or (r.width, r.height)),
                        )
                    elif buf is None:
                        r.draw(feed)
                    else:
                        r.render(feed, out=buf, out_size=size)
                    drew = time.monotonic()
                    if pool is not None:
                        pool.submit(buf)
                    else:
                        for s in readers:
                            s.write(buf)
                    for s in self.sinks:
                        s.mirror(r)
                    if not lit:  # nothing was on the panel to light up before
                        lit = True
                        for s in self.sinks:
                            s.turn_on()
                    self.frame += 1
                    self.drawn += 1
                    end = time.monotonic()
                    self.elapsed = end - started
                    self.rate.tick(end)
                    if self.on_frame is not None:
                        self.on_frame(self, end - t0)
                    if self.pacer is not None:
                        now = time.monotonic()
                        time.sleep(self.pacer.sleep_for(now - drew, now - t0))
                    elif self.rest:
                        time.sleep(self.rest)
                except Exception as e:
                    if self.on_error is None or not self.on_error(e):
                        raise
                    time.sleep(0.25)
        finally:
            for s in self.sinks:
                s.clear()
                s.turn_off()

    def _done(self):
        return self.stopping is not None and bool(self.stopping())

    def _feed(self, frame):
        """The array provider for one frame: the controller first, the host's
        source (or the placeholder) for anything it does not answer."""
        ctrl = self.controller
        provider = self.provider
        if ctrl is None:
            if provider is None:
                return None
            return lambda name, size: provider(name, size, frame)

        def feed(name, size):
            data = ctrl.provide(name, size, frame)
            if data is not None:
                return data
            return None if provider is None else provider(name, size, frame)

        return feed

    @staticmethod
    def _readback(readers):
        """The size the frame is read back at for this set of readers: None for
        the render size read back directly, else the largest downscale asked
        for (a smaller consumer downscales further itself)."""
        sizes = [s.readback() for s in readers]
        if not sizes or any(x is None for x in sizes):
            return None
        return max(x[0] for x in sizes), max(x[1] for x in sizes)

    @staticmethod
    def _pool(readers, size):
        """The one sink the loop may render straight into: a pooled sink alone
        on the frame, at the render size. Anything else takes a copy."""
        if len(readers) == 1 and readers[0].pooled and size is None:
            return readers[0]
        return None

    def _target(self, readers, size, pool):
        """The buffer this frame is read back into, or None for no readback."""
        if not readers:
            return None
        r = self.renderer
        shape = (
            (r.height, r.width, r.components)
            if size is None
            else (size[1], size[0], r.components)
        )
        if pool is not None:
            return pool.acquire(shape)
        if all(s.copies for s in readers):
            return self._reuse(shape)
        # a sink keeps the frame past write() (hands it to another thread), so
        # the next readback must not land in the same memory
        return np.empty(shape, dtype=np.uint8)

    def _reuse(self, shape):
        if self._buf is None or self._buf.shape != shape:
            self._buf = np.empty(shape, dtype=np.uint8)
        return self._buf
