"""The render loop: what it asks the renderer for, and what each sink gets.

One loop feeds every output ghee has, so the sink list is what decides whether a
frame is read back, at what size, into whose buffer, and whether a clean frame
may be skipped at all. Driven against a recording renderer, so none of it needs
a GPU.
"""

import numpy as np
import pytest

from ghee.loop import Loop, should_draw
from ghee.output import Output, Scanout
from ghee.procedural import SceneController
from ghee.stats import DEST_FILE, DEST_KMS, DEST_PANEL


class Recorder:
    """A Renderer stand-in that records what the loop asked of it."""

    width, height, components = 64, 48, 4
    error_line = error_pass = None

    def __init__(self):
        self.calls = []  # (kind, buffer shape or None, out_size)
        self.buffers = []  # the buffer object handed to each readback
        self.dts = []
        self.times = []
        self.updates = []
        self.fed = []

    def setup(self, scene):
        return ""

    def display_vrefresh(self):
        return 0

    def _note(self, kind, out, out_size, provider):
        self.calls.append((kind, None if out is None else out.shape, out_size))
        self.buffers.append(out)
        if provider is not None:
            self.fed.append(provider("vals", 4))

    def render(self, provider=None, out=None, out_size=None):
        self._note("render", out, out_size, provider)
        return out

    def draw(self, provider=None):
        self._note("draw", None, None, provider)

    def present(self, provider=None, out=None, out_size=None):
        self._note("present", out, out_size, provider)

    def set_scene_param(self, *a):
        pass

    def set_param(self, *a):
        pass

    def set_dt(self, value):
        self.dts.append(value)

    def set_time(self, t):
        self.times.append(t)


class Sink(Output):
    """A recording sink; the class attributes under test are constructor args."""

    dest = "test"

    def __init__(self, dest="test", size=None, take=True, **flags):
        self.dest = dest
        self._size = size
        self._take = take
        self.frames = []
        self.mirrors = 0
        self.on = self.off = self.cleared = 0
        for k, v in flags.items():
            setattr(self, k, v)

    def wants(self, frame):
        return self._take

    def readback(self):
        return self._size

    def write(self, frame):
        self.frames.append(frame)

    def mirror(self, renderer):
        self.mirrors += 1

    def turn_on(self):
        self.on += 1

    def turn_off(self):
        self.off += 1

    def clear(self):
        self.cleared += 1


class Pool(Sink):
    """A sink the loop may render straight into, like ImageSequence."""

    pooled = True

    def __init__(self, **kw):
        super().__init__(**kw)
        self.lent = []
        self.submitted = []

    def acquire(self, shape):
        buf = np.zeros(shape, dtype=np.uint8)
        self.lent.append(buf)
        return buf

    def submit(self, buf):
        self.submitted.append(buf)


class Clean(SceneController):
    def dirty(self, frame, t):
        return False


def _run(sinks, **kw):
    r = Recorder()
    loop = Loop(r, sinks, **kw)
    loop.run()
    return r, loop


# what reaches a sink


def test_one_frame_reaches_every_sink():
    a, b = Sink(dest="a"), Sink(dest="b")
    r, _ = _run([a, b], frames=3)
    assert len(a.frames) == len(b.frames) == 3
    assert a.frames[0] is b.frames[0]  # one readback, one buffer, both sinks


def test_a_sink_that_takes_no_frame_gets_no_readback():
    s = Sink(take=False)
    r, _ = _run([s], frames=2)
    assert [c[0] for c in r.calls] == ["draw", "draw"]
    assert s.frames == []


def test_the_readback_is_the_largest_downscale_asked_for():
    r, _ = _run([Sink(size=(32, 24)), Sink(size=(20, 40))], frames=1)
    assert r.calls[0] == ("render", (40, 32, 4), (32, 40))


def test_a_sink_needing_the_whole_frame_wins_over_a_downscale():
    """None is the render size read back directly; a smaller consumer
    downscales the big frame itself rather than costing it detail."""
    r, _ = _run([Sink(size=None), Sink(size=(32, 24))], frames=1)
    assert r.calls[0] == ("render", (48, 64, 4), None)


def test_a_pooled_sink_alone_is_rendered_straight_into():
    p = Pool()
    r, _ = _run([p], frames=3)
    assert len(p.submitted) == 3
    assert p.lent == p.submitted  # the loop drew into the sink's own buffers
    assert r.buffers[0] is p.lent[0]
    assert p.frames == []  # write() never ran: no copy anywhere


def test_a_pooled_sink_beside_another_takes_a_copy():
    p, s = Pool(), Sink()
    r, _ = _run([p, s], frames=2)
    assert p.submitted == []  # the pool cannot own a buffer two sinks read
    assert len(p.frames) == len(s.frames) == 2


def test_a_pooled_sink_asked_for_a_downscale_takes_a_copy():
    p = Pool(size=(16, 12))
    r, _ = _run([p], frames=1)
    assert p.submitted == []
    assert p.frames[0].shape == (12, 16, 4)


def test_a_sink_that_keeps_the_frame_gets_a_buffer_of_its_own():
    s = Sink(copies=False)
    _run([s], frames=3)
    assert len({id(f) for f in s.frames}) == 3  # nothing overwrote a held frame


def test_a_sink_that_copies_gets_one_buffer_reused():
    s = Sink()
    _run([s], frames=3)
    assert len({id(f) for f in s.frames}) == 1


# the scanout


def test_a_scanout_presents_and_never_takes_a_frame():
    k = Scanout()
    r, loop = _run([k], frames=2)
    assert [c[0] for c in r.calls] == ["present", "present"]
    assert r.calls[0] == ("present", None, None)
    assert loop.out == [DEST_KMS]


def test_a_scanout_beside_a_reader_reads_the_frame_back_too():
    """present() has no direct full-frame readback, so the size is explicit."""
    r, _ = _run([Scanout(), Sink()], frames=1)
    assert r.calls[0] == ("present", (48, 64, 4), (64, 48))


def test_a_scanout_beside_a_downscaling_reader_downscales_on_the_gpu():
    r, _ = _run([Scanout(), Sink(size=(32, 24))], frames=1)
    assert r.calls[0] == ("present", (24, 32, 4), (32, 24))


# a mirror sink samples the GPU itself


def test_a_mirror_sink_runs_after_the_draw():
    m = Sink(take=False)
    r, _ = _run([m], frames=3)
    assert m.mirrors == 3
    assert m.frames == []


# DT and time, fed in one place


def test_dt_is_the_measured_frame_period():
    r, _ = _run([Sink()], frames=4)
    assert len(r.dts) == 3  # frame 0 has no previous frame to measure against
    assert all(0.0 < v < 5.0 for v in r.dts)
    assert r.times == []  # the engine's own clock still runs the scene


def test_a_pinned_timestep_is_both_the_dt_and_the_time():
    r, _ = _run([Sink()], frames=3, timestep=0.5)
    assert r.dts == [0.5, 0.5, 0.5]
    assert r.times == [0.0, 0.5, 1.0]


def test_the_first_frame_keeps_the_engine_default_dt():
    r, _ = _run([Sink()], frames=1)
    assert r.dts == []


# skipping a clean frame


def test_a_clean_frame_is_skipped_when_every_sink_holds_its_last():
    s = Sink(persists=True)
    stop = {"n": 0}

    def stopping():
        stop["n"] += 1
        return stop["n"] > 4

    r, loop = _run([s], controller=Clean(), stopping=stopping, idle=lambda p: 0.0)
    assert loop.frame > 1
    assert loop.drawn == 1  # the first frame draws, the rest are skipped
    assert len(s.frames) == 1


def test_a_sink_that_does_not_hold_its_last_frame_is_never_skipped():
    s = Sink(persists=False)
    r, loop = _run([s], controller=Clean(), frames=4)
    assert loop.drawn == 4
    assert len(s.frames) == 4


def test_a_finite_render_draws_every_frame_even_for_a_clean_scene():
    s = Sink(persists=True)
    r, loop = _run([s], controller=Clean(), frames=5)
    assert loop.drawn == 5


def test_a_skipped_frame_still_advances_the_frame_number():
    seen = []

    class Watcher(SceneController):
        def dirty(self, frame, t):
            seen.append(frame)
            return False

    stop = {"n": 0}
    _run(
        [Sink(persists=True)],
        controller=Watcher(),
        stopping=lambda: stop.__setitem__("n", stop["n"] + 1) or stop["n"] > 4,
        idle=lambda p: 0.0,
    )
    assert seen == sorted(set(seen)) and len(seen) > 1


def test_the_idle_wait_falls_back_to_the_measured_period():
    """Without a pacer or a host's own interval there is still an answer: a
    scanout render with no --fps has neither."""
    loop = Loop(Recorder(), [Scanout()])
    assert loop.idle() == pytest.approx(1.0 / 60.0)
    loop.period = 0.02
    assert loop.idle() == 0.02


# the frame budget and what it publishes


def test_frames_is_the_budget_and_drawn_is_what_landed():
    r, loop = _run([Sink()], frames=7)
    assert (loop.frame, loop.drawn) == (7, 7)
    assert loop.elapsed > 0


def test_an_endless_loop_runs_until_the_host_stops_it():
    stop = {"n": 0}

    def stopping():
        stop["n"] += 1
        return stop["n"] > 3

    r, loop = _run([Sink()], stopping=stopping)
    assert loop.drawn == 3


def test_out_names_every_destination_the_frames_reach():
    loop = Loop(Recorder(), [Scanout(), Sink(dest=DEST_PANEL), Sink(dest=DEST_FILE)])
    assert loop.out == [DEST_KMS, DEST_PANEL, DEST_FILE]
    assert Loop(Recorder(), [Sink(dest="")]).out == []


def test_on_frame_reports_each_frames_own_cost():
    costs = []
    _run([Sink()], frames=3, on_frame=lambda lp, cost: costs.append(cost))
    assert len(costs) == 3 and all(c >= 0 for c in costs)


# the sinks' power state


def test_a_sink_is_turned_on_once_there_is_a_frame_to_show():
    s = Sink()
    _run([s], frames=3)
    assert s.on == 1
    assert s.frames  # the backlight came up over a drawn frame, not a blank panel


def test_a_sink_is_blanked_and_turned_off_on_the_way_out():
    s = Sink()
    _run([s], frames=2)
    assert (s.cleared, s.off) == (1, 1)


def test_a_run_that_drew_nothing_never_turns_a_sink_on():
    s = Sink()
    _run([s], frames=0, stopping=lambda: True)
    assert (s.on, s.cleared, s.off) == (0, 1, 1)


# the controller


def test_the_controller_can_be_swapped_mid_run():
    class First(SceneController):
        def update(self, renderer, frame, t):
            renderer.updates.append("first")

    class Second(SceneController):
        def update(self, renderer, frame, t):
            renderer.updates.append("second")

    r = Recorder()
    loop = Loop(r, [Sink()], controller=First(), frames=4)
    loop.on_frame = lambda lp, cost: setattr(lp, "controller", Second())
    loop.run()
    assert r.updates == ["first", "second", "second", "second"]


def test_a_controller_answers_for_its_own_arrays_and_the_provider_for_the_rest():
    class Half(SceneController):
        def provide(self, name, size, frame):
            return [1.0] if name == "mine" else None

    r = Recorder()
    Loop(
        r,
        [Sink()],
        controller=Half(),
        provider=lambda name, size, frame: ["placeholder"],
        frames=1,
    ).run()
    assert r.fed == [["placeholder"]]  # the recorder asks for "vals"


def test_no_provider_and_no_controller_leaves_the_arrays_alone():
    r = Recorder()
    Loop(r, [Sink()], frames=1).run()
    assert r.fed == []  # Renderer._feed(None) skips every upload


# errors


def test_an_error_ends_the_run_by_default():
    class Boom(Sink):
        def write(self, frame):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _run([Boom()], frames=2)


def test_a_host_may_keep_the_loop_alive_through_an_error():
    seen = []

    class Boom(Sink):
        def write(self, frame):
            raise RuntimeError("boom")

    stop = {"n": 0}

    def stopping():
        stop["n"] += 1
        return stop["n"] > 2

    _run(
        [Boom()],
        stopping=stopping,
        on_error=lambda e: seen.append(str(e)) or True,
    )
    assert seen == ["boom", "boom"]


def test_a_sink_is_still_turned_off_when_a_frame_raises():
    class Boom(Sink):
        def write(self, frame):
            raise RuntimeError("boom")

    s = Boom()
    with pytest.raises(RuntimeError):
        _run([s], frames=1)
    assert (s.cleared, s.off) == (1, 1)


# the skip decision itself


def test_should_draw_is_the_one_rule_for_it():
    assert should_draw(True, Clean(), 5, 0.1) is True  # a queued command
    assert should_draw(False, Clean(), 0, 0.0) is True  # the first frame
    assert should_draw(False, None, 5, 0.1) is True  # no controller
    assert should_draw(False, Clean(), 5, 0.1) is False
