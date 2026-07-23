"""On-demand rendering: the skip decision, the poll interval, and pacer idle.

A scene whose controller reports dirty() False may have its frame skipped
entirely; these cover the pure decision logic without touching GL.
"""

from ghee.console.server import _should_render, _skip_interval
from ghee.engine import Renderer
from ghee.output import AdaptivePacer, FixedPacer
from ghee.procedural import SceneController


class Clean(SceneController):
    def dirty(self, frame, t):
        return False


class Dirty(SceneController):
    def dirty(self, frame, t):
        return True


def test_should_render_dirty_renders():
    assert _should_render(False, Dirty(), 5, 0.1) is True


def test_should_render_clean_skips():
    assert _should_render(False, Clean(), 5, 0.1) is False


def test_should_render_queued_command_forces():
    assert _should_render(True, Clean(), 5, 0.1) is True


def test_should_render_no_controller_free_runs():
    assert _should_render(False, None, 5, 0.1) is True


def test_should_render_first_frame_always_draws():
    # something must reach the sink once, even for a scene born clean
    assert _should_render(False, Clean(), 0, 0.0) is True


def test_default_controller_is_always_dirty():
    # unchanged behavior for every scene that doesn't opt in
    assert _should_render(False, SceneController(), 5, 0.1) is True


def test_interval_display_uses_vrefresh():
    assert _skip_interval(60, 0, 15, 0.05) == 1.0 / 60.0
    assert _skip_interval(75, 3, 15, 0.05) == 1.0 / 75.0  # display wins over preview


def test_interval_no_clients_blocks():
    assert _skip_interval(0, 0, 15, 0.05) is None


def test_interval_capped_preview():
    assert _skip_interval(0, 1, 15, 0.05) == 1.0 / 15.0


def test_interval_uncapped_preview_uses_render_period():
    assert _skip_interval(0, 1, 0, 0.021) == 0.021


def test_fixed_pacer_idle_period():
    assert FixedPacer(30).idle_period() == 1.0 / 30.0


def test_adaptive_pacer_idle_period_tracks_writes():
    p = AdaptivePacer(factor=2.0, max_fps=60)
    assert p.idle_period() == p.min_period  # nothing measured yet
    p.sleep_for(0.05, 0.06)  # a 50ms sink write
    assert p.idle_period() == 0.05 * 2.0


class _FeedHarness:
    """Renderer._feed on a fake engine: records texture uploads."""

    def __init__(self):
        self.calls = []
        harness = self

        class Lib:
            def ghee_mp_set_float_texture(self, handle, pi, enc, data, w, h):
                harness.calls.append((pi, enc, len(data), w, h))

        self.lib = Lib()
        self.handle = None
        self._feeds = {"text": [(0, (4, 2))]}

    def feed(self, provider):
        Renderer._feed(self, provider)


def test_unchanged_is_exported():
    import ghee
    from ghee.engine import UNCHANGED

    assert ghee.UNCHANGED is UNCHANGED  # one identity, package-level
    assert repr(ghee.UNCHANGED) == "ghee.UNCHANGED"


def test_unchanged_skips_upload():
    import ghee

    h = _FeedHarness()
    h.feed(lambda name, size: ghee.UNCHANGED)
    assert h.calls == []  # sentinel: the texture already holds the data


def test_real_data_still_uploads():
    h = _FeedHarness()
    h.feed(lambda name, size: [0.5] * 8)
    assert len(h.calls) == 1
    _, _, n, w, hh = h.calls[0]
    assert (n, w, hh) == (8, 4, 2)
