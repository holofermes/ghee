"""Name-set ratchets for the surface embedding hosts import. Every name pinned
here is reached from code outside this repo, so renaming or dropping one is an
API break for external consumers rather than an internal refactor, and it has to
be a deliberate, versioned decision; when one of these fails the fix is to
restore the name or ship the break knowingly, never to edit the assert until it
passes.
"""

import inspect
import os
import queue
from pathlib import Path

import ghee
import ghee.feeds
from ghee.console import web_dir
from ghee.console.core import ConsoleHost, static_path
from ghee.console.server import H264Fanout, run
from ghee.engine import Renderer, probe_display
from ghee.stats import FrameRate

EXPORTS = {
    "UNCHANGED",
    "Scene",
    "Pass",
    "Param",
    "Channel",
    "ArrayInput",
    "Arg",
    "SceneController",
    "load_scene",
    "load_filters",
    "apply_filters",
    "__version__",
}

RUN_KWARGS = {
    "arrays",
    "hwinfo",
    "scenes_dir",
    "size",
    "port",
    "host",
    "display",
    "rgb565",
    "sinks",
    "store",
    "lib_path",
}

HOST_SURFACE = {
    "writable",
    "save_refused",
    "stats",
    "scene_names",
    "active_name",
    "scene_def",
    "filters",
    "hwinfo",
    "set_param",
    "switch",
    "step",
    "apply_scene",
    "save_scene",
}

RENDERER_CALLED = {
    "setup",
    "render",
    "draw",
    "present",
    "set_param",
    "set_scene_param",
    "pack",
    "display_vrefresh",
    "max_texture_units",
    "data_texture_linear",
    "timings",
    "set_profiling",
    "set_time",
    "set_dt",
    "last_error",
    "close",
}

RENDERER_SURFACE = RENDERER_CALLED | {"reset"}

FANOUT_SURFACE = {"size", "color", "nclients", "want_key", "push", "add", "remove"}

JS_SURFACE = (
    "function attachH264(canvas, url, opts)",
    "onconfig",
    "onstats",
    "onfail",
    "decoded",
    "fps:",
    "queue:",
)


def public(obj):
    return {n for n in vars(obj) if not n.startswith("_")}


def test_the_package_exports_exactly_these_names():
    assert set(ghee.__all__) == EXPORTS
    assert [n for n in ghee.__all__ if not hasattr(ghee, n)] == []


def test_the_console_run_takes_the_embedding_keywords():
    params = inspect.signature(run).parameters
    assert RUN_KWARGS <= set(params)
    for name in sorted(RUN_KWARGS):
        p = params[name]
        assert p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY), name
        assert p.default is not inspect.Parameter.empty, name


def test_the_console_host_surface_is_what_a_host_subclasses():
    assert public(ConsoleHost) == HOST_SURFACE


def test_probe_display_takes_the_device_and_the_lib_path():
    assert set(inspect.signature(probe_display).parameters) == {
        "dri_device",
        "lib_path",
    }


def test_the_renderer_surface_is_what_hosts_and_benches_call():
    assert RENDERER_CALLED <= public(Renderer)
    assert public(Renderer) == RENDERER_SURFACE


def test_the_console_web_root_holds_the_files_a_host_serves():
    root = web_dir()
    assert os.path.isdir(root)
    assert os.path.isfile(os.path.join(root, "index.html"))
    assert static_path(root, "/h264client.js")


def test_the_fanout_hands_a_client_a_queue_to_drain():
    """A client's .q is private-shaped and drained across a repo boundary: an
    embedding host's own /h264 handler reads it directly, so it is load-bearing
    for external consumers and pinned as it stands."""
    fanout = H264Fanout(rate=FrameRate())
    client = fanout.add()
    assert isinstance(client.q, queue.Queue)
    assert FANOUT_SURFACE <= (public(H264Fanout) | public(fanout))


def test_the_older_feeds_import_path_still_resolves():
    from ghee.console.feeds import demo_feed

    assert demo_feed is ghee.feeds.demo_feed


def test_the_js_client_keeps_its_callback_surface():
    """An external page attaches to /h264 through this file and reads the
    onstats payload as {decoded, fps, queue}."""
    src = Path(web_dir(), "h264client.js").read_text()
    assert [s for s in JS_SURFACE if s not in src] == []
