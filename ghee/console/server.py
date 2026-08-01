"""ghee console render server.

Attaches the console to a render loop (ghee.loop): it owns the active scene, the
scene list and the command queue the loop drains, and serves the JSON API and
the static PWA. The loop itself is not the console's, so a render can have one
without a browser and a browser without giving up its panel.

Standalone (animated placeholder data, no host):
    python -m ghee.console.server [scenes_dir] [WxH] [port]

Embedded (a host injects real data):
    from ghee.console.server import run
    run(arrays={"levels": my_source}, hwinfo=host_hwinfo)

API:
  GET  /stream[?vid=]     MJPEG, one part per rendered frame
  GET  /h264[?vid=]       framed H264 stream ([u8 type][u32 len]: 0=config JSON,
                          1=key AU, 2=delta AU, AVCC) for WebCodecs; one shared
                          encoder fans out to all viewers. Hardware zero-copy
                          where /dev/video11 exists, ffmpeg/libx264 otherwise.
  POST /ping?vid=         heartbeat for the streams opened under that vid
  GET  /frame.png         one lossless PNG of the next frame, at the render size.
                          The request rides the command queue, so a /set that
                          returned before it is in the frame. 409 while the scene
                          has an error, 503 when the loop draws nothing.
  GET  /stats             {fps, frame_ms, stream_fps, out, error, error_line,
                          error_pass, active, display, stale}
  GET  /scenes            {scenes:[name...], active}
  GET  /filters           {name: filter def} registry for the "+ filter" picker
  GET  /scene?name=       resolved scene JSON (shader files inlined, filters
                          expanded). The active scene answers with the document
                          that is rendering: applied edits and /set values
                          included. Any other name answers from disk.
  POST /set?pass=&name=&value=   live uniform, no recompile (pass=scene = scene-scoped)
  POST /next /prev        switch active scene
  POST /switch?name=      switch to a named scene; the active name reloads it,
                          which is how a stale scene picks up its files again
  POST /scene             full scene JSON -> recompile active (apply working copy)
  POST /save              write the scene as a modified copy in <scenes>/modified/;
                          a procedural scene saves a .values.json sidecar (param
                          values + filters) applied over its defaults at load;
                          the original file is never touched (demo scenes refuse)
  GET  /<file>            static file from web/
"""

import io
import os
import queue
import select
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from PIL import Image

from ghee.console import examples_dir, web_dir
from ghee.console.core import (
    ConsoleAPI,
    ConsoleHost,
    NotFound,
    Response,
    ServerError,
    parse_query,
)
from ghee.console.feeds import default_hwinfo
from ghee.console.store import SceneStore
from ghee.engine import Renderer, probe_dri_devices
from ghee.feeds import demo_feed
from ghee.h264 import (
    HW_COLOR,
    NAL_IDR,
    NAL_PPS,
    NAL_SEI,
    NAL_SLICE,
    NAL_SPS,
    SW_COLOR,
    HardwareH264,
    avcc,
    build_config,
    split_annexb,
    wire_frame,
)
from ghee.loop import Loop
from ghee.output import Output, Scanout
from ghee.sources import Snapshot, collect
from ghee.stats import DEST_H264, DEST_KMS, DEST_MJPEG, FrameRate

# None -> Renderer auto-locates the .so bundled in the ghee package.
DEFAULT_LIB = os.environ.get("GHEE_LIB")

# Set by run(); the render thread and request handler read them as globals.
_lib_path = None
WEB: str = ""
STORE: SceneStore = None
W = H = 0
_provider = None
_hwinfo = default_hwinfo
_display = False  # also scan out to a connected KMS display (headless Pi)
_rgb565 = False  # display scanout in 16-bit color (halves GPU store + scanout bytes)
_sinks = ()  # output sinks beside the console's own: a panel, files, a pipe
_components = 4  # frame components the renderer reads back (3 = a panel packs RGB)
_frames = 0  # 0 = render until stop(); a count ends the console with the render
_timestep = None  # pin engine time to frame * this, for a reproducible render
_dri = None  # DRI device override, else probed
_preview_max = (
    480  # longest side of the browser preview (downscaled off the render thread)
)
PREVIEW_QUALITY = 75  # JPEG quality for the MJPEG preview
_preview_fps = 15  # browser preview refresh cap; the display/panel stays full speed
_raw = {"rgb": None, "seq": 0}  # latest preview frame handed to the encoder thread
_raw_lock = threading.Condition()
_httpd = None
_stopping = threading.Event()
_fatal = ""  # a failure the console cannot render through; run() raises it
# Seconds run() gives the render thread to unwind: blank and power down its
# sinks, drain a queued image writer, close them. Bounded, so a frame wedged on
# a bus cannot hold the process. A blank over a slow panel bus is under a second
# and a default write buffer of 720p PNGs about a second; a deeper --write-buffer
# can outlast this, and loses the tail of its queue when it does.
RENDER_JOIN_TIMEOUT = 10.0


def _set_item(scope, name, value):
    """Render-queue item for a /set. scope="scene" targets the scene's shared
    params: a distinct item type the loop applies to every pass. Anything
    else is a pass index."""
    if scope == "scene":
        return ("scene_param", name, value)
    return ("param", int(scope), name, value)


def _coerce(ptype, val):
    """The param's declared type wins over whatever the wire carried: an int
    rounds, a bool is a real bool. A color or vec arrives as a list and
    passes through."""
    if isinstance(val, list):
        return val
    if ptype == "int":
        return int(round(val))
    if ptype == "bool":
        return bool(val)
    return val


_running = {"cur": None}
# The files the running scene was built from. Only a load off disk replaces it,
# so an applied working copy leaves a scene whose files moved on still stale.
_sources = {"cur": None}

_latest = {"jpg": None, "seq": 0}
_lock = threading.Condition()
_nclients = 0
_nclients_lock = threading.Lock()
_q = queue.Queue()
_stat = {
    "frame_ms": 0.0,
    "error": "",
    "active": None,
    "display": {},
    "error_line": None,
    "error_pass": None,
}
_frame_rate = FrameRate()  # loop iterations that drew
_stream_rate = FrameRate()  # frames handed to viewers, whichever codec


def _out():
    """Where this console's frames are going right now: the scanout and the
    output sinks it was given, plus whichever streams have a viewer."""
    out = []
    if _display:
        out.append(DEST_KMS)
    out.extend(s.dest for s in _sinks if s.dest)
    if _fanout.nclients:
        out.append(DEST_H264)
    if _nclients:
        out.append(DEST_MJPEG)
    return out


def _format():
    """The pixel format the frames end up in: a panel's own, else the readback's."""
    for s in _sinks:
        fmt = getattr(s, "native_format", None)
        if fmt:
            return fmt
    return "rgba"


def stats():
    """The /stats payload. The rates are computed on read, so an idle loop's
    number falls with the clock instead of being written by a frame that never
    happened."""
    snap = _sources["cur"]
    return dict(
        _stat,
        fps=_frame_rate.value(),
        stream_fps=_stream_rate.value(),
        out=_out(),
        stale=snap is not None and snap.stale(),
    )


def _reset_stats():
    """Everything published describes the scene running now, not the one
    before: the rate windows, the last frame's cost, the compile error."""
    _frame_rate.reset()
    _stream_rate.reset()
    _stat["frame_ms"] = 0.0
    _stat["error"] = ""
    _stat["error_line"] = None
    _stat["error_pass"] = None


def _publish(scene):
    """Republish the running document from the scene the loop holds. Called by
    the render thread after anything that changes what is rendering: a load, an
    applied scene that compiled, a /set that landed."""
    _running["cur"] = (_stat["active"], scene.to_dict())


def _count_clients(delta):
    """MJPEG viewers, adjusted from handler threads and read by the render and
    encoder threads."""
    global _nclients
    with _nclients_lock:
        _nclients += delta
        return _nclients


def _make_provider(arrays):
    """arrays: {name -> callable(size, frame) -> floats}. Returns provide(name,
    size, frame) that uses the host's source for a name or the placeholder."""

    def provide(name, size, frame):
        src = arrays.get(name)
        return src(size, frame) if src is not None else demo_feed(name, size, frame)

    return provide


def _make_renderer():
    """Create the render-thread Renderer, probing DRI devices so the console runs on
    the Pi (card0) and on desktop Mesa (renderD128). components defaults to 4 because
    desktop Mesa rejects an RGB glReadPixels; a panel sink asks for 3, which is what
    it packs from."""
    if _display:
        return Renderer(
            W,
            H,
            components=_components,
            display=True,
            rgb565=_rgb565,
            dri_device=_dri or os.environ.get("GHEE_DRI") or "/dev/dri/card0",
            lib_path=_lib_path,
        )
    errs = []
    for dev in [_dri] if _dri else probe_dri_devices():
        try:
            return Renderer(
                W, H, components=_components, dri_device=dev, lib_path=_lib_path
            )
        except Exception as e:
            errs.append(f"{dev}: {e}")
    raise RuntimeError("no GL device for the console:\n" + "\n".join(errs))


def encoder_loop():
    # JPEG is expensive, so it runs OFF the render thread: the render loop drops
    # its readback here and this encodes it, downscaling only when the frame was
    # sized for another consumer. The display/panel never waits. Raw frames may
    # flow for the h264 path; JPEG is only produced while a viewer is attached.
    last_seq = -1
    while not _stopping.is_set():
        with _raw_lock:
            if _nclients == 0 or _raw["seq"] == last_seq:
                _raw_lock.wait(0.25)
            fresh = _nclients > 0 and _raw["seq"] != last_seq
            rgb = _raw["rgb"] if fresh else None
            last_seq = _raw["seq"]
        if rgb is None:
            continue
        mode = "RGBA" if rgb.shape[2] == 4 else "RGB"
        img = Image.fromarray(rgb, mode).convert("RGB")
        w, h = img.size
        scale = _preview_max / max(w, h)
        if scale < 1.0:  # sized for the software H264 stream, or a bigger host frame
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=PREVIEW_QUALITY)
        with _lock:
            _latest["jpg"] = buf.getvalue()
            _latest["seq"] += 1
            _lock.notify_all()
        _stream_rate.tick()


# A lagging client's backlog never drains on its own (producer and consumer
# run at the same rate), so it becomes permanent added latency. Past this many
# queued frames the client is dropped to resync at a keyframe instead.
H264_LAG_FRAMES = 10


class _H264Client:
    def __init__(self):
        self.q = queue.Queue(maxsize=32)
        self.started = False
        self.lock = threading.Lock()

    def offer(self, frame, config):
        # frame/config are wire-framed once by the fanout and shared across
        # clients; bytes are immutable.
        with self.lock:
            if self.started and self.q.qsize() >= H264_LAG_FRAMES:
                self.started = False  # lagging: drop the rest of this GOP
                while not self.q.empty():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        break
            if not self.started:
                if config is not None:
                    self.q.put(config)
                    self.q.put(frame)
                    self.started = True
                return
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                self.started = False  # backstop; the lag check above fires first


class H264Fanout:
    """One shared H264 stream, every /h264 viewer. A new viewer starts at a
    keyframe (hardware forces one on join; software joins at the next GOP).

    rate is the FrameRate the pushes are counted into. A host serving another
    codec as well passes the counter it publishes as stream_fps, so one number
    covers every viewer it has.
    """

    def __init__(self, rate=None):
        self.clients = []
        self.lock = threading.Lock()
        self.cfg = None
        self.size = (0, 0)  # encoded dimensions, set by whoever starts the encoder
        self.color = None  # decoder colorSpace hint (HW_COLOR/SW_COLOR), same owner
        self.want_key = False
        self.rate = rate if rate is not None else FrameRate()

    @property
    def nclients(self):
        return len(self.clients)

    def stream_fps(self):
        return self.rate.value()

    def add(self):
        c = _H264Client()
        with self.lock:
            self.clients.append(c)
        self.want_key = True
        return c

    def remove(self, c):
        with self.lock:
            if c in self.clients:
                self.clients.remove(c)

    def push(self, au):
        nals = split_annexb(au)
        types = [x[0] & 0x1F for x in nals]
        sps = next((x for x in nals if x[0] & 0x1F == NAL_SPS), None)
        pps = next((x for x in nals if x[0] & 0x1F == NAL_PPS), None)
        if sps and pps:
            self.cfg = build_config(sps, pps, *self.size, color=self.color)
        # keep VCL + SEI; drop AUD and parameter sets (the config carries them)
        chunk = avcc(
            [x for x in nals if (x[0] & 0x1F) in (NAL_SLICE, NAL_IDR, NAL_SEI)]
        )
        if not chunk:
            return
        is_key = NAL_IDR in types
        frame = wire_frame(1 if is_key else 2, chunk)
        config = wire_frame(0, self.cfg) if is_key and self.cfg else None
        with self.lock:
            for c in list(self.clients):
                c.offer(frame, config)
            if self.clients:
                self.rate.tick()
            # a waiting (new or resyncing) client rejoins at the next keyframe;
            # keep requesting one until everyone is streaming
            if any(not c.started for c in self.clients):
                self.want_key = True


_fanout = H264Fanout(rate=_stream_rate)
# Hardware encoding failed for the stream running now, so h264_sw_loop owns it.
# Cleared once the last viewer leaves: a stall or a busy codec belongs to the
# stream it happened in, and the software lane costs a readback the hardware
# one does not.
_h264_hw_dead = False
_h264_bitrate = 10_000_000
_H264_FPS = 30  # software-encode cadence (hardware follows the display/loop)
_stream_max = 0  # optional cap on the h264 stream's longest side; 0 = full size


H264_BPP = 0.15  # bits per pixel per frame the stream aims for
H264_BITRATE_CEILING = 80_000_000


def _stream_size(w, h, longest):
    """Stream dimensions: fit the longest side, keep aspect, force even
    (yuv420 needs even dims)."""
    s = min(1.0, longest / max(w, h)) if longest > 0 else 1.0
    return max(2, round(w * s / 2) * 2), max(2, round(h * s / 2) * 2)


def stream_bitrate(w, h, fps, floor):
    derived = w * h * max(1, fps) * H264_BPP
    return int(max(floor, min(derived, H264_BITRATE_CEILING)))


def h264_sw_loop():
    """Software H264 (boards with no /dev/video11: Pi 5, desktops): encode the
    same readback frames the MJPEG preview uses, off the render thread."""
    from ghee.h264 import SoftwareH264

    enc = None
    size = None
    last_seq = -1
    while not _stopping.is_set():
        if not (_h264_hw_dead and _fanout.nclients > 0):
            if enc is not None:
                enc.close()
                enc = None
            with _raw_lock:
                _raw_lock.wait(0.25)
            continue
        with _raw_lock:
            if _raw["seq"] == last_seq:
                _raw_lock.wait(0.25)
            fresh = _raw["seq"] != last_seq
            rgb = _raw["rgb"] if fresh else None
            last_seq = _raw["seq"]
        if rgb is None:
            continue
        h, w = rgb.shape[:2]
        if enc is not None and size != (w, h):
            enc.close()
            enc = None
        if enc is None:
            try:
                enc = SoftwareH264(
                    w,
                    h,
                    fps=_H264_FPS,
                    bitrate=stream_bitrate(w, h, _H264_FPS, _h264_bitrate),
                    components=rgb.shape[2],
                )
            except RuntimeError as e:
                print(f"h264: software encoder unavailable: {e}", file=sys.stderr)
                time.sleep(5)
                continue
            size = (w, h)
            _fanout.size = (w, h)
            _fanout.color = SW_COLOR
        try:
            enc.feed(rgb)
            au, _ = enc.get_au(timeout=1.0)
        except (BrokenPipeError, OSError) as e:
            # a dead encoder process must not kill this thread: respawn
            print(f"h264: software encoder died ({e}); respawning", file=sys.stderr)
            enc.close()
            enc = None
            continue
        if au:
            _fanout.push(au)


def _skip_interval(display_hz, nclients, preview_fps, render_period):
    """How long a clean (dirty=False) scene sleeps before re-polling, derived
    from whoever consumes frames: the display's refresh, else the preview cap,
    else the last measured render period (uncapped preview). None = nobody
    consumes anything, block on the command queue instead."""
    if display_hz > 0:
        return 1.0 / display_hz
    if nclients <= 0:
        return None
    if preview_fps > 0:
        return 1.0 / preview_fps
    return render_period


# How long GET /frame.png waits for the render loop to answer its request.
FRAME_TIMEOUT = 4.0


class FrameShot(Output):
    """One whole frame for a client that asked for a picture (GET /frame.png).

    Requests are generations. request() hands one out and the handler puts a
    token on the command queue; the loop arms the shot where that token lands,
    which takes every generation asked for so far into the next frame. So a
    /set that returned before the request was drained before the arm and is in
    the frame, while a request made after the arm waits for the following draw.
    A request made while another one is still being answered joins it instead
    of asking for a draw of its own, so requests that overlap cost one frame.
    A command queued since the arm ends that: the frame in flight is a command
    short of what the caller has already been told landed, so it asks anew.
    """

    dest = ""  # a still is not somewhere the frames are going; see _out()
    copies = False  # the frame crosses to the handler thread that encodes it
    persists = True  # nothing here waits on frames, so a clean scene may skip

    def __init__(self):
        self.cv = threading.Condition()
        self.asked = 0  # generations handed out
        self.armed = 0  # generations the next frame answers
        self.done = 0  # generations the published frame answers
        self.holders = 0  # requests being answered right now
        self.moved = False  # a command was queued after the arm
        self.frame = None

    def request(self):
        with self.cv:
            if self.holders and not self.moved:
                return self.asked
            self.asked += 1
            return self.asked

    def hold(self):
        """Count a request from here until it has its answer; one made in the
        meantime shares its frame."""
        with self.cv:
            self.holders += 1

    def drop(self):
        with self.cv:
            self.holders -= 1

    def touch(self):
        """A command is on the queue: whatever is being drawn now predates it."""
        with self.cv:
            self.moved = True

    def arm(self):
        with self.cv:
            self.armed = self.asked
            self.moved = False

    def wait(self, gen, timeout):
        """The frame answering this generation, or None if none was published
        in time. Woken early by a shutdown, which publishes nothing more."""
        end = time.monotonic() + timeout
        with self.cv:
            while self.done < gen:
                left = end - time.monotonic()
                if left <= 0 or _stopping.is_set():
                    return None
                self.cv.wait(min(0.25, left))
            return self.frame

    def wants(self, frame):
        return self.armed > self.done

    def readback(self):
        return None  # the render size, read back directly

    def write(self, frame):
        with self.cv:
            self.frame = frame
            self.done = self.armed
            self.cv.notify_all()


_shot = FrameShot()


def _command(item):
    """Queue something that changes what is rendering. Queued first, then
    counted, so a still can only join a frame that is going to carry it."""
    _q.put(item)
    _shot.touch()


class ConsoleSink(Output):
    """The console as an output sink: the throttled, downscaled readback the
    browser preview and the software H264 encoder share, and the hardware H264
    mirror. It reads this module's viewer counts and caps every frame, so what
    it asks the loop for follows viewers coming and going.
    """

    dest = ""  # the console's destinations are its viewers; see _out()
    copies = False  # the readback crosses to the encoder threads
    persists = True  # a browser holds the last frame it was sent
    hz = 0  # display refresh, the rate a hardware encoder is created at

    def __init__(self):
        self.enc = None  # hardware encoder; render-thread owned (it issues GL calls)
        self._last = 0.0  # when the preview was last refreshed
        self._now = 0.0

    def _software(self):
        """Software H264 eats the same readback frames the MJPEG preview does,
        so its viewers count as readback consumers; hardware does not (it
        samples the GPU texture directly)."""
        return _h264_hw_dead and _fanout.nclients > 0

    def wants(self, frame):
        # Refresh the browser preview at most _preview_fps, and only while
        # watched. The display/panel renders every frame regardless; the preview
        # is a throttled, downscaled mirror handled by the encoder thread. The
        # software encoder has its own cadence (_H264_FPS); the fastest active
        # consumer wins.
        if _shot.wants(frame):
            # this frame is read back whole for the still; taking it here would
            # resize the software encoder to the render size and back again
            return False
        rates = []
        if _nclients > 0:
            rates.append(_preview_fps)  # <= 0 means every frame
        if self._software():
            rates.append(_H264_FPS)
        self._now = time.monotonic()
        return bool(rates) and (
            any(r <= 0 for r in rates) or (self._now - self._last) >= 1.0 / max(rates)
        )

    def readback(self):
        # One readback feeds both consumers, so it is sized for whoever is
        # attached: the MJPEG preview can downscale a bigger frame, the software
        # encoder cannot recover a smaller one.
        scale = min(1.0, _preview_max / max(W, H))
        pw, ph = max(1, round(W * scale)), max(1, round(H * scale))
        if self._software():
            sw, sh = _stream_size(W, H, _stream_max)
            return max(pw, sw), max(ph, sh)
        return pw, ph

    def write(self, frame):
        self._last = self._now
        with _raw_lock:
            _raw["rgb"] = frame
            _raw["seq"] += 1
            _raw_lock.notify_all()

    def mirror(self, renderer):
        """Hardware H264 mirror: encode the frame just produced, zero readback.
        keep_last in the engine holds the final pass in its FBO while the
        encoder exists, so encode_last always has a texture to sample."""
        global _h264_hw_dead
        want = _fanout.nclients > 0
        if not want:
            _h264_hw_dead = False  # the next stream tries hardware again
            if self.enc is not None:
                self.enc.close()  # restore direct-scanout speed while unwatched
                self.enc = None
            return
        if self.enc is None and not _h264_hw_dead:
            try:
                w, h = _stream_size(W, H, _stream_max)
                hz = self.hz or 30
                self.enc = HardwareH264(
                    renderer,
                    w,
                    h,
                    fps=hz,
                    bitrate=stream_bitrate(w, h, hz, _h264_bitrate),
                )
                _fanout.size = (w, h)
                _fanout.color = HW_COLOR
            except RuntimeError as e:
                _h264_hw_dead = True  # h264_sw_loop takes this stream over
                print(f"h264: {e}; using software encoder", file=sys.stderr)
        if self.enc is None:
            return
        if _fanout.want_key:
            _fanout.want_key = False
            self.enc.force_keyframe()
        try:
            au, _ = self.enc.encode_last()
        except RuntimeError as e:
            self.enc.close()
            self.enc = None
            _h264_hw_dead = True
            print(f"h264: {e}; using software encoder", file=sys.stderr)
        else:
            if au:
                _fanout.push(au)

    def close(self):
        if self.enc is not None:
            self.enc.close()
            self.enc = None


def render_loop():
    """Run the console's render loop on this thread. Everything the console adds
    to a plain render is here: the command queue, the scene it publishes, the
    stale snapshot, and a frame error that must not kill the thread."""
    # GL is thread-affine: create the Renderer here, in the thread that renders.
    try:
        renderer = _make_renderer()
    except Exception as e:
        return _fail(f"renderer: {e}")
    st: dict = {"scene": None, "controller": None}
    console = ConsoleSink()
    console.hz = renderer.display_vrefresh() if _display else 0
    sinks = ([Scanout()] if _display else []) + list(_sinks) + [console, _shot]
    reported = [""]

    def set_scene(scene, name=None, controller=None, snap=None):
        """The compile error, or "" when this scene is the one now rendering."""
        err = renderer.setup(scene)
        if err:
            _stat["error"] = err
            _stat["error_line"] = renderer.error_line
            _stat["error_pass"] = renderer.error_pass
            if st["scene"] is not None:
                renderer.setup(st["scene"])  # keep the last good scene rendering
            return err
        st["scene"] = scene
        st["controller"] = controller
        loop.controller = controller
        if controller is not None:
            controller.setup(renderer)
            # string params are host-side: hand every current value to the
            # controller so saved/overlaid strings take effect on load
            groups = [scene.params] + [p.params for p in scene.passes]
            for params in groups:
                for nm, pm in params.items():
                    if pm.type == "string":
                        controller.on_param(renderer, nm, pm.value)
        _reset_stats()
        if name:
            _stat["active"] = name
        # before the publish: a /stats between the two must not see the new
        # scene with the old (or no) snapshot and answer stale from it
        if snap is not None:
            _sources["cur"] = snap
        _publish(scene)
        return ""

    def load(name):
        try:
            with collect() as read:
                scene, controller = STORE.resolve(name)
            set_scene(scene, name, controller, snap=Snapshot(read))
        except Exception as e:
            _stat["error"] = f"load {name}: {e}"

    def drain():
        """Apply every queued command, and report whether one landed: a queued
        command (switch, param, scene edit, a new preview client) forces a draw
        this iteration so edits always show."""
        force = False
        edited = False  # a /set landed: the running document has to follow
        while not _q.empty():
            item = _q.get()
            force = True
            if item[0] == "switch":
                load(item[1])
            elif item[0] == "param":
                _, pi, nm, val = item
                scene = st["scene"]
                if scene is None:
                    continue
                try:
                    p = scene.passes[pi]
                    pm = p.params[nm]
                    if pm.type == "string":
                        pm.value = val
                        edited = True
                        ctrl = st["controller"]
                        if ctrl is not None:
                            ctrl.on_param(renderer, nm, val)
                        continue
                    v = _coerce(pm.type, val)
                    renderer.set_param(p.name, nm, pm.type, v)
                    pm.value = v
                    edited = True
                except Exception as e:
                    print(f"set param {nm}: {e}", file=sys.stderr)
            elif item[0] == "scene_param":
                _, nm, val = item
                scene = st["scene"]
                if scene is None:
                    continue
                try:
                    pm = scene.params[nm]
                    if pm.type == "string":
                        pm.value = val
                        edited = True
                        ctrl = st["controller"]
                        if ctrl is not None:
                            ctrl.on_param(renderer, nm, val)
                        continue
                    v = _coerce(pm.type, val)
                    renderer.set_scene_param(nm, pm.type, v)
                    pm.value = v
                    edited = True
                except Exception as e:
                    print(f"set scene param {nm}: {e}", file=sys.stderr)
            elif item[0] == "shot":
                # in queue order, so a /set ahead of it is applied to the frame
                # this arms
                _shot.arm()
            elif item[0] == "scene_obj":
                set_scene(item[1], controller=st["controller"])
        if edited and st["scene"] is not None:
            _publish(st["scene"])
        return force

    def idle(period):
        interval = _skip_interval(
            console.hz, _nclients + _fanout.nclients, _preview_fps, period
        )
        if interval is None:
            _q.put(_q.get())  # nobody consumes frames: sleep until a command
            return 0.0
        return interval

    def on_frame(loop, cost):
        _stat["frame_ms"] = cost * 1000.0

    def on_error(e):
        # one bad frame (or scene, or controller) must not take the thread with
        # it: report it and keep rendering whatever last worked
        msg = f"render: {e}"
        _stat["error"] = msg
        if msg != reported[0]:
            print(f"ghee console: {msg}", file=sys.stderr)
            reported[0] = msg
        return True

    loop = Loop(
        renderer,
        sinks,
        provider=_provider or demo_feed,
        frames=_frames,
        timestep=_timestep,
        rate=_frame_rate,
        rest=0.003,  # yield to the handler and encoder threads
        poll=drain,
        idle=idle,
        stopping=_stopping.is_set,
        on_frame=on_frame,
        on_error=on_error,
    )
    load(STORE.names[0])
    try:
        loop.run()
    finally:
        try:
            for s in sinks:
                s.close()
        finally:
            # whichever way this thread ended there is nothing rendering to
            # serve, and run() is waiting on it
            stop()


class _RenderLoopHost(ConsoleHost):
    """This module's render loop as a console host. Handler threads only read
    _stat and put items on _q; the render thread owns the renderer and drains
    the queue, so nothing here touches GL."""

    save_refused = "demo scenes are read-only; serve your own directory to save"

    @property
    def writable(self):
        return not STORE.read_only

    def stats(self):
        return stats()

    def scene_names(self):
        return STORE.names

    def active_name(self):
        return _stat["active"]

    def scene_def(self, name=None):
        name = name or _stat["active"]
        if name not in STORE:
            raise NotFound("no scene")
        cur = _running["cur"]
        if cur is not None and cur[0] == name:
            return cur[1]
        try:
            scene, _ = STORE.resolve(name)
        except Exception as e:
            raise ServerError(f"build {name}: {e}") from e
        return scene.to_dict()

    def hwinfo(self):
        return _hwinfo()

    def set_param(self, scope, name, value):
        _command(_set_item(scope, name, value))

    def switch(self, name):
        if name not in STORE:
            raise NotFound("no scene")
        _command(("switch", name))
        return name

    def apply_scene(self, scene):
        _command(("scene_obj", scene))

    def save_scene(self, name, scene):
        STORE.save(name, scene)


def _build_api():
    """Rebuilt by run() (and by the tests) because the web root is module state."""
    return ConsoleAPI(_RenderLoopHost(), web=WEB)


API = _build_api()


def _line(msg):
    """First line only: a status body stays one line, whatever it reports."""
    return msg.splitlines()[0] if msg else msg


def _peer_gone(sock):
    """Whether a stream viewer's connection has ended. A viewer sends nothing
    after its request, so anything readable on it is the close.

    Without this a stream handler learns its viewer left from a write that
    fails, and no write comes while the encoder is handing out nothing. A
    viewer nobody notices leaving holds the console in the mode its stream
    asked for: the readback the software encoder needs, every frame, for the
    rest of the process."""
    try:
        return bool(select.select([sock], [], [], 0)[0])
    except OSError:
        return True


# A page opens its stream as /h264?vid=<id> or /stream?vid=<id> and repeats
# POST /ping?vid=<id> while it watches. Consumption is not evidence a viewer is
# there: a page frozen in a browser's back/forward cache stays connected and
# keeps draining its socket, so its stream reads as watched for as long as the
# process lives. The ping stopping is what separates the two, and it is
# measured on its own clock so a viewer on a congested link is never dropped
# for being slow.
PING_INTERVAL = 3.0  # the cadence the console page pings at
VIEWER_DEADLINE = 10.0  # three intervals, so two lost pings are not a departure


class _Viewer:
    """One stream connection and when its client last said it was there. A
    connection that named no vid is not pinged for and never expires, which is
    what keeps `curl /h264` and any page older than the heartbeat streaming."""

    __slots__ = ("vid", "last")

    def __init__(self, vid):
        self.vid = vid or ""
        self.last = time.monotonic()


class Heartbeat:
    """The stream viewers expected to say they are still there.

    Connections are held one by one rather than one per vid, so a page that
    reopens its stream (an h264 viewer falling back to MJPEG) leaves nothing
    behind that outlives the connection it belonged to.
    """

    def __init__(self, deadline=VIEWER_DEADLINE):
        self.deadline = deadline
        self._lock = threading.Lock()
        self._by_vid = {}

    def join(self, vid):
        v = _Viewer(vid)
        if v.vid:
            with self._lock:
                self._by_vid.setdefault(v.vid, []).append(v)
        return v

    def leave(self, v):
        if not v.vid:
            return
        with self._lock:
            group = self._by_vid.get(v.vid)
            if group is None:
                return
            if v in group:
                group.remove(v)
            if not group:
                del self._by_vid[v.vid]

    def ping(self, vid):
        """Mark every connection opened under this vid alive. Returns how many,
        which is zero for a page that pings before its stream is up."""
        now = time.monotonic()
        with self._lock:
            group = list(self._by_vid.get(vid or "", ()))
        for v in group:
            v.last = now
        return len(group)

    def gone(self, v):
        return bool(v.vid) and (time.monotonic() - v.last) > self.deadline


_viewers = Heartbeat()


class Handler(BaseHTTPRequestHandler):
    """The blocking transport. Streams are its own; everything else is API."""

    protocol_version = "HTTP/1.1"  # /h264 streams chunked (proxy-friendly)

    def log_message(self, *args, **kwargs):
        pass

    def _send(self, resp):
        try:
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.content_type)
            self.send_header("Content-Length", str(len(resp.body)))
            self.end_headers()
            if resp.body:
                self.wfile.write(resp.body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _refused(self, method, path):
        """A request turned away before its body was read leaves the connection
        desynced, so every refusal but the method check closes it."""
        resp = API.precheck(method, path, self.headers)
        if resp is None:
            return False
        self.close_connection = resp.status != 405
        self._send(resp)
        return True

    def do_GET(self):
        u = urlparse(self.path)
        if self._refused("GET", u.path):
            return
        q = parse_query(u.query)
        if u.path == "/stream":
            return self._stream(q)
        if u.path == "/h264":
            return self._h264(q)
        if u.path == "/frame.png":
            return self._frame()
        self._send(API.dispatch("GET", u.path, q, self.headers))

    def do_POST(self):
        u = urlparse(self.path)
        if self._refused("POST", u.path):
            return
        n = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(n) if n else b""
        q = parse_query(u.query)
        if u.path == "/ping":
            _viewers.ping(q.get("vid"))
            return self._send(Response.text(200, "ok"))
        self._send(API.dispatch("POST", u.path, q, self.headers, body))

    def _stream(self, query):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        viewer = _viewers.join(query.get("vid"))
        _count_clients(1)
        _q.put(("preview",))  # nudge an idle (dirty=False) scene to draw a frame
        last_seq = -1
        try:
            while not _stopping.is_set():
                if _peer_gone(self.connection) or _viewers.gone(viewer):
                    break
                with _lock:
                    if _latest["seq"] == last_seq:
                        _lock.wait(1.0)
                    jpg = _latest["jpg"]
                    seq = _latest["seq"]
                if jpg is None or seq == last_seq:
                    continue
                last_seq = seq
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                    % len(jpg)
                )
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            _count_clients(-1)
            _viewers.leave(viewer)

    def _frame(self):
        # A scene that does not compile has no frame to hand out: say so instead
        # of answering with whatever was rendering before it.
        if _stat["error"]:
            return self._send(
                Response.text(409, f"scene error: {_line(_stat['error'])}")
            )
        gen = _shot.request()
        _shot.hold()
        try:
            _q.put(("shot",))  # forces a draw, and wakes a loop parked on the queue
            frame = _shot.wait(gen, FRAME_TIMEOUT)
            if frame is None:
                why = _fatal or f"nothing drawn in {FRAME_TIMEOUT:g}s"
                return self._send(Response.text(503, f"no frame: {_line(why)}"))
            # PNG on this thread, not the render thread's
            mode = "RGBA" if frame.shape[2] == 4 else "RGB"
            buf = io.BytesIO()
            Image.fromarray(frame, mode).save(buf, "PNG")
            png = buf.getvalue()
        finally:
            _shot.drop()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(png)

    def _h264(self, query):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        c = _fanout.add()
        viewer = _viewers.join(query.get("vid"))
        _q.put(("preview",))  # wake an idle (dirty=False) scene so the stream starts
        try:
            while not _stopping.is_set():
                if _peer_gone(self.connection) or _viewers.gone(viewer):
                    break
                try:
                    blob = c.q.get(timeout=1.0)
                except queue.Empty:
                    continue
                self.wfile.write(b"%X\r\n" % len(blob) + blob + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            _fanout.remove(c)
            _viewers.leave(viewer)
            # Symmetric with the wake on connect, and load-bearing: the encoder
            # is torn down on the next drawn frame, but a scene that reports
            # clean skips before the loop reaches that line. An encoder left
            # open holds keep_last, which costs the display direct scanout for
            # a stream nobody is watching.
            _q.put(("preview",))


def _fail(msg):
    """A failure the console cannot render through (no GL device, no engine):
    stop, and let run() raise it instead of leaving a server with no renderer."""
    global _fatal
    _fatal = msg
    _stat["error"] = msg
    print(f"ghee console: {msg}", file=sys.stderr)
    stop()


def stop():
    """Shut the server down and signal the render/encode threads. run() calls it
    on the way out; a host that runs the console in a thread calls it to end one.
    It waits for no thread of its own: the render thread calls it, and so can a
    request handler. run() is what waits, so by the time it returns the render
    loop has closed its sinks."""
    global _httpd
    _stopping.set()
    _q.put(("stop",))  # wake a render loop parked on the command queue
    with _raw_lock:
        _raw_lock.notify_all()
    with _lock:
        _lock.notify_all()
    httpd, _httpd = _httpd, None
    if httpd is not None:
        httpd.shutdown()
        httpd.server_close()


def run(
    arrays=None,
    hwinfo=None,
    scenes_dir=None,
    size=(320, 240),
    port=8090,
    lib_path=None,
    host="127.0.0.1",
    display=False,
    rgb565=False,
    preview_max=480,
    preview_fps=15,
    stream_max=0,
    bitrate=10_000_000,
    scene_args=None,
    store=None,
    sinks=(),
    components=4,
    frames=0,
    timestep=None,
    dri_device=None,
):
    """Start the console and serve until stop() (or Ctrl-C). arrays:
    {array_name -> callable(size, frame) -> floats} hwinfo:
    callable -> dict for the device-info readout.
    scene_args: build-time arguments (the CLI's tokens after `--`) offered to
    every scene the console loads; one no scene declares is an error here.
    store: a prebuilt SceneStore instead of one over scenes_dir.
    sinks: output sinks (ghee.output.Output) the frames also reach, so a console
    can drive a panel or write files; `display` is the KMS scanout instead,
    because that one is a mode the renderer is created in.
    frames: 0 serves until stop(); a count ends the console with the render.
    host defaults to loopback; the console has no auth and compiles POSTed GLSL,
    so pass "0.0.0.0" only to expose it on a network you trust."""
    global _lib_path, WEB, STORE, W, H, _provider, _hwinfo, API
    global _display, _rgb565, _preview_max, _preview_fps
    global _stream_max, _httpd, _fatal, _h264_bitrate, _h264_hw_dead
    global _sinks, _components, _frames, _timestep, _dri
    _h264_hw_dead = False  # a previous run's failure is not this renderer's
    _display = display
    _rgb565 = rgb565
    _preview_max = preview_max
    _preview_fps = preview_fps
    _stream_max = stream_max
    _h264_bitrate = bitrate
    _sinks = tuple(sinks)
    _components = components
    _frames = frames
    _timestep = timestep
    _dri = dri_device
    _provider = _make_provider(arrays or {})
    _hwinfo = hwinfo or default_hwinfo
    WEB = web_dir()
    W, H = size
    if store is None:
        read_only = scenes_dir is None
        scenes_dir = os.path.abspath(scenes_dir) if scenes_dir else examples_dir()
        store = SceneStore(scenes_dir, read_only=read_only, args=scene_args or ())
    STORE = store
    names = STORE.names
    if not names:
        raise ValueError(f"no .scene.json in {STORE.dir}")
    if scene_args:
        from ghee.procedural import check_scene_args

        check_scene_args({n: STORE.path(n) for n in names}, scene_args, "ghee serve --")
    _reset_stats()
    _running["cur"] = None  # the render thread publishes one when it loads
    _sources["cur"] = None
    _stat["active"] = names[0]
    _stat["display"] = {"w": W, "h": H, "format": _format(), "rotation": 0}
    _lib_path = lib_path or DEFAULT_LIB
    _fatal = ""
    _stopping.clear()
    API = _build_api()
    httpd = _httpd = ThreadingHTTPServer((host, port), Handler)
    render = threading.Thread(target=render_loop, daemon=True)
    render.start()
    threading.Thread(target=encoder_loop, daemon=True).start()
    threading.Thread(target=h264_sw_loop, daemon=True).start()
    print(
        f"ghee console: {len(names)} scenes {names} on {host}:{port} "
        f"(scenes={STORE.dir}, preview {W}x{H} {_format()})"
    )
    try:
        httpd.serve_forever()
    finally:
        stop()  # the server is down and the loop is signalled; viewers are gone
        if render.is_alive():
            render.join(RENDER_JOIN_TIMEOUT)
            if render.is_alive():
                print(
                    f"ghee console: render thread still going after "
                    f"{RENDER_JOIN_TIMEOUT:g}s; its sinks are as it left them",
                    file=sys.stderr,
                )
    if _fatal:
        raise RuntimeError(_fatal)


if __name__ == "__main__":
    sd = sys.argv[1] if len(sys.argv) > 1 else None
    sz = (
        tuple(int(x) for x in sys.argv[2].split("x"))
        if len(sys.argv) > 2
        else (320, 240)
    )
    pt = int(sys.argv[3]) if len(sys.argv) > 3 else 8090
    run(scenes_dir=sd, size=sz, port=pt)
