"""ghee console render server.

Owns the render loop (ghee is thread-affine), the active scene, and the scene list.
Serves the JSON API and the static PWA.

Standalone (animated placeholder data, no host):
    python -m ghee.console.server [scenes_dir] [WxH] [port]

Embedded (a host injects real data):
    from ghee.console.server import run
    run(arrays={"levels": my_source}, hwinfo=host_hwinfo)

API:
  GET  /stream            MJPEG, one part per rendered frame
  GET  /h264              framed H264 stream ([u8 type][u32 len]: 0=config JSON,
                          1=key AU, 2=delta AU, AVCC) for WebCodecs; one shared
                          encoder fans out to all viewers. Hardware zero-copy
                          where /dev/video11 exists, ffmpeg/libx264 otherwise.
  GET  /stats             {fps, render_ms, encode_ms, error, active}
  GET  /scenes            {scenes:[name...], active}
  GET  /filters           {name: filter def} registry for the "+ filter" picker
  GET  /scene?name=       resolved scene JSON (shader files inlined, filters expanded)
  POST /set?pass=&name=&value=   live uniform, no recompile (pass=scene = scene-scoped)
  POST /next /prev        switch active scene
  POST /switch?name=      switch to a named scene
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
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from ghee.console import examples_dir, web_dir
from ghee.console.core import (
    ConsoleAPI,
    ConsoleHost,
    NotFound,
    ServerError,
    parse_query,
)
from ghee.console.feeds import default_hwinfo, demo_feed
from ghee.console.store import SceneStore
from ghee.engine import Renderer, probe_dri_devices
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

# None -> Renderer auto-locates the .so bundled in the ghee package.
DEFAULT_LIB = os.environ.get("GHEE_LIB")

# Set by run(); the render thread and request handler read them as globals.
_lib_path = None
WEB: str = ""
STORE: SceneStore = None
W = H = 0
_provider = None
_hwinfo = default_hwinfo
_display = False  # also scan out to a connected KMS display (headless Pi, root)
_rgb565 = False  # display scanout in 16-bit color (halves GPU store + scanout bytes)
_preview_max = (
    480  # longest side of the browser preview (downscaled off the render thread)
)
_preview_fps = 15  # browser preview refresh cap; the display/panel stays full speed
_raw = {"rgb": None, "seq": 0}  # latest preview frame handed to the encoder thread
_raw_lock = threading.Condition()
_httpd = None
_stopping = threading.Event()
_fatal = ""  # a failure the console cannot render through; run() raises it


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


_latest = {"jpg": None, "seq": 0}
_lock = threading.Condition()
_nclients = 0
_nclients_lock = threading.Lock()
_q = queue.Queue()
_stat = {
    "fps": 0.0,
    "render_ms": 0.0,
    "encode_ms": 0.0,
    "error": "",
    "active": None,
    "display": {},
    "error_line": None,
    "error_pass": None,
}


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
    the Pi (card0) and on desktop Mesa (renderD128). components=4 because desktop Mesa
    rejects an RGB glReadPixels; the frame is encoded to RGB for the browser anyway."""
    if _display:
        return Renderer(
            W,
            H,
            components=4,
            display=True,
            rgb565=_rgb565,
            dri_device=os.environ.get("GHEE_DRI") or "/dev/dri/card0",
            lib_path=_lib_path,
        )
    errs = []
    for dev in probe_dri_devices():
        try:
            return Renderer(W, H, components=4, dri_device=dev, lib_path=_lib_path)
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
        t0 = time.monotonic()
        mode = "RGBA" if rgb.shape[2] == 4 else "RGB"
        img = Image.fromarray(rgb, mode).convert("RGB")
        w, h = img.size
        scale = _preview_max / max(w, h)
        if scale < 1.0:  # sized for the software H264 stream, or a bigger host frame
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=50)
        _stat["encode_ms"] = (time.monotonic() - t0) * 1000.0
        with _lock:
            _latest["jpg"] = buf.getvalue()
            _latest["seq"] += 1
            _lock.notify_all()


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
    keyframe (hardware forces one on join; software joins at the next GOP)."""

    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()
        self.cfg = None
        self.size = (0, 0)  # encoded dimensions, set by whoever starts the encoder
        self.color = None  # decoder colorSpace hint (HW_COLOR/SW_COLOR), same owner
        self.want_key = False

    @property
    def nclients(self):
        return len(self.clients)

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
            # a waiting (new or resyncing) client rejoins at the next keyframe;
            # keep requesting one until everyone is streaming
            if any(not c.started for c in self.clients):
                self.want_key = True


_fanout = H264Fanout()
_h264_hw_dead = False  # hardware create failed once: h264_sw_loop owns the stream
_h264_bitrate = 4_000_000
_H264_FPS = 30  # software-encode cadence (hardware follows the display/loop)
_stream_max = 0  # optional cap on the h264 stream's longest side; 0 = full size


def _stream_size(w, h, longest):
    """Stream dimensions: fit the longest side, keep aspect, force even
    (yuv420 needs even dims)."""
    s = min(1.0, longest / max(w, h)) if longest > 0 else 1.0
    return max(2, round(w * s / 2) * 2), max(2, round(h * s / 2) * 2)


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
                    bitrate=_h264_bitrate,
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


def _should_render(force, ctrl, frame, t):
    """A frame draws when something demands it: a queued command, the very
    first frame (the sink must show something once), a scene with no
    controller (free-running), or a controller reporting dirty."""
    if force or frame == 0 or ctrl is None:
        return True
    return bool(ctrl.dirty(frame, t))


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


def render_loop():
    global _h264_hw_dead
    # GL is thread-affine: create the Renderer here, in the thread that renders.
    try:
        renderer = _make_renderer()
    except Exception as e:
        return _fail(f"renderer: {e}")
    st: dict = {"scene": None, "controller": None}
    h264_enc = None  # hardware encoder; render-thread owned (it issues GL calls)

    def set_scene(scene, name=None, controller=None):
        err = renderer.setup(scene)
        if err:
            _stat["error"] = err
            _stat["error_line"] = renderer.error_line
            _stat["error_pass"] = renderer.error_pass
            if st["scene"] is not None:
                renderer.setup(st["scene"])  # keep the last good scene rendering
            return
        st["scene"] = scene
        st["controller"] = controller
        if controller is not None:
            controller.setup(renderer)
            # string params are host-side: hand every current value to the
            # controller so saved/overlaid strings take effect on load
            groups = [scene.params] + [p.params for p in scene.passes]
            for params in groups:
                for nm, pm in params.items():
                    if pm.type == "string":
                        controller.on_param(renderer, nm, pm.value)
        _stat["error"] = ""
        _stat["error_line"] = None
        _stat["error_pass"] = None
        if name:
            _stat["active"] = name

    def load(name):
        try:
            scene, controller = STORE.resolve(name)
            set_scene(scene, name, controller)
        except Exception as e:
            _stat["error"] = f"load {name}: {e}"

    load(STORE.names[0])
    last = time.monotonic()
    last_preview = 0.0
    frame = 0
    # poll cadence for clean scenes: the display's own refresh when scanning
    # out, else consumer-derived in _skip_interval
    display_hz = renderer.display_vrefresh() if _display else 0
    last_period = None  # measured frame period; set before any skip can happen
    # preview readback size: display fit into _preview_max, aspect preserved
    _pscale = min(1.0, _preview_max / max(W, H))
    pw, ph = max(1, round(W * _pscale)), max(1, round(H * _pscale))
    sw_size = _stream_size(W, H, _stream_max)  # software H264 reads these frames back
    provider = _provider or demo_feed
    reported = ""
    while not _stopping.is_set():
        try:
            force = False  # any queued command below forces a draw this iteration
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
                            ctrl = st["controller"]
                            if ctrl is not None:
                                ctrl.on_param(renderer, nm, val)
                            continue
                        v = _coerce(pm.type, val)
                        renderer.set_param(p.name, nm, pm.type, v)
                        pm.value = v
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
                            ctrl = st["controller"]
                            if ctrl is not None:
                                ctrl.on_param(renderer, nm, val)
                            continue
                        v = _coerce(pm.type, val)
                        renderer.set_scene_param(nm, pm.type, v)
                        pm.value = v
                    except Exception as e:
                        print(f"set scene param {nm}: {e}", file=sys.stderr)
                elif item[0] == "scene_obj":
                    set_scene(item[1], controller=st["controller"])

            ctrl = st["controller"]
            # A controller that knows this frame would look identical may skip it
            # entirely: the display keeps scanning out the last flipped buffer.
            # Queued commands (switch/param/scene edit/new preview client) force a
            # draw so edits always show. `frame` advances on skipped frames too.
            # (t = frame/60 assumes a 60Hz cadence; non-60Hz panels get a
            # proportionally scaled t.)
            if not _should_render(force, ctrl, frame, frame / 60.0):
                _stat["fps"] *= 0.98
                frame += 1
                interval = _skip_interval(
                    display_hz, _nclients + _fanout.nclients, _preview_fps, last_period
                )
                if interval is None:
                    _q.put(_q.get())  # nobody consumes frames: sleep until a command
                    continue
                time.sleep(interval)
                continue
            if ctrl is not None:
                ctrl.update(renderer, frame, frame / 60.0)

            def feed(name, size, frame=frame, ctrl=ctrl):
                if ctrl is not None:
                    data = ctrl.provide(name, size, frame)
                    if data is not None:
                        return data
                return provider(name, size, frame)

            # Refresh the browser preview at most _preview_fps, and only when watched.
            # The display/panel renders every frame regardless; the preview is a
            # throttled, downscaled mirror handled by the encoder thread.
            now = time.monotonic()
            # software H264 eats the same readback frames as the MJPEG preview, so
            # its viewers count as readback consumers at their own cadence
            # (_H264_FPS, not the MJPEG preview cap); hardware H264 does not (it
            # samples the GPU texture directly). Fastest active consumer wins.
            sw_h264 = _h264_hw_dead and _fanout.nclients > 0
            rates = []
            if _nclients > 0:
                rates.append(_preview_fps)  # <= 0 means every frame
            if sw_h264:
                rates.append(_H264_FPS)
            want_preview = bool(rates) and (
                any(r <= 0 for r in rates) or (now - last_preview) >= 1.0 / max(rates)
            )
            # one readback feeds both consumers, so it is sized for whoever is
            # attached: the MJPEG preview can downscale a bigger frame, the
            # software encoder cannot recover a smaller one
            rw, rh = (max(pw, sw_size[0]), max(ph, sw_size[1])) if sw_h264 else (pw, ph)
            # each frame gets its own buffer: the encoder thread reads it while
            # the next readback is already running, and the size can change
            # under it when a viewer comes or goes
            readback = np.empty((rh, rw, 4), dtype=np.uint8) if want_preview else None

            t0 = time.monotonic()
            if _display:
                # scan straight out; read back a small GPU-downscaled preview only
                # when refreshing (keeps the flip cadence at full speed).
                renderer.present(
                    feed, out=readback, out_size=((rw, rh) if want_preview else None)
                )
            elif want_preview:
                renderer.render(feed, out=readback, out_size=(rw, rh))
            else:
                renderer.draw(feed)  # nothing reads a frame this iteration
            frame += 1
            t1 = time.monotonic()

            if want_preview:
                with _raw_lock:
                    _raw["rgb"] = readback
                    _raw["seq"] += 1
                    _raw_lock.notify_all()
                last_preview = now

            # hardware H264 mirror: encode the frame just produced, zero readback.
            # keep_last in the engine holds the final pass in its FBO while the
            # encoder exists, so encode_last always has a texture to sample.
            want_h264 = _fanout.nclients > 0
            if want_h264 and h264_enc is None and not _h264_hw_dead:
                try:
                    sw_, sh_ = _stream_size(W, H, _stream_max)
                    h264_enc = HardwareH264(
                        renderer, sw_, sh_, fps=display_hz or 30, bitrate=_h264_bitrate
                    )
                    _fanout.size = (sw_, sh_)
                    _fanout.color = HW_COLOR
                except RuntimeError as e:
                    _h264_hw_dead = True  # no /dev/video11: h264_sw_loop takes over
                    print(f"h264: {e}; using software encoder", file=sys.stderr)
            if h264_enc is not None and not want_h264:
                h264_enc.close()  # restore direct-scanout speed while unwatched
                h264_enc = None
            elif h264_enc is not None:
                if _fanout.want_key:
                    _fanout.want_key = False
                    h264_enc.force_keyframe()
                au, _ = h264_enc.encode_last()
                if au:
                    _fanout.push(au)

            dt = t1 - last
            last = t1
            if dt > 0:
                last_period = dt
                inst = 1.0 / dt
                _stat["fps"] = (
                    inst if _stat["fps"] == 0 else _stat["fps"] * 0.9 + inst * 0.1
                )
                _stat["render_ms"] = (t1 - t0) * 1000.0
            time.sleep(0.003)
        except Exception as e:
            # one bad frame (or scene, or controller) must not take the thread with
            # it: report it and keep rendering whatever last worked
            msg = f"render: {e}"
            _stat["error"] = msg
            if msg != reported:
                print(f"ghee console: {msg}", file=sys.stderr)
                reported = msg
            time.sleep(0.25)


class _RenderLoopHost(ConsoleHost):
    """This module's render loop as a console host. Handler threads only read
    _stat and put items on _q; the render thread owns the renderer and drains
    the queue, so nothing here touches GL."""

    save_refused = "demo scenes are read-only; serve your own directory to save"

    @property
    def writable(self):
        return not STORE.read_only

    def stats(self):
        return dict(_stat)  # snapshot: the render thread writes into it

    def scene_names(self):
        return STORE.names

    def active_name(self):
        return _stat["active"]

    def scene_def(self, name=None):
        name = name or _stat["active"]
        if name not in STORE:
            raise NotFound("no scene")
        try:
            scene, _ = STORE.resolve(name)
        except Exception as e:
            raise ServerError(f"build {name}: {e}") from e
        return scene.to_dict()

    def hwinfo(self):
        return _hwinfo()

    def set_param(self, scope, name, value):
        _q.put(_set_item(scope, name, value))

    def switch(self, name):
        if name not in STORE:
            raise NotFound("no scene")
        _q.put(("switch", name))
        return name

    def apply_scene(self, scene):
        _q.put(("scene_obj", scene))

    def save_scene(self, name, scene):
        STORE.save(name, scene)


def _build_api():
    """Rebuilt by run() (and by the tests) because the web root is module state."""
    return ConsoleAPI(_RenderLoopHost(), web=WEB)


API = _build_api()


class Handler(BaseHTTPRequestHandler):
    """The blocking transport. Streams are its own; everything else is API."""

    protocol_version = "HTTP/1.1"  # /h264 streams chunked (proxy-friendly)

    def log_message(self, *args, **kwargs):
        pass

    def _send(self, resp):
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.content_type)
        self.send_header("Content-Length", str(len(resp.body)))
        self.end_headers()
        if resp.body:
            self.wfile.write(resp.body)

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
        if u.path == "/stream":
            return self._stream()
        if u.path == "/h264":
            return self._h264()
        self._send(API.dispatch("GET", u.path, parse_query(u.query), self.headers))

    def do_POST(self):
        u = urlparse(self.path)
        if self._refused("POST", u.path):
            return
        n = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(n) if n else b""
        self._send(
            API.dispatch("POST", u.path, parse_query(u.query), self.headers, body)
        )

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        _count_clients(1)
        _q.put(("preview",))  # nudge an idle (dirty=False) scene to draw a frame
        last_seq = -1
        try:
            while not _stopping.is_set():
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

    def _h264(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        c = _fanout.add()
        _q.put(("preview",))  # wake an idle (dirty=False) scene so the stream starts
        try:
            while not _stopping.is_set():
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


def _fail(msg):
    """A failure the console cannot render through (no GL device, no engine):
    stop, and let run() raise it instead of leaving a server with no renderer."""
    global _fatal
    _fatal = msg
    _stat["error"] = msg
    print(f"ghee console: {msg}", file=sys.stderr)
    stop()


def stop():
    """Shut the server and the render/encode threads down. run() calls it on the
    way out; a host that runs the console in a thread calls it to end one."""
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
):
    """Start the console and serve until stop() (or Ctrl-C). arrays:
    {array_name -> callable(size, frame) -> floats} hwinfo:
    callable -> dict for the device-info readout.
    host defaults to loopback; the console has no auth and compiles POSTed GLSL,
    so pass "0.0.0.0" only to expose it on a network you trust."""
    global _lib_path, WEB, STORE, W, H, _provider, _hwinfo, API
    global _display, _rgb565, _preview_max, _preview_fps
    global _stream_max, _httpd, _fatal
    _display = display
    _rgb565 = rgb565
    _preview_max = preview_max
    _preview_fps = preview_fps
    _stream_max = stream_max
    _provider = _make_provider(arrays or {})
    _hwinfo = hwinfo or default_hwinfo
    WEB = web_dir()
    W, H = size
    read_only = scenes_dir is None
    scenes_dir = os.path.abspath(scenes_dir) if scenes_dir else examples_dir()
    STORE = SceneStore(scenes_dir, read_only=read_only)
    names = STORE.names
    if not names:
        raise ValueError(f"no .scene.json in {scenes_dir}")
    _stat["active"] = names[0]
    _stat["display"] = {
        "name": "preview",
        "w": W,
        "h": H,
        "format": "rgba",
        "rotation": 0,
    }
    _lib_path = lib_path or DEFAULT_LIB
    _fatal = ""
    _stopping.clear()
    API = _build_api()
    httpd = _httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=render_loop, daemon=True).start()
    threading.Thread(target=encoder_loop, daemon=True).start()
    threading.Thread(target=h264_sw_loop, daemon=True).start()
    print(
        f"ghee console: {len(names)} scenes {names} on {host}:{port} "
        f"(scenes={scenes_dir}, preview {W}x{H} rgba)"
    )
    try:
        httpd.serve_forever()
    finally:
        stop()
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
