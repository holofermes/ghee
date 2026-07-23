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
  GET  /stats             {fps, render_ms, encode_ms, error, active}
  GET  /scenes            {scenes:[name...], active}
  GET  /filters           {name: filter def} registry for the "+ filter" picker
  GET  /scene?name=       resolved scene JSON (shader files inlined, filters expanded)
  GET  /set?pass=&name=&value=   live uniform, no recompile
  GET  /next /prev        switch active scene
  POST /scene             full scene JSON -> recompile active (apply working copy)
  POST /save              write the scene as a modified copy in <scenes>/modified/;
                          the original file is never touched (demo scenes refuse)
  GET  /<file>            static file from web/
"""

import glob
import io
import json
import mimetypes
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from ghee import Scene, apply_filters, load_filters, load_scene
from ghee.console import examples_dir, web_dir
from ghee.console.feeds import default_hwinfo, demo_feed
from ghee.engine import Renderer

# None -> Renderer auto-locates the .so bundled in the ghee package.
DEFAULT_LIB = os.environ.get("GHEE_LIB")

# Set by run(); the render thread and request handler read them as globals.
_lib_path = None
WEB: str = ""
SCENES = {}
NAMES = []
W = H = 0
_provider = None
_hwinfo = default_hwinfo
_scenes_dir = ""
_read_only = False  # bundled examples live in the package; never write there
_display = False    # also scan out to a connected KMS display (headless Pi, root)
_rgb565 = False     # display scanout in 16-bit color (halves GPU store + scanout bytes)
_preview_max = 480  # longest side of the browser preview (downscaled off the render thread)
_preview_fps = 15   # browser preview refresh cap; the display/panel stays full speed
_raw = {"rgb": None, "seq": 0}   # latest full frame handed to the encoder thread
_raw_lock = threading.Lock()


def _modified_path(name):
    return os.path.join(_scenes_dir, "modified", f"{name}.scene.json")


def _scene_path(name):
    """A saved modified copy shadows the original; the original is never written."""
    p = SCENES[name]
    if p.endswith(".scene.json") and os.path.isfile(_modified_path(name)):
        return _modified_path(name)
    return p


def _load_resolved(name):
    """Scene ready to render/edit: shader files inlined, declared filters expanded."""
    scene, controller = load_scene(_scene_path(name))
    apply_filters(scene, scene.filters)
    return scene, controller

_latest = {"jpg": None, "seq": 0}
_lock = threading.Lock()
_nclients = 0
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


def _make_provider(arrays):
    """arrays: {name -> callable(size, frame) -> floats}. Returns provide(name,
    size, frame) that uses the host's source for a name or the placeholder."""

    def provide(name, size, frame):
        src = arrays.get(name)
        return src(size, frame) if src is not None else demo_feed(name, size, frame)

    return provide


def _scan_scenes(scenes_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(scenes_dir, "*.scene.json"))):
        out[os.path.basename(p)[: -len(".scene.json")]] = p
    for p in sorted(glob.glob(os.path.join(scenes_dir, "*.py"))):
        name = os.path.basename(p)[:-3]
        out.setdefault(name, p)  # a .py beside its .scene.json: load_scene finds both
    return out


def _make_renderer():
    """Create the render-thread Renderer, probing DRI devices so the console runs on
    the Pi (card0) and on desktop Mesa (renderD128). components=4 because desktop Mesa
    rejects an RGB glReadPixels; the frame is encoded to RGB for the browser anyway."""
    if _display:
        return Renderer(W, H, components=4, display=True, rgb565=_rgb565,
                        dri_device=os.environ.get("GHEE_DRI") or "/dev/dri/card0",
                        lib_path=_lib_path)
    errs = []
    for dev in [
        os.environ.get("GHEE_DRI"),
        None,
        "/dev/dri/renderD128",
        "/dev/dri/card1",
    ]:
        try:
            return Renderer(W, H, components=4, dri_device=dev, lib_path=_lib_path)
        except Exception as e:
            errs.append(f"{dev or 'card0'}: {e}")
    raise RuntimeError("no GL device for the console:\n" + "\n".join(errs))


def encoder_loop():
    # JPEG is expensive at display resolution, so it runs OFF the render thread:
    # the render loop drops the latest full frame here, this downscales it to
    # _preview_max and encodes at _preview_fps. The display/panel never waits.
    last_seq = -1
    while True:
        with _raw_lock:
            fresh = _raw["seq"] != last_seq and _raw["rgb"] is not None
            rgb = _raw["rgb"] if fresh else None
            if fresh:
                last_seq = _raw["seq"]
        if rgb is None:
            time.sleep(0.01)
            continue
        mode = "RGBA" if rgb.shape[2] == 4 else "RGB"
        img = Image.fromarray(rgb, mode).convert("RGB")
        w, h = img.size
        scale = _preview_max / max(w, h)
        if scale < 1.0:
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=50)
        with _lock:
            _latest["jpg"] = buf.getvalue()
            _latest["seq"] += 1


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
    # GL is thread-affine: create the Renderer here, in the thread that renders.
    renderer = _make_renderer()
    st: dict = {"scene": None, "controller": None}

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
        _stat["error"] = ""
        _stat["error_line"] = None
        _stat["error_pass"] = None
        if name:
            _stat["active"] = name

    def load(name):
        try:
            scene, controller = _load_resolved(name)
            set_scene(scene, name, controller)
        except Exception as e:
            _stat["error"] = f"load {name}: {e}"

    load(NAMES[0])
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
    preview_buf = np.zeros((ph, pw, 4), dtype=np.uint8)
    provider = _provider or demo_feed
    while True:
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
                    v = int(round(val)) if pm.type == "int" else val
                    renderer.set_param(p.name, nm, pm.type, v)
                    pm.value = v
                except Exception as e:
                    print(f"set param {nm}: {e}", file=sys.stderr)
            elif item[0] == "scene_obj":
                try:
                    set_scene(Scene.from_dict(item[1]), controller=st["controller"])
                except Exception as e:
                    _stat["error"] = f"scene: {e}"

        ctrl = st["controller"]
        # A controller that knows this frame would look identical may skip it
        # entirely: the display keeps scanning out the last flipped buffer.
        # Queued commands (switch/param/scene edit/new preview client) force a
        # draw so edits always show. `frame` advances on skipped frames too.
        # (t = frame/60 predates this loop and assumes a 60Hz cadence; with the
        # skip interval derived from the real refresh it tracks wall clock on a
        # 60Hz panel, and is off for other panels exactly as it always was.)
        if not _should_render(force, ctrl, frame, frame / 60.0):
            _stat["fps"] *= 0.98
            frame += 1
            interval = _skip_interval(display_hz, _nclients, _preview_fps, last_period)
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
        want_preview = _nclients > 0 and (
            _preview_fps <= 0 or (now - last_preview) >= 1.0 / _preview_fps
        )

        t0 = time.monotonic()
        if _display:
            # scan straight out; read back a small GPU-downscaled preview only
            # when refreshing (keeps the flip cadence at full speed).
            renderer.present(
                feed,
                out=(preview_buf if want_preview else None),
                out_size=((pw, ph) if want_preview else None),
            )
            preview_rgb = preview_buf
        else:
            preview_rgb = renderer.render(feed)
        frame += 1
        t1 = time.monotonic()

        if want_preview:
            with _raw_lock:
                _raw["rgb"] = preview_rgb.copy()
                _raw["seq"] += 1
            last_preview = now

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


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):
        pass

    def _send(self, code, body: bytes | str = b"", ctype="text/plain"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/stream":
            return self._stream()
        if u.path == "/stats":
            return self._send(200, json.dumps(_stat), "application/json")
        if u.path == "/scenes":
            return self._send(
                200,
                json.dumps({"scenes": NAMES, "active": _stat["active"]}),
                "application/json",
            )
        if u.path == "/hwinfo":
            return self._send(200, json.dumps(_hwinfo()), "application/json")
        if u.path == "/filters":
            return self._send(200, json.dumps(load_filters()), "application/json")
        if u.path == "/scene":
            name = q.get("name", [_stat["active"]])[0]
            if name not in SCENES:
                return self._send(404, "no scene")
            try:
                scene, _ = _load_resolved(name)
                return self._send(200, json.dumps(scene.to_dict()), "application/json")
            except Exception as e:
                return self._send(500, f"build {name}: {e}")
        if u.path == "/set":
            try:
                item = ("param", int(q["pass"][0]), q["name"][0], float(q["value"][0]))
            except (KeyError, ValueError) as e:
                return self._send(400, f"bad /set: {e}")
            _q.put(item)
            return self._send(200, "ok")
        if u.path in ("/next", "/prev"):
            i = NAMES.index(_stat["active"]) if _stat["active"] in NAMES else 0
            i = (i + (1 if u.path == "/next" else -1)) % len(NAMES)
            _q.put(("switch", NAMES[i]))
            return self._send(200, json.dumps({"active": NAMES[i]}), "application/json")
        if u.path == "/switch":
            name = q.get("name", [None])[0]
            if name not in SCENES:
                return self._send(404, "no scene")
            _q.put(("switch", name))
            return self._send(200, json.dumps({"active": name}), "application/json")
        return self._static(u.path)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        if u.path == "/scene":
            try:
                _q.put(("scene_obj", json.loads(body.decode("utf-8"))))
            except Exception as e:
                return self._send(400, str(e))
            return self._send(200, "ok")
        if u.path == "/save":
            return self._save(body)
        return self._send(404, "no")

    def _save(self, body):
        if _read_only:
            return self._send(403, "demo scenes are read-only; serve your own directory to save")
        try:
            d = json.loads(body.decode("utf-8"))
            name = d["name"]
            scene = Scene.from_dict(d["scene"])
        except Exception as e:
            return self._send(400, f"bad /save: {e}")
        if name not in SCENES:
            return self._send(404, "no scene")
        if not SCENES[name].endswith(".scene.json"):
            return self._send(400, "procedural scene, nothing to save")
        # store unexpanded: filter passes come back at load time from the filters list
        scene.passes = [p for p in scene.passes if not p.filter]
        path = _modified_path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(scene.to_dict(), f, indent=2)
        os.replace(tmp, path)
        return self._send(200, "ok")

    def _static(self, path):
        if path in ("", "/"):
            path = "/index.html"
        fp = os.path.normpath(os.path.join(WEB, path.lstrip("/")))
        if not fp.startswith(WEB) or not os.path.isfile(fp):
            return self._send(404, "not found")
        ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
        with open(fp, "rb") as f:
            return self._send(200, f.read(), ctype)

    def _stream(self):
        global _nclients
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        _nclients += 1
        _q.put(("preview",))  # nudge an idle (dirty=False) scene to draw a frame
        last_seq = -1
        try:
            while True:
                with _lock:
                    jpg = _latest["jpg"]
                    seq = _latest["seq"]
                if jpg is not None and seq != last_seq:
                    last_seq = seq
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                        % len(jpg)
                    )
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                else:
                    time.sleep(0.002)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            _nclients -= 1


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
):
    """Start the console. arrays: {array_name -> callable(size, frame) -> floats}
    maps a scene's declared arrays to real data; unmapped names get an animated
    placeholder. hwinfo: callable -> dict for the device-info readout.
    host defaults to loopback; the console has no auth and compiles POSTed GLSL,
    so pass "0.0.0.0" only to expose it on a network you trust."""
    global _lib_path, WEB, SCENES, NAMES, W, H, _provider, _hwinfo
    global _scenes_dir, _read_only, _display, _rgb565, _preview_max, _preview_fps
    _display = display
    _rgb565 = rgb565
    _preview_max = preview_max
    _preview_fps = preview_fps
    _provider = _make_provider(arrays or {})
    _hwinfo = hwinfo or default_hwinfo
    WEB = web_dir()
    W, H = size
    _read_only = scenes_dir is None
    scenes_dir = os.path.abspath(scenes_dir) if scenes_dir else examples_dir()
    _scenes_dir = scenes_dir
    SCENES = _scan_scenes(scenes_dir)
    NAMES = list(SCENES.keys())
    if not NAMES:
        raise SystemExit(f"no .scene.json in {scenes_dir}")
    _stat["active"] = NAMES[0]
    _stat["display"] = {"name": "preview", "w": W, "h": H, "format": "rgba", "rotation": 0}
    _lib_path = lib_path or DEFAULT_LIB
    threading.Thread(target=render_loop, daemon=True).start()
    threading.Thread(target=encoder_loop, daemon=True).start()
    print(
        f"ghee console: {len(NAMES)} scenes {NAMES} on {host}:{port} "
        f"(scenes={scenes_dir}, preview {W}x{H} rgba)"
    )
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    sd = sys.argv[1] if len(sys.argv) > 1 else None
    sz = (
        tuple(int(x) for x in sys.argv[2].split("x"))
        if len(sys.argv) > 2
        else (320, 240)
    )
    pt = int(sys.argv[3]) if len(sys.argv) > 3 else 8090
    run(scenes_dir=sd, size=sz, port=pt)
