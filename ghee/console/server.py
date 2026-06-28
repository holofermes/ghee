"""ghee console — the render server.

Owns the render loop (GL is thread-affine), the active scene, and the scene list.
Serves the JSON API and the static PWA under web/. Rendering goes through the shared
ghee.Renderer (same binding the host app uses); this server only manages scenes
and the web surface.

Standalone (animated placeholder data, no host):
    python -m ghee.console.server [scenes_dir] [WxH] [port]

Embedded (a host injects real data):
    from ghee.console.server import run
    run(arrays={"levels": my_source}, hwinfo=host_hwinfo)

A scene declares named float arrays; the engine fills each one every frame. The
host maps array names to sources via `arrays` (name -> callable(size, frame) ->
floats); unmapped names fall back to an animated placeholder. ghee never
interprets what a name means.

API:
  GET  /stream            MJPEG, one part per rendered frame
  GET  /stats             {fps, render_ms, encode_ms, error, active}
  GET  /scenes            {scenes:[name...], active}
  GET  /scene?name=       pristine scene JSON from disk
  GET  /set?pass=&name=&value=   live uniform, no recompile
  GET  /next /prev        switch active scene
  POST /scene             full scene JSON -> recompile active (apply working copy)
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
from urllib.parse import urlparse, parse_qs

from PIL import Image

from ghee import Scene, DisplayProfile
from ghee.engine import Renderer
from ghee.console import web_dir, examples_dir
from ghee.console.feeds import demo_feed, default_hwinfo

# None -> Renderer auto-locates the .so bundled in the ghee package.
DEFAULT_LIB = os.environ.get("GHEE_LIB")

# Set by run(); the render thread and request handler read them as globals.
renderer = None
_lib_path = None
WEB = None
SCENES = {}
NAMES = []
profile = None
W = H = 0
_provider = None
_hwinfo = default_hwinfo

_latest = {"jpg": None, "seq": 0}
_lock = threading.Lock()
_nclients = 0  # MJPEG viewers; the render loop skips JPEG encoding when 0
_q = queue.Queue()
_stat = {"fps": 0.0, "render_ms": 0.0, "encode_ms": 0.0, "error": "", "active": None,
         "display": {}, "error_line": None, "error_pass": None}


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
        out[os.path.basename(p)[:-len(".scene.json")]] = p
    return out


def render_loop():
    # GL is thread-affine: create the Renderer here, in the thread that renders.
    global renderer
    renderer = Renderer(W, H, components=3, lib_path=_lib_path)
    st = {"scene": None}

    def set_scene(scene, name=None):
        err = renderer.setup(scene, profile)
        if err:
            _stat["error"] = err
            _stat["error_line"] = renderer.error_line
            _stat["error_pass"] = renderer.error_pass
            if st["scene"] is not None:
                renderer.setup(st["scene"], profile)   # keep the last good scene rendering
            return
        st["scene"] = scene
        _stat["error"] = ""; _stat["error_line"] = None; _stat["error_pass"] = None
        if name:
            _stat["active"] = name

    def load(name):
        try:
            set_scene(Scene.load(SCENES[name]), name)
        except Exception as e:
            _stat["error"] = f"load {name}: {e}"

    load(NAMES[0])
    last = time.monotonic(); frame = 0
    while True:
        while not _q.empty():
            item = _q.get()
            if item[0] == "switch":
                load(item[1])
            elif item[0] == "param":
                _, pi, nm, val = item
                try:
                    p = st["scene"].passes[pi]; pm = p.params[nm]
                    v = int(round(val)) if pm.type == "int" else val
                    renderer.set_param(p.name, nm, pm.type, v)
                    pm.value = v
                except Exception:
                    pass
            elif item[0] == "scene_obj":
                try:
                    set_scene(Scene.from_dict(item[1]))
                except Exception as e:
                    _stat["error"] = f"scene: {e}"

        t0 = time.monotonic()
        rgb = renderer.render(lambda name, size: _provider(name, size, frame))
        frame += 1
        t1 = time.monotonic()
        if _nclients > 0:  # only encode when a browser is actually watching
            buf = io.BytesIO()
            Image.fromarray(rgb, "RGB").save(buf, "JPEG", quality=50)  # baseline ~3ms; optimize+progressive cost ~10ms more for ~900B
            with _lock:
                _latest["jpg"] = buf.getvalue(); _latest["seq"] += 1
        t2 = time.monotonic()
        dt = t2 - last; last = t2
        if dt > 0:
            inst = 1.0 / dt
            _stat["fps"] = inst if _stat["fps"] == 0 else _stat["fps"] * 0.9 + inst * 0.1
            _stat["render_ms"] = (t1 - t0) * 1000.0
            _stat["encode_ms"] = (t2 - t1) * 1000.0
        time.sleep(0.003)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", ctype="text/plain"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/stream":
            return self._stream()
        if u.path == "/stats":
            return self._send(200, json.dumps(_stat), "application/json")
        if u.path == "/scenes":
            return self._send(200, json.dumps({"scenes": NAMES, "active": _stat["active"]}), "application/json")
        if u.path == "/hwinfo":
            return self._send(200, json.dumps(_hwinfo()), "application/json")
        if u.path == "/scene":
            name = q.get("name", [_stat["active"]])[0]
            if name not in SCENES:
                return self._send(404, "no scene")
            with open(SCENES[name], "r", encoding="utf-8") as f:
                return self._send(200, f.read(), "application/json")
        if u.path == "/set":
            try:
                _q.put(("param", int(q["pass"][0]), q["name"][0], float(q["value"][0])))
            except Exception:
                pass
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
        return self._send(404, "no")

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
        last_seq = -1
        try:
            while True:
                with _lock:
                    jpg = _latest["jpg"]; seq = _latest["seq"]
                if jpg is not None and seq != last_seq:
                    last_seq = seq
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                else:
                    time.sleep(0.002)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            _nclients -= 1


def run(arrays=None, hwinfo=None, scenes_dir=None, size=(320, 240), port=8090, lib_path=None):
    """Start the console. arrays: {array_name -> callable(size, frame) -> floats}
    maps a scene's declared arrays to real data; unmapped names get an animated
    placeholder. hwinfo: callable -> dict for the device-info readout."""
    global _lib_path, WEB, SCENES, NAMES, profile, W, H, _provider, _hwinfo
    _provider = _make_provider(arrays or {})
    _hwinfo = hwinfo or default_hwinfo
    WEB = web_dir()
    W, H = size
    scenes_dir = os.path.abspath(scenes_dir) if scenes_dir else examples_dir()
    SCENES = _scan_scenes(scenes_dir)
    NAMES = list(SCENES.keys())
    if not NAMES:
        raise SystemExit(f"no .scene.json in {scenes_dir}")
    profile = DisplayProfile(width=W, height=H)
    _stat["active"] = NAMES[0]
    _stat["display"] = {"w": W, "h": H, "orientation": profile.orientation, "bw": profile.bw, "rotation": 0}
    _lib_path = lib_path or DEFAULT_LIB
    threading.Thread(target=render_loop, daemon=True).start()
    print(f"ghee console: {len(NAMES)} scenes {NAMES} on :{port} (scenes={scenes_dir})")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    sd = sys.argv[1] if len(sys.argv) > 1 else None
    sz = tuple(int(x) for x in sys.argv[2].split("x")) if len(sys.argv) > 2 else (320, 240)
    pt = int(sys.argv[3]) if len(sys.argv) > 3 else 8090
    run(scenes_dir=sd, size=sz, port=pt)
