"""The console API with no transport under it.

A host implements ConsoleHost over whatever it is already rendering and hands
it to ConsoleAPI. The transport turns its own request objects into
(method, path, query, headers, body) and turns the Response back into whatever
it serves:

    api = ConsoleAPI(MyHost(), web=web_dir())
    refused = api.precheck(method, path, headers)   # before reading a body
    resp = api.dispatch(method, path, query, headers, body)

query is any mapping of name -> first value (parse_query() builds one from a
raw query string; aiohttp's rel_url.query is already one). headers is any
case-insensitive mapping.

Streams (/stream, /h264) and anything a host adds of its own stay with the
transport: route them before dispatch reaches its static-file fallback.

Everything the console refuses lives here, so a new host inherits the refusals
instead of reimplementing them: the Origin check on anything that mutates, the
body cap, POST-only state changes, the JSON content type, and the static root
check.

Thread safety: an API holds no mutable state and can be called from any number
of threads at once. What a host does under set_param/switch/apply_scene is the
host's contract to state: ghee's own host hands the work to its render thread
through a queue, spikerer's runs on the loop thread that owns its renderer.
"""

import json
import mimetypes
import os
from typing import NamedTuple
from urllib.parse import parse_qs, urlsplit

from ghee import Scene, load_filters

MAX_BODY = 1 << 20  # a POSTed scene is JSON text

# Everything that changes the renderer is POST: a GET is reachable from a
# hostile page's <img src>, which carries no Origin to refuse.
POST_ONLY = ("/set", "/switch", "/next", "/prev")

_VEC_ARITY = {"vec2": 2, "vec3": 3, "vec4": 4}
_BOOLS = {"true": True, "1": True, "false": False, "0": False}


class Response(NamedTuple):
    status: int
    body: bytes
    content_type: str = "text/plain"

    @classmethod
    def text(cls, status, msg=""):
        return cls(status, msg.encode("utf-8"), "text/plain")

    @classmethod
    def json(cls, obj, status=200):
        return cls(status, json.dumps(obj).encode("utf-8"), "application/json")


class ConsoleError(Exception):
    """Raised by a host or the API; the status it carries is what the client
    gets, with str(e) as the body."""

    status = 400


class BadRequest(ConsoleError):
    status = 400


class ReadOnly(ConsoleError):
    status = 403


class NotFound(ConsoleError):
    status = 404


class UnsupportedMedia(ConsoleError):
    status = 415


class ServerError(ConsoleError):
    status = 500


def origin_allowed(origin, host):
    """An Origin on a state-changing request must be this server's own. Requests
    with no Origin at all (curl, the CLI, a script) pass."""
    if not origin:
        return True
    return urlsplit(origin).netloc.lower() == (host or "").lower()


def parse_query(qs):
    """A raw query string as {name: first value}. The console never repeats a key."""
    return {k: v[0] for k, v in parse_qs(qs).items()}


def parse_param_value(ptype, raw):
    """Turn a /set raw query string into the value set_param expects. A "color"
    is "r,g,b" -> [float, float, float]; a "vecK" is K comma-separated floats ->
    [float, ...]; a "bool" is true/false (or 1/0) -> a real bool, which is what
    the scene model demands; anything else is a single float. Raises ValueError
    on malformed input (a short/long color/vec, a non-number, a non-bool)."""
    if ptype == "bool":
        try:
            return _BOOLS[raw.strip().lower()]
        except KeyError:
            raise ValueError(f"bool needs true or false, got {raw!r}") from None
    if ptype == "color":
        parts = [float(x) for x in raw.split(",")]
        if len(parts) != 3:
            raise ValueError(f"color needs three comma-separated floats, got {raw!r}")
        return parts
    if ptype in _VEC_ARITY:
        n = _VEC_ARITY[ptype]
        parts = [float(x) for x in raw.split(",")]
        if len(parts) != n:
            raise ValueError(f"{ptype} needs {n} comma-separated floats, got {raw!r}")
        return parts
    if ptype == "string":
        return raw
    return float(raw)


def parse_scope(raw):
    """A /set pass= target: the literal "scene" for the scene's shared params,
    else a pass index (ValueError if it is not an int)."""
    return "scene" if raw == "scene" else int(raw)


def static_path(root, urlpath):
    """The file a URL path names under root, or None. commonpath, not
    startswith: a sibling /webXYZ shares the /web prefix."""
    if urlpath in ("", "/"):
        urlpath = "/index.html"
    root = os.path.normpath(root)
    fp = os.path.normpath(os.path.join(root, urlpath.lstrip("/")))
    try:
        if os.path.commonpath([root, fp]) != root:
            return None
    except ValueError:
        return None
    return fp if os.path.isfile(fp) else None


class ConsoleHost:
    """What the console needs from whatever is rendering.

    Subclass it, implement the pieces your host has, mount a ConsoleAPI on your
    server. Raise NotFound/BadRequest/ReadOnly/ServerError from any method and
    the client gets that status with the message.
    """

    # /save is refused unless a host opts in; save_refused is the refusal text
    writable = False
    save_refused = "this console cannot save"

    def stats(self):
        """{fps, frame_ms, stream_fps, out, error, error_line, error_pass,
        active, display, stale}.

        fps: render-loop frames completed per real elapsed second, over a
        trailing window (ghee.stats.FrameRate). frame_ms: one loop iteration,
        the frame's own work with no paced sleep in it. stream_fps: frames
        handed to viewers per real second, counted where the handoff happens,
        every codec in one number. out: the destination names the frames reach
        right now (ghee.stats.DEST_*), a sink's own for a sink. error_pass: the
        pass index a compile error landed in, or "common" for the scene's
        common block; error_line is the line within it. stale: the files the
        active scene was built from now hold code it was not compiled from, so
        a reload would render something else (ghee.sources); a host whose
        scenes have no files behind them answers False.

        Read by a poll loop in the browser, so it must not block."""
        raise NotImplementedError

    def scene_names(self):
        raise NotImplementedError

    def active_name(self):
        raise NotImplementedError

    def scene_def(self, name=None):
        """Resolved scene JSON (shader files inlined, filters expanded) for a
        name, or for the active scene when name is None."""
        raise NotImplementedError

    def filters(self):
        """name -> filter def, for the "+ filter" picker."""
        return load_filters()

    def hwinfo(self):
        return {}

    def set_param(self, scope, name, value):
        """Live uniform, no recompile. scope is a pass index, or the string
        "scene" for a scene-scoped param."""
        raise NotImplementedError

    def switch(self, name):
        """Make name active and return the new active name."""
        raise NotImplementedError

    def step(self, direction):
        """Move to the next or previous of scene_names()."""
        names = self.scene_names()
        if not names:
            raise NotFound("no scenes")
        active = self.active_name()
        i = names.index(active) if active in names else 0
        return self.switch(names[(i + (1 if direction == "next" else -1)) % len(names)])

    def apply_scene(self, scene):
        """Recompile the running renderer onto a validated ghee.Scene."""
        raise NotImplementedError

    def save_scene(self, name, scene):
        raise ReadOnly(self.save_refused)


class ConsoleAPI:
    """Routes one transport-free request onto a ConsoleHost."""

    def __init__(self, host, web=None, post_only=POST_ONLY, max_body=MAX_BODY):
        self.host = host
        self.web = web  # static root, or None when the transport serves its own
        self.post_only = tuple(post_only)
        self.max_body = max_body

    def precheck(self, method, path, headers):
        """The refusals that have to land before a body is read: a Response to
        send instead of handling the request, or None. Pure, and dispatch runs
        it again, so a transport that skips it is still refused."""
        if method == "POST":
            if not origin_allowed(headers.get("Origin") or "", headers.get("Host")):
                return Response.text(403, "cross-origin request")
            raw = headers.get("Content-Length") or "0"
            n = int(raw) if raw.isdigit() else -1
            if n < 0:
                return Response.text(400, f"bad Content-Length {raw!r}")
            if n > self.max_body:
                return Response.text(
                    413, f"body of {n} bytes, limit is {self.max_body}"
                )
        if method == "GET" and path in self.post_only:
            return Response.text(405, f"{path} is POST only")
        return None

    def dispatch(self, method, path, query=None, headers=None, body=b""):
        headers = headers if headers is not None else {}
        refused = self.precheck(method, path, headers)
        if refused is not None:
            return refused
        try:
            if method == "GET":
                return self._get(path, query or {})
            if method == "POST":
                return self._post(path, query or {}, headers, body)
        except ConsoleError as e:
            return Response.text(e.status, str(e))
        return Response.text(405, f"{method} is not allowed")

    def _get(self, path, query):
        h = self.host
        if path == "/stats":
            return Response.json(h.stats())
        if path == "/scenes":
            return Response.json({"scenes": h.scene_names(), "active": h.active_name()})
        if path == "/hwinfo":
            return Response.json(h.hwinfo())
        if path == "/filters":
            return Response.json(h.filters())
        if path == "/scene":
            return Response.json(h.scene_def(query.get("name") or None))
        return self.static(path)

    def _post(self, path, query, headers, body):
        h = self.host
        if path == "/set":
            try:
                value = parse_param_value(query.get("type") or "float", query["value"])
                scope = parse_scope(query["pass"])
                name = query["name"]
            except (KeyError, ValueError) as e:
                return Response.text(400, f"bad /set: {e}")
            h.set_param(scope, name, value)
            return Response.text(200, "ok")
        if path in ("/next", "/prev"):
            return Response.json({"active": h.step(path[1:])})
        if path == "/switch":
            return Response.json({"active": h.switch(query.get("name"))})
        if path == "/scene":
            scene = self._scene(self._json(headers, body, "/scene"), "/scene")
            h.apply_scene(scene)
            return Response.text(200, "ok")
        if path == "/save":
            d = self._json(headers, body, "/save")
            if not h.writable:
                raise ReadOnly(h.save_refused)
            try:
                name = d["name"]
            except Exception as e:
                raise BadRequest(f"bad /save: {e}") from e
            h.save_scene(name, self._scene(d.get("scene"), "/save"))
            return Response.text(200, "ok")
        return Response.text(404, "no")

    def static(self, path):
        if self.web is None:
            return Response.text(404, "not found")
        fp = static_path(self.web, path)
        if fp is None:
            return Response.text(404, "not found")
        ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
        with open(fp, "rb") as f:
            return Response(200, f.read(), ctype)

    @staticmethod
    def _json(headers, body, what):
        ct = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ct != "application/json":
            raise UnsupportedMedia(f"expected application/json, got {ct!r}")
        try:
            return json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)
        except Exception as e:
            raise BadRequest(f"bad {what}: {e}") from e

    @staticmethod
    def _scene(d, what):
        """A posted scene is a Scene before anything downstream sees it: the
        renderer never gets a dict nobody validated."""
        try:
            return Scene.from_dict(d)
        except Exception as e:
            raise BadRequest(f"bad {what}: {e}") from e
