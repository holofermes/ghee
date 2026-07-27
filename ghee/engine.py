"""ghee.Renderer drives the C engine from Python.

Wraps the ghee_* ABI.

    with Renderer(320, 240, lib_path="libghee.so") as r:
        err = r.setup(scene)                         # "" or a compile error
        rgb = r.render(lambda name, size: data)      # (h, w, components) uint8
        r.set_param("main", "decay", "float", 0.9)   # live uniform, no recompile

`components` (3=RGB, 4=RGBA) and `lib_path` are host choices. Packing a frame to a
1bpp/RGB565 output is the optional pack() helper.
"""

import ctypes
import glob
import importlib.resources
import os
import re

import numpy as np

from ghee.cparam import CParameter, create_c_parameter, set_c_name


def probe_dri_devices():
    """DRI nodes to try for offscreen rendering, in order: $GHEE_DRI, the render
    nodes, then the primary cards. First one that creates a renderer wins. One
    list for every caller, so the console, the tests and the Renderer itself all
    land on the same GPU.

    Render nodes come first because a primary node carries DRM master. A render node
    has no master to contend for."""
    devices = [os.environ.get("GHEE_DRI"), "/dev/dri/renderD128"]
    devices += sorted(glob.glob("/dev/dri/renderD*"))
    devices += ["/dev/dri/card0", *sorted(glob.glob("/dev/dri/card*"))]
    return list(dict.fromkeys(d for d in devices if d))


def probe_dri_cards():
    """Primary DRI nodes, $GHEE_DRI first. Display work needs one of these: a
    render node has no connectors and cannot take DRM master."""
    devices = [os.environ.get("GHEE_DRI"), *sorted(glob.glob("/dev/dri/card*"))]
    devices = list(dict.fromkeys(d for d in devices if d))
    return devices or ["/dev/dri/card0"]


def bundled_lib():
    """Path to the .so shipped inside the package (built by setup.py into lib/),
    or None if it was not built (e.g. installed on a host without the toolchain).
    Matched by glob: packaging tools may tag the name (debian renames plain .so
    files to libghee.cpython-<ver>-<arch>.so)."""
    try:
        d = importlib.resources.files("ghee") / "lib"
        hits = sorted(
            str(p)
            for p in (d.iterdir() if d.is_dir() else [])
            if p.name.startswith("libghee") and p.name.endswith(".so")
        )
        return hits[0] if hits else None
    except Exception:
        return None


def probe_display_device(dri_device=None, lib_path=None):
    """The card node driving a connected display, as (device, width, height), or
    None when nothing is connected."""
    lib_path = lib_path or os.environ.get("GHEE_LIB") or bundled_lib()
    if not lib_path:
        raise ValueError(
            "no engine .so: build ghee on this host, or pass lib_path / set $GHEE_LIB"
        )
    lib = ctypes.CDLL(lib_path)
    lib.ghee_display_probe.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.ghee_display_probe.restype = ctypes.c_int
    for dev in [dri_device] if dri_device else probe_dri_cards():
        w, h = ctypes.c_int(), ctypes.c_int()
        if not lib.ghee_display_probe(dev.encode(), ctypes.byref(w), ctypes.byref(h)):
            return (dev, w.value, h.value)
    return None


def probe_display(dri_device=None, lib_path=None):
    """Size of the mode the connected display is driving, as (width, height), or
    None when there is no card or nothing connected."""
    found = probe_display_device(dri_device=dri_device, lib_path=lib_path)
    return None if found is None else found[1:]


class _Unchanged:
    """Sentinel a provider returns for an array whose texture the engine
    already holds: the upload is skipped entirely. The scene declares it
    explicitly; the engine never caches or compares data to guess. After
    setup() rebuilds a scene the textures are empty, so a controller must
    push real data again first (its setup() hook marks that moment)."""

    __slots__ = ()

    def __repr__(self):
        return "ghee.UNCHANGED"


UNCHANGED = _Unchanged()

_U8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
_F32_1 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")


def packed_size(width, height, bw):
    """Bytes a frame packs down to: 1bpp (bw) vs RGB565."""
    return width * ((height + 7) // 8) if bw else width * height * 2


def _buf(arr):
    """Pointer to a frame buffer the engine writes pixels into."""
    if arr.dtype != np.uint8 or not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("frame buffer must be a C-contiguous uint8 array")
    return arr.ctypes.data_as(ctypes.c_void_p)


def _pack_with(fn, rgb, width, height, components, bw, out):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if out is None:
        out = np.zeros(packed_size(width, height, bw), dtype=np.uint8)
    else:
        out.fill(0)
    fn(width, height, components, 1 if bw else 0, rgb, out)
    return out


_PACK = None


def _packer():
    global _PACK
    if _PACK is None:
        path = os.environ.get("GHEE_LIB") or bundled_lib()
        if not path:
            raise ValueError("no engine .so for packing: build ghee or set $GHEE_LIB")
        lib = ctypes.CDLL(path)
        lib.ghee_pack_frame.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            _U8,
            _U8,
        ]
        _PACK = lib.ghee_pack_frame
    return _PACK


def pack_frame(rgb, width, height, components=3, bw=False, out=None):
    """Pack an RGB(A) frame to 1bpp (bw) or RGB565, with no Renderer. For output
    sinks that hold a frame but not an engine. Allocates `out` if not given."""
    return _pack_with(_packer(), rgb, width, height, components, bw, out)


def _user_line(err, glsl, code):
    """Map a GL compile-error line (full shader) to the user's code line by
    subtracting the generated preamble's length."""
    m = re.search(r":(\d+)\(", err)
    if not m:
        return None
    pos = glsl.find(code)
    offset = glsl[:pos].count("\n") if pos >= 0 else 0
    return max(1, int(m.group(1)) - offset)


class Renderer:
    """A multipass renderer bound to one engine.

    ghee is thread-affine: construct and use a Renderer on a single thread (the
    context is made current at create time). A worker that renders in its own
    thread must create the Renderer in that thread.
    """

    def __init__(
        self,
        width,
        height,
        components=3,
        lib_path=None,
        dri_device=None,
        display=False,
        rgb565=False,
    ):
        self.width = width
        self.height = height
        self.components = components
        self.scene = None
        self.error = ""  # last setup() compile error text ("" = ok)
        self.error_line = (
            None  # last setup() compile error: user-code line / pass index
        )
        self.error_pass = None
        self._mp = {}  # pass name -> engine pass index
        self._feeds = {}  # array name -> size
        self._rgb = np.zeros((height, width, components), dtype=np.uint8)

        lib_path = lib_path or os.environ.get("GHEE_LIB") or bundled_lib()
        if not lib_path:
            raise ValueError(
                "no engine .so: build ghee on this host, or pass "
                "lib_path / set $GHEE_LIB"
            )
        self.lib: ctypes.CDLL = ctypes.CDLL(lib_path)
        self._declare()
        self.display = display
        err = ctypes.c_char_p()
        if display:
            # scan out to a KMS connector: needs the card (KMS) node, not a
            # render node, and DRM master.
            dev = dri_device or os.environ.get("GHEE_DRI")
            if not dev:
                found = probe_display_device(lib_path=lib_path)
                dev = found[0] if found else "/dev/dri/card0"
            self.handle = self.lib.ghee_display_create(
                width, height, dev.encode(), 1 if rgb565 else 0, ctypes.byref(err)
            )
            if not self.handle:
                raise RuntimeError(
                    f"ghee_display_create failed: "
                    f"{err.value.decode() if err.value else 'unknown'}"
                )
            return
        devices = [dri_device] if dri_device else probe_dri_devices()
        errs = []
        self.handle = None
        for dev in devices:
            err = ctypes.c_char_p()
            self.handle = self.lib.ghee_create(
                width, height, components, dev.encode(), ctypes.byref(err)
            )
            if self.handle:
                break
            errs.append(f"{dev}: {err.value.decode() if err.value else 'unknown'}")
        if not self.handle:
            raise RuntimeError("ghee_create failed: " + "; ".join(errs))

    def _declare(self):
        L = self.lib
        L.ghee_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        L.ghee_create.restype = ctypes.c_void_p
        L.ghee_display_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        L.ghee_display_create.restype = ctypes.c_void_p
        L.ghee_mp_present.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        L.ghee_destroy.argtypes = [ctypes.c_void_p]
        L.ghee_last_error.argtypes = [ctypes.c_void_p]
        L.ghee_last_error.restype = ctypes.c_char_p
        L.ghee_mp_reset.argtypes = [ctypes.c_void_p]
        L.ghee_mp_add_pass.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.ghee_mp_add_pass.restype = ctypes.c_int
        L.ghee_mp_set_pass_size.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        L.ghee_mp_set_pass_size.restype = ctypes.c_bool
        L.ghee_mp_commit.argtypes = [ctypes.c_void_p]
        L.ghee_mp_commit.restype = ctypes.c_int
        L.ghee_mp_abort.argtypes = [ctypes.c_void_p]
        L.ghee_mp_set_channel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        L.ghee_mp_set_channel.restype = ctypes.c_bool
        L.ghee_mp_set_uniform.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(CParameter),
        ]
        L.ghee_mp_set_uniform.restype = ctypes.c_bool
        L.ghee_mp_declare_array.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_char_p,
        ]
        L.ghee_mp_declare_array.restype = ctypes.c_int
        L.ghee_mp_set_array.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            _F32_1,
            ctypes.c_int,
            ctypes.c_int,
        ]
        L.ghee_max_texture_units.argtypes = [ctypes.c_void_p]
        L.ghee_max_texture_units.restype = ctypes.c_int
        L.ghee_mp_render.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        L.ghee_pack_frame.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            _U8,
            _U8,
        ]
        L.ghee_display_vrefresh.argtypes = [ctypes.c_void_p]
        L.ghee_display_vrefresh.restype = ctypes.c_int
        L.ghee_encoder_create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        L.ghee_encoder_create.restype = ctypes.c_void_p
        _enc_io = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        L.ghee_mp_render_encoded.argtypes = _enc_io
        L.ghee_mp_render_encoded.restype = ctypes.c_int
        L.ghee_encoder_encode_last.argtypes = _enc_io
        L.ghee_encoder_encode_last.restype = ctypes.c_int
        L.ghee_encoder_flush.argtypes = _enc_io
        L.ghee_encoder_flush.restype = ctypes.c_int
        L.ghee_encoder_force_keyframe.argtypes = [ctypes.c_void_p]
        L.ghee_encoder_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.ghee_set_profiling.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        L.ghee_set_time.argtypes = [ctypes.c_void_p, ctypes.c_float]
        L.ghee_clear_time.argtypes = [ctypes.c_void_p]
        L.ghee_get_timings.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]

    def setup(self, scene):
        """Build the scene's passes. Returns "" on success or an error string (a compile
        error, or a scene the engine refuses). On a compile error error_line/error_pass
        point at the offending user-code line."""
        self.error = ""
        self.error_line = None
        self.error_pass = None
        if not scene.passes:
            self.error = "scene has no passes"
            return self.error
        # An array name is one logical input, so it must have a single size. Reject a
        # scene that declares the same name at two sizes before touching the engine.
        seen = {}
        for p in scene.passes:
            for a in p.arrays:
                if seen.setdefault(a.name, a.size) != a.size:
                    self.error = (
                        f"array '{a.name}' declared at conflicting sizes "
                        f"({seen[a.name]} and {a.size})"
                    )
                    return self.error
        # The output pass renders at the renderer size (its buffer feeds
        # readback/display/encoder). Checked here against the final resolved pass
        # list, so a scene whose own last pass is sized stays valid once filters
        # append a full-size pass after it.
        if scene.passes[-1].size is not None:
            last = scene.passes[-1]
            self.error = (
                f"output pass '{last.name}' must not set a size; it renders at "
                f"the renderer size ({self.width}x{self.height})"
            )
            return self.error
        # A channel binds to a pass by name. A name that is no pass is an error
        # buffer=None is the deliberate unbound channel and reads black.
        names = {p.name for p in scene.passes}
        for p in scene.passes:
            for c, ch in enumerate(p.channels):
                if ch.buffer is not None and ch.buffer not in names:
                    self.error = (
                        f"pass '{p.name}': channel {c} binds to buffer "
                        f"'{ch.buffer}', which is not a pass in this scene"
                    )
                    return self.error
        units = self.max_texture_units()
        for p in scene.passes:
            want = len(p.channels) + len({a.name for a in p.arrays})
            if want > units:
                self.error = (
                    f"pass '{p.name}' needs {want} texture units; this GPU has {units}"
                )
                return self.error
        mp = {}
        for i, p in enumerate(scene.passes):
            glsl = p.glsl(scene.params)
            idx = self.lib.ghee_mp_add_pass(self.handle, glsl.encode("utf-8"))
            if idx >= 0 and p.size is not None:
                if not self.lib.ghee_mp_set_pass_size(
                    self.handle, idx, p.size[0], p.size[1]
                ):
                    idx = -1
            if idx < 0:
                err = self.last_error() or "compile error"
                self.lib.ghee_mp_abort(self.handle)
                self.error = err
                self.error_line = _user_line(err, glsl, p.code)
                self.error_pass = i
                return err
            mp[p.name] = idx
        if self.lib.ghee_mp_commit(self.handle) < 0:
            self.error = self.last_error() or "commit failed"
            return self.error
        self._mp = mp
        for p in scene.passes:
            for c, ch in enumerate(p.channels):
                src = mp[ch.buffer] if ch.buffer is not None else -1
                if not self.lib.ghee_mp_set_channel(self.handle, mp[p.name], c, src):
                    self.error = self.last_error() or f"pass '{p.name}': bad channel"
                    return self.error
            for nm, pm in p.params.items():
                self.set_param(p.name, nm, pm.type, pm.value)
        # scene params reach every pass at once (they sit in every preamble)
        for nm, pm in scene.params.items():
            self.set_scene_param(nm, pm.type, pm.value)
        self.scene = scene

        feeds = {}
        for p in scene.passes:
            for a in p.arrays:
                if (
                    self.lib.ghee_mp_declare_array(
                        self.handle, mp[p.name], a.name.encode("utf-8")
                    )
                    < 0
                ):
                    self.error = (
                        self.last_error()
                        or f"pass '{p.name}': array '{a.name}' refused"
                    )
                    return self.error
                feeds[a.name] = a.size
        self._feeds = feeds
        return ""

    def render(self, provider=None, out=None, out_size=None):
        """Render all passes; final pass -> the returned RGB(A) frame (h,w,components).
        provider(name, size) -> floats fills each declared array each frame or pass None
        to leave arrays untouched. `out` lets the host render into its own buffer.
        out_size=(w, h) reads back a GPU-downscaled frame at that size instead of the full
        one (as present() does), allocating the buffer when `out` is None. draw() renders
        with no readback at all."""
        self._feed(provider)
        if out_size is not None:
            pw, ph = out_size
            if out is None:
                out = np.zeros((ph, pw, self.components), dtype=np.uint8)
        else:
            pw, ph = 0, 0
            if out is None:
                out = self._rgb
        self.lib.ghee_mp_render(self.handle, _buf(out), pw, ph)
        return out

    def draw(self, provider=None):
        """Render all passes with no readback."""
        self._feed(provider)
        self.lib.ghee_mp_render(self.handle, None, 0, 0)

    def _feed(self, provider):
        """Upload each declared array for this frame (called once per unique name).
        A render uploads whatever the provider returns; the one exception is the
        explicit UNCHANGED sentinel, a scene's declaration that the texture
        already holds this array (a partially dynamic scene skips its static
        mask that way). The engine never caches or diffs data on its own."""
        if provider is None:
            return
        for name, size in self._feeds.items():
            raw = provider(name, size)
            if raw is UNCHANGED:
                continue
            data = np.ascontiguousarray(raw, dtype=np.float32).reshape(-1)
            if isinstance(size, int):
                w, h = min(len(data), size), 1
            else:
                w, h = size
                if len(data) < w * h:  # short feed: zero-pad to the full mask
                    buf = np.zeros(w * h, dtype=np.float32)
                    buf[: len(data)] = data
                    data = buf
            self.lib.ghee_mp_set_array(self.handle, name.encode("utf-8"), data, w, h)

    def present(self, provider=None, out=None, out_size=None):
        """Display mode only: feed arrays, render the passes, and scan the final
        buffer straight out to the connected display. Pass out=<h,w,4 buffer> and
        out_size=(w, h) to also read back a GPU-downscaled preview (a few hundred
        KB, not the full frame); None skips the readback."""
        self._feed(provider)
        if out is not None and out_size is not None:
            buf = _buf(out)
            pw, ph = out_size
        else:
            buf, pw, ph = None, 0, 0
        self.lib.ghee_mp_present(self.handle, buf, pw, ph)

    def set_param(self, pass_name, name, ptype, value):
        """Set one uniform on a pass live (no recompile). False if the pass/uniform
        is absent. A "color" param is [r, g, b]: it sets the three decomposed
        float uniforms name_r/_g/_b (the engine has no color type). A "vecK" param
        is K floats: it sets the decomposed float uniforms name_x/_y/_z/_w.
        A "string" param is host-side (the scene's controller consumes it):
        nothing to set here."""
        if ptype == "string":
            return False
        idx = self._mp.get(pass_name)
        if idx is None:
            return False
        if ptype == "color":
            r, g, b = value
            oks = []
            for suffix, comp in (("_r", r), ("_g", g), ("_b", b)):
                cp = create_c_parameter({"type": "float", "value": comp})
                set_c_name(cp, name + suffix)
                oks.append(
                    self.lib.ghee_mp_set_uniform(self.handle, idx, ctypes.byref(cp))
                )
            return all(oks)
        if ptype in ("vec2", "vec3", "vec4"):
            oks = []
            for suffix, comp in zip(("_x", "_y", "_z", "_w"), value, strict=False):
                cp = create_c_parameter({"type": "float", "value": comp})
                set_c_name(cp, name + suffix)
                oks.append(
                    self.lib.ghee_mp_set_uniform(self.handle, idx, ctypes.byref(cp))
                )
            return all(oks)
        cp = create_c_parameter({"type": ptype, "value": value})
        set_c_name(cp, name)
        return self.lib.ghee_mp_set_uniform(self.handle, idx, ctypes.byref(cp))

    def set_scene_param(self, name, ptype, value):
        """Set one scene-scoped uniform on every pass live (no recompile). A scene
        param sits in every pass's preamble, so the set fans out to all of them;
        a pass that never reads it has an inactive uniform and is skipped. True if
        at least one pass held the uniform and took the value. The list is built
        before any(): every pass must be set, not just up to the first that holds
        the uniform."""
        oks = [self.set_param(pn, name, ptype, value) for pn in self._mp]
        return any(oks)

    def pack(self, rgb, bw=False, out=None):
        """Optional: pack an RGB(A) frame to 1bpp (bw) or RGB565.
        Allocates `out` if not given; zeros it first."""
        return _pack_with(
            self.lib.ghee_pack_frame,
            rgb,
            self.width,
            self.height,
            self.components,
            bw,
            out,
        )

    def display_vrefresh(self):
        """The connected display mode's refresh rate in Hz, 0 when not in
        display mode."""
        return int(self.lib.ghee_display_vrefresh(self.handle))

    def set_profiling(self, on):
        """Enable per-frame draw/readback timing (adds a glFinish; off = zero cost)."""
        self.lib.ghee_set_profiling(self.handle, bool(on))

    def set_time(self, t):
        """Pin the `time` uniform to a fixed value, for reproducible renders. Default
        is the monotonic clock since the engine was created; t=None hands it back to
        that clock."""
        if t is None:
            self.lib.ghee_clear_time(self.handle)
        else:
            self.lib.ghee_set_time(self.handle, float(t))

    def max_texture_units(self):
        """Fragment texture units this GPU exposes: a pass's channels plus its array
        inputs have to fit (8 on VC4)."""
        return int(self.lib.ghee_max_texture_units(self.handle))

    def timings(self):
        """Last frame's split as (draw_ms, read_ms): GPU render of all passes vs the
        glReadPixels readback. Both 0 until a frame renders with profiling on."""
        d = ctypes.c_double(0.0)
        r = ctypes.c_double(0.0)
        self.lib.ghee_get_timings(self.handle, ctypes.byref(d), ctypes.byref(r))
        return d.value, r.value

    def last_error(self):
        e = self.lib.ghee_last_error(self.handle)
        return e.decode("utf-8") if e else ""

    def reset(self):
        self.lib.ghee_mp_reset(self.handle)
        self._mp = {}
        self._feeds = {}

    def close(self):
        if getattr(self, "handle", None):
            self.lib.ghee_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
