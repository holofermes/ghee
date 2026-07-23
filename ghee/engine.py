"""ghee.Renderer drives the C engine from Python.

Wraps the ghee_* ABI.

    r = Renderer(320, 240, lib_path="libghee.so")
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

from ghee.cparam import CParameter, create_c_parameter


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
_U8_3 = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3, flags="C_CONTIGUOUS")
_F32_1 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")


def packed_size(width, height, bw):
    """Bytes a frame packs down to: 1bpp (bw) vs RGB565."""
    return width * ((height + 7) // 8) if bw else width * height * 2


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
        self._feeds = {}  # array name -> [(pass index, size), ...]
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
            # render node, and DRM master (run as root).
            dev = dri_device or os.environ.get("GHEE_DRI") or "/dev/dri/card0"
            self.handle = self.lib.ghee_display_create(
                width, height, dev.encode(), 1 if rgb565 else 0, ctypes.byref(err)
            )
            if not self.handle:
                raise RuntimeError(
                    f"ghee_display_create failed: "
                    f"{err.value.decode() if err.value else 'unknown'}"
                )
            return
        # No explicit device: probe $GHEE_DRI, the Pi's card0, the desktop
        # render node, then any remaining cards. First one that creates wins.
        if dri_device:
            devices = [dri_device]
        else:
            devices = [os.environ.get("GHEE_DRI"), "/dev/dri/card0"]
            devices += ["/dev/dri/renderD128", *sorted(glob.glob("/dev/dri/card*"))]
            devices = list(dict.fromkeys(d for d in devices if d))
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
        L.ghee_mp_set_channel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        L.ghee_mp_set_uniform.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(CParameter),
        ]
        L.ghee_mp_set_uniform.restype = ctypes.c_bool
        L.ghee_mp_set_float_texture.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_char_p,
            _F32_1,
            ctypes.c_int,
            ctypes.c_int,
        ]
        L.ghee_mp_render.argtypes = [ctypes.c_void_p, _U8_3]
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
        L.ghee_set_profiling.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        L.ghee_set_time.argtypes = [ctypes.c_void_p, ctypes.c_float]
        L.ghee_get_timings.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]

    def setup(self, scene):
        """Build the scene's passes. Returns "" on success or an error string (a compile
        error, or a scene the engine refuses). On a compile error error_line/error_pass
        point at the offending user-code line. The engine is untouched on a returned
        error, so the caller may keep rendering the previous scene."""
        self.error = ""
        self.error_line = None
        self.error_pass = None
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
        self.lib.ghee_mp_reset(self.handle)
        mp = {}
        for i, p in enumerate(scene.passes):
            glsl = p.glsl()
            idx = self.lib.ghee_mp_add_pass(self.handle, glsl.encode("utf-8"))
            if idx < 0:
                err = self.last_error() or "compile error"
                self.error = err
                self.error_line = _user_line(err, glsl, p.code)
                self.error_pass = i
                return err
            mp[p.name] = idx
        self._mp = mp
        for p in scene.passes:
            for c, ch in enumerate(p.channels):
                if ch.buffer is not None and ch.buffer in mp:
                    self.lib.ghee_mp_set_channel(
                        self.handle, mp[p.name], c, mp[ch.buffer]
                    )
            for nm, pm in p.params.items():
                self.set_param(p.name, nm, pm.type, pm.value)
        self.scene = scene
        # Group array feeds by name: an array name is one logical input, so the provider
        # is asked for it once per frame and the result goes to every pass that declares
        # it (sizes are validated equal at the top of setup).
        feeds = {}
        for p in scene.passes:
            for a in p.arrays:
                feeds.setdefault(a.name, []).append((mp[p.name], a.size))
        self._feeds = feeds
        return ""

    def render(self, provider=None, out=None):
        """Render all passes; final pass -> the returned RGB(A) frame (h,w,components).
        provider(name, size) -> floats fills each declared array each frame (called once
        per unique array name); pass None to leave arrays untouched. `out` lets the host
        render into its own buffer."""
        if out is None:
            out = self._rgb
        self._feed(provider)
        self.lib.ghee_mp_render(self.handle, out)
        return out

    def _feed(self, provider):
        """Upload each declared array for this frame (called once per unique name).
        A render uploads whatever the provider returns; the one exception is the
        explicit UNCHANGED sentinel, a scene's declaration that the texture
        already holds this array (a partially dynamic scene skips its static
        mask that way). The engine never caches or diffs data on its own."""
        if provider is None:
            return
        for name, targets in self._feeds.items():
            size = targets[0][1]
            raw = provider(name, size)
            if raw is UNCHANGED:
                continue
            data = np.ascontiguousarray(raw, dtype=np.float32).reshape(-1)
            enc = name.encode("utf-8")
            if isinstance(size, int):
                w, h = min(len(data), size), 1
            else:
                w, h = size
                if len(data) < w * h:  # short feed: zero-pad to the full mask
                    buf = np.zeros(w * h, dtype=np.float32)
                    buf[: len(data)] = data
                    data = buf
            for pi, _sz in targets:
                self.lib.ghee_mp_set_float_texture(self.handle, pi, enc, data, w, h)

    def present(self, provider=None, out=None, out_size=None):
        """Display mode only: feed arrays, render the passes, and scan the final
        buffer straight out to the connected display. Pass out=<h,w,4 buffer> and
        out_size=(w, h) to also read back a GPU-downscaled preview (a few hundred
        KB, not the full frame); None skips the readback."""
        self._feed(provider)
        if out is not None and out_size is not None:
            buf = out.ctypes.data_as(ctypes.c_void_p)
            pw, ph = out_size
        else:
            buf, pw, ph = None, 0, 0
        self.lib.ghee_mp_present(self.handle, buf, pw, ph)

    def set_param(self, pass_name, name, ptype, value):
        """Set one uniform on a pass live (no recompile). False if the pass/uniform
        is absent."""
        idx = self._mp.get(pass_name)
        if idx is None:
            return False
        cp = create_c_parameter({"type": ptype, "value": value})
        cp.name = name.encode("utf-8")
        return self.lib.ghee_mp_set_uniform(self.handle, idx, ctypes.byref(cp))

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
        is the monotonic clock since the engine was created."""
        self.lib.ghee_set_time(self.handle, float(t))

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
