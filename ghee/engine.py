"""ghee.Renderer drives the C engine from Python.

Wraps the ghee_* ABI and the multipass orchestration in one place: build a Scene's
passes, wire channels, set params, feed named arrays, render to RGB. Data-agnostic:
the host supplies array data by name through a provider; the engine names nothing.

    r = Renderer(320, 240, lib_path="libghee.so")
    err = r.setup(scene, profile)               # "" or a compile error
    rgb = r.render(lambda name, size: data)      # (h, w, components) uint8
    r.set_param("main", "decay", "float", 0.9)   # live uniform, no recompile

`components` (3=RGB, 4=RGBA) and `lib_path` are host choices; ghee has no opinion
on the platform. Packing a frame to a 1bpp/RGB565 output is the optional pack() helper.
"""
import ctypes
import importlib.resources
import os
import re

import numpy as np

from ghee import DisplayProfile
from ghee.cparam import CParameter, create_c_parameter


def bundled_lib():
    """Path to the .so shipped inside the package (built by setup.py into lib/),
    or None if it was not built (e.g. installed on a host without the toolchain)."""
    try:
        p = importlib.resources.files("ghee") / "lib" / "libghee.so"
        return str(p) if p.is_file() else None
    except Exception:
        return None

_U8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
_U8_3 = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3, flags="C_CONTIGUOUS")
_F32_1 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")


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
    """A multipass renderer bound to one C-engine instance (one GL context).

    GL is thread-affine: construct and use a Renderer on a single thread (the
    context is made current at create time). A worker that renders in its own
    thread must create the Renderer in that thread, not hand one in from another.
    """

    def __init__(self, width, height, components=3, lib_path=None, dri_device=None):
        self.width = width
        self.height = height
        self.components = components
        self.scene = None
        self.error_line = None     # last setup() compile error: user-code line / pass index
        self.error_pass = None
        self._mp = {}              # pass name -> engine pass index
        self._feeds = []           # (pass index, array name, size)
        self._rgb = np.zeros((height, width, components), dtype=np.uint8)

        lib_path = lib_path or os.environ.get("GHEE_LIB") or bundled_lib()
        if not lib_path:
            raise ValueError("no engine .so: build ghee on this host, or pass "
                             "lib_path / set $GHEE_LIB")
        self._stdlib = ctypes.CDLL("")
        self.lib = ctypes.CDLL(lib_path)
        self._declare()
        err = ctypes.c_char_p()
        self.handle = self.lib.ghee_create(width, height, components,
                                          dri_device.encode() if dri_device else None,
                                          ctypes.byref(err))
        if not self.handle:
            raise RuntimeError(f"ghee_create failed: {err.value.decode() if err.value else 'unknown'}")

    def _declare(self):
        L = self.lib
        L.ghee_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]
        L.ghee_create.restype = ctypes.c_void_p
        L.ghee_destroy.argtypes = [ctypes.c_void_p]
        L.ghee_last_error.argtypes = [ctypes.c_void_p]; L.ghee_last_error.restype = ctypes.c_char_p
        L.ghee_mp_reset.argtypes = [ctypes.c_void_p]
        L.ghee_mp_add_pass.argtypes = [ctypes.c_void_p, ctypes.c_char_p]; L.ghee_mp_add_pass.restype = ctypes.c_int
        L.ghee_mp_set_channel.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.ghee_mp_set_uniform.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(CParameter)]; L.ghee_mp_set_uniform.restype = ctypes.c_bool
        L.ghee_mp_set_float_array.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, _F32_1, ctypes.c_int]
        L.ghee_mp_render.argtypes = [ctypes.c_void_p, _U8_3]
        L.ghee_pack_frame.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, _U8, _U8]
        L.ghee_set_profiling.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        L.ghee_get_timings.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

    def setup(self, scene, profile=None):
        """Build the scene's passes. Returns "" on success or a compile-error
        string (engine is left reset; the caller may restore the previous scene).
        On failure error_line/error_pass point at the offending user-code line."""
        profile = profile or DisplayProfile(width=self.width, height=self.height)
        self.error_line = None
        self.error_pass = None
        self.lib.ghee_mp_reset(self.handle)
        mp = {}
        for i, p in enumerate(scene.passes):
            glsl = p.glsl(profile)
            idx = self.lib.ghee_mp_add_pass(self.handle, glsl.encode("utf-8"))
            if idx < 0:
                err = self.last_error() or "compile error"
                self.error_line = _user_line(err, glsl, p.code)
                self.error_pass = i
                return err
            mp[p.name] = idx
        self._mp = mp
        for p in scene.passes:
            for c, ch in enumerate(p.channels):
                if ch.buffer is not None and ch.buffer in mp:
                    self.lib.ghee_mp_set_channel(self.handle, mp[p.name], c, mp[ch.buffer])
            for nm, pm in p.params.items():
                self.set_param(p.name, nm, pm.type, pm.value)
        self.scene = scene
        self._feeds = [(mp[p.name], a.name, a.size) for p in scene.passes for a in p.arrays]
        return ""

    def render(self, provider=None, out=None):
        """Render all passes; final pass -> the returned RGB(A) frame (h,w,components).
        provider(name, size) -> floats fills each declared array each frame; pass None
        to leave arrays untouched. `out` lets the host render into its own buffer."""
        if out is None:
            out = self._rgb
        if provider is not None:
            for pi, name, size in self._feeds:
                data = np.ascontiguousarray(provider(name, size), dtype=np.float32)
                self.lib.ghee_mp_set_float_array(self.handle, pi, name.encode("utf-8"), data, min(len(data), size))
        self.lib.ghee_mp_render(self.handle, out)
        return out

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
        if out is None:
            size = self.width * ((self.height + 7) // 8) if bw else self.width * self.height * 2
            out = np.zeros(size, dtype=np.uint8)
        else:
            out.fill(0)
        self.lib.ghee_pack_frame(self.width, self.height, self.components, 1 if bw else 0, rgb, out)
        return out

    def set_profiling(self, on):
        """Enable per-frame draw/readback timing (adds a glFinish; off = zero cost)."""
        self.lib.ghee_set_profiling(self.handle, bool(on))

    def timings(self):
        """Last frame's split as (draw_ms, read_ms): GPU render of all passes vs the
        glReadPixels readback. Both 0 until a frame renders with profiling on."""
        d = ctypes.c_double(0.0); r = ctypes.c_double(0.0)
        self.lib.ghee_get_timings(self.handle, ctypes.byref(d), ctypes.byref(r))
        return d.value, r.value

    def last_error(self):
        e = self.lib.ghee_last_error(self.handle)
        return e.decode("utf-8") if e else ""

    def reset(self):
        self.lib.ghee_mp_reset(self.handle)
        self._mp = {}
        self._feeds = []

    def close(self):
        if getattr(self, "handle", None):
            self.lib.ghee_destroy(self.handle)
            self.handle = None
            try:
                self._stdlib.dlclose(self.lib._handle)
            except Exception:
                pass
            self.lib = None
