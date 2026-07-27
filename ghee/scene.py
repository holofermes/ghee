"""ghee scene model for the multipass shader renderer.

A Scene is options + an ordered list of passes. The host fills each pass's arrays each frame.

A pass's inputs. The names below are the contract between this preamble and the ghee
engine: ghee_mp_render binds `time`, `resolution`, and `iChannel0..N` by these exact
literal names. Arrays and params are bound by the author's chosen name.
  * built-ins, always available: `time` (float), `resolution` (vec2).
  * ARRAYS: named uniform float arrays {name, size}. Indexed via <name>_at()
  * CHANNELS: numbered samplers iChannel0..N, each bound to another pass's output
    buffer, a pass can also point at itself. A forward ref reads
    that pass's current frame; a self/later ref reads its previous frame.
  * PARAMS: named typed attributes.

Image/texture channels and per-pass float buffer formats are not modeled.

A pass may set `size` [w, h] to render at its own resolution; consumers sample it
through normalized UVs so mixed sizes compose, and the output (last) pass omits it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

_GLSL_TYPE = {
    "float": "float",
    "int": "int",
    "bool": "bool",
}

# vecK params have no engine type: like color, the preamble decomposes them to K
# float uniforms named by these suffixes, so they never reach the C ABI as one value.
_VEC_ARITY = {"vec2": 2, "vec3": 3, "vec4": 4}
_VEC_SUFFIXES = ("_x", "_y", "_z", "_w")

_PARAM_TYPES = ("float", "int", "bool", "color", "string", *_VEC_ARITY)

# Arrays and passes become GL textures, so a size is bounded by what a texture
# can be: one dimension caps at 16384 (above any GLES2 GL_MAX_TEXTURE_SIZE) and
# a 2D input or a sized pass caps at 16M texels. A scene naming more than that
# is malformed, and would otherwise reach np.arange / glTexImage2D as an
# allocation the size of the number.
_MAX_DIM = 16384
_MAX_TEXELS = 1 << 24


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_dim(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and 0 < x <= _MAX_DIM


def _int(x):
    """A JSON size is an int or a whole float. Anything else passes through
    untouched, so the size check reports what the scene actually said."""
    return int(x) if _is_num(x) and float(x).is_integer() else x


def _list(d: dict, key: str) -> list:
    v = d.get(key, [])
    if not isinstance(v, list):
        raise ValueError(f"'{key}' needs a list, got {v!r}")
    return v


def _obj(d: dict, key: str) -> dict:
    v = d.get(key, {})
    if not isinstance(v, dict):
        raise ValueError(f"'{key}' needs an object of name -> definition, got {v!r}")
    return v


def _read_shader(path: str, base_dir: str | None, pass_name: str) -> str:
    """A pass's `shader` file, read from the scene's own directory. A scene with
    no directory (one posted to the console, one built in memory) cannot name
    files at all, and one that has a directory cannot reach outside it: either
    would turn a scene into a read of any file on the host."""
    if not isinstance(path, str):
        raise ValueError(f"pass '{pass_name}' shader needs a filename, got {path!r}")
    if not base_dir:
        raise ValueError(
            f"pass '{pass_name}' names a shader file ({path!r}) but the scene has "
            "no directory to resolve it against; inline the GLSL in 'code'"
        )
    root = os.path.normpath(os.path.abspath(base_dir))
    full = os.path.normpath(os.path.join(root, path))
    if full != root and not full.startswith(root + os.sep):
        raise ValueError(
            f"pass '{pass_name}' shader {path!r} is outside the scene directory"
        )
    try:
        with open(full, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise ValueError(
            f"pass '{pass_name}' shader {path!r}: {e.strerror or e}"
        ) from None


def _param_uniform_lines(p: Param) -> list[str]:
    """Preamble uniform declarations for one param. A color decomposes to three
    float uniforms plus a vec3 alias; a vecK to K floats plus a vecK alias (the
    engine has no color/vec type, it only ever sets floats). A scalar is one
    typed uniform. A string is host-side only (the scene's controller consumes
    it): no uniform at all. Shared by pass-scoped and scene-scoped params."""
    if p.type == "string":
        return []
    if p.type == "color":
        return [
            f"uniform float {p.name}_r;",
            f"uniform float {p.name}_g;",
            f"uniform float {p.name}_b;",
            f"#define {p.name} vec3({p.name}_r, {p.name}_g, {p.name}_b)",
        ]
    if p.type in _VEC_ARITY:
        sfx = _VEC_SUFFIXES[: _VEC_ARITY[p.type]]
        args = ", ".join(f"{p.name}{s}" for s in sfx)
        return [f"uniform float {p.name}{s};" for s in sfx] + [
            f"#define {p.name} {p.type}({args})"
        ]
    return [f"uniform {p.glsl_type} {p.name};"]


@dataclass
class Param:
    name: str
    type: str
    value: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    # string params only: offered choices, rendered as a dropdown (free text
    # input when absent)
    options: list[str] | None = None

    def __post_init__(self) -> None:
        if self.type not in _PARAM_TYPES:
            raise ValueError(
                f"param '{self.name}' has unknown type {self.type!r}; "
                f"known types: {', '.join(_PARAM_TYPES)}"
            )
        # A color is [r, g, b] floats 0..1. It has no engine type: the preamble
        # decomposes it to three float uniforms, so it never reaches glsl_type
        # or the C ABI as one value.
        if self.type == "color":
            v = self.value
            if (
                not isinstance(v, (list, tuple))
                or len(v) != 3
                or not all(_is_num(x) for x in v)
            ):
                raise ValueError(
                    f"color param '{self.name}' needs three numbers [r, g, b], "
                    f"got {v!r}"
                )
            self.value = [float(x) for x in v]
        elif self.type in _VEC_ARITY:
            n = _VEC_ARITY[self.type]
            v = self.value
            if (
                not isinstance(v, (list, tuple))
                or len(v) != n
                or not all(_is_num(x) for x in v)
            ):
                raise ValueError(
                    f"{self.type} param '{self.name}' needs {n} numbers, got {v!r}"
                )
            self.value = [float(x) for x in v]
        elif self.type == "string":
            if not isinstance(self.value, str):
                raise ValueError(
                    f"string param '{self.name}' needs a string, got {self.value!r}"
                )
        # A bool is `uniform bool` in the shader and an int over the C ABI (the
        # engine has no bool type; glUniform1i sets a bool uniform).
        elif self.type == "bool":
            if not isinstance(self.value, bool):
                raise ValueError(
                    f"bool param '{self.name}' needs true or false, got {self.value!r}"
                )
        else:
            if not _is_num(self.value):
                raise ValueError(
                    f"{self.type} param '{self.name}' needs a number, "
                    f"got {self.value!r}"
                )
            self.value = int(self.value) if self.type == "int" else float(self.value)
        for k in ("min", "max", "step"):
            r = getattr(self, k)
            if r is not None and not _is_num(r):
                raise ValueError(f"param '{self.name}' {k} needs a number, got {r!r}")
        if self.options is not None:
            if self.type != "string":
                raise ValueError(
                    f"param '{self.name}': options belong to string params only"
                )
            if not isinstance(self.options, (list, tuple)) or not all(
                isinstance(o, str) for o in self.options
            ):
                raise ValueError(
                    f"string param '{self.name}' options need a list of strings"
                )
            self.options = list(self.options)

    @property
    def glsl_type(self) -> str:
        if self.type == "string":
            raise ValueError("a string param is host-side; it has no glsl_type")
        if self.type == "color":
            raise ValueError(
                "color decomposes to three float uniforms; it has no glsl_type"
            )
        if self.type in _VEC_ARITY:
            raise ValueError(
                f"{self.type} decomposes to {_VEC_ARITY[self.type]} float uniforms; "
                "it has no glsl_type"
            )
        return _GLSL_TYPE[self.type]

    @classmethod
    def from_dict(cls, name: str, d: dict) -> Param:
        if not isinstance(d, dict):
            raise ValueError(f"param '{name}' needs an object, got {d!r}")
        for k in ("type", "value"):
            if k not in d:
                raise ValueError(f"param '{name}' has no '{k}'")
        return cls(
            name,
            d["type"],
            d["value"],
            d.get("min"),
            d.get("max"),
            d.get("step"),
            d.get("options"),
        )

    def to_dict(self) -> dict:
        d: dict = {"type": self.type, "value": self.value}
        for k in ("min", "max", "step", "options"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        return d


@dataclass
class ArrayInput:
    """A named float input, fed by the host and bound as a texture sampler.
    size N is a 1xN array: one fetch per computed index (no VC4 uniform-array
    unroll or size cap), read as <name>_at(i) or <name>_sample(t). size [w, h]
    is a 2D w x h mask (host-rasterized text, images), read as
    <name>_sample2(uv)."""

    name: str
    size: int | tuple[int, int]

    def __post_init__(self) -> None:
        s = self.size
        if isinstance(s, (list, tuple)):
            if len(s) != 2 or not all(_is_dim(x) for x in s):
                raise ValueError(
                    f"array '{self.name}' size needs two ints in 1..{_MAX_DIM} "
                    f"[w, h], got {s!r}"
                )
            if s[0] * s[1] > _MAX_TEXELS:
                raise ValueError(
                    f"array '{self.name}' is {s[0]}x{s[1]}, over the "
                    f"{_MAX_TEXELS} element cap"
                )
            self.size = (s[0], s[1])
        elif not _is_dim(s):
            raise ValueError(
                f"array '{self.name}' size needs an int in 1..{_MAX_DIM}, got {s!r}"
            )

    @property
    def is_2d(self) -> bool:
        return not isinstance(self.size, int)

    @classmethod
    def from_dict(cls, d: dict) -> ArrayInput:
        if not isinstance(d, dict) or "name" not in d or "size" not in d:
            raise ValueError(f"an array needs 'name' and 'size', got {d!r}")
        sz = d["size"]
        if isinstance(sz, (list, tuple)):
            if len(sz) != 2:
                raise ValueError(
                    f"array '{d['name']}' size needs two ints [w, h], got {sz!r}"
                )
            return cls(name=d["name"], size=(_int(sz[0]), _int(sz[1])))
        return cls(name=d["name"], size=_int(sz))

    def to_dict(self) -> dict:
        return {"name": self.name, "size": list(self.size) if self.is_2d else self.size}


@dataclass
class Channel:
    """A numbered sampler -> iChannel<index>, bound to a pass's output buffer."""

    buffer: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Channel:
        if not isinstance(d, dict):
            raise ValueError(
                f"a channel needs an object like {{'buffer': 'pass'}}, got {d!r}"
            )
        return cls(buffer=d.get("buffer"))

    def to_dict(self) -> dict:
        return {"buffer": self.buffer}


@dataclass
class Pass:
    name: str
    code: str
    arrays: list[ArrayInput] = field(default_factory=list)
    channels: list[Channel] = field(default_factory=list)
    params: dict[str, Param] = field(default_factory=dict)
    filter: str = ""  # non-empty = this pass came from a named filter (read-only in UI)
    # optional [w, h] this pass renders at; None = the renderer's size. A sized
    # pass is sampled through normalized UVs, so consumers ignore its size. The
    # output (last) pass must leave this None (it feeds readback/display/encoder).
    size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.size is None:
            return
        s = self.size
        if (
            not isinstance(s, (list, tuple))
            or len(s) != 2
            or not all(_is_dim(x) for x in s)
        ):
            raise ValueError(
                f"pass '{self.name}' size needs two positive ints [w, h] up to "
                f"{_MAX_DIM}, got {s!r}"
            )
        if s[0] * s[1] > _MAX_TEXELS:
            raise ValueError(
                f"pass '{self.name}' is {s[0]}x{s[1]}, over the {_MAX_TEXELS} pixel cap"
            )
        self.size = (s[0], s[1])

    @property
    def reads_self(self) -> bool:
        """True if any channel feeds back from this pass's own buffer."""
        return any(c.buffer == self.name for c in self.channels)

    @classmethod
    def from_dict(cls, d: dict, base_dir: str | None = None) -> Pass:
        if not isinstance(d, dict):
            raise ValueError(f"a pass needs a JSON object, got {d!r}")
        name = d.get("name", "main")
        code = d.get("code")
        if code is None:
            if "shader" not in d:
                raise ValueError(
                    f"pass '{name}' has neither 'code' (inline GLSL) nor "
                    "'shader' (a file next to the scene)"
                )
            code = _read_shader(d["shader"], base_dir, name)
        try:
            arrays = [ArrayInput.from_dict(a) for a in _list(d, "arrays")]
            channels = [Channel.from_dict(c) for c in _list(d, "channels")]
            params = {n: Param.from_dict(n, pd) for n, pd in _obj(d, "params").items()}
        except ValueError as e:
            raise ValueError(f"pass '{name}': {e}") from None
        return cls(
            name=name,
            code=code,
            arrays=arrays,
            channels=channels,
            params=params,
            filter=d.get("filter", ""),
            size=d.get("size"),
        )

    def to_dict(self) -> dict:
        out: dict = {"name": self.name, "code": self.code}
        if self.size is not None:
            out["size"] = list(self.size)
        if self.arrays:
            out["arrays"] = [a.to_dict() for a in self.arrays]
        if self.channels:
            out["channels"] = [c.to_dict() for c in self.channels]
        if self.params:
            out["params"] = {n: p.to_dict() for n, p in self.params.items()}
        if self.filter:
            out["filter"] = self.filter
        return out

    def glsl(self, scene_params: dict[str, Param] | None = None) -> str:
        L = [
            "#version 100",
            # highp where the fragment stage supports it (VC4 does), else mediump.
            # mediump cannot represent texel coords like (col+0.5)/w precisely, so a
            # feedback pass that resamples its own output drifts a fraction of a pixel
            # per frame and shears over time. highp keeps ping-pong buffers stable.
            "#ifdef GL_FRAGMENT_PRECISION_HIGH",
            "precision highp float;",
            "#else",
            "precision mediump float;",
            "#endif",
            "varying vec2 v_texCoord;",
            "uniform vec2 resolution;",
            "uniform float time;",
            # VC4's gl_FragCoord.xy is sheared ~1/3 px per row when a pass samples
            # its own output as feedback (the interpolated varying is exact). Map
            # gl_FragCoord to the varying so scene code (gl_FragCoord.xy/resolution)
            # samples feedback without drifting. No effect on non-feedback passes.
            "#define gl_FragCoord vec4(v_texCoord * resolution, 0.0, 1.0)",
        ]
        for a in self.arrays:
            L.append(f"uniform sampler2D {a.name}_tex;")
            if not a.is_2d:
                L.append(f"const int {a.name}_size = {a.size};")
        for i in range(len(self.channels)):
            L.append(f"uniform sampler2D iChannel{i};")
        # Scene-scoped params are emitted into every pass identically (explicit
        # sharing, no name-matching), ahead of this pass's own params.
        for p in (scene_params or {}).values():
            L += _param_uniform_lines(p)
        for p in self.params.values():
            L += _param_uniform_lines(p)

        # A 1xN data texture: _at(i) is one fetch with a computed coord (no VC4
        # unroll or size cap), _sample(t) interpolates between elements for free.
        # A 2D input is one normalized-uv fetch: _sample2(uv).
        for a in self.arrays:
            if a.is_2d:
                L.append(
                    f"float {a.name}_sample2(vec2 uv){{ "
                    f"return texture2D({a.name}_tex, uv).r; }}"
                )
                continue
            L += [
                f"float {a.name}_at(int i0){{ return texture2D({a.name}_tex, "
                f"vec2((float(i0)+0.5)/float({a.name}_size), 0.5)).r; }}",
                f"float {a.name}_sample(float t){{ "
                f"return texture2D({a.name}_tex, vec2(t, 0.5)).r; }}",
            ]

        return "\n".join(L) + "\n\n" + self.code


@dataclass
class Scene:
    passes: list[Pass] = field(default_factory=list)
    version: int = 1
    # named filters (post-process passes) to append after the scene's own passes,
    # in order; the last one is what reaches the display. Resolved via ghee.filters.
    filters: list[str] = field(default_factory=list)
    # scene-scoped params: emitted into every pass's preamble and set on all of
    # them at once. Same shape as a pass's params.
    params: dict[str, Param] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for p in self.passes:
            if p.name in seen:
                raise ValueError(
                    f"pass '{p.name}' is declared twice; a pass name is the "
                    "buffer name channels bind to, so it must be unique"
                )
            seen.add(p.name)
        # A name belongs to one scope only. Sharing a scene param with a pass is
        # explicit (it reaches every pass), so a pass declaring the same name is
        # an ambiguous redeclaration, not a link. Reject it by name and pass.
        for p in self.passes:
            for nm in p.params:
                if nm in self.params:
                    raise ValueError(
                        f"param '{nm}' is declared at scene scope and in pass "
                        f"'{p.name}'; a name belongs to one scope only"
                    )

    @property
    def output(self) -> Pass:
        return self.passes[-1]

    @classmethod
    def from_dict(
        cls, d: dict, base_dir: str | None = None, path: str | None = None
    ) -> Scene:
        """`path` is only used to name the scene in error messages."""
        src = path or "scene"
        if not isinstance(d, dict):
            raise ValueError(f"{src}: needs a JSON object, got {d!r}")
        try:
            filters = _list(d, "filters")
            if not all(isinstance(n, str) for n in filters):
                raise ValueError(f"'filters' needs a list of names, got {filters!r}")
            return cls(
                version=d.get("version", 1),
                passes=[Pass.from_dict(pd, base_dir) for pd in _list(d, "passes")],
                filters=list(filters),
                params={
                    n: Param.from_dict(n, pd) for n, pd in _obj(d, "params").items()
                },
            )
        except ValueError as e:
            raise ValueError(f"{src}: {e}") from None

    def to_dict(self) -> dict:
        out = {"version": self.version, "passes": [p.to_dict() for p in self.passes]}
        if self.filters:
            out["filters"] = self.filters
        if self.params:
            out["params"] = {n: p.to_dict() for n, p in self.params.items()}
        return out

    @classmethod
    def load(cls, path: str) -> Scene:
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except ValueError as e:
                raise ValueError(f"{path}: invalid JSON: {e}") from None
        return cls.from_dict(
            data, base_dir=os.path.dirname(os.path.abspath(path)), path=path
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
