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

Image/texture channels and per-pass float buffer formats are not modeled yet :(
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

_GLSL_TYPE = {
    "float": "float",
    "int": "int",
    "vec2": "vec2",
    "vec3": "vec3",
    "vec4": "vec4",
}


@dataclass
class Param:
    name: str
    type: str
    value: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None

    @property
    def glsl_type(self) -> str:
        return _GLSL_TYPE[self.type]

    @classmethod
    def from_dict(cls, name: str, d: dict) -> Param:
        return cls(
            name, d["type"], d["value"], d.get("min"), d.get("max"), d.get("step")
        )

    def to_dict(self) -> dict:
        d: dict = {"type": self.type, "value": self.value}
        for k in ("min", "max", "step"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        return d


@dataclass
class ArrayInput:
    """A named uniform float array."""

    name: str
    size: int

    @classmethod
    def from_dict(cls, d: dict) -> ArrayInput:
        return cls(name=d["name"], size=int(d["size"]))

    def to_dict(self) -> dict:
        return {"name": self.name, "size": self.size}


@dataclass
class Channel:
    """A numbered sampler -> iChannel<index>, bound to a pass's output buffer."""

    buffer: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Channel:
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

    @property
    def reads_self(self) -> bool:
        """True if any channel feeds back from this pass's own buffer."""
        return any(c.buffer == self.name for c in self.channels)

    @classmethod
    def from_dict(cls, d: dict, base_dir: str | None = None) -> Pass:
        code = d.get("code")
        if code is None and "shader" in d:
            path = d["shader"]
            if base_dir:
                path = os.path.join(base_dir, path)
            with open(path, encoding="utf-8") as f:
                code = f.read()
        return cls(
            name=d.get("name", "main"),
            code=code or "",
            arrays=[ArrayInput.from_dict(a) for a in d.get("arrays", [])],
            channels=[Channel.from_dict(c) for c in d.get("channels", [])],
            params={n: Param.from_dict(n, pd) for n, pd in d.get("params", {}).items()},
        )

    def to_dict(self) -> dict:
        out: dict = {"name": self.name, "code": self.code}
        if self.arrays:
            out["arrays"] = [a.to_dict() for a in self.arrays]
        if self.channels:
            out["channels"] = [c.to_dict() for c in self.channels]
        if self.params:
            out["params"] = {n: p.to_dict() for n, p in self.params.items()}
        return out

    def glsl(self) -> str:
        L = [
            "#version 100",
            "precision mediump float;",
            "varying vec2 v_texCoord;",
            "uniform vec2 resolution;",
            "uniform float time;",
        ]
        for a in self.arrays:
            L.append(f"uniform float {a.name}[{a.size}];")
            L.append(f"const int {a.name}_size = {a.size};")
        for i in range(len(self.channels)):
            L.append(f"uniform sampler2D iChannel{i};")
        for p in self.params.values():
            L.append(f"uniform {p.glsl_type} {p.name};")

        # Constant-index loop when a sampler is in scope - Pi 3 and older VC4 can't dynamically index :/
        loop = bool(self.channels)
        for a in self.arrays:
            if loop:
                L += [
                    f"float {a.name}_at(int i0){{",
                    f"for (int i=0;i<{a.size};i++){{ if (i==i0) return {a.name}[i]; }}",
                    "return 0.0; }",
                ]
            else:
                L += [f"float {a.name}_at(int i0){{ return {a.name}[i0]; }}"]

        return "\n".join(L) + "\n\n" + self.code


@dataclass
class Scene:
    passes: list[Pass] = field(default_factory=list)
    version: int = 1

    @property
    def output(self) -> Pass:
        return self.passes[-1]

    @classmethod
    def from_dict(cls, d: dict, base_dir: str | None = None) -> Scene:
        return cls(
            version=d.get("version", 1),
            passes=[Pass.from_dict(pd, base_dir) for pd in d.get("passes", [])],
        )

    def to_dict(self) -> dict:
        return {"version": self.version, "passes": [p.to_dict() for p in self.passes]}

    @classmethod
    def load(cls, path: str) -> Scene:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, base_dir=os.path.dirname(os.path.abspath(path)))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
