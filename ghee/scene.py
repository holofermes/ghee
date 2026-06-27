"""ghee — scene model for the audio-reactive shader renderer.

A Scene is options + an ordered LIST of passes (last pass -> screen). It's generic:
it knows nothing about audio/cava/"amplitudes". Those are the *app's* concern — the
host fills a pass's arrays with whatever data it has (cava bars, etc.) each frame.

A pass's inputs:
  * built-ins, always available, just use them: `time` (float), `resolution` (vec2).
  * ARRAYS: named uniform float arrays {name, size}. The author names them; the host
    fills them. Indexed via <name>_at() (constant-index loop when a sampler is in
    scope — the VC4 dynamic-index workaround).
  * CHANNELS: numbered samplers iChannel0..N, each bound to an image (URI) or a
    buffer (another pass's output; a pass pointing at ITSELF = feedback).
  * PARAMS: named typed sliders (decay, tint, ...).

There is no special "feedback"/"ping-pong"/"source" concept: feedback is just a
channel bound to its own pass's buffer; the engine double-buffers that internally.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

_GLSL_TYPE = {"float": "float", "int": "int", "vec2": "vec2",
              "vec3": "vec3", "vec4": "vec4"}


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
    def from_dict(cls, name: str, d: dict) -> "Param":
        return cls(name, d["type"], d["value"], d.get("min"), d.get("max"), d.get("step"))

    def to_dict(self) -> dict:
        d: dict = {"type": self.type, "value": self.value}
        for k in ("min", "max", "step"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        return d


@dataclass
class ArrayInput:
    """A named uniform float array. The host fills it each frame; ghee/the engine
    don't know or care what the data means."""
    name: str
    size: int

    @classmethod
    def from_dict(cls, d: dict) -> "ArrayInput":
        return cls(name=d["name"], size=int(d["size"]))

    def to_dict(self) -> dict:
        return {"name": self.name, "size": self.size}


@dataclass
class Channel:
    """A numbered sampler -> iChannel<index>, bound to exactly one of:
       image (a URI/path) or buffer (a pass name; self = feedback)."""
    image: str | None = None
    buffer: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Channel":
        return cls(image=d.get("image"), buffer=d.get("buffer"))

    def to_dict(self) -> dict:
        return {"image": self.image} if self.image is not None else {"buffer": self.buffer}


@dataclass
class DisplayProfile:
    """The target display. Supplied by the renderer, not stored in the scene file.
    orientation goes into the preamble as a #define so the shader picks its axis. bw is
    applied when the frame is packed for a 1-bit panel; the shader renders color.
    """
    orientation: str = "HORIZONTAL"
    bw: bool = False
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "DisplayProfile":
        return cls(d.get("orientation", "HORIZONTAL"), d.get("bw", False),
                   d.get("width"), d.get("height"))

    def to_dict(self) -> dict:
        out: dict = {"orientation": self.orientation, "bw": self.bw}
        if self.width is not None:
            out["width"] = self.width
        if self.height is not None:
            out["height"] = self.height
        return out


@dataclass
class Pass:
    name: str
    code: str
    format: str = "rgba8"
    arrays: list[ArrayInput] = field(default_factory=list)
    channels: list[Channel] = field(default_factory=list)
    params: dict[str, Param] = field(default_factory=dict)

    @property
    def reads_self(self) -> bool:
        """True if any channel feeds back from this pass's own buffer."""
        return any(c.buffer == self.name for c in self.channels)

    @classmethod
    def from_dict(cls, d: dict, base_dir: str | None = None) -> "Pass":
        code = d.get("code")
        if code is None and "shader" in d:
            path = d["shader"]
            if base_dir:
                path = os.path.join(base_dir, path)
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
        return cls(
            name=d.get("name", "main"),
            code=code or "",
            format=d.get("format", "rgba8"),
            arrays=[ArrayInput.from_dict(a) for a in d.get("arrays", [])],
            channels=[Channel.from_dict(c) for c in d.get("channels", [])],
            params={n: Param.from_dict(n, pd) for n, pd in d.get("params", {}).items()},
        )

    def to_dict(self) -> dict:
        out: dict = {"name": self.name, "code": self.code, "format": self.format}
        if self.arrays:
            out["arrays"] = [a.to_dict() for a in self.arrays]
        if self.channels:
            out["channels"] = [c.to_dict() for c in self.channels]
        if self.params:
            out["params"] = {n: p.to_dict() for n, p in self.params.items()}
        return out

    def glsl(self, profile: "DisplayProfile | None" = None) -> str:
        profile = profile or DisplayProfile()
        L = [
            "#version 100",
            "precision mediump float;",
            "varying vec2 v_texCoord;",
            "uniform vec2 resolution;",
            "uniform float time;",
            f'#define {"VERTICAL" if profile.orientation == "VERTICAL" else "HORIZONTAL"}',
        ]
        for a in self.arrays:
            L.append(f"uniform float {a.name}[{a.size}];")
            L.append(f"const int {a.name}_size = {a.size};")
        for i in range(len(self.channels)):
            L.append(f"uniform sampler2D iChannel{i};")
        for p in self.params.values():
            L.append(f"uniform {p.glsl_type} {p.name};")

        # Constant-index loop when a sampler is in scope (VC4 can't dynamically index).
        loop = bool(self.channels)
        for a in self.arrays:
            if loop:
                L += [f"float {a.name}_at(int i0){{",
                      f"for (int i=0;i<{a.size};i++){{ if (i==i0) return {a.name}[i]; }}",
                      "return 0.0; }"]
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
    def from_dict(cls, d: dict, base_dir: str | None = None) -> "Scene":
        return cls(version=d.get("version", 1),
                   passes=[Pass.from_dict(pd, base_dir) for pd in d.get("passes", [])])

    def to_dict(self) -> dict:
        return {"version": self.version,
                "passes": [p.to_dict() for p in self.passes]}

    @classmethod
    def load(cls, path: str) -> "Scene":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, base_dir=os.path.dirname(os.path.abspath(path)))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
