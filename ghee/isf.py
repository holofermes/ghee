"""Import an ISF (Interactive Shader Format) shader as a ghee scene or filter.

An ISF file is a fragment shader with a JSON header in a leading block comment:
INPUTS (typed uniforms plus image and audio samplers), PASSES (the render
targets and the size each one draws at), and a body written against ISF's own
built-ins.

A generated shim maps ISF onto the ghee preamble:

    RENDERSIZE -> resolution         isf_FragNormCoord -> v_texCoord
    TIME       -> time               an image input / PASSES target -> iChannelN
    IMG_*      -> texture2D()        an audio input -> a ghee array
    PASSINDEX  -> a per-pass const int, so the compiler folds the branches

ISF puts one body in every pass and branches on PASSINDEX, so each ghee pass
gets that same body with its own PASSINDEX constant.

Two things the body cannot keep. A file-scope initializer that reads a uniform
is legal in the GLSL 1.20 ISF is written against and illegal in ESSL 1.00, so
it is hoisted into main. IMG_SIZE cannot be a function (a sampler2D carries no
size), so IMG_SIZE(x) becomes the constant x_isf_size.

Refused, by name and reason: FLOAT (32-bit) pass buffers, which ghee's 8-bit
buffers would saturate silently; a custom ISF vertex shader, whose varyings
ghee's fixed vertex shader does not emit; IMPORTED external images.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field

from ghee.scene import ArrayInput, Channel, Param, Pass, Scene

# An audio input that names no MAX gets this many bins.
_AUDIO_BINS = 256

# Fetch estimate over this many per fragment: too slow to hold 30fps on VC4.
_FETCH_BUDGET = 200

# Texture units on VC4, the smallest GPU ghee targets.
_VC4_UNITS = 8

# Names ghee's own preamble owns. An ISF input or target using one would shadow
# it, so it is renamed here and in the body.
_RESERVED = frozenset({"time", "resolution", "v_texCoord"})

_IDENT = re.compile(r"[A-Za-z_]\w*$")
_CHANNEL = re.compile(r"iChannel\d+$")
_VARYING = re.compile(r"^[ \t]*(?:varying|in)[ \t]+\w+[ \t]+(\w+)", re.M)
_MAIN = re.compile(r"\bvoid\s+main\s*\([^)]*\)\s*\{")
_FETCH = re.compile(r"\b(?:IMG_(?:NORM_)?(?:THIS_)?(?:NORM_)?PIXEL|texture2D)\s*\(")
_FOR = re.compile(r"\bfor\s*\(")
_LIMIT = re.compile(r"<=?\s*(\d+)")

_GLSL_TYPES = frozenset(
    "float int bool vec2 vec3 vec4 ivec2 ivec3 ivec4 bvec2 bvec3 bvec4 "
    "mat2 mat3 mat4".split()
)

_IMG_ACCESSORS = (
    "vec4 IMG_PIXEL(vec2 sz, sampler2D s, vec2 p) { return texture2D(s, p / sz); }",
    "vec4 IMG_NORM_PIXEL(sampler2D s, vec2 p) { return texture2D(s, p); }",
    "vec4 IMG_THIS_PIXEL(sampler2D s) { return texture2D(s, v_texCoord); }",
    "vec4 IMG_THIS_NORM_PIXEL(sampler2D s) { return texture2D(s, v_texCoord); }",
    "vec4 IMG_NORM_THIS_PIXEL(sampler2D s) { return texture2D(s, v_texCoord); }",
)

# What an unbound image input reads in scene mode: a checkered gradient with a
# disc, so a filter imported as a scene shows what it does to something.
SOURCE_PASS = "isf_source"
_SOURCE_CODE = """void main() {
    vec2 uv = v_texCoord;
    vec2 g = floor(uv * 8.0);
    float checker = mod(g.x + g.y, 2.0);
    float disc = smoothstep(0.30, 0.28, distance(uv, vec2(0.5)));
    vec3 col = vec3(uv.x, uv.y, 1.0 - uv.y) * (0.4 + 0.45 * checker);
    gl_FragColor = vec4(mix(col, vec3(1.0, 0.85, 0.2), disc), 1.0);
}
"""


class ISFError(ValueError):
    """An ISF file ghee cannot import, with the reason and the file it came from."""


def read_isf(path: str) -> tuple[str, str, bool]:
    """An ISF source, a name to call it, and whether a sibling .vs sits next to
    it. `path` is a local file or an http(s) URL. A URL has no sibling to look
    for; a custom vertex shader there is caught by the varyings it leaves in the
    body instead."""
    if path.startswith(("http://", "https://")):
        import urllib.parse
        import urllib.request

        with urllib.request.urlopen(path, timeout=30) as r:
            text = r.read().decode("utf-8", "replace")
        name = urllib.parse.unquote(os.path.basename(path.split("?")[0]))
        return text, name or path, False
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    stem, _ = os.path.splitext(path)
    return text, os.path.basename(path), os.path.exists(stem + ".vs")


def split_header(text: str) -> tuple[dict, str]:
    """The leading /*{ ... }*/ JSON header and the GLSL after it."""
    m = re.search(r"/\*(.*?)\*/", text, re.S)
    if m is None or not m.group(1).lstrip().startswith("{"):
        raise ISFError("no leading /*{ ... }*/ JSON header, so it is not ISF")
    try:
        meta = json.loads(m.group(1))
    except ValueError as e:
        raise ISFError(f"the JSON header does not parse: {e}") from None
    if not isinstance(meta, dict):
        raise ISFError(f"the JSON header needs an object, got {meta!r}")
    return meta, text[m.end() :]


def eval_size(expr, values: dict) -> int:
    """An ISF WIDTH/HEIGHT: a number, or an equation over $WIDTH, $HEIGHT and
    $<inputName> with + - * / and floor/ceil/abs/min/max. Rounds down to a
    positive int, which is what a render target can be."""
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        v = float(expr)
    elif isinstance(expr, str):
        v = _Expr(expr, values).parse()
    else:
        raise ISFError(f"size {expr!r} is neither a number nor an equation")
    if not math.isfinite(v):
        raise ISFError(f"size equation {expr!r} does not evaluate to a number")
    return max(1, int(v))


def hoist_globals(body: str) -> tuple[str, list[str]]:
    """Move every non-const file-scope initializer into the top of main.
    ESSL 1.00 wants a constant expression there and GLSL 1.20 does not, so
    `float ratio = RENDERSIZE.x / RENDERSIZE.y;` at file scope is the single
    most common reason an ISF body will not compile. Returns the body and the
    names hoisted."""
    mask = _mask(body)
    edits: list[tuple[int, int, str]] = []
    names: list[str] = []
    assigns: list[str] = []
    depth = par = start = 0
    for i, c in enumerate(mask):
        if c in "([":
            par += 1
        elif c in ")]":
            par -= 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                start, par = i + 1, 0
        elif c == ";" and depth == 0 and par == 0:
            # from the masked text: a statement's slice starts after the last
            # `;` or `}`, so it can carry a comment block the rewrite must keep
            head = len(mask[start:i]) - len(mask[start:i].lstrip())
            split = _split_decl(mask[start + head : i])
            if split is not None:
                decl, sets = split
                edits.append((start + head, i + 1, decl))
                names += [nm for nm, _ in sets]
                assigns += [f"    {nm} = {rhs};" for nm, rhs in sets]
            start = i + 1
    if not assigns:
        return body, []
    m = _MAIN.search(mask)
    if m is None:
        raise ISFError("no void main() to hoist the file-scope initializers into")
    edits.append((m.end(), m.end(), "\n" + "\n".join(assigns)))
    out = body
    for a, b, text in sorted(edits, reverse=True):
        out = out[:a] + text + out[b:]
    return out, names


def fetch_estimate(body: str) -> int:
    """Texture fetches one fragment makes: every IMG_*/texture2D call site times
    the trip count of each literal-bounded for loop around it. A loop bounded by
    a uniform counts once, so this is a floor, not a bound."""
    mask = _mask(body)
    total, mult = 0, 1
    stack: list[tuple[int, int]] = []
    depth = i = 0
    while i < len(mask):
        c = mask[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            while stack and stack[-1][0] > depth:
                mult //= stack.pop()[1]
        elif (m := _FOR.match(mask, i)) is not None:
            close = _match_paren(mask, m.end() - 1)
            brace = mask.find("{", close)
            if brace >= 0 and not mask[close + 1 : brace].strip():
                trips = _trips(mask[m.end() : close])
                stack.append((depth + 1, trips))
                mult *= trips
            i = close
        elif (m := _FETCH.match(mask, i)) is not None:
            total += mult
            i = m.end() - 1
        i += 1
    return total


def to_scene(
    text: str, name: str = "shader", size=(320, 240), vertex_shader: bool = False
) -> tuple[Scene, list[str]]:
    """Import ISF source as a ghee Scene. Returns the scene and any warnings."""
    scene, _, warnings = _convert(text, name, size, False, vertex_shader)
    return scene, warnings


def to_filter(
    text: str, name: str = "shader", size=(320, 240), vertex_shader: bool = False
) -> tuple[dict, list[str]]:
    """Import a single-pass ISF filter as a ghee filter definition (the dict a
    `<name>.filter.json` holds). Its one image input becomes iChannel0, which is
    what ghee hands a filter: the frame so far."""
    _, flt, warnings = _convert(text, name, size, True, vertex_shader)
    return flt, warnings


def slug(name: str) -> str:
    """A file/pass/filter name from an ISF title: 'Radial Spectrogram.fs' ->
    'radial_spectrogram'."""
    stem = os.path.splitext(os.path.basename(name))[0]
    out = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return out or "isf"


@dataclass
class _Input:
    name: str
    type: str
    param: Param | None = None
    size: tuple[int, int] | None = None  # audio inputs: the array they become


@dataclass
class _Sized:
    """Everything the shim has to name: samplers and the size each one is."""

    channels: list[tuple[str, str | None]] = field(default_factory=list)
    arrays: list[str] = field(default_factory=list)
    sizes: dict[str, tuple[int, int]] = field(default_factory=dict)


def _convert(text, src, size, as_filter, vertex_shader):
    try:
        return _build(text, src, size, as_filter, vertex_shader)
    except ISFError as e:
        raise ISFError(f"{src}: {e}") from None


def _build(text, src, size, as_filter, vertex_shader):
    meta, body = split_header(text)
    _refuse_unrunnable(meta, body, vertex_shader)
    warn: list[str] = []
    size = (int(size[0]), int(size[1]))

    inputs = _inputs(meta)
    pass_meta = _passes(meta)
    body, renames = _rename_reserved(body, inputs, pass_meta)
    warn += [f"'{a}' is a ghee preamble name, renamed to '{b}'" for a, b in renames]

    names, targets = _pass_names(pass_meta, len(pass_meta) or 1)
    resolved, pass_sizes = _pass_sizes(pass_meta, size, inputs, len(names), warn)

    images = [i for i in inputs if i.type == "image"]
    arrays = [i for i in inputs if i.size is not None]
    params = {i.name: i.param for i in inputs if i.param is not None}
    code = _mask(body, directives=False)
    refs = [t for t in targets if t and re.search(rf"\b{t}\b", code)]

    if as_filter:
        _refuse_unfilterable(pass_meta, images, arrays, refs)

    sized = _Sized(arrays=[a.name for a in arrays])
    for i in images:
        sized.channels.append((i.name, None if as_filter else SOURCE_PASS))
        sized.sizes[i.name] = size
    for t in refs:
        sized.channels.append((t, t))
        sized.sizes[t] = resolved[names.index(t)]
    for a in arrays:
        sized.sizes[a.name] = a.size

    body = _rewrite_img(body, set(sized.sizes))
    body, hoisted = hoist_globals(body)
    if hoisted:
        warn.append(
            f"hoisted file-scope initializer(s) into main: {', '.join(hoisted)}"
        )

    units = len(sized.channels) + len(sized.arrays)
    if units > _VC4_UNITS:
        warn.append(
            f"binds {units} textures; VC4-class GPUs (Zero..Pi 4) have {_VC4_UNITS}"
        )
    fetches = fetch_estimate(body)
    if fetches > _FETCH_BUDGET:
        warn.append(
            f"about {fetches} texture fetches per fragment; VC4 misses 30fps "
            f"past roughly 100"
        )
    if re.search(r"\bDATE\b", body):
        warn.append("DATE is a fixed vec4(0, 0, 0, time); ghee has no calendar clock")

    title = slug(src)
    if as_filter:
        flt = {
            "name": title,
            "code": _shim(src, 0, sized) + "\n" + body,
            "params": {n: p.to_dict() for n, p in params.items()},
        }
        desc = _description(meta)
        if desc:
            flt["description"] = desc
        return None, flt, warn

    passes = []
    if images:
        passes.append(Pass(name=SOURCE_PASS, code=_SOURCE_CODE))
        warn.append(
            f"image input(s) {', '.join(i.name for i in images)} have no producer; "
            f"bound to a generated '{SOURCE_PASS}' test pattern pass"
        )
    for i, nm in enumerate(names):
        passes.append(
            Pass(
                name=nm,
                code=_shim(src, i, sized) + "\n" + body,
                arrays=[ArrayInput(a.name, a.size[0]) for a in arrays],
                channels=[Channel(buffer=b) for _, b in sized.channels],
                size=pass_sizes[i],
            )
        )
    return Scene(passes=passes, params=params), None, warn


def _refuse_unrunnable(meta, body, vertex_shader):
    if vertex_shader:
        raise ISFError(
            "ships a custom ISF vertex shader (.vs); ghee's vertex shader is "
            "fixed and emits v_texCoord only, so this cannot link"
        )
    m = _VARYING.search(body)
    if m is not None:
        raise ISFError(
            f"reads the custom varying '{m.group(1)}', which only an ISF vertex "
            "shader can write; ghee's vertex shader emits v_texCoord only"
        )
    imported = meta.get("IMPORTED")
    if imported:
        which = ", ".join(imported) if isinstance(imported, dict) else str(imported)
        raise ISFError(
            f"IMPORTED loads external image(s) ({which}); ghee scenes have no "
            "image inputs, only pass buffers and host-fed arrays"
        )
    floats = [
        p.get("TARGET") or f"pass {i}"
        for i, p in enumerate(meta.get("PASSES") or [])
        if isinstance(p, dict) and p.get("FLOAT")
    ]
    if floats:
        raise ISFError(
            f"buffer(s) {', '.join(floats)} are FLOAT; ghee's pass buffers are "
            "8-bit RGB, so an accumulator saturates within a few seconds and "
            "the scene freezes without ever reporting an error"
        )


def _refuse_unfilterable(pass_meta, images, arrays, refs):
    tail = "Import it as a scene instead"
    if len(pass_meta) > 1:
        raise ISFError(
            f"has {len(pass_meta)} passes; a ghee filter is one pass. {tail}"
        )
    if len(images) != 1:
        what = "no image input" if not images else f"{len(images)} image inputs"
        raise ISFError(
            f"has {what}; a ghee filter reads exactly one, iChannel0 (the frame "
            f"so far). {tail}"
        )
    if refs:
        raise ISFError(
            f"reads its own buffer ({', '.join(refs)}) for feedback; a ghee "
            f"filter binds iChannel0 only. {tail}"
        )
    if arrays:
        raise ISFError(
            f"reads audio ({', '.join(a.name for a in arrays)}); a ghee filter "
            f"declares no arrays. {tail}"
        )


def _inputs(meta) -> list[_Input]:
    raw = meta.get("INPUTS") or []
    if not isinstance(raw, list):
        raise ISFError(f"INPUTS needs a list, got {raw!r}")
    out = []
    for d in raw:
        if not isinstance(d, dict):
            raise ISFError(f"an INPUTS entry needs an object, got {d!r}")
        name, typ = d.get("NAME"), d.get("TYPE")
        if not isinstance(name, str) or not _IDENT.match(name):
            raise ISFError(f"input name {name!r} is not a GLSL identifier")
        inp = _Input(name=name, type=typ)
        if typ in ("audio", "audioFFT"):
            n = d.get("MAX")
            n = int(n) if isinstance(n, (int, float)) and n else _AUDIO_BINS
            inp.size = (n, 1)
        elif typ != "image":
            inp.param = _param(name, typ, d)
        out.append(inp)
    return out


def _param(name, typ, d) -> Param:
    """One INPUTS entry as a ghee Param. ISF colors are RGBA, so they become a
    vec4 rather than ghee's three-component color; a long is an int over the
    range its VALUES span (ghee has no labelled int dropdown)."""
    default = d.get("DEFAULT")
    if typ == "float":
        return Param(
            name, "float", _num(default, d.get("MIN"), 0.0), d.get("MIN"), d.get("MAX")
        )
    if typ in ("bool", "event"):
        return Param(name, "bool", bool(default))
    if typ == "long":
        values = [v for v in (d.get("VALUES") or []) if isinstance(v, (int, float))]
        lo, hi = (min(values), max(values)) if values else (None, None)
        step = 1 if values else None
        return Param(name, "int", int(_num(default, lo, 0)), lo, hi, step)
    if typ == "color":
        return Param(name, "vec4", _vec(default, 4, [0.0, 0.0, 0.0, 1.0]))
    if typ == "point2D":
        return Param(name, "vec2", _vec(default, 2, [0.0, 0.0]))
    raise ISFError(f"input '{name}' has unsupported TYPE {typ!r}")


def _num(*candidates):
    for c in candidates:
        if isinstance(c, (int, float)) and not isinstance(c, bool):
            return float(c)
    return 0.0


def _vec(value, n, fallback):
    if not isinstance(value, (list, tuple)):
        return fallback
    v = [float(x) if isinstance(x, (int, float)) else 0.0 for x in value]
    return (v + fallback[len(v) :])[:n]


def _passes(meta) -> list[dict]:
    raw = meta.get("PASSES") or []
    if not isinstance(raw, list) or not all(isinstance(p, dict) for p in raw):
        raise ISFError(f"PASSES needs a list of objects, got {raw!r}")
    return raw


def _rename_reserved(body, inputs, pass_meta):
    """An ISF name that is also a ghee preamble name (a target called `time`)
    would shadow the uniform the shim maps TIME onto. Rename it, and every
    reference the body makes to it, before anything else reads the body."""
    taken = [i.name for i in inputs] + [
        p["TARGET"] for p in pass_meta if isinstance(p.get("TARGET"), str)
    ]
    renames = []
    for old in taken:
        if old in _RESERVED or _CHANNEL.match(old):
            new = "isf_" + old
            body = re.sub(rf"\b{re.escape(old)}\b", new, body)
            renames.append((old, new))
            for i in inputs:
                if i.name == old:
                    i.name = new
            for p in pass_meta:
                if p.get("TARGET") == old:
                    p["TARGET"] = new
    return body, renames


def _pass_names(pass_meta, count):
    names, targets, used = [], [], set()
    for i in range(count):
        target = pass_meta[i].get("TARGET") if i < len(pass_meta) else None
        if target is not None and not (
            isinstance(target, str) and _IDENT.match(target)
        ):
            raise ISFError(f"PASSES[{i}] TARGET {target!r} is not a GLSL identifier")
        name = target or ("out" if i == count - 1 else f"pass{i}")
        while name in used:
            name += "_"
        used.add(name)
        names.append(name)
        targets.append(target)
    return names, targets


def _pass_sizes(pass_meta, size, inputs, count, warn):
    """Each pass's WIDTH/HEIGHT equations evaluated against the target size.
    Returns the resolved sizes (what the shim reports as x_isf_size) and the
    sizes the scene declares, where None means the renderer's own size."""
    values = {"WIDTH": float(size[0]), "HEIGHT": float(size[1])}
    for i in inputs:
        if i.param is not None and i.param.type in ("float", "int"):
            values[i.name] = float(i.param.value)
        elif i.param is not None and i.param.type == "bool":
            values[i.name] = 1.0 if i.param.value else 0.0
    resolved, declared = [], []
    for i in range(count):
        p = pass_meta[i] if i < len(pass_meta) else {}
        w = eval_size(p["WIDTH"], values) if "WIDTH" in p else size[0]
        h = eval_size(p["HEIGHT"], values) if "HEIGHT" in p else size[1]
        resolved.append((w, h))
        declared.append(None if (w, h) == size else (w, h))
    if declared and declared[-1] is not None:
        warn.append(
            f"last pass asks for {resolved[-1][0]}x{resolved[-1][1]}; ghee's "
            f"output pass renders at the render size, so it takes {size[0]}x{size[1]}"
        )
        resolved[-1] = size
        declared[-1] = None
    return resolved, declared


def _rewrite_img(body, samplers):
    """IMG_SIZE(x) -> the constant x_isf_size, and IMG_PIXEL(x, p) -> the same
    call with x's size handed in front, since a sampler2D argument carries no
    size of its own."""
    unknown = set()

    def size_of(m):
        if m.group(1) not in samplers:
            unknown.add(m.group(1))
        return m.group(1) + "_isf_size"

    body = _sub_code(r"\bIMG_SIZE\s*\(\s*([A-Za-z_]\w*)\s*\)", size_of, body)
    body = _sub_code(
        r"\bIMG_PIXEL\s*\(\s*([A-Za-z_]\w*)\s*,",
        lambda m: f"IMG_PIXEL({size_of(m)}, {m.group(1)},",
        body,
    )
    if unknown:
        raise ISFError(
            f"IMG_PIXEL/IMG_SIZE reads {', '.join(sorted(unknown))}, which is "
            "not an image input or a pass target, so its size is unknown"
        )
    return body


def _sub_code(pattern, repl, body):
    """re.sub over the code only: the match runs against the body with its
    comments blanked, so a call left behind in a commented-out line is neither
    rewritten nor reported."""
    out = body
    for m in reversed(list(re.finditer(pattern, _mask(body, directives=False)))):
        out = out[: m.start()] + repl(m) + out[m.end() :]
    return out


def _shim(src, index, sized: _Sized) -> str:
    L = [
        f"// ISF shim for '{src}', pass {index}",
        "#define RENDERSIZE resolution",
        "#define TIME time",
        "#define TIMEDELTA (1.0 / 60.0)",
        "#define FRAMEINDEX int(time * 60.0)",
        "#define DATE vec4(0.0, 0.0, 0.0, time)",
        "#define isf_FragNormCoord v_texCoord",
        "#define vv_FragNormCoord v_texCoord",
        f"const int PASSINDEX = {index};",
        *_IMG_ACCESSORS,
    ]
    for i, (name, _) in enumerate(sized.channels):
        L.append(f"#define {name} iChannel{i}")
    for name in sized.arrays:
        L.append(f"#define {name} {name}_tex")
    # ISF also exposes each image's placement in its texture. Nothing ghee binds
    # is atlased or flipped, so the rect is the whole buffer.
    for name, (w, h) in sized.sizes.items():
        L += [
            f"const vec2 {name}_isf_size = vec2({float(w)}, {float(h)});",
            f"const vec2 _{name}_imgSize = vec2({float(w)}, {float(h)});",
            f"const vec4 _{name}_imgRect = vec4(0.0, 0.0, {float(w)}, {float(h)});",
            f"const bool _{name}_flip = false;",
        ]
    return "\n".join(L) + "\n"


def _description(meta) -> str:
    bits = [str(meta.get(k, "")).strip() for k in ("DESCRIPTION", "CREDIT")]
    return " ".join(b for b in bits if b)


def _mask(src: str, directives: bool = True) -> str:
    """src with every comment blanked (and every preprocessor directive too,
    unless `directives` says otherwise), so a scan sees code only and every
    index still lines up with the original."""
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        two = src[i : i + 2]
        if two == "//":
            while i < n and src[i] != "\n":
                out[i] = " "
                i += 1
        elif two == "/*":
            end = src.find("*/", i + 2)
            end = n if end < 0 else end + 2
            for k in range(i, end):
                if src[k] != "\n":
                    out[k] = " "
            i = end
        elif directives and src[i] == "#" and _line_start(src, i):
            more = True
            while more and i < n:
                eol = src.find("\n", i)
                eol = n if eol < 0 else eol
                more = src[i:eol].rstrip().endswith("\\")
                for k in range(i, eol):
                    out[k] = " "
                i = eol + 1
        else:
            i += 1
    return "".join(out)


def _line_start(src, i):
    return not src[src.rfind("\n", 0, i) + 1 : i].strip()


def _split_decl(stmt: str):
    """A file-scope `<type> a = x, b;` as (declaration, [(name, initializer)]),
    or None when the statement declares no initialized variable (a prototype, a
    const, a uniform, a varying, an interface block)."""
    s = stmt.strip()
    m = re.match(r"^((?:lowp|mediump|highp)\s+)?([A-Za-z_]\w*)\s+(.*)$", s, re.S)
    if m is None or m.group(2) not in _GLSL_TYPES:
        return None
    decls, sets = [], []
    for part in _split_commas(m.group(3)):
        lhs, eq, rhs = _split_assign(part)
        if not eq:
            decls.append(part.strip())
            continue
        if "[" in lhs:  # an array initializer is not ESSL 1.00 to begin with
            return None
        decls.append(lhs.strip())
        sets.append((lhs.strip(), rhs.strip()))
    if not sets:
        return None
    return f"{m.group(1) or ''}{m.group(2)} {', '.join(decls)};", sets


def _split_commas(text):
    parts, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _split_assign(part):
    depth = 0
    for i, c in enumerate(part):
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "=" and depth == 0 and part[i : i + 2] != "==":
            return part[:i], True, part[i + 1 :]
    return part, False, ""


def _match_paren(src, i):
    depth = 0
    for k in range(i, len(src)):
        if src[k] == "(":
            depth += 1
        elif src[k] == ")":
            depth -= 1
            if depth == 0:
                return k
    return len(src) - 1


def _trips(header):
    m = _LIMIT.search(header)
    return int(m.group(1)) if m else 1


class _Expr:
    """Recursive descent over an ISF size equation. The whole vocabulary is
    $names, numbers, + - * /, and floor/ceil/abs/min/max."""

    _FN = {
        "floor": lambda a: float(math.floor(a)),
        "ceil": lambda a: float(math.ceil(a)),
        "abs": abs,
        "min": min,
        "max": max,
    }

    def __init__(self, text, values):
        self.text, self.values = text, values
        self.tok = [
            t
            for t in re.findall(r"\$?[A-Za-z_]\w*|\d*\.\d+|\d+\.?|[-+*/(),]|\S", text)
            if t.strip()
        ]
        self.i = 0

    def parse(self):
        v = self._sum()
        if self.i < len(self.tok):
            raise ISFError(
                f"size equation {self.text!r}: unexpected {self.tok[self.i]!r}"
            )
        return v

    def _peek(self):
        return self.tok[self.i] if self.i < len(self.tok) else None

    def _take(self, what=None):
        t = self._peek()
        if t is None or (what is not None and t != what):
            raise ISFError(f"size equation {self.text!r}: expected {what or 'a value'}")
        self.i += 1
        return t

    def _sum(self):
        v = self._product()
        while self._peek() in ("+", "-"):
            v = v + self._product() if self._take() == "+" else v - self._product()
        return v

    def _product(self):
        v = self._unary()
        while self._peek() in ("*", "/"):
            if self._take() == "*":
                v *= self._unary()
            else:
                d = self._unary()
                if d == 0:
                    raise ISFError(f"size equation {self.text!r}: divide by zero")
                v /= d
        return v

    def _unary(self):
        if self._peek() in ("+", "-"):
            return -self._unary() if self._take() == "-" else self._unary()
        return self._atom()

    def _atom(self):
        t = self._take()
        if t == "(":
            v = self._sum()
            self._take(")")
            return v
        if t.startswith("$"):
            name = t[1:]
            if name not in self.values:
                raise ISFError(
                    f"size equation {self.text!r} reads ${name}, which is not "
                    "WIDTH, HEIGHT or a numeric input"
                )
            return self.values[name]
        if t in self._FN:
            self._take("(")
            args = [self._sum()]
            while self._peek() == ",":
                self._take(",")
                args.append(self._sum())
            self._take(")")
            return float(self._FN[t](*args))
        try:
            return float(t)
        except ValueError:
            raise ISFError(
                f"size equation {self.text!r}: {t!r} is not a value"
            ) from None
