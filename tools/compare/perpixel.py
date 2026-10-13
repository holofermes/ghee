"""Per-pixel programs on the glass, each beside a ghee version of its picture.

- chroma-luma: luma.examples' chroma.py as written, through luma.core's
  linux_framebuffer device with `--framebuffer full_frame --bgr`. Every
  sample of every frame is Python math at half the panel's size, drawn as a
  2x2 Pillow rectangle, and the device converts the frame and writes it to
  the fbdev one row at a time.
- chroma-ghee: chroma.glsl, the same four effects and crossfades in one
  fragment shader, scanned out to the panel's card.
- liquid-pygame: pygame's liquid.py ported to the panel's width: the block
  size, the amplitude, the shift and the wavelength scaled by width / 640,
  the bitmap made a PNG with the fewest scale2x passes that cover the reach
  of the blocks, and each frame written whole to the fbdev.
- liquid-ghee: liquid.glsl, the same ripple over the same PNG.
- perfloop-luma: luma.examples' perfloop.py, one picture sent again and
  again through the same device, luma's floor.

The examples come from their repositories at pinned commits on the first
run, into --sources. A luma program runs its own loop, so the runner's
phases sit in the calls that end its frames: chroma.py's `time.sleep(0.01)`
and perfloop.py's frame regulator. Paced, that call waits out the rest of
the frame's 1/30 s, and free, it returns at once, so chroma.py's own 10 ms
sleep is left out.

The DRM fbdev helper before Linux commit c28c22c8cfbd ("drm/fb-helper: Fix
clipping when damage area spans a single scanline") gives a write of exactly
one whole line an empty damage rect, and flushes nothing to the panel, so
on those kernels the device's row writes put nothing on the glass. The
runner gathers them into one write of the frame, timed from its start.

The check draws both sides of each pair at the same steps headless and
compares them, in RGB and in the levels that the panel's driver keeps:
16 for a grey panel, 2 for a mono one, from the kernel's BT.601 grey.
chroma.py's frames are kept as PNGs in the check's directory and taken from
there when present, since reaching a late step runs every frame before it.
"""

import colorsys
import fcntl
import json
import os
import select
import sys
import time
import urllib.request
from math import sin
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SOURCES = [
    (
        "rm-hull/luma.examples",
        "cfb170e8a81d05a61d929613dd57c5e5f557a981",
        [
            "examples/chroma.py",
            "examples/demo_opts.py",
            "examples/demo.py",
            "examples/perfloop.py",
        ],
    ),
    (
        "pygame/pygame",
        "85fda3f719d437cf27106afae8c890e6b88ba5f5",
        ["examples/data/liquid.bmp"],
    ),
]
PROGRAMS = [
    "chroma-luma",
    "chroma-ghee",
    "liquid-pygame",
    "liquid-ghee",
    "perfloop-luma",
]
QUEUE = {"chroma-ghee", "liquid-ghee"}
WARM = 10
FBIOBLANK = 0x4611


def default_sources():
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return str(Path(cache, "ghee-compare"))


def fetch(sources):
    """Every source file, downloaded once, by repository name."""
    for repo, sha, paths in SOURCES:
        for path in paths:
            out = Path(sources, repo.split("/")[1], Path(path).name)
            if out.exists():
                continue
            url = f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)


def liquid_geometry(w):
    """liquid.py's block size, amplitude, shift and wavelength, at 640
    wide, scaled to w."""
    return round(20 * w / 640), 15 * w / 640, 20 * w / 640, 0.01 * 640 / w


def liquid_png(sources, w, h):
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    out = Path(sources, f"liquid-{w}x{h}.png")
    if out.exists():
        return out
    _, amp, shift, _ = liquid_geometry(w)
    bitmap = pygame.image.load(str(Path(sources, "pygame", "liquid.bmp")))
    while bitmap.get_width() < w + amp + shift or bitmap.get_height() < h + amp + shift:
        bitmap = pygame.transform.scale2x(bitmap)
    pygame.image.save(bitmap, str(out))
    return out


def chroma_palette(sources):
    """chroma.py's colours as it floors them, one row for each step mod 255:
    the tunnel's bright and dark stripe, each without and with its ring, then
    the checker's half and full square."""
    from PIL import Image

    out = Path(sources, "chroma-palette.png")
    if out.exists():
        return out

    def floored(rgb, add=0.0):
        return tuple(int(max(0, min(255, (c + add) * 1 * 255))) for c in rgb)

    rows = []
    for k in range(255):
        h = k / 255.0
        bright, dark = colorsys.hsv_to_rgb(h, 1, 0.8), colorsys.hsv_to_rgb(h, 1, 0.3)
        rows += [floored(bright), floored(bright, 0.3), floored(dark)]
        rows += [floored(dark, 0.3)]
        rows += [floored(colorsys.hsv_to_rgb(h, 1, v)) for v in (0.5, 1)]
    img = Image.new("RGB", (6, 255))
    img.putdata(rows)
    img.save(out)
    return out


def chroma_scene(sources):
    return {
        "version": 1,
        "assets": {"palette": {"file": str(chroma_palette(sources))}},
        "common": {"file": str(HERE / "trig.glsl")},
        "passes": [
            {
                "name": "main",
                "fragment": {"file": str(HERE / "chroma.glsl")},
                "params": {"first": {"type": "float", "value": 0.0}},
            }
        ],
    }


def liquid_scene(png, w):
    block, amp, shift, freq = liquid_geometry(w)
    params = {"first": 0.0, "blocksize": block, "amplitude": amp}
    params |= {"shift": shift, "freq": freq}
    return {
        "version": 1,
        "assets": {"liquid": {"file": str(png)}},
        "common": {"file": str(HERE / "trig.glsl")},
        "passes": [
            {
                "name": "main",
                "fragment": {"file": str(HERE / "liquid.glsl")},
                "params": {
                    k: {"type": "float", "value": float(v)} for k, v in params.items()
                },
            }
        ],
    }


class Control:
    """The runner's phases, run from the call that ends each frame of a
    program: the warm-up, then the frame held until the runner says paced or
    free, each phase's frames reported when it ends.

    begin() marks the start of a frame's draw and tick() its end, and a
    writer sets `sent` to the start of its frame's last write."""

    def __init__(self, say, pace, drain):
        self.say, self.pace, self.drain = say, pace, drain
        self.mode, self.count, self.warm = "warm", 0, []
        self.frames, self.sent, self.first, self.start = [], None, None, None

    def begin(self):
        self.start = time.monotonic()
        if self.first is None:
            self.first = self.start

    def tick(self, *_):
        now = time.monotonic()
        frame = (self.start, self.sent or self.start, now)
        self.sent = None
        if self.mode == "warm":
            if self.count:
                self.warm.append(now - self.start)
            self.count += 1
            if self.count == WARM:
                time.sleep(self.drain)
                self.say({"first": self.first, "warm": max(self.warm)})
                self.command()
        elif self.mode == "paced":
            self.frames.append(frame)
            self.due += 1 / self.pace
            if select.select([0], [], [], max(0.0, self.due - time.monotonic()))[0]:
                os.read(0, 64)
                self.report()
        else:
            self.frames.append(frame)
            if now >= self.end:
                self.report()
        self.start = time.monotonic()

    def report(self):
        time.sleep(self.drain)
        self.say({"frames": self.frames})
        self.frames = []
        self.command()

    def command(self):
        cmd = os.read(0, 64).decode().split()
        if not cmd or cmd[0] == "quit":
            raise SystemExit(0)
        self.mode = cmd[0]
        self.due = time.monotonic()
        if self.mode == "free":
            self.end = self.due + float(cmd[1])


class Gathered:
    """A luma device's framebuffer file, its writes gathered into one write
    of the rows that they cover when the device flushes."""

    def __init__(self, f, ctl):
        self.fd, self.ctl, self.close = f.fileno(), ctl, f.close
        self.buf, self.pos, self.lo, self.hi = bytearray(), 0, None, 0

    def seek(self, pos, whence=0):
        self.pos = pos

    def write(self, data):
        end = self.pos + len(data)
        if len(self.buf) < end:
            self.buf.extend(bytes(end - len(self.buf)))
        self.buf[self.pos : end] = data
        self.lo = self.pos if self.lo is None else min(self.lo, self.pos)
        self.hi, self.pos = max(self.hi, end), end
        return len(data)

    def flush(self):
        if self.lo is None:
            return
        self.ctl.sent = time.monotonic()
        os.pwrite(self.fd, memoryview(self.buf)[self.lo : self.hi], self.lo)
        self.lo, self.hi = None, 0


class Regulator:
    """perfloop.py's frame regulator, the frame's begin and end."""

    def __init__(self, ctl):
        self.ctl, self.called = ctl, 0

    def __enter__(self):
        self.ctl.begin()
        return self

    def __exit__(self, kind, *_):
        self.called += 1
        if kind is None:
            self.ctl.tick()
        return False

    def effective_FPS(self):
        return 0.0

    def average_transit_time(self):
        return 0.0


def unblank(fb):
    fd = os.open(fb, os.O_WRONLY)
    try:
        fcntl.ioctl(fd, FBIOBLANK, 0)
    finally:
        os.close(fd)


def luma_argv(name, fb):
    return [name, "--display", "linux_framebuffer", "--framebuffer-device", fb]


def run_luma(name, panel, sources, ctl):
    sys.path.insert(0, str(Path(sources, "luma.examples")))
    sys.argv = luma_argv(name, panel.fb) + ["--framebuffer", "full_frame", "--bgr"]
    unblank(panel.fb)
    import demo_opts

    def device():
        d = demo_opts.get_device()
        f = d._linux_framebuffer__file_handle
        d._linux_framebuffer__file_handle = Gathered(f, ctl)
        return d

    if name == "chroma-luma":
        import chroma

        chroma.device = device()
        chroma.time = SimpleNamespace(sleep=ctl.tick)
        ctl.begin()
        chroma.main()
    else:
        import perfloop

        perfloop.get_device = device
        perfloop.time = SimpleNamespace(sleep=lambda _: None)
        perfloop.framerate_regulator = lambda fps: Regulator(ctl)
        perfloop.main()


class Liquid:
    """liquid.py's loop at the panel's size, on SDL's dummy driver."""

    def __init__(self, w, h, png):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame as pg

        self.pg = pg
        pg.display.init()
        self.screen = pg.display.set_mode((w, h), depth=32)
        self.bitmap = pg.image.load(str(png)).convert()
        self.block, self.amp, self.shift, self.freq = liquid_geometry(w)
        self.xblocks, self.yblocks = range(0, w, self.block), range(0, h, self.block)
        self.anim = 0.0

    def draw(self):
        screen, bitmap, b = self.screen, self.bitmap, self.block
        amp, shift, freq = self.amp, self.shift, self.freq
        for _ in self.pg.event.get():
            pass
        self.anim = self.anim + 0.02
        for x in self.xblocks:
            xpos = (x + (sin(self.anim + x * freq) * amp)) + shift
            for y in self.yblocks:
                ypos = (y + (sin(self.anim + y * freq) * amp)) + shift
                screen.blit(bitmap, (x, y), (xpos, ypos, b, b))


def run_liquid(panel, sources, ctl):
    from fbdev import Framebuffer

    png = Path(sources, f"liquid-{panel.w}x{panel.h}.png")
    liquid = Liquid(panel.w, panel.h, png)
    fb = Framebuffer(panel.fb, panel.w, panel.h)
    ctl.begin()
    while True:
        liquid.draw()
        fb.write(liquid.screen.get_buffer())
        ctl.sent = fb.sent
        ctl.tick()


def run_ghee(name, panel, sources, ctl):
    import ghee

    if name == "chroma-ghee":
        scene = chroma_scene(sources)
    else:
        scene = liquid_scene(Path(sources, f"liquid-{panel.w}x{panel.h}.png"), panel.w)
    h = ghee.open(scene, display="kms", dri=panel.card)
    ctl.begin()
    while True:
        h.frame()
        ctl.tick()


def child(name, card, fb, sources, pace, drain):
    """One program on the panel, its frames reported to the runner on
    stdout. The program's own output goes to stderr."""
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)

    def say(msg):
        print(json.dumps(msg), file=out, flush=True)

    w, h = (
        int(v)
        for v in Path("/sys/class/graphics", Path(fb).name, "virtual_size")
        .read_text()
        .split(",")
    )
    panel = SimpleNamespace(card=card, fb=fb, w=w, h=h)
    ctl = Control(say, pace, drain)
    if name.endswith("-luma"):
        run_luma(name, panel, sources, ctl)
    elif name == "liquid-pygame":
        run_liquid(panel, sources, ctl)
    else:
        run_ghee(name, panel, sources, ctl)


class Done(Exception):
    pass


def luma_frames(sources, w, h, steps):
    """chroma.py's frames at the steps, from luma.core's dummy device."""
    sys.path.insert(0, str(Path(sources, "luma.examples")))
    import chroma
    from luma.core.device import dummy

    chroma.device = dummy(width=w, height=h, mode="RGB")
    got, step = {}, 0

    def sleep(_):
        nonlocal step
        if step in steps:
            got[step] = chroma.device.image.copy()
        step += 1
        if len(got) == len(steps):
            raise Done

    chroma.time = SimpleNamespace(sleep=sleep)
    try:
        chroma.main()
    except Done:
        pass
    return got


def references(out, sources, w, h, steps):
    """chroma.py's frames at the steps, kept in out as PNGs, so a slow
    machine can take them from a fast one."""
    from PIL import Image

    paths = {s: Path(out, f"luma-chroma-{w}x{h}-{s:05d}.png") for s in steps}
    if not all(p.exists() for p in paths.values()):
        for s, img in luma_frames(sources, w, h, set(steps)).items():
            img.save(paths[s])
    return {s: Image.open(p).convert("RGB") for s, p in paths.items()}


def pygame_frames(png, w, h, steps):
    from PIL import Image

    liquid = Liquid(w, h, png)
    got = {}
    for step in range(max(steps) + 1):
        liquid.draw()
        if step in steps:
            raw = liquid.pg.image.tobytes(liquid.screen, "RGB")
            got[step] = Image.frombytes("RGB", (w, h), raw)
    return got


def ghee_frames(scene, w, h, steps):
    import ghee

    h_ = ghee.open(scene, width=w, height=h)
    got = {}
    for n, step in enumerate(sorted(steps)):
        h_.set_param("first", float(step - n))
        h_.frame()
        got[step] = h_.image().convert("RGB")
    h_.close()
    return got


def pixels(img):
    data = img.tobytes()
    return list(zip(data[0::3], data[1::3], data[2::3]))


def levels(img, n):
    """The panel's level of each pixel: the kernel's BT.601 grey, cut to n
    levels, 16 or 2."""
    out = []
    for r, g, b in pixels(img):
        grey = (77 * r + 150 * g + 29 * b) // 256
        out.append(grey >> 4 if n == 16 else int(grey >= 128))
    return out


def side_by_side(a, b, la, lb, n, w, h, scale=3):
    from PIL import Image

    def show(lv):
        img = Image.new("L", (w, h))
        img.putdata([v * 255 // (n - 1) for v in lv])
        return img.convert("RGB")

    diff = Image.new("RGB", (w, h))
    diff.putdata([(255, 0, 0) if x != y else (0, 0, 0) for x, y in zip(la, lb)])
    gap = 4
    sheet = Image.new("RGB", (3 * w + 2 * gap, 2 * h + gap), (40, 40, 40))
    sheet.paste(a, (0, 0))
    sheet.paste(b, (w + gap, 0))
    sheet.paste(show(la), (0, h + gap))
    sheet.paste(show(lb), (w + gap, h + gap))
    sheet.paste(diff, (2 * (w + gap), h + gap))
    return sheet.resize((sheet.width * scale, sheet.height * scale), Image.NEAREST)


def check(out, sources, w, h, n, steps, pairs):
    """Draw each pair at the steps, save one sheet per step, and give the
    pixels that differ in RGB and in the panel's levels."""
    os.makedirs(out, exist_ok=True)
    rows = []
    for pair in pairs:
        if pair == "chroma":
            a = references(out, sources, w, h, steps)
            b = ghee_frames(chroma_scene(sources), w, h, steps)
        else:
            png = liquid_png(sources, w, h)
            a = pygame_frames(png, w, h, set(steps))
            b = ghee_frames(liquid_scene(png, w), w, h, steps)
        for step in steps:
            pa, pb = pixels(a[step]), pixels(b[step])
            rgb = sum(1 for x, y in zip(pa, pb) if x != y)
            most = max(abs(c - d) for x, y in zip(pa, pb) for c, d in zip(x, y))
            la, lb = levels(a[step], n), levels(b[step], n)
            lv = sum(1 for x, y in zip(la, lb) if x != y)
            sheet = side_by_side(a[step], b[step], la, lb, n, w, h)
            path = Path(out, f"{pair}-{w}x{h}-{step:05d}.png")
            sheet.save(path)
            rows.append(
                {"pair": pair, "step": step, "rgb": rgb, "most": most, "levels": lv}
            )
            print(
                f"{pair} step {step}: {rgb} of {w * h} pixels differ in RGB, "
                f"by {most} at most, {lv} in {n} levels  {path}"
            )
    Path(out, f"check-{w}x{h}.json").write_text(json.dumps(rows, indent=1))
