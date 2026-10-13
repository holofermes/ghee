# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render", "pillow", "pygame", "raylib_sdl"]
# ///
"""Time ghee and the Python drawing libraries on one status screen.

Each library draws the stats demo's screen: the time, the load as a share of
the cores with a bar beside it, and a blinking HOT past 90%. The layout is
the demo's 128x32 one, scaled by width / 128 and centred vertically. The
clock and the load change on every frame. The runner starts one process for
each library and size, and runs it in one of two modes.

Into memory, the default: each library draws its frame into memory, ghee
reading it back from the GPU. The process measures:

- the time from the start of the process to its first frame;
- the mean and the 95th percentile frame time over --frames frames, run free;
- the share of one core, paced to 30 fps for --paced seconds;
- its peak RSS.

On a Raspberry Pi 5 the runner also reads, while a library runs paced at 30
fps, the v3d busy share from the GPU's gpu_stats and the board's power from
`vcgencmd pmic_read_adc`: the median sum over the PMIC rails of volts times
amps, beside the same figure idle.

raylib comes from its SDL build, raylib_sdl, which runs headless on SDL's
offscreen driver.

Onto the glass, with --glass CARD: each library puts its frames on the SPI
or I2C panel of that DRM card. ghee scans out to the card, and Pillow and
pygame write each frame whole to the card's fbdev, which flushes it to the
panel at once. The runner times every bus transfer to the panel from the
kernel's tracepoints, through `sudo -n`, and measures for each library:

- the time from the start of the process to the end of its first frame's
  transfer, then 120 frames to draw every glyph once;
- paced to 30 fps: the frames on the glass a second, the latency from the
  start of a frame's draw to the end of the transfer that carries it, mean
  and 95th percentile, and the CPU time of every task on the system over
  --window seconds, less the same figure with the library's first frame held
  on the panel and nothing drawing, taken once the ondemand governor has let
  the clock down after the start, and the library's own process alone;
- on a Pi 5, the board's power over --power-window seconds, held and paced;
- run free for --free seconds: the frames drawn and the frames on the glass
  a second, and the same latency.

    uv run run.py
    uv run run.py --libs ghee,pillow --sizes 128x32 --png out
    uv run run.py --glass /dev/dri/card1
"""

import argparse
import importlib
import json
import os
import resource
import select
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import glass

HERE = Path(__file__).resolve().parent
FONTS = HERE.parents[1] / "ghee" / "examples" / "fonts"
LIBS = ["ghee", "pillow", "pygame", "raylib"]
GLASS_LIBS = ["ghee", "pillow", "pygame"]
SIZES = ["128x32", "256x64", "320x240"]
PACE = 30
DRAIN = 0.2
WARM = 120
SETTLE = 1.0


def layout(w, h, fonts):
    s = w / 128
    top = round((h - 32 * s) / 2)

    def px(v):
        return round(v * s)

    return SimpleNamespace(
        w=w,
        h=h,
        screen="#0d201a",
        phosphor="#3ce94d",
        mint="#a1fee3",
        sans=str(fonts / "DejaVuSans-Latin.ttf"),
        pixel=str(fonts / "PixelOperator8.ttf"),
        clock=(0, top, px(16)),
        load=(0, top + px(20), px(8)),
        hot=(px(100), top, px(12)),
        frame=(px(64), top + px(20), px(64), px(8)),
        border=max(1, round(s)),
        bar=(px(66), top + px(22), px(60), px(4)),
    )


def values(i, t):
    clock = f"{i // 60 % 24:02d}:{i % 60:02d}"
    load = i % 120 / 100
    return clock, load, t * 2 % 1 < 0.5


def say(msg):
    print(json.dumps(msg), flush=True)


def glass_child(screen, args):
    t0 = time.monotonic()
    screen.draw(*values(0, 0.0))
    warm = []
    for i in range(1, WARM):
        a = time.monotonic()
        screen.draw(*values(i, a - t0))
        warm.append(time.monotonic() - a)
    time.sleep(DRAIN)
    say({"first": t0, "warm": max(warm)})
    i = WARM
    while True:
        cmd = os.read(0, 64).decode().split()
        if not cmd or cmd[0] == "quit":
            return
        frames = []
        due = time.monotonic()
        end = due + float(cmd[1]) if cmd[0] == "free" else None
        fb = getattr(screen, "fb", None)
        while True:
            a = time.monotonic()
            screen.draw(*values(i, a - t0))
            frames.append((a, fb.sent if fb else a, time.monotonic()))
            i += 1
            if end is not None:
                if frames[-1][2] >= end:
                    break
                continue
            due += 1 / PACE
            if select.select([0], [], [], max(0.0, due - time.monotonic()))[0]:
                os.read(0, 64)
                break
        time.sleep(DRAIN)
        say({"frames": frames})


def child(args):
    w, h = map(int, args.size.split("x"))
    lib = importlib.import_module(f"{args.lib}_screen")
    panel = SimpleNamespace(card=args.card, fb=args.fb) if args.card else None
    lay = layout(w, h, Path(args.fonts))
    if panel:
        return glass_child(lib.Screen(lay, panel), args)
    screen = lib.Screen(lay)
    if args.save:
        screen.draw("12:34", 0.95, True)
        screen.save(args.save)
        return
    t0 = time.monotonic()
    screen.draw(*values(0, 0.0))
    out = {"startup": time.monotonic() - args.spawned}
    for i in range(1, 11):
        screen.draw(*values(i, time.monotonic() - t0))
    times = []
    for i in range(11, 11 + args.frames):
        a = time.perf_counter()
        screen.draw(*values(i, time.monotonic() - t0))
        times.append(time.perf_counter() - a)
    times.sort()
    out["mean"] = sum(times) / len(times)
    out["p95"] = times[int(len(times) * 0.95)]
    print(json.dumps({"paced": True}), flush=True)
    n = int(args.paced * PACE)
    start = time.monotonic()
    cpu = time.process_time()
    due = start
    for i in range(n):
        screen.draw(*values(i, time.monotonic() - t0))
        due += 1 / PACE
        wait = due - time.monotonic()
        if wait > 0:
            time.sleep(wait)
    wall = time.monotonic() - start
    out["cpu"] = (time.process_time() - cpu) / wall
    out["paced_fps"] = n / wall
    out["rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print(json.dumps(out), flush=True)


def gpu_stats_path():
    for p in sorted(Path("/sys/class/drm").glob("*/device/gpu_stats")):
        return p
    return None


def gpu_time(path):
    stamp = busy = 0
    for line in path.read_text().splitlines()[1:]:
        _, ts, _, runtime = line.split()
        stamp, busy = int(ts), busy + int(runtime)
    return stamp, busy


def board_watts():
    out = subprocess.run(
        ["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, check=True
    ).stdout
    amps, volts = {}, {}
    for line in out.splitlines():
        name, _, value = line.strip().partition(" ")
        value = float(value.split("=")[1].rstrip("AV"))
        if name.endswith("_A"):
            amps[name[:-2]] = value
        elif name.endswith("_V"):
            volts[name[:-2]] = value
    return sum(a * volts[r] for r, a in amps.items() if r in volts)


def has_pmic():
    try:
        return board_watts() > 0
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return False


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


class Sampler:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        while not self.stop.wait(0.05):
            self.samples.append(board_watts())

    def finish(self):
        self.stop.set()
        self.thread.join()
        return median(self.samples)


def idle_watts(seconds):
    s = Sampler()
    time.sleep(seconds)
    return s.finish()


def measure(lib, size, args, gpu, power):
    cmd = [sys.executable, __file__, "--child", lib, "--size", size]
    cmd += ["--frames", str(args.frames), "--paced", str(args.paced)]
    cmd += ["--fonts", args.fonts, "--spawned", repr(time.monotonic())]
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, text=True)
    result, sampler, g0 = None, None, None
    for line in proc.stdout:
        if not line.startswith("{"):
            continue
        msg = json.loads(line)
        if msg.get("paced"):
            sampler = Sampler() if power else None
            g0 = gpu_time(gpu) if gpu else None
        else:
            if g0:
                g1 = gpu_time(gpu)
                msg["gpu"] = (g1[1] - g0[1]) / (g1[0] - g0[0])
            if sampler:
                msg["watts"] = sampler.finish()
                sampler = None
            result = msg
    if sampler:
        sampler.finish()
    if proc.wait() != 0 or result is None:
        return None
    return result


def measure_glass(lib, p, args, power):
    tracer = glass.Tracer(p.bus)
    tracer.start()
    out, proc = {}, None
    try:
        cmd = [sys.executable, __file__, "--child", lib, "--size", p.size]
        cmd += ["--card", p.card, "--fb", p.fb, "--fonts", args.fonts]
        spawned = time.monotonic()
        proc = subprocess.Popen(
            cmd, cwd=HERE, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )

        def ask(line=None):
            if line:
                proc.stdin.write(line + "\n")
                proc.stdin.flush()
            for reply in proc.stdout:
                if reply.startswith("{"):
                    return json.loads(reply)
            raise RuntimeError(f"{lib} exited with status {proc.wait()}")

        hello = ask()
        first, out["warm"] = hello["first"], hello["warm"]
        time.sleep(glass.settle())
        out["idle_cpu"], _ = glass.cores(args.window, proc.pid)
        if power:
            out["idle_watts"] = watts(args.power_window)
        proc.stdin.write("paced\n")
        proc.stdin.flush()
        time.sleep(SETTLE)
        cpu, out["proc_cpu"] = glass.cores(args.window, proc.pid)
        out["cpu"] = cpu - out["idle_cpu"]
        if power:
            out["watts"] = watts(args.power_window)
        paced = ask("stop")["frames"]
        free = ask(f"free {args.free}")["frames"]
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.wait()
    except (RuntimeError, BrokenPipeError) as e:
        print(f"{lib}: {e}", file=sys.stderr)
        return None
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait()
        transfers, out["lost"] = tracer.stop()
    ups = glass.updates(transfers)
    if args.raw:
        raw = {"spawned": spawned, "first": first, "paced": paced, "free": free}
        raw["transfers"] = transfers
        Path(args.raw, f"{lib}.json").write_text(json.dumps(raw))
    queue = getattr(importlib.import_module(f"{lib}_screen").Screen, "QUEUE", False)
    out["startup"] = next(u[1] for u in ups if u[0] > first) - spawned
    out["free"] = glass.phase(free, ups, queue, free[-1][2] + DRAIN)
    out["paced"] = glass.phase(paced, ups, queue, paced[-1][2] + DRAIN)
    return out


def watts(seconds):
    s = Sampler()
    time.sleep(seconds)
    return s.finish()


def glass_table(rows, power):
    head = ["library", "panel", "start s", "drawn/s", "glass/s", "ms", "p95 ms"]
    head += ["glass/s @30", "ms @30", "p95 @30", "CPU @30", "process @30"]
    if power:
        head += ["W held", "W @30"]
    lines = [head]
    for r in rows:
        m = r.get("m")
        if m is None:
            lines.append([r["lib"], r["size"], "failed"])
            continue
        f, pc = m["free"], m["paced"]
        cells = [r["lib"], r["size"], f"{m['startup']:.2f}", f"{f['drawn']:.0f}"]
        cells += [f"{f['glass']:.0f}", f"{f['latency'] * 1e3:.2f}"]
        cells += [f"{f['latency_p95'] * 1e3:.2f}", f"{pc['glass']:.1f}"]
        cells += [f"{pc['latency'] * 1e3:.2f}", f"{pc['latency_p95'] * 1e3:.2f}"]
        cells += [f"{m['cpu']:.1%}", f"{m['proc_cpu']:.1%}"]
        if power:
            cells += [f"{m['idle_watts']:.2f}", f"{m['watts']:.2f}"]
        lines.append(cells)
        for name, ph in (("free", f), ("paced", pc)):
            if ph["unmatched"]:
                print(f"{r['lib']} {name}: {ph['unmatched']:+d} updates against frames")
        if m["lost"]:
            print(f"{r['lib']}: the trace buffer lost events")
    print_rows(lines)


def print_rows(lines):
    widths = [
        max(len(row[i]) for row in lines if i < len(row)) for i in range(len(lines[0]))
    ]
    for row in lines:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip())


def table(rows, idle):
    head = [
        "library",
        "size",
        "mean ms",
        "p95 ms",
        "fps",
        "CPU @30",
        "RSS MB",
        "start s",
    ]
    if any(r["m"] and "gpu" in r["m"] for r in rows):
        head.append("GPU @30")
    if idle is not None:
        head.append("W @30")
    lines = [head]
    for r in rows:
        m = r.get("m")
        if m is None:
            lines.append([r["lib"], r["size"], "failed"])
            continue
        cells = [r["lib"], r["size"], f"{m['mean'] * 1e3:.2f}", f"{m['p95'] * 1e3:.2f}"]
        cells += [f"{1 / m['mean']:.0f}", f"{m['cpu']:.0%}", f"{m['rss'] / 2**20:.0f}"]
        cells.append(f"{m['startup']:.2f}")
        if "gpu" in m:
            cells.append(f"{m['gpu']:.1%}")
        if idle is not None:
            cells.append(f"{m['watts']:.2f}")
        lines.append(cells)
    print_rows(lines)
    if idle is not None:
        print(f"board power idle: {idle:.2f} W, PMIC rails summed")


def glass_main(args):
    p = glass.panel(args.glass)
    if args.raw:
        os.makedirs(args.raw, exist_ok=True)
    libs = args.libs.split(",") if args.libs else GLASS_LIBS
    if set(libs) - set(GLASS_LIBS):
        raise SystemExit(f"--glass takes {', '.join(GLASS_LIBS)}")
    power = has_pmic()
    rows = [
        {"lib": lib, "size": p.size, "m": measure_glass(lib, p, args, power)}
        for lib in libs
    ]
    if args.json:
        Path(args.json).write_text(
            json.dumps({"panel": p.size, "rows": rows}, indent=1)
        )
    glass_table(rows, power)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--libs")
    ap.add_argument("--sizes", default=",".join(SIZES))
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--paced", type=float, default=2.0, help="seconds at 30 fps")
    ap.add_argument("--fonts", default=str(FONTS))
    ap.add_argument("--png", help="save one frame of each library and size here")
    ap.add_argument("--json", help="write the results here")
    ap.add_argument(
        "--glass", metavar="CARD", help="the DRM card of a SPI or I2C panel"
    )
    ap.add_argument("--free", type=float, default=2.0, help="seconds run free")
    ap.add_argument(
        "--window", type=float, default=4.0, help="seconds of each CPU window"
    )
    ap.add_argument("--power-window", type=float, default=2.0)
    ap.add_argument("--raw", help="write each library's frames and transfers here")
    ap.add_argument("--child", dest="lib", help=argparse.SUPPRESS)
    ap.add_argument("--card", help=argparse.SUPPRESS)
    ap.add_argument("--fb", help=argparse.SUPPRESS)
    ap.add_argument("--size", help=argparse.SUPPRESS)
    ap.add_argument("--spawned", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--save", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.lib:
        return child(args)
    args.fonts = os.path.abspath(args.fonts)
    if args.glass:
        return glass_main(args)
    libs = (args.libs or ",".join(LIBS)).split(",")
    sizes = args.sizes.split(",")
    if args.png:
        os.makedirs(args.png, exist_ok=True)
        for lib in libs:
            for size in sizes:
                path = os.path.abspath(f"{args.png}/{lib}-{size}.png")
                cmd = [sys.executable, __file__, "--child", lib, "--size", size]
                cmd += ["--fonts", args.fonts, "--save", path]
                code = subprocess.run(cmd, cwd=HERE, check=False).returncode
                print(path if code == 0 else f"{path}: failed, exit status {code}")
        return
    gpu = gpu_stats_path()
    power = has_pmic()
    idle = idle_watts(3.0) if power else None
    rows = []
    for size in sizes:
        for lib in libs:
            rows.append(
                {"lib": lib, "size": size, "m": measure(lib, size, args, gpu, power)}
            )
    if args.json:
        Path(args.json).write_text(
            json.dumps({"idle_watts": idle, "rows": rows}, indent=1)
        )
    table(rows, idle)


if __name__ == "__main__":
    main()
