# /// script
# requires-python = ">=3.10"
# dependencies = ["ghee-render", "pillow", "pygame", "raylib_sdl", "luma.core"]
# ///
"""Time ghee and the Python drawing libraries on one status screen.

Each library draws the status screen of docs/python.md: the time, the load
as a share of the cores with a bar beside it, and a blinking HOT past 90%.
The layout is that screen's 128x32 one, scaled by width / 128 and centred
vertically. The clock and the load change on every frame. The runner starts
one process for each library and size, and runs it in one of two modes.

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
or I2C panel of that DRM card. ghee scans out to the card, ghee_drop with
drop=True, and Pillow and pygame write each frame whole to the card's fbdev,
which flushes it to the panel at once. The runner times every bus transfer
to the panel from the kernel's tracepoints, through `sudo -n`, and runs each
library --repeats times, the libraries in turn (ABAB), each run a fresh
process. Around the runs it:

- sets every CPU policy's governor to performance, and puts the old one back
  at the end, on an error or Ctrl-C too;
- before each run, waits until the SoC reads under --cool degrees C
  (`vcgencmd measure_temp`), and gives up after --cool-wait seconds;
- after each run, reads `vcgencmd get_throttled`, and marks the run if a
  flag holds at its end or came up during it. The summary leaves marked runs
  out;
- blanks the panel through its fbdev before each run and after the last.

It measures for each run:

- on a Pi 5, the board's power with no library running and the panel
  blanked, over --power-window seconds;
- the time from the start of the process to the end of the transfer that
  puts its first frame on the panel, then 120 frames to draw every glyph
  once;
- paced to 30 fps: the frames on the glass a second, the latency from the
  start of a frame's draw to the end of the transfer that carries it, mean
  and 95th percentile, and the CPU time of every task on the system over
  --window seconds, less the same figure with the library's last frame held
  on the panel and nothing drawing, and the library's own process alone;
- the busy share of each of the GPU's engines for the library's process at
  30 fps, from the DRM driver's fdinfo (drm-engine-*), where the driver
  gives it;
- on a Pi 5, the board's power over --power-window seconds, held and paced;
- run free for --free seconds: the frames drawn and the frames on the glass
  a second, the same latency, and the CPU time of every task on the system
  and of the library's process.

--cadence second changes the screen once a second with HOT hidden, and
--cadence never leaves it as it is. The latency then runs from the moment
the screen changes to the end of the transfer that carries the change.

With --perpixel, --glass runs per-pixel programs in place of the status
screen, and --check draws each pair of them headless at the same steps and
compares the pictures: see perpixel.py.

    uv run run.py
    uv run run.py --libs ghee,pillow --sizes 128x32 --png out
    uv run run.py --glass /dev/dri/card1 --cool 45
    uv run run.py --glass /dev/dri/card1 --cool 45 --perpixel
    uv run run.py --perpixel --check out --sizes 256x64 --levels 16
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

import board
import glass
import perpixel

HERE = Path(__file__).resolve().parent
FONTS = HERE.parents[1] / "ghee" / "examples" / "fonts"
LIBS = ["ghee", "pillow", "pygame", "raylib"]
GLASS_LIBS = ["ghee", "ghee_drop", "pillow", "pygame"]
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


def ticked(i, t, cadence):
    """values() at frame i, t seconds in, as often as `cadence` changes them:
    every frame, once a second with HOT hidden, or never."""
    if cadence == "frame":
        return values(i, t)
    k = int(t) if cadence == "second" else 0
    return values(k % 90, 0.0)


def say(msg):
    print(json.dumps(msg), flush=True)


def glass_child(screen, args):
    t0 = time.monotonic()
    screen.draw(*values(0, 0.0))
    warm = []
    for i in range(1, WARM):
        a = time.monotonic()
        shown = values(i, a - t0)
        screen.draw(*shown)
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
        start = True
        while True:
            a = time.monotonic()
            now = ticked(i, a - t0, args.cadence)
            screen.draw(*now)
            b = time.monotonic()
            sent, since, changed = b, a, now != shown
            if fb:
                sent = fb.sent
            if args.cadence != "frame":
                # The change at a phase's start came while nothing drew.
                sent = getattr(screen, "sent", sent)
                since = t0 + int(a - t0)
                changed = changed and not start
            frames.append((a, sent, b, changed, since))
            shown, start = now, False
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
    if args.lib in perpixel.PROGRAMS:
        return perpixel.child(args.lib, args.card, args.fb, args.sources, PACE, DRAIN)
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


def rail_watts():
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
    return {r: a * volts[r] for r, a in amps.items() if r in volts}


def board_watts():
    return sum(rail_watts().values())


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
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.wait(0.05):
            self.samples.append(rail_watts())

    def finish(self):
        """The median board power, and each rail's median in `rails`."""
        self.stop.set()
        self.thread.join()
        self.rails = {r: median([s[r] for s in self.samples]) for r in self.samples[0]}
        return median([sum(s.values()) for s in self.samples])


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


def measure_glass(lib, p, args, power, run):
    out, proc = {}, None
    glass.blank(p.fb)
    if power:
        out["board_watts"], out["board_rails"] = watts(args.power_window)
    tracer = glass.Tracer(p.bus)
    tracer.start()
    try:
        cmd = [sys.executable, __file__, "--child", lib, "--size", p.size]
        cmd += ["--card", p.card, "--fb", p.fb, "--fonts", args.fonts]
        cmd += ["--sources", args.sources, "--cadence", args.cadence]
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
        warmed = time.monotonic()
        first, out["warm"] = hello["first"], hello["warm"]
        time.sleep(glass.settle())
        out["idle_cpu"], _ = glass.cores(args.window, proc.pid)
        if power:
            out["idle_watts"], out["idle_rails"] = watts(args.power_window)
        proc.stdin.write("paced\n")
        proc.stdin.flush()
        time.sleep(SETTLE)
        g0, t0 = board.gpu_ns(proc.pid), time.monotonic()
        cpu, out["proc_cpu"] = glass.cores(args.window, proc.pid)
        g1, t1 = board.gpu_ns(proc.pid), time.monotonic()
        out["gpu"] = board.gpu_share(g0, g1, t1 - t0)
        out["cpu"] = cpu - out["idle_cpu"]
        if power:
            out["watts"], out["rails"] = watts(args.power_window)
        paced = ask("stop")["frames"]
        proc.stdin.write(f"free {args.free}\n")
        proc.stdin.flush()
        time.sleep(0.1)
        g0, t0 = board.gpu_ns(proc.pid), time.monotonic()
        out["free_cpu"], out["free_proc_cpu"] = glass.cores(args.free - 0.3, proc.pid)
        g1, t1 = board.gpu_ns(proc.pid), time.monotonic()
        out["free_gpu"] = board.gpu_share(g0, g1, t1 - t0)
        free = ask()["frames"]
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
        Path(args.raw, f"{lib}-{run + 1}.json").write_text(json.dumps(raw))
    if lib in perpixel.PROGRAMS:
        queue, warm = lib in perpixel.QUEUE, perpixel.WARM
    else:
        queue = getattr(importlib.import_module(f"{lib}_screen").Screen, "QUEUE", False)
        warm = WARM
    # A frame that changes nothing may reach the panel or not, so an update
    # carries the last frame that changed and was handed over before it.
    queue = queue and args.cadence == "frame"
    out["startup"] = next(u[1] for u in ups if u[0] > first) - spawned
    out["warm_updates"] = sum(1 for u in ups if first < u[0] < warmed)
    if queue and out["warm_updates"] != warm:
        print(f"{lib}: {out['warm_updates']} updates for {warm} warm-up frames")
    out["free"] = glass.phase(free, ups, queue, free[-1][2] + DRAIN)
    out["paced"] = glass.phase(paced, ups, queue, paced[-1][2] + DRAIN)
    return out


def watts(seconds):
    s = Sampler()
    time.sleep(seconds)
    return s.finish(), s.rails


def busiest(share):
    """The busiest GPU engine's share, or None with no figure. v3d's cpu
    queue runs on the CPU."""
    if not share:
        return None
    return max(v for k, v in share.items() if k != "cpu")


def glass_table(rows, power):
    head = ["run", "library", "panel", "C", "first s", "drawn/s", "glass/s"]
    head += ["updates/s", "ms", "p95 ms", "CPU free", "process free"]
    head += ["drawn/s @30", "glass/s @30", "updates/s @30", "ms @30"]
    head += ["p95 @30", "CPU @30", "process @30", "GPU @30"]
    if power:
        head += ["W idle", "W held", "W @30"]
    head.append("throttled")
    lines = [head]
    for r in rows:
        m = r.get("m")
        if m is None:
            lines.append([str(r["run"]), r["lib"], r["size"], "failed"])
            continue
        f, pc = m["free"], m["paced"]
        gpu = busiest(m["gpu"])
        cells = [str(r["run"]), r["lib"], r["size"], f"{r['temp']:.1f}"]
        cells += [f"{m['startup']:.2f}", f"{f['drawn']:.0f}"]
        cells += [f"{f['glass']:.0f}", f"{f['updates']:.0f}"]
        cells += [f"{f['latency'] * 1e3:.2f}", f"{f['latency_p95'] * 1e3:.2f}"]
        cells += [f"{m['free_cpu']:.1%}", f"{m['free_proc_cpu']:.1%}"]
        cells += [f"{pc['drawn']:.1f}", f"{pc['glass']:.1f}", f"{pc['updates']:.1f}"]
        cells += [f"{pc['latency'] * 1e3:.2f}", f"{pc['latency_p95'] * 1e3:.2f}"]
        cells += [f"{m['cpu']:.1%}", f"{m['proc_cpu']:.1%}"]
        cells.append("-" if gpu is None else f"{gpu:.1%}")
        if power:
            cells += [f"{m['board_watts']:.2f}", f"{m['idle_watts']:.2f}"]
            cells.append(f"{m['watts']:.2f}")
        cells.append(", ".join(r["throttled"]) or "no")
        lines.append(cells)
        for name, ph in (("free", f), ("paced", pc)):
            if ph["unmatched"]:
                print(f"{r['lib']} {name}: {ph['unmatched']:+d} updates against frames")
        if m["lost"]:
            print(f"{r['lib']}: the trace buffer lost events")
    print_rows(lines)


def spread(xs, fmt):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return "-"
    mid = (
        xs[len(xs) // 2]
        if len(xs) % 2
        else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2
    )
    if len(xs) == 1 or fmt.format(xs[0]) == fmt.format(xs[-1]):
        return fmt.format(mid)
    return f"{fmt.format(mid)} ({fmt.format(xs[0])} to {fmt.format(xs[-1])})"


def glass_summary(rows, power):
    """Each library's median over its runs that neither failed nor
    throttled, with the lowest and highest in brackets."""
    head = ["library", "runs", "first s", "ms @30", "CPU @30", "process @30"]
    head += ["GPU @30"]
    if power:
        head += ["W idle", "W @30"]
    lines = [head]
    for lib in dict.fromkeys(r["lib"] for r in rows):
        ms = [r["m"] for r in rows if r["lib"] == lib and r["m"] and not r["throttled"]]
        n = sum(1 for r in rows if r["lib"] == lib)
        cells = [lib, f"{len(ms)} of {n}"]
        cells.append(spread([m["startup"] for m in ms], "{:.2f}"))
        cells.append(spread([m["paced"]["latency"] * 1e3 for m in ms], "{:.2f}"))
        cells.append(spread([m["cpu"] * 100 for m in ms], "{:.1f}%"))
        cells.append(spread([m["proc_cpu"] * 100 for m in ms], "{:.1f}%"))
        gpu = [busiest(m["gpu"]) for m in ms]
        cells.append(spread([g * 100 for g in gpu if g is not None], "{:.1f}%"))
        if power:
            cells.append(spread([m["board_watts"] for m in ms], "{:.2f}"))
            cells.append(spread([m["watts"] for m in ms], "{:.2f}"))
        lines.append(cells)
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
    if args.cool is None:
        raise SystemExit(
            "--glass takes --cool C: the SoC temperature to wait for before "
            "each run, a little over the board's own at idle"
        )
    if args.repeats < 2:
        raise SystemExit("--glass takes --repeats 2 or more")
    p = glass.panel(args.glass)
    if args.raw:
        os.makedirs(args.raw, exist_ok=True)
    known = perpixel.PROGRAMS if args.perpixel else GLASS_LIBS
    libs = args.libs.split(",") if args.libs else known
    if set(libs) - set(known):
        raise SystemExit(f"--glass takes {', '.join(known)}")
    if args.perpixel:
        perpixel.fetch(args.sources)
        perpixel.liquid_png(args.sources, p.w, p.h)
        perpixel.chroma_palette(args.sources)
    power = has_pmic()
    flags = board.throttled()
    print(f"get_throttled: {flags:#x}")
    if flags >> 16:
        print("  flags up since boot count for a run only if they hold at its end")
    rows = []
    with board.governor("performance"), glass.dark(p.fb):
        for run in range(1, args.repeats + 1):
            for lib in libs:
                found, temp, waited = board.cool(args.cool, args.cool_wait)
                cooled = f", {waited:.0f} s to cool" if waited >= 1 else ""
                print(f"run {run} {lib}: SoC {found:.1f} C{cooled}", flush=True)
                before = board.throttled()
                m = measure_glass(lib, p, args, power, run - 1)
                events = board.throttle_events(before, board.throttled())
                if events:
                    print(f"run {run} {lib}: {', '.join(events)}, marked", flush=True)
                rows.append(
                    {"run": run, "lib": lib, "size": p.size, "temp": temp}
                    | {"waited": waited, "throttled": events, "m": m}
                )
                if args.json:
                    Path(args.json).write_text(
                        json.dumps({"panel": p.size, "rows": rows}, indent=1)
                    )
    glass_table(rows, power)
    print()
    glass_summary(rows, power)


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
        "--cadence",
        choices=["frame", "second", "never"],
        default="frame",
        help="how often the status screen changes on the glass",
    )
    ap.add_argument(
        "--window", type=float, default=4.0, help="seconds of each CPU window"
    )
    ap.add_argument("--power-window", type=float, default=2.0)
    ap.add_argument(
        "--repeats", type=int, default=2, help="runs of each library on the glass"
    )
    ap.add_argument(
        "--cool",
        type=float,
        metavar="C",
        help="SoC temperature to wait for before each run on the glass",
    )
    ap.add_argument(
        "--cool-wait",
        type=float,
        default=600.0,
        metavar="S",
        help="seconds to wait for --cool before giving up",
    )
    ap.add_argument("--raw", help="write each library's frames and transfers here")
    ap.add_argument(
        "--perpixel", action="store_true", help="the per-pixel programs, perpixel.py"
    )
    ap.add_argument("--check", metavar="DIR", help="compare each pair's pictures")
    ap.add_argument("--steps", default=",".join(str(s) for s in range(25, 2000, 50)))
    ap.add_argument("--pairs", default="chroma,liquid")
    ap.add_argument(
        "--levels", type=int, default=16, help="the panel's levels, 16 or 2"
    )
    ap.add_argument("--sources", default=perpixel.default_sources())
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
    args.sources = os.path.abspath(args.sources)
    if args.check:
        perpixel.fetch(args.sources)
        steps = [int(s) for s in args.steps.split(",")]
        for size in args.sizes.split(","):
            w, h = map(int, size.split("x"))
            perpixel.check(
                args.check,
                args.sources,
                w,
                h,
                args.levels,
                steps,
                args.pairs.split(","),
            )
        return
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
