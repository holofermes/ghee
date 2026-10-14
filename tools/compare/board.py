"""The board around each run on the glass: the CPU governor, the SoC's
temperature and throttle flags from `vcgencmd`, and the GPU time of one
process from the DRM driver's fdinfo."""

import contextlib
import os
import re
import signal
import subprocess
import time
from pathlib import Path

CPUFREQ = Path("/sys/devices/system/cpu/cpufreq")
# get_throttled's bits: 0 to 3 hold now, 16 to 19 have held since boot.
FLAGS = ["under-voltage", "arm frequency capped", "throttled", "soft temp limit"]


def vcgencmd(*args):
    try:
        return subprocess.run(
            ["vcgencmd", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"vcgencmd {' '.join(args)}: {e}") from e


def temperature():
    return float(re.fullmatch(r"temp=([\d.]+)'C", vcgencmd("measure_temp"))[1])


def cool(limit, timeout):
    """Wait until the SoC reads under `limit` degrees C, and give the
    temperature found, the one waited for and the seconds waited."""
    t0 = time.monotonic()
    found = now = temperature()
    while now >= limit:
        if time.monotonic() - t0 > timeout:
            raise SystemExit(
                f"the SoC stayed at {now:.1f} C, not under {limit:.1f} C, "
                f"for {timeout:.0f} s: giving up"
            )
        time.sleep(2)
        now = temperature()
    return found, now, time.monotonic() - t0


def throttled():
    return int(vcgencmd("get_throttled").split("=")[1], 16)


def throttle_events(before, after):
    """The flags that held at the end of a run or first came up during it.
    A flag already up since boot before the run counts only if it holds at
    the end."""
    now = after & 0xF
    new = (after >> 16) & ~(before >> 16) & 0xF
    return [name for i, name in enumerate(FLAGS) if (now | new) >> i & 1]


def write_sys(path, value):
    subprocess.run(
        ["sudo", "-n", "tee", str(path)],
        input=f"{value}\n",
        text=True,
        stdout=subprocess.DEVNULL,
        check=True,
    )


def _exit_on(signum, _frame):
    raise SystemExit(128 + signum)


@contextlib.contextmanager
def governor(name):
    """Every CPU policy on `name` inside the block, and back on its own
    governor after, on an error, Ctrl-C, SIGTERM or SIGHUP too."""
    paths = sorted(CPUFREQ.glob("policy*/scaling_governor"))
    if not paths:
        raise SystemExit(f"{CPUFREQ}: no cpufreq policies, no governor to set")
    old = {p: p.read_text().strip() for p in paths}
    handlers = {s: signal.signal(s, _exit_on) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        for p in paths:
            write_sys(p, name)
        was = ", ".join(sorted(set(old.values())))
        print(f"governor: {name} on {len(paths)} policies, was {was}", flush=True)
        yield
    finally:
        for p, g in old.items():
            write_sys(p, g)
        back = {p.read_text().strip() for p in paths}
        print(f"governor: back on {', '.join(sorted(back))}", flush=True)
        for s, h in handlers.items():
            signal.signal(s, h)


def gpu_ns(pid):
    """Busy nanoseconds per engine of every DRM client that the process
    holds open, from fdinfo's drm-engine-* lines, or None when no file
    gives them. A client open on two descriptors counts once."""
    clients = {}
    try:
        fds = os.listdir(f"/proc/{pid}/fdinfo")
    except OSError:
        return None
    for fd in fds:
        try:
            text = Path(f"/proc/{pid}/fdinfo/{fd}").read_text()
        except OSError:
            continue
        fields = {
            k.strip(): v.strip()
            for k, _, v in (line.partition(":") for line in text.splitlines())
        }
        engines = {
            k.removeprefix("drm-engine-"): int(v.split()[0])
            for k, v in fields.items()
            if k.startswith("drm-engine-") and not k.startswith("drm-engine-capacity")
        }
        if engines and "drm-client-id" in fields:
            clients[fields["drm-client-id"]] = engines
    if not clients:
        return None
    total = {}
    for engines in clients.values():
        for k, v in engines.items():
            total[k] = total.get(k, 0) + v
    return total


def gpu_share(before, after, wall):
    """Each engine's busy share of `wall` seconds between two gpu_ns()."""
    if before is None or after is None:
        return None
    return {k: (after[k] - before.get(k, 0)) / 1e9 / wall for k in after}
