"""The panel side of the runner: a SPI or I2C panel under a kernel DRM driver.

The kernel's spi or i2c tracepoints time each bus transfer to the panel on
CLOCK_MONOTONIC, the clock of the frames' own stamps. One update of the
panel is the run of transfers that ends with a transfer of pixel data, 64
bytes or more, and the update starts at the first of the transfers that
lead up to that one with less than a millisecond between each.
"""

import bisect
import re
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

TRACE = Path("/sys/kernel/tracing")
DATA = 64
GAP = 0.001
LINE = re.compile(r"^\s*.+?-\d+\s+\[\d+\]\s+\S+\s+(\d+\.\d+): (\w+): (.*)$")


def panel(card):
    """The card's fbdev, its size and the bus device of the panel."""
    dev = Path("/sys/class/drm") / Path(card).resolve().name / "device"
    fbs = sorted((dev / "graphics").glob("fb*"))
    if not fbs:
        raise SystemExit(f"{card}: no fbdev for this card")
    w, h = (int(v) for v in (fbs[0] / "virtual_size").read_text().split(","))
    real = dev.resolve()
    kind = (real / "subsystem").resolve().name
    if kind == "spi":
        bus, cs = re.fullmatch(r"spi(\d+)\.(\d+)", real.name).groups()
        bus = SimpleNamespace(kind="spi", name=real.name, bus=int(bus), cs=int(cs))
    elif kind == "i2c":
        adapter, addr = real.name.split("-")
        bus = SimpleNamespace(kind="i2c", adapter=int(adapter), addr=int(addr, 16))
    else:
        raise SystemExit(f"{card}: the panel sits on {kind}, not spi or i2c")
    return SimpleNamespace(
        card=str(card), fb=f"/dev/{fbs[0].name}", w=w, h=h, size=f"{w}x{h}", bus=bus
    )


def tracefs(path, value=None):
    if value is None:
        return subprocess.run(
            ["sudo", "-n", "cat", str(TRACE / path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    subprocess.run(
        ["sudo", "-n", "tee", str(TRACE / path)],
        input=f"{value}\n",
        text=True,
        stdout=subprocess.DEVNULL,
        check=True,
    )


class Tracer:
    """The bus tracepoints of one panel, on for one run and off after."""

    def __init__(self, bus, buffer_kb=2048):
        self.bus = bus
        self.buffer_kb = buffer_kb
        if bus.kind == "spi":
            self.events = ["spi/spi_transfer_start", "spi/spi_transfer_stop"]
            self.filter = f"bus_num == {bus.bus} && chip_select == {bus.cs}"
        else:
            self.events = ["i2c/i2c_write", "i2c/i2c_result"]
            self.filter = f"adapter_nr == {bus.adapter}"

    def start(self):
        self.clock = re.search(r"\[(\w+)\]", tracefs("trace_clock"))[1]
        self.size = tracefs("buffer_size_kb").split()[0]
        tracefs("trace_clock", "mono")
        tracefs("buffer_size_kb", self.buffer_kb)
        tracefs("trace", "")
        for e in self.events:
            tracefs(f"events/{e}/filter", self.filter)
            tracefs(f"events/{e}/enable", 1)

    def stop(self):
        """Turn the events off, put the settings back, and give the
        transfers as (start, stop, bytes), and whether the buffer lost any."""
        try:
            for e in self.events:
                tracefs(f"events/{e}/enable", 0)
            text = tracefs("trace")
        finally:
            for e in self.events:
                tracefs(f"events/{e}/filter", 0)
            tracefs("trace", "")
            tracefs("trace_clock", self.clock)
            tracefs("buffer_size_kb", self.size)
        counts = re.search(r"entries-in-buffer/entries-written: (\d+)/(\d+)", text)
        lost = counts is not None and counts[1] != counts[2]
        parse = spi_transfers if self.bus.kind == "spi" else i2c_transfers
        return parse(text, self.bus), lost


def spi_transfers(text, bus):
    started, out = {}, []
    for line in text.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        ts, event, rest = float(m[1]), m[2], m[3].split()
        if rest[0] != bus.name:
            continue
        if event == "spi_transfer_start":
            started[rest[1]] = ts
        elif event == "spi_transfer_stop" and rest[1] in started:
            out.append((started.pop(rest[1]), ts, int(rest[2].split("=")[1])))
    return out


def i2c_transfers(text, bus):
    pending, out = None, []
    for line in text.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        ts, event, rest = float(m[1]), m[2], m[3]
        if event == "i2c_write":
            f = re.match(r"i2c-\d+ #(\d+) a=([0-9a-f]+) f=\w+ l=(\d+)", rest)
            if f[1] == "0":
                pending = [ts, 0] if int(f[2], 16) == bus.addr else None
            if pending:
                pending[1] += int(f[3])
        elif event == "i2c_result" and pending:
            out.append((pending[0], ts, pending[1]))
            pending = None
    return out


def updates(transfers):
    """(start, stop) of each update, in order."""
    out, first, last = [], None, None
    for start, stop, n in transfers:
        if first is None or start - last > GAP:
            first = start
        last = stop
        if n >= DATA:
            out.append((first, stop))
            first = None
    return out


def carried(frames, ups, queue, end):
    """Each frame that reached the glass, as (frame, update).

    frames are (draw start, write start, draw end) in order, after a pause
    with no update in flight, and ups the updates that start after the first
    frame's draw and before end. A queue hands every frame to the panel in
    order, so the nth update carries the nth frame. Otherwise the panel takes
    the latest frame: an update carries the last frame whose write to the
    framebuffer began before the update started.
    """
    ups = [u for u in ups if frames[0][0] < u[0] < end]
    if queue:
        return list(zip(frames, ups)), len(ups) - len(frames)
    sent = [f[1] for f in frames]
    out, last = [], -1
    for u in ups:
        i = bisect.bisect_right(sent, u[0]) - 1
        if i > last:
            out.append((frames[i], u))
            last = i
    return out, 0


def stats(xs):
    xs = sorted(xs)
    return sum(xs) / len(xs), xs[int(len(xs) * 0.95)]


def phase(frames, ups, queue, end):
    """Frames drawn and frames on the glass a second, and the latency from
    a frame's draw to the end of its update, over one phase."""
    pairs, extra = carried(frames, ups, queue, end)
    t0, t1 = frames[0][0], frames[-1][2]
    glass = sum(1 for _, u in pairs if u[1] <= t1)
    mean, p95 = stats([u[1] - f[0] for f, u in pairs])
    return {
        "drawn": len(frames) / (t1 - t0),
        "glass": glass / (t1 - t0),
        "latency": mean,
        "latency_p95": p95,
        "unmatched": extra,
    }


def settle():
    """Seconds that the ondemand governor holds a raised clock after a
    burst, plus one, or one under any other governor."""
    cpufreq = Path("/sys/devices/system/cpu/cpufreq")
    gov = cpufreq / "policy0" / "scaling_governor"
    if not gov.exists() or gov.read_text().strip() != "ondemand":
        return 1.0
    d = next(
        p
        for p in (cpufreq / "ondemand", cpufreq / "policy0" / "ondemand")
        if p.exists()
    )
    hold = int((d / "sampling_rate").read_text()) * int(
        (d / "sampling_down_factor").read_text()
    )
    return hold / 1e6 + 1


def run_ns():
    """Time spent by every task on every CPU, from /proc/schedstat."""
    lines = Path("/proc/schedstat").read_text().splitlines()
    return sum(int(line.split()[7]) for line in lines if line.startswith("cpu"))


def task_ns(pid):
    """Time spent by every thread of one process."""
    tasks = Path(f"/proc/{pid}/task").iterdir()
    return sum(int((t / "schedstat").read_text().split()[0]) for t in tasks)


def cores(seconds, pid):
    """Cores' worth of task time over the next `seconds`, of the whole system
    and of one process."""
    n0, p0, t0 = run_ns(), task_ns(pid), time.monotonic()
    time.sleep(seconds)
    wall = time.monotonic() - t0
    return (run_ns() - n0) / 1e9 / wall, (task_ns(pid) - p0) / 1e9 / wall
