"""The bundled panel drivers, against faked buses.

No panel and no kernel here: spidev, gpiod and smbus2 are replaced with
recorders, so what is under test is the bytes and the pin levels each driver
would put on the wire, and what it does when a line it wants is already taken.
gpiod ships two incompatible python APIs (1.x on bookworm, 2.x on trixie) and
both are exercised.
"""

import errno
import importlib
import os
import sys
import types

import numpy as np
import pytest

from ghee.engine import bundled_lib

CHIPS = {"/dev/gpiochip0": "gpio-brcmstb", "/dev/gpiochip4": "pinctrl-rp1"}


class Bus:
    """What the gpiod fakes below record, whichever API is in play. Levels are
    normalised to bools, so a test does not care that v1 says 1 and v2 says
    Value.ACTIVE."""

    def __init__(self, busy=()):
        self.busy = set(busy)
        self.claimed = set()
        self.requests = []  # (consumer, offsets) per successful request
        self.events = []  # (offset, level) in order
        self.released = []
        self.closed = []

    @property
    def levels(self):
        return dict(self.events)

    def drive(self, offset, level):
        self.events.append((offset, level))


def fake_gpiod_v2(bus, chips=CHIPS):
    """gpiod 2.x, as trixie ships it: one atomic request_lines() per group,
    chip metadata on a ChipInfo, and the enums in a gpiod.line submodule."""
    line = types.ModuleType("gpiod.line")
    line.Direction = types.SimpleNamespace(OUTPUT="output")
    line.Value = types.SimpleNamespace(ACTIVE="active", INACTIVE="inactive")

    class Chip:
        def __init__(self, path):
            if path not in chips:
                raise OSError(errno.ENODEV, "no such device")
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_info(self):
            return types.SimpleNamespace(label=chips[self.path])

    class Request:
        def __init__(self, offsets):
            self.offsets = offsets

        def set_value(self, offset, value):
            bus.drive(offset, value == line.Value.ACTIVE)

        def release(self):
            bus.released.append(tuple(self.offsets))
            bus.claimed.difference_update(self.offsets)

    def request_lines(path, consumer=None, config=None):
        offsets = list(config)
        taken = [o for o in offsets if o in bus.busy]
        if taken:
            raise OSError(errno.EBUSY, f"GPIO{taken[0]} is busy")
        bus.claimed.update(offsets)
        bus.requests.append((consumer, offsets))
        return Request(offsets)

    mod = types.ModuleType("gpiod")
    mod.line = line
    mod.Chip = Chip
    mod.LineSettings = lambda **kw: kw
    mod.request_lines = request_lines
    return mod


def fake_gpiod_v1(bus, chips=CHIPS):
    """gpiod 1.6.x, as bookworm ships it: chip.get_line() then line.request(),
    one line at a time, and chip metadata through methods."""

    class Line:
        def __init__(self, offset):
            self.offset = offset

        def request(self, consumer=None, type=None, default_val=0):
            if self.offset in bus.busy:
                raise OSError(errno.EBUSY, f"GPIO{self.offset} is busy")
            bus.claimed.add(self.offset)
            bus.requests.append((consumer, [self.offset]))

        def set_value(self, value):
            bus.drive(self.offset, bool(value))

        def release(self):
            bus.released.append(self.offset)
            bus.claimed.discard(self.offset)

    class Chip:
        def __init__(self, path):
            if path not in chips:
                raise OSError(errno.ENODEV, "no such device")
            self.path = path

        def label(self):
            return chips[self.path]

        def get_line(self, offset):
            return Line(offset)

        def close(self):
            bus.closed.append(self.path)

    mod = types.ModuleType("gpiod")
    mod.LINE_REQ_DIR_OUT = 1
    mod.Chip = Chip
    return mod


class FakeSpi:
    """spidev.SpiDev, recording (dc level, bytes) per transfer. The DC level
    comes back out of the Bus the gpiod fake drives."""

    def __init__(self, bus, dc):
        self.bus = bus
        self.dc = dc
        self.transfers = []
        self.opened = None
        self.mode = None
        self.max_speed_hz = None
        self.closed = False

    def open(self, spi_bus, device):
        self.opened = (spi_bus, device)

    def writebytes2(self, data):
        self.transfers.append((self.bus.levels.get(self.dc), bytes(data)))

    def close(self):
        self.closed = True


class FakeSMBus:
    def __init__(self, number):
        self.number = number
        self.writes = []  # ("cmd", addr, control, value) or ("rdwr", addr, bytes)
        self.closed = False

    def write_byte_data(self, addr, control, value):
        self.writes.append(("cmd", addr, control, value))

    def i2c_rdwr(self, *msgs):
        for addr, data in msgs:
            self.writes.append(("rdwr", addr, bytes(data)))

    def close(self):
        self.closed = True

    def commands(self):
        return [w for w in self.writes if w[0] == "cmd"]

    def transfers(self):
        return [w for w in self.writes if w[0] == "rdwr"]


def fake_smbus2(made):
    mod = types.ModuleType("smbus2")

    def SMBus(number):
        made["bus"] = FakeSMBus(number)
        return made["bus"]

    mod.SMBus = SMBus
    mod.i2c_msg = types.SimpleNamespace(write=lambda addr, data: (addr, data))
    return mod


@pytest.fixture
def load(monkeypatch):
    """Import a driver module against fake buses, and drop it again afterwards
    so the fakes never leak into another test."""
    loaded = []

    def _load(name, **fakes):
        for mod_name, mod in fakes.items():
            monkeypatch.setitem(sys.modules, mod_name, mod)
            if hasattr(mod, "line"):
                monkeypatch.setitem(sys.modules, f"{mod_name}.line", mod.line)
        for mod_name in dict.fromkeys(["ghee.drivers.gpio", f"ghee.drivers.{name}"]):
            sys.modules.pop(mod_name, None)
            loaded.append(mod_name)
        mod = importlib.import_module(f"ghee.drivers.{name}")
        gpio = sys.modules.get("ghee.drivers.gpio")
        if gpio is not None:
            monkeypatch.setattr(gpio, "_chip_paths", lambda: list(CHIPS))
        return mod

    yield _load
    from ghee.drivers.base import _REGISTRY

    for mod_name in loaded:
        sys.modules.pop(mod_name, None)
        _REGISTRY.pop(mod_name.rsplit(".", 1)[-1], None)


@pytest.fixture(params=[fake_gpiod_v1, fake_gpiod_v2], ids=["gpiod-v1", "gpiod-v2"])
def gpiod_api(request):
    return request.param


# the gpiod shim: two bindings, one call surface


def test_backend_follows_the_installed_binding(load, gpiod_api):
    bus = Bus()
    gpio = load("gpio", gpiod=gpiod_api(bus))
    assert gpio.V2 is (gpiod_api is fake_gpiod_v2)

    pins = gpio.Pins([25, None, 22], consumer="ghee-test")
    assert pins.offsets == [25, 22]  # a None pin is not a line
    assert bus.claimed == {25, 22}
    assert {consumer for consumer, _ in bus.requests} == {"ghee-test"}

    pins.set(25, True)
    pins.set(22, False)
    assert bus.levels == {25: True, 22: False}
    pins.close()
    assert bus.claimed == set()


def test_chip_is_found_by_label_prefix_not_by_number(load, gpiod_api):
    """The Pi 3 is pinctrl-bcm2835 on gpiochip0, the Pi 5 is pinctrl-rp1 on
    gpiochip4, with unrelated chips in between."""
    gpio = load("gpio", gpiod=gpiod_api(Bus()))
    chip = gpio.soc_chip()
    assert getattr(chip, "path", chip) == "/dev/gpiochip4"


def test_an_unopenable_chip_is_skipped(load, gpiod_api, monkeypatch):
    gpio = load("gpio", gpiod=gpiod_api(Bus()))
    monkeypatch.setattr(
        gpio, "_chip_paths", lambda: ["/dev/gpiochip9", "/dev/gpiochip4"]
    )
    chip = gpio.soc_chip()
    assert getattr(chip, "path", chip) == "/dev/gpiochip4"


def test_no_soc_chip_is_an_error(load, gpiod_api, monkeypatch):
    gpio = load("gpio", gpiod=gpiod_api(Bus()))
    monkeypatch.setattr(gpio, "_chip_paths", lambda: ["/dev/gpiochip0"])
    with pytest.raises(RuntimeError, match="pinctrl-"):
        gpio.soc_chip()


def test_a_failed_group_leaves_nothing_claimed(load, gpiod_api):
    """v2 requests a group atomically; v1 claims line by line and would hold the
    earlier ones until the process exits, so it unwinds them itself."""
    bus = Bus(busy=[22])
    gpio = load("gpio", gpiod=gpiod_api(bus))
    with pytest.raises(OSError):
        gpio.Pins([25, 22])
    assert bus.claimed == set()


# ST7789


def st7789_driver(load, gpiod_api, busy=(), **kw):
    bus = Bus(busy=busy)
    spidev = types.ModuleType("spidev")
    spi = FakeSpi(bus, dc=kw.get("dc", 25))
    spidev.SpiDev = lambda: spi
    mod = load("st7789", gpiod=gpiod_api(bus), spidev=spidev)
    return mod.ST7789(**kw), bus, spi


def wire(spi):
    return [(dc, data.hex()) for dc, data in spi.transfers]


# What a Blinka-driven adafruit_rgb_display.st7789 puts on the wire for this
# panel, recorded off the real one: (DC level, bytes).
INIT_ON_THE_WIRE = [
    (False, "01"),  # SWRESET
    (False, "11"),  # SLPOUT
    (False, "3a"),  # COLMOD
    (True, "55"),
    (False, "36"),  # MADCTL
    (True, "08"),
    (False, "2a"),  # CASET
    (True, "003500bc"),
    (False, "2b"),  # RASET
    (True, "00280118"),
    (False, "21"),  # INVON
    (False, "13"),  # NORON
    (False, "29"),  # DISPON
    (False, "36"),  # MADCTL
    (True, "c0"),
]

FIRST_FRAME_ON_THE_WIRE = [
    (False, "2a"),
    (True, "003500bb"),
    (False, "2b"),
    (True, "00280117"),
    (False, "2c"),  # RAMWR
]


def test_st7789_init_matches_the_recorded_panel_stream(load, gpiod_api):
    d, _, spi = st7789_driver(load, gpiod_api)
    assert wire(spi) == INIT_ON_THE_WIRE
    assert spi.opened == (0, 0)
    assert spi.mode == 0
    assert spi.max_speed_hz == 99000000
    assert (d.width, d.height) == (135, 240)
    assert d._buf.size == 135 * 240 * 2


def test_st7789_frame_sets_the_window_once_then_blits(load, gpiod_api):
    if not (os.environ.get("GHEE_LIB") or bundled_lib()):
        pytest.skip("no engine .so to pack a frame with")
    d, _, spi = st7789_driver(load, gpiod_api)
    frame = np.zeros((240, 135, 3), dtype=np.uint8)
    frame[:, :, 0] = 255

    del spi.transfers[:]
    d.write(frame)
    dc, blit = spi.transfers[-1]
    assert wire(spi)[:-1] == FIRST_FRAME_ON_THE_WIRE
    assert dc is True and len(blit) == 135 * 240 * 2
    assert blit[:2] == b"\xf8\x00"  # red, RGB565 big endian

    del spi.transfers[:]
    d.write(frame)
    assert wire(spi)[:-1] == [(False, "2c")]  # the window is already set


def test_st7789_geometry_and_pins_are_constructor_arguments(load, gpiod_api):
    d, bus, spi = st7789_driver(
        load,
        gpiod_api,
        width=240,
        height=240,
        x_offset=0,
        y_offset=80,
        dc=24,
        rst=27,
        backlight=None,
        baudrate=32000000,
        spi_bus=1,
        spi_device=1,
    )
    assert (d.width, d.height) == (240, 240)
    assert d._buf.size == 240 * 240 * 2
    assert spi.opened == (1, 1) and spi.max_speed_hz == 32000000
    assert bus.claimed == {24, 27}  # no backlight line requested
    assert d.bl_pins is None
    assert wire(spi)[6:10] == [
        (False, "2a"),
        (True, "000000f0"),
        (False, "2b"),
        (True, "00500140"),
    ]
    d.turn_on()  # a panel with no backlight pin: a no-op, not a crash
    d.turn_off()
    assert 22 not in bus.levels


def test_st7789_madctl_is_the_last_init_write(load, gpiod_api):
    """A 240x240 board came up rotated 180 under the 0xc0 default, so the
    rotation/mirroring byte has to reach the panel from the constructor."""
    d, _, spi = st7789_driver(load, gpiod_api, madctl=0x00)
    assert wire(spi)[-2:] == [(False, "36"), (True, "00")]
    assert wire(spi)[:-1] == INIT_ON_THE_WIRE[:-1]  # only that byte changed
    assert d.madctl == 0x00


def test_st7789_pulses_reset_before_talking_to_the_panel(load, gpiod_api):
    d, bus, spi = st7789_driver(load, gpiod_api, rst=27)
    assert 27 in bus.claimed
    assert [lvl for off, lvl in bus.events if off == 27][:2] == [False, True]
    assert bus.events.index((27, True)) < bus.events.index((25, False))
    assert wire(spi) == INIT_ON_THE_WIRE


def test_st7789_survives_a_backlight_another_driver_owns(load, gpiod_api, capsys):
    """GPIO22 is this panel's backlight and this board's DigiAMP+ mute, so the
    iqaudio kernel driver holds it and libgpiod is refused."""
    d, bus, spi = st7789_driver(load, gpiod_api, busy=[22])
    assert d.bl_pins is None
    assert bus.claimed == {25}  # DC survived the refusal
    assert wire(spi) == INIT_ON_THE_WIRE
    err = capsys.readouterr().err
    assert "GPIO22" in err and "backlight" in err
    d.turn_on()
    d.turn_off()
    assert 22 not in bus.levels


def test_st7789_backlight_follows_turn_on_and_off(load, gpiod_api):
    d, bus, _ = st7789_driver(load, gpiod_api)
    assert bus.levels[22] is True  # the constructor leaves the panel lit
    d.turn_off()
    assert bus.levels[22] is False
    d.turn_on()
    assert bus.levels[22] is True


def test_st7789_close_releases_the_bus_and_every_line(load, gpiod_api):
    d, bus, spi = st7789_driver(load, gpiod_api, rst=27)
    d.close()
    assert spi.closed and bus.claimed == set()


# SSD1306


def ssd1306_driver(load, **kw):
    made = {}
    mod = load("ssd1306", smbus2=fake_smbus2(made))
    return mod.SSD1306(**kw), made["bus"]


# The command bytes adafruit_ssd1306 sends a 128x32 panel on internal Vcc,
# poweron() first, then the init table, then show()'s addressing window.
INIT_COMMANDS = [
    0xAF,
    0xAE,
    0x20,
    0x00,
    0x40,
    0xA1,
    0xA8,
    0x1F,
    0xC8,
    0xD3,
    0x00,
    0xDA,
    0x02,
    0xD5,
    0x80,
    0xD9,
    0xF1,
    0xDB,
    0x30,
    0x81,
    0xFF,
    0xA4,
    0xA6,
    0xAD,
    0x30,
    0x8D,
    0x14,
    0xAF,
]
WINDOW_COMMANDS = [0x21, 0, 127, 0x22, 0, 3]


def test_ssd1306_init_is_commands_behind_the_command_control_byte(load):
    d, bus = ssd1306_driver(load)
    cmds = bus.commands()
    assert all(addr == 0x3C and control == 0x80 for _, addr, control, _ in cmds)
    assert [value for *_, value in cmds] == INIT_COMMANDS + WINDOW_COMMANDS
    assert (d.width, d.height) == (128, 32)


def test_ssd1306_shows_one_transfer_behind_the_data_control_byte(load):
    _, bus = ssd1306_driver(load)
    _, addr, data = bus.transfers()[0]
    assert addr == 0x3C
    assert len(data) == 1 + 128 * 32 // 8
    assert data[0] == 0x40
    assert set(data[1:]) == {0}  # the panel comes up cleared


def test_ssd1306_frame_lands_after_the_control_byte(load):
    if not (os.environ.get("GHEE_LIB") or bundled_lib()):
        pytest.skip("no engine .so to pack a frame with")
    d, bus = ssd1306_driver(load)
    del bus.writes[:]
    d.write(np.full((32, 128, 3), 255, dtype=np.uint8))
    assert [w[0] for w in bus.writes] == ["cmd"] * len(WINDOW_COMMANDS) + ["rdwr"]
    assert [w[3] for w in bus.commands()] == WINDOW_COMMANDS
    data = bus.transfers()[0][2]
    assert data[0] == 0x40 and set(data[1:]) == {0xFF}


def test_ssd1306_power_commands(load):
    d, bus = ssd1306_driver(load)
    del bus.writes[:]
    d.turn_off()
    d.turn_on()
    assert [(w[2], w[3]) for w in bus.commands()] == [(0x80, 0xAE), (0x80, 0xAF)]
    d.close()
    assert bus.closed


def test_ssd1306_bus_and_address_are_constructor_arguments(load):
    _, bus = ssd1306_driver(load, bus=0, addr=0x3D)
    assert bus.number == 0
    assert all(addr == 0x3D for _, addr, *_ in bus.writes)


# what `ghee displays` reports


def test_displays_lists_every_option_a_driver_takes(load, capsys):
    """Undiscoverable otherwise: the only other way to learn y_offset is to read
    the driver."""
    from ghee.cli import main

    load("st7789", gpiod=fake_gpiod_v2(Bus()), spidev=types.ModuleType("spidev"))
    load("ssd1306", smbus2=fake_smbus2({}))
    main(["displays"])
    lines = [ln.strip() for ln in capsys.readouterr().out.splitlines()]
    assert [
        " ".join(ln.split()) for ln in lines if ln.startswith(("st7789", "ssd1306"))
    ] == ["st7789 ok", "ssd1306 ok"]
    assert (
        "--display-opts width=135,height=240,x_offset=53,y_offset=40,dc=25,"
        "rst=None,backlight=22,baudrate=99000000,spi_bus=0,spi_device=0,"
        "madctl=192" in lines
    )
    assert "--display-opts bus=1,addr=60" in lines
