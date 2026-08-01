"""Output pins over libgpiod, v1 (bookworm) or v2 (trixie).

The two python bindings share a name and nothing else: v1 is chip.get_line() +
line.request(type=LINE_REQ_DIR_OUT), v2 is gpiod.request_lines(path,
config={offset: LineSettings(...)}). Chip metadata is methods in v1 and
properties on a ChipInfo in v2.

Offsets are BCM numbers. A Pins group is one request, so lines that may fail
independently (an optional backlight) belong in their own group.
"""

import glob

import gpiod

V2 = hasattr(gpiod, "request_lines")
CHIP_LABEL = "pinctrl-"  # bcm2835 / bcm2711 / rp1: only the prefix is stable


def _chip_paths():
    return sorted(glob.glob("/dev/gpiochip*"))


if V2:
    from gpiod.line import Direction, Value

    def soc_chip():
        """Path of the SoC gpiochip. Numbering moves between boards (the Pi 5
        has four), the label prefix does not."""
        for path in _chip_paths():
            try:
                with gpiod.Chip(path) as chip:
                    if chip.get_info().label.startswith(CHIP_LABEL):
                        return path
            except OSError:
                continue
        raise RuntimeError(f"no gpiochip labelled {CHIP_LABEL}*")

    class Pins:
        def __init__(self, offsets, consumer="ghee"):
            self.offsets = [o for o in offsets if o is not None]
            self.req = gpiod.request_lines(
                soc_chip(),
                consumer=consumer,
                config={
                    o: gpiod.LineSettings(
                        direction=Direction.OUTPUT, output_value=Value.INACTIVE
                    )
                    for o in self.offsets
                },
            )

        def set(self, offset, high):
            self.req.set_value(offset, Value.ACTIVE if high else Value.INACTIVE)

        def close(self):
            self.req.release()

else:

    def soc_chip():
        """The SoC gpiochip, open. Numbering moves between boards (the Pi 5 has
        four), the label prefix does not."""
        for path in _chip_paths():
            try:
                chip = gpiod.Chip(path)
            except OSError:
                continue
            if chip.label().startswith(CHIP_LABEL):
                return chip
            chip.close()
        raise RuntimeError(f"no gpiochip labelled {CHIP_LABEL}*")

    class Pins:
        def __init__(self, offsets, consumer="ghee"):
            self.offsets = [o for o in offsets if o is not None]
            self.chip = soc_chip()
            self.lines = {}
            try:
                for o in self.offsets:
                    line = self.chip.get_line(o)
                    line.request(
                        consumer=consumer, type=gpiod.LINE_REQ_DIR_OUT, default_val=0
                    )
                    self.lines[o] = line
            except BaseException:
                # v1 claims one line per call, so a failure part way through
                # holds the earlier lines until the process exits.
                self.close()
                raise

        def set(self, offset, high):
            self.lines[offset].set_value(1 if high else 0)

        def close(self):
            for line in self.lines.values():
                line.release()
            self.lines.clear()
            if self.chip is not None:
                self.chip.close()
                self.chip = None
