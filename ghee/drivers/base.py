"""Display driver base and registry.

A driver is an Output bound to a physical panel: it owns the panel geometry and the
native pixel format, and packs each RGB frame before pushing it. Register one with
register(name, cls). A third-party package exposes its driver through a "ghee.drivers"
entry point whose module calls register() on import.
"""

import abc

from ghee.output import Output
from ghee.stats import DEST_PANEL

_REGISTRY = {}


def register(name, cls):
    _REGISTRY[name] = cls


class DisplayDriver(Output):
    dest = DEST_PANEL
    persists = True  # the panel holds its last frame with no bus traffic
    width = 0
    height = 0
    components = 3
    native_format = "rgb565"  # "rgb565" or "mono1"

    @property
    def bw(self):
        return self.native_format == "mono1"

    @abc.abstractmethod
    def write(self, frame):
        pass
