"""The `ghee` command of the ghee-render wheel, compiled into the native
module."""

import signal
import sys

from . import _ext


def main() -> int:
    # Python's own handler only sets a flag, which native code never reads.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    return _ext()._command(sys.argv[1:])
