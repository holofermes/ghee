"""The `ghee` command of the ghee-render wheel, compiled into the native
module."""

import sys

from . import _ext


def main() -> int:
    return _ext()._command(sys.argv[1:])
