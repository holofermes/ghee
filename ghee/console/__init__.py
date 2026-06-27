"""ghee console: the shared shader-editor PWA (web/) and the standalone dev
render server. web/ ships as package data, so every server serves one copy from
wherever ghee is installed instead of keeping its own."""
import importlib.resources


def web_dir():
    """Filesystem path to the console web assets, resolved from the installed
    package (the live source tree under an editable install)."""
    return str(importlib.resources.files(__package__) / "web")


def examples_dir():
    """Filesystem path to the bundled example scenes."""
    return str(importlib.resources.files("ghee") / "examples")
