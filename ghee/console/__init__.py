"""ghee console"""

import importlib.resources


def web_dir():
    return str(importlib.resources.files("ghee.console") / "web")


def examples_dir():
    return str(importlib.resources.files("ghee") / "examples")
