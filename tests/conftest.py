import os

import pytest

from ghee.engine import Renderer

W, H, COMPONENTS = 64, 48, 4

# None falls through to the engine default (/dev/dri/card0, the Pi). renderD128 is the
# desktop-Mesa render node. First device that constructs wins.
_DEVICES = [os.environ.get("GHEE_DRI"), None, "/dev/dri/renderD128", "/dev/dri/card1"]


@pytest.fixture(scope="session")
def renderer():
    r, errs = None, []
    for dev in _DEVICES:
        if dev is None and any(e.startswith("card0") for e in errs):
            continue
        try:
            r = Renderer(W, H, components=COMPONENTS, dri_device=dev)
            break
        except Exception as e:
            errs.append(f"{dev or 'card0'}: {e}")
    if r is None:
        pytest.skip("no GL device for rendering:\n" + "\n".join(errs))
    yield r
    r.close()
