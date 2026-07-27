import pytest

from ghee.engine import Renderer, probe_dri_devices

W, H, COMPONENTS = 64, 48, 4


@pytest.fixture(scope="session")
def renderer():
    r, errs = None, []
    for dev in probe_dri_devices():
        try:
            r = Renderer(W, H, components=COMPONENTS, dri_device=dev)
            break
        except Exception as e:
            errs.append(f"{dev}: {e}")
    if r is None:
        pytest.skip("no GL device for rendering:\n" + "\n".join(errs))
    yield r
    r.close()
