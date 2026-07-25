from ghee.engine import probe_display


def test_probe_missing_device_is_none():
    assert probe_display(dri_device="/dev/dri/does-not-exist") is None


def test_probe_reports_mode_or_none():
    # Headless hosts (CI, render-node-only cards) legitimately probe to None;
    # anything else must be a positive WxH.
    size = probe_display()
    if size is not None:
        w, h = size
        assert w > 0 and h > 0
