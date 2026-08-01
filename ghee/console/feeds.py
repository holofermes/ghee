"""What the console fills in for itself: host info, and the array placeholder."""

from ghee.feeds import demo_feed

# demo_feed lives in ghee.feeds; re-exported here for the older import path.
__all__ = ["default_hwinfo", "demo_feed"]


def default_hwinfo():
    """Generic host info when no real provider is injected."""
    import platform

    return {"model": platform.platform(), "gpu": "n/a"}
