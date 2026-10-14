"""A frame's tick, as a handle's frame() and a scene's tick hooks get it.

It imports nothing, so a handle hands one out without loading the shim.
"""


class Tick:
    """One frame's tick.

    `frame` counts the ticks, and `time` and `dt` are in seconds: `dt` is
    the time since the tick before, 1/60 for the first tick unless a timestep
    pins it.
    `params` holds the params that changed, each under its bare name, or as
    `pass.name` when the bare name sets another param, each value as a scene
    file writes it, and a text layer's words as its `text`, such as
    `title.text`. The first tick holds them all.
    """

    __slots__ = ("dt", "frame", "params", "time")

    def __init__(self, frame: int, time: float, dt: float, params: dict) -> None:
        self.frame = frame
        self.time = time
        self.dt = dt
        self.params = params

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tick):
            return NotImplemented
        return (self.frame, self.time, self.dt, self.params) == (
            other.frame,
            other.time,
            other.dt,
            other.params,
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"Tick(frame={self.frame!r}, time={self.time!r}, dt={self.dt!r}, "
            f"params={self.params!r})"
        )
