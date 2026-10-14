# The clock of docs/python.md at a fixed time.
import ghee


@ghee.tick
def clock(tick):
    with ghee.draw() as d:
        d.text("21:47:05")


ghee.run()
