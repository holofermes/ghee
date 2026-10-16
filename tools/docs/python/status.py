# The status loop of docs/python.md, headless at a 128x32 panel's size, at 95%
# load and a fixed time: one blink of HOT, 2 a second, in ten frames.
import ghee

screen = ghee.open(width=128, height=32, timestep=0.05)
load = 0.95
for _ in range(10):
    with screen.draw() as d:
        d.text("21:47", x=0, y=0, font_size=16, render="mono")
        d.text(
            f"load {load:.0%}",
            x=0,
            y=20,
            font_size=8,
            font_family="Pixel Operator 8",
            render="mono",
        )
        d.rect(
            x=64,
            y=20,
            width=64,
            height=8,
            color="transparent",
            border={"width": 1, "color": "white"},
        )
        d.rect(x=66, y=22, width=60 * min(load, 1), height=4)
        if load > 0.9:
            w, _ = screen.measure("HOT", font_size=12)
            d.text(
                "HOT", x=128 - w, y=0, font_size=12, render="mono", blink={"rate": 2}
            )
    screen.save("f.####.png")
