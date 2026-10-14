import ghee

screen = ghee.open(width=128, height=64)
with screen.draw() as d:
    d.text("hello", color="orange", glow={})
    d.line((8, 56), (120, 56), color="orange")
screen.save("hello.png")
