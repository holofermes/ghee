import ghee

screen = ghee.open("bars.scene.json", width=128, height=64)
screen.feed("levels", [(i + 1) / 16 for i in range(16)])
screen.frame()
screen.save("ramp.png")
