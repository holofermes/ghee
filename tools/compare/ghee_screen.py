import ghee


class Screen:
    QUEUE = True

    def __init__(self, lay, panel=None):
        self.lay = lay
        self.panel = panel
        if panel:
            self.h = ghee.open(display="kms", dri=panel.card)
        else:
            self.h = ghee.open(width=lay.w, height=lay.h)

    def draw(self, clock, load, blink):
        lay = self.lay
        x, y, size = lay.clock
        lx, ly, lsize = lay.load
        hx, hy, hsize = lay.hot
        fx, fy, fw, fh = lay.frame
        bx, by, bw, bh = lay.bar
        width = round(bw * min(load, 1))
        with self.h.draw() as d:
            d.rect(color=lay.screen)
            d.text(
                clock,
                x=x,
                y=y,
                font_size=size,
                font_family=lay.sans,
                color=lay.phosphor,
            )
            d.text(
                f"load {load:.0%}",
                x=lx,
                y=ly,
                font_size=lsize,
                font_family=lay.pixel,
                color=lay.phosphor,
            )
            d.rect(
                x=fx,
                y=fy,
                width=fw,
                height=fh,
                color="transparent",
                border={"width": lay.border, "color": lay.phosphor},
            )
            d.rect(
                x=bx, y=by, width=width, height=bh, color=lay.mint, visible=width > 0
            )
            d.text(
                "HOT",
                x=hx,
                y=hy,
                font_size=hsize,
                font_family=lay.sans,
                color=lay.mint,
                blink={"rate": 2},
                visible=load > 0.9,
            )
        if not self.panel:
            return self.h.pixels()

    def save(self, path):
        self.h.save(path)
