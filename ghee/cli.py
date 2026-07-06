"""ghee command line.

    ghee render scene.json -o frame.####.jpg --frames 500   # image sequence
    ghee render scene.json --display st7789                  # to a panel, until Ctrl-C
    ghee serve scenes/                                       # editor console
    ghee displays                                            # list display drivers

Scene arrays get an animated placeholder; no real input is wired yet.
"""

import argparse
import sys


def _size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def cmd_render(a):
    from ghee import load_scene
    from ghee.console.feeds import demo_feed
    from ghee.engine import Renderer

    scene, ctrl = load_scene(a.scene)
    if a.display:
        from ghee.drivers import get_driver

        sink = get_driver(a.display)()
        w, h = sink.width, sink.height
    elif a.output:
        from ghee.output import ImageSequence

        sink = ImageSequence(a.output)
        w, h = a.size
    else:
        sys.exit("render: pass -o PATTERN or --display NAME")

    components = sink.components if a.display else a.components
    r = Renderer(w, h, components=components, lib_path=a.lib, dri_device=a.dri)
    err = r.setup(scene)
    if err:
        sys.exit(f"compile error: {err}")
    if ctrl is not None:
        ctrl.setup(r)

    def provider(name, size, frame):
        if ctrl is not None:
            data = ctrl.provide(name, size, frame)
            if data is not None:
                return data
        return demo_feed(name, size, frame)

    frames = a.frames if a.frames is not None else (0 if a.display else 1)
    if a.display:
        sink.turn_on()
    i = 0
    try:
        while frames == 0 or i < frames:
            if ctrl is not None:
                ctrl.update(r, i, i / 60.0)
            rgb = r.render(lambda name, size, i=i: provider(name, size, i))
            sink.write(rgb)
            i += 1
    except KeyboardInterrupt:
        pass
    finally:
        if a.display:
            sink.turn_off()
        sink.close()
        r.close()
    if a.output:
        print(f"wrote {i} frame(s) to {a.output}")


def cmd_serve(a):
    from ghee.console.server import run

    run(scenes_dir=a.scenes, size=a.size, port=a.port, lib_path=a.lib, host=a.host)


def cmd_displays(a):
    from ghee.drivers import available_drivers

    for name, status in available_drivers().items():
        print(f"{name:12} {status}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="ghee")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="render a scene to images or a display")
    r.add_argument("scene")
    r.add_argument("-o", "--output", help="image pattern, e.g. frame.####.jpg")
    r.add_argument("--display", help="render to a hardware display by name")
    r.add_argument(
        "--frames",
        type=int,
        help="frame count (default 1 for images, loop for a display)",
    )
    r.add_argument(
        "--size", type=_size, default=(320, 240), help="WxH for image output"
    )
    r.add_argument(
        "--components", type=int, default=3, choices=(3, 4), help="3=RGB, 4=RGBA"
    )
    r.add_argument("--dri", help="DRI device, e.g. /dev/dri/renderD128")
    r.add_argument("--lib", help="path to libghee.so")
    r.set_defaults(fn=cmd_render)

    s = sub.add_parser("serve", help="run the editor console")
    s.add_argument("scenes", nargs="?", help="scenes directory")
    s.add_argument("--size", type=_size, default=(320, 240))
    s.add_argument("--port", type=int, default=8090)
    s.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; 0.0.0.0 exposes the console (no auth) to the network",
    )
    s.add_argument("--lib")
    s.set_defaults(fn=cmd_serve)

    d = sub.add_parser("displays", help="list display drivers and availability")
    d.set_defaults(fn=cmd_displays)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
