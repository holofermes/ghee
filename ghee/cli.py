"""ghee command line.

    ghee render scene.json -o frame.####.jpg --frames 500   # image sequence
    ghee render scene.json --display st7789                  # to a panel, until Ctrl-C
    ghee serve scenes/                                       # editor console
    ghee serve --demo                                        # console on the bundled examples
    ghee displays                                            # list display drivers

Scene arrays get an animated placeholder; no real input is wired yet.
"""

import argparse
import os
import signal
import sys
import time


def _size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def _find_scene(arg):
    """A path as given, or a scene name looked up in $GHEE_SCENES (colon-
    separated directories), then the bundled examples."""

    def hit(stem):
        return any(
            os.path.exists(p) for p in (stem, stem + ".scene.json", stem + ".py")
        )

    if hit(arg):
        return arg
    from ghee.console import examples_dir

    dirs = [d for d in os.environ.get("GHEE_SCENES", "").split(":") if d]
    dirs.append(examples_dir())
    for d in dirs:
        p = os.path.join(d, arg)
        if hit(p):
            return p
    sys.exit(f"no scene '{arg}' (searched: cwd, {', '.join(dirs)})")


def cmd_render(a):
    from ghee import apply_filters, load_filters, load_scene
    from ghee.console.feeds import demo_feed
    from ghee.engine import Renderer

    scene, ctrl = load_scene(_find_scene(a.scene))
    names = list(scene.filters)
    if a.filters:
        names += [n.strip() for n in a.filters.split(",") if n.strip()]
    names = list(dict.fromkeys(names))
    reg = load_filters()
    unknown = [n for n in names if n not in reg and n not in scene.filters]
    if unknown:
        sys.exit(f"unknown filter: {', '.join(unknown)} (known: {', '.join(reg)})")
    apply_filters(scene, names, registry=reg)
    if a.display:
        from ghee.drivers import get_driver

        sink = get_driver(a.display)()
        w, h = sink.width, sink.height
    elif a.output:
        from ghee.output import ImageSequence, expand_pattern

        if a.frames and a.frames > 1 and expand_pattern(a.output, 0) == a.output:
            sys.exit(
                "render: multiple frames into one filename; number them with "
                "a # run (frame.####.png)"
            )
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
    pacer = None
    if a.display:
        from ghee.output import AdaptivePacer, FixedPacer

        pacer = FixedPacer(a.fps) if a.fps else AdaptivePacer()
        sink.turn_on()
    # An endless display run may skip frames a controller reports clean
    # (dirty=False): no render, no sink write, the panel keeps its last frame.
    # Finite renders (--frames, -o) draw every frame regardless.
    gated = a.display and frames == 0 and ctrl is not None
    i = 0
    try:
        while frames == 0 or i < frames:
            t0 = time.monotonic()
            if gated and i > 0 and not ctrl.dirty(i, i / 60.0):
                i += 1
                time.sleep(pacer.idle_period())
                continue
            if ctrl is not None:
                ctrl.update(r, i, i / 60.0)
            rgb = r.render(lambda name, size, i=i: provider(name, size, i))
            tw = time.monotonic()
            sink.write(rgb)
            if pacer is not None:
                tn = time.monotonic()
                time.sleep(pacer.sleep_for(tn - tw, tn - t0))
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

    if a.demo and a.scenes:
        sys.exit("serve: pass a scenes directory or --demo, not both")
    if not a.demo and not a.scenes:
        sys.exit("serve: pass a scenes directory, or --demo for the bundled examples")
    run(
        scenes_dir=a.scenes,
        size=a.size,
        port=a.port,
        lib_path=a.lib,
        host=a.host,
        display=a.display,
        rgb565=a.rgb565,
        preview_max=a.preview_max,
        preview_fps=a.preview_fps,
    )


def cmd_displays(a):
    from ghee.drivers import available_drivers

    for name, status in available_drivers().items():
        print(f"{name:12} {status}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="ghee")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="render a scene to images or a display")
    r.add_argument(
        "scene", help="scene file, or a name from $GHEE_SCENES / the examples"
    )
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
        "--components",
        type=int,
        default=4,
        choices=(3, 4),
        help="3=RGB, 4=RGBA (desktop Mesa can only read back RGBA)",
    )
    r.add_argument("--dri", help="DRI device, e.g. /dev/dri/renderD128")
    r.add_argument("--lib", help="path to libghee.so")
    r.add_argument(
        "--fps",
        type=float,
        help="fixed display frame rate (default: adapt to the panel's own speed)",
    )
    r.add_argument(
        "--filters",
        help="comma-separated filter names applied after the scene's own",
    )
    r.set_defaults(fn=cmd_render)

    s = sub.add_parser("serve", help="run the editor console")
    s.add_argument("scenes", nargs="?", help="scenes directory")
    s.add_argument(
        "--demo", action="store_true", help="serve the bundled example scenes"
    )
    s.add_argument("--size", type=_size, default=(320, 240))
    s.add_argument("--port", type=int, default=8090)
    s.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; 0.0.0.0 exposes the console (no auth) to the network",
    )
    s.add_argument("--lib")
    s.add_argument(
        "--display",
        action="store_true",
        help="also scan out to a connected display (headless Pi, run as root)",
    )
    s.add_argument(
        "--rgb565",
        action="store_true",
        help="display scanout in 16-bit color: halves the GPU store and scanout "
        "bandwidth (worth it on VC4-class boards); slight banding in gradients",
    )
    s.add_argument(
        "--preview-max",
        type=int,
        default=480,
        help="longest side of the browser preview in px (downscaled; default 480)",
    )
    s.add_argument(
        "--preview-fps",
        type=int,
        default=15,
        help="browser preview refresh cap; the display stays full speed (default 15)",
    )
    s.set_defaults(fn=cmd_serve)

    d = sub.add_parser("displays", help="list display drivers and availability")
    d.set_defaults(fn=cmd_displays)

    a = p.parse_args(argv)
    # SIGTERM (timeout, systemd stop) must unwind like Ctrl-C so the finally
    # blocks run and a display sink gets turned off, not left on a stale frame.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    a.fn(a)


if __name__ == "__main__":
    main()
