"""ghee command line.

    ghee render scene.json -o frame.####.jpg --frames 500   # image sequence
    ghee render scene.json -o - --frames 300 | ffmpeg ...    # raw frames to a pipe
    ghee render scene.json --display st7789                  # to a panel, until Ctrl-C
    ghee render scene.json --h264 - | mpv -                  # H264 to a pipe
    ghee render scene.json --display st7789 --serve           # panel plus the console
    ghee serve scenes/                                       # editor console
    ghee serve --demo                                        # console on the bundled examples
    ghee render --demo bars -o frame.png                     # a bundled example by name
    ghee isf import shader.fs -o shader.scene.json           # import an ISF shader
    ghee displays                                            # list display drivers

Outputs combine: -o, --display and --h264 are a list of sinks one render feeds,
and --serve attaches the console to any of them. `serve` is `render --serve`
over a directory of scenes.

Everything after `--` goes to the scene's own controller, and `-- --help` lists
what it takes:

    ghee render --demo text -- --text "hello there"
    ghee render --demo bluesun --display kms -- --emit 480x300

Scene arrays animate on a placeholder unless a scene controller feeds them.
"""

import argparse
import os
import signal
import sys


def _size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def _parse_kv(s, what):
    """ "k=v,k2=v2" -> a dict. Each value goes through ast.literal_eval (90 ->
    int, True -> bool) and falls back to the raw string when that fails. `what`
    names the flag a parse error came from."""
    import ast

    opts = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            sys.exit(f"{what} wants k=v pairs, got '{pair}'")
        k, v = pair.split("=", 1)
        try:
            opts[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            opts[k.strip()] = v.strip()
    return opts


def _split_scene_args(argv):
    """(ghee's own argv, the scene's tokens). The first `--` ends ghee's flags:
    everything past it belongs to the scene, further `--` included."""
    if "--" not in argv:
        return argv, []
    i = argv.index("--")
    return argv[:i], argv[i + 1 :]


def _find_scene(arg, demo=False):
    """A path as given, or a scene name looked up in $GHEE_SCENES (colon-
    separated directories), plus the bundled examples when --demo asks for
    them."""

    def hit(stem):
        return any(
            os.path.exists(p) for p in (stem, stem + ".scene.json", stem + ".py")
        )

    if hit(arg):
        return arg
    dirs = [d for d in os.environ.get("GHEE_SCENES", "").split(":") if d]
    if demo:
        from ghee.console import examples_dir

        dirs.append(examples_dir())
    for d in dirs:
        p = os.path.join(d, arg)
        if hit(p):
            return p
    sys.exit(f"no scene '{arg}' (searched: {', '.join(['cwd'] + dirs)})")


def _scene(a):
    """The scene this invocation renders, with its filters applied: the ones it
    declares, then whatever --filters adds."""
    from ghee import apply_filters, load_filters, load_scene

    scene, ctrl = load_scene(_find_scene(a.scene, a.demo), a.scene_args)
    names = list(scene.filters)
    if a.filters:
        names += [n.strip() for n in a.filters.split(",") if n.strip()]
    names = list(dict.fromkeys(names))
    reg = load_filters()
    unknown = [n for n in names if n not in reg]
    if unknown:
        sys.exit(f"unknown filter: {', '.join(unknown)} (known: {', '.join(reg)})")
    apply_filters(scene, names, registry=reg)
    return scene, ctrl


def _panel(a):
    """What --display names: the string "kms" for the scanout, else the driver
    instance, built with whatever --display-opts sets."""
    if not a.display:
        if a.display_opts:
            sys.exit(f"{a.cmd}: --display-opts without --display")
        return None
    if a.display == "kms":
        if a.display_opts:
            sys.exit(f"{a.cmd}: --display-opts with --display kms; kms has no driver")
        return "kms"
    from ghee.drivers import driver_options, get_driver

    try:
        cls = get_driver(a.display)
    except KeyError:
        # --display takes an optional value, so a path written after it is read
        # as a driver name
        hint = (
            ", and a scenes directory goes before it"
            if os.path.isdir(a.display)
            else ""
        )
        sys.exit(f"{a.cmd}: no display driver '{a.display}' (see: ghee displays){hint}")
    if not a.display_opts:
        return cls()
    opts = _parse_kv(a.display_opts, f"{a.cmd}: --display-opts")
    # checked before the driver runs: a constructor claims GPIO lines and the
    # SPI bus before it reaches a keyword it does not take
    known = driver_options(cls)
    unknown = [k for k in opts if k not in known]
    if unknown:
        sys.exit(
            f"{a.cmd}: --display-opts: {a.display} has no option "
            f"{', '.join(unknown)} (takes: {', '.join(known)})"
        )
    return cls(**opts)


def _geometry(a, panel):
    """(size, components) the render runs at. A scanout renders at the connected
    mode and a panel at its own geometry and pixel layout, either of which wins
    over --size; everything else takes --size."""
    if panel == "kms":
        from ghee.engine import probe_display

        size = probe_display(dri_device=a.dri, lib_path=a.lib) or a.size
        if size is None:
            sys.exit(f"{a.cmd}: no connected display to size from; pass --size WxH")
        return size, a.components
    if panel is not None:
        return (panel.width, panel.height), panel.components
    return a.size or (320, 240), a.components


def _file_sink(a):
    """The -o sink: raw frames to stdout, or numbered image files."""
    if a.output == "-":
        from ghee.output import RawStream

        return RawStream(sys.stdout.buffer)
    if not a.output:
        return None
    from ghee.output import ImageSequence, expand_pattern

    if a.frames and a.frames > 1 and expand_pattern(a.output, 0) == a.output:
        sys.exit(
            "render: multiple frames into one filename; number them with "
            "a # run (frame.####.png)"
        )
    save_opts = _parse_kv(a.save_opts, f"{a.cmd}: --save-opts") if a.save_opts else None
    return ImageSequence(a.output, save_opts=save_opts, depth=a.write_buffer)


def _pacer(a, panel):
    """Nothing paces a file or a pipe: that render runs as fast as it draws. A
    panel paces on its own bus speed, the KMS flip paces itself at the display's
    refresh, and an H264 stream holds a fixed rate for its timestamps."""
    from ghee.output import AdaptivePacer, FixedPacer

    if panel == "kms":
        return FixedPacer(a.fps) if a.fps else None
    if panel is not None:
        return FixedPacer(a.fps) if a.fps else AdaptivePacer()
    if a.h264:
        return FixedPacer(_h264_fps(a))
    return None


def _h264_fps(a):
    return int(a.fps) if a.fps else 30


def _frames(a):
    """--frames as given, else one image and an endless display, stream or
    console."""
    if a.frames is not None:
        return a.frames
    return 0 if (a.display or a.h264 or a.serve) else 1


def cmd_render(a):
    """One render into however many sinks the flags ask for."""
    from ghee.engine import Renderer
    from ghee.feeds import demo_feed
    from ghee.loop import Loop
    from ghee.output import Scanout

    if a.serve and a.filters:
        sys.exit(f"{a.cmd}: --filters with --serve; the console has a filter picker")
    if a.serve and a.h264:
        sys.exit(f"{a.cmd}: --h264 with --serve; the console serves /h264 already")
    if a.output == "-" and a.h264 == "-":
        sys.exit(f"{a.cmd}: -o - and --h264 - both want stdout")
    # the scene loads first: it owns the tokens after `--`, including their
    # --help, and a scene the loader rejects must fail before a panel is opened
    scene = None if a.serve else _scene(a)
    panel = _panel(a)
    kms = panel == "kms"
    if panel is None and not a.output and not a.h264 and not a.serve:
        sys.exit(f"{a.cmd}: pass -o PATTERN, --display NAME, --h264 PATH or --serve")
    size, components = _geometry(a, panel)
    sinks = [] if panel is None else [Scanout() if kms else panel]
    out = _file_sink(a)
    if out is not None:
        sinks.append(out)
    if a.serve:
        # the console creates the Renderer on its own render thread (GL belongs
        # to the thread that draws), so the sinks are all that is handed over
        return _serve(a, [s for s in sinks if not s.scanout], size, components, kms)

    scene, ctrl = scene
    w, h = size
    r = Renderer(
        w,
        h,
        components=components,
        lib_path=a.lib,
        dri_device=a.dri,
        display=kms,
        rgb565=a.rgb565,
    )
    err = r.setup(scene)
    if err:
        sys.exit(f"compile error: {err}")
    if ctrl is not None:
        ctrl.setup(r)
    stream = None
    if a.h264:
        from ghee.h264 import H264Stream

        fps = _h264_fps(a)
        stream = sys.stdout.buffer if a.h264 == "-" else open(a.h264, "wb")
        h264 = H264Stream(
            stream, r, w, h, fps=fps, bitrate=a.bitrate, components=components
        )
        print(
            f"h264: {'hardware zero-copy' if h264.hw else 'software libx264'} "
            f"{w}x{h}@{fps} {a.bitrate}bps -> {a.h264}",
            file=sys.stderr,
        )
        sinks.append(h264)
    times = []
    loop = Loop(
        r,
        sinks,
        controller=ctrl,
        provider=demo_feed,
        frames=_frames(a),
        timestep=a.timestep,
        pacer=_pacer(a, panel),
        on_frame=(lambda lp, cost: times.append(cost)) if a.stats else None,
    )
    pipe = "-" in (a.output, a.h264)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    except (BrokenPipeError, ConnectionResetError):
        if not pipe:
            raise
        # devnull stdout: the interpreter's exit flush would re-raise
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    finally:
        for s in sinks:
            s.close()
        if stream is not None and stream is not sys.stdout.buffer:
            stream.close()
        r.close()
    if a.output and a.output != "-":
        print(f"wrote {loop.drawn} frame(s) to {a.output}")
    if a.h264 and a.h264 != "-":
        print(f"wrote {loop.drawn} frame(s) to {a.h264}", file=sys.stderr)
    _print_stats(a, times, loop.elapsed, loop.out)


def _serve(a, sinks, size, components, kms):
    """Attach the console to this render. A scene serves itself alone; a
    directory (or --demo) serves everything in it."""
    from ghee.console.server import run

    scenes_dir, store = None, None
    if a.scene is not None:
        path = _find_scene(a.scene, a.demo)
        if os.path.isdir(path):
            scenes_dir = path
        else:
            store = _one_scene_store(a, path)
    try:
        run(
            scenes_dir=scenes_dir,
            store=store,
            size=size,
            components=components,
            sinks=sinks,
            display=kms,
            rgb565=a.rgb565,
            port=a.port,
            host=a.host,
            lib_path=a.lib,
            dri_device=a.dri,
            preview_max=a.preview_max,
            preview_fps=a.preview_fps,
            stream_max=a.stream_max,
            bitrate=a.bitrate,
            scene_args=a.scene_args,
            frames=_frames(a),
            timestep=a.timestep,
        )
    except KeyboardInterrupt:
        pass


def _one_scene_store(a, path):
    """A store holding the single scene `render --serve` was pointed at. The
    bundled examples are never written, whichever way they are reached."""
    from ghee.console import examples_dir
    from ghee.console.core import NotFound
    from ghee.console.store import SceneStore

    bundled = os.path.dirname(os.path.abspath(path)) == os.path.abspath(examples_dir())
    try:
        return SceneStore.one(path, read_only=bundled, args=a.scene_args or ())
    except NotFound as e:
        sys.exit(f"{a.cmd}: {e}")


def cmd_serve(a):
    """`serve` is `render --serve` over a directory of scenes."""
    if a.demo and a.scenes:
        sys.exit("serve: pass a scenes directory or --demo, not both")
    if not a.demo and not a.scenes:
        if a.display and os.path.isdir(a.display):
            # --display takes an optional value, so it swallowed the directory
            sys.exit(
                f"serve: --display took '{a.display}' as the display name; "
                f"put the scenes directory before it"
            )
        sys.exit("serve: pass a scenes directory, or --demo for the bundled examples")
    a.scene = a.scenes  # --demo leaves it None: the bundled examples, read-only
    return cmd_render(a)


def _print_stats(a, times, wall, out):
    """--stats: one summary line after a render. ms/frame is the per-frame cost
    with paced sleeps excluded; fps is frames over real elapsed time, the same
    number the console publishes."""
    from ghee.stats import rate

    if not a.stats or not times:
        return
    tot = sum(times)
    print(
        f"stats: {len(times)} frame(s) in {wall:.2f}s, "
        f"{1000 * tot / len(times):.2f} ms/frame "
        f"(min {1000 * min(times):.2f}, max {1000 * max(times):.2f}), "
        f"{rate(len(times), wall):.1f} fps -> {'+'.join(out)}",
        file=sys.stderr,
    )


def cmd_isf_import(a):
    import json

    from ghee.isf import ISFError, read_isf, slug, to_filter, to_scene

    try:
        text, name, vs = read_isf(a.source)
    except OSError as e:
        sys.exit(f"isf import: {a.source}: {e.strerror or e}")
    try:
        if a.as_filter:
            out, warnings = to_filter(text, name, a.size, vs)
            body = json.dumps(out, indent=2)
            what = f"filter '{out['name']}'"
        else:
            scene, warnings = to_scene(text, name, a.size, vs)
            body = json.dumps(scene.to_dict(), indent=2)
            what = f"scene, {len(scene.passes)} pass(es)"
    except ISFError as e:
        sys.exit(f"isf import: {e}")
    for w in warnings:
        print(f"isf import: {name}: {w}", file=sys.stderr)
    path = a.output or f"{slug(name)}.{'filter' if a.as_filter else 'scene'}.json"
    with open(path, "w", encoding="utf-8") as f:
        f.write(body + "\n")
    print(f"wrote {what} to {path}")


def cmd_displays(a):
    import inspect

    from ghee.drivers import available_drivers, driver_options, get_driver
    from ghee.engine import probe_display

    try:
        size = probe_display()
        kms = f"connected, {size[0]}x{size[1]}" if size else "no connected display"
    except Exception as e:
        kms = f"unavailable ({e})"
    print(f"{'kms':12} {kms}")
    for name, status in available_drivers().items():
        print(f"{name:12} {status}")
        if status != "ok":
            continue
        opts = ",".join(
            k if v is inspect.Parameter.empty else f"{k}={v!r}"
            for k, v in driver_options(get_driver(name)).items()
        )
        if opts:
            print(f"{'':12} --display-opts {opts}")


def _render_flags(p):
    """What every render takes, whichever command asked for it."""
    p.add_argument(
        "--display",
        nargs="?",
        const="kms",
        metavar="NAME",
        help="render to a hardware display: an SPI/I2C driver name, or bare "
        "(or 'kms') for the connected HDMI/KMS display (zero-copy scanout; "
        "needs DRM master)",
    )
    p.add_argument(
        "--display-opts",
        metavar="K=V,...",
        help="options for the --display driver, e.g. width=240,height=240,"
        "y_offset=80,backlight=26; `ghee displays` lists what each one takes",
    )
    p.add_argument(
        "--rgb565",
        action="store_true",
        help="display scanout in 16-bit color: halves the GPU store and scanout "
        "bandwidth (worth it on VC4-class boards); slight banding in gradients",
    )
    p.add_argument(
        "--frames",
        type=int,
        help="frame count (default 1 for images, loop for a display, a stream "
        "or the console)",
    )
    p.add_argument(
        "--timestep",
        type=float,
        metavar="S",
        help="pin engine time to frame*S seconds instead of the wall clock, so "
        "renders are reproducible (frame N always renders at time N*S)",
    )
    p.add_argument(
        "--size",
        type=_size,
        default=None,
        help="WxH to render at; a display or panel sets its own (default 320x240)",
    )
    p.add_argument("--dri", help="DRI device, e.g. /dev/dri/renderD128")
    p.add_argument("--lib", help="path to libghee.so")


def _console_flags(p, bitrate):
    """The console's own flags: what `serve` offers and `render --serve` needs."""
    p.add_argument("--port", type=int, default=8090)
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; 0.0.0.0 exposes the console (no auth) to the network",
    )
    p.add_argument(
        "--preview-max",
        type=int,
        default=480,
        help="longest side of the browser preview in px (downscaled; default 480)",
    )
    p.add_argument(
        "--preview-fps",
        type=int,
        default=15,
        help="browser preview refresh cap; the display stays full speed (default 15)",
    )
    p.add_argument(
        "--stream-max",
        type=int,
        default=0,
        help="cap the /h264 stream's longest side (GPU downscale; "
        "default 0 = full render size)",
    )
    p.add_argument(
        "--bitrate",
        type=int,
        default=bitrate,
        help=f"H264 bitrate in bits/s: the --h264 stream's rate, or the floor "
        f"for the console's /h264 (default {bitrate})",
    )


def _parser():
    from ghee import __version__

    p = argparse.ArgumentParser(prog="ghee")
    p.add_argument("--version", action="version", version=f"ghee {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser(
        "render",
        help="render a scene to images, a display, a stream, or several at once",
        epilog="anything after `--` goes to the scene's controller "
        "(ghee render --demo text -- --text hi); `-- --help` lists what it takes",
    )
    r.add_argument("scene", help="scene file, or a name from $GHEE_SCENES")
    r.add_argument(
        "--demo",
        action="store_true",
        help="look the scene name up in the bundled examples too",
    )
    r.add_argument(
        "-o",
        "--output",
        help="image pattern (frame.####.jpg), or '-' for raw RGB(A) frames to "
        "stdout (h*w*components, no header; pipe to ffmpeg/mpv)",
    )
    r.add_argument(
        "--h264",
        metavar="PATH",
        help="write an Annex-B H264 stream to PATH ('-' = stdout, pipe to mpv); "
        "hardware zero-copy on Pi 1..Pi 4, ffmpeg/libx264 elsewhere",
    )
    r.add_argument(
        "--serve",
        action="store_true",
        help="also run the editor console on this render (see: ghee serve)",
    )
    r.add_argument(
        "--components",
        type=int,
        default=4,
        choices=(3, 4),
        help="3=RGB, 4=RGBA (desktop Mesa can only read back RGBA)",
    )
    r.add_argument(
        "--save-opts",
        metavar="K=V,...",
        help="extra options passed to the image encoder, e.g. quality=90 or "
        "lossless=True (ignored for -o -)",
    )
    r.add_argument(
        "--write-buffer",
        type=int,
        default=16,
        metavar="N",
        help="frames buffered for the async image writer; the render only waits "
        "when the writer falls N frames behind; memory cost is N * w * h * "
        "components bytes",
    )
    r.add_argument(
        "--fps",
        type=float,
        help="fixed display frame rate (default: adapt to the panel's own speed)",
    )
    r.add_argument(
        "--filters",
        help="comma-separated filter names applied after the scene's own",
    )
    r.add_argument(
        "--stats",
        action="store_true",
        help="print a timing summary when the render ends",
    )
    _render_flags(r)
    _console_flags(r, bitrate=4_000_000)
    r.set_defaults(fn=cmd_render)

    s = sub.add_parser(
        "serve",
        help="run the editor console",
        epilog="the console is plain HTTP: GET /stats /scene /scenes /frame.png "
        "/stream /h264, POST /set /switch /next /prev /scene /save /ping. "
        "Anything after `--` goes to every scene the console loads, and "
        "each keeps what it declares; `-- --help` lists what takes what",
    )
    s.add_argument("scenes", nargs="?", help="scenes directory")
    s.add_argument(
        "--demo", action="store_true", help="serve the bundled example scenes"
    )
    _render_flags(s)
    _console_flags(s, bitrate=10_000_000)
    # serve is render --serve over a directory, so it carries render's whole
    # namespace with the flags it does not offer left at their defaults
    s.set_defaults(
        fn=cmd_serve,
        serve=True,
        output=None,
        h264=None,
        components=4,
        save_opts=None,
        write_buffer=16,
        fps=None,
        filters=None,
        stats=False,
    )

    i = sub.add_parser("isf", help="Interactive Shader Format (ISF) shaders")
    isub = i.add_subparsers(dest="isf_cmd", required=True)
    imp = isub.add_parser("import", help="convert an ISF shader to a scene or filter")
    imp.add_argument("source", help="a .fs file, or an http(s) URL to one")
    imp.add_argument(
        "-o",
        "--output",
        help="output path (default: the shader's name, .scene.json or .filter.json)",
    )
    imp.add_argument(
        "--size",
        type=_size,
        default=(320, 240),
        help="WxH the pass WIDTH/HEIGHT equations are evaluated against, since "
        "ghee pass sizes are fixed at import (default 320x240)",
    )
    imp.add_argument(
        "--as-filter",
        action="store_true",
        help="write a ghee filter (a single-pass ISF whose image input becomes "
        "iChannel0, the frame so far) instead of a scene",
    )
    imp.set_defaults(fn=cmd_isf_import)

    d = sub.add_parser("displays", help="list display drivers and availability")
    d.set_defaults(fn=cmd_displays)

    return p


def main(argv=None):
    p = _parser()
    argv, scene_args = _split_scene_args(list(sys.argv[1:] if argv is None else argv))
    a = p.parse_args(argv)
    if scene_args and a.cmd not in ("render", "serve"):
        sys.exit(f"{a.cmd}: takes no scene arguments after --")
    a.scene_args = scene_args
    # SIGTERM (timeout, systemd stop) must unwind like Ctrl-C so the finally
    # blocks run and a display sink gets turned off, not left on a stale frame.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        a.fn(a)
    except (ValueError, RuntimeError, OSError, ImportError) as e:
        # scene, engine, encoder and missing-dependency failures reach the user
        # the way the argument errors above do: one line, no stack. Set
        # $GHEE_TRACEBACK to get the stack back.
        if os.environ.get("GHEE_TRACEBACK"):
            raise
        sys.exit(f"{a.cmd}: {e}")


if __name__ == "__main__":
    main()
