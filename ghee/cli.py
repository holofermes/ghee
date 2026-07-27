"""ghee command line.

    ghee render scene.json -o frame.####.jpg --frames 500   # image sequence
    ghee render scene.json -o - --frames 300 | ffmpeg ...    # raw frames to a pipe
    ghee render scene.json --display st7789                  # to a panel, until Ctrl-C
    ghee serve scenes/                                       # editor console
    ghee serve --demo                                        # console on the bundled examples
    ghee isf import shader.fs -o shader.scene.json           # import an ISF shader
    ghee displays                                            # list display drivers

Scene arrays animate on a placeholder unless a scene controller feeds them.
"""

import argparse
import os
import signal
import sys
import time


def _size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def _parse_save_opts(s):
    """ "k=v,k2=v2" -> a dict for Image.save. Each value goes through
    ast.literal_eval (90 -> int, True -> bool) and falls back to the raw string
    when that fails."""
    import ast

    opts = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            sys.exit(f"render: --save-opts wants k=v pairs, got '{pair}'")
        k, v = pair.split("=", 1)
        try:
            opts[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            opts[k.strip()] = v.strip()
    return opts


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
    unknown = [n for n in names if n not in reg]
    if unknown:
        sys.exit(f"unknown filter: {', '.join(unknown)} (known: {', '.join(reg)})")
    apply_filters(scene, names, registry=reg)
    if a.h264:
        if a.display or a.output:
            sys.exit("render: --h264 excludes -o/--display")
        return _render_h264(a, scene, ctrl)
    if a.output == "-" and a.display:
        sys.exit("render: -o - (stdout) excludes --display")
    kms = a.display == "kms"
    if kms:
        from ghee.engine import probe_display

        size = probe_display(dri_device=a.dri, lib_path=a.lib)
        if size is None:
            sys.exit("kms: no card or no connected display")
        w, h = size
        sink = None
    elif a.display:
        from ghee.drivers import get_driver

        try:
            sink = get_driver(a.display)()
        except KeyError:
            sys.exit(f"render: no display driver '{a.display}' (see: ghee displays)")
        w, h = sink.width, sink.height
    elif a.output == "-":
        from ghee.output import RawStream

        sink = RawStream(sys.stdout.buffer)
        w, h = a.size
    elif a.output:
        from ghee.output import ImageSequence, expand_pattern

        if a.frames and a.frames > 1 and expand_pattern(a.output, 0) == a.output:
            sys.exit(
                "render: multiple frames into one filename; number them with "
                "a # run (frame.####.png)"
            )
        save_opts = _parse_save_opts(a.save_opts) if a.save_opts else None
        sink = ImageSequence(a.output, save_opts=save_opts, depth=a.write_buffer)
        w, h = a.size
    else:
        sys.exit("render: pass -o PATTERN or --display NAME")

    components = sink.components if a.display and not kms else a.components
    r = Renderer(
        w, h, components=components, lib_path=a.lib, dri_device=a.dri, display=kms
    )
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
    if kms:
        # the flip paces at the panel's refresh by itself; --fps slows it down
        if a.fps:
            from ghee.output import FixedPacer

            pacer = FixedPacer(a.fps)
    elif a.display:
        from ghee.output import AdaptivePacer, FixedPacer

        pacer = FixedPacer(a.fps) if a.fps else AdaptivePacer()
        sink.turn_on()
    # An endless display run may skip frames a controller reports clean
    # (dirty=False): no render, no sink write, the panel keeps its last frame.
    # Finite renders (--frames, -o) draw every frame regardless.
    gated = a.display and frames == 0 and ctrl is not None
    raw = a.output == "-"
    pooled = bool(a.output) and not raw  # image sink: render into its buffer pool
    i = 0
    times = []
    try:
        while frames == 0 or i < frames:
            t0 = time.monotonic()
            t = i * a.timestep if a.timestep is not None else i / 60.0
            if gated and i > 0 and not ctrl.dirty(i, t):
                i += 1
                time.sleep(pacer.idle_period())
                continue
            if ctrl is not None:
                ctrl.update(r, i, t)
            if a.timestep is not None:
                r.set_time(t)
            if kms:
                r.present(lambda name, size, i=i: provider(name, size, i))
                tw = time.monotonic()
            elif pooled:
                buf = sink.acquire((h, w, components))
                r.render(lambda name, size, i=i: provider(name, size, i), out=buf)
                tw = time.monotonic()
                sink.submit(buf)
            else:
                rgb = r.render(lambda name, size, i=i: provider(name, size, i))
                tw = time.monotonic()
                sink.write(rgb)
            if a.stats:
                times.append(time.monotonic() - t0)
            if pacer is not None:
                tn = time.monotonic()
                time.sleep(pacer.sleep_for(tn - tw, tn - t0))
            i += 1
    except KeyboardInterrupt:
        pass
    except (BrokenPipeError, ConnectionResetError):
        if not raw:
            raise
        # devnull stdout: the interpreter's exit flush would re-raise
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    finally:
        if a.display and not kms:
            sink.turn_off()
        if sink is not None:
            sink.close()
        r.close()
    if a.output and not raw:
        print(f"wrote {i} frame(s) to {a.output}")
    _print_stats(a, times)


def _print_stats(a, times):
    """--stats: one summary line after a render (paced sleeps excluded, so a
    display run reports its render+write cost, not the frame interval)."""
    if not a.stats or not times:
        return
    tot = sum(times)
    print(
        f"stats: {len(times)} frame(s) in {tot:.1f}s, "
        f"{1000 * tot / len(times):.2f} ms/frame "
        f"(min {1000 * min(times):.2f}, max {1000 * max(times):.2f}), "
        f"{len(times) / tot:.1f} fps",
        file=sys.stderr,
    )


def _render_h264(a, scene, ctrl):
    """Annex-B H264 to a file or stdout ("-"): pipe it to mpv/ffplay, record
    it, or push it through ffmpeg. Hardware zero-copy where the board has an
    encoder (Zero..Pi 4), ffmpeg/libx264 on the readback elsewhere (Pi 5,
    desktops). Every frame encodes at a fixed rate: H264's own delta coding is
    what makes static frames cheap, so dirty() gating does not apply here."""
    from ghee.console.feeds import demo_feed
    from ghee.engine import Renderer
    from ghee.h264 import HardwareH264, make_encoder

    w, h = a.size
    fps = int(a.fps) if a.fps else 30
    r = Renderer(w, h, components=4, lib_path=a.lib, dri_device=a.dri)
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

    enc = make_encoder(r, w, h, fps=fps, bitrate=a.bitrate, components=4)
    hw = isinstance(enc, HardwareH264)
    print(
        f"h264: {'hardware zero-copy' if hw else 'software libx264'} "
        f"{w}x{h}@{fps} {a.bitrate}bps -> {a.h264}",
        file=sys.stderr,
    )
    out = sys.stdout.buffer if a.h264 == "-" else open(a.h264, "wb")
    frames = a.frames if a.frames is not None else 0
    period = 1.0 / fps
    i = 0
    times = []
    try:
        while frames == 0 or i < frames:
            t0 = time.monotonic()
            t = i * a.timestep if a.timestep is not None else i / 60.0
            if ctrl is not None:
                ctrl.update(r, i, t)
            if a.timestep is not None:
                r.set_time(t)
            if hw:
                au, _ = enc.encode_frame(
                    lambda name, size, i=i: provider(name, size, i)
                )
            else:
                enc.feed(r.render(lambda name, size, i=i: provider(name, size, i)))
                au, _ = enc.get_au(timeout=2.0 if i == 0 else 0.5)
            if au:
                out.write(au)
                out.flush()
            if a.stats:
                times.append(time.monotonic() - t0)
            i += 1
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    except (BrokenPipeError, ConnectionResetError):
        # devnull stdout: the interpreter's exit flush would re-raise
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    finally:
        try:
            for au in enc.drain():
                out.write(au)
            out.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            enc.close()
            if out is not sys.stdout.buffer:
                out.close()
            r.close()
    if a.h264 != "-":
        print(f"wrote {i} frame(s) to {a.h264}", file=sys.stderr)
    _print_stats(a, times)


def cmd_serve(a):
    from ghee.console.server import run

    if a.demo and a.scenes:
        sys.exit("serve: pass a scenes directory or --demo, not both")
    if not a.demo and not a.scenes:
        sys.exit("serve: pass a scenes directory, or --demo for the bundled examples")
    size = a.size
    if size is None:
        if a.display:
            from ghee.engine import probe_display

            size = probe_display(lib_path=a.lib)
            if size is None:
                sys.exit("serve: no connected display to size from; pass --size WxH")
        else:
            size = (320, 240)
    try:
        run(
            scenes_dir=a.scenes,
            size=size,
            port=a.port,
            lib_path=a.lib,
            host=a.host,
            display=a.display,
            rgb565=a.rgb565,
            preview_max=a.preview_max,
            preview_fps=a.preview_fps,
            stream_max=a.stream_max,
        )
    except KeyboardInterrupt:
        pass


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
    from ghee.drivers import available_drivers
    from ghee.engine import probe_display

    try:
        size = probe_display()
        kms = f"connected, {size[0]}x{size[1]}" if size else "no connected display"
    except Exception as e:
        kms = f"unavailable ({e})"
    print(f"{'kms':12} {kms}")
    for name, status in available_drivers().items():
        print(f"{name:12} {status}")


def main(argv=None):
    from ghee import __version__

    p = argparse.ArgumentParser(prog="ghee")
    p.add_argument("--version", action="version", version=f"ghee {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="render a scene to images or a display")
    r.add_argument(
        "scene", help="scene file, or a name from $GHEE_SCENES / the examples"
    )
    r.add_argument(
        "-o",
        "--output",
        help="image pattern (frame.####.jpg), or '-' for raw RGB(A) frames to "
        "stdout (h*w*components, no header; pipe to ffmpeg/mpv)",
    )
    r.add_argument(
        "--display",
        help="render to a hardware display: an SPI driver name, or 'kms' for "
        "the connected HDMI/KMS display (zero-copy scanout; needs DRM master, "
        "so root and no desktop session on the card)",
    )
    r.add_argument(
        "--frames",
        type=int,
        help="frame count (default 1 for images, loop for a display)",
    )
    r.add_argument(
        "--timestep",
        type=float,
        metavar="S",
        help="pin engine time to frame*S seconds instead of the wall clock, so "
        "renders are reproducible (frame N always renders at time N*S)",
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
    r.add_argument(
        "--h264",
        metavar="PATH",
        help="write an Annex-B H264 stream to PATH ('-' = stdout, pipe to mpv); "
        "hardware zero-copy on Zero..Pi 4, ffmpeg/libx264 elsewhere",
    )
    r.add_argument(
        "--bitrate",
        type=int,
        default=4_000_000,
        help="H264 bitrate in bits/s (default 4000000)",
    )
    r.add_argument(
        "--stats",
        action="store_true",
        help="print a timing summary when the render ends",
    )
    r.set_defaults(fn=cmd_render)

    s = sub.add_parser("serve", help="run the editor console")
    s.add_argument("scenes", nargs="?", help="scenes directory")
    s.add_argument(
        "--demo", action="store_true", help="serve the bundled example scenes"
    )
    s.add_argument(
        "--size",
        type=_size,
        default=None,
        help="WxH; default is the connected display's mode with --display, "
        "else 320x240",
    )
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
    s.add_argument(
        "--stream-max",
        type=int,
        default=0,
        help="cap the /h264 stream's longest side (GPU downscale; "
        "default 0 = full render size)",
    )
    s.set_defaults(fn=cmd_serve)

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

    a = p.parse_args(argv)
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
