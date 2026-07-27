"""H264 output. Two backends behind one shape, both emitting Annex-B access
units, so the CLI pipe, the serve /h264 endpoint, and mpv see identical bytes:

- HardwareH264: zero-copy GL -> linear dmabuf -> V4L2 M2M (bcm2835-codec,
  /dev/video11, original Zero through Pi 4). The render itself lands in the
  encoder's buffer; no glReadPixels anywhere.
- SoftwareH264: an ffmpeg/libx264 subprocess fed the readback frames (Pi 5 has
  no hardware video encoder at all; desktops likewise via this path).
"""

import base64
import ctypes
import json
import queue
import shutil
import struct
import subprocess
import threading

NAL_SLICE, NAL_IDR, NAL_SEI, NAL_SPS, NAL_PPS, NAL_AUD = 1, 5, 6, 7, 8, 9

# WebCodecs VideoDecoderConfig.colorSpace per backend. The VPU converts with a
# fixed BT.601 full-range matrix and writes no VUI, so unhinted browsers guess
# BT.709 limited. The software lane is x264 tagged BT.601 limited.
HW_COLOR = {
    "primaries": "bt709",
    "transfer": "iec61966-2-1",
    "matrix": "smpte170m",
    "fullRange": True,
}
SW_COLOR = {
    "primaries": "bt709",
    "transfer": "bt709",
    "matrix": "smpte170m",
    "fullRange": False,
}


def split_annexb(au: bytes):
    """Split an Annex-B access unit into raw NAL units (no start codes).
    find() keeps the scan in C; a byte-at-a-time Python loop costs
    milliseconds per AU on small ARM cores. A 4-byte start code's leading 00
    lands on the previous NAL's tail and is dropped by the rstrip, same as
    trailing_zero_8bits padding."""
    nals = []
    i = au.find(b"\x00\x00\x01")
    while i >= 0:
        start = i + 3
        j = au.find(b"\x00\x00\x01", start)
        end = j if j >= 0 else len(au)
        nals.append(au[start:end].rstrip(b"\x00"))
        i = j
    return [x for x in nals if x]


def avcc(nals):
    """Length-prefixed (AVCC) concatenation of NAL units, for VideoDecoder."""
    return b"".join(struct.pack(">I", len(x)) + x for x in nals)


def build_config(sps: bytes, pps: bytes, w: int, h: int, color=None) -> bytes:
    """WebCodecs decoder config JSON: codec string + avcC description."""
    codec = f"avc1.{sps[1]:02x}{sps[2]:02x}{sps[3]:02x}"
    box = (
        bytes([1, sps[1], sps[2], sps[3], 0xFF, 0xE1])
        + struct.pack(">H", len(sps))
        + sps
        + bytes([1])
        + struct.pack(">H", len(pps))
        + pps
    )
    cfg = {
        "codec": codec,
        "description": base64.b64encode(box).decode(),
        "codedWidth": w,
        "codedHeight": h,
    }
    if color:
        cfg["colorSpace"] = color
    return json.dumps(cfg).encode()


def wire_frame(t: int, payload: bytes) -> bytes:
    """Browser wire framing: [u8 type][u32 BE len][payload]."""
    return bytes([t]) + struct.pack(">I", len(payload)) + payload


class HardwareH264:
    """Zero-copy V4L2 encoder bound to a Renderer's GL context. All methods
    must run on the renderer's thread (they issue GL calls)."""

    def __init__(self, renderer, w, h, fps=30, bitrate=4_000_000):
        self.r = renderer
        self.w, self.h = w, h
        err = ctypes.c_char_p()
        self.handle = renderer.lib.ghee_encoder_create(
            renderer.handle, w, h, fps, bitrate, ctypes.byref(err)
        )
        if not self.handle:
            raise RuntimeError(
                (err.value or b"encoder create failed").decode(errors="replace")
            )

    def _collect(self, rc, nal, ln, key):
        if rc < 0:
            raise RuntimeError(f"hardware encode failed: {self.r.last_error()}")
        if rc > 0:
            return None, False  # pipeline priming: no compressed output yet
        return ctypes.string_at(nal, ln.value), bool(key.value)

    def encode_frame(self, provider=None):
        """Render the scene and encode it, no readback. -> (au|None, is_key)"""
        self.r._feed(provider)
        nal = ctypes.POINTER(ctypes.c_ubyte)()
        ln, key = ctypes.c_int(), ctypes.c_int()
        rc = self.r.lib.ghee_mp_render_encoded(
            self.r.handle,
            self.handle,
            ctypes.byref(nal),
            ctypes.byref(ln),
            ctypes.byref(key),
        )
        return self._collect(rc, nal, ln, key)

    def encode_last(self):
        """Encode the frame the last render()/present() produced (display
        mirror: one render feeds screen and stream). -> (au|None, is_key)"""
        nal = ctypes.POINTER(ctypes.c_ubyte)()
        ln, key = ctypes.c_int(), ctypes.c_int()
        rc = self.r.lib.ghee_encoder_encode_last(
            self.r.handle,
            self.handle,
            ctypes.byref(nal),
            ctypes.byref(ln),
            ctypes.byref(key),
        )
        return self._collect(rc, nal, ln, key)

    def force_keyframe(self):
        self.r.lib.ghee_encoder_force_keyframe(self.handle)

    def drain(self):
        """End of stream: yield the access units the encoder still owes, one per
        flush, until it reports drained. Without it a capture ends mid-GOP and
        loses its last frames."""
        if not getattr(self, "handle", None):
            return
        nal = ctypes.POINTER(ctypes.c_ubyte)()
        ln, key = ctypes.c_int(), ctypes.c_int()
        while True:
            rc = self.r.lib.ghee_encoder_flush(
                self.r.handle,
                self.handle,
                ctypes.byref(nal),
                ctypes.byref(ln),
                ctypes.byref(key),
            )
            if rc < 0:
                raise RuntimeError(f"hardware flush failed: {self.r.last_error()}")
            if rc > 0:
                return
            yield ctypes.string_at(nal, ln.value)

    def close(self):
        if getattr(self, "handle", None):
            self.r.lib.ghee_encoder_destroy(self.r.handle, self.handle)
            self.handle = None


class SoftwareH264:
    """ffmpeg/libx264 subprocess. feed() it readback frames; get_au() returns
    complete access units (x264 runs with aud=1 so AU boundaries are explicit)."""

    def __init__(self, w, h, fps=30, bitrate=4_000_000, components=4):
        if not shutil.which("ffmpeg"):
            raise RuntimeError("no ffmpeg on PATH (needed for software H264)")
        self.w, self.h = w, h
        pix = "rgba" if components == 4 else "rgb24"
        gop = fps * 2
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                pix,
                "-s",
                f"{w}x{h}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-pix_fmt",
                "yuv420p",  # 4:4:4 would break WebCodecs/hw decoders
                # pin the RGB->YUV matrix/range and tag the stream to match;
                # untagged streams decode with guessed colorimetry
                "-vf",
                "scale=out_color_matrix=bt601:out_range=tv",
                "-colorspace",
                "smpte170m",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-color_range",
                "tv",
                "-x264-params",
                f"aud=1:keyint={gop}:min-keyint={gop}",
                "-b:v",
                str(bitrate),
                "-f",
                "h264",
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        self._q = queue.Queue(maxsize=256)
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self):
        """Split the raw stream into access units at AUD start codes."""
        buf = b""
        try:
            while True:
                chunk = self.proc.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                while True:
                    first = self._find_aud(buf, 0)
                    if first is None:
                        break
                    nxt = self._find_aud(buf, first + 4)
                    if nxt is None:
                        break
                    self._q.put(buf[first:nxt])
                    buf = buf[nxt:]
            if buf and self._find_aud(buf, 0) is not None:
                self._q.put(buf)  # trailing AU after EOF (stream flushed on close)
        finally:
            self._q.put(None)

    @staticmethod
    def _find_aud(buf, start):
        i = start
        while True:
            i = buf.find(b"\x00\x00\x01", i)
            if i < 0 or i + 3 >= len(buf):
                return None
            if buf[i + 3] & 0x1F == NAL_AUD:
                return i
            i += 3

    def feed(self, rgb):
        """One raw frame (numpy array or bytes, matching w/h/components)."""
        self.proc.stdin.write(rgb)  # buffer protocol: no per-frame copy

    def get_au(self, timeout=2.0):
        """-> (au|None, is_key). None = nothing ready within timeout / ended."""
        try:
            au = self._q.get(timeout=timeout)
        except queue.Empty:
            return None, False
        if au is None:
            self._q.put(None)  # recycle the end sentinel: every later get sees it
            return None, False
        is_key = any(x[0] & 0x1F == NAL_IDR for x in split_annexb(au))
        return au, is_key

    def force_keyframe(self):
        pass  # no mid-stream control over the pipe; viewers join at the next GOP

    def drain(self, timeout=5.0):
        """Close input and yield every remaining access unit."""
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
            self.proc.stdin = None
        while True:
            try:
                au = self._q.get(timeout=timeout)
            except queue.Empty:
                return
            if au is None:
                self._q.put(None)  # keep the sentinel: drain/close stay idempotent
                return
            yield au

    def close(self, timeout=5.0):
        for _ in self.drain(timeout=timeout):
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=timeout)


def make_encoder(renderer, w, h, fps=30, bitrate=4_000_000, components=4):
    """Hardware when the board has it, software otherwise. Raises only when
    neither exists."""
    try:
        return HardwareH264(renderer, w, h, fps=fps, bitrate=bitrate)
    except RuntimeError as hw_err:
        try:
            return SoftwareH264(w, h, fps=fps, bitrate=bitrate, components=components)
        except RuntimeError as sw_err:
            raise RuntimeError(f"no H264 encoder: {hw_err}; {sw_err}") from None
