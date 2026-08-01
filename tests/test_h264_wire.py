"""What a fresh /h264 viewer receives from the fanout: the framing, the order,
and the config payload. External consumers forward these bytes verbatim, so the
wire is a contract and not an implementation detail. Pure: no encoder, no
hardware, no ffmpeg."""

import json
import queue
import struct

from ghee.console.server import H264Fanout

SC = b"\x00\x00\x00\x01"
SPS, PPS = b"\x67\x42\x00\x1e", b"\x68\xce\x06\xe2"
CONFIG, KEY, DELTA = 0, 1, 2


def key_au():
    return SC + SPS + SC + PPS + SC + b"\x65\x88\x80"


def delta_au():
    return SC + b"\x41\x88\x80"


def fanout():
    f = H264Fanout()
    f.size = (320, 240)
    return f


def drain(client):
    out = []
    while True:
        try:
            out.append(client.q.get_nowait())
        except queue.Empty:
            return out


def unframe(msg):
    return msg[0], struct.unpack(">I", msg[1:5])[0], msg[5:]


def test_a_viewer_gets_config_then_the_keyframe():
    f = fanout()
    c = f.add()
    f.push(key_au())
    types = [unframe(m)[0] for m in drain(c)]
    assert types == [CONFIG, KEY]


def test_every_wire_frame_declares_its_own_length():
    f = fanout()
    c = f.add()
    f.push(key_au())
    f.push(delta_au())
    msgs = drain(c)
    assert len(msgs) == 3
    for m in msgs:
        t, n, payload = unframe(m)
        assert t in (CONFIG, KEY, DELTA)
        assert n == len(payload)
        assert n > 0


def test_the_config_payload_is_a_decoder_config():
    f = fanout()
    c = f.add()
    f.push(key_au())
    t, _, payload = unframe(drain(c)[0])
    assert t == CONFIG
    cfg = json.loads(payload)
    assert {"codec", "description", "codedWidth", "codedHeight"} <= set(cfg)
    assert (cfg["codedWidth"], cfg["codedHeight"]) == (320, 240)


def test_a_delta_access_unit_arrives_as_its_own_type():
    f = fanout()
    c = f.add()
    f.push(key_au())
    drain(c)
    f.push(delta_au())
    assert [unframe(m)[0] for m in drain(c)] == [DELTA]


def test_a_viewer_joining_mid_stream_waits_for_the_next_keyframe():
    f = fanout()
    first = f.add()
    f.push(key_au())
    drain(first)
    late = f.add()
    f.push(delta_au())
    assert drain(late) == []
    assert f.want_key is True
    f.push(key_au())
    assert [unframe(m)[0] for m in drain(late)] == [CONFIG, KEY]
