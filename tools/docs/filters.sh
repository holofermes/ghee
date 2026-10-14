#!/bin/bash
# Render the pictures of docs/filters.md into media/filters/, from
# tools/docs/filters/panel.scene.json: a glowing clock over a grey ramp, as a
# 1-bit panel and a 16-level panel show it with no filter, and with each
# filter. A blinking filter's 16-frame cycle becomes a looping WebP, and the
# mean of the cycle a still: what the eye sees at the panel's frame rate.
# Stills are PNGs at three times the render's size, with nearest sampling.
# Needs python3 and ffmpeg with libwebp.
# usage: tools/docs/filters.sh [GHEE]       target/release/ghee by default
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
ghee=${1:-$root/target/release/ghee}
scene=$root/tools/docs/filters/panel.scene.json
out=$root/media/filters
if [ ! -x "$ghee" ]; then
    echo "filters.sh: no ghee at $ghee, run cargo build --release" >&2
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out"

# raw NAME SIZE FRAMES [FILTER]: FRAMES frames of the scene at SIZE as raw
# RGBA, one frame for each 1/200 s, as a fast panel takes them
raw() {
    local f=()
    if [ -n "${4:-}" ]; then f=(--filters "$4"); fi
    "$ghee" render "$scene" -o - --size "$2" --frames "$3" --timestep 0.005 "${f[@]}" -q \
        > "$tmp/$1.rgba"
}

# grey NAME SIZE OP: $tmp/NAME.rgba to one grey frame per input frame in
# $tmp/NAME-OP.gray. OP is "frames" (each frame's luma), "1bit" (luma cut at
# 128, as a 1-bit panel's driver cuts it), "4bit" (luma's top four bits, 16
# levels) or "mean" (the mean of all frames)
grey() {
    python3 - "$tmp/$1.rgba" "$2" "$3" "$tmp/$1-$3.gray" <<'PY'
import sys
src, size, op, dst = sys.argv[1:]
w, h = (int(v) for v in size.split("x"))
data = open(src, "rb").read()
n = w * h
frames = [data[i * n * 4:(i + 1) * n * 4] for i in range(len(data) // (n * 4))]

def luma(f):
    return [round(0.299 * f[i] + 0.587 * f[i + 1] + 0.114 * f[i + 2]) for i in range(0, n * 4, 4)]

out = bytearray()
if op == "mean":
    sums = [0] * n
    for f in frames:
        for i, v in enumerate(luma(f)):
            sums[i] += v
    out += bytes(round(s / len(frames)) for s in sums)
else:
    for f in frames:
        for v in luma(f):
            if op == "1bit":
                v = 255 if v >= 128 else 0
            elif op == "4bit":
                v = (v >> 4) * 17
            out.append(v)
open(dst, "wb").write(out)
PY
}

# png NAME-OP SIZE: the first grey frame of $tmp/NAME-OP.gray, three times its size
png() {
    ffmpeg -v error -y -f rawvideo -pix_fmt gray -s "$2" -i "$tmp/$1.gray" -frames:v 1 \
        -vf scale=iw*3:ih*3:flags=neighbor "$out/$3.png"
    echo "media/filters/$3.png"
}

# webp NAME-OP SIZE OUT: every grey frame, looping at 8 frames a second
webp() {
    ffmpeg -v error -y -f rawvideo -pix_fmt gray -framerate 8 -s "$2" -i "$tmp/$1.gray" \
        -vf scale=iw*3:ih*3:flags=neighbor -loop 0 -lossless 1 "$out/$3.webp"
    echo "media/filters/$3.webp"
}

small=128x32
big=256x64

raw plain "$small" 1
grey plain "$small" 1bit
png plain-1bit "$small" none-1bit

raw dither "$small" 1 dither
grey dither "$small" frames
png dither-frames "$small" dither

raw frc "$small" 16 frc
grey frc "$small" frames
grey frc "$small" mean
webp frc-frames "$small" frc
png frc-mean "$small" frc-seen

raw plain16 "$big" 1
grey plain16 "$big" 4bit
png plain16-4bit "$big" none-16

raw frc256 "$big" 16 frc256
grey frc256 "$big" mean
png frc256-mean "$big" frc256-seen
