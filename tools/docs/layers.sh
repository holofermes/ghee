#!/bin/bash
# Render the pictures of docs/layers.md into media/layers/, from the scenes in
# tools/docs/layers/. A still is a PNG and a motion a looping WebP, both at
# three times the render's size, with nearest sampling.
# Needs python3, and ffmpeg with libwebp.
# usage: tools/docs/layers.sh [GHEE]       target/release/ghee by default
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
scenes=$root/tools/docs/layers
ghee=${1:-$root/target/release/ghee}
out=$root/media/layers
if [ ! -x "$ghee" ]; then
    echo "layers.sh: no ghee at $ghee, run cargo build --release" >&2
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out"

# The 16x16 icon of image.png and image-in-shader.png: white, and fully
# transparent around it.
python3 - "$tmp/wifi.png" <<'EOF'
import struct, sys, zlib

ART = """
................
................
.....######.....
...##########...
.###........###.
##............##
.....######.....
...##########...
..##........##..
................
......####......
.....######.....
................
.......##.......
.......##.......
................
"""


def chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


rows = ART.split()
raw = b"".join(
    b"\0" + b"".join(b"\xff\xff\xff\xff" if c == "#" else b"\0\0\0\0" for c in row) for row in rows
)
head = struct.pack(">IIBBBBB", len(rows[0]), len(rows), 8, 6, 0, 0, 0)
with open(sys.argv[1], "wb") as f:
    f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
EOF
export GHEE_ASSETS=$tmp

# render NAME SIZE FRAMES FPS: NAME's scene into $tmp/NAME/, frame N at N/FPS s
render() {
    rm -rf "${tmp:?}/$1"
    mkdir -p "$tmp/$1"
    "$ghee" render "$scenes/$1.scene.json" -o "$tmp/$1/f.####.png" --size "$2" \
        --frames "$3" --timestep "$(python3 -c "print(1 / $4)")" -q
}

# still NAME SIZE
still() {
    render "$1" "$2" 1 1
    ffmpeg -v error -y -i "$tmp/$1/f.0000.png" -vf scale=iw*3:ih*3:flags=neighbor "$out/$1.png"
    echo "media/layers/$1.png"
}

# loop NAME FPS: the frames already in $tmp/NAME/ as a WebP that loops
loop() {
    ffmpeg -v error -y -framerate "$2" -i "$tmp/$1/f.%04d.png" \
        -vf scale=iw*3:ih*3:flags=neighbor -c:v libwebp_anim -lossless 1 \
        -compression_level 6 -loop 0 "$out/$1.webp"
    echo "media/layers/$1.webp"
}

still hello 128x64
still gradient 128x64
still transparent-glow 128x64
still blends 512x64
still text_shadow 128x64
still glow 128x64
still text_stroke 128x64
still karaoke 128x64
still image 128x32
still shapes 128x32
still render 256x48
still text-in-shader 256x64
still image-in-shader 128x64

# One wave at the default speed, 0.6 waves a second, and one blink at the
# default rate, 1 a second.
render wave 128x64 50 30
loop wave 30
render blink 128x64 10 10
loop blink 10

# A scroll starts with the words off the right edge, so its first frame shows
# none of them. The loop ends before the next frame that shows none, once the
# words have crossed.
render scroll 128x64 300 30
frames=("$tmp"/scroll/f.*.png)
end=0
crossed=0
for i in "${!frames[@]}"; do
    [ "$i" -gt 0 ] || continue
    if cmp -s "${frames[0]}" "${frames[$i]}"; then
        if [ "$crossed" = 1 ]; then
            end=$i
            break
        fi
    else
        crossed=1
    fi
done
if [ "$end" = 0 ]; then
    echo "layers.sh: the scroll did not cross the frame in ${#frames[@]} frames" >&2
    exit 1
fi
rm -f "${frames[@]:$end}"
loop scroll 30
