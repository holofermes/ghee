#!/bin/bash
# Render the pictures of docs/procedural.md into media/procedural/, from the
# programs in tools/docs/procedural/, each run by `ghee render`. A motion is a
# looping WebP at three times the render's size, with nearest sampling.
# Needs python3, and ffmpeg with libwebp.
# usage: tools/docs/procedural.sh [GHEE]       target/release/ghee by default
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
ghee=${1:-$root/target/release/ghee}
out=$root/media/procedural
if [ ! -x "$ghee" ]; then
    echo "procedural.sh: no ghee at $ghee, run cargo build --release" >&2
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out"

# loop NAME SIZE FRAMES STEP FPS: NAME's program rendered FRAMES frames at a
# STEP s timestep, as a WebP that loops at FPS
loop() {
    mkdir -p "$tmp/$1"
    cp "$root/tools/docs/procedural/$1.py" "$tmp/$1/"
    "$ghee" render "$tmp/$1/$1.py" -o "$tmp/$1/f.####.png" --size "$2" \
        --frames "$3" --timestep "$4" -q
    ffmpeg -v error -y -framerate "$5" -i "$tmp/$1/f.%04d.png" \
        -vf scale=iw*3:ih*3:flags=neighbor -c:v libwebp_anim -lossless 1 \
        -compression_level 6 -loop 0 "$out/$1.webp"
    echo "media/procedural/$1.webp"
}

loop bars 128x64 63 0.05 20
