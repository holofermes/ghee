#!/bin/bash
# Render the pictures of docs/displays.md into media/displays/: a frame of the
# bundled neon example, and tools/docs/displays/rotate.scene.json at each
# --rotate. A still is a PNG at three times the render's size, with nearest
# sampling.
# Needs ffmpeg. $GHEE_CACHE, when set, takes the examples that --demo unpacks.
# usage: tools/docs/displays.sh [GHEE]       target/release/ghee by default
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
ghee=${1:-$root/target/release/ghee}
out=$root/media/displays
if [ ! -x "$ghee" ]; then
    echo "displays.sh: no ghee at $ghee, run cargo build --release" >&2
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out"

# png NAME FRAME: frame FRAME of $tmp/NAME/, three times its size
png() {
    ffmpeg -v error -y -i "$tmp/$1/f.$(printf %04d "$2").png" \
        -vf scale=iw*3:ih*3:flags=neighbor "$out/$1.png"
    echo "media/displays/$1.png"
}

mkdir -p "$tmp/neon"
"$ghee" render --demo neon -o "$tmp/neon/f.####.png" --frames 31 --timestep 0.0333 -q
png neon 30

for turn in 0 90 180; do
    mkdir -p "$tmp/rotate-$turn"
    "$ghee" render "$root/tools/docs/displays/rotate.scene.json" -o "$tmp/rotate-$turn/f.####.png" \
        --size 128x64 --rotate "$turn" -q
    png "rotate-$turn" 0
done
