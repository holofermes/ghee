#!/bin/bash
# Render the pictures of docs/scenes.md into media/scenes/, from the scenes in
# tools/docs/scenes/. A still is a PNG and a motion a looping WebP, both at
# three times the render's size, with nearest sampling.
# Needs python3, and ffmpeg with libwebp.
# usage: tools/docs/scenes.sh [GHEE]       target/release/ghee by default
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
ghee=${1:-$root/target/release/ghee}
out=$root/media/scenes
if [ ! -x "$ghee" ]; then
    echo "scenes.sh: no ghee at $ghee, run cargo build --release" >&2
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out"
scenes=$tmp/scenes
cp -r "$root/tools/docs/scenes" "$scenes"

# variant NAME FROM PASS KEY VALUE: FROM's scene with KEY of pass PASS set to
# VALUE, a JSON value, or with param KEY set when KEY is params.NAME
variant() {
    python3 - "$scenes/$2.scene.json" "$scenes/$1.scene.json" "$3" "$4" "$5" <<'EOF'
import json, sys

src, dst, pass_name, key, value = sys.argv[1:]
scene = json.load(open(src))
p = next(p for p in scene["passes"] if p["name"] == pass_name)
if key.startswith("params."):
    p["params"][key[len("params."):]]["value"] = json.loads(value)
else:
    p[key] = json.loads(value)
json.dump(scene, open(dst, "w"), indent=2)
EOF
}

# render NAME SIZE FRAMES STEP: NAME's scene into $tmp/NAME/, frame N at N*STEP s
render() {
    rm -rf "${tmp:?}/$1"
    mkdir -p "$tmp/$1"
    "$ghee" render "$scenes/$1.scene.json" -o "$tmp/$1/f.####.png" --size "$2" \
        --frames "$3" --timestep "$4" -q
}

# still NAME SIZE [FRAME]: frame FRAME of NAME's scene, the first by default
still() {
    local frame=${3:-0}
    render "$1" "$2" "$((frame + 1))" 0.05
    ffmpeg -v error -y -i "$tmp/$1/f.$(printf %04d "$frame").png" \
        -vf scale=iw*3:ih*3:flags=neighbor "$out/$1.png"
    echo "media/scenes/$1.png"
}

# loop NAME FPS FIRST: the frames in $tmp/NAME/ from FIRST on, as a WebP that
# loops
loop() {
    ffmpeg -v error -y -framerate "$2" -start_number "$3" -i "$tmp/$1/f.%04d.png" \
        -vf scale=iw*3:ih*3:flags=neighbor -c:v libwebp_anim -lossless 1 \
        -compression_level 6 -loop 0 "$out/$1.webp"
    echo "media/scenes/$1.webp"
}

still uv 128x64
still shadow 128x64
still bars 128x64

for n in 2 4 8; do
    variant "rings-$n" rings main params.rings "$n"
    still "rings-$n" 128x64
done

for s in nearest linear trilinear; do
    variant "sampling-$s" sampling checker sampling "\"$s\""
    still "sampling-$s" 128x64
done

# The dot goes round once in 60 frames. The trail of the first round fades out
# within it, so the second round loops.
render trail 128x64 120 "$(python3 -c 'import math; print(2 * math.pi / 3 / 60)')"
loop trail 30 60
