#!/bin/bash
# Render the pictures of docs/python.md into media/python/, from the programs
# in tools/docs/python/. Each runs in a folder of its own and writes its
# picture there. A still is a PNG and a motion a looping WebP, both at three
# times the render's size, with nearest sampling.
# Needs ffmpeg with libwebp, and a Python that imports ghee-render.
# usage: tools/docs/python.sh [PYTHON]       python3 by default
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
programs=$root/tools/docs/python
python=${1:-python3}
out=$root/media/python
if ! "$python" -c 'import ghee; ghee.open' 2>/dev/null; then
    echo "python.sh: $python does not import ghee-render" >&2
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out"

# run NAME [ARGS...]: tools/docs/python/NAME.py in $tmp/NAME/
run() {
    local name=$1
    shift
    mkdir -p "$tmp/$name"
    (cd "$tmp/$name" && "$python" "$programs/$name.py" "$@")
}

# still NAME FILE: FILE of NAME's folder as media/python/NAME.png
still() {
    ffmpeg -v error -y -i "$tmp/$1/$2" -vf scale=iw*3:ih*3:flags=neighbor "$out/$1.png"
    echo "media/python/$1.png"
}

# loop NAME FPS: the frames f.NNNN.png of NAME's folder as a WebP that loops
loop() {
    ffmpeg -v error -y -framerate "$2" -i "$tmp/$1/f.%04d.png" \
        -vf scale=iw*3:ih*3:flags=neighbor -c:v libwebp_anim -lossless 1 \
        -compression_level 6 -loop 0 "$out/$1.webp"
    echo "media/python/$1.webp"
}

run hello
still hello hello.png

run status
loop status 20

# The bars scene of docs/scenes.md.
mkdir -p "$tmp/ramp"
cp "$root/tools/docs/scenes/bars.scene.json" "$root/tools/docs/scenes/bars.glsl" "$tmp/ramp/"
run ramp
still ramp ramp.png

run clock -o clock.png --size 128x64
still clock clock.png

run scene
still scene scene.png
