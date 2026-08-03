#!/bin/bash
# Make a ghee cross-build sysroot.
# usage: tools/mksysroot.sh <rootfs-dir> <out-dir>
#        tools/mksysroot.sh <repo-url> <dist> <arch> <out-dir>
#
# The first form copies from a target filesystem: a mounted rootfs or
# a directory rsynced from a device that has the dev packages.
# The second form downloads the packages from an apt repository and
# extracts them, with no root and no target device:
#   tools/mksysroot.sh http://raspbian.raspberrypi.com/raspbian trixie armhf sysroots/armv6
#   tools/mksysroot.sh http://deb.debian.org/debian trixie arm64 sysroots/aarch64
#
# The sysroot carries what the engine compiles and links against: the
# EGL/GLES/KHR/libdrm headers, gbm.h, xf86drm*.h, and the four GL-stack
# libraries with their symlinks.
set -euo pipefail

PKGS="libdrm2 libdrm-dev libegl1 libegl-dev libgbm1 libgbm-dev
      libgl-dev libgles2 libgles-dev"

usage="usage: mksysroot.sh <rootfs-dir> <out> | <repo-url> <dist> <arch> <out>"

tmp=
cleanup() {
    [ -n "$tmp" ] && rm -rf "$tmp"
    return 0
}
trap cleanup EXIT

fetch() {
    repo=$1 dist=$2 arch=$3
    tmp=$(mktemp -d)
    curl -LsS "$repo/dists/$dist/main/binary-$arch/Packages.xz" |
        xz -d >"$tmp/Packages"
    # shellcheck disable=SC2086
    files=$(awk -v pkgs="$(echo $PKGS)" '
        BEGIN { n = split(pkgs, a, " "); for (i = 1; i <= n; i++) want[a[i]] = 1 }
        /^Package: / { p = $2 }
        /^Filename: / { if (p in want && !(p in got)) got[p] = $2 }
        END {
            bad = 0
            for (p in want)
                if (p in got) print got[p]
                else { print "mksysroot.sh: " p ": not in " pkgs > "/dev/stderr"; bad = 1 }
            exit bad
        }' "$tmp/Packages")
    for f in $files; do
        curl -LsSo "$tmp/$(basename "$f")" "$repo/$f"
        dpkg-deb -x "$tmp/$(basename "$f")" "$tmp/root"
    done
    echo "$tmp/root"
}

case $# in
2) root=$1 out=$2 ;;
4) out=$4; root=$(fetch "$1" "$2" "$3") ;;
*) echo "$usage" >&2; exit 2 ;;
esac

libdir=$(cd "$root"/usr/lib/*-linux-gnu* && pwd)
triplet=$(basename "$libdir")

mkdir -p "$out/usr/include" "$out/usr/lib/$triplet"
cp -a "$root/usr/include/EGL" "$root/usr/include/GLES" \
    "$root/usr/include/GLES2" "$root/usr/include/GLES3" \
    "$root/usr/include/KHR" "$root/usr/include/libdrm" \
    "$root/usr/include/gbm.h" "$root/usr/include/xf86drm.h" \
    "$root/usr/include/xf86drmMode.h" "$out/usr/include/"
for lib in libdrm libgbm libEGL libGLESv2; do
    cp -a "$libdir/$lib".so* "$out/usr/lib/$triplet/"
done

echo "$out: $(find "$out" -type f,l | wc -l) files, $(du -sh "$out" | cut -f1)"
