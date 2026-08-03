#!/bin/bash
# Build the ghee wheels for the Pi targets.
# usage: tools/xwheels.sh <template-wheel> [out-dir]
#
# The template is a native maturin wheel (maturin build --release
# --compatibility linux, run in python/). The script rebuilds the
# extension for each target with cargo-zigbuild, then repacks the
# template with the new module and platform tag. maturin cannot do
# this itself: it rejects glibc-suffixed targets and its manylinux
# audit wants to bundle the GL stack.
#
# GHEE_SYSROOTS names the sysroot directory (default /root/xcheck/sysroots,
# the build container). Each sysroot is the target's /usr/lib + /usr/include
# for libdrm, gbm, EGL and GLESv2.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
tmpl=${1:?usage: xwheels.sh <template-wheel> [out-dir]}
out=${2:-$root/python/target/wheels}
sysroots=${GHEE_SYSROOTS:-/root/xcheck/sysroots}

build() {
    target=$1 sysroot=$2
    shift 2
    env "$@" GHEE_SYSROOT="$sysroots/$sysroot" \
        cargo zigbuild --release --manifest-path "$root/python/Cargo.toml" \
        --target "$target"
}

# glibc pins: 2.36 = Bookworm. 2.38 = the floor for Trixie's libdrm on armv6.
# The Zero W is an ARM1176; zig spells the cpu with underscores.
build aarch64-unknown-linux-gnu.2.36 aarch64
build arm-unknown-linux-gnueabihf.2.38 armv6 \
    CFLAGS_arm_unknown_linux_gnueabihf=-mcpu=arm1176jzf_s

mkdir -p "$out"
python3 - "$tmpl" "$out" "$root" <<'EOF'
import base64, hashlib, os, re, struct, sys, zipfile

tmpl, out, root = sys.argv[1:4]
name = os.path.basename(tmpl)
m = re.match(r"(.+)-(cp\d+-abi\d+)-(.+)\.whl$", name)
if not m:
    sys.exit("not a wheel name: %s" % name)
distver, pytag, plat = m.groups()

targets = [
    ("aarch64-unknown-linux-gnu", "linux_aarch64", 183),
    ("arm-unknown-linux-gnueabihf", "linux_armv6l", 40),
]

def digest(data):
    h = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return h.rstrip(b"=").decode()

for triple, newplat, e_machine in targets:
    so = os.path.join(root, "python", "target", triple, "release", "lib_native.so")
    blob = open(so, "rb").read()
    machine = struct.unpack_from("<H", blob, 18)[0]
    if machine != e_machine:
        sys.exit("%s: ELF machine %d, want %d" % (so, machine, e_machine))
    wheel = os.path.join(out, "%s-%s-%s.whl" % (distver, pytag, newplat))
    zin = zipfile.ZipFile(tmpl)
    swapped = {}
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as zout:
        record = None
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename.endswith("ghee/_native.abi3.so"):
                data = blob
                swapped[info.filename] = data
            elif info.filename.endswith(".dist-info/WHEEL"):
                data = data.replace(plat.encode(), newplat.encode())
                swapped[info.filename] = data
            elif info.filename.endswith(".dist-info/RECORD"):
                record = info
                continue
            zout.writestr(info, data)
        lines = []
        for line in zin.read(record.filename).decode().splitlines():
            path = line.split(",")[0]
            if path in swapped:
                data = swapped[path]
                line = "%s,sha256=%s,%d" % (path, digest(data), len(data))
            lines.append(line)
        zout.writestr(record, "\n".join(lines) + "\n")
    print(wheel)
EOF
