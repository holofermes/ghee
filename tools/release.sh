#!/bin/bash
# Write one version into every site, and regenerate THIRD-PARTY-NOTICES.
# The tree carries X.Y.Z-dev. The release flip is the last commit before the
# tag, the tag builds the artifacts, and the next commit flips to the next -dev.
# usage: tools/release.sh <version>       0.3.1, or 0.3.1-dev
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
version=${1:?usage: release.sh <version>}
case $version in
    [0-9]*.[0-9]*.[0-9]*) ;;
    *) echo "release.sh: '$version' is no X.Y.Z or X.Y.Z-dev" >&2; exit 1 ;;
esac

manifests=("$root/Cargo.toml" "$root/python/Cargo.toml")
locks=("$root/Cargo.lock" "$root/python/Cargo.lock")

check() {
    if [ "$2" != "$version" ]; then
        echo "release.sh: ${1#"$root/"} says '$2'" >&2
        exit 1
    fi
    printf '%-20s %s\n' "${1#"$root/"}" "$2"
}

for manifest in "${manifests[@]}"; do
    sed -i "0,/^version = \".*\"$/s//version = \"$version\"/" "$manifest"
    # the lock names the version too. --workspace leaves every other package
    # where it stands.
    cargo update --workspace --offline --quiet --manifest-path "$manifest"
done

for manifest in "${manifests[@]}"; do
    check "$manifest" "$(grep -m1 '^version = ' "$manifest" | cut -d'"' -f2)"
done

for lock in "${locks[@]}"; do
    for got in $(grep -A1 -E '^name = "ghee(-python)?"$' "$lock" |
        grep '^version = ' | cut -d'"' -f2); do
        check "$lock" "$got"
    done
done

# THIRD-PARTY-NOTICES lists the crates of python/Cargo.lock, written above
cargo about generate --locked --manifest-path "$root/python/Cargo.toml" \
    --config "$root/tools/about.toml" --output-file "$root/THIRD-PARTY-NOTICES" \
    "$root/tools/about.hbs"

# the Debian spelling the deb build derives: 0.3.1-dev sorts after 0.3.1, and
# 0.3.1~dev sorts before it
printf '%-20s %s\n' "deb (derived)" "$(printf %s "$version" | tr - '~')"
