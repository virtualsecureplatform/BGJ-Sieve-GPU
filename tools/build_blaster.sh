#!/bin/bash

set -euo pipefail

if (( $# != 3 )); then
    echo "usage: $0 PREFIX SOURCE EXPECTED_COMMIT" >&2
    exit 2
fi
PREFIX="$1"
SOURCE="$2"
EXPECTED_COMMIT="$3"
ACTUAL_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || {
    echo "BLASter revision mismatch: expected $EXPECTED_COMMIT, found $ACTUAL_COMMIT" >&2
    exit 2
}
if compgen -G "${PREFIX}/blaster/_core*.so" >/dev/null; then
    echo "Reusing BLASter ${EXPECTED_COMMIT} at ${PREFIX}"
    exit 0
fi
mkdir -p "$PREFIX/blaster"
cp "$SOURCE"/src/blaster/*.py "$PREFIX/blaster/"
(
    cd "$SOURCE"
    CC=gcc CXX=g++ python3 setup.py build_ext --build-lib "$PREFIX"
)
compgen -G "${PREFIX}/blaster/_core*.so" >/dev/null
echo "Built BLASter ${EXPECTED_COMMIT} at ${PREFIX}"
