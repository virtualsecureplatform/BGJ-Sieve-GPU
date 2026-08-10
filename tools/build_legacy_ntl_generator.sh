#!/bin/bash

set -euo pipefail

if (( $# != 3 )); then
    echo "usage: $0 PREFIX NTL_ARCHIVE GENERATOR_SOURCE" >&2
    exit 2
fi

PREFIX="$1"
ARCHIVE="$2"
GENERATOR_SOURCE="$3"
EXPECTED_ARCHIVE_SHA256="8f31508a9176b3fc843f08468b1632017f2450677bfd5147ead5136e0f24b68f"

[[ "$(sha256sum "$ARCHIVE" | awk '{print $1}')" == "$EXPECTED_ARCHIVE_SHA256" ]] || {
    echo "Unexpected NTL 9.3.0 archive hash" >&2
    exit 2
}
if [[ -x "$PREFIX/bin/svp_challenge_generator" ]]; then
    "$PREFIX/bin/svp_challenge_generator" 1 0 >/dev/null
    echo "Reusing legacy challenge generator: $PREFIX/bin/svp_challenge_generator"
    exit 0
fi

WORK="${PREFIX}.build-${SLURM_JOB_ID:-$$}"
mkdir -p "$WORK" "$PREFIX/bin"
tar -xzf "$ARCHIVE" -C "$WORK"
(
    cd "$WORK/ntl-9.3.0/src"
    ./configure PREFIX="$PREFIX" NTL_GMP_LIP=on SHARED=off NTL_THREADS=off
    make -j "${SLURM_CPUS_PER_TASK:-4}"
    make install
)
g++ -O2 -std=c++11 -I"$PREFIX/include" "$GENERATOR_SOURCE" \
    "$PREFIX/lib/libntl.a" -lgmp -lm -o "$PREFIX/bin/svp_challenge_generator"
"$PREFIX/bin/svp_challenge_generator" 1 0 >/dev/null
echo "Built NTL 9.3.0 challenge generator: $PREFIX/bin/svp_challenge_generator"
