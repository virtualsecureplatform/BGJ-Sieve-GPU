#!/bin/bash

# Build the current dep/fplll revision in a private prefix. The container
# supplies only ordinary build prerequisites; fplll itself comes from the
# checked-out source tree. This works both in a Slurm allocation and locally.
set -euo pipefail

if (( $# != 3 )); then
    echo "usage: $0 PREFIX FPLLL_SOURCE EXPECTED_COMMIT" >&2
    exit 2
fi

PREFIX="$1"
FPLLL_SOURCE="$2"
EXPECTED_COMMIT="$3"
DEPS_ROOT="$(dirname -- "$PREFIX")"
STRATEGY_SHA256="f516b0a6f0c580cff72e1e2c3562c44dc6f17e8f99613e9e4020e35481b27a18"

if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Invalid expected fplll commit: $EXPECTED_COMMIT" >&2
    exit 2
fi
ACTUAL_COMMIT="$(git -C "$FPLLL_SOURCE" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
    echo "fplll source moved: expected $EXPECTED_COMMIT, found $ACTUAL_COMMIT" >&2
    exit 2
fi
for tool in autoreconf automake libtoolize make gcc g++ pkg-config; do
    if ! command -v "$tool" >/dev/null; then
        echo "Missing container build prerequisite: $tool" >&2
        exit 2
    fi
done
if [[ ! -f /usr/include/mpfr.h ]]; then
    echo "Missing container build prerequisite: /usr/include/mpfr.h" >&2
    exit 2
fi

mkdir -p "$DEPS_ROOT"
exec 9>"${DEPS_ROOT}/.fplll-build.lock"
flock -x 9
export LD_LIBRARY_PATH="${PREFIX}/lib:${LD_LIBRARY_PATH:-}"
if [[ -x "${PREFIX}/bin/fplll" && -f "${PREFIX}/build-manifest.txt" ]] &&
   grep -Fqx "fplll_commit=${EXPECTED_COMMIT}" "${PREFIX}/build-manifest.txt" &&
   echo "${STRATEGY_SHA256}  ${PREFIX}/share/fplll/strategies/default.json" | sha256sum -c -; then
    echo "[fplll-build] reusing ${PREFIX} ($("${PREFIX}/bin/fplll" --version))"
    exit 0
fi

STAGE="$(mktemp -d "${DEPS_ROOT}/.build-fplll.${SLURM_JOB_ID:-local}.XXXXXX")"
cleanup() {
    case "$STAGE" in
        "${DEPS_ROOT}"/.build-fplll.*) rm -rf -- "$STAGE" ;;
        *) echo "Refusing to remove unexpected build directory: $STAGE" >&2 ;;
    esac
}
trap cleanup EXIT

cp -a "$FPLLL_SOURCE" "${STAGE}/fplll-src"
BUILD_PREFIX="${STAGE}/prefix"
BUILD_JOBS="${SLURM_CPUS_ON_NODE:-$(nproc)}"
if (( BUILD_JOBS > 16 )); then
    BUILD_JOBS=16
fi
(
    cd "${STAGE}/fplll-src"
    # Cluster module environments may export CC=nvc even inside Apptainer.
    # Use the Ubuntu toolchain whose headers and libraries are in this image.
    export CC=gcc
    export CXX=g++
    ./autogen.sh
    ./configure --prefix="$BUILD_PREFIX" --disable-static --enable-shared
    make -j"$BUILD_JOBS"
    make install
    export LD_LIBRARY_PATH="${BUILD_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    "${BUILD_PREFIX}/bin/fplll" --version
    echo "${STRATEGY_SHA256}  ${BUILD_PREFIX}/share/fplll/strategies/default.json" | sha256sum -c -
)

if [[ -e "$PREFIX" ]]; then
    echo "Unexpected incomplete dependency prefix exists: $PREFIX" >&2
    exit 2
fi
{
    echo "fplll_commit=${EXPECTED_COMMIT}"
    echo "fplll_version=$(LD_LIBRARY_PATH="${BUILD_PREFIX}/lib" "${BUILD_PREFIX}/bin/fplll" --version)"
} > "${BUILD_PREFIX}/build-manifest.txt"
mv "$BUILD_PREFIX" "$PREFIX"
export LD_LIBRARY_PATH="${PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "[fplll-build] installed ${EXPECTED_COMMIT} at ${PREFIX} ($("${PREFIX}/bin/fplll" --version))"
