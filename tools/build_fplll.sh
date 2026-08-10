#!/bin/bash

# Build the current dep/fplll revision privately on a compute node. Bootstrap
# tools and MPFR are also private, so the immutable SIF needs no fplll install.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to build fplll outside a Slurm allocation" >&2
    exit 2
fi
if (( $# != 3 )); then
    echo "usage: $0 PREFIX FPLLL_SOURCE EXPECTED_COMMIT" >&2
    exit 2
fi

PREFIX="$1"
FPLLL_SOURCE="$2"
EXPECTED_COMMIT="$3"
DEPS_ROOT="$(dirname -- "$PREFIX")"
M4_URL="https://ftp.gnu.org/gnu/m4/m4-1.4.19.tar.xz"
M4_SHA256="63aede5c6d33b6d9b13511cd0be2cac046f2e70fd0a07aa9573a04a82783af96"
AUTOCONF_URL="https://ftp.gnu.org/gnu/autoconf/autoconf-2.72.tar.xz"
AUTOCONF_SHA256="ba885c1319578d6c94d46e9b0dceb4014caafe2490e437a0dbca3f270a223f5a"
AUTOMAKE_URL="https://ftp.gnu.org/gnu/automake/automake-1.16.5.tar.xz"
AUTOMAKE_SHA256="f01d58cd6d9d77fbdca9eb4bbd5ead1988228fdb73d6f7a201f5f8d6b118b469"
LIBTOOL_URL="https://ftp.gnu.org/gnu/libtool/libtool-2.4.7.tar.xz"
LIBTOOL_SHA256="4f7f217f057ce655ff22559ad221a0fd8ef84ad1fc5fcb6990cecc333aa1635d"
MPFR_URL="https://www.mpfr.org/mpfr-4.2.1/mpfr-4.2.1.tar.xz"
MPFR_SHA256="277807353a6726978996945af13e52829e3abd7a9a5b7fb2793894e18f1fcbb2"
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

STAGE="$(mktemp -d "${DEPS_ROOT}/.build-fplll.${SLURM_JOB_ID}.XXXXXX")"
cleanup() {
    case "$STAGE" in
        "${DEPS_ROOT}"/.build-fplll.*) rm -rf -- "$STAGE" ;;
        *) echo "Refusing to remove unexpected build directory: $STAGE" >&2 ;;
    esac
}
trap cleanup EXIT

download() {
    python3 - "$1" "$2" <<'PY'
import shutil
import sys
import urllib.request

request = urllib.request.Request(sys.argv[1], headers={"User-Agent": "BGJ-Sieve-GPU dependency builder"})
with urllib.request.urlopen(request, timeout=120) as response, open(sys.argv[2], "wb") as output:
    shutil.copyfileobj(response, output)
PY
}

download "$M4_URL" "${STAGE}/m4-1.4.19.tar.xz"
download "$AUTOCONF_URL" "${STAGE}/autoconf-2.72.tar.xz"
download "$AUTOMAKE_URL" "${STAGE}/automake-1.16.5.tar.xz"
download "$LIBTOOL_URL" "${STAGE}/libtool-2.4.7.tar.xz"
download "$MPFR_URL" "${STAGE}/mpfr-4.2.1.tar.xz"
(
    cd "$STAGE"
    echo "${M4_SHA256}  m4-1.4.19.tar.xz" | sha256sum -c -
    echo "${AUTOCONF_SHA256}  autoconf-2.72.tar.xz" | sha256sum -c -
    echo "${AUTOMAKE_SHA256}  automake-1.16.5.tar.xz" | sha256sum -c -
    echo "${LIBTOOL_SHA256}  libtool-2.4.7.tar.xz" | sha256sum -c -
    echo "${MPFR_SHA256}  mpfr-4.2.1.tar.xz" | sha256sum -c -
    tar -xf m4-1.4.19.tar.xz
    tar -xf autoconf-2.72.tar.xz
    tar -xf automake-1.16.5.tar.xz
    tar -xf libtool-2.4.7.tar.xz
    tar -xf mpfr-4.2.1.tar.xz
)

TOOLS_PREFIX="${STAGE}/tools"
BUILD_PREFIX="${STAGE}/prefix"
BUILD_JOBS="${SLURM_CPUS_ON_NODE:-16}"
if (( BUILD_JOBS > 16 )); then
    BUILD_JOBS=16
fi

for package in m4-1.4.19 autoconf-2.72 automake-1.16.5 libtool-2.4.7; do
    (
        cd "${STAGE}/${package}"
        PATH="${TOOLS_PREFIX}/bin:${PATH}" ./configure --prefix="$TOOLS_PREFIX"
        PATH="${TOOLS_PREFIX}/bin:${PATH}" make -j"$BUILD_JOBS"
        PATH="${TOOLS_PREFIX}/bin:${PATH}" make install
    )
done
(
    cd "${STAGE}/mpfr-4.2.1"
    ./configure --prefix="$BUILD_PREFIX" --disable-static --enable-shared
    make -j"$BUILD_JOBS"
    make install
)

cp -a "$FPLLL_SOURCE" "${STAGE}/fplll-src"
(
    cd "${STAGE}/fplll-src"
    export PATH="${TOOLS_PREFIX}/bin:${PATH}"
    export CPPFLAGS="-I${BUILD_PREFIX}/include"
    export LDFLAGS="-L${BUILD_PREFIX}/lib"
    export LD_LIBRARY_PATH="${BUILD_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    ./autogen.sh
    ./configure --prefix="$BUILD_PREFIX" --disable-static --enable-shared
    make -j"$BUILD_JOBS"
    make install
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
    echo "mpfr_version=4.2.1"
} > "${BUILD_PREFIX}/build-manifest.txt"
mv "$BUILD_PREFIX" "$PREFIX"
export LD_LIBRARY_PATH="${PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "[fplll-build] installed ${EXPECTED_COMMIT} at ${PREFIX} ($("${PREFIX}/bin/fplll" --version))"
