#!/bin/bash
#SBATCH --job-name=svp164-progbkz74-cuda
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp163-164/slurm-%x-%j.out

set -euo pipefail

SOURCE="${SLURM_SUBMIT_DIR}"
LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SIF="${SIF:-${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif}"
INPUT="${INPUT:-${LARGE_BASE}/inputs/svp164-seed0/L_164_0.potlll-heuristic-mpfr256}"
OUTPUT="${OUTPUT:-${LARGE_BASE}/inputs/svp164-seed0/L_164_0.potlll-heuristic-mpfr256.progressive-bkz74-cuda.pre}"
OLD_BKZ72="${LARGE_BASE}/inputs/svp164-seed0/L_164_0.potlll-heuristic-mpfr256.bkz.p72l8.pre"
BUILD_ROOT="${LARGE_BASE}/builds/fplll-progressive-bkz74-cuda-${SLURM_JOB_ID}"
RUN_ROOT="${LARGE_BASE}/runs/svp164-seed0/progressive-bkz74-cuda/job-${SLURM_JOB_ID}"
LOG_ROOT="${LARGE_BASE}/logs/svp163-164"

mkdir -p "${BUILD_ROOT}" "${RUN_ROOT}" "${LOG_ROOT}" "$(dirname "${OUTPUT}")"

# Build isolated source copies so Autotools and object files never modify the
# login-node checkout.  Both compilation and reduction execute on this node.
cp -a "${SOURCE}/deps/fplll-cuda-enumeration" "${BUILD_ROOT}/cudaenum-src"
cp -a "${SOURCE}/dep/fplll" "${BUILD_ROOT}/fplll-src"

apptainer exec --nv \
    --bind "${SOURCE}","${BUILD_ROOT}","${RUN_ROOT}","$(dirname "${INPUT}")" \
    --pwd "${RUN_ROOT}" "${SIF}" bash -lc "
set -euo pipefail
cd '${BUILD_ROOT}/cudaenum-src'
./configure --enable-a100 CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    NVCC=/usr/local/cuda/bin/nvcc \
    CUDA_CXXFLAGS='-D NDEBUG' \
    CUDA_LDFLAGS='-L/usr/local/cuda/lib64'
make -j 16

cd '${BUILD_ROOT}/fplll-src'
./autogen.sh
LDFLAGS='-L/usr/local/cuda/lib64 -Wl,-rpath,${BUILD_ROOT}/cudaenum-src/.libs -Wl,-rpath,/usr/local/cuda/lib64' \
./configure \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    --with-extenum-dir='${BUILD_ROOT}/cudaenum-src/.libs' \
    --with-extenum-lib=cudaenum \
    --with-extenum-func=fplll_cuda_enum \
    CXXFLAGS='-O3 -march=native'
make -j 16

g++ -O2 -I'${SOURCE}/include' '${SOURCE}/tools/compare_basis_quality.cpp' \
    -L'${SOURCE}/src' -lllib -lntl -lgmp -lm -fopenmp -pthread \
    -o '${BUILD_ROOT}/compare_basis_quality'
"

export APPTAINERENV_LD_LIBRARY_PATH="${BUILD_ROOT}/cudaenum-src/.libs:${BUILD_ROOT}/fplll-src/.libs:/usr/local/cuda/lib64"
export APPTAINERENV_CUDAENUM_DEVICES="0,1,2,3"
export APPTAINERENV_CUDAENUM_PROFILE=a100
export APPTAINERENV_CUDAENUM_MEMORY_FRACTION=0.75
export APPTAINERENV_CUDAENUM_MULTI_GPU_MIN_DIM=65
export APPTAINERENV_CUDAENUM_VERBOSE=1

TMP_OUTPUT="${OUTPUT}.tmp.${SLURM_JOB_ID}"
rm -f "${TMP_OUTPUT}"
echo "phase=progressive_bkz_start epoch=$(date +%s) input=${INPUT} output=${OUTPUT} beta=74 devices=0,1,2,3"
timeout --signal=TERM --kill-after=60s 86130s \
    apptainer exec --nv \
        --bind "${BUILD_ROOT}","${RUN_ROOT}","$(dirname "${INPUT}")" \
        --pwd "${RUN_ROOT}" "${SIF}" \
        "${BUILD_ROOT}/fplll-src/fplll/fplll" \
        -a bkz -b 74 -bkzprogressive auto -nolll -bkzboundedlll \
        -f mpfr -p 256 \
        -s "${BUILD_ROOT}/fplll-src/strategies/default.json" \
        -of b "${INPUT}" >"${TMP_OUTPUT}"
echo "phase=progressive_bkz_end epoch=$(date +%s)"

echo "phase=determinant_check_start epoch=$(date +%s)"
apptainer exec \
    --bind "${SOURCE}","${BUILD_ROOT}","$(dirname "${INPUT}")" \
    "${SIF}" python3 "${SOURCE}/tools/check_basis_det_mod.py" "${INPUT}" "${TMP_OUTPUT}"
echo "phase=determinant_check_end epoch=$(date +%s)"
mv "${TMP_OUTPUT}" "${OUTPUT}"
echo "phase=quality_compare_start epoch=$(date +%s)"
apptainer exec \
    --bind "${SOURCE}","${BUILD_ROOT}","$(dirname "${INPUT}")" \
    "${SIF}" "${BUILD_ROOT}/compare_basis_quality" \
    potlll "${INPUT}" \
    old_fplll_bkz72 "${OLD_BKZ72}" \
    progressive_bkz74_cuda "${OUTPUT}"
echo "phase=quality_compare_end epoch=$(date +%s)"
