#!/bin/bash
#SBATCH --job-name=bounded-bkz-fix
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=00:30:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/bounded-bkz-fix-%j.out

set -euo pipefail

SOURCE="${SLURM_SUBMIT_DIR}"
LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SIF="${SIF:-${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif}"
CUDA_BUILD="${LARGE_BASE}/builds/cudaenum-correctness-294990/src"
BUILD_ROOT="${LARGE_BASE}/builds/bounded-bkz-fix-${SLURM_JOB_ID}"
INPUT="${LARGE_BASE}/inputs/svp164-seed0/L_164_0.potlll-heuristic-mpfr256"
OUTPUT="${BUILD_ROOT}/progressive-bkz40.pre"
mkdir -p "${BUILD_ROOT}" "${LARGE_BASE}/logs"
cp -a "${SOURCE}/dep/fplll" "${BUILD_ROOT}/fplll-src"

apptainer exec --nv --bind "${SOURCE}","${BUILD_ROOT}","${CUDA_BUILD}","$(dirname "${INPUT}")" \
  --pwd "${BUILD_ROOT}/fplll-src" "${SIF}" bash -lc "
set -euo pipefail
./autogen.sh
LDFLAGS='-L/usr/local/cuda/lib64 -Wl,-rpath,${CUDA_BUILD}/.libs -Wl,-rpath,/usr/local/cuda/lib64' \\
./configure CC=/usr/bin/gcc CXX=/usr/bin/g++ \\
  --with-extenum-dir='${CUDA_BUILD}/.libs' --with-extenum-lib=cudaenum \\
  --with-extenum-func=fplll_cuda_enum CXXFLAGS='-O3 -march=native'
make -j 16
make check
CUDAENUM_DEVICES=0,1,2,3 CUDAENUM_PROFILE=a100 CUDAENUM_MULTI_GPU_MIN_DIM=1 \\
  CUDAENUM_VERBOSE=0 LD_LIBRARY_PATH='${CUDA_BUILD}/.libs:/.singularity.d/libs:/usr/local/cuda/lib64' \\
  ./fplll/fplll -a bkz -b 40 -bkzprogressive 10,20,30,40 -bkzmaxloops 1 \\
    -nolll -bkzboundedlll -f mpfr -p 256 -s ./strategies/default.json \\
    -of b '${INPUT}' > '${OUTPUT}'
python3 '${SOURCE}/tools/check_basis_det_mod.py' '${INPUT}' '${OUTPUT}'
"
