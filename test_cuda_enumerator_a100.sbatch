#!/bin/bash
#SBATCH --job-name=cudaenum-correctness
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=00:30:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/cudaenum-correctness-%j.out

set -euo pipefail

SOURCE="${SLURM_SUBMIT_DIR}"
LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SIF="${SIF:-${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif}"
BUILD_ROOT="${LARGE_BASE}/builds/cudaenum-correctness-${SLURM_JOB_ID}"
mkdir -p "${BUILD_ROOT}" "${LARGE_BASE}/logs"
cp -a "${SOURCE}/deps/fplll-cuda-enumeration" "${BUILD_ROOT}/src"

apptainer exec --nv --bind "${BUILD_ROOT}" --pwd "${BUILD_ROOT}/src" "${SIF}" bash -lc '
set -euo pipefail
./configure --enable-a100 CC=/usr/bin/gcc CXX=/usr/bin/g++ \
  NVCC=/usr/local/cuda/bin/nvcc CUDA_CXXFLAGS="-D NDEBUG" \
  CUDA_LDFLAGS="-L/usr/local/cuda/lib64"
make -j 16
make check
CUDAENUM_DEVICES=0 CUDAENUM_VERBOSE=0 ./cudaenum_test
CUDAENUM_DEVICES=0,1,2,3 CUDAENUM_MULTI_GPU_MIN_DIM=1 CUDAENUM_VERBOSE=0 ./cudaenum_test
'
