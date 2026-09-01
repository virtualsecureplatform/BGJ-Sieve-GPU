#!/bin/bash
#SBATCH --job-name=svp164-basis-quality
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=1
#SBATCH --time=00:10:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp163-164/slurm-%x-%j.out

set -euo pipefail

SOURCE="${SLURM_SUBMIT_DIR}"
LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SIF="${SIF:-${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif}"
BUILD_ROOT="${LARGE_BASE}/builds/basis-quality-${SLURM_JOB_ID}"
FPLLL_BASIS="${LARGE_BASE}/inputs/svp164-seed0/L_164_0.potlll-heuristic-mpfr256.bkz.p72l8.pre"
HD_BASIS="${LARGE_BASE}/inputs/svp164-seed0/L_164_0.hdbkz-b120-j8-f32.pre"

mkdir -p "${BUILD_ROOT}" "${LARGE_BASE}/logs/svp163-164"
apptainer exec --bind "${SOURCE}","${BUILD_ROOT}","$(dirname "${FPLLL_BASIS}")" \
    "${SIF}" bash -lc \
    "g++ -O2 -I'${SOURCE}/include' '${SOURCE}/tools/compare_basis_quality.cpp' \
         -L'${SOURCE}/src' -lllib -lntl -lgmp -lm -fopenmp -pthread \
         -o '${BUILD_ROOT}/compare_basis_quality' && \
     '${BUILD_ROOT}/compare_basis_quality' '${FPLLL_BASIS}' '${HD_BASIS}'"

