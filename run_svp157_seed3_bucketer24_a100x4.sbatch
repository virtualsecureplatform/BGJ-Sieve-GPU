#!/bin/bash
#SBATCH --job-name=svp157-s3-buc24
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=06:00:00

set -euo pipefail

SOURCE="${SLURM_SUBMIT_DIR}"
export SVP_DIMENSION=157
export SVP142_LATTICE_SEED=3
export SVP_INSTANCE=svp157-seed3
export SVP142_INPUT_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/inputs/${SVP_INSTANCE}"
export SVP142_RUN_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/runs/${SVP_INSTANCE}/variants/noprofile-buc24/job-${SLURM_JOB_ID}"
export SVP142_REPRO="${SOURCE}/reproduce/svp142-seed0/reproduce.py"
export SVP142_TARGET_NORM2=10701946
export SVP142_RAW_SHA256=GENERATE
export SVP142_KNOWN_VECTOR="${SOURCE}/reproduce/svp158-search/no-known-vector.txt"
export SVP_PREPROCESS_MODE=potlll-bkz
export SVP_BKZ_BETA=70
export SVP_BKZ_LOOPS=8
export BGJ_ENABLE_PROFILING=0
export SVP_SIEVE_SEED=0
export TARGET_SIEVING_DIM=139
export MIN_LIFTING_DIM=120
export SVP_TIMEOUT_SECONDS=21300
export APPTAINERENV_HD_BUCKET_THREADS=24
export APPTAINERENV_HD_GPU_NATIVE_BWC=0
export APPTAINERENV_HD_GPU_NATIVE_BWC_P2P=0
export APPTAINERENV_HD_GPU_NATIVE_BWC_VERIFY=0

exec bash "${SOURCE}/run_svp142_seed0_a100x4.sbatch"
