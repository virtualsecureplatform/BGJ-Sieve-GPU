#!/bin/bash
#SBATCH --job-name=svp163-s0-6-bkz70
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --array=0-6%1
#SBATCH --time=12:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp163-164/slurm-%x-%A_%a.out

set -euo pipefail

SOURCE="${SLURM_SUBMIT_DIR}"
SEED="${SLURM_ARRAY_TASK_ID}"
export SVP_DIMENSION=163
export SVP142_LATTICE_SEED="$SEED"
export SVP_INSTANCE="svp163-seed${SEED}"
export SVP142_INPUT_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/inputs/${SVP_INSTANCE}"
export SVP142_RUN_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/runs/${SVP_INSTANCE}/pwc364-bkz70/job-${SLURM_ARRAY_JOB_ID}_${SEED}"
export SVP142_REPRO="${SOURCE}/reproduce/svp163-search/reproduce.py"
# Strict improvement over the official norm-3333 record.
export SVP142_TARGET_NORM2=11108888
export SVP142_RAW_SHA256=GENERATE
export SVP142_KNOWN_VECTOR="${SOURCE}/reproduce/svp158-search/no-known-vector.txt"
export SVP_PREPROCESS_MODE=potlll-bkz
export SVP_BKZ_BETA=70
export SVP_BKZ_LOOPS=8
export BGJ_ENABLE_PROFILING=0
export SVP_SIEVE_SEED=0
export TARGET_SIEVING_DIM=144
export MIN_LIFTING_DIM=120
export SVP_TIMEOUT_SECONDS=42930
export APPTAINERENV_HD_BUCKET_THREADS=20
export APPTAINERENV_HD_GPU_NATIVE_BWC=1
export APPTAINERENV_HD_GPU_NATIVE_BWC_P2P=1
export APPTAINERENV_HD_GPU_NATIVE_BWC_VERIFY=0
export APPTAINERENV_HD_HBM_BWC_GB=24

exec bash "${SOURCE}/run_svp142_seed0_a100x4.sbatch"
