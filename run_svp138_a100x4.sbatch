#!/bin/bash
#SBATCH --job-name=svp138-c586-4g
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=02:00:00
#SBATCH --output=svp138-c586-4g_%j.log

set -euo pipefail

LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SOURCE="${SLURM_SUBMIT_DIR}"
REPRO_SOURCE="${REPRO_SOURCE:-${LARGE_BASE}/worktrees/repro-a100-2gpu}"
IMAGE="${BGJ_IMAGE:-${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif}"
INPUT_SOURCE="${SVP138_INPUT_DIR:-${LARGE_BASE}/runs/svp138-seed2-289059}"
WORK="${LARGE_BASE}/c586-a100x4-repro/job-${SLURM_JOB_ID}"
REPRO="${REPRO_SOURCE}/reproduce/svp138-seed2/reproduce.py"
BINARY="${SOURCE}/app/hd_sieve"

mkdir -p "$WORK"
cp "$INPUT_SOURCE/svp138-seed2.raw.basis" "$WORK/L_138_2.raw"
cp "$INPUT_SOURCE/svp138-seed2.lll.basis" "$WORK/L_138_2.lll"
cp "$INPUT_SOURCE/svp138-seed2.bkz60.basis" "$WORK/L_138_2.p60l8.pre"

echo "node=$(hostname) job=${SLURM_JOB_ID} visible_gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "source=${SOURCE}"
echo "work=${WORK}"
sha256sum "$WORK/L_138_2.raw" "$WORK/L_138_2.lll" "$WORK/L_138_2.p60l8.pre"

# Compilation and the sieve both run only inside the GPU allocation.
apptainer exec --nv \
    --bind "$LARGE_BASE:$LARGE_BASE" \
    --pwd "$SOURCE" \
    "$IMAGE" bash -lc '
        set -euo pipefail
        make -C src clean NVCC=nvcc
        make -C app clean NVCC=nvcc
        make -C src INC_DIR=-DHD_SVP140_CACHE_PROFILE=1 NVCC=nvcc
        make -C app hd_sieve INC_DIR=-DHD_SVP140_CACHE_PROFILE=1 NVCC=nvcc
    '

sha256sum "$BINARY"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

apptainer exec --nv \
    --bind "$LARGE_BASE:$LARGE_BASE" \
    --pwd "$REPRO_SOURCE" \
    "$IMAGE" python3 "$REPRO" run \
        --workdir "$WORK" \
        --binary "$BINARY" \
        --timeout 6900 \
        --stale 0
