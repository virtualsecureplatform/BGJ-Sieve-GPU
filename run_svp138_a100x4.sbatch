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
BUILD="${BUILD:-1}"
POOL_THREADS="${POOL_THREADS:-8}"
POOL_BUFFER_CACHE="${POOL_BUFFER_CACHE:-0}"
PRE_SHA256="${SVP138_PRE_SHA256:-f902fd5b8b72171b084f8d62202c31a05ef18b9547fe99b23eb39446d355faeb}"

mkdir -p "$WORK"
cp "$INPUT_SOURCE/svp138-seed2.raw.basis" "$WORK/L_138_2.raw"
cp "$INPUT_SOURCE/svp138-seed2.lll.basis" "$WORK/L_138_2.lll"
cp "$INPUT_SOURCE/svp138-seed2.bkz60.basis" "$WORK/L_138_2.p60l8.pre"

echo "node=$(hostname) job=${SLURM_JOB_ID} visible_gpus=${CUDA_VISIBLE_DEVICES:-unset} build=${BUILD} pool_threads=${POOL_THREADS} pool_buffer_cache=${POOL_BUFFER_CACHE}"
echo "source=${SOURCE}"
echo "work=${WORK}"
sha256sum "$WORK/L_138_2.raw" "$WORK/L_138_2.lll" "$WORK/L_138_2.p60l8.pre"

# Compilation and the sieve both run only inside the GPU allocation. Set
# BUILD=0 only after this exact source/configuration has been built once.
if [[ "$BUILD" == 1 ]]; then
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
elif [[ "$BUILD" != 0 ]]; then
    echo "BUILD must be 0 or 1" >&2
    exit 2
fi

if [[ ! "$POOL_THREADS" =~ ^[0-9]+$ ]] || (( POOL_THREADS < 4 || POOL_THREADS > 64 || POOL_THREADS % 4 != 0 )); then
    echo "POOL_THREADS must be a multiple of 4 in [4, 64]" >&2
    exit 2
fi
if [[ "$POOL_BUFFER_CACHE" != 0 && "$POOL_BUFFER_CACHE" != 1 ]]; then
    echo "POOL_BUFFER_CACHE must be 0 or 1" >&2
    exit 2
fi

sha256sum "$BINARY"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Summary profiling remains enabled; omit only the per-bucket D2H copy and
# stream synchronization that the verified production experiment disabled.
export APPTAINERENV_HD_PROFILE_BUCKET_STATS=0
export APPTAINERENV_HD_POOL_THREADS="$POOL_THREADS"
export APPTAINERENV_HD_POOL_BUFFER_CACHE="$POOL_BUFFER_CACHE"

apptainer exec --nv \
    --bind "$LARGE_BASE:$LARGE_BASE" \
    --pwd "$REPRO_SOURCE" \
    "$IMAGE" python3 "$REPRO" run \
        --workdir "$WORK" \
        --binary "$BINARY" \
        --pre-sha256 "$PRE_SHA256" \
        --timeout 6900 \
        --stale 0
