#!/bin/bash
#SBATCH --job-name=svp138-c586-4g
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=02:00:00
#SBATCH --output=svp138-c586-4g_%j.log

set -euo pipefail

LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SOURCE="${SLURM_SUBMIT_DIR}"
REPRO_SOURCE="${REPRO_SOURCE:-${SOURCE}}"
IMAGE="${BGJ_IMAGE:-${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif}"
INPUT_SOURCE="${SVP138_INPUT_DIR:-${LARGE_BASE}/runs/svp138-seed2-289059}"
WORK="${LARGE_BASE}/c586-a100x4-repro/job-${SLURM_JOB_ID}"
REPRO="${REPRO_SOURCE}/reproduce/svp138-seed2/reproduce.py"
ACTUAL_SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
SOURCE_COMMIT="${SOURCE_COMMIT:-${ACTUAL_SOURCE_COMMIT}}"
FILTER_TASK_VECS="${FILTER_TASK_VECS:-524288}"
BGJ2_REDUCER_THREADS="${BGJ2_REDUCER_THREADS:-48}"
TARGET_SIEVING_DIM="${TARGET_SIEVING_DIM:-128}"
CONTINUE_AFTER_TARGET="${CONTINUE_AFTER_TARGET:-0}"
GPU_UID_SHARED_LOAD_PCT="${GPU_UID_SHARED_LOAD_PCT:-60}"
BACKPRESSURE_PROFILE="${BACKPRESSURE_PROFILE:-tiered}"
BINARY_ROOT="${BGJ_BINARY_ROOT:-${LARGE_BASE}/builds}"
BINARY="${BGJ_BINARY:-${BINARY_ROOT}/hd_sieve-${SOURCE_COMMIT}-ftv${FILTER_TASK_VECS}}"
BINARY_MANIFEST="${BINARY}.manifest"
BUILD="${BUILD:-0}"
POOL_THREADS="${POOL_THREADS:-8}"
POOL_BUFFER_CACHE="${POOL_BUFFER_CACHE:-1}"
PRE_SHA256="${SVP138_PRE_SHA256:-f902fd5b8b72171b084f8d62202c31a05ef18b9547fe99b23eb39446d355faeb}"

if [[ "$ACTUAL_SOURCE_COMMIT" != "$SOURCE_COMMIT" ]]; then
    echo "Source moved after submission: expected $SOURCE_COMMIT, found $ACTUAL_SOURCE_COMMIT" >&2
    exit 2
fi
if [[ "$BUILD" != 0 && "$BUILD" != 1 ]]; then
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
if [[ ! "$BGJ2_REDUCER_THREADS" =~ ^[0-9]+$ ]] || (( BGJ2_REDUCER_THREADS < 4 || BGJ2_REDUCER_THREADS > 64 || BGJ2_REDUCER_THREADS % 4 != 0 )); then
    echo "BGJ2_REDUCER_THREADS must be a multiple of 4 in [4, 64]" >&2
    exit 2
fi
if [[ ! "$TARGET_SIEVING_DIM" =~ ^[0-9]+$ ]] || (( TARGET_SIEVING_DIM < 114 || TARGET_SIEVING_DIM > 138 )); then
    echo "TARGET_SIEVING_DIM must be in [114, 138]" >&2
    exit 2
fi
if [[ "$CONTINUE_AFTER_TARGET" != 0 && "$CONTINUE_AFTER_TARGET" != 1 ]]; then
    echo "CONTINUE_AFTER_TARGET must be 0 or 1" >&2
    exit 2
fi
if [[ ! "$GPU_UID_SHARED_LOAD_PCT" =~ ^[0-9]+$ ]] || (( GPU_UID_SHARED_LOAD_PCT < 25 || GPU_UID_SHARED_LOAD_PCT > 80 )); then
    echo "GPU_UID_SHARED_LOAD_PCT must be in [25, 80]" >&2
    exit 2
fi
if [[ "$BACKPRESSURE_PROFILE" != tiered && "$BACKPRESSURE_PROFILE" != smooth ]]; then
    echo "BACKPRESSURE_PROFILE must be tiered or smooth" >&2
    exit 2
fi
if [[ ! "$FILTER_TASK_VECS" =~ ^[0-9]+$ ]] || (( FILTER_TASK_VECS < 65536 || (FILTER_TASK_VECS & (FILTER_TASK_VECS - 1)) != 0 )); then
    echo "FILTER_TASK_VECS must be a power of two >= 65536" >&2
    exit 2
fi
if [[ ! -f "$IMAGE" ]]; then
    echo "Missing Apptainer image: $IMAGE" >&2
    exit 2
fi

mkdir -p "$WORK"
cp "$INPUT_SOURCE/svp138-seed2.raw.basis" "$WORK/L_138_2.raw"
cp "$INPUT_SOURCE/svp138-seed2.lll.basis" "$WORK/L_138_2.lll"
cp "$INPUT_SOURCE/svp138-seed2.bkz60.basis" "$WORK/L_138_2.p60l8.pre"

echo "node=$(hostname) job=${SLURM_JOB_ID} visible_gpus=${CUDA_VISIBLE_DEVICES:-unset} build=${BUILD} pool_threads=${POOL_THREADS} pool_buffer_cache=${POOL_BUFFER_CACHE} reducer_threads=${BGJ2_REDUCER_THREADS} filter_task_vecs=${FILTER_TASK_VECS} tsd=${TARGET_SIEVING_DIM} continue_after_target=${CONTINUE_AFTER_TARGET} uid_load_pct=${GPU_UID_SHARED_LOAD_PCT} backpressure=${BACKPRESSURE_PROFILE}"
echo "source=${SOURCE}"
echo "source_commit=${SOURCE_COMMIT}"
echo "binary=${BINARY}"
echo "work=${WORK}"
sha256sum "$WORK/L_138_2.raw" "$WORK/L_138_2.lll" "$WORK/L_138_2.p60l8.pre"

# Compilation and the sieve both run only inside Slurm allocations. The
# submit_svp138_a100x4.sh wrapper builds a commit/config-keyed binary first and
# submits this job with BUILD=0 after that build succeeds.
if [[ "$BUILD" == 1 ]]; then
    mkdir -p "$BINARY_ROOT"
    apptainer exec --nv \
        --bind "$LARGE_BASE:$LARGE_BASE" \
        --pwd "$SOURCE" \
        "$IMAGE" bash -lc '
            set -euo pipefail
            make -C src clean NVCC=nvcc
            make -C app clean NVCC=nvcc
            make -C src INC_DIR="-DHD_SVP140_CACHE_PROFILE=1 -DHD_FILTER_TASK_VECS=$1" NVCC=nvcc
            make -C app hd_sieve INC_DIR="-DHD_SVP140_CACHE_PROFILE=1 -DHD_FILTER_TASK_VECS=$1" NVCC=nvcc
        ' _ "$FILTER_TASK_VECS"
    BINARY_TMP="${BINARY}.tmp-${SLURM_JOB_ID}"
    cp "$SOURCE/app/hd_sieve" "$BINARY_TMP"
    chmod 0755 "$BINARY_TMP"
    mv -f "$BINARY_TMP" "$BINARY"
    {
        echo "source_commit=${SOURCE_COMMIT}"
        echo "filter_task_vecs=${FILTER_TASK_VECS}"
        sha256sum "$BINARY"
    } > "${BINARY_MANIFEST}.tmp-${SLURM_JOB_ID}"
    mv -f "${BINARY_MANIFEST}.tmp-${SLURM_JOB_ID}" "$BINARY_MANIFEST"
fi

if [[ ! -x "$BINARY" || ! -f "$BINARY_MANIFEST" ]]; then
    echo "Missing verified compute-built binary or manifest: $BINARY" >&2
    echo "Submit with ./submit_svp138_a100x4.sh, or set BUILD=1." >&2
    exit 2
fi
grep -Fqx "source_commit=${SOURCE_COMMIT}" "$BINARY_MANIFEST"
grep -Fqx "filter_task_vecs=${FILTER_TASK_VECS}" "$BINARY_MANIFEST"

sha256sum "$BINARY"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Summary profiling remains enabled; omit only the per-bucket D2H copy and
# stream synchronization that the verified production experiment disabled.
export APPTAINERENV_HD_PROFILE_BUCKET_STATS=0
export APPTAINERENV_HD_POOL_THREADS="$POOL_THREADS"
export APPTAINERENV_HD_POOL_BUFFER_CACHE="$POOL_BUFFER_CACHE"
export APPTAINERENV_HD_BGJ2_REDUCER_THREADS="$BGJ2_REDUCER_THREADS"
export APPTAINERENV_HD_GPU_UID_SHARED_LOAD_PCT="$GPU_UID_SHARED_LOAD_PCT"
export APPTAINERENV_HD_BACKPRESSURE_PROFILE="$BACKPRESSURE_PROFILE"

reproduce_args=(
    run
    --workdir "$WORK"
    --binary "$BINARY"
    --pre-sha256 "$PRE_SHA256"
    --timeout 6900
    --stale 0
    --tsd "$TARGET_SIEVING_DIM"
)
if [[ "$CONTINUE_AFTER_TARGET" == 1 ]]; then
    reproduce_args+=(--continue-after-target)
fi

apptainer exec --nv \
    --bind "$LARGE_BASE:$LARGE_BASE" \
    --pwd "$REPRO_SOURCE" \
    "$IMAGE" python3 "$REPRO" "${reproduce_args[@]}"
