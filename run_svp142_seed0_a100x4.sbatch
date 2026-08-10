#!/bin/bash
#SBATCH --job-name=svp142-s0-4g
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=03:00:00
#SBATCH --output=svp142-4g_%j.log

set -euo pipefail

LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SOURCE="${SLURM_SUBMIT_DIR}"
ACTUAL_SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
SOURCE_COMMIT="${SOURCE_COMMIT:-${ACTUAL_SOURCE_COMMIT}}"
FILTER_TASK_VECS="${FILTER_TASK_VECS:-524288}"
LATTICE_SEED="${SVP142_LATTICE_SEED:-0}"
IMAGE="${BGJ_IMAGE:-${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif}"
INPUT_DIR="${SVP142_INPUT_DIR:-${LARGE_BASE}/inputs/svp142-seed${LATTICE_SEED}}"
RUN_DIR="${SVP142_RUN_DIR:-${LARGE_BASE}/runs/svp142-seed${LATTICE_SEED}/job-${SLURM_JOB_ID}}"
BINARY_ROOT="${BGJ_BINARY_ROOT:-${LARGE_BASE}/builds}"
BINARY="${BGJ_BINARY:-${BINARY_ROOT}/hd_sieve-${SOURCE_COMMIT}-svp142-500g-ftv${FILTER_TASK_VECS}}"
BINARY_MANIFEST="${BINARY}.manifest"
REPRO="${SVP142_REPRO:-${SOURCE}/reproduce/svp142-seed${LATTICE_SEED}/reproduce.py}"
TARGET_SIEVING_DIM="${TARGET_SIEVING_DIM:-132}"
MIN_LIFTING_DIM="${MIN_LIFTING_DIM:-118}"
BGJ2_REDUCER_THREADS="${BGJ2_REDUCER_THREADS:-48}"
POOL_THREADS="${POOL_THREADS:-8}"
GPU_UID_SHARED_LOAD_PCT="${GPU_UID_SHARED_LOAD_PCT:-60}"
BACKPRESSURE_PROFILE="${BACKPRESSURE_PROFILE:-tiered}"

if [[ "$ACTUAL_SOURCE_COMMIT" != "$SOURCE_COMMIT" ]]; then
    echo "Source moved after submission: expected $SOURCE_COMMIT, found $ACTUAL_SOURCE_COMMIT" >&2
    exit 2
fi
if [[ ! "$TARGET_SIEVING_DIM" =~ ^[0-9]+$ ]] || (( TARGET_SIEVING_DIM < 118 || TARGET_SIEVING_DIM > 142 )); then
    echo "TARGET_SIEVING_DIM must be in [118, 142]" >&2
    exit 2
fi
if [[ ! "$MIN_LIFTING_DIM" =~ ^[0-9]+$ ]] || (( MIN_LIFTING_DIM < 100 || MIN_LIFTING_DIM > TARGET_SIEVING_DIM )); then
    echo "MIN_LIFTING_DIM must be in [100, TARGET_SIEVING_DIM]" >&2
    exit 2
fi
if [[ ! "$BGJ2_REDUCER_THREADS" =~ ^[0-9]+$ ]] || (( BGJ2_REDUCER_THREADS < 4 || BGJ2_REDUCER_THREADS > 64 || BGJ2_REDUCER_THREADS % 4 != 0 )); then
    echo "BGJ2_REDUCER_THREADS must be a multiple of 4 in [4, 64]" >&2
    exit 2
fi
if [[ ! -x "$BINARY" || ! -f "$BINARY_MANIFEST" ]]; then
    echo "Missing compute-built binary or manifest: $BINARY" >&2
    exit 2
fi
grep -Fqx "source_commit=${SOURCE_COMMIT}" "$BINARY_MANIFEST"
grep -Fqx "filter_task_vecs=${FILTER_TASK_VECS}" "$BINARY_MANIFEST"
grep -Fqx "cache_profile=svp142-500g" "$BINARY_MANIFEST"

echo "node=$(hostname) job=${SLURM_JOB_ID} visible_gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "source_commit=${SOURCE_COMMIT} binary=${BINARY}"
echo "input_dir=${INPUT_DIR} run_dir=${RUN_DIR} MLD=${MIN_LIFTING_DIM} TSD=${TARGET_SIEVING_DIM}"
echo "reducer_threads=${BGJ2_REDUCER_THREADS} pool_threads=${POOL_THREADS} uid_load_pct=${GPU_UID_SHARED_LOAD_PCT} backpressure=${BACKPRESSURE_PROFILE}"
sha256sum "$BINARY"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export APPTAINERENV_HD_PROFILE_BUCKET_STATS=0
export APPTAINERENV_HD_POOL_THREADS="$POOL_THREADS"
export APPTAINERENV_HD_POOL_BUFFER_CACHE=1
export APPTAINERENV_HD_BGJ2_REDUCER_THREADS="$BGJ2_REDUCER_THREADS"
export APPTAINERENV_HD_GPU_UID_SHARED_LOAD_PCT="$GPU_UID_SHARED_LOAD_PCT"
export APPTAINERENV_HD_BACKPRESSURE_PROFILE="$BACKPRESSURE_PROFILE"

apptainer exec --nv \
    --bind "$LARGE_BASE:$LARGE_BASE" \
    --pwd "$SOURCE" \
    "$IMAGE" python3 "$REPRO" run \
        --input-dir "$INPUT_DIR" \
        --run-dir "$RUN_DIR" \
        --binary "$BINARY" \
        --timeout 10500 \
        --mld "$MIN_LIFTING_DIM" \
        --tsd "$TARGET_SIEVING_DIM"
