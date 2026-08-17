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
BGJ_ENABLE_PROFILING="${BGJ_ENABLE_PROFILING:-1}"
DIMENSION="${SVP_DIMENSION:-142}"
LATTICE_SEED="${SVP142_LATTICE_SEED:-0}"
INSTANCE="${SVP_INSTANCE:-svp142-seed${LATTICE_SEED}}"
IMAGE="${BGJ_IMAGE:-${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif}"
INPUT_DIR="${SVP142_INPUT_DIR:-${LARGE_BASE}/inputs/${INSTANCE}}"
RUN_DIR="${SVP142_RUN_DIR:-${LARGE_BASE}/runs/${INSTANCE}/job-${SLURM_JOB_ID}}"
BINARY_ROOT="${BGJ_BINARY_ROOT:-${LARGE_BASE}/builds}"
SOLVER_KEY="$("${SOURCE}/tools/solver_cache_key.sh" "$SOURCE" "$SOURCE_COMMIT" "$FILTER_TASK_VECS" "$BGJ_ENABLE_PROFILING" | sha256sum | awk '{print $1}')"
BINARY="${BGJ_BINARY:-${BINARY_ROOT}/hd_sieve-${SOLVER_KEY}-a100x4-ftv${FILTER_TASK_VECS}}"
BINARY_MANIFEST="${BINARY}.manifest"
REPRO="${SVP142_REPRO:-${SOURCE}/reproduce/svp142-seed${LATTICE_SEED}/reproduce.py}"
TARGET_SIEVING_DIM="${TARGET_SIEVING_DIM:-132}"
MIN_LIFTING_DIM="${MIN_LIFTING_DIM:-118}"
BGJ2_REDUCER_THREADS="${BGJ2_REDUCER_THREADS:-48}"
POOL_THREADS="${POOL_THREADS:-8}"
GPU_UID_SHARED_LOAD_PCT="${GPU_UID_SHARED_LOAD_PCT:-60}"
BACKPRESSURE_PROFILE="${BACKPRESSURE_PROFILE:-tiered}"
CONTINUE_AFTER_TARGET="${CONTINUE_AFTER_TARGET:-0}"
SVP_TIMEOUT_SECONDS="${SVP_TIMEOUT_SECONDS:-10500}"

if [[ "$ACTUAL_SOURCE_COMMIT" != "$SOURCE_COMMIT" ]]; then
    echo "Source moved after submission: expected $SOURCE_COMMIT, found $ACTUAL_SOURCE_COMMIT" >&2
    exit 2
fi
if [[ "$BGJ_ENABLE_PROFILING" != 0 && "$BGJ_ENABLE_PROFILING" != 1 ]]; then
    echo "BGJ_ENABLE_PROFILING must be 0 or 1" >&2
    exit 2
fi
if [[ ! "$TARGET_SIEVING_DIM" =~ ^[0-9]+$ ]] || (( TARGET_SIEVING_DIM < 118 || TARGET_SIEVING_DIM > DIMENSION )); then
    echo "TARGET_SIEVING_DIM must be in [118, ${DIMENSION}]" >&2
    exit 2
fi
if [[ ! "$MIN_LIFTING_DIM" =~ ^[0-9]+$ ]] || (( MIN_LIFTING_DIM < 100 || MIN_LIFTING_DIM > TARGET_SIEVING_DIM )); then
    echo "MIN_LIFTING_DIM must be in [100, TARGET_SIEVING_DIM]" >&2
    exit 2
fi
if [[ "$CONTINUE_AFTER_TARGET" != 0 && "$CONTINUE_AFTER_TARGET" != 1 ]]; then
    echo "CONTINUE_AFTER_TARGET must be 0 or 1" >&2
    exit 2
fi
if [[ ! "$SVP_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || (( SVP_TIMEOUT_SECONDS < 60 )); then
    echo "SVP_TIMEOUT_SECONDS must be an integer >= 60" >&2
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
grep -Fqx "solver_key=${SOLVER_KEY}" "$BINARY_MANIFEST"
grep -Fqx "filter_task_vecs=${FILTER_TASK_VECS}" "$BINARY_MANIFEST"
grep -Fqx "enable_profiling=${BGJ_ENABLE_PROFILING}" "$BINARY_MANIFEST"
grep -Fqx "cache_profile=svp142-500g" "$BINARY_MANIFEST"

echo "node=$(hostname) job=${SLURM_JOB_ID} visible_gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "source_commit=${SOURCE_COMMIT} solver_key=${SOLVER_KEY} enable_profiling=${BGJ_ENABLE_PROFILING} binary=${BINARY}"
echo "input_dir=${INPUT_DIR} run_dir=${RUN_DIR} MLD=${MIN_LIFTING_DIM} TSD=${TARGET_SIEVING_DIM} timeout=${SVP_TIMEOUT_SECONDS}s continue_after_target=${CONTINUE_AFTER_TARGET}"
echo "reducer_threads=${BGJ2_REDUCER_THREADS} pool_threads=${POOL_THREADS} bucket_threads=${APPTAINERENV_HD_BUCKET_THREADS:-auto} uid_load_pct=${GPU_UID_SHARED_LOAD_PCT} backpressure=${BACKPRESSURE_PROFILE}"
sha256sum "$BINARY"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export APPTAINERENV_HD_PROFILE_BUCKET_STATS=0
export APPTAINERENV_HD_POOL_THREADS="$POOL_THREADS"
export APPTAINERENV_HD_POOL_BUFFER_CACHE=1
export APPTAINERENV_HD_BGJ2_REDUCER_THREADS="$BGJ2_REDUCER_THREADS"
export APPTAINERENV_HD_GPU_UID_SHARED_LOAD_PCT="$GPU_UID_SHARED_LOAD_PCT"
export APPTAINERENV_HD_BACKPRESSURE_PROFILE="$BACKPRESSURE_PROFILE"

reproduce_args=(
    run
    --input-dir "$INPUT_DIR"
    --run-dir "$RUN_DIR"
    --binary "$BINARY"
    --timeout "$SVP_TIMEOUT_SECONDS"
    --mld "$MIN_LIFTING_DIM"
    --tsd "$TARGET_SIEVING_DIM"
)
if [[ "$CONTINUE_AFTER_TARGET" == 1 ]]; then
    reproduce_args+=(--continue-after-target)
fi

apptainer exec --nv \
    --bind "$LARGE_BASE:$LARGE_BASE" \
    --pwd "$SOURCE" \
    "$IMAGE" python3 "$REPRO" "${reproduce_args[@]}"
