#!/bin/bash
#SBATCH --job-name=bgj-build-a100
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=1
#SBATCH --time=00:20:00
#SBATCH --output=bgj-build-a100_%j.log

set -euo pipefail

LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SOURCE="${SLURM_SUBMIT_DIR}"
ACTUAL_SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
SOURCE_COMMIT="${SOURCE_COMMIT:-${ACTUAL_SOURCE_COMMIT}}"
FILTER_TASK_VECS="${FILTER_TASK_VECS:-524288}"
IMAGE="${BGJ_IMAGE:-${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif}"
BINARY_ROOT="${BGJ_BINARY_ROOT:-${LARGE_BASE}/builds}"
BINARY="${BGJ_BINARY:-${BINARY_ROOT}/hd_sieve-${SOURCE_COMMIT}-ftv${FILTER_TASK_VECS}}"
MANIFEST="${BINARY}.manifest"

if [[ "$ACTUAL_SOURCE_COMMIT" != "$SOURCE_COMMIT" ]]; then
    echo "Source moved after submission: expected $SOURCE_COMMIT, found $ACTUAL_SOURCE_COMMIT" >&2
    exit 2
fi
if ! git -C "$SOURCE" diff --quiet || ! git -C "$SOURCE" diff --cached --quiet; then
    echo "Refusing to build tracked, uncommitted source changes" >&2
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

mkdir -p "$BINARY_ROOT"
echo "node=$(hostname) job=${SLURM_JOB_ID} source_commit=${SOURCE_COMMIT} filter_task_vecs=${FILTER_TASK_VECS}"
echo "binary=${BINARY}"

if [[ -x "$BINARY" && -f "$MANIFEST" ]] &&
   grep -Fqx "source_commit=${SOURCE_COMMIT}" "$MANIFEST" &&
   grep -Fqx "filter_task_vecs=${FILTER_TASK_VECS}" "$MANIFEST"; then
    echo "Reusing existing verified compute-built binary"
    sha256sum "$BINARY"
    exit 0
fi

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
MANIFEST_TMP="${MANIFEST}.tmp-${SLURM_JOB_ID}"
cp "$SOURCE/app/hd_sieve" "$BINARY_TMP"
chmod 0755 "$BINARY_TMP"
mv -f "$BINARY_TMP" "$BINARY"
{
    echo "source_commit=${SOURCE_COMMIT}"
    echo "filter_task_vecs=${FILTER_TASK_VECS}"
    sha256sum "$BINARY"
} > "$MANIFEST_TMP"
mv -f "$MANIFEST_TMP" "$MANIFEST"
sha256sum "$BINARY"
