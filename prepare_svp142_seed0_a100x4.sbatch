#!/bin/bash
#SBATCH --job-name=svp142-prep
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=1
#SBATCH --time=00:45:00
#SBATCH --output=svp142-prep_%j.log

set -euo pipefail

LARGE_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
SOURCE="${SLURM_SUBMIT_DIR}"
ACTUAL_SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
SOURCE_COMMIT="${SOURCE_COMMIT:-${ACTUAL_SOURCE_COMMIT}}"
FILTER_TASK_VECS="${FILTER_TASK_VECS:-524288}"
LATTICE_SEED="${SVP142_LATTICE_SEED:-0}"
INSTANCE="${SVP_INSTANCE:-svp142-seed${LATTICE_SEED}}"
IMAGE="${BGJ_IMAGE:-${LARGE_BASE}/bgj-sieve-gpu-cuda13.3.sif}"
PREPROCESS_IMAGE="${SVP_PREPROCESS_IMAGE:-${IMAGE}}"
INPUT_DIR="${SVP142_INPUT_DIR:-${LARGE_BASE}/inputs/${INSTANCE}}"
BINARY_ROOT="${BGJ_BINARY_ROOT:-${LARGE_BASE}/builds}"
SOLVER_KEY="$("${SOURCE}/tools/solver_cache_key.sh" "$SOURCE" "$SOURCE_COMMIT" "$FILTER_TASK_VECS" | sha256sum | awk '{print $1}')"
BINARY="${BGJ_BINARY:-${BINARY_ROOT}/hd_sieve-${SOLVER_KEY}-a100x4-ftv${FILTER_TASK_VECS}}"
MANIFEST="${BINARY}.manifest"
REPRO="${SVP142_REPRO:-${SOURCE}/reproduce/svp142-seed${LATTICE_SEED}/reproduce.py}"
DEFINES="-DHD_A100X4_500G_CACHE_PROFILE=1 -DHD_FILTER_TASK_VECS=${FILTER_TASK_VECS}"
FPLLL_COMMIT="$(git -C "${SOURCE}/dep/fplll" rev-parse HEAD)"
FPLLL_PREFIX="${BGJ_FPLLL_PREFIX:-${LARGE_BASE}/deps/fplll-${FPLLL_COMMIT:0:12}}"
FPLLL_BUILDER="${SOURCE}/tools/build_fplll.sh"
LEGACY_NTL_PREFIX="${LARGE_BASE}/deps/ntl-9.3.0-svp-generator"
PREPROCESS_MODE="${SVP_PREPROCESS_MODE:-lll-bkz}"
BLASTER_COMMIT="$(git -C "${SOURCE}/dep/BLASter" rev-parse HEAD)"
BLASTER_PREFIX="${LARGE_BASE}/deps/BLASter-${BLASTER_COMMIT:0:12}"

if [[ "$ACTUAL_SOURCE_COMMIT" != "$SOURCE_COMMIT" ]]; then
    echo "Source moved after submission: expected $SOURCE_COMMIT, found $ACTUAL_SOURCE_COMMIT" >&2
    exit 2
fi
if ! git -C "$SOURCE" diff --quiet || ! git -C "$SOURCE" diff --cached --quiet; then
    echo "Refusing to use tracked, uncommitted source changes" >&2
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
if [[ ! "$LATTICE_SEED" =~ ^[0-9]+$ ]]; then
    echo "SVP142_LATTICE_SEED must be a nonnegative integer" >&2
    exit 2
fi

mkdir -p "$INPUT_DIR" "$BINARY_ROOT"
echo "node=$(hostname) job=${SLURM_JOB_ID} source_commit=${SOURCE_COMMIT}"
echo "image=${IMAGE}"
echo "preprocess_image=${PREPROCESS_IMAGE} preprocess_mode=${PREPROCESS_MODE}"
echo "input_dir=${INPUT_DIR}"
echo "binary=${BINARY}"
echo "solver_key=${SOLVER_KEY}"
echo "fplll_commit=${FPLLL_COMMIT} fplll_prefix=${FPLLL_PREFIX}"

if [[ "$LATTICE_SEED" != "0" ]]; then
    apptainer exec --bind "$LARGE_BASE:$LARGE_BASE" --pwd "$SOURCE" "$IMAGE" \
        "${SOURCE}/tools/build_legacy_ntl_generator.sh" \
        "$LEGACY_NTL_PREFIX" "${SOURCE}/dep/dist/ntl-9.3.0.tar.gz" \
        "${SOURCE}/tools/svp_challenge_generator.cpp"
    export APPTAINERENV_SVP142_LEGACY_GENERATOR="${LEGACY_NTL_PREFIX}/bin/svp_challenge_generator"
fi
if [[ "$PREPROCESS_MODE" == "lll-deeplll-bkz" ]]; then
    apptainer exec --bind "$LARGE_BASE:$LARGE_BASE" --pwd "$SOURCE" "$PREPROCESS_IMAGE" \
        "${SOURCE}/tools/build_blaster.sh" "$BLASTER_PREFIX" \
        "${SOURCE}/dep/BLASter" "$BLASTER_COMMIT"
    export APPTAINERENV_PYTHONPATH="${BLASTER_PREFIX}:${SOURCE}/dep/BLASter/src"
    export APPTAINERENV_BGJ_BLASTER_COMMIT="$BLASTER_COMMIT"
    export APPTAINERENV_SVP_BLASTER_APP="${BLASTER_PREFIX}/app.py"
fi

# Build the current vendored fplll revision under LARGE0 if it is not cached. This
# keeps the immutable container unchanged while pinning the actual CLI used by
# preprocessing. Download, build, and both fplll stages stay in this allocation.
apptainer exec --nv \
    --bind "$LARGE_BASE:$LARGE_BASE" \
    --pwd "$SOURCE" \
    "$IMAGE" "$FPLLL_BUILDER" "$FPLLL_PREFIX" "${SOURCE}/dep/fplll" "$FPLLL_COMMIT"

export APPTAINERENV_PATH="${FPLLL_PREFIX}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export APPTAINERENV_LD_LIBRARY_PATH="${FPLLL_PREFIX}/lib:/usr/local/cuda/lib64"
export APPTAINERENV_BGJ_FPLLL_COMMIT="$FPLLL_COMMIT"
apptainer exec --nv \
    --bind "$LARGE_BASE:$LARGE_BASE" \
    --pwd "$SOURCE" \
    "$PREPROCESS_IMAGE" python3 "$REPRO" prepare \
        --input-dir "$INPUT_DIR" \
        --strategy "${FPLLL_PREFIX}/share/fplll/strategies/default.json"

if [[ -x "$BINARY" && -f "$MANIFEST" ]] &&
   grep -Fqx "solver_key=${SOLVER_KEY}" "$MANIFEST" &&
   grep -Fqx "filter_task_vecs=${FILTER_TASK_VECS}" "$MANIFEST" &&
   grep -Fqx "cache_profile=svp142-500g" "$MANIFEST"; then
    echo "Reusing existing verified compute-built binary"
    sha256sum "$BINARY"
    exit 0
fi

# Promote a verified commit-keyed binary when its actual solver trees and
# compile configuration match. This makes the new content cache effective
# immediately without recompiling unchanged CUDA/C++ sources.
for old_manifest in "${BINARY_ROOT}"/hd_sieve-*-svp142-500g-ftv"${FILTER_TASK_VECS}".manifest; do
    [[ -f "$old_manifest" ]] || continue
    old_commit="$(sed -n 's/^source_commit=//p' "$old_manifest" | head -1)"
    [[ "$old_commit" =~ ^[0-9a-f]{40}$ ]] || continue
    old_key="$("${SOURCE}/tools/solver_cache_key.sh" "$SOURCE" "$old_commit" "$FILTER_TASK_VECS" 2>/dev/null | sha256sum | awk '{print $1}')" || continue
    [[ "$old_key" == "$SOLVER_KEY" ]] || continue
    old_binary="${old_manifest%.manifest}"
    [[ -x "$old_binary" ]] || continue
    expected_sha="$(tail -1 "$old_manifest" | awk '{print $1}')"
    [[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || continue
    [[ "$(sha256sum "$old_binary" | awk '{print $1}')" == "$expected_sha" ]] || continue
    cp "$old_binary" "$BINARY"
    chmod 0755 "$BINARY"
    {
        echo "solver_key=${SOLVER_KEY}"
        echo "source_commit_built=${old_commit}"
        echo "filter_task_vecs=${FILTER_TASK_VECS}"
        echo "cache_profile=svp142-500g"
        sha256sum "$BINARY"
    } > "$MANIFEST"
    echo "Promoted matching verified binary from ${old_binary}"
    sha256sum "$BINARY"
    exit 0
done

apptainer exec --nv \
    --bind "$LARGE_BASE:$LARGE_BASE" \
    --pwd "$SOURCE" \
    "$IMAGE" bash -lc '
        set -euo pipefail
        make -C src clean NVCC=nvcc
        make -C app clean NVCC=nvcc
        make -C src INC_DIR="$1" NVCC=nvcc
        make -C app hd_sieve INC_DIR="$1" NVCC=nvcc
    ' _ "$DEFINES"

BINARY_TMP="${BINARY}.tmp-${SLURM_JOB_ID}"
MANIFEST_TMP="${MANIFEST}.tmp-${SLURM_JOB_ID}"
cp "$SOURCE/app/hd_sieve" "$BINARY_TMP"
chmod 0755 "$BINARY_TMP"
mv -f "$BINARY_TMP" "$BINARY"
{
    echo "solver_key=${SOLVER_KEY}"
    echo "source_commit_built=${SOURCE_COMMIT}"
    echo "filter_task_vecs=${FILTER_TASK_VECS}"
    echo "cache_profile=svp142-500g"
    sha256sum "$BINARY"
} > "$MANIFEST_TMP"
mv -f "$MANIFEST_TMP" "$MANIFEST"
sha256sum "$BINARY"
