#!/bin/bash

set -euo pipefail

SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"

if ! git -C "$SOURCE" diff --quiet || ! git -C "$SOURCE" diff --cached --quiet; then
    echo "Refusing to submit tracked, uncommitted source changes" >&2
    exit 2
fi

cd "$SOURCE"

# The two builds are serialized because both Makefiles use the source tree for
# intermediate files. Runs use commit/batch-keyed binaries under /LARGE0.
build_524="$({ sbatch --parsable \
    --job-name=bgj-build-f524 \
    --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT},FILTER_TASK_VECS=524288" \
    build_svp138_a100x4.sbatch; } | cut -d';' -f1)"
build_1m="$({ sbatch --parsable \
    --job-name=bgj-build-f1m \
    --dependency="afterok:${build_524}" \
    --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT},FILTER_TASK_VECS=1048576" \
    build_svp138_a100x4.sbatch; } | cut -d';' -f1)"

submit_trial() {
    local name="$1"
    local dependency="$2"
    local filter_task_vecs="$3"
    local reducer_threads="$4"
    local backpressure_profile="$5"
    local uid_load_pct="$6"
    local job

    job="$({ sbatch --parsable \
        --job-name="$name" \
        --time=01:00:00 \
        --dependency="afterok:${dependency}" \
        --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT},FILTER_TASK_VECS=${filter_task_vecs},BGJ2_REDUCER_THREADS=${reducer_threads},BACKPRESSURE_PROFILE=${backpressure_profile},GPU_UID_SHARED_LOAD_PCT=${uid_load_pct},TARGET_SIEVING_DIM=127,CONTINUE_AFTER_TARGET=1,BUILD=0" \
        run_svp138_a100x4.sbatch; } | cut -d';' -f1)"
    echo "trial=${name} job=${job} dependency=afterok:${dependency} filter=${filter_task_vecs} reducers=${reducer_threads} backpressure=${backpressure_profile} uid_load=${uid_load_pct}"
}

echo "source_commit=${SOURCE_COMMIT}"
echo "build_524=${build_524}"
echo "build_1m=${build_1m} dependency=afterok:${build_524}"
submit_trial a4-ctl127 "$build_524" 524288 48 tiered 60
submit_trial a4-red44 "$build_524" 524288 44 tiered 60
submit_trial a4-smooth "$build_524" 524288 48 smooth 60
submit_trial a4-uid80 "$build_524" 524288 48 tiered 80
submit_trial a4-wide1m "$build_1m" 1048576 48 tiered 60
