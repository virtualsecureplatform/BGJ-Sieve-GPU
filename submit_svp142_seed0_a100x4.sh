#!/bin/bash

set -euo pipefail

SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
FILTER_TASK_VECS="${FILTER_TASK_VECS:-524288}"
BGJ2_REDUCER_THREADS="${BGJ2_REDUCER_THREADS:-48}"
TARGET_SIEVING_DIM="${TARGET_SIEVING_DIM:-132}"
MIN_LIFTING_DIM="${MIN_LIFTING_DIM:-118}"

if ! git -C "$SOURCE" diff --quiet || ! git -C "$SOURCE" diff --cached --quiet; then
    echo "Refusing to submit tracked, uncommitted source changes" >&2
    exit 2
fi

cd "$SOURCE"
prepare_job="$({ sbatch --parsable \
    --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT},FILTER_TASK_VECS=${FILTER_TASK_VECS}" \
    prepare_svp142_seed0_a100x4.sbatch; } | cut -d';' -f1)"
run_job="$({ sbatch --parsable \
    --dependency="afterok:${prepare_job}" \
    --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT},FILTER_TASK_VECS=${FILTER_TASK_VECS},BGJ2_REDUCER_THREADS=${BGJ2_REDUCER_THREADS},TARGET_SIEVING_DIM=${TARGET_SIEVING_DIM},MIN_LIFTING_DIM=${MIN_LIFTING_DIM}" \
    run_svp142_seed0_a100x4.sbatch; } | cut -d';' -f1)"

echo "prepare_job=${prepare_job}"
echo "run_job=${run_job} dependency=afterok:${prepare_job}"
