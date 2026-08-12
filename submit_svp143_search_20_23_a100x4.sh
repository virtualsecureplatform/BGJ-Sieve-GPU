#!/bin/bash

set -euo pipefail

SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"

if ! git -C "$SOURCE" diff --quiet || ! git -C "$SOURCE" diff --cached --quiet; then
    echo "Refusing to submit tracked, uncommitted source changes" >&2
    exit 2
fi

cd "$SOURCE"
prep="$(sbatch --parsable \
    --array=20-23%1 \
    --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT}" \
    prepare_svp143_search_a100x4.sbatch | cut -d';' -f1)"
run="$(sbatch --parsable \
    --array=20-23%2 \
    --dependency="afterok:${prep}" \
    --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT}" \
    run_svp143_search_a100x4.sbatch | cut -d';' -f1)"

echo "prepare_array=${prep} seeds=20-23 concurrency=1"
echo "run_array=${run} seeds=20-23 concurrency=2 dependency=afterok:${prep}"
echo "preprocess=LLL-DeepLLL4-BKZ64x8 MLD=118 TSD=134 target_norm2=8793148"
