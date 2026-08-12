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
    --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT}" \
    prepare_svp144_search_a100x4.sbatch | cut -d';' -f1)"
run="$(sbatch --parsable \
    --dependency="afterok:${prep}" \
    --export="ALL,SOURCE_COMMIT=${SOURCE_COMMIT}" \
    run_svp144_search_a100x4.sbatch | cut -d';' -f1)"

echo "prepare_array=${prep} seeds=8-20 concurrency=2"
echo "run_array=${run} seeds=8-20 concurrency=2 dependency=afterok:${prep}"
echo "preprocess=LLL-heuristic-mpfr256-then-PotLLLBKZ64x8 MLD=119 TSD=135 target_norm2=8849681"
