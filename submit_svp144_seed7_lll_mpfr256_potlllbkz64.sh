#!/bin/bash

set -euo pipefail

SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
common="SOURCE_COMMIT=${SOURCE_COMMIT},SVP_DIMENSION=144,SVP142_LATTICE_SEED=7,SVP_INSTANCE=svp144-seed7,SVP142_REPRO=${SOURCE}/reproduce/svp144-seed7/reproduce.py,SVP_PREPROCESS_MODE=lll-potlllbkz,SVP_BKZ_BETA=64,SVP_BKZ_LOOPS=8"

if ! git -C "$SOURCE" diff --quiet || ! git -C "$SOURCE" diff --cached --quiet; then
    echo "Refusing to submit tracked, uncommitted source changes" >&2
    exit 2
fi

cd "$SOURCE"
prep="$(sbatch --parsable --job-name=svp144-s7-lpotprep --time=01:15:00 \
    --export="ALL,${common}" prepare_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"
run="$(sbatch --parsable --job-name=svp144-s7-lpot64 \
    --dependency="afterok:${prep}" \
    --export="ALL,${common},SVP_SIEVE_SEED=0,TARGET_SIEVING_DIM=135,MIN_LIFTING_DIM=119" \
    run_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"

echo "prepare_job=${prep} run_job=${run} dependency=afterok:${prep}"
echo "preprocess=LLL-heuristic-mpfr256-then-PotLLLBKZ64x8 MLD=119 TSD=135 target_norm2=8849681"
