#!/bin/bash

set -euo pipefail

SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
common="SOURCE_COMMIT=${SOURCE_COMMIT},SVP_DIMENSION=146,SVP142_LATTICE_SEED=14,SVP_INSTANCE=svp146-seed14,SVP142_REPRO=${SOURCE}/reproduce/svp146-seed14/reproduce.py,SVP_PREPROCESS_MODE=lll-potlllbkz,SVP_BKZ_BETA=64,SVP_BKZ_LOOPS=8"

if ! git -C "$SOURCE" diff --quiet || ! git -C "$SOURCE" diff --cached --quiet; then
    echo "Refusing to submit tracked, uncommitted source changes" >&2
    exit 2
fi

cd "$SOURCE"
prep="$(sbatch --parsable --job-name=svp146-s14-lpotprep --time=01:15:00 \
    --export="ALL,${common}" prepare_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"
run="$(sbatch --parsable --job-name=svp146-s14-lpot64 --time=04:00:00 \
    --dependency="afterok:${prep}" \
    --export="ALL,${common},SVP_SIEVE_SEED=0,TARGET_SIEVING_DIM=136,MIN_LIFTING_DIM=120,SVP_TIMEOUT_SECONDS=14100" \
    run_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"

echo "prepare_job=${prep} run_job=${run} dependency=afterok:${prep}"
echo "preprocess=LLL-heuristic-mpfr256-then-PotLLLBKZ64x8 MLD=120 TSD=136 target_norm2=9142369"
