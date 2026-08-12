#!/bin/bash

set -euo pipefail

SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
common="SOURCE_COMMIT=${SOURCE_COMMIT},SVP_DIMENSION=143,SVP142_LATTICE_SEED=13,SVP_INSTANCE=svp143-seed13,SVP142_REPRO=${SOURCE}/reproduce/svp143-seed13/reproduce.py,SVP_PREPROCESS_MODE=lll-potlllbkz,SVP_BKZ_BETA=64,SVP_BKZ_LOOPS=8"

if ! git -C "$SOURCE" diff --quiet || ! git -C "$SOURCE" diff --cached --quiet; then
    echo "Refusing to submit tracked, uncommitted source changes" >&2
    exit 2
fi

cd "$SOURCE"
prep="$(sbatch --parsable --job-name=svp143-s13-lpotprep --time=01:15:00 \
    --export="ALL,${common}" prepare_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"
run="$(sbatch --parsable --job-name=svp143-s13-lpot64 \
    --dependency="afterok:${prep}" \
    --export="ALL,${common},SVP_SIEVE_SEED=0,TARGET_SIEVING_DIM=134,MIN_LIFTING_DIM=118" \
    run_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"

echo "prepare_job=${prep} run_job=${run} dependency=afterok:${prep}"
echo "preprocess=LLL-heuristic-mpfr256-then-PotLLLBKZ64x8 MLD=118 TSD=134 target_norm2=8793148"
