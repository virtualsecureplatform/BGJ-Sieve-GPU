#!/bin/bash

set -euo pipefail

SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
common="SOURCE_COMMIT=${SOURCE_COMMIT},SVP142_LATTICE_SEED=43,SVP_INSTANCE=svp142-seed43,SVP142_REPRO=${SOURCE}/reproduce/svp142-search/reproduce.py"
prep="$(sbatch --parsable \
    --job-name="svp142-s43-cache" \
    --export="ALL,${common}" \
    prepare_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"
echo "cache_prepare_job=${prep}"
for pct in 70 80; do
    job="$(sbatch --parsable \
        --job-name="svp142-s43-u${pct}" \
        --dependency="afterok:${prep}" \
        --export="ALL,${common},GPU_UID_SHARED_LOAD_PCT=${pct}" \
        run_svp142_seed0_a100x4.sbatch | cut -d';' -f1)"
    echo "uid_load_pct=${pct} job=${job}"
done
