#!/bin/bash
set -euo pipefail

SOURCE="$(cd "$(dirname "$0")" && pwd)"
LOG_ROOT="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/logs/svp163-164"
mkdir -p "$LOG_ROOT"
cd "$SOURCE"

prep_job="$(sbatch --parsable "${SOURCE}/prepare_svp163_164_bkz70_a100x4.sbatch")"
svp164_job="$(sbatch --parsable --dependency="afterok:${prep_job}" "${SOURCE}/run_svp164_seed0_bkz70_a100x4.sbatch")"
svp163_job="$(sbatch --parsable --dependency="afterok:${prep_job}" "${SOURCE}/run_svp163_search_bkz70_a100x4.sbatch")"

echo "preparation job: ${prep_job}"
echo "SVP-164 seed-0 job: ${svp164_job}"
echo "SVP-163 seed-0..6 array: ${svp163_job}"
