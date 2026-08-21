#!/bin/bash
set -euo pipefail

SOURCE="$(cd "$(dirname "$0")" && pwd)"
LOG_ROOT="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/logs/svp163-164"
mkdir -p "$LOG_ROOT"
cd "$SOURCE"

prep_job="$(sbatch --parsable "${SOURCE}/prepare_svp163_164_bkz70_a100x4.sbatch")"
svp164_job="$(sbatch --parsable --dependency="afterok:${prep_job}" "${SOURCE}/run_svp164_seed0_bkz70_a100x4.sbatch")"
# Do not allocate the two SVP-163 four-GPU sieves until the SVP-164 sieve
# has actually started.  `after:` is deliberately used (rather than
# `afterok:`): it releases this array on job start, while the seed inputs are
# already guaranteed by SVP-164's afterok dependency on the shared prep job.
svp163_job="$(sbatch --parsable --dependency="after:${svp164_job}" "${SOURCE}/run_svp163_search_bkz70_a100x4.sbatch")"

echo "preparation job: ${prep_job}"
echo "SVP-164 seed-0 job: ${svp164_job}"
echo "SVP-163 seed-0..6 array: ${svp163_job}"
