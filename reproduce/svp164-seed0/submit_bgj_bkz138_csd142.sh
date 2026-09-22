#!/bin/bash
set -euo pipefail

SOURCE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${SOURCE}"

# Slurm opens #SBATCH --output before executing the batch script, so the
# directory cannot be created from inside the job itself.
mkdir -p "/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/logs/svp164-bgj"

prep_job="$(sbatch --parsable reproduce/svp164-seed0/preprocess_bgj_bkz138_a100x4.sbatch)"
sieve138_job="$(sbatch --parsable --dependency="afterok:${prep_job}" --export=ALL,TSD=138 reproduce/svp164-seed0/sieve_a100x4.sbatch)"
sieve142_job="$(sbatch --parsable --dependency="afterok:${sieve138_job}" --export=ALL,TSD=142 reproduce/svp164-seed0/sieve_a100x4.sbatch)"
printf 'submitted BKZ-138 job %s, CSD-138 job %s, and resumable CSD-142 job %s\n' \
    "${prep_job}" "${sieve138_job}" "${sieve142_job}"
