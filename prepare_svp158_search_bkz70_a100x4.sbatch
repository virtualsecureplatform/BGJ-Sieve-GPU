#!/bin/bash
#SBATCH --job-name=svp158-p70-prep
#SBATCH --partition=gr20100g
#SBATCH --rsc=p=1:c=8:t=1:m=32G
#SBATCH --time=03:00:00

set -euo pipefail

SOURCE="${SLURM_SUBMIT_DIR}"
LOG_ROOT="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/logs/svp158-search-bkz70"
mkdir -p "$LOG_ROOT"

pids=()
seeds=()
for seed in $(seq 1 8); do
    (
        export SVP_DIMENSION=158
        export SVP142_LATTICE_SEED="$seed"
        export SVP_INSTANCE="svp158-seed${seed}"
        export SVP142_INPUT_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/inputs/${SVP_INSTANCE}"
        export SVP142_REPRO="${SOURCE}/reproduce/svp158-search/reproduce.py"
        export SVP_PREPROCESS_MODE=potlll-bkz
        export SVP_BKZ_BETA=70
        export SVP_BKZ_LOOPS=8
        exec bash "${SOURCE}/prepare_svp142_seed0_a100x4.sbatch"
    ) >"${LOG_ROOT}/prep_${SLURM_JOB_ID}_${seed}.log" 2>&1 &
    pids+=("$!")
    seeds+=("$seed")
done

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "seed ${seeds[$i]} preprocessing completed"
    else
        rc=$?
        echo "seed ${seeds[$i]} preprocessing failed with exit ${rc}" >&2
        failed=1
    fi
done
exit "$failed"
