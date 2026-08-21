#!/bin/bash
#SBATCH --job-name=svp163-164-prep
#SBATCH --partition=gr20100g
#SBATCH --rsc=p=1:c=16:t=1:m=64G
#SBATCH --time=06:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp163-164/slurm-%x-%j.out

set -euo pipefail

SOURCE="${SLURM_SUBMIT_DIR}"
LOG_ROOT="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/logs/svp163-164"
mkdir -p "$LOG_ROOT"

prepare_one() {
    local dimension="$1"
    local seed="$2"
    local repro="$3"
    local target_norm2="$4"
    local host_vec_max_dim="${5:-144}"
    local instance="svp${dimension}-seed${seed}"
    export SVP_DIMENSION="$dimension"
    export SVP142_LATTICE_SEED="$seed"
    export SVP_INSTANCE="$instance"
    export SVP142_INPUT_DIR="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/inputs/${instance}"
    export SVP142_REPRO="${SOURCE}/${repro}"
    export SVP142_TARGET_NORM2="$target_norm2"
    export SVP142_RAW_SHA256=GENERATE
    export SVP142_KNOWN_VECTOR="${SOURCE}/reproduce/svp158-search/no-known-vector.txt"
    export SVP_PREPROCESS_MODE=potlll-bkz
    export SVP_BKZ_BETA=70
    export SVP_BKZ_LOOPS=8
    export BGJ_ENABLE_PROFILING=0
    export POOL_HOST_VEC_MAX_DIM="$host_vec_max_dim"
    bash "${SOURCE}/prepare_svp142_seed0_a100x4.sbatch"
}

# Build the CSD145 host-slot binary while preparing SVP-164.
prepare_one 164 0 reproduce/svp164-seed0/reproduce.py 11009124 145 \
    >"${LOG_ROOT}/prep_${SLURM_JOB_ID}_svp164_0.log" 2>&1

# Then build the default CSD144 binary before parallelizing the remaining
# SVP-163 preprocessing tasks.
prepare_one 163 0 reproduce/svp163-search/reproduce.py 11108888 144 \
    >"${LOG_ROOT}/prep_${SLURM_JOB_ID}_svp163_0.log" 2>&1

pids=()
seeds=()
for seed in {1..6}; do
    (
        prepare_one 163 "$seed" reproduce/svp163-search/reproduce.py 11108888 144
    ) >"${LOG_ROOT}/prep_${SLURM_JOB_ID}_svp163_${seed}.log" 2>&1 &
    pids+=("$!")
    seeds+=("$seed")
done

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "SVP-163 seed ${seeds[$i]} preprocessing completed"
    else
        rc=$?
        echo "SVP-163 seed ${seeds[$i]} preprocessing failed with exit ${rc}" >&2
        failed=1
    fi
done
exit "$failed"
