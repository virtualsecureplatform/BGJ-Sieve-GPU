#!/bin/bash
#SBATCH --job-name=svp167-potlll-sim
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=1
#SBATCH --time=1-00:00:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out
set -euo pipefail
BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
RUN="${BASE}/runs/svp167-search/potlll-simulation-${SLURM_JOB_ID}"
SIF="${BASE}/bgj-sieve-gpu-cuda13.3.sif"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "${SLURM_SUBMIT_DIR}" rev-parse HEAD)}"
FPLLL="${BASE}/deps/fplll-b0e38e31085e"
GENERATOR="${BASE}/deps/ntl-9.3.0-svp-generator/bin/svp_challenge_generator"
mkdir -p "${RUN}/source"
git -C "${SLURM_SUBMIT_DIR}" archive "${SOURCE_COMMIT}" | tar -x -C "${RUN}/source"
SOURCE="${RUN}/source"
cp "${SLURM_SUBMIT_DIR}/../pro-pnj-bkz/PnJBKZ_Simulator/Fig2_Fig3_Fig4_Verification_Experiments_of_PnjBKZ_Simulator_2025_LLL_practice.py" "${RUN}/reference-simulator.py"
echo "Source ${SOURCE_COMMIT}; one GPU allocation; eight concurrent single-threaded PotLLL processes"
export APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1
apptainer exec --bind "${BASE}" "${SIF}" env LD_LIBRARY_PATH="${FPLLL}/lib" \
    "${FPLLL}/bin/fplll" --help | grep -q potlll
# Build only CPU objects needed for GSO extraction, on the compute node.
apptainer exec --bind "${BASE}" "${SIF}" bash -lc \
    "make -C '${SOURCE}/src' -j 4 lattice.o quad.o lll.o vec.o utils.o && g++ -O2 -fopenmp '${SOURCE}/tools/dump_gso.cpp' '${SOURCE}/src/lattice.o' '${SOURCE}/src/quad.o' '${SOURCE}/src/lll.o' '${SOURCE}/src/vec.o' '${SOURCE}/src/utils.o' -lntl -lgmp -lm -o '${RUN}/dump_gso'"
prepare_seed() {
    local seed="$1"
    local DIR
    DIR="${RUN}/seed-${seed}"
    mkdir -p "${DIR}"
    apptainer exec --bind "${BASE}" "${SIF}" "${GENERATOR}" 167 "${seed}" > "${DIR}/input.raw.tmp"
    mv "${DIR}/input.raw.tmp" "${DIR}/input.raw"
    echo "PotLLL seed=${seed} start=$(date -Is)"
    /usr/bin/time -o "${DIR}/potlll.time" -f 'elapsed_seconds=%e user_seconds=%U system_seconds=%S' \
        apptainer exec --bind "${BASE}" "${SIF}" env LD_LIBRARY_PATH="${FPLLL}/lib" \
        "${FPLLL}/bin/fplll" -a potlll -m heuristic -f mpfr -p 256 "${DIR}/input.raw" > "${DIR}/potlll.basis.tmp"
    mv "${DIR}/potlll.basis.tmp" "${DIR}/potlll.basis"
    cat "${DIR}/potlll.time"
    apptainer exec --bind "${BASE}" "${SIF}" python3 "${SOURCE}/tools/check_basis_det_mod.py" "${DIR}/input.raw" "${DIR}/potlll.basis"
    apptainer exec --bind "${BASE}" "${SIF}" "${RUN}/dump_gso" "${DIR}/potlll.basis" > "${DIR}/potlll.gso"
}
PIDS=()
for seed in 0 1 2 3 4 5 6 7; do
    mkdir -p "${RUN}/seed-${seed}"
    prepare_seed "${seed}" > "${RUN}/seed-${seed}/prepare.log" 2>&1 &
    PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then FAILED=1; fi
done
for seed in 0 1 2 3 4 5 6 7; do
    echo "=== seed ${seed} ==="
    cat "${RUN}/seed-${seed}/prepare.log"
done
if [[ "${FAILED}" == 1 ]]; then
    echo "At least one seed failed; skipping simulation and BKZ submission." >&2
    exit 1
fi
apptainer exec --bind "${BASE}" "${SIF}" python3 "${SOURCE}/tools/plan_svp167_search.py" "${RUN}" "${RUN}/reference-simulator.py"
SEED=$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["selected"]; print(s["seed"] if s else "none")' "${RUN}/summary.json")
if [[ "${SEED}" != none ]]; then
    # Leave this one-GPU allocation before the four-GPU solver starts.
    sbatch --dependency="afterok:${SLURM_JOB_ID}" \
        --export="ALL,SEARCH_RUN=${RUN},SELECTED_SEED=${SEED},SOURCE_COMMIT=${SOURCE_COMMIT}" \
        "${SLURM_SUBMIT_DIR}/reproduce/svp167-search/bkz_candidate_a100x4.sbatch"
else
    echo "No configuration predicted to reach 3405 with BSD <=143; no BKZ submitted."
fi
