#!/bin/bash
#SBATCH --job-name=bgj-mpi-transport-probe
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=8
#SBATCH --time=00:10:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/mpi/slurm-%x-%j.out
set -euo pipefail
: "${SOURCE_COMMIT:?}"
export MPI_BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU"
export MPI_SIF="${MPI_BASE}/bgj-sieve-gpu-cuda13.3.sif"
RUN="${MPI_BASE}/diagnostics/mpi-transport-${SLURM_JOB_ID}"
mkdir -p "${RUN}/source"
git -C "${SLURM_SUBMIT_DIR}" archive "${SOURCE_COMMIT}" | tar -x -C "${RUN}/source"
SOURCE="${RUN}/source"
MPI_PREFIX="$(dirname "$(dirname "$(readlink -e "$(command -v mpicxx)")")")"
export MPI_LIBDIR="$(mpicxx --showme:link | tr ' ' '\n' | sed -n 's/^-L//p' | head -1)"
mpicxx -O2 "${SOURCE}/tools/mpi_transport_probe.cpp" -o "${RUN}/probe"
export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx OMPI_MCA_pml_ucx_tls=any OMPI_MCA_pml_ucx_devices=any PMIX_MCA_gds=hash
launch=(srun --mpi=pmix_v2 --immediate=60 --nodes=2 --ntasks=2 --ntasks-per-node=1
        --gpus=8 --gres=gpu:4 --cpus-per-task=128 --cpu-bind=none --kill-on-bad-exit=1)
"${launch[@]}" "${MPI_PREFIX}/bin/ucx_info" -d > "${RUN}/host-devices.log" 2>&1
"${launch[@]}" bash "${SOURCE}/tools/mpi_device_probe.sh" > "${RUN}/host-access.log" 2>&1
for mode in baseline host-rdma; do
    export HD_MPI_HOST_RDMA=0
    if [[ "${mode}" == host-rdma ]]; then export HD_MPI_HOST_RDMA=1; fi
    export MPI_BINARY=/bin/bash
    "${launch[@]}" bash "${SOURCE}/reproduce/svp167-search/mpi_rank.sh" \
        "${SOURCE}/tools/mpi_device_probe.sh" > "${RUN}/${mode}-access.log" 2>&1
    export MPI_BINARY="${MPI_PREFIX}/bin/ucx_info"
    "${launch[@]}" bash "${SOURCE}/reproduce/svp167-search/mpi_rank.sh" -d > "${RUN}/${mode}-devices.log" 2>&1
    export MPI_BINARY=/usr/bin/ldd
    "${launch[@]}" bash "${SOURCE}/reproduce/svp167-search/mpi_rank.sh" \
        "${MPI_LIBDIR}/ucx/libuct_ib.so" > "${RUN}/${mode}-ib-libraries.log" 2>&1 || true
    export MPI_BINARY="${RUN}/probe" APPTAINERENV_UCX_LOG_LEVEL=info
    rc=0
    timeout --signal=TERM --kill-after=10 90 "${launch[@]}" bash \
        "${SOURCE}/reproduce/svp167-search/mpi_rank.sh" > "${RUN}/${mode}-bandwidth.log" 2>&1 || rc=$?
    echo "${mode} returncode=${rc}"
    rg 'TRANSPORT-PROBE|ep_cfg|ERROR|WARN' "${RUN}/${mode}-bandwidth.log" || true
done
echo "Probe logs: ${RUN}"
