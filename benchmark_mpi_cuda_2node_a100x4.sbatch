#!/bin/bash
#SBATCH --job-name=bgj-mpi-gpu-probe
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=8
#SBATCH --time=00:20:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/mpi-probe/slurm-%x-%j.out

set -euo pipefail

SOURCE="${SLURM_SUBMIT_DIR}"
RESULT_ROOT="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/mpi-probe/job-${SLURM_JOB_ID}"
mkdir -p "${RESULT_ROOT}" "/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/logs/mpi-probe"

nvcc -ccbin mpicxx -O3 -std=c++17 \
    "${SOURCE}/tools/mpi_cuda_pingpong.cu" \
    -o "${RESULT_ROOT}/mpi_cuda_pingpong"

echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST}"
echo "mpi=$(mpirun --version | head -1)"
echo "cuda=$(nvcc --version | tail -1)"
ompi_info --parsable --all | grep -E \
    'mpi_built_with_cuda_support:value|opal_built_with_cuda_support:value|pml:ucx:version' || true
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -lc \
    'hostname; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; nvidia-smi topo -m'

export UCX_LOG_LEVEL=warn
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx

echo "probe=one_gpu_per_node"
srun --mpi=pmix_v2 --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    "${RESULT_ROOT}/mpi_cuda_pingpong" | tee "${RESULT_ROOT}/one-gpu-per-node.csv"

echo "probe=four_gpu_pairs"
srun --mpi=pmix_v2 --nodes=2 --ntasks=8 --ntasks-per-node=4 \
    "${RESULT_ROOT}/mpi_cuda_pingpong" | tee "${RESULT_ROOT}/four-gpu-pairs.csv"

echo "result_root=${RESULT_ROOT}"
