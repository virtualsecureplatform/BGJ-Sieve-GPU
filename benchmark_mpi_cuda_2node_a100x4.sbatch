#!/bin/bash
#SBATCH --job-name=bgj-mpi-gpu-probe
#SBATCH --partition=gr20100g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=16G
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
mpirun --bind-to none --map-by ppr:1:node -np 2 \
    "${RESULT_ROOT}/mpi_cuda_pingpong" | tee "${RESULT_ROOT}/one-gpu-per-node.csv"

echo "probe=four_gpu_pairs"
mpirun --bind-to none --map-by ppr:4:node -np 8 \
    "${RESULT_ROOT}/mpi_cuda_pingpong" | tee "${RESULT_ROOT}/four-gpu-pairs.csv"

echo "result_root=${RESULT_ROOT}"
