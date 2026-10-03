#!/bin/bash
#SBATCH --job-name=svp164-csd143-inspect
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=4
#SBATCH --time=00:10:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp164-bgj/slurm-%x-%j.out
set -euo pipefail
: "${DIAG_JOB_ID:?set the running allocation to inspect}"
timeout 90s srun --jobid="${DIAG_JOB_ID}" --overlap -N1 -n1 bash -lc '
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    for pid in $(pgrep -u "$USER" -x hd_sieve); do
        exe=$(readlink "/proc/$pid/exe")
        timeout 60s gdb -batch -ex "set sysroot /proc/$pid/root" \
            -ex "set pagination off" -ex "thread apply all bt 10" "$exe" -p "$pid"
    done
'
