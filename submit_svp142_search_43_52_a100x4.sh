#!/bin/bash

set -euo pipefail
SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SOURCE"
prep="$(sbatch --parsable prepare_svp142_search_a100x4.sbatch | cut -d';' -f1)"
run="$(sbatch --parsable --dependency="afterok:${prep}" run_svp142_search_a100x4.sbatch | cut -d';' -f1)"
echo "prepare_array=${prep} seeds=43-52 concurrency=1"
echo "run_array=${run} seeds=43-52 concurrency=2 dependency=afterok:${prep}"
