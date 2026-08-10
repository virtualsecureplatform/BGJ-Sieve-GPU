#!/bin/bash

set -euo pipefail

SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SVP142_LATTICE_SEED=42
exec "${SOURCE}/submit_svp142_seed0_a100x4.sh"
