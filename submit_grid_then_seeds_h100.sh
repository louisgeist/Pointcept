#!/bin/bash
# Submit grid-then-seed on H100 with dataset-aware Slurm walltime:
#   H3D 4 h | DALES 8 h | ECLAIR 12 h (inferred from grid config path).
#
# Usage (same args as sbatch_grid_then_seeds_h100.sh):
#   ./submit_grid_then_seeds_h100.sh <grid_config> <weight.pth> [exp_name]
#   GRID_CONFIG=... WEIGHT=... [N_SEEDS=10] ./submit_grid_then_seeds_h100.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/grid_then_seeds_resolve_time.sh
source "${SCRIPT_DIR}/scripts/grid_then_seeds_resolve_time.sh"
# shellcheck source=scripts/grid_then_seeds_submit.sh
source "${SCRIPT_DIR}/scripts/grid_then_seeds_submit.sh"

grid_then_seeds_submit "${SCRIPT_DIR}/sbatch_grid_then_seeds_h100.sh" "$@"
