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

GRID_CONFIG="${GRID_CONFIG:-${1:?grid config required (arg1 or GRID_CONFIG=)}}"
SLURM_TIME="${SLURM_TIME:-$(grid_then_seeds_resolve_time "$GRID_CONFIG")}"

echo "grid config: ${GRID_CONFIG}"
echo "slurm --time: ${SLURM_TIME}"

exec sbatch --time="${SLURM_TIME}" --export=ALL,SLURM_TIME="${SLURM_TIME}" \
  "${SCRIPT_DIR}/sbatch_grid_then_seeds_h100.sh" "$@"
