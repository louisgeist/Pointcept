#!/bin/bash
# Submit grid-then-seed on H100 with dataset-aware Slurm walltime:
#   H3D 4 h | DALES 8 h | ECLAIR 12 h (inferred from grid config path).
#
# Usage (same args as sbatch_grid_then_seeds_h100.sh):
#   ./submit_grid_then_seeds_h100.sh <grid_config> [weight.pth] [exp_name]
#   GRID_CONFIG=... [WEIGHT=...] [N_SEEDS=10] ./submit_grid_then_seeds_h100.sh
#   Random-init grid configs: omit weight (or WEIGHT=).
#
# Self-contained (no extra scripts/ imports required on Jean-Zay).
# Prefers native /usr/bin/sbatch (--time on CLI); falls back to patching
# #SBATCH --time in a temp copy when only the IMAGINE wrapper is available.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/sbatch_grid_then_seeds_h100.sh"

grid_then_seeds_resolve_time() {
    local cfg="${1:?grid config path required}"
    case "$cfg" in
        */h3d/* | *h3d* | */H3D/* | *H3D*) echo "04:00:00" ;;
        */dales/* | *dales* | */DALES/* | *DALES*) echo "08:00:00" ;;
        */eclair/* | *eclair* | */ECLAIR/* | *ECLAIR*) echo "12:00:00" ;;
        *) echo "05:00:00" ;;
    esac
}

grid_then_seeds_sbatch_supports_cli_flags() {
    local sbatch_cmd="${1:?}"
    "${sbatch_cmd}" --help 2>&1 | grep -qE '(^|[[:space:]])--time'
}

grid_then_seeds_resolve_sbatch() {
    if [[ -n "${SBATCH_CMD:-}" ]]; then
        echo "${SBATCH_CMD}"
        return
    fi
    local cand
    for cand in /usr/bin/sbatch "$(command -v sbatch 2>/dev/null)"; do
        [[ -n "${cand}" && -x "${cand}" ]] || continue
        if grid_then_seeds_sbatch_supports_cli_flags "${cand}"; then
            echo "${cand}"
            return
        fi
    done
    echo "sbatch"
}

GRID_CONFIG="${GRID_CONFIG:-${1:?grid config required (arg1 or GRID_CONFIG=)}}"
SLURM_TIME="${SLURM_TIME:-$(grid_then_seeds_resolve_time "$GRID_CONFIG")}"
export SLURM_TIME="${SLURM_TIME}"

SBATCH_CMD_RESOLVED="$(grid_then_seeds_resolve_sbatch)"

echo "grid config: ${GRID_CONFIG}"
echo "slurm --time: ${SLURM_TIME}"
echo "sbatch cmd: ${SBATCH_CMD_RESOLVED}"

if grid_then_seeds_sbatch_supports_cli_flags "${SBATCH_CMD_RESOLVED}"; then
    echo "submit mode: native sbatch (--time on CLI)"
    exec "${SBATCH_CMD_RESOLVED}" --time="${SLURM_TIME}" --export=ALL,SLURM_TIME="${SLURM_TIME}" \
        "${TEMPLATE}" "$@"
fi

tmp="$(mktemp "${TMPDIR:-/tmp}/grid_then_seeds_submit.XXXXXX.slurm")"
sed "s/^#SBATCH --time=.*/#SBATCH --time=${SLURM_TIME}/" "${TEMPLATE}" > "${tmp}"
chmod +x "${tmp}"

echo "submit mode: IMAGINE wrapper (temp script with #SBATCH --time)"
echo "submit script: ${tmp}"

"${SBATCH_CMD_RESOLVED}" "${tmp}" "$@"
rc=$?
rm -f "${tmp}"
exit "${rc}"
