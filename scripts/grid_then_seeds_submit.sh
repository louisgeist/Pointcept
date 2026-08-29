#!/bin/bash
# Shared submit helper for grid-then-seed jobs.
#
# Jean-Zay IMAGINE compute-accounting wraps `sbatch` in PATH with a narrow CLI
# (script path + args only — no native Slurm flags like --time/--export).
# When a native Slurm sbatch is available (typically /usr/bin/sbatch), prefer it
# and pass --time on the CLI. Otherwise inject #SBATCH --time into a temp copy.

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

grid_then_seeds_sbatch_supports_cli_flags() {
    local sbatch_cmd="${1:?}"
    "${sbatch_cmd}" --help 2>&1 | grep -qE '(^|[[:space:]])--time'
}

grid_then_seeds_submit() {
    local template="${1:?sbatch template required}"
    shift

    local grid_config="${GRID_CONFIG:-${1:?grid config required (arg1 or GRID_CONFIG=)}}"
    local slurm_time="${SLURM_TIME:-$(grid_then_seeds_resolve_time "$grid_config")}"
    export SLURM_TIME="${slurm_time}"

    local sbatch_cmd
    sbatch_cmd="$(grid_then_seeds_resolve_sbatch)"

    echo "grid config: ${grid_config}"
    echo "slurm --time: ${slurm_time}"
    echo "sbatch cmd: ${sbatch_cmd}"

    if grid_then_seeds_sbatch_supports_cli_flags "${sbatch_cmd}"; then
        echo "submit mode: native sbatch (--time on CLI)"
        "${sbatch_cmd}" --time="${slurm_time}" --export=ALL,SLURM_TIME="${slurm_time}" \
            "${template}" "$@"
        return $?
    fi

    local tmp
    tmp="$(mktemp "${TMPDIR:-/tmp}/grid_then_seeds_submit.XXXXXX.slurm")"
    sed "s/^#SBATCH --time=.*/#SBATCH --time=${slurm_time}/" "${template}" > "${tmp}"
    chmod +x "${tmp}"

    echo "submit mode: IMAGINE wrapper (temp script with #SBATCH --time)"
    echo "submit script: ${tmp}"

    "${sbatch_cmd}" "${tmp}" "$@"
    local rc=$?
    rm -f "${tmp}"
    return "${rc}"
}
