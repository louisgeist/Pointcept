#!/bin/bash
# Resolve Slurm walltime for grid-then-seed jobs from the grid config path.
# Source this file or call grid_then_seeds_resolve_time <grid_config>.

grid_then_seeds_resolve_time() {
    local cfg="${1:?grid config path required}"
    case "$cfg" in
        */h3d/* | *h3d* | */H3D/* | *H3D*)
            echo "04:00:00"
            ;;
        */dales/* | *dales* | */DALES/* | *DALES*)
            echo "08:00:00"
            ;;
        */eclair/* | *eclair* | */ECLAIR/* | *ECLAIR*)
            echo "12:00:00"
            ;;
        *)
            echo "05:00:00"
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    grid_then_seeds_resolve_time "$@"
fi
