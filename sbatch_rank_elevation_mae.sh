#!/bin/bash
# Rank elevation MAE per ROI from an already-dumped test `result/` directory.
# Numpy-only (no GPU compute) but billed on unv@h100 -- no CPU hours on this
# allocation. Predictions are `{tile}_reg_elevation.npy` written by
# MultiTaskTester; GT is the preprocessed `elevation.npy`.
#
# Usage (Jean Zay):
#   sbatch sbatch_rank_elevation_mae.sh 873542
#   sbatch sbatch_rank_elevation_mae.sh /lustre/.../logs/slurm/873542
#
# Env overrides:
#   PRED_ROOT=/path/to/result          # default: <job>/result
#   DATA_ROOT=/path/to/flair3d_plus    # default: $REPO_ROOT/data/flair3d_plus
#   DEPARTMENTS=D068,D075              # empty = every department in the dump
#   TOP_SUBTILES=15
#   OUT_DIR=...                        # default: this Slurm job's log dir
#
# After a restricted test dump (include_names), only those ROIs are ranked.

# Account must be first so sbatch parses it (variables in #SBATCH may not expand at submit time)
#SBATCH -A unv@h100
#SBATCH -C h100
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err

#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

#SBATCH --job-name=elev_mae_rank

set -eo pipefail

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept

ARG="${1:-${SOURCE_JOB_ID:-873542}}"
if [[ "${ARG}" == */* ]]; then
    SAVE_PATH="${ARG%/}"
    SOURCE_JOB_ID="$(basename "${SAVE_PATH}")"
else
    SOURCE_JOB_ID="${ARG}"
    SAVE_PATH="${REPO_ROOT}/logs/slurm/${SOURCE_JOB_ID}"
fi

PRED_ROOT="${PRED_ROOT:-${SAVE_PATH}/result}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/flair3d_plus}"
DEPARTMENTS="${DEPARTMENTS-D068,D075}"
TOP_SUBTILES="${TOP_SUBTILES:-15}"

JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p "${JOB_DIR}"
OUT_DIR="${OUT_DIR:-${JOB_DIR}}"

cp "$0" "${JOB_DIR}/script.slurm"

{
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "Source dump job: ${SOURCE_JOB_ID}"
    echo "SAVE_PATH:  ${SAVE_PATH}"
    echo "PRED_ROOT:  ${PRED_ROOT}"
    echo "DATA_ROOT:  ${DATA_ROOT}"
    echo "DEPARTMENTS:${DEPARTMENTS:-all}"
    echo "OUT_DIR:    ${OUT_DIR}"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
} > "${JOB_DIR}/job_info.log"

if [ ! -d "${PRED_ROOT}" ]; then
    echo "ERROR: pred root not found: ${PRED_ROOT}" >&2
    echo "Pass a train/test job id whose result/ holds *_reg_elevation.npy" >&2
    exit 1
fi

module purge
module load miniforge/24.9.0

conda deactivate && while [ -n "${CONDA_DEFAULT_ENV:-}" ]; do conda deactivate; done

module purge
module load arch/h100
module load miniforge/24.9.0
conda activate pointcept

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"

python scripts/rank_elevation_mae.py \
    --pred-root "${PRED_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --departments "${DEPARTMENTS}" \
    --top-subtiles "${TOP_SUBTILES}"

echo "Done. Ranking CSVs in ${OUT_DIR}" | tee -a "${JOB_DIR}/job_info.log"
