#!/bin/bash
# Nathab tile-distribution naive baselines (KL + TV) on the national test split.
#
# Numpy-only (no GPU compute) but submitted on unv@h100 -- this allocation has no
# unv@cpu hours; same pattern as sbatch_rank_elevation_mae.sh.
#
# Usage (Jean Zay, from repo root):
#   sbatch sbatch_nathab_baseline_metrics.sh
#
# Outputs:
#   stats/flair3d/nathab_baseline_metrics/test/{summary.csv,results.json}
#   stats/flair3d/label_distribution_national/train/  (step 1, train marginals)

# Account must be first so sbatch parses it (variables in #SBATCH may not expand at submit time)
#SBATCH -A unv@h100
#SBATCH -C h100
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err

#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

#SBATCH --job-name=nathab_baseline

set -eo pipefail

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
} > ${JOB_DIR}/job_info.log

module purge
module load miniforge/24.9.0

conda deactivate && while [ -n "${CONDA_DEFAULT_ENV:-}" ]; do conda deactivate; done

module purge
module load arch/h100
module load cuda/12.4.1
module load miniforge/24.9.0
conda activate pointcept_124

cd ${REPO_ROOT}
# Pointops built for H100 (same as other unv@h100 jobs; harmless once count script no longer
# imports the full training stack, but kept for consistency with sbatch_pretrain_h100.sh).
POINTOPS_PATH=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_h100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
export PYTHONPATH="${REPO_ROOT}:${POINTOPS_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

# 1) train's global per-axis marginal (pi_hat_train), needed for the KL/TV(qbar_test, pi_hat_train)
#    column -- cheap relative to step 2, only aggregate counts are used, not per-tile.
python scripts/count_flair3d_train_label_distribution.py \
    --data_root data/flair3d_plus \
    --csv_manifest data/flair3d_plus/raw/scene_split_manifest.csv \
    --split train \
    --num_workers ${SLURM_CPUS_PER_TASK} \
    --output_dir stats/flair3d/label_distribution_national/train

# 2) main computation: H_a / H_a^TV, KL/TV(qbar_test,U), KL/TV(qbar_test,pi_hat_train) on test.
python scripts/compute_nathab_baseline_metrics.py \
    --data_root data/flair3d_plus \
    --csv_manifest data/flair3d_plus/raw/scene_split_manifest.csv \
    --split test \
    --num_workers ${SLURM_CPUS_PER_TASK} \
    --no_require_local_dir \
    --extra_pi_hat_csv_dir stats/flair3d/label_distribution_national/train \
    --extra_pi_hat_name train \
    --output_dir stats/flair3d/nathab_baseline_metrics

echo "Done. Outputs in ${REPO_ROOT}/stats/flair3d/nathab_baseline_metrics/test/" | tee -a ${JOB_DIR}/job_info.log
