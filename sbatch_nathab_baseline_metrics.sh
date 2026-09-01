#!/bin/bash
# Nathab tile-distribution naive baselines (KL + TV) on the national test split.
#
# Usage (Jean Zay, from repo root):
#   sbatch sbatch_nathab_baseline_metrics.sh
#
# Outputs:
#   stats/flair3d/nathab_baseline_metrics/test/{summary.csv,results.json}
#   stats/flair3d/label_distribution_national/train/  (step 1, train marginals)

# Account must be first so sbatch parses it (variables in #SBATCH may not expand at submit time)
#SBATCH -A unv@cpu
#SBATCH --partition=cpu_p1
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err

#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

#SBATCH --job-name=nathab_baseline

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

module purge
module load miniforge/24.9.0

conda deactivate && while [ ! -z "$CONDA_DEFAULT_ENV" ]; do conda deactivate; done

module purge
module load miniforge/24.9.0
conda activate pointcept

cd ${REPO_ROOT}
export PYTHONPATH="${REPO_ROOT}"

# 1) train's global per-axis marginal (pi_hat_train), needed for the KL(qbar_test, pi_hat_train)
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
    --extra_pi_hat_csv_dir stats/flair3d/label_distribution_national/train \
    --extra_pi_hat_name train \
    --output_dir stats/flair3d/nathab_baseline_metrics

echo "Done. Outputs in ${REPO_ROOT}/stats/flair3d/nathab_baseline_metrics/test/"
