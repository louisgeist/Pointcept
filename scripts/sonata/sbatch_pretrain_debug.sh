#!/bin/bash

# DEBUG smoke: Sonata-v1m2 pretrain on Flair3D+ (2× A100, ~20′).
# Submits one lin-probe debug job via LinProbeSbatchHook at epoch_2.
# Usage: sbatch scripts/sonata/sbatch_pretrain_debug.sh [exp_name]
#
# Jean-Zay compute-accounting tags (IMAGINE wrapper):
#   https://github.com/Archiel19/compute-accounting
# Keep --comment set so non-interactive sbatch never prompts.

#SBATCH -A uhn@a100
#SBATCH -C a100
#SBATCH --comment=flair3d,explore,pre-train
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err

#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread

#SBATCH --job-name=sonata_pre_dbg

EXP_NAME="${1:-sonata_pretrain_debug}"

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Exp name: $EXP_NAME"
    echo "Config: flair3d_default/pretrain-sonata-v1m2-flair3d-debug"
    echo "Hardware: 2× A100 (debug smoke)"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    echo "Working directory: $(pwd)"
    nvidia-smi
} > ${JOB_DIR}/job_info.log

module purge
module load arch/a100
module load cuda/12.1.0
module load miniforge/24.9.0

conda deactivate && while [ ! -z "$CONDA_DEFAULT_ENV" ]; do conda deactivate; done
conda activate pointcept_124

conda list > ${JOB_DIR}/conda_env.txt

export WANDB_MODE=offline
cd ${REPO_ROOT}

START_TIME=$(date +%s)

export JOB_DIR
sh scripts/train.sh -g 2 -d flair3d_default -c pretrain-sonata-v1m2-flair3d-debug -n "$EXP_NAME"

echo "Exp dir: ${JOB_DIR}" >> "${JOB_DIR}/job_info.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log
