#!/bin/bash

# Sonata-v1m2 pretrain on Flair3D+ (8× A100).
# Usage: sbatch scripts/sonata/sbatch_pretrain.sh [exp_name]

#SBATCH -A uhn@a100
#SBATCH -C a100
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err

#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread

#SBATCH --job-name=sonata_pretrain

EXP_NAME="${1:-sonata_pretrain_flair3dplus}"

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Exp name: $EXP_NAME"
    echo "Config: flair3d_default/pretrain-sonata-v1m2-flair3d"
    echo "Hardware: 8× A100"
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
conda activate pointcept

conda list > ${JOB_DIR}/conda_env.txt

export WANDB_MODE=offline
cd ${REPO_ROOT}

START_TIME=$(date +%s)

export JOB_DIR
sh scripts/train.sh -g 8 -d flair3d_default -c pretrain-sonata-v1m2-flair3d -n "$EXP_NAME"

echo "Exp dir: ${JOB_DIR}" >> "${JOB_DIR}/job_info.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log
