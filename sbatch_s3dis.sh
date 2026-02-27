#!/bin/bash

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept

#SBATCH --output=${REPO_ROOT}/logs/slurm/%j/slurm.out
#SBATCH --error=${REPO_ROOT}/logs/slurm/%j/slurm.err
#SBATCH -A ppm@h100
#SBATCH -C h100

#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread


#SBATCH --job-name=s3dis_ptv3_test

JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    echo "Working directory: $(pwd)"
    echo "Python executable: $(which python)"
    nvidia-smi  # État initial du GPU
} > ${JOB_DIR}/job_info.log

module purge
module load miniforge/24.9.0

conda deactivate && while [ ! -z "$CONDA_DEFAULT_ENV" ]; do conda deactivate; done

module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0

mamba activate pointcept

conda list > ${JOB_DIR}/conda_env.txt

export WANDB_MODE=offline
cd ${REPO_ROOT}

START_TIME=$(date +%s)


EXTRA_OPTIONS="save_path=exp/s3dis/ptv3_nonormal eval_epoch=2 epoch=10 data.train.max_sample=100 data.val.max_sample=100 data.test.max_sample=10"
sh scripts/train.sh -g 1 -d s3dis -c ptv3_nonormal -n ptv3_nonormal

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    nvidia-smi  # État final du GPU
} >> ${JOB_DIR}/job_info.log
