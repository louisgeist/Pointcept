#!/bin/bash

# Short Sonata linear probe on Flair3D+ segment (1× H100).
# Usage:
#   sbatch scripts/sonata/sbatch_lin_probe_h100.sh <weight.pth> [exp_name]
# Or with env vars (LinProbeSbatchHook / watcher):
#   WEIGHT=... EXP_NAME=... PRETRAIN_JOB_DIR=... PRETRAIN_EPOCH=... PRETRAIN_ITERS=... \
#     sbatch scripts/sonata/sbatch_lin_probe_h100.sh
#
# Jean-Zay compute-accounting tags (IMAGINE wrapper):
#   https://github.com/Archiel19/compute-accounting
# --comment is required so LinProbeSbatchHook auto-submits never hang on interactive prompts.

#SBATCH -A uhn@h100
#SBATCH -C h100
#SBATCH --comment=flair3d,explore,evaluate
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err

#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# H100 Jean-Zay: 24 CPU/GPU (gpu_p6, 96 CPUs / 4 GPUs).
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

#SBATCH --job-name=sonata_lin_h100

WEIGHT="${WEIGHT:-${1:?WEIGHT path required (arg1 or WEIGHT=)}}"
EXP_NAME="${EXP_NAME:-${2:-sonata_lin_probe}}"
PRETRAIN_JOB_DIR="${PRETRAIN_JOB_DIR:-}"
PRETRAIN_EPOCH="${PRETRAIN_EPOCH:-0}"
PRETRAIN_ITERS="${PRETRAIN_ITERS:-0}"

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Exp name: $EXP_NAME"
    echo "Weight: $WEIGHT"
    echo "Pretrain job dir: ${PRETRAIN_JOB_DIR:-<none>}"
    echo "Pretrain epoch/iters: ${PRETRAIN_EPOCH}/${PRETRAIN_ITERS}"
    echo "Config: flair3d_default/segment/sonata-v1m2-flair3d-lin"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    nvidia-smi
} > ${JOB_DIR}/job_info.log

module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0

conda deactivate && while [ ! -z "$CONDA_DEFAULT_ENV" ]; do conda deactivate; done
conda activate pointcept_124

conda list > ${JOB_DIR}/conda_env.txt

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ${REPO_ROOT}

# Pointops built for H100 (train.sh prepends CODE_DIR but does not override an existing PYTHONPATH)
POINTOPS_PATH=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_h100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
export PYTHONPATH="${POINTOPS_PATH}${PYTHONPATH:+:$PYTHONPATH}"

START_TIME=$(date +%s)

export JOB_DIR
TRAIN_RC=0
sh scripts/train.sh \
  -g 1 \
  -d flair3d_default \
  -c segment/sonata-v1m2-flair3d-lin \
  -n "$EXP_NAME" \
  -w "$WEIGHT" || TRAIN_RC=$?

echo "Exp dir: ${JOB_DIR}" >> "${JOB_DIR}/job_info.log"
echo "WEIGHT=${WEIGHT}" >> "${JOB_DIR}/job_info.log"
echo "TRAIN_RC=${TRAIN_RC}" >> "${JOB_DIR}/job_info.log"

# Append mIoU to the pretrain CSV when launched from LinProbeSbatchHook / watcher.
if [ -n "${PRETRAIN_JOB_DIR}" ]; then
  python scripts/sonata/append_lin_probe_result.py \
    --pretrain_job_dir "${PRETRAIN_JOB_DIR}" \
    --probe_job_dir "${JOB_DIR}" \
    --ckpt "${WEIGHT}" \
    --pretrain_epoch "${PRETRAIN_EPOCH}" \
    --pretrain_iters "${PRETRAIN_ITERS}" \
    --train_exit_code "${TRAIN_RC}" \
    >> "${JOB_DIR}/job_info.log" 2>&1 || true
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log

exit ${TRAIN_RC}
