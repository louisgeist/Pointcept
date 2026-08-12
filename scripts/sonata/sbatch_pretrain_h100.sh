#!/bin/bash

# Sonata-v1m2 pretrain on Flair3D+ (default: 6 nodes × 4 H100 = 24 GPUs).
# Usage: sbatch scripts/sonata/sbatch_pretrain_h100.sh [exp_name]
#
# Multi-node: bump --nodes AND --ntasks together (1 Slurm task per node).
# --gres=gpu:N is per node; NUM_GPUS below must match N.
# H100 nodes have 4 GPUs / 96 CPUs (vs 8 / 64 on A100).
#
# Jean-Zay compute-accounting tags (IMAGINE wrapper):
#   https://github.com/Archiel19/compute-accounting
# Keep --comment set so non-interactive sbatch never prompts.

#SBATCH -A uhn@h100
#SBATCH -C h100
#SBATCH --comment=flair3d,baseline,pre-train
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err

#SBATCH --time=19:50:00
#SBATCH --signal=B:USR1@120
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread

#SBATCH --job-name=sonata_pretrain_h100

EXP_NAME="${1:-sonata_pretrain_flair3dplus_h100}"
NUM_GPUS=4  # must match --gres=gpu:N (per node); total GPUs = nodes × NUM_GPUS
# LinProbeSbatchHook (hooks[6]) keeps its default sbatch_script
# (scripts/sonata/sbatch_lin_probe.sh, A100): probe jobs are cheap/short, so
# they run on A100 even when pretraining itself is on H100, to save H100 hours.

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
# Pass an existing job dir as $2 to resume it (train.sh auto-resumes when
# JOB_DIR/model/model_last.pth exists), e.g.:
#   sbatch scripts/sonata/sbatch_pretrain_h100.sh sonata_pretrain_flair3dplus_h100 logs/slurm/<OLD_JOB_ID>
# (not `sbatch --export=...`: the Jean-Zay sbatch wrapper doesn't forward that flag.)
JOB_DIR=${2:-${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Exp name: $EXP_NAME"
    echo "Config: flair3d_default/pretrain-sonata-v1m2-flair3d"
    echo "Nodes: ${SLURM_NNODES} × ${NUM_GPUS} H100"
    echo "EXTRA_OPTIONS: ${EXTRA_OPTIONS}"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    echo "Working directory: $(pwd)"
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
export POINTCEPT_SLURM_REQUEUE=1

# Auto-requeue near walltime end (Slurm sends SIGUSR1 120s before kill via --signal=B:USR1@120 above):
# - slurm_requeue_trap.sh: batch-script-level trap, requeues immediately on SIGUSR1.
# - slurm_requeue_watchdog.sh: background poll of squeue time-left, in case the signal is missed.
# - tools/train.py additionally installs an in-process poll/signal handler (pointcept/utils/slurm_requeue.py),
#   gated by POINTCEPT_SLURM_REQUEUE=1 above, which flushes wandb/runtime state before requeuing.
. scripts/slurm_requeue_trap.sh
sh scripts/slurm_requeue_watchdog.sh &

# Per-node nvidia-smi snapshot before training starts: the job_info.log nvidia-smi above only runs
# on node 0 (batch script), so it can't show a bad/busy GPU on another node. One file per node.
mkdir -p "${JOB_DIR}/nvidia-smi_pre"
srun --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  bash -c "nvidia-smi > \"${JOB_DIR}/nvidia-smi_pre/\$(hostname).log\" 2>&1"

# Batch script runs on node 0 only; srun starts one task per node so every node runs train.sh.
# train.sh uses SLURM_NODEID for --machine-rank and SLURM_NODELIST for the dist URL.
#
# Wrapped in a retry loop: if training crashes (e.g. a bad sample poisons one rank and
# srun cancels the whole step), retry in place within THIS SAME allocation instead of
# resubmitting a new sbatch job (which would queue for a fresh allocation). JOB_DIR is
# unchanged across retries, so scripts/train.sh auto-resumes from model_last.pth if a
# checkpoint was already saved. Bounded by should_retry_after_crash (see
# scripts/slurm_crash_retry.sh) so a systematic crash doesn't spin forever.
. scripts/slurm_crash_retry.sh
while true; do
  ATTEMPT_START=$(date +%s)
  srun --unbuffered \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    bash -c "
cd ${REPO_ROOT} || exit 1
export JOB_DIR=\"${JOB_DIR}\"
export PYTHONPATH=\"${PYTHONPATH}\"
export WANDB_MODE=\"${WANDB_MODE}\"
export PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF}\"
export POINTCEPT_SLURM_REQUEUE=\"${POINTCEPT_SLURM_REQUEUE}\"
export EXTRA_OPTIONS=\"${EXTRA_OPTIONS}\"
sh scripts/train.sh \
  -g ${NUM_GPUS} \
  -m ${SLURM_NNODES} \
  -d flair3d_default \
  -c pretrain-sonata-v1m2-flair3d \
  -n \"${EXP_NAME}\"
"
  TRAIN_EXIT_CODE=$?
  ATTEMPT_DURATION=$(( $(date +%s) - ATTEMPT_START ))
  [ "$TRAIN_EXIT_CODE" -eq 0 ] && break
  should_retry_after_crash "$ATTEMPT_DURATION" "${JOB_DIR}" || break
done

echo "Exp dir: ${JOB_DIR}" >> "${JOB_DIR}/job_info.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log
