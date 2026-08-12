#!/bin/bash

# Resume a Sonata-v1m2 pretrain run under a NEW config (e.g. a VRAM-safety
# variant) from an EXISTING checkpoint from a different (e.g. crashed) job,
# preserving optimizer/scheduler/epoch state (true resume, not just weight
# loading). H100 variant (6 nodes × 4 H100 = 24 GPUs).
#
# scripts/train.sh only auto-resumes when $JOB_DIR/model/model_last.pth
# already exists when it starts (see its JOB_DIR branch) — but a fresh sbatch
# submission gets a brand-new JOB_DIR (logs/slurm/<new_job_id>/), so the old
# job's checkpoint is never there on its own. This script pre-seeds the new
# JOB_DIR with:
#   - $JOB_DIR/model/model_last.pth  <- copied from RESUME_CKPT
#   - $JOB_DIR/config.py             <- the NEW config, resolved via
#     Config.fromfile()+dump() so it has no _base_ (matches what a normal
#     fresh run itself writes at pointcept/engines/defaults.py:185, since
#     train.sh's resume branch loads $JOB_DIR/config.py directly rather than
#     re-resolving configs/<...>.py)
#   - $JOB_DIR/code/                 <- snapshot of scripts/tools/pointcept
#     (normally made by train.sh's fresh-run branch, skipped on resume)
# so that train.sh's JOB_DIR-checkpoint-present branch fires and resumes with
# RESUME=true, WEIGHT=model_last.pth.
#
# Usage: sbatch scripts/sonata/sbatch_pretrain_resume_h100.sh <exp_name> <config_rel_no_ext> <resume_ckpt_abs_path>
# Example:
#   sbatch scripts/sonata/sbatch_pretrain_resume_h100.sh \
#     sonata_vram_bs1 \
#     experiment/w107/1/sonata_vram/bs1 \
#     /lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/546886/model/model_last.pth
#
# Multi-node: bump --nodes AND --ntasks together (1 Slurm task per node).
# --gres=gpu:N is per node; NUM_GPUS below must match N.
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

#SBATCH --job-name=sonata_pretrain_resume_h100

EXP_NAME="${1:?usage: sbatch sbatch_pretrain_resume_h100.sh <exp_name> <config_rel_no_ext> <resume_ckpt_abs_path>}"
CONFIG_REL="${2:?usage: sbatch sbatch_pretrain_resume_h100.sh <exp_name> <config_rel_no_ext> <resume_ckpt_abs_path>}"
RESUME_CKPT="${3:?usage: sbatch sbatch_pretrain_resume_h100.sh <exp_name> <config_rel_no_ext> <resume_ckpt_abs_path>}"
NUM_GPUS=4  # must match --gres=gpu:N (per node); total GPUs = nodes × NUM_GPUS
# LinProbeSbatchHook (hooks[6]) keeps its default sbatch_script
# (scripts/sonata/sbatch_lin_probe.sh, A100): probe jobs are cheap/short, so
# they run on A100 even when pretraining itself is on H100, to save H100 hours.

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Exp name: $EXP_NAME"
    echo "Config: ${CONFIG_REL}"
    echo "Resuming from: ${RESUME_CKPT}"
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

if [ ! -f "${RESUME_CKPT}" ]; then
  echo "ERROR: resume checkpoint not found: ${RESUME_CKPT}" >&2
  exit 1
fi
if [ ! -f "configs/${CONFIG_REL}.py" ]; then
  echo "ERROR: config not found: configs/${CONFIG_REL}.py" >&2
  exit 1
fi

# Pre-seed JOB_DIR so scripts/train.sh's JOB_DIR-checkpoint-present branch
# auto-resumes (RESUME=true, WEIGHT=model_last.pth) instead of starting fresh.
mkdir -p ${JOB_DIR}/model ${JOB_DIR}/code
cp -r scripts tools pointcept ${JOB_DIR}/code/
python -c "
from pointcept.utils.config import Config
cfg = Config.fromfile('configs/${CONFIG_REL}.py')
cfg.dump('${JOB_DIR}/config.py')
"
cp ${RESUME_CKPT} ${JOB_DIR}/model/model_last.pth

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
# -c ${CONFIG_REL} only matters if the JOB_DIR-resume pre-seed above didn't fire (see train.sh);
# with model_last.pth already present, train.sh overrides CONFIG_DIR to $JOB_DIR/config.py itself.
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
  -c ${CONFIG_REL} \
  -n \"${EXP_NAME}\"
"

echo "Exp dir: ${JOB_DIR}" >> "${JOB_DIR}/job_info.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log
