#!/bin/bash
# Sonata pretraining of the coord-fix retrain campaign: 6 nodes x 4 H100 = 24 GPUs (as job 862680).
#   sbatch scripts/jz/sonata_pretrain.slurm runs/<name> [extra train.py --options ...]
# Another config (ablations): sbatch --export=ALL,CONFIG=<config> --job-name=rc-sonata-<x> ...
# Resumes from RUN_DIR/model/model_last.pth when present. Requeues itself 120 s before the
# walltime (same job id), as on flair3d: batch trap on SIGUSR1 + squeue watchdog + in-process
# handler of tools/train.py (POINTCEPT_SLURM_REQUEUE=1); a crash is retried in place
# (scripts/jz/slurm_crash_retry.sh). Exits at once when RUN_DIR/DONE exists. Every 10th epoch
# checkpoint is probed by scripts/jz/sonata_probe.sh (results in RUN_DIR/probes/summary.csv).
#SBATCH -A unv@h100
#SBATCH -C h100
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=19:50:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=rc-sonata-pretrain
#SBATCH --comment=malibu3d_retrain,baseline,pre-train
set -o pipefail
RUN_DIR=$1; shift
REPO=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept-malibu3d
CONFIG=${CONFIG:-configs/malibu3d/pretrain-sonata-v1m2-malibu3d.py}
if [ -f "$RUN_DIR/DONE" ]; then echo "DONE exists"; exit 0; fi
mkdir -p "$RUN_DIR/jobs"
cp "$0" "$RUN_DIR/jobs/${SLURM_JOB_ID}.slurm"
# Markers of the previous segment (same job id after a requeue) must not block crash retries.
rm -f "$RUN_DIR/.requeue_triggered" "$RUN_DIR/.crash_retry_count"
{ echo "job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0}) on $SLURM_JOB_NODELIST at $(date)"; git -C $REPO log --oneline -1; } >> "$RUN_DIR/jobs/history.log"

module purge
module load arch/h100 cuda/12.4.1 miniforge/24.9.0
conda activate pointcept_124
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$REPO:/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_h100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
cd $REPO

export JOB_DIR=$RUN_DIR
export POINTCEPT_SLURM_REQUEUE=1
. scripts/jz/slurm_requeue_trap.sh
sh scripts/jz/slurm_requeue_watchdog.sh &
. scripts/jz/slurm_crash_retry.sh

MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
while true; do
    ATTEMPT_START=$(date +%s)
    OPTS="save_path=$RUN_DIR probe_save_freq=10 $*"
    if [ -f "$RUN_DIR/model/model_last.pth" ]; then
        OPTS="$OPTS resume=True weight=$RUN_DIR/model/model_last.pth"
    fi
    # New port per attempt: the previous attempt's rendezvous may still hold the old one.
    PORT=$((20000 + (SLURM_JOB_ID + ATTEMPT_START) % 20000))
    srun --unbuffered --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
        --cpus-per-task=$SLURM_CPUS_PER_TASK bash -c "
cd $REPO
export JOB_DIR=$RUN_DIR POINTCEPT_SLURM_REQUEUE=1
python tools/train.py --config-file $CONFIG --num-gpus 4 --num-machines $SLURM_NNODES \
    --machine-rank \$SLURM_NODEID --dist-url tcp://$MASTER:$PORT \
    --options $OPTS probe_command='sh scripts/jz/sonata_probe.sh {weight} {probe_dir}'
"
    STATUS=$?
    echo "[$(date)] exit $STATUS" >> "$RUN_DIR/jobs/history.log"
    [ $STATUS -eq 0 ] && break
    should_retry_after_crash "$(( $(date +%s) - ATTEMPT_START ))" "$RUN_DIR" || break
done
[ $STATUS -eq 0 ] && touch "$RUN_DIR/DONE"
exit $STATUS
