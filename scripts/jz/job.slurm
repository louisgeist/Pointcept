#!/bin/bash
# Generic single-node Jean Zay job (coord-fix retrain campaign, Volt...). It runs the code of the
# repo it was submitted from (SLURM_SUBMIT_DIR, kept across requeues).
# Submitted by scripts/jz/submit.sh, which sets the #SBATCH resources on the command line and
# passes: MODE RUN_DIR CONFIG WEIGHT NGPU [extra train.py --options ...].
#   MODE=train       tools/train.py; resumes from RUN_DIR/model/model_last.pth when present
#   MODE=test        tools/test.py on WEIGHT (RUN_DIR = output dir)
#   MODE=sweep       tools/sweep_then_seeds.py (grid -> 10 seeds); resumes on its own
#   MODE=pureforest  tools/probe_pureforest.py (pooled features + L-BFGS)
# Walltime: the job requeues itself 120 s before the end (same job id, the next segment resumes),
# as on flair3d: batch trap on SIGUSR1 + squeue watchdog + in-process handler of tools/train.py
# (POINTCEPT_SLURM_REQUEUE=1). A crash is retried in place (scripts/jz/slurm_crash_retry.sh).
# On success RUN_DIR/DONE is written, so a later requeue or resubmission exits at once.
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --open-mode=append
set -o pipefail
MODE=$1; RUN_DIR=$2; CONFIG=$3; WEIGHT=$4; NGPU=$5; shift 5
EXTRA=("$@")
case $MODE in train|test|sweep|pureforest) ;; *) echo "unknown mode $MODE"; exit 2 ;; esac

REPO=$SLURM_SUBMIT_DIR
if [ ! -f "$REPO/tools/train.py" ]; then
    echo "SLURM_SUBMIT_DIR=$REPO is not a Pointcept repo root (submit with scripts/jz/submit.sh)"
    exit 2
fi
if [ -f "$RUN_DIR/DONE" ]; then
    echo "[$(date)] $RUN_DIR/DONE exists: nothing to do."
    exit 0
fi
mkdir -p "$RUN_DIR/jobs"
cp "$0" "$RUN_DIR/jobs/${SLURM_JOB_ID}.slurm"
# Markers of the previous segment (same job id after a requeue) must not block crash retries.
rm -f "$RUN_DIR/.requeue_triggered" "$RUN_DIR/.crash_retry_count"
{
    echo "job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0}) on $(hostname) at $(date)"
    echo "mode=$MODE config=$CONFIG weight=$WEIGHT ngpu=$NGPU extra=${EXTRA[*]}"
    git -C $REPO log --oneline -1
} >> "$RUN_DIR/jobs/history.log"

module purge
module load arch/a100 cuda/12.4.1 miniforge/24.9.0
conda activate pointcept_124
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$REPO:/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_a100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
cd $REPO

export JOB_DIR=$RUN_DIR
export POINTCEPT_SLURM_REQUEUE=1
. scripts/jz/slurm_requeue_trap.sh
sh scripts/jz/slurm_requeue_watchdog.sh &
. scripts/jz/slurm_crash_retry.sh

run_mode() {
    case $MODE in
    train)
        OPTS=(save_path=$RUN_DIR)
        if [ -f "$RUN_DIR/model/model_last.pth" ]; then
            OPTS+=(resume=True weight=$RUN_DIR/model/model_last.pth)
        elif [ "$WEIGHT" != "-" ]; then
            OPTS+=(weight=$WEIGHT)
        fi
        python tools/train.py --config-file "$CONFIG" --num-gpus "$NGPU" --options "${OPTS[@]}" "${EXTRA[@]}"
        ;;
    test)
        python tools/test.py --config-file "$CONFIG" --num-gpus "$NGPU" --options save_path=$RUN_DIR weight=$WEIGHT "${EXTRA[@]}"
        ;;
    sweep)
        ARGS=(--grid-config "$CONFIG" --save-root "$RUN_DIR" --num-gpus "$NGPU")
        [ "$WEIGHT" != "-" ] && ARGS+=(--weight "$WEIGHT")
        [ ${#EXTRA[@]} -gt 0 ] && ARGS+=(--extra-options "${EXTRA[*]}")
        python tools/sweep_then_seeds.py "${ARGS[@]}"
        ;;
    pureforest)
        python tools/probe_pureforest.py --config-file "$CONFIG" --work-dir "$RUN_DIR" --options weight=$WEIGHT "${EXTRA[@]}"
        ;;
    esac
}

while true; do
    ATTEMPT_START=$(date +%s)
    run_mode
    STATUS=$?
    echo "[$(date)] exit $STATUS" >> "$RUN_DIR/jobs/history.log"
    [ $STATUS -eq 0 ] && break
    should_retry_after_crash "$(( $(date +%s) - ATTEMPT_START ))" "$RUN_DIR" || break
done
[ $STATUS -eq 0 ] && touch "$RUN_DIR/DONE"
exit $STATUS
