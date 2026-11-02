#!/bin/bash
# Submit one run of the coord-fix retrain campaign on Jean Zay, plus CHAIN follow-up jobs that
# resume it (afterany dependency) if it hits the walltime. Run from the repo root on Jean Zay, in a
# login shell (bash -lc). Runs live in runs/<name>/ (stable across the chained jobs).
#
#   scripts/jz/submit.sh NAME MODE CONFIG [options] [-- extra --options for train.py]
#     --weight W     checkpoint (train: initial weight; test/sweep/pureforest: backbone to probe)
#     --gpus N       GPUs on one node (default 1)
#     --account A    default uhn@a100        --constraint C   default a100 (h100 for unv@h100)
#     --time T       default 19:50:00        --chain K        follow-up resume jobs (default 1)
#     --cpus C       cpus per task (default 8 per GPU)
#     --tags T       compute-accounting tags (default baseline,train)
set -eo pipefail
NAME=$1; MODE=$2; CONFIG=$3; shift 3
WEIGHT=-; NGPU=1; ACCOUNT=uhn@a100; CONSTRAINT=a100; TIME=19:50:00; CHAIN=1; CPUS=; TAGS=baseline,train
while [ $# -gt 0 ]; do
    case $1 in
    --weight) WEIGHT=$2; shift 2 ;;
    --gpus) NGPU=$2; shift 2 ;;
    --account) ACCOUNT=$2; shift 2 ;;
    --constraint) CONSTRAINT=$2; shift 2 ;;
    --time) TIME=$2; shift 2 ;;
    --chain) CHAIN=$2; shift 2 ;;
    --cpus) CPUS=$2; shift 2 ;;
    --tags) TAGS=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option $1"; exit 2 ;;
    esac
done
# A100 nodes: 8 GPUs / 64 cores; H100 nodes: 4 GPUs / 96 cores.
[ "$CONSTRAINT" = h100 ] && CPUS=${CPUS:-$((24 * NGPU))}
CPUS=${CPUS:-$((8 * NGPU))}
REPO=$(pwd)
RUN_DIR=$REPO/runs/$NAME
mkdir -p "$RUN_DIR/jobs"
SB=(-A "$ACCOUNT" -C "$CONSTRAINT" --nodes=1 --ntasks=1 --gres=gpu:$NGPU --cpus-per-task=$CPUS
    --time=$TIME --job-name="rc-$NAME" --comment="malibu3d_retrain,$TAGS"
    --output="$RUN_DIR/jobs/%j.out" --error="$RUN_DIR/jobs/%j.err")
JID=$(sbatch --parsable "${SB[@]}" scripts/jz/job.slurm "$MODE" "$RUN_DIR" "$CONFIG" "$WEIGHT" "$NGPU" "$@")
IDS=$JID
for _ in $(seq 1 "$CHAIN"); do
    JID=$(sbatch --parsable "${SB[@]}" --dependency=afterany:$JID scripts/jz/job.slurm "$MODE" "$RUN_DIR" "$CONFIG" "$WEIGHT" "$NGPU" "$@")
    IDS="$IDS $JID"
done
echo "$NAME $MODE jobs: $IDS -> $RUN_DIR"
