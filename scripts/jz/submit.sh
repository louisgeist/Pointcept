#!/bin/bash
# Submit one run of the coord-fix retrain campaign on Jean Zay. The job requeues itself before the
# walltime and resumes (see scripts/jz/job.slurm). Run from the repo root on Jean Zay, in a login
# shell (bash -lc). Runs live in runs/<name>/ (stable across requeued segments).
#
#   scripts/jz/submit.sh NAME MODE CONFIG [options] [-- extra --options for train.py]
#     --weight W     checkpoint (train: initial weight; test/sweep/pureforest: backbone to probe)
#     --gpus N       GPUs on one node (default 1)
#     --account A    default uhn@a100        --constraint C   default a100 (h100 for unv@h100)
#     --time T       walltime of one segment (default 19:50:00)
#     --cpus C       cpus per task (default 8 per GPU)
#     --tags T       compute-accounting tags (default baseline,train)
#     --campaign C   first word of the Slurm comment (default malibu3d_retrain)
#     --prefix P     job-name prefix, the job is named P-NAME (default rc)
#     --test-only    validate with sbatch --test-only, submit nothing
# The job runs the code of the repo it is submitted from (Pointcept-malibu3d, Pointcept-volt...).
set -eo pipefail
NAME=$1; MODE=$2; CONFIG=$3; shift 3
WEIGHT=-; NGPU=1; ACCOUNT=uhn@a100; CONSTRAINT=a100; TIME=19:50:00; CPUS=; TAGS=baseline,train
CAMPAIGN=malibu3d_retrain; PREFIX=rc; TEST_ONLY=
while [ $# -gt 0 ]; do
    case $1 in
    --weight) WEIGHT=$2; shift 2 ;;
    --gpus) NGPU=$2; shift 2 ;;
    --account) ACCOUNT=$2; shift 2 ;;
    --constraint) CONSTRAINT=$2; shift 2 ;;
    --time) TIME=$2; shift 2 ;;
    --cpus) CPUS=$2; shift 2 ;;
    --tags) TAGS=$2; shift 2 ;;
    --campaign) CAMPAIGN=$2; shift 2 ;;
    --prefix) PREFIX=$2; shift 2 ;;
    --test-only) TEST_ONLY=--test-only; shift ;;
    --) shift; break ;;
    *) echo "unknown option $1"; exit 2 ;;
    esac
done
# A100 nodes: 8 GPUs / 64 cores; H100 nodes: 4 GPUs / 96 cores.
[ "$CONSTRAINT" = h100 ] && CPUS=${CPUS:-$((24 * NGPU))}
CPUS=${CPUS:-$((8 * NGPU))}
REPO=$(pwd)
[ -f "$REPO/tools/train.py" ] && [ -f "$REPO/scripts/jz/job.slurm" ] || { echo "run from the repo root"; exit 2; }
RUN_DIR=$REPO/runs/$NAME
mkdir -p "$RUN_DIR/jobs"
SB=(-A "$ACCOUNT" -C "$CONSTRAINT" --nodes=1 --ntasks=1 --gres=gpu:$NGPU --cpus-per-task=$CPUS
    --time=$TIME --job-name="$PREFIX-$NAME" --comment="$CAMPAIGN,$TAGS"
    --output="$RUN_DIR/jobs/%j.out" --error="$RUN_DIR/jobs/%j.err")
if [ -n "$TEST_ONLY" ]; then
    sbatch --test-only "${SB[@]}" scripts/jz/job.slurm "$MODE" "$RUN_DIR" "$CONFIG" "$WEIGHT" "$NGPU" "$@"
    exit
fi
JID=$(sbatch --parsable "${SB[@]}" scripts/jz/job.slurm "$MODE" "$RUN_DIR" "$CONFIG" "$WEIGHT" "$NGPU" "$@")
echo "$NAME $MODE job: $JID -> $RUN_DIR"
