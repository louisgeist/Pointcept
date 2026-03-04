#!/bin/sh

# Copied from train.sh, adapted to work with repo `jz_utils` for cluster job submission.

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TRAIN_CODE=train.py

# In `train.sh`, the variable DATASET is used to determine:
# - the experiment directory - which is NOT used in `jz_utils`, as it is the log_dir
# - the config file - which is NOT used in `jz_utils`, to simplify the writing and submitting of jobs.
DATASET=scannet 
CONFIG="None"
EXP_NAME=debug
WEIGHT="None"
RESUME=false
NUM_GPU=None
NUM_MACHINE=1
DIST_URL="auto"


while getopts "p:d:c:n:w:g:m:r:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      echo "Note that with `train_jz_utils.sh`, giving a dataset argument with `-d` has no effect" >&2
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      echo "Note that with `train_jz_utils.sh`, giving a GPU number argument with `-g` has no effect. (`launch_experiments_v2` expects the number of GPUs to be set in the config file (.py file))" >&2
      NUM_GPU=$OPTARG
      ;;
    m)
      NUM_MACHINE=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

# if [ "${NUM_GPU}" = 'None' ]
# then
#   NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
# fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
# echo "Dataset: $DATASET"
echo "Config: $CONFIG"
# echo "GPU Num: $NUM_GPU"
echo "Machine Num: $NUM_MACHINE"

if [ -n "$SLURM_NODELIST" ]; then
  MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
  MASTER_ADDR=$(getent hosts "$MASTER_HOSTNAME" | awk '{ print $1 }')
  MASTER_PORT=$((10000 + 0x$(echo -n "${EXP_NAME}" | md5sum | cut -c 1-4 | awk '{print $1}') % 20000))
  DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT
fi

echo "Dist URL: $DIST_URL"

# train_jz_utils.sh expects JOB_DIR (set by sbatch wrapper) so exp lives under logs/slurm/%j/
if [ -z "${JOB_DIR}" ]; then
  echo "Error: JOB_DIR is not set. Run this script via your jz_utils sbatch wrapper, which exports JOB_DIR." >&2
  exit 1
fi
EXP_DIR=${JOB_DIR}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/experiment/${CONFIG}


echo " =========> CREATE EXP DIR <========="
case "$EXP_DIR" in /*) echo "Experiment dir: $EXP_DIR" ;; *) echo "Experiment dir: $ROOT_DIR/$EXP_DIR" ;; esac
if [ "${RESUME}" = true ] && [ -d "$EXP_DIR" ]
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  RESUME=false
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  cp -r scripts tools pointcept "$CODE_DIR"
fi

echo "Loading config in:" $CONFIG_DIR
# Use absolute path so Python finds pointcept when CODE_DIR is under logs/slurm/%j/
export PYTHONPATH=$(cd "$CODE_DIR" && pwd)
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="
ulimit -n 65536
# Extra options for Python (e.g. eval_epoch=1 epoch=34 for smoke test)
OPTS="save_path=$EXP_DIR"
[ -n "${EXTRA_OPTIONS-}" ] && OPTS="$OPTS $EXTRA_OPTIONS"

# Note that `--num-gpus` is not passed: it is expected to be already set in the config file (.py file) for proper slurm job submission.
if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-machines "$NUM_MACHINE" \
    --machine-rank ${SLURM_NODEID:-0} \
    --dist-url ${DIST_URL} \
    --options $OPTS
else
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-machines "$NUM_MACHINE" \
    --machine-rank ${SLURM_NODEID:-0} \
    --dist-url ${DIST_URL} \
    --options save_path="$EXP_DIR" resume="$RESUME" weight="$WEIGHT"
fi
