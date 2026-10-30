#!/bin/sh
# Test a trained run. Usage (from any directory):
#   sh scripts/test.sh -g <num_gpus> -d <dataset> -n <exp_name> [-c <config>] [-w <weight>]
# Reads exp/<dataset>/<exp_name>/model/<weight>.pth (default model_best). Without -c, the config
# saved by training (exp/<dataset>/<exp_name>/config.py) is used; -c takes the same values as in
# scripts/train.sh (a path under configs/ without ".py", or a config name under configs/<dataset>/).

cd $(dirname $(dirname "$0")) || exit
PYTHON=python

TEST_CODE=test.py

DATASET=malibu3d
CONFIG="None"
EXP_NAME=debug
WEIGHT=model_best
NUM_GPU=None
NUM_MACHINE=1
DIST_URL="auto"

while getopts "p:d:c:n:w:g:m:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
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
    g)
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

if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "GPU Num: $NUM_GPU"
echo "Machine Num: $NUM_MACHINE"

if [ -n "$SLURM_NODELIST" ]; then
  MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
  # Prefer IPv4 for torch.distributed rendezvous (see scripts/train.sh).
  MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOSTNAME" | awk 'NR==1 { print $1 }')
  if [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR=$(getent hosts "$MASTER_HOSTNAME" | head -n 1 | awk '{ print $1 }')
  fi
  MASTER_PORT=$((10000 + 0x$(echo -n "${DATASET}/${EXP_NAME}" | md5sum | cut -c 1-4 | awk '{print $1}') % 20000))
  case "$MASTER_ADDR" in
    *:*) DIST_URL="tcp://[${MASTER_ADDR}]:${MASTER_PORT}" ;;
    *)   DIST_URL="tcp://${MASTER_ADDR}:${MASTER_PORT}" ;;
  esac
fi

echo "Dist URL: $DIST_URL"

EXP_DIR=exp/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
if [ "${CONFIG}" = "None" ]; then
  CONFIG_DIR=${EXP_DIR}/config.py
elif [ -f "configs/${CONFIG}.py" ]; then
  CONFIG_DIR=configs/${CONFIG}.py
else
  CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./${PYTHONPATH:+:$PYTHONPATH}


echo " =========> RUN TASK <========="
ulimit -n 65536
$PYTHON -u tools/$TEST_CODE \
  --config-file "$CONFIG_DIR" \
  --num-gpus "$NUM_GPU" \
  --num-machines "$NUM_MACHINE" \
  --machine-rank ${SLURM_NODEID:-0} \
  --dist-url ${DIST_URL} \
  --options save_path="$EXP_DIR" weight="${MODEL_DIR}"/"${WEIGHT}".pth
