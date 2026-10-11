#!/bin/sh
# Train a config. Usage (from any directory):
#   sh scripts/train.sh -g <num_gpus> -d <dataset> -c <config> -n <exp_name> [-w <weight>] [-r true]
# -c is a path under configs/ without ".py" (e.g. malibu3d/multi-litept-b-v1m0-malibu3d) or a
# config name under configs/<dataset>/. The run goes to exp/<dataset>/<exp_name> (or to $JOB_DIR
# when set, e.g. by a Slurm batch script), with a snapshot of the code under <exp_dir>/code.
# -r true resumes from <exp_dir>/model/model_last.pth; under Slurm with $JOB_DIR, an existing
# model_last.pth is resumed automatically (requeued jobs). $EXTRA_OPTIONS is appended to --options.

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TRAIN_CODE=train.py

DATASET=malibu3d
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
echo "Config: $CONFIG"
echo "GPU Num: $NUM_GPU"
echo "Machine Num: $NUM_MACHINE"

if [ -n "$SLURM_NODELIST" ]; then
  MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
  # Prefer IPv4 for torch.distributed rendezvous on SLURM clusters.
  # getent hosts may return IPv6 first (often link-local, e.g. fe80::/10),
  # which can break c10d socket connection depending on node networking.
  MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOSTNAME" | awk 'NR==1 { print $1 }')
  if [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR=$(getent hosts "$MASTER_HOSTNAME" | head -n 1 | awk '{ print $1 }')
  fi
  MASTER_PORT=$((10000 + 0x$(echo -n "${DATASET}/${EXP_NAME}" | md5sum | cut -c 1-4 | awk '{print $1}') % 20000))
  # IPv6 addresses need brackets in URL: tcp://[fe80::1]:port
  case "$MASTER_ADDR" in
    *:*) DIST_URL="tcp://[${MASTER_ADDR}]:${MASTER_PORT}" ;;
    *)   DIST_URL="tcp://${MASTER_ADDR}:${MASTER_PORT}" ;;
  esac
fi

echo "Dist URL: $DIST_URL"

# Use JOB_DIR if set (e.g. by a Slurm batch script), else exp/<dataset>/<exp_name>
EXP_DIR=exp/${DATASET}/${EXP_NAME}
[ -n "${JOB_DIR}" ] && EXP_DIR=${JOB_DIR}  # override when running under Slurm with JOB_DIR exported
[ -n "${EXP_DIR}" ] && export POINTCEPT_SAVE_PATH="${EXP_DIR}"
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
if [ -f "configs/${CONFIG}.py" ]; then
  CONFIG_DIR=configs/${CONFIG}.py
elif [ -f "configs/${DATASET}/${CONFIG}.py" ]; then
  CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
else
  CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
fi


echo " =========> CREATE EXP DIR <========="
case "$EXP_DIR" in /*) echo "Experiment dir: $EXP_DIR" ;; *) echo "Experiment dir: $ROOT_DIR/$EXP_DIR" ;; esac
if [ -n "${JOB_DIR}" ] && [ -f "${MODEL_DIR}/model_last.pth" ]; then
  RESUME=true
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=${MODEL_DIR}/model_last.pth
elif [ "${RESUME}" = true ] && [ -d "$EXP_DIR" ]; then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  RESUME=false
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  if [ "${SLURM_NODEID:-0}" = "0" ]; then
    cp -r scripts tools pointcept "$CODE_DIR"
    touch "${CODE_DIR}/.code_ready"
  else
    # Every node under a multi-node `srun --ntasks-per-node=1` runs this script and shares
    # CODE_DIR on the cluster filesystem; only node 0 snapshots the code, others wait for it
    # instead of racing their own `cp -r` into the same destination (which can leave a
    # subpackage like pointcept/engines/ incompletely copied when this node's python starts).
    WAIT_ELAPSED=0
    WAIT_TIMEOUT=600
    while [ ! -f "${CODE_DIR}/.code_ready" ]; do
      sleep 2
      WAIT_ELAPSED=$((WAIT_ELAPSED + 2))
      if [ "$WAIT_ELAPSED" -ge "$WAIT_TIMEOUT" ]; then
        echo "Error: timed out waiting for node 0 to finish code snapshot in ${CODE_DIR}" >&2
        exit 1
      fi
    done
  fi
fi

echo "Loading config in:" $CONFIG_DIR
# Absolute path, so Python finds the code snapshot wherever CODE_DIR lives.
# An existing PYTHONPATH is kept after it, not overridden: compiled extensions such as
# `pointops` may be built per GPU architecture and exported by the launcher beforehand.
export PYTHONPATH="$(cd "$CODE_DIR" && pwd)${PYTHONPATH:+:$PYTHONPATH}"

echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="
ulimit -n 65536
# Extra options for Python (e.g. eval_epoch=1 epoch=34 for smoke test)
OPTS="save_path=$EXP_DIR"
[ -n "${EXTRA_OPTIONS-}" ] && OPTS="$OPTS $EXTRA_OPTIONS"

if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$NUM_GPU" \
    --num-machines "$NUM_MACHINE" \
    --machine-rank ${SLURM_NODEID:-0} \
    --dist-url ${DIST_URL} \
    --options $OPTS
else
    OPTS="save_path=$EXP_DIR resume=$RESUME weight=$WEIGHT"
    [ -n "${EXTRA_OPTIONS-}" ] && OPTS="$OPTS $EXTRA_OPTIONS"
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$NUM_GPU" \
    --num-machines "$NUM_MACHINE" \
    --machine-rank ${SLURM_NODEID:-0} \
    --dist-url ${DIST_URL} \
    --options $OPTS
fi
