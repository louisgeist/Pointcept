#!/bin/bash
# Submit a Volt training on Jean Zay with the self-requeuing launcher of scripts/jz (resumes
# from runs/<name>/model/model_last.pth, EMA included). Run from the Pointcept-volt repo root,
# in a login shell (bash -lc). Runs land in runs/<name>/ (runs -> $SCRATCH).
#
#   scripts/volt/submit_jz.sh NAME CONFIG [scripts/jz/submit.sh options] [-- train.py --options]
#   e.g. scripts/volt/submit_jz.sh pool3-k5 configs/malibu3d/ablations/multi-volt-s-v1m0-malibu3d-pool3-k5.py
#        scripts/volt/submit_jz.sh pool3-k5 <config> --test-only
set -eo pipefail
NAME=$1; CONFIG=$2; shift 2
[ -f "$CONFIG" ] || { echo "no config $CONFIG"; exit 2; }
exec scripts/jz/submit.sh "$NAME" train "$CONFIG" --campaign volt --prefix volt --tags volt,train "$@"
