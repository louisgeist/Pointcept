#!/usr/bin/env bash
# Run SpUNet toy elevation hparam ablation (configs 2..21, sequential).
#
# Each run writes to exp/flair3d/<exp_name>/:
#   model/model_best.pth, model/model_last.pth
#   result/<tile>_reg_elevation.npy  (after PreciseEvaluator test)
if [ -z "${BASH_VERSION:-}" ]; then
  exec /bin/bash "$0" "$@"
fi
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${ROOT_DIR}"

NUM_GPU="${1:-1}"

get_slug() {
  case "$1" in
    1)  echo "overfit_minimal" ;;
    2)  echo "warmup_50" ;;
    3)  echo "aug_off" ;;
    4)  echo "no_z_random_offset" ;;
    5)  echo "lr_5e-3" ;;
    6)  echo "lr_1e-2" ;;
    7)  echo "weight_decay_0" ;;
    8)  echo "amp_off" ;;
    9)  echo "target_scale_0.1" ;;
    10) echo "target_scale_1.0" ;;
    11) echo "feat_no_coord" ;;
    12) echo "coord_scale_0.1" ;;
    13) echo "coord_scale_1.0" ;;
    14) echo "masked_feat_off" ;;
    15) echo "smoothl1_beta_0.1" ;;
    16) echo "smoothl1_beta_1.0" ;;
    17) echo "long_run" ;;
    18) echo "lr_5e-4" ;;
    19) echo "lr_1e-4" ;;
    20) echo "loss_l1" ;;
    21) echo "loss_mse" ;;
    *)  echo "unknown" ;;
  esac
}

run_one() {
  local num_exp="$1"
  local slug
  slug="$(get_slug "${num_exp}")"
  local config="experiment/w103/7/elevation_hparam/spunet_toy_${num_exp}"
  local exp_name
  printf -v exp_name "spunet_toy_hparam_%02d_%s" "${num_exp}" "${slug}"
  echo "========== Training: ${config} (exp: ${exp_name}) =========="
  echo "Save path: exp/flair3d/${exp_name}/"
  sh scripts/train.sh -g "${NUM_GPU}" -d flair3d -c "${config}" -n "${exp_name}"
}

for num_exp in $(seq 2 21); do
  run_one "${num_exp}"
done

echo "========== All runs finished =========="
echo "Weights:  exp/flair3d/spunet_toy_hparam_<NN>_<slug>/model/model_best.pth"
echo "Preds:    exp/flair3d/spunet_toy_hparam_<NN>_<slug>/result/*_reg_elevation.npy"
