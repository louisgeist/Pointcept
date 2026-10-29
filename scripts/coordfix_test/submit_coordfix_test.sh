#!/bin/bash
# Test-only re-evaluation of the 5 paper multitask checkpoints (LitePT-B, PTv3, SpUNet, KPConvX,
# Sonata-FT) on the coord-fixed Flair3D+ (data/flair3d_plus_v2), one Slurm job per model.
# Configs: configs/experiment/w118/4/coordfix_test/ (weight + v2 paths baked in, wandb disabled).
# Each job runs tools/test.py (MultiTaskTester + NetworkAPLSEvaluator) into logs/slurm/<jobid>/;
# metrics end up in logs/slurm/<jobid>/slurm.err -> collect with
# scripts/coordfix_test/collect_coordfix_test.py.
#
# Usage (from the repo root on Jean Zay, in a login shell so `module`/conda are inherited):
#   bash -lc 'cd $WORK/Pointcept && bash scripts/coordfix_test/submit_coordfix_test.sh [account] [model ...]'
# account defaults to uhn@a100; models default to all five.
set -euo pipefail

ACCOUNT=${1:-uhn@a100}
shift || true
REPO=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
CFG_DIR=configs/experiment/w118/4/coordfix_test

# model -> "config num_gpu walltime". Reference test+APLS wall times on 1 A100 (quantized data):
# LitePT-B 2.7 h + 10 min (873542), Sonata-FT 5.0 h + 37 min (2126589); v2 tiles hold ~10 % more
# voxels. KPConvX is ~4x slower than Sonata per point (bench 1659402) -> 4 GPUs.
declare -A SPEC=(
  [litept-b]="multi-litept-b-v1m0-flair3d-coordfix-test_1.py 1 08:00:00"
  [ptv3]="multi-ptv3-v1m0-flair3d-coordfix-test_2.py 1 10:00:00"
  [spunet]="multi-spunet-v1m0-flair3d-coordfix-test_3.py 1 08:00:00"
  [kpconvx]="multi-kpconvx-v1m0-flair3d-coordfix-test_4.py 4 12:00:00"
  [sonata-ft]="multi-sonata-ft-v1m0-flair3d-coordfix-test_5.py 1 12:00:00"
)
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(litept-b ptv3 spunet kpconvx sonata-ft)

GPU=${ACCOUNT#*@}
for m in "${MODELS[@]}"; do
  read -r CFG NGPU TIME <<< "${SPEC[$m]}"
  [ -f "${REPO}/${CFG_DIR}/${CFG}" ] || { echo "missing ${CFG_DIR}/${CFG}" >&2; exit 1; }
  sbatch <<EOF
#!/bin/bash
#SBATCH --output=${REPO}/logs/slurm/%j/slurm.out
#SBATCH --error=${REPO}/logs/slurm/%j/slurm.err
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${GPU}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:${NGPU}
#SBATCH --cpus-per-task=$((8 * NGPU))
#SBATCH --hint=nomultithread
#SBATCH --job-name=test-coordfix-${m}
#SBATCH --time=${TIME}
#SBATCH --comment=flair3d_multi,final,evaluate

JOB_DIR=${REPO}/logs/slurm/\${SLURM_JOB_ID}
mkdir -p \${JOB_DIR}
cp \$0 \${JOB_DIR}/script.slurm

module purge
module load miniforge/24.9.0
conda deactivate && while [ ! -z "\$CONDA_DEFAULT_ENV" ]; do conda deactivate; done
module purge
module load arch/${GPU}
module load cuda/12.4.1
module load miniforge/24.9.0
conda activate pointcept_124

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${REPO}/pointops_build_${GPU}/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg:${REPO}
cd ${REPO}
cp ${CFG_DIR}/${CFG} \${JOB_DIR}/config.py
git rev-parse HEAD > \${JOB_DIR}/git_commit.txt

python tools/test.py --config-file ${CFG_DIR}/${CFG} --num-gpus ${NGPU} \\
  --options save_path=\${JOB_DIR}
EOF
done
