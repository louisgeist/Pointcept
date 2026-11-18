#!/bin/bash
#SBATCH -A uhn@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --comment=malibu3d_retrain,baseline,evaluate
# Stand-alone re-test of a finished OpenGF seed ensemble on regions T3, T2, T1 (new sweeps do it themselves
# at the end of `job.slurm sweep`; use this for old reference checkpoints). Run from the repo root:
#   sbatch --job-name=rc-ogf-regions-<tag> --output=runs/opengf-regions/<tag>_%j.out \
#       scripts/jz/opengf_regions.slurm <seed_config.py> <seeds_model_dir> <out_dir>
CONFIG=$1; MODEL_DIR=$2; OUT_DIR=$3
cd $SLURM_SUBMIT_DIR
module purge; module load arch/a100 cuda/12.4.1 miniforge/24.9.0; conda activate pointcept_124
export PYTHONPATH=$PWD:/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_a100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
python scripts/jz/retest_opengf_regions.py "$CONFIG" "$MODEL_DIR" "$OUT_DIR" T3_ T2_ T1_
