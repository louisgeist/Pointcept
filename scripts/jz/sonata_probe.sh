#!/bin/bash
# Probe command of the Sonata pretraining (PeriodicProbeHook): submits the short MALiBU3D
# land-cover linear probe of one checkpoint as its own 1-GPU job.
#   probe_command="sh scripts/jz/sonata_probe.sh {weight} {probe_dir}"
WEIGHT=$1; PROBE_DIR=$2
mkdir -p "$PROBE_DIR"
sbatch -A uhn@a100 -C a100 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --time=03:00:00 \
    --job-name=rc-sonata-probe --comment=malibu3d_retrain,baseline,evaluate \
    --output="$PROBE_DIR/%j.out" --error="$PROBE_DIR/%j.err" \
    scripts/jz/job.slurm train "$PROBE_DIR" configs/malibu3d/probe/sonata-v1m2-malibu3d-lin.py "$WEIGHT" 1
