# RandLA-Net port to PointCept

This branch adds a native RandLA-Net backbone and two canonical configs:

- `configs/h3d/semseg-randla-v1m0-h3d.py` — semantic segmentation on H3D
- `configs/pureforest/cls-randla-v1m0-pureforest.py` — classification on PureForest

## What is ported

- RandLA-Net backbone (`pointcept/models/randla_net/`)
  - Local Feature Aggregation + attentive pooling
  - Random decimation in the encoder
  - Nearest-neighbor upsampling in the decoder
- PointCept integration via `DefaultSegmentor` (H3D) and `DefaultClassifier` (PureForest)
- Model registration: `type="RandLA-Net"`

## What is NOT ported yet (vs Myria3D)

### Scheduler / optimizer recipe

Myria3D default for RandLA-Net:

- Optimizer: **Adam**, lr ≈ **0.00393**
- Scheduler: **ReduceLROnPlateau** (`mode=min`, `factor=0.5`, `patience=20`, `cooldown=5`)
- Monitoring: **val/loss_epoch**

Current PointCept configs use PointCept-native recipes instead:

- H3D: **AdamW** + **OneCycleLR**
- PureForest: **AdamW** + **LinearLR** warmup

`ReduceLROnPlateau` is not registered in `pointcept/utils/scheduler.py` today.

### Transforms / data pipeline

Myria3D-specific preprocessing is not reproduced yet, including:

- PyG `GridSampling(voxel=0.25)` semantics
- min/max point budget (`MinimumNumNodes(300)`, `MaximumNumNodes(40000)`)
- Myria3D normalizations (`NullifyLowestZ`, `NormalizePos(subtile_width)`, `StandardizeRGBAndIntensity`)
- train/eval split with full-cloud KNN interpolation (`CopyFullPos`, `knn_interpolate`, etc.)

Current configs use standard PointCept transforms:

- `GridSample` + `SphereCrop`
- `CenterShift`, `NormalizeColor`
- light geometric augmentations (`RandomFlip`, `RandomRotate`)
- standard PointCept eval (`SemSegEvaluator`, fragment + `inverse`), not Myria3D full-cloud interpolation

## How to run

```bash
# H3D semantic segmentation
python tools/train.py --config-file configs/h3d/semseg-randla-v1m0-h3d.py --num-gpus 1

# PureForest classification
python tools/train.py --config-file configs/pureforest/cls-randla-v1m0-pureforest.py --num-gpus 1
```

## Dependencies

- Recommended: `pointops` (fast KNN)
- Fallback: pure PyTorch KNN (slower, for smoke tests)
- Standard PointCept deps still required (`torch_scatter`, etc.)

## Next steps (if aiming for Myria3D parity)

1. Add `ReduceLROnPlateau` scheduler support in PointCept
2. Port Myria3D transform chain (or explicit PointCept equivalents)
3. Optional phase 2: full-cloud KNN evaluation/interpolation pipeline
