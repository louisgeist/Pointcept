"""Smoke tests for SpUNet-v1m1 pooling stride (does not change stage count).

Run with: PYTHONPATH=./ pytest tests/test_spunet_stride.py
"""

import unittest

import torch

try:
    import spconv.pytorch  # noqa: F401

    HAS_SPCONV = True
except ImportError:
    HAS_SPCONV = False

_CUDA_SPCONV = torch.cuda.is_available() and HAS_SPCONV


@unittest.skipUnless(_CUDA_SPCONV, "CUDA + spconv required")
class TestSpUNetPoolingStride(unittest.TestCase):
    def test_stride3_broadcasts_to_four_poolings_and_forwards(self):
        from pointcept.models.sparse_unet.spconv_unet_v1m1_base import (
            SpUNetBase,
            _as_stage_strides,
        )

        self.assertEqual(_as_stage_strides(3, 4), (3, 3, 3, 3))
        with self.assertRaises(AssertionError):
            _as_stage_strides((3, 3, 3), 4)

        model = SpUNetBase(in_channels=7, num_classes=0, stride=3).cuda()
        self.assertEqual(len(model.down), 4)
        self.assertEqual(len(model.up), 4)
        self.assertEqual(model.stride, (3, 3, 3, 3))

        # Coords span > 3^4 so 4x stride-3 pooling still occupies several voxels.
        zs, ys, xs = torch.meshgrid(
            torch.arange(0, 162, 27, device="cuda"),
            torch.arange(0, 162, 27, device="cuda"),
            torch.arange(0, 162, 27, device="cuda"),
            indexing="ij",
        )
        grid_coord = torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)
        n = grid_coord.shape[0]
        feat = torch.randn(n, 7, device="cuda")
        offset = torch.tensor([n], device="cuda")
        model.eval()
        with torch.no_grad():
            out = model(dict(grid_coord=grid_coord, feat=feat, offset=offset))
        self.assertEqual(tuple(out.shape), (n, 96))
