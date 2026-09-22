"""Correctness tests for SpUNet-v1m1's point_mode (per-point encoder
multiscale output for GridProbeSegmentorV2), independent of _pooling_inverse
itself: rebuilds the fine->coarse voxel correspondence from scratch via a
plain Python coordinate dict and cross-checks the broadcast features.

Run with: PYTHONPATH=./ pytest tests/test_spunet_point_mode.py
"""

import unittest

import torch

try:
    import spconv.pytorch  # noqa: F401

    HAS_SPCONV = True
except ImportError:
    HAS_SPCONV = False

_CUDA_SPCONV = torch.cuda.is_available() and HAS_SPCONV


def _reference_pooling_inverse(fine_indices, coarse_indices, stride):
    """Ground-truth fine->coarse row map via a plain Python dict keyed on
    (batch, z//stride, y//stride, x//stride), independent of the
    torch.unique-based implementation under test."""
    coarse_row_of = {
        tuple(row.tolist()): i for i, row in enumerate(coarse_indices.cpu())
    }
    out = torch.empty(fine_indices.shape[0], dtype=torch.long)
    for i, row in enumerate(fine_indices.cpu()):
        b, z, y, x = row.tolist()
        key = (b, z // stride, y // stride, x // stride)
        assert key in coarse_row_of, f"fine voxel {key} has no coarse match"
        out[i] = coarse_row_of[key]
    return out


def _walk_like_grid_probe(point):
    """Reproduces grid_probe.py's `_forward_backbone` walk exactly (decoder
    side via unpooling_parent, then encoder side via pooling_parent), so
    tests exercise the same code path GridProbeSegmentorV2 actually uses."""
    point_list = [point]
    while "unpooling_parent" in point_list[-1].keys():
        point_list.append(point_list[-1].pop("unpooling_parent"))
    for i in reversed(range(1, len(point_list))):
        child, parent = point_list[i], point_list[i - 1]
        assert "pooling_inverse" in child.keys()
        parent.feat = torch.cat([parent.feat, child.feat[child.pooling_inverse]], dim=-1)
    point = point_list[0]
    while "pooling_parent" in point.keys():
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point.feat


@unittest.skipUnless(_CUDA_SPCONV, "CUDA + spconv required")
class TestSpUNetPointMode(unittest.TestCase):
    def _make_model_and_input(self, stride=3, point_mode=True, dec_point_mode=False):
        from pointcept.models.sparse_unet.spconv_unet_v1m1_base import SpUNetBase

        model = SpUNetBase(
            in_channels=7,
            num_classes=0,
            channels=(32, 64, 128, 256, 256, 128, 96, 96),
            layers=(2, 3, 4, 6, 2, 2, 2, 2),
            stride=stride,
            point_mode=point_mode,
            dec_point_mode=dec_point_mode,
        ).cuda()
        model.eval()

        torch.manual_seed(0)
        zs, ys, xs = torch.meshgrid(
            torch.arange(0, 9 * stride, stride, device="cuda"),
            torch.arange(0, 9 * stride, stride, device="cuda"),
            torch.arange(0, 9 * stride, stride, device="cuda"),
            indexing="ij",
        )
        grid_coord = torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)
        n = grid_coord.shape[0]
        feat = torch.randn(n, 7, device="cuda")
        offset = torch.tensor([n], device="cuda")
        return model, dict(grid_coord=grid_coord, feat=feat, offset=offset), n

    def test_pooling_inverse_matches_independent_reference(self):
        from pointcept.models.sparse_unet.spconv_unet_v1m1_base import (
            _pooling_inverse,
        )

        model, input_dict, n = self._make_model_and_input(stride=3)
        with torch.no_grad():
            grid_coord = input_dict["grid_coord"]
            feat = input_dict["feat"]
            offset = input_dict["offset"]
            from pointcept.models.utils import offset2batch
            import spconv.pytorch as spconv

            batch = offset2batch(offset)
            sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
            x = spconv.SparseConvTensor(
                features=feat,
                indices=torch.cat(
                    [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
                ).contiguous(),
                spatial_shape=sparse_shape,
                batch_size=batch[-1].tolist() + 1,
            )
            x = model.conv_input(x)
            skips = [x]
            for s in range(model.num_stages):
                x = model.down[s](x)
                x = model.enc[s](x)
                skips.append(x)

        for s in range(1, len(skips)):
            got = _pooling_inverse(
                skips[s - 1].indices, skips[s].indices, model.stride[s - 1]
            )
            want = _reference_pooling_inverse(
                skips[s - 1].indices, skips[s].indices, model.stride[s - 1]
            ).to(got.device)
            self.assertTrue(
                torch.equal(got, want),
                f"stage {s}: pooling_inverse mismatch vs independent reference",
            )
            # Every fine voxel's broadcast feature must equal its actual
            # coarse voxel's feature (not just a same-shaped row).
            self.assertTrue(
                torch.equal(
                    skips[s].features[got], skips[s].features[want]
                )
            )

    def test_forward_returns_point_chain_with_expected_shape(self):
        from pointcept.models.utils.structure import Point

        model, input_dict, n = self._make_model_and_input(
            stride=3, point_mode=True, dec_point_mode=False
        )
        with torch.no_grad():
            point = model(input_dict)
        self.assertIsInstance(point, Point)

        feat = _walk_like_grid_probe(point)
        # stem(32) + stage0(32) + stage1(64) + stage2(128) + stage3/bottleneck(256)
        expected_channels = 32 + 32 + 64 + 128 + 256
        self.assertEqual(feat.shape, (n, expected_channels))

    def test_dec_point_mode_unpooling_inverse_matches_reference(self):
        from pointcept.models.sparse_unet.spconv_unet_v1m1_base import (
            _pooling_inverse,
        )

        model, input_dict, n = self._make_model_and_input(
            stride=3, point_mode=False, dec_point_mode=True
        )
        with torch.no_grad():
            grid_coord = input_dict["grid_coord"]
            feat = input_dict["feat"]
            offset = input_dict["offset"]
            from pointcept.models.utils import offset2batch
            import spconv.pytorch as spconv

            batch = offset2batch(offset)
            sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
            x = spconv.SparseConvTensor(
                features=feat,
                indices=torch.cat(
                    [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
                ).contiguous(),
                spatial_shape=sparse_shape,
                batch_size=batch[-1].tolist() + 1,
            )
            x = model.conv_input(x)
            skips = [x]
            for s in range(model.num_stages):
                x = model.down[s](x)
                x = model.enc[s](x)
                skips.append(x)

            # Replay the decoder loop manually to get each stage's raw
            # SparseConvTensor (dec_point_mode consumes `skips` internally).
            dec_tensors = [skips[-1]]  # bottleneck
            x = skips[-1]
            skips_copy = list(skips[:-1])
            for s in reversed(range(model.num_stages)):
                x = model.up[s](x)
                skip = skips_copy.pop(-1)
                x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
                x = model.dec[s](x)
                dec_tensors.append(x)
            # dec_tensors: [bottleneck, dec(num_stages-1), ..., dec(0)/finest]

        for k in range(len(dec_tensors) - 1, 0, -1):
            finer, coarser = dec_tensors[k], dec_tensors[k - 1]
            stride = model.stride[model.num_stages - k]
            got = _pooling_inverse(finer.indices, coarser.indices, stride)
            want = _reference_pooling_inverse(finer.indices, coarser.indices, stride).to(
                got.device
            )
            self.assertTrue(
                torch.equal(got, want),
                f"decoder level {k}: pooling_inverse mismatch vs independent reference",
            )

    def test_dec_point_mode_forward_shape(self):
        from pointcept.models.utils.structure import Point

        model, input_dict, n = self._make_model_and_input(
            stride=3, point_mode=False, dec_point_mode=True
        )
        with torch.no_grad():
            point = model(input_dict)
        self.assertIsInstance(point, Point)

        feat = _walk_like_grid_probe(point)
        # dec0(96) + dec1(96) + dec2(128) + dec3(256) + bottleneck(256)
        expected_channels = 96 + 96 + 128 + 256 + 256
        self.assertEqual(feat.shape, (n, expected_channels))

    def test_combined_point_mode_and_dec_point_mode_shape_and_alignment(self):
        from pointcept.models.utils.structure import Point

        model, input_dict, n = self._make_model_and_input(
            stride=3, point_mode=True, dec_point_mode=True
        )
        with torch.no_grad():
            point = model(input_dict)
        self.assertIsInstance(point, Point)

        feat = _walk_like_grid_probe(point)
        # dec0(96)+dec1(96)+dec2(128)+dec3(256)+bottleneck(256) [832]
        # + stem(32)+stage0(32)+stage1(64)+stage2(128) [256, bottleneck dropped]
        expected_channels = (96 + 96 + 128 + 256 + 256) + (32 + 32 + 64 + 128)
        self.assertEqual(expected_channels, 1088)
        self.assertEqual(feat.shape, (n, expected_channels))

    def test_dec_stage0_output_row_order_matches_stem(self):
        """The concat in forward() assumes dec-stage-0's output rows already
        align with skips[0] (stem) row-for-row — same invariant the plain
        (non-point-mode) decoder relies on for its own skip-connection concat.
        Verify it explicitly via an independent coordinate-based check rather
        than trusting it."""
        model, input_dict, n = self._make_model_and_input(
            stride=3, point_mode=False, dec_point_mode=False
        )
        with torch.no_grad():
            grid_coord = input_dict["grid_coord"]
            feat = input_dict["feat"]
            offset = input_dict["offset"]
            from pointcept.models.utils import offset2batch
            import spconv.pytorch as spconv

            batch = offset2batch(offset)
            sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
            x = spconv.SparseConvTensor(
                features=feat,
                indices=torch.cat(
                    [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
                ).contiguous(),
                spatial_shape=sparse_shape,
                batch_size=batch[-1].tolist() + 1,
            )
            x = model.conv_input(x)
            stem = x
            skips = [x]
            for s in range(model.num_stages):
                x = model.down[s](x)
                x = model.enc[s](x)
                skips.append(x)
            x = skips.pop(-1)
            for s in reversed(range(model.num_stages)):
                x = model.up[s](x)
                skip = skips.pop(-1)
                x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
                x = model.dec[s](x)
            dec0_out = x

        self.assertTrue(torch.equal(stem.indices, dec0_out.indices))


if __name__ == "__main__":
    unittest.main()
