"""Volt options: sparse tokenizer / detokenizer == dense ones (outputs and gradients),
and grid_pool == Volt run on pre-pooled coarse voxels with outputs broadcast back."""

import copy
import unittest

import torch

from pointcept.models.volt.decoder import Detokenizer
from pointcept.models.volt.volt_base import Tokenizer


def _sparse_scene(n_tokens_side=6, K=5, n_vox=3000, n_batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    side = n_tokens_side * K
    idx = torch.cat(
        [
            torch.randint(0, n_batch, (n_vox, 1), generator=g),
            torch.randint(0, side, (n_vox, 3), generator=g),
        ],
        1,
    )
    idx = torch.unique(idx, dim=0)
    return idx.int()


class TestSparsePatches(unittest.TestCase):
    def _check_pair(self, K, C_in=7, D=48, C_out=16):
        torch.manual_seed(K)
        idx = _sparse_scene(K=K)
        x = torch.randn(idx.shape[0], C_in, requires_grad=True)
        tok_d = Tokenizer(C_in, D, K)
        tok_s = copy.deepcopy(tok_d)
        tok_s.sparse = True
        a = tok_d(x, idx)
        x2 = x.detach().clone().requires_grad_(True)
        b = tok_s(x2, idx)
        torch.testing.assert_close(b[0], a[0], rtol=1e-4, atol=1e-5)
        for i in (1, 2, 3):
            self.assertTrue(torch.equal(a[i], b[i]))

        det_d = Detokenizer(D, C_out, K)
        det_s = copy.deepcopy(det_d)
        det_s.sparse = True
        with torch.no_grad():
            det_d.bias.normal_()
            det_s.bias.copy_(det_d.bias)
        out_d = det_d(a[0], a[2], a[3])
        out_s = det_s(b[0], b[2], b[3])
        torch.testing.assert_close(out_s, out_d, rtol=1e-4, atol=1e-5)

        w = torch.randn_like(out_d)
        (out_d * w).sum().backward()
        (out_s * w).sum().backward()
        torch.testing.assert_close(x2.grad, x.grad, rtol=1e-4, atol=1e-5)
        for m_d, m_s in ((tok_d, tok_s), (det_d, det_s)):
            for (n, p_d), (_, p_s) in zip(m_d.named_parameters(), m_s.named_parameters()):
                torch.testing.assert_close(p_s.grad, p_d.grad, rtol=1e-4, atol=1e-4, msg=n)

    def test_k5(self):
        self._check_pair(5)

    def test_k15(self):
        self._check_pair(15)

    def test_state_dict_unchanged(self):
        self.assertEqual(
            Tokenizer(7, 48, 5, sparse=True).state_dict().keys(),
            Tokenizer(7, 48, 5).state_dict().keys(),
        )


def _cuda_flash():
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_cuda_flash(), "needs CUDA + flash_attn")
class TestGridPool(unittest.TestCase):
    def test_grid_pool_equals_prepooled(self):
        import torch_scatter

        from pointcept.models.volt.volt_base import Volt

        torch.manual_seed(0)
        kw = dict(in_channels=7, embed_dim=96, depth=2, num_heads=2, drop_path=0.0,
                  stride=5, kernel_size=5, up_mlp_dim=32, qk_norm=True)
        pooled = Volt(grid_pool=3, sparse_patches=True, **kw).cuda().eval()
        plain = copy.deepcopy(pooled)
        plain.grid_pool = 1
        idx = _sparse_scene(n_tokens_side=8, K=15, n_vox=20000).cuda()
        batch, grid = idx[:, 0].long(), idx[:, 1:]
        feat = torch.randn(idx.shape[0], 7, device="cuda")
        with torch.no_grad():
            out = pooled(dict(grid_coord=grid, feat=feat, batch=batch))
            key = torch.cat([batch[:, None], grid // 3], 1)
            key, inv = torch.unique(key, dim=0, sorted=True, return_inverse=True)
            ref = plain(
                dict(
                    grid_coord=key[:, 1:],
                    feat=torch_scatter.scatter_mean(feat, inv, dim=0),
                    batch=key[:, 0],
                )
            )[inv]
        self.assertEqual(out.shape, (idx.shape[0], 32))
        torch.testing.assert_close(out, ref)
        # Voxels of one coarse voxel share their output.
        same = inv == inv[0]
        self.assertTrue(torch.equal(out[same], out[0].expand(int(same.sum()), -1)))


if __name__ == "__main__":
    unittest.main()
