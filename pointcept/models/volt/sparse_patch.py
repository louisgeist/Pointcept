"""Sparse (occupied-voxel-only) versions of Volt's Tokenizer / Detokenizer products.

Volt's Tokenizer scatters each voxel into a dense (tokens, K^3, C_in) patch tensor and
applies Linear(K^3 * C_in -> D); its Detokenizer computes Linear(D -> K^3 * C_out) for
every token, i.e. a dense (tokens, K^3, C_out) tensor, then reads back the occupied
slots. On aerial LiDAR the patches are nearly empty (~2-20 occupied slots out of K^3),
so the detokenizer's memory grows with K^3. Splitting each weight matrix into its K^3
per-slot blocks W[o] gives the same products over occupied voxels only:

    token_t  = b + sum_{v in t} W_tok[o_v] x_v          (W_tok[o]: C_in x D)
    output_v =     W_det[o_v] token_{t(v)}  (+ bias)     (W_det[o]: D x C_out)

computed as one grouped matmul per slot (pyg_lib segment_matmul) over the voxels sorted
by slot. Same parameters (state_dict unchanged), same math as the dense path up to
floating-point summation order; memory and FLOPs scale with the number of voxels.
The grouped matmul runs in float32 (also under autocast) and the result is cast back.
"""

import torch
from pyg_lib.ops import segment_matmul


def _linear_out_dtype(x):
    """dtype nn.Linear would return for input x (autocast-aware)."""
    if x.is_cuda and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return x.dtype


def _group_by_slot(offset_id, num_slots):
    order = torch.argsort(offset_id, stable=True)
    counts = torch.bincount(offset_id, minlength=num_slots)
    ptr = torch.zeros(num_slots + 1, dtype=torch.long, device=offset_id.device)
    ptr[1:] = torch.cumsum(counts, dim=0)
    return order, ptr


def sparse_tokenize(proj, features, inverse, offset_id, num_tokens, kernel_size):
    """== proj(dense_patches.flatten(1)) of Volt's Tokenizer, over occupied voxels."""
    k3 = kernel_size**3
    out_dim, in_flat = proj.weight.shape
    in_dim = in_flat // k3
    order, ptr = _group_by_slot(offset_id, k3)
    # weight[d, o * C_in + c] -> W[o] of shape (C_in, D)
    weight = proj.weight.float().view(out_dim, k3, in_dim).permute(1, 2, 0)
    contrib = segment_matmul(features[order].float(), ptr, weight.contiguous())
    tokens = torch.zeros(
        num_tokens, out_dim, device=features.device, dtype=contrib.dtype
    )
    tokens.index_add_(0, inverse[order], contrib)
    if proj.bias is not None:
        tokens = tokens + proj.bias.float()
    return tokens.to(_linear_out_dtype(features))


def sparse_detokenize(proj, coarse_features, inverse, offset_id, kernel_size, out_dim):
    """== proj(coarse).view(-1, K^3, C_out)[inverse, offset_id] of Volt's Detokenizer."""
    k3 = kernel_size**3
    in_dim = proj.weight.shape[1]
    order, ptr = _group_by_slot(offset_id, k3)
    # weight[o * C_out + c, d] -> W[o] of shape (D, C_out)
    weight = proj.weight.float().view(k3, out_dim, in_dim).transpose(1, 2)
    x = coarse_features[inverse[order]].float()
    y = segment_matmul(x, ptr, weight.contiguous())
    out = torch.empty_like(y)
    out[order] = y
    return out.to(coarse_features.dtype)
