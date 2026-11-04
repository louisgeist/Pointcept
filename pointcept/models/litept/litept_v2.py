"""
LitePT-v2: LitePT with the Sonata changes.

LitePT-v2 follows the same changes as PT-v3m1 -> PT-v3m2 (Sonata, Wu et al. 2025),
transposed to LitePT. LitePT-v1 already had three of them (traceable GridPooling
with ``pooling_parent`` / ``pooling_inverse``, ``enc_mode``, LayerNorm in the
blocks). LitePT-v2 adds the others, each behind an option:

(a) ``norm="ln"``: LayerNorm instead of BatchNorm in ``Embedding``, ``GridPooling``
    and ``GridUnpooling`` (``"bn"`` = v1: BatchNorm1d, eps 1e-3, momentum 0.01).
(b) ``stem="linear"``: the stem is ``nn.Linear`` + norm + GELU instead of
    ``SubMConv3d`` k=5 + norm + GELU (``"conv"`` = v1). The k=3 conv of the first
    block keeps the locality.
(c) ``mask_token=True``: a learned token replaces the stem output of the points
    flagged by ``point.mask`` (Sonata masking). It is applied after the stem, so
    with ``stem="conv"`` the stem still mixes masked and visible neighbours.
(d) ``init="trunc_normal"``: ``trunc_normal_(std=0.02)`` for the weights of every
    ``nn.Linear`` and ``SubMConv3d``, zero biases (``"default"`` = v1: PyTorch and
    spconv defaults).
(e) ``layer_scale``: optional LayerScale (``gamma`` initialised to this value) on
    the attention and MLP branches of the attention blocks (None = off).
(f) ``freeze_encoder=True``: freezes the stem and the encoder (decoder probing).

And one change from PT-v3m3 (Utonia), not in Sonata:

(g) RoPE coordinate augmentation at train time (``rope_shift``, ``rope_jitter``,
    ``rope_rescale``; off by default): before PointROPE, the positions of each
    attention layer (``grid_coord`` of the stage, in voxels of that stage) are
    shifted by U(-rope_shift, rope_shift) voxels per axis, multiplied by
    exp(U(-log j, log j)) per axis (``rope_jitter`` = j) and by
    exp(U(-log r, log r)) globally (``rope_rescale`` = r), one draw per layer and
    forward. The pointrope CUDA kernel only takes int64 positions, so augmented
    (float) positions go through a PyTorch implementation of the same rotation.
    In eval mode the positions are not augmented.

Defaults are the PT-v3m2 ones: ``norm="ln"``, ``stem="linear"``,
``init="trunc_normal"``, ``mask_token=False``, ``layer_scale=None``,
``freeze_encoder=False``, and no RoPE augmentation. With ``norm="bn"``,
``stem="conv"``, ``mask_token=False``, ``init="default"`` (and the other options
off), LitePT-v2 has the same state_dict keys as LitePT-v1 and gives the same
outputs for the same weights, so MALiBU3D LitePT-v1 checkpoints load into it.
"""

import math
from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import spconv.pytorch as spconv
from timm.layers import DropPath

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential

from .litept_v1 import (
    MLP,
    GridPooling,
    GridUnpooling,
    PointROPEAttention,
    flash_attn,
)


def float_pointrope(tokens, positions, base, F0=1.0):
    """PointROPE for float positions (same rotation as the pointrope kernel).

    tokens: [B, H, N, head_dim] (head_dim divisible by 6), positions: [B, N, 3].
    Each third of head_dim is rotated by one axis: its first and second halves
    are the (u, v) pairs, rotated by angle ``pos * F0 / base ** (i / Q)``.
    """
    D = tokens.size(-1) // 3
    inv_freq = F0 / (base ** (torch.arange(0, D, 2, device=tokens.device).float() / D))
    out = []
    for axis, x in enumerate(tokens.chunk(3, dim=-1)):
        angle = positions[..., axis, None].float() * inv_freq  # [B, N, D/2]
        angle = torch.cat((angle, angle), dim=-1)[:, None]  # [B, 1, N, D]
        x1, x2 = x[..., : D // 2], x[..., D // 2 :]
        out.append(x * angle.cos() + torch.cat((-x2, x1), dim=-1) * angle.sin())
    return torch.cat(out, dim=-1)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PointROPEAttentionV2(PointROPEAttention):
    """LitePT-v1 attention with optional train-time RoPE coordinate augmentation."""

    def __init__(
        self, *args, rope_shift=None, rope_jitter=None, rope_rescale=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rope_shift = rope_shift
        self.rope_jitter = rope_jitter
        self.rope_rescale = rope_rescale

    @property
    def augment_rope(self):
        return self.training and (
            (self.rope_shift is not None and self.rope_shift > 0)
            or (self.rope_jitter is not None and self.rope_jitter > 1.0)
            or (self.rope_rescale is not None and self.rope_rescale > 1.0)
        )

    def augment_positions(self, pos):
        pos = pos.float()
        dd = {"device": pos.device, "dtype": pos.dtype}
        if self.rope_shift is not None and self.rope_shift > 0:
            pos = pos + torch.empty(3, **dd).uniform_(-self.rope_shift, self.rope_shift)
        if self.rope_jitter is not None and self.rope_jitter > 1.0:
            j = math.log(self.rope_jitter)
            pos = pos * torch.empty(3, **dd).uniform_(-j, j).exp()
        if self.rope_rescale is not None and self.rope_rescale > 1.0:
            r = math.log(self.rope_rescale)
            pos = pos * torch.empty(1, **dd).uniform_(-r, r).exp()
        return pos

    def forward(self, point):
        if not self.augment_rope:
            return super().forward(point)

        H = self.num_heads
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        qkv = self.qkv(point.feat)[order]  # [N, C]

        pos = self.augment_positions(point.grid_coord[order]).unsqueeze(0)  # [1, N, 3]

        q, k, v = qkv.half().chunk(3, dim=-1)
        q = q.reshape(-1, H, C // H).transpose(0, 1)[None]  # [1, H, N, head_dim]
        k = k.reshape(-1, H, C // H).transpose(0, 1)[None]  # [1, H, N, head_dim]
        q = float_pointrope(q.float(), pos, self.rope.base, self.rope.F0).to(q.dtype)
        k = float_pointrope(k.float(), pos, self.rope.base, self.rope.F0).to(k.dtype)

        qkv_rotated = torch.stack(
            [
                q.squeeze(0).transpose(0, 1),
                k.squeeze(0).transpose(0, 1),
                v.reshape(-1, H, C // H),
            ],
            dim=1,
        )  # [N, 3, H, head_dim]

        if flash_attn is None:
            raise ImportError(
                "LitePT forward requires flash_attn, but it is not installed. "
                "Please install flash_attn to run LitePT attention."
            )

        feat = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv_rotated,
            cu_seqlens,
            max_seqlen=self.patch_size,
            dropout_p=self.attn_drop if self.training else 0,
            softmax_scale=self.scale,
        ).reshape(-1, C)

        feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class Block(PointModule):
    """LitePT-v1 block plus optional LayerScale and RoPE augmentation.

    ``ls1`` / ``ls2`` hold no parameters when ``layer_scale`` is None, so the
    state_dict keys are those of the LitePT-v1 block.
    """

    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_conv=True,
        enable_attn=True,
        rope_freq=100.0,
        rope_shift=None,
        rope_jitter=None,
        rope_rescale=None,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.enable_conv = enable_conv
        self.enable_attn = enable_attn

        if self.enable_conv:
            self.conv = PointSequential(
                spconv.SubMConv3d(
                    channels,
                    channels,
                    kernel_size=3,
                    bias=True,
                    indice_key=cpe_indice_key,
                ),
                nn.Linear(channels, channels),
                norm_layer(channels),
            )
        else:
            self.norm0 = PointSequential(
                norm_layer(channels),
            )

        if self.enable_attn:
            self.norm1 = PointSequential(norm_layer(channels))
            self.ls1 = PointSequential(
                LayerScale(channels, init_values=layer_scale)
                if layer_scale is not None
                else nn.Identity()
            )
            self.attn = PointROPEAttentionV2(
                channels=channels,
                patch_size=patch_size,
                rope_freq=rope_freq,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                order_index=order_index,
                rope_shift=rope_shift,
                rope_jitter=rope_jitter,
                rope_rescale=rope_rescale,
            )
            self.norm2 = PointSequential(norm_layer(channels))
            self.ls2 = PointSequential(
                LayerScale(channels, init_values=layer_scale)
                if layer_scale is not None
                else nn.Identity()
            )
            self.mlp = PointSequential(
                MLP(
                    in_channels=channels,
                    hidden_channels=int(channels * mlp_ratio),
                    out_channels=channels,
                    act_layer=act_layer,
                    drop=proj_drop,
                )
            )
            self.drop_path = PointSequential(
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )

    def forward(self, point: Point):
        if self.enable_conv:
            shortcut = point.feat
            point = self.conv(point)
            point.feat = shortcut + point.feat
        else:
            point = self.norm0(point)

        if self.enable_attn:
            shortcut = point.feat
            if self.pre_norm:
                point = self.norm1(point)
            point = self.drop_path(self.ls1(self.attn(point)))
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm1(point)

            shortcut = point.feat
            if self.pre_norm:
                point = self.norm2(point)
            point = self.drop_path(self.ls2(self.mlp(point)))
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm2(point)

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class Embedding(PointModule):
    """Stem (``"conv"``: SubMConv3d k=5 as in LitePT-v1; ``"linear"``: nn.Linear as
    in Sonata) + norm + act, then the optional mask token."""

    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
        stem="linear",
        mask_token=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        if stem == "conv":
            self.stem = PointSequential(
                conv=spconv.SubMConv3d(
                    in_channels,
                    embed_channels,
                    kernel_size=5,
                    padding=1,
                    bias=False,
                    indice_key="stem",
                )
            )
        elif stem == "linear":
            self.stem = PointSequential(linear=nn.Linear(in_channels, embed_channels))
        else:
            raise ValueError(f"stem must be 'conv' or 'linear', got {stem!r}")
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

        if mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, embed_channels))
        else:
            self.mask_token = None

    def forward(self, point: Point):
        point = self.stem(point)
        if self.mask_token is not None and "mask" in point.keys():
            point.feat = torch.where(
                point.mask.unsqueeze(-1),
                self.mask_token.to(point.feat.dtype),
                point.feat,
            )
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


@MODELS.register_module("LitePT-v2")
class LitePT(PointModule):
    def __init__(
        self,
        in_channels=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enc_mode=False,
        dec_traceable=False,
        # LitePT-v2 options (see the module docstring)
        norm="ln",
        stem="linear",
        mask_token=False,
        init="trunc_normal",
        layer_scale=None,
        freeze_encoder=False,
        rope_shift=None,
        rope_jitter=None,
        rope_rescale=None,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.enc_mode = enc_mode
        self.dec_traceable = dec_traceable
        self.shuffle_orders = shuffle_orders
        self.freeze_encoder = freeze_encoder

        self.enc_conv = enc_conv
        self.enc_attn = enc_attn
        self.dec_conv = dec_conv
        self.dec_attn = dec_attn

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1
        assert self.enc_mode or self.num_stages == len(dec_channels) + 1
        assert self.enc_mode or self.num_stages == len(dec_num_head) + 1
        assert self.enc_mode or self.num_stages == len(dec_patch_size) + 1
        assert init in ("trunc_normal", "default"), init

        # norm layers: blocks always use LayerNorm; `norm` sets the stem,
        # pooling and unpooling norm
        if norm == "bn":
            io_norm_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        elif norm == "ln":
            io_norm_layer = nn.LayerNorm
        else:
            raise ValueError(f"norm must be 'bn' or 'ln', got {norm!r}")
        ln_layer = nn.LayerNorm

        # activation layers
        act_layer = nn.GELU

        rope_aug = dict(
            rope_shift=rope_shift, rope_jitter=rope_jitter, rope_rescale=rope_rescale
        )

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=io_norm_layer,
            act_layer=act_layer,
            stem=stem,
            mask_token=mask_token,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=io_norm_layer,
                        act_layer=act_layer,
                        re_serialization=enc_attn[s],
                        serialization_order=self.order,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        layer_scale=layer_scale,
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_conv=enc_conv[s],
                        enable_attn=enc_attn[s],
                        rope_freq=enc_rope_freq[s],
                        **rope_aug,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.enc_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=io_norm_layer,
                        act_layer=act_layer,
                        traceable=self.dec_traceable,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            layer_scale=layer_scale,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_conv=dec_conv[s],
                            enable_attn=dec_attn[s],
                            rope_freq=dec_rope_freq[s],
                            **rope_aug,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

        if init == "trunc_normal":
            self.apply(self._init_weights)
        if self.freeze_encoder:
            for p in self.embedding.parameters():
                p.requires_grad = False
            for p in self.enc.parameters():
                p.requires_grad = False

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, spconv.SubMConv3d)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, data_dict):
        """
        data_dict is the batched input point cloud, it should contain as least:
        1. feat [N, input_dim]: input feature for the point cloud
        2. grid_coord [N, 3]: voxelized coordinate after grid sampling
           or/and
           coord [N, 3]: original coordinate + grid_size: grid_size used for grid sampling
        3. offset [batch_size]: separator of point clouds in batched data
           or/and
           batch [N]: batch index of each point
        4. (optional) mask [N] bool: points whose stem output is replaced by the
           mask token (when ``mask_token=True``)
        """
        point = Point(data_dict)
        if self.enc_attn[0]:
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)

        if not self.enc_mode:
            point = self.dec(point)

        return point
