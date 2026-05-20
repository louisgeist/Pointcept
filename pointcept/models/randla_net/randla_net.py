"""
RandLA-Net backbone for Pointcept.

This implementation is native to Pointcept input contracts (coord/feat/offset)
and keeps the main RandLA-Net design principles:
- local feature aggregation with attentive pooling,
- random decimation in the encoder,
- nearest-neighbor interpolation in the decoder.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS

try:
    import pointops
except ImportError:
    pointops = None


def _iter_offset_ranges(offset: torch.Tensor) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = 0
    for end in offset.tolist():
        end_i = int(end)
        ranges.append((start, end_i))
        start = end_i
    return ranges


def _safe_index(feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # pointops may return -1 placeholders when neighborhoods are incomplete.
    pad = torch.zeros((1, feat.shape[1]), device=feat.device, dtype=feat.dtype)
    feat_pad = torch.cat([feat, pad], dim=0)
    safe = idx.clone().long()
    safe[safe < 0] = feat.shape[0]
    return feat_pad[safe]


def _safe_index_xyz(xyz: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    pad = torch.zeros((1, 3), device=xyz.device, dtype=xyz.dtype)
    xyz_pad = torch.cat([xyz, pad], dim=0)
    safe = idx.clone().long()
    safe[safe < 0] = xyz.shape[0]
    return xyz_pad[safe]


def _knn_fallback(
    xyz: torch.Tensor,
    offset: torch.Tensor,
    query_xyz: torch.Tensor,
    query_offset: torch.Tensor,
    k: int,
) -> torch.Tensor:
    # CPU/GPU-safe fallback used when pointops is unavailable.
    # This branch is intended for compatibility/smoke checks, not peak speed.
    device = xyz.device
    out = torch.full((query_xyz.shape[0], k), -1, device=device, dtype=torch.long)
    src_ranges = _iter_offset_ranges(offset)
    qry_ranges = _iter_offset_ranges(query_offset)
    for (s0, s1), (q0, q1) in zip(src_ranges, qry_ranges):
        src = xyz[s0:s1]
        qry = query_xyz[q0:q1]
        if src.numel() == 0 or qry.numel() == 0:
            continue
        kk = min(k, src.shape[0])
        dist = torch.cdist(qry, src)
        idx_local = dist.topk(k=kk, dim=1, largest=False).indices
        idx_global = idx_local + s0
        out[q0:q1, :kk] = idx_global
        if kk < k:
            out[q0:q1, kk:] = idx_global[:, -1:].expand(-1, k - kk)
    return out


def knn_query(
    xyz: torch.Tensor,
    offset: torch.Tensor,
    query_xyz: torch.Tensor,
    query_offset: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if pointops is not None:
        idx, _ = pointops.knn_query(
            k,
            xyz.contiguous(),
            offset.contiguous().int(),
            query_xyz.contiguous(),
            query_offset.contiguous().int(),
        )
        return idx.long()
    return _knn_fallback(xyz, offset, query_xyz, query_offset, k)


class PointBatchNorm(nn.Module):
    def __init__(self, channels: int, momentum: float = 0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels, eps=1e-6, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.bn(x)
        if x.dim() == 3:
            # [N, K, C] -> [N, C, K] -> BN -> back
            return self.bn(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        raise ValueError(f"Unsupported tensor shape for PointBatchNorm: {tuple(x.shape)}")


class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: bool = True,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = PointBatchNorm(out_channels, momentum=momentum)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class AttentivePooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            PointBatchNorm(in_channels, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_channels, in_channels, bias=False),
        )
        self.post = SharedMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=True,
            momentum=momentum,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, K, C]
        score = self.score_fn(x)
        attn = torch.softmax(score, dim=1)
        feat = torch.sum(attn * x, dim=1)  # [N, C]
        return self.post(feat)


class LocalFeatureAggregation(nn.Module):
    def __init__(self, channels: int, neighbors: int, momentum: float = 0.01):
        super().__init__()
        self.neighbors = int(neighbors)
        self.pos_mlp = SharedMLP(10, channels // 2, momentum=momentum)
        self.mlp1 = SharedMLP(channels + channels // 2, channels // 2, momentum=momentum)
        self.pool1 = AttentivePooling(channels // 2, channels // 2, momentum=momentum)
        self.mlp2 = SharedMLP(channels, channels // 2, momentum=momentum)
        self.pool2 = AttentivePooling(channels // 2, channels, momentum=momentum)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        idx = knn_query(xyz, offset, xyz, offset, self.neighbors)
        neigh_xyz = _safe_index_xyz(xyz, idx)
        center_xyz = xyz.unsqueeze(1).expand(-1, self.neighbors, -1)
        rel_xyz = neigh_xyz - center_xyz
        rel_dist = torch.norm(rel_xyz, p=2, dim=-1, keepdim=True)
        pos_enc = torch.cat([rel_xyz, rel_dist, center_xyz, neigh_xyz], dim=-1)
        pos_enc = self.pos_mlp(pos_enc)

        neigh_feat = _safe_index(feat, idx)
        feat1 = self.mlp1(torch.cat([neigh_feat, pos_enc], dim=-1))
        feat1 = self.pool1(feat1)  # [N, C//2]

        feat1_expand = feat1.unsqueeze(1).expand(-1, self.neighbors, -1)
        feat2 = self.mlp2(torch.cat([feat1_expand, pos_enc], dim=-1))
        feat2 = self.pool2(feat2)  # [N, C]
        return feat2


class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, neighbors: int, momentum: float = 0.01):
        super().__init__()
        mid = out_channels // 2
        self.pre = SharedMLP(in_channels, mid, momentum=momentum)
        self.lfa = LocalFeatureAggregation(mid, neighbors=neighbors, momentum=momentum)
        self.post = SharedMLP(mid, out_channels, activation=False, momentum=momentum)
        self.shortcut = (
            SharedMLP(in_channels, out_channels, activation=False, momentum=momentum)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        x = self.pre(feat)
        x = self.lfa(xyz, x, offset)
        x = self.post(x)
        shortcut = self.shortcut(feat)
        return self.act(x + shortcut)


def random_decimate(
    xyz: torch.Tensor,
    feat: torch.Tensor,
    offset: torch.Tensor,
    decimation: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if decimation <= 1:
        return xyz, feat, offset
    xyz_out = []
    feat_out = []
    new_offsets = []
    running = 0
    for start, end in _iter_offset_ranges(offset):
        n = end - start
        keep = max(1, n // decimation)
        perm = torch.randperm(n, device=xyz.device)[:keep] + start
        xyz_out.append(xyz[perm])
        feat_out.append(feat[perm])
        running += keep
        new_offsets.append(running)
    return (
        torch.cat(xyz_out, dim=0),
        torch.cat(feat_out, dim=0),
        torch.tensor(new_offsets, device=offset.device, dtype=offset.dtype),
    )


def nearest_interpolate(
    src_xyz: torch.Tensor,
    src_feat: torch.Tensor,
    src_offset: torch.Tensor,
    dst_xyz: torch.Tensor,
    dst_offset: torch.Tensor,
) -> torch.Tensor:
    idx = knn_query(src_xyz, src_offset, dst_xyz, dst_offset, 1).squeeze(1)
    gathered = _safe_index(src_feat, idx.unsqueeze(1)).squeeze(1)
    return gathered


@MODELS.register_module("RandLA-Net")
class RandLANet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        task: str = "cloud_segmentation",
        encoder_channels: Tuple[int, int, int, int] = (32, 64, 128, 256),
        decoder_channels: Tuple[int, int, int] = (128, 64, 32),
        neighbors: int = 16,
        decimation: int = 4,
        bn_momentum: float = 0.01,
        dropout: float = 0.0,
    ):
        super().__init__()
        if task not in {"cloud_segmentation", "classification"}:
            raise ValueError(f"Unsupported task={task!r}")
        self.task = task
        self.num_classes = int(num_classes)
        self.decimation = int(decimation)

        self.stem = SharedMLP(
            in_channels=input_channels,
            out_channels=encoder_channels[0],
            momentum=bn_momentum,
        )

        self.encoder = nn.ModuleList()
        in_c = encoder_channels[0]
        for out_c in encoder_channels:
            self.encoder.append(
                DilatedResidualBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    neighbors=neighbors,
                    momentum=bn_momentum,
                )
            )
            in_c = out_c

        self.decoder_fuse = nn.ModuleList()
        current_c = encoder_channels[-1]
        skip_channels = list(encoder_channels[:-1])[::-1]
        for out_c, skip_c in zip(decoder_channels, skip_channels):
            self.decoder_fuse.append(
                SharedMLP(
                    in_channels=current_c + skip_c,
                    out_channels=out_c,
                    momentum=bn_momentum,
                )
            )
            current_c = out_c

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.final = nn.Linear(current_c, self.num_classes) if self.num_classes > 0 else None

    def forward(self, data_dict: dict) -> torch.Tensor:
        xyz = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()

        feat = self.stem(feat)
        xyz_stack = [xyz]
        feat_stack = [feat]
        offset_stack = [offset]

        # Encoder with random decimation between stages.
        for i, block in enumerate(self.encoder):
            feat = block(xyz, feat, offset)
            xyz_stack.append(xyz)
            feat_stack.append(feat)
            offset_stack.append(offset)
            if i < len(self.encoder) - 1:
                xyz, feat, offset = random_decimate(
                    xyz=xyz,
                    feat=feat,
                    offset=offset,
                    decimation=self.decimation,
                )

        # Decoder (nearest interpolation + skip fusion).
        for i, fuse in enumerate(self.decoder_fuse):
            src_xyz, src_feat, src_offset = xyz, feat, offset
            dst_xyz = xyz_stack[-(i + 2)]
            dst_feat = feat_stack[-(i + 2)]
            dst_offset = offset_stack[-(i + 2)]
            up_feat = nearest_interpolate(src_xyz, src_feat, src_offset, dst_xyz, dst_offset)
            feat = fuse(torch.cat([up_feat, dst_feat], dim=-1))
            xyz, offset = dst_xyz, dst_offset

        feat = self.dropout(feat)

        if self.task == "classification":
            pooled = []
            for start, end in _iter_offset_ranges(offset):
                pooled.append(feat[start:end].mean(dim=0, keepdim=True))
            out = torch.cat(pooled, dim=0)
            if self.final is not None:
                out = self.final(out)
            return out

        if self.final is None:
            return feat
        return self.final(feat)
