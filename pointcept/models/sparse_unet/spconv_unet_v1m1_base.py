"""
SparseUNet Driven by SpConv (recommend)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import spconv.pytorch as spconv
from torch_geometric.utils import scatter

from timm.layers import trunc_normal_

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch
from pointcept.models.utils.structure import Point


def _pooling_inverse(fine_indices, coarse_indices, stride):
    """Fine-to-coarse voxel mapping of a SpUNet down-sampling step.

    For each row of `fine_indices` ([Nf, 4] = batch, z, y, x), returns the row
    of `coarse_indices` ([Nc, 4]) it pools into, for a SparseConv3d with
    kernel_size == stride and no padding (a non-overlapping partition:
    coarse_coord = fine_coord // stride, batch unchanged). The match is done
    against the coarse tensor actually produced by spconv, so it relies only on
    the pooling being a partition, not on spconv's output-coordinate convention.
    """
    fine_key = fine_indices.clone()
    fine_key[:, 1:] = torch.div(fine_indices[:, 1:], stride, rounding_mode="floor")
    n_coarse = coarse_indices.shape[0]
    combined = torch.cat([coarse_indices, fine_key], dim=0)
    unique_rows, inverse = torch.unique(combined, sorted=True, return_inverse=True, dim=0)
    coarse_group = inverse[:n_coarse]
    fine_group = inverse[n_coarse:]
    assert unique_rows.shape[0] == n_coarse, (
        "fine-to-coarse voxel correspondence mismatch: some fine voxels have "
        "no matching coarse voxel under coord // stride — SpUNet's down-conv "
        "stride/padding no longer matches the assumed clean-partition pooling."
    )
    row_of_group = torch.empty(n_coarse, dtype=torch.long, device=fine_indices.device)
    row_of_group[coarse_group] = torch.arange(n_coarse, device=fine_indices.device)
    return row_of_group[fine_group]


def _walk_pooling_parent_chain(point):
    """Consume a pooling_parent/pooling_inverse chain (built by
    SpUNetBase._point_mode_forward) and return the finest level's concatenated
    features. Same walk as DefaultSegmentorV2 / GridProbeSegmentorV2, done here
    so the encoder features can be appended to the dec_point_mode output.
    """
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point.feat


def _as_stage_strides(stride, num_stages):
    """Broadcast an int to every pooling stage, or validate a per-stage sequence.

    Does not change the number of pooling layers: a sequence must match
    ``num_stages`` (``len(layers) // 2``).
    """
    if isinstance(stride, int):
        return (int(stride),) * num_stages
    stride = tuple(int(s) for s in stride)
    assert len(stride) == num_stages, (
        f"stride length {len(stride)} != num_stages {num_stages}; "
        "stride does not change the number of pooling layers"
    )
    return stride


class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        in_channels,
        embed_channels,
        stride=1,
        norm_fn=None,
        indice_key=None,
        bias=False,
    ):
        super().__init__()

        assert norm_fn is not None

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, embed_channels, kernel_size=1, bias=False
                ),
                norm_fn(embed_channels),
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out


@MODELS.register_module("SpUNet-v1m1")
class SpUNetBase(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        # MALiBU3D additions: `stride` (int, or one value per stage) sets the
        # kernel size and stride of the down/up-sampling convs (upstream: 2);
        # `point_mode` / `dec_point_mode` are described below.
        stride=2,
        enc_mode=False,
        point_mode=False,
        dec_point_mode=False,
    ):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        assert not (enc_mode and (point_mode or dec_point_mode)), (
            "enc_mode (global per-scene pooling, for classification) is mutually "
            "exclusive with point_mode/dec_point_mode (per-point features for "
            "GridProbeSegmentorV2) — enc_mode never builds the decoder these need."
        )
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        self.stride = _as_stage_strides(stride, self.num_stages)
        self.enc_mode = enc_mode
        # point_mode: encoder multi-scale features for linear probing. Instead
        # of logits, forward returns the encoder levels (stem, stage0, ...,
        # bottleneck) as a Point chain linked by `pooling_parent` /
        # `pooling_inverse`, like PT-v3's GridPooling, so that
        # DefaultSegmentorV2 / GridProbeSegmentorV2 concatenate every level
        # onto the finest voxels (unlike enc_mode, which returns one pooled
        # vector per scene). No new parameters: works with any SpUNet checkpoint.
        self.point_mode = point_mode
        # dec_point_mode: the decoder counterpart (like LitePT dec_traceable):
        # forward returns the decoder stage outputs and the bottleneck as a
        # Point chain linked by `unpooling_parent` / `pooling_inverse`, giving
        # a decoder hypercolumn (832 channels with the default `channels`).
        # Decoder stage s lives on the same voxels as skips[s] (up[s] reuses
        # down[s]'s indice_key), so the mapping is computed with
        # _pooling_inverse as in point_mode. Combined with point_mode=True, the
        # encoder levels except the bottleneck (already included) are appended
        # (1088 channels with the default `channels`).
        self.dec_point_mode = dec_point_mode

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.ReLU(),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList() if not self.enc_mode else None

        for s in range(self.num_stages):
            pool_stride = self.stride[s]
            # encode num_stages
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(
                        enc_channels,
                        channels[s],
                        kernel_size=pool_stride,
                        stride=pool_stride,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                    ),
                    norm_fn(channels[s]),
                    nn.ReLU(),
                )
            )
            self.enc.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            # (f"block{i}", block(enc_channels, channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                            # if i == 0 else
                            (
                                f"block{i}",
                                block(
                                    channels[s],
                                    channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}",
                                ),
                            )
                            for i in range(layers[s])
                        ]
                    )
                )
            )
            if not self.enc_mode:
                # decode num_stages
                self.up.append(
                    spconv.SparseSequential(
                        spconv.SparseInverseConv3d(
                            channels[len(channels) - s - 2],
                            dec_channels,
                            kernel_size=pool_stride,
                            bias=False,
                            indice_key=f"spconv{s + 1}",
                        ),
                        norm_fn(dec_channels),
                        nn.ReLU(),
                    )
                )
                self.dec.append(
                    spconv.SparseSequential(
                        OrderedDict(
                            [
                                (
                                    (
                                        f"block{i}",
                                        block(
                                            dec_channels + enc_channels,
                                            dec_channels,
                                            norm_fn=norm_fn,
                                            indice_key=f"subm{s}",
                                        ),
                                    )
                                    if i == 0
                                    else (
                                        f"block{i}",
                                        block(
                                            dec_channels,
                                            dec_channels,
                                            norm_fn=norm_fn,
                                            indice_key=f"subm{s}",
                                        ),
                                    )
                                )
                                for i in range(layers[len(channels) - s - 1])
                            ]
                        )
                    )
                )

            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        final_in_channels = (
            channels[-1] if not self.enc_mode else channels[self.num_stages - 1]
        )
        self.final = (
            spconv.SubMConv3d(
                final_in_channels, num_classes, kernel_size=1, padding=1, bias=True
            )
            if num_classes > 0
            else spconv.Identity()
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_dict):
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]

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
        x = self.conv_input(x)
        skips = [x]
        # enc forward
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)

        if self.point_mode or self.dec_point_mode:
            enc_extra_feat = None
            if self.point_mode:
                # With dec_point_mode, the decoder chain already contains the
                # bottleneck: leave it out here so it is not concatenated twice.
                enc_levels = skips[:-1] if self.dec_point_mode else skips
                enc_point = self._point_mode_forward(enc_levels)
                if not self.dec_point_mode:
                    return enc_point
                # Concatenate the encoder levels now (enc_point.feat alone is
                # only the coarsest level) to append them to the decoder output.
                enc_extra_feat = _walk_pooling_parent_chain(enc_point)
            point = self._dec_point_mode_forward(skips)
            if enc_extra_feat is not None:
                point.feat = torch.cat([point.feat, enc_extra_feat], dim=-1)
            return point

        x = skips.pop(-1)
        if not self.enc_mode:
            # dec forward
            for s in reversed(range(self.num_stages)):
                x = self.up[s](x)
                skip = skips.pop(-1)
                x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
                x = self.dec[s](x)

        x = self.final(x)
        if self.enc_mode:
            x = x.replace_feature(
                scatter(x.features, x.indices[:, 0].long(), reduce="mean", dim=0)
            )
        return x.features

    def _point_mode_forward(self, skips):
        """Build the stem -> ... -> bottleneck Point chain for point_mode.

        `skips`: the encoder SparseConvTensors, finest (stem output, before any
        pooling) first and coarsest last. Returns the coarsest Point; its
        `pooling_parent` chain gives the finest-first concatenation order of
        LitePT / PT-v3 enc_mode.
        """
        points = [Point(feat=level.features) for level in skips]
        for s in range(len(skips) - 1, 0, -1):
            points[s]["pooling_parent"] = points[s - 1]
            points[s]["pooling_inverse"] = _pooling_inverse(
                skips[s - 1].indices, skips[s].indices, self.stride[s - 1]
            )
        return points[-1]

    def _dec_point_mode_forward(self, skips):
        """Run the decoder and build its Point chain for dec_point_mode.

        Mirrors PT-v3's GridUnpooling(traceable=True): each finer point links
        to the coarser one through `unpooling_parent`, and the coarser point
        holds the `pooling_inverse` mapping from the finer rows to its own.
        The hypercolumn walk then concatenates dec-stage0, dec-stage1, ...,
        and the bottleneck, finest first. `skips` is consumed (popped), as in
        the plain decoder forward.
        """
        x = skips.pop(-1)  # bottleneck
        point = Point(feat=x.features)
        for s in reversed(range(self.num_stages)):
            prev_x, prev_point = x, point
            x = self.up[s](x)
            skip = skips.pop(-1)
            x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
            x = self.dec[s](x)
            point = Point(feat=x.features)
            point["unpooling_parent"] = prev_point
            prev_point["pooling_inverse"] = _pooling_inverse(
                x.indices, prev_x.indices, self.stride[s]
            )
        return point


