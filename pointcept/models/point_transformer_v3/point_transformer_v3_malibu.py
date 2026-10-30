"""
Point Transformer V3, MALiBU3D variant (registered as "PT-v3-malibu")

PT-v3m2 (point_transformer_v3m2_sonata.py, by Xiaoyang Wu) with one change: the
stem is a SubMConv3d with kernel_size=5 instead of PT-v3m2's Linear, followed by
LayerNorm + GELU, and it therefore runs after serialization/sparsification.
Pooling is PT-v3m2's GridPooling, which accepts any integer stride (e.g. 3).

Stem weights are not interchangeable with PT-v3m2 / Sonata checkpoints.
"""

import spconv.pytorch as spconv
import torch.nn as nn

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointSequential
from .point_transformer_v3m2_sonata import Embedding, PointTransformerV3


class SubMConvEmbedding(Embedding):
    def __init__(self, in_channels, embed_channels, **kwargs):
        super().__init__(in_channels, embed_channels, **kwargs)
        stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        for name, module in self.stem.named_children():
            if name != "linear":
                stem.add(module, name=name)
        self.stem = stem


@MODELS.register_module("PT-v3-malibu")
class PointTransformerV3Malibu(PointTransformerV3):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        embedding = self.embedding
        self.embedding = SubMConvEmbedding(
            embedding.in_channels,
            embedding.embed_channels,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            mask_token=embedding.mask_token is not None,
        )
        self.embedding.apply(self._init_weights)
        if self.freeze_encoder:
            for p in self.embedding.parameters():
                p.requires_grad = False

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)
        return point
