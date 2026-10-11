"""
Grid-Search Linear Probing

Trains N independently configured linear "probe" heads on top of a single
shared frozen-backbone forward pass, so a hyperparameter grid search over
loss/optimizer/scheduler/dropout/normalization pays the backbone cost once
per batch instead of once per grid point.

  - GridProbeSegmentorV2: semantic segmentation probes (per-point logits,
    target e.g. "segment").
  - GridProbeClassifier: tile/scene classification probes (multi-scale
    features -> scene pooling -> linear heads, target e.g. "category").

This module only defines the probe heads and their losses. Per-probe
optimizers/schedulers are handled by GridProbeTrainer (pointcept/engines/train.py),
validation and winner selection by the GridProbe hooks
(pointcept/engines/hooks/grid_probe.py).
"""

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model
from .default import LearnedMaskedFeatMixin, _safe_segment_csr_mean

_INPUT_NORM_P = {
    "l1": 1,
    "l2": 2,
    "linf": float("inf"),
    "l1_block": 1,
    "l2_block": 2,
    "linf_block": float("inf"),
}
_FEAT_NORM_TYPES = frozenset((None, "batchnorm", "layernorm"))


def _pnorm_scale(x, p, eps):
    """x / ||x||_p along the last dim (norm clamped at `eps`).

    Under autocast, .norm() returns fp32; the norm is cast back to x's dtype so
    that the result keeps x's dtype instead of being promoted to fp32.
    """
    norm = x.norm(p=p, dim=-1, keepdim=True).clamp_min(eps)
    return x / norm.to(x.dtype)


def _unitsphere(feat, kind, eps=1e-6, channel_blocks=None):
    """Row-wise rescaling onto the unit sphere.

    None returns `feat` unchanged. "l1"/"l2"/"linf" divide the whole feature
    vector by its p-norm. "l1_block"/"l2_block"/"linf_block" split `feat` into
    `channel_blocks` contiguous chunks along the last dim (in the finest-first
    concat order of _forward_backbone, e.g. dec_channels + (bottleneck_channels,))
    and rescale each chunk independently, so that every backbone level of a
    multi-scale feature gets the same weight.
    """
    if kind is None:
        return feat
    if kind not in _INPUT_NORM_P:
        raise ValueError(
            f"input_norm must be one of None/l1/l2/linf/l1_block/l2_block/linf_block, got {kind!r}."
        )
    p = _INPUT_NORM_P[kind]
    if kind.endswith("_block"):
        if channel_blocks is None:
            raise ValueError(
                f"input_norm={kind!r} requires channel_blocks to be set on the model."
            )
        # Fill a preallocated output instead of torch.cat-ing per-block
        # results: peak extra memory stays ~1x feat instead of ~2x.
        out = torch.empty_like(feat)
        offset = 0
        for size in channel_blocks:
            out[..., offset : offset + size] = _pnorm_scale(
                feat[..., offset : offset + size], p, eps
            )
            offset += size
        return out
    return _pnorm_scale(feat, p, eps)


def _prepare_shared_feat(feat):
    """One contiguous (+ AMP-dtype) copy of backbone feat for every probe head."""
    feat = feat.contiguous()
    if torch.is_autocast_enabled():
        feat = feat.to(dtype=torch.get_autocast_gpu_dtype())
    return feat


def _input_norm_cache(feat, heads, names, channel_blocks=None):
    """Normalized features keyed by (input_norm, eps), computed once per key
    among the heads in `names` (heads sharing an input_norm share one copy)."""
    cache = {}
    for name in names:
        head = heads[name]
        key = (head.input_norm, head.eps)
        if key not in cache:
            cache[key] = _unitsphere(feat, head.input_norm, head.eps, channel_blocks)
    return cache


class ProbeHead(nn.Module):
    """One grid-search point: optional input-feature normalization, optional
    feature norm (BatchNorm1d/LayerNorm), optional dropout, then a linear
    classifier. All knobs are independent per probe.
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        input_norm=None,
        feat_norm=None,
        dropout=0.0,
        eps=1e-6,
        channel_blocks=None,
    ):
        super().__init__()
        if input_norm not in (None,) and input_norm not in _INPUT_NORM_P:
            raise ValueError(
                f"input_norm must be one of None/l1/l2/linf/l1_block/l2_block/linf_block, got {input_norm!r}."
            )
        if feat_norm not in _FEAT_NORM_TYPES:
            raise ValueError(f"feat_norm must be one of {sorted(str(x) for x in _FEAT_NORM_TYPES)}, got {feat_norm!r}.")
        self.input_norm = input_norm
        self.eps = eps
        self.channel_blocks = channel_blocks
        if feat_norm == "batchnorm":
            self.norm = nn.BatchNorm1d(in_channels)
        elif feat_norm == "layernorm":
            self.norm = nn.LayerNorm(in_channels)
        else:
            self.norm = nn.Identity()
        self.dropout = nn.Dropout(p=float(dropout)) if dropout else nn.Identity()
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, feat, apply_input_norm=True):
        """apply_input_norm=False when `feat` comes from the shared
        _input_norm_cache (already normalized)."""
        if apply_input_norm:
            feat = _unitsphere(feat, self.input_norm, self.eps, self.channel_blocks)
        feat = self.norm(feat)
        feat = self.dropout(feat)
        return self.linear(feat)


@MODELS.register_module()
class GridProbeSegmentorV2(nn.Module, LearnedMaskedFeatMixin):
    """N linear probe heads sharing one frozen-backbone forward pass.

    probes: Mapping[str, dict], each entry:
      - input_norm: None | "l1" | "l2" | "linf" | "l1_block" | "l2_block" |
        "linf_block" (see _unitsphere); the "_block" kinds need channel_blocks.
      - feat_norm: None | "batchnorm" | "layernorm"
      - dropout: float
      - criteria: list of loss cfgs (same format as DefaultSegmentorV2.criteria)
      - num_classes: optional per-probe override of ``num_classes``
    Optimizer/scheduler/grad_clip may also live in each probe dict; they are
    consumed by GridProbeTrainer, not this class.

    drop_leading_channels: model-level (not per-probe); drops the first N
    channels of the shared backbone feat before every probe head. Multi-scale
    features are concatenated finest level first (see _forward_backbone), so
    e.g. drop_leading_channels=dec_channels[0] removes the finest decoder level.

    channel_blocks: model-level (not per-probe); channel count of each
    concatenated level in the shared backbone feat, finest first (e.g.
    dec_channels + (bottleneck_channels,) for a decoder hypercolumn, or
    enc_channels for encoder multi-scale features), after
    drop_leading_channels: sum(channel_blocks) must equal
    backbone_out_channels - drop_leading_channels. Required only when a probe
    uses a "*_block" input_norm.

    freeze_backbone=True (default) runs the backbone forward under
    torch.no_grad() and keeps the whole backbone in eval() mode, re-applied on
    every .train() call (same contract as FrozenBackboneEvalMixin in
    default.py): BatchNorm running statistics do not drift toward the
    downstream dataset, and DropPath / Dropout (including KPConvX's
    DropPathPack) stay inactive. Learned masked-feat fill values are frozen
    too. There is deliberately no opt-out.
    """

    def __init__(
        self,
        probes,
        backbone_out_channels,
        num_classes,
        ignore_index=-1,
        backbone=None,
        target_key="segment",
        freeze_backbone=True,
        feature_mask_values=None,
        drop_leading_channels=0,
        channel_blocks=None,
    ):
        super().__init__()
        if not isinstance(probes, Mapping) or len(probes) == 0:
            raise ValueError("probes must be a non-empty mapping of probe_name -> config.")
        self.probe_configs = {str(k): dict(v) for k, v in probes.items()}
        self.probe_names = tuple(self.probe_configs.keys())
        self.target_key = target_key
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        # None: all probes are computed. GridProbeWinnerSelector sets it to the
        # winning probe, which then runs alone (see forward()).
        self.active_probe = None

        self.drop_leading_channels = int(drop_leading_channels)
        if not 0 <= self.drop_leading_channels < backbone_out_channels:
            raise ValueError(
                f"drop_leading_channels must be in [0, {backbone_out_channels}), "
                f"got {self.drop_leading_channels}."
            )
        probe_in_channels = backbone_out_channels - self.drop_leading_channels

        if channel_blocks is not None:
            channel_blocks = tuple(int(c) for c in channel_blocks)
            if any(c <= 0 for c in channel_blocks):
                raise ValueError(f"channel_blocks entries must all be positive, got {channel_blocks}.")
            if sum(channel_blocks) != probe_in_channels:
                raise ValueError(
                    f"sum(channel_blocks)={sum(channel_blocks)} must equal "
                    f"backbone_out_channels - drop_leading_channels={probe_in_channels}."
                )
        self.channel_blocks = channel_blocks

        uses_block_norm = any(
            str(probe_cfg.get("input_norm")).endswith("_block")
            for probe_cfg in self.probe_configs.values()
        )
        if uses_block_norm and self.channel_blocks is None:
            raise ValueError("channel_blocks must be set when any probe uses a '*_block' input_norm.")

        self.backbone = build_model(backbone)
        self.heads = nn.ModuleDict()
        self.criteria_by_task = {}
        for name, probe_cfg in self.probe_configs.items():
            self.heads[name] = ProbeHead(
                in_channels=probe_in_channels,
                num_classes=probe_cfg.get("num_classes", self.num_classes),
                input_norm=probe_cfg.get("input_norm"),
                feat_norm=probe_cfg.get("feat_norm"),
                dropout=probe_cfg.get("dropout", 0.0),
                channel_blocks=self.channel_blocks,
            )
            self.criteria_by_task[name] = build_criteria(probe_cfg.get("criteria"))

        self._init_learned_masked_feat(feature_mask_values=feature_mask_values)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            # Learned masked-feat fill values (e.g. color_mask_value) belong to
            # the backbone input: frozen with it.
            if self.enable_learned_masked_feat:
                for feat_key in self.learned_masked_feat_keys:
                    mask_value = getattr(self, f"{feat_key}_mask_value", None)
                    if mask_value is not None:
                        mask_value.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            # nn.Module.train() just switched every backbone submodule to
            # `mode`; a frozen backbone must stay in eval() (BatchNorm running
            # stats, DropPath/DropPathPack, Dropout).
            self.backbone.eval()
        return self

    def backbone_parameters(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def probe_head_parameters(self, name):
        """Trainable parameters of one probe (GridProbeTrainer builds one
        optimizer per probe from them)."""
        return [p for p in self.heads[name].parameters() if p.requires_grad]

    def _active_probe_names(self):
        if self.active_probe is not None:
            return (self.active_probe,)
        return self.probe_names

    def _forward_backbone(self, input_dict):
        """Per-point backbone features, with the decoder hypercolumn and the
        encoder multi-scale features concatenated finest level first, exactly
        as in DefaultSegmentorV2.forward (see the comments there)."""
        point = Point(input_dict)
        point = self.backbone(point)
        if isinstance(point, Point):
            # Decoder hypercolumn (traceable decoders only; no-op otherwise).
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                child, parent = point_list[i], point_list[i - 1]
                assert "pooling_inverse" in child.keys()
                parent.feat = torch.cat([parent.feat, child.feat[child.pooling_inverse]], dim=-1)
            point = point_list[0]
            # Encoder multi-scale features (enc_mode / SpUNet point_mode).
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        return feat, point

    def prepare_batch(self, input_dict):
        """Run the backbone once; return (features by input_norm, Point, active probe names)."""
        self._fill_masked_feat_with_learned_value(input_dict)
        if self.freeze_backbone:
            with torch.no_grad():
                feat, point = self._forward_backbone(input_dict)
        else:
            feat, point = self._forward_backbone(input_dict)
        if self.drop_leading_channels:
            feat = feat[..., self.drop_leading_channels:]
        active = self._active_probe_names()
        feat = _prepare_shared_feat(feat)
        feat_by_norm = _input_norm_cache(feat, self.heads, active, self.channel_blocks)
        return feat_by_norm, point, active

    def probe_logits(self, name, feat_by_norm):
        head = self.heads[name]
        return head(
            feat_by_norm[(head.input_norm, head.eps)],
            apply_input_norm=False,
        )

    def forward(self, input_dict, return_point=False):
        feat_by_norm, point, active = self.prepare_batch(input_dict)
        return_dict = {}
        if return_point:
            return_dict["point"] = point
        else:
            # Release the backbone's internal state (serialization orders,
            # padding indices, sparse tensors, ...) before running the heads;
            # nothing below reads `point`.
            del point
        seg_logits_by_task = {
            name: self.probe_logits(name, feat_by_norm) for name in active
        }
        return_dict["seg_logits_by_task"] = seg_logits_by_task

        has_target = self.target_key in input_dict
        if has_target:
            target = input_dict[self.target_key]
            loss_by_task = {}
            total_loss = None
            for name in active:
                task_loss = self.criteria_by_task[name](seg_logits_by_task[name], target)
                loss_by_task[name] = task_loss
                total_loss = task_loss if total_loss is None else total_loss + task_loss
            return_dict["loss"] = total_loss
            return_dict["loss_by_task"] = loss_by_task

        if self.training and has_target:
            # Expose the shared target under each probe's name: InformationWriter
            # accumulates per-task train mIoU from input_dict[task_name].
            # (GridProbeEvaluator reads target_key directly.)
            for name in active:
                input_dict.setdefault(name, target)
            with torch.no_grad():
                return_dict["pred_by_task"] = {
                    name: logits.argmax(dim=1) for name, logits in seg_logits_by_task.items()
                }

        if self.active_probe is not None:
            # Single-probe outputs, so that the standard testers/evaluators
            # (which read "seg_logits"/"pred"/"loss") work unchanged.
            return_dict["seg_logits"] = seg_logits_by_task[self.active_probe]
            if "loss" in return_dict:
                return_dict["loss"] = loss_by_task[self.active_probe]
            if "pred_by_task" in return_dict:
                return_dict["pred"] = return_dict["pred_by_task"][self.active_probe]
            elif not self.training:
                return_dict["pred"] = return_dict["seg_logits"].argmax(dim=1)

        return return_dict


@MODELS.register_module()
class GridProbeClassifier(GridProbeSegmentorV2):
    """N linear probe heads for tile/scene classification.

    Same probe grid and shared frozen backbone as GridProbeSegmentorV2, but the
    per-point (multi-scale) features are pooled per scene (``pooling``: mean or
    max) to [B, C] before the heads. Target key defaults to ``category``
    (e.g. PureForest).

    Use with GridProbeTrainer + GridProbeEvaluator (reads ``target_key``) and
    ``test=dict(type="ClsTester")`` through GridProbeWinnerSelector
    (``cls_logits`` is exposed once ``active_probe`` is set).

    Encoder multi-scale features, as for the segmentation probes:
      - LitePT / PT-v3 / KPConvX: ``enc_mode=True``
      - SpUNet: ``point_mode=True`` (its ``enc_mode`` scene-pools inside the
        backbone and gives only the last-stage features)
    """

    def __init__(
        self,
        probes,
        backbone_out_channels,
        num_classes,
        ignore_index=-1,
        backbone=None,
        target_key="category",
        freeze_backbone=True,
        feature_mask_values=None,
        drop_leading_channels=0,
        channel_blocks=None,
        pooling="mean",
    ):
        super().__init__(
            probes=probes,
            backbone_out_channels=backbone_out_channels,
            num_classes=num_classes,
            ignore_index=ignore_index,
            backbone=backbone,
            target_key=target_key,
            freeze_backbone=freeze_backbone,
            feature_mask_values=feature_mask_values,
            drop_leading_channels=drop_leading_channels,
            channel_blocks=channel_blocks,
        )
        pooling = str(pooling)
        if pooling not in ("mean", "max"):
            raise ValueError(
                f"GridProbeClassifier.pooling must be 'mean' or 'max', got {pooling!r}."
            )
        self.pooling = pooling

    def _pool_scene_feat(self, feat, offset):
        indptr = F.pad(offset, (1, 0))
        if self.pooling == "mean":
            return _safe_segment_csr_mean(feat, indptr)
        return torch_scatter.segment_csr(feat, indptr, reduce="max")

    def prepare_batch(self, input_dict):
        """As GridProbeSegmentorV2.prepare_batch, with scene pooling before the input norms."""
        self._fill_masked_feat_with_learned_value(input_dict)
        if self.freeze_backbone:
            with torch.no_grad():
                feat, point = self._forward_backbone(input_dict)
        else:
            feat, point = self._forward_backbone(input_dict)
        if self.drop_leading_channels:
            feat = feat[..., self.drop_leading_channels :]

        offset = input_dict["offset"]
        if isinstance(point, Point) and "offset" in point.keys():
            offset = point.offset
        n_scenes = int(offset.numel())
        n_points = int(offset[-1].item()) if n_scenes > 0 else 0
        if feat.shape[0] == n_scenes:
            # Already scene-pooled (e.g. SpUNet enc_mode=True).
            pass
        elif feat.shape[0] == n_points:
            feat = self._pool_scene_feat(feat, offset)
        else:
            raise ValueError(
                "Unexpected backbone feature shape "
                f"{tuple(feat.shape)} for offset with {n_scenes} scenes and "
                f"{n_points} points (pooling={self.pooling!r})."
            )

        active = self._active_probe_names()
        feat = _prepare_shared_feat(feat)
        feat_by_norm = _input_norm_cache(feat, self.heads, active, self.channel_blocks)
        return feat_by_norm, point, active

    def forward(self, input_dict, return_point=False):
        return_dict = super().forward(input_dict, return_point=return_point)
        # ClsTester and the classification logging read cls_logits(_by_task).
        if "seg_logits_by_task" in return_dict:
            return_dict["cls_logits_by_task"] = return_dict["seg_logits_by_task"]
        if "seg_logits" in return_dict:
            return_dict["cls_logits"] = return_dict["seg_logits"]
        return return_dict
