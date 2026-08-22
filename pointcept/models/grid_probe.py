"""
Grid-Search Linear Probing

Trains N independently-configured linear "probe" heads on top of a single
shared frozen-backbone forward pass, so a hyperparameter grid search over
loss/optimizer/scheduler/dropout/normalization only pays the backbone cost
once per batch instead of once per grid point.

Scope: semantic segmentation probes only, all probes read the same target
key (e.g. "segment"). Optimizer/scheduler heterogeneity across probes is
handled by GridProbeTrainer (pointcept/engines/train.py), not here — this
module only owns probe architecture + per-probe loss.
"""

from collections.abc import Mapping

import torch
import torch.nn as nn
from timm.layers import DropPath

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model
from .default import LearnedMaskedFeatMixin

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
    """x / ||x||_p along the last dim (clamped at `eps`).

    Under autocast, .norm() runs (and returns) in fp32 for numerical
    stability; dividing fp16 x by an fp32 norm silently promotes the whole
    result to fp32 instead of matching x's own dtype, hence the explicit
    cast back.
    """
    norm = x.norm(p=p, dim=-1, keepdim=True).clamp_min(eps)
    return x / norm.to(x.dtype)


def _unitsphere(feat, kind, eps=1e-6, channel_blocks=None):
    """Row-wise unitsphere rescale.

    None returns `feat` unchanged. "l1"/"l2"/"linf" divide the whole feature
    vector by its p-norm. "l1_block"/"l2_block"/"linf_block" instead split
    `feat` into `channel_blocks` contiguous chunks along the last dim (sizes
    in the same finest-stage-first order as _forward_backbone's concat, e.g.
    dec_channels + (bottleneck_channels,)) and unit-sphere-rescale each chunk
    independently before re-concatenating — normalizes each backbone
    stage/level's own sub-vector instead of the pooled hypercolumn as a
    whole. Shared by ProbeHead and GridProbeSegmentorV2's per-kind cache so N
    heads with the same input_norm do not each materialize a [P, C] copy.
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
        # Write each block's rescaled slice directly into a preallocated
        # output instead of torch.cat-ing a list of per-block tensors: cat
        # would need every per-block result alive at once *plus* its own
        # freshly-allocated [P, C] output (~2x feat's memory, transient) —
        # this keeps the extra cost at ~1x, matching the non-block path.
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
    """At most one unitsphere tensor per (input_norm, eps) among `names`."""
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
        "linf_block" (applied to backbone feat). The non-"_block" kinds
        unit-sphere-rescale the whole concatenated feature vector; the
        "_block" kinds instead rescale each channel_blocks chunk (e.g. each
        decoder stage / encoder level of a hypercolumn) independently —
        requires channel_blocks to be set (see below).
      - feat_norm: None | "batchnorm" | "layernorm"
      - dropout: float
      - criteria: list of loss cfgs (same shape as DefaultSegmentorV2.criteria)
    Optimizer/scheduler/grad_clip may also live in each probe dict; they are
    consumed by GridProbeTrainer, not this class.

    drop_leading_channels: model-level (not per-probe) — drops the first N
    channels of the shared backbone feat before every probe head sees it.
    For dec_traceable/enc_mode multiscale concat, the finest stage is always
    concatenated first (see _forward_backbone), so this ablates that stage
    out of every probe's input, e.g. drop_leading_channels=dec_channels[0].

    channel_blocks: model-level (not per-probe) — sizes of each concatenated
    stage/level in the shared backbone feat, in the same finest-first concat
    order as _forward_backbone (e.g. dec_channels + (bottleneck_channels,)
    for a decoder hypercolumn, or enc_channels for an encoder-multiscale
    concat), AFTER accounting for drop_leading_channels — i.e.
    sum(channel_blocks) must equal backbone_out_channels -
    drop_leading_channels. Only required when at least one probe uses an
    input_norm ending in "_block"; ignored otherwise.

    Unlike DefaultSegmentorV2/MultiTaskSegmentorV2 (whose freeze_backbone
    only sets requires_grad=False and otherwise lets the backbone follow the
    trainer's normal train()/eval() propagation), freeze_backbone=True here
    also wraps the backbone forward in torch.no_grad() (the VRAM/compute
    saving) and, by default, pins the backbone's BatchNorm and DropPath
    submodules to eval-mode *behavior* (nothing to do with requires_grad)
    even while mode=True (i.e. during training) — re-applied on every
    .train() call.

    This eval-pinning turned out to matter a lot for backbones with
    BatchNorm (e.g. LitePT-v1): DefaultSegmentorV2 never forces eval, so its
    "frozen" backbone still lets BatchNorm running_mean/var drift toward the
    downstream dataset on every training step (a free, gradient-free domain
    recalibration — see project memory / README discussion), and DropPath
    stays stochastically active. `bn_eval_mode` / `drop_path_eval_mode` let
    each of those two effects be toggled independently of the other, for
    ablating which one actually drives a given backbone's linear-probe
    quality gap against DefaultSegmentorV2 (both default True = original,
    fully-eval-pinned behavior, unchanged from before these flags existed).
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
        bn_eval_mode=True,
        drop_path_eval_mode=True,
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
        self.active_probe = None  # None => all probes computed; set by winner-selection.

        # For decoder-hypercolumn/encoder-multiscale backbones, the concat
        # order in _forward_backbone always puts the finest stage first
        # (see that method's docstring), so dropping the first N channels
        # drops that stage from every probe's input — e.g. set this to the
        # finest decoder stage's channel count to ablate it out of the
        # hypercolumn without touching the backbone.
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
        self.bn_eval_mode = bn_eval_mode
        self.drop_path_eval_mode = drop_path_eval_mode
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            # Learned masked-feat fill values (e.g. color_mask_value) are part of
            # how raw features are prepared for the backbone.
            if self.enable_learned_masked_feat:
                for feat_key in self.learned_masked_feat_keys:
                    mask_value = getattr(self, f"{feat_key}_mask_value", None)
                    if mask_value is not None:
                        mask_value.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            # nn.Module.train() just recursively flipped every backbone
            # submodule (BatchNorm, DropPath, LayerNorm, ...) to `mode`. Put
            # back to eval() only the pieces this instance is configured to
            # pin — independently, so BN running-stat drift and DropPath's
            # stochastic depth can each be toggled on/off on its own (see
            # class docstring). Everything else (LayerNorm, Linear, ...) is
            # train/eval-invariant so leaving it at `mode` is harmless.
            if self.bn_eval_mode or self.drop_path_eval_mode:
                for m in self.backbone.modules():
                    if self.bn_eval_mode and isinstance(
                        m, nn.modules.batchnorm._BatchNorm
                    ):
                        m.eval()
                    if self.drop_path_eval_mode and isinstance(m, DropPath):
                        m.eval()
        return self

    def backbone_parameters(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def probe_head_parameters(self, name):
        return [p for p in self.heads[name].parameters() if p.requires_grad]

    def _active_probe_names(self):
        if self.active_probe is not None:
            return (self.active_probe,)
        return self.probe_names

    def _forward_backbone(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        if isinstance(point, Point):
            # Decoder-side multiscale concat: only populated when the backbone's
            # decoder was built with dec_traceable=True (e.g. LitePT's
            # `dec_traceable`). See DefaultSegmentorV2.forward for the same
            # pattern/rationale. No-op for any backbone that never sets
            # dec_traceable=True.
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                child, parent = point_list[i], point_list[i - 1]
                assert "pooling_inverse" in child.keys()
                parent.feat = torch.cat([parent.feat, child.feat[child.pooling_inverse]], dim=-1)
            point = point_list[0]
            # Encoder-side multiscale concat (enc_mode=True backbones).
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
        """Shared backbone feat + per-kind unitsphere cache for the active heads."""
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
            # Backbone-internal-only state (serialized order/inverse, per-stage
            # pad/unpad/cu_seqlens, sparse_conv_feat, ...) -- same reasoning as
            # GridProbeTrainer.run_step: nothing below reads `point`, but the
            # reference would otherwise sit in this frame's locals (kept alive
            # by Python until forward() returns) for every active probe below,
            # e.g. GridProbeEvaluator's per-batch self.trainer.model(input_dict).
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
            # Alias the shared target under each active probe's own name so
            # InformationWriter's existing per-task train-mIoU accumulation
            # (which reads input_dict[task_name]) picks these probes up for
            # free via cfg.data.task_configs — all probes share one target,
            # this just gives each one a name-matching view of it. Eval/test
            # paths don't need this (GridProbeEvaluator reads target_key directly).
            for name in active:
                input_dict.setdefault(name, target)
            with torch.no_grad():
                return_dict["pred_by_task"] = {
                    name: logits.argmax(dim=1) for name, logits in seg_logits_by_task.items()
                }

        if self.active_probe is not None:
            # Single-probe compatibility surface: lets SemSegTester / SemSegEvaluator
            # (which only know about "seg_logits"/"pred"/"loss") run unmodified.
            return_dict["seg_logits"] = seg_logits_by_task[self.active_probe]
            if "loss" in return_dict:
                return_dict["loss"] = loss_by_task[self.active_probe]
            if "pred_by_task" in return_dict:
                return_dict["pred"] = return_dict["pred_by_task"][self.active_probe]
            elif not self.training:
                return_dict["pred"] = return_dict["seg_logits"].argmax(dim=1)

        return return_dict
