"""
Generic task wrappers around a backbone.

- DefaultSegmentor: the backbone owns its output classifier (e.g. SpUNet, KPConvX).
- DefaultSegmentorV2: backbone + linear point-wise ``seg_head``.
- MultiTaskSegmentorV2: one shared backbone and one head per MALiBU3D task
  (point-wise semantics, tile-level class distributions, 2D raster masks, regression).
- DefaultClassifier: backbone + scene pooling + linear ``cls_head`` (e.g. PureForest).

All of them accept ``freeze_backbone`` (see FrozenBackboneEvalMixin) and
``feature_mask_values`` (see LearnedMaskedFeatMixin).
"""

import torch
import torch.nn as nn
import torch_scatter
from collections.abc import Mapping
from pointcept.utils.logger import get_root_logger
from pointcept.models.losses import build_criteria
from pointcept.utils.misc import pool_axis_distribution_from_probs
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from .builder import MODELS, build_model

logger = get_root_logger()

_POOLING_MODES = frozenset(("mean", "max", "attention"))


def _safe_segment_csr_mean(feat, indptr):
    """Per-scene mean pool in float32 (AMP-safe); empty segments -> 0."""
    pooled = torch_scatter.segment_csr(feat.float(), indptr, reduce="mean")
    return torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)


class FrozenBackboneEvalMixin:
    """Keeps a frozen backbone in eval() mode, whatever train()/eval() is called.

    ``requires_grad=False`` (or ``torch.no_grad()``) only stops gradient updates
    of the *parameters*. BatchNorm running statistics are buffers: they keep
    updating in the forward pass whenever the module is in train mode, and
    DropPath / Dropout stay stochastic. A one-off ``backbone.eval()`` in
    ``__init__`` is not enough either, since the trainer calls ``model.train()``
    afterwards, which recursively flips the backbone back to train mode.

    Hence, for classes using this mixin, ``freeze_backbone=True`` means
    ``requires_grad=False`` *and* a backbone that is re-pinned to eval() on
    every ``.train()`` call. There is deliberately no opt-out flag.

    The host class must define ``self.backbone`` and ``self.freeze_backbone``.
    The mixin must come before ``nn.Module`` in the bases (so that its
    ``train`` overrides ``nn.Module.train``).
    """

    def _pin_frozen_backbone_eval(self):
        backbone = getattr(self, "backbone", None)
        if getattr(self, "freeze_backbone", False) and backbone is not None:
            backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self._pin_frozen_backbone_eval()
        return self


class LearnedMaskedFeatMixin:
    """Replaces missing input features by a learned per-modality value.

    Some points have no valid value for a feature modality (e.g. no RGB, or a
    color dropped by ``RandomDropColor(keep_mask=True)``). ``Collect`` then
    provides a boolean ``{key}_mask`` (True = missing) and the slice
    ``[{key}_feat_start, {key}_feat_end)`` of that modality inside ``feat``.
    When enabled, masked rows of that slice are overwritten by a learnable
    vector ``{key}_mask_value`` before the backbone forward, instead of the
    constant fill value written by the data pipeline.

    ``feature_mask_values``: ``dict(enable=bool, masked_feat_keys=(...))``;
    supported keys are ``color`` (3 ch), ``normal`` (3 ch) and ``strength`` (1 ch).
    """

    def _init_learned_masked_feat(self, feature_mask_values=None):
        cfg = feature_mask_values or {}
        self.enable_learned_masked_feat = bool(cfg.get("enable", False))
        self.learned_masked_feat_keys = tuple(
            cfg.get("masked_feat_keys", ("color", "normal", "strength"))
        )

        if not self.enable_learned_masked_feat:
            return

        if "color" in self.learned_masked_feat_keys:
            self.color_mask_value = nn.Parameter(torch.zeros(1, 3))
        if "normal" in self.learned_masked_feat_keys:
            self.normal_mask_value = nn.Parameter(torch.zeros(1, 3))
        if "strength" in self.learned_masked_feat_keys:
            self.strength_mask_value = nn.Parameter(torch.zeros(1, 1))

    def _fill_masked_feat_with_learned_value(self, input_dict):
        if not self.enable_learned_masked_feat:
            return
        assert "feat" in input_dict, "'feat' is required in input_dict."
        expected_dims = {"color": 3, "normal": 3, "strength": 1}
        feat = input_dict["feat"]
        for feat_key in self.learned_masked_feat_keys:
            mask_key = f"{feat_key}_mask"

            if mask_key not in input_dict:
                continue

            # Modalities are already concatenated in "feat"; Collect records
            # where each one starts and ends.
            start_key = f"{feat_key}_feat_start"
            end_key = f"{feat_key}_feat_end"

            start_value = input_dict[start_key]
            end_value = input_dict[end_key]
            start = (
                int(start_value[0].item())
                if torch.is_tensor(start_value)
                else int(start_value)
            )
            end = (
                int(end_value[0].item())
                if torch.is_tensor(end_value)
                else int(end_value)
            )
            feat_dim = end - start
            if feat_dim <= 0:
                continue

            expected_dim = expected_dims.get(feat_key, feat_dim)
            assert (
                feat_dim == expected_dim
            ), f"Unexpected {feat_key} feature dim: expected {expected_dim}, got {feat_dim}."

            mask = input_dict[mask_key].bool().unsqueeze(-1)
            assert hasattr(
                self, f"{feat_key}_mask_value"
            ), f"Missing learned parameter '{feat_key}_mask_value'."
            learned_mask_value = getattr(self, f"{feat_key}_mask_value").to(feat.dtype)
            filled = torch.where(
                mask,
                learned_mask_value,
                feat[:, start:end],
            )
            # When the batch mask is all-False, torch.where gives no grad to the
            # learned value; add a zero term so DDP still marks it as used.
            if self.training:
                filled = filled + learned_mask_value.sum() * 0.0
            feat[:, start:end] = filled

    def _print_learned_masked_feat(self):
        """Print the learned fill values (called by the trainer at the end of training)."""
        if not self.enable_learned_masked_feat:
            print("Learned masked feat is not enabled.")
            return
        print(f"Learned masked feat keys: {self.learned_masked_feat_keys}")
        for feat_key in self.learned_masked_feat_keys:
            if hasattr(self, f"{feat_key}_mask_value"):
                value = (
                    getattr(self, f"{feat_key}_mask_value")
                    .detach()
                    .cpu()
                    .flatten()
                    .tolist()
                )
                print(f"{feat_key.capitalize()} mask value: " f"{value}")


@MODELS.register_module()
class DefaultSegmentor(FrozenBackboneEvalMixin, nn.Module, LearnedMaskedFeatMixin):
    """Segmentor whose backbone owns the output classifier (the backbone returns logits).

    freeze_backbone=True means ALL of the following (there is no separate flag):
      * every backbone parameter has requires_grad=False (no gradient, never
        updated by the optimizer);
      * the backbone is kept in eval() mode *permanently*: every .train() call
        (the trainer issues one per epoch) re-pins it. BatchNorm therefore
        uses its stored running statistics and does NOT update them, and
        DropPath / Dropout are inactive. The backbone is a fixed, deterministic
        feature extractor; only the heads are in train mode.
      * exception: parameters under ``final.`` (the backbone-owned output
        classifier of e.g. SpUNet / KPConvX) stay trainable; the module is
        nevertheless in eval(), which is harmless for a linear classifier.
    """

    def __init__(
        self,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        feature_mask_values=None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self._init_learned_masked_feat(feature_mask_values=feature_mask_values)
        if self.freeze_backbone:
            # The backbone-owned output layer ("final" in SpUNet and KPConvX)
            # stays trainable.
            keep_segmentation_head_prefixes = "final."
            for name, param in self.backbone.named_parameters():
                if name.startswith(keep_segmentation_head_prefixes):
                    continue
                param.requires_grad = False
            self._pin_frozen_backbone_eval()

    def forward(self, input_dict):
        self._fill_masked_feat_with_learned_value(input_dict)
        if "condition" in input_dict.keys():
            # Dataset condition for PDNorm (PPT, https://arxiv.org/abs/2308.09718);
            # only one condition per batch is supported.
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict = dict(loss=loss)
            with torch.no_grad():
                # Used by the hooks to accumulate the train confusion matrix
                # (train mIoU); the writer only logs scalars.
                return_dict["pred"] = seg_logits.argmax(dim=1)
            return return_dict
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(FrozenBackboneEvalMixin, nn.Module, LearnedMaskedFeatMixin):
    """Backbone + linear ``seg_head`` point-wise segmentor.

    freeze_backbone=True means ALL of the following (there is no separate flag):
      * every backbone parameter has requires_grad=False (no gradient, never
        updated by the optimizer);
      * the backbone is kept in eval() mode *permanently*: every .train() call
        (the trainer issues one per epoch) re-pins it. BatchNorm therefore
        uses its stored running statistics and does NOT update them, and
        DropPath / Dropout are inactive. The backbone is a fixed, deterministic
        feature extractor; only the heads are in train mode.
      * learned masked-feat fill values (feature_mask_values) are frozen too.
    """

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        feature_mask_values=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self._init_learned_masked_feat(feature_mask_values=feature_mask_values)
        self._ignore_index = None
        # Cache ignore_index at init to keep runtime deterministic.
        self.get_ignore_index()
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Learned masked-feat fill values (e.g. color_mask_value) belong to
            # the backbone input, not to the head: freeze them too so a linear
            # probe sees the inputs the backbone was trained with.
            if self.enable_learned_masked_feat:
                for feat_key in self.learned_masked_feat_keys:
                    mask_value = getattr(self, f"{feat_key}_mask_value", None)
                    if mask_value is not None:
                        mask_value.requires_grad = False
            self._pin_frozen_backbone_eval()

    def get_ignore_index(self) -> int:
        """ignore_index of the first criterion that defines one (-1 otherwise)."""
        if self._ignore_index is not None:
            return self._ignore_index
        for c in getattr(self.criteria, "criteria", []):
            if hasattr(c, "ignore_index"):
                self._ignore_index = int(getattr(c, "ignore_index"))
                return self._ignore_index

        logger.warning("No ignore_index found in criteria, using -1 as default.")
        self._ignore_index = -1
        return self._ignore_index

    def forward(self, input_dict, return_point=False):
        self._fill_masked_feat_with_learned_value(input_dict)
        point = Point(input_dict)
        point = self.backbone(point)
        # Point-based backbones (PT-v3, LitePT, SpUNet in point modes) return a
        # Point; the others (SpUNet, KPConvX) return a per-point feature tensor.
        if isinstance(point, Point):
            # Decoder hypercolumn, only present when the decoder is traceable
            # (LitePT dec_traceable=True, PT-v3m2 / PT-v3-malibu traceable=True,
            # SpUNet dec_point_mode=True): follow the `unpooling_parent` links
            # from the finest decoder output to the encoder bottleneck, then
            # concatenate every level's output gathered onto the finest points,
            # finest first. No-op for a non-traceable decoder.
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                child, parent = point_list[i], point_list[i - 1]
                assert "pooling_inverse" in child.keys()
                parent.feat = torch.cat(
                    [parent.feat, child.feat[child.pooling_inverse]], dim=-1
                )
            point = point_list[0]
            # Encoder multi-scale features (enc_mode=True, or SpUNet
            # point_mode=True): the backbone returns the coarsest point; follow
            # the `pooling_parent` links down to the finest level, concatenating
            # each coarser feature gathered onto the finer points (finest first).
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # Expose the output Point (feat, coord, ...), e.g. for feature extraction.
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            with torch.no_grad():
                # Used by the hooks to accumulate the train confusion matrix
                # (train mIoU); the writer only logs scalars.
                return_dict["pred"] = seg_logits.argmax(dim=1)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class MultiTaskSegmentorV2(FrozenBackboneEvalMixin, nn.Module, LearnedMaskedFeatMixin):
    """Multi-task model: one shared backbone, one linear head per task (MALiBU3D).

    The multi-task counterpart of DefaultSegmentorV2. ``task_configs`` maps each
    task name (also its target key in ``input_dict``) to a config whose
    ``task_type`` is one of:

    - ``semantic``: point-wise classification (e.g. ``segment``); logits in
      ``seg_logits_by_task``.
    - ``tile_distribution``: point-wise logits (``seg_logits_by_task``) trained
      at tile level: per-tile mean predicted probabilities vs. the tile's
      empirical label distribution, with the task criteria (WeightedKLDivLoss).
      With ``pointwise_supervision=True`` the criteria are applied point-wise
      instead (e.g. natural-habitat axes).
    - ``pixel_semantic``: 2D raster masks (e.g. ``network`` roads/railroads,
      ``forest_2d``): point features are pooled into the 1 m raster cells and
      one head predicts ``num_networks`` channels of ``num_classes`` logits.
      Needs the ``{task}_cell`` / ``{task}_pix`` keys written by
      ``NetworkRasterToPointLabels``.
    - ``regression``: point-wise scalar (e.g. ``elevation``); predictions in
      ``reg_pred_by_task``.

    ``main_task`` (default: first task; semantic, tile_distribution or
    pixel_semantic) provides ``seg_logits`` / ``pred``, which the single-task
    hooks use (e.g. checkpoint selection on its mIoU). ``task_weights`` scale
    each task loss in the total ``loss``; ``task_criteria`` gives per-task loss
    configs (``criteria`` is the fallback).

    Outputs: ``seg_logits``, ``seg_logits_by_task``, ``reg_pred_by_task``,
    ``pixel_seg_{logits,target}_by_task`` (per cell) and
    ``pixel_seg_{logits,target}_dense_by_task`` (per-scene rasters, see
    ``_compute_pixel_logits``); ``loss`` and ``loss_by_task`` when any target is
    in the batch; ``pred`` / ``pred_by_task`` in training.

    freeze_backbone=True means ALL of the following (there is no separate flag):
      * every backbone parameter has requires_grad=False (no gradient, never
        updated by the optimizer);
      * the backbone is kept in eval() mode *permanently*: every .train() call
        (the trainer issues one per epoch) re-pins it. BatchNorm therefore
        uses its stored running statistics and does NOT update them, and
        DropPath / Dropout are inactive. The backbone is a fixed, deterministic
        feature extractor; only the heads are in train mode.
    """

    _TASK_TYPES = frozenset(
        (
            "semantic",
            "regression",
            "pixel_semantic",
            "tile_distribution",
        )
    )

    def __init__(
        self,
        task_configs,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        task_criteria=None,
        task_weights=None,
        main_task=None,
        freeze_backbone=False,
        feature_mask_values=None,
    ):
        super().__init__()
        if not isinstance(task_configs, Mapping) or len(task_configs) == 0:
            raise ValueError(
                "task_configs must be a non-empty mapping of task_name -> config."
            )
        self.task_configs = {str(k): dict(v) for k, v in task_configs.items()}
        self.tasks = tuple(self.task_configs.keys())
        self.main_task = str(main_task or self.tasks[0])
        if self.main_task not in self.task_configs:
            raise ValueError("main_task must be one of task_configs keys.")
        main_type = self._task_type(self.task_configs[self.main_task])
        if main_type not in ("semantic", "pixel_semantic", "tile_distribution"):
            raise ValueError(
                "main_task must be a semantic, pixel_semantic, or tile_distribution "
                f"task (got task_type={main_type!r})."
            )

        self.backbone = build_model(backbone)
        self.seg_heads = nn.ModuleDict()
        self.pixel_seg_heads = nn.ModuleDict()
        self.reg_heads = nn.ModuleDict()
        self.criteria_by_task = {}
        self.task_weights = {}
        default_weight = 1.0
        task_weights = task_weights or {}

        for task_name, task_config in self.task_configs.items():
            task_type = self._task_type(task_config)
            task_criteria_cfg = None
            if isinstance(task_criteria, Mapping) and task_name in task_criteria:
                task_criteria_cfg = task_criteria[task_name]
            elif criteria is not None:
                task_criteria_cfg = criteria
            self.criteria_by_task[task_name] = build_criteria(task_criteria_cfg)
            self.task_weights[task_name] = float(
                task_weights.get(task_name, default_weight)
            )

            if task_type in ("semantic", "tile_distribution"):
                num_classes = int(task_config["num_classes"])
                if num_classes <= 0:
                    raise ValueError(
                        f"num_classes must be > 0 for semantic task '{task_name}'."
                    )
                self.seg_heads[task_name] = nn.Linear(
                    backbone_out_channels, num_classes
                )
            elif task_type == "pixel_semantic":
                num_classes = int(task_config["num_classes"])
                num_networks = int(task_config.get("num_networks", 1))
                if num_classes <= 0 or num_networks <= 0:
                    raise ValueError(
                        f"num_classes and num_networks must be > 0 for "
                        f"pixel_semantic task '{task_name}'."
                    )
                # One Linear(C -> r * num_classes), viewed as (P, r, num_classes)
                # in forward (r = num_networks channels).
                self.pixel_seg_heads[task_name] = nn.Linear(
                    backbone_out_channels, num_classes * num_networks
                )
            elif task_type == "regression":
                self.reg_heads[task_name] = nn.Linear(backbone_out_channels, 1)

        self._init_learned_masked_feat(feature_mask_values=feature_mask_values)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self._pin_frozen_backbone_eval()

    # The three accessors below are used by the GradNorm loss balancing and the
    # per-task gradient-norm logging (pointcept/utils/gradient_norm.py).
    def backbone_parameters(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def last_backbone_layer_parameters(self):
        from pointcept.utils.gradient_norm import resolve_last_backbone_layer_params

        return resolve_last_backbone_layer_params(self.backbone)

    def task_head_parameters(self, task_name):
        params = []
        if task_name in self.seg_heads:
            params.extend(self.seg_heads[task_name].parameters())
        if task_name in self.pixel_seg_heads:
            params.extend(self.pixel_seg_heads[task_name].parameters())
        if task_name in self.reg_heads:
            params.extend(self.reg_heads[task_name].parameters())
        return [p for p in params if p.requires_grad]

    @classmethod
    def _task_type(cls, task_config):
        tt = task_config.get("task_type")
        if tt not in cls._TASK_TYPES:
            raise ValueError(
                "Each task_configs entry must set task_type to 'semantic', "
                "'pixel_semantic', 'regression' or 'tile_distribution' "
                f"(got {tt!r})."
            )
        return tt

    @staticmethod
    def _as_scalar_tensor(value, device, dtype=torch.float32):
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype).reshape(-1)
        return torch.tensor([float(value)], device=device, dtype=dtype)

    def _pixel_pool_and_gather(
        self,
        feat,
        cell,
        pix,
        offset,
        point_labels,
        num_networks,
        height=None,
        width=None,
        ignore_index=2,
        pooling="max",
    ):
        """Pool point features into raster cells (pixels) and gather each cell's label.

        ``point_labels`` ``(N, r)``: per-point raster labels written by
        ``NetworkRasterToPointLabels``. ``cell`` ``(N, 2)``: absolute Lambert
        ``(iy, ix)``, used as pooling keys (unique across scenes mixed by Mix3D).
        ``pix`` ``(N, 2)``: ``(iy, ix)`` relative to the scene's GT raster, used to
        scatter predictions back onto it. A cell takes the label of its first point.
        ``pooling``: ``"max"`` suits thin structures where one strong point
        response is enough (``network``); ``"mean"`` suits area-coverage classes
        (``forest_2d``).
        The dense-raster meta is only filled when one ``(height, width)`` is
        available per scene, i.e. not when Mix3D merged scenes.

        Returns
        -------
        pooled_feat : (P, C)
        targets : (P, r) long
        meta_parts : list of None or (n_pix, iy_u, ix_u, height, width, in_grid)
        """
        if pooling not in ("max", "mean"):
            raise ValueError(f"pooling must be 'max' or 'mean', got {pooling!r}")
        device = feat.device
        if not torch.is_tensor(point_labels):
            point_labels = torch.as_tensor(point_labels, device=device)
        else:
            point_labels = point_labels.to(device=device)
        if point_labels.ndim != 2 or point_labels.shape[0] != feat.shape[0]:
            raise ValueError(
                f"point-wise network labels expected (N={feat.shape[0]}, r), "
                f"got {tuple(point_labels.shape)}"
            )
        if point_labels.shape[1] != num_networks:
            raise ValueError(
                f"network labels expected r={num_networks} channels, "
                f"got {point_labels.shape[1]}"
            )
        if not torch.is_tensor(cell):
            cell = torch.as_tensor(cell, device=device)
        else:
            cell = cell.to(device=device)
        if not torch.is_tensor(pix):
            pix = torch.as_tensor(pix, device=device)
        else:
            pix = pix.to(device=device)
        if cell.ndim != 2 or cell.shape[0] != feat.shape[0] or cell.shape[1] < 2:
            raise ValueError(
                f"cell expected (N={feat.shape[0]}, 2), got {tuple(cell.shape)}"
            )
        if pix.ndim != 2 or pix.shape[0] != feat.shape[0] or pix.shape[1] < 2:
            raise ValueError(
                f"pix expected (N={feat.shape[0]}, 2), got {tuple(pix.shape)}"
            )

        heights = widths = None
        can_dense = False
        if height is not None and width is not None:
            heights = self._as_scalar_tensor(height, device, dtype=torch.float32)
            widths = self._as_scalar_tensor(width, device, dtype=torch.float32)
            if heights.numel() == 1 and offset.numel() > 1:
                heights = heights.expand(offset.numel())
                widths = widths.expand(offset.numel())
            can_dense = (
                heights.numel() == offset.numel() and widths.numel() == offset.numel()
            )

        batch_idx = offset2batch(offset)
        pooled_parts = []
        target_parts = []
        meta_parts = []
        n_scenes = int(offset.numel())
        for b in range(n_scenes):
            point_mask = batch_idx == b
            if not point_mask.any():
                meta_parts.append(None)
                continue
            cells_b = cell[point_mask][:, :2].long()
            pix_b = pix[point_mask][:, :2].long()
            f = feat[point_mask]
            labels_pt = point_labels[point_mask]

            _, inv = torch.unique(cells_b, dim=0, return_inverse=True)
            if pooling == "mean":
                pooled = torch_scatter.scatter_mean(f, inv, dim=0)
            else:
                pooled = torch_scatter.scatter_max(f, inv, dim=0)[0]
            idx = torch.arange(inv.numel(), device=device)
            first = torch_scatter.scatter_min(idx, inv, dim=0)[0]
            labels = labels_pt[first].long()

            pooled_parts.append(pooled)
            target_parts.append(labels)

            n_pix = int(pooled.shape[0])
            if can_dense:
                h = int(heights[b].item())
                w = int(widths[b].item())
                iy_u = pix_b[first, 0]
                ix_u = pix_b[first, 1]
                in_grid = (ix_u >= 0) & (iy_u >= 0) & (ix_u < w) & (iy_u < h)
                meta_parts.append((n_pix, iy_u, ix_u, h, w, in_grid))
            else:
                meta_parts.append((n_pix, None, None, None, None, None))

        if not pooled_parts:
            c = feat.shape[1]
            return (
                feat.new_zeros((0, c)),
                feat.new_zeros((0, num_networks), dtype=torch.long),
                meta_parts,
            )
        return (
            torch.cat(pooled_parts, dim=0),
            torch.cat(target_parts, dim=0),
            meta_parts,
        )

    def _compute_pixel_logits(self, feat, input_dict):
        """Per-cell logits ``(P, r, num_classes)`` and targets ``(P, r)`` of each
        pixel_semantic task present in the batch, plus per-scene dense rasters
        (``None`` when they cannot be built)."""
        logits_by_task = {}
        targets_by_task = {}
        dense_by_task = {}
        dense_target_by_task = {}
        for task_name, head in self.pixel_seg_heads.items():
            task_config = self.task_configs[task_name]
            num_networks = int(task_config["num_networks"])
            num_classes = int(task_config["num_classes"])
            if task_name not in input_dict:
                continue
            cell_key = f"{task_name}_cell"
            pix_key = f"{task_name}_pix"
            if cell_key not in input_dict or pix_key not in input_dict:
                raise KeyError(
                    f"pixel_semantic task '{task_name}' requires '{cell_key}' and "
                    f"'{pix_key}' in input_dict (add "
                    f"NetworkRasterToPointLabels(target_key='{task_name}'))."
                )
            pooled, targets, meta_parts = self._pixel_pool_and_gather(
                feat,
                input_dict[cell_key],
                input_dict[pix_key],
                input_dict["offset"],
                input_dict[task_name],
                num_networks,
                height=input_dict.get(f"{task_name}_height"),
                width=input_dict.get(f"{task_name}_width"),
                ignore_index=int(task_config.get("ignore_index", 2)),
                pooling=str(task_config.get("pooling", "max")),
            )
            raw = head(pooled)  # (P, r * C)
            logits = raw.view(-1, num_networks, num_classes)
            logits_by_task[task_name] = logits
            targets_by_task[task_name] = targets

            # Dense foreground probabilities (r, H, W) on the scene's GT raster:
            # softmax class-1 probability for observed pixels (at least one point
            # in the 1 m cell), NaN for unobserved ones. Not built under Mix3D.
            # dense_target is the GT counterpart: the cell label for observed
            # pixels (possibly ignore_index), -1 for unobserved ones (ignore_index
            # is a valid observed value, so it cannot also mean "unobserved").
            dense_list = []
            dense_target_list = []
            cursor = 0
            for meta in meta_parts:
                if meta is None:
                    dense_list.append(None)
                    dense_target_list.append(None)
                    continue
                n_pix, iy_u, ix_u, height, width, in_grid = meta
                part = logits[cursor : cursor + n_pix]
                target_part = targets[cursor : cursor + n_pix]
                cursor += n_pix
                if in_grid is None or height is None:
                    dense_list.append(None)
                    dense_target_list.append(None)
                    continue
                # float32: under AMP, softmax promotes to fp32 while part may be Half.
                dense = torch.full(
                    (num_networks, int(height), int(width)),
                    float("nan"),
                    device=part.device,
                    dtype=torch.float32,
                )
                dense_target = torch.full(
                    (num_networks, int(height), int(width)),
                    -1,
                    device=target_part.device,
                    dtype=torch.long,
                )
                if in_grid.any():
                    probs_fg = torch.softmax(part[in_grid].float(), dim=-1)[..., 1]
                    dense[:, iy_u[in_grid], ix_u[in_grid]] = probs_fg.transpose(0, 1)
                    dense_target[:, iy_u[in_grid], ix_u[in_grid]] = target_part[
                        in_grid
                    ].transpose(0, 1)
                dense_list.append(dense)
                dense_target_list.append(dense_target)
            dense_by_task[task_name] = dense_list
            dense_target_by_task[task_name] = dense_target_list
        return logits_by_task, targets_by_task, dense_by_task, dense_target_by_task

    def _forward_backbone(self, input_dict):
        """Per-point backbone features; encoder multi-scale features (enc_mode)
        are concatenated as in DefaultSegmentorV2 (no decoder hypercolumn)."""
        point = Point(input_dict)
        point = self.backbone(point)
        if isinstance(point, Point):
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

    def _compute_logits(self, feat):
        return {task_name: head(feat) for task_name, head in self.seg_heads.items()}

    def _compute_reg_preds(self, feat):
        return {
            task_name: head(feat).squeeze(-1)
            for task_name, head in self.reg_heads.items()
        }

    def _has_any_target(self, input_dict):
        for task_name in self.tasks:
            if task_name in input_dict:
                return True
        return False

    def _compute_loss(
        self,
        logits_by_task,
        reg_pred_by_task,
        input_dict,
        pixel_logits_by_task=None,
        pixel_targets_by_task=None,
    ):
        """Weighted sum of the losses of the tasks whose target is in the batch.

        Returns ``(total_loss, loss_by_task)`` (unweighted per-task losses); raises
        if no task target is present.
        """
        total_loss = None
        loss_by_task = {}
        pixel_logits_by_task = pixel_logits_by_task or {}
        pixel_targets_by_task = pixel_targets_by_task or {}
        for task_name in self.tasks:
            task_config = self.task_configs[task_name]
            tt = self._task_type(task_config)
            w = self.task_weights[task_name]
            if tt == "semantic":
                if task_name not in logits_by_task:
                    continue
                if task_name not in input_dict:
                    continue
                logits = logits_by_task[task_name]
                task_loss = self.criteria_by_task[task_name](
                    logits, input_dict[task_name]
                )
            elif tt == "tile_distribution":
                if task_name not in logits_by_task:
                    continue
                if task_name not in input_dict:
                    continue
                logits = logits_by_task[task_name]
                if task_config.get("pointwise_supervision"):
                    task_loss = self.criteria_by_task[task_name](
                        logits, input_dict[task_name]
                    )
                else:
                    # Tile level: per-tile mean predicted distribution vs. the
                    # tile's label histogram, weighted by its labelled-point count.
                    target = input_dict[task_name].reshape(-1).long()
                    num_classes = int(task_config["num_classes"])
                    ignore_index = int(task_config["ignore_index"])
                    offset = input_dict["offset"]
                    indptr = nn.functional.pad(offset, (1, 0))
                    probs = torch.softmax(logits.float(), dim=-1)
                    pi_hat, q_t, n_t = pool_axis_distribution_from_probs(
                        probs, target, indptr, ignore_index, num_classes
                    )
                    keep = n_t > 0
                    if not keep.any():
                        # No labelled point in the batch: keep the logits in the
                        # graph so DDP still sees a (zero) gradient for this head.
                        task_loss = logits.sum() * 0.0
                    else:
                        task_loss = self.criteria_by_task[task_name](
                            pi_hat[keep], q_t[keep], weight=n_t[keep]
                        )
            elif tt == "pixel_semantic":
                if task_name not in pixel_logits_by_task:
                    continue
                logits = pixel_logits_by_task[task_name]  # (P, r, C)
                targets = pixel_targets_by_task[task_name]  # (P, r)
                num_networks = int(task_config["num_networks"])
                channel_losses = []
                for c in range(num_networks):
                    channel_losses.append(
                        self.criteria_by_task[task_name](logits[:, c, :], targets[:, c])
                    )
                task_loss = sum(channel_losses) / float(num_networks)
            else:
                if task_name not in reg_pred_by_task:
                    continue
                if task_name not in input_dict:
                    continue
                pred = reg_pred_by_task[task_name].reshape(-1)
                target = input_dict[task_name].float().reshape(-1)
                task_loss = self.criteria_by_task[task_name](pred, target)
            loss_by_task[task_name] = task_loss
            weighted = task_loss * w
            total_loss = weighted if total_loss is None else total_loss + weighted
        if total_loss is None:
            raise KeyError(
                "No multitask targets found in input_dict for tasks: "
                f"{', '.join(self.tasks)}."
            )
        return total_loss, loss_by_task

    def forward(self, input_dict, return_point=False):
        self._fill_masked_feat_with_learned_value(input_dict)
        feat, point = self._forward_backbone(input_dict)
        logits_by_task = self._compute_logits(feat)
        (
            pixel_logits_by_task,
            pixel_targets_by_task,
            pixel_dense_by_task,
            pixel_dense_target_by_task,
        ) = self._compute_pixel_logits(feat, input_dict)
        reg_pred_by_task = self._compute_reg_preds(feat)

        if self.main_task in logits_by_task:
            main_logits = logits_by_task[self.main_task]
        elif self.main_task in pixel_logits_by_task:
            # pixel_semantic main task: expose its first channel (per cell).
            main_logits = pixel_logits_by_task[self.main_task][:, 0, :]
        else:
            # main_task absent from this batch: fall back to another task's
            # logits so that seg_logits always exists.
            main_logits = next(iter(logits_by_task.values()), None)
            if main_logits is None and pixel_logits_by_task:
                main_logits = next(iter(pixel_logits_by_task.values()))[:, 0, :]
            if main_logits is None:
                main_logits = feat.new_zeros((feat.shape[0], 1))

        return_dict = {
            "seg_logits": main_logits,
            "seg_logits_by_task": logits_by_task,
            "pixel_seg_logits_by_task": pixel_logits_by_task,
            "pixel_seg_target_by_task": pixel_targets_by_task,
            "pixel_seg_logits_dense_by_task": pixel_dense_by_task,
            "pixel_seg_target_dense_by_task": pixel_dense_target_by_task,
            "reg_pred_by_task": reg_pred_by_task,
        }
        if return_point:
            return_dict["point"] = point

        if self.training or self._has_any_target(input_dict):
            total_loss, loss_by_task = self._compute_loss(
                logits_by_task,
                reg_pred_by_task,
                input_dict,
                pixel_logits_by_task=pixel_logits_by_task,
                pixel_targets_by_task=pixel_targets_by_task,
            )
            return_dict["loss"] = total_loss
            return_dict["loss_by_task"] = loss_by_task

        if self.training:
            with torch.no_grad():
                return_dict["pred"] = main_logits.argmax(dim=1)
                return_dict["pred_by_task"] = {
                    task_name: logits.argmax(dim=1)
                    for task_name, logits in logits_by_task.items()
                }
                for task_name, logits in pixel_logits_by_task.items():
                    return_dict["pred_by_task"][task_name] = logits.argmax(dim=-1)

        return return_dict


class AttentiveScenePool(nn.Module):
    """Cross-attention pooling: one learnable query aggregates per-point K/V into a scene token."""

    def __init__(self, embed_dim, attn_drop=0.0):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0.0 else nn.Identity()

    def forward(self, feat, offset):
        """feat: (N, C) per-point features; offset: (B,) cumulative point counts
        per scene. Returns (B, C) attention-weighted scene features."""
        indptr = nn.functional.pad(offset, (1, 0))
        k = self.key_proj(feat)
        v = self.value_proj(feat)
        scores = (k @ self.query.T).squeeze(-1) * self.scale
        batch = offset2batch(offset)
        max_scores = torch_scatter.segment_csr(scores, indptr, reduce="max")
        scores = scores - max_scores[batch]
        exp_scores = torch.exp(scores)
        sum_exp = torch_scatter.segment_csr(exp_scores, indptr, reduce="sum")
        weights = exp_scores / (sum_exp[batch] + 1e-8)
        weights = self.attn_drop(weights)
        return torch_scatter.segment_csr(
            weights.unsqueeze(-1) * v, indptr, reduce="sum"
        )


@MODELS.register_module()
class DefaultClassifier(FrozenBackboneEvalMixin, nn.Module, LearnedMaskedFeatMixin):
    """Backbone + scene pooling + linear ``cls_head`` classifier.

    ``pooling``: ``mean``, ``max`` or ``attention`` (AttentiveScenePool) over the
    per-point backbone features of each scene; target key ``category``.

    freeze_backbone=True means ALL of the following (there is no separate flag):
      * every backbone parameter has requires_grad=False (no gradient, never
        updated by the optimizer);
      * the backbone is kept in eval() mode *permanently*: every .train() call
        (the trainer issues one per epoch) re-pins it. BatchNorm therefore
        uses its stored running statistics and does NOT update them, and
        DropPath / Dropout are inactive. The backbone is a fixed, deterministic
        feature extractor; only the heads are in train mode.
      * the frozen backbone forward additionally runs under torch.no_grad().
    """

    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
        freeze_backbone=False,
        feature_mask_values=None,
        pooling="mean",
    ):
        super().__init__()
        pooling = str(pooling)
        if pooling not in _POOLING_MODES:
            raise ValueError(
                f"pooling must be one of {sorted(_POOLING_MODES)}, got {pooling!r}."
            )
        self.pooling = pooling
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.freeze_backbone = freeze_backbone
        self._init_learned_masked_feat(feature_mask_values=feature_mask_values)
        self.attn_pool = (
            AttentiveScenePool(backbone_embed_dim) if pooling == "attention" else None
        )
        self.cls_head = nn.Linear(backbone_embed_dim, num_classes)
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self._pin_frozen_backbone_eval()

    def _pool_scene_feat(self, feat, offset):
        indptr = nn.functional.pad(offset, (1, 0))
        if self.pooling == "mean":
            return _safe_segment_csr_mean(feat, indptr)
        if self.pooling == "max":
            return torch_scatter.segment_csr(feat, indptr, reduce="max")
        return self.attn_pool(feat, offset)

    def forward(self, input_dict):
        self._fill_masked_feat_with_learned_value(input_dict)
        point = Point(input_dict)
        offset = input_dict["offset"]
        if self.freeze_backbone:
            with torch.no_grad():
                backbone_out = self.backbone(point)
        else:
            backbone_out = self.backbone(point)
        # Point-based backbones return a Point (pooled here, on its own offset,
        # i.e. at the coarsest level for enc_mode); the others return a tensor,
        # either per point or already scene-pooled (e.g. SpUNet enc_mode=True).
        if isinstance(backbone_out, Point):
            feat = self._pool_scene_feat(backbone_out.feat, backbone_out.offset)
        elif isinstance(backbone_out, torch.Tensor):
            # Per-point features
            if backbone_out.shape[0] == offset[-1]:
                feat = self._pool_scene_feat(backbone_out, offset)

            # Already mean-pooled by the backbone
            elif backbone_out.shape[0] == offset.numel():
                if self.pooling != "mean":
                    raise ValueError(
                        "Backbone returned a pre-pooled tensor (B, C), but "
                        f"pooling={self.pooling!r} requires per-point features. "
                        "Use enc_mode=False on the backbone or set pooling='mean'."
                    )
                feat = backbone_out
            else:
                raise ValueError(
                    "Unexpected backbone output shape "
                    f"{tuple(backbone_out.shape)} for offset with "
                    f"{offset.numel()} scenes and {offset[-1].item()} points."
                )
        else:
            raise TypeError(
                f"Backbone must return Point or torch.Tensor, got {type(backbone_out)!r}."
            )
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits.float(), input_dict["category"])
            return_dict = dict(loss=loss)
            with torch.no_grad():
                # Used by the hooks to accumulate the train confusion matrix.
                return_dict["pred"] = cls_logits.argmax(dim=1)
            return return_dict
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits.float(), input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
