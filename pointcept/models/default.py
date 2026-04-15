import torch
import torch.nn as nn
import torch_scatter
import torch_cluster
from peft import LoraConfig, get_peft_model
from collections import OrderedDict
from pointcept.utils.logger import get_root_logger
from pointcept.models.losses import build_criteria
from pointcept.utils.misc import intersection_and_union_gpu
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from .builder import MODELS, build_model

logger = get_root_logger()


class LearnedMaskedFeatMixin:
    def _init_learned_masked_feat(self, feature_mask_values=None):
        cfg = feature_mask_values or {}
        self.enable_learned_masked_feat = bool(cfg.get("enable", False))
        self.learned_masked_feat_keys = tuple(
            cfg.get("masked_feat_keys", ("color", "normal", "strength"))
        )

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
            # Get the mask of points where the feature is masked.
            mask_key = f"{feat_key}_mask" 
            
            if mask_key not in input_dict:
                # If no mask, then there is no need to apply filling with the learned value.
                continue
            
            # The features (color, normal, coord, etc.) are already concatenated
            # in the "feat" tensor. Thus, the start and end keys are necessary to
            # locate the slice of the feature in the "feat" tensor.
            start_key = f"{feat_key}_feat_start"
            end_key = f"{feat_key}_feat_end"

            start_value = input_dict[start_key]
            end_value = input_dict[end_key]
            start = int(start_value[0].item()) if torch.is_tensor(start_value) else int(start_value)
            end = int(end_value[0].item()) if torch.is_tensor(end_value) else int(end_value)
            feat_dim = end - start
            if feat_dim <= 0:
                continue

            expected_dim = expected_dims.get(feat_key, feat_dim)
            assert feat_dim == expected_dim, (
                f"Unexpected {feat_key} feature dim: expected {expected_dim}, got {feat_dim}."
            )

            mask = input_dict[mask_key].bool().unsqueeze(-1)
            assert hasattr(self, f"{feat_key}_mask_value"), (
                f"Missing learned parameter '{feat_key}_mask_value'."
            )
            learned_mask_value = getattr(self, f"{feat_key}_mask_value").to(feat.dtype)
            feat[:, start:end] = torch.where(
                mask,
                learned_mask_value,
                feat[:, start:end],
            )
            
    def _print_learned_masked_feat(self):
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
                print(
                    f"{feat_key.capitalize()} mask value: "
                    f"{value}"
                )


@MODELS.register_module()
class DefaultSegmentor(nn.Module, LearnedMaskedFeatMixin):
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
            # Keep segmentation heads trainable for backbones that own their
            # own output classifier (e.g., SpUNet: "final", KPConvX: "final").
            keep_segmentation_head_prefixes = ("final.")
            for name, param in self.backbone.named_parameters():
                if name.startswith(keep_segmentation_head_prefixes):
                    continue
                param.requires_grad = False

    def forward(self, input_dict):
        self._fill_masked_feat_with_learned_value(input_dict)
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict = dict(loss=loss)
            with torch.no_grad():
                # Expose predictions for epoch-level confusion accumulation in hooks. (for train/miou)
                # (We avoid logging this tensor directly; the writer filters non-scalars.)
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
class DefaultSegmentorV2(nn.Module, LearnedMaskedFeatMixin):
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
    
    def get_ignore_index(self) -> int:
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
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
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
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            with torch.no_grad():
                # Expose predictions for epoch-level confusion accumulation in hooks.
                # (We avoid logging this tensor directly; the writer filters non-scalars.)
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
class DefaultLORASegmentorV2(nn.Module, LearnedMaskedFeatMixin):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        backbone_path=None,
        keywords=None,
        replacements=None,
        feature_mask_values=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.keywords = keywords
        self.replacements = replacements
        self.backbone = build_model(backbone)
        backbone_weight = torch.load(
            backbone_path,
            map_location=lambda storage, loc: storage.cuda(),
        )
        self.backbone_load(backbone_weight)

        self.criteria = build_criteria(criteria)
        self._init_learned_masked_feat(feature_mask_values=feature_mask_values)
        self._ignore_index = None
        # Cache ignore_index at init to keep runtime deterministic.
        self.get_ignore_index()
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        if self.use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv"],
                # target_modules=["query", "value"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone.enc = get_peft_model(self.backbone.enc, lora_config)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.use_lora:
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        self.backbone.enc.print_trainable_parameters()
    
    def get_ignore_index(self) -> int:
        if self._ignore_index is not None:
            return self._ignore_index
        for c in getattr(self.criteria, "criteria", []):
            if hasattr(c, "ignore_index"):
                self._ignore_index = int(getattr(c, "ignore_index"))
                return self._ignore_index
        logger.warning("No ignore_index found in criteria, using -1 as default.")
        self._ignore_index = -1
        return self._ignore_index

    def backbone_load(self, checkpoint):
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if not key.startswith("module."):
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
            # Now all keys contain "module." no matter DDP or not.
            if self.keywords in key:
                key = key.replace(self.keywords, self.replacements)
            key = key[7:]  # module.xxx.xxx -> xxx.xxx
            if key.startswith("backbone."):
                key = key[9:]
            weight[key] = value
        load_state_info = self.backbone.load_state_dict(weight, strict=False)
        print(f"Missing keys: {load_state_info[0]}")
        print(f"Unexpected keys: {load_state_info[1]}")

    def forward(self, input_dict, return_point=False):
        self._fill_masked_feat_with_learned_value(input_dict)
        point = Point(input_dict)
        if self.freeze_backbone and not self.use_lora:
            with torch.no_grad():
                point = self.backbone(point)
        else:
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

        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            with torch.no_grad(): # To compute metrics
                return_dict["pred"] = seg_logits.argmax(dim=1)
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module, LearnedMaskedFeatMixin):
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
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self._init_learned_masked_feat(feature_mask_values=feature_mask_values)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        self._fill_masked_feat_with_learned_value(input_dict)
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["origin_segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["origin_segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module, LearnedMaskedFeatMixin):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
        freeze_backbone=False,
        feature_mask_values=None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.freeze_backbone = freeze_backbone
        self._init_learned_masked_feat(feature_mask_values=feature_mask_values)
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(self, input_dict):
        self._fill_masked_feat_with_learned_value(input_dict)
        point = Point(input_dict)
        if self.freeze_backbone:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
