"""
Ablation (Tab. 4, w/o RGB): evaluate the LitePT-B multitask checkpoint (../multi-litept-b-v1m0-malibu3d.py,
which this config inherits) with RGB removed at validation and test time. Every color value is replaced
by the learned color mask value (RandomDropColor with drop_ratio=1, keep_mask=True); intensity is
unchanged. Evaluation only: run with tools/test.py, do not train with this config.
"""

_base_ = ["../multi-litept-b-v1m0-malibu3d.py"]

weight = "ckpt/litept-b/model.pth"

wandb_run_name = "LPT-B multi test-noRGB 1.1 iter=200000"

# A child config replaces lists instead of merging them: the val transform and the test
# post_transform are restated in full, with the base values below.
grid_size = 0.1
feat_keys = ["coord", "color", "strength"]
coord_feat_scale = 0.01

from pointcept.datasets.malibu3d_tasks import (
    MALIBU3D_COLLECT_PREFIX_LITEPT,
    MALIBU3D_TARGET_KEYS,
    init_multitask_collect_keys,
)

(
    train_multitask_keys,
    val_multitask_keys,
    test_multitask_keys,
    multitask_index_valid_keys,
) = init_multitask_collect_keys(
    MALIBU3D_TARGET_KEYS, collect_prefix_keys=MALIBU3D_COLLECT_PREFIX_LITEPT
)

del MALIBU3D_COLLECT_PREFIX_LITEPT, init_multitask_collect_keys

# Replaces every color value by the learned color mask value.
drop_all_color = dict(
    type="RandomDropColor",
    drop_ratio=1.0,
    drop_application_ratio=1.0,
    keep_mask=True,
)

data = dict(
    val=dict(
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(multitask_index_valid_keys)},
            ),
            dict(type="ExtractAbsXY"),
            dict(type="CenterShift", apply_z=True),
            dict(type="Z_MinShift"),
            dict(
                type="Copy",
                keys_dict={t: f"origin_{t}" for t in MALIBU3D_TARGET_KEYS},
            ),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            drop_all_color,
            dict(type="NetworkRasterToPointLabels"),
            dict(type="NetworkRasterToPointLabels", target_key="forest_2d"),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="Collect",
                keys=val_multitask_keys,
                feat_keys=feat_keys,
                feat_scales=dict(coord=coord_feat_scale),
            ),
        ],
    ),
    test=dict(
        test_cfg=dict(
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                drop_all_color,
                dict(type="NetworkRasterToPointLabels"),
                dict(type="NetworkRasterToPointLabels", target_key="forest_2d"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=test_multitask_keys,
                    optional_keys=("inverse",),
                    feat_keys=feat_keys,
                    feat_scales=dict(coord=coord_feat_scale),
                ),
            ],
        ),
    ),
)

del MALIBU3D_TARGET_KEYS, drop_all_color
