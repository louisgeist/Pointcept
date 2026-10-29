"""malibu3d_tasks: class names and task configs of the released MALiBU3D labels, and the
unpacking of natural_habitat.npy (N, 4) into the four nathab_* axes.

Run with: PYTHONPATH=. pytest tests/test_malibu3d_tasks.py
"""

import unittest

import numpy as np

from pointcept.datasets.malibu3d_tasks import (
    NATHAB_AXES,
    NATHAB_AXIS_KEYS,
    SEGMENT_NAMES,
    get_missing_target_fill_value,
    init_task_configs,
    unpack_nathab_axes,
)


class TestMalibu3dLabels(unittest.TestCase):
    def test_segment_config(self):
        cfg = init_task_configs(("segment",))["segment"]
        self.assertEqual(cfg["num_classes"], 15)
        self.assertEqual(cfg["ignore_index"], 15)
        self.assertEqual(cfg["names"], list(SEGMENT_NAMES))
        self.assertEqual(cfg["task_type"], "semantic")
        self.assertEqual(get_missing_target_fill_value("segment"), 15)

    def test_nathab_configs(self):
        cfgs = init_task_configs(NATHAB_AXIS_KEYS)
        for key, ignore in zip(NATHAB_AXIS_KEYS, (4, 3, 2, 3)):
            self.assertEqual(cfgs[key]["ignore_index"], ignore)
            self.assertEqual(cfgs[key]["num_classes"], ignore)
            self.assertEqual(cfgs[key]["names"], list(NATHAB_AXES[key]))
            self.assertEqual(cfgs[key]["task_type"], "tile_distribution")
            self.assertEqual(get_missing_target_fill_value(key), ignore)

    def test_unpack_axes(self):
        axes = np.array([[0, 1, 0, 2], [4, 3, 2, 3]], dtype=np.uint8)
        out = unpack_nathab_axes(axes)
        self.assertEqual(list(out), list(NATHAB_AXIS_KEYS))
        self.assertEqual(out["nathab_moisture_regime"].tolist(), [1, 3])
        self.assertEqual(out["nathab_bioclimatic_zone"].dtype, np.int32)

    def test_unpack_rejects_carhab_ids_and_out_of_range(self):
        with self.assertRaises(ValueError):
            unpack_nathab_axes(np.zeros(5, dtype=np.int16))
        with self.assertRaises(ValueError):
            unpack_nathab_axes(np.array([[5, 0, 0, 0]], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
