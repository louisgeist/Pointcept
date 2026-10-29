"""forest_2d's entry in MALIBU3D_PIXEL_SEMANTIC_TASKS, and init_multitask_collect_keys emitting per-task
"<task>_*" raster keys for each pixel_semantic task (network, forest_2d) without collisions.

Run with: PYTHONPATH=. pytest tests/test_pixel_semantic_collect_keys.py
"""

import unittest

from pointcept.datasets.malibu3d_tasks import (
    MALIBU3D_PIXEL_SEMANTIC_TASKS,
    MALIBU3D_TARGET_KEYS,
    get_task_config,
    init_multitask_collect_keys,
)


class TestForestTwoDRegistration(unittest.TestCase):
    def test_forest_2d_registered_with_expected_fields(self):
        cfg = get_task_config("forest_2d")
        self.assertEqual(cfg["task_type"], "pixel_semantic")
        self.assertEqual(cfg["num_networks"], 1)
        self.assertEqual(cfg["num_classes"], 2)
        self.assertEqual(cfg["ignore_index"], 2)
        self.assertEqual(cfg["channel_names"], ["FOREST"])
        self.assertEqual(cfg["names"], ["Not Forest", "Forest", "Void"])
        self.assertEqual(cfg["pooling"], "mean")
        self.assertFalse(cfg["enable_dilated_prf"])

    def test_get_task_config_returns_independent_copies(self):
        a = get_task_config("forest_2d")
        b = get_task_config("forest_2d")
        a["num_classes"] = 999
        self.assertEqual(b["num_classes"], 2)

    def test_network_defaults_unchanged(self):
        cfg = MALIBU3D_PIXEL_SEMANTIC_TASKS["network"]
        self.assertNotIn("pooling", cfg)
        self.assertTrue(cfg["enable_dilated_prf"])


class TestInitMultitaskCollectKeys(unittest.TestCase):
    def test_network_only_matches_historical_key_set(self):
        train_keys, val_keys, _, _ = init_multitask_collect_keys(("network",))
        expected_extra = (
            "network_cell",
            "network_pix",
            "network_origin_x",
            "network_origin_y",
            "network_pixel_m",
            "network_height",
            "network_width",
        )
        for key in expected_extra:
            self.assertIn(key, train_keys)
            self.assertIn(key, val_keys)
        self.assertIn("network", train_keys)

    def test_forest_2d_only_gets_its_own_keys_not_network(self):
        train_keys, val_keys, _, _ = init_multitask_collect_keys(("forest_2d",))
        self.assertIn("forest_2d_cell", train_keys)
        self.assertIn("forest_2d_pix", train_keys)
        self.assertIn("forest_2d_origin_x", train_keys)
        self.assertIn("forest_2d_origin_y", train_keys)
        self.assertIn("forest_2d_pixel_m", train_keys)
        self.assertIn("forest_2d_height", train_keys)
        self.assertIn("forest_2d_width", train_keys)
        self.assertNotIn("network_cell", train_keys)
        self.assertNotIn("network_pix", train_keys)

    def test_network_and_forest_2d_coexist_without_collision(self):
        train_keys, val_keys, _, _ = init_multitask_collect_keys(
            ("network", "forest_2d")
        )
        for prefix in ("network", "forest_2d"):
            for suffix in (
                "cell",
                "pix",
                "origin_x",
                "origin_y",
                "pixel_m",
                "height",
                "width",
            ):
                self.assertIn(f"{prefix}_{suffix}", train_keys)
        # Both raw target names present, each exactly once.
        self.assertEqual(train_keys.count("network"), 1)
        self.assertEqual(train_keys.count("forest_2d"), 1)

    def test_forest_2d_is_scene_level_in_val_keys(self):
        # pixel_semantic targets are "scene-level": val Collect keys must NOT
        # include an origin_forest_2d entry (unlike point-wise targets).
        _, val_keys, _, _ = init_multitask_collect_keys(("forest_2d",))
        self.assertIn("forest_2d", val_keys)
        self.assertNotIn("origin_forest_2d", val_keys)

    def test_test_keys_of_the_benchmark(self):
        # Fragment Collect of the multitask test pipeline: grid_coord but not grid_size,
        # no point-wise target, pixel targets in registry order (network, forest_2d).
        grid = ("cell", "pix", "origin_x", "origin_y", "pixel_m", "height", "width")
        _, _, test_keys, _ = init_multitask_collect_keys(
            MALIBU3D_TARGET_KEYS, collect_prefix_keys=("grid_coord", "grid_size")
        )
        expected = ("coord", "grid_coord", "index")
        for key in ("network", "forest_2d"):
            expected += (key,) + tuple(f"{key}_{s}" for s in grid)
        self.assertEqual(test_keys, expected)
        _, _, test_keys, _ = init_multitask_collect_keys(("segment",))
        self.assertEqual(test_keys, ("coord", "index"))


if __name__ == "__main__":
    unittest.main()
