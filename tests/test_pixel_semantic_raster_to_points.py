"""
Tests for NetworkRasterToPointLabels's target_key generalization: the transform
used to be hardcoded to the literal "network" field name; it must now support an
arbitrary pixel_semantic target_key (e.g. "forest_2d") while preserving its exact
historical behavior when called without target_key (defaults to "network").

Also verifies the multi-task ordering requirement: since a real multi-task pipeline
runs this transform once per pixel_semantic task on the *same* data_dict, the first
call must not consume "abs_xy" so the second call can still use it.

Run with: PYTHONPATH=./ pytest tests/test_pixel_semantic_raster_to_points.py
"""

import unittest

import numpy as np

from pointcept.datasets.transform import NetworkRasterToPointLabels


def _make_data_dict(raster_key, raster, origin_x, origin_y, pixel_m, abs_xy):
    return {
        raster_key: raster,
        f"{raster_key}_origin_x": np.asarray([origin_x], dtype=np.float64),
        f"{raster_key}_origin_y": np.asarray([origin_y], dtype=np.float64),
        f"{raster_key}_pixel_m": np.asarray([pixel_m], dtype=np.float64),
        "abs_xy": abs_xy,
        "coord": np.zeros((abs_xy.shape[0], 3), dtype=np.float32),
        "index_valid_keys": ["coord", "abs_xy"],
    }


class TestDefaultTargetKeyIsNetwork(unittest.TestCase):
    def test_default_target_key_behaves_like_historical_network(self):
        # 2x2 grid, origin (0, 0), pixel_m=1. Points sit in each of the 4 cells.
        raster = np.array([[[0, 1], [1, 0]]], dtype=np.uint8)  # (r=1, H=2, W=2)
        abs_xy = np.array(
            [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]], dtype=np.float64
        )
        data_dict = _make_data_dict("network", raster, 0.0, 0.0, 1.0, abs_xy)

        out = NetworkRasterToPointLabels()(data_dict)

        np.testing.assert_array_equal(out["network"], [[0], [1], [1], [0]])
        np.testing.assert_array_equal(out["network_cell"], [[0, 0], [0, 1], [1, 0], [1, 1]])
        np.testing.assert_array_equal(out["network_pix"], [[0, 0], [0, 1], [1, 0], [1, 1]])
        np.testing.assert_array_equal(out["network_height"], [2])
        np.testing.assert_array_equal(out["network_width"], [2])
        self.assertIn("network", out["index_valid_keys"])
        self.assertIn("network_cell", out["index_valid_keys"])
        self.assertIn("network_pix", out["index_valid_keys"])


class TestExplicitTargetKey(unittest.TestCase):
    def test_forest_2d_target_key_produces_independent_fields(self):
        raster = np.array([[[1, 0], [0, 1]]], dtype=np.uint8)
        abs_xy = np.array([[0.25, 0.25], [0.75, 0.75]], dtype=np.float64)
        data_dict = _make_data_dict("forest_2d", raster, 0.0, 0.0, 0.5, abs_xy)

        out = NetworkRasterToPointLabels(target_key="forest_2d")(data_dict)

        np.testing.assert_array_equal(out["forest_2d"], [[1], [1]])
        self.assertNotIn("network_cell", out)
        self.assertNotIn("network_pix", out)
        self.assertIn("forest_2d_cell", out)
        self.assertIn("forest_2d_pix", out)


class TestTwoPixelSemanticTasksInSequence(unittest.TestCase):
    def test_network_then_forest_2d_both_succeed_on_same_data_dict(self):
        # Regression test: the original implementation popped "abs_xy" after use,
        # which would silently no-op the *second* NetworkRasterToPointLabels call
        # in a multi-task pipeline (network then forest_2d), leaving forest_2d's
        # raster un-converted (still dense) all the way to Collect.
        abs_xy = np.array([[0.5, 0.5]], dtype=np.float64)
        network_raster = np.array([[[1]]], dtype=np.uint8)  # (1, 1, 1)
        forest_raster = np.array([[[0]]], dtype=np.uint8)  # (1, 1, 1)
        data_dict = {
            "network": network_raster,
            "network_origin_x": np.asarray([0.0]),
            "network_origin_y": np.asarray([0.0]),
            "network_pixel_m": np.asarray([1.0]),
            "forest_2d": forest_raster,
            "forest_2d_origin_x": np.asarray([0.0]),
            "forest_2d_origin_y": np.asarray([0.0]),
            "forest_2d_pixel_m": np.asarray([1.0]),
            "abs_xy": abs_xy,
            "coord": np.zeros((1, 3), dtype=np.float32),
            "index_valid_keys": ["coord", "abs_xy"],
        }

        data_dict = NetworkRasterToPointLabels(target_key="network")(data_dict)
        data_dict = NetworkRasterToPointLabels(target_key="forest_2d")(data_dict)

        np.testing.assert_array_equal(data_dict["network"], [[1]])
        np.testing.assert_array_equal(data_dict["forest_2d"], [[0]])
        self.assertIn("forest_2d_cell", data_dict)
        self.assertIn("forest_2d_pix", data_dict)


if __name__ == "__main__":
    unittest.main()
