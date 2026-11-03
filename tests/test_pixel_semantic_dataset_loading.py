"""MALiBU3DDataset._load_pixel_semantic_label for "network" and "forest_2d": raster + georeferencing from
tiles.csv (raster_origins), missing-raster handling, channel count check. The dataset is built with
object.__new__ (its __init__ needs on-disk manifests); only optional_target_keys and raster_origins are set.

Run with: PYTHONPATH=. pytest tests/test_pixel_semantic_dataset_loading.py
"""

import os
import tempfile
import unittest

import numpy as np

from pointcept.datasets.malibu3d import MALiBU3DDataset

TILE = "D000-2020_XX-S1-1_1-1"


def _bare_dataset(optional_target_keys=(), origin=(10.0, 20.0)):
    ds = object.__new__(MALiBU3DDataset)
    ds.optional_target_keys = tuple(optional_target_keys)
    ds.raster_origins = {TILE: origin}
    return ds


def _scene(root):
    scene = os.path.join(root, TILE)
    os.makedirs(scene)
    return scene


class TestLoadPixelSemanticLabelForestTwoD(unittest.TestCase):
    def test_loads_npy_with_tiles_csv_grid(self):
        ds = _bare_dataset()
        with tempfile.TemporaryDirectory() as root:
            raster = np.ones((1, 3, 4), dtype=np.uint8)
            out = ds._load_pixel_semantic_label(
                {"forest_2d": raster}, _scene(root), target_key="forest_2d"
            )
            np.testing.assert_array_equal(out["forest_2d"], raster)
            self.assertEqual(out["forest_2d_origin_x"], [10.0])
            self.assertEqual(out["forest_2d_origin_y"], [20.0])
            self.assertEqual(out["forest_2d_pixel_m"], [1.0])

    def test_missing_and_not_optional_raises(self):
        ds = _bare_dataset()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                ds._load_pixel_semantic_label({}, _scene(root), target_key="forest_2d")

    def test_missing_and_optional_uses_fill_value(self):
        ds = _bare_dataset(optional_target_keys=("forest_2d",))
        with tempfile.TemporaryDirectory() as root:
            out = ds._load_pixel_semantic_label(
                {}, _scene(root), target_key="forest_2d"
            )
            self.assertEqual(out["forest_2d"].shape[0], 1)

    def test_tile_without_grid_raises(self):
        ds = _bare_dataset()
        ds.raster_origins = {}
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(KeyError):
                ds._load_pixel_semantic_label(
                    {"forest_2d": np.ones((1, 3, 4), dtype=np.uint8)},
                    _scene(root),
                    target_key="forest_2d",
                )


class TestNetworkRaster(unittest.TestCase):
    def test_default_target_key_loads_network(self):
        ds = _bare_dataset(origin=(1.0, 2.0))
        with tempfile.TemporaryDirectory() as root:
            raster = np.ones((1, 2, 2), dtype=np.uint8)
            out = ds._load_pixel_semantic_label({"network": raster}, _scene(root))
            np.testing.assert_array_equal(out["network"], raster)
            self.assertEqual(out["network_origin_x"], [1.0])
            self.assertEqual(out["network_origin_y"], [2.0])

    def test_missing_network_points_to_rasterize_script(self):
        ds = _bare_dataset()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(FileNotFoundError, "rasterize_network.py"):
                ds._load_pixel_semantic_label({}, _scene(root))

    def test_channel_count_mismatch_raises(self):
        ds = _bare_dataset()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                ds._load_pixel_semantic_label(
                    {"network": np.zeros((3, 2, 2), dtype=np.uint8)}, _scene(root)
                )


if __name__ == "__main__":
    unittest.main()
