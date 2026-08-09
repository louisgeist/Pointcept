"""
Tests for Flair3DDataset's pixel-semantic asset loading, generalized from a
network-only _load_network_label to a target_key-parametrized
_load_pixel_semantic_label so both "network" and "forest_2d" (or any other
FLAIR3D_PIXEL_SEMANTIC_TASKS entry) load through the same code path.

Constructs a bare Flair3DDataset instance via object.__new__ (bypassing __init__,
which needs real on-disk manifests) and sets only the attributes the method under
test actually reads (self.optional_target_keys), following the same lightweight
pattern as tests/test_tile_distribution_pooling.py's direct model construction.

Run with: PYTHONPATH=./ pytest tests/test_pixel_semantic_dataset_loading.py
"""

import json
import os
import tempfile
import unittest

import numpy as np

from pointcept.datasets.flair3d import Flair3DDataset


def _bare_dataset(optional_target_keys=()):
    ds = object.__new__(Flair3DDataset)
    ds.optional_target_keys = tuple(optional_target_keys)
    return ds


class TestLoadPixelSemanticLabelForestTwoD(unittest.TestCase):
    def test_loads_npy_already_in_data_dict(self):
        ds = _bare_dataset()
        with tempfile.TemporaryDirectory() as scene:
            meta = {
                "forest_2d": {
                    "origin_x": 10.0,
                    "origin_y": 20.0,
                    "pixel_m": 0.5,
                    "width": 4,
                    "height": 3,
                    "channel_order": ["FOREST"],
                }
            }
            with open(os.path.join(scene, "meta.json"), "w") as f:
                json.dump(meta, f)
            raster = np.ones((1, 3, 4), dtype=np.uint8)
            data_dict = {"forest_2d": raster}

            out = ds._load_pixel_semantic_label(data_dict, scene, target_key="forest_2d")

            np.testing.assert_array_equal(out["forest_2d"], raster)
            self.assertEqual(out["forest_2d_origin_x"], [10.0])
            self.assertEqual(out["forest_2d_origin_y"], [20.0])
            self.assertEqual(out["forest_2d_pixel_m"], [0.5])

    def test_meta_only_empty_tile_synthesizes_zeros(self):
        ds = _bare_dataset()
        with tempfile.TemporaryDirectory() as scene:
            meta = {
                "forest_2d": {
                    "origin_x": 0.0,
                    "origin_y": 0.0,
                    "pixel_m": 0.5,
                    "width": 4,
                    "height": 3,
                }
            }
            with open(os.path.join(scene, "meta.json"), "w") as f:
                json.dump(meta, f)
            data_dict = {}

            out = ds._load_pixel_semantic_label(data_dict, scene, target_key="forest_2d")

            self.assertEqual(out["forest_2d"].shape, (1, 3, 4))
            self.assertTrue((out["forest_2d"] == 0).all())

    def test_missing_and_not_optional_raises(self):
        ds = _bare_dataset(optional_target_keys=())
        with tempfile.TemporaryDirectory() as scene:
            data_dict = {}
            with self.assertRaises(FileNotFoundError):
                ds._load_pixel_semantic_label(data_dict, scene, target_key="forest_2d")

    def test_missing_and_optional_uses_fill_value(self):
        ds = _bare_dataset(optional_target_keys=("forest_2d",))
        with tempfile.TemporaryDirectory() as scene:
            data_dict = {}
            out = ds._load_pixel_semantic_label(data_dict, scene, target_key="forest_2d")
            self.assertEqual(out["forest_2d"].shape[0], 1)


class TestNetworkStillWorksUnchanged(unittest.TestCase):
    def test_default_target_key_loads_network(self):
        ds = _bare_dataset()
        with tempfile.TemporaryDirectory() as scene:
            meta = {
                "network": {
                    "origin_x": 1.0,
                    "origin_y": 2.0,
                    "pixel_m": 1.0,
                    "width": 2,
                    "height": 2,
                    "channel_order": ["ROADS", "RAILROADS", "TRANSMISSION_LINES"],
                }
            }
            with open(os.path.join(scene, "meta.json"), "w") as f:
                json.dump(meta, f)
            raster = np.zeros((3, 2, 2), dtype=np.uint8)
            raster[0] = 1  # ROADS channel
            data_dict = {"network": raster}

            out = ds._load_pixel_semantic_label(data_dict, scene)

            self.assertEqual(out["network"].shape, (2, 2, 2))  # sliced to r=2
            np.testing.assert_array_equal(out["network"][0], np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
