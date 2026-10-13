"""Hugging Face release layout: tiles have no meta.json, so raster georeferencing is read from tiles.csv.

Run with: PYTHONPATH=. pytest tests/test_release_layout.py
"""

import csv
import os
import tempfile
import unittest

import numpy as np

from pointcept.datasets.malibu3d import MALiBU3DDataset, RELEASE_RASTER_PIXEL_M
from pointcept.datasets.preprocessing.malibu3d.tile_catalog import iter_tiles

HEADER = [
    "split",
    "dept_year",
    "roi",
    "scene_i_j",
    "tile_id",
    "zip_path",
    "has_natural_habitat",
    "has_elevation",
    "has_roads_graph",
    "roads_gpkg",
    "n_points",
    "n_voxels",
    "forest_origin_x",
    "forest_origin_y",
    "forest_width",
    "forest_height",
]


def _write_tiles_csv(path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerow(
            [
                "test",
                "D075-2021",
                "UU-S1-4",
                "1-1",
                "D075-2021_UU-S1-4_1-1",
                "x.zip",
                "True",
                "True",
                "True",
                "D075_UU-S1-4_ROADS_graph.gpkg",
                "10",
                "8",
                "651984.0",
                "6862561.0",
                "103",
                "103",
            ]
        )


class TestTileCatalogRasterColumns(unittest.TestCase):
    def test_raster_georef_is_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiles.csv")
            _write_tiles_csv(path)
            (tile,) = list(iter_tiles(path, ["test"]))
            self.assertEqual(
                (tile.raster_origin_x, tile.raster_origin_y), (651984.0, 6862561.0)
            )
            self.assertEqual((tile.raster_width, tile.raster_height), (103, 103))


class TestRasterOriginWithoutMeta(unittest.TestCase):
    def _dataset(self, origins):
        ds = object.__new__(MALiBU3DDataset)
        ds.optional_target_keys = ("network",)
        ds.raster_origins = origins
        return ds

    def test_forest_2d_uses_tiles_csv_origin(self):
        ds = self._dataset({"D075-2021_UU-S1-4_1-1": (651984.0, 6862561.0)})
        with tempfile.TemporaryDirectory() as tmp:
            scene = os.path.join(tmp, "D075-2021_UU-S1-4_1-1")
            os.makedirs(scene)
            out = ds._load_pixel_semantic_label(
                {"forest_2d": np.zeros((1, 103, 103), dtype=np.uint8)},
                scene,
                target_key="forest_2d",
            )
            self.assertEqual(out["forest_2d_origin_x"], [651984.0])
            self.assertEqual(out["forest_2d_origin_y"], [6862561.0])
            self.assertEqual(out["forest_2d_pixel_m"], [RELEASE_RASTER_PIXEL_M])

    def test_missing_network_stays_unlocated(self):
        # A missing raster must not be anchored at the tile origin: its fill would label real points.
        ds = self._dataset({"D075-2021_UU-S1-4_1-1": (651984.0, 6862561.0)})
        with tempfile.TemporaryDirectory() as tmp:
            scene = os.path.join(tmp, "D075-2021_UU-S1-4_1-1")
            os.makedirs(scene)
            out = ds._load_pixel_semantic_label({}, scene, target_key="network")
            self.assertEqual(out["network_origin_x"], [0.0])
            self.assertEqual(out["network_origin_y"], [0.0])


if __name__ == "__main__":
    unittest.main()
