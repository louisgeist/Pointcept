"""tile_catalog: tiles.csv rows are read into Tile records, and the loaders built on it (voxel sizes,
road rasterization) read the same tiles.

Run with: PYTHONPATH=. pytest tests/test_tile_catalog.py
"""

import dataclasses
import os
import pathlib
import tempfile
import unittest

from pointcept.datasets.preprocessing.malibu3d.tile_catalog import (
    Tile,
    read_tiles,
)

TILES_CSV = (
    "split,dept_year,roi,scene_i_j,tile_id,zip_path,has_natural_habitat,has_elevation,"
    "has_roads_graph,roads_gpkg,date_aerial_rgb,date_lidarhd,date_gap_days,n_points,"
    "n_voxels,forest_origin_x,forest_origin_y,forest_width,forest_height\n"
    "train,D004-2021,AA-S1-32,1-1,D004-2021_AA-S1-32_1-1,"
    "data/train/D004-2021_LIDARHD/AA-S1-32.zip,False,True,True,"
    "D004_AA-S1-32_ROADS_graph.gpkg,2021-06-27,2021-06-26,1.0,261926,177624,"
    "927440.0,6302945.0,103,103\n"
    "val,D075-2021,UU-S1-4,1-2,D075-2021_UU-S1-4_1-2,"
    "data/val/D075-2021_LIDARHD/UU-S1-4.zip,True,False,False,"
    "D075_UU-S1-4_ROADS_graph.gpkg,2021-05-01,2021-04-01,30.0,1000,800,"
    "1.0,2.0,3,4\n"
)

EXPECTED = [
    Tile(
        split="train",
        dept_year="D004-2021",
        roi="AA-S1-32",
        scene_i_j="1-1",
        tile_id="D004-2021_AA-S1-32_1-1",
        has_natural_habitat=False,
        has_elevation=True,
        has_roads_graph=True,
        date_gap_days=1.0,
        n_points=261926,
        n_voxels=177624,
    ),
    Tile(
        split="val",
        dept_year="D075-2021",
        roi="UU-S1-4",
        scene_i_j="1-2",
        tile_id="D075-2021_UU-S1-4_1-2",
        has_natural_habitat=True,
        has_elevation=False,
        has_roads_graph=False,
        date_gap_days=30.0,
        n_points=1000,
        n_voxels=800,
    ),
]


RASTER_GEOREF = [
    dict(raster_origin_x=927440.0, raster_origin_y=6302945.0, raster_width=103, raster_height=103),
    dict(raster_origin_x=1.0, raster_origin_y=2.0, raster_width=3, raster_height=4),
]
EXPECTED_TILES_CSV = [dataclasses.replace(t, **g) for t, g in zip(EXPECTED, RASTER_GEOREF)]


class TestTileCatalog(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tiles_csv = os.path.join(self._tmp.name, "tiles.csv")
        with open(self.tiles_csv, "w", encoding="utf-8") as f:
            f.write(TILES_CSV)

    def tearDown(self):
        self._tmp.cleanup()

    def test_read_tiles(self):
        self.assertEqual(read_tiles(self.tiles_csv), EXPECTED_TILES_CSV)

    def test_split_filter(self):
        self.assertEqual(read_tiles(self.tiles_csv, ["val"]), EXPECTED_TILES_CSV[1:])

    def test_tile_dir(self):
        self.assertEqual(
            EXPECTED[0].tile_dir("/root"),
            "/root/train/D004-2021_LIDARHD/AA-S1-32/D004-2021_AA-S1-32_1-1",
        )

    def test_missing_optional_columns_are_none(self):
        path = os.path.join(self._tmp.name, "minimal.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("split,dept_year,roi,tile_id\ntrain,D1-2021,R,D1-2021_R_1-1\n")
        (tile,) = read_tiles(path)
        self.assertIsNone(tile.has_roads_graph)
        self.assertIsNone(tile.n_points)

    def test_missing_required_column_raises(self):
        path = os.path.join(self._tmp.name, "bad.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("split,dept_year,roi\ntrain,D1-2021,R\n")
        with self.assertRaises(ValueError):
            read_tiles(path)

    def test_invalid_bool_raises(self):
        path = os.path.join(self._tmp.name, "badbool.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("split,dept_year,roi,tile_id,has_elevation\ntrain,D,R,D_R_1-1,maybe\n")
        with self.assertRaises(ValueError):
            read_tiles(path)

    def test_voxel_sizes(self):
        from pointcept.datasets.utils import load_voxel_size_csv

        expected = {"D004-2021_AA-S1-32_1-1": 177624, "D075-2021_UU-S1-4_1-2": 800}
        self.assertEqual(load_voxel_size_csv(self.tiles_csv), expected)

    def test_rasterize_network_loader(self):
        from pointcept.datasets.preprocessing.malibu3d.rasterize_network import (
            load_manifest_patches,
        )

        patches, n_skipped = load_manifest_patches(pathlib.Path(self.tiles_csv))
        self.assertEqual(n_skipped, 0)
        self.assertEqual(
            [(p.patch_id, p.flags_dict) for p in patches],
            [
                ("D004-2021_AA-S1-32_1-1", {"ROADS": True}),
                ("D075-2021_UU-S1-4_1-2", {"ROADS": False}),
            ],
        )


if __name__ == "__main__":
    unittest.main()
