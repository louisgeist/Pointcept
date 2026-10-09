"""
Tests for tile_catalog.iter_tiles: the MALiBU3D tiles.csv and the legacy
scene_split_manifest.csv must normalize to the same Tile records, and the
preprocessing loaders built on it must accept both schemas.

Run with: PYTHONPATH=./ pytest tests/test_tile_catalog.py
"""

import os
import pathlib
import tempfile
import unittest

from pointcept.datasets.preprocessing.flair3d_plus.tile_catalog import (
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

LEGACY_CSV = (
    "split,dept_year,roi,scene_i_j,patch_id,LIDARHD,NATURAL_HABITAT,LAND_USE,DEM_ELEV,"
    "ROADS,RAILROADS,TRANSMISSION_LINES,date_aerial_rgb,date_lidarhd,date_gap_days,"
    "n_points,n_voxels\n"
    "train,D004-2021,AA-S1-32,1-1,D004-2021_AA-S1-32_1-1,True,False,True,True,True,"
    "False,False,2021-06-27,2021-06-26,1.0,261926,177624\n"
    "train,D004-2021,AA-S1-32,1-2,D004-2021_AA-S1-32_1-2,False,False,False,False,,,,"
    "2021-06-27,,<NA>,,\n"
    "val,D075-2021,UU-S1-4,1-2,D075-2021_UU-S1-4_1-2,True,True,False,False,False,"
    "True,True,2021-05-01,2021-04-01,30.0,1000,800\n"
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


class TestTileCatalog(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tiles_csv = os.path.join(self._tmp.name, "tiles.csv")
        self.legacy_csv = os.path.join(self._tmp.name, "scene_split_manifest.csv")
        with open(self.tiles_csv, "w", encoding="utf-8") as f:
            f.write(TILES_CSV)
        with open(self.legacy_csv, "w", encoding="utf-8") as f:
            f.write(LEGACY_CSV)

    def tearDown(self):
        self._tmp.cleanup()

    def test_both_schemas_normalize_identically(self):
        self.assertEqual(read_tiles(self.tiles_csv), EXPECTED)
        # Legacy LIDARHD=False row is dropped; extra legacy columns are ignored.
        self.assertEqual(read_tiles(self.legacy_csv), EXPECTED)

    def test_split_filter(self):
        self.assertEqual(read_tiles(self.tiles_csv, ["val"]), EXPECTED[1:])
        self.assertEqual(read_tiles(self.legacy_csv, ["train"]), EXPECTED[:1])

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

    def test_voxel_sizes_accept_tile_id(self):
        from pointcept.datasets.utils import load_voxel_size_csv

        expected = {"D004-2021_AA-S1-32_1-1": 177624, "D075-2021_UU-S1-4_1-2": 800}
        self.assertEqual(load_voxel_size_csv(self.tiles_csv), expected)
        self.assertEqual(load_voxel_size_csv(self.legacy_csv), expected)

    def test_rasterize_network_loader_both_schemas(self):
        from pointcept.datasets.preprocessing.flair3d_plus.rasterize_network import (
            load_manifest_patches,
        )

        for path in (self.tiles_csv, self.legacy_csv):
            patches, n_skipped = load_manifest_patches(pathlib.Path(path))
            self.assertEqual(n_skipped, 0)
            self.assertEqual(
                [(p.patch_id, p.flags_dict) for p in patches],
                [
                    ("D004-2021_AA-S1-32_1-1", {"ROADS": True}),
                    ("D075-2021_UU-S1-4_1-2", {"ROADS": False}),
                ],
            )

    def test_rasterize_forest_loader_both_schemas(self):
        from pointcept.datasets.preprocessing.flair3d_plus.rasterize_forest import (
            load_manifest_patches,
        )

        for path in (self.tiles_csv, self.legacy_csv):
            patches, _ = load_manifest_patches(pathlib.Path(path), splits=["train"])
            self.assertEqual(
                [p.lidar_patch_stem() for p in patches],
                ["D004-2021_LIDARHD_AA-S1-32_1-1"],
            )

    def test_preprocess_loader_both_schemas(self):
        try:
            from pointcept.datasets.preprocessing.flair3d_plus.preprocess_flair3d_v2 import (
                load_manifest_tasks,
            )
        except ImportError as exc:  # plyfile / rasterio missing
            self.skipTest(str(exc))

        for path in (self.tiles_csv, self.legacy_csv):
            tasks = sorted(load_manifest_tasks(path, ["train", "val"]), key=lambda t: t.patch_id)
            self.assertEqual(
                [(t.patch_id, t.has_natural_habitat, t.has_dem_elev, t.date_gap_days) for t in tasks],
                [
                    ("D004-2021_AA-S1-32_1-1", False, True, 1.0),
                    ("D075-2021_UU-S1-4_1-2", True, False, 30.0),
                ],
            )


if __name__ == "__main__":
    unittest.main()
