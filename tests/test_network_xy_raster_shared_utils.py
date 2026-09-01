"""
Tests for preprocessing utilities shared between rasterize_network.py and
rasterize_forest.py: abs_xy_bounds_from_coord, load_known_missing_tiles, and
default_missing_coord_details_csv. Extracted from rasterize_network.py
(previously private, single-use helpers with no test coverage) into
network_xy_raster_utils.py so rasterize_forest.py can reuse them without
duplication.

Run with: PYTHONPATH=./ pytest tests/test_network_xy_raster_shared_utils.py
"""

import csv
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pointcept.datasets.preprocessing.malibu3d_plus.network_xy_raster_utils import (
    abs_xy_bounds_from_coord,
    default_missing_coord_details_csv,
    load_known_missing_tiles,
)


class TestAbsXyBoundsFromCoord(unittest.TestCase):
    def test_computes_bounds_with_translation(self):
        with tempfile.TemporaryDirectory() as patch_dir:
            coord = np.array(
                [[0.0, 0.0, 0.0], [10.0, 5.0, 0.0], [3.0, -2.0, 0.0]], dtype=np.float32
            )
            np.save(os.path.join(patch_dir, "coord.npy"), coord)
            np.save(
                os.path.join(patch_dir, "coord_translation.npy"),
                np.array([1000.0, 2000.0, 0.0], dtype=np.float64),
            )
            xmin, ymin, xmax, ymax = abs_xy_bounds_from_coord(Path(patch_dir))
            self.assertEqual((xmin, ymin, xmax, ymax), (1000.0, 1998.0, 1010.0, 2005.0))

    def test_missing_translation_file_raises(self):
        with tempfile.TemporaryDirectory() as patch_dir:
            np.save(
                os.path.join(patch_dir, "coord.npy"),
                np.zeros((1, 3), dtype=np.float32),
            )
            with self.assertRaises(FileNotFoundError):
                abs_xy_bounds_from_coord(Path(patch_dir))


class TestLoadKnownMissingTiles(unittest.TestCase):
    def test_none_path_returns_empty_set(self):
        self.assertEqual(load_known_missing_tiles(None), set())

    def test_reads_details_csv_reason_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "missing_coord_tiles.details.csv")
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["split", "patch_id", "reason"])
                writer.writeheader()
                writer.writerow(
                    {"split": "Train", "patch_id": "A-1", "reason": "missing_coord_file"}
                )
                writer.writerow(
                    {"split": "Train", "patch_id": "B-2", "reason": "other_reason"}
                )
            result = load_known_missing_tiles(Path(path))
            self.assertEqual(result, {("train", "A-1")})

    def test_reads_plain_text_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "missing_ply_preflight.txt")
            with open(path, "w") as f:
                f.write("# comment\nVal,C-3,some note\n")
            result = load_known_missing_tiles(Path(path))
            self.assertEqual(result, {("val", "C-3")})


class TestDefaultMissingCoordDetailsCsv(unittest.TestCase):
    def test_points_under_data_malibu3d_plus(self):
        path = default_missing_coord_details_csv()
        parts = path.parts
        self.assertIn("data", parts)
        self.assertIn("malibu3d_plus", parts)
        self.assertEqual(path.name, "missing_coord_tiles.details.csv")


if __name__ == "__main__":
    unittest.main()
