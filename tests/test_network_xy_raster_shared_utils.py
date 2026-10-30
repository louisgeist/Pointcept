"""Preprocessing helper of rasterize_network.py (pointcept/evaluation/network/xy_raster_utils.py):
abs_xy_bounds_from_coord.

Run with: PYTHONPATH=. pytest tests/test_network_xy_raster_shared_utils.py
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pointcept.evaluation.network.xy_raster_utils import (
    abs_xy_bounds_from_coord,
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


if __name__ == "__main__":
    unittest.main()
