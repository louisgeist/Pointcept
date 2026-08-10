"""
Tests for rasterize_forest.process_patch: reads a small synthetic GeoTIFF window,
resamples it to the target grid, and writes forest_2d.npy + meta.json.

Requires rasterio (already a dependency for Flair3D+ preprocessing). Skips cleanly
if rasterio is not importable in the current environment.

Run with: PYTHONPATH=./ pytest tests/test_rasterize_forest.py
"""

import json
import os
import tempfile
import unittest

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from pointcept.datasets.preprocessing.flair3d_plus.rasterize_forest import (
    ManifestPatch,
    load_manifest_patches,
    process_patch,
)


@unittest.skipUnless(HAS_RASTERIO, "rasterio not installed")
class TestProcessPatch(unittest.TestCase):
    def _write_synthetic_tiff(self, path, xmin, ymax, pixel_m, array, nodata=None):
        # array is already in "north-up" (row 0 = north) orientation, as a real
        # GeoTIFF read would return it.
        transform = from_origin(xmin, ymax, pixel_m, pixel_m)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=array.dtype,
            crs="EPSG:2154",
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(array, 1)

    def test_row_orientation_is_flipped_to_south_up(self):
        # Native tiff at 0.5m (== target pixel_m, so no resampling ambiguity):
        # top-left (north-west) quadrant is forest (1), rest is non-forest (0).
        # In a north-up array, "top-left" is array[0, 0].
        native = np.zeros((4, 4), dtype=np.uint8)
        native[0, 0] = 1

        with tempfile.TemporaryDirectory() as tmp:
            tiff_path = os.path.join(tmp, "FOREST.tif")
            # xmin=0, ymax=2.0 (4 rows * 0.5m), so the grid spans x in [0,2), y in [0,2).
            self._write_synthetic_tiff(tiff_path, xmin=0.0, ymax=2.0, pixel_m=0.5, array=native)

            patch_dir = os.path.join(tmp, "patch")
            os.makedirs(patch_dir)
            coord = np.array([[0.1, 0.1, 0.0], [1.9, 1.9, 0.0]], dtype=np.float32)
            np.save(os.path.join(patch_dir, "coord.npy"), coord)
            np.save(
                os.path.join(patch_dir, "coord_translation.npy"),
                np.array([0.0, 0.0, 0.0], dtype=np.float64),
            )

            process_patch(
                patch_dir, tiff_path, pixel_m=0.5, ignore_index=2,
            )

            forest = np.load(os.path.join(patch_dir, "forest_2d.npy"))
            self.assertEqual(forest.shape, (1, 4, 4))
            # North-west corner (native[0,0]=1) is at max-y, min-x -- in the
            # south-up output grid that is the LAST row, FIRST column.
            self.assertEqual(int(forest[0, -1, 0]), 1)
            # Everywhere else should be 0.
            self.assertEqual(int(forest.sum()), 1)

            with open(os.path.join(patch_dir, "meta.json")) as f:
                meta = json.load(f)
            self.assertEqual(meta["forest_2d"]["pixel_m"], 0.5)
            self.assertEqual(meta["forest_2d"]["width"], 4)
            self.assertEqual(meta["forest_2d"]["height"], 4)
            self.assertEqual(meta["forest_2d"]["channel_order"], ["FOREST"])

    def test_resamples_non_integer_ratio_with_majority_vote(self):
        # Native 0.2m tiff, target 0.5m grid (2.5x downsample, non-integer ratio).
        # Fully-forest native tiff -> every output cell should be forest.
        native = np.ones((10, 10), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            tiff_path = os.path.join(tmp, "FOREST.tif")
            self._write_synthetic_tiff(tiff_path, xmin=0.0, ymax=2.0, pixel_m=0.2, array=native)

            patch_dir = os.path.join(tmp, "patch")
            os.makedirs(patch_dir)
            coord = np.array([[0.1, 0.1, 0.0], [1.9, 1.9, 0.0]], dtype=np.float32)
            np.save(os.path.join(patch_dir, "coord.npy"), coord)
            np.save(
                os.path.join(patch_dir, "coord_translation.npy"),
                np.array([0.0, 0.0, 0.0], dtype=np.float64),
            )

            process_patch(patch_dir, tiff_path, pixel_m=0.5, ignore_index=2)

            forest = np.load(os.path.join(patch_dir, "forest_2d.npy"))
            self.assertTrue((forest == 1).all())

    def test_nodata_pixel_mapped_to_ignore_index(self):
        # Native tiff declares nodata=255 and has one pixel set to that
        # sentinel; the rest is valid forest (1) / non-forest (0) data.
        native = np.zeros((4, 4), dtype=np.uint8)
        native[0, 0] = 255  # nodata sentinel, north-west corner
        native[1, 1] = 1

        with tempfile.TemporaryDirectory() as tmp:
            tiff_path = os.path.join(tmp, "FOREST.tif")
            self._write_synthetic_tiff(
                tiff_path, xmin=0.0, ymax=2.0, pixel_m=0.5, array=native, nodata=255,
            )

            patch_dir = os.path.join(tmp, "patch")
            os.makedirs(patch_dir)
            coord = np.array([[0.1, 0.1, 0.0], [1.9, 1.9, 0.0]], dtype=np.float32)
            np.save(os.path.join(patch_dir, "coord.npy"), coord)
            np.save(
                os.path.join(patch_dir, "coord_translation.npy"),
                np.array([0.0, 0.0, 0.0], dtype=np.float64),
            )

            process_patch(patch_dir, tiff_path, pixel_m=0.5, ignore_index=2)

            forest = np.load(os.path.join(patch_dir, "forest_2d.npy"))
            # The raw nodata sentinel (255) must never survive into the
            # written array -- it should be remapped to ignore_index.
            self.assertNotIn(255, np.unique(forest).tolist())
            # native[0, 0] (north-west) lands at south-up row -1, col 0.
            self.assertEqual(int(forest[0, -1, 0]), 2)

    def test_raises_on_unexpected_pixel_values(self):
        # A raster with no declared nodata but an out-of-range pixel value
        # (e.g. a corrupt/mislabeled source tiff) must fail loudly at
        # preprocessing time rather than silently write an invalid label.
        native = np.zeros((4, 4), dtype=np.uint8)
        native[2, 2] = 7  # not in {0, 1, ignore_index}

        with tempfile.TemporaryDirectory() as tmp:
            tiff_path = os.path.join(tmp, "FOREST.tif")
            self._write_synthetic_tiff(
                tiff_path, xmin=0.0, ymax=2.0, pixel_m=0.5, array=native,
            )

            patch_dir = os.path.join(tmp, "patch")
            os.makedirs(patch_dir)
            coord = np.array([[0.1, 0.1, 0.0], [1.9, 1.9, 0.0]], dtype=np.float32)
            np.save(os.path.join(patch_dir, "coord.npy"), coord)
            np.save(
                os.path.join(patch_dir, "coord_translation.npy"),
                np.array([0.0, 0.0, 0.0], dtype=np.float64),
            )

            with self.assertRaises(ValueError):
                process_patch(patch_dir, tiff_path, pixel_m=0.5, ignore_index=2)


class TestManifestPatchLidarStem(unittest.TestCase):
    def test_lidar_patch_stem_does_not_duplicate_dept_year_and_roi(self):
        # patch_id is already f"{dept_year}_{roi}_{scene_i_j}" (see
        # preprocess_flair3d_v2.py's manifest convention) -- lidar_patch_stem
        # must be built from scene_i_j alone, not from patch_id, or
        # dept_year/roi get duplicated in the resulting FOREST tiff path.
        patch = ManifestPatch(
            split="train",
            dept_year="D026-2020",
            roi="AA-S2-1",
            scene_i_j="5-8",
            patch_id="D026-2020_AA-S2-1_5-8",
        )
        self.assertEqual(patch.lidar_patch_stem(), "D026-2020_LIDARHD_AA-S2-1_5-8")

    def test_load_manifest_patches_reads_scene_i_j_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "manifest.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("split,dept_year,roi,scene_i_j,patch_id,LIDARHD\n")
                f.write("train,D026-2020,AA-S2-1,5-8,D026-2020_AA-S2-1_5-8,True\n")

            import pathlib

            patches, n_skipped = load_manifest_patches(pathlib.Path(csv_path))
            self.assertEqual(n_skipped, 0)
            self.assertEqual(len(patches), 1)
            self.assertEqual(patches[0].lidar_patch_stem(), "D026-2020_LIDARHD_AA-S2-1_5-8")


if __name__ == "__main__":
    unittest.main()
