"""
Tests for apply_subset_selection: stratified sidecar filter and max_sample head slice.

Run with: PYTHONPATH=./ pytest tests/test_apply_subset_selection.py
"""

import csv
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Load subset_utils by path to avoid pointcept.datasets.__init__ (torch deps).
_SUBSET_UTILS_PATH = (
    Path(__file__).resolve().parents[1] / "pointcept" / "datasets" / "subset_utils.py"
)
_spec = importlib.util.spec_from_file_location(
    "_subset_utils_under_test", _SUBSET_UTILS_PATH
)
_subset_utils = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _subset_utils
_spec.loader.exec_module(_subset_utils)
apply_subset_selection = _subset_utils.apply_subset_selection
scene_matches_include_name = _subset_utils.scene_matches_include_name


def _write_sidecar(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "patch_id"])
        writer.writeheader()
        for split, patch_id in rows:
            writer.writerow({"split": split, "patch_id": patch_id})


class TestApplySubsetSelection(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = self._tmpdir.name
        # Paths whose basenames are the patch ids used in the sidecar.
        self.data_list = [
            os.path.join(self.root, "tile_a"),
            os.path.join(self.root, "tile_b"),
            os.path.join(self.root, "tile_c"),
            os.path.join(self.root, "tile_d"),
            os.path.join(self.root, "tile_e"),
        ]
        self.sidecar = os.path.join(self.root, "subset.csv")
        _write_sidecar(
            self.sidecar,
            [
                ("val", "tile_a"),
                ("val", "tile_c"),
                ("val", "tile_d"),
                ("val", "tile_e"),
                ("train", "tile_b"),  # wrong split — must be ignored for val
            ],
        )

    def test_sidecar_only(self):
        result = apply_subset_selection(
            self.data_list,
            split="val",
            stratified_subset_manifest=self.sidecar,
        )
        # Intersection of val data_list with sidecar (sorted by basename).
        self.assertEqual(
            [os.path.basename(p) for p in result],
            ["tile_a", "tile_c", "tile_d", "tile_e"],
        )

    def test_sidecar_then_max_sample(self):
        result = apply_subset_selection(
            self.data_list,
            split="val",
            max_sample=2,
            stratified_subset_manifest=self.sidecar,
        )
        # After sidecar (4 scenes, sorted), head slice to 2.
        self.assertEqual(len(result), 2)
        self.assertEqual(
            [os.path.basename(p) for p in result],
            ["tile_a", "tile_c"],
        )


class TestSceneMatchesIncludeName(unittest.TestCase):
    def test_exact_patch_id(self):
        path = "/data/malibu3d_plus/test/D075-2021_LIDARHD/AA/D075-2021_AA-S2-2"
        self.assertTrue(scene_matches_include_name(path, "D075-2021_AA-S2-2"))

    def test_lidarhd_token_stripped(self):
        path = "/data/malibu3d_plus/test/D075-2021_LIDARHD/AA/D075-2021_AA-S2-2"
        self.assertTrue(scene_matches_include_name(path, "D075-2021_LIDARHD_AA-S2-2"))

    def test_dept_plus_roi_suffix(self):
        path = "/data/malibu3d_plus/train/D075-2021_LIDARHD/UF/D075-2021_UF-S1-2"
        self.assertTrue(scene_matches_include_name(path, "D075_UF-S1-2"))

    def test_no_match(self):
        path = "/data/malibu3d_plus/test/D075-2021_LIDARHD/AA/D075-2021_AA-S2-2"
        self.assertFalse(scene_matches_include_name(path, "D068_UN-S1-28"))


class TestIncludeNamesFilter(unittest.TestCase):
    def setUp(self):
        self.data_list = [
            "/data/malibu3d_plus/test/D075-2021_LIDARHD/AA/D075-2021_AA-S2-2",
            "/data/malibu3d_plus/test/D075-2021_LIDARHD/UU/D075-2021_UU-S1-4",
            "/data/malibu3d_plus/train/D068-2021_LIDARHD/FA/D068-2021_FA-S1-26",
            "/data/malibu3d_plus/val/D049-2021_LIDARHD/AA/D049-2021_AA-S1-1",
        ]

    def test_include_names_keeps_matches(self):
        result = apply_subset_selection(
            self.data_list,
            split=["train", "val", "test"],
            include_names=[
                "D075-2021_AA-S2-2",
                "D075-2021_LIDARHD_UU-S1-4",
                "D068_FA-S1-26",
                "D075_UF-S1-2",  # unmatched — should warn, not crash
            ],
        )
        self.assertEqual(
            [os.path.basename(p) for p in result],
            [
                "D075-2021_AA-S2-2",
                "D075-2021_UU-S1-4",
                "D068-2021_FA-S1-26",
            ],
        )

    def test_multi_split_without_include_names_does_not_raise(self):
        result = apply_subset_selection(
            self.data_list,
            split=["train", "val", "test"],
        )
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
