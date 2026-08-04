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


if __name__ == "__main__":
    unittest.main()
