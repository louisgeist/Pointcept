"""Ground-truth road graphs are found either under network_graphs_root or in each ROI directory.

Run with: PYTHONPATH=. pytest tests/test_network_graph_paths.py
"""

import tempfile
import unittest
from pathlib import Path

from pointcept.datasets.preprocessing.malibu3d import network_label_utils as nlu


class TestGraphPaths(unittest.TestCase):
    def test_graph_in_roi_directory_when_root_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            roi_dir = Path(tmp) / "test" / "D075-2021_LIDARHD" / "AA-S2-1"
            roi_dir.mkdir(parents=True)
            gpkg = roi_dir / "D075_AA-S2-1_ROADS_graph.gpkg"
            gpkg.touch()
            self.assertEqual(nlu.resolve_exported_graph_gpkg(None, roi_dir, "ROADS"), gpkg)
            self.assertEqual(nlu.expected_exported_graph_path(None, roi_dir, "ROADS"), gpkg)
            self.assertIsNone(nlu.resolve_exported_graph_gpkg(None, roi_dir, "RAILROADS"))

    def test_graph_under_explicit_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "graphs"
            root.mkdir()
            roi_dir = Path(tmp) / "test" / "D075-2021_LIDARHD" / "AA-S2-1"
            gpkg = root / "D075_AA-S2-1_ROADS_graph.gpkg"
            gpkg.touch()
            self.assertEqual(nlu.resolve_exported_graph_gpkg(root, roi_dir, "ROADS"), gpkg)
            self.assertEqual(nlu.expected_exported_graph_path(str(root), roi_dir, "ROADS"), gpkg)


if __name__ == "__main__":
    unittest.main()
