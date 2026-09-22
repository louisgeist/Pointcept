"""Roundtrip PixelGraph -> GeoPackage -> load_roi_exported_network_graph.

Also checks the APLS sidecar dump written next to predicted graphs.

Run with: PYTHONPATH=./ pytest tests/test_pred_graph_gpkg.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import geopandas as gpd

try:
    from osgeo import ogr  # noqa: F401

    HAS_OGR = True
except ImportError:
    HAS_OGR = False

from pointcept.datasets.preprocessing.flair3d_plus.apls_metric import ApsSymmetricResult
from pointcept.datasets.preprocessing.flair3d_plus.network_label_utils import (
    load_roi_exported_network_graph,
    pred_graph_output_stem_paths,
    write_pixel_graph_gpkg,
)
from pointcept.datasets.preprocessing.flair3d_plus.network_xy_raster_utils import (
    GridSpec,
    PixelGraph,
    pixel_centers_xy,
)


def _import_eval_network_apls():
    tools_dir = str(Path(__file__).resolve().parents[1] / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import eval_network_apls

    return eval_network_apls


def _grid() -> GridSpec:
    return GridSpec(
        origin_x=650000.0,
        origin_y=6860000.0,
        width=10,
        height=10,
        pixel_m=1.0,
    )


def _line_graph() -> PixelGraph:
    grid = _grid()
    node_rc = np.array([[1, 1], [1, 3], [4, 3]], dtype=np.int64)
    node_xy = pixel_centers_xy(node_rc, grid)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    weights = np.array([2.0, 3.0], dtype=np.float64)
    return PixelGraph(
        node_rc=node_rc,
        node_xy=node_xy,
        edges=edges,
        grid=grid,
        edge_weights=weights,
    )


def _empty_graph() -> PixelGraph:
    grid = _grid()
    return PixelGraph(
        node_rc=np.empty((0, 2), dtype=np.int64),
        node_xy=np.empty((0, 2), dtype=np.float64),
        edges=np.empty((0, 2), dtype=np.int64),
        grid=grid,
        edge_weights=np.empty((0,), dtype=np.float64),
    )


def _roi_dir(tmp: Path) -> Path:
    roi = tmp / "val" / "D075-2021_LIDARHD" / "UU-S1-4"
    roi.mkdir(parents=True)
    return roi


@unittest.skipUnless(HAS_OGR, "GDAL/OGR not installed")
class TestPixelGraphGpkgRoundtrip(unittest.TestCase):
    def test_roundtrip_preserves_nodes_edges_and_distance(self):
        graph = _line_graph()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.gpkg"
            write_pixel_graph_gpkg(path, graph, metadata={"network_type": "ROADS"})
            loaded = load_roi_exported_network_graph(path)

        np.testing.assert_allclose(loaded.node_xy, graph.node_xy)
        self.assertEqual(loaded.edges.shape, graph.edges.shape)
        self.assertTrue(np.all(loaded.edges[:, 0] < loaded.edges[:, 1]))
        np.testing.assert_array_equal(loaded.edges, graph.edges)

        expected_dist = np.linalg.norm(
            graph.node_xy[graph.edges[:, 1]] - graph.node_xy[graph.edges[:, 0]],
            axis=1,
        )
        np.testing.assert_allclose(loaded.edge_length_m, expected_dist)

    def test_empty_graph_writes_empty_layers(self):
        graph = _empty_graph()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.gpkg"
            write_pixel_graph_gpkg(path, graph, metadata={})
            loaded = load_roi_exported_network_graph(path)

        self.assertEqual(loaded.node_xy.shape, (0, 2))
        self.assertEqual(loaded.edges.shape, (0, 2))
        self.assertEqual(loaded.edge_length_m.shape, (0,))


class TestPredGraphPaths(unittest.TestCase):
    def test_stem_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            roi = _roi_dir(Path(tmp))
            gpkg, sidecar, stem = pred_graph_output_stem_paths(
                Path(tmp) / "out", roi, "ROADS"
            )
        self.assertEqual(stem, "D075_UU-S1-4")
        self.assertEqual(gpkg.name, "D075_UU-S1-4_ROADS_pred_graph.gpkg")
        self.assertEqual(sidecar.name, "D075_UU-S1-4_ROADS_apls.json")

    def test_no_save_pred_gpkg_cli(self):
        eval_network_apls = _import_eval_network_apls()
        parser = eval_network_apls.build_argparser()
        args = parser.parse_args(
            [
                "--data_root",
                "x",
                "--save_path",
                "x",
                "--network_graphs_root",
                "x",
                "--split_manifest_csv",
                "x",
                "--out_dir",
                "x",
                "--no_save_pred_gpkg",
            ]
        )
        self.assertTrue(args.no_save_pred_gpkg)


@unittest.skipUnless(HAS_OGR, "GDAL/OGR not installed")
class TestPredGraphDump(unittest.TestCase):
    def test_dump_writes_gpkg_sidecar_and_metadata(self):
        eval_network_apls = _import_eval_network_apls()
        graph = _line_graph()
        result = ApsSymmetricResult(
            roi="UU-S1-4",
            network_type="ROADS",
            score=0.5,
            score_gt_to_pred=0.4,
            score_pred_to_gt=0.8,
            numerator=1.0,
            denom=2,
            numerator_pred_to_gt=1.0,
            denom_pred_to_gt=2,
            n_nodes_gt=10,
            n_nodes_pred=3,
            n_edges_gt=9,
            n_edges_pred=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            roi = _roi_dir(tmp_path)
            out_dir = tmp_path / "result"
            gpkg_path, json_path = eval_network_apls.dump_pred_graph_artifacts(
                out_dir,
                roi,
                "D075-2021",
                "ROADS",
                graph,
                result,
            )
            self.assertTrue(gpkg_path.is_file())
            self.assertTrue(json_path.is_file())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["stem"], "D075_UU-S1-4")
            self.assertEqual(payload["roi"], "UU-S1-4")
            self.assertEqual(payload["department"], "D075-2021")
            self.assertEqual(payload["network_type"], "ROADS")
            self.assertEqual(payload["score"], 0.5)
            self.assertEqual(payload["score_gt_to_pred"], 0.4)
            self.assertEqual(payload["score_pred_to_gt"], 0.8)
            self.assertEqual(payload["n_nodes_gt"], 10)
            self.assertEqual(payload["n_nodes_pred"], 3)
            self.assertEqual(payload["n_edges_gt"], 9)
            self.assertEqual(payload["n_edges_pred"], 2)

            meta = gpd.read_file(gpkg_path, layer="metadata")
            meta_map = dict(zip(meta["key"].tolist(), meta["value"].tolist()))
            self.assertEqual(meta_map["score"], "0.5")
            self.assertEqual(meta_map["score_gt_to_pred"], "0.4")
            self.assertEqual(meta_map["score_pred_to_gt"], "0.8")


if __name__ == "__main__":
    unittest.main()
