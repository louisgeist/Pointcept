"""The APLS metric is a fixed protocol: parameters are constants, not knobs.

Guards the public API (no tuning parameters), the recorded constants, stale-config rejection,
and a few behavioural invariants that follow from the protocol values.
"""

from __future__ import annotations

import inspect
import re
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pointcept.datasets.preprocessing.flair3d_plus import apls_metric as apls
from pointcept.utils.network_apls import validate_network_apls_eval_cfg

_REMOVED_KEYS = (
    "apls_densify",
    "apls_snap_to_edge",
    "apls_symmetric",
    "apls_min_path_length_m",
    "apls_max_nodes_exact",
)


def _line_graph(y: float, length_m: float = 100.0) -> apls.AplsGraph:
    return apls.AplsGraph(
        node_xy=np.array([[0.0, y], [length_m, y]], dtype=np.float64),
        edges=np.array([[0, 1]], dtype=np.int64),
        edge_length_m=np.array([length_m], dtype=np.float64),
    )


def _score(gt, pred) -> apls.AplsResult:
    return apls.apls_symmetric_score(gt, pred, roi="r", network_type="ROADS")


class TestProtocolConstants(unittest.TestCase):
    def test_values(self):
        self.assertEqual(apls.APLS_DENSIFY_M, 50.0)
        self.assertEqual(apls.APLS_SNAP_TO_EDGE_M, 4.0)
        self.assertEqual(apls.APLS_MIN_PATH_LENGTH_M, 5.0)
        self.assertEqual(
            apls.APLS_PROTOCOL,
            {
                "version": apls.APLS_PROTOCOL_VERSION,
                "densify_m": 50.0,
                "snap_to_edge_m": 4.0,
                "min_path_length_m": 5.0,
                "bidirectional": True,
                "aggregation": "harmonic_mean",
            },
        )


class TestPublicApiHasNoKnobs(unittest.TestCase):
    def test_public_signatures(self):
        for fn in (apls.apls_symmetric_score, apls.apls_pair_diagnostics):
            params = set(inspect.signature(fn).parameters)
            self.assertEqual(params, {"gt", "pred", "roi", "network_type"}, fn.__name__)

    def test_eval_entrypoint_has_no_apls_params(self):
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            import eval_network_apls
        finally:
            sys.path.pop(0)
        params = set(inspect.signature(eval_network_apls.run).parameters)
        self.assertFalse(params & set(_REMOVED_KEYS), params & set(_REMOVED_KEYS))
        parser_opts = {
            o for a in eval_network_apls.build_argparser()._actions for o in a.option_strings
        }
        for key in _REMOVED_KEYS:
            self.assertNotIn(f"--{key}", parser_opts)
        self.assertNotIn("--no_apls_symmetric", parser_opts)

    def test_no_repo_config_sets_fixed_keys(self):
        pattern = re.compile(r"^\s*(%s)\s*=" % "|".join(_REMOVED_KEYS), re.MULTILINE)
        offenders = [
            str(p.relative_to(REPO_ROOT))
            for p in (REPO_ROOT / "configs").rglob("*.py")
            if pattern.search(p.read_text())
        ]
        self.assertEqual(offenders, [])


class TestPredictionDefaultsMatchConfigs(unittest.TestCase):
    """eval_network_apls.run() defaults == the mask->graph recipe of the reference config."""

    KEYS = (
        "threshold", "overlap_combine", "connectivity", "rdp_epsilon_m", "endpoint_fix_enabled",
        "endpoint_fix_stage", "merge_hop_threshold", "radius_fix_radius_m",
        "remove_small_objects_enabled", "remove_small_objects_min_size_px", "skeletonize_enabled",
        "open_iterations", "close_iterations", "morph_connectivity", "min_component_nodes",
    )

    def test_defaults_equal_reference_config(self):
        from pointcept.utils.config import Config

        sys.path.insert(0, str(REPO_ROOT / "tools"))
        try:
            import eval_network_apls
        finally:
            sys.path.pop(0)
        cfg = Config.fromfile(str(REPO_ROOT / "configs/flair3d_default/multi-ptv3-v1m0-flair3d.py"))
        defaults = {
            k: v.default for k, v in inspect.signature(eval_network_apls.run).parameters.items()
        }
        for key in self.KEYS:
            self.assertEqual(defaults[key], cfg.network_apls_eval[key], key)


class TestConfigValidation(unittest.TestCase):
    def test_accepts_clean_and_absent(self):
        validate_network_apls_eval_cfg({})
        validate_network_apls_eval_cfg({"network_apls_eval": dict(threshold=0.2)})

    def test_rejects_each_removed_key(self):
        for key in _REMOVED_KEYS:
            with self.assertRaises(ValueError, msg=key):
                validate_network_apls_eval_cfg({"network_apls_eval": {key: 1}})


class TestProtocolBehaviour(unittest.TestCase):
    def test_identical_graphs_score_one(self):
        res = _score(_line_graph(0.0), _line_graph(0.0))
        self.assertAlmostEqual(res.score, 1.0, places=6)
        self.assertAlmostEqual(res.score_gt_to_pred, 1.0, places=6)
        self.assertAlmostEqual(res.score_pred_to_gt, 1.0, places=6)

    def test_offset_within_snap_radius_is_forgiven(self):
        self.assertGreater(_score(_line_graph(0.0), _line_graph(2.0)).score, 0.99)

    def test_offset_beyond_snap_radius_scores_zero(self):
        self.assertEqual(_score(_line_graph(0.0), _line_graph(10.0)).score, 0.0)

    def test_min_path_length_filters_short_pairs(self):
        # 4 m < 5 m: the only GT pair is ignored; 6 m >= 5 m: it is scored.
        self.assertEqual(_score(_line_graph(0.0, 4.0), _line_graph(0.0, 4.0)).denom, 0)
        self.assertGreater(_score(_line_graph(0.0, 6.0), _line_graph(0.0, 6.0)).denom, 0)

    def test_score_is_harmonic_mean_of_both_directions(self):
        # Pred covers only the first half of GT: pred->GT is perfect, GT->pred is not.
        gt = _line_graph(0.0, 100.0)
        pred = _line_graph(0.0, 50.0)
        res = _score(gt, pred)
        self.assertLess(res.score_gt_to_pred, res.score_pred_to_gt)
        a, b = res.score_gt_to_pred, res.score_pred_to_gt
        self.assertAlmostEqual(res.score, 2 * a * b / (a + b), places=9)

    def test_empty_graph_conventions(self):
        empty = apls.AplsGraph(
            node_xy=np.empty((0, 2)), edges=np.empty((0, 2), dtype=np.int64),
            edge_length_m=np.empty((0,)),
        )
        self.assertEqual(_score(empty, empty).score, 1.0)
        self.assertEqual(_score(_line_graph(0.0), empty).score, 0.0)
        self.assertEqual(_score(empty, _line_graph(0.0)).score, 0.0)

    def test_diagnostics_match_gt_to_pred_direction(self):
        gt, pred = _line_graph(0.0, 100.0), _line_graph(1.0, 60.0)
        diag = apls.apls_pair_diagnostics(gt, pred, roi="r", network_type="ROADS")
        sym = _score(gt, pred)
        self.assertIsNotNone(diag)
        self.assertAlmostEqual(diag.score, sym.score_gt_to_pred, places=9)


if __name__ == "__main__":
    unittest.main()
