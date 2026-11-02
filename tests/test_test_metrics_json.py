"""TesterBase writes the test metrics (per-class included) to save_path/test_metrics.json."""
import json
import logging
import os
import tempfile
import time
import unittest
from types import SimpleNamespace

import numpy as np

from pointcept.engines.test import TesterBase as _TesterBase


def _tester(save_path):
    tester = _TesterBase.__new__(_TesterBase)
    tester.cfg = SimpleNamespace(save_path=save_path, weight="model/model_best.pth")
    tester.logger = logging.getLogger("test_metrics_json")
    tester.checkpoint_epoch = 7
    tester._test_wall_start = time.perf_counter()
    return tester


class TestMetricsJson(unittest.TestCase):
    def test_writes_per_class_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            log_dict = {
                "test/segment/mIoU": 0.71,
                "test/segment/iou/building": np.float32(0.9),
                "test/reg/elevation/mae": np.float64(1.02),
            }
            _tester(d).end_test_timing_and_log(extra_log_dict=log_dict)
            with open(f"{d}/test_metrics.json") as f:
                out = json.load(f)
            self.assertAlmostEqual(out["test/segment/iou/building"], 0.9, places=6)
            self.assertAlmostEqual(out["test/reg/elevation/mae"], 1.02)
            self.assertEqual(out["epoch"], 7)
            self.assertEqual(out["weight"], "model/model_best.pth")
            self.assertIn("test/s", out)

    def test_no_file_without_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            _tester(d).end_test_timing_and_log()
            self.assertFalse(os.path.exists(f"{d}/test_metrics.json"))


if __name__ == "__main__":
    unittest.main()
