"""dilated_prf_enabled: per-task switch for buffer ("relaxed") precision/recall/F1, on for thin curvilinear
pixel_semantic tasks (network) and off for area-coverage ones (forest_2d).

Run with: PYTHONPATH=. pytest tests/test_dilated_prf_opt_out.py
"""

import unittest

from pointcept.datasets.malibu3d_config_utils import get_task_config
from pointcept.utils.dilated_metrics import dilated_prf_enabled


class TestDilatedPrfEnabled(unittest.TestCase):
    def test_defaults_to_true_when_key_absent(self):
        self.assertTrue(dilated_prf_enabled({}))
        self.assertTrue(dilated_prf_enabled({"task_type": "pixel_semantic"}))

    def test_false_when_explicitly_disabled(self):
        self.assertFalse(dilated_prf_enabled({"enable_dilated_prf": False}))

    def test_true_when_explicitly_enabled(self):
        self.assertTrue(dilated_prf_enabled({"enable_dilated_prf": True}))

    def test_network_enabled_forest_2d_opted_out(self):
        self.assertTrue(dilated_prf_enabled(get_task_config("network")))
        self.assertFalse(dilated_prf_enabled(get_task_config("forest_2d")))


if __name__ == "__main__":
    unittest.main()
