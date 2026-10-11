"""
Tests for the natural-habitat ecological-axis label definitions and the CarHab -> axes bake.

Run with: PYTHONPATH=./ pytest tests/test_nathab_axis_label_remap.py
"""

import unittest

import numpy as np

from pointcept.datasets.preprocessing.malibu3d import malibu3d_label_remap as fr
from pointcept.datasets.preprocessing.malibu3d import nathab_axes as na


class TestNathabAxesHelper(unittest.TestCase):
    def test_luts_match_stored_to_train(self):
        storage = fr.get_definition("natural_habitat", "default")
        for lut, definition in zip(na.NATHAB_AXIS_LUTS, na.NATHAB_AXIS_DEFINITIONS):
            target = fr.get_definition("natural_habitat", definition)
            expected = fr.build_stored_to_train_lut(storage, target)
            np.testing.assert_array_equal(
                np.asarray(lut, dtype=np.int32),
                expected,
                err_msg=f"LUT mismatch for {definition}",
            )

    def test_carhab_to_axes_matches_fanout(self):
        raw = np.array([0, 6, 13, 25, 36, 37, 40, 42, 43], dtype=np.int32)
        axes = na.carhab_to_nathab_axes(raw)
        self.assertEqual(axes.dtype, np.uint8)
        self.assertEqual(axes.shape, (raw.shape[0], 4))
        unpacked = na.unpack_nathab_axes(axes)
        np.testing.assert_array_equal(
            unpacked["nathab_habitat_type"], [0, 1, 0, 0, 2, 2, 4, 4, 4]
        )
        np.testing.assert_array_equal(
            unpacked["nathab_moisture_regime"], [0, 0, 1, 1, 3, 3, 3, 3, 3]
        )
        np.testing.assert_array_equal(
            unpacked["nathab_soil_chemistry"], [0, 0, 0, 0, 0, 1, 2, 2, 2]
        )
        np.testing.assert_array_equal(
            unpacked["nathab_bioclimatic_zone"], [0, 0, 1, 2, 3, 3, 3, 3, 3]
        )

    def test_already_baked_axes_validated(self):
        baked = np.array([[0, 0, 0, 0], [1, 2, 1, 2]], dtype=np.uint8)
        out = na.carhab_to_nathab_axes(baked)
        np.testing.assert_array_equal(out, baked)

    def test_rejects_out_of_range_carhab(self):
        with self.assertRaises(ValueError):
            na.carhab_to_nathab_axes(np.array([44], dtype=np.int32))


class TestNathabAxisLabelDefinitions(unittest.TestCase):
    def test_habitat_type_ecological(self):
        d = fr.get_definition("natural_habitat", "by_habitat_type_ecological")
        self.assertEqual(d.ignore_index, 4)
        self.assertEqual(d.names, ("Open", "Forest", "Mineral", "Aquatic", "Void"))
        # open ids (e.g. 0), forest ids (e.g. 6), mineral (36/37), aquatic (38/39),
        # everything else (cultivated=40, built=41, N/A=42, routes=43) -> void.
        self.assertEqual(d.lut[0], 0)
        self.assertEqual(d.lut[6], 1)
        self.assertEqual(d.lut[36], 2)
        self.assertEqual(d.lut[37], 2)
        self.assertEqual(d.lut[38], 3)
        self.assertEqual(d.lut[39], 3)
        for raw_id in (40, 41, 42, 43):
            self.assertEqual(d.lut[raw_id], 4)

    def test_soil_chemistry(self):
        d = fr.get_definition("natural_habitat", "by_soil_chemistry")
        self.assertEqual(d.ignore_index, 2)
        self.assertEqual(d.names, ("Acidic", "Alkaline", "Void"))
        # substrate parity for ids 0-35 (block of 3: acid,acid,acid,basic,basic,basic,...)
        self.assertEqual(d.lut[0], 0)
        self.assertEqual(d.lut[3], 1)
        # mineral/aquatic acid/basic pairs
        self.assertEqual(d.lut[36], 0)  # mineral acid
        self.assertEqual(d.lut[37], 1)  # mineral basic
        self.assertEqual(d.lut[38], 0)  # aquatic acid
        self.assertEqual(d.lut[39], 1)  # aquatic basic
        for raw_id in (40, 41, 42, 43):
            self.assertEqual(d.lut[raw_id], 2)

    def test_moisture_regime(self):
        d = fr.get_definition("natural_habitat", "by_moisture_regime")
        self.assertEqual(d.ignore_index, 3)
        self.assertEqual(d.names, ("Humide", "Mesique", "Sec", "Void"))
        for raw_id in range(36):
            self.assertEqual(d.lut[raw_id], raw_id % 3)
        for raw_id in (36, 37, 38, 39, 40, 41, 42, 43):
            self.assertEqual(d.lut[raw_id], 3)

    def test_bioclimatic_zone_reuses_by_climatic_domain(self):
        d = fr.get_definition("natural_habitat", "by_climatic_domain")
        self.assertEqual(d.names, ("Temperate", "Mediterranean", "Alpine", "Void"))


if __name__ == "__main__":
    unittest.main()
