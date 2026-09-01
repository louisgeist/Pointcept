"""
Tests for the nathab ecological-axis LabelDefinitions and the Malibu3DLabelRemap
fan-out extension (one on-disk source field -> several differently-named
per-axis output fields in a single transform call).

Run with: PYTHONPATH=./ pytest tests/test_nathab_axis_label_remap.py
"""

import unittest

import numpy as np

from pointcept.datasets.preprocessing.malibu3d_plus import malibu3d_label_remap as fr
from pointcept.datasets.transform import Malibu3DLabelRemap


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


class TestMalibu3DLabelRemapFanOut(unittest.TestCase):
    def _make_fanout_remap(self):
        return Malibu3DLabelRemap(
            remaps=dict(
                natural_habitat="default",
                nathab_habitat_type=("natural_habitat", "by_habitat_type_ecological"),
                nathab_moisture_regime=("natural_habitat", "by_moisture_regime"),
                nathab_soil_chemistry=("natural_habitat", "by_soil_chemistry"),
                nathab_bioclimatic_zone=("natural_habitat", "by_climatic_domain"),
            ),
            storage_definitions=dict(natural_habitat="default"),
        )

    def test_source_key_left_untouched(self):
        remap = self._make_fanout_remap()
        raw = np.array([0, 6, 13, 25, 36, 37, 40, 42, 43], dtype=np.int32)
        data = {"natural_habitat": raw.copy()}
        out = remap(data)
        np.testing.assert_array_equal(out["natural_habitat"], raw)

    def test_fanout_values_match_hand_derived_luts(self):
        remap = self._make_fanout_remap()
        raw = np.array([0, 6, 13, 25, 36, 37, 40, 42, 43], dtype=np.int32)
        out = remap({"natural_habitat": raw.copy()})
        np.testing.assert_array_equal(
            out["nathab_habitat_type"], [0, 1, 0, 0, 2, 2, 4, 4, 4]
        )
        np.testing.assert_array_equal(
            out["nathab_moisture_regime"], [0, 0, 1, 1, 3, 3, 3, 3, 3]
        )
        np.testing.assert_array_equal(
            out["nathab_soil_chemistry"], [0, 0, 0, 0, 0, 1, 2, 2, 2]
        )
        np.testing.assert_array_equal(
            out["nathab_bioclimatic_zone"], [0, 0, 1, 2, 3, 3, 3, 3, 3]
        )

    def test_fanout_reads_pristine_value_even_when_source_also_remapped_in_place(self):
        # Regression test for the fan-out-before-in-place ordering guarantee:
        # if the same source key is ALSO remapped in place in the same
        # transform instance, fan-out outputs must still be derived from the
        # untouched on-disk (storage-space) value, not the in-place output.
        remap = Malibu3DLabelRemap(
            remaps=dict(
                natural_habitat="by_moisture_v3",  # in-place mutation
                nathab_habitat_type=("natural_habitat", "by_habitat_type_ecological"),
            ),
            storage_definitions=dict(natural_habitat="default"),
        )
        data = {"natural_habitat": np.array([6], dtype=np.int32)}  # forest/temperate
        out = remap(data)
        # in-place: raw 6 -> by_moisture_v3 (id % 3 = 0, "Humide")
        self.assertEqual(int(out["natural_habitat"][0]), 0)
        # fan-out: must still see pristine raw=6 -> forest (1), not the
        # already-mutated in-place value (0, which would incorrectly resolve
        # to "open" if read after mutation).
        self.assertEqual(int(out["nathab_habitat_type"][0]), 1)

    def test_bare_string_in_place_form_unchanged(self):
        # Existing single-key in-place usage (e.g. nathab_moisture configs)
        # must keep working exactly as before this extension.
        remap = Malibu3DLabelRemap(
            remaps={"natural_habitat": "by_moisture_v3"},
            storage_definitions={"natural_habitat": "default"},
        )
        data = {"natural_habitat": np.array([0, 3, 6, 42, 43], dtype=np.int32)}
        out = remap(data)
        np.testing.assert_array_equal(out["natural_habitat"], [0, 0, 0, 5, 5])

    def test_invalid_spec_shape_raises(self):
        with self.assertRaises(ValueError):
            Malibu3DLabelRemap(remaps={"bad_key": (1, 2, 3)})


if __name__ == "__main__":
    unittest.main()
