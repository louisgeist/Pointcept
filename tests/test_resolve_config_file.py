"""Tests for config-path resolution (hyphenated filenames vs CWD).

Run with: PYTHONPATH=./ pytest tests/test_resolve_config_file.py
"""

import os
import tempfile
import unittest

from pointcept.engines.defaults import _legacy_hyphen_path, _resolve_config_file, _repo_root


class TestResolveConfigFile(unittest.TestCase):
    def test_hyphen_rewrite_splits_on_first_dash(self):
        self.assertEqual(
            _legacy_hyphen_path("scannet/semseg-pt-v1.py"),
            os.path.join("scannet/semseg", "pt-v1.py"),
        )
        self.assertIsNone(_legacy_hyphen_path("configs/scannet/semseg.py"))

    def test_hyphenated_repo_relative_path_is_not_rewritten(self):
        rel = "configs/experiment/w109/2/kpconv_bs/multi-kpconvx-v1m0-flair3d_1.py"
        expected = os.path.join(_repo_root(), rel)
        self.assertTrue(os.path.isfile(expected), f"fixture missing: {expected}")

        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                resolved = _resolve_config_file(rel)
        finally:
            os.chdir(cwd)

        self.assertEqual(os.path.abspath(resolved), os.path.abspath(expected))

    def test_missing_file_error_keeps_original_path(self):
        rel = "configs/experiment/w109/3/debug/sonata-v1m2-flair3d-lin-grid_2.py"
        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                with self.assertRaises(FileNotFoundError) as ctx:
                    _resolve_config_file(rel)
        finally:
            os.chdir(cwd)

        msg = str(ctx.exception)
        self.assertIn(rel, msg)
        self.assertNotIn(os.path.join("sonata", "v1m2-flair3d-lin-grid_2.py"), msg)

    def test_legacy_hyphen_shorthand_still_works_when_file_exists(self):
        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                nested = os.path.join(tmp, "scannet", "semseg")
                os.makedirs(nested)
                target = os.path.join(nested, "pt-v1.py")
                with open(target, "w", encoding="utf-8") as f:
                    f.write("# placeholder\n")
                os.chdir(tmp)
                resolved = os.path.abspath(_resolve_config_file("scannet/semseg-pt-v1.py"))
        finally:
            os.chdir(cwd)

        self.assertEqual(resolved, os.path.abspath(target))
