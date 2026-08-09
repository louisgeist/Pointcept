"""
pytest bootstrap: make the handful of pointcept modules these tests need
importable without a full CUDA/spconv/pointops/custom-C++-extension build.

Why this exists: `pointcept/datasets/__init__.py` and `pointcept/models/__init__.py`
eagerly import *every* dataset and *every* backbone (nuScenes, Waymo, KPConvX,
SparseUNet, ...). Several of those transitively require a compiled CUDA
extension (spconv, pointops, KPConvX's cpp_subsampling) that this environment
does not have installed (see docs/superpowers/plans/2026-08-09-forest-2d-task.md
for context: no local machine currently has the project's real
`pointcept-torch2.5.0-cu12.4` conda env). None of that machinery is needed to
test the forest_2d task's pure-Python/CPU-tensor logic.

This module pre-registers lightweight stand-ins for `pointcept`,
`pointcept.datasets`, and `pointcept.models` in `sys.modules` *before* pytest
imports any test file, then loads the small set of real, needed leaf modules
directly from their source files (bypassing the heavy package `__init__.py`
files) and re-attaches the names other real modules expect to find on the
stubbed packages (e.g. `pointcept.datasets.build_dataset`). Everything loaded
this way is the *real* implementation — only the two package `__init__.py`
bodies are skipped, not any of the logic under test.

**Safety property — this is a fallback, not a replacement:** on a machine with
the real environment (spconv/pointops/etc. all installed, e.g. the project's
actual `pointcept-torch2.5.0-cu12.4` conda env), `import pointcept.datasets`
and `import pointcept.models` below succeed normally on the *first* try, so
this module does nothing at all — the real `__init__.py` files run exactly as
they always have, and every backbone/dataset still registers itself in the
`MODELS`/`DATASETS` registries. The stubbing path only activates when the
plain import fails (e.g. `ModuleNotFoundError: spconv`), and even then it
loads the *real* leaf modules this test suite actually needs — it never fakes
behavior for code under test, only skips two package `__init__.py` bodies
whose only job here would be registering backbones nothing in this suite
uses. Do not remove the `try/except ImportError` below — without it, this
file would silently break `MODELS.build(...)`/`DATASETS.build(...)` for every
other backbone on a real GPU machine where the plain import would have
worked fine.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

# tests/conftest.py is one level below the repo root; resolve paths from here
# so the fallback stubs work regardless of pytest's invocation cwd (e.g.
# `cd /tmp && pytest /path/to/repo/tests/`), not just when run from the repo
# root.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _stub_pkg(name, path):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    sys.modules[name] = mod
    return mod


def _load_real(name, path):
    if name in sys.modules:
        return sys.modules[name]
    is_pkg = os.path.isdir(path)
    file_path = os.path.join(path, "__init__.py") if is_pkg else path
    submodule_locations = [path] if is_pkg else None
    spec = importlib.util.spec_from_file_location(
        name, file_path, submodule_search_locations=submodule_locations
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_fallback_stubs():
    _stub_pkg("pointcept", str(_REPO_ROOT / "pointcept"))
    datasets_pkg = _stub_pkg("pointcept.datasets", str(_REPO_ROOT / "pointcept" / "datasets"))
    models_pkg = _stub_pkg("pointcept.models", str(_REPO_ROOT / "pointcept" / "models"))

    # --- pointcept.models: real leaf modules only, no backbones ---
    _load_real("pointcept.models.utils", str(_REPO_ROOT / "pointcept" / "models" / "utils"))
    models_builder = _load_real(
        "pointcept.models.builder", str(_REPO_ROOT / "pointcept" / "models" / "builder.py")
    )
    _load_real("pointcept.models.losses", str(_REPO_ROOT / "pointcept" / "models" / "losses"))
    models_default = _load_real(
        "pointcept.models.default", str(_REPO_ROOT / "pointcept" / "models" / "default.py")
    )
    models_pkg.build_model = models_builder.build_model
    models_pkg.MultiTaskSegmentorV2 = models_default.MultiTaskSegmentorV2

    # --- pointcept.datasets: real leaf modules only, no per-dataset backends ---
    datasets_defaults = _load_real(
        "pointcept.datasets.defaults", str(_REPO_ROOT / "pointcept" / "datasets" / "defaults.py")
    )
    datasets_builder = _load_real(
        "pointcept.datasets.builder", str(_REPO_ROOT / "pointcept" / "datasets" / "builder.py")
    )
    _load_real(
        "pointcept.datasets.transform", str(_REPO_ROOT / "pointcept" / "datasets" / "transform.py")
    )
    _load_real(
        "pointcept.datasets.flair3d_config_utils",
        str(_REPO_ROOT / "pointcept" / "datasets" / "flair3d_config_utils.py"),
    )
    datasets_flair3d = _load_real(
        "pointcept.datasets.flair3d", str(_REPO_ROOT / "pointcept" / "datasets" / "flair3d.py")
    )
    datasets_utils = _load_real(
        "pointcept.datasets.utils", str(_REPO_ROOT / "pointcept" / "datasets" / "utils.py")
    )
    datasets_pkg.build_dataset = datasets_builder.build_dataset
    datasets_pkg.collate_fn = datasets_utils.collate_fn
    datasets_pkg.point_collate_fn = datasets_utils.point_collate_fn
    datasets_pkg.Flair3DDataset = datasets_flair3d.Flair3DDataset
    datasets_pkg.DefaultDataset = datasets_defaults.DefaultDataset


try:
    import pointcept.datasets  # noqa: F401
    import pointcept.models  # noqa: F401
except ImportError as exc:
    print(
        f"tests/conftest.py: real import failed ({exc!r}); "
        "falling back to lightweight leaf-module stubs for this test session.",
        file=sys.stderr,
    )
    _install_fallback_stubs()
