"""pytest bootstrap: lets the CPU-only tests import pointcept without compiled CUDA extensions.

`pointcept/datasets/__init__.py` and `pointcept/models/__init__.py` import every dataset and backbone, and some
of those need compiled extensions (spconv, pointops, KPConvX's cpp_subsampling). If that plain import fails,
this file registers empty package stand-ins for `pointcept`, `pointcept.datasets` and `pointcept.models` and
loads the real leaf modules the suite needs straight from their source files. Only the two package
`__init__.py` bodies are skipped; the code under test is never faked.

In a full environment the plain import succeeds and this file does nothing. Keep the `try/except ImportError`:
installing the stubs unconditionally would leave most backbones unregistered in `MODELS`/`DATASETS`.

Run the suite with: PYTHONPATH=. pytest tests/
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

# Resolve paths from this file so the fallback works whatever pytest's working directory is.
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
    datasets_pkg = _stub_pkg(
        "pointcept.datasets", str(_REPO_ROOT / "pointcept" / "datasets")
    )
    models_pkg = _stub_pkg("pointcept.models", str(_REPO_ROOT / "pointcept" / "models"))

    # --- pointcept.models: real leaf modules only, no backbones ---
    _load_real(
        "pointcept.models.utils", str(_REPO_ROOT / "pointcept" / "models" / "utils")
    )
    models_builder = _load_real(
        "pointcept.models.builder",
        str(_REPO_ROOT / "pointcept" / "models" / "builder.py"),
    )
    _load_real(
        "pointcept.models.losses", str(_REPO_ROOT / "pointcept" / "models" / "losses")
    )
    models_default = _load_real(
        "pointcept.models.default",
        str(_REPO_ROOT / "pointcept" / "models" / "default.py"),
    )
    models_pkg.build_model = models_builder.build_model
    models_pkg.MultiTaskSegmentorV2 = models_default.MultiTaskSegmentorV2

    # --- pointcept.datasets: real leaf modules only, no per-dataset backends ---
    datasets_defaults = _load_real(
        "pointcept.datasets.defaults",
        str(_REPO_ROOT / "pointcept" / "datasets" / "defaults.py"),
    )
    datasets_builder = _load_real(
        "pointcept.datasets.builder",
        str(_REPO_ROOT / "pointcept" / "datasets" / "builder.py"),
    )
    _load_real(
        "pointcept.datasets.transform",
        str(_REPO_ROOT / "pointcept" / "datasets" / "transform.py"),
    )
    _load_real(
        "pointcept.datasets.malibu3d_config_utils",
        str(_REPO_ROOT / "pointcept" / "datasets" / "malibu3d_config_utils.py"),
    )
    datasets_malibu3d = _load_real(
        "pointcept.datasets.malibu3d",
        str(_REPO_ROOT / "pointcept" / "datasets" / "malibu3d.py"),
    )
    datasets_utils = _load_real(
        "pointcept.datasets.utils",
        str(_REPO_ROOT / "pointcept" / "datasets" / "utils.py"),
    )
    datasets_pkg.build_dataset = datasets_builder.build_dataset
    datasets_pkg.collate_fn = datasets_utils.collate_fn
    datasets_pkg.point_collate_fn = datasets_utils.point_collate_fn
    datasets_pkg.MALiBU3DDataset = datasets_malibu3d.MALiBU3DDataset
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
