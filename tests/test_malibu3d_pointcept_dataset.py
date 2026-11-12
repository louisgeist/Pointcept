"""MALiBU3D land-cover dataset shipped under pointcept-dataset/.

The module is loaded without registering another ``MALiBU3DDataset``: that name
belongs to the multitask dataset. Instantiate ``module.MALiBU3DDataset`` directly.

Run with: PYTHONPATH=. pytest tests/test_malibu3d_pointcept_dataset.py
"""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pointcept.datasets.builder import DATASETS
from pointcept.datasets.malibu3d import MALiBU3DDataset as MultitaskMALiBU3DDataset

_DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "pointcept-dataset"
    / "pointcept"
    / "datasets"
    / "malibu3d.py"
)
_MODULE = None


def pointcept_dataset():
    """Load the Pointcept dataset module once, under a registry name that is not MALiBU3DDataset."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    original = DATASETS.register_module

    def redirected(name=None, force=False, module=None):
        return original(name="MALiBU3DLandCoverDataset", force=True, module=module)

    DATASETS.register_module = redirected
    try:
        spec = importlib.util.spec_from_file_location(
            "malibu3d_pointcept_dataset", _DATASET_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        DATASETS.register_module = original
    assert DATASETS.get("MALiBU3DDataset") is MultitaskMALiBU3DDataset
    assert DATASETS.get("MALiBU3DLandCoverDataset") is module.MALiBU3DDataset
    _MODULE = module
    return module


def _write_tile(root, split, tile_id, segment):
    path = root / split / "D001-2020_LIDARHD" / "AA" / tile_id
    path.mkdir(parents=True)
    n = len(segment)
    coord = np.zeros((n, 3), dtype=np.float32)
    coord[:, 0] = np.arange(n, dtype=np.float32)
    np.save(path / "coord.npy", coord)
    np.save(path / "segment.npy", np.asarray(segment, dtype=np.uint8))
    np.save(path / "strength.npy", np.linspace(0, 1, n, dtype=np.float32))
    np.save(path / "natural_habitat.npy", np.zeros((n, 4), dtype=np.uint8))
    return path


def _write_catalog(root, rows):
    path = root / "tiles.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "dept_year", "roi", "tile_id", "n_points"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": row[0],
                    "dept_year": "D001-2020",
                    "roi": "AA",
                    "tile_id": row[1],
                    "n_points": row[2],
                }
            )
    return path


def _dataset(root, split, **kwargs):
    return pointcept_dataset().MALiBU3DDataset(
        split=split,
        data_root=str(root),
        csv_manifest=str(root / "tiles.csv"),
        **kwargs,
    )


def _names(dataset):
    return sorted(Path(path).name for path in dataset.data_list)


def test_min_points_filters_train_only(tmp_path):
    segment = [0, 15, 1]
    for tile_id in ("train_keep", "train_drop", "val_small", "test_small"):
        _write_tile(tmp_path, tile_id.split("_")[0], tile_id, segment)
    _write_catalog(
        tmp_path,
        [
            ("train", "train_keep", 1000),
            ("train", "train_drop", 999),
            ("val", "val_small", 10),
            ("test", "test_small", 1),
        ],
    )

    train = _dataset(tmp_path, "train")
    val = _dataset(tmp_path, "val")
    test = _dataset(tmp_path, "test")

    assert _names(train) == ["train_keep"]
    assert _names(val) == ["val_small"]
    assert _names(test) == ["test_small"]
    assert train.ignore_index == 15
    assert train.min_points == {"train": 1000}

    data = train.get_data(0)
    assert data["segment"].tolist() == segment
    assert "natural_habitat" not in data
    assert train.get_split_name(0) == "train"


def test_missing_tile_directory_raises(tmp_path):
    _write_tile(tmp_path, "train", "present", [0, 1])
    _write_catalog(
        tmp_path,
        [
            ("train", "present", 2000),
            ("train", "ghost", 2000),
        ],
    )
    with pytest.raises(FileNotFoundError, match="ghost"):
        _dataset(tmp_path, "train")


def test_subset_manifest_filters_by_tile_id_and_split(tmp_path):
    for tile_id in ("val_a", "val_b"):
        _write_tile(tmp_path, "val", tile_id, [0, 15])
    _write_catalog(
        tmp_path,
        [
            ("val", "val_a", 50),
            ("val", "val_b", 50),
        ],
    )
    subset = tmp_path / "subset.csv"
    subset.write_text("split,tile_id\nval,val_a\ntrain,val_b\n", encoding="utf-8")

    dataset = _dataset(tmp_path, "val", subset_manifest=str(subset))
    assert _names(dataset) == ["val_a"]


def test_segment_ids_outside_0_15_raise(tmp_path):
    _write_tile(tmp_path, "train", "bad", [0, 16])
    _write_catalog(tmp_path, [("train", "bad", 2000)])
    dataset = _dataset(tmp_path, "train")
    with pytest.raises(ValueError, match="outside"):
        dataset.get_data(0)


def test_min_points_rejects_test_split():
    dataset_cls = pointcept_dataset().MALiBU3DDataset
    with pytest.raises(ValueError, match="test"):
        dataset_cls(min_points={"test": 1}, split="test", data_root="unused")


def test_prepare_feat_scales_coord_and_reshapes_strength():
    transform = pointcept_dataset().MALiBU3DPrepareFeat(coord_scale=0.01)
    coord = np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 5.0]], dtype=np.float32)
    strength = np.array([0.25, 0.5], dtype=np.float32)
    out = transform({"coord": coord.copy(), "strength": strength})

    np.testing.assert_array_equal(out["coord"], coord)
    np.testing.assert_allclose(out["coord_feat"], coord * np.float32(0.01))
    assert out["strength"].shape == (2, 1)
    np.testing.assert_allclose(out["strength"][:, 0], strength)


def test_registry_keeps_the_multitask_dataset():
    pointcept_dataset()
    assert DATASETS.get("MALiBU3DDataset") is MultitaskMALiBU3DDataset
    assert "target_keys" in MultitaskMALiBU3DDataset.__init__.__code__.co_varnames
