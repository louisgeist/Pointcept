"""
Preprocessing script for the OpenGF ground-filtering dataset
(https://github.com/Nathan-UW/OpenGF, Qin et al. CVPRW 2021 / ISPRS P&RS 2023).

Raw layout (unzip the distributed archives until it appears)::

  <dataset_root>/
    Training/<Terrain>/<Scene>/<Scene>_<idx>.laz   (151 files, ~500x500 m tiles)
    Validation/<Scene>_v.laz                        (9 files, one per training scene)
    Test/T{1,2,3}.laz                               (3 files, large irregular regions)

Usage (CPU only)::

  ln -sfn /path/to/OpenGF data/opengf/raw
  python pointcept/datasets/preprocessing/opengf/preprocess_opengf.py \
      --dataset_root data/opengf/raw \
      --output_root data/opengf \
      --num_workers 8

Writes per-scene folders under output_root/{train,val,test}/<scene_id>/:
  coord.npy, strength.npy, segment.npy, meta.json (with the float64 ``coord_translation``
  such that absolute xyz = coord + coord_translation)

Label semantics (LAS ``classification`` field, values {0, 1, 2} only):
  raw 2 = GR (bare earth, clean water)   -> stored segment 0 ("Ground")
  raw 1 = NG (buildings, vegetation, cars, other unclassified objects)
                                          -> stored segment 1 ("Non-ground")
  raw 0 = "Unclassified" (low/high outliers, Qin et al. CVPRW 2021 Sec 3.3)
                                          -> stored segment 2 ("Outlier")

Outliers are kept as their own on-disk label (not merged, not dropped) so configs
choose how to treat them:
  - ``RemapSegment(mapping={2: 1})`` merges outliers into Non-ground, the paper's
    training / "Test II (w outliers)" convention (Sec 4.4: "we merged class 0 into
    class 1, so that outliers are treated as NG points").
  - ``DropSegmentClass(labels=[2])`` physically removes outlier points before
    GridSample, as in "Test II (w/o outliers)" (Sec 4.5: outliers are "deleted",
    which changes the neighborhood of the remaining points, unlike ``ignore_index``).
  - ``ignore_index=2`` keeps outliers as context but excludes them from loss/metrics.
Only Test/T2.laz has many outlier points (Qin et al.: "Unlike Test I, there are a large
number of outliers in Test II"; T1 and T3 have none); Training/Validation tiles have a few.

``intensity`` is written as ``strength``. RGB fields, when present, hold a constant
dummy value and are not written (as for DALES).

Tiles are split with ``split_scene_xy_by_chunk_size`` (a physical tile size) rather
than a fixed N x N factor because Test regions are much larger and uneven (T1 ~2635 x
2504 m, T2 ~939 x 1226 m) than the uniform 500 x 500 m Training/Validation tiles. The
default 166.67 m (= 500 / 3) gives an exact 3 x 3 split of Training/Validation (like
DALES' chunking=3, ~500k points per subtile) and the same footprint on Test, which
keeps subtiles small enough for validation/test memory at grid_size 0.1.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Load chunking helper without importing pointcept.datasets (avoids torch deps).
_chunking_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "xy_grid_chunking.py"
)
_spec = importlib.util.spec_from_file_location("xy_grid_chunking", _chunking_path)
_chunking_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_chunking_mod)
split_scene_xy_by_chunk_size = _chunking_mod.split_scene_xy_by_chunk_size

try:
    import laspy
except ImportError as error:
    raise ImportError(
        "Please install 'laspy' to preprocess OpenGF LAZ files."
    ) from error

# Raw LAS classification {0, 1, 2} -> stored segment {2, 1, 0}, indexed by raw id.
RAW2SEGMENT = np.asarray([2, 1, 0], dtype=np.int32)


def build_scene(laz_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Return (scene, coord_translation); coord = absolute xyz - coord_translation."""
    las = laspy.read(laz_path)
    dim_names = set(las.point_format.dimension_names)

    xyz = np.stack(
        [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)],
        axis=1,
    ).astype(np.float64)
    # Recenter in float64 BEFORE the float32 cast: absolute OpenGF coordinates
    # (~1.3e6 / 5.0e6 m) have a float32 spacing of 0.125 m in X and 0.5 m in Y,
    # which would snap every point onto a coarse XY lattice.
    translation = np.floor(xyz.min(axis=0))
    coord = (xyz - translation).astype(np.float32)

    if "classification" not in dim_names:
        raise KeyError(f"Missing 'classification' field in {laz_path}")
    segment_raw = np.asarray(las.classification).astype(np.int32, copy=False)
    if np.any(segment_raw < 0) or np.any(segment_raw >= len(RAW2SEGMENT)):
        raise ValueError(
            f"Label out of range in {laz_path}. Expected raw classification in "
            f"[0, {len(RAW2SEGMENT) - 1}], got {sorted(set(segment_raw.tolist()))}."
        )
    segment = RAW2SEGMENT[segment_raw]

    if "intensity" not in dim_names:
        raise KeyError(f"Missing 'intensity' field in {laz_path}")
    strength = np.asarray(las.intensity).astype(np.float32)

    return {"coord": coord, "segment": segment, "strength": strength}, translation


def save_scene(
    output_scene_dir: str,
    scene: Dict[str, np.ndarray],
    meta: Dict[str, Any],
) -> None:
    os.makedirs(output_scene_dir, exist_ok=True)
    np.save(
        os.path.join(output_scene_dir, "coord.npy"), scene["coord"].astype(np.float32)
    )
    np.save(
        os.path.join(output_scene_dir, "strength.npy"),
        scene["strength"].astype(np.float32),
    )
    np.save(
        os.path.join(output_scene_dir, "segment.npy"), scene["segment"].astype(np.int32)
    )
    with open(os.path.join(output_scene_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def process_one_file(
    laz_path: str,
    output_root: str,
    split: str,
    scene_name: str,
    terrain: Optional[str],
    chunk_size: float,
) -> Tuple[str, int]:
    scene, translation = build_scene(laz_path=laz_path)
    scene_id = os.path.splitext(os.path.basename(laz_path))[0]
    meta = {
        "split": split,
        "scene": scene_name,
        "terrain": terrain,
        "source_file": os.path.basename(laz_path),
        # absolute xyz = coord + coord_translation (float64, whole source file)
        "coord_translation": translation.tolist(),
    }
    sub_scenes = split_scene_xy_by_chunk_size(scene=scene, chunk_size=chunk_size)
    single_tile = len(sub_scenes) == 1
    written = 0
    for suffix, sub_scene in sub_scenes:
        # "0-0" also occurs as a genuine grid cell when chunk_size actually splits the
        # scene, not just as the no-op sentinel -- key off len(sub_scenes), not the label.
        tiled_scene_id = scene_id if single_tile else f"{scene_id}_{suffix}"
        output_scene_dir = os.path.join(output_root, split, tiled_scene_id)
        save_scene(output_scene_dir, sub_scene, meta)
        written += 1
    return scene_id, written


def collect_jobs(dataset_root: str) -> List[Tuple[str, str, str, Optional[str]]]:
    """Return (laz_path, split, scene_name, terrain) for every raw file."""
    jobs: List[Tuple[str, str, str, Optional[str]]] = []

    for laz_path in sorted(
        glob.glob(os.path.join(dataset_root, "Training", "*", "*", "*.laz"))
    ):
        terrain = os.path.basename(os.path.dirname(os.path.dirname(laz_path)))
        scene_name = os.path.basename(os.path.dirname(laz_path))
        jobs.append((laz_path, "train", scene_name, terrain))

    for laz_path in sorted(
        glob.glob(os.path.join(dataset_root, "Validation", "*.laz"))
    ):
        scene_name = os.path.splitext(os.path.basename(laz_path))[0]
        jobs.append((laz_path, "val", scene_name, None))

    for laz_path in sorted(glob.glob(os.path.join(dataset_root, "Test", "*.laz"))):
        scene_name = os.path.splitext(os.path.basename(laz_path))[0]
        jobs.append((laz_path, "test", scene_name, None))

    return jobs


def main_process():
    parser = argparse.ArgumentParser(
        description="Preprocess OpenGF LAZ tiles for Pointcept."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Directory containing Training/, Validation/, Test/.",
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument(
        "--chunk_size",
        default=500.0 / 3,
        type=float,
        help=(
            "Physical target tile size in meters (adaptive nx x ny split, see "
            "split_scene_xy_by_chunk_size). Default 166.67 m = an exact 3x3 split on "
            "Training/Validation's uniform 500x500 m tiles (DALES' own chunking=3 "
            "convention), and the same absolute footprint on Test's larger, unevenly "
            "sized regions -- keeps every split's per-tile point count in the same "
            "ballpark instead of a fixed N x N factor over-sizing Test's subtiles."
        ),
    )
    parser.add_argument(
        "--max_files",
        default=None,
        type=int,
        help="Optional cap on number of source files (smoke tests).",
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive.")

    jobs = collect_jobs(args.dataset_root)
    if args.max_files is not None:
        jobs = jobs[: args.max_files]

    os.makedirs(args.output_root, exist_ok=True)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(args.output_root, split), exist_ok=True)

    total = len(jobs)
    print(f"Found {total} source LAZ file(s) under {args.dataset_root}")
    print(f"chunk_size={args.chunk_size}, num_workers={args.num_workers}")
    if total == 0:
        return

    scenes_written = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [
            pool.submit(
                process_one_file,
                laz_path,
                args.output_root,
                split,
                scene_name,
                terrain,
                args.chunk_size,
            )
            for laz_path, split, scene_name, terrain in jobs
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            _, n_written = future.result()
            scenes_written += n_written
            print(f"\rProgress: {idx}/{total}", end="", flush=True)
    print()
    print(
        f"Done. {scenes_written} scene folder(s) after chunking "
        f"({total} source LAZ file(s)) -> {args.output_root}"
    )


if __name__ == "__main__":
    main_process()
