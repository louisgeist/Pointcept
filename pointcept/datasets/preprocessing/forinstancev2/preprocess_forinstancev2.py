"""
Preprocessing script for the FOR-instance V2 dataset (individual-tree LiDAR
benchmark; multi-country / multi-sensor forest plots + one large merged
terrestrial/mobile scan contributed by "Yuchen").

Raw layout (as staged on hecate, `/data/geist/datasets/ForInstanceV2/`)::

    <dataset_root>/
        train_val_data/*.ply   (train + val, split encoded in filename)
        test_data/*.ply        (test)

Each PLY is binary little-endian with per-vertex properties
``x, y, z, semantic_seg, treeID`` (confirmed on every file in a local
download 2026-09) -- no color/intensity, so this is geometry-only.

Filenames follow ``<source>_<original_name>_<split>.ply``, e.g.
``NIBIO2_NIBIO2_plot12_annotated_train.ply`` (source=NIBIO2) or
``Yuchen_2023_dls_merged_230209_panoptic_val.ply`` (source=Yuchen). ``source``
identifies the contributing site/sensor (`BlueCat`, `CULS`, `NIBIO`,
`NIBIO2`, `NIBIO_MLS`, `RMIT`, `SCION`, `TUWIEN`, `Yuchen`) and is written to
each scene's ``meta.json`` so ``ForInstanceV2Dataset(sources=[...])`` can
filter by it at train time (e.g. `sources=["Yuchen"]` for the one ULS-style
site, `Yuchen`, kept easy to select on its own per the maintainer's request).

Label semantics (`semantic_seg`, values 1-3 confirmed across the dataset,
consistent with the FOR-instance benchmark paper's convention -- verify
against your own copy if this ever changes):
    1 = ground      -> stored segment 0
    2 = low vegetation -> stored segment 1
    3 = tree (stem + crown) -> stored segment 2
`treeID` (0 = no tree instance, i.e. ground; >0 = per-tree instance id) is
kept as-is in `instance.npy` for future instance/panoptic use -- current
configs only consume `segment`.

`BlueCat_RN_merged_trees_*` is a single merged terrestrial/mobile scan far
denser and larger than every other file (~16,000 pts/m^2 over ~25,000 m^2,
407M points for the train split alone, vs <6M points / <2,000 pts/m^2 for
every other source) -- writing it out raw would make multi-GB .npy files.
Files above `--big_file_threshold` raw points get BOTH treatments together
(checked once, per source file -- i.e. per split -- so BlueCat's train/val/
test all get the same treatment even though their point counts differ
enough that one alone might fall under the threshold post-voxelization):

1. Voxel-subsampled at `--voxel_size` (default 0.1m, matching every
   downstream config's `grid_size`): one real point kept per occupied
   voxel, picked uniformly at random -- the exact same selection
   `GridSample(mode="train")` does at train time (see
   `pointcept/datasets/transform.py`), just run once offline instead of on
   every epoch. This is *not* how Pointcept preprocessing normally works
   (datasets are usually kept at native resolution and voxelized live), but
   is necessary here purely to bring the on-disk size down; since a voxel
   holds exactly one point afterwards, a live `GridSample` at the same grid
   size is a no-op, so downstream training behaves identically to the
   unvoxelized case (measured ~10.6x point reduction on a sample file).
2. XY-chunked into `--chunk_size`-meter tiles via
   `split_scene_xy_by_chunk_size` (same helper DALES/H3D/ECLAIR use),
   purely to keep individual scene folders in the same ballpark as the
   rest of the dataset. A regular grid over an irregularly-shaped point
   cloud (BlueCat's footprint is a diagonal scan swath, not a filled
   rectangle -- confirmed by rasterizing its raw XY density on a local
   download) leaves some edge/corner tiles nearly empty; tiles with fewer
   than `--min_points_per_tile` points after chunking are dropped.

Every other (already plot-sized) file is kept as a single, unvoxelized,
unchunked scene folder, matching how those datasets are laid out upstream.

Usage:
    ln -sfn /data/geist/datasets/ForInstanceV2 data/forinstancev2/raw
    python pointcept/datasets/preprocessing/forinstancev2/preprocess_forinstancev2.py \
        --dataset_root data/forinstancev2/raw \
        --output_root data/forinstancev2 \
        --num_workers 8

Writes per-scene folders under output_root/{train,val,test}/<scene_id>/:
    coord.npy, segment.npy, instance.npy, meta.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import importlib.util

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
    from plyfile import PlyData
except ImportError as error:
    raise ImportError(
        "Please install 'plyfile' to preprocess ForInstanceV2 PLY files."
    ) from error

# Longest-prefix-first so compound prefixes (e.g. NIBIO_MLS) win over their
# shorter substring (NIBIO). Matched against "<prefix>_" at the filename start.
KNOWN_SOURCES = sorted(
    ["BlueCat", "CULS", "NIBIO2", "NIBIO_MLS", "NIBIO", "RMIT", "SCION", "TUWIEN", "Yuchen"],
    key=len,
    reverse=True,
)

SEGMENT_ID2TRAINID = {1: 0, 2: 1, 3: 2}
CLASS_NAMES = ["Ground", "Low vegetation", "Tree"]

VALID_SPLITS = ("train", "val", "test")


def voxelize_scene(
    scene: Dict[str, np.ndarray], voxel_size: float, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    """Keep one real point per occupied `voxel_size` voxel, picked uniformly at random.

    Mirrors `GridSample(mode="train")`'s point-selection exactly (see
    `pointcept/datasets/transform.py`): points are grouped by integer voxel
    coordinate, one index per group is drawn at random. No averaging/majority
    voting -- every kept point (and its segment/instance label) is untouched,
    only redundant points within the same voxel are dropped.
    """
    coord = scene["coord"]
    grid_coord = np.floor(coord / voxel_size).astype(np.int64)
    grid_coord -= grid_coord.min(0)
    dims = grid_coord.max(0) + 1
    key = (grid_coord[:, 0] * dims[1] + grid_coord[:, 1]) * dims[2] + grid_coord[:, 2]

    idx_sort = np.argsort(key, kind="stable")
    key_sort = key[idx_sort]
    _, start_idx, count = np.unique(key_sort, return_index=True, return_counts=True)
    idx_select = start_idx + (rng.integers(0, count.max(), size=count.size) % count)
    idx_unique = idx_sort[idx_select]

    return {k: v[idx_unique] for k, v in scene.items()}


def parse_source_and_split(filename: str) -> Tuple[str, str]:
    """Parse `<source>_..._<split>.ply` -> (source, split)."""
    stem = os.path.splitext(filename)[0]
    source = next((s for s in KNOWN_SOURCES if stem.startswith(s + "_")), None)
    if source is None:
        raise ValueError(
            f"Could not identify source for {filename!r}; known sources: {KNOWN_SOURCES}"
        )
    split = stem.rsplit("_", 1)[-1]
    if split not in VALID_SPLITS:
        raise ValueError(f"Could not identify split (train/val/test) for {filename!r}")
    return source, split


def build_scene(ply_path: str) -> Dict[str, np.ndarray]:
    ply_data = PlyData.read(ply_path)
    element = ply_data.elements[0].data
    attributes = {name: np.asarray(element[name]) for name in element.dtype.names}

    for axis in ("x", "y", "z"):
        if axis not in attributes:
            raise KeyError(f"Missing '{axis}' in {ply_path}")
    coord = np.stack(
        [attributes["x"], attributes["y"], attributes["z"]], axis=1
    ).astype(np.float32)

    if "semantic_seg" not in attributes:
        raise KeyError(f"Missing 'semantic_seg' field in {ply_path}")
    segment_raw = attributes["semantic_seg"].astype(np.int32, copy=False)
    unknown = set(np.unique(segment_raw).tolist()) - set(SEGMENT_ID2TRAINID)
    if unknown:
        raise ValueError(
            f"Unexpected semantic_seg value(s) {sorted(unknown)} in {ply_path}; "
            f"expected only {sorted(SEGMENT_ID2TRAINID)}."
        )
    segment = np.vectorize(SEGMENT_ID2TRAINID.get)(segment_raw).astype(np.int32)

    if "treeID" not in attributes:
        raise KeyError(f"Missing 'treeID' field in {ply_path}")
    instance = attributes["treeID"].astype(np.int32, copy=False)

    return {"coord": coord, "segment": segment, "instance": instance}


def save_scene(output_scene_dir: str, scene: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
    os.makedirs(output_scene_dir, exist_ok=True)
    np.save(os.path.join(output_scene_dir, "coord.npy"), scene["coord"].astype(np.float32))
    np.save(os.path.join(output_scene_dir, "segment.npy"), scene["segment"].astype(np.int32))
    np.save(os.path.join(output_scene_dir, "instance.npy"), scene["instance"].astype(np.int32))
    with open(os.path.join(output_scene_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def process_one_file(
    ply_path: str,
    output_root: str,
    split: str,
    source: str,
    big_file_threshold: int,
    voxel_size: float,
    chunk_size: float,
    min_points_per_tile: int,
    seed: int,
) -> Tuple[str, int, int]:
    scene = build_scene(ply_path)
    scene_id = os.path.splitext(os.path.basename(ply_path))[0]
    n_points_raw = scene["coord"].shape[0]

    # A single gate decides both: files small enough to skip voxelization are
    # also small enough to stay a single scene folder (matching every other,
    # already plot-sized source). This keeps all splits of the *same* big
    # source (e.g. BlueCat train/val/test) treated identically, instead of
    # each being judged independently against a post-voxel point count.
    is_big = n_points_raw > big_file_threshold
    if is_big:
        # Reseed per file (not shared across the process pool) but deterministic
        # across runs regardless of worker scheduling order.
        rng = np.random.default_rng(seed + zlib.crc32(scene_id.encode("utf-8")))
        scene = voxelize_scene(scene, voxel_size=voxel_size, rng=rng)

    meta_base = {
        "source": source,
        "split": split,
        "orig_file": os.path.basename(ply_path),
        "voxelized": is_big,
        "voxel_size": voxel_size if is_big else None,
    }

    effective_chunk_size = chunk_size if is_big else float("inf")
    sub_scenes = split_scene_xy_by_chunk_size(scene=scene, chunk_size=effective_chunk_size)

    written = 0
    dropped = 0
    for suffix, sub_scene in sub_scenes:
        if sub_scene["coord"].shape[0] < min_points_per_tile:
            dropped += 1
            continue
        tiled_scene_id = scene_id if suffix == "0-0" else f"{scene_id}_{suffix}"
        output_scene_dir = os.path.join(output_root, split, tiled_scene_id)
        save_scene(output_scene_dir, sub_scene, meta_base)
        written += 1
    return scene_id, written, dropped


def collect_jobs(dataset_root: str) -> List[Tuple[str, str, str]]:
    """Return list of (ply_path, split, source) across train_val_data/ and test_data/."""
    jobs: List[Tuple[str, str, str]] = []
    for subdir in ("train_val_data", "test_data"):
        split_dir = os.path.join(dataset_root, subdir)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Directory not found, skipped: {split_dir}")
            continue
        for ply_path in sorted(glob.glob(os.path.join(split_dir, "*.ply"))):
            source, split = parse_source_and_split(os.path.basename(ply_path))
            jobs.append((ply_path, split, source))
    return jobs


def main_process():
    parser = argparse.ArgumentParser(description="Preprocess ForInstanceV2 PLY tiles for Pointcept.")
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Directory containing train_val_data/ and test_data/.",
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument(
        "--big_file_threshold",
        default=5_000_000,
        type=int,
        help="Files with more raw points than this are voxel-subsampled AND XY-chunked "
        "(default: 5,000,000; only BlueCat_RN_merged_trees_* currently exceeds this -- "
        "note this is checked per source *file*, i.e. per split, so all of a big source's "
        "train/val/test splits get the same treatment, even if one split alone would fall "
        "under the threshold after voxelization).",
    )
    parser.add_argument(
        "--voxel_size",
        default=0.1,
        type=float,
        help="Voxel size in meters for offline subsampling of files above --big_file_threshold "
        "(default: 0.1, matching every downstream config's GridSample grid_size -- keep "
        "them in sync, or a live GridSample at a *finer* grid size than this would be a "
        "no-op, silently capping resolution below what the config asks for).",
    )
    parser.add_argument(
        "--chunk_size",
        default=20.0,
        type=float,
        help="XY tile size in meters applied to files above --big_file_threshold (default: "
        "20.0; chosen for BlueCat's post-voxelization density, roughly on par with other "
        "sources' plot sizes/point counts). A regular grid over an irregularly-shaped point "
        "cloud (e.g. a diagonal scan swath, not a filled rectangle) produces near-empty edge "
        "tiles -- see --min_points_per_tile.",
    )
    parser.add_argument(
        "--min_points_per_tile",
        default=1_000,
        type=int,
        help="Drop chunked tiles with fewer points than this (default: 1,000) -- filters out "
        "near-empty edge/corner tiles from XY-chunking an irregularly-shaped point cloud "
        "against its rectangular bounding box. Only affects chunked (big) files.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Base RNG seed for voxel-subsampling point selection (per-file, deterministic "
        "regardless of worker scheduling order).",
    )
    parser.add_argument(
        "--max_files",
        default=None,
        type=int,
        help="Optional cap on number of source files (smoke tests).",
    )
    args = parser.parse_args()

    jobs = collect_jobs(args.dataset_root)
    if args.max_files is not None:
        jobs = jobs[: args.max_files]

    os.makedirs(args.output_root, exist_ok=True)
    for split in VALID_SPLITS:
        os.makedirs(os.path.join(args.output_root, split), exist_ok=True)

    total = len(jobs)
    print(f"Found {total} source PLY file(s) under {args.dataset_root}")
    print(
        f"big_file_threshold={args.big_file_threshold}, voxel_size={args.voxel_size}, "
        f"chunk_size={args.chunk_size}, min_points_per_tile={args.min_points_per_tile}, "
        f"num_workers={args.num_workers}"
    )
    if total == 0:
        return

    scenes_written = 0
    tiles_dropped = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [
            pool.submit(
                process_one_file,
                ply_path,
                args.output_root,
                split,
                source,
                args.big_file_threshold,
                args.voxel_size,
                args.chunk_size,
                args.min_points_per_tile,
                args.seed,
            )
            for ply_path, split, source in jobs
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            scene_id, n_written, n_dropped = future.result()
            scenes_written += n_written
            tiles_dropped += n_dropped
            suffix = f" ({n_dropped} tile(s) dropped, <min_points_per_tile)" if n_dropped else ""
            print(f"\rProgress: {idx}/{total} (last: {scene_id} -> {n_written} tile(s){suffix})", end="", flush=True)
    print()
    print(
        f"Done. {scenes_written} scene folder(s) after chunking ({total} source PLY file(s), "
        f"{tiles_dropped} tile(s) dropped as near-empty) -> {args.output_root}"
    )


if __name__ == "__main__":
    main_process()
