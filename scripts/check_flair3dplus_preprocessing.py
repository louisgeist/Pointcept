"""
Validate Flair3D+ preprocessed scenes listed in the split manifest.

The script follows the same scene-selection logic as Flair3DDataset:
- rows with LIDARHD=True only
- same split filtering
- same hardcoded excluded tiles
- optional runtime exclusions from a missing tiles manifest

For each non-excluded scene, the checker validates:
- point count consistency across available modalities
- minimum number of points
- required assets presence
- load errors and basic shape checks
- NaN/Inf for selected float modalities
- optional warnings for low segment label cardinality

python scripts/check_flair3dplus_preprocessing.py \
  --data_root data/flair3d_plus \
  --csv_manifest data/flair3d_plus/raw/scene_split_manifest.csv \
  --splits train,val,test \
  --min_points 10000 \
  --num_workers 24

"""

from __future__ import annotations

import argparse
import ast
import csv
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAIR3D_DATASET_FILE = os.path.join(REPO_ROOT, "pointcept", "datasets", "flair3d.py")

MODALITIES = (
    "coord",
    "color",
    "elevation",
    "forest",
    "land_use",
    "natural_habitat",
    "segment",
    "strength",
)
REQUIRED_MODALITIES = ("coord", "color", "segment")
FINITE_CHECK_MODALITIES = ("coord", "strength") #"elevation",


@dataclass(frozen=True)
class SceneRecord:
    split: str
    patch_id: str
    scene_path: str


@dataclass(frozen=True)
class CheckResult:
    split: str
    patch_id: str
    scene_path: str
    severity: str
    reason: str
    details: str


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def parse_splits(splits_arg: str) -> set[str]:
    return {token.strip() for token in splits_arg.split(",") if token.strip()}


def parse_manifest_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def build_scene_path(data_root: str, split: str, patch_id: str, dept_year: str, roi: str) -> str:
    return os.path.join(data_root, split, f"{dept_year}_LIDARHD", roi, patch_id)


def load_hardcoded_excluded_tiles() -> set[tuple[str, str]]:
    """
    Read HARDCODED_EXCLUDED_TILES from pointcept/datasets/flair3d.py using AST.

    This avoids importing the whole package and optional runtime dependencies.
    """
    def _load_missing_lidarhd_tiles_default() -> set[tuple[str, str]]:
        details_csv = os.path.join(REPO_ROOT, "data", "flair3d_plus", "missing_coord_tiles.details.csv")
        missing_tiles: set[tuple[str, str]] = set()
        if not os.path.exists(details_csv):
            return missing_tiles

        with open(details_csv, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("reason") != "missing_coord_file":
                    continue
                split = (row.get("split") or "").strip()
                patch_id = (row.get("patch_id") or "").strip()
                if split and patch_id:
                    missing_tiles.add((split, patch_id))
        return missing_tiles

    def _eval_expr(node: ast.AST, env: dict[str, object]) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(_eval_expr(elt, env) for elt in node.elts)
        if isinstance(node, ast.List):
            return [_eval_expr(elt, env) for elt in node.elts]
        if isinstance(node, ast.Set):
            return {_eval_expr(elt, env) for elt in node.elts}
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise KeyError(node.id)
            return env[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = _eval_expr(node.left, env)
            right = _eval_expr(node.right, env)
            if not isinstance(left, set) or not isinstance(right, set):
                raise TypeError("BitOr is only supported for sets in this parser.")
            return left | right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            # Mirror Flair3D dataset's default missing-tiles source.
            if node.func.id == "_load_missing_lidarhd_tiles":
                return _load_missing_lidarhd_tiles_default()
        raise TypeError(f"Unsupported AST node for exclusion parsing: {node.__class__.__name__}")

    with open(FLAIR3D_DATASET_FILE, "r", encoding="utf-8") as handle:
        module_ast = ast.parse(handle.read(), filename=FLAIR3D_DATASET_FILE)

    for node in module_ast.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Flair3DDataset":
            continue

        env: dict[str, object] = {}
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                continue
            target_name = stmt.targets[0].id
            try:
                env[target_name] = _eval_expr(stmt.value, env)
            except Exception:
                continue
            if target_name == "HARDCODED_EXCLUDED_TILES":
                value = env[target_name]
                if not isinstance(value, set):
                    break
                return {(str(split), str(patch_id)) for split, patch_id in value}
    raise RuntimeError("Could not locate Flair3DDataset.HARDCODED_EXCLUDED_TILES in flair3d.py.")


def load_missing_tiles_manifest(path: str | None) -> set[tuple[str, str]]:
    missing_tiles: set[tuple[str, str]] = set()
    if not path:
        return missing_tiles
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing_tiles_manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [part.strip() for part in stripped.split(",", 2)]
            if len(parts) < 2:
                continue
            split, patch_id = parts[0], parts[1]
            if split and patch_id:
                missing_tiles.add((split, patch_id))
    return missing_tiles


def load_scene_records(
    data_root: str,
    csv_manifest: str,
    target_splits: set[str],
    excluded_tiles: set[tuple[str, str]],
) -> list[SceneRecord]:
    scene_records: list[SceneRecord] = []

    with open(csv_manifest, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "patch_id", "LIDARHD"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise KeyError(f"Missing required columns in manifest: {sorted(missing_cols)}")

        for row in reader:
            split = str(row["split"]).strip()
            patch_id = str(row["patch_id"]).strip()
            if not split or not patch_id:
                continue
            if split not in target_splits:
                continue
            if not parse_manifest_bool(row.get("LIDARHD")):
                continue

            if (split, patch_id) in excluded_tiles:
                continue

            dept_year = (row.get("dept_year") or "").strip() or patch_id.split("_", 2)[0]
            roi = (row.get("roi") or "").strip() or patch_id.split("_", 2)[1]
            scene_path = build_scene_path(data_root, split, patch_id, dept_year, roi)
            scene_records.append(SceneRecord(split=split, patch_id=patch_id, scene_path=scene_path))

    return scene_records


def _get_point_count(modality: str, arr: np.ndarray) -> tuple[int | None, str | None]:
    if arr.ndim == 0:
        return None, f"{modality} is scalar, expected array with first dimension as points."

    if modality == "coord":
        if arr.ndim != 2 or arr.shape[1] != 3:
            return None, f"coord shape {tuple(arr.shape)} is invalid; expected (N, 3)."
        return int(arr.shape[0]), None

    if modality == "color":
        if arr.ndim != 2 or arr.shape[1] != 3:
            return None, f"color shape {tuple(arr.shape)} is invalid; expected (N, 3)."
        return int(arr.shape[0]), None

    if modality == "segment":
        if arr.ndim > 2:
            return None, f"segment shape {tuple(arr.shape)} is invalid; expected vector-like."
        return int(arr.reshape(-1).shape[0]), None

    return int(arr.shape[0]), None


def _result(scene: SceneRecord, severity: str, reason: str, details: str) -> CheckResult:
    return CheckResult(
        split=scene.split,
        patch_id=scene.patch_id,
        scene_path=scene.scene_path,
        severity=severity,
        reason=reason,
        details=details,
    )


def _check_one_scene(task: tuple[SceneRecord, int]) -> list[CheckResult]:
    scene, min_points = task
    issues: list[CheckResult] = []

    if not os.path.isdir(scene.scene_path):
        issues.append(
            _result(
                scene,
                "error",
                "missing_scene_dir",
                "Scene directory does not exist.",
            )
        )
        return issues

    loaded: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    for modality in MODALITIES:
        file_path = os.path.join(scene.scene_path, f"{modality}.npy")
        if not os.path.isfile(file_path):
            if modality in REQUIRED_MODALITIES:
                issues.append(
                    _result(
                        scene,
                        "error",
                        "missing_required_modality",
                        f"{modality}.npy is missing.",
                    )
                )
            continue

        try:
            arr = np.load(file_path, mmap_mode="r")
        except Exception as exc:  # noqa: BLE001
            issues.append(
                _result(
                    scene,
                    "error",
                    "modality_load_error",
                    f"{modality}.npy failed to load: {exc.__class__.__name__}",
                )
            )
            continue

        count, shape_error = _get_point_count(modality, arr)
        if shape_error is not None:
            issues.append(_result(scene, "error", "invalid_modality_shape", shape_error))
            continue

        loaded[modality] = arr
        counts[modality] = int(count)

    if not counts:
        issues.append(
            _result(
                scene,
                "error",
                "no_supported_modalities",
                "None of the supported modalities could be loaded.",
            )
        )
        return issues

    if len(set(counts.values())) > 1:
        details = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        issues.append(
            _result(
                scene,
                "error",
                "inconsistent_point_count",
                f"Point counts do not match across modalities: {details}",
            )
        )

    reference_count = counts["coord"] if "coord" in counts else next(iter(counts.values()))
    if reference_count < min_points:
        issues.append(
            _result(
                scene,
                "warning",
                "too_few_points",
                f"Scene has {reference_count} points (< {min_points}).",
            )
        )

    for modality in FINITE_CHECK_MODALITIES:
        arr = loaded.get(modality)
        if arr is None:
            continue
        finite_mask = np.isfinite(arr)
        if not bool(np.all(finite_mask)):
            non_finite = int(arr.size - np.count_nonzero(finite_mask))
            issues.append(
                _result(
                    scene,
                    "error",
                    "non_finite_values",
                    f"{modality}.npy contains {non_finite} non-finite values.",
                )
            )

    strength = loaded.get("strength")
    if strength is not None and strength.size > 0:
        min_val = float(np.min(strength))
        max_val = float(np.max(strength))
        if min_val < 0.0 or max_val > 1.0:
            issues.append(
                _result(
                    scene,
                    "warning",
                    "strength_out_of_range",
                    f"strength.npy min/max are [{min_val:.6f}, {max_val:.6f}] (expected in [0, 1]).",
                )
            )

    return issues


def scan_scenes(
    scene_records: list[SceneRecord],
    min_points: int,
    num_workers: int,
) -> list[CheckResult]:
    tasks = [(scene, min_points) for scene in scene_records]
    if num_workers <= 1:
        nested = [_check_one_scene(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            nested = list(pool.map(_check_one_scene, tasks))

    flat = [item for sub in nested for item in sub]
    flat.sort(key=lambda row: (row.severity, row.reason, row.split, row.patch_id))
    return flat


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_detailed_csv(path: str, rows: Iterable[CheckResult]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "patch_id",
                "severity",
                "reason",
                "details",
                "scene_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": row.split,
                    "patch_id": row.patch_id,
                    "severity": row.severity,
                    "reason": row.reason,
                    "details": row.details,
                    "scene_path": row.scene_path,
                }
            )


def write_manifest_output(path: str, rows: Iterable[CheckResult], include_warnings: bool) -> int:
    if include_warnings:
        selected = [row for row in rows if row.severity in {"error", "warning"}]
    else:
        selected = [row for row in rows if row.severity == "error"]

    uniq = sorted({(row.split, row.patch_id) for row in selected})
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        for split, patch_id in uniq:
            handle.write(f"{split},{patch_id}\n")
    return len(uniq)


def summarize_results(
    scene_records: list[SceneRecord],
    issues: list[CheckResult],
) -> str:
    scanned_by_split = Counter(scene.split for scene in scene_records)
    issue_by_split = Counter(row.split for row in issues)
    reason_count = Counter((row.severity, row.reason) for row in issues)
    severity_count = Counter(row.severity for row in issues)

    ordered_splits = sorted(set(scanned_by_split) | set(issue_by_split))
    lines = [
        f"Scanned scenes: {len(scene_records)}",
        f"Issues: total={len(issues)}, errors={severity_count['error']}, warnings={severity_count['warning']}",
        "Per split:",
    ]
    for split in ordered_splits:
        lines.append(
            f"  - {split}: scanned={scanned_by_split[split]}, "
            f"issues={issue_by_split[split]}"
        )

    if reason_count:
        lines.append("Issue breakdown:")
        for (severity, reason), count in sorted(reason_count.items()):
            lines.append(f"  - {severity}:{reason}: {count}")

    return "\n".join(lines)


def default_report_path(output_manifest: str) -> str:
    root, ext = os.path.splitext(output_manifest)
    if ext:
        return f"{root}.details.csv"
    return f"{output_manifest}.details.csv"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Flair3D+ preprocessed scene assets.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.join("data", "flair3d_plus"),
        help="Preprocessed Flair3D+ root (relative to repo root if not absolute).",
    )
    parser.add_argument(
        "--csv_manifest",
        type=str,
        default=os.path.join("data", "flair3d_plus", "raw", "scene_split_manifest.csv"),
        help="Path to scene_split_manifest.csv (relative to repo root if not absolute).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to scan.",
    )
    parser.add_argument(
        "--missing_tiles_manifest",
        type=str,
        default="",
        help="Optional extra excluded tiles file with lines: split,patch_id.",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=10000,
        help="Warn when the scene point count is below this threshold.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--output_report_csv",
        type=str,
        default=os.path.join("data", "flair3d_plus", "preprocessing_checks", "flair3dplus_preprocess_check.csv"),
        help="Detailed CSV output path.",
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        default=os.path.join("data", "flair3d_plus", "preprocessing_checks", "flair3dplus_preprocess_issues.txt"),
        help="Output split,patch_id manifest for problematic scenes.",
    )
    parser.add_argument(
        "--manifest_include_warnings",
        action="store_true",
        help="Also include warning-level issues in output manifest.",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()

    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1.")
    if args.min_points < 0:
        raise ValueError("--min_points must be >= 0.")
    data_root = resolve_repo_path(args.data_root)
    csv_manifest = resolve_repo_path(args.csv_manifest)
    output_report = resolve_repo_path(args.output_report_csv)
    output_manifest = resolve_repo_path(args.output_manifest)
    target_splits = parse_splits(args.splits)

    hardcoded_excluded = load_hardcoded_excluded_tiles()
    extra_excluded = load_missing_tiles_manifest(resolve_repo_path(args.missing_tiles_manifest)) if args.missing_tiles_manifest else set()
    excluded_tiles = hardcoded_excluded | extra_excluded

    scene_records = load_scene_records(
        data_root=data_root,
        csv_manifest=csv_manifest,
        target_splits=target_splits,
        excluded_tiles=excluded_tiles,
    )

    issues = scan_scenes(
        scene_records=scene_records,
        min_points=args.min_points,
        num_workers=args.num_workers,
    )

    write_detailed_csv(output_report, issues)
    issue_count = write_manifest_output(
        output_manifest,
        issues,
        include_warnings=args.manifest_include_warnings,
    )

    print(summarize_results(scene_records, issues))
    print(f"Wrote detailed report: {output_report}")
    print(
        f"Wrote issue manifest: {output_manifest} "
        f"({issue_count} unique split,patch_id; include_warnings={args.manifest_include_warnings})"
    )


if __name__ == "__main__":
    main()
