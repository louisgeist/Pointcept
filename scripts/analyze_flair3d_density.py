#!/usr/bin/env python3
"""Compute per-ROI and per-subtile point density from a Flair3D+ manifest CSV.

Two density definitions at ROI level:
  - theoretical:  surface = grid bounding-box  (max_i-min_i+1)*(max_j-min_j+1) * SUBTILE_AREA
  - adjusted:     surface = n_subtiles_present * SUBTILE_AREA

At subtile level a single density is reported (surface = SUBTILE_AREA).

Usage:
python scripts/analyze_flair3d_density.py \
    --manifest scene_split_manifest_JZ_110826.csv \
    --output-roi density_roi.csv \
    --output-subtile density_subtile.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

SUBTILE_AREA_KM2 = 0.01048576  # 1024 * 1024 * 0.01 m² = 10 485.76 m²
SUBTILE_AREA_M2 = 10485.76


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, type=Path, help="Path to scene_split_manifest CSV")
    p.add_argument("--output-roi", required=True, type=Path, help="Output CSV path (per-ROI)")
    p.add_argument("--output-subtile", required=True, type=Path, help="Output CSV path (per-subtile)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- accumulate per-ROI stats ----
    roi_stats: dict[str, dict] = {}  # key = dept_year + "_" + roi
    subtile_rows: list[dict] = []

    with open(args.manifest, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["LIDARHD"] != "True":
                continue

            dept_year = row["dept_year"]
            roi = row["roi"]
            scene_ij = row["scene_i_j"]
            patch_id = row["patch_id"]
            split = row["split"]
            n_points = int(row["n_points"])

            i_str, j_str = scene_ij.split("-")
            i, j = int(i_str), int(j_str)

            dept = dept_year.split("-")[0]

            # per-subtile row
            subtile_rows.append(
                dict(
                    dept=dept,
                    dept_year=dept_year,
                    roi=roi,
                    scene_i_j=scene_ij,
                    patch_id=patch_id,
                    split=split,
                    n_points=n_points,
                    density_pts_m2=round(n_points / SUBTILE_AREA_M2, 2),
                )
            )

            # per-ROI accumulation
            key = f"{dept_year}_{roi}"
            if key not in roi_stats:
                roi_stats[key] = dict(
                    dept=dept,
                    dept_year=dept_year,
                    roi=roi,
                    split=split,
                    n_points=0,
                    n_subtiles=0,
                    min_i=i,
                    max_i=i,
                    min_j=j,
                    max_j=j,
                )
            s = roi_stats[key]
            s["n_points"] += n_points
            s["n_subtiles"] += 1
            s["min_i"] = min(s["min_i"], i)
            s["max_i"] = max(s["max_i"], i)
            s["min_j"] = min(s["min_j"], j)
            s["max_j"] = max(s["max_j"], j)

    # ---- write per-ROI CSV ----
    roi_fields = [
        "dept",
        "dept_year",
        "roi",
        "split",
        "n_points",
        "n_subtiles",
        "theoretical_surface_km2",
        "subtile_adjusted_surface_km2",
        "density_theoretical_pts_m2",
        "density_adjusted_pts_m2",
    ]

    total_points = 0
    total_subtiles = 0
    total_theo_surface = 0.0
    total_adj_surface = 0.0

    args.output_roi.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_roi, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=roi_fields)
        w.writeheader()

        for key in sorted(roi_stats):
            s = roi_stats[key]
            grid_area = (s["max_i"] - s["min_i"] + 1) * (s["max_j"] - s["min_j"] + 1)
            theo_surf = grid_area * SUBTILE_AREA_KM2
            adj_surf = s["n_subtiles"] * SUBTILE_AREA_KM2

            total_points += s["n_points"]
            total_subtiles += s["n_subtiles"]
            total_theo_surface += theo_surf
            total_adj_surface += adj_surf

            w.writerow(
                dict(
                    dept=s["dept"],
                    dept_year=s["dept_year"],
                    roi=s["roi"],
                    split=s["split"],
                    n_points=s["n_points"],
                    n_subtiles=s["n_subtiles"],
                    theoretical_surface_km2=round(theo_surf, 6),
                    subtile_adjusted_surface_km2=round(adj_surf, 6),
                    density_theoretical_pts_m2=round(s["n_points"] / (grid_area * SUBTILE_AREA_M2), 2),
                    density_adjusted_pts_m2=round(s["n_points"] / (s["n_subtiles"] * SUBTILE_AREA_M2), 2),
                )
            )

        w.writerow(
            dict(
                dept="TOTAL",
                dept_year="",
                roi="",
                split="",
                n_points=total_points,
                n_subtiles=total_subtiles,
                theoretical_surface_km2=round(total_theo_surface, 6),
                subtile_adjusted_surface_km2=round(total_adj_surface, 6),
                density_theoretical_pts_m2=round(total_points / (total_theo_surface * 1e6), 2) if total_theo_surface else 0,
                density_adjusted_pts_m2=round(total_points / (total_adj_surface * 1e6), 2) if total_adj_surface else 0,
            )
        )

    print(f"ROI CSV written: {args.output_roi}  ({len(roi_stats)} ROIs + TOTAL)")

    # ---- write per-subtile CSV ----
    subtile_fields = ["dept", "dept_year", "roi", "scene_i_j", "patch_id", "split", "n_points", "density_pts_m2"]

    args.output_subtile.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_subtile, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=subtile_fields)
        w.writeheader()
        for row in subtile_rows:
            w.writerow(row)

    print(f"Subtile CSV written: {args.output_subtile}  ({len(subtile_rows)} subtiles)")


if __name__ == "__main__":
    main()
