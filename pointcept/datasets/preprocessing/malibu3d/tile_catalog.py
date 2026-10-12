"""Read the MALiBU3D tile table (``tiles.csv``: one row per LiDAR tile) into ``Tile`` records.

Pure stdlib, so preprocessing scripts can import it without torch.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

REQUIRED_COLUMNS = ("split", "dept_year", "roi", "tile_id")

_TRUE_TOKENS = frozenset({"true", "1", "yes"})
_FALSE_TOKENS = frozenset({"false", "0", "no", ""})
_NA_TOKENS = frozenset({"", "<na>", "na", "none", "n/a", "nan", "null"})


@dataclass(frozen=True)
class Tile:
    """One LiDAR tile. Flag / count fields are ``None`` when the CSV lacks the column."""

    split: str
    dept_year: str
    roi: str
    scene_i_j: str
    tile_id: str
    has_natural_habitat: Optional[bool] = None
    has_elevation: Optional[bool] = None
    has_roads_graph: Optional[bool] = None
    date_gap_days: Optional[float] = None
    n_points: Optional[int] = None
    n_voxels: Optional[int] = None
    # Georeferencing of the tile's 1 m rasters (forest_2d, and network.npy once rasterized):
    # Lambert-93 lower-left origin and size in pixels, from the forest_* columns of tiles.csv.
    raster_origin_x: Optional[float] = None
    raster_origin_y: Optional[float] = None
    raster_width: Optional[int] = None
    raster_height: Optional[int] = None

    def tile_dir(self, data_root: str) -> str:
        """Pointcept on-disk scene dir: ``{root}/{split}/{dept_year}_LIDARHD/{roi}/{tile_id}``."""
        return os.path.join(
            data_root, self.split, f"{self.dept_year}_LIDARHD", self.roi, self.tile_id
        )


def _parse_bool(raw: Optional[str], column: str, tile_id: str, csv_path: str) -> bool:
    token = (raw or "").strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise ValueError(
        f"Invalid boolean {raw!r} in column {column!r} for tile {tile_id!r} ({csv_path})"
    )


def _parse_float(raw: Optional[str]) -> Optional[float]:
    token = (raw or "").strip()
    if token.lower() in _NA_TOKENS:
        return None
    return float(token)


def _parse_int(raw: Optional[str]) -> Optional[int]:
    value = _parse_float(raw)
    return None if value is None else int(value)


def iter_tiles(
    csv_path: str,
    splits: Optional[Sequence[str]] = None,
) -> Iterator[Tile]:
    """Yield the tiles of ``tiles.csv`` (or of a CSV with a subset of its rows).

    ``splits`` filters on the (lower-cased) ``split`` column. Rows with an empty id
    field are skipped silently. Invalid booleans raise ``ValueError``.
    """
    csv_path = str(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Tile CSV not found: {csv_path}")
    splits_set = {s.strip().lower() for s in splits} if splits else None

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV or no header row: {csv_path}")
        columns = set(reader.fieldnames)
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise ValueError(f"Tile CSV {csv_path} is missing columns {missing}")

        for row in reader:
            split = (row.get("split") or "").strip().lower()
            dept_year = (row.get("dept_year") or "").strip()
            roi = (row.get("roi") or "").strip()
            tile_id = (row.get("tile_id") or "").strip()
            if not split or not dept_year or not roi or not tile_id:
                continue
            if splits_set is not None and split not in splits_set:
                continue

            def flag(column: str) -> Optional[bool]:
                if column not in columns:
                    return None
                return _parse_bool(row.get(column), column, tile_id, csv_path)

            yield Tile(
                split=split,
                dept_year=dept_year,
                roi=roi,
                scene_i_j=(row.get("scene_i_j") or "").strip(),
                tile_id=tile_id,
                has_natural_habitat=flag("has_natural_habitat"),
                has_elevation=flag("has_elevation"),
                has_roads_graph=flag("has_roads_graph"),
                date_gap_days=_parse_float(row.get("date_gap_days")),
                n_points=_parse_int(row.get("n_points")),
                n_voxels=_parse_int(row.get("n_voxels")),
                raster_origin_x=_parse_float(row.get("forest_origin_x")),
                raster_origin_y=_parse_float(row.get("forest_origin_y")),
                raster_width=_parse_int(row.get("forest_width")),
                raster_height=_parse_int(row.get("forest_height")),
            )


def read_tiles(csv_path: str, splits: Optional[Sequence[str]] = None) -> List[Tile]:
    return list(iter_tiles(csv_path, splits))


def tile_csv_columns(csv_path: str) -> set:
    """Column names of ``csv_path``."""
    with open(str(csv_path), "r", encoding="utf-8", newline="") as f:
        return set(next(csv.reader(f), []))
