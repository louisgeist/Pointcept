"""Read the Flair3D+ / MALiBU3D tile list from either CSV schema.

Two schemas describe the same tiles:

- ``tiles.csv`` -- the MALiBU3D Hugging Face release catalog: one row per
  shipped tile (every row has LiDAR HD). This is the canonical file.
- ``scene_split_manifest.csv`` -- the legacy Pointcept split table: one row per
  FLAIR-HUB patch, including ``LIDARHD=False`` rows, with upper-case flags.

:func:`iter_tiles` normalizes both to the catalog column names, so callers can
point ``csv_manifest`` / ``--split_manifest_csv`` at either file:

==================  ===================  ===========================
catalog column      legacy column        notes
==================  ===================  ===========================
``tile_id``         ``patch_id``         ``{dept_year}_{roi}_{scene_i_j}``
``has_natural_habitat``  ``NATURAL_HABITAT``
``has_elevation``   ``DEM_ELEV``
``has_roads_graph`` ``ROADS``
(implicit True)     ``LIDARHD``          legacy rows with False are skipped
==================  ===================  ===========================

Other legacy columns are ignored (modalities that are not part of the release).

Pure stdlib so preprocessing scripts can import it without torch.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

LEGACY_COLUMN_ALIASES = {
    "patch_id": "tile_id",
    "NATURAL_HABITAT": "has_natural_habitat",
    "DEM_ELEV": "has_elevation",
    "ROADS": "has_roads_graph",
}

REQUIRED_COLUMNS = ("split", "dept_year", "roi", "tile_id")

_TRUE_TOKENS = frozenset({"true", "1", "yes"})
_FALSE_TOKENS = frozenset({"false", "0", "no", ""})
_NA_TOKENS = frozenset({"", "<na>", "na", "none", "n/a", "nan", "null"})


@dataclass(frozen=True)
class Tile:
    """One LiDAR tile, normalized from either CSV schema.

    Flag / count fields are ``None`` when the source CSV lacks the column.
    """

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
    """Yield LiDAR tiles from ``tiles.csv`` or a legacy split manifest.

    ``splits`` filters on the (lower-cased) ``split`` column. Legacy rows with
    ``LIDARHD`` other than True are skipped; rows with an empty id field are
    skipped silently.
    """
    csv_path = str(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Tile CSV not found: {csv_path}")
    splits_set = {s.strip().lower() for s in splits} if splits else None

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV or no header row: {csv_path}")
        rename = {
            name: LEGACY_COLUMN_ALIASES.get(name, name) for name in reader.fieldnames
        }
        columns = set(rename.values())
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise ValueError(
                f"Tile CSV {csv_path} is missing columns {missing} "
                "(expected tiles.csv or scene_split_manifest.csv)"
            )
        has_lidarhd = "LIDARHD" in columns

        for raw in reader:
            row = {rename[k]: v for k, v in raw.items() if k in rename}
            split = (row.get("split") or "").strip().lower()
            dept_year = (row.get("dept_year") or "").strip()
            roi = (row.get("roi") or "").strip()
            tile_id = (row.get("tile_id") or "").strip()
            if not split or not dept_year or not roi or not tile_id:
                continue
            if splits_set is not None and split not in splits_set:
                continue
            if has_lidarhd and not _parse_bool(
                row.get("LIDARHD"), "LIDARHD", tile_id, csv_path
            ):
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
            )


def read_tiles(csv_path: str, splits: Optional[Sequence[str]] = None) -> List[Tile]:
    return list(iter_tiles(csv_path, splits))


def tile_csv_columns(csv_path: str) -> set:
    """Header of ``csv_path`` with legacy names mapped to catalog names."""
    with open(str(csv_path), "r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f), [])
    return {LEGACY_COLUMN_ALIASES.get(name, name) for name in header}
