# Expected `raw/` Folder Structure (produced by `unzip_flair3d.py`)

```text
raw/
├── LIDARHD/
│   └── <DEPT_YEAR>_LIDARHD/
│       └── <ROI>/
│           └── <PATCH>.ply
├── FOREST/
│   └── <DEPT_YEAR>_FOREST/
│       └── <ROI>/
│           └── <PATCH>.tif
├── LAND_USE/
│   └── <DEPT_YEAR>_LAND_USE/
│       └── <ROI>/
│           └── <PATCH>.tif
├── NATURAL_HABITAT/
│   └── <DEPT_YEAR>_NATURAL_HABITAT/
│       └── <ROI>/
│           └── <PATCH>.tif
└── DEM_ELEV/
    └── <DEPT_YEAR>_DEM_ELEV/
        └── <ROI>/
            └── <PATCH>.tif
```

## Naming Convention

- `<DEPT_YEAR>` example: `D010-2019`
- Raster folder suffixes: `_FOREST`, `_LAND_USE`, `_NATURAL_HABITAT`, `_DEM_ELEV`
- `<ROI>` (Region Of Interest) example: `AA-S1-1`
- `<PATCH>=<DEPT_YEAR>_<LABELNAME>_<ROI>_<I-J>` example: `D010-2019_LIDARHD_AA-S1-1_1-1` (same patch id across modalities, only extension changes)
    - with `<I-J>`being the subtitle coordinates.

## Preprocessing flow (manifest-driven)

`preprocess_flair3d.py` is driven entirely by a split manifest CSV (built by
`scripts/build_csv_manifest.py`). The script never globs the `raw/` tree to
discover scenes: every patch to process is taken from the manifest, and any
on-disk discrepancy is reported in `<output_root>/missing_scenes.txt`.

Required manifest columns:

```
split, dept_year, roi, scene_i_j, patch_id,
LIDARHD, NATURAL_HABITAT, LAND_USE, DEM_ELEV,
date_gap_days
```

Rules applied per row:

- `LIDARHD=False`: row skipped silently (no scene output).
- `LIDARHD=True` with PLY missing on disk: reported as `Missing PLY`.
- `FOREST` is sampled for every kept patch (no manifest column — assumed
  available everywhere); a missing FOREST raster is reported as
  `Missing modality raster`.
- `NATURAL_HABITAT` / `LAND_USE` / `DEM_ELEV`: sampled only when the
  corresponding column is `True`; a missing raster despite `True` is reported
  as `Missing modality raster`.
- `date_gap_days` is read directly from the manifest and stored in
  `<scene>/meta.json` (no GeoPackage dependency).
