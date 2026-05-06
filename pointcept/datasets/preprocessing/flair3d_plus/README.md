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
on-disk discrepancy is reported in dedicated text files under `<output_root>`.

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

## Reports

- `<output_root>/missing_ply_preflight.txt`: written immediately after the
  PLY existence pre-flight (before any scene is processed). Lists every
  patch with `LIDARHD=True` whose `.ply` is missing on disk. Useful when
  you need the list right away on a long run. Override path with
  `--missing_ply_preflight_file`.
- `<output_root>/missing_scenes.txt`: written at the end of the run.
  Consolidates missing PLY, missing modality rasters, and failed
  preprocessing tasks. Override path with `--missing_scenes_file`.

## Resume / re-run

By default, the script skips scenes whose `coord.npy` already exists in the
output directory. `coord.npy` is written last, so its presence reliably
indicates that every other array and `meta.json` were fully persisted. This
makes re-runs incremental: only patches still missing or interrupted mid-write
are processed.

Use `--force` to reprocess every patch unconditionally. This is required when:

- `--label_definition` is changed between runs.
- Manifest modality flags (`NATURAL_HABITAT`, `LAND_USE`, `DEM_ELEV`) are
  toggled, since stale outputs from a previous run would otherwise be kept.
