# FLAIR-HUB Extension — New Geospatial Modalities

This repository is an **extension of [FLAIR-HUB](https://huggingface.co/datasets/IGNF/FLAIR-HUB)**, the large-scale multimodal dataset for land cover and crop mapping produced by IGN (Institut national de l'information géographique et forestière). It augments FLAIR-HUB's 2,822 zones across 74 (département, year) couples with four new modalities derived from national French geospatial products.

---

## Repository Structure

This repository follows the same directory structure as FLAIR-HUB, with the modality name used as the top-level directory identifier:

```
data/
├── LIDARHD/
│   ├── <ROI>/
│   │   ├── <Patch>.ply
│   │   ├── ...
│   └── ...
├── FOREST/
│   ├── <ROI>/
│   │   ├── <Patch>.tif
│   │   ├── ...
│   └── ...
├── LAND_USE/
│   ├── <ROI>/
│   │   ├── <Patch>.tif
│   │   ├── ...
│   └── ...
└── NATURAL_HABITAT/
    ├── <ROI>/
    │   ├── <Patch>.tif
    │   ├── ...
    └── ...
```

Patch naming and ROI identifiers are consistent with the corresponding FLAIR-HUB patches. The only structural difference is the use of the `.ply` extension for LiDAR point cloud files instead of `.tif`.

---

## New Modalities

### LIDARHD — LiDAR HD Point Clouds

**Source:** French national program [LiDAR HD](https://geoservices.ign.fr/lidarhd)  
**Format:** Binary little-endian PLY 1.0 (`.ply`)  
**Coverage:** Partial — see coverage notes below

Each `.ply` file is a 3D point cloud aligned spatially with the corresponding FLAIR-HUB patch. Points carry the following per-point attributes:

| Attribute | Description |
|---|---|
| `x`, `y`, `z` | 3D coordinates |
| `intensity` | Laser return intensity |
| `red`, `green`, `blue` | RGB color, naively projected from the FLAIR-HUB `AERIAL_RGBI` raster |
| `cosia_class` | Land cover class, naively projected from `AERIAL_LABEL-COSIA` |
| `lidarhd_class` | Native LiDAR HD classification |

> **Note on projection:** RGB colors and CoSIA class labels were obtained by projecting the corresponding FLAIR-HUB raster values onto each 3D point using its (x, y) coordinates. This projection is naive (no interpolation or geometric refinement) and may introduce minor misalignments near object boundaries.

**Coverage details:**

- Not all FLAIR-HUB patches have a corresponding LiDAR point cloud.
- `zone_completeness.json`: for each zone, the percentage of tiles covered by a LiDAR `.ply` file.
- `missing_in_zone.json`: for each zone, the count of tiles not covered by a LiDAR `.ply` file.

---

### FOREST — Forest Mask

**Source:** [Masque Forêt](https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_MASQUE-FORET), IGN  
**Format:** GeoTIFF raster, 20 cm/px  
**Coverage:** All 74 (département, year) couples — complete

A binary rasterization of the national vector forest mask product. Pixel values indicate:

| Value | Meaning |
|---|---|
| `0` | Non-forest |
| `1` | Forest |

The forest/non-forest definition follows the **FAO (Food and Agriculture Organization)** standard.

---

### LAND_USE — Land Use Map

**Source:** Layer *Usage* of [OCS GE](https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_OCS-GE) (Occupation du Sol à Grande Échelle), IGN  
**Format:** GeoTIFF raster, 20 cm/px  
**Coverage:** 73 out of 74 (département, year) couples — when present, coverage within the couple is complete  
**Classes:** 20 classes — see [`land_use_classes.txt`](land_use_classes.txt)

A rasterized version of the *Usage* (functional land use) layer of OCS GE, a national high-resolution land occupation product. Each pixel is assigned one of 20 land use classes encoding the functional purpose of the land (agriculture, housing, industry, transport, etc.).

---

### NATURAL_HABITAT — Natural Habitat Map

**Source:** [CarHab](https://cartes.gouv.fr/rechercher-une-donnee/dataset/INPN-CARHAB_HABITATS), INPN (Inventaire National du Patrimoine Naturel)  
**Format:** GeoTIFF raster, 20 cm/px  
**Coverage:** 55 out of 74 (département, year) couples — when present, coverage within the couple is complete

A rasterized version of the CarHab product, which maps natural and semi-natural vegetation habitats across France. Each pixel is assigned one of **44 habitat classes** — see [`natural_habitat_classes.txt`](natural_habitat_classes.txt).

> **Note on class 43 (`Autre`):** This residual class is most likely dominated by roads and railway infrastructure, which are not explicitly classified in the CarHab product.

---

### DEM_ELEV — Elevation Data (from FLAIR-HUB)

**Source:** Copied as-is from [FLAIR-HUB](https://huggingface.co/datasets/IGNF/FLAIR-HUB)  
**Format:** GeoTIFF raster, 20 cm/px (DSM) / 1 m (DTM), Float32  
**Coverage:** All 74 (département, year) couples — complete

This modality is a verbatim copy of the `DEM_ELEV` modality from FLAIR-HUB. It provides elevation data with two channels: a Digital Surface Model (DSM) at 20 cm resolution and a Digital Terrain Model (DTM) at 1 m resolution (resampled). The difference DSM − DTM gives an estimate of object heights (buildings, vegetation, etc.). It is included here for convenience to avoid requiring a separate download of FLAIR-HUB when working with this extension.

---

### Acquisition Date and LIDARHD / AERIAL_RGBI Temporal Gap

The file `lidarhd_aerial_date_gap.gpkg` records, for each patch, the acquisition dates of both the LiDAR HD point cloud (`date_lidarhd`) and the aerial RGBI imagery (`date_aerial_rgb`), along with the temporal gap between them expressed in days (`date_gap_days`).

---

## Coverage Summary

| Modality | Couples covered | Patch-level completeness |
|---|---|---|
| DEM_ELEV | All 74 | Complete (copied from FLAIR-HUB) |
| LIDARHD | All 74 | Partial — see `zone_completeness.json` |
| FOREST | All 74 | Complete |
| LAND_USE | 73 / 74 | Complete when present |
| NATURAL_HABITAT | 55 / 74 | Complete when present |
| LIDARHD / AERIAL_RGBI date gap | All 74 | Complete — patches with no LiDAR acquisition have `date_lidarhd` = `<NA>` and `date_gap_days` = `NULL` |

---

## Relation to FLAIR-HUB

This dataset is designed to be used jointly with FLAIR-HUB. All patches in this extension correspond directly to patches in FLAIR-HUB (same spatial extent, same naming convention). To work with the full multimodal data, users should download both repositories and align patches by their shared identifiers.

