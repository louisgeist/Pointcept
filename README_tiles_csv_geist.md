# `tiles.csv` remplace `scene_split_manifest.csv`

Mini-doc de migration pour tout repo qui lisait le manifest Flair3D+
(`scene_split_manifest.csv`). À donner telle quelle à un agent dans l'autre repo.

## Ce qui change

- **Un seul fichier de tuiles** : `tiles.csv` (et `tiles.parquet`, même table) à la racine
  du dataset HF [`LouisGeist/MALiBU3D`](https://huggingface.co/datasets/LouisGeist/MALiBU3D).
  Sur Jean Zay, `unzip_hf_dataset.py` le copie à la racine de l'arbre extrait
  (`.../MALiBU3D/tiles.csv`).
- **Une ligne par tuile livrée, toutes avec LiDAR** : il n'y a plus de colonne `LIDARHD`,
  et plus de lignes FLAIR-HUB sans LiDAR. Un patch FLAIR-HUB absent de `tiles.csv` n'a pas
  de LiDAR HD (ou, rarement, des coordonnées manquantes).
- `scene_split_manifest.csv` n'est plus publié. C'est une entrée interne de
  Flair3D-build, à ne plus lire ailleurs. La copie locale de Pointcept
  (`data/flair3d_plus/raw/scene_split_manifest.csv`, juillet) a d'ailleurs des flags
  périmés.

## Correspondance des colonnes

| ancien (`scene_split_manifest.csv`) | nouveau (`tiles.csv`) |
|---|---|
| `split`, `dept_year`, `roi`, `scene_i_j` | inchangés |
| `patch_id` | `tile_id` (même valeur : `{dept_year}_{roi}_{scene_i_j}`) |
| `LIDARHD` | supprimée (toujours vraie) → retirer le filtre `LIDARHD == 'True'` |
| `NATURAL_HABITAT` | `has_natural_habitat` |
| `DEM_ELEV` | `has_elevation` |
| `ROADS` | `has_roads_graph` (le GPKG des routes est dans le zip de la ROI) |
| `LAND_USE`, `RAILROADS`, `TRANSMISSION_LINES` | **supprimées, sans remplaçant** (modalités non livrées) |
| `date_aerial_rgb`, `date_lidarhd`, `date_gap_days` | inchangés |
| `n_points`, `n_voxels` | inchangés, remplis pour toutes les tuiles (`n_voxels` : présent sur HF après la prochaine ré-upload de `tiles.csv`) |
| — | nouveaux : `zip_path`, `roads_gpkg`, `forest_origin_x/y`, `forest_width/height` |

Les booléens valent `True` / `False` (chaînes en CSV, bool en parquet).

## Lecture

Chemin d'une tuile (layout Pointcept, inchangé) :
`{data_root}/{split}/{dept_year}_LIDARHD/{roi}/{tile_id}`

```python
import pandas as pd

tiles = pd.read_csv("tiles.csv")
val = tiles[tiles.split == "val"]                      # plus de filtre LIDARHD
with_dtm = tiles[tiles.has_elevation]
paths = [f"{root}/{r.split}/{r.dept_year}_LIDARHD/{r.roi}/{r.tile_id}" for r in val.itertuples()]
```

Pour accepter **les deux formats** pendant la transition, Pointcept utilise
`pointcept/datasets/preprocessing/flair3d_plus/tile_catalog.py` : stdlib uniquement,
copiable tel quel. `iter_tiles(path, splits)` ramène l'ancien manifest au schéma
`tiles.csv` et saute les lignes `LIDARHD=False`.

## Checklist pour mettre à jour un repo

1. `grep -rn "scene_split_manifest\|patch_id\|LIDARHD\|NATURAL_HABITAT\|DEM_ELEV\|LAND_USE\|RAILROADS\|TRANSMISSION_LINES\|\"ROADS\"" .`
2. Pointer les chemins vers `tiles.csv`, renommer les colonnes selon la table ci-dessus,
   retirer les filtres `LIDARHD`.
3. Tout code qui dépendait de `LAND_USE` / `RAILROADS` / `TRANSMISSION_LINES` : à
   supprimer, ou bien à faire reposer sur la présence du fichier source (raster ou graphe),
   comme le fait maintenant le preprocessing Pointcept.
4. Vérifier le nombre de tuiles : 211 047 au total (train 130 847, val 33 231,
   test 46 969).
