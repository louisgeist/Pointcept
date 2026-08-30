# Elevation parity plot (pred vs GT) — LitePT-B sur D075

Refait la figure « élévation prédite = f(vérité terrain) » (parity / scatter density plot) pour le
modèle **LitePT-B Flair3D+ multitâche (job 873542)** — le baseline supervisé « classique » — sur deux
zones du split test :

- `D075-2021_AA-S2-2` (péri-urbain / agricole, relief modéré)
- `D075-2021_UU-S1-4` (urbain dense, bâtiments jusqu'à ~62 m)

Recalcule **MAE / RMSE / R²** par zone et exporte tous les couples `(gt, pred)` dans des formats
directement exploitables dans Overleaf (pgfplots). Le montage final de la figure se fait côté LaTeX —
ce script ne produit que les données + un aperçu PNG de contrôle + un squelette `.tex`.

La cible « élévation » = **hauteur au-dessus du MNT** (`z_LiDAR − DTM`, mètres), pas l'altitude absolue.
Cf. `pointcept/datasets/preprocessing/flair3d_plus/preprocess_flair3d_v2.py` (~l.721).

## Résultats (2026-08-28, `--density-bins 200 --subsample 50000`)

| zone | n points (finis) | NaN-GT écartés | MAE | RMSE | R² | biais (pred−gt) | fit OLS (pred ≈ a·gt + b) |
|---|---|---|---|---|---|---|---|
| D075-2021_AA-S2-2 | 4 921 091 | 6 890 | **0.190 m** | **0.314 m** | **0.9910** | +0.045 m | 0.954·gt + 0.101 |
| D075-2021_UU-S1-4 | 4 536 854 | 6 671 | **0.792 m** | **1.177 m** | **0.9877** | −0.683 m | 0.935·gt + 0.076 |

UU-S1-4 est plus dure (zone urbaine) et le modèle **sous-estime les structures hautes** (biais négatif,
pente < 1) — visible sur `preview.png`. Métriques micro-moyennées sur tous les points des 25 sous-tuiles
poolés, masque `isfinite(pred) & isfinite(gt)` (convention repo `accumulate_regression_errors`).

## Données sources

Ni les prédictions ni le dump ne sont dans le repo Pointcept — ils viennent du run de test JZ
`sbatch test_flair3d_resume.sh 873542` (cf. `README_geist.md` §« Nathab inference dumps », l.303-332),
scp'és en local dans le repo frère :

| | chemin | format |
|---|---|---|
| prédictions | `/data/geist/superpixel_transformer_dev/local/temp/873542/D075_{AA-S2-2,UU-S1-4}/<scene>_<r>-<c>_reg_elevation.npy` | `float64`, **déjà en mètres**, résolution pleine, row-aligned avec `coord.npy`. 25 sous-tuiles 5×5 par zone. |
| vérité terrain | `data/flair3d_plus/test/D075-2021_LIDARHD/{AA-S2-2,UU-S1-4}/<scene>_<r>-<c>/elevation.npy` | `float32`, mètres, `NaN` là où le raster MNT est nodata. |

Les `_reg_elevation.npy` sont **dénormalisés avant `np.save`** par le tester
(`_denorm_regression_pred_np` dans `pointcept/engines/test.py`), donc pas de `×100` à appliquer même si
le multitâche entraîne en interne avec `ELEVATION_TARGET_SCALE = 0.01`. Le même dossier contient aussi
`_pred_segment`, `_pred_nathab_*`, `_logits_{network,forest_2d}`, `_ROADS_apls.json` (non utilisés ici).

Pour régénérer le dump depuis JZ (si perdu) : voir `test_flair3d_resume.sh` +
`EXTRA_OPTIONS='data.test.split=[train,val,test] data.test.include_names=[D075-2021_AA-S2-2,D075-2021_UU-S1-4]'`.

## Scripts

`scripts/export_elevation_parity.py` — **pur numpy**, autonome (aucune dépendance au registre Pointcept,
pas besoin de `PYTHONPATH`, pas de GPU). matplotlib optionnel (juste pour `preview.png`, dégradé propre
en warning si absent). Dump TikZ / CSV / npz.

`scripts/visualize_elevation_scatter.py` — hexbin matplotlib (viridis, `y = x` + fit OLS), port
du script Hydra de l’autre repo. Lit les `pairs.npz` déjà exportés (défaut) ou un couple
`--roi` / `--result-dir`. Métriques sur **tous** les points finis ; `--max-points` ne sous-échantillonne
que le rendu.

`scripts/rank_elevation_mae.py` — classe les 7 ROI du dump 873542 (D068 + D075) par MAE
d’élévation. Découvre les `*_reg_elevation.npy`, résout le GT sous
`data/flair3d_plus/{train,val,test}/<dept>_LIDARHD/<roi>/`, une passe d’accumulateurs
(pas de concat). ~4 s pour 375 sous-tuiles.

```bash
cd /data/geist/Pointcept
python scripts/export_elevation_parity.py
```

Options :

| flag | défaut | rôle |
|---|---|---|
| `--pred-root` | `/data/geist/superpixel_transformer_dev/local/temp/873542` | dossier des dumps de prédiction |
| `--gt-root` | `data/flair3d_plus/test/D075-2021_LIDARHD` | racine des tuiles préprocessées (GT) |
| `--out-dir` | `stats/flair3d/elevation_parity` | où écrire les sorties |
| `--subsample` | `50000` | taille du sous-échantillon `scatter_*.dat` par zone (`0` = désactive) |
| `--seed` | `0` | graine du sous-échantillon |
| `--density-bins` | `200` | résolution de la grille de densité (par axe) — **levier de vitesse de compilation LaTeX** |
| `--no-preview` | — | saute `preview.png` |

Les deux zones sont codées en dur dans `ZONES` en tête de script (tuple
`{scene, pred_subdir, gt_subdir}`) — pour d'autres zones, ajouter une entrée (les 5 autres tuiles du
dump 873542 : `D068-2021_UF-S1-23`, `D068-2021_UU-S1-12`, `D075_UF-S1-2`, `D068_FA-S1-26`,
`D068_UN-S1-28` — attention, le préfixe dossier GT diffère selon la tuile).

Runtime export : ~1 min 20 (dominé par l'écriture des `pairs.csv.gz` de ~5 M lignes).

Hexbin (après l’export, ou tout seul si les `pairs.npz` sont déjà là) :

```bash
python scripts/visualize_elevation_scatter.py --combine

python scripts/visualize_elevation_scatter.py \
  --roi data/flair3d_plus/test/D075-2021_LIDARHD/AA-S2-2 \
  --result-dir /data/geist/superpixel_transformer_dev/local/temp/873542/D075_AA-S2-2 \
  --output /tmp/AA-S2-2_elevation_hexbin.png
```

| flag | défaut | rôle |
|---|---|---|
| `--pairs` | les 2 `pairs.npz` D075 | cache `(gt, pred)` déjà exporté |
| `--roi` + `--result-dir` | — | charge les sous-tuiles (GT `elevation.npy` + `*_reg_elevation.npy`) |
| `--hexbin-gridsize` | `80` | résolution des hexagones |
| `--max-points` | `0` (tous) | sous-échantillon **rendu seulement** |
| `--combine` | off | figure côte-à-côte si plusieurs zones |

Classement MAE des 7 ROI du dump (D068 + D075) :

```bash
python scripts/rank_elevation_mae.py
python scripts/rank_elevation_mae.py --top-subtiles 20
```

Jean Zay (`unv@h100`, dumps dans `logs/slurm/<job>/result/`) :

```bash
sbatch sbatch_rank_elevation_mae.sh 873542
DEPARTMENTS= sbatch sbatch_rank_elevation_mae.sh 873542   # toutes les ROI du dump, pas seulement D068/D075
```

CSV écrits dans `logs/slurm/$SLURM_JOB_ID/` (`roi_mae_ranking.csv`, `per_subtile_all.csv`).

Hexbin de la ROI la plus dure :

```bash
python scripts/visualize_elevation_scatter.py \
  --roi data/flair3d_plus/test/D068-2021_LIDARHD/UF-S1-23 \
  --result-dir /data/geist/superpixel_transformer_dev/local/temp/873542/D068_UF-S1-23
```

## Ranking MAE — 7 ROI du dump 873542 (2026-08-30)

| rank | ROI | n subtiles | n finis | MAE | RMSE | R² | biais |
|---|---|---|---|---|---|---|---|
| 1 | D068-2021_UF-S1-23 | 50 | 20 208 281 | **1.276 m** | **3.547 m** | 0.8737 | −0.283 m |
| 2 | D075-2021_UU-S1-4 | 25 | 4 536 854 | 0.792 m | 1.177 m | 0.9877 | −0.683 m |
| 3 | D068-2021_UU-S1-12 | 50 | 9 511 592 | 0.436 m | 0.753 m | 0.9882 | −0.237 m |
| 4 | D068-2021_FA-S1-26 | 50 | 14 905 038 | 0.386 m | 0.704 m | 0.9896 | −0.162 m |
| 5 | D075-2021_UF-S1-2 | 25 | 5 325 925 | 0.358 m | 0.591 m | 0.9901 | −0.141 m |
| 6 | D068-2021_UN-S1-28 | 150 | 28 309 840 | 0.246 m | 0.430 m | 0.9849 | −0.100 m |
| 7 | D075-2021_AA-S2-2 | 25 | 4 921 091 | 0.190 m | 0.314 m | 0.9910 | +0.045 m |

UF-S1-23 est un outlier : quelques sous-tuiles du bloc `3-5`/`3-6`/`3-7` ont un MAE de 6–16 m
(R² ~ 0.07–0.17) — à inspecter (GT MNT / prédiction, pas juste « zone urbaine difficile »).
UU-S1-4 reste la 2e plus dure, cohérent avec le parity plot. AA-S2-2 et UN-S1-28 sont les plus faciles.

## Fichiers produits — `stats/flair3d/elevation_parity/`

```
summary.csv              1 ligne / zone : toutes les métriques scalaires (n, MAE, RMSE, R², biais,
                         médiane/p90/p99 de |err|, pearson_r, ols_slope/intercept, min/max/mean/std gt+pred)
metrics.json             idem, imbriqué par zone
per_subtile.csv          MAE/RMSE/R²/n par sous-tuile 5×5 (25 lignes / zone) — pour repérer les sous-zones dures
PARITY_TIKZ_SNIPPET.tex  bloc pgfplots prêt à coller, une figure par zone, valeurs déjà substituées
hexbin_combined.png      hexbin côte-à-côte des 2 zones (`--combine`)
roi_mae_ranking.csv      1 ligne / ROI du dump 873542, tri MAE décroissant (7 ROI D068+D075)
per_subtile_all.csv      MAE/RMSE/R²/biais par sous-tuile, 375 lignes, tri MAE décroissant

<zone>/
  pairs.csv.gz           TOUS les couples (gt,pred) finis — en-tête "gt,pred", ~4.5-4.9 M lignes, 4 décimales
  pairs.npz              idem en arrays float32 (gt, pred) — rechargement Python rapide sans parser le texte
  density_2d.dat         grille de densité <bins>×<bins>, colonnes "x y count logcount"
                         (x = centre bin gt, y = centre bin pred, logcount = log10(count+1)) ;
                         ligne vide entre blocs gt, bin pred variant le plus vite
                         → \addplot3[surf, mesh/ordering=y varies] table[x=x,y=y,z=logcount]
  density_matrix.txt     même densité en matrice brute : <bins> lignes (pred ↑) × <bins> colonnes (gt ↑)
                         → alternative via `matrix plot` ou chargée comme image
  density_meta.json      nx, ny, range_lo/range_hi (étendue carrée commune aux 2 axes), bin_width, count_max
  scatter_50k.dat        sous-échantillon aléatoire seedé, colonnes "gt pred" → \addplot table (marks)
  preview.png            contrôle visuel : imshow(log densité) + diagonale y=x + droite OLS + encart métriques
  hexbin.png             hexbin matplotlib (viridis, y=x crimson, OLS bleu pointillé, encart MAE/RMSE/R²/bias)
```

Étendue de la grille : carré `[floor(min(gt,pred)), ceil(max(gt,pred))]` par zone (`[-3, 27]` pour
AA-S2-2, `[-9, 63]` pour UU-S1-4) — **rien n'est tronqué**, la queue haute (bâtiments) est incluse.

## Faire la figure dans Overleaf

1. Copier `stats/flair3d/elevation_parity/` (ou au moins `<zone>/density_2d.dat` + `PARITY_TIKZ_SNIPPET.tex`)
   dans le projet Overleaf.
2. Préambule : `\usepackage{pgfplots}` + `\pgfplotsset{compat=1.18}`.
3. Coller le bloc voulu depuis `PARITY_TIKZ_SNIPPET.tex` (heatmap `surf` vue de dessus `view={0}{90}`,
   couleur = `log10(count+1)`, + diagonale `y=x` cyan + droite d'ajustement OLS rouge pointillée + `\node`
   avec MAE/RMSE/R²). Ajuster `width`/`height`/`colormap`/`title` au besoin.

Leviers :
- **Compilation lente ?** Un `surf` 200×200 = 40 000 patches, ça peut prendre 10-30 s sous pdflatex.
  Relancer le script avec `--density-bins 120` (ou moins), ou basculer sur `density_matrix.txt` via
  `matrix plot`, ou pré-rendre la densité en PNG et `\addplot graphics`.
- Échelle de couleur linéaire : remplacer `z=logcount` par `z=count` dans le `\addplot3`.
- Nuage de points plutôt que heatmap : `\addplot table[x=gt, y=pred] {<zone>/scatter_50k.dat};` avec
  `only marks, mark size=0.15pt, opacity=0.3`.

## Notes

- **R² n'est calculé nulle part dans le repo** (les evaluators/testers ne font que MAE/RMSE sur erreur
  absolue) — défini dans le script comme coefficient de détermination
  `1 − Σ(pred−gt)² / Σ(gt − mean(gt))²`.
- Le masque ne retire que les points **non-finis** (GT `NaN` ≈ MNT manquant, ~0.14 % des points ici).
  Pas de notion de « void » pour la régression.
- **Taille disque** : `stats/flair3d/elevation_parity/` fait **~100 MB**, presque tout dans les `pairs.*`
  (~48 MB de `.csv.gz` + ~54 MB de `.npz`). À garder en local ou à gitignorer — ne pas commiter
  aveuglément. Le reste (`density_*`, `scatter_*`, `summary`, `preview`, `.tex`) fait < 3 MB.
- Config d'entraînement de référence : `configs/flair3d_default/multi-litept-b-v1m0-flair3d.py`
  (multitâche) ; le mono-tâche `configs/flair3d_default/elevation/litept-b-v1m0-flair3d.py` utilise
  `target_scales={}` (mètres bruts, pas de `0.01`).
