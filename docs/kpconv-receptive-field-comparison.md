# Comparaison KPConv / KPConvX — ScanNet vs DALES / Malibu3D (ALS)

Notes pour adapter KPConvX à l’ALS (Malibu3D+). `dl` = taille de cellule du premier sub-échantillonnage, en mètres (`first_subsampling_dl` chez KPConv ; `grid_size` / `subsample_size` chez KPConvX).

**Sources**

- KPConv ScanNet : [HuguesTHOMAS/KPConv `training_Scannet.py`](https://github.com/HuguesTHOMAS/KPConv/blob/master/training_Scannet.py)
- KPConvX ScanNet : `configs/scannet/semseg-kpconvx-base.py`
- KPConv DALES : [Arjun-NA/KPConv_for_DALES `training_DALES.py`](https://github.com/Arjun-NA/KPConv_for_DALES/blob/master/training_DALES.py)
- KPConvX Malibu3D : `configs/malibu3d_default/multi-kpconvx-v1m0-malibu3d.py`

## Paramètres

| Paramètre | Nom code (KPConvX) | KPConv ScanNet | KPConvX ScanNet | KPConv DALES | KPConvX Malibu3D |
|---|---|---|---|---|---|
| Maille 1er étage | `subsample_size` / `grid_size` | 0.04 m (`first_subsampling_dl`) | 0.02 m | 0.25 m (`first_subsampling_dl`) | 0.10 m |
| Rayon kernel (en cellules) | `kp_radius` | **2.5** rigid / **5.0** deformable (`conv_radius` / `density_parameter`) | 2.3 | **2.5** rigid / **5.0** deformable | 3.2 |
| Influence d’un kernel point (en cellules) | `kp_sigma` | 1.0 (`KP_extent`) | 2.3 | 1.0 (`KP_extent`) | 3.2 |
| Zoom inter-étages | `radius_scaling` | 2.0 (doublement de grille) | 2.2 | 2.0 | 3.0 |
| Fonction d’influence | `kp_influence` | `linear` (`KP_influence`) | `constant` | `linear` | `constant` |
| Agrégation | `kp_aggregation` | `sum` (`convolution_mode`) | `nearest` | `sum` | `nearest` |
| Crop d’entrée | `SphereCrop` / `point_max` | sphère **2 m** (`in_radius`) | 40 000 pts | sphère **20 m** (`in_radius`) | 40 000 pts |
| Voisins | `neighbor_limits` | recherche **par rayon** | kNN (12, 16, 20, 20, 20) | recherche **par rayon** | kNN (12, 16, 20, 20, 20) |
| Points de kernel | `shell_sizes` | 15 (`num_kernel_points`) | (1, 14, 28) → 43 | 15 | (1, 14, 28) → 43 |
| Profondeur encodeur | `layer_blocks` | ~2 blocs / layer, 5 layers | (3, 3, 9, 12, 3) | ~2 blocs / layer, 5 layers | (3, 3, 9, 12, 3) |
| Pooling | `grid_pool` | strided KPConv, ×2 | `True` | strided KPConv, ×2 | `True` |

Chez KPConv original, il n’y a pas un seul `kp_radius` : rigid = 2.5 cellules, deformable = 5.0 (`density_parameter`). Avec `kp_influence='constant'` (KPConvX), `kp_sigma` ne pondère pas.

## Batch size / GPU

Chez KPConv original, `batch_num` = nombre de **sphères** empilées dans un batch TensorFlow (1 GPU). Chez Pointcept, `batch_size` = total de scènes **sur tous les GPU** ; le par-GPU vaut `batch_size / num_gpu`. Mix3D (`mix_prob=0.8` KPConvX) mélange deux scènes dans un slot, ça ne double pas le batch vu par l’optimiseur.

| Paramètre | Nom code | KPConv ScanNet | KPConvX ScanNet | KPConv DALES | KPConvX Malibu3D |
|---|---|---|---|---|---|
| GPU | `GPU_ID` / `num_gpu` / `-g` | **1** (`GPU_ID='0'`) | **1** (exemple `train.sh -g 1`) | **1** (TF ; `GPU_ID` commenté) | **1** (`num_gpu=1`) |
| Batch config | `batch_num` / `batch_size` | 10 | **2** | 4 | `2 * num_gpu` → **2** |
| Batch effectif (optimiseur) | — | **10** | **2** | **4** | **2** |
| Batch / GPU | — | 10 | 2 | 4 | 2 |
| Gradient accumulation | `gradient_accumulation_steps` | — (TF, pas d’accum) | 1 (défaut Pointcept) | — | 1 |
| Unité d’un slot | — | sphère **2 m** | jusqu’à 40k pts | sphère **20 m** | jusqu’à 40k pts |

Pointcept impose `batch_size % world_size == 0` (`default_setup`). Un commentaire local `batch_size=12` / 8 GPU est **invalide** (12 ≱ 8) : ignoré.

Papier KPConvX (ScanNet / S3DIS, pipeline standalone, pas cette config Pointcept) : micro-batch **4**, accum **6** → effectif **24**. La config Pointcept `semseg-kpconvx-base.py` est un exemple minimal (`batch_size=2`), distinct du papier.

## Grandeurs dérivées (mètres)

| Grandeur | Formule | KPConv ScanNet | KPConvX ScanNet | KPConv DALES | KPConvX Malibu3D |
|---|---|---|---|---|---|
| Premier rayon kernel \(r_0\) | `kp_radius × dl` | 0.10 m (rigid) | 0.046 m | 0.625 m (rigid) | 0.32 m |
| \(\sigma_0\) | `kp_sigma × dl` | 0.04 m | 0.046 m | 0.25 m | 0.32 m |
| Ratio crop / `dl` | — | **50×** (2 / 0.04) | variable (nb de pts) | **80×** (20 / 0.25) | variable (nb de pts) |
| Rayon kernel L5 | \(r_0 \times s^{4}\) (KPConvX) ; \(5 \times dl_5\) (KPConv deform) | **3.20 m** | 1.08 m | **20 m** | 25.92 m |
| RF rayon théorique (encodeur, 1 conv/bloc, hors pooling) | \(\sum n_\ell r_\ell\) | ~8.7 m | ~12 m | ~54 m | ~211 m |

Le RF théorique est un **rayon** (pas un diamètre), **borne supérieure** : chaque résiduel = une conv, chaque hop saturé au rayon kernel. Hors pooling (~+3–4 m) et hors décodeur (~+4 m sur KPConvX). Avec kNN, le hop réel peut être plus petit que \(r_\ell\).

## Échelles par layer

| Layer | KPConv ScanNet `dl=0.04` | KPConvX ScanNet `dl=0.02` | KPConv DALES `dl=0.25` | KPConvX Malibu3D `dl=0.10` |
|---|---|---|---|---|
| 1 | voxel **0.04** · r **0.10** (rigid) | voxel 0.020 · r **0.046** · k=12 | voxel **0.25** · r **0.63** (rigid) | voxel 0.10 · r **0.32** · k=12 |
| 2 | **0.08** · r **0.20** (rigid) | 0.044 · r **0.10** · k=16 | **0.50** · r **1.25** (rigid) | 0.30 · r **0.96** · k=16 |
| 3 | **0.16** · r **0.80** (deform) | 0.097 · r **0.22** · k=20 | **1.00** · r **5.0** (deform) | 0.90 · r **2.88** · k=20 |
| 4 | **0.32** · r **1.60** (deform) | 0.213 · r **0.49** · k=20 | **2.00** · r **10** (deform) | 2.70 · r **8.64** · k=20 |
| 5 | **0.64** · r **3.20** (deform) | 0.469 · r **1.08** · k=20 | **4.00** · r **20** (deform) | 8.10 · r **25.92** · k=20 |

KPConv ScanNet / DALES : mêmes `kp_radius` / `KP_extent` / `density_parameter` / architecture. Seuls `dl` et `in_radius` changent (règle Hugues : `in_radius ≈ 50 × dl` ; DALES va à 80×).

KPConvX ScanNet → Malibu3D : `dl` passe de 2 cm à 10 cm, et le default multi élargit aussi kernel et pooling (`kp_radius=3.2`, `s=3.0`, 40k pts) pour que le kernel dépasse un peu un hop de pooling. Ablation w109/2 `_1` : recette ScanNet `kp_radius=2.3`, `s=2.2` (~5.4 m au fond). Pas de sphère métrique fixe, pas de kernels deformable plus larges au fond.

## Glossaire des leviers RF

| Nom | Rôle |
|---|---|
| `dl` / `grid_size` / `subsample_size` | Maille en mètres. Tous les rayons = multiplicateur × `dl`. |
| `kp_radius` | Rayon de la boule où sont posés les points de kernel, en **cellules**. \(r_0 = \texttt{kp\_radius} \times dl\). |
| `kp_sigma` | Portée d’influence d’un kernel point, en cellules. En `linear` : \(w=\max(0,1-d/\sigma)\). Ignoré si `constant`. |
| `radius_scaling` | Homothétie entre étages : kernel, \(\sigma\), et grille de pooling. KPConv original = 2.0 ; KPConvX papier / ScanNet = 2.2 ; Malibu3D default = 3.0. |

## Recette ALS (KPConv DALES vs KPConvX Malibu3D actuel)

La recette DALES ne scale pas l’indoor 2 cm → 10 cm. Elle coarsifie **et** impose un contexte métrique :

1. `dl = 0.25 m` (vs 0.10 m KPConvX Malibu3D).
2. Sphère **20 m**, pas un budget de points. À 40k pts et `dl=0.1 m`, le crop XY réel est plutôt ~5–15 m en canopée.
3. Deformable dès le 3ᵉ étage (`ρ=5`) : dernier kernel = **20 m** = `in_radius`. KPConvX Malibu3D default : `kp_radius=3.2` partout (~26 m au fond). Ablation `_1` : `kp_radius=2.3` (~5.4 m).
4. Voisins géométriques + influence linéaire, vs kNN + `constant`.
