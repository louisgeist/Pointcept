# UMAP des features (Sonata vs LitePT-B, encodeur vs décodeur)

Compare visuellement, coloré par classe, les features gelées de :
- **Sonata** (PT-v3m2, encodeur multiscale, 1232ch)
- **LitePT-B décodeur** (hypercolumn dec + bottleneck, 1404ch)
- **LitePT-B encodeur** (multiscale, 1386ch)

Trois jeux de données couverts :
- **DALES** (8 classes, transfert cross-domain via GridProbe) — `scripts/extract_dales_grid_probe_features.py`
- **Flair3D+** (15 classes segment v20/finer12, domaine natif des 2 checkpoints, D068/D075 par défaut) —
  `scripts/extract_flair3d_grid_probe_features.py`
- **H3D** (Hessigheim 3D, 11 classes, transfert cross-domain via GridProbe, split `val` — 4 tuiles déjà
  mirrorées localement) — `scripts/extract_h3d_grid_probe_features.py`

Tout tourne **en local sur hecate** (2x A6000) — pas besoin de JZ à part pour récupérer les checkpoints
une fois. Les trois scripts d'extraction partagent leur logique (shim flash-attn, contournement spconv,
échantillonnage stratifié) via `scripts/_grid_probe_extract_common.py`.

Rangement sous `stats/umap/` (dossier intermédiaire dédié à cette analyse, séparé du reste de `stats/`
— ex. `stats/flair3d/` contient par ailleurs des stats sans rapport, label distributions etc.) : un
sous-dossier par dataset (`stats/umap/dales/`, `stats/umap/flair3d/`, `stats/umap/h3d/`), avec dans
chacun un sous-dossier `data/` pour les `.npz` de features (sortie des scripts d'extraction 1a/1b/1c) et,
au même niveau, un sous-dossier `plots/` pour les figures produites par le script de visu (2) —
`visualize_dales_umap_features.py` déduit `plots/` automatiquement à partir du dossier `data/` du premier
`--features` passé (sibling de `data/`), donc pas besoin de préciser `--output` pour que ça range au bon
endroit.

## Checkpoints

Déjà rapatriés dans `ckpt/` (mêmes checkpoints pour DALES, Flair3D+ et H3D — ce sont les checkpoints natifs
Flair3D+, DALES et H3D sont juste des transferts cross-domain dessus) :
- `ckpt/862680/epoch_120.pth` (Sonata pretrain, job 862680)
- `ckpt/873542/model_best.pth` (LitePT-B Flair3D+ multitask supervised pretrain, job 873542)
- `ckpt/1095469/model_best.pth` (PT-v3-malibu Flair3D+ multitask supervised pretrain, job 1095469)

Pour re-télécharger depuis JZ si besoin :
```bash
scp -J passerelle usi32yh@jean-zay.idris.fr:/lustre/fsn1/projects/rech/unv/usi32yh/logs/pointcept_logs/slurm/862680/model/epoch_120.pth \
  lgeist@hecate:/data/geist/Pointcept/ckpt/862680/epoch_120.pth
scp -J passerelle usi32yh@jean-zay.idris.fr:/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/873542/model/model_best.pth \
  lgeist@hecate:/data/geist/Pointcept/ckpt/873542/model_best.pth
```

## 1a. Extraction DALES (`scripts/extract_dales_grid_probe_features.py`)

Reprend directement les configs GridProbe DALES existants (backbone + CheckpointLoader rename/exclude).
Tourne sur le split `test` DALES (déjà mirroré localement, `data/dales/test/`), échantillonnage stratifié
par classe (8 classes, `Unknown`/ignore exclu), sauvegarde `coord`/`feat`(fp16)/`segment`/`tile_idx` dans
un `.npz`.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pointcept
cd /data/geist/Pointcept
export PYTHONPATH="$PWD"

python3 scripts/extract_dales_grid_probe_features.py \
  --config configs/dales/sonata-v1m2-dales-lin-grid.py \
  --weight ckpt/862680/epoch_120.pth \
  --output stats/umap/dales/data/sonata_enc.npz \
  --points-per-class 3000 --max-draws 200 --seed 0 --device cuda:0

python3 scripts/extract_dales_grid_probe_features.py \
  --config configs/dales/litept-b-v1m0-dales-lin-grid.py \
  --weight ckpt/873542/model_best.pth \
  --output stats/umap/dales/data/litept_dec.npz \
  --points-per-class 3000 --max-draws 200 --seed 0 --device cuda:0

python3 scripts/extract_dales_grid_probe_features.py \
  --config configs/dales/litept-b-v1m0-dales-lin-grid-enc.py \
  --weight ckpt/873542/model_best.pth \
  --output stats/umap/dales/data/litept_enc.npz \
  --points-per-class 3000 --max-draws 200 --seed 0 --device cuda:0
```

Chaque run prend ~10-15 min (11 tuiles DALES, ~50-75 tirages typiquement pour remplir les 8 classes).

## 1b. Extraction Flair3D+ (`scripts/extract_flair3d_grid_probe_features.py`)

Lit directement les configs *natifs* Flair3D+ (pas de GridProbe ici — ces checkpoints ont été
pretrainés/finetunés directement sur Flair3D+, pas de transfert). Réduit chaque config à une lecture
mono-tâche `segment` (Flair3DDataset le supporte nativement), et pointe sur un manifeste local
D068+D075 (déjà mirroré, `data/flair3d_plus/test/{D068,D075}-2021_LIDARHD/`, ~4750 sous-tuiles combinées
— bien plus petites que les tuiles DALES donc pas besoin d'autant de tirages par classe).

```bash
python3 scripts/extract_flair3d_grid_probe_features.py \
  --config configs/flair3d_default/probe/sonata-v1m2-flair3d-lin-grid.py \
  --weight ckpt/862680/epoch_120.pth \
  --output stats/umap/flair3d/data/sonata_enc.npz \
  --points-per-class 3000 --max-draws 400 --seed 0 --device cuda:0

python3 scripts/extract_flair3d_grid_probe_features.py \
  --config configs/flair3d_default/multi-litept-b-v1m0-flair3d.py \
  --weight ckpt/873542/model_best.pth \
  --output stats/umap/flair3d/data/litept_dec.npz \
  --points-per-class 3000 --max-draws 400 --seed 0 --device cuda:0

python3 scripts/extract_flair3d_grid_probe_features.py \
  --config configs/flair3d_default/multi-litept-b-v1m0-flair3d.py \
  --weight ckpt/873542/model_best.pth \
  --output stats/umap/flair3d/data/litept_enc.npz \
  --points-per-class 3000 --max-draws 400 --seed 0 --device cuda:0 --enc-mode

# PT-v3-malibu (Malibu3D = Flair3D+ native domain), job 1095469
python3 scripts/extract_flair3d_grid_probe_features.py \
  --config configs/flair3d_default/multi-ptv3-v1m0-flair3d.py \
  --weight ckpt/1095469/model_best.pth \
  --output stats/umap/flair3d/data/ptv3_dec.npz \
  --points-per-class 3000 --max-draws 400 --seed 0 --device cuda:0

python3 scripts/extract_flair3d_grid_probe_features.py \
  --config configs/flair3d_default/multi-ptv3-v1m0-flair3d.py \
  --weight ckpt/1095469/model_best.pth \
  --output stats/umap/flair3d/data/ptv3_enc.npz \
  --points-per-class 3000 --max-draws 400 --seed 0 --device cuda:0 --enc-mode
```

Notes :
- `--enc-mode` bascule le backbone en `enc_mode=True` (LitePT-B 1386ch / PT-v3-malibu 992ch,
  hypercolumn encodeur) au lieu du décodeur natif (LitePT-B 1404ch / PT-v3-malibu 1024ch). Pour le
  décodeur, le script force `dec_traceable=True` (LitePT) ou `traceable=True` (PT-v3-malibu) — absent
  du config natif multitask, qui ne lit que la sortie du dernier étage decoder — pur ajout de
  bookkeeping, ne change aucun poids ni aucune valeur calculée (voir docstring du script).
  Malibu3D désigne ici le domaine natif Flair3D+ du backbone `PT-v3-malibu`.
- `--csv-manifest` (défaut `data/flair3d_plus/raw/scene_split_manifest_D068_D075.csv`, construit une fois
  en concaténant les manifestes par département — `Flair3DDataset` n'accepte qu'un seul fichier). Pour un
  autre combo de départements, reconstruire ce CSV (voir le script pour l'exemple `csv`/`pandas`).
- 15 classes segment v20/finer12 directement (pas de regroupement) : Building, Greenhouse, Impervious
  surface, Other soil, Herbaceous, Vineyard, Brushwood, Other infrastructures, Swimming pool, Water,
  Deciduous, Coniferous, Bridge, Agricultural soil, Soil under vegetation.
- Chaque run prend quelques minutes (tuiles beaucoup plus petites que DALES : ~100-280k points bruts par
  sous-tuile contre ~11M pour un tuile DALES entière).

## 1c. Extraction H3D (`scripts/extract_h3d_grid_probe_features.py`)

Reprend les configs GridProbe H3D existants sous `configs/experiment/w109/5/13h_adamw_h3d/` (backbone +
CheckpointLoader rename/exclude — pas de config canonique sous `configs/h3d/` pour ces variantes lin-grid,
seulement les `semseg-*.py` d'entraînement complet). Tourne sur le split `val` H3D (déjà mirroré
localement, `data/h3d/val/`, 4 tuiles), échantillonnage stratifié par classe (11 classes, `Void`/ignore
exclu), sauvegarde `coord`/`feat`(fp16)/`segment`/`tile_idx` dans un `.npz`.

```bash
python3 scripts/extract_h3d_grid_probe_features.py \
  --config configs/experiment/w109/5/13h_adamw_h3d/sonata-v1m2-h3d-lin-grid_6.py \
  --weight ckpt/862680/epoch_120.pth \
  --output stats/umap/h3d/data/sonata_enc.npz \
  --points-per-class 3000 --max-draws 100 --seed 0 --device cuda:0

python3 scripts/extract_h3d_grid_probe_features.py \
  --config configs/experiment/w109/5/13h_adamw_h3d/litept-b-v1m0-h3d-lin_4.py \
  --weight ckpt/873542/model_best.pth \
  --output stats/umap/h3d/data/litept_dec.npz \
  --points-per-class 3000 --max-draws 100 --seed 0 --device cuda:0

python3 scripts/extract_h3d_grid_probe_features.py \
  --config configs/experiment/w109/5/13h_adamw_h3d/litept-b-v1m0-h3d-lin_5.py \
  --weight ckpt/873542/model_best.pth \
  --output stats/umap/h3d/data/litept_enc.npz \
  --points-per-class 3000 --max-draws 100 --seed 0 --device cuda:0
```

Notes :
- Contrairement à DALES/Flair3D+, H3D n'a pas de `strength`/intensité sur disque (seulement
  `coord`/`color`/`segment`) — zero-fill via `FillMissingFeat`, comme documenté dans les configs `*_h3d`
  sources (`enc_mode=True` pour Sonata et LitePT-B encodeur, `enc_mode=False, dec_traceable=True` déjà
  baké dans le config LitePT-B décodeur — pas de surcharge nécessaire côté script, contrairement au cas
  Flair3D+ natif en 1b).
- 11 classes (`Void`/ignore exclu) : Low Vegetation, Impervious Surface, Vehicle, Urban Furniture, Roof,
  Façade, Shrub, Tree, Soil or Gravel, Vertical Surface, Chimney.
- Chaque run prend quelques minutes (4 tuiles, ~35-40 tirages typiquement pour remplir les 11 classes).

## 1d. Baseline aléatoire (`--rand-init`, sur les 4 scripts d'extraction)

Pour comparer les features pré-entraînées/finetunées à une baseline "from scratch", passer `--rand-init`
au lieu de `--weight` : le backbone est construit puis directement passé en forward sans passer par le
`CheckpointLoader` — il garde son init aléatoire fraîche (celle de `nn.Module`/spconv à la construction).
`--weight` devient optionnel (ignoré si fourni avec `--rand-init`) ; erreur explicite si ni l'un ni
l'autre n'est passé.

```bash
python3 scripts/extract_dales_grid_probe_features.py \
  --config configs/dales/litept-b-v1m0-dales-lin-grid-enc.py \
  --rand-init \
  --output stats/umap/dales/data/litept_enc_randinit.npz \
  --points-per-class 3000 --max-draws 200 --seed 0 --device cuda:0
```

Même flag sur `extract_eclair_grid_probe_features.py`, `extract_h3d_grid_probe_features.py` et
`extract_flair3d_grid_probe_features.py` (ajouter `--enc-mode` sur eclair/flair3d si on veut la variante
encodeur LitePT-B plutôt que le décodeur natif, comme pour une extraction normale). Le `.npz` produit
garde une trace de la provenance (`rand_init=True`, `weight_path=""` dans les métadonnées) — mêmes clés
`coord`/`feat`/`segment`/`class_names` que d'habitude, donc utilisable directement dans le script de visu
(étape 2) en lui donnant un nom de panneau du genre `litept_enc_randinit`.

Options communes aux quatre scripts d'extraction : `--weight` (obligatoire sauf si `--rand-init`),
`--points-per-class` (plafond par classe), `--point-max` (taille du SphereCrop par tirage, ne pas monter —
voir gotchas), `--max-draws` (nb de tirages tuile+crop-aléatoire avant d'abandonner sur les classes rares),
`--seed`. **Toujours `--device cuda:0`** (voir gotchas). Les extractions d'un même run sont indépendantes →
peuvent tourner en parallèle, mais toutes sur `cuda:0`.

## 2. Visualisation UMAP (`scripts/visualize_dales_umap_features.py`)

100% local/offline, ne relit que les `.npz` (pas besoin de modèle/checkpoint). Fonctionne pour DALES,
Flair3D+ et H3D indifféremment (dataset-agnostic, lit `class_names` depuis le `.npz`). Fait exprès pour
être bidouillé directement (court et plat).

### Flair3D
```bash
python3 scripts/visualize_dales_umap_features.py \
  --features "Sonata (enc, 1232ch)"=stats/umap/flair3d/data/sonata_enc.npz \
             "LitePT-B (dec, 1404ch)"=stats/umap/flair3d/data/litept_dec.npz \
             "LitePT-B (enc, 1386ch)"=stats/umap/flair3d/data/litept_enc.npz \
             "PT-v3-malibu (dec, 1024ch)"=stats/umap/flair3d/data/ptv3_dec.npz \
  --n-neighbors 50 --min-dist 1 --metric euclidean --seed 0 --format pdf --points-per-class 1000
```
### DALES
```bash
python3 scripts/visualize_dales_umap_features.py \
  --features "Sonata (enc, 1232ch)"=stats/umap/dales/data/sonata_enc.npz \
             "LitePT-B (dec, 1404ch)"=stats/umap/dales/data/litept_dec.npz \
             "LitePT-B (enc, 1386ch)"=stats/umap/dales/data/litept_enc.npz \
  --n-neighbors 50 --min-dist 1 --metric euclidean --seed 0 --format pdf --points-per-class 2000
```
### H3D
```bash
python3 scripts/visualize_dales_umap_features.py \
  --features "Sonata (enc, 1232ch)"=stats/umap/h3d/data/sonata_enc.npz \
             "LitePT-B (dec, 1404ch)"=stats/umap/h3d/data/litept_dec.npz \
             "LitePT-B (enc, 1386ch)"=stats/umap/h3d/data/litept_enc.npz \
  --n-neighbors 50 --min-dist 1 --metric euclidean --seed 0 --format pdf --points-per-class 2000
```

### ECLAIR
```bash
python3 scripts/visualize_dales_umap_features.py \
  --features "Sonata (enc, 1232ch)"=stats/umap/eclair/data/sonata_enc.npz \
             "LitePT-B (dec, 1404ch)"=stats/umap/eclair/data/litept_dec.npz \
             "LitePT-B (enc, 1386ch)"=stats/umap/eclair/data/litept_enc.npz \
  --n-neighbors 50 --min-dist 1 --metric euclidean --seed 0 --format pdf --points-per-class 1500
```

### Baseline aléatoire vs pré-entraîné
Comparer côte à côte un panneau pré-entraîné et son équivalent `--rand-init` (voir 1d) — même dataset,
même mode enc/dec, juste ajouter le panneau `..._randinit` à `--features` :
```bash
python3 scripts/visualize_dales_umap_features.py \
  --features "LitePT-B enc (pretrained)"=stats/umap/dales/data/litept_enc.npz \
             "LitePT-B enc (rand init)"=stats/umap/dales/data/litept_enc_randinit.npz \
  --n-neighbors 50 --min-dist 1 --metric euclidean --seed 0 --format pdf --points-per-class 2000
```

### Liste commandes
Paramètres à explorer :
- `--n-neighbors` (défaut 30) — le plus impactant. Petit (5-15) = structure locale/fine ; grand (50-100) = structure globale.
- `--min-dist` (défaut 0.1) — 0 = clusters denses/collés ; proche de 1 = plus étalé.
- `--metric` (défaut `cosine`) — tester aussi `euclidean`.
- `--points-per-class` (défaut : tout le `.npz`) — sous-échantillonne avant de fitter UMAP, pratique pour
  itérer vite sans attendre le fit sur toutes les données (utile surtout pour Flair3D+, 15 classes × 3000
  = 45k points par panneau).
- `--format` (défaut `png`) — `pdf`/`svg`/... pour le nom de fichier par défaut.
- `--seed` — vérifier qu'un pattern observé n'est pas juste un artefact d'init.

`--output` par défaut encode le dataset + les params dans le nom
(`<dataset>_umap_nn<n>_md<d>_<metric>_n<total>.<format>`, `<dataset>` déduit du dossier `data/` du premier
`--features`) donc pas besoin de le préciser à chaque essai — les sweeps ne s'écrasent pas entre eux. Le
titre en haut de la figure rappelle aussi `n_neighbors`/`min_dist`/`metric`.

Couleurs par classe définies dans `CLASS_COLORS` en haut du script — palette DALES officielle (Ground
beige, Vegetation vert foncé, Cars magenta, Trucks jaune, Power lines rose grisé, Fences vert vif, Poles
orange, Buildings rouge brique, Unknown bleu nuit) + Flair3D segment v20 + ECLAIR + H3D, avec des teintes
partagées entre datasets pour les classes équivalentes (`Ground`/`Vegetation`/`Buildings`/`Poles` DALES↔ECLAIR,
`Vehicle` ECLAIR↔H3D, etc. — voir les commentaires en tête du dict). Fallback sur `tab20` pour toute classe
non listée.

### Mode papier (`--minimal`)

Pour des figures prêtes à insérer dans le papier (pas de légende, pas de titre, pas de cadre) : passer
`--minimal`. Dans ce mode le script ne produit plus une figure combinée à plusieurs panneaux — chaque
panneau (`--features name=path`) est sauvegardé comme sa **propre image individuelle**, sans légende, sans
titre (ni par panneau ni le titre `n_neighbors=...` en haut), sans cadre d'axes (spines retirées).

#### Flair3D (Malibu3D pour PT-v3-malibu)
```bash
python3 scripts/visualize_dales_umap_features.py \
  --features sonata_enc=stats/umap/flair3d/data/sonata_enc.npz \
             litept_dec=stats/umap/flair3d/data/litept_dec.npz \
             litept_enc=stats/umap/flair3d/data/litept_enc.npz \
             litept_enc_randinit=stats/umap/flair3d/data/litept_enc_randinit.npz \
             ptv3_enc=stats/umap/flair3d/data/ptv3_enc.npz \
             ptv3_dec=stats/umap/flair3d/data/ptv3_dec.npz \
  --n-neighbors 50 --min-dist 1 --metric euclidean --seed 0 --format pdf --points-per-class 1000 \
  --minimal --save-embeddings
```

#### DALES
```bash
python3 scripts/visualize_dales_umap_features.py \
  --features sonata_enc=stats/umap/dales/data/sonata_enc.npz \
             litept_dec=stats/umap/dales/data/litept_dec.npz \
             litept_enc=stats/umap/dales/data/litept_enc.npz \
             litept_enc_randinit=stats/umap/dales/data/litept_enc_randinit.npz \
             ptv3_enc=stats/umap/dales/data/ptv3_enc.npz \
  --n-neighbors 50 --min-dist 1 --metric euclidean --seed 0 --format pdf --points-per-class 2000 \
  --minimal --save-embeddings
```

- `--output` en mode `--minimal` désigne un **dossier** (pas un fichier) où écrire les panneaux — optionnel,
  par défaut le même `plots/` habituel (sibling de `data/` du premier `--features`).
- Nom de fichier par panneau : `<dataset>_<nom_panneau>_umap_nn<n>_md<d>_<metric>_n<points>.<format>` — donc
  préférer des noms de panneau courts sans espace/parenthèse en mode `--minimal` (`litept_enc` plutôt que
  `"LitePT-B (enc, 1386ch)"`) puisqu'ils finissent tels quels dans le nom de fichier.
- Le mode combiné par défaut (sans `--minimal`) est inchangé (légende + titre + cadre + une seule figure).

### Sauvegarder les données brutes post-UMAP (`--save-embeddings`)

Pour retoucher le style d'une figure (couleurs, taille des points, légende...) sans avoir à refitter UMAP
(l'étape coûteuse), passer `--save-embeddings` : écrit un `.npz` par panneau (indépendant de `--minimal`,
compatible avec le mode combiné comme avec le mode papier) contenant l'embedding 2D + le nécessaire pour
retracer la figure telle quelle :
- `embedding` (N, 2, float32), `segment` (N,), `class_names`
- métadonnées : `n_neighbors`, `min_dist`, `metric`, `seed`, `feat_channels`, `source` (chemin du `.npz` de
  features d'origine)

```bash
python3 scripts/visualize_dales_umap_features.py \
  --features sonata_enc=stats/umap/dales/data/sonata_enc.npz \
  --n-neighbors 50 --min-dist 1 --metric euclidean --seed 0 \
  --save-embeddings
```

Écrit par défaut dans `stats/umap/<dataset>/embeddings/` (sibling de `data/`/`plots/`), nommage identique
au reste (`<dataset>_<panneau>_umap_nn<n>_md<d>_<metric>_n<points>.npz`) ; `--embeddings-output` pour
choisir un autre dossier.

## Gotchas locaux hecate (déjà gérés dans `scripts/_grid_probe_extract_common.py`, pas besoin d'y toucher)

- **Pas besoin d'installer flash-attn.** LitePT-v1 (et PT-v3m2 en usage classique) le requièrent, mais
  le shim commun monkeypatch `flash_attn` par une implémentation basée sur `scaled_dot_product_attention`
  (même calcul, juste pas le kernel fusionné — indifférent pour une extraction ponctuelle).
- **`cuda:1` est cassé pour tout op spconv** sur cette machine (illegal memory access, même sur un
  `SubMConv3d` isolé trivial) — toujours `--device cuda:0` pour LitePT-v1/PT-v3m2 en local.
- **L'auto-tuner spconv (`MaskImplicitGemm`) plante** dès qu'une 2e forme de conv distincte est utilisée
  dans le process (`can't find suitable algorithm`) — `spconv-cu118==2.3.8` local vs `spconv-cu124` dans
  `environment.yml` (donc probablement OK sur JZ). Contourné en forçant `algo=ConvAlgo.Native` sur
  chaque couche spconv après construction du modèle (`force_native_conv_algo`).
- **Tuiles DALES entières trop grosses pour un seul forward** (~11M points post-GridSample) → crash
  spconv différent (illegal memory access dans le stem). D'où le crop `SphereCrop(point_max=102400)`
  (la valeur déjà utilisée partout ailleurs dans ce repo) + plusieurs tirages aléatoires par tuile pour
  quand même couvrir les classes rares. (Non-problème sur Flair3D+ : les sous-tuiles y sont déjà petites.)

Aucun de ces trois derniers points n'est lié à une modif locale du repo (vérifié via `git log` sur
`litept_v1.py`/`modules.py`) — probablement juste `spconv-cu118` vs `spconv-cu124` + GPU local (A6000,
sm_86) vs JZ (A100/H100).
