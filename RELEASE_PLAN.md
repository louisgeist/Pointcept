# Plan de release MALiBU3D

Suivi du nettoyage de la branche `malibu3d` avant la création du repo public.
**Fichier de travail : à supprimer avant le squash vers le repo public.**

- Snapshot de la soumission : tag `malibu3d-submission` (commit 430f1da, = `flair3d` au 2026-10-10).
- Release publique : nouveau repo propre (un seul commit), créé à la fin à partir de `malibu3d`.
- Après chaque phase : relancer le harnais et le comparer à la référence (voir Phase 0), puis faire un commit.

## Où on en est (2026-10-11, fin de session)

- Phases 0 à 6 terminées. Phase 6 : commentaires/docstrings (`cce7089`, poussé), puis un commit « valeurs de code » d'après les rapports des agents (**committé en local, PAS poussé** : voir la question 0).
- Rapports des agents de la Phase 6 (valeurs de code à changer + bugs suspects) : `local/release_inventory/phase6_reports.md`. Ce qui reste à trancher est dans « Questions pour Louis » ; ce qui relève des Phases 7/8 est listé à la fin de la Phase 6.
- Prochaine étape : Phase 7 (README, CITATION.cff, LICENSE, environment.yml). La question 0 est réglée (miroir à la fin, plus de push d'ici là) : on peut la commencer.

### Reprendre sans l'historique de la conversation

- Tout se passe sur hecate, dans `/data/geist/Pointcept`, branche `malibu3d` : les données locales, les checkpoints (`ckpt_cleaned/`) et la référence du harnais (`local/regression/ref`) n'existent que là. Données locales au format HF : `data/malibu3d/{train,val,test}` pointent vers `~/data/malibu3d_hecate` (miroir de `data/flair3d_plus` en liens symboliques, `natural_habitat.npy` converti en `(N, 4)` par `local/release_inventory/make_hf_format_mirror.py`, conversion identique aux 25 tuiles du toy HF). `data/flair3d_plus` reste au format de la branche `flair3d`. Référence du harnais recapturée sur ces données (`local/regression/ref` ; l'ancienne : `ref_pre_hf_layout`).
- Notes de travail non versionnées (`local/` est dans `.gitignore`), sur hecate seulement : `local/release_inventory/phase6_reports.md` (rapports détaillés des agents de la Phase 6, avec `fichier:ligne`), `paper_to_runs.md` / `paper_experiments.md` (chiffres du papier ↔ runs wandb/jobs), `code_inventory.md` / `files_inventory.md` (inventaire de la Phase 1), `ast_same.py` (vérifie qu'un diff ne touche que les commentaires et docstrings).
- Commandes :
  - tests : `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.local/share/mamba/envs/pointcept/bin/python -m pytest -q tests -p no:warnings --ignore=tests/regression`
  - harnais (~1 h) : `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $PY tests/regression/harness.py capture --stage all --out local/regression/new`, puis `$PY tests/regression/harness.py compare local/regression/ref local/regression/new` ; après vérification d'un changement voulu : `$PY tests/regression/harness.py accept local/regression/ref local/regression/new <configs ou cas>`
  - GPU : toujours `cuda:0` sur hecate (spconv plante sur `cuda:1`)

### Questions pour Louis (réponses du 2026-10-12)

0. **Fork public** (`github.com/louisgeist/Pointcept`, double aveugle) → **on ne touche à rien pour l'instant**. Le `git push --mirror` vers un repo privé non-fork, puis le nettoyage du fork public, se feront **à la toute fin**. D'ici là : **ne plus pousser** sur `origin` (le miroir partira du dépôt local).
   **Nom du repo de code (2026-10-13) : `github.com/louisgeist/malibu3d`.** Ce nom est aujourd'hui pris par le repo privé du site (GitHub Pages, `louisgeist.github.io/malibu3d/`) ; Louis le renomme en `malibu` (le site passera à `louisgeist.github.io/malibu/`, l'ancienne URL ne sera pas redirigée). Ensuite : créer `louisgeist/malibu3d` (privé, pas un fork), y pousser la branche nettoyée (pas de `--mirror` brut), passer `origin` dessus, puis nettoyer le fork public. Liens à mettre à jour : `CITATION.cff` (fait), carte du dataset HF (« Training code », TODO `MALiBU3D`), carte des checkpoints (`ckpt_cleaned/README.md`, « TODO: GitHub URL »).
1. **Noms de classes en anglais** → fait avec le point 8 : axes en anglais, alignés sur `labels.json` (« Humid / Mesic / Dry ») ; les 43 noms CarHab disparaissent avec le preprocessing.
2. **DALES sans split val** → rien dans le papier ni le README ; la phrase « DALES has no validation split, so validation runs on the test split » est déjà dans la docstring des 14 configs DALES (passe de commentaires de la Phase 6).
3. **Probes PureForest** → fait. `GridProbeClassifier` et l'entraînement en ligne supprimés ; configs `configs/pureforest/probe-*.py` (modèle `PooledFeatureExtractor` + pipeline d'extraction + grille du probe) ; un seul point d'entrée `tools/probe_pureforest.py` (extraction puis L-BFGS, mêmes réglages que le papier). Vérifié : configs identiques aux anciennes (backbone, masques, pipeline, datasets) ; extraction identique au bit près à l'ancien script à seed égale sur `data/pureforest_toy` (LitePT-B, SpUNet, Sonata, KPConvX ; PTv3 non déterministe sur GPU même à seed fixée, écart ancien/nouveau = écart entre deux runs) ; fit relancé sur les embeddings stockés du papier pour les 10 backbones (5 du Tab. 3, Sonata indoor, noRGB, noGNL, monoLC, preECLAIR) : même pooling, même C, même val et test mIoU que l'ancien script.
4. **`enable_wandb = True`** → on garde la valeur amont, rien dans le README.
5. **`black`** → fait : black 26.10.1 (« stable », comme le workflow amont ; garde intact le code amont du merge-base) sur les 194 fichiers .py du fork, sauf `pointcept/models/kpconvx/` (code Apple) et les fichiers amont non modifiés. AST inchangé sur les 141 fichiers reformatés, pytest et harnais (build) OK.
8. **Preprocessing obsolète** → fait. Supprimés : `preprocess_malibu3d.py`, `unzip_raw_archives.py`, `rasterize_forest.py`, `malibu3d_label_remap.py`, `nathab_axes.py`, `scripts/analyze_malibu3d_test_point_voxel_counts.py`, la lecture de `scene_split_manifest.csv` / `patch_id` (catalogue, budgets de voxels, sous-ensemble de val) et la branche CarHab `(N,)` du dataset. Remplacés par `pointcept/datasets/malibu3d_labels.py` (noms du land cover et des 4 axes, = `labels.json`). Plus aucun « v20 » ni `label_definitions` dans les configs. `tile_catalog.py` reste dans `preprocessing/malibu3d/` (importé en voisin par `rasterize_network.py`, sans torch). README du dossier réécrit autour de `rasterize_network.py` et d'APLS.
6. `train.sh` sans `rm -rf` → OK.
7. Config DALES Sonata remise en grille de 12 lr → OK.

## Décisions

| Sujet | Décision |
|---|---|
| Nom | Flair3D / Flair3D+ → MALiBU3D. Classes `MALiBU3D*` (`MALiBU3DDataset`), fichiers et chemins `malibu3d`. Ne **jamais** toucher à « FLAIR-HUB » (le dataset d'imagerie IGN). |
| Datasets gardés | MALiBU3D, DALES, H3D, ECLAIR, OpenGF, PureForest |
| Datasets supprimés | indoor (ScanNet, S3DIS, Structured3D, …), SensatUrban, FOR-instanceV2, SemanticKITTI, nuScenes, Waymo |
| Modèles gardés | LitePT, PTv3 (m1/m2/malibu), SpUNet, KPConvX, Sonata ; `default.py` et `grid_probe.py` |
| Modèles supprimés | Concerto, Utonia, OctFormer, Swin3D, PTv1/v2, Stratified Transformer, SPVCNN, OA-CNNs, PointGroup, PPT, MSC, SGIFormer, context-aware classifier, `kpnext/` (doublon de `kpconvx/utils`) |
| RandLA-Net | Jamais présent sur cette branche ; rien à faire |
| KPConvX | Source : https://github.com/apple-aiml-research/ml-kpconvx/tree/main/Pointcept-wrapper. Copier leur LICENSE dans `pointcept/models/kpconvx/` et les citer |
| `natural_habitat` mono-tâche par point | Config `configs/flair3d_default/natural_habitat/litept-v1m0-flair3d.py` obsolète → supprimer. Garder le code de la cible `natural_habitat` par point seulement si l'ablation A.3 du papier en dépend (à vérifier en Phase 1) |
| Configs cassées par `bn_eval_mode` | Ignorées ; elles partent avec `configs/experiment/` |
| DALES / ECLAIR | Pas de prétraitement local. Couverts en local par l'étage build seulement ; étage forward lancé une fois sur Jean Zay (tag vs version finale) en Phase 8 |
| Pointcept | Licence MIT : garder `LICENSE` + le copyright amont, ajouter « Built on Pointcept » (commit amont) dans le README |

## À livrer sur Hugging Face avec le dataset

- [x] `val_subset_2000tiles.csv` (publié sur HF le 2026-10-12, commit 728b1b4, avec une phrase dans la carte) (renommé, sans « dev ») : les 2000 tuiles val de Jean Zay (`$WORK/Pointcept/data/flair3d_plus/manifests/val_dev_subset_2000.csv`, 9 juillet), réduites aux colonnes `split,tile_id`. À la racine du repo HF, à côté de `tiles.csv`. Fichier prêt : `local/release_inventory/hf_upload/val_subset_2000tiles.csv` (dossier de travail ; original JZ à côté) ; sur hecate, `data/malibu3d/val_subset_2000tiles.csv` (lien vers `data/flair3d_plus/manifests/`). Configs mises à jour, doc courte `docs/val_subset.md` (à intégrer au README / carte HF en Phase 7). La copie de hecate `val_dev_subset_2000.csv` (773 lignes = tout le val de D067) est obsolète. Sélection calculée sur les labels de juillet (land cover v18 + multilabel) : non reconstructible depuis la release, le CSV fait référence.
- [ ] **Lien vers le code dans la carte du dataset HF** (`LouisGeist/MALiBU3D`, `README.md`, ligne « Training code (GitHub, TODO) ») : remplacer `https://github.com/louisgeist/MALiBU3D` par `https://github.com/louisgeist/malibu3d`, au moment où le repo de code devient public. Source de la carte : `$WORK/malibu3d_hf/README.md` sur Jean Zay (identique à la version en ligne).
- [ ] **Lien vers le code dans la carte des checkpoints** (`LouisGeist/MALiBU3D-backbones`, encore privé ; source locale `ckpt_cleaned/README.md`, « `TODO: GitHub URL` ») : mettre `https://github.com/louisgeist/malibu3d`, en même temps.
- [ ] **Release v2 (coordonnées corrigées)** : repack depuis `$SCRATCH/data_pointcept/flair3d_plus_v2` avec `Flair3D-build/scripts/rebuild_v2/jz/repack_hf.sh` (branche `rebuild-flair3d-plus-v2`), journaux dans `$WORK/malibu3d_hf_v2`. Puis upload `--skip-pack --force`, `super_squash_history`, tag `v1.0`. Le squash invalide le hash HF `c064cc7` cité plus bas.
- [ ] **Taille totale du dataset** à recalculer après le pack v2 (somme des zips + fichiers racine ; v1 = 651 GB, v2 ≈ +18 % car `coord.npy` cm-précis se compresse moins bien) et à reporter dans le papier et dans `README.md` (« 650 GB of zipped regions »).
- [ ] Vérifier que le `tiles.csv` publié contient bien la colonne `n_voxels` (utilisée par les budgets de voxels en val/test). Elle est présente dans la copie locale.

## Phase 0 — Snapshot et harnais de non-régression

- [x] Commiter les fichiers restants sur `flair3d`, poser le tag `malibu3d-submission`, créer la branche `malibu3d`
- [x] Écrire le harnais `tests/regression/` (`harness.py`, `cases.py`)
- [x] Capturer la référence dans `local/regression/ref` (build : 1 084 configs ; forward : 49 cas ; train : 15 cas)
- [x] Mesurer le bruit avec deux captures `train` de référence (`ref`, `ref_noise`) et régler les tolérances : loss totale à 2e-3 près à l'itération 1, à 5 % près aux itérations 2 à 10
- [x] Mesurer le bruit de l'étage forward : spconv forcé en `ConvAlgo.Native` (le choix d'algorithme variait d'un processus à l'autre) → ~1e-6 sauf LitePT (bruit diffus GPU : écart moyen relatif ≤ 2e-4, argmax identique ≥ 99,98 %). Critère : tolérance stricte, ou écart moyen relatif ≤ 1e-3 et accord des logits ≥ 99,9 %

Références : `local/regression/ref_tag` = capture du tag, jamais modifiée ; `local/regression/ref` = référence courante, mise à jour seulement pour les changements voulus et vérifiés (configs promues ou corrigées), cas par cas.

Durée : build ~15 min (~3 min une fois `configs/experiment` supprimé), forward ~4 min, train ~15 min.
Usage : build + forward après chaque commit de nettoyage, `all` à la fin de chaque phase.

Commande :
```
PY=~/.local/share/mamba/envs/pointcept/bin/python
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $PY tests/regression/harness.py capture --stage all --out local/regression/new
$PY tests/regression/harness.py compare local/regression/ref local/regression/new [--map map.json]
```

## Phase 1 — Inventaire « à garder » (validé par Louis avant toute suppression)

- [ ] Associer chaque tableau du papier → run wandb → config (point de départ : release HF, commit c064cc7). Ne **pas** utiliser la branche `suppmat` (ancien nettoyage rapide, non fiable)
- [ ] Lister les configs de `configs/experiment/w*` qui sont derrière des résultats du papier
- [ ] Calculer la liste des types de registry et des modules réellement utilisés par ces configs
- [ ] Faire relire et valider l'inventaire par Louis

Constats en cours de route :
- `configs/flair3d_default/natural_habitat/litept-v1m0-flair3d.py` (mono-tâche `natural_habitat` par point) plante déjà au tag : assertion `nll_loss`, labels hors bornes avec les données D067. Le papier utilise les tâches `nathab_*` → candidate à la suppression.
- Les anciennes configs `configs/experiment/w109`–`w110` qui utilisent `bn_eval_mode` ne se construisent plus depuis f869214 (attendu) : 184 configs d'expérience sur 972 au total.
- `configs/pureforest/{kpconvx,spunet}-toy.py` héritent de `cls-{kpconvx,spunet}-v1m0-pureforest.py`, qui n'existent plus → à supprimer.
- `/data` est plein (16 Go libres) : impossible de prétraiter DALES et ECLAIR en local pour l'instant, donc leurs probes ne sont pas couvertes par l'étage forward. Les checkpoints d'ablation OpenGF (monoLC, noGNL, realGN) ne sont que sur Jean Zay.

## Phase 2 — Suppressions en masse

- [x] Promouvoir les configs du papier (vérifiées par comparaison sémantique avec la config qui a produit chaque run) :
  - ablations MALiBU3D, dérivées de la config canonique (format de données actuel) : `configs/flair3d_default/ablations/` (noGNL, GradNorm, natural habitat par point, land cover seul, test sans RGB)
  - probes sans RGB : `configs/{h3d,eclair}/litept-b-v1m0-*-lin_enc-norgb.py`
  - LitePT depuis zéro : `configs/eclair/semseg-litept-b-v1m0-eclair.py` (nouveau) ; `configs/h3d/semseg-litept-b-v1m0-h3d.py` corrigé (4000 époques, lr 5e-4)
  - OpenGF : `mix_prob = 0` partout (comme dans le papier) ; probe KPConvX avec `point_max = 60000` (comme le run du papier)
  - pas de configs de seeds à lr fixé (D1) ; les probes des backbones d'ablation et le PTv3 aléatoire = grilles canoniques avec `weight=` surchargé (à documenter)
- [x] Supprimer `configs/experiment/` (+ `tools/gen_grid_seed_configs.py`, + `tests/test_resolve_config_file.py`, déjà cassé au tag)
- [x] Supprimer les configs amont (scannet, s3dis, concerto, utonia, sonata indoor, nuscenes, waymo, forinstancev2, `_base_/dataset`), `configs/flair3d/` (ancien dev) et les configs debug/toy/LitePT-S de `flair3d_default`, `dales`, `h3d`, `pureforest`
- [x] Supprimer les modèles, datasets, preprocessing, `libs/` et tests inutilisés (`PDNorm` déplacé dans `point_transformer_v3/` ; `Flair3DClimaticDomainDataset` supprimé, plus utilisé ; `environment.yml` installe maintenant `libs/pointrope`, qui manquait)
- [x] Supprimer les scripts racine (`sbatch_*`, `test_*.sh`, `submit_*`), `stats/`, `docs/superpowers`, `docs/opengf_*`, les CSV temporaires. Reportés : les `README_*_geist.md` et `README_flair3dplus.md` (Phase 7, après migration de leur contenu utile) et `docs/bug_review` (Phase 5)
- [x] Trier `scripts/` : garder l'outillage utilisateur (unzip, sous-ensemble de val, comptage des voxels, bench d'inférence, probe PureForest sklearn, exemple de forward HF), supprimer le reste (D4). `LinProbeSbatchHook` retiré des configs de pré-entraînement Sonata (D6)
- [x] Harnais complet + pytest : 0 échec après acceptation des 3 changements voulus (hooks Sonata, `mix_prob=0` OpenGF vérifié en relançant avec `mix_prob=0.8`). Corrigé au passage dans le harnais : en mode `all`, chaque étage tourne dans son propre processus (sinon l'étage build décalait les forwards PT-v3m1 de ~1e-3)

## Phase 3 — Réorganisation

- [x] (fait avec le renommage, Phase 4) Ranger les configs en `configs/malibu3d/{segment,forest,natural_habitat,network,elevation,multitask,probe,pretrain}` et `configs/{dales,h3d,eclair,opengf,pureforest}`
- [x] Remplacer les chemins `weight=` (`/lustre/...`, numéros de job, `ckpt/malibu3d/*`) par la convention de `README_checkpoints.md` : `ckpt/<nom>/model.pth` (873542 → `litept-b`, 1095469 → `ptv3`, 1052217 → `spunet`, 1159986 → `kpconvx`, 862680 → `sonata-pretrained`). Backbones d'ablation non publiés (noGNL, monoLC, realGN, preECLAIR) : les 3 configs OpenGF qui ne différaient que par `weight` sont supprimées → surcharge `--options weight=...` à documenter (Phase 7). Le checkpoint Sonata indoor officiel garde `ckpt/sonata/pretrain-sonata-v1m1-0-base.pth` (à documenter)
- [x] `network_graphs_root` (APLS) pointe vers `/lustre/...` : les graphes GT sont dans le dataset HF, à la racine de chaque zip de ROI (`data/{split}/{dept}_LIDARHD/{roi}/*_ROADS_graph.gpkg`), alors que `network_label_utils.py` attend un dossier plat `{root}/{dept}_{zone}_{NETWORK}_graph.gpkg` → `network_graphs_root=None` lit chaque graphe dans le dossier de sa ROI (nouveau défaut des configs, test unitaire `tests/test_network_graph_paths.py`) ; APLS vérifié de bout en bout sur hecate avec un dossier explicite, et désormais exercé par le harnais
- [x] Valeurs par défaut personnelles retirées du code (`preprocess_flair3d_v2.py`) ; les docstrings et le lien wandb sont traités en Phase 6, l'audit global en Phase 8. ~~Retirer les chemins `/lustre` et `/data/geist` des docstrings/defaults (`preprocess_flair3d_v2.py`, `rasterize_forest.py`, `rasterize_network.py`, `unzip_malibu3d.py`), le lien wandb de `multi-litept-b-v1m0-flair3d.py`~~
- [x] Lanceurs Slurm : tous supprimés en Phase 2 (spécifiques à Jean Zay) ; le README expliquera l'usage générique
- [ ] Harnais (avec une table de correspondance des chemins) + commit

## Phase 4 — Renommage Flair3D → MALiBU3D

- [x] Table de renommage appliquée par script (`local/release_inventory/rename_malibu3d.py`, ordre : `Flair3D-build` protégé ; `data/flair3d_plus/raw/tiles` → `data/malibu3d/tiles` ; `preprocess_flair3d_v2` → `preprocess_malibu3d` ; `unzip_flair3d` → `unzip_raw_archives` ; `flair3d_plus`/`flair3d_default` → `malibu3d` ; `Flair3D+`/`Flair3D` → `MALiBU3D` ; `FLAIR3D` → `MALIBU3D` ; `flair3d` → `malibu3d`) : 138 fichiers modifiés, 37 déplacés (`git mv`), un seul commit
- [x] `configs/flair3d_default/` → `configs/malibu3d/` ; `MALiBU3DDataset`, `MALiBU3DLabelRemap`, `MALIBU3D_*` ; `pointcept/datasets/malibu3d*.py` ; `preprocessing/malibu3d/`
- [x] Layout des données = layout HF : `data/malibu3d/{train,val,test}`, `data/malibu3d/tiles.csv`, `data/malibu3d/manifests/`. En local (hecate) : `data/malibu3d/` = liens symboliques vers `data/flair3d_plus/` (à refaire sur Jean Zay)
- [x] Clés de `meta.json` sur disque : aucune ne contient « flair3d » → format des données inchangé ; clés des checkpoints inchangées
- [x] `grep -i flair3d` vide hors `Flair3D-build` (Phase 6), `RELEASE_*` et READMEs internes ; FLAIR-HUB intact
- [x] Harnais complet avec la table de renommage : 0 échec (configs, poids initiaux, datasets, forwards, entraînements). `local/regression/ref` = capture post-renommage
- [x] `scripts/unzip_malibu3d.py` écrit dans `{output_root}/data/{split}` alors que les configs lisent `data/malibu3d/{split}` → aligné en Phase 5 (`b7c4288`)
- [ ] Régénérer / vérifier les configs citées dans le README des checkpoints HF (`multi-*-v1m0-malibu3d.py`) au moment de la mise à jour de la model card

## Phase 5 — Code obsolète dans les fichiers conservés

- [x] Définitions de labels legacy (v18/v19/v20), compatibilité `scene_split_manifest.csv` : supprimées avec le preprocessing (question 8). Options et branches mortes, trainers inutilisés : voir plus bas
- [x] Classes enregistrées jamais utilisées supprimées (20 transforms, 2 datasets image, `ConcatDataset`, 2 trainers, 5 testers, 3 évaluateurs, 4 hooks, segmentors LoRA/DINO, 3 losses, 3 schedulers, `SpUNetNoSkipBase`) + imports devenus inutiles (dont `peft`)
- [x] Code mort restant (couverture du harnais + `vulture`, en tenant compte des registres) : 5 classes enregistrées jamais référencées (`ExtractAbsZ`, `ElasticDistortion`, `MSELoss`, `LinProbeSbatchHook`, `SpconvNativeConvAlgo`) et 19 fonctions propres au fork ; utilitaires génériques amont conservés
- [x] Revue de bugs : les correctifs 02 à 07 sont déjà dans le code ; `docs/bug_review/` supprimé, test renommé `tests/test_regressions.py`. **Reste ouvert (décision Louis, hors code)** : 01, coordonnées MALiBU3D quantifiées à 0,5 m en Y dans les PLY sources et la release HF
- [x] Harnais complet + pytest + preprocessing octet par octet après chaque étape

### Compatibilité avec la release HF (testée sur une ROI téléchargée : D075 UU-S1-4 + `toy/`)

La release HF ne livre **ni** `meta.json`, **ni** `network.npy`, **ni** `forest.npy` par point, **ni** `land_use.npy`, **ni** `climatic_domain.npy`, **ni** `natural_habitat_multilabel.npy` (README du dataset HF).
- [x] `forest_2d` était entièrement ignoré (géoréférencement lu dans `meta.json`) → lu dans `tiles.csv` (`c88ff4e`)
- [x] `network.npy` à générer : `rasterize_network.py` lit maintenant les graphes dans les dossiers de ROI ; avertissement si absent (`b7c4288`, `c88ff4e`)
- [x] `unzip_malibu3d.py` extrait dans `{output_root}/{split}/…` (`b7c4288`)
- [x] Checkpoint LitePT-B publié sur ces tuiles : 0,97 de précision land cover
- [x] Supprimer les cibles inutilisables avec les données publiées : multilabel natural habitat, classification par tuile (domaine climatique), `natural_habitat` par point (+ `MALiBU3DLabelRemap`), `forest` par point (+ config mono-tâche `forest`, + `forest.npy` du preprocessing) ; définitions de labels réduites à segment default/v20 et natural habitat default + 4 axes (`c6b3d96`, `f5ae663`, `3408077`). Preprocessing vérifié octet par octet contre le tag sur 3 tuiles brutes D068
- [x] `scripts/build_stratified_subset.py` supprimé (dépendait de `natural_habitat_multilabel.npy`, non livré) ; le sous-ensemble est livré tel quel sur HF ; `subset_utils.py` ne garde que son application

## Phase 6 — Commentaires et docstrings

- [x] Passe module par module (7 agents en parallèle, consignes : `local/release_inventory/phase6_brief.md`) : anglais, docstrings courtes, plus de références internes (semaines, jobs, dates, machines, chemins perso, Flair3D-build). Docstrings de configs : modèle, tâche, protocole, table du papier, checkpoint `ckpt/<nom>/model.pth`. Énoncés faux corrigés au passage (voir rapports)
- [x] AST inchangé sur les 184 .py modifiés (`local/release_inventory/ast_same.py`) ; les 2 .sh : seules des lignes `#` ont changé
- [x] pytest (232) + harnais complet → commit
- [x] Commit séparé « valeurs de code » d'après les rapports (harnais : seuls les écarts voulus à l'étape build, acceptés dans `local/regression/ref` ; pytest OK). Au passage, Sonata-FT ne pouvait pas construire son split train avec l'ancien manifest (`min_points` exige `n_points`), ce que corrige le passage à `tiles.csv`. Contenu : noms de runs wandb (H100/A100/epoch_120/warmup0%, suffixes des ablations), `csv_manifest` de Sonata-FT → `tiles.csv`, messages et textes d'aide internes (hecate/JZ/Flair3D-build/mmap fallback/legacy GradScaler/pre-macro-F1), id de run wandb réel dans un test, `train.sh`/`test.sh` (wipe + `DATASET=scannet`), branches de test des datasets supprimés dans `SemSegTester`, petits bugs morts (`FocalLoss.total()`, print de debug, `if …: pass`), config DALES Sonata remise en grille
- [x] `black` (question 5)
- Restent pour la Phase 7/8 (dans les rapports) : `tests/test_checkpoint_forward.py` (chemins de checkpoints locaux, manifests régionaux, clés Sonata à revérifier), `scripts/build_image.sh` (installe OCNN/CLIP/Swin3D), noms TAGS de `scripts/pureforest/run_sklearn_probes_gpu.sh`, en-tête de `malibu3d_leakage_excluded_test_tiles.txt`, `scripts/analyze_malibu3d_test_point_voxel_counts.py` (ne lit pas `tiles.csv`), documenter la compilation de `cpp_wrappers` KPConvX (distutils → Python ≤ 3.11) et le téléchargement du checkpoint Sonata officiel

## Phase 7 — Documentation de release

- [x] README réécrit (remplace celui de Pointcept amont) : présentation, installation, données (téléchargement HF, `unzip_malibu3d.py`, `rasterize_network.py`, sous-ensemble de val), entraînement et évaluation, table des configs, checkpoints, évaluation cross-domaine, résultats (Tab. 2 et 3, chiffres de la colonne « paper » de `paper_to_runs.md`), citation, licence et crédits
- [x] `docs/checkpoints.md` (ex-`README_checkpoints.md`) et `docs/probing.md` (jeux de données en aval, prétraitement, protocole grille → graines, PureForest). `docs/val_subset.md` fondu dans le README
- [x] README internes supprimés (`README_*_geist.md`, `README_grid_then_seed.md`, `README_flair3dplus.md`), ainsi que `scripts/build_image.sh` et `scripts/create_tars.sh` (Docker amont : OCNN/CLIP/Swin3D)
- [x] `LICENSE` : MIT, copyright Pointcept + « The MALiBU3D authors » ; licence MIT d'Apple copiée dans `pointcept/models/kpconvx/LICENSE`
- [x] `CITATION.cff` : 10 auteurs (ordre de Louis), sans affiliations ; TODO : URL du repo, arXiv
- [x] `environment.yml` : env `malibu3d` ; retirés open3d, peft, ocnn, CLIP, torch-sparse, torch-cluster, h5py, ftfy, einops, google-sparsehash, canaux pyg/bioconda ; ajoutés huggingface_hub, laspy + lazrs-python, gdal. **Non testé** : installation propre en Phase 8
- [x] Workflow `formatter.yml` : black 26.10.1 épinglé, `pointcept/models/kpconvx` exclu
- [x] Probes automatiques réintroduites, sans dépendance à Slurm : `PeriodicProbeHook` (`pointcept/engines/hooks/misc.py`) lance en tâche de fond une commande au choix (locale ou `sbatch`) après chaque `epoch_N.pth`, formatée avec `{weight}`, `{epoch}`, `{iter}`, `{probe_dir}`, `{save_path}` ; pas de relance après reprise (`probes/launched.json`) ; résumé `probes/summary.csv`. Ajouté aux deux configs de pré-entraînement Sonata (désactivé par défaut, `probe_command = None`) ; `--options` garde maintenant les espaces des chaînes simples (`DictAction`). Testé de bout en bout sur hecate (pré-entraînement court + 2 vraies probes locales)
- [ ] Sonata pré-entraîné sur ECLAIR (Tab. 4, case TODO du papier) : relancé sur Jean Zay (job 48309, 6×4 H100, ~moitié faite au 2026-10-13, probe 1012572 en attente). Config du job = `configs/eclair/pretrain-sonata-v1m2-eclair.py` de `malibu3d` (au hook `LinProbeSbatchHook` près, retiré exprès) ; `ECLAIRDataset` identique (filtre `min_points={"train": 2000}` contre les vues dégénérées qui donnaient les NaN). Quand le run finit : probes grille → graines DALES/H3D/OpenGF/PureForest avec ce checkpoint, ajouter la ligne aux résultats si elle va dans le papier
- [ ] TODO avant publication : URL du repo dans les cartes HF (dataset et checkpoints ; `CITATION.cff` fait), arXiv, affiliations éventuelles

## Phase 8 — Hygiène et validation finale

- [x] Recherche de traces personnelles (2026-10-13) sur tous les fichiers suivis : chemins (`/lustre`, `/data/geist`, `/home`), `usi32yh`, hecate/Jean Zay/IDRIS, emails, tokens wandb/HF, numéros de jobs, dossiers de semaine, français, TODO, gros fichiers. Propre, sauf `tests/regression/` (harnais : chemins et données de hecate) et la section « # Geist » de `.gitignore` (nettoyée ; entrées personnelles déplacées dans `.git/info/exclude`). Restent volontairement : les TODO arXiv, les repos HF `LouisGeist/...`
- [ ] Au commit final : retirer `tests/regression/` (outil de travail lié à hecate et à `local/regression/ref`)
- [x] Installation propre dans un env neuf (2026-10-16, `~/data/envs/malibu3d_test`, micromamba, CUDA 12.4 / PyTorch 2.5.0 cu124 / spconv-cu124), puis pytest (225 OK) et harnais. `environment.yml` réécrit : conda pour Python, le toolkit CUDA, les bibliothèques géospatiales ; pip pour torch, torch-scatter, spconv. Corrections trouvées par l'installation : `cudnn` de l'amont ne se résolvait plus, `easydict` manquait (KPConvX), `numpy<2` (API C numpy 1.x des enveloppes C++ de KPConvX), `torch_geometric==2.7.0` + `torch-cluster` (la 2.8 exige pyg-lib pour `voxel_grid`), `pytest`. Extensions dans le README (`pip install --no-build-isolation` pour pointops, pointrope, flash-attn 2.7.3 ; `setup.py build_ext --inplace` pour KPConvX, dont les `setup.py` passent de distutils à setuptools). Harnais, à données égales : nouvel env = ancien env au bruit près (2 forwards PTv3 OpenGF à 0,0011 d'écart relatif moyen, argmax identique).
- Référence du harnais recapturée le 2026-10-16 (ancien env) : les rasters `forest_2d` locaux de D067 ont été remplacés par la version HF le 2026-10-15 (autre session) ; ancienne référence dans `local/regression/ref_pre_forest2d_hf`
- [ ] Reproduire un ou deux chiffres du papier sur Jean Zay à partir des checkpoints HF
- [ ] Sur Jean Zay, lancer l'étage forward du harnais sur les probes DALES et ECLAIR, au tag puis sur la version finale, et comparer
- [x] Louis a renommé le repo du site `louisgeist/malibu3d` en `malibu` (2026-10-13 ; site → `louisgeist.github.io/malibu/`)
- [ ] Créer `louisgeist/malibu3d` (privé, pas un fork), y pousser la branche nettoyée (squash, pas de `--mirror` brut), supprimer ce fichier
- [ ] Le jour de la publication : passer `louisgeist/malibu3d` en public, puis mettre à jour les deux liens des cartes HF (dataset et checkpoints, voir « À livrer sur Hugging Face »)
