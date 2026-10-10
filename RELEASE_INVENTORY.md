# Inventaire de la Phase 1 — à valider

Synthèse des trois inventaires détaillés, rangés dans `local/release_inventory/` (dossier ignoré par git) :
- `paper_to_runs.md` : chaque chiffre du papier → run wandb → config ;
- `code_inventory.md` : code de `pointcept/` et `libs/` utilisé ou non ;
- `files_inventory.md` : scripts, outils, tests, docs et fichiers racine.

**Fichier de travail, à supprimer avant le squash.** Pour chaque décision, ma recommandation est en gras. Réponds seulement là où tu n'es pas d'accord.

---

## 1. Ce qui a produit les chiffres du papier

| Résultat | Origine | Config dans le repo |
|---|---|---|
| Tab. 2 (5 modèles multitâches) | runs `era2p5gs`, `uzhxltcv`, `wznbvbgm`, `8t6q5nnv`, `u3j665te` (= checkpoints HF) | `configs/flair3d_default/multi-*` (équivalentes à l'entraînement) ✅ |
| Tab. 2 débit d'inférence | pas dans wandb ; le bench local du 2026-08-27 donne d'autres valeurs | `scripts/bench_inference_speed.py` |
| Tab. 3 H3D | jobs 139391–139395 | `configs/h3d/*` ✅ (sauf LitePT, voir §2) |
| Tab. 3 DALES / ECLAIR | anciennes seeds à lr fixé (`w110/2/grid_seed_*`) | **pas** les configs canoniques (voir §2) |
| Tab. 3 OpenGF | probes sans Mix3D (`w116/1/opengf_lin_nomix`) | les configs canoniques ont encore `mix_prob=0.8` |
| Tab. 3 PureForest | probe sklearn, `stats/pureforest/sklearn_probe/*/metrics.json` (10/10 identiques) | `configs/pureforest/cls-*-lin-grid-enc.py` + scripts sklearn ✅ |
| Tab. 3 LitePT depuis zéro | `1zv81u3h`, `m13e6i7e`, `kwbhlruh`, `2wp8kn77`, `2wmafl7w` | DALES et PureForest canoniques ; H3D, ECLAIR et OpenGF seulement dans `w*` |
| Tab. 4 / A.2 / A.3 / A.7 | noGNL `2n5rn10b`, monoLC `ox7xgu13`, realGN `ldry5inn`, natural habitat par point `6qiksdcc`, PTv3 aléatoire | seulement dans `w*` |

## 2. Décisions

**D1. Probes DALES / ECLAIR.** Les chiffres publiés viennent des configs de seeds dont le lr a été fixé à la main d'après une grille plus ancienne (`w110/2`). Relancées par `grid_then_seeds`, les grilles canoniques donnent d'autres chiffres. Même chose pour LitePT sur H3D : la grille choisit 5e-2, qui donne 72.1 ; le papier utilise 5e-3, forcé à la main, qui donne 75.0.
→ **Livrer les grilles canoniques (le protocole décrit dans le papier) ET promouvoir les configs de seeds à lr fixé dans `configs/<dataset>/paper/` pour reproduire les chiffres exacts. Le README dira honnêtement que ces lr viennent d'une grille antérieure.**

**D2. OpenGF.** → **Passer `mix_prob=0` dans `configs/opengf/*`, ce qui correspond au papier, au lieu de promouvoir les 9 configs `w116/1`. Même chose pour la config depuis zéro.**

**D3. Ablations** (noGNL, realGN, natural habitat par point A.3, land cover seul, noRGB, pré-entraînement sur ECLAIR, PTv3 aléatoire). Ce sont 38 configs de `w*` au total.
→ **Les promouvoir dans `configs/malibu3d/ablations/` et `configs/<dataset>/ablations/`, en retirant `bn_eval_mode` (qui plante à HEAD). Les probes des backbones d'ablation seront des surcharges de `weight=` documentées, pas des configs dupliquées.**

**D4. Scripts à trancher.**
- `bench_inference_speed.py` → **garder, en le retravaillant** (il reproduit le débit du Tab. 2).
- Baselines natural habitat avec le comptage des labels, `oracle_landcover_to_forest.py`, probes PureForest par échelle, viewers viser (OpenGF, PureForest) → **supprimer** : ils ne sont pas dans le papier.
- `visualize_network_mask.py` → **supprimer** (on pourra le réintroduire si on veut un outil de visualisation APLS).
- `sbatch_grid_then_seeds_h100.sh` → **supprimer** ; le README expliquera l'usage sous Slurm.
- `README_flair3dplus.md` → **migrer les descriptions de modalités dans la doc dataset, puis supprimer.**

**D5. Code.**
- `PDNorm`, importé par PT-v3m1 depuis `point_prompt_training` → **déplacer `PDNorm` dans `point_transformer_v3/`, puis supprimer PPT.**
- FOR-instanceV2 → **supprimer** (dataset, preprocessing, configs `w115/5`).
- `configs/_base_/dataset/` et `configs/sonata/` (Sonata indoor amont) → **supprimer**, avec `preprocessing/scannet`.
- `preprocessing/flair3d/` (v1, remplacé par `flair3d_plus`) → **supprimer.**
- `kpconvx/cpp_wrappers/cpp_neighbors` → **garder** (arbre KPConvX amont intact).
- `DefaultLORASegmentorV2` et l'import `peft` → **supprimer** ; retirer open3d, MinkowskiEngine, clip, transformers et torchsparse de l'environnement.
- Les classes mortes des fichiers conservés (20 transforms, 2 trainers, 5 testers, 3 evaluators, 4 hooks, 3 losses, 3 schedulers) → **supprimer en Phase 5**, avec le harnais.

**D6. Artefacts à livrer** (déjà dans le plan) : `val_dev_subset_2000.csv` sur HF. Vérifier `n_voxels` dans `tiles.csv`. Retirer `LinProbeSbatchHook` des configs de pré-entraînement Sonata.

## 3. Questions ouvertes (pour toi, pas bloquantes pour la Phase 2)

1. **noRGB multitâche (LC 65.3 / APLS 42.1)** : aucun run dans wandb, rien trouvé dans les logs JZ. La config probable est `w110/4/nocolor_test/multi-litept-b-v1m0-flair3d-nocolor_1.py`, un test seul sur 873542. Te souviens-tu du job ?
2. **Débit du Tab. 2** : de quel bench viennent 1.67 / 1.94 / 1.72 / 1.77 / 0.54 ? Et pour Sonata, est-ce le probe ou `multi-sonata-ft` ?

## 4. Écarts papier ↔ runs (à corriger pour la camera-ready, hors nettoyage)

- **H3D, KPConvX 75.1 et noGNL 75.0** : les meilleurs runs donnent 75.050 et 74.945 → 75.1 et 74.9 en arrondi standard.
- **ECLAIR depuis zéro 78.7** : correspond au run de **400** époques. Le papier dit 800, et le run de 800 époques donne 80.6.
- **OpenGF depuis zéro 87.1** : c'est le mIoU de test. Le reste de la colonne OpenGF est l'IoU Ground, qui vaut 87.0 pour ce run.
- **LitePT H3D 75.0** : obtenu avec un lr forcé à 5e-3. La grille seule donne 72.1, et le même config relancé le 08-27 donnait 73.3 → résultat peu stable.
- **ECLAIR Sonata** : toujours TODO dans le Tab. 4 (le pré-entraînement a divergé en NaN).
