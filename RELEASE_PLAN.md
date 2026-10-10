# Plan de release MALiBU3D

Suivi du nettoyage de la branche `malibu3d` avant la création du repo public.
**Fichier de travail : à supprimer avant le squash vers le repo public.**

- Snapshot de la soumission : tag `malibu3d-submission` (commit 430f1da, = `flair3d` au 2026-10-10).
- Release publique : nouveau repo propre (un seul commit), créé à la fin à partir de `malibu3d`.
- Après chaque phase : relancer le harnais et le comparer à la référence (voir Phase 0), puis faire un commit.

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

- [ ] `manifests/val_dev_subset_2000.csv` (sous-ensemble stratifié de validation lu par les configs par défaut via `val_stratified_subset_manifest`). Aujourd'hui : `data/flair3d_plus/manifests/val_dev_subset_2000.csv`, ignoré par git. Une fois publié, faire pointer les configs et la doc vers le fichier HF.
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

- [ ] (fait avec le renommage, Phase 4) Ranger les configs en `configs/malibu3d/{segment,forest,natural_habitat,network,elevation,multitask,probe,pretrain}` et `configs/{dales,h3d,eclair,opengf,pureforest}`
- [x] Remplacer les chemins `weight=` (`/lustre/...`, numéros de job, `ckpt/malibu3d/*`) par la convention de `README_checkpoints.md` : `ckpt/<nom>/model.pth` (873542 → `litept-b`, 1095469 → `ptv3`, 1052217 → `spunet`, 1159986 → `kpconvx`, 862680 → `sonata-pretrained`). Backbones d'ablation non publiés (noGNL, monoLC, realGN, preECLAIR) : les 3 configs OpenGF qui ne différaient que par `weight` sont supprimées → surcharge `--options weight=...` à documenter (Phase 7). Le checkpoint Sonata indoor officiel garde `ckpt/sonata/pretrain-sonata-v1m1-0-base.pth` (à documenter)
- [x] `network_graphs_root` (APLS) pointe vers `/lustre/...` : les graphes GT sont dans le dataset HF, à la racine de chaque zip de ROI (`data/{split}/{dept}_LIDARHD/{roi}/*_ROADS_graph.gpkg`), alors que `network_label_utils.py` attend un dossier plat `{root}/{dept}_{zone}_{NETWORK}_graph.gpkg` → `network_graphs_root=None` lit chaque graphe dans le dossier de sa ROI (nouveau défaut des configs, test unitaire `tests/test_network_graph_paths.py`) ; APLS vérifié de bout en bout sur hecate avec un dossier explicite, et désormais exercé par le harnais
- [x] Valeurs par défaut personnelles retirées du code (`preprocess_flair3d_v2.py`) ; les docstrings et le lien wandb sont traités en Phase 6, l'audit global en Phase 8. ~~Retirer les chemins `/lustre` et `/data/geist` des docstrings/defaults (`preprocess_flair3d_v2.py`, `rasterize_forest.py`, `rasterize_network.py`, `unzip_malibu3d.py`), le lien wandb de `multi-litept-b-v1m0-flair3d.py`~~
- [x] Lanceurs Slurm : tous supprimés en Phase 2 (spécifiques à Jean Zay) ; le README expliquera l'usage générique
- [ ] Harnais (avec une table de correspondance des chemins) + commit

## Phase 4 — Renommage Flair3D → MALiBU3D

- [ ] Générer la liste exhaustive des identifiants, chemins et chaînes qui contiennent `flair3d` → table de correspondance relue par Louis
- [ ] Appliquer la table par script et faire les `git mv` (un seul commit dédié)
- [ ] Régénérer les configs livrées avec les checkpoints HF
- [ ] Vérifier les clés de `meta.json` lues par le dataset ; symlink `data/malibu3d → flair3d_plus` pour les données existantes
- [ ] Vérifier que `grep -ri flair3d` ne renvoie plus rien et que FLAIR-HUB est intact
- [ ] Harnais (table `strings`) + commit

## Phase 5 — Code obsolète dans les fichiers conservés

- [ ] Définitions de labels legacy (v18/v19), compatibilité `scene_split_manifest.csv`, options et branches mortes, trainers inutilisés
- [ ] Repérer le code mort avec la couverture du harnais (`coverage`) et `vulture`
- [ ] Décider quels correctifs de `docs/bug_review` (01 à 08) entrent dans la release
- [ ] Harnais + commit (une suppression logique par commit si c'est risqué)

## Phase 6 — Commentaires et docstrings

- [ ] Passe module par module : anglais, docstrings courtes, plus de références internes (semaines, jobs, dates)
- [ ] Vérifier que l'AST est inchangé (seuls les commentaires et docstrings ont bougé) ; formatter avec `black`
- [ ] Harnais + commit

## Phase 7 — Documentation de release

- [ ] Réécrire le README : installation, téléchargement depuis HF, preprocessing, entraînement, évaluation du benchmark, checkpoints, résultats, citation
- [ ] Ajouter `CITATION.cff`, mettre à jour `LICENSE` (Pointcept + MALiBU3D), les crédits (Pointcept, PTv3, Sonata, LitePT, KPConvX)
- [ ] Nettoyer `environment.yml`

## Phase 8 — Hygiène et validation finale

- [ ] Rechercher les chemins en dur, `usi32yh`, les comptes JZ, les emails, les tokens wandb/HF, les gros fichiers
- [ ] Installation propre dans un env neuf, puis lancer le harnais
- [ ] Reproduire un ou deux chiffres du papier sur Jean Zay à partir des checkpoints HF
- [ ] Sur Jean Zay, lancer l'étage forward du harnais sur les probes DALES et ECLAIR, au tag puis sur la version finale, et comparer
- [ ] Créer le repo public (squash), supprimer ce fichier
