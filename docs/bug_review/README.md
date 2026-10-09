# Revue de bugs — 2026-09-27

Revue du code propre au fork (diff vs `upstream/main`, ~47k lignes) : boucle
d'entraînement, modèles multi-tâche, datasets/transforms, évaluateurs, GridProbe,
testers, preprocessing, utilitaires de reprise. Chaque suspect a été confronté
aux configs réelles et, quand c'était possible, reproduit sur les données
locales ou par un test.

| # | Problème | Sévérité | Statut |
|---|----------|----------|--------|
| [01](01_flair3d_coord_quantization.md) | Coordonnées Flair3D+ quantifiées à 0,5 m en Y (PLY source en float32 absolu, release HF incluse) | critique | **décision requise** |
| [02](02_opengf_eclair_float32_coords.md) | OpenGF (ECLAIR ?) : cast float32 des coordonnées absolues au preprocessing | haute | corrigé, re-preprocess à lancer |
| [03](03_gridprobe_shared_gradscaler.md) | `GridProbeTrainer` : un seul GradScaler pour N probes | moyenne (conditionnelle) | corrigé |
| [04](04_multitask_tester_classification_metrics.md) | `MultiTaskTester` jette les métriques test des tâches `classification` (+ garde multi-fragments) | moyenne | corrigé |
| [05](05_clstester_test_metrics.md) | `ClsTester` sans `test_metrics` → test `null` dans `grid_search_results.json` (PureForest) | basse–moyenne | corrigé |
| [06](06_cross_entropy_all_ignore.md) | `CrossEntropyLoss` : zéro détaché si tout est ignoré | basse | corrigé |
| [07](07_climatic_domain_dataset_crash.md) | `Flair3DClimaticDomainDataset` plante (kwargs retirés) | basse | corrigé |
| [08](08_minor_and_latent.md) | Points mineurs / latents / à vérifier sur JZ | — | mixte |

## Fichiers modifiés (non commités)

- `pointcept/models/losses/misc.py`, `pointcept/utils/gradient_norm.py` (commentaires) — 06
- `pointcept/engines/test.py` — 04, 05
- `pointcept/engines/train.py`, `pointcept/engines/hooks/misc.py` — 03
  (GradScaler par probe + checkpoint dict / reprise mono-scaler)
- `pointcept/datasets/transform.py` — 08 (`RemapSegment`)
- `pointcept/datasets/flair3d_climatic_domain.py` — 07
- `pointcept/datasets/preprocessing/opengf/preprocess_opengf.py`,
  `pointcept/datasets/preprocessing/eclair/preprocess_eclair.py`,
  `scripts/eclair/visualize_excluded_tile.py` — 02
- `tests/test_bug_review_2026_09_27.py` — tests de non-régression (dont
  `TestGridProbePerProbeGradScaler` pour 03)

Suite complète : `PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py"`
→ 189 tests, seule erreur le test obsolète `test_resolve_config_file` (préexistant, voir 08).
