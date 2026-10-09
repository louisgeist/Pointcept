# 03 — `GridProbeTrainer` : un seul `GradScaler` pour toutes les probes

**Sévérité : moyenne (conditionnelle)** · **Statut : corrigé**

## Correctif

`GridProbeTrainer` construit désormais un `GradScaler` **par probe**
(`self.scaler: dict[str, GradScaler]`), scale chaque perte individuellement,
fait un seul `backward` sur la somme, puis `unscale_/step/update` + décision
de scheduler **par probe**. `CheckpointSaver`/`CheckpointLoader` sérialisent
le dict ; un ancien ckpt mono-scaler est rebroadcasté sur chaque probe au
resume. Test : `TestGridProbePerProbeGradScaler` dans
`tests/test_bug_review_2026_09_27.py`.

## Constat (pré-correctif)

`pointcept/engines/train.py`, `GridProbeTrainer.run_step` (branche AMP) : un
backward sur la somme des pertes, puis `unscale_/step` par optimizer et **un
seul** `scaler.update()` ; les schedulers de *toutes* les probes ne sont stepés
que si l'échelle n'a pas baissé. 185 configs GridProbe ont `enable_amp=True` en
`float16` (défaut) avec des balayages de lr jusqu'à 5e-1.

Si une seule probe produit des logits non finis (débordement fp16 > 65504 →
perte NaN), à chaque step :
- son optimizer est sauté (ses poids restent figés → elle reste non finie) ;
- `scaler.update()` divise l'échelle par 2 → elle s'effondre vers 0 ;
- `step_schedulers` est faux → **aucun scheduler n'avance** ;
- avec une échelle ≈ 0, les gradients fp16 des probes saines deviennent nuls.

## Reproduction (GPU, logique identique à `run_step`)

Deux têtes linéaires sur des features fixes, AdamW + OneCycleLR, `grad_clip=3`,
une tête « divergée » (poids ×1e6) :

```
bad_probe=False  final_scale=6.554e+04  healthy_train_acc=0.990
bad_probe=True   final_scale=0.000e+00  healthy_train_acc=0.128   # hasard = 0.125
```

**Déclenchement réel non démontré** : sur des features synthétiques
(C=1232, 15 classes, 20 % de bruit), même lr=0,5 n'atteint que |logit| ≈ 1,7e3
(une seule baisse d'échelle en 3000 steps). Cela dépend de l'amplitude réelle
des features du backbone (non normalisées en `input_norm=None`).

## Vérifier les runs existants (JZ)

Le scalaire console `loss` est la somme de toutes les probes : une probe non
finie le rend NaN.

```bash
grep -l "loss: nan" logs/slurm/*/train.log   # runs GridProbe concernés
```

## Options de correction (historique)

1. **Un `GradScaler` par probe** (**retenu / implémenté**) : `Σ_p scaler_p.scale(loss_p)`,
   un seul backward (têtes disjointes), puis `unscale_/step/update` et
   décision de scheduler par probe.
2. Garder un scaler, mais détecter une probe non finie et la retirer de
   `active` (probe « morte », loggée comme telle) — non retenu.
3. Passer ces configs en `amp_dtype="bfloat16"` (pas de loss scaling) — non
   retenu (changerait le numérique des runs futurs).
