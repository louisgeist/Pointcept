# decision.md

Log of ambiguous points Claude had to resolve on its own judgment while working in this repo —
not a changelog of everything implemented. Newest entries at the top. Terse bullets: what was
left unclear, what was chosen (and why, only if not obvious).

---

## 2026-09-18 — OpenGF dataset integration: outliers kept as a 3rd on-disk label, not merged/dropped at preprocessing time

**Decision:** `preprocess_opengf.py` writes `segment.npy` with 3 values (`0=Ground, 1=Non-ground,
2=Outlier`), remapped straight from the raw LAS `classification` field (`{2,1,0}` respectively).
Two new generic transforms were added to `pointcept/datasets/transform.py` — `RemapSegment(mapping)`
(in-place id->id merge) and `DropSegmentClass(labels)` (physical point removal via `index_operator`,
run before `GridSample` so it actually changes the geometric neighborhood fed to the network, not
just the loss mask) — so a GridProbe config decides on the fly whether outliers are merged into
Non-ground, physically dropped, or left as `ignore_index`. The shipped
`configs/opengf/litept-b-v1m0-opengf-lin-grid-enc.py` uses `RemapSegment({2: 1})` in train/val/test.

**Why:** Initially assumed (wrongly) OpenGF had no val split and that raw label `0` should be
`ignore_index` — both corrected by the user. Checked the actual local download
(`/data/geist/datasets/OpenGF`, 163 LAZ files) and the CVPRW 2021 paper (`arxiv.org/pdf/2101.09641`,
fetched and OCR'd via `pdftotext` since WebFetch can't read PDF binaries) directly: OpenGF *does*
ship its own `Validation/` (9 scenes, one per training terrain, distinct from `Test/`), and raw
label `0` is "outliers" (low/high LiDAR noise), with the paper's official baseline convention being
"merge outliers into NG" for training/"Test II (w outliers)", vs. a separate "Test II (w/o outliers)"
protocol that **physically deletes** outlier points before inference (Sec 4.4-4.5) — not an
evaluation-time mask, since removing points changes the remaining points' neighborhood too. The user
explicitly asked to keep outliers as their own label and decide per-config rather than bake one
choice into preprocessing, since GridProbe configs may want any of the three treatments later.

**How to apply:** Confirmed by full-dataset smoke test (`preprocess_opengf.py` run end-to-end on the
local 163-file download, 217 output scene folders, point counts and per-class counts cross-checked
against raw `laspy` reads — no data loss). One preprocessing bug caught during that smoke test and
fixed: `split_scene_xy_by_chunk_size` labels its first grid cell `"0-0"` regardless of whether the
scene was actually split — the original `tiled_scene_id = scene_id if suffix == "0-0" else ...`
silently collided that first real subtile onto the unsuffixed scene folder name whenever a Test tile
split into >1 piece (only Test/T1/T2/T3 are large enough to actually split; Training/Validation's
uniform 500x500 m tiles are a no-op). Fixed to key off `len(sub_scenes) == 1` instead of the label
string — this class of bug (a coordinate label doubling as a sentinel) would resurface for any other
dataset's preprocessing script reusing `split_scene_xy_by_chunk_size` with the same shortcut; check
`len(sub_scenes)` there too, not the returned suffix. Only one GridProbe config was scaffolded
(LitePT-B encoder-multiscale, checkpoint job 873542 — same frozen backbone as the DALES/H3D/ECLAIR
siblings) per the user's explicit choice to validate the pipeline before generating the full
per-backbone matrix (SpUNet/PTv3/KPConvX/Sonata variants) other datasets have; `strength_feat_scale =
1/60000` (DALES' convention) is an unverified assumption for OpenGF (only checked that the observed
intensity range is the same order of magnitude, ~60000) — revisit if probe LRs look off.

---

## 2026-09-16 — Sonata multi-task FT config (`configs/experiment/w113/3/sonata_ft/`)

- "Look at the litept-b multitask config" — read as copy the multi-task *wiring*
  (task/criteria/hooks/Collect), not the backbone: the Sonata checkpoint only matches a
  `PT-v3m2` backbone built with the exact pretrain encoder dims, not `LitePT-v1`.
- Encoder-only vs. decoder-plus-FT was left unspecified — first pass reused a prior
  encoder-only decision from a stashed 2026-08-19 attempt; user corrected to a fresh,
  randomly-initialized decoder (dims from the upstream `configs/sonata/semseg-sonata-v1m1-0c-scannet-ft.py`
  reference). Should have asked instead of assuming the old decision still applied.
- Checkpoint pinned to job 862680/`epoch_120.pth` (not final `epoch_150.pth`) — matches the
  `W_SONATA` convention already used elsewhere (README_grid_then_seed.md), not stated by the
  user.
- `batch_size=12` carried over from the LitePT baselines, unrecalibrated for this backbone —
  flagging, not re-tuned.

---

## 2026-09-14 — Removed dead per-point offset computation in `Collect.__call__` (repo-wide, not PureForest-specific)

**Decision:** Deleted `pointcept/datasets/transform.py:116-118`
(`data_dict["offset"] = torch.cumsum(torch.tensor([data.shape[0] for data in data_dict["coord"]]), dim=0)`)
from `Collect.__call__`, no replacement needed.

**Why:** Found while profiling why PureForest pooled-embedding extraction was projected at
~17h for one backbone's train split (post coord-denormalization fix, tiles now ~100k pts
instead of ~3.5k). Isolated the cost: this one line was ~150-250ms per single-scene sample
(iterating a `(N,3)` tensor row-by-row in Python — `.shape[0]` per row is nonsense, always `3`)
— >90% of the whole `Collect` call, and >30% of total per-tile wall time in the (single-threaded,
no-DataLoader-overlap) extraction script. Proved it's dead code, not just slow: the result is
written to `data_dict` (the local input dict), never to `data` (the dict `Collect` actually
returns) — the real per-scene offset is computed correctly two lines later
(`data["offset"] = torch.tensor([data_dict[value].shape[0]])` via `offset_keys_dict`, default
`{"offset": "coord"}`) and unconditionally overwrites whatever the deleted line produced. That
second line also only works if `data_dict["coord"]` is a single array (`.shape` access) — so if
`data_dict["coord"]` were ever genuinely a Python list of multiple per-fragment arrays (the
shape the deleted line's per-element loop implies it was written for), the line right after it
would already crash with `AttributeError`. So the deleted line was unconditionally dead for
every call site in the repo, not just this one. Verified before/after: `data["offset"]`
identical, `post_transform` (GridSample→Collect) on a 6-tile PureForest batch went
2.33s→0.077s (~30x) on the Collect stage alone.

**How to apply:** This affects every training/val/test call through `Collect` (basically every
dataset in the repo, PureForest included) — expect a small-to-moderate free CPU-side speedup
project-wide, proportionally largest on dense point clouds (fine `grid_size`, high native
density) processed without `DataLoader` worker overlap, i.e. exactly this kind of offline
single-threaded extraction/probing script; negligible where GPU compute or worker-parallel
loading already dominates wall time (most normal multi-GPU training). No config or checkpoint
compatibility impact — `offset` output is byte-identical, just computed once instead of twice
(once uselessly). Not yet fixed: the PureForest OOM hit on `cuda:0` at `--batch-size 6` (GPU0
shares ~11GB free with another user's job on hecate right now) — that needs a smaller
`--batch-size` and/or `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, unrelated to this fix.

---

## 2026-09-14 — PureForest `coord.npy` denormalization moved into the dataset loader, not preprocessing

**Decision:** Fixed the "`grid_size=0.1` silently means 2.5m voxels on PureForest" bug (on-disk
`coord.npy` is stored ~[-1,1]-normalized — mean-centered XY, min-shifted Z, all divided by
`COORD_SCALE_M=25.0` — per the PureForest paper's own baseline preprocessing) by denormalizing
(`* COORD_SCALE_M`) inside `PureForestDataset.get_data()` in `pointcept/datasets/pureforest.py`,
rather than re-running `preprocess_pureforest.py` to rewrite `coord.npy` in real meters on disk.

**Why:** User picked this option explicitly (denormalize-at-load vs. rescale every config's
`grid_size`/`coord_feat_scale` by ÷25). It's a one-line fix in one place, requires no
reprocessing of the ~135k already-preprocessed LAZ tiles, and immediately makes every existing
config (`grid_size=0.1`, `coord_feat_scale=0.01`, `point_max=...`) mean what it already assumes
elsewhere in the repo (real meters), with zero config edits needed.

**How to apply / caveat:** This is a pragmatic fix, not the clean one — `coord.npy` on disk still
stores the paper's normalized units, and every consumer of that file (not just
`PureForestDataset`) needs to know to multiply by `COORD_SCALE_M` before treating `coord` as
metric. `scripts/extract_pureforest_pooled_embeddings.py` and any other script reading
`coord.npy` directly (bypassing `PureForestDataset`) is still affected and was **not** patched
here — audit those before trusting their output. **For an eventual official repo release, this
should be done properly in `preprocess_pureforest.py` instead**: either store `coord.npy` in real
meters from the start (dropping `normalize_tile_coord`'s `/= COORD_SCALE_M`, since Pointcept's
own transform pipeline — `CenterShift`, `Z_MinShift` — already reproduces the paper's centering
behavior on real coordinates, matching every other dataset in this repo), or bake the
denormalization into preprocessing output so no downstream consumer has to remember the ÷25
convention. The loader-side fix here is a stopgap so existing configs work correctly today; it
should not be treated as the final design.

---

## 2026-09-14 — PureForest test-set leakage vs Flair3D+ trainval: exclude at forest level, not tile level

**Decision:** Built the exclusion list at the `bdforetv2_id` (PureForest's own "forest"/annotation-polygon
unit, 449 total) granularity, not at the individual-50m-tile granularity. A geometric intersection
(EPSG:2154) found 231 PureForest test tiles directly overlapping Flair3D+ trainval tiles, but those
231 tiles belong to only 2 forests (`bdforetv2_id` 275 and 442); excluding those two forests wholesale
removes 2170 / 52935 test tiles (4.10%) instead of just 231 (0.44%).

**Why:** PureForest's own train/val/test split is done at the forest-polygon level specifically "to
account for spatial autocorrelation" (per the dataset paper) — tiles from the same forest are not
independent. A tile-level-only exclusion would leave ~1900 tiles from the same 2 leaking forests in
the test set, which are still spatially/texturally correlated with the tiles that did leak into
Flair3D+ training, even though their footprint doesn't literally intersect. Matching PureForest's own
split logic was the user's explicit ask ("pour être en accord avec la logique de base de pureforest").

**How to apply:** The static list lives at
`pointcept/datasets/preprocessing/pureforest/flair3d_leakage_excluded_test_tiles.txt` (tracked in git —
`data/` is gitignored wholesale, so it couldn't live next to the preprocessed tiles). It's loaded by
`PureForestDataset` via a new `exclude_flair3d_leakage_tiles` constructor param, default `False` (opt-in,
so any other/future use of the dataset class doesn't silently change). Set to `True` in the `test=dict(...)`
block of all 10 non-toy configs under `configs/pureforest/` (5 GridProbe/lin-probe + 5 from-scratch) —
these are the canonical base configs referenced by the "copy a reference config" convention
(`experiment-config-generation.mdc`). Did **not** retroactively patch already-existing standalone configs
under `configs/experiment/**/pureforest*` or `pf_scratch/**` (w112/w113) — those are frozen per-day
snapshots by convention, and several may already have completed runs; re-copy from the updated base configs
for any new experiment going forward. If this list needs to change later (e.g. after re-checking val-split
leakage — 14/13523 val tiles also intersect Flair3D+ trainval and were left untouched here), regenerate it
the same way (see the file's own header comment for the exact join) rather than hand-editing it.

---

## 2026-09-14 — Recomputed PureForest sklearn-probe test metrics on the leakage-filtered test set

**Decision:** Reused the already-saved `best_test_predictions.npz` (`names`/`category`/`pred` on the full
old test set) for each of the 6 probes in `stats/pureforest/sklearn_probe/summary.html`, filtered out the
2170 excluded names, and recomputed `mIoU`/`allAcc`/`mAcc`/`macro_f1`/per-class via the same
`intersection_and_union`-based formula the sweep script uses — instead of reloading `best_so_far.pkl` and
re-running inference from embeddings. In each run's `metrics.json`, moved the original (leaked) `best.test`
block to `best.test_pre_leakage_fix` rather than deleting it, and backed up the whole pre-edit file to
`metrics.json.pre_leakage_fix.bak`. Regenerated `summary.html` by calling `summarize_sklearn_probes.py`'s
`_load_runs`/`render_html` directly with an explicit 6-run allowlist, not its default glob-all-subdirs
`main()`.

**Why:** Predictions were already on disk and match 1:1 with the exclusion list (verified: exactly 2170
names removed per run) — re-running GPU inference through the saved linear weights would reproduce the
identical predictions for a non-leaked point (the model itself didn't change) at the cost of a GPU pass,
so it was redundant. Kept the old numbers under a renamed key instead of overwriting them outright, in
case the paper draft or an earlier note (`stats/pureforest/pureforest_flair3d_leakage.md`) needs to cite
the "with leakage" figure for contrast. Used an explicit allowlist for the HTML regen because
`stats/pureforest/sklearn_probe/` now also holds 5 unrelated in-progress MLP scale-probe runs
(`kpconvx_malibu3d_ms_mlp*`, a separate exploration) that would otherwise silently appear in a table whose
whole point was a like-for-like before/after on the 6 runs the user was already looking at.

**How to apply:** All 6 mIoU deltas are small (-0.07 to -0.18 pts) and the ranking is unchanged, but
`allAcc` drops more (-1.2 to -1.5 pts) because the leaked `deciduous_oak` tiles were being predicted
correctly at an above-average rate — expected if the backbone partly memorized that ground during Flair3D+
training. If new probe runs get added to this summary later, regenerate the same way (recompute from
`best_test_predictions.npz` + the exclusion list, not from scratch) rather than re-running the full
train/val sweep, since the sweep's agg/C selection depends only on train/val (untouched by this leakage).

---

## 2026-09-14 — Started this file at `decision.md` (repo root, not gitignored by default)

**Decision:** Created as a plain root-level markdown file, tracked like any other repo file
(not under `scripts/` or a dotfile). Named `decision.md` (singular) per the user's exact
wording, even though the convention elsewhere in the repo (e.g. `stats/`, `README_*.md`) might
suggest a plural or prefixed name.

**Why:** User asked explicitly for a `decision.md` file to log coding decisions going forward,
in French, as a standing process for this session and future ones.

**How to apply:** From now on, whenever a coding task involves a non-obvious judgment call
(ambiguous requirement resolved one way, a convention invented because none existed, a
tradeoff picked without being asked), state it plainly in the chat response *and* append an
entry here in the same turn. Skip purely mechanical choices (variable names, which existing
helper to call) — this is for calls a reasonable person could disagree with.

---

## 2026-09-14 — CLAUDE.md refresh scope

**Decision:** Updated CLAUDE.md to cover everything added since its last edit (2026-08-10,
~30 commits: PureForest, DALES/H3D/ECLAIR cross-domain GridProbe benchmarking,
`tools/grid_then_seeds.py`, real GradNorm ablation, the network/APLS pipeline, two new
`.cursor/rules/*.mdc` files, and pointers to the 4 new maintainer READMEs). Left out:
one-off experiment details (specific job IDs, ROI names, exact metric values) — those stay
in the READMEs/configs themselves; CLAUDE.md only points to them.

**Why:** CLAUDE.md is meant to be a stable orientation doc for future Claude Code sessions,
not a running log — duplicating volatile specifics here would just make it another thing to
keep in sync (and it would go stale the same way it just had).

**How to apply:** When something new lands that's *structural* (a new dataset integration, a
new mutually-exclusive config mode, a new cross-repo pipeline, a new cursor rule), add a short
pointer here. When it's a one-off result or a specific run's parameters, put it in the
relevant README or config comment instead, not here.
