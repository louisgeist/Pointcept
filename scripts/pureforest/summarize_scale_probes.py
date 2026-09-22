#!/usr/bin/env python3
"""Build a standalone HTML report comparing PureForest linear probes across
encoder-scale slices (see ``run_sklearn_scale_slice_probes_gpu.sh``).

For each backbone tag under ``--probe-root`` (default
``stats/pureforest/sklearn_probe_scales``), reads every ``scale_*/metrics.json``
and renders one section: which subset of concatenated encoder stages (a
"scale slice" of the pooled multi-scale embedding) gives the best linear-probe
accuracy, and how the winning (agg, C) config compares across slices.

Example::

    python scripts/pureforest/summarize_scale_probes.py
    python scripts/pureforest/summarize_scale_probes.py \\
      --probe-root stats/pureforest/sklearn_probe_scales \\
      --out stats/pureforest/sklearn_probe_scales/summary.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


def _pct(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "—"
    return f"{100.0 * x:.{digits}f}"


def _escape(s: object) -> str:
    return html.escape(str(s), quote=True)


def _scale_sort_key(slug: str) -> tuple:
    """Order: p0-1..p0-(L-1) (growing prefix), then full (= p0-L, all stages),
    then s1-end..s(L-1)-end (shrinking suffix) — full sits right where it
    belongs, as the last and largest prefix."""
    if slug == "full":
        return (1, 1_000_000)
    m = re.fullmatch(r"p0-(\d+)", slug)
    if m:
        return (1, int(m.group(1)))
    m = re.fullmatch(r"s(\d+)-end", slug)
    if m:
        return (2, int(m.group(1)))
    return (3, slug)


def _load_tag(tag_dir: Path) -> dict | None:
    scale_dirs = sorted(tag_dir.glob("scale_*"))
    rows = []
    channel_blocks: list[int] | None = None
    for scale_dir in scale_dirs:
        metrics_path = scale_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        data = json.loads(metrics_path.read_text())
        best = data.get("best") or {}
        if "test" not in best or "val" not in best:
            continue
        channel_blocks = channel_blocks or list(data.get("channel_blocks") or [])
        block_indices = data.get("block_indices")
        n_blocks = len(channel_blocks)
        included = (
            list(range(n_blocks)) if block_indices is None else list(block_indices)
        )
        slug = scale_dir.name.removeprefix("scale_")
        rows.append(
            {
                "slug": slug,
                "scale_slice": data.get("scale_slice"),
                "included": included,
                "agg": best.get("agg"),
                "C": best.get("C"),
                "feat_dim": best.get("feat_dim"),
                "train": best.get("train") or {},
                "val": best["val"],
                "test": best["test"],
            }
        )
    if not rows or channel_blocks is None:
        return None
    rows.sort(key=lambda r: _scale_sort_key(r["slug"]))
    return {"tag": tag_dir.name, "channel_blocks": channel_blocks, "rows": rows}


def _stage_strip(channel_blocks: list[int], included: list[int]) -> str:
    """Compact per-row visualization of which encoder stages are kept."""
    weights = [max(c, 1) ** 0.5 for c in channel_blocks]
    total = sum(weights)
    segs = []
    for i, (w, c) in enumerate(zip(weights, channel_blocks)):
        pct = 100.0 * w / total
        state = "on" if i in included else "off"
        segs.append(
            f"<span class='seg {state}' style='flex-basis:{pct:.2f}%' "
            f"title='stage {i} · {c} ch'>{c}</span>"
        )
    return f"<div class='strip'>{''.join(segs)}</div>"


def render_html(tags: list[dict], skipped: list[str], out_path: Path) -> str:
    sections = []
    for t in tags:
        rows = t["rows"]
        channel_blocks = t["channel_blocks"]
        best_idx = max(range(len(rows)), key=lambda i: rows[i]["val"]["mIoU"])
        legend = "".join(
            f"<span class='seg on' style='flex-basis:{100.0 * (max(c,1)**0.5) / sum(max(x,1)**0.5 for x in channel_blocks):.2f}%'>"
            f"s{i}·{c}</span>"
            for i, c in enumerate(channel_blocks)
        )

        body_rows = []
        for i, r in enumerate(rows):
            best_cls = " class='best'" if i == best_idx else ""
            strip = _stage_strip(channel_blocks, r["included"])
            train_miou = r["train"].get("mIoU")
            body_rows.append(
                f"<tr{best_cls}>"
                f"<td><code>{_escape(r['scale_slice'])}</code></td>"
                f"<td>{strip}</td>"
                f"<td class='num'>{r['feat_dim']}</td>"
                f"<td class='num'>{_escape(r['agg'])}</td>"
                f"<td class='num'>{_escape(r['C'])}</td>"
                f"<td class='num'>{_pct(train_miou)}</td>"
                f"<td class='num'>{_pct(r['val']['mIoU'])}</td>"
                f"<td class='num hl'>{_pct(r['test']['mIoU'])}</td>"
                f"<td class='num'>{_pct(r['test']['allAcc'])}</td>"
                "</tr>"
            )

        best = rows[best_idx]
        worst_idx = min(range(len(rows)), key=lambda i: rows[i]["val"]["mIoU"])
        worst = rows[worst_idx]
        spread = best["val"]["mIoU"] - worst["val"]["mIoU"]

        sections.append(
            f"""
      <section class="tag">
        <h2><code>{_escape(t['tag'])}</code></h2>
        <p class="lede">
          Encodeur à {len(channel_blocks)} stages, largeurs
          <code>{'·'.join(str(c) for c in channel_blocks)}</code> canaux
          (fin&nbsp;→&nbsp;grossier). Meilleure tranche&nbsp;:
          <strong>{_escape(best['scale_slice'])}</strong>
          (stages {'+'.join(str(i) for i in best['included'])},
          {best['feat_dim']}&nbsp;dim) — val mIoU <strong>{_pct(best['val']['mIoU'])}%</strong>,
          test mIoU {_pct(best['test']['mIoU'])}%. Écart val mIoU sur
          l'ensemble des tranches testées&nbsp;: {_pct(spread)}&nbsp;pts
          (min {_escape(worst['scale_slice'])} = {_pct(worst['val']['mIoU'])}%).
        </p>
        <div class="legend">
          <span class="legend-label">stages&nbsp;→</span>
          <div class="strip legend-strip">{legend}</div>
        </div>
        <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>tranche</th>
              <th>stages retenus</th>
              <th class="num">dim</th>
              <th class="num">agg</th>
              <th class="num">C</th>
              <th class="num">train mIoU</th>
              <th class="num">val mIoU</th>
              <th class="num hl">test mIoU</th>
              <th class="num">test OA</th>
            </tr>
          </thead>
          <tbody>
            {''.join(body_rows)}
          </tbody>
        </table>
        </div>
        <p class="muted">
          Sélection de (agg, C) sur val mIoU pour chaque tranche ; ligne en
          surbrillance = meilleure tranche pour ce backbone. Valeurs en %.
        </p>
      </section>
"""
        )

    skipped_note = ""
    if skipped:
        names = ", ".join(f"<code>{_escape(s)}</code>" for s in skipped)
        skipped_note = (
            f"<p class='muted skipped'>Sweep pas encore terminé (aucune tranche "
            f"complète) pour&nbsp;: {names}.</p>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PureForest — probes par échelle</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,440..600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  :root {{
    color-scheme: light dark;
    --paper: #f6f7f1;
    --paper-raised: #ffffff;
    --ink: #17211a;
    --muted: #5b6a5a;
    --line: #dadfd0;
    --accent: #2f6b4f;
    --accent-soft: #dcebe1;
    --accent-ink: #1c4432;
    --bad: #a24b3c;
    --best-bg: #e8f2ea;
    --seg-off: #e4e7dd;
    --seg-off-ink: #8b968a;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --paper: #10140f;
      --paper-raised: #161c15;
      --ink: #e7ebe0;
      --muted: #8fa08c;
      --line: #2b332a;
      --accent: #7fd1a4;
      --accent-soft: #1d2e23;
      --accent-ink: #bfe9d1;
      --bad: #e08d7c;
      --best-bg: #17281d;
      --seg-off: #232922;
      --seg-off-ink: #6d7a6a;
    }}
  }}
  :root[data-theme="dark"] {{
    --paper: #10140f;
    --paper-raised: #161c15;
    --ink: #e7ebe0;
    --muted: #8fa08c;
    --line: #2b332a;
    --accent: #7fd1a4;
    --accent-soft: #1d2e23;
    --accent-ink: #bfe9d1;
    --bad: #e08d7c;
    --best-bg: #17281d;
    --seg-off: #232922;
    --seg-off-ink: #6d7a6a;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 2.5rem 1.25rem 4rem;
    background: var(--paper);
    color: var(--ink);
    font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
    line-height: 1.5;
  }}
  main {{ max-width: 980px; margin: 0 auto; }}
  h1, h2 {{
    font-family: "Fraunces", Georgia, serif;
    font-weight: 560;
    text-wrap: balance;
    margin: 0;
  }}
  h1 {{ font-size: 2rem; letter-spacing: -0.01em; }}
  h2 {{ font-size: 1.3rem; color: var(--accent-ink); }}
  .kicker {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 0.5rem;
  }}
  .standfirst {{
    max-width: 62ch;
    color: var(--muted);
    font-size: 0.98rem;
    margin: 0.9rem 0 0;
  }}
  header {{
    border-bottom: 1px solid var(--line);
    padding-bottom: 1.6rem;
    margin-bottom: 2.2rem;
  }}
  .method {{
    background: var(--paper-raised);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 1.4rem 0 0;
    font-size: 0.88rem;
    color: var(--muted);
  }}
  .method strong {{ color: var(--ink); }}
  section.tag {{
    margin-top: 2.6rem;
    padding-top: 1.8rem;
    border-top: 1px solid var(--line);
  }}
  section.tag:first-of-type {{ border-top: none; padding-top: 0; }}
  .lede {{
    max-width: 68ch;
    font-size: 0.95rem;
    margin: 0.6rem 0 1.1rem;
  }}
  .lede code, table code {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.88em;
    background: var(--accent-soft);
    color: var(--accent-ink);
    padding: 0.08em 0.34em;
    border-radius: 4px;
  }}
  .legend {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
  }}
  .legend-label {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.72rem;
    color: var(--muted);
    white-space: nowrap;
  }}
  .legend-strip {{ flex: 1; height: 1.6rem; }}
  .legend-strip .seg {{ font-size: 0.68rem; }}
  .strip {{
    display: flex;
    gap: 2px;
    height: 1.35rem;
    min-width: 220px;
  }}
  .strip .seg {{
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 3px;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.62rem;
    font-variant-numeric: tabular-nums;
    color: var(--paper-raised);
    background: var(--accent);
    white-space: nowrap;
    overflow: hidden;
  }}
  .strip .seg.off {{
    background: var(--seg-off);
    color: var(--seg-off-ink);
  }}
  .scroll {{ overflow-x: auto; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.86rem;
    background: var(--paper-raised);
  }}
  th, td {{
    border-bottom: 1px solid var(--line);
    padding: 0.5rem 0.6rem;
    text-align: left;
    white-space: nowrap;
  }}
  th {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.68rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 500;
  }}
  td.num, th.num {{
    text-align: right;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
  }}
  tr.best {{ background: var(--best-bg); }}
  tr.best td:first-child code {{
    background: var(--accent);
    color: var(--paper-raised);
  }}
  th.hl, td.hl {{ background: var(--accent-soft); }}
  th.hl {{ color: var(--accent-ink); }}
  td.hl {{ color: var(--accent-ink); font-weight: 600; }}
  .muted {{ color: var(--muted); font-size: 0.82rem; }}
  .skipped {{ margin-top: 2.4rem; }}
  footer {{
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 0.8rem;
  }}
  footer code {{ font-family: "IBM Plex Mono", ui-monospace, monospace; }}
  @media (max-width: 640px) {{
    h1 {{ font-size: 1.6rem; }}
    .strip {{ min-width: 160px; }}
  }}
</style>
</head>
<body>
<main>
  <header>
    <p class="kicker">Flair3D+ &middot; linear probing &middot; PureForest (13 essences)</p>
    <h1>Quelle échelle de l'encodeur porte le signal&nbsp;?</h1>
    <p class="standfirst">
      Chaque backbone produit un embedding multi-échelle par pooling (mean/max)
      des sorties de chacun de ses stages, du plus fin (résolution la plus
      dense) au plus grossier. On tranche cet embedding concaténé à
      différentes profondeurs et on réentraîne une tête linéaire à chaque
      fois : est-ce que les stages profonds aident, ou bruitent le signal
      utile pour distinguer les essences&nbsp;?
    </p>
    <div class="method">
      <strong>Méthode.</strong> Pour chaque tranche, grille (agg&nbsp;∈
      {{mean, max, concat, sum}} × C&nbsp;∈ {{1e-3…1e3}}), config retenue sur
      <strong>val mIoU</strong>. Les colonnes train/test sont lues pour cette
      config gagnante, pas re-optimisées.
      <code>scripts/pureforest/run_sklearn_scale_slice_probes_gpu.sh</code> →
      <code>scripts/probe_pureforest_sklearn.py --scale-slice …</code>
    </div>
  </header>

  {''.join(sections)}
  {skipped_note}

  <footer>
    Généré par <code>scripts/pureforest/summarize_scale_probes.py</code>
    → <code>{_escape(out_path)}</code>.
  </footer>
</main>
</body>
</html>
"""
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=Path("stats/pureforest/sklearn_probe_scales"),
        help="Directory containing <tag>/scale_*/metrics.json folders",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path (default: <probe-root>/summary.html)",
    )
    args = parser.parse_args()
    probe_root = args.probe_root
    out_path = args.out or (probe_root / "summary.html")

    tags = []
    skipped = []
    for tag_dir in sorted(p for p in probe_root.iterdir() if p.is_dir()):
        loaded = _load_tag(tag_dir)
        if loaded is None:
            skipped.append(tag_dir.name)
        else:
            tags.append(loaded)

    if not tags:
        raise SystemExit(f"No complete scale sweep found under {probe_root}")

    html_doc = render_html(tags, skipped, out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {out_path} ({len(tags)} tag(s), {len(skipped)} skipped)")


if __name__ == "__main__":
    main()
