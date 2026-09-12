#!/usr/bin/env python3
"""Build a standalone HTML summary of PureForest sklearn probe runs.

Reads every ``*/metrics.json`` under a probe root, ranks by test mIoU, and
injects the PureForest paper Lidar baseline (Gaydon et al., arXiv:2404.12064)
for OA / mIoU / per-class IoU only.

Example::

    python scripts/pureforest/summarize_sklearn_probes.py
    python scripts/pureforest/summarize_sklearn_probes.py \\
      --probe-root stats/pureforest/sklearn_probe \\
      --out stats/pureforest/sklearn_probe/summary.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

# Gaydon et al., PureForest (WACV 2025 / arXiv:2404.12064) — Lidar baseline test.
# Global: Table of modality results. Per-class IoU: Figure 8 / accompanying text.
PAPER_BASELINE = {
    "name": "baseline PureForest (papier)",
    "source": "Gaydon et al., PureForest, WACV 2025 (arXiv:2404.12064) — Lidar / RandLA-Net",
    "test_oa": 0.803,
    "test_miou": 0.551,
    "per_class_iou": {
        "deciduous_oak": 0.734,
        "evergreen_oak": 0.594,
        "beech": 0.888,
        "chestnut": 0.565,
        "black_locust": 0.581,
        "maritime_pine": 0.629,
        "scotch_pine": 0.586,
        "black_pine": 0.462,
        "aleppo_pine": 0.393,
        "fir": 0.0,
        "spruce": 0.858,
        "larch": 0.506,
        "douglas": 0.365,
    },
}


def _pct(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "—"
    return f"{100.0 * x:.{digits}f}"


def _load_runs(probe_root: Path) -> list[dict]:
    runs: list[dict] = []
    for metrics_path in sorted(probe_root.glob("*/metrics.json")):
        data = json.loads(metrics_path.read_text())
        best = data.get("best") or {}
        if "test" not in best or "val" not in best:
            continue
        run_name = metrics_path.parent.name
        class_names = list(data.get("class_names") or [])
        test = best["test"]
        val = best["val"]
        train = best.get("train") or {}
        runs.append(
            {
                "name": run_name,
                "embeddings_dir": data.get("embeddings_dir"),
                "backend": data.get("backend"),
                "agg": best.get("agg"),
                "C": best.get("C"),
                "feat_dim": best.get("feat_dim"),
                "class_names": class_names,
                "train": train,
                "val": val,
                "test": test,
                "test_miou": float(test["mIoU"]),
            }
        )
    runs.sort(key=lambda r: -r["test_miou"])
    return runs


def _class_names(runs: list[dict]) -> list[str]:
    for run in runs:
        if run["class_names"]:
            return run["class_names"]
    return list(PAPER_BASELINE["per_class_iou"].keys())


def _heatmap_bg(iou: float | None) -> str:
    """Return a light green→white CSS background from IoU in [0, 1]."""
    if iou is None:
        return "transparent"
    t = max(0.0, min(1.0, float(iou)))
    r = int(220 - 140 * t)
    g = int(120 + 100 * t)
    b = int(120 + 40 * t)
    return f"rgba({r},{g},{b},0.35)"


def _escape(s: object) -> str:
    return html.escape(str(s), quote=True)


def _metric_cell(values: list[float | None], idx: int, higher_better: bool = True) -> str:
    v = values[idx]
    if v is None:
        return "<td>—</td>"
    present = [x for x in values if x is not None]
    is_best = present and (v == (max(present) if higher_better else min(present)))
    cls = ' class="best"' if is_best else ""
    return f"<td{cls}>{_pct(v)}</td>"


def render_html(runs: list[dict], out_path: Path) -> str:
    classes = _class_names(runs)
    paper = PAPER_BASELINE
    paper_ious = [paper["per_class_iou"].get(c) for c in classes]

    test_miou = [r["test_miou"] for r in runs]
    test_oa = [float(r["test"]["allAcc"]) for r in runs]
    test_macc = [float(r["test"]["mAcc"]) for r in runs]
    test_f1 = [float(r["test"]["macro_f1"]) for r in runs]
    val_miou = [float(r["val"]["mIoU"]) for r in runs]
    train_miou = [
        float(r["train"]["mIoU"]) if "mIoU" in r["train"] else None for r in runs
    ]

    max_bar = max([paper["test_miou"], *test_miou]) if runs else paper["test_miou"]
    paper_bar_w = 100.0 * paper["test_miou"] / max_bar if max_bar > 0 else 0.0

    ranking_rows = []
    # Paper baseline first, not ranked among probes.
    ranking_rows.append(
        "<tr class='baseline'>"
        f"<td>ref</td>"
        f"<td><strong>{_escape(paper['name'])}</strong>"
        f"<div class='bar'><span style='width:{paper_bar_w:.1f}%'></span></div>"
        f"<div class='muted'>{_escape(paper['source'])}</div></td>"
        f"<td>—</td>"
        f"<td>—</td>"
        f"<td>—</td>"
        f"<td>—</td>"
        f"<td>—</td>"
        f"<td>{_pct(paper['test_miou'])}</td>"
        f"<td>{_pct(paper['test_oa'])}</td>"
        f"<td>—</td>"
        f"<td>—</td>"
        "</tr>"
    )
    for i, run in enumerate(runs, start=1):
        bar_w = 100.0 * run["test_miou"] / max_bar if max_bar > 0 else 0.0
        ranking_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td><code>{_escape(run['name'])}</code>"
            f"<div class='bar'><span style='width:{bar_w:.1f}%'></span></div></td>"
            f"<td>{_escape(run['agg'])}</td>"
            f"<td>{_escape(run['C'])}</td>"
            f"<td>{_escape(run['feat_dim'])}</td>"
            f"{_metric_cell(train_miou, i - 1)}"
            f"{_metric_cell(val_miou, i - 1)}"
            f"{_metric_cell(test_miou, i - 1)}"
            f"{_metric_cell(test_oa, i - 1)}"
            f"{_metric_cell(test_macc, i - 1)}"
            f"{_metric_cell(test_f1, i - 1)}"
            "</tr>"
        )

    class_header = "".join(f"<th>{_escape(c)}</th>" for c in classes)
    per_class_rows = []
    cells = []
    for iou in paper_ious:
        cells.append(
            f"<td style='background:{_heatmap_bg(iou)}'>{_pct(iou, 1)}</td>"
        )
    per_class_rows.append(
        "<tr class='baseline'>"
        f"<td><strong>{_escape(paper['name'])}</strong></td>"
        f"<td>{_pct(paper['test_miou'])}</td>"
        + "".join(cells)
        + "</tr>"
    )
    for run in runs:
        ious_list = run["test"].get("per_class_iou")
        if ious_list is None:
            cells_html = ["<td>—</td>"] * len(classes)
        else:
            cells_html = []
            for iou in ious_list:
                cells_html.append(
                    f"<td style='background:{_heatmap_bg(iou)}'>{_pct(float(iou), 1)}</td>"
                )
        per_class_rows.append(
            "<tr>"
            f"<td><code>{_escape(run['name'])}</code></td>"
            f"<td>{_pct(run['test_miou'])}</td>"
            + "".join(cells_html)
            + "</tr>"
        )

    best_probe = runs[0]["name"] if runs else "—"
    best_probe_miou = _pct(runs[0]["test_miou"]) if runs else "—"
    gap = (runs[0]["test_miou"] - paper["test_miou"]) if runs else None

    doc = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PureForest sklearn probe — récapitulatif</title>
<style>
  :root {{
    color-scheme: light dark;
    --fg: #1a1a1a;
    --muted: #666;
    --border: #ccc;
    --bg: #fff;
    --bg-alt: #f6f6f6;
    --baseline: #eef3ff;
    --best: #1b7f3a;
    --bar: #3b82f6;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --fg: #eee;
      --muted: #aaa;
      --border: #444;
      --bg: #121212;
      --bg-alt: #1c1c1c;
      --baseline: #1a2438;
      --best: #6ee7a0;
      --bar: #60a5fa;
    }}
  }}
  body {{
    font-family: ui-sans-serif, system-ui, sans-serif;
    margin: 1.5rem auto;
    max-width: 1200px;
    padding: 0 1rem 3rem;
    color: var(--fg);
    background: var(--bg);
    line-height: 1.4;
  }}
  h1 {{ font-size: 1.35rem; font-weight: 600; margin: 0 0 0.4rem; }}
  h2 {{ font-size: 1.1rem; font-weight: 600; margin: 2rem 0 0.6rem; }}
  .muted {{ color: var(--muted); font-size: 0.9rem; }}
  .summary {{
    background: var(--bg-alt);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.85rem 1rem;
    margin: 1rem 0 1.5rem;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
  }}
  th, td {{
    border: 1px solid var(--border);
    padding: 0.35rem 0.45rem;
    text-align: right;
    white-space: nowrap;
  }}
  th:first-child, td:first-child,
  th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  th {{ background: var(--bg-alt); font-weight: 600; }}
  tr.baseline {{ background: var(--baseline); }}
  td.best {{ font-weight: 600; color: var(--best); }}
  code {{ font-size: 0.84em; }}
  .bar {{
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 0.3rem;
    overflow: hidden;
  }}
  .bar > span {{
    display: block;
    height: 100%;
    background: var(--bar);
  }}
  .scroll {{ overflow-x: auto; }}
  footer {{ margin-top: 2rem; color: var(--muted); font-size: 0.85rem; }}
</style>
</head>
<body>
  <h1>PureForest — sklearn linear probes</h1>
  <p class="muted">
    Grille (agg × C) sélectionnée sur <strong>val mIoU</strong> ; métriques ci-dessous pour le
    meilleur couple. Comparaison au baseline Lidar du papier PureForest (test OA / mIoU uniquement).
  </p>

  <div class="summary">
    <strong>Verdict :</strong> meilleur probe =
    <code>{_escape(best_probe)}</code> (test mIoU {best_probe_miou}%) ;
    baseline papier = {_pct(paper['test_miou'])}% mIoU / {_pct(paper['test_oa'])}% OA.
    Écart mIoU vs papier : {_pct(gap)} pts.
  </div>

  <h2>Classement (test mIoU)</h2>
  <div class="scroll">
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Run</th>
        <th>agg</th>
        <th>C</th>
        <th>dim</th>
        <th>train mIoU</th>
        <th>val mIoU</th>
        <th>test mIoU</th>
        <th>test OA</th>
        <th>test mAcc</th>
        <th>test macro-F1</th>
      </tr>
    </thead>
    <tbody>
      {''.join(ranking_rows)}
    </tbody>
  </table>
  </div>
  <p class="muted">Valeurs en %. Cellules vertes = meilleur parmi les probes (baseline hors compétition).</p>

  <h2>IoU par classe (test)</h2>
  <div class="scroll">
  <table>
    <thead>
      <tr>
        <th>Run</th>
        <th>mIoU</th>
        {class_header}
      </tr>
    </thead>
    <tbody>
      {''.join(per_class_rows)}
    </tbody>
  </table>
  </div>

  <footer>
    Baseline : {_escape(paper['source'])}.
    IoU par classe du papier d’après Fig. 8 (texte).
    Généré par <code>scripts/pureforest/summarize_sklearn_probes.py</code>
    → <code>{_escape(out_path)}</code>.
  </footer>
</body>
</html>
"""
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=Path("stats/pureforest/sklearn_probe"),
        help="Directory containing per-run probe folders with metrics.json",
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

    runs = _load_runs(probe_root)
    if not runs:
        raise SystemExit(f"No metrics.json found under {probe_root}")

    html_doc = render_html(runs, out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {out_path} ({len(runs)} runs + paper baseline)")


if __name__ == "__main__":
    main()
