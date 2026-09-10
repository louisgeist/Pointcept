#!/usr/bin/env python3
"""Archive best multi-seed linear-probe checkpoints for external benchmarks.

For each row in the jobs CSV (H3D / DALES / ECLAIR × Sonata + MS-Enc
backbones), locate the Jean-Zay job directory, pick the seed with the highest
test metric, and copy (or emit scp commands for) that checkpoint plus
lightweight provenance JSON.

Supports two on-disk layouts:

- **nested** (``grid_then_seeds``): ``seeds/seed_ensemble_results.json`` +
  ``seeds/model/probe_best_seedK.pth`` (typical H3D).
- **flat** (legacy seed-ensemble): ``seed_ensemble_results.json`` +
  ``model/probe_best_seedK.pth`` at the job root (typical DALES / ECLAIR).

Example (on Jean-Zay):

  python scripts/archive_external_lin_probe_ckpts.py --dry-run
  python scripts/archive_external_lin_probe_ckpts.py --copy \\
    --dest $WORK/Pointcept/ckpt/lin_probe_pack

Example (emit scp lines for hecate):

  python scripts/archive_external_lin_probe_ckpts.py --emit-scp \\
    --hecate-dest lgeist@hecate:/data/geist/Pointcept/ckpt/lin_probe
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS = Path(__file__).resolve().parent / "lin_probe_external_jobs.csv"
DEFAULT_SRC_ROOT = Path(
    "/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm"
)
DEFAULT_DEST = REPO_ROOT / "ckpt" / "lin_probe"
DEFAULT_JZ_HOST = "usi32yh@jean-zay.idris.fr"
DEFAULT_HECATE_DEST = "lgeist@hecate:/data/geist/Pointcept/ckpt/lin_probe"

SEED_RESULT = "seed_ensemble_results.json"
GRID_RESULT = "grid_search_results.json"
SUMMARY_CSV = "grid_then_seeds_summary.csv"
JOB_INFO = "job_info.log"

# Metric key inside per_probe rows of seed_ensemble_results.json
METRIC_BY_DATASET = {
    "h3d": "test/f1_macro",
    "dales": "test/mIoU",
    "eclair": "test/mIoU",
}


@dataclass
class JobRow:
    dataset: str
    backbone: str
    exp_name: str
    job_id: str
    wandb_id: str = ""
    test_miou_reported: str = ""
    lr: str = ""


@dataclass
class JobLayout:
    """On-disk layout of a grid→seed (or legacy seed-ensemble) job.

    ``nested``: ``grid_then_seeds`` (``seeds/``, ``grid/`` subdirs) — H3D.
    ``flat``: single job dir with ``seed_ensemble_results.json`` + ``model/``
    at the root — older DALES / ECLAIR seed-ensemble runs.
    """

    kind: str
    seed_json: Path
    model_dir: Path
    grid_json: Optional[Path]
    summary_csv: Optional[Path]


@dataclass
class Selection:
    row: JobRow
    job_dir: Path
    seed_name: str
    metric_key: str
    metric_value: float
    ckpt_path: Path
    select_metric: Optional[str]
    test_mIoU_mean: Optional[float]
    test_mIoU_std: Optional[float]
    test_f1_macro_mean: Optional[float]
    test_f1_macro_std: Optional[float]
    status: str
    detail: str = ""
    layout: Optional[JobLayout] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_jobs(path: Path) -> list[JobRow]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"dataset", "backbone", "exp_name", "job_id"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(
                f"{path}: missing columns {sorted(required - set(reader.fieldnames or []))}"
            )
        rows = []
        for raw in reader:
            rows.append(
                JobRow(
                    dataset=raw["dataset"].strip().lower(),
                    backbone=raw["backbone"].strip().lower(),
                    exp_name=raw["exp_name"].strip(),
                    job_id=str(raw["job_id"]).strip(),
                    wandb_id=(raw.get("wandb_id") or "").strip(),
                    test_miou_reported=(raw.get("test_miou_reported") or "").strip(),
                    lr=(raw.get("lr") or "").strip(),
                )
            )
    return rows


def parse_job_info_exp_name(job_dir: Path) -> Optional[str]:
    path = job_dir / JOB_INFO
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^Exp name:\s*(.+)$", text, flags=re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip()


def _as_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pick_best_seed(
    seed_results: dict[str, Any],
    dataset: str,
) -> tuple[str, str, float, Optional[str]]:
    """Return (seed_name, metric_key, metric_value, select_metric_from_json)."""
    select_metric = seed_results.get("select_metric")
    per_probe = seed_results.get("per_probe") or {}
    if not per_probe:
        raise ValueError("seed_ensemble_results.json has empty per_probe")

    # Prefer baked-in select_metric; else dataset default.
    if select_metric == "macro_f1":
        metric_key = "test/f1_macro"
    elif select_metric == "mIoU":
        metric_key = "test/mIoU"
    else:
        metric_key = METRIC_BY_DATASET.get(dataset, "test/mIoU")

    best_name: Optional[str] = None
    best_val: Optional[float] = None
    for name, row in per_probe.items():
        val = _as_float(row.get(metric_key))
        if val is None:
            continue
        if best_val is None or val > best_val:
            best_val = val
            best_name = name

    if best_name is None or best_val is None:
        raise ValueError(
            f"no per_probe entry has a numeric {metric_key!r} "
            f"(select_metric={select_metric!r})"
        )
    return best_name, metric_key, best_val, select_metric


def discover_job_layout(job_dir: Path) -> Optional[JobLayout]:
    """Resolve nested (``seeds/``) vs flat (job-root) seed-ensemble layouts."""
    nested_json = job_dir / "seeds" / SEED_RESULT
    flat_json = job_dir / SEED_RESULT

    if nested_json.is_file():
        model_dir = job_dir / "seeds" / "model"
        grid = job_dir / "grid" / GRID_RESULT
        summary = job_dir / SUMMARY_CSV
        return JobLayout(
            kind="nested",
            seed_json=nested_json,
            model_dir=model_dir,
            grid_json=grid if grid.is_file() else None,
            summary_csv=summary if summary.is_file() else None,
        )

    if flat_json.is_file():
        model_dir = job_dir / "model"
        grid = job_dir / GRID_RESULT
        if not grid.is_file():
            alt = job_dir / "grid" / GRID_RESULT
            grid = alt if alt.is_file() else grid
        summary = job_dir / SUMMARY_CSV
        return JobLayout(
            kind="flat",
            seed_json=flat_json,
            model_dir=model_dir,
            grid_json=grid if grid.is_file() else None,
            summary_csv=summary if summary.is_file() else None,
        )

    return None


def _fail_selection(
    row: JobRow,
    job_dir: Path,
    status: str,
    detail: str,
) -> Selection:
    return Selection(
        row=row,
        job_dir=job_dir,
        seed_name="",
        metric_key="",
        metric_value=float("nan"),
        ckpt_path=Path(),
        select_metric=None,
        test_mIoU_mean=None,
        test_mIoU_std=None,
        test_f1_macro_mean=None,
        test_f1_macro_std=None,
        status=status,
        detail=detail,
    )


def resolve_selection(
    row: JobRow,
    src_root: Path,
    *,
    verify_exp_name: bool,
) -> Selection:
    job_dir = src_root / row.job_id
    if not job_dir.is_dir():
        return _fail_selection(row, job_dir, "missing_job_dir", str(job_dir))

    if verify_exp_name:
        logged = parse_job_info_exp_name(job_dir)
        if logged is not None and logged != row.exp_name:
            return _fail_selection(
                row,
                job_dir,
                "exp_name_mismatch",
                f"job_info={logged!r} csv={row.exp_name!r}",
            )

    layout = discover_job_layout(job_dir)
    if layout is None:
        return _fail_selection(
            row,
            job_dir,
            "missing_seed_results",
            f"neither seeds/{SEED_RESULT} nor {SEED_RESULT} under {job_dir}",
        )

    try:
        data = json.loads(layout.seed_json.read_text(encoding="utf-8"))
        seed_name, metric_key, metric_value, select_metric = pick_best_seed(
            data, row.dataset
        )
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        return _fail_selection(row, job_dir, "bad_seed_results", str(exc))

    ckpt_path = layout.model_dir / f"probe_best_{seed_name}.pth"
    means = dict(
        test_mIoU_mean=_as_float(data.get("test_mIoU_mean")),
        test_mIoU_std=_as_float(data.get("test_mIoU_std")),
        test_f1_macro_mean=_as_float(data.get("test_f1_macro_mean")),
        test_f1_macro_std=_as_float(data.get("test_f1_macro_std")),
    )
    if not ckpt_path.is_file():
        return Selection(
            row=row,
            job_dir=job_dir,
            seed_name=seed_name,
            metric_key=metric_key,
            metric_value=metric_value,
            ckpt_path=ckpt_path,
            select_metric=select_metric,
            status="missing_ckpt",
            detail=str(ckpt_path),
            layout=layout,
            **means,
        )

    return Selection(
        row=row,
        job_dir=job_dir,
        seed_name=seed_name,
        metric_key=metric_key,
        metric_value=metric_value,
        ckpt_path=ckpt_path,
        select_metric=select_metric,
        status="ok",
        detail=layout.kind,
        layout=layout,
        **means,
    )


def slot_dir(dest: Path, row: JobRow) -> Path:
    return dest / row.dataset / row.backbone


def build_meta(sel: Selection) -> dict[str, Any]:
    row = sel.row
    return {
        "archived_at": _utc_now(),
        "dataset": row.dataset,
        "backbone": row.backbone,
        "exp_name": row.exp_name,
        "job_id": row.job_id,
        "wandb_id": row.wandb_id,
        "test_miou_reported": row.test_miou_reported,
        "lr": row.lr,
        "job_dir": str(sel.job_dir),
        "layout": sel.layout.kind if sel.layout else None,
        "seed_name": sel.seed_name,
        "metric_key": sel.metric_key,
        "metric_value": sel.metric_value,
        "select_metric": sel.select_metric,
        "test_mIoU_mean": sel.test_mIoU_mean,
        "test_mIoU_std": sel.test_mIoU_std,
        "test_f1_macro_mean": sel.test_f1_macro_mean,
        "test_f1_macro_std": sel.test_f1_macro_std,
        "source_ckpt": str(sel.ckpt_path),
        "archived_as": "probe_best.pth",
    }


def copy_slot(sel: Selection, dest: Path) -> Path:
    out = slot_dir(dest, sel.row)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sel.ckpt_path, out / "probe_best.pth")
    (out / "meta.json").write_text(
        json.dumps(build_meta(sel), indent=2) + "\n", encoding="utf-8"
    )
    layout = sel.layout
    if layout is not None:
        shutil.copy2(layout.seed_json, out / SEED_RESULT)
        if layout.grid_json is not None:
            shutil.copy2(layout.grid_json, out / GRID_RESULT)
        if layout.summary_csv is not None:
            shutil.copy2(layout.summary_csv, out / SUMMARY_CSV)
    return out


def write_manifest(path: Path, selections: list[Selection]) -> None:
    fields = [
        "status",
        "dataset",
        "backbone",
        "exp_name",
        "job_id",
        "wandb_id",
        "seed_name",
        "metric_key",
        "metric_value",
        "select_metric",
        "test_mIoU_mean",
        "test_mIoU_std",
        "test_f1_macro_mean",
        "test_f1_macro_std",
        "source_ckpt",
        "detail",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sel in selections:
            writer.writerow(
                {
                    "status": sel.status,
                    "dataset": sel.row.dataset,
                    "backbone": sel.row.backbone,
                    "exp_name": sel.row.exp_name,
                    "job_id": sel.row.job_id,
                    "wandb_id": sel.row.wandb_id,
                    "seed_name": sel.seed_name,
                    "metric_key": sel.metric_key,
                    "metric_value": (
                        f"{sel.metric_value:.6f}" if sel.status == "ok" else ""
                    ),
                    "select_metric": sel.select_metric or "",
                    "test_mIoU_mean": (
                        f"{sel.test_mIoU_mean:.6f}"
                        if sel.test_mIoU_mean is not None
                        else ""
                    ),
                    "test_mIoU_std": (
                        f"{sel.test_mIoU_std:.6f}"
                        if sel.test_mIoU_std is not None
                        else ""
                    ),
                    "test_f1_macro_mean": (
                        f"{sel.test_f1_macro_mean:.6f}"
                        if sel.test_f1_macro_mean is not None
                        else ""
                    ),
                    "test_f1_macro_std": (
                        f"{sel.test_f1_macro_std:.6f}"
                        if sel.test_f1_macro_std is not None
                        else ""
                    ),
                    "source_ckpt": str(sel.ckpt_path) if sel.ckpt_path else "",
                    "detail": sel.detail,
                }
            )


def emit_scp_commands(
    selections: list[Selection],
    *,
    jz_host: str,
    hecate_dest: str,
) -> list[str]:
    """Emit per-file scp commands (fallback if the pack workflow is unavailable)."""
    lines: list[str] = [
        "# Create destination dirs on hecate first, then run these scp lines.",
    ]
    base = hecate_dest.rstrip("/")
    datasets = sorted({s.row.dataset for s in selections if s.status == "ok"})
    backbones = sorted({s.row.backbone for s in selections if s.status == "ok"})
    for ds in datasets:
        for bb in backbones:
            local_hint = base.split(":", 1)[-1]
            lines.append(f"# mkdir -p {local_hint}/{ds}/{bb}")
    for sel in selections:
        if sel.status != "ok":
            lines.append(f"# SKIP {sel.row.exp_name}: {sel.status} {sel.detail}")
            continue
        slot = f"{base}/{sel.row.dataset}/{sel.row.backbone}"
        lines.append(
            f"scp -J passerelle {jz_host}:{sel.ckpt_path} {slot}/probe_best.pth"
        )
        layout = sel.layout
        if layout is not None:
            lines.append(
                f"scp -J passerelle {jz_host}:{layout.seed_json} {slot}/{SEED_RESULT}"
            )
            if layout.grid_json is not None:
                lines.append(
                    f"scp -J passerelle {jz_host}:{layout.grid_json} {slot}/{GRID_RESULT}"
                )
            if layout.summary_csv is not None:
                lines.append(
                    f"scp -J passerelle {jz_host}:{layout.summary_csv} {slot}/{SUMMARY_CSV}"
                )
        else:
            for name, rel in (
                (SEED_RESULT, Path("seeds") / SEED_RESULT),
                (GRID_RESULT, Path("grid") / GRID_RESULT),
                (SUMMARY_CSV, Path(SUMMARY_CSV)),
            ):
                src = sel.job_dir / rel
                lines.append(f"scp -J passerelle {jz_host}:{src} {slot}/{name}")
    return lines


def emit_scp_pack_workflow(
    *,
    jz_host: str,
    pack_remote: str,
    hecate_dest: str,
) -> str:
    """Recommended workflow: pack on JZ, then one recursive scp."""
    return "\n".join(
        [
            "# Recommended: build the pack on Jean-Zay, then pull once to hecate:",
            f"#   ssh {jz_host}",
            "#   cd /lustre/fswork/projects/rech/unv/usi32yh/Pointcept",
            "#   python scripts/archive_external_lin_probe_ckpts.py --copy \\",
            f"#     --dest {pack_remote}",
            f"mkdir -p {hecate_dest.split(':', 1)[-1]}",
            f"scp -J passerelle -r {jz_host}:{pack_remote}/. {hecate_dest}/",
        ]
    )


def print_selection(sel: Selection) -> None:
    row = sel.row
    if sel.status == "ok":
        layout = sel.layout.kind if sel.layout else "?"
        print(
            f"[ok] {row.exp_name:22s} job={row.job_id} layout={layout} "
            f"{sel.seed_name} {sel.metric_key}={sel.metric_value:.4f} "
            f"<- {sel.ckpt_path}",
            flush=True,
        )
    else:
        print(
            f"[{sel.status}] {row.exp_name:22s} job={row.job_id} {sel.detail}",
            flush=True,
        )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--jobs",
        type=Path,
        default=DEFAULT_JOBS,
        help=f"jobs CSV (default: {DEFAULT_JOBS})",
    )
    ap.add_argument(
        "--src-root",
        type=Path,
        default=DEFAULT_SRC_ROOT,
        help="Jean-Zay logs/slurm root",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="local archive destination (for --copy)",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve best seeds and write manifest only (default if no mode set)",
    )
    mode.add_argument(
        "--copy",
        action="store_true",
        help="copy best ckpts + provenance into --dest",
    )
    mode.add_argument(
        "--emit-scp",
        action="store_true",
        help="print scp commands / pack workflow for hecate",
    )
    ap.add_argument(
        "--verify-exp-name",
        action="store_true",
        help="require job_info.log Exp name to match CSV exp_name",
    )
    ap.add_argument("--jz-host", default=DEFAULT_JZ_HOST)
    ap.add_argument("--hecate-dest", default=DEFAULT_HECATE_DEST)
    ap.add_argument(
        "--pack-remote",
        default="/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/ckpt/lin_probe_pack",
        help="JZ pack path used in the recommended scp workflow",
    )
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not (args.dry_run or args.copy or args.emit_scp):
        args.dry_run = True

    jobs = load_jobs(args.jobs)
    selections = [
        resolve_selection(row, args.src_root, verify_exp_name=args.verify_exp_name)
        for row in jobs
    ]

    for sel in selections:
        print_selection(sel)

    n_ok = sum(1 for s in selections if s.status == "ok")
    n_fail = len(selections) - n_ok
    print(f"\n{n_ok}/{len(selections)} ok, {n_fail} failed", flush=True)

    if args.emit_scp:
        print(
            "\n"
            + emit_scp_pack_workflow(
                jz_host=args.jz_host,
                pack_remote=args.pack_remote,
                hecate_dest=args.hecate_dest,
            )
        )
        # Synthesize expected source paths when the JZ tree is not mounted
        # locally, so --emit-scp remains useful from hecate.
        for sel in selections:
            if sel.status == "ok":
                continue
            if sel.status != "missing_job_dir":
                continue
            # Assume standard layout; seed name unknown until dry-run on JZ.
            sel.ckpt_path = (
                args.src_root
                / sel.row.job_id
                / "seeds"
                / "model"
                / "probe_best_SEED.pth"
            )
            sel.job_dir = args.src_root / sel.row.job_id
            sel.status = "ok"
            sel.detail = "path assumed (run --dry-run on JZ to resolve seed)"
            sel.seed_name = "SEED"
        print("\n# Per-file fallback (replace SEED after JZ dry-run):\n")
        for line in emit_scp_commands(
            selections, jz_host=args.jz_host, hecate_dest=args.hecate_dest
        ):
            print(line)
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    # Always keep a copy of the jobs map next to the archive.
    shutil.copy2(args.jobs, args.dest / "jobs.csv")
    write_manifest(args.dest / "manifest.csv", selections)

    if args.copy:
        for sel in selections:
            if sel.status != "ok":
                continue
            out = copy_slot(sel, args.dest)
            print(f"copied -> {out}", flush=True)
        print(f"wrote {args.dest / 'manifest.csv'}", flush=True)
    else:
        print(f"dry-run manifest -> {args.dest / 'manifest.csv'}", flush=True)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
