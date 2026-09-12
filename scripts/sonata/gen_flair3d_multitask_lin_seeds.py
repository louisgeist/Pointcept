#!/usr/bin/env python3
"""Generate 10 multi-task Sonata lin-probe seed configs from a GridProbe winner.

Reads ``grid_search_results.json`` (or an explicit ``--lr``) and clones
``configs/experiment/w112/6/sonata_flair3d_lin/multi-sonata-v1m2-flair3d-lin-seed_1.py``
into ``_1.py`` … ``_10.py``, patching ``lr`` / ``seed`` / ``num_exp``.

``wandb_run_name`` is an f-string over those knobs, so it updates for free.

Examples:

  python scripts/sonata/gen_flair3d_multitask_lin_seeds.py \\
    --grid-dir logs/slurm/<GRID_JOB>

  python scripts/sonata/gen_flair3d_multitask_lin_seeds.py --lr 2e-2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = REPO_ROOT / "configs/experiment/w112/6/sonata_flair3d_lin"
TEMPLATE_NAME = "multi-sonata-v1m2-flair3d-lin-seed_1.py"
GRID_RESULT = "grid_search_results.json"
N_SEEDS = 10

# Compact literals matching the 12-LR GridProbe axis.
_LR_LITERALS = {
    1e-4: "1e-4",
    2e-4: "2e-4",
    5e-4: "5e-4",
    1e-3: "1e-3",
    2e-3: "2e-3",
    5e-3: "5e-3",
    1e-2: "1e-2",
    2e-2: "2e-2",
    5e-2: "5e-2",
    1e-1: "1e-1",
    2e-1: "2e-1",
    5e-1: "5e-1",
}

NUM_EXP_RE = re.compile(r"^num_exp = \d+", re.MULTILINE)
SEED_RE = re.compile(r"^seed = \d+", re.MULTILINE)
LR_RE = re.compile(r"^lr = .+$", re.MULTILINE)


def format_lr(lr: float) -> str:
    for value, literal in _LR_LITERALS.items():
        if abs(lr - value) / value < 1e-8:
            return literal
    return repr(lr)


def lr_from_probe_config(probe_config: dict) -> float:
    optimizer = probe_config.get("optimizer") or {}
    if optimizer.get("lr") is not None:
        return float(optimizer["lr"])
    scheduler = probe_config.get("scheduler") or {}
    max_lr = scheduler.get("max_lr")
    if max_lr is not None:
        if isinstance(max_lr, (list, tuple)):
            max_lr = max_lr[0]
        return float(max_lr)
    raise ValueError(
        "winner probe_config has neither optimizer.lr nor scheduler.max_lr"
    )


def lr_from_grid_dir(grid_dir: Path) -> tuple[float, str]:
    path = grid_dir / GRID_RESULT
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    winner = data.get("winner") or {}
    name = winner.get("probe_name") or "<unknown>"
    probe_config = winner.get("probe_config")
    if not probe_config:
        raise ValueError(f"{path}: no usable winner.probe_config ({winner!r})")
    return lr_from_probe_config(probe_config), str(name)


def patch_template(text: str, *, lr: float, seed: int, num_exp: int) -> str:
    if NUM_EXP_RE.search(text) is None:
        raise ValueError("template is missing a 'num_exp = N' assignment")
    if SEED_RE.search(text) is None:
        raise ValueError("template is missing a 'seed = N' assignment")
    if LR_RE.search(text) is None:
        raise ValueError("template is missing an 'lr = ...' assignment")
    text = NUM_EXP_RE.sub(f"num_exp = {num_exp}", text, count=1)
    text = SEED_RE.sub(f"seed = {seed}", text, count=1)
    text = LR_RE.sub(f"lr = {format_lr(lr)}  # GridProbe winner lr", text, count=1)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=None,
        help=f"GridProbe save dir containing {GRID_RESULT}",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override winner lr (skip reading grid_search_results.json)",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=DEFAULT_DIR,
        help="Directory holding the seed_1.py template (and write target)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=N_SEEDS,
        help=f"Number of seed configs to write (default {N_SEEDS})",
    )
    args = parser.parse_args()

    if args.n_seeds < 1:
        print("error: --n-seeds must be >= 1", file=sys.stderr)
        return 2
    if args.lr is None and args.grid_dir is None:
        print("error: pass --grid-dir or --lr", file=sys.stderr)
        return 2

    template_path = args.template_dir / TEMPLATE_NAME
    if not template_path.is_file():
        print(f"error: template not found: {template_path}", file=sys.stderr)
        return 1

    winner_name = None
    if args.lr is not None:
        lr = float(args.lr)
    else:
        grid_dir = args.grid_dir
        if not grid_dir.is_absolute():
            grid_dir = (REPO_ROOT / grid_dir).resolve()
        lr, winner_name = lr_from_grid_dir(grid_dir)

    source = template_path.read_text(encoding="utf-8")
    stem = "multi-sonata-v1m2-flair3d-lin-seed"
    written = []
    for num_exp in range(1, args.n_seeds + 1):
        seed = num_exp - 1
        out_path = args.template_dir / f"{stem}_{num_exp}.py"
        out_path.write_text(
            patch_template(source, lr=lr, seed=seed, num_exp=num_exp),
            encoding="utf-8",
        )
        written.append(out_path)

    print(f"lr={format_lr(lr)}" + (f"  winner={winner_name}" if winner_name else ""))
    for path in written:
        try:
            shown = path.resolve().relative_to(REPO_ROOT)
        except ValueError:
            shown = path
        print(f"  wrote {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
