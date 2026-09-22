"""Survey per-tile point counts in a preprocessed ECLAIR split, broken down by
review_category (approved/rejected), to pick a min_points threshold for
ECLAIRDataset instead of guessing/copying one from another dataset.

Background: a handful of ECLAIR train tiles are near-empty outliers (isolated
from the rest of the size distribution by an order of magnitude); feeding
these into Sonata pretraining's MultiViewGenerator produces degenerate views
(a few hundred points) that destabilize teacher-student matching and can
crash training with a non-finite loss. See configs/eclair/pretrain-sonata-v1m2-eclair.py.

Usage:
    python scripts/eclair/analyze_point_counts.py --root data/eclair/train
"""

import argparse
import glob
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="data/eclair/train", help="Preprocessed split directory"
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 5000, 10000],
        help="Candidate min_points thresholds to report counts below",
    )
    parser.add_argument(
        "--smallest", type=int, default=15, help="How many smallest tiles to list"
    )
    args = parser.parse_args()

    counts = []
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        meta_path = os.path.join(d, "meta.json")
        coord_path = os.path.join(d, "coord.npy")
        if not (os.path.isfile(meta_path) and os.path.isfile(coord_path)):
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        n = np.load(coord_path, mmap_mode="r").shape[0]
        counts.append((meta.get("review_category", "?"), n, os.path.basename(d)))

    counts.sort(key=lambda x: x[1])
    print(f"total tiles: {len(counts)}")

    for cat in sorted(set(c[0] for c in counts)):
        sub = np.array([n for c, n, _ in counts if c == cat])
        print(f"\n[{cat}] n_tiles={len(sub)}")
        for pct in (1, 5, 25, 50):
            print(f"  p{pct}: {np.percentile(sub, pct):.0f}")
        for thr in args.thresholds:
            print(f"  below {thr}: {(sub < thr).sum()}")

    print(f"\n{args.smallest} smallest tiles overall:")
    for cat, n, name in counts[: args.smallest]:
        print(f"  {name}  n_points={n}  review_category={cat}")


if __name__ == "__main__":
    main()
