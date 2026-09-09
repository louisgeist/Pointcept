#!/usr/bin/env python3
"""Offline sklearn linear probe on PureForest pooled embeddings.

Loads ``{train,val,test}.npz`` produced by
``scripts/extract_pureforest_pooled_embeddings.py``, builds four feature views
(mean / max / concat / sum), fits ``LogisticRegression`` with an L2 ``C`` grid
(weight-decay equivalent), selects the best ``(agg, C)`` on val, then reports
test once.

Usage::

    python scripts/probe_pureforest_sklearn.py \\
      --embeddings-dir stats/pureforest/embeddings/sonata_outdoor \\
      --output-dir stats/pureforest/sklearn_probe/sonata_outdoor

    python scripts/probe_pureforest_sklearn.py \\
      --embeddings-dir stats/pureforest/embeddings/sonata_outdoor_toy \\
      --class-weight balanced \\
      --Cs 0.01 0.1 1 10 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


AGG_NAMES = ("mean", "max", "concat", "sum")


def load_split(embeddings_dir: Path, split: str):
    path = embeddings_dir / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing embeddings file: {path}")
    with np.load(path, allow_pickle=True) as data:
        payload = {
            "names": data["names"],
            "category": data["category"].astype(np.int64),
            "mean_feat": data["mean_feat"].astype(np.float32),
            "max_feat": data["max_feat"].astype(np.float32),
        }
        class_names = (
            data["class_names"].tolist()
            if "class_names" in data.files
            else None
        )
    return payload, class_names


def build_features(mean_feat, max_feat, agg: str):
    if agg == "mean":
        return mean_feat
    if agg == "max":
        return max_feat
    if agg == "concat":
        return np.concatenate([mean_feat, max_feat], axis=1)
    if agg == "sum":
        return mean_feat + max_feat
    raise ValueError(f"Unknown aggregation {agg!r}; expected one of {AGG_NAMES}.")


def classification_metrics(y_true, y_pred, num_classes, ignore_index=-1):
    from pointcept.utils.misc import (
        f1_scores_from_hist,
        intersection_and_union,
        mean_acc_from_hist,
        mean_iou_from_hist,
    )

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    intersection, union, target = intersection_and_union(
        y_pred, y_true, num_classes, ignore_index=ignore_index
    )
    m_iou = float(mean_iou_from_hist(intersection, union))
    m_acc = float(mean_acc_from_hist(intersection, target, union=union))
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    f1, macro_f1 = f1_scores_from_hist(intersection, union, target)
    return {
        "allAcc": acc,
        "mAcc": m_acc,
        "mIoU": m_iou,
        "macro_f1": float(macro_f1),
        "per_class_f1": f1.tolist(),
        "intersection": intersection.tolist(),
        "union": union.tolist(),
        "target": target.tolist(),
    }


def _make_logistic_regression(c_value, class_weight, max_iter, n_jobs, seed):
    """Build LogisticRegression with kwargs compatible across sklearn versions.

    sklearn >= 1.8/1.9 deprecated/removed ``multi_class``, ``penalty='l2'``, and
    ``n_jobs`` for this estimator; older envs still expect some of them.
    """
    from sklearn.linear_model import LogisticRegression
    import inspect

    params = inspect.signature(LogisticRegression.__init__).parameters
    kwargs = {
        "C": float(c_value),
        "solver": "lbfgs",
        "max_iter": int(max_iter),
        "random_state": int(seed),
    }
    if "class_weight" in params:
        kwargs["class_weight"] = class_weight
    if "multi_class" in params:
        kwargs["multi_class"] = "multinomial"
    if "n_jobs" in params and n_jobs is not None:
        # n_jobs has no effect on lbfgs since sklearn 1.8; skip to avoid warnings.
        import sklearn

        major_minor = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        if major_minor < (1, 8):
            kwargs["n_jobs"] = n_jobs
    return LogisticRegression(**kwargs)


def fit_and_eval(
    *,
    x_train,
    y_train,
    x_val,
    y_val,
    c_value,
    class_weight,
    max_iter,
    n_jobs,
    seed,
):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)

    clf = _make_logistic_regression(
        c_value=c_value,
        class_weight=class_weight,
        max_iter=max_iter,
        n_jobs=n_jobs,
        seed=seed,
    )
    clf.fit(x_train_s, y_train)
    pred_val = clf.predict(x_val_s)
    return clf, scaler, pred_val


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--embeddings-dir",
        required=True,
        help="Directory with train.npz / val.npz [/ test.npz].",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write metrics.json (default: embeddings-dir/sklearn_probe).",
    )
    parser.add_argument(
        "--aggs",
        nargs="+",
        default=list(AGG_NAMES),
        choices=list(AGG_NAMES),
        help="Feature aggregations to evaluate.",
    )
    parser.add_argument(
        "--Cs",
        nargs="+",
        type=float,
        default=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3],
        help="Inverse L2 strength grid (sklearn C). Larger C = less regularization.",
    )
    parser.add_argument(
        "--class-weight",
        default="none",
        choices=["none", "balanced"],
        help="Pass 'balanced' to LogisticRegression (default: none).",
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--select-metric",
        default="mIoU",
        choices=["mIoU", "allAcc", "mAcc", "macro_f1"],
        help="Val metric used to pick best (agg, C).",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Only run train/val selection (no test.npz required).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is required. Install with: pip install scikit-learn"
        ) from exc

    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir or (embeddings_dir / "sklearn_probe"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train, class_names_train = load_split(embeddings_dir, "train")
    val, class_names_val = load_split(embeddings_dir, "val")
    class_names = class_names_train or class_names_val
    if class_names is None:
        from pointcept.datasets.preprocessing.pureforest.pureforest_classes import (
            CLASS_NAMES,
        )

        class_names = list(CLASS_NAMES)
    num_classes = len(class_names)

    class_weight = None if args.class_weight == "none" else args.class_weight
    results = []
    best = None

    print(
        f"[probe] train={train['category'].shape[0]}  val={val['category'].shape[0]}  "
        f"feat_dim={train['mean_feat'].shape[1]}  aggs={args.aggs}  Cs={args.Cs}"
    )

    for agg in args.aggs:
        x_train = build_features(train["mean_feat"], train["max_feat"], agg)
        x_val = build_features(val["mean_feat"], val["max_feat"], agg)
        for c_value in args.Cs:
            clf, scaler, pred_val = fit_and_eval(
                x_train=x_train,
                y_train=train["category"],
                x_val=x_val,
                y_val=val["category"],
                c_value=c_value,
                class_weight=class_weight,
                max_iter=args.max_iter,
                n_jobs=args.n_jobs,
                seed=args.seed,
            )
            metrics_val = classification_metrics(
                val["category"], pred_val, num_classes=num_classes
            )
            row = {
                "agg": agg,
                "C": float(c_value),
                "feat_dim": int(x_train.shape[1]),
                "val": metrics_val,
            }
            results.append(row)
            score = metrics_val[args.select_metric]
            print(
                f"[probe] agg={agg:6s} C={c_value:<8g} "
                f"val/{args.select_metric}={score:.4f}  "
                f"allAcc={metrics_val['allAcc']:.4f}  "
                f"mIoU={metrics_val['mIoU']:.4f}  "
                f"macro_f1={metrics_val['macro_f1']:.4f}"
            )
            if best is None or score > best["score"]:
                best = {
                    "score": float(score),
                    "agg": agg,
                    "C": float(c_value),
                    "clf": clf,
                    "scaler": scaler,
                    "val": metrics_val,
                    "feat_dim": int(x_train.shape[1]),
                }

    assert best is not None
    print(
        f"[probe] best on val: agg={best['agg']} C={best['C']} "
        f"{args.select_metric}={best['score']:.4f}"
    )

    report = {
        "embeddings_dir": str(embeddings_dir),
        "class_names": class_names,
        "num_classes": num_classes,
        "select_metric": args.select_metric,
        "class_weight": class_weight,
        "Cs": [float(c) for c in args.Cs],
        "aggs": list(args.aggs),
        "grid_results": [
            {
                "agg": r["agg"],
                "C": r["C"],
                "feat_dim": r["feat_dim"],
                "val": {
                    k: r["val"][k]
                    for k in ("allAcc", "mAcc", "mIoU", "macro_f1")
                },
            }
            for r in results
        ],
        "best": {
            "agg": best["agg"],
            "C": best["C"],
            "feat_dim": best["feat_dim"],
            "val": {
                k: best["val"][k]
                for k in ("allAcc", "mAcc", "mIoU", "macro_f1", "per_class_f1")
            },
        },
    }

    if not args.skip_test:
        test_path = embeddings_dir / "test.npz"
        if not test_path.is_file():
            print(f"[probe] no test.npz at {test_path} -- skipping test.")
        else:
            test, _ = load_split(embeddings_dir, "test")
            x_test = build_features(test["mean_feat"], test["max_feat"], best["agg"])
            x_test_s = best["scaler"].transform(x_test)
            pred_test = best["clf"].predict(x_test_s)
            metrics_test = classification_metrics(
                test["category"], pred_test, num_classes=num_classes
            )
            report["best"]["test"] = {
                k: metrics_test[k]
                for k in ("allAcc", "mAcc", "mIoU", "macro_f1", "per_class_f1")
            }
            print(
                f"[probe] test (winner only): "
                f"allAcc={metrics_test['allAcc']:.4f}  "
                f"mIoU={metrics_test['mIoU']:.4f}  "
                f"macro_f1={metrics_test['macro_f1']:.4f}"
            )

            # Optional: dump predictions for the winner.
            pred_path = output_dir / "best_test_predictions.npz"
            np.savez_compressed(
                pred_path,
                names=test["names"],
                category=test["category"],
                pred=pred_test.astype(np.int64),
                agg=best["agg"],
                C=best["C"],
            )
            print(f"[probe] wrote {pred_path}")

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[probe] wrote {metrics_path}")


if __name__ == "__main__":
    main()
