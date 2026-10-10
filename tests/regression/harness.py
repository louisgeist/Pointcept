"""Non-regression harness for the release cleanup.

Capture a reference before a cleanup step, capture again after, compare:

  PY=~/.local/share/mamba/envs/pointcept/bin/python
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $PY tests/regression/harness.py capture --stage build   --out local/regression/ref
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $PY tests/regression/harness.py capture --stage forward --out local/regression/ref
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $PY tests/regression/harness.py capture --stage train   --out local/regression/ref
  $PY tests/regression/harness.py compare local/regression/ref local/regression/new [--map map.json]
  $PY tests/regression/harness.py accept local/regression/ref local/regression/new <config path or case>...

local/regression/ref_tag is the capture of the submission tag and never changes; local/regression/ref is the
rolling reference, updated with `accept` only for intended changes that were checked one by one.

Stages (cases in tests/regression/cases.py):
  build    every config: parsed config, model keys/shapes/seeded-init hashes, dataset type/length/sample list
  forward  released checkpoints on fixed local samples: multitask heads, GridProbe backbone features
  train    short tools/train.py runs (train + val + test), seed 0: loss curve and metrics

--map json: {"paths": [[old, new], ...], "strings": [[old, new], ...]} substring replacements applied to
the reference (config paths / config values) so renamed configs still line up.
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases  # noqa: E402

MAX_POINTS = 30000
# Measured on two identical reference runs (2026-10-10): step-1 loss differs by <= 7e-4 relative,
# steps 2-10 by <= 1.1e-2 (KPConvX, SpUNet probe), later steps diverge.
TRAIN_STRICT_STEPS = 10
FORWARD_MEAN_RTOL = 1e-3
FORWARD_MIN_AGREEMENT = 0.999
TRAIN_STEP1_RTOL = 2e-3
TRAIN_RTOL = 5e-2
FEAT_ROWS = 2048


# ----------------------------------------------------------------------------- helpers
def seed_all(seed=0):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if callable(x):
        return f"<callable {getattr(x, '__name__', type(x).__name__)}>"
    return f"<{type(x).__name__} {x}>"


def sha(b):
    return hashlib.sha1(b).hexdigest()[:16]


def tensor_sha(t):
    return sha(t.detach().cpu().contiguous().numpy().tobytes())


def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def read_jsonl(path):
    out = {}
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                out[r["key"]] = r
    return out


def quiet_logs():
    import logging

    logging.disable(logging.WARNING)


# ----------------------------------------------------------------------------- stage build
def capture_build(out, only=None):
    import torch

    from pointcept.datasets import build_dataset
    from pointcept.models import build_model
    from pointcept.utils.config import Config

    quiet_logs()
    path = os.path.join(out, "build.jsonl")
    done = read_jsonl(path)
    ds_cache = {}
    paths = [p for p in cases.build_configs() if not only or re.search(only, p)]
    for i, p in enumerate(paths):
        if p in done:
            continue
        t0 = time.time()
        rec = dict(key=p)
        try:
            cfg = Config.fromfile(p)
        except Exception as e:
            rec["cfg_error"] = f"{type(e).__name__}: {e}"[:500]
            append_jsonl(path, rec)
            continue
        cfgd = jsonable(dict(cfg._cfg_dict))
        rec["cfg"] = cfgd
        if isinstance(cfgd.get("model"), dict):
            try:
                seed_all(0)
                model = build_model(cfg.model)
                sd = model.state_dict()
                rec["model"] = dict(
                    type=cfg.model.get("type"),
                    n_params=int(sum(v.numel() for v in model.parameters())),
                    tensors={k: [list(v.shape), tensor_sha(v)] for k, v in sd.items()},
                )
                del model, sd
            except Exception as e:
                rec["model_error"] = f"{type(e).__name__}: {e}"[:500]
        data = cfg.get("data")
        if isinstance(data, dict):
            rec["data"] = {}
            for split in ("train", "val", "test"):
                dcfg = data.get(split)
                if not isinstance(dcfg, dict) or "type" not in dcfg:
                    continue
                key = json.dumps(jsonable(dict(dcfg)), sort_keys=True)
                if key not in ds_cache:
                    try:
                        ds = build_dataset(dcfg)
                        root = str(dcfg.get("data_root", ""))
                        # Relative to data_root, so moving the data directory does not change the hash.
                        names = [os.path.relpath(str(n), root) if root and str(n).startswith(root) else str(n)
                                 for n in getattr(ds, "data_list", [])]
                        ds_cache[key] = dict(type=dcfg["type"], len=len(ds), n_list=len(names),
                                             list_sha=sha("\n".join(names).encode()), head=names[:3])
                        del ds
                    except Exception as e:
                        ds_cache[key] = dict(error=f"{type(e).__name__}: {e}"[:500])
                rec["data"][split] = ds_cache[key]
        rec["seconds"] = round(time.time() - t0, 1)
        append_jsonl(path, rec)
        torch.cuda.empty_cache()
        print(f"[build {i + 1}/{len(paths)}] {p} {rec['seconds']}s"
              + (" MODEL_ERROR" if "model_error" in rec else ""), flush=True)


# ----------------------------------------------------------------------------- stage forward
def load_ckpt(model, path, keywords="", replacement=None, exclude_keys=()):
    """Same key handling as the CheckpointLoader hook (single process)."""
    import torch

    replacement = keywords if replacement is None else replacement
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    weight = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        if not k.startswith("module."):
            k = "module." + k
        if exclude_keys and any(e in k for e in exclude_keys):
            continue
        if keywords in k:
            k = k.replace(keywords, replacement, 1)
        weight[k[len("module."):]] = v
    own = model.state_dict()
    bad_shape = sorted(k for k, v in weight.items() if k in own and own[k].shape != v.shape)
    for k in bad_shape:
        weight.pop(k)
    res = model.load_state_dict(weight, strict=False)
    return dict(missing=sorted(res.missing_keys), unexpected=sorted(res.unexpected_keys),
                shape_mismatch=bad_shape)


def make_input(dcfg, pick, ds_cache):
    """pick = tile name or sample index. Seeded val pipeline + centre crop to MAX_POINTS."""
    import torch

    from pointcept.datasets import build_dataset
    from pointcept.datasets.utils import collate_fn

    key = json.dumps(jsonable(dict(dcfg)), sort_keys=True)
    if key not in ds_cache:
        ds_cache[key] = build_dataset(dcfg)
    ds = ds_cache[key]
    if len(ds.data_list) == 0:
        raise LookupError(f"no local data under {dcfg.get('data_root')} ({dcfg.get('split')})")
    if isinstance(pick, str):
        names = [os.path.basename(str(p)) for p in ds.data_list]
        idx = names.index(pick)
    else:
        idx = pick
    seed_all(0)
    item = ds[idx]
    name = ds.get_data_name(idx) if hasattr(ds, "get_data_name") else str(idx)
    n = item["coord"].shape[0]
    if n > MAX_POINTS:
        c = item["coord"][:, :2]
        d = ((c - c.mean(0, keepdim=True)) ** 2).sum(1)
        keep = torch.argsort(d)[:MAX_POINTS].sort().values
        for k, v in list(item.items()):
            if k.startswith("origin_") or k == "inverse":
                item.pop(k)
            elif torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == n:
                item[k] = v[keep]
        item["offset"] = torch.tensor([keep.numel()])
    item = collate_fn([item])
    hashes = {k: [list(v.shape), tensor_sha(v)] for k, v in item.items() if torch.is_tensor(v)}
    item = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in item.items()}
    return item, hashes, str(name)


def flatten_out(out, prefix=""):
    import torch

    flat = {}
    for k, v in out.items():
        if torch.is_tensor(v):
            flat[prefix + k] = v.detach().float().cpu().numpy()
        elif isinstance(v, dict):
            flat.update(flatten_out(v, prefix + k + "/"))
    return flat


def flair_dcfg(cfg, split):
    dcfg = dict(cfg.data.val)
    dcfg.pop("stratified_subset_manifest", None)
    dcfg.pop("max_sample", None)
    dcfg.update(split=split, csv_manifest=cases.FLAIR3D_MANIFEST[split], min_points=None)
    return dcfg


def resolve_probe_ckpt(weight):
    if not weight:
        return None
    for k, name in cases.JOB_TO_CKPT.items():
        if f"/{k}/" in weight:
            return cases.CKPT[name]
    return weight if os.path.isfile(weight) else None


def capture_forward(out, only=None):
    import spconv_native  # noqa: F401  (deterministic spconv algorithm choice)
    import torch

    from pointcept.models import build_model
    from pointcept.utils.config import Config

    quiet_logs()
    fdir = os.path.join(out, "forward")
    os.makedirs(fdir, exist_ok=True)
    path = os.path.join(out, "forward.jsonl")
    done = read_jsonl(path)
    ds_cache = {}

    jobs = [("multitask", n, v) for n, v in cases.MULTITASK.items()]
    jobs += [("probe", n, p) for n, p in cases.probe_cases().items()]
    for kind, name, spec in jobs:
        if (only and not re.search(only, name)) or any(k.startswith(name + "@") or k == name for k in done):
            continue
        t0 = time.time()
        try:
            if kind == "multitask":
                cfg_path, ck = spec
                cfg = Config.fromfile(cfg_path)
                seed_all(0)
                model = build_model(cfg.model).cuda().eval()
                load = load_ckpt(model, cases.CKPT[ck])
                for tile, split in cases.FLAIR3D_TILES:
                    inp, hashes, _ = make_input(flair_dcfg(cfg, split), tile, ds_cache)
                    with torch.no_grad():
                        flat = flatten_out(model(inp))
                    key = f"{name}@{tile}"
                    np.savez_compressed(os.path.join(fdir, key + ".npz"), **flat)
                    append_jsonl(path, dict(key=key, config=cfg_path, ckpt=cases.CKPT[ck], load=load,
                                            inputs=hashes, outputs=sorted(flat)))
            else:
                cfg = Config.fromfile(spec)
                ckpt = (cases.CKPT[cases.PROBE_CKPT_OVERRIDE[name]] if name in cases.PROBE_CKPT_OVERRIDE
                        else resolve_probe_ckpt(cfg.get("weight")))
                if ckpt is None:
                    append_jsonl(path, dict(key=name, config=spec, skipped=f"no local ckpt for {cfg.get('weight')}"))
                    continue
                loader = next((h for h in cfg.hooks if h["type"] == "CheckpointLoader"), {})
                seed_all(0)
                model = build_model(cfg.model).cuda().eval()
                load = load_ckpt(model, ckpt, loader.get("keywords", ""), loader.get("replacement"),
                                 tuple(loader.get("exclude_keys") or ()))
                load["missing"] = [k for k in load["missing"] if k.startswith("backbone.")]
                if cfg.data.val.get("type") == "Flair3DDataset":
                    dcfg, pick = flair_dcfg(cfg, "val"), cases.FLAIR3D_TILES[0][0]
                else:
                    dcfg, pick = dict(cfg.data.val), 0
                inp, hashes, sample = make_input(dcfg, pick, ds_cache)
                with torch.no_grad():
                    feat_by_norm = model.prepare_batch(inp)[0]
                flat = {}
                rng = np.random.default_rng(0)
                for norm, f in feat_by_norm.items():
                    f = f.float()
                    flat[f"{norm}/mean"] = f.mean(0).cpu().numpy()
                    flat[f"{norm}/std"] = f.std(0).cpu().numpy() if f.shape[0] > 1 else f[0].cpu().numpy()
                    rows = np.sort(rng.choice(f.shape[0], min(FEAT_ROWS, f.shape[0]), replace=False))
                    flat[f"{norm}/rows"] = f[torch.from_numpy(rows).cuda()].cpu().numpy()
                np.savez_compressed(os.path.join(fdir, name.replace("/", "__") + ".npz"), **flat)
                append_jsonl(path, dict(key=name, config=spec, ckpt=ckpt, sample=sample, load=load,
                                        inputs=hashes, outputs=sorted(flat)))
            print(f"[forward] {name} {time.time() - t0:.1f}s", flush=True)
        except LookupError as e:
            append_jsonl(path, dict(key=name, skipped=str(e)))
            print(f"[forward] {name} skipped: {e}", flush=True)
        except Exception as e:
            append_jsonl(path, dict(key=name, error=f"{type(e).__name__}: {e}"[:800],
                                    tb=traceback.format_exc()[-2000:]))
            print(f"[forward] {name} ERROR {type(e).__name__}: {e}", flush=True)
        finally:
            model = None
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- stage train
def read_scalars(logdir):
    """All TensorBoard scalars written by the run: {tag: [[step, value], ...]}."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    return {t: [[e.step, e.value] for e in ea.Scalars(t)] for t in ea.Tags()["scalars"]}


def eval_lines(text):
    return [re.sub(r"^\[[^\]]*\]\s*", "", l).strip() for l in text.splitlines()
            if re.search(r"(Val|Test) (result|regression)|precision/recall/f1|macro_apls", l)]


def capture_train(out, only=None):
    py = sys.executable
    path = os.path.join(out, "train.jsonl")
    done = read_jsonl(path)
    for name, (cfg_path, opts) in cases.TRAIN.items():
        if name in done or (only and not re.search(only, name)):
            continue
        save = os.path.abspath(os.path.join(out, "train", name))
        os.makedirs(save, exist_ok=True)
        cmd = [py, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_entry.py"), "--config-file", cfg_path, "--num-gpus", "1",
               "--options", f"save_path={save}", *opts]
        t0 = time.time()
        with open(os.path.join(save, "stdout.log"), "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONPATH=os.getcwd())).returncode
        shutil.rmtree(os.path.join(save, "model"), ignore_errors=True)  # checkpoints: 0.2-1.5 GB per case
        scalars = read_scalars(save) if glob.glob(os.path.join(save, "events.out.tfevents*")) else {}
        evals = eval_lines(open(os.path.join(save, "stdout.log"), errors="replace").read())
        append_jsonl(path, dict(key=name, config=cfg_path, options=opts, returncode=rc,
                                scalars=scalars, evals=evals, seconds=round(time.time() - t0)))
        n_it = len(scalars.get("train_batch/loss", []))
        print(f"[train] {name} rc={rc} iters={n_it} {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- compare
def apply_map(s, pairs):
    for a, b in pairs:
        s = s.replace(a, b)
    return s


def compare(ref, new, mapping, rtol, atol):
    paths_map, str_map = mapping.get("paths", []), mapping.get("strings", [])
    fails, notes = [], []

    # build
    r, n = read_jsonl(os.path.join(ref, "build.jsonl")), read_jsonl(os.path.join(new, "build.jsonl"))
    if r and n:
        gone = []
        for p, rr in r.items():
            np_ = apply_map(p, paths_map)
            if np_ not in n:
                gone.append(p)
                continue
            nn = n[np_]
            rc = json.loads(apply_map(json.dumps(rr.get("cfg")), str_map))
            if rc != nn.get("cfg"):
                diff = [k for k in set(rc or {}) | set(nn.get("cfg") or {})
                        if (rc or {}).get(k) != (nn.get("cfg") or {}).get(k)]
                fails.append(f"build {np_}: config differs at keys {sorted(diff)[:10]}")
            for k in ("cfg_error", "model_error"):
                if rr.get(k) != nn.get(k) and nn.get(k):
                    fails.append(f"build {np_}: new {k}: {nn[k][:200]}")
            rm, nm = rr.get("model"), nn.get("model")
            if rm and nm:
                rt = {apply_map(k, str_map): v for k, v in rm["tensors"].items()}
                if rt != nm["tensors"]:
                    ks = [k for k in set(rt) | set(nm["tensors"]) if rt.get(k) != nm["tensors"].get(k)]
                    fails.append(f"build {np_}: model state differs ({len(ks)} tensors, e.g. {sorted(ks)[:3]})")
            for split, rd in (rr.get("data") or {}).items():
                nd = (nn.get("data") or {}).get(split)
                rd = json.loads(apply_map(json.dumps(rd), str_map))
                if rd != nd:
                    fails.append(f"build {np_}: dataset[{split}] {rd} -> {nd}")
        notes.append(f"build: {len(r)} ref configs, {len(gone)} not present in new (removed/unmapped)")
        if gone:
            with open(os.path.join(new, "removed_configs.txt"), "w") as f:
                f.write("\n".join(gone) + "\n")

    # forward (skipped when the new capture did not run this stage)
    r, n = read_jsonl(os.path.join(ref, "forward.jsonl")), read_jsonl(os.path.join(new, "forward.jsonl"))
    if not n:
        r = {}
    for key, rr in r.items():
        if "error" in rr or "skipped" in rr:
            continue
        nkey = apply_map(key, paths_map)
        nn = n.get(nkey)
        if nn is None and not os.path.isfile(apply_map(rr.get("config", ""), paths_map)):
            notes.append(f"forward {key}: config removed")
            continue
        if nn is None or "error" in nn:
            fails.append(f"forward {key}: " + ("missing" if nn is None else nn["error"][:200]))
            continue
        if json.dumps(rr["load"]) != json.dumps(nn["load"]):
            fails.append(f"forward {key}: checkpoint load report differs {nn['load']}")
        if rr["inputs"] != nn["inputs"]:
            ks = [k for k in set(rr["inputs"]) | set(nn["inputs"]) if rr["inputs"].get(k) != nn["inputs"].get(k)]
            fails.append(f"forward {key}: input tensors differ {sorted(ks)}")
        a = np.load(os.path.join(ref, "forward", key.replace("/", "__") + ".npz"))
        b = np.load(os.path.join(new, "forward", nkey.replace("/", "__") + ".npz"))
        worst = worst_rel = 0.0
        for k in a.files:
            if k not in b.files or a[k].shape != b[k].shape:
                fails.append(f"forward {key}: output {k} missing or reshaped")
                continue
            x, y = a[k], b[k]
            d = np.abs(x - y)
            rel = float(d.mean() / max(float(np.abs(x).mean()), 1e-12)) if d.size else 0.0
            worst, worst_rel = max(worst, float(d.max(initial=0))), max(worst_rel, rel)
            if float(np.max(d - (atol + rtol * np.abs(x)), initial=-np.inf)) <= 0:
                continue
            # LitePT forwards are not bit-reproducible on GPU (measured run-to-run: mean relative
            # diff <= 2e-4, argmax agreement >= 0.9998), so accept a small diffuse difference.
            is_logits = "logits" in k and x.ndim == 2 and x.shape[1] > 1
            agree = float(np.mean(x.argmax(1) == y.argmax(1))) if is_logits else 1.0
            if rel > FORWARD_MEAN_RTOL or agree < FORWARD_MIN_AGREEMENT:
                fails.append(f"forward {key}: {k} max|diff|={d.max():.3g} mean rel={rel:.2g} argmax agree={agree:.5f}")
        notes.append(f"forward {key}: max|diff|={worst:.3g} mean rel={worst_rel:.2g}")

    # train (skipped when the new capture did not run this stage)
    r, n = read_jsonl(os.path.join(ref, "train.jsonl")), read_jsonl(os.path.join(new, "train.jsonl"))
    if not n:
        r = {}
    for key, rr in r.items():
        nn = n.get(apply_map(key, paths_map))
        if nn is None and apply_map(key, paths_map) not in cases.TRAIN:
            notes.append(f"train {key}: case removed")
            continue
        if nn is None:
            fails.append(f"train {key}: missing")
            continue
        if nn["returncode"] != rr["returncode"]:
            fails.append(f"train {key}: returncode {rr['returncode']} -> {nn['returncode']}")
        worst, worst_tag = 0.0, None
        for tag, series in rr["scalars"].items():
            other = nn["scalars"].get(tag)
            if other is None or len(other) != len(series):
                fails.append(f"train {key}: scalar {tag} missing or different length")
                continue
            for (step, a), (_, b) in zip(series, other):
                d = abs(a - b) / max(abs(a), 1e-6)
                if d > worst:
                    worst, worst_tag = d, tag
                # GPU non-determinism (spconv atomics, AMP): only the first steps of the total train loss
                # are asserted; step 1 (init + data + loss) tightly, steps 2-10 loosely.
                tol = TRAIN_STEP1_RTOL if step == 1 else TRAIN_RTOL
                if tag == "train_batch/loss" and step <= TRAIN_STRICT_STEPS and d > tol:
                    fails.append(f"train {key}: {tag} step {step}: {a:.5g} -> {b:.5g}")
        notes.append(f"train {key}: max relative scalar diff {worst:.3g} ({worst_tag})")
        n_diff = sum(x != y for x, y in zip(rr["evals"], nn["evals"])) + abs(len(rr["evals"]) - len(nn["evals"]))
        if n_diff:
            # Val/test metrics after 40 noisy steps are informational only.
            notes.append(f"train {key}: {n_diff} eval line(s) differ (informational)")

    print("\n".join(notes))
    print(f"\n{len(fails)} failure(s)")
    print("\n".join(fails))
    return 1 if fails else 0


def accept(ref, new, keys):
    """Copy the given entries (config paths / case names) from `new` into the rolling reference `ref`.

    Only for intended, verified changes. Build entries of configs that no longer exist are dropped.
    """
    for stage in ("build", "forward", "train"):
        rpath, npath = os.path.join(ref, f"{stage}.jsonl"), os.path.join(new, f"{stage}.jsonl")
        if not os.path.isfile(npath):
            continue
        r, n = read_jsonl(rpath), read_jsonl(npath)
        for k in keys:
            if k in n:
                r[k] = n[k]
                if stage == "forward":
                    fn = k.replace("/", "__") + ".npz"
                    shutil.copy(os.path.join(new, "forward", fn), os.path.join(ref, "forward", fn))
                print(f"accepted {stage} {k}")
        if stage == "build":
            r = {k: v for k, v in r.items() if os.path.isfile(k)}
        with open(rpath, "w") as f:
            f.writelines(json.dumps(v) + "\n" for v in r.values())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--stage", choices=["build", "forward", "train", "all"], required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--only", default=None, help="regex on config path / case name")
    k = sub.add_parser("compare")
    k.add_argument("ref")
    k.add_argument("new")
    k.add_argument("--map", default=None)
    k.add_argument("--rtol", type=float, default=1e-3)
    k.add_argument("--atol", type=float, default=1e-4)
    a = sub.add_parser("accept", help="copy intended, verified changes from a capture into the rolling reference")
    a.add_argument("ref")
    a.add_argument("new")
    a.add_argument("keys", nargs="+")
    args = ap.parse_args()
    if args.cmd == "capture":
        os.makedirs(args.out, exist_ok=True)
        if args.stage == "all":
            # One process per stage: building every config leaves global CUDA/spconv state behind that shifts
            # PT-v3m1 forwards by ~1e-3 relative when the forward stage runs in the same process.
            for stage in ("build", "forward", "train"):
                cmd = [sys.executable, os.path.abspath(__file__), "capture", "--stage", stage, "--out", args.out]
                subprocess.run(cmd + (["--only", args.only] if args.only else []), check=True)
        else:
            dict(build=capture_build, forward=capture_forward, train=capture_train)[args.stage](args.out, args.only)
    elif args.cmd == "accept":
        accept(args.ref, args.new, args.keys)
    else:
        mapping = json.load(open(args.map)) if args.map else {}
        sys.exit(compare(args.ref, args.new, mapping, args.rtol, args.atol))


if __name__ == "__main__":
    main()
