"""Build slim, Hugging Face-ready checkpoints under ``ckpt_cleaned/``.

Training checkpoints carry optimizer / scheduler / scaler / hook state (~2/3 of the file) and, for Sonata,
the teacher + mask/unmask heads. A release keeps only what inference and frozen-backbone probing need:

* ``multitask`` entries (LitePT-B / PTv3 / SpUNet / KPConvX / Sonata-FT): the full ``state_dict`` of
  ``MultiTaskSegmentorV2`` (backbone + task heads + mask values), ``module.`` prefix stripped.
* ``ssl`` entries (Sonata pretrained SSL): ``module.student.backbone.*`` only, renamed ``backbone.*`` (this
  is exactly what the GridProbe configs load; ``CheckpointLoader(keywords="module.student.backbone")``
  simply no longer matches and the keys load as-is).

Output (``<out>/<name>/``): ``model.pth`` (tensors + plain numbers only, ``torch.load(weights_only=True)``
safe), ``meta.json``. No config is shipped: the model is rebuilt from the default config in this repo
(``cfg`` of each ENTRY, all under ``configs/flair3d_default/``).
Also ``<out>/SHA256SUMS`` and ``<out>/_local/provenance.json`` (source paths + hashes; NOT for upload).

Every output is verified: tensor-by-tensor equality with the source, then (unless --no-strict-load) a
``strict=True`` ``load_state_dict`` into the model rebuilt from that repo config.

    export PYTHONPATH=$PWD
    CUDA_VISIBLE_DEVICES=0 python scripts/hf_release/prepare_release.py            # all
    CUDA_VISIBLE_DEVICES=0 python scripts/hf_release/prepare_release.py --only ptv3 --dry-run
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from pointcept.models import build_model
from pointcept.utils.config import Config

REPO = Path(__file__).resolve().parents[2]

# name -> source ckpt, config, kind. Sources under ckpt/malibu3d/ are byte-identical to the original
# Jean Zay job checkpoints (sha256 checked), so they are used as the canonical copies.
ENTRIES = [
    dict(name="litept-b", wandb="louisgeist-ENPC/flair3d_multi/uzhxltcv", kind="multitask",
         src="ckpt/malibu3d/litept_b_multitask/model_best.pth",
         cfg="configs/flair3d_default/multi-litept-b-v1m0-flair3d.py"),
    dict(name="ptv3", wandb="louisgeist-ENPC/flair3d_multi/wznbvbgm", kind="multitask",
         src="ckpt/malibu3d/ptv3_multitask/model_best.pth",
         cfg="configs/flair3d_default/multi-ptv3-v1m0-flair3d.py"),
    dict(name="spunet", wandb="louisgeist-ENPC/flair3d_multi/8t6q5nnv", kind="multitask",
         src="ckpt/malibu3d/spunet_multitask/model_best.pth",
         cfg="configs/flair3d_default/multi-spunet-v1m0-flair3d.py"),
    dict(name="kpconvx", wandb="louisgeist-ENPC/flair3d_multi/u3j665te", kind="multitask",
         src="ckpt/malibu3d/kpconvx_multitask/model_best.pth",
         cfg="configs/flair3d_default/multi-kpconvx-v1m0-flair3d.py"),
    dict(name="sonata-pretrained", wandb="louisgeist-ENPC/flair3d_sonata/4fc00tcw", kind="ssl",
         src="ckpt/malibu3d/sonata_outdoor/epoch_120.pth",
         cfg="configs/flair3d_default/probe/sonata-v1m2-flair3d-lin-grid.py"),
    # Multitask fine-tuning of sonata-pretrained (JZ job 2126589).
    dict(name="sonata-ft", wandb="louisgeist-ENPC/flair3d_multi/era2p5gs", kind="multitask",
         src="ckpt/2126589/model_best.pth",
         cfg="configs/flair3d_default/multi-sonata-ft-v1m0-flair3d.py"),
]

SSL_PREFIX = "module.student.backbone."


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def to_plain(x):
    if isinstance(x, dict):
        return {k: to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_plain(v) for v in x)
    return x


def strip_module(k):
    return k[len("module."):] if k.startswith("module.") else k


def clean_state_dict(entry, ckpt):
    sd = ckpt["state_dict"]
    if entry["kind"] == "ssl":
        out = {}
        for k, v in sd.items():
            k = k if k.startswith("module.") else "module." + k
            if k.startswith(SSL_PREFIX):
                out["backbone." + k[len(SSL_PREFIX):]] = v
        if not out:
            raise RuntimeError(f"{entry['name']}: no '{SSL_PREFIX}*' keys in checkpoint")
    else:
        out = {strip_module(k): v for k, v in sd.items()}
    # detach + clone: drop any shared storage / autograd metadata so the file holds exactly the tensors
    return {k: v.detach().clone().contiguous() for k, v in out.items()}


def model_cfg(entry):
    """Return the model dict (``model`` or SSL ``backbone``) of the repo's default config, to rebuild it."""
    cfg = Config.fromfile(str(REPO / entry["cfg"]))
    return to_plain(cfg.model.backbone if entry["kind"] == "ssl" else cfg.model)


def verify_equal(entry, src_ckpt, cleaned_path):
    new = torch.load(cleaned_path, map_location="cpu", weights_only=True)
    expected = clean_state_dict(entry, src_ckpt)
    got = new["state_dict"]
    assert set(got) == set(expected), (
        f"key mismatch: only-cleaned={sorted(set(got) - set(expected))[:3]} "
        f"only-source={sorted(set(expected) - set(got))[:3]}")
    for k, v in expected.items():
        assert got[k].dtype == v.dtype and torch.equal(got[k], v), f"tensor differs: {k}"
    assert not any(k.startswith("module.") for k in got), "module. prefix left in keys"
    return new


def verify_strict_load(entry, block, state_dict):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = build_model(block).to(device)
    if entry["kind"] == "ssl":
        sd = {k[len("backbone."):]: v for k, v in state_dict.items()}
        res = model.load_state_dict(sd, strict=False)
        # embedding.mask_token only exists in the pretraining model (masked SSL); kept in the release
        # (like Meta's official Sonata ckpt) so pretraining can be resumed, ignored by probe models.
        assert not res.missing_keys, f"missing keys: {res.missing_keys[:5]}"
        assert set(res.unexpected_keys) <= {"embedding.mask_token"}, res.unexpected_keys
    else:
        model.load_state_dict(state_dict, strict=True)  # raises on missing / unexpected / shape mismatch
    return device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="ckpt_cleaned")
    ap.add_argument("--only", nargs="*", help="entry names to (re)build; default all")
    ap.add_argument("--dry-run", action="store_true", help="report sizes / keys, write nothing")
    ap.add_argument("--no-strict-load", action="store_true", help="skip the model rebuild + strict load")
    args = ap.parse_args()

    out_root = REPO / args.out
    entries = [e for e in ENTRIES if not args.only or e["name"] in args.only]
    unknown = set(args.only or []) - {e["name"] for e in ENTRIES}
    if unknown:
        sys.exit(f"unknown entries: {sorted(unknown)} (known: {[e['name'] for e in ENTRIES]})")

    provenance = {}
    for e in entries:
        src = REPO / e["src"]
        print(f"\n=== {e['name']} ({e['kind']}) <- {e['src']}")
        src_ckpt = torch.load(src, map_location="cpu", weights_only=False)
        sd = clean_state_dict(e, src_ckpt)
        n_params = sum(v.numel() for v in sd.values())
        block = model_cfg(e)
        dtypes = sorted({str(v.dtype) for v in sd.values()})
        print(f"  tensors={len(sd)} params={n_params / 1e6:.2f}M dtypes={dtypes} "
              f"epoch={src_ckpt.get('epoch')} best_metric_value={src_ckpt.get('best_metric_value')}")
        if args.dry_run:
            continue

        dst = out_root / e["name"]
        dst.mkdir(parents=True, exist_ok=True)
        payload = {"state_dict": sd, "epoch": int(src_ckpt["epoch"])}
        if e["kind"] == "multitask":
            payload["best_metric_value"] = float(src_ckpt["best_metric_value"])
        torch.save(payload, dst / "model.pth")

        verify_equal(e, src_ckpt, dst / "model.pth")
        print("  [ok] tensor-by-tensor equal to source, weights_only=True loadable")
        if not args.no_strict_load:
            dev = verify_strict_load(e, block, torch.load(dst / "model.pth", weights_only=True)["state_dict"])
            print(f"  [ok] strict=True load into rebuilt model ({dev})")

        meta = dict(
            name=e["name"], kind=e["kind"],
            backbone_type=block["backbone"]["type"] if e["kind"] == "multitask" else block["type"],
            n_params=n_params, n_tensors=len(sd), epoch=payload["epoch"],
            size_bytes=(dst / "model.pth").stat().st_size,
        )
        if e["kind"] == "multitask":
            meta["best_val_mIoU_segment"] = payload["best_metric_value"]
            meta["tasks"] = list(block["task_configs"])
        (dst / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        provenance[e["name"]] = dict(src=e["src"], src_sha256=sha256(src), src_size=src.stat().st_size,
                                     cfg=e["cfg"], wandb=e["wandb"])

    if args.dry_run:
        return
    sums = []
    for p in sorted(out_root.glob("*/*")):
        if p.is_file() and p.parent.name != "_local":
            sums.append(f"{sha256(p)}  {p.relative_to(out_root).as_posix()}")
    (out_root / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    prov_path = out_root / "_local" / "provenance.json"
    prov_path.parent.mkdir(exist_ok=True)
    old = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    old.update(provenance)
    prov_path.write_text(json.dumps(old, indent=2) + "\n")
    print(f"\nDone -> {out_root}  ({len(sums)} files hashed in SHA256SUMS)")


if __name__ == "__main__":
    main()
