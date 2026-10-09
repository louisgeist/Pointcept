"""Upload ``ckpt_cleaned/`` to a **private** Hugging Face model repo, then verify the round trip.

Prerequisites: run ``scripts/hf_release/prepare_release.py`` first, and log in once yourself with
``hf auth login`` (a token with write access to the account; never paste it anywhere).

    python scripts/hf_release/upload.py --dry-run          # list what would be sent, check SHA256SUMS
    python scripts/hf_release/upload.py                    # create private repo + upload + verify
    python scripts/hf_release/upload.py --verify-only      # re-download and re-check hashes only

The repo is always created private; make it public later in the repo settings. ``_local/`` (source paths and
hashes of the original training checkpoints) is never uploaded. Remote files that no longer exist locally
(renamed / removed models, old ``model_config.py``) are deleted in the same commit, so the Hub mirrors
``ckpt_cleaned/`` (``.gitattributes`` is kept); ``--keep-stale`` disables that. Unchanged files are not re-sent.
"""

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = Path(__file__).resolve().parents[2]
IGNORE = ["_local/*", "_local/**", ".git*", "__pycache__/*"]


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_sums(folder):
    sums = {}
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        sums[rel] = digest
    return sums


def check_local(folder, sums):
    bad = [rel for rel, d in sums.items() if not (folder / rel).is_file() or sha256(folder / rel) != d]
    if bad:
        sys.exit(f"local files do not match SHA256SUMS (re-run prepare_release.py): {bad}")
    print(f"[ok] {len(sums)} local files match SHA256SUMS")


def verify_remote(api, repo_id, sums):
    with tempfile.TemporaryDirectory(prefix="hf_verify_") as tmp:
        for rel, digest in sums.items():
            p = hf_hub_download(repo_id, rel, repo_type="model", cache_dir=tmp, token=api.token)
            if sha256(p) != digest:
                sys.exit(f"REMOTE HASH MISMATCH: {rel}")
            print(f"  [ok] {rel}")
    print("[ok] remote files match SHA256SUMS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="LouisGeist/MALiBU3D-backbones")
    ap.add_argument("--dir", default="ckpt_cleaned")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--keep-stale", action="store_true", help="do not delete remote files absent locally")
    args = ap.parse_args()

    folder = REPO / args.dir
    if not (folder / "SHA256SUMS").is_file():
        sys.exit(f"{folder}/SHA256SUMS missing: run scripts/hf_release/prepare_release.py first")
    sums = read_sums(folder)
    check_local(folder, sums)

    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception:
        if args.dry_run:
            who = "<not logged in>"
        else:
            sys.exit("Not logged in to Hugging Face: run `hf auth login` in your own terminal, then retry.")
    print(f"Hugging Face user: {who}  ->  repo: {args.repo_id} (private)")

    if args.verify_only:
        verify_remote(api, args.repo_id, sums)
        return

    files = sorted(p for p in folder.rglob("*") if p.is_file() and "_local" not in p.relative_to(folder).parts)
    total = sum(p.stat().st_size for p in files)
    for p in files:
        print(f"  {p.relative_to(folder).as_posix():45s} {p.stat().st_size / 1e6:8.1f} MB")
    print(f"  {'total':45s} {total / 1e6:8.1f} MB  ({len(files)} files)")

    stale = []
    if not args.keep_stale:
        try:
            local = {p.relative_to(folder).as_posix() for p in files}
            stale = sorted(f for f in api.list_repo_files(args.repo_id, repo_type="model")
                           if f not in local and f != ".gitattributes")
        except Exception:
            pass  # repo does not exist yet (or not logged in during a dry-run): nothing to delete
    for f in stale:
        print(f"  DELETE (remote only) {f}")
    if args.dry_run:
        print("dry-run: nothing created, nothing uploaded, nothing deleted")
        return

    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder), repo_id=args.repo_id, repo_type="model",
        ignore_patterns=IGNORE, delete_patterns=stale or None,
        commit_message="Update MALiBU3D backbones: sonata-pretrained, configs from the repo, new card",
    )
    print("[ok] uploaded; verifying by re-download ...")
    verify_remote(api, args.repo_id, sums)
    print(f"Done. Private repo: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
