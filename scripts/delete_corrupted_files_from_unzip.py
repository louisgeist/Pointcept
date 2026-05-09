from pathlib import Path
import ast
import re
import shutil

dataset_file = Path("pointcept/datasets/flair3d.py")
unzip_root = Path("data/flair3d_plus/raw/unzip")
dry_run = True  # Set to False to actually delete

text = dataset_file.read_text(encoding="utf-8")

# Extract CORRUPTED_TILES set from flair3d.py
match = re.search(r"CORRUPTED_TILES\s*=\s*\{(.*?)\n\s*\}", text, re.S)
if not match:
    raise RuntimeError("Could not find CORRUPTED_TILES in flair3d.py")

tiles = ast.literal_eval("{" + match.group(1) + "}")
if len(tiles) != 22:
    raise RuntimeError(f"Expected 22 tiles, found {len(tiles)}. Aborting.")

to_delete = []
for split, tile in sorted(tiles):
    # Try both common layouts
    candidates = [
        unzip_root / split / tile,  # e.g. .../unzip/train/<tile>
        unzip_root / tile,          # e.g. .../unzip/<tile>
    ]
    for path in candidates:
        if path.exists():
            to_delete.append(path)

if not to_delete:
    print("No matching unzip directories found.")
    raise SystemExit(0)

print("Directories targeted:")
for path in to_delete:
    print(f" - {path}")

if dry_run:
    print("\nDry-run only. Set dry_run = False to delete.")
else:
    for path in to_delete:
        shutil.rmtree(path)
    print(f"\nDeleted {len(to_delete)} directories.")
