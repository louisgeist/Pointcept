# MALiBU3D validation subset (`val_subset_2000tiles.csv`)

The training configs validate on a fixed subset of 2,000 validation tiles, selected to be representative of
the label distribution of the whole validation split. Test metrics are always computed on the full test split.

The file ships with the dataset on Hugging Face as `manifests/val_subset_2000tiles.csv` (columns `split`,
`tile_id`). The configs expect it at `data/malibu3d/manifests/val_subset_2000tiles.csv`.

To validate on the whole validation split instead:

```bash
python tools/train.py --config-file <config> --options data.val.stratified_subset_manifest=False
```
