

Attention à bien donner les bonnes wheels pour les package torch-scatter and co :
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
```

Remarque: dans ce repo, y\'a que une manière de faire le preprocessing -> ou alors il faut changer manuellement le nom du folder où on save les tiles.

### S3DIS check
#### PP
Le fichier à fix manuellement: 
`/data/geist/datasets/s3dis/Stanford3dDataset_v1.2/Area_5/office_19/Annotations/ceiling_1.txt`

```bash
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py \
  --splits Area_1 Area_2 Area_3 Area_4 Area_5 Area_6 \
  --dataset_root /data/geist/datasets/s3dis/Stanford3dDataset_v1.2 \
  --output_root /data/geist/Pointcept/data/s3dis \
  --align_angle
```

#### PP on JZ
Copy the fixed file
```bash
scp -r -J passerelle lgeist@hecate:/data/geist/datasets/s3dis/Stanford3dDataset_v1.2/Area_5/office_19/Annotations/ceiling_1.txt usi32yh@jean-zay.idris.fr:/lustre/fsn1/projects/rech/unv/usi32yh/data/s3dis/Stanford3dDataset_v1.2/Area_5/office_19/Annotations/ceiling_1.txt
```


```bash
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py   --splits Area_1 Area_2 Area_3 Area_4 Area_5 Area_6   --dataset_root /lustre/fsn1/projects/rech/unv/usi32yh/data/s3dis/Stanford3dDataset_v1.2   --output_root data/s3dis   --align_angle
```

#### Train
train S3DIS without the normal features:
```bash
sh scripts/train.sh -g 1 -d s3dis -c ptv3_nonormal -n ptv3_nonormal
```

### Flair3D
- Preprocessing in debug mode (sample data, 1 worker):
```bash
sh scripts/preprocess_flair3d_debug.sh
# or with custom paths:
DATASET_ROOT=/path/to/data OUTPUT_ROOT=/path/to/out sh scripts/preprocess_flair3d_debug.sh
```

- Preprocessing of all splits on hecate:
```bash
python pointcept/datasets/preprocessing/flair3d/preprocess_flair3d.py \
 --dataset_root /data/geist/datasets/sample_flairhub_3d \
 --output_root /data/geist/Pointcept/data/flair3d \
 --split val test train \
 --mode subtile \
 --label_definition inter_finerall6
 --save_strength
 --num_workers 24
```

- with other label definition
```bash
python pointcept/datasets/preprocessing/flair3d/preprocess_flair3d.py \
 --dataset_root /data/geist/datasets/sample_flairhub_3d \
 --output_root /data/geist/Pointcept/data/flair3d_lab7 \
 --split val test train \
 --mode subtile \
 --label_definition inter_finerall7
 --save_strength
 --num_workers 10
```

- PP on JZ:
```bash
python pointcept/datasets/preprocessing/flair3d/preprocess_flair3d.py \
 --dataset_root /lustre/fsn1/projects/rech/unv/usi32yh/data/flair3d/FLAIR-HUB \
 --output_root $WORK/Pointcept/data/flair3d \
 --split val test train \
 --mode subtile \
 --label_definition inter_finerall6 \
 --save_strength \
 --num_workers 12
```

python pointcept/datasets/preprocessing/flair3d/preprocess_flair3d.py \
 --dataset_root /lustre/fsn1/projects/rech/unv/usi32yh/data/flair3d/FLAIR-HUB \
 --output_root $WORK/Pointcept/data/flair3d_lab7 \
 --split val test train \
 --mode subtile \
 --label_definition inter_finerall7 \
 --save_strength \
 --num_workers 12
#### Train Flair3D

On hecate:
```bash
sh scripts/train.sh -g 1 -d flair3d -c ptv3_nonormal_subtile -n ptv3_nonormal_subtile
```
ou regarder script debug mode

Or directly with Python (from repo root):

```bash
python tools/train.py --config-file configs/flair3d/ptv3_nonormal_subtile.py --num-gpus 1
```

Mini-dataset smoke test (10 epochs, eval every 10, 2 train + 2 val samples).

With the script (recommended; `EXTRA_OPTIONS` is passed as `--options` to the Python command):

```bash
export EXTRA_OPTIONS="epoch=10 eval_epoch=10 max_sample_train=2 max_sample_val=2"
sh scripts/train.sh -g 1 -d flair3d -c ptv3_nonormal_subtile -n ptv3_nonormal_subtile
```

Train directement une config dans experiment:
```bash
cd /lustre/fswork/projects/rech/unv/usi32yh/Pointcept
python -m tools.train \
  --config-file configs/experiment/w88/1/check_scheduler/1_ptv3_harmonized-transforms.py \
  --num-gpus 1 \
  --num-machines 1 \
  --machine-rank 0 \
  --dist-url auto \
  --options save_path=logs/local/test
```


# Brouillon
python -m tools.train \
  --config-file configs/experiment/w88/5/kpconv_debug/6.py \
  --num-gpus 1 \
  --num-machines 1 \
  --machine-rank 0 \
  --dist-url auto \
  --options batch_size=2