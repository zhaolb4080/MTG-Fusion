### Quick Test with Our Prepared Data

The pretrained checkpoints are under `checkpoints/`, and the prepared test manifests, text descriptions, and text caches are under `data/`. The source images are not included because of their file sizes; therefore, `datasets/` is empty by default.

#### Download the Datasets

Download our prepared training and test splits:

* [Download the prepared datasets](https://pan.baidu.com/s/1TUk6dKbdgTJcDMZ2keTSrg?pwd=7nmd)

Extract the downloaded package into the repository root. The expected structure is:

```text
datasets/
├── MSRS/
│   ├── train/{ir,vi}
│   └── test/{ir,vi}
├── RoadScene/
│   ├── train/{ir,vi}
│   └── test/{ir,vi}
├── M3FD/
│   ├── train/{Ir,Vis}
│   └── test/{Ir,Vis}
├── CT-MRI/
│   ├── train/{CT,MRI}
│   └── test/{CT,MRI}
├── PET-MRI/
│   ├── train/{PET,MRI}
│   └── test/{PET,MRI}
└── SPECT-MRI/
    ├── train/{SPECT,MRI}
    └── test/{SPECT,MRI}
```

The pretrained checkpoints are organized as:

```text
checkpoints/
├── msrs.pt
├── roadscene.pt
├── m3fd.pt
├── ct_mri.pt
├── pet_mri.pt
└── spect_mri.pt
```

Folder names are case-sensitive and should not be changed because the prepared manifests use relative paths rooted at `datasets/`.

#### Run Inference

For example, test MTG-Fusion on MSRS:

```bash
python infer.py \
  --config configs/vis_ir.yaml \
  --checkpoint checkpoints/msrs.pt \
  --manifest data/msrs_test_text.jsonl \
  --text-cache data/msrs_test_text.pt \
  --output results/MSRS \
  --restore-color
```

Use the corresponding files for the other datasets:

| Dataset   | Config                | Checkpoint                 | Test Manifest                    | Text Cache                    |
| --------- | --------------------- | -------------------------- | -------------------------------- | ----------------------------- |
| MSRS      | `configs/vis_ir.yaml` | `checkpoints/msrs.pt`      | `data/msrs_test_text.jsonl`      | `data/msrs_test_text.pt`      |
| RoadScene | `configs/vis_ir.yaml` | `checkpoints/roadscene.pt` | `data/roadscene_test_text.jsonl` | `data/roadscene_test_text.pt` |
| M3FD      | `configs/vis_ir.yaml` | `checkpoints/m3fd.pt`      | `data/m3fd_test_text.jsonl`      | `data/m3fd_test_text.pt`      |
| CT-MRI    | `configs/mif.yaml`    | `checkpoints/ct_mri.pt`    | `data/ct_mri_test_text.jsonl`    | `data/ct_mri_test_text.pt`    |
| PET-MRI   | `configs/mif.yaml`    | `checkpoints/pet_mri.pt`   | `data/pet_mri_test_text.jsonl`   | `data/pet_mri_test_text.pt`   |
| SPECT-MRI | `configs/mif.yaml`    | `checkpoints/spect_mri.pt` | `data/spect_mri_test_text.jsonl` | `data/spect_mri_test_text.pt` |

Use `--restore-color` for MSRS, RoadScene, M3FD, PET-MRI, and SPECT-MRI. It can be omitted for grayscale CT-MRI fusion.

### Test on Your Own Data

To test your own data, organize the image pairs, build a manifest, generate descriptions, cache the text embeddings, and run inference. We generated our prepared descriptions using the ChatGPT-4.0 API. For your convenience, a local Qwen model is recommended without API credentials.

#### 1. Organize the Image Pairs

```text
datasets/custom/
├── source_a/
│   ├── 00001.png
│   └── ...
└── source_b/
    ├── 00001.png
    └── ...
```

#### 2. Build the Manifest

```bash
python tools/build_manifest.py \
  --ir-dir datasets/custom/source_a \
  --vis-dir datasets/custom/source_b \
  --color-source b \
  --output data/custom_test.jsonl
```

Here, `--ir-dir` and `--vis-dir` represent source A and source B. Use color source `b` for infrared–visible fusion and `a` for PET–MRI or SPECT–MRI. Add `--recursive` for nested directories.

#### 3. Generate and Cache Text

```bash
python tools/generate_descriptions.py \
  --input data/custom_test.jsonl \
  --output data/custom_test_text.jsonl \
  --backend qwen-local \
  --task vis-ir \
  --qwen-model Qwen/Qwen2.5-VL-3B-Instruct \
  --device cuda \
  --batch-size 1 \
  --resume
```

```bash
python tools/cache_text_embeddings.py \
  --manifest data/custom_test_text.jsonl \
  --output data/custom_test_text.pt \
  --device cuda
```

#### 4. Run Inference

Example for infrared–visible fusion:

```bash
python infer.py \
  --config configs/vis_ir.yaml \
  --checkpoint checkpoints/msrs.pt \
  --manifest data/custom_test_text.jsonl \
  --text-cache data/custom_test_text.pt \
  --output results/custom \
  --restore-color
```

