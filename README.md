### Quick Test with Our Prepared Data

We provide the test manifests, text descriptions, and precomputed text caches for all supported datasets. After downloading the corresponding dataset and pretrained checkpoint, inference can be performed directly without regenerating text descriptions or embeddings.

Run the following command from the repository root:

```bash
python infer.py \
  --config <CONFIG> \
  --checkpoint <CHECKPOINT> \
  --manifest <TEST_MANIFEST> \
  --text-cache <TEXT_CACHE> \
  --output <OUTPUT_DIR> \
  --restore-color
```

The prepared files for each dataset are listed below:

| Dataset   | Config                | Test Manifest                    | Text Cache                    |
| --------- | --------------------- | -------------------------------- | ----------------------------- |
| MSRS      | `configs/vis_ir.yaml` | `data/msrs_test_text.jsonl`      | `data/msrs_test_text.pt`      |
| RoadScene | `configs/vis_ir.yaml` | `data/roadscene_test_text.jsonl` | `data/roadscene_test_text.pt` |
| M3FD      | `configs/vis_ir.yaml` | `data/m3fd_test_text.jsonl`      | `data/m3fd_test_text.pt`      |
| CT-MRI    | `configs/mif.yaml`    | `data/ct_mri_test_text.jsonl`    | `data/ct_mri_test_text.pt`    |
| PET-MRI   | `configs/mif.yaml`    | `data/pet_mri_test_text.jsonl`   | `data/pet_mri_test_text.pt`   |
| SPECT-MRI | `configs/mif.yaml`    | `data/spect_mri_test_text.jsonl` | `data/spect_mri_test_text.pt` |

For example, testing the MSRS model:

```bash
python infer.py \
  --config configs/vis_ir.yaml \
  --checkpoint pretrained_models/MSRS/best.pt \
  --manifest data/msrs_test_text.jsonl \
  --text-cache data/msrs_test_text.pt \
  --output results/MSRS \
  --restore-color
```

The fused images will be saved to the specified output directory. The `--restore-color` option preserves the chrominance information of the designated color source. It may be omitted when grayscale output is required.
