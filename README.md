# APCF-Net: Adaptive Piecewise Curve and Multi-Color Attention Fusion Network

Official implementation package for the manuscript:

**APCF-Net: Adaptive Piecewise Curve and Multi-Color Attention Fusion Network for Underwater Image Restoration and Enhancement**

This repository is prepared for journal submission to **The Visual Computer**. It provides the code, configuration files, directory layout, sample paper images, and reproducibility instructions needed to train, infer, and evaluate APCF-Net.

> **Important dataset note.** This repository is configured for the authors' raw/non-paired underwater image setting. It does not redistribute the full UIEB, UCCS/RUIE, or NYU-v2 datasets. Download the datasets from their original providers and place the images in the directories described below.

---

## 1. Repository contents

```text
APCF-Net-VisualComputer/
├── configs/
│   ├── apcfnet_raw_uieb.yaml          # main raw/non-paired training configuration
│   └── apcfnet_paired_optional.yaml   # optional paired/reference training configuration
├── data/
│   ├── raw/
│   │   ├── train/                     # raw underwater training images
│   │   ├── val/                       # raw validation images for preview
│   │   └── test/                      # raw test images for inference
│   ├── paired/train/                  # optional paired input/reference folders
│   └── reference_optional/test/        # optional references for PSNR/SSIM only
├── examples/
│   └── paper_samples/
│       ├── input/                     # five small input samples extracted from paper figures
│       ├── enhanced/                  # corresponding enhanced samples shown in paper figures
│       └── sample_pairs.csv
├── saved_models/
│   └── apcfnet/                       # best.pth and last.pth are saved here after training
├── results/
│   ├── enhanced/                      # inference outputs
│   ├── metrics/                       # CSV metric files
│   ├── figures/                       # preview/qualitative figures
│   └── paper_samples/enhanced/         # paper sample enhanced images
├── logs/                              # training logs
├── scripts/                           # training, inference, evaluation, dataset utilities
├── src/apcfnet/                       # APCF-Net source package
├── tests/                             # smoke tests

```

---

## 2. Method overview

APCF-Net contains the following computational modules:

- RGB, HSV, and CIELab color-space processing
- Modified U-Net multi-scale restoration backbone
- MIRNet-inspired residual attention refinement
- MobileNetV2-style inverted residual bottleneck branch
- Triple Channel-Wise Attention fusion
- Adaptive piecewise curve-guided enhancement
- Raw/non-paired zero-reference training objective
- Optional paired multi-color-space perceptual loss when valid references are available

The default training script uses only raw underwater images and does not require paired reference targets.

---

## 3. Installation

A CUDA-enabled GPU is recommended for full training.

```bash
conda create -n apcfnet python=3.10 -y
conda activate apcfnet
pip install -r requirements.txt
```

Check installation:

```bash
python -m pytest tests
```

---

## 4. Dataset download links

Download the datasets from their original providers:

### UIEB

Official project page:

```text
https://li-chongyi.github.io/proj_benchmark.html
```

Use the raw underwater images for the non-paired setting. If you use references for PSNR/SSIM, place them in `data/reference_optional/test/` with matching file names.

### UCCS / RUIE benchmark

RUIE benchmark repository:

```text
https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark
```

The UCCS subset can be used as raw underwater images for no-reference testing and qualitative comparison.

### NYU Depth V2

Official NYU Depth V2 page:

```text
https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
```

Use the RGB images according to the experimental protocol described in the manuscript.

See also `DATASETS.md` for additional preparation notes.

---

## 5. Preparing raw/non-paired data

Place raw underwater images as follows:

```text
data/raw/train/*.jpg
data/raw/val/*.jpg
data/raw/test/*.jpg
```

You can automatically split a raw image folder:

```bash
python scripts/split_raw_images.py \
  --input_dir /path/to/raw_underwater_images \
  --output_root data/raw \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --copy
```

To copy five UIEB examples locally after you download UIEB:

```bash
python scripts/prepare_uieb_samples.py \
  --uieb_raw_dir /path/to/UIEB/raw-890 \
  --output_dir examples/uieb_raw_5 \
  --num_images 5
```

---

## 6. Paper sample images

Five small sample input/enhanced pairs are included under:

```text
examples/paper_samples/input/
examples/paper_samples/enhanced/
results/paper_samples/enhanced/
results/figures/paper_sample_pairs.png
```

These files are extracted from the qualitative figures in the manuscript for visual verification of the repository structure. They are not intended to replace the full benchmark datasets.

---

## 7. Raw/non-paired training

Train APCF-Net using the default raw-image configuration:

```bash
python scripts/train_raw.py --config configs/apcfnet_raw_uieb.yaml
```

Outputs are written to:

```text
saved_models/apcfnet/best.pth
saved_models/apcfnet/last.pth
saved_models/apcfnet/config_snapshot.yaml
logs/train_log.csv
results/figures/validation_preview_epoch_*.png
```

---

## 8. Inference

Run inference using a trained checkpoint:

```bash
python scripts/infer.py \
  --config configs/apcfnet_raw_uieb.yaml \
  --checkpoint saved_models/apcfnet/best.pth \
  --input_dir data/raw/test \
  --output_dir results/enhanced
```

For a quick structural check using the paper samples, copy the paper input samples into `data/raw/test/` or set `--input_dir examples/paper_samples/input`.

---

## 9. No-reference evaluation

For raw/non-paired enhancement results:

```bash
python scripts/evaluate_no_reference.py \
  --input_dir results/enhanced \
  --output_csv results/metrics/no_reference_metrics.csv
```

Implemented no-reference measures include AG, CCF, UCIQE, UIQM, UICM, UISM, UIConM, entropy, and Laplacian sharpness.

---

## 10. Optional reference-based evaluation

Use this only when valid reference images are available with identical file names:

```bash
python scripts/evaluate_reference.py \
  --enhanced_dir results/enhanced \
  --reference_dir data/reference_optional/test \
  --output_csv results/metrics/reference_metrics.csv
```

This computes PSNR and SSIM. The repository intentionally does not include random or placeholder PCQI values. If PCQI is reported in the manuscript, use a validated PCQI implementation and document it in the repository.

---

## 11. Optional paired training

If a paired/reference subset is used:

```text
data/paired/train/input/*.jpg
data/paired/train/reference/*.jpg
```

Run:

```bash
python scripts/train_paired.py --config configs/apcfnet_paired_optional.yaml
```

This mode uses the multi-color-space and perceptual objective:

```text
L_total = lambda_RGB L_RGB + lambda_HSV L_HSV + lambda_Lab L_Lab + lambda_Per L_Perceptual
```

For the authors' raw/non-paired setting, use `scripts/train_raw.py` instead.

---

## 12. Reproducibility checklist

1. Use the YAML configuration file corresponding to the reported experiment.
2. Save `config_snapshot.yaml` with each checkpoint.
3. Record the exact dataset split and image list used for each table.
4. Use no-reference metrics for raw-only datasets.
5. Use PSNR/SSIM only where matched references exist.
6. Do not report placeholder metrics.
7. Archive the GitHub release on Zenodo and cite the Zenodo DOI in the Code Availability Statement.

---

## 13. Suggested The Visual Computer statements

Use the templates in:

```text
journal_materials/DataAvailabilityStatement.md
journal_materials/CodeAvailabilityStatement.md
journal_materials/OnlineResourceCaption.md
```

---

## 14. License

Code is released under the MIT License. Dataset licenses and redistribution conditions remain governed by the original dataset owners.

---

## 15. Citation

After acceptance, update `CITATION.cff` with the final article DOI. If archived through Zenodo, cite the generated Zenodo DOI in the manuscript and repository.
