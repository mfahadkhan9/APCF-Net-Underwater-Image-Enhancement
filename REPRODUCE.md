# Reproducing APCF-Net experiments

## 1. Prepare the environment

```bash
conda create -n apcfnet python=3.10 -y
conda activate apcfnet
pip install -r requirements.txt
```

## 2. Prepare raw/non-paired data

Download raw underwater images from the dataset provider. For UIEB, use the official raw image download port. Because redistribution is prohibited by the UIEB dataset owner, raw images are not included in this repository.

Split the raw images:

```bash
python scripts/split_raw_images.py \
  --input_dir /path/to/raw_underwater_images \
  --output_root data/raw \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --copy
```

## 3. Train

```bash
python scripts/train_raw.py --config configs/apcfnet_raw_uieb.yaml
```

## 4. Inference

```bash
python scripts/infer.py \
  --config configs/apcfnet_raw_uieb.yaml \
  --checkpoint saved_models/apcfnet/best.pth \
  --input_dir data/raw/test \
  --output_dir results/enhanced
```

## 5. Evaluate

No-reference metrics:

```bash
python scripts/evaluate_no_reference.py \
  --input_dir results/enhanced \
  --output_csv results/metrics/no_reference_metrics.csv
```

Optional reference metrics:

```bash
python scripts/evaluate_reference.py \
  --enhanced_dir results/enhanced \
  --reference_dir data/reference_optional/test \
  --output_csv results/metrics/reference_metrics.csv
```

## 6. Archive for publication

Before publication, create a GitHub release and archive the release with Zenodo. Cite the Zenodo DOI in the manuscript's Code Availability Statement.
