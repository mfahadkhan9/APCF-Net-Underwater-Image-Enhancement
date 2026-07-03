# Data directory

This repository is configured for raw/non-paired underwater image enhancement.

Expected layout:

```text
data/
├── raw/
│   ├── train/        # raw underwater images for non-paired training
│   ├── val/          # raw underwater images for validation/preview
│   └── test/         # raw underwater images for inference/evaluation
└── reference_optional/
    └── test/         # optional reference images with identical filenames, only for PSNR/SSIM
```

No dataset is redistributed in this repository. Download each dataset from its original provider and copy the images into the relevant folders.
