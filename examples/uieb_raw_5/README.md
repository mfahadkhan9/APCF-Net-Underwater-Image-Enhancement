# UIEB raw sample images

This folder is intended for five local UIEB raw sample images used only for quick qualitative inspection.

The UIEB project page states that redistribution of the UIEB dataset is forbidden. Therefore, the repository does not include UIEB image files by default. After downloading UIEB raw images from the official project page, create the local sample folder using:

```bash
python scripts/prepare_uieb_samples.py \
  --uieb_raw_dir /path/to/UIEB/raw-890 \
  --output_dir examples/uieb_raw_5 \
  --num_images 5
```

Before uploading these sample images to a public repository, verify that you have explicit permission from the dataset owner.
