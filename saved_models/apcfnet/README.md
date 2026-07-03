# APCF-Net checkpoints

This folder is intentionally prepared for final trained APCF-Net checkpoints.

Expected files after training:

- `best.pth`
- `last.pth`
- `config_snapshot.yaml`
- optional `epoch_XXXX.pth`

Run:

```bash
python scripts/train_raw.py --config configs/apcfnet_raw_uieb.yaml
```
