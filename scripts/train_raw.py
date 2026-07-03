#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import argparse
import csv
import shutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from apcfnet.data.raw_dataset import RawUnderwaterDataset
from apcfnet.losses.zero_reference import ZeroReferenceLoss
from apcfnet.models.apcf_net import APCFNet
from apcfnet.utils.config import load_config, save_config
from apcfnet.utils.io import save_tensor_image
from apcfnet.utils.seed import set_seed


def build_model(cfg):
    m = cfg['model']
    return APCFNet(
        base_channels=int(m.get('base_channels', 32)),
        num_curve_segments=int(m.get('num_curve_segments', 8)),
        use_tca=bool(m.get('use_tca', True)),
        use_curve=bool(m.get('use_curve', True)),
    )


def main():
    parser = argparse.ArgumentParser(description='Train APCF-Net on raw/non-paired underwater images.')
    parser.add_argument('--config', required=True, help='Path to YAML configuration.')
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg['project'].get('seed', 42)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = cfg['paths']
    ckpt_dir = Path(paths['checkpoint_dir'])
    log_dir = Path(paths['log_dir'])
    fig_dir = Path(paths['figure_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, ckpt_dir / 'config_snapshot.yaml')

    data_cfg = cfg['data']
    train_ds = RawUnderwaterDataset(data_cfg['train_dir'], data_cfg['image_size'], data_cfg['extensions'], augment=True)
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=data_cfg.get('num_workers', 2), pin_memory=True)

    val_loader = None
    val_dir = Path(data_cfg.get('val_dir', ''))
    if val_dir.exists() and any(val_dir.iterdir()):
        val_ds = RawUnderwaterDataset(val_dir, data_cfg['image_size'], data_cfg['extensions'], augment=False)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['learning_rate']), weight_decay=float(cfg['training'].get('weight_decay', 0.0)))
    criterion = ZeroReferenceLoss(**cfg['loss']).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg['training'].get('mixed_precision', False)) and device.type == 'cuda')

    log_path = log_dir / 'train_log.csv'
    with log_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss_total', 'loss_spatial', 'loss_exposure', 'loss_color', 'loss_tv', 'loss_curve'])
        writer.writeheader()

    best_loss = float('inf')
    epochs = int(cfg['training']['epochs'])
    for epoch in range(1, epochs + 1):
        model.train()
        meter = {}
        n_batches = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
        for images, _ in pbar:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(images)
                loss, losses = criterion(images, outputs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            for k, v in losses.items():
                meter[k] = meter.get(k, 0.0) + float(v)
            n_batches += 1
            pbar.set_postfix({'loss': f"{losses['loss_total']:.4f}"})

        avg = {k: v / max(n_batches, 1) for k, v in meter.items()}
        avg['epoch'] = epoch
        with log_path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'loss_total', 'loss_spatial', 'loss_exposure', 'loss_color', 'loss_tv', 'loss_curve'])
            writer.writerow(avg)

        checkpoint = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'config': cfg, 'loss': avg['loss_total']}
        torch.save(checkpoint, ckpt_dir / 'last.pth')
        if avg['loss_total'] < best_loss:
            best_loss = avg['loss_total']
            shutil.copyfile(ckpt_dir / 'last.pth', ckpt_dir / 'best.pth')

        if epoch % int(cfg['training'].get('save_every', 10)) == 0:
            torch.save(checkpoint, ckpt_dir / f'epoch_{epoch:04d}.pth')

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for i, (img, name) in enumerate(val_loader):
                    if i >= 4:
                        break
                    out = model(img.to(device))['enhanced'][0]
                    save_tensor_image(out, fig_dir / f'validation_preview_epoch_{epoch:04d}_{name[0]}')

    print(f'Training complete. Best loss: {best_loss:.6f}')
    print(f'Checkpoints: {ckpt_dir}')


if __name__ == '__main__':
    main()
