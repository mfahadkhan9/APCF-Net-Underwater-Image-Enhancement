#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from apcfnet.data.raw_dataset import RawUnderwaterDataset
from apcfnet.models.apcf_net import APCFNet
from apcfnet.utils.config import load_config
from apcfnet.utils.io import save_tensor_image


def build_model(cfg):
    m = cfg['model']
    return APCFNet(
        base_channels=int(m.get('base_channels', 32)),
        num_curve_segments=int(m.get('num_curve_segments', 8)),
        use_tca=bool(m.get('use_tca', True)),
        use_curve=bool(m.get('use_curve', True)),
    )


def main():
    parser = argparse.ArgumentParser(description='Run APCF-Net inference on raw underwater images.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    ds = RawUnderwaterDataset(args.input_dir, image_size=cfg['data']['image_size'], extensions=cfg['data']['extensions'], augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for img, name in tqdm(loader, desc='Enhancing'):
            out = model(img.to(device))['enhanced'][0]
            save_tensor_image(out, output_dir / name[0])
    print(f'Enhanced images saved to: {output_dir}')


if __name__ == '__main__':
    main()
