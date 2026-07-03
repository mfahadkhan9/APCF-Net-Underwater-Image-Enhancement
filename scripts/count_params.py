#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import argparse
from apcfnet.models.apcf_net import APCFNet
from apcfnet.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description='Count APCF-Net trainable parameters.')
    parser.add_argument('--config', default='configs/apcfnet_raw_uieb.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    m = cfg['model']
    model = APCFNet(m['base_channels'], m['num_curve_segments'], m['use_tca'], m['use_curve'])
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n:,} ({n / 1e6:.3f} M)')


if __name__ == '__main__':
    main()
