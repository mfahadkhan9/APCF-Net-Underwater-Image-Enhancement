#!/usr/bin/env python
from pathlib import Path
import argparse
import random
import shutil


def main():
    parser = argparse.ArgumentParser(description='Split raw underwater images into train/val/test folders.')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_root', default='data/raw')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--copy', action='store_true', help='Copy files instead of symlinking.')
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp'])
    args = parser.parse_args()
    paths = sorted([p for p in Path(args.input_dir).rglob('*') if p.suffix.lower() in args.extensions])
    random.Random(args.seed).shuffle(paths)
    n = len(paths)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    splits = {
        'train': paths[:n_train],
        'val': paths[n_train:n_train + n_val],
        'test': paths[n_train + n_val:],
    }
    root = Path(args.output_root)
    for split, items in splits.items():
        out_dir = root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in items:
            dst = out_dir / p.name
            if dst.exists():
                continue
            if args.copy:
                shutil.copy2(p, dst)
            else:
                try:
                    dst.symlink_to(p.resolve())
                except OSError:
                    shutil.copy2(p, dst)
        print(f'{split}: {len(items)} images -> {out_dir}')


if __name__ == '__main__':
    main()
