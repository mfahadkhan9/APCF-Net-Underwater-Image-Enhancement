#!/usr/bin/env python
from pathlib import Path
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser(description='Copy five locally downloaded UIEB raw images into examples/uieb_raw_5.')
    parser.add_argument('--uieb_raw_dir', required=True, help='Path to locally downloaded UIEB raw images.')
    parser.add_argument('--output_dir', default='examples/uieb_raw_5', help='Output folder for local sample images.')
    parser.add_argument('--num_images', type=int, default=5)
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp'])
    args = parser.parse_args()
    src = Path(args.uieb_raw_dir)
    dst = Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    paths = sorted([p for p in src.rglob('*') if p.suffix.lower() in args.extensions])
    if len(paths) < args.num_images:
        raise RuntimeError(f'Found only {len(paths)} images in {src}, but {args.num_images} were requested.')
    for p in paths[:args.num_images]:
        shutil.copy2(p, dst / p.name)
        print(f'Copied {p.name}')
    print(f'Done. Local sample folder: {dst}')
    print('Check UIEB licence conditions before uploading these images publicly.')


if __name__ == '__main__':
    main()
