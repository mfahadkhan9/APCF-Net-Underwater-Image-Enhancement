#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import argparse
import pandas as pd
from tqdm import tqdm

from apcfnet.metrics.reference import compute_reference_metrics
from apcfnet.utils.io import list_images


def main():
    parser = argparse.ArgumentParser(description='Evaluate PSNR/SSIM using matched references.')
    parser.add_argument('--enhanced_dir', required=True)
    parser.add_argument('--reference_dir', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp'])
    args = parser.parse_args()
    enhanced = list_images(args.enhanced_dir, args.extensions)
    ref_dir = Path(args.reference_dir)
    rows = []
    for ep in tqdm(enhanced, desc='Reference metrics'):
        rp = ref_dir / ep.name
        if not rp.exists():
            print(f'Skipping {ep.name}: reference not found')
            continue
        m = compute_reference_metrics(ep, rp)
        m['filename'] = ep.name
        rows.append(m)
    if not rows:
        raise RuntimeError('No matched enhanced/reference pairs were found.')
    df = pd.DataFrame(rows)
    df = df[['filename'] + [c for c in df.columns if c != 'filename']]
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    df.drop(columns=['filename']).mean(numeric_only=True).to_csv(out.with_name(out.stem + '_mean.csv'))
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
