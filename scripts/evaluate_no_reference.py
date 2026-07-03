#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import argparse
import pandas as pd
from tqdm import tqdm

from apcfnet.metrics.no_reference import compute_no_reference_metrics
from apcfnet.utils.io import list_images


def main():
    parser = argparse.ArgumentParser(description='Evaluate no-reference underwater image quality metrics.')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp'])
    args = parser.parse_args()
    paths = list_images(args.input_dir, args.extensions)
    if not paths:
        raise FileNotFoundError(f'No images found in {args.input_dir}')
    rows = []
    for p in tqdm(paths, desc='Metrics'):
        m = compute_no_reference_metrics(p)
        m['filename'] = p.name
        rows.append(m)
    df = pd.DataFrame(rows)
    cols = ['filename'] + [c for c in df.columns if c != 'filename']
    df = df[cols]
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = df.drop(columns=['filename']).mean(numeric_only=True)
    summary.to_csv(out.with_name(out.stem + '_mean.csv'))
    print(summary)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
