# Dataset preparation

This repository does not redistribute full benchmark datasets. Download data from the official sources and place images into the expected folders.

## UIEB

Official URL:

```text
https://li-chongyi.github.io/proj_benchmark.html
```

Suggested layout for raw/non-paired training:

```text
data/raw/train/*.jpg
data/raw/val/*.jpg
data/raw/test/*.jpg
```

Optional reference images for PSNR/SSIM:

```text
data/reference_optional/test/*.jpg
```

Reference images must have the same file names as enhanced outputs.

## UCCS / RUIE

Repository URL:

```text
https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark
```

Use UCCS images as raw inputs for no-reference evaluation and qualitative comparison.

## NYU Depth V2

Official URL:

```text
https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
```

Use the RGB images according to your experimental protocol.

## Important note on redistribution

Do not upload full UIEB, UCCS/RUIE, or NYU-v2 data to a public GitHub repository unless their license explicitly permits redistribution. For public release, provide dataset URLs and preparation scripts instead.
