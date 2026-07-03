from pathlib import Path
from typing import Dict
import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from apcfnet.metrics.no_reference import read_rgb


def compute_reference_metrics(enhanced_path: str | Path, reference_path: str | Path) -> Dict[str, float]:
    enh = read_rgb(enhanced_path)
    ref = read_rgb(reference_path)
    if enh.shape != ref.shape:
        ref = cv2.resize(ref, (enh.shape[1], enh.shape[0]), interpolation=cv2.INTER_CUBIC)
    psnr = peak_signal_noise_ratio(ref, enh, data_range=255)
    ssim = structural_similarity(ref, enh, channel_axis=2, data_range=255)
    return {'PSNR': float(psnr), 'SSIM': float(ssim)}
