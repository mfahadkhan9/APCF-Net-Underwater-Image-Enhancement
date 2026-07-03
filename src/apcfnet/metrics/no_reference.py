from pathlib import Path
from typing import Dict
import cv2
import numpy as np
from skimage import color, filters, measure


def read_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Could not read image: {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def average_gradient(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def colorfulness_contrast_factor(rgb: np.ndarray) -> float:
    lab = color.rgb2lab(rgb / 255.0)
    a = lab[..., 1]
    b = lab[..., 2]
    return float(np.sqrt(np.var(a) + np.var(b)))


def uciqe(rgb: np.ndarray) -> float:
    lab = color.rgb2lab(rgb / 255.0)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    saturation = chroma / (lab[..., 0] + 1e-6)
    contrast_l = np.percentile(lab[..., 0], 99) - np.percentile(lab[..., 0], 1)
    return float(0.4680 * np.std(chroma) + 0.2745 * contrast_l + 0.2576 * np.mean(saturation))


def uicm(rgb: np.ndarray) -> float:
    rgb = rgb.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(-0.0268 * np.mean(rg) + 0.1586 * np.std(rg) - 0.0986 * np.mean(yb) + 0.2356 * np.std(yb))


def uism(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = filters.sobel(gray.astype(np.float32) / 255.0)
    return float(np.mean(edges) * 100.0)


def uiconm(rgb: np.ndarray, block_size: int = 8) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    vals = []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = gray[y:y + block_size, x:x + block_size]
            mx, mn = float(block.max()), float(block.min())
            if mx + mn > 1e-6:
                vals.append((mx - mn) / (mx + mn))
    return float(np.mean(vals)) if vals else 0.0


def uiqm(rgb: np.ndarray) -> float:
    return float(0.0282 * uicm(rgb) + 0.2953 * uism(rgb) + 3.5753 * uiconm(rgb))


def entropy_score(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(measure.shannon_entropy(gray))


def sharpness_laplacian(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_no_reference_metrics(path: str | Path) -> Dict[str, float]:
    rgb = read_rgb(path)
    return {
        'AG': average_gradient(rgb),
        'CCF': colorfulness_contrast_factor(rgb),
        'UCIQE': uciqe(rgb),
        'UIQM': uiqm(rgb),
        'UICM': uicm(rgb),
        'UISM': uism(rgb),
        'UIConM': uiconm(rgb),
        'Entropy': entropy_score(rgb),
        'Sharpness': sharpness_laplacian(rgb),
    }
