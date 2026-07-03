from pathlib import Path
from typing import Iterable, List
from PIL import Image
import torch
import numpy as np


def list_images(root: str | Path, extensions: Iterable[str]) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    exts = tuple(e.lower() for e in extensions)
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in exts])


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Convert a tensor in [0,1] with shape CxHxW to PIL RGB."""
    x = x.detach().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def save_tensor_image(x: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(x).save(path)
