from pathlib import Path
from typing import Iterable, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from apcfnet.utils.io import list_images


class PairedUnderwaterDataset(Dataset):
    """Paired underwater image dataset for optional supervised experiments.

    The input and reference folders must contain images with identical file names.
    """

    def __init__(self, input_dir: str | Path, reference_dir: str | Path, image_size: int = 256,
                 extensions: Iterable[str] = ('.jpg', '.jpeg', '.png', '.bmp'), augment: bool = False):
        self.input_dir = Path(input_dir)
        self.reference_dir = Path(reference_dir)
        self.paths = list_images(self.input_dir, extensions)
        if not self.paths:
            raise FileNotFoundError(f'No input images found in {self.input_dir}.')
        self.ref_paths = []
        for p in self.paths:
            ref = self.reference_dir / p.name
            if not ref.exists():
                raise FileNotFoundError(f'Missing reference for {p.name}: {ref}')
            self.ref_paths.append(ref)
        ops = [transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)]
        if augment:
            ops += [transforms.RandomHorizontalFlip(p=0.5)]
        ops += [transforms.ToTensor()]
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[object, object, str]:
        inp = Image.open(self.paths[idx]).convert('RGB')
        ref = Image.open(self.ref_paths[idx]).convert('RGB')
        return self.transform(inp), self.transform(ref), self.paths[idx].name
