from pathlib import Path
from typing import Iterable, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from apcfnet.utils.io import list_images


class RawUnderwaterDataset(Dataset):
    """Raw/non-paired underwater image dataset.

    The dataset returns only the degraded underwater image and its filename.
    No paired target/reference image is required.
    """

    def __init__(self, root: str | Path, image_size: int = 256, extensions: Iterable[str] = ('.jpg', '.jpeg', '.png', '.bmp'), augment: bool = False):
        self.root = Path(root)
        self.paths = list_images(self.root, extensions)
        if not self.paths:
            raise FileNotFoundError(f'No images found in {self.root}. Expected extensions: {extensions}')
        ops = [transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)]
        if augment:
            ops += [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.1)]
        ops += [transforms.ToTensor()]
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[object, str]:
        path = self.paths[idx]
        image = Image.open(path).convert('RGB')
        return self.transform(image), path.name
