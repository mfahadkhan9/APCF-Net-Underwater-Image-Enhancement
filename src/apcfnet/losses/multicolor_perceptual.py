from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

from apcfnet.data.color import rgb_to_hsv, rgb_to_lab


class MultiColorPerceptualLoss(nn.Module):
    """Optional supervised loss used when paired references are available.

    L_total = lambda_rgb L_rgb + lambda_hsv L_hsv + lambda_lab L_lab + lambda_per L_perceptual

    This loss is provided for reproducibility of paired/reference experiments. For raw/non-paired
    training, use ZeroReferenceLoss.
    """

    def __init__(self, lambda_rgb: float = 1.0, lambda_hsv: float = 1.0, lambda_lab: float = 1.0,
                 lambda_perceptual: float = 0.1, use_imagenet_weights: bool = True):
        super().__init__()
        self.lambda_rgb = lambda_rgb
        self.lambda_hsv = lambda_hsv
        self.lambda_lab = lambda_lab
        self.lambda_perceptual = lambda_perceptual
        self.l1 = nn.L1Loss()
        weights = VGG19_Weights.IMAGENET1K_V1 if use_imagenet_weights else None
        self.vgg = vgg19(weights=weights).features[:16].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _vgg_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, 1)
        return (x - self.mean) / self.std

    def perceptual(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out_f = self.vgg(self._vgg_preprocess(out))
        tgt_f = self.vgg(self._vgg_preprocess(target))
        return self.l1(out_f, tgt_f)

    def forward(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = outputs['enhanced']
        l_rgb = self.l1(out, target)
        l_hsv = self.l1(rgb_to_hsv(out), rgb_to_hsv(target))
        l_lab = self.l1(rgb_to_lab(out), rgb_to_lab(target))
        l_per = self.perceptual(out, target)
        total = (self.lambda_rgb * l_rgb + self.lambda_hsv * l_hsv +
                 self.lambda_lab * l_lab + self.lambda_perceptual * l_per)
        return total, {
            'loss_total': float(total.detach().cpu()),
            'loss_rgb': float(l_rgb.detach().cpu()),
            'loss_hsv': float(l_hsv.detach().cpu()),
            'loss_lab': float(l_lab.detach().cpu()),
            'loss_perceptual': float(l_per.detach().cpu()),
        }
