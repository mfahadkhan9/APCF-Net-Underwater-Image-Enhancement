import torch
import torch.nn as nn
import torch.nn.functional as F

from apcfnet.data.color import rgb_to_hsv, rgb_to_lab
from apcfnet.models.blocks import UNetBlock, UpBlock, InvertedResidual, TripleChannelWiseAttention, piecewise_curve_transform


class APCFNet(nn.Module):
    """APCF-Net for raw/non-paired underwater image enhancement."""

    def __init__(self, base_channels: int = 32, num_curve_segments: int = 8, use_tca: bool = True, use_curve: bool = True):
        super().__init__()
        self.num_curve_segments = num_curve_segments
        self.use_tca = use_tca
        self.use_curve = use_curve
        c = base_channels

        self.rgb_stem = UNetBlock(3, c)
        self.hsv_stem = UNetBlock(3, c)
        self.lab_stem = UNetBlock(3, c)
        self.tca = TripleChannelWiseAttention(c) if use_tca else None

        self.pool = nn.MaxPool2d(2)
        self.enc1 = UNetBlock(c, c)
        self.enc2 = UNetBlock(c, c * 2)
        self.enc3 = UNetBlock(c * 2, c * 4)
        self.enc4 = UNetBlock(c * 4, c * 8)

        self.bottleneck = nn.Sequential(
            InvertedResidual(c * 8, c * 8),
            InvertedResidual(c * 8, c * 8),
        )

        self.up3 = UpBlock(c * 8, c * 4, c * 4)
        self.up2 = UpBlock(c * 4, c * 2, c * 2)
        self.up1 = UpBlock(c * 2, c, c)
        self.up0 = UpBlock(c, c, c)

        self.reconstruction = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 3, kernel_size=1),
            nn.Sigmoid(),
        )

        self.curve_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 8, c * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, 3 * num_curve_segments, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        x = x.clamp(0, 1)
        hsv = rgb_to_hsv(x)
        lab = rgb_to_lab(x)

        f_rgb = self.rgb_stem(x)
        f_hsv = self.hsv_stem(hsv)
        f_lab = self.lab_stem(lab)

        if self.use_tca:
            f0, att_weights = self.tca(f_rgb, f_hsv, f_lab)
        else:
            f0 = (f_rgb + f_hsv + f_lab) / 3.0
            att_weights = None

        e1 = self.enc1(f0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(e4)

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        d0 = self.up0(d1, f0)

        base = self.reconstruction(d0)
        curve_params = self.curve_head(b).view(x.shape[0], 3, self.num_curve_segments)
        if self.use_curve:
            enhanced = piecewise_curve_transform(base, curve_params, self.num_curve_segments)
        else:
            enhanced = base
        return {
            'enhanced': enhanced.clamp(0, 1),
            'base': base.clamp(0, 1),
            'curve_params': curve_params,
            'attention_weights': att_weights,
        }
