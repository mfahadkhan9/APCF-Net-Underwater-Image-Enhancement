import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MIRNetRefinementBlock(nn.Module):
    """MIRNet-inspired residual attention refinement block."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = ConvBNReLU(channels, channels)
        self.conv3 = ConvBNReLU(channels, channels)
        self.att = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.conv3(self.conv2(self.conv1(x)))
        a = self.att(r)
        return x + a * r


class UNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
            MIRNetRefinementBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.proj = ConvBNReLU(in_ch, out_ch)
        self.refine = UNetBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.proj(x)
        return self.refine(torch.cat([x, skip], dim=1))


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual bottleneck."""

    def __init__(self, in_ch: int, out_ch: int, expansion: int = 4):
        super().__init__()
        hidden = in_ch * expansion
        self.use_residual = in_ch == out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.use_residual else y


class TripleChannelWiseAttention(nn.Module):
    """Adaptive fusion of RGB, HSV, and Lab feature maps."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fuse = ConvBNReLU(channels * 3, channels)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )

    def forward(self, f_rgb: torch.Tensor, f_hsv: torch.Tensor, f_lab: torch.Tensor):
        fused = self.fuse(torch.cat([f_rgb, f_hsv, f_lab], dim=1))
        logits = self.gate(fused)  # Bx3x1x1
        weights = torch.softmax(logits, dim=1)
        w_rgb, w_hsv, w_lab = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        out = w_rgb * f_rgb + w_hsv * f_hsv + w_lab * f_lab
        return out, weights


def piecewise_curve_transform(x: torch.Tensor, params: torch.Tensor, num_segments: int = 8) -> torch.Tensor:
    """Apply adaptive monotonic piecewise curve to RGB tensor.

    Args:
        x: Bx3xHxW image in [0,1].
        params: Bx3xK curve parameters. Larger values increase local enhancement.
        num_segments: K.
    """
    x = x.clamp(0, 1)
    params = torch.sigmoid(params).unsqueeze(-1).unsqueeze(-1)  # Bx3xKx1x1
    y = x.unsqueeze(2).repeat(1, 1, num_segments, 1, 1)
    idx = torch.arange(num_segments, device=x.device, dtype=x.dtype).view(1, 1, num_segments, 1, 1)
    delta = torch.clamp(num_segments * y - idx, 0.0, 1.0)
    increments = torch.tanh(delta * (1.0 + 2.0 * params)) / float(num_segments)
    out = increments.sum(dim=2)
    return out.clamp(0, 1)
