import torch


def rgb_to_hsv(rgb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable RGB to HSV conversion for tensors in [0,1].

    Args:
        rgb: Tensor of shape Bx3xHxW.
    Returns:
        HSV tensor in [0,1], same shape.
    """
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    maxc, _ = rgb.max(dim=1, keepdim=True)
    minc, _ = rgb.min(dim=1, keepdim=True)
    v = maxc
    delta = maxc - minc
    s = delta / (maxc + eps)

    rc = (maxc - r) / (delta + eps)
    gc = (maxc - g) / (delta + eps)
    bc = (maxc - b) / (delta + eps)

    h = torch.zeros_like(maxc)
    h = torch.where((maxc == r) & (delta > eps), (bc - gc) % 6.0, h)
    h = torch.where((maxc == g) & (delta > eps), 2.0 + rc - bc, h)
    h = torch.where((maxc == b) & (delta > eps), 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    return torch.cat([h, s, v], dim=1).clamp(0, 1)


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp_min(0) ** 2.4)


def rgb_to_lab(rgb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Approximate differentiable RGB to CIELab conversion.

    Output is normalized to [0,1] per channel:
    L*: [0,100] -> [0,1], a*/b*: approximately [-128,127] -> [0,1].
    """
    rgb_lin = _srgb_to_linear(rgb.clamp(0, 1))
    r, g, b = rgb_lin[:, 0:1], rgb_lin[:, 1:2], rgb_lin[:, 2:3]

    # sRGB D65 conversion
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    x = x / 0.95047
    z = z / 1.08883

    delta = 6.0 / 29.0

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > delta ** 3, t.clamp_min(eps) ** (1.0 / 3.0), t / (3 * delta ** 2) + 4.0 / 29.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    bb = 200 * (fy - fz)

    L = (L / 100.0).clamp(0, 1)
    a = ((a + 128.0) / 255.0).clamp(0, 1)
    bb = ((bb + 128.0) / 255.0).clamp(0, 1)
    return torch.cat([L, a, bb], dim=1)
