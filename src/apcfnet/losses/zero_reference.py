from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroReferenceLoss(nn.Module):
    """Zero-reference objective for non-paired underwater image enhancement.

    The loss does not require reference images. It combines spatial consistency,
    exposure control, gray-world color constancy, total variation, and curve
    smoothness regularization.
    """

    def __init__(self, lambda_spatial: float = 1.0, lambda_exposure: float = 10.0, lambda_color: float = 5.0,
                 lambda_tv: float = 1.0, lambda_curve: float = 0.1, exposure_target: float = 0.60, patch_size: int = 16):
        super().__init__()
        self.lambda_spatial = lambda_spatial
        self.lambda_exposure = lambda_exposure
        self.lambda_color = lambda_color
        self.lambda_tv = lambda_tv
        self.lambda_curve = lambda_curve
        self.exposure_target = exposure_target
        self.patch_size = patch_size
        kernel_left = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_right = torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_up = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_down = torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel_left', kernel_left)
        self.register_buffer('kernel_right', kernel_right)
        self.register_buffer('kernel_up', kernel_up)
        self.register_buffer('kernel_down', kernel_down)

    def spatial_consistency(self, inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        inp_gray = inp.mean(dim=1, keepdim=True)
        out_gray = out.mean(dim=1, keepdim=True)
        inp_pool = F.avg_pool2d(inp_gray, kernel_size=4, stride=4)
        out_pool = F.avg_pool2d(out_gray, kernel_size=4, stride=4)
        loss = 0.0
        for k in [self.kernel_left, self.kernel_right, self.kernel_up, self.kernel_down]:
            d_in = F.conv2d(inp_pool, k, padding=1)
            d_out = F.conv2d(out_pool, k, padding=1)
            loss = loss + torch.mean((d_out - d_in) ** 2)
        return loss / 4.0

    def exposure_control(self, out: torch.Tensor) -> torch.Tensor:
        gray = out.mean(dim=1, keepdim=True)
        mean = F.avg_pool2d(gray, kernel_size=self.patch_size, stride=self.patch_size)
        return torch.mean(torch.abs(mean - self.exposure_target))

    @staticmethod
    def color_constancy(out: torch.Tensor) -> torch.Tensor:
        mean_rgb = out.mean(dim=(2, 3))
        mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        return torch.mean((mr - mg) ** 2 + (mr - mb) ** 2 + (mg - mb) ** 2)

    @staticmethod
    def total_variation(out: torch.Tensor) -> torch.Tensor:
        tv_h = torch.mean(torch.abs(out[:, :, 1:, :] - out[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(out[:, :, :, 1:] - out[:, :, :, :-1]))
        return tv_h + tv_w

    @staticmethod
    def curve_smoothness(curve_params: torch.Tensor) -> torch.Tensor:
        if curve_params.shape[-1] < 2:
            return curve_params.new_tensor(0.0)
        return torch.mean(torch.abs(curve_params[:, :, 1:] - curve_params[:, :, :-1]))

    def forward(self, inp: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = outputs['enhanced']
        curve_params = outputs['curve_params']
        l_spa = self.spatial_consistency(inp, out)
        l_exp = self.exposure_control(out)
        l_col = self.color_constancy(out)
        l_tv = self.total_variation(out)
        l_curve = self.curve_smoothness(curve_params)
        total = (self.lambda_spatial * l_spa + self.lambda_exposure * l_exp + self.lambda_color * l_col +
                 self.lambda_tv * l_tv + self.lambda_curve * l_curve)
        return total, {
            'loss_total': float(total.detach().cpu()),
            'loss_spatial': float(l_spa.detach().cpu()),
            'loss_exposure': float(l_exp.detach().cpu()),
            'loss_color': float(l_col.detach().cpu()),
            'loss_tv': float(l_tv.detach().cpu()),
            'loss_curve': float(l_curve.detach().cpu()),
        }
