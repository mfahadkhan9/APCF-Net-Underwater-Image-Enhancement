import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from apcfnet.models.apcf_net import APCFNet
from apcfnet.losses.zero_reference import ZeroReferenceLoss


def test_forward_and_loss():
    torch.set_num_threads(1)
    model = APCFNet(base_channels=4, num_curve_segments=2)
    x = torch.rand(1, 3, 32, 32)
    outputs = model(x)
    assert outputs['enhanced'].shape == x.shape
    assert outputs['curve_params'].shape == (1, 3, 2)
    loss, parts = ZeroReferenceLoss(patch_size=8)(x, outputs)
    assert torch.isfinite(loss)
    assert 'loss_total' in parts
