import copy

import torch
from torch import nn


def create_target(net: nn.Module):
    target = copy.deepcopy(net)
    for p in target.parameters():
        p.requires_grad = False
    return target


@torch.inference_mode()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """
    Softly update the parameters of target module
    towards the parameters of source module.
    """

    for tgt, src in zip(target.parameters(), source.parameters(), strict=True):
        tgt.data.mul_(1 - tau).add_(src.data, alpha=tau)
