import copy
from itertools import chain
from typing import Iterable

import torch.nn as nn

__all__ = ["clone_net", "freeze"]


def clone_net(source: nn.Module, requires_grad: bool = True) -> nn.Module:
    """Clone a given network or a given `nn.Module`
    Args:
        source: the network or `nn.Module` that you want to clone.
            :type source: nn.Module
        requires_grad: a boolean flag that determines whether the
            cloned network still keeps the gradients information.
            If `requires_grad` is set to `True`, the parameters of
            the cloned network will be included in the computation
            of gradients during back propagation.
            :type requires_grad: bool
                Default: True
    Returns:
        a cloned network.
            :type `nn.Module`
    """
    net = copy.deepcopy(source)
    for param in net.parameters():
        param.requires_grad = requires_grad
    return net


class freeze:
    """Context manager for freezing a given neural network module."""

    def __init__(self, *modules: Iterable[nn.Module]):
        self._modules = modules

    def __enter__(self):
        for param in chain.from_iterable(module.parameters() for module in self._modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for param in chain.from_iterable(module.parameters() for module in self._modules):
            param.requires_grad = True
