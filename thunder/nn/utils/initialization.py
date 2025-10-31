import math
from itertools import chain
from typing import Iterable

import torch.nn as nn

__all__ = ["orthogonal_modules_", "xavier_normal_modules_"]


def orthogonal_modules_(*modules: Iterable[nn.Module], gain: float = 2.0) -> None:
    """Initialize the network with orthogonal method

    Args:
        modules: The neural network modules
        gain: The gain of the initial params
            Default: 2.0
    Usually it is only used in MLP and RNN
    """
    for layer in chain.from_iterable(module.modules() for module in modules):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(gain))
        elif isinstance(layer, nn.LSTM):
            for i in range(layer.num_layers):
                nn.init.orthogonal_(getattr(layer, f"weight_ih_l{i}"), gain=math.sqrt(gain))
                nn.init.orthogonal_(getattr(layer, f"weight_hh_l{i}"), gain=math.sqrt(gain))


def xavier_normal_modules_(*modules: Iterable[nn.Module], gain: float = 1.0) -> None:
    """Initialize the network with xavier normal distribution
    Args:
        modules: The neural network modules
        gain: The gain of the initial params
            Default: 1.0
    """
    for layer in chain.from_iterable(module.modules() for module in modules):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight, gain=gain)
