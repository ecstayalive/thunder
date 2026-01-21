import math
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from thunder.nn.torch.mapping import ACTIVATION_CLS_NAME

from .activation import Sin

__all__ = ["LinearBlock", "SirenBlock"]


class LinearBlock(nn.Module):
    """Multi-Layer Perception Block

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        hidden_features: For example, one tuple (256, 126, 10) stands that
            there are three hidden layer, which sizes are 256, 126, 10.
        activation: The type of activation function used in
            this mlp block
        activation_output: Whether the output needs to be activated


    Note: Use orthogonal method to initialize the linear weight.
        For details: https://arxiv.org/pdf/1609.07093.pdf and
        https://arxiv.org/pdf/math-ph/0609050.pdf
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Iterator[int] = None,
        activation: str = "softsign",
        activate_output: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # get block attributions
        activation_fn_name = ACTIVATION_CLS_NAME[activation.lower()]
        activation_cls = getattr(nn, activation_fn_name)
        if hidden_features is not None:
            arch = (in_features, *hidden_features, out_features)
        else:
            arch = (in_features, out_features)
        # create linear block
        layers = []
        for in_dimension, out_dimension in zip(arch[:-2], arch[1:-1]):
            layers.extend(
                (
                    nn.Linear(in_dimension, out_dimension, **factory_kwargs),
                    activation_cls(),
                )
            )
        layers.append(nn.Linear(arch[-2], arch[-1], **factory_kwargs))
        if activate_output:
            layers.append(activation_cls())
        self.linear_block = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0) -> None:
        for layer in self.linear_block:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(gain))

    def forward(self, input: Tensor) -> Tensor:
        return self.linear_block(input)


class SirenBlock(nn.Module):
    """Siren is introduced in "Implicit Neural Representations with Periodic Activation Functions "
    For detail: https://arxiv.org/abs/2006.09661
    Args:
        in_features:
        out_features:
        hidden_features:
        omega:
        device:
        dtype:
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Iterator[int],
        activate_output: bool = False,
        omega: float = 30.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # get block attributions
        activation_cls = Sin
        self.arch = (in_features, *hidden_features, out_features)
        # First layer is very special
        self.siren_head_weight = nn.Parameter(
            torch.empty((self.arch[1], self.arch[0]), **factory_kwargs)
        )
        self.siren_head_bias = nn.Parameter(torch.empty(self.arch[1], **factory_kwargs))
        self.siren_head_out_features = self.arch[1]
        # Other layers
        tail_layers = []
        for in_dimension, out_dimension in zip(self.arch[1:-1], self.arch[2:]):
            tail_layers.extend(
                (
                    activation_cls(),
                    nn.Linear(in_dimension, out_dimension, **factory_kwargs),
                )
            )
        if activate_output:
            tail_layers.append(activation_cls())
        self.siren_tail = nn.Sequential(*tail_layers)
        self.omega = omega
        self.reset_parameters()

    def reset_parameters(self, c: float = 6):
        """For details: https://arxiv.org/abs/2006.09661"""
        nn.init.uniform_(
            self.siren_head_weight,
            -1 / self.siren_head_out_features,
            1 / self.siren_head_out_features,
        )
        nn.init.uniform_(
            self.siren_head_bias,
            -1 / self.siren_head_out_features,
            1 / self.siren_head_out_features,
        )
        for layer in self.siren_tail:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(
                    layer.weight,
                    -math.sqrt(c / layer.out_features),
                    math.sqrt(c / layer.out_features),
                )
                nn.init.uniform_(
                    layer.bias,
                    -math.sqrt(c / layer.out_features),
                    math.sqrt(c / layer.out_features),
                )

    def forward(self, input: Tensor) -> Tensor:
        input = F.linear(input, self.omega * self.siren_head_weight, self.siren_head_bias)
        return self.siren_tail(input)

    def extra_repr(self) -> str:
        return f"(siren_head): Linear(in_features={self.arch[0]}, out_features={self.arch[1]}, bias=True)"
        return f"(siren_head): Linear(in_features={self.arch[0]}, out_features={self.arch[1]}, bias=True)"
