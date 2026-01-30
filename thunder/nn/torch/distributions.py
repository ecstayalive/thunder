import math
from typing import Iterable

import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

from .functional import inverse_softplus
from .modules import LinearBlock


class NeuralDistribution(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NeuralNormal(NeuralDistribution):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Iterable[int] = None,
        activation: str = "mish",
        init_std: float = 1.0,
        min_std: float = 0.01,
        max_std: float = 20.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ffn = LinearBlock(
            in_features=in_features,
            out_features=2 * out_features,
            hidden_features=hidden_features,
            activation=activation,
            activate_output=False,
            **factory_kwargs,
        )
        self.out_features = out_features
        self.init_std = init_std
        self.min_std = min_std
        self.max_std = max_std
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0):
        self.ffn.reset_parameters(gain=gain)
        last_layer: nn.Linear = self.ffn.linear_block[-1]
        with torch.no_grad():
            nn.init.orthogonal_(last_layer.weight[: self.out_features], gain=math.sqrt(gain))
            nn.init.orthogonal_(last_layer.weight[self.out_features :], gain=0.01 * math.sqrt(gain))
            out_std = max(self.init_std - self.min_std, 0.01)
            std_bias = inverse_softplus(
                torch.tensor(
                    out_std, device=last_layer.weight.device, dtype=last_layer.weight.dtype
                )
            )
            last_layer.bias[self.out_features :].fill_(std_bias)
            last_layer.bias[: self.out_features].fill_(0.0)

    def forward(self, features: torch.Tensor):
        mean, inv_std = torch.chunk(self.ffn(features), 2, -1)
        std = torch.clamp(F.softplus(inv_std) + self.min_std, max=self.max_std)
        return distributions.Normal(mean, std)


class NeuralConsistentNormal(NeuralDistribution):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Iterable[int] = None,
        activation: str = "mish",
        init_std: float = 1.0,
        min_std: float = 0.01,
        max_std: float = 20.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ffn = LinearBlock(
            in_features,
            out_features,
            hidden_features,
            activation,
            False,
            device=device,
            dtype=dtype,
        )
        self.inv_std = nn.Parameter(
            torch.ones(out_features, device=device, dtype=dtype) * math.log(init_std)
        )
        self.init_std = init_std
        self.min_std = min_std
        self.max_std = max_std
        self.device = device
        self.dtype = dtype
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0):
        self.ffn.reset_parameters(gain)
        out_std = max(self.init_std - self.min_std, 0.01)
        std_bias = inverse_softplus(torch.tensor(out_std, device=self.device, dtype=self.dtype))
        with torch.no_grad():
            self.inv_std.fill_(std_bias)

    def forward(self, features: torch.Tensor):
        std = torch.clamp(F.softplus(self.inv_std) + self.min_std, max=self.max_std)
        return distributions.Normal(self.ffn(features), std)
