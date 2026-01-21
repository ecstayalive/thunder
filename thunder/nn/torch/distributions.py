import math
from abc import ABC, abstractmethod

import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

from thunder.nn.torch.functional import inverse_softplus


class Distributions(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Normal(Distributions):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_std: float = 1.0,
        min_std: float = 0.01,
        max_std: float = 20.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.init_std = init_std
        self.min_std = min_std
        self.max_std = max_std

        self.projector = nn.Linear(in_features, 2 * out_features, **factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0):
        nn.init.orthogonal_(self.projector.weight[: self.out_features], gain=math.sqrt(gain))
        nn.init.orthogonal_(self.projector.weight[self.out_features :], gain=0.1 * math.sqrt(gain))
        # nn.init.constant_(self.projector.bias, 0.0)
        out_std = max(self.init_std - self.min_std, 0.0)
        std_bias = inverse_softplus(
            torch.tensor(
                out_std, device=self.projector.weight.device, dtype=self.projector.weight.dtype
            )
        )
        with torch.no_grad():
            self.projector.bias[self.out_features :].fill_(std_bias)

    def forward(self, features: torch.Tensor):
        mean, log_std = torch.chunk(self.projector(features), 2, -1)
        std = torch.clamp(F.softplus(log_std) + self.min_std, max=self.max_std)
        return distributions.Normal(mean, std)


class ConsistentNormal(Distributions):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_std: float = 1.0,
        min_std: float = 0.01,
        max_std: float = 20.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.init_std = init_std
        self.min_std = min_std
        self.max_std = max_std
        self.projector = nn.Linear(in_features, out_features, **factory_kwargs)
        self.log_std = nn.Parameter(torch.ones(out_features, **factory_kwargs) * math.log(init_std))
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0):
        nn.init.orthogonal_(self.projector.weight, gain=math.sqrt(gain))
        # nn.init.constant_(self.projector.bias, 0.0)
        out_std = max(self.init_std - self.min_std, 0.0)
        std_bias = inverse_softplus(
            torch.tensor(
                out_std, device=self.projector.weight.device, dtype=self.projector.weight.dtype
            )
        )
        with torch.no_grad():
            self.log_std.fill_(std_bias)

    def forward(self, features: torch.Tensor):
        std = torch.clamp(F.softplus(self.log_std) + self.min_std, max=self.max_std)
        return distributions.Normal(self.projector(features), std)
