import math
from typing import Iterable, Tuple

import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

from .functional import inverse_softplus
from .modules import LinearBlock


class Distribution:
    has_rsample = True

    def __init__(
        self, batch_shape: torch.Size = torch.Size(), event_shape: torch.Size = torch.Size()
    ):
        self.event_shape = event_shape

    def sample(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def rsample(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def mean(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def entropy(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def std(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def var(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class Normal(Distribution):
    def __init__(self, loc: torch.Tensor, stddev: torch.Tensor):
        self.loc = loc
        self.stdev = stddev

    def sample(self):
        with torch.no_grad():
            return self.rsample()

    def rsample(self):
        return self.mean + self.std * torch.randn_like(self.mean)


class DistributionHead(nn.Module):
    def forward(self, *args, **kwargs) -> Distribution:
        raise NotImplementedError


class NeuralNormal(DistributionHead):
    def __init__(
        self,
        in_features: int,
        out_size: int,
        hidden_features: Iterable[int] = None,
        activation: str = "mish",
        init_std: float = 1.0,
        min_std: float = 0.01,
        max_std: float = 20.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.out_size = out_size
        self.init_std = init_std
        self.min_std = min_std
        self.max_std = max_std
        self.ffn = LinearBlock(
            in_features=in_features,
            out_features=2 * out_size,
            hidden_features=hidden_features,
            activation=activation,
            activate_output=False,
            device=device,
            dtype=dtype,
        )
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0):
        last_layer: nn.Linear = self.ffn.linear_block[-1]
        with torch.no_grad():
            nn.init.orthogonal_(last_layer.weight[: self.out_size], gain=math.sqrt(gain))
            nn.init.orthogonal_(last_layer.weight[self.out_size :], gain=0.01 * math.sqrt(gain))
            out_std = max(self.init_std - self.min_std, 0.01)
            std_bias = inverse_softplus(
                torch.tensor(
                    out_std, device=last_layer.weight.device, dtype=last_layer.weight.dtype
                )
            )
            last_layer.bias[self.out_size :].fill_(std_bias)
            last_layer.bias[: self.out_size].fill_(0.0)

    def forward(self, features: torch.Tensor) -> distributions.Normal:
        mean, inv_std = torch.chunk(self.ffn(features), 2, -1)
        std = torch.clamp(F.softplus(inv_std) + self.min_std, max=self.max_std)
        return distributions.Normal(mean, std)


class NeuralTransformedDist(DistributionHead):
    def __init__(
        self,
        in_features: int,
        out_size: int,
        hidden_features: Iterable[int] = None,
        activation: str = "mish",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ffn = LinearBlock(
            in_features=in_features,
            out_features=out_size,
            hidden_features=hidden_features,
            activation=activation,
            activate_output=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, *args, **kwargs):
        pass


class NeuralConsistentNormal(DistributionHead):
    def __init__(
        self,
        in_features: int,
        out_size: int,
        hidden_features: Iterable[int] = None,
        activation: str = "mish",
        init_std: float = 1.0,
        min_std: float = 0.01,
        max_std: float = 20.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.init_std = init_std
        self.min_std = min_std
        self.max_std = max_std
        self.ffn = LinearBlock(
            in_features=in_features,
            out_features=out_size,
            hidden_features=hidden_features,
            activation=activation,
            activate_output=False,
            device=device,
            dtype=dtype,
        )
        self.inv_std = nn.Parameter(
            torch.ones(out_size, device=device, dtype=dtype) * math.log(init_std)
        )
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0):
        out_std = max(self.init_std - self.min_std, 0.01)
        std_bias = inverse_softplus(
            torch.tensor(out_std, device=self.inv_std.device, dtype=self.inv_std.dtype)
        )
        with torch.no_grad():
            self.inv_std.fill_(std_bias)

    def forward(self, features: torch.Tensor):
        std = torch.clamp(F.softplus(self.inv_std) + self.min_std, max=self.max_std)
        return distributions.Normal(self.ffn(features), std)
