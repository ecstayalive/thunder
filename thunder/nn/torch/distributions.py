import math
from abc import ABC
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import inverse_softplus
from .modules import LinearBlock

LOG_2PI = math.log(2.0 * math.pi)


class Distribution(ABC):
    has_rsample = True

    def __init__(
        self, batch_shape: torch.Size = torch.Size(), event_shape: torch.Size = torch.Size()
    ):
        self.batch_shape = torch.Size(batch_shape)
        self.event_shape = torch.Size(event_shape)
        self._event_ndims = len(self.event_shape)
        self._event_numel = math.prod(self.event_shape) if self.event_shape else 1

    def _reduce_event(self, value: torch.Tensor) -> torch.Tensor:
        if self._event_ndims == 0:
            return value
        return value.reshape(value.shape[: -self._event_ndims] + (-1,)).sum(-1)

    def _reduce_event_log_det(
        self, log_det: torch.Tensor, target_shape: torch.Size
    ) -> torch.Tensor:
        if self._event_ndims == 0:
            return log_det
        if log_det.ndim == 0:
            return log_det * self._event_numel
        if log_det.shape[-self._event_ndims :] == target_shape[-self._event_ndims :]:
            return log_det.reshape(log_det.shape[: -self._event_ndims] + (-1,)).sum(-1)
        log_det = log_det.expand(target_shape)
        return log_det.reshape(log_det.shape[: -self._event_ndims] + (-1,)).sum(-1)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def mean(self) -> torch.Tensor:
        raise NotImplementedError

    def entropy(self) -> torch.Tensor:
        raise NotImplementedError

    def std(self) -> torch.Tensor:
        raise NotImplementedError

    def var(self) -> torch.Tensor:
        raise NotImplementedError


class Normal(Distribution):
    def __init__(
        self, loc: torch.Tensor, stddev: torch.Tensor, event_shape: torch.Size = torch.Size()
    ):
        loc, stddev = torch.broadcast_tensors(loc, stddev)
        event_shape = torch.Size(event_shape)
        param_shape = torch.Size(loc.shape)
        event_ndims = len(event_shape)
        if event_ndims > len(param_shape) or (
            event_ndims > 0 and param_shape[-event_ndims:] != event_shape
        ):
            raise ValueError(
                f"event_shape {event_shape} must match the rightmost dimensions "
                f"of broadcast parameter shape {param_shape}."
            )
        batch_shape = param_shape[: len(param_shape) - event_ndims] if event_ndims else param_shape
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)
        self.loc = loc
        self.stddev = stddev
        self.variance = stddev.square()
        self._param_shape = param_shape

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = torch.Size(sample_shape) + self._param_shape
        noise = torch.randn(shape, device=self.loc.device, dtype=self.loc.dtype)
        sample = self.loc + self.stddev * noise
        log_prob = self.log_prob(sample)
        return sample, log_prob

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        log_scale = torch.log(self.stddev)
        log_prob = -0.5 * (((value - self.loc) ** 2) / self.variance + LOG_2PI) - log_scale
        return self._reduce_event(log_prob)

    def mean(self) -> torch.Tensor:
        return self.loc

    def entropy(self) -> torch.Tensor:
        entropy = 0.5 + 0.5 * LOG_2PI + torch.log(self.stddev)
        return self._reduce_event(entropy)

    def std(self) -> torch.Tensor:
        return self.stddev

    def var(self) -> torch.Tensor:
        return self.variance


class Transform:
    bijective = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._call(x)

    def inv(self, y: torch.Tensor) -> torch.Tensor:
        return self._inverse(y)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ComposeTransform(Transform):
    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = tuple(transforms)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        for transform in reversed(self.transforms):
            y = transform.inv(y)
        return y

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(x)
        current = x
        for transform in self.transforms:
            next_value = transform(current)
            total = total + transform.log_abs_det_jacobian(current, next_value)
            current = next_value
        return total


class TanhTransform(Transform):
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        y = y.clamp(min=-1.0 + self.eps, max=1.0 - self.eps)
        return 0.5 * (torch.log1p(y) - torch.log1p(-y))

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del y
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class AffineTransform(Transform):
    def __init__(self, loc: float | torch.Tensor = 0.0, scale: float | torch.Tensor = 1.0):
        self.loc = loc
        self.scale = scale

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.loc

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del y
        scale = torch.as_tensor(self.scale, device=x.device, dtype=x.dtype)
        return torch.log(scale.abs())


class TransformedDistribution(Distribution):
    def __init__(self, base_dist: Distribution, transforms: Sequence[Transform]):
        super().__init__(batch_shape=base_dist.batch_shape, event_shape=base_dist.event_shape)
        self.base_dist = base_dist
        self.transforms = tuple(transforms)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tuple[torch.Tensor, torch.Tensor]:
        x, log_prob = self.base_dist.rsample(sample_shape)
        y = x
        for transform in self.transforms:
            next_y = transform(y)
            log_det = self._reduce_event_log_det(
                transform.log_abs_det_jacobian(y, next_y), y.shape
            )
            log_prob = log_prob - log_det
            y = next_y
        return y, log_prob

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        y = value
        log_prob = 0.0
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            log_det = self._reduce_event_log_det(
                transform.log_abs_det_jacobian(x, y), x.shape
            )
            log_prob = log_prob - log_det
            y = x
        return self.base_dist.log_prob(y) + log_prob

    def mean(self) -> torch.Tensor:
        y = self.base_dist.mean()
        for transform in self.transforms:
            y = transform(y)
        return y

    def entropy(self) -> torch.Tensor:
        sample, log_prob = self.rsample()
        del sample
        return -log_prob

    def std(self) -> torch.Tensor:
        raise NotImplementedError(
            "std is not analytically defined for generic transformed distributions."
        )

    def var(self) -> torch.Tensor:
        raise NotImplementedError(
            "var is not analytically defined for generic transformed distributions."
        )


class DistributionHead(nn.Module):
    def __init__(self, event_shape: torch.Size = torch.Size()):
        super().__init__()
        self.event_shape = torch.Size(event_shape)

    def forward(self, *args, **kwargs) -> Distribution:
        raise NotImplementedError


class NormalHead(DistributionHead):
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
        event_shape: torch.Size = torch.Size(),
    ):
        super().__init__(event_shape=event_shape)
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

    def forward(self, features: torch.Tensor) -> Normal:
        mean, inv_std = torch.chunk(self.ffn(features), 2, -1)
        std = torch.clamp(F.softplus(inv_std) + self.min_std, max=self.max_std)
        return Normal(mean, std, self.event_shape)


class TransformedDistHead(DistributionHead):
    def __init__(
        self,
        in_features: int,
        out_size: int,
        hidden_features: Iterable[int] = None,
        activation: str = "mish",
        init_std: float = 1.0,
        min_std: float = 0.01,
        max_std: float = 20.0,
        transforms: Sequence[Transform] | None = None,
        consistent_std: bool = False,
        device=None,
        dtype=None,
        event_shape: torch.Size = torch.Size(),
    ):
        super().__init__(event_shape=event_shape)
        head_cls = ConsistentNormalHead if consistent_std else NormalHead
        self.base_head = head_cls(
            in_features=in_features,
            out_size=out_size,
            hidden_features=hidden_features,
            activation=activation,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            device=device,
            dtype=dtype,
            event_shape=event_shape,
        )
        self.transforms = tuple(transforms) if transforms is not None else (TanhTransform(),)

    def forward(self, features: torch.Tensor) -> TransformedDistribution:
        return TransformedDistribution(self.base_head(features), self.transforms)


class ConsistentNormalHead(DistributionHead):
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
        event_shape: torch.Size = torch.Size(),
    ):
        super().__init__(event_shape=event_shape)
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

    def forward(self, features: torch.Tensor) -> Normal:
        std = torch.clamp(F.softplus(self.inv_std) + self.min_std, max=self.max_std)
        return Normal(self.ffn(features), std, self.event_shape)
