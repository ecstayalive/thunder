from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Dict, Iterable, Literal, Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import inverse_softplus
from .modules import LinearBlock

LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class DistributionParams(ABC):
    """Typed, detached parameters that fully reconstruct a distribution.

    Concrete subclasses live next to their distribution and carry only that
    family's natural parameters, so ``stats``/``rebuild`` stay type-safe.
    """


class Distribution(ABC):
    has_rsample = True

    def __init__(
        self,
        batch_shape: torch.Size = torch.Size(),
        event_shape: torch.Size = torch.Size(),
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

    def sample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def mean(self) -> torch.Tensor:
        raise NotImplementedError

    def mode(self) -> torch.Tensor:
        raise NotImplementedError

    def entropy(self) -> torch.Tensor:
        raise NotImplementedError

    def std(self) -> torch.Tensor:
        raise NotImplementedError

    def var(self) -> torch.Tensor:
        raise NotImplementedError

    def kl_divergence(self, other: "Distribution") -> torch.Tensor:
        """Event-reduced KL(self || other) between two same-family distributions."""
        raise NotImplementedError

    def stats(self) -> DistributionParams:
        """Detached parameters enough to rebuild this distribution."""
        raise NotImplementedError

    def rebuild(self, params: DistributionParams) -> "Distribution":
        """Build a sibling distribution (same fixed config) from ``params``."""
        raise NotImplementedError


class Normal(Distribution):
    def __init__(
        self,
        loc: torch.Tensor,
        stddev: torch.Tensor,
        event_shape: torch.Size = torch.Size(),
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
        batch_shape = (
            param_shape[: len(param_shape) - event_ndims]
            if event_ndims
            else param_shape
        )
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)
        self.loc = loc
        self.stddev = stddev
        self.variance = stddev.square()
        self._param_shape = param_shape

    def sample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = torch.Size(sample_shape) + self._param_shape
        noise = torch.randn(shape, device=self.loc.device, dtype=self.loc.dtype)
        sample = self.loc + self.stddev * noise
        log_prob = self.log_prob(sample)
        return sample, log_prob

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        log_scale = torch.log(self.stddev)
        log_prob = (
            -0.5 * (((value - self.loc) ** 2) / self.variance + LOG_2PI) - log_scale
        )
        return self._reduce_event(log_prob)

    def mean(self) -> torch.Tensor:
        return self.loc

    def mode(self) -> torch.Tensor:
        return self.loc

    def entropy(self) -> torch.Tensor:
        entropy = 0.5 + 0.5 * LOG_2PI + torch.log(self.stddev)
        return self._reduce_event(entropy)

    def std(self) -> torch.Tensor:
        return self.stddev

    def var(self) -> torch.Tensor:
        return self.variance

    def kl_divergence(self, other: "Normal") -> torch.Tensor:
        kl = (
            torch.log(other.stddev / self.stddev)
            + (self.variance + (self.loc - other.loc).square()) / (2.0 * other.variance)
            - 0.5
        )
        return self._reduce_event(kl)

    def stats(self) -> "NormalParams":
        return NormalParams(loc=self.loc.detach(), scale=self.stddev.detach())

    def rebuild(self, params: "NormalParams") -> "Normal":
        return Normal(params.loc, params.scale, event_shape=self.event_shape)


@dataclass(frozen=True)
class NormalParams(DistributionParams):
    loc: torch.Tensor
    scale: torch.Tensor


class ScaledBeta(Distribution):
    """Beta(concentration1, concentration0) on ``[0, 1]`` mapped by a fixed affine
    transform onto ``[low, high]``.

    For bounded action spaces this stays fully analytic — unlike a squashed Gaussian,
    mean/std/var/entropy/log_prob/KL all have closed forms — so it drops into the PPO
    actor in place of :class:`Normal`. The shared affine map cancels in the KL, leaving
    the unit-interval Beta KL.
    """

    has_rsample = True

    def __init__(
        self,
        concentration1: torch.Tensor,
        concentration0: torch.Tensor,
        low: float = -1.0,
        high: float = 1.0,
        event_shape: torch.Size = torch.Size(),
        eps: float = 1e-3,
    ):
        if high <= low:
            raise ValueError(f"expected low < high, got low={low}, high={high}.")
        c1, c0 = torch.broadcast_tensors(concentration1, concentration0)
        event_shape = torch.Size(event_shape)
        param_shape = torch.Size(c1.shape)
        event_ndims = len(event_shape)
        if event_ndims > len(param_shape) or (
            event_ndims > 0 and param_shape[-event_ndims:] != event_shape
        ):
            raise ValueError(
                f"event_shape {event_shape} must match the rightmost dimensions "
                f"of broadcast parameter shape {param_shape}."
            )
        batch_shape = (
            param_shape[: len(param_shape) - event_ndims]
            if event_ndims
            else param_shape
        )
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)
        self.concentration1 = c1
        self.concentration0 = c0
        self.total = c1 + c0
        self.low = float(low)
        self.high = float(high)
        self.scale = self.high - self.low
        self.log_scale = math.log(self.scale)
        self.eps = eps
        self._param_shape = param_shape

    @staticmethod
    def log_beta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    def rsample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = torch.Size(sample_shape) + self._param_shape
        # Reparameterized Beta via two Gammas: X = G1 / (G1 + G2).
        g1 = torch._standard_gamma(self.concentration1.expand(shape))
        g2 = torch._standard_gamma(self.concentration0.expand(shape))
        sample = self.low + self.scale * g1 / (g1 + g2)
        return sample, self.log_prob(sample)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        x = ((value - self.low) / self.scale).clamp(self.eps, 1.0 - self.eps)
        log_prob = (
            (self.concentration1 - 1.0) * torch.log(x)
            + (self.concentration0 - 1.0) * torch.log1p(-x)
            - self.log_beta(self.concentration1, self.concentration0)
            - self.log_scale
        )
        return self._reduce_event(log_prob)

    def mean(self) -> torch.Tensor:
        return self.low + self.scale * self.concentration1 / self.total

    def mode(self) -> torch.Tensor:
        # True density mode of the unimodal Beta (concentrations > 1, as BetaHead
        # guarantees); the affine map's constant Jacobian carries it through exactly.
        # The clamps degrade gracefully toward the nearest edge for the boundary /
        # uniform cases a directly-constructed ScaledBeta may produce.
        unit_mode = (
            (self.concentration1 - 1.0) / (self.total - 2.0).clamp_min(self.eps)
        ).clamp(0.0, 1.0)
        return self.low + self.scale * unit_mode

    def var(self) -> torch.Tensor:
        unit_var = (
            self.concentration1
            * self.concentration0
            / (self.total.square() * (self.total + 1.0))
        )
        return self.scale * self.scale * unit_var

    def std(self) -> torch.Tensor:
        return self.var().sqrt()

    def entropy(self) -> torch.Tensor:
        a, b = self.concentration1, self.concentration0
        entropy = (
            self.log_beta(a, b)
            - (a - 1.0) * torch.digamma(a)
            - (b - 1.0) * torch.digamma(b)
            + (self.total - 2.0) * torch.digamma(self.total)
            + self.log_scale
        )
        return self._reduce_event(entropy)

    def kl_divergence(self, other: ScaledBeta) -> torch.Tensor:
        pa, pb, qa, qb = (
            self.concentration1,
            self.concentration0,
            other.concentration1,
            other.concentration0,
        )
        kl = (
            self.log_beta(qa, qb)
            - self.log_beta(pa, pb)
            + (pa - qa) * torch.digamma(pa)
            + (pb - qb) * torch.digamma(pb)
            + (qa - pa + qb - pb) * torch.digamma(self.total)
        )
        return self._reduce_event(kl)

    def stats(self) -> ScaledBetaParams:
        return ScaledBetaParams(
            concentration1=self.concentration1.detach(),
            concentration0=self.concentration0.detach(),
        )

    def rebuild(self, params: ScaledBetaParams) -> ScaledBeta:
        return ScaledBeta(
            params.concentration1,
            params.concentration0,
            low=self.low,
            high=self.high,
            event_shape=self.event_shape,
            eps=self.eps,
        )


@dataclass(frozen=True)
class ScaledBetaParams(DistributionParams):
    concentration1: torch.Tensor
    concentration0: torch.Tensor


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
    def __init__(
        self, loc: float | torch.Tensor = 0.0, scale: float | torch.Tensor = 1.0
    ):
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
        super().__init__(
            batch_shape=base_dist.batch_shape, event_shape=base_dist.event_shape
        )
        self.base_dist = base_dist
        self.transforms = tuple(transforms)

    def sample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def mode(self) -> torch.Tensor:
        raise NotImplementedError(
            "mode is not analytically defined for generic transformed distributions."
        )


class DistributionHead(nn.Module):
    def __init__(self, event_shape: torch.Size = torch.Size()):
        super().__init__()
        self.event_shape = torch.Size(event_shape)

    def forward(self, *args, **kwargs) -> Distribution:
        raise NotImplementedError


class NormalHead(DistributionHead):
    __constants__ = [
        "out_size",
        "std_mode",
        "min_std",
        "max_std",
        "min_log_std",
        "max_log_std",
    ]

    def __init__(
        self,
        in_features: int,
        out_size: int,
        hidden_features: Iterable[int] = None,
        event_shape: torch.Size = torch.Size(),
        activation: str = "mish",
        init_std: float = 1.0,
        min_std: float = 0.01,
        max_std: float = 20.0,
        std_mode: Literal["softplus", "exp"] = "exp",
        device=None,
        dtype=None,
    ):
        super().__init__(event_shape=event_shape)
        if std_mode not in ("softplus", "exp"):
            raise ValueError(f"std_mode must be 'softplus' or 'exp', got {std_mode}.")
        if min_std <= 0.0 or max_std <= min_std:
            raise ValueError(
                f"expected 0 < min_std < max_std, got {min_std}, {max_std}."
            )
        self.out_size = out_size
        self.init_std = init_std
        self.min_std = min_std
        self.max_std = max_std
        self.min_log_std = math.log(min_std)
        self.max_log_std = math.log(max_std)
        self.std_mode = std_mode
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
            nn.init.orthogonal_(
                last_layer.weight[: self.out_size], gain=math.sqrt(gain)
            )
            nn.init.orthogonal_(
                last_layer.weight[self.out_size :], gain=0.01 * math.sqrt(gain)
            )
            if self.std_mode == "softplus":
                out_std = max(self.init_std - self.min_std, 0.01)
                std_bias = torch.tensor(
                    out_std,
                    device=last_layer.weight.device,
                    dtype=last_layer.weight.dtype,
                )
                std_bias = inverse_softplus(std_bias)
            else:
                init_std = min(max(self.init_std, self.min_std), self.max_std)
                std_bias = torch.tensor(
                    math.log(init_std),
                    device=last_layer.weight.device,
                    dtype=last_layer.weight.dtype,
                )
            last_layer.bias[self.out_size :].fill_(std_bias)
            last_layer.bias[: self.out_size].fill_(0.0)

    def forward(self, features: torch.Tensor) -> Normal:
        mean, std_param = torch.chunk(self.ffn(features), 2, -1)
        if self.std_mode == "softplus":
            std = torch.clamp(F.softplus(std_param) + self.min_std, max=self.max_std)
        else:
            log_std = torch.clamp(std_param, min=self.min_log_std, max=self.max_log_std)
            std = torch.exp(log_std)
        return Normal(mean, std, self.event_shape)


class TransformedDistHead(DistributionHead):
    def __init__(
        self,
        in_features: int,
        out_size: int,
        hidden_features: Iterable[int] = None,
        event_shape: torch.Size = torch.Size(),
        activation: str = "mish",
        init_std: float = 1.0,
        min_std: float = 0.01,
        max_std: float = 20.0,
        std_mode: Literal["softplus", "exp"] = "exp",
        transforms: Sequence[Transform] | None = None,
        consistent_std: bool = False,
        device=None,
        dtype=None,
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
            std_mode=std_mode,
            device=device,
            dtype=dtype,
            event_shape=event_shape,
        )
        self.transforms = (
            tuple(transforms) if transforms is not None else (TanhTransform(),)
        )

    def forward(self, features: torch.Tensor) -> TransformedDistribution:
        return TransformedDistribution(self.base_head(features), self.transforms)


class ConsistentNormalHead(DistributionHead):
    __constants__ = ["std_mode", "min_std", "max_std", "min_log_std", "max_log_std"]

    def __init__(
        self,
        in_features: int,
        out_size: int,
        hidden_features: Iterable[int] = None,
        event_shape: torch.Size = torch.Size(),
        activation: str = "mish",
        init_std: float = 1.0,
        min_std: float = 0.01,
        max_std: float = 20.0,
        std_mode: Literal["softplus", "exp"] = "exp",
        device=None,
        dtype=None,
    ):
        super().__init__(event_shape=event_shape)
        if std_mode not in ("softplus", "exp"):
            raise ValueError(f"std_mode must be 'softplus' or 'exp', got {std_mode}.")
        if min_std <= 0.0 or max_std <= min_std:
            raise ValueError(
                f"expected 0 < min_std < max_std, got {min_std}, {max_std}."
            )
        self.init_std = init_std
        self.min_std = min_std
        self.max_std = max_std
        self.min_log_std = math.log(min_std)
        self.max_log_std = math.log(max_std)
        self.std_mode = std_mode
        self.ffn = LinearBlock(
            in_features=in_features,
            out_features=out_size,
            hidden_features=hidden_features,
            activation=activation,
            activate_output=False,
            device=device,
            dtype=dtype,
        )
        self.std_param = nn.Parameter(
            torch.ones(out_size, device=device, dtype=dtype) * math.log(init_std)
        )
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0):
        if self.std_mode == "softplus":
            out_std = max(self.init_std - self.min_std, 0.01)
            std_bias = torch.tensor(
                out_std, device=self.std_param.device, dtype=self.std_param.dtype
            )
            std_bias = inverse_softplus(std_bias)
        else:
            init_std = min(max(self.init_std, self.min_std), self.max_std)
            std_bias = torch.tensor(
                math.log(init_std),
                device=self.std_param.device,
                dtype=self.std_param.dtype,
            )
        with torch.no_grad():
            last_layer: nn.Linear = self.ffn.linear_block[-1]
            nn.init.orthogonal_(last_layer.weight, gain=math.sqrt(gain))
            last_layer.bias.fill_(0.0)
            self.std_param.fill_(std_bias)

    def forward(self, features: torch.Tensor) -> Normal:
        if self.std_mode == "softplus":
            std = torch.clamp(
                F.softplus(self.std_param) + self.min_std, max=self.max_std
            )
        else:
            log_std = torch.clamp(
                self.std_param, min=self.min_log_std, max=self.max_log_std
            )
            std = torch.exp(log_std)
        return Normal(self.ffn(features), std, self.event_shape)


class BetaHead(DistributionHead):
    """Emit a :class:`ScaledBeta` over ``[low, high]`` for bounded action spaces."""

    __constants__ = ["out_size", "low", "high", "concentration_offset"]

    def __init__(
        self,
        in_features: int,
        out_size: int,
        hidden_features: Iterable[int] = None,
        event_shape: torch.Size = torch.Size(),
        activation: str = "mish",
        low: float = -1.0,
        high: float = 1.0,
        concentration_offset: float = 1.0,
        device=None,
        dtype=None,
    ):
        super().__init__(event_shape=event_shape)
        if high <= low:
            raise ValueError(f"expected low < high, got low={low}, high={high}.")
        if concentration_offset < 1.0:
            raise ValueError("concentration_offset must be >= 1 for a unimodal Beta.")
        self.out_size = out_size
        self.low = float(low)
        self.high = float(high)
        self.concentration_offset = float(concentration_offset)
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
            nn.init.orthogonal_(last_layer.weight, gain=0.01 * math.sqrt(gain))
            last_layer.bias.fill_(0.0)

    def forward(self, features: torch.Tensor) -> ScaledBeta:
        c1, c0 = torch.chunk(self.ffn(features), 2, -1)
        c1 = F.softplus(c1) + self.concentration_offset
        c0 = F.softplus(c0) + self.concentration_offset
        return ScaledBeta(c1, c0, self.low, self.high, self.event_shape)


@dataclass
class DistributionHeadSpec(ABC):
    """Base contract for a self-building distribution head, colocated with the heads."""

    class_type: ClassVar[Type[DistributionHead]]

    @abstractmethod
    def factory(self, in_shape: int, action_dim: int, **ctx) -> DistributionHead:
        raise NotImplementedError


@dataclass
class NormalHeadSpec(DistributionHeadSpec):
    hidden_features: Tuple[int, ...] = (256, 128)
    activation: str = "mish"
    init_std: float = 0.5
    min_std: float = 0.01
    max_std: float = 10.0
    std_mode: Literal["softplus", "exp"] = "exp"

    class_type: ClassVar[Type[DistributionHead]] = NormalHead

    def factory(self, in_shape: int, action_dim: int, **ctx) -> NormalHead:
        return self.class_type(
            in_features=in_shape,
            out_size=action_dim,
            hidden_features=list(self.hidden_features),
            event_shape=torch.Size([action_dim]),
            activation=self.activation,
            init_std=self.init_std,
            min_std=self.min_std,
            max_std=self.max_std,
            std_mode=self.std_mode,
        )


@dataclass
class ConsistentNormalHeadSpec(DistributionHeadSpec):
    hidden_features: Tuple[int, ...] = (256, 128)
    activation: str = "mish"
    init_std: float = 0.5
    min_std: float = 0.01
    max_std: float = 10.0
    std_mode: Literal["softplus", "exp"] = "exp"

    class_type: ClassVar[Type[DistributionHead]] = ConsistentNormalHead

    def factory(self, in_shape: int, action_dim: int, **ctx) -> ConsistentNormalHead:
        return self.class_type(
            in_features=in_shape,
            out_size=action_dim,
            hidden_features=list(self.hidden_features),
            event_shape=torch.Size([action_dim]),
            activation=self.activation,
            init_std=self.init_std,
            min_std=self.min_std,
            max_std=self.max_std,
            std_mode=self.std_mode,
        )


@dataclass
class TransformedDistHeadSpec(DistributionHeadSpec):
    hidden_features: Tuple[int, ...] = (256, 128)
    activation: str = "mish"
    init_std: float = 0.5
    min_std: float = 0.01
    max_std: float = 10.0
    std_mode: Literal["softplus", "exp"] = "exp"
    consistent_std: bool = False

    class_type: ClassVar[Type[DistributionHead]] = TransformedDistHead

    def factory(self, in_shape: int, action_dim: int, **ctx) -> TransformedDistHead:
        return self.class_type(
            in_features=in_shape,
            out_size=action_dim,
            hidden_features=list(self.hidden_features),
            event_shape=torch.Size([action_dim]),
            activation=self.activation,
            init_std=self.init_std,
            min_std=self.min_std,
            max_std=self.max_std,
            std_mode=self.std_mode,
            consistent_std=self.consistent_std,
        )


@dataclass
class BetaHeadSpec(DistributionHeadSpec):
    hidden_features: Tuple[int, ...] = (256, 128)
    activation: str = "mish"
    low: float = -1.0
    high: float = 1.0
    concentration_offset: float = 1.0

    class_type: ClassVar[Type[DistributionHead]] = BetaHead

    def factory(self, in_shape: int, action_dim: int, **ctx) -> BetaHead:
        return self.class_type(
            in_features=in_shape,
            out_size=action_dim,
            hidden_features=list(self.hidden_features),
            event_shape=torch.Size([action_dim]),
            activation=self.activation,
            low=self.low,
            high=self.high,
            concentration_offset=self.concentration_offset,
        )
