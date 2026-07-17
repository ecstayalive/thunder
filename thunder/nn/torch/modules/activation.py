from typing import ClassVar, Dict, Type

import torch
import torch.nn as nn

from ..functional import squash, swiglu

__all__ = [
    "ACTIVATION_CLS",
    "Cos",
    "Sin",
    "SoftThreshold",
    "Squash",
    "SwiGLU",
    "resolve_activation",
]


class Sin(nn.Module):
    """ """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input)


class Cos(nn.Module):
    """ """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.cos(input)


class Squash(nn.Module):
    """ """

    def __init__(self, dim: int = -1, keepdim: bool = True) -> None:
        """"""
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return squash(input, self.dim, self.keepdim)


class SwiGLU(nn.Module):
    r"""SwiGLU gated activation, ``silu(gate) * value`` over the two halves of
    ``dim``: ``(..., 2d) -> (..., d)``.
    For details: https://arxiv.org/abs/2002.05202

    Args:
        dim: the dimension split into gate and value halves

    Attributes:
        halves_dim: Marks activations that halve their feature dimension, so
            width-aware blocks (e.g. ``LinearBlock``) double the projection
            feeding them.
    """

    halves_dim: ClassVar[bool] = True

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return swiglu(input, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


ACTIVATION_CLS: Dict[str, Type[nn.Module]] = {
    "leaky_relu": nn.LeakyReLU,
    "leakyrelu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "softsign": nn.Softsign,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "tanh": nn.Tanh,
    "hard_tanh": nn.Hardtanh,
    "hardtanh": nn.Hardtanh,
    "swiglu": SwiGLU,
}


def resolve_activation(name: str, allow_halving: bool = False) -> Type[nn.Module]:
    """Resolves an activation name to its class from ``ACTIVATION_CLS``.

    Args:
        name: Registry key, case-insensitive.
        allow_halving: Whether dim-halving activations (``halves_dim=True``,
            e.g. ``swiglu``) are acceptable. Width-aware blocks that double
            the projections feeding the activation opt in; dim-preserving
            contexts keep the default and get a ``ValueError``.
    """
    key = name.lower()
    try:
        activation_cls = ACTIVATION_CLS[key]
    except KeyError:
        raise KeyError(
            f"Unknown activation '{name}', expected one of {sorted(ACTIVATION_CLS)}"
        ) from None
    if not allow_halving and getattr(activation_cls, "halves_dim", False):
        raise ValueError(
            f"Activation '{name}' halves the feature dim and requires a width-aware block"
        )
    return activation_cls


class SoftThreshold(nn.Module):
    """The functional of soft threshold can be seen:
    https://welts.xyz/2022/01/22/soft_thresholding
    In this implementation, we use a small network to learn the threshold value.

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ """
        ...
