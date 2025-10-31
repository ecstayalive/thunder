import torch
import torch.nn as nn

from ..functional import squash

__all__ = ["Sin", "Cos", "Squash"]


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
