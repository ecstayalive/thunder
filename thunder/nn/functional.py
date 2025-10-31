import torch
from torch import Tensor

__all__ = ["squash"]


def squash(input: Tensor, dim: int = -1, keepdim: bool = True) -> Tensor:
    r"""Non-Linear activation function used in Capsule Network

    Args:
        input: input feature vectors
        dim: the squashing axis

    Returns:
        The squashing feature vectors

    Examples:

        >>> input = torch.randn(1, 10, 8)
        >>> output = squash(input)
        >>> print(output.shape)
            torch.Size([1, 10, 8])
    """
    norm = torch.norm(input, p=2, dim=dim, keepdim=keepdim)
    scale = norm / (1 + norm**2)

    return scale * input
