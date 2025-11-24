import torch
from torch import Tensor

__all__ = ["squash"]


@torch.jit.script
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


@torch.jit.script
def position_embedding_2d(channels: int, height: int, width: int, temperature: int = 10000):
    """This is position embedding for spatial attention.
    For spatial attention, d_model can be considered as the channel dimension.
    Half channel stores its height information, half channel stores its width information.
    ..math:

    """
    if channels % 2 != 0:
        raise ValueError("Channels must be divisible by 2")
    half_dim = channels // 2
    omega = torch.arange(half_dim // 2, dtype=torch.float32)
    omega /= half_dim // 2
    omega = 1.0 / (temperature**omega)
    y = torch.arange(height, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32)
    y_ang = torch.outer(y, omega)
    x_ang = torch.outer(x, omega)
    pe_y = torch.zeros(height, half_dim, dtype=torch.float32)
    pe_y[:, 0::2] = y_ang.sin()
    pe_y[:, 1::2] = y_ang.cos()
    pe_x = torch.zeros(width, half_dim, dtype=torch.float32)
    pe_x[:, 0::2] = x_ang.sin()
    pe_x[:, 1::2] = x_ang.cos()
    pe_full = torch.cat(
        [pe_y.unsqueeze(1).expand(-1, width, -1), pe_x.unsqueeze(0).expand(height, -1, -1)], dim=-1
    )
    # [H, W, C] -> [C, H, W]
    return pe_full.permute(2, 0, 1)
