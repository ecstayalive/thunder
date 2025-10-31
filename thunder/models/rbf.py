import torch
import torch.nn as nn
import torch.nn.functional as F


class Rbf(nn.Module):
    """Radial basis function network, a universal approximator on
    a compact subset of :math:`R^n`.
    For details: https://en.wikipedia.org/wiki/Radial_basis_function_network
    Args:
        in_features:
        out_features:
        kernel:
        normalized:
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel: nn.Module,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

    def reset_parameters(self):
        """"""
        ...


class GaussianRbf(nn.Module):
    """Radial basis function network, which uses Gaussian kernel
    For details: https://en.wikipedia.org/wiki/Radial_basis_function_network
    Args:
        in_features:
        out_features:
        kernel_num:
        normalized:
        device:
        dtype:

    Input:
        input:

    Examples:
        >>> rbf_net = GaussianRbf(10, 5, 20)
        >>> x = torch.randn(9, 10)
        >>> y = rbf_net(x)
        >>> print(y.shape)
            torch.Size([9, 5])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_num: nn.Module,
        normalized: bool = False,
        norm_order: float = 2,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.norm_order = norm_order
        self.normalized = normalized
        self.basis_center = nn.Parameter(torch.empty((kernel_num, in_features), **factory_kwargs))
        # TODO: The selection of this part of parameters requires more experiments
        self.beta_log = nn.Parameter(torch.empty(kernel_num, **factory_kwargs))
        self.weight = nn.Parameter(torch.empty((out_features, kernel_num), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.basis_center)
        nn.init.normal_(self.beta_log)
        nn.init.xavier_normal_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        distance = self.basis_center - input.unsqueeze(-2)
        distance = torch.norm(distance, self.norm_order, -1)
        if self.normalized:
            distance = distance / torch.sum(distance, -1, keepdim=True)
        probability = torch.exp(-torch.exp(self.beta_log) * distance)
        return F.linear(probability, self.weight)
