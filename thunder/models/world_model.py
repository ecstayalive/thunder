from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from thunder.nn import LinearBlock


class RepresentationModel(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_features: int, state_size: int, hidden_size, dtype=None, device=None):
        super().__init__()

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None): ...


class TransitionModel(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, state_size: int, dtype=None, device=None):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.state_size = state_size

    def forward(self, state: torch.Tensor): ...

    def state0(self, batch_size: int):
        return torch.zeros(batch_size, self.state_size).to(self.factory_kwargs["device"])


class TransitionEnsembleModel(nn.Module):
    def __init__(self, state_size: int, k_head: int = 5, dtype=None, device=None):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.state_size = state_size
        self.k_head = k_head

    def forward(self, state: torch.Tensor) -> Normal:
        """
        Args:
            x (torch.Tensor): shape: (B, H_dim)
        """
        ...
