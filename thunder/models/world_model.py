from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from thunder.nn import LinearBlock
from thunder.nn.mapping import ACTIVATION_CLS_NAME


class RSSM(nn.Module):
    def __init__(
        self, r_model: RecurrentModel, t_model: TransitionModel, repr_model: RepresentationModel
    ):
        super().__init__()
        self.r_model = r_model
        self.t_model = t_model
        self.repr_model = repr_model

    def recurrent_model_input_init(self, batch_size):
        return self.t_model.state0(batch_size), self.r_model.deterministic0(batch_size)


class RecurrentModel(nn.Module):
    def __init__(
        self,
        deterministic_size: int,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        activation: str = "relu",
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.deterministic_size = deterministic_size
        self.state_size = state_size
        self.action_size = action_size
        activation_fn_name = ACTIVATION_CLS_NAME[activation.lower()]
        self.activation = getattr(nn, activation_fn_name)
        self.linear = nn.Linear(state_size + action_size, hidden_size, **self.factory_kwargs)
        self.recurrent = nn.GRUCell(hidden_size, self.deterministic_size, **self.factory_kwargs)

    def forward(self, s_embed: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor):
        x = torch.cat((s_embed, action), 1)
        x = self.activation(self.linear(x))
        x = self.recurrent(x, hidden)
        return x

    def deterministic0(self, batch_size: int):
        return torch.zeros(batch_size, self.deterministic_size, **self.factory_kwargs)


class TransitionModel(nn.Module):
    def __init__(
        self,
        deterministic_size: int,
        state_size: int,
        hidden_size: Iterable[int] = [128],
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.state_size = state_size
        self.deterministic_size = deterministic_size
        self.network = LinearBlock(
            deterministic_size, hidden_size, state_size * 2, **self.factory_kwargs
        )

    def forward(self, x: torch.tensor):
        x = self.network(x)
        mu, log_std = torch.chunk(x, 2, dim=-1)
        return mu, log_std

    def state0(self, batch_size):
        return torch.zeros(batch_size, self.state_size).to(self.factory_kwargs["device"])


class TransitionEnsembleModel(nn.Module):
    def __init__(
        self,
        deterministic_size: int,
        state_size: int,
        hidden_size: Iterable[int] = [128],
        k_head: int = 5,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.state_size = state_size
        self.deterministic_size = deterministic_size
        self.k_head = k_head
        self.network = LinearBlock(
            deterministic_size, hidden_size, 2 * self.k_head * state_size, **self.factory_kwargs
        )

    def forward(self, x: torch.Tensor) -> Normal:
        """
        Args:
            x (torch.Tensor): shape: (B, H_dim)
        """
        x = self.network(x)

        B = x.shape[0]
        network_out = self.network(x)
        # (B, K, S_dim * 2)
        reshaped_out = network_out.view(B, self.k, self.stochastic_size * 2)
        # (K, B, S_dim * 2)
        transposed_out = reshaped_out.permute(1, 0, 2)
        mu, log_std = torch.chunk(transposed_out, 2, -1)

        return mu, log_std


class RepresentationModel(nn.Module):
    def __init__(
        self,
        deterministic_size: int,
        o_embed_size: int,
        state_size: int,
        hidden_size: Iterable[int] = [128],
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.factory_kwargs = {"dtype": dtype, "device": device}
        self.deterministic_size = deterministic_size
        self.state_size = state_size
        self.o_embed_size = o_embed_size

        self.network = LinearBlock(
            self.o_embed_size + self.deterministic_size, self.state_size * 2, hidden_size
        )

    def forward(self, o_embed: torch.Tensor, deterministic):
        x = self.network(torch.cat([deterministic, o_embed], dim=-1))
        mu, log_std = torch.chunk(x, 2 - 1)
        return mu, log_std
