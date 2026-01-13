from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple

import torch
from torch import nn

from thunder.nn import LinearBlock, RunningNorm1d

__all__ = ["GeneralVNet", "GeneralQNet", "MultiHeadVNet", "MultiHeadQNet", "MultiVNet", "MultiQNet"]


class Critic(ABC, nn.Module): ...


class GeneralVNet(nn.Module):
    """ """

    def __init__(self, kernel: LinearBlock):
        super().__init__()
        self.kernel = kernel
        if self.is_recurrent:
            self.forward = self.rnn_forward

    def evaluate(self, obs, *args, **kwargs):
        v_value, _ = self(obs, *args, **kwargs)
        return v_value

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.kernel(obs), None

    def rnn_forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]]:
        return self.kernel(obs, hidden)


class GeneralQNet(nn.Module):

    def __init__(self, kernel: LinearBlock):
        super().__init__()
        self.kernel = kernel

    def evaluate(self, obs, act, *args, **kwargs):
        q_value, _ = self(obs, act, *args, **kwargs)
        return q_value


class MultiHeadVNet(nn.Module):
    """ """

    def __init__(self, encoder: LinearBlock, decoders: Iterator[LinearBlock]):
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, None]:
        latent = self.encoder(obs)
        output = [m(latent) for m in self.decoders]
        return torch.cat(output, dim=-1), None

    def rnn_forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]]:
        latent, hidden = self.encoder(obs, hidden)
        output = [m(latent) for m in self.decoders]
        return torch.cat(output, dim=-1), hidden


class MultiHeadQNet(nn.Module):
    """ """

    def __init__(self, encoder: LinearBlock, decoders: Iterator[LinearBlock]):
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, None]:
        latent = self.encoder(torch.cat((obs, act), dim=-1))
        output = [m(latent) for m in self.decoders]
        return torch.cat(output, dim=-1), None

    def rnn_forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        hidden: Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]]:
        latent, hidden = self.encoder(torch.cat((obs, act), dim=-1), hidden)
        output = [m(latent) for m in self.decoders]
        return torch.cat(output, dim=-1), hidden


class MultiVNet(nn.Module):
    """ """

    def __init__(self, kernels: Iterator[LinearBlock]):
        super().__init__()
        self.v_nets = nn.ModuleList(kernels)

    def forward(self, obs: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], None]:
        return torch.cat([m(obs) for m in self.v_nets], dim=-1), None

    def rnn_forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[
            Tuple[Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]], ...]
        ] = None,
    ) -> Tuple[
        Tuple[torch.Tensor, ...], Tuple[torch.Tensor | Tuple[torch.Tensor, torch.Tensor], ...]
    ]:
        if hidden is None:
            output, hx = zip(*[m(obs) for m in self.v_nets])
        else:
            output, hx = zip(*[m(obs, h) for m, h in zip(self.v_nets, hidden)])
        return torch.cat(output, dim=-1), tuple(hx)


class MultiQNet(nn.Module):
    """ """

    def __init__(self, kernels: Iterator[LinearBlock]):
        super().__init__()
        self.q_nets = nn.ModuleList(kernels)

    @classmethod
    def make(cls, cfg, obs_dim, action_dim, n_critics=2):
        modules = []
        return cls(modules, obs_dim)

    def evaluate(self, obs: torch.Tensor, act: torch.Tensor, *args, **kwargs):
        q_value, _ = self(obs, act, *args, **kwargs)
        return q_value

    def forward(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], None]:
        q_obs = torch.cat((obs, act), dim=-1)
        return torch.cat([m(q_obs) for m in self.q_nets], dim=-1), None

    def rnn_forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        hidden: Optional[
            Tuple[Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]], ...]
        ] = None,
    ) -> Tuple[
        Tuple[torch.Tensor, ...], Tuple[torch.Tensor | Tuple[torch.Tensor, torch.Tensor], ...]
    ]:
        q_obs = torch.cat((obs, act), dim=-1)
        if hidden is None:
            output, hx = zip(*[m(q_obs) for m in self.q_nets])
        else:
            output, hx = zip(*[m(q_obs, h) for m, h in zip(self.q_nets, hidden)])
        return torch.cat(output, dim=-1), tuple(hx)
