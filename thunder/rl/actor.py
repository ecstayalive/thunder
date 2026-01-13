from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Mapping

import torch
import torch.nn as nn

from thunder.nn import LinearBlock, Normalization, RunningNorm1d
from thunder.rl.distributions import *


class Actor(ABC, nn.Module):
    """Abstract base class for actor."""

    @abstractmethod
    def explore(self, *args, **kwargs):
        """
        Args:
            *args: Additional arguments.
            **kwargs: Additional arguments.
        """
        ...

    @abstractmethod
    def act(self, *args, **kwargs): ...


class GeneralActor(nn.Module):
    """ """

    def __init__(self, encoder: LinearBlock, action_dec: Distribution) -> None:
        """
        `enc` is abbreviation for `encoder`, and `dec` is abbreviation for `decoder`.
        Args:
            encoder: The encoder module.
            action_dec: The action decoder module.
        """
        super().__init__()
        self.encoder = encoder
        self.action_dec = action_dec

    @classmethod
    def make(cls, cfg, obs_dim: int, action_dim: int, with_actor_info: bool = False):
        cfg = cfg.copy()
        encoder, encoder_info = NetFactory.make(obs_dim, modules_info=cfg["encoder"])
        if encoder_info.is_recurrent:
            encoder = DimAdaptRMlp(encoder)
        distribution = make_distribution(
            action_dim,
            latent_dim=encoder_info.out_features,
            obs_dim=obs_dim,
            **cfg["distribution"],
        )
        obj = cls(encoder, distribution)
        if with_actor_info:
            return obj, encoder_info
        return obj

    def explore(self, obs: torch.Tensor, hx=None, sample=True, *, return_joint_prob: bool = True):
        if self.is_recurrent:
            latent, hidden = self.encoder(obs, hx)
        else:
            latent, hidden = self.encoder(obs), None
        if isinstance(self.action_dec, SDE):
            latent = (latent, obs)
        if not sample:
            mean, std = self.action_dec(latent)
            return (mean, std), hidden
        mean, std, action, log_prob = self.action_dec.sample_log_prob(latent, return_joint_prob)
        return (mean, std), (action, log_prob), hidden

    def act_log_prob(self, obs, *args, **kwargs):
        _, (action, log_prob), _ = self.explore(obs, sample=True, *args, **kwargs)
        return action, log_prob

    def act_stochastic(self, obs, *args, **kwargs):
        _, (action, _), hidden = self.explore(obs, sample=True, *args, **kwargs)
        return action, hidden

    def calc_log_prob_entropy(self, mean, std, sample, *, return_joint_prob: bool = True):
        log_prob = self.action_dec.calc_log_prob(mean, std, sample, return_joint_prob)
        entropy = self.action_dec.calc_entropy(std, return_joint_prob)
        return log_prob, entropy

    def forward(self, obs: torch.Tensor, *args, **kwargs):
        latent = self.encoder(obs)
        return self.action_dec.determine(latent), None

    def rnn_forward(self, obs: torch.Tensor, hx=None):
        latent, hidden = self.encoder(obs, hx)
        return self.action_dec.determine(latent), hidden

    @property
    def obs_shape(self):
        if hasattr(self.encoder, "in_features"):  # RecurrentMlp
            return [self.encoder.in_features]
        elif hasattr(self.encoder, "input_size"):  # nn.RNNBase
            return [self.encoder.input_size]
        return self.encoder.input_shape

    @property
    def action_shape(self):
        return self.action_dec.action_shape

    def set_exploration_std(self, std):
        if hasattr(self.action_dec, "set_std"):
            self.action_dec.set_std(std)

    def clamp_exploration_std(self, min=None, max=None, indices: slice = None):
        if isinstance(self.action_dec, Gaussian):
            self.action_dec.clamp_std(min, max, indices)

    def resample_distribution(self):
        if isinstance(self.action_dec, SDE | GeneralizedSDE):
            self.action_dec.resample_weights()

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Override ``state_dict`` method in ``torch.nn.Module``
        This method support ``torch.compile`` function
        .. warning::
            Currently ``state_dict()`` also accepts positional arguments for
            ``destination``, ``prefix`` and ``keep_vars`` in order. However,
            this is being deprecated and keyword arguments will be enforced in
            future releases.
            For details: https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.state_dict
        """
        if destination is None:
            destination = OrderedDict()
        destination.update(
            [
                ("encoder", self.encoder.state_dict()),
                ("action_dec", self.action_dec.state_dict()),
            ]
        )
        return destination

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        self.encoder.load_state_dict(state_dict.get("encoder", {}), strict=strict, assign=assign)
        if "distribution" in state_dict:  # for compatibility
            self.action_dec.load_state_dict(
                state_dict["distribution"], strict=strict, assign=assign
            )
        else:
            self.action_dec.load_state_dict(
                state_dict.get("action_dec", {}), strict=strict, assign=assign
            )

    def restore(self, model_path, prefix="actor"):
        device = next(self.parameters()).device
        state_dict = torch.load(model_path, map_location=device)
        if prefix is not None:
            state_dict = state_dict[prefix]
        self.load_state_dict(state_dict)
        return self

    def scriptable(
        self,
        normalizer: Normalization | RunningNorm1d = None,
        denormalizer: Normalization | RunningNorm1d = None,
    ):
        encoder = self.encoder
        if self.is_recurrent:
            encoder = self.encoder.scriptable()

    def inference(self):
        return InferenceActor(self)


class InferenceActor(nn.Module):
    def __init__(self, actor: GeneralActor):
        super().__init__()
        self.actor = actor
        self._hidden = None
        param = next(self.actor.parameters())
        self._datatype, self._device = param.dtype, param.device

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, *, stochastic=False):
        x = torch.as_tensor(x, dtype=self._datatype, device=self._device)
        if stochastic:
            y, self._hidden = self.actor.act_stochastic(x, self._hidden)
        else:
            y, self._hidden = self.actor(x, self._hidden)
        return y.cpu().numpy()

    @torch.inference_mode()
    def reset(self, indices=None):
        if indices is None or self._hidden is None:
            self._hidden = None
            return
        if isinstance(self._hidden, torch.Tensor):  # gru
            self._hidden[..., indices, :] = 0.0
        else:
            for hidden in self._hidden:  # lstm
                hidden[..., indices, :] = 0.0
