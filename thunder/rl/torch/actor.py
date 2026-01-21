from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from thunder.core import ThunderModule
from thunder.nn.torch.distributions import Distributions as Dist

from ..types import ActorStep


class Actor(ThunderModule):
    """ """

    def __init__(self, backbone: nn.Module, dist: Dist):
        super().__init__()
        self.backbone = backbone
        self.dist = dist

    def reset(self, indices: Optional[Sequence[int]] = None):
        """
        Manages the lifecycle of internal hidden states.
        Args:
            indices: The indices of the environments that hit 'done'.
                     For VectorEnvs, we only clear the memory of finished agents.
        """
        pass

    def forward(
        self,
        embedding: torch.Tensor | Dict[str, torch.Tensor],
        carry=None,
        backbones_kwargs=None,
        dist_kwargs=None,
    ):
        embedding, carry = self.backbone(
            embedding, carry, **backbones_kwargs if backbones_kwargs is not None else {}
        )
        dist = self.dist(embedding, **dist_kwargs if dist_kwargs is not None else {})
        return dist, carry

    def explore(
        self,
        embedding: torch.Tensor | Dict[str, torch.Tensor],
        carry=None,
        backbones_kwargs=None,
        dist_kwargs=None,
    ) -> ActorStep:
        """
        Args:
            *args: Additional arguments.
            **kwargs: Additional arguments.
        Returns:
            action, log_prob, action distribution and other information.
        Raises:
            NotImplementedError: _description_
        """
        dist, carry = self.forward(embedding, carry, backbones_kwargs, dist_kwargs)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        return ActorStep(
            action=action,
            log_prob=log_prob,
            distribution=dist,
            carry=carry,
            backbone_kwargs=backbones_kwargs,
            dist_kwargs=dist_kwargs,
        )

    def decision(
        self,
        embedding: torch.Tensor | Dict[str, torch.Tensor],
        carry=None,
        backbones_kwargs=None,
        dist_kwargs=None,
    ) -> ActorStep:
        """
        Args:
            *args: Additional arguments.
            **kwargs: Additional arguments.
        Returns:
            action, log_prob and action distribution.
        Raises:
            NotImplementedError: _description_
        """
        dist = self.forward(embedding, carry)
        action = dist.mean()
        log_prob = dist.log_prob(action)
        return ActorStep(action=action, log_prob=log_prob, distribution=dist, carry=carry)

    def transform_action(fn: callable, fn_inv: callable, action_step: ActorStep):
        """
        Docstring for transform_action

        :param self: Description
        :param action: Description
        """
        ...
