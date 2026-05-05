from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from thunder.core import ThunderModule
from thunder.nn.torch.distributions import DistributionHead
from thunder.nn.torch.models import RepresentModel

from ..types import ActorStep


class Actor(ThunderModule):
    """ """

    def __init__(self, backbone: nn.Module, dist: DistributionHead):
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
        backbone_kwargs=None,
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
        dist, carry = self.forward(embedding, carry, backbone_kwargs, dist_kwargs)
        action, log_prob = dist.rsample()
        return ActorStep(
            action=action,
            log_prob=log_prob,
            distribution=dist,
            carry=carry,
            backbone_kwargs=backbone_kwargs,
            dist_kwargs=dist_kwargs,
        )

    def determine(
        self,
        embedding: torch.Tensor | Dict[str, torch.Tensor],
        carry=None,
        backbone_kwargs=None,
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
        dist, carry = self.forward(embedding, carry, backbone_kwargs, dist_kwargs)
        action = dist.mean()
        log_prob = dist.log_prob(action)
        return ActorStep(action=action, log_prob=log_prob, distribution=dist, carry=carry)
