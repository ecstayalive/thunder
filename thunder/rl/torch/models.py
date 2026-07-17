from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn as nn

from thunder.core import ThunderModule
from thunder.nn.torch import (
    ConsistentNormalHeadSpec,
    DistributionHeadSpec,
    GruMlpSpec,
    ModelSpec,
)
from thunder.nn.torch.distributions import Distribution, DistributionHead

from ..types import ActorStep


class Actor(ThunderModule):
    """Backbone + distribution head. Pulls its input stream(s) from the obs dict by
    ``obs_keys`` and feeds them to the backbone as positional args."""

    def __init__(
        self, backbone: nn.Module, dist: DistributionHead, obs_keys: Tuple[str, ...]
    ):
        super().__init__()
        self.backbone = backbone
        self.dist = dist
        self.obs_keys = tuple(obs_keys)

    def forward(self, obs: Dict[str, torch.Tensor], carry=None):
        feature, carry = self.backbone(*(obs[k] for k in self.obs_keys), carry)
        return self.dist(feature), carry

    def explore(self, obs: Dict[str, torch.Tensor], carry=None) -> ActorStep:
        """Sample an action from the policy distribution."""
        dist, carry = self.forward(obs, carry)
        dist: Distribution
        action, log_prob = dist.rsample()
        return ActorStep(
            action=action, log_prob=log_prob, distribution=dist, carry=carry
        )

    def determine(self, obs: Dict[str, torch.Tensor], carry=None) -> ActorStep:
        """Take the deterministic (mode) action -- the peak of the policy density."""
        dist, carry = self.forward(obs, carry)
        dist: Distribution
        action = dist.mode()
        log_prob = dist.log_prob(action)
        return ActorStep(
            action=action, log_prob=log_prob, distribution=dist, carry=carry
        )

    def encode(self, obs: Dict[str, torch.Tensor], carry=None):
        """Encode the obs into a feature vector."""
        feature, carry = self.backbone(*(obs[k] for k in self.obs_keys), carry)
        return feature, carry

    def decode(self, feature: torch.Tensor):
        """Decode a feature vector into a distribution."""
        dist = self.dist(feature)
        return dist


class Critic(ThunderModule):
    """Backbone emitting a value. Pulls its input stream(s) from the obs dict by
    ``obs_keys``."""

    def __init__(self, backbone: nn.Module, obs_keys: Tuple[str, ...]):
        super().__init__()
        self.backbone = backbone
        self.obs_keys = tuple(obs_keys)

    def forward(self, obs: Dict[str, torch.Tensor], carry=None):
        return self.backbone(*(obs[k] for k in self.obs_keys), carry)


@dataclass
class ActorSpec:
    """Declarative spec for an :class:`Actor` (backbone + distribution head).

    ``obs_keys`` selects which obs streams feed the backbone (one positional input
    each); ``factory`` resolves their feature shapes from ``obs_shapes``.
    """

    obs_keys: Tuple[str, ...] = ("policy",)
    backbone: ModelSpec = field(
        default_factory=lambda: GruMlpSpec(out_shape=256, mlp_shape=())
    )
    head: DistributionHeadSpec = field(default_factory=ConsistentNormalHeadSpec)

    def factory(self, obs_shapes: Dict[str, int], action_dim: int, **ctx) -> Actor:
        in_shapes = [obs_shapes[k] for k in self.obs_keys]
        backbone = self.backbone.factory(*in_shapes, **ctx)
        head = self.head.factory(self.backbone.out_shape, action_dim, **ctx)
        return Actor(backbone, head, self.obs_keys)


@dataclass
class CriticSpec:
    """Declarative spec for a :class:`Critic` (backbone emitting ``out_shape`` values)."""

    obs_keys: Tuple[str, ...] = ("policy",)
    backbone: ModelSpec = field(
        default_factory=lambda: GruMlpSpec(out_shape=1, mlp_shape=(256, 128))
    )

    def factory(self, obs_shapes: Dict[str, int], **ctx) -> Critic:
        in_shapes = [obs_shapes[k] for k in self.obs_keys]
        return Critic(self.backbone.factory(*in_shapes, **ctx), self.obs_keys)
