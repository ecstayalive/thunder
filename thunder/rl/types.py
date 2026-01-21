import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

_BACKEND = os.getenv("THUNDER_BACKEND", "torch")


@dataclass(slots=True)
class ActorStep:
    """
    Standardized output for Environment Interaction.
    Args:
        action:
        log_prob:
        distribution: torch.distributions for `torch` and jax.distributions for `jax`
        carry:
        extra:
    """

    action: Any
    log_prob: Optional[Any] = None
    distribution: Optional[Any] = None
    carry: Optional[Any] = None
    backbone_kwargs: Optional[Dict] = field(default_factory=dict)
    dist_kwargs: Optional[Dict] = field(default_factory=dict)


if _BACKEND == "torch":
    import torch.utils._pytree as pytree

elif _BACKEND == "jax":
    import jax

    def _flatten_actor_step(node: ActorStep):
        children = (
            node.action,
            node.log_prob,
            node.distribution,
            node.carry,
            node.backbone_kwargs,
            node.dist_kwargs,
        )
        aux_data = None
        return children, aux_data

    def _unflatten_actor_step(aux_data, children):
        return ActorStep(
            action=children[0],
            log_prob=children[1],
            distribution=children[2],
            carry=children[3],
            backbone_kwargs=children[4],
            dist_kwargs=children[5],
        )

    jax.tree_util.register_pytree_node(ActorStep, _flatten_actor_step, _unflatten_actor_step)
