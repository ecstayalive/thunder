import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

_BACKEND = os.getenv("THUNDER_BACKEND", "torch")


@dataclass(slots=True)
class ActorStep:
    """
    Standardized output for Environment Interaction.
    Args:
        action:
        log_prob:
        distribution: torch.distributions for `torch` and jax.distributions for `jax`
        hidden:
        extra:
    """

    action: Any
    log_prob: Optional[Any] = None
    distribution: Optional[Any] = None
    hidden: Optional[Any] = None
    extra: Optional[dict] = field(default_factory=dict)


class Actor(ABC):
    """ """

    @abstractmethod
    def reset(self, indices: Optional[Sequence[int]] = None):
        """
        Manages the lifecycle of internal hidden states.
        Args:
            indices: The indices of the environments that hit 'done'.
                     For VectorEnvs, we only clear the memory of finished agents.
        """
        raise NotImplementedError()

    @abstractmethod
    def explore(self, *args, **kwargs) -> ActorStep:
        """
        Args:
            *args: Additional arguments.
            **kwargs: Additional arguments.
        Returns:
            Tuple[Any, Tuple[Any, Any], Any]: action, log_prob, action distribution and other information.
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    @abstractmethod
    def decision(self, *args, **kwargs) -> Tuple[Any, Tuple[Any, Any]]:
        """
        Args:
            *args: Additional arguments.
            **kwargs: Additional arguments.
        Returns:
            Tuple[Any, Tuple[Any, Any]]: action, log_prob and action distribution.
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()


if _BACKEND == "torch":
    import torch.utils._pytree as pytree

elif _BACKEND == "jax":
    import jax

    def _flatten_actor_step(node: ActorStep):
        children = (node.action, node.log_prob, node.distribution, node.hidden, node.extra)
        aux_data = None
        return children, aux_data

    def _unflatten_actor_step(aux_data, children):
        return ActorStep(
            action=children[0],
            log_prob=children[1],
            distribution=children[2],
            hidden=children[3],
            extra=children[4],
        )

    jax.tree_util.register_pytree_node(ActorStep, _flatten_actor_step, _unflatten_actor_step)
