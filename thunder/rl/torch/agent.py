from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

from thunder.core import Algorithm, Executor, ModelPack, Operation, Pipeline

from .buffer import Buffer

if TYPE_CHECKING:
    from .actor import Actor
    from .operations import Rollout


class Agent(Algorithm):
    """
    Args:
        ABC (_type_): _description_
    """

    def __init__(
        self,
        models: ModelPack,
        buffer: Optional[Buffer] = None,
        executor: Optional[Executor] = None,
        optim_config: Optional[Dict[str, Any]] = None,
        pipeline: Optional[Pipeline | Iterable[Operation]] = None,
        name: str = "agent",
    ):
        super().__init__(models, executor, optim_config, pipeline, name=name)
        self.buffer = buffer if buffer is not None else Buffer()

    def act(self, obs):
        """_summary_

        Args:
            obs (_type_): _description_
        """
        pass

    def collect(self, **kwargs):
        """_summary_

        Args:
            **kwargs: _description_
        """
        pass

    def reset(self, indices):
        """_summary_

        Args:
            indices (int): _description_
        """
        pass
