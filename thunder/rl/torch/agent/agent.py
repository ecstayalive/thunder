from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Dict, Iterable, Literal, Optional, TYPE_CHECKING

from thunder.core import Algorithm, Executor, ModelPack, Operation, Pipeline

from ..buffer import Buffer

if TYPE_CHECKING:
    from thunder.env.env import ThunderEnv


@dataclass
class AgentSpec:
    name: str
    device: str = "cuda:0"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"


class Agent(Algorithm):
    """ """

    def __init__(
        self,
        models: ModelPack,
        buffer: Optional[Buffer] = None,
        executor: Optional[Executor] = None,
        optim_config: Optional[Dict[str, Any]] = None,
        pipeline: Optional[Pipeline | Iterable[Operation]] = None,
        name: str = "",
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

    @classmethod
    def factory(cls, env: ThunderEnv, spec: AgentSpec, **kwargs) -> Agent:
        """_summary_

        Args:
            env (ThunderEnvWrapper): _description_
            **kwargs: _description_
        """
        pass
