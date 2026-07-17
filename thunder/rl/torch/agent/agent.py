from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Dict, Iterable, Literal, Optional, TYPE_CHECKING

from thunder.core import Algorithm, AttrData, Executor, ModelPack, Operation, Pipeline

from ..buffer import Buffer

if TYPE_CHECKING:
    from thunder.env.env import ThunderEnv


@dataclass
class AgentSpec:
    name: str
    device: str = "cuda:0"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    compile: bool = False
    resume: str | None = None


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

    def act(self, obs, explore: bool = True):
        """_summary_

        Args:
            obs (_type_): _description_
        """
        raise NotImplementedError

    def infer(self, obs, explore: bool = False):
        """Inference-only action selection: no value estimation and no
        trajectory bookkeeping. Used by evaluation/play pipelines.

        Args:
            obs: Batched observations from the environment.
            explore: Sample from the policy distribution instead of its mode.
        """
        raise NotImplementedError

    def collect(self, **kwargs):
        """_summary_

        Args:
            **kwargs: _description_
        """
        return

    def reset(self, dones):
        """_summary_

        Args:
            dones: Boolean mask of environments whose agent state should reset.
        """
        return

    def snapshot(self) -> AttrData:
        return AttrData()

    @classmethod
    def factory(cls, env: ThunderEnv, spec: AgentSpec, **kwargs) -> Agent:
        """_summary_

        Args:
            env (ThunderEnvWrapper): _description_
            **kwargs: _description_
        """
        raise NotImplementedError
