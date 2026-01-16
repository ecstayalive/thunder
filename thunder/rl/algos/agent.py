from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional

from thunder.core import Algorithm, Executor, ModelPack, Operation


class Agent(Algorithm, ABC):
    def __init__(
        self,
        models: ModelPack,
        executor: Optional[Executor] = None,
        opt_config: Optional[Dict[str, Any]] = None,
        pipeline: Optional[Iterable[Operation]] = None,
    ):
        super().__init__(models, executor, opt_config, pipeline)

    @abstractmethod
    def interact(env):
        raise NotImplementedError
