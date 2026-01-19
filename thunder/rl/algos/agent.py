from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional

from thunder.core import Algorithm, Executor, ModelPack, Operation
from thunder.rl.buffer.torch import Buffer, Transition
from thunder.utils import ArgBase


class Agent(Algorithm):
    """_summary_

    Args:
        Algorithm (_type_): _description_
    """

    def __init__(self, models: ModelPack, buffer: Buffer, executor: Optional[Executor] = None):
        self.models = models
        self.buffer = buffer
        super().__init__(self.models, executor=executor)
