from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Tuple

from .context import ExecutionContext
from .executor import Executor

if TYPE_CHECKING:
    from .data import Batch
    from .executor.interface import ExecutorProtocol
    from .module import ModelPack
    from .operation import Operation


class GraphAlgorithm(ABC):
    def __init__(
        self,
        models: ModelPack,
        executor: ExecutorProtocol,
        pipeline: Optional[Iterable[Operation]] = None,
    ):
        self.models = models
        self.executor = executor
        if pipeline:
            self.setup_pipeline(pipeline)
        self.ctx: Optional[ExecutionContext] = None

    def build(self, optim_config: Dict[str, Any]) -> None:
        """Build the algorithm by initializing the execution context.
        Args:
            sample_batch (Batch): _description_
            optim_config (Dict[str, Any]): _description_
        """
        self.ctx = self.executor.init(self.models, optim_config)
        self.models = self.ctx.models

    def setup_pipeline(self, pipeline: Iterable[Operation]) -> None:
        """_summary_

        Args:
            pipeline (Iterable[Operation]): _description_
        """
        self.pipeline = pipeline
        self._jit_step: callable = Executor.jit(partial(self._step, pipeline=tuple(self.pipeline)))

    def step(self, batch: Batch, jit: bool = True) -> Dict[str, Any]:
        """_summary_
        Args:
            batch (Batch):
        Raises:
            RuntimeError:
        Returns:
            Dict[str, Any]:
        """
        if self.ctx is None:
            raise RuntimeError("Algorithm not built. Please call .build() first.")
        if self.pipeline is None:
            raise RuntimeError("No pipeline defined for the algorithm.")

        self.ctx = self.ctx.replace(batch=batch)
        if jit:
            self.ctx, metrics = self._jit_step(self.ctx)
        else:
            self.ctx, metrics = self._step(self.ctx, tuple(self.pipeline))
        self.ctx = self.ctx.replace(step=self.ctx.step + 1)
        return metrics

    @staticmethod
    def _step(ctx: ExecutionContext, pipeline: Tuple[Operation, ...]):
        metrics = {}
        for op in pipeline:
            ctx, m = op(ctx)
            metrics.update(m)
        return ctx, metrics


class Agent(GraphAlgorithm):
    def __init__(self, models, executor, pipeline):
        super().__init__(models, executor, pipeline)

    def act(self, obs: Dict[str, Any]): ...

    def explore(self, obs: Dict[str, Any]): ...
