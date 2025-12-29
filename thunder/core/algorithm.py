from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from .context import ExecutionContext

if TYPE_CHECKING:
    from .data import Batch, ModelPack
    from .executor.interface import ExecutorProtocol
    from .operation import Operation


class GraphAlgorithm(ABC):
    def __init__(
        self,
        models: ModelPack,
        executor: ExecutorProtocol,
        pipeline: Optional[Iterable[Operation]] = None,
        ctx: Optional[ExecutionContext] = None,
    ):
        self.models = models
        self.executor = executor
        self.ctx: Optional[ExecutionContext] = ctx
        self.pipeline: Operation[Iterable[Operation]] = pipeline

    def build(self, sample_batch: Batch, optim_config: Dict[str, Any]) -> None:
        """Build the algorithm by initializing the execution context.
        Args:
            sample_batch (Batch): _description_
            optim_config (Dict[str, Any]): _description_
        """
        self.ctx = self.executor.init(self.models, sample_batch, optim_config)
        self.models = self.ctx.models

    def setup_pipeline(self, pipeline: Iterable[Operation]) -> None:
        """_summary_

        Args:
            pipeline (Iterable[Operation]): _description_
        """
        self.pipeline = pipeline

    def step(self, batch: Batch) -> Dict[str, Any]:
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
        metrics = {}
        self.ctx = self.ctx.replace(batch=batch)
        for op in self.pipeline:
            self.ctx, m = op(self.ctx)
            metrics.update(m)
        self.ctx = self.ctx.replace(step=self.ctx.step + 1)
        return metrics


class Agent(GraphAlgorithm):
    def __init__(self, models, executor, pipeline):
        super().__init__(models, executor, pipeline)

    def act(self, obs: Dict[str, Any]): ...
    def explore(self, obs: Dict[str, Any]): ...
