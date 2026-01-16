from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Tuple

from .context import ExecutionContext
from .executor import Executor

if TYPE_CHECKING:
    from .data import Batch
    from .executor.interface import Executor
    from .module import ModelPack
    from .operation import Operation


class Algorithm(ABC):
    def __init__(
        self,
        models: ModelPack,
        executor: Optional[Executor] = None,
        optim_config: Optional[Dict[str, Any]] = None,
        pipeline: Optional[Iterable[Operation]] = None,
    ):
        self.models = models
        self.executor = executor if executor is not None else Executor()
        if optim_config is not None:
            self.build(optim_config)
        if pipeline is not None:
            self.setup_pipeline(pipeline)

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
        self._jit_step: Callable[[ExecutionContext], Tuple[ExecutionContext, Dict]] = Executor.jit(
            partial(self._step, pipeline=tuple(self.pipeline))
        )

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
