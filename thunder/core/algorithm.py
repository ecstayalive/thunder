from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

from .executor import Executor
from .operation import Pipeline

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
        pipeline: Optional[Pipeline | Iterable[Operation]] = None,
    ):
        self.models = models
        self.ctx = None
        self.executor = executor if executor is not None else Executor()
        self.setup_pipeline(pipeline)
        if optim_config is not None:
            self.build(optim_config)

    def build(self, optim_config: Dict[str, Any]) -> None:
        """Build the algorithm by initializing the execution context.
        Args:
            sample_batch (Batch): _description_
            optim_config (Dict[str, Any]): _description_
        """
        self.ctx = self.executor.init(self.models, optim_config)
        self.optim_config = optim_config
        self.models = self.ctx.models

    def setup_pipeline(self, pipeline: Iterable[Operation], jit: bool = False) -> None:
        """_summary_

        Args:
            pipeline (Iterable[Operation]): _description_
        """
        if pipeline is None:
            self.pipeline = Pipeline([], jit=False)
        elif isinstance(pipeline, Pipeline):
            self.pipeline = pipeline
        else:
            self.pipeline = Pipeline(pipeline, name="", jit=jit)

    def step(self, batch: Optional[Batch] = None) -> Dict[str, Any]:
        """_summary_
        Raises:
            RuntimeError:
        Returns:
            Dict[str, Any]:
        """
        if self.ctx is None:
            raise RuntimeError("Algorithm not built. Please call .build() first.")
        if self.pipeline is None:
            raise RuntimeError(
                "No pipeline defined for the algorithm. Please call .setup_pipeline() first."
            )
        if batch is not None:
            self.ctx = self.ctx.replace(batch=batch)
        with self.ctx.manager:
            self.ctx, metrics = self.pipeline(self.ctx)
        self.ctx = self.ctx.replace(step=self.ctx.step + 1)
        return metrics

    def __repr__(self):
        pass
