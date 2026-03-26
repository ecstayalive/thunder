from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

from .data import Cache
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
        name: str = "algorithm",
    ):
        self.name = name
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

    def setup_pipeline(
        self, pipeline: Optional[Pipeline | Iterable[Operation]], jit: bool = False
    ) -> None:
        """_summary_

        Args:
            pipeline (Iterable[Operation]): _description_
        """
        if pipeline is None:
            self.pipeline = Pipeline([], name=self.name, jit=False)
        elif isinstance(pipeline, Pipeline):
            pipeline.name = self.name
            pipeline._prefix = "" if not self.name else f"{self.name}/"
            pipeline.setup()
            self.pipeline = pipeline
        else:
            self.pipeline = Pipeline(pipeline, name=self.name, jit=jit)

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
        changes = {"cache": Cache()}
        if batch is not None:
            changes["batch"] = batch
        self.ctx = self.ctx.replace(**changes)
        with self.ctx.manager:
            self.ctx, metrics = self.pipeline(self.ctx)
        self.ctx = self.ctx.replace(step=self.ctx.step + 1)
        metrics.update({"execution_context": self.ctx})
        return metrics

    def __repr__(self):
        parts = [f"name={self.name!r}"]
        parts.append(f"executor={type(self.executor).__name__}()")
        parts.append(f"models={type(self.models).__name__}()")
        parts.append(f"built={self.ctx is not None!r}")
        if self.ctx is not None:
            parts.append(f"step={self.ctx.step!r}")
        if self.pipeline is not None:
            parts.append(f"pipeline={self.pipeline!r}")
        return f"{type(self).__name__}({', '.join(parts)})"
