from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .context import ExecutionContext

if TYPE_CHECKING:
    from .data import Batch
    from .executor.executor import ExecutorProtocol
    from .module import ThunderModule
    from .operation import Operation


class GraphAlgorithm(ABC):
    def __init__(self, model: ThunderModule, executor: ExecutorProtocol, pipeline: List[Operation]):
        self.model = model
        self.executor = executor
        self.ctx: Optional[ExecutionContext] = None
        self.pipeline = pipeline

    def build(self, sample_batch: Batch, optim_config: Dict[str, Any]) -> None:
        params, opt_states, meta = self.executor.init_state(
            model=self.model, batch=sample_batch, optim_config=optim_config
        )
        self.ctx = ExecutionContext.create(executor=self.executor, model=self.model, batch=None)
        self.ctx = self.ctx.replace(params=params, opt_states=opt_states)
        self.ctx.update_meta(**meta)

    def step(self, batch: Batch) -> Dict[str, Any]:
        if self.ctx is None:
            raise RuntimeError("Algorithm not built. Please call .build() first.")
        # Load data
        metrics = {}
        self.ctx = self.ctx.replace(batch=batch)
        for op in self.pipeline:
            self.ctx, m = op(self.ctx)
            metrics.update(m)
        self.ctx = self.ctx.replace(step=self.ctx.step + 1)
        return metrics
