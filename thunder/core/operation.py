from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .context import ExecutionContext
    from .data import Batch, ModelPack
    from .module import ThunderModule


class Operation(ABC):
    """ """

    def __init__(self, name: str = "op", interval: int = 1):
        self.name = name
        self.interval = interval

    def __call__(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        if ctx.step % self.interval != 0:
            return ctx, {}
        new_ctx, raw_metrics = self.forward(ctx)
        if raw_metrics:
            prefix = f"{self.name}/"
            metrics = {f"{prefix}{k}": v for k, v in raw_metrics.items()}
        else:
            metrics = {}

        return new_ctx, metrics

    @abstractmethod
    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        pass


class Objective(Operation):
    """Objective is a special read-only Operation.
    When executed directly within a Pipeline. It functions as
    a `Logger`, computing Loss and recording Metrics without
    updating the model. When it is aggregated by an OptimizeOp.
    The OptimizeOp invokes its `compute` method
    to obtain gradient signals.
    Args:

    """

    def __init__(self, name: str, weight: float = 1.0):
        super().__init__(name=name, interval=1)
        self.weight = weight

    def __call__(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        if ctx.step % self.interval != 0:
            return ctx, {}
        _, metrics = self.forward(ctx.batch, ctx.models, ctx.params)
        return ctx, metrics

    def forward(self, batch: Batch, model: ModelPack, params: Any) -> Tuple[Any, Dict[str, Any]]:
        loss, metrics = self.compute(batch, model, params)
        weighted_loss = self.weight * loss
        metrics = {
            f"{self.name}/loss": loss,
            f"{self.name}/weighted_loss": weighted_loss,
            **metrics,
        }
        return weighted_loss, metrics

    @abstractmethod
    def compute(self, batch: Batch, model: ModelPack, params: Any) -> Tuple[Any, Dict[str, Any]]:
        pass


class OptimizeOp(Operation):
    """ """

    def __init__(
        self,
        target: str,
        opt: str,
        objectives: List[Objective],
        max_grad_norm: float = 1.0,
        name: str = "grad_op",
        interval: int = 1,
    ):
        super().__init__(name=name, interval=interval)
        self.target = target
        self.opt = opt
        self.objectives = objectives
        self.max_grad_norm = max_grad_norm

    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        metrics, new_params, new_opt_state = ctx.executor.optimize(
            ctx=ctx,
            target=self.target,
            opt=self.opt,
            objectives=self.objectives,
            max_grad_norm=self.max_grad_norm,
        )
        new_ctx = ctx.apply_gradients(
            target=self.target, opt=self.opt, new_params=new_params, new_opt_state=new_opt_state
        )
        return new_ctx, metrics


class CallbackOp(Operation):
    def __init__(self, fn: Callable, name="callback", interval=1):
        super().__init__(name, interval)
        self.fn = fn

    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        return self.fn(ctx)
