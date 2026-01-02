from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

if TYPE_CHECKING:
    from .context import ExecutionContext
    from .data import Batch
    from .module import ModelPack


class Operation(ABC):
    """ """

    def __init__(self, name: str = "op", interval: int = 1, **kwargs):
        self.name = name
        self.interval = interval
        self.kwargs = kwargs

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

    def __init__(self, name: str, weight: float = 1.0, **kwargs):
        super().__init__(name=name, interval=1, **kwargs)
        self.weight = weight

    def __call__(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        if ctx.step % self.interval != 0:
            return ctx, {}
        _, metrics = self.forward(ctx.batch, ctx.models)
        return ctx, metrics

    def forward(self, batch: Batch, model: ModelPack) -> Tuple[Any, Dict[str, Any]]:
        loss, metrics = self.compute(batch, model)
        weighted_loss = self.weight * loss
        metrics = {
            f"{self.name}/loss": loss,
            f"{self.name}/weighted_loss": weighted_loss,
            **metrics,
        }
        return weighted_loss, metrics

    @abstractmethod
    def compute(self, batch: Batch, model: ModelPack) -> Tuple[Any, Dict[str, Any]]:
        pass


class OptimizeOp(Operation):
    """ """

    def __init__(
        self,
        opt: str,
        objectives: List[Objective],
        max_grad_norm: float = 1.0,
        name: str = "grad_op",
        interval: int = 1,
    ):
        super().__init__(name=name, interval=interval)
        self.opt = opt
        self.objectives = objectives
        self.max_grad_norm = max_grad_norm

    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        metrics = ctx.executor.optimize(
            ctx=ctx, opt=self.opt, objectives=self.objectives, max_grad_norm=self.max_grad_norm
        )
        return ctx, metrics


class CallableOp(Operation):
    """
    Args:
        fn: fn(ctx: ExecutionContext, **kwargs) -> Tuple[ExecutionContext, Dict[str, Any]]
    """

    def __init__(self, fn: Callable, name="callable", interval=1, **kwargs):
        super().__init__(name, interval, **kwargs)
        self.fn = fn

    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        return self.fn(ctx, **self.kwargs)


class CallableObjective(Objective):
    """
    Args:
        fn: fn(batch: Batch, model: ModelPack, params: Any, **kwargs) -> Tuple[Loss, Dict[str, Any]]
    """

    def __init__(self, fn: Callable, name="callable_objective", weight=1.0, **kwargs):
        super().__init__(name, weight, **kwargs)
        self.fn = fn

    def compute(self, batch: Batch, model: ModelPack, params: Any) -> Tuple[Any, Dict[str, Any]]:
        return self.fn(batch, model, params, **self.kwargs)
