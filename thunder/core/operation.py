from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Tuple

from .executor import Executor

if TYPE_CHECKING:
    from .context import ExecutionContext
    from .data import Batch
    from .module import ModelPack


class Operation(ABC):
    """ """

    def __init__(self, name: str = "operation", **kwargs):
        self.name = name
        self.kwargs = kwargs
        self._prefix = "" if not name else f"{name}/"

    def __call__(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        """ """
        ctx, metrics = self.forward(ctx)
        metrics = {f"{self._prefix}{k}": v for k, v in metrics.items()}
        return ctx, metrics

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

    def __init__(self, name: str = "objective", weight: float = 1.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight = weight

    def __call__(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        loss, metrics = self.forward(ctx.batch, ctx.models)
        return ctx, metrics

    def forward(self, batch: Batch, model: ModelPack) -> Tuple[Any, Dict[str, Any]]:
        loss, metrics = self.compute(batch, model)
        weighted_loss = self.weight * loss
        metrics = {
            f"{self._prefix}loss": loss,
            f"{self._prefix}weighted_loss": weighted_loss,
            **metrics,
        }
        return weighted_loss, metrics

    @abstractmethod
    def compute(self, batch: Batch, model: ModelPack) -> Tuple[Any, Dict[str, Any]]:
        """_summary_

        Args:
            batch (Batch): _description_
            model (ModelPack): _description_

        Returns:
            Tuple[Any, Dict[str, Any]]: _description_
        """
        pass


class Pipeline(Operation):
    """_summary_
    Args:
    """

    def __init__(self, pipeline: Iterable[Operation], name="pipeline", jit: bool = False, **kwargs):
        super().__init__(name, **kwargs)
        self.jit = jit
        self.pipeline = list(pipeline)
        self.setup()

    def __call__(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        """ """
        if self.jit:
            ctx, metrics = self._jit_forward(ctx)
        else:
            ctx, metrics = self.forward(ctx)
        return ctx, metrics

    def forward(self, ctx: ExecutionContext):
        return self._forward(ctx, tuple(self.pipeline), self._prefix)

    @staticmethod
    def _forward(ctx: ExecutionContext, pipeline: Tuple[Operation, ...], prefix: str):
        metrics = {}
        for op in pipeline:
            ctx, m = op(ctx)
            metrics.update(m)
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        return ctx, metrics

    def setup(self):
        self._jit_forward = Executor.jit(
            partial(self._forward, pipeline=tuple(self.pipeline), prefix=self._prefix)
        )

    def __iter__(self):
        return iter(self.pipeline)

    def __len__(self):
        return len(self.pipeline)

    def __getitem__(self, index):
        return self.pipeline[index]

    def __setitem__(self, index, value):
        self.pipeline[index] = value
        self.setup()

    def insert(self, index: int, op: Operation):
        self.pipeline.insert(index, op)
        self.setup()

    def remove(self, index: int):
        self.pipeline.pop(index)
        self.setup()

    def append(self, op: Operation):
        self.pipeline.append(op)
        self.setup()


class OptimizeOp(Operation):
    """ """

    def __init__(
        self,
        opt: str,
        objectives: Iterable[Objective],
        max_grad_norm: float = 1.0,
        name: str = "optimize",
    ):
        super().__init__(name=name)
        self.opt = opt
        self.objectives = tuple(objectives)
        self.max_grad_norm = max_grad_norm

    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        metrics = ctx.executor.optimize(
            ctx=ctx, opt=self.opt, objectives=self.objectives, max_grad_norm=self.max_grad_norm
        )
        return ctx, metrics


class CallableOp(Operation):
    __slots__ = ("_fn",)

    def __init__(self, fn: Callable, name="callable_op", returns=None, **bindings):
        super().__init__(name=name)
        self._fn = self._jit_compile(fn, bindings, returns)

    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict]:
        return self._fn(ctx)

    def _jit_compile(self, fn, bindings, returns):
        closure_vars = {"_fn": fn, "replace": dataclasses.replace}
        args_code = []
        for name, value in bindings.items():
            var_name = f"_var_{name}"
            closure_vars[var_name] = value
            if hasattr(value, "path") or callable(value):
                args_code.append(f"{name}={var_name}(ctx)")
            else:
                args_code.append(f"{name}={var_name}")
        arg_str = ", ".join(args_code)
        body = f"res = _fn({arg_str})"
        if returns is None:
            body += "; return ctx, (res if isinstance(res, dict) else {})"
        else:
            target_path = returns._path
            update_code = self._build_replace_chain("ctx", target_path, "res")
            body += f"; new_ctx = {update_code}; return new_ctx, {{}}"
        factory_args = ", ".join(closure_vars.keys())
        fn_name = f"_jit_{self.name}"
        lines = [
            f"def factory({factory_args}):",  # Outer Factory
            f"    def {fn_name}(ctx):",  # Inner JIT Function
            f"        {body}",  # The Logic
            f"    return {fn_name}",  # Return the inner function
        ]

        full_source = "\n".join(lines)
        local_scope = {}
        exec(full_source, globals(), local_scope)
        factory = local_scope["factory"]
        return factory(**closure_vars)

    def _build_replace_chain(self, root_var, path, value_var):
        """ """
        if len(path) == 0:
            return value_var
        op, key = path[0]
        if len(path) == 1:
            return f"replace({root_var}, {key}={value_var})"
        else:
            child_accessor = f"{root_var}.{key}" if op == 0 else f"{root_var}[{repr(key)}]"
            inner_update = self._build_replace_chain(child_accessor, path[1:], value_var)
            return f"replace({root_var}, {key}={inner_update})"


class CallableObjective(Objective):
    """
    Adapts any function into a `Thunder` Objective.
    The 'fn' can have ANY signature. You use 'bindings' to map
    Thunder's data (ctx.batch, ctx.models) to the function's arguments.
    """

    __slots__ = ("_compute_fn",)

    def __init__(self, fn: Callable, name="callable_objective", weight=1.0, **bindings):
        super().__init__(name, weight)
        self._compute_fn = self._jit_compile(fn, bindings)

    def compute(self, batch: Batch, model: ModelPack) -> Tuple[Any, Dict[str, Any]]:
        return self._compute_fn(batch, model)

    def _jit_compile(self, fn, bindings):
        closure_vars = {"_fn": fn}
        from collections import namedtuple

        closure_vars["_CtxSim"] = namedtuple("CtxSim", ["batch", "models"])
        args_code = []
        for name, value in bindings.items():
            var_name = f"_var_{name}"
            closure_vars[var_name] = value
            if hasattr(value, "path") or callable(value):
                args_code.append(f"{name}={var_name}(_CtxSim(batch, model))")
            else:
                args_code.append(f"{name}={var_name}")
        arg_str = ", ".join(args_code)
        code = f"lambda batch, model: _fn({arg_str})"

        factory_args = ", ".join(closure_vars.keys())
        full_code = f"lambda {factory_args}: {code}"
        return eval(full_code)(**closure_vars)
