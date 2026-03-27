from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Optional,
    Tuple,
)

from .context import Ref, replace_ref_path
from .executor import Executor

if TYPE_CHECKING:
    from .context import ExecutionContext
    from .data import Batch
    from .module import ModelPack


_RefSpec = Any


def _normalize_refs(refs: Iterable[_RefSpec] | None) -> frozenset[Ref]:
    """ """
    if refs is None:
        return frozenset()
    return frozenset(ref if isinstance(ref, Ref) else Ref(ref) for ref in refs)


class PipelineValidationError(ValueError):
    pass


_SYSTEM_PREFIX_REFS = _normalize_refs(
    (
        "batch.obs",
        "batch.actions",
        "batch.rewards",
        "batch.dones",
        "batch.timeouts",
        "batch.mask",
        "batch.next_obs",
    )
)
_SYSTEM_EXACT_REFS = _normalize_refs(
    (
        "step",
        "batch",
        "cache",
        "models",
        "executor",
        "manager",
        "opt_groups",
        "meta",
    )
)


@dataclass(slots=True)
class _RefNode:
    terminal: bool = False
    children: Dict[Any, "_RefNode"] = field(default_factory=dict)


class RefIndex:
    __slots__ = ("_prefix_root", "_exact")

    def __init__(
        self,
        exact_refs: Iterable[_RefSpec] = (),
        prefix_refs: Iterable[_RefSpec] = (),
    ):
        self._prefix_root = _RefNode()
        self._exact: set[Ref] = set()
        self.update_exact(exact_refs)
        self.update_prefix(prefix_refs)

    def add_exact(self, ref: _RefSpec) -> None:
        self._exact.add(ref if isinstance(ref, Ref) else Ref(ref))

    def add_prefix(self, ref: _RefSpec) -> None:
        ref = ref if isinstance(ref, Ref) else Ref(ref)
        node = self._prefix_root
        for step in ref.path:
            node = node.children.setdefault(step, _RefNode())
        node.terminal = True

    def update_exact(self, refs: Iterable[_RefSpec]) -> None:
        for ref in refs:
            self.add_exact(ref)

    def update_prefix(self, refs: Iterable[_RefSpec]) -> None:
        for ref in refs:
            self.add_prefix(ref)

    def covers(self, ref: _RefSpec) -> bool:
        ref = ref if isinstance(ref, Ref) else Ref(ref)
        if ref in self._exact:
            return True
        node = self._prefix_root
        if node.terminal:
            return True
        for step in ref.path:
            node = node.children.get(step)
            if node is None:
                return False
            if node.terminal:
                return True
        return False


class Operation(ABC):
    """ """

    requires: ClassVar[frozenset[Ref]] = frozenset()
    provides: ClassVar[frozenset[Ref]] = frozenset()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.requires = _normalize_refs(getattr(cls, "requires", ()))
        cls.provides = _normalize_refs(getattr(cls, "provides", ()))

    def __init__(self, name: str = "operation", **kwargs):
        self.name = name
        self.kwargs = kwargs
        self._prefix = "" if not name else f"{name}/"

    def __call__(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        """ """
        ctx, metrics = self.forward(ctx)
        metrics = {f"{self._prefix}{k}": v for k, v in metrics.items()}
        return ctx, metrics

    def _repr_fields(self) -> Dict[str, Any]:
        fields = {"name": self.name}
        if self.kwargs:
            fields["kwargs"] = self.kwargs
        return fields

    def __repr__(self):
        parts = []
        for key, value in self._repr_fields().items():
            if value in (None, (), {}, frozenset()):
                continue
            parts.append(f"{key}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    @abstractmethod
    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        pass


class NullOperation(Operation):
    """ """

    def __init__(self, name: str = "null", **kwargs):
        super().__init__(name=name, **kwargs)

    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        return ctx, {}


class Objective(Operation):
    """Objective is a special read-only Operation.
    When executed directly within a Pipeline. It functions as
    a `Logger`, computing Loss and recording Metrics without
    updating the model. When it is aggregated by an OptimizeOp.
    The OptimizeOp invokes its `compute` method
    to obtain gradient signals.
    Args:

    """

    def __init__(self, weight: float = 1.0, name: str = "objective", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight = weight

    def __call__(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        loss, metrics = self.forward(ctx.batch, ctx.models)
        return ctx, metrics

    def _repr_fields(self) -> Dict[str, Any]:
        fields = super()._repr_fields()
        fields["weight"] = self.weight
        return fields

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
        self._refresh_contracts()
        self.validate()
        self._jit_forward = Executor.jit(
            partial(self._forward, pipeline=tuple(self.pipeline), prefix=self._prefix)
        )

    def _refresh_contracts(self) -> None:
        requires, provides = self._analyze_contract()
        self.requires = requires
        self.provides = provides

    def _analyze_contract(
        self,
        exact_refs: Iterable[_RefSpec] = _SYSTEM_EXACT_REFS,
        prefix_refs: Iterable[_RefSpec] = _SYSTEM_PREFIX_REFS,
    ) -> Tuple[frozenset[Ref], frozenset[Ref]]:
        available = RefIndex(exact_refs=exact_refs, prefix_refs=prefix_refs)
        external_requires: list[Ref] = []
        provided_refs: list[Ref] = []
        for op in self.pipeline:
            missing = [ref for ref in op.requires if not available.covers(ref)]
            external_requires.extend(missing)
            available.update_prefix(missing)
            available.update_prefix(op.provides)
            provided_refs.extend(op.provides)
        return frozenset(external_requires), frozenset(provided_refs)

    def validate(
        self,
        initial_exact_refs: Iterable[_RefSpec] = _SYSTEM_EXACT_REFS,
        initial_prefix_refs: Iterable[_RefSpec] = _SYSTEM_PREFIX_REFS,
        mode: str = "error",
    ) -> None:
        available = RefIndex(exact_refs=initial_exact_refs, prefix_refs=initial_prefix_refs)
        for idx, op in enumerate(self.pipeline):
            missing = tuple(ref for ref in op.requires if not available.covers(ref))
            if missing:
                message = (
                    f"Pipeline '{self.name}' validation failed at op[{idx}] '{op.name}'. "
                    f"Missing requirements: {', '.join(map(repr, missing))}"
                )
                if mode == "warn":
                    print(f"[Thunder][Pipeline Warning] {message}")
                else:
                    raise PipelineValidationError(message)
            available.update_prefix(op.provides)

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

    def _repr_fields(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "jit": self.jit,
            "size": len(self.pipeline),
            "ops": tuple(type(op).__name__ for op in self.pipeline),
        }


class OptimizeOp(Operation):
    """ """

    def __init__(
        self,
        opt: str,
        objectives: Iterable[Objective],
        max_grad_norm: float = 1.0,
        name: Optional[str] = None,
    ):
        self.opt = opt
        super().__init__(name=name if name is not None else opt)
        self.objectives = tuple(objectives)
        self.max_grad_norm = max_grad_norm

        objective_requires = []
        for obj in self.objectives:
            objective_requires.extend(obj.requires)
        self.requires = frozenset(objective_requires)
        self.provides = frozenset()

    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        metrics = ctx.executor.optimize(
            ctx=ctx, opt=self.opt, objectives=self.objectives, max_grad_norm=self.max_grad_norm
        )
        return ctx, metrics

    def _repr_fields(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "opt": self.opt,
            "objectives": tuple(type(obj).__name__ for obj in self.objectives),
            "max_grad_norm": self.max_grad_norm,
        }


class CallableOp(Operation):
    __slots__ = ("_fn",)

    def __init__(self, fn: Callable, name="callable_op", returns=None, **bindings):
        super().__init__(name=name)
        self._fn = self._jit_compile(fn, bindings, returns)

    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict]:
        return self._fn(ctx)

    def _repr_fields(self) -> Dict[str, Any]:
        fields = super()._repr_fields()
        fields["callable"] = getattr(self._fn, "__name__", type(self._fn).__name__)
        return fields

    def _jit_compile(self, fn, bindings, returns):
        closure_vars = {"_fn": fn, "_replace_path": replace_ref_path}
        args_code = []
        for name, value in bindings.items():
            var_name = f"_var_{name}"
            closure_vars[var_name] = value
            if isinstance(value, Ref) or callable(value):
                args_code.append(f"{name}={var_name}(ctx)")
            else:
                args_code.append(f"{name}={var_name}")
        arg_str = ", ".join(args_code)
        body = f"res = _fn({arg_str})"
        if returns is None:
            body += "; return ctx, (res if isinstance(res, dict) else {})"
        else:
            closure_vars["_returns_path"] = returns._path
            body += "; new_ctx = _replace_path(ctx, _returns_path, res); return new_ctx, {}"
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

    def _repr_fields(self) -> Dict[str, Any]:
        fields = super()._repr_fields()
        fields["callable"] = getattr(self._compute_fn, "__name__", type(self._compute_fn).__name__)
        return fields

    def _jit_compile(self, fn, bindings):
        closure_vars = {"_fn": fn}
        from collections import namedtuple

        closure_vars["_CtxSim"] = namedtuple("CtxSim", ["batch", "models"])
        args_code = []
        for name, value in bindings.items():
            var_name = f"_var_{name}"
            closure_vars[var_name] = value
            if isinstance(value, Ref) or callable(value):
                args_code.append(f"{name}={var_name}(_CtxSim(batch, model))")
            else:
                args_code.append(f"{name}={var_name}")
        arg_str = ", ".join(args_code)
        code = f"lambda batch, model: _fn({arg_str})"

        factory_args = ", ".join(closure_vars.keys())
        full_code = f"lambda {factory_args}: {code}"
        return eval(full_code)(**closure_vars)
