from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, ClassVar, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

from .context import Ref
from .executor import Executor
from .scheme import normalize_refs, PipelineRuntimeError, PipelineValidator, RefSpec

if TYPE_CHECKING:
    from .context import ExecutionContext


class Operation(ABC):
    """ """

    requires: ClassVar[frozenset[Ref]] = frozenset()
    provides: ClassVar[frozenset[Ref]] = frozenset()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.requires = normalize_refs(getattr(cls, "requires", ()))
        cls.provides = normalize_refs(getattr(cls, "provides", ()))

    def __init__(self, name: str = "", **kwargs):
        self.name = name
        self.kwargs = kwargs
        self._prefix = "" if not name else f"{name}/"

    def __call__(
        self, ctx: ExecutionContext
    ) -> Tuple[ExecutionContext, Dict[str, Any]]:
        """ """
        try:
            ctx, metrics = self.forward(ctx)
        except PipelineRuntimeError:
            raise
        except Exception as exc:
            raise PipelineRuntimeError(self._fmt_runtime_error(ctx, exc)) from exc
        metrics = {f"{self._prefix}{k}": v for k, v in metrics.items()}
        return ctx, metrics

    def _fmt_runtime_error(self, ctx: ExecutionContext, exc: Exception) -> str:
        op_name = f"'{self.name}'" if self.name else "<unnamed>"
        requires = ", ".join(map(repr, self.requires)) or "<none>"
        provides = ", ".join(map(repr, self.provides)) or "<none>"
        batch = getattr(ctx, "batch", None)
        cache = getattr(ctx, "cache", None)
        return (
            f"\033[91m{type(self).__name__}({op_name}) failed during execution.\n"
            f"Original error: {type(exc).__name__}: {exc}\n"
            f"Declared requires: {requires}\n"
            f"Declared provides: {provides}\n"
            f"Context snapshot: batch={batch!r}, cache={cache!r}\033[0m"
        )

    def _repr_fields(self) -> Dict[str, Any]:
        fields = {"name": self.name}
        if self.kwargs:
            fields["kwargs"] = self.kwargs
        return fields

    def _repr_children(self) -> Iterable[Tuple[str, "Operation"]]:
        return ()

    def _repr_field_items(self) -> Iterable[Tuple[str, Any]]:
        for key, value in self._repr_fields().items():
            if value is None:
                continue
            if isinstance(value, (tuple, list, dict, set, frozenset)) and not value:
                continue
            yield key, value

    def extra_repr(self) -> str:
        return ", ".join(f"{key}={value!r}" for key, value in self._repr_field_items())

    def _repr_child_line(self, key: str, child: "Operation") -> str:
        child_repr = repr(child).replace("\n", "\n  ")
        return f"({key}): {child_repr}"

    def _repr_lines(self) -> Tuple[list[str], bool]:
        extra = self.extra_repr()
        lines = extra.splitlines() if extra else []
        children = tuple(self._repr_children())
        lines.extend(self._repr_child_line(key, child) for key, child in children)
        return lines, bool(children)

    def __repr__(self):
        lines, has_children = self._repr_lines()
        if not lines:
            return f"{type(self).__name__}()"
        if len(lines) == 1 and not has_children:
            return f"{type(self).__name__}({lines[0]})"
        body = "\n  ".join(lines)
        return f"{type(self).__name__}(\n  {body}\n)"

    @abstractmethod
    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        raise NotImplementedError


class Objective(Operation):
    """Objective is a special read-only Operation.
    When executed directly within a Pipeline. It functions as
    a `Logger`, computing Loss and recording Metrics without
    updating the model. When it is aggregated by an OptimizeOp.
    The OptimizeOp invokes its `compute` method
    to obtain gradient signals.
    """

    def __init__(self, weight: float = 1.0, name: str = "objective", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight = weight
        #
        self.exports: Dict[str, Any] = {}

    def _repr_fields(self) -> Dict[str, Any]:
        fields = super()._repr_fields()
        fields["weight"] = self.weight
        return fields

    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        loss, metrics = self.evaluate(ctx)
        return ctx, metrics

    def evaluate(self, ctx: ExecutionContext) -> Tuple[Any, Dict[str, Any]]:
        from thunder.utils.logging import Scalar

        self.exports.clear()
        loss, metrics = self.compute(ctx)
        weighted_loss = self.curriculum(ctx) * self.weight * loss
        metrics = {f"{self._prefix}{k}": v for k, v in metrics.items()}
        metrics = {
            f"{self._prefix}loss": Scalar(Executor.detach(loss)),
            f"{self._prefix}weighted_loss": Scalar(Executor.detach(weighted_loss)),
            **metrics,
        }
        return weighted_loss, metrics

    @abstractmethod
    def compute(self, ctx: ExecutionContext) -> Tuple[Any, Dict[str, Any]]:
        """

        Args:
            ctx (ExecutionContext):

        Returns:
            Tuple[Any, Dict[str, Any]]:
        """
        raise NotImplementedError

    def curriculum(self, ctx: ExecutionContext) -> float:
        return 1.0

    def export(self) -> Dict[str, Any]:
        return self.exports


class Pipeline(Operation):
    """_summary_
    Args:
    """

    forward_fn: callable

    def __init__(
        self, pipeline: Iterable[Operation], name="", jit: bool = False, **kwargs
    ):
        super().__init__(name, **kwargs)
        self.jit = jit
        self.pipeline = list(pipeline)
        self.setup()

    def forward(self, ctx: ExecutionContext):
        return self.forward_fn(ctx)

    @staticmethod
    def _forward(
        ctx: ExecutionContext,
        pipeline: Tuple[Operation, ...],
        prefix: str,
    ):
        metrics = {}
        for op in pipeline:
            ctx, m = op(ctx)
            metrics.update(m)
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        return ctx, metrics

    def setup(self):
        self._pipeline = tuple(self.pipeline)
        requires, provides = self._analyze_contract()
        self.requires = requires
        self.provides = provides
        self.validate(initial_refs=requires)
        self.forward_fn = self._compile_forward()

    def _compile_forward(self):
        forward_fn = partial(
            self._forward,
            pipeline=self._pipeline,
            prefix=self._prefix,
        )
        return Executor.jit(forward_fn) if self.jit else forward_fn

    def _refresh_contracts(self) -> None:
        requires, provides = self._analyze_contract()
        self.requires = requires
        self.provides = provides

    def analyze_contract(
        self,
        initial_refs: Iterable[RefSpec] = (),
    ) -> Tuple[frozenset[Ref], frozenset[Ref]]:
        return self._analyze_contract(initial_refs=initial_refs)

    def _analyze_contract(
        self,
        initial_refs: Iterable[RefSpec] = (),
        _seen: Optional[frozenset] = None,
    ) -> Tuple[frozenset[Ref], frozenset[Ref]]:
        if _seen is None:
            _seen = frozenset()
        if id(self) in _seen:
            return frozenset(), frozenset()
        _seen = _seen | frozenset({id(self)})

        ops = []
        for op in self.pipeline:
            if isinstance(op, Pipeline) and id(op) not in _seen:
                sub_req, sub_prov = op._analyze_contract(_seen=_seen)
                ops.append((sub_req, sub_prov))
            else:
                ops.append(op)
        return PipelineValidator(ops, name=self.name).analyze(initial_refs)

    def validate(
        self,
        initial_refs: Iterable[RefSpec] = (),
        mode: str = "error",
    ) -> None:
        PipelineValidator(self.pipeline, name=self.name).validate(initial_refs, mode)

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

    def extend(self, ops: Iterable[Operation]):
        self.pipeline.extend(ops)
        self.setup()

    def _repr_fields(self) -> Dict[str, Any]:
        return {"jit": self.jit, "size": len(self.pipeline)}

    def _repr_children(self) -> Iterable[Tuple[str, Operation]]:
        return ((str(i), op) for i, op in enumerate(self.pipeline))


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
        return ctx.executor.optimize(
            ctx=ctx,
            opt=self.opt,
            objectives=self.objectives,
            max_grad_norm=self.max_grad_norm,
        )

    def _repr_fields(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "opt": self.opt,
            "max_grad_norm": self.max_grad_norm,
        }

    def _repr_children(self) -> Iterable[Tuple[str, Operation]]:
        return ((str(i), obj) for i, obj in enumerate(self.objectives))


class NullOperation(Operation):
    """ """

    def __init__(self, name: str = "null", **kwargs):
        super().__init__(name=name, **kwargs)

    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        return ctx, {}
