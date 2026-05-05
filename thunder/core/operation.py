from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Optional,
    Tuple,
    TYPE_CHECKING,
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
        "batch.terminated",
        "batch.timeouts",
        "batch.mask",
        "batch.next_obs",
        "batch.values",
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
SYSTEM_EXACT_REFS = _SYSTEM_EXACT_REFS
SYSTEM_PREFIX_REFS = _SYSTEM_PREFIX_REFS


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

    def __init__(self, name: str = "", **kwargs):
        self.name = name
        self.kwargs = kwargs
        self._prefix = "" if not name else f"{name}/"

    def __call__(
        self, ctx: ExecutionContext
    ) -> Tuple[ExecutionContext, Dict[str, Any]]:
        """ """
        ctx, metrics = self.forward(ctx)
        metrics = {f"{self._prefix}{k}": v for k, v in metrics.items()}
        return ctx, metrics

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
        self.exports.clear()
        loss, metrics = self.compute(ctx)
        weighted_loss = self.curriculum(ctx) * self.weight * loss
        metrics = {
            f"loss": loss,
            f"weighted_loss": weighted_loss,
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
        self,
        pipeline: Iterable[Operation],
        name="",
        jit: bool = False,
        validate: str | None = "error",
        initial_exact_refs: Iterable[_RefSpec] = _SYSTEM_EXACT_REFS,
        initial_prefix_refs: Iterable[_RefSpec] = _SYSTEM_PREFIX_REFS,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.jit = jit
        self.pipeline = list(pipeline)
        self._validate_mode = validate
        self._initial_exact_refs = tuple(initial_exact_refs)
        self._initial_prefix_refs = tuple(initial_prefix_refs)
        self.setup()

    def __call__(
        self, ctx: ExecutionContext
    ) -> Tuple[ExecutionContext, Dict[str, Any]]:
        return self.forward(ctx)

    def forward(self, ctx: ExecutionContext):
        return self.forward_fn(ctx)

    @staticmethod
    def _forward(ctx: ExecutionContext, pipeline: Tuple[Operation, ...], prefix: str):
        metrics = {}
        for op in pipeline:
            ctx, m = op(ctx)
            metrics.update(m)
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        return ctx, metrics

    def setup(self):
        self._pipeline = tuple(self.pipeline)
        self._refresh_contracts()
        self._validate_contract(
            initial_exact_refs=self._initial_exact_refs,
            initial_prefix_refs=self._initial_prefix_refs,
            mode=self._validate_mode,
        )
        self.forward_fn = self._compile_forward()

    def _compile_forward(self):
        forward_fn = partial(
            self._forward, pipeline=self._pipeline, prefix=self._prefix
        )
        return Executor.jit(forward_fn) if self.jit else forward_fn

    def _validate_contract(
        self,
        initial_exact_refs: Iterable[_RefSpec],
        initial_prefix_refs: Iterable[_RefSpec],
        mode: str | None,
    ) -> None:
        if mode is None:
            return
        self.validate(
            initial_exact_refs=initial_exact_refs,
            initial_prefix_refs=initial_prefix_refs,
            mode=mode,
        )

    def _refresh_contracts(self) -> None:
        requires, provides = self._analyze_contract(
            exact_refs=self._initial_exact_refs,
            prefix_refs=self._initial_prefix_refs,
        )
        self.requires = requires
        self.provides = provides

    def analyze_contract(
        self,
        exact_refs: Iterable[_RefSpec] = _SYSTEM_EXACT_REFS,
        prefix_refs: Iterable[_RefSpec] = _SYSTEM_PREFIX_REFS,
    ) -> Tuple[frozenset[Ref], frozenset[Ref]]:
        return self._analyze_contract(exact_refs=exact_refs, prefix_refs=prefix_refs)

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
        available = RefIndex(
            exact_refs=initial_exact_refs, prefix_refs=initial_prefix_refs
        )
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


class CallableOp(Operation):
    __slots__ = ("_fn",)

    def __init__(self, fn: Callable, name="callable_op", returns=None, **bindings):
        super().__init__(name=name)
        self._fn = self._compile(fn, bindings, returns)

    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        return self._fn(ctx)

    def _repr_fields(self) -> Dict[str, Any]:
        fields = super()._repr_fields()
        fields["callable"] = getattr(self._fn, "__name__", type(self._fn).__name__)
        return fields

    def _compile(self, fn: Callable, bindings: Dict[str, Any], returns: Ref | None):
        def _resolve(value, ctx):
            if isinstance(value, Ref):
                return value(ctx)
            if callable(value):
                return value(ctx)
            return value

        def _call(ctx):
            kwargs = {key: _resolve(value, ctx) for key, value in bindings.items()}
            result = fn(**kwargs)
            if returns is not None:
                return replace_ref_path(ctx, returns.path, result), {}
            if isinstance(result, tuple) and len(result) == 2:
                return result
            return ctx, result if isinstance(result, dict) else {}

        return _call


class NullOperation(Operation):
    """ """

    def __init__(self, name: str = "null", **kwargs):
        super().__init__(name=name, **kwargs)

    def forward(self, ctx: ExecutionContext) -> Tuple[ExecutionContext, Dict[str, Any]]:
        return ctx, {}
