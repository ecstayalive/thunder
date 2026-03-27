from __future__ import annotations

import ast
import contextlib
import os
import sys
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ContextManager, Dict, Hashable, Optional, Tuple

from .data import Cache

if TYPE_CHECKING:
    from .data import Batch
    from .executor.interface import Executor
    from .module import ModelPack, ThunderModule

_BACKEND = os.getenv("THUNDER_BACKEND", "torch").lower()


class ComposedContextManager(ContextManager):
    def __init__(self, *contexts):
        self.contexts = contexts
        self._stack = None

    def __enter__(self):
        self._stack = contextlib.ExitStack()
        self._stack.__enter__()

        try:
            for ctx in self.contexts:
                self._stack.enter_context(ctx)
        except:
            self._stack.__exit__(*sys.exc_info())
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._stack.__exit__(exc_type, exc_val, exc_tb)

    def __eq__(self, other):
        if not isinstance(other, ComposedContextManager):
            return False
        return self.contexts == other.contexts

    def __hash__(self):
        return hash(self.contexts)


@dataclass(slots=True)
class OptimGroup:
    """
    Args:
        name:
        targets:
        params:
        optimizer: `nnx.Optimizer` for jax, `torch.optim.Optimizer` for torch
        scheduler: learning rate scheduler, None for `jax`
        scaler: The scaler object (optax.amp.DynamicScale or torch.cuda.amp.GradScaler)
        scaler_state: The dynamic state of the scaler (JAX only, None for Torch)
    """

    name: str
    targets: Tuple[str, ...]
    optimizer: Any
    scheduler: Optional[Any] = None
    scaler: Optional[Any] = None
    scaler_state: Optional[Any] = None


@dataclass(slots=True)
class ExecutionContextManager:
    """
    Handles Mixed Precision AND Distributed Contexts.
    Args:

    """

    _context_manager: ContextManager | ComposedContextManager
    compute_dtype: Any
    device: Any

    distributed: bool = False
    rank: int = 0
    world_size: int = 1
    mesh: Any = None

    def __enter__(self):
        """ """
        return self._context_manager.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._context_manager.__exit__(exc_type, exc_val, exc_tb)

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


@dataclass(slots=True)
class ExecutionContext:
    """
    Args:
        step:
        batch:
        opt_groups:
        executor:
        model:
        cache:
        meta:
    """

    step: int
    models: ModelPack
    opt_groups: Dict[str, OptimGroup]
    executor: Executor
    manager: ExecutionContextManager
    batch: Optional[Batch] = None
    cache: Cache = field(default_factory=Cache)
    meta: Dict[str, Any] = field(default_factory=dict)

    def replace(self, **changes) -> ExecutionContext:
        return replace(self, **changes)

    def get_model(self, key: str) -> ThunderModule:
        """ """
        try:
            return getattr(self.models, key)
        except AttributeError:
            raise ValueError(
                f"Model '{key}' not found in Context. "
                f"Available models: {list(self.models.keys())}"
            )

    def update_meta(self, **kwargs) -> None:
        """ """
        self.meta.update(kwargs)

    @classmethod
    def create(
        cls,
        models: ModelPack,
        executor: Executor,
        manager: ExecutionContextManager,
        opt_groups: Dict[str, OptimGroup],
    ) -> ExecutionContext:
        """ """
        return cls(step=0, models=models, opt_groups=opt_groups, executor=executor, manager=manager)


@dataclass(frozen=True, slots=True)
class _RefAttr:
    name: str


@dataclass(frozen=True, slots=True)
class _RefKey:
    key: Hashable


_RefPath = Tuple[_RefAttr | _RefKey, ...]


class Ref:

    __slots__ = ("_path", "_compiled_fn")

    def __init__(self, path: str | _RefPath):
        raw_path = self._parse_expr(path) if isinstance(path, str) else tuple(path)
        self._path = self._normalize_path(raw_path)
        self._compiled_fn = _compile_ref_accessor(self._path)

    def __call__(self, ctx: ExecutionContext):
        return self._compiled_fn(ctx)

    def __eq__(self, other):
        if not isinstance(other, Ref):
            return False
        return self._path == other._path

    def __hash__(self):
        return hash(self._path)

    def __reduce__(self):
        # Rebuild from the canonical path so pickling does not try to serialize
        # the cached accessor lambda created by `_compile_ref_accessor`.
        return (type(self), (self._path,))

    @property
    def path(self) -> _RefPath:
        return self._path

    def __str__(self):
        if not self._path:
            return "ctx"
        parts = []
        for step in self._path:
            if isinstance(step, _RefAttr):
                parts.append(step.name if not parts else f".{step.name}")
            else:
                parts.append(f"[{repr(step.key)}]")
        return "".join(parts)

    def __repr__(self):
        if not self._path:
            return "CtxRef"
        return f'Ref("{self}")'

    @staticmethod
    @lru_cache(maxsize=None)
    def _parse_expr(expr: str) -> _RefPath:
        expr = expr.strip()
        if not expr:
            return ()
        node = ast.parse(expr, mode="eval").body
        return Ref._parse_node(node)

    @classmethod
    def _parse_node(cls, node: ast.AST) -> _RefPath:
        if isinstance(node, ast.Name):
            return () if node.id == "ctx" else (_RefAttr(name=node.id),)
        if isinstance(node, ast.Attribute):
            return cls._parse_node(node.value) + (_RefAttr(node.attr),)
        if isinstance(node, ast.Subscript):
            return cls._parse_node(node.value) + (_RefKey(cls._parse_key(node.slice)),)
        raise ValueError(
            "Ref only supports attribute and subscript access, "
            f"got unsupported expression: {ast.dump(node)}"
        )

    @staticmethod
    def _parse_key(node: ast.AST) -> Hashable:
        try:
            key = ast.literal_eval(node)
        except Exception as exc:
            raise ValueError(
                "Ref subscript keys must be Python literals, "
                f"got unsupported expression: {ast.dump(node)}"
            ) from exc
        if not isinstance(key, Hashable):
            raise ValueError(f"Ref subscript key must be hashable, got: {type(key).__name__}")
        return key

    @staticmethod
    def _normalize_path(path: _RefPath) -> _RefPath:
        """Normalize equivalent access patterns for Thunder containers.

        `Batch` and `Cache` support both attribute access and string-key access
        for their first-level entries. Canonicalizing these paths avoids false
        negatives during contract validation, e.g. `batch.embedding` versus
        `batch["embedding"]`.
        """
        if len(path) < 2:
            return path
        root, first_child = path[0], path[1]
        if (
            isinstance(root, _RefAttr)
            and root.name in {"batch", "cache"}
            and isinstance(first_child, _RefKey)
            and isinstance(first_child.key, str)
            and first_child.key.isidentifier()
        ):
            return (root, _RefAttr(first_child.key), *path[2:])
        return path


@lru_cache(maxsize=None)
def _compile_ref_accessor(path: _RefPath):
    expr = "ctx"
    safe_locals = {}
    for i, step in enumerate(path):
        if isinstance(step, _RefAttr):
            expr += f".{step.name}"
        else:
            key = step.key
            if isinstance(key, (str, int)):
                expr += f"[{repr(key)}]"
            else:
                var_name = f"_k{i}"
                safe_locals[var_name] = key
                expr += f"[{var_name}]"
    if not safe_locals:
        return eval(f"lambda ctx: {expr}")
    args = ", ".join(safe_locals.keys())
    factory = eval(f"lambda {args}: lambda ctx: {expr}")
    return factory(**safe_locals)


def replace_ref_path(root: Any, path: _RefPath, value: Any) -> Any:
    if not path:
        return value
    step, rest = path[0], path[1:]
    if isinstance(step, _RefAttr):
        child = getattr(root, step.name)
        new_child = replace_ref_path(child, rest, value)
        if hasattr(root, "replace") and callable(root.replace):
            return root.replace(**{step.name: new_child})
        return replace(root, **{step.name: new_child})

    child = root[step.key]
    new_child = replace_ref_path(child, rest, value)
    if hasattr(root, "replace") and callable(root.replace) and isinstance(step.key, str):
        return root.replace(**{step.key: new_child})
    if isinstance(root, dict):
        updated = root.copy()
        updated[step.key] = new_child
        return updated
    if isinstance(root, list):
        updated = list(root)
        updated[step.key] = new_child
        return updated
    if isinstance(root, tuple):
        updated = list(root)
        updated[step.key] = new_child
        if hasattr(root, "_fields"):
            return type(root)(*updated)
        return tuple(updated)
    raise TypeError(
        "Ref path updates only support dataclass attributes, dicts, lists, and tuples. "
        f"Cannot update key {step.key!r} on {type(root).__name__}."
    )


CtxRef = Ref(())

if _BACKEND == "torch":
    import torch.utils._pytree as pytree

    def _flatten_optim_group(obj: OptimGroup):
        children = [obj.optimizer, obj.scaler_state]
        aux_data = (obj.name, obj.targets, obj.scheduler, obj.scaler)
        return children, aux_data

    def _unflatten_optim_group(children, aux_data):
        return OptimGroup(
            name=aux_data[0],
            targets=aux_data[1],
            optimizer=children[0],
            scheduler=aux_data[2],
            scaler=aux_data[3],
            scaler_state=children[1],
        )

    pytree.register_pytree_node(OptimGroup, _flatten_optim_group, _unflatten_optim_group)

    def _flatten_manager(obj: ExecutionContextManager):
        children = []
        aux_data = (
            obj._context_manager,
            obj.compute_dtype,
            obj.device,
            obj.distributed,
            obj.rank,
            obj.world_size,
            obj.mesh,
        )
        return children, aux_data

    def _unflatten_manager(children, aux_data):
        return ExecutionContextManager(
            _context_manager=aux_data[0],
            compute_dtype=aux_data[1],
            device=aux_data[2],
            distributed=aux_data[3],
            rank=aux_data[4],
            world_size=aux_data[5],
            mesh=aux_data[6],
        )

    pytree.register_pytree_node(ExecutionContextManager, _flatten_manager, _unflatten_manager)

    def _flatten_context(obj: ExecutionContext):
        children = [obj.models, obj.opt_groups, obj.step, obj.batch, obj.cache, obj.meta]
        aux_data = (obj.executor, obj.manager)

        return children, aux_data

    def _unflatten_context(children, aux_data):
        executor, manager = aux_data
        models, opt_groups, step, batch, cache, meta = children

        return ExecutionContext(
            step=step,
            batch=batch,
            cache=cache,
            models=models,
            opt_groups=opt_groups,
            executor=executor,
            manager=manager,
            meta=meta,
        )

    pytree.register_pytree_node(ExecutionContext, _flatten_context, _unflatten_context)

if _BACKEND == "jax":
    import flax.nnx as nnx
    import jax.tree_util as jtu

    def _flatten_optim_group(obj: OptimGroup):
        children = (obj.optimizer, obj.scaler_state)
        aux_data = (obj.name, obj.targets, obj.scheduler, obj.scaler)
        return children, aux_data

    def _unflatten_optim_group(aux_data, children):
        return OptimGroup(
            name=aux_data[0],
            targets=aux_data[1],
            optimizer=children[0],
            scheduler=aux_data[2],
            scaler=aux_data[3],
            scaler_state=children[1],
        )

    jtu.register_pytree_node(OptimGroup, _flatten_optim_group, _unflatten_optim_group)

    def _flatten_manager(obj: ExecutionContextManager):
        children = []
        aux_data = (
            obj._context_manager,
            obj.compute_dtype,
            obj.device,
            obj.distributed,
            obj.rank,
            obj.world_size,
            obj.mesh,
        )
        return children, aux_data

    def _unflatten_manager(aux_data, children):
        return ExecutionContextManager(
            _context_manager=aux_data[0],
            compute_dtype=aux_data[1],
            device=aux_data[2],
            distributed=aux_data[3],
            rank=aux_data[4],
            world_size=aux_data[5],
            mesh=aux_data[6],
        )

    jtu.register_pytree_node(ExecutionContextManager, _flatten_manager, _unflatten_manager)

    def _flatten_context(obj: ExecutionContext):
        """ """
        nnx_containers = (obj.models, obj.opt_groups)
        graphdef, state = nnx.split(nnx_containers)
        children = (obj.step, obj.batch, obj.cache, state, obj.meta)
        aux_data = (obj.executor, graphdef, obj.manager)

        return children, aux_data

    def _unflatten_context(aux_data, children):
        executor, graphdef, manager = aux_data
        step, batch, cache, state, meta = children
        models, opt_groups = nnx.merge(graphdef, state)
        return ExecutionContext(
            step=step,
            batch=batch,
            cache=cache,
            models=models,
            opt_groups=opt_groups,
            executor=executor,
            meta=meta,
            manager=manager,
        )

    jtu.register_pytree_node(ExecutionContext, _flatten_context, _unflatten_context)

__all__ = [
    "ComposedContextManager",
    "OptimGroup",
    "ExecutionContextManager",
    "ExecutionContext",
    "CtxRef",
    "Ref",
]
