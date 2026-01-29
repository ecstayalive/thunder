from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ContextManager, Dict, Optional, Tuple

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
        meta:
    """

    step: int
    batch: Optional[Batch]
    models: ModelPack
    opt_groups: Dict[str, OptimGroup]
    executor: Executor
    manager: ExecutionContextManager
    meta: Dict[str, Any] = field(default_factory=dict)

    def replace(self, **changes) -> ExecutionContext:
        return replace(self, **changes)

    def get_model(self, key: str) -> ThunderModule:
        """ """
        try:
            return getattr(self.models, key)
        except KeyError:
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
        return cls(
            step=0,
            batch=None,
            models=models,
            opt_groups=opt_groups,
            executor=executor,
            manager=manager,
        )


class _ContextRef:
    __slots__ = ("_path", "_compiled_fn")

    def __init__(self, path=()):
        self._path = path
        self._compiled_fn = None

    def __getattr__(self, name: str):
        return _ContextRef(self._path + ((0, name),))

    def __getitem__(self, key: str):
        return _ContextRef(self._path + ((1, key),))

    def __call__(self, ctx: ExecutionContext):
        if self._compiled_fn:
            return self._compiled_fn(ctx)
        self._compiled_fn = self._jit_compile()
        return self._compiled_fn(ctx)

    def _jit_compile(self):
        """ """
        expr = "ctx"
        safe_locals = {}
        for i, (op, key) in enumerate(self._path):
            if op == 0:
                expr += f".{key}"
            else:
                if isinstance(key, (str, int)):
                    expr += f"[{repr(key)}]"
                else:
                    var_name = f"_k{i}"
                    safe_locals[var_name] = key
                    expr += f"[{var_name}]"
        if not safe_locals:
            code = f"lambda ctx: {expr}"
            return eval(code)
        else:
            args = ", ".join(safe_locals.keys())
            code = f"lambda {args}: lambda ctx: {expr}"
            factory = eval(code)
            return factory(**safe_locals)


CtxRef = _ContextRef()

if _BACKEND == "torch":
    import torch.utils._pytree as pytree

    def _optim_group_flatten(obj: OptimGroup):
        children = [obj.optimizer, obj.scaler_state]
        aux_data = (obj.name, obj.targets, obj.scheduler, obj.scaler)
        return children, aux_data

    def _optim_group_unflatten(children, aux_data):
        return OptimGroup(
            name=aux_data[0],
            targets=aux_data[1],
            optimizer=children[0],
            scheduler=aux_data[2],
            scaler=aux_data[3],
            scaler_state=children[1],
        )

    pytree.register_pytree_node(OptimGroup, _optim_group_flatten, _optim_group_unflatten)

    def _manager_flatten(obj: ExecutionContextManager):
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

    def _manager_unflatten(children, aux_data):
        return ExecutionContextManager(
            _context_manager=aux_data[0],
            compute_dtype=aux_data[1],
            device=aux_data[2],
            distributed=aux_data[3],
            rank=aux_data[4],
            world_size=aux_data[5],
            mesh=aux_data[6],
        )

    pytree.register_pytree_node(ExecutionContextManager, _manager_flatten, _manager_unflatten)

    def _context_flatten(obj: ExecutionContext):
        children = [obj.models, obj.opt_groups, obj.step, obj.batch, obj.meta]
        aux_data = (obj.executor, obj.manager)

        return children, aux_data

    def _context_unflatten(children, aux_data):
        executor, manager = aux_data
        models, opt_groups, step, batch, meta = children

        return ExecutionContext(
            step=step,
            batch=batch,
            models=models,
            opt_groups=opt_groups,
            executor=executor,
            manager=manager,
            meta=meta,
        )

    pytree.register_pytree_node(ExecutionContext, _context_flatten, _context_unflatten)

if _BACKEND == "jax":
    import flax.nnx as nnx
    import jax.tree_util as jtu

    def _optim_group_flatten(obj: OptimGroup):
        children = (obj.optimizer, obj.scaler_state)
        aux_data = (obj.name, obj.targets, obj.scheduler, obj.scaler)
        return children, aux_data

    def _optim_group_unflatten(aux_data, children):
        return OptimGroup(
            name=aux_data[0],
            targets=aux_data[1],
            optimizer=children[0],
            scheduler=aux_data[2],
            scaler=aux_data[3],
            scaler_state=children[1],
        )

    jtu.register_pytree_node(OptimGroup, _optim_group_flatten, _optim_group_unflatten)

    def _manager_flatten(obj: ExecutionContextManager):
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

    def _manager_unflatten(aux_data, children):
        return ExecutionContextManager(
            _context_manager=aux_data[0],
            compute_dtype=aux_data[1],
            device=aux_data[2],
            distributed=aux_data[3],
            rank=aux_data[4],
            world_size=aux_data[5],
            mesh=aux_data[6],
        )

    jtu.register_pytree_node(ExecutionContextManager, _manager_flatten, _manager_unflatten)

    def _context_flatten(obj: ExecutionContext):
        """ """
        nnx_containers = (obj.models, obj.opt_groups)
        graphdef, state = nnx.split(nnx_containers)
        children = (obj.step, obj.batch, state, obj.meta)
        aux_data = (obj.executor, graphdef, obj.manager)

        return children, aux_data

    def _context_unflatten(aux_data, children):
        executor, graphdef, manager = aux_data
        step, batch, state, meta = children
        models, opt_groups = nnx.merge(graphdef, state)
        return ExecutionContext(
            step=step,
            batch=batch,
            models=models,
            opt_groups=opt_groups,
            executor=executor,
            meta=meta,
            manager=manager,
        )

    jtu.register_pytree_node(ExecutionContext, _context_flatten, _context_unflatten)

__all__ = [
    "ComposedContextManager",
    "OptimGroup",
    "ExecutionContextManager",
    "ExecutionContext",
    "CtxRef",
]
