from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from .data import Batch
    from .executor.interface import ExecutorProtocol
    from .module import ModelPack, ThunderModule

_BACKEND = os.getenv("THUNDER_BACKEND", "torch").lower()


@dataclass(slots=True)
class OptimGroup:
    """
    Args:
        name:
        targets:
        params:
        optimizer: `nnx.Optimizer` for jax, `torch.optim.Optimizer` for torch
        scheduler: learning rate scheduler, None for `jax`
    """

    name: str
    targets: Tuple[str, ...]
    optimizer: Any
    scheduler: Optional[Any] = None


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
    executor: ExecutorProtocol
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
        cls, executor: ExecutorProtocol, models: ModelPack, opt_groups: Dict[str, OptimGroup]
    ) -> ExecutionContext:
        """ """
        return cls(
            step=0, batch=None, models=models, opt_groups=opt_groups, executor=executor, meta={}
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
        children = [obj.optimizer]
        aux_data = (obj.name, obj.targets, obj.scheduler)
        return children, aux_data

    def _optim_group_unflatten(aux_data, children):
        return OptimGroup(
            name=aux_data[0], targets=aux_data[1], optimizer=children[0], scheduler=aux_data[3]
        )

    pytree.register_pytree_node(OptimGroup, _optim_group_flatten, _optim_group_unflatten)

    def _context_flatten(obj: ExecutionContext):
        children = [obj.models, obj.opt_groups, obj.step, obj.batch, obj.meta]
        aux_data = obj.executor

        return children, aux_data

    def _context_unflatten(aux_data, children):
        executor = aux_data
        models, opt_groups, step, batch, meta = children

        return ExecutionContext(
            step=step,
            batch=batch,
            models=models,
            opt_groups=opt_groups,
            executor=executor,
            meta=meta,
        )

    pytree.register_pytree_node(ExecutionContext, _context_flatten, _context_unflatten)

if _BACKEND == "jax":
    import flax.nnx as nnx
    import jax.tree_util as jtu

    def _optim_group_flatten(obj: OptimGroup):
        children = (obj.optimizer,)
        aux_data = (obj.name, obj.targets, obj.scheduler)
        return children, aux_data

    def _optim_group_unflatten(aux_data, children):
        return OptimGroup(
            name=aux_data[0], targets=aux_data[1], optimizer=children[0], scheduler=aux_data[2]
        )

    jtu.register_pytree_node(OptimGroup, _optim_group_flatten, _optim_group_unflatten)

    def _context_flatten(obj: ExecutionContext):
        """ """
        nnx_containers = (obj.models, obj.opt_groups)
        graphdef, state = nnx.split(nnx_containers)
        children = (obj.step, obj.batch, state, obj.meta)
        aux_data = (obj.executor, graphdef)

        return children, aux_data

    def _context_unflatten(aux_data, children):
        executor, graphdef = aux_data
        step, batch, state, meta = children
        models, opt_groups = nnx.merge(graphdef, state)
        return ExecutionContext(
            step=step,
            batch=batch,
            models=models,
            opt_groups=opt_groups,
            executor=executor,
            meta=meta,
        )

    jtu.register_pytree_node(ExecutionContext, _context_flatten, _context_unflatten)

__all__ = ["OptimGroup", "ExecutionContext", "CtxRef"]
