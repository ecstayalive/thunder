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


if _BACKEND == "jax":
    import flax.nnx as nnx

    def _optim_group_flatten(obj: OptimGroup):
        children = (obj.optimizer,)
        aux_data = (obj.name, obj.targets, obj.scheduler)
        return children, aux_data

    def _optim_group_unflatten(aux_data, children):
        return OptimGroup(
            name=aux_data[0],
            targets=aux_data[1],
            optimizer=children[0],
            scheduler=aux_data[2],
        )

    def _context_flatten(obj: ExecutionContext):
        children = (obj.step, obj.batch, obj.models, obj.opt_groups)
        aux_data = (obj.executor, obj.meta)
        return children, aux_data

    def _context_unflatten(aux_data, children):
        return ExecutionContext(
            step=children[0],
            batch=children[1],
            models=children[2],
            opt_groups=children[3],
            executor=aux_data[0],
            meta=aux_data[1],
        )
