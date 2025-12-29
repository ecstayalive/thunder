from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from .data import Batch
    from .executor.interface import ExecutorProtocol
    from .module import ModelPack, ThunderModule


@dataclass(slots=True)
class OptimGroup:
    """
    Args:
        name:
        targets:
        params:
        opt_state:
        tx: `optax.GradientTransformation` for `jax,` None for `torch`
        scheduler: learning rate scheduler, None for `jax`
    """

    name: str
    targets: Tuple[str, ...]
    params: Dict[str, Any]
    opt_state: Any
    tx: Optional[Any] = None
    scheduler: Optional[Any] = None


@dataclass(slots=True)
class ExecutionContext:
    """
    Args:
        step:
        batch:
        params:
        opt_groups:
        executor:
        model:
        meta:
    """

    step: int
    batch: Optional[Batch]
    models: ModelPack
    params: Dict[str, Any]
    opt_groups: Dict[str, OptimGroup]
    executor: ExecutorProtocol
    meta: Dict[str, Any] = field(default_factory=dict)

    def replace(self, **changes) -> ExecutionContext:
        return replace(self, **changes)

    def apply_gradients(
        self, opt: str, new_params_subset: Optional[Any], new_opt_state: Optional[Any]
    ) -> ExecutionContext:
        """
        Unified handling of gradient back propagation logic.
        Args:
            opt: Corresponding optimizer name (e.g. “actor_opt”)
            new_params:
                - Torch: Typically None (as Torch Optimizer modifies references in-place)
                - JAX: New parameters PyTree
            new_opt_state:
                - Torch: Typically None
                - JAX: New opt_state PyTree
        Returns:
            ExecutionContext:
                - Torch: Returns self (reference unchanged)
                - JAX: Returns a new Context instance (parameters replaced)
        """
        if new_params_subset is None:
            return self
        new_full_params = self.params.copy()
        new_full_params.update(new_params_subset)
        target_group = self.opt_groups[opt]
        new_group = replace(target_group, params=new_params_subset, opt_state=new_opt_state)
        new_groups_dict = self.opt_groups.copy()
        new_groups_dict[opt] = new_group

        return self.replace(params=new_full_params, opt_groups=new_groups_dict)

    def set_param(self, key: str, value: Any) -> None:
        """In-place update params."""
        self.params[key] = value

    def get_param(self, key: str) -> Any:
        """ """
        try:
            return self.params[key]
        except KeyError:
            raise ValueError(
                f"Parameter group '{key}' not found in Context. Available: {list(self.params.keys())}"
            )

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
        cls, executor: ExecutorProtocol, models: ModelPack, batch: Optional[Batch] = None
    ) -> ExecutionContext:
        """ """
        return cls(
            step=0,
            batch=batch,
            models=models,
            params={},
            opt_groups=None,
            executor=executor,
            meta={},
        )
