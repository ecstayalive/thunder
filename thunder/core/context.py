from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .data import Batch
    from .executor.executor import ExecutorProtocol
    from .module import ThunderModule


@dataclass
class ExecutionContext:
    """ """

    step: int
    batch: Optional[Batch]
    params: Dict[str, Any]
    opt_states: Dict[str, Any]
    executor: ExecutorProtocol
    model: ThunderModule
    meta: Dict[str, Any] = field(default_factory=dict)

    def replace(self, **changes) -> ExecutionContext:
        return replace(self, **changes)

    def apply_gradients(
        self, target: str, opt: str, new_params: Optional[Any], new_opt_state: Optional[Any]
    ) -> ExecutionContext:
        """
        Unified handling of gradient back propagation logic.
        Args:
            target: Name of the target parameter group to update (e.g. “actor”)
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
        if new_params is None and new_opt_state is None:
            return self
        changes = {}
        if new_params is not None:
            updated_params = self.params.copy()
            updated_params[target] = new_params
            changes["params"] = updated_params
        if new_opt_state is not None:
            updated_opt_states = self.opt_states.copy()
            updated_opt_states[opt] = new_opt_state
            changes["opt_states"] = updated_opt_states
        return self.replace(**changes)

    def set_param(self, key: str, value: Any) -> None:
        """In-place update params."""
        self.params[key] = value

    def get_param(self, key: str) -> Any:
        """ """
        if key not in self.params:
            raise ValueError(
                f"Parameter group '{key}' not found in Context. Available: {list(self.params.keys())}"
            )
        return self.params[key]

    def update_meta(self, **kwargs) -> None:
        """ """
        self.meta.update(kwargs)

    @classmethod
    def create(
        cls, executor: ExecutorProtocol, model: ThunderModule, batch: Optional[Batch] = None
    ) -> ExecutionContext:
        """ """
        return cls(
            step=0, batch=batch, params={}, opt_states={}, executor=executor, model=model, meta={}
        )
