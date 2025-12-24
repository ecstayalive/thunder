from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Protocol, Tuple, runtime_checkable

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from ..data import Batch
    from ..module import ThunderModule
    from ..operation import Objective


@runtime_checkable
class ExecutorProtocol(Protocol):
    """ """

    def optimize(
        self,
        ctx: ExecutionContext,
        target: str,
        opt: str,
        objectives: list[Objective],
        max_grad_norm: float = 1.0,
    ) -> tuple[dict[str, Any], Any, Any]:
        """ """
        ...

    def init_state(
        self, model: ThunderModule, batch: Batch, optim_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Returns:
            (params, opt_states, meta)
        """
        ...

    def to_device(self, data: Any) -> Any: ...

    def to_numpy(self, data: Any) -> Any: ...
