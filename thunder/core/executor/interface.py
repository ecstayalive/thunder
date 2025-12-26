from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Protocol, Tuple, runtime_checkable

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from ..data import Batch, ModelPack
    from ..module import ThunderModule
    from ..operation import Objective


@runtime_checkable
class ExecutorProtocol(Protocol):
    """ """

    @abstractmethod
    def call(self, model: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """ """
        ...

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

    def init(
        self, model: ThunderModule, batch: Batch, optim_config: Dict[str, Any]
    ) -> ExecutionContext:
        """ """
        ...

    def to_device(self, data: Any) -> Any: ...

    def to_numpy(self, data: Any) -> Any: ...
