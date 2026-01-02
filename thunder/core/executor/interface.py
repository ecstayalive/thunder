from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from ..data import Batch
    from ..module import ModelPack, ThunderModule
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
        opt: str,
        objectives: list[Objective],
        max_grad_norm: float = 1.0,
    ) -> Dict[str, Any]:
        """ """
        ...

    def init(self, model: ThunderModule, optim_config: Dict[str, Any]) -> ExecutionContext:
        """ """
        ...

    def jit(self, fn: Callable): ...

    def to_device(self, data: Any) -> Any: ...

    def to_numpy(self, data: Any) -> Any: ...
