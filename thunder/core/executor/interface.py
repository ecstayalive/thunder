from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from ..data import Batch
    from ..module import ModelPack
    from ..operation import Objective


@runtime_checkable
class ExecutorProtocol(Protocol):
    """ """

    def init(self, model: ModelPack, optim_config: Dict[str, Any]) -> ExecutionContext:
        """ """
        ...

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

    def cond(self, predicate: Any, fn: Callable[[Any], Any], operand: Any) -> Any: ...

    def to_device(self, data: Any) -> Any: ...

    def to_numpy(self, data: Any) -> Any: ...

    @staticmethod
    def jit(fn: Callable): ...
