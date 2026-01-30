from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from ..data import Batch
    from ..module import ModelPack
    from ..operation import Objective


@runtime_checkable
class Executor(Protocol):
    """ """

    backend: str

    def init(
        self,
        model: ModelPack,
        optim_config: Dict[str, Any],
        distributed_strategy: Optional[Callable] = None,
    ) -> ExecutionContext:
        """ """
        raise NotImplementedError

    def optimize(
        self,
        ctx: ExecutionContext,
        opt: str,
        objectives: Tuple[Objective],
        max_grad_norm: float = 1.0,
    ) -> Dict[str, Any]:
        """ """
        ...

    @staticmethod
    def jit(fn: Callable): ...

    @staticmethod
    def devices(backend: str): ...
    @staticmethod
    def default_device(device): ...
    @staticmethod
    def to_device(data: Any, device) -> Any: ...
    @staticmethod
    def to_numpy(data: Any) -> Any: ...
    @staticmethod
    def to_jax(data: Any) -> Any: ...
    @staticmethod
    def to_torch(data: Any) -> Any: ...
    @staticmethod
    def to_warp(data: Any) -> Any: ...
    @staticmethod
    def to_dlpack(data: Any) -> Any: ...
    @staticmethod
    def to(data: Any, *args, **kwargs) -> Any: ...
    @staticmethod
    def from_numpy(data: Any) -> Any: ...
    @staticmethod
    def from_jax(data: Any) -> Any: ...
    @staticmethod
    def from_torch(data: Any) -> Any: ...
    @staticmethod
    def from_warp(data: Any) -> Any: ...
    @staticmethod
    def from_dlpack(data: Any) -> Any: ...
