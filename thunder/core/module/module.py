from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class ModuleProtocol(Protocol):
    """ """

    _module: Any
    _backend: str

    def __init__(self, module: Any, backend: str = ...) -> None: ...

    def __call__(self, x: Any, state: Any = None, carry: Any = None, **kwargs: Any) -> Any: ...

    def get_params(self) -> Any: ...
