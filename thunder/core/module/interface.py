from __future__ import annotations

from typing import Any, Iterator, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class ModulePackProtocol(Protocol):
    def __init__(self, **kwargs): ...

    def __getitem__(self, key: str) -> Any: ...

    def get(self, name: str) -> Any: ...

    def keys(self) -> Iterator[str]: ...

    def items(self) -> Iterator[Tuple[str, Any]]: ...

    @property
    def _fields(self): ...


@runtime_checkable
class ModuleProtocol(Protocol):
    """ """

    _module: Any
    _backend: str

    def __init__(self, module: Any, backend: str = ...) -> None: ...

    def __call__(self, x: Any, state: Any = None, carry: Any = None, **kwargs: Any) -> Any: ...

    def get_params(self) -> Any: ...
