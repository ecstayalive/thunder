from typing import Any, Iterator, Set, Tuple

from flax import nnx


class JaxModelPack(nnx.Module):
    """ """

    def __init__(self, **kwargs):
        for name, module in kwargs.items():
            setattr(self, name, module)

        self._keys = tuple(kwargs.keys())

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def keys(self) -> Iterator[str]:
        return iter(self._keys)

    def items(self) -> Iterator[Tuple[str, Any]]:
        for k in self._keys:
            yield k, getattr(self, k)

    @property
    def _fields(self):
        return self._keys


class JaxModule(nnx.Module):
    """ """

    backend: str = "jax"
