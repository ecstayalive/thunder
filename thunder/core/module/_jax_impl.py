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

    def __init__(self, module: nnx.Module, _backend: str = "jax"):
        self._module = module
        self._backend = _backend

        self._base_methods: Set[str] = set(dir(nnx.Module()))
        self._bind_methods()

    def __call__(self, *args, **kwargs) -> Any:
        """ """
        return self._module(*args, **kwargs)

    def _bind_methods(self):
        """ """
        for name in dir(self._module):
            if not name.startswith("_") and name not in self._base_methods and name != "forward":
                attr = getattr(self._module, name)
                if callable(attr):
                    setattr(self, name, attr)

    def __getattr__(self, name: str) -> Any:
        """ """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self._module, name)
