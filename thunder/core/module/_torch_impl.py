from typing import Any, Iterator, Tuple

import torch
import torch.nn as nn


class TorchModelPack(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for name, module in kwargs.items():
            self.add_module(name, module)

        self._keys = list(kwargs.keys())

    def __getitem__(self, key: str) -> nn.Module:
        """ """
        return getattr(self, key)

    def keys(self) -> Iterator[str]:
        return iter(self._keys)

    def items(self) -> Iterator[Tuple[str, nn.Module]]:
        for k in self._keys:
            yield k, getattr(self, k)

    @property
    def _fields(self):
        return self._keys


class TorchModule(nn.Module):
    """ """

    def __init__(self, module: nn.Module, _backend: str = "torch"):
        super().__init__()
        self.add_module("_module", module)
        self._module: nn.Module
        self._backend = _backend
        self._base_methods = set(dir(nn.Module()))
        self._bind_methods()

    def _bind_methods(self):
        """ """
        for name in dir(self._module):
            if not name.startswith("_") and name not in self._base_methods and name != "forward":
                method = getattr(self._module, name)
                if callable(method):
                    setattr(self, name, method)

    def forward(self, *args, **kwargs):
        """ """
        return self._module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """ """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name)

    def to_ddp(self, **kwargs) -> nn.Module:
        """ """
        device_ids = kwargs.pop("device_ids", None)
        if device_ids is None and next(self.parameters()).is_cuda:
            device_ids = [next(self.parameters()).device.index]

        return torch.nn.parallel.DistributedDataParallel(self, device_ids=device_ids, **kwargs)

    def compile(self, **kwargs):
        """ """
        return torch.compile(self, **kwargs)
