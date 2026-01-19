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

    def get(self, name: str) -> nn.Module:
        self.get_submodule(name)

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

    backend: str = "torch"
