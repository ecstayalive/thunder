from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Mapping, Tuple

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

    # def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
    #     super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    # def load_state_dict(
    #     self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    # ):
    #     super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)

    @property
    def _fields(self):
        return self._keys


class TorchModule(nn.Module, ABC):
    """ """

    backend: str = "torch"

    @abstractmethod
    def forward(
        self, embedding: torch.Tensor | Dict[str, torch.Tensor], carry: Any = None, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass
