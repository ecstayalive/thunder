from typing import Any, Optional

import torch.nn as nn


class TorchModule(nn.Module):
    def __init__(self, module: nn.Module, backend: str = "torch"):
        super().__init__()
        self._backend = backend
        self._module = module

    def forward(self, x, state: Optional[Any] = None, carry: Optional[Any] = None, **kwargs):
        if carry is not None:
            return self._module(x, carry, **kwargs)
        return self._module(x, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name)
