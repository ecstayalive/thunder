from functools import wraps
from typing import Any, Callable, Optional

import torch.nn as nn


class TorchModule(nn.Module):
    def __init__(self, module: nn.Module, name: str, backend: str = "torch"):
        super().__init__()
        self.add_module("_module", module)
        self._module: nn.Module
        self._name = name
        self._backend = backend
        self._bind_methods(module)

    def _bind_methods(self, module: nn.Module):
        """ """
        base_methods = set(dir(nn.Module()))
        for name in dir(module):
            if not name.startswith("_") and name not in base_methods and name != "forward":
                method = getattr(module, name)
                if callable(method):
                    setattr(self, name, self._wrap_ignoring_state(method))

    @staticmethod
    def _wrap_ignoring_state(fn: Callable):
        """ """

        @wraps(fn)
        def wrapper(*args, state=None, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    def forward(self, x, state=None, carry=None, **kwargs):
        """ """
        if carry is not None:
            return self._module(x, carry, **kwargs)
        return self._module(x, **kwargs)
