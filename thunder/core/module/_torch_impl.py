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
        return self.get_submodule(name)

    def keys(self) -> Iterator[str]:
        return iter(self._keys)

    def items(self) -> Iterator[Tuple[str, nn.Module]]:
        for k in self._keys:
            yield k, getattr(self, k)

    @staticmethod
    def _unwrap(module: nn.Module) -> nn.Module:
        """Peel the executor's runtime wrappers down to the canonical module."""
        while True:
            if isinstance(module, nn.parallel.DistributedDataParallel):
                module = module.module
            elif hasattr(module, "_orig_mod"):
                module = module._orig_mod
            else:
                return module

    def state_dict(self, *, keep_vars: bool = False) -> Dict[str, torch.Tensor]:
        out = {}
        for name, module in self.items():
            sub = self._unwrap(module).state_dict(keep_vars=keep_vars)
            out.update({f"{name}.{key}": value for key, value in sub.items()})
        return out

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # Canonicalize first so checkpoints written by older versions (with
        # wrapper segments baked into the keys) keep loading.
        def canonical(key: str) -> str:
            segments = (s for s in key.split(".") if s not in ("module", "_orig_mod"))
            return ".".join(segments)

        state_dict = {canonical(key): value for key, value in state_dict.items()}
        if strict:
            stray = [k for k in state_dict if k.split(".", 1)[0] not in self._keys]
            if stray:
                raise KeyError(f"Unexpected model keys in state_dict: {stray}")
        for name, module in self.items():
            prefix = f"{name}."
            sub = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            self._unwrap(module).load_state_dict(sub, strict=strict)

    @property
    def _fields(self):
        return self._keys


class TorchModule(nn.Module, ABC):
    """ """

    backend: str = "torch"

    @abstractmethod
    def forward(
        self,
        embedding: torch.Tensor | Dict[str, torch.Tensor],
        carry: Any = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass
