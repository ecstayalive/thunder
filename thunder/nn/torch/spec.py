from dataclasses import dataclass
import torch.nn as nn


@dataclass
class ModelSpec:
    entry: nn.Module
    kwargs: dict

    def __call__(self, **kwargs) -> nn.Module:
        return self.entry(**self.kwargs, **kwargs)
