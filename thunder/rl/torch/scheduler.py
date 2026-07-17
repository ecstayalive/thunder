from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import torch

from thunder.utils import Scalar


@dataclass(slots=True)
class AdaptiveKlSchedulerSpec:
    cls: type["AdaptiveKlScheduler"] = field(
        default_factory=lambda: AdaptiveKlScheduler
    )
    key: str = "kl"
    desired_kl: float = 0.01
    min_lr: float = 1e-5
    max_lr: float = 1e-2
    factor: float = 1.2
    strict: bool = True


class AdaptiveKlScheduler:
    """Adjust Torch optimizer learning rates from an exported KL value.

    The scheduler is intentionally small and Torch-like: it owns the optimizer
    and `step` receives the exports produced by objectives during the current
    optimization call.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        key: str = "kl",
        desired_kl: float = 0.01,
        min_lr: float = 1e-5,
        max_lr: float = 1e-2,
        factor: float = 1.2,
        strict: bool = True,
    ):
        if factor <= 1.0:
            raise ValueError("AdaptiveKlScheduler requires factor > 1.0.")
        if min_lr <= 0.0:
            raise ValueError("AdaptiveKlScheduler requires min_lr > 0.")
        if max_lr < min_lr:
            raise ValueError("AdaptiveKlScheduler requires max_lr >= min_lr.")
        if desired_kl <= 0.0:
            raise ValueError("AdaptiveKlScheduler requires desired_kl > 0.")

        self.optimizer = optimizer
        self.key = key
        self.desired_kl = desired_kl
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.factor = factor
        self.strict = strict

    @classmethod
    def factory(
        cls,
        optimizer: torch.optim.Optimizer,
        spec: AdaptiveKlSchedulerSpec,
    ) -> AdaptiveKlScheduler:
        return cls(
            optimizer=optimizer,
            key=spec.key,
            desired_kl=spec.desired_kl,
            min_lr=spec.min_lr,
            max_lr=spec.max_lr,
            factor=spec.factor,
            strict=spec.strict,
        )

    def step(self, exports: Dict[str, Any]) -> Dict[str, Any]:
        if self.key not in exports:
            if self.strict:
                raise KeyError(
                    f"AdaptiveKlScheduler expected export {self.key!r}, "
                    f"available exports: {tuple(exports)}."
                )
            return {}

        kl = exports[self.key].detach()
        if not self.optimizer.param_groups:
            raise RuntimeError("Optimizer has no parameter groups.")
        lr = self.optimizer.param_groups[0]["lr"]
        if not isinstance(lr, torch.Tensor):
            raise TypeError("AdaptiveKlScheduler requires Tensor optimizer lr.")
        min_lr = torch.as_tensor(self.min_lr, device=lr.device, dtype=lr.dtype)
        max_lr = torch.as_tensor(self.max_lr, device=lr.device, dtype=lr.dtype)
        factor = torch.as_tensor(self.factor, device=lr.device, dtype=lr.dtype)
        desired_kl = torch.as_tensor(self.desired_kl, device=kl.device, dtype=kl.dtype)
        decrease = torch.maximum(min_lr, lr / factor)
        increase = torch.minimum(max_lr, lr * factor)
        new_lr = torch.where(
            kl > desired_kl * 2.0,
            decrease,
            torch.where((kl > 0.0) & (kl < desired_kl / 2.0), increase, lr),
        )
        for group in self.optimizer.param_groups:
            group_lr: torch.Tensor = group["lr"]
            group_lr.copy_(new_lr.to(device=group_lr.device, dtype=group_lr.dtype))
        return {"scheduler/lr": Scalar(new_lr.detach())}
