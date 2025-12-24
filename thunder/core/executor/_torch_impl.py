from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from ..data import Batch
    from ..module._torch_impl import TorchModule
    from ..operation import Objective


class TorchExecutor:
    def __init__(self, device: str = None, mixed_precision: bool = False, compile: bool = True):
        """ """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda":
            if torch.cuda.get_device_capability() >= (8, 0):
                torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
        self.mixed_precision = mixed_precision
        self.use_scaler = self.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_scaler)
        self._compile_enabled = compile
        if self._compile_enabled:
            self._compiled_loss_fn = torch.compile(self._forward)
        else:
            self._compiled_loss_fn = self._forward

    def to_device(self, data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_device(v) for v in data)
        return data

    def to_numpy(self, data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data

    def _forward(
        self, objectives: list[Objective], batch: Batch, model: nn.Module, target_params_ref: Any
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_loss = torch.tensor(0.0, device=self.device)
        metrics = {}
        for obj in objectives:
            loss, m = obj.forward(batch, model, target_params_ref)
            total_loss += loss
            metrics.update(m)
        return total_loss, metrics

    def optimize(
        self,
        ctx: ExecutionContext,
        target: str,
        opt: str,
        objectives: list[Objective],
        max_grad_norm: float = 1.0,
    ) -> tuple[dict, Any, Any]:
        target_module_or_params = ctx.params.get(target)
        optimizer: torch.optim.Optimizer = ctx.opt_states[opt]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            loss, metrics = self._compiled_loss_fn(
                objectives, ctx.batch, ctx.model, target_module_or_params
            )
        metrics["loss_total"] = loss
        self.scaler.scale(loss).backward()
        if max_grad_norm > 0:
            self.scaler.unscale_(optimizer)
            if isinstance(target_module_or_params, nn.Module):
                torch.nn.utils.clip_grad_norm_(target_module_or_params.parameters(), max_grad_norm)
            else:
                parameters = [p for group in optimizer.param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        self.scaler.step(optimizer)
        self.scaler.update()
        return metrics, None, None

    def init_state(
        self, model: TorchModule, batch: Batch, optim_config: Dict[str, Any]
    ) -> Tuple[Dict, Dict, Dict]:
        """ """
        ctx_params_refs = {"default": model}
        opt_states = {}
        for opt_key, cfg in optim_config.items():
            cfg = cfg.copy()
            target_path = cfg.pop("target", "default")
            if target_path == "default":
                target_obj = model
            else:
                try:
                    target_obj = getattr(model, target_path)
                except AttributeError as e:
                    raise ValueError(
                        f"Optimizer '{opt_key}' targets '{target_path}', "
                        f"but model has no such attribute."
                    ) from e
            ctx_params_refs[target_path] = target_obj
            cls_name = cfg.pop("class", "Adam")
            if not hasattr(torch.optim, cls_name):
                raise ValueError(f"Unknown torch optimizer: {cls_name}")
            OptimCls = getattr(torch.optim, cls_name)
            trainable_params = (
                target_obj.parameters() if isinstance(target_obj, nn.Module) else target_obj
            )
            opt_states[opt_key] = OptimCls(trainable_params, **cfg)
        return ctx_params_refs, opt_states, {}
