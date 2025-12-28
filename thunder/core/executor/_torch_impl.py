from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from thunder.core.context import ExecutionContext

if TYPE_CHECKING:
    from ..data import Batch, ModelPack
    from ..module._torch_impl import TorchModule
    from ..operation import Objective


class TorchExecutor:
    """
    Args:
    """

    def __init__(
        self,
        device: str = None,
        mixed_precision: bool = False,
        compile: bool = True,
        compile_mode: str = "default",
        distributed: bool = False,
        **kwargs,
    ):
        """ """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda":
            if torch.cuda.get_device_capability() >= (8, 0):
                torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
        self.mixed_precision = mixed_precision
        self.distributed = distributed
        self.use_scaler = self.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_scaler)
        self._compile_enabled = compile
        self._compile_mode = compile_mode
        self._compiled_forward = None

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
        self, objectives: Tuple[Objective], batch: Batch, models: ModelPack, params: Any
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_loss = torch.tensor(0.0, device=self.device)
        metrics = {}
        for obj in objectives:
            loss, m = obj.forward(batch, models, params)
            total_loss += loss
            metrics.update(m)
        return total_loss, metrics

    def optimize(
        self,
        ctx: ExecutionContext,
        opt: str,
        objectives: Tuple[Objective],
        max_grad_norm: float = 1.0,
    ) -> Tuple[dict, Any, Any]:
        optimizer: torch.optim.Optimizer = ctx.opt_states[opt]
        if self._compile_enabled and self._compiled_forward is None:
            self._compiled_forward = torch.compile(self._forward, mode=self._compile_mode)
        forward_fn = self._compiled_forward if self._compile_enabled else self._forward
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            loss, metrics = forward_fn(objectives, ctx.batch, ctx.models, ctx.params)
        metrics["loss_total"] = loss
        self.scaler.scale(loss).backward()
        if max_grad_norm > 0:
            self.scaler.unscale_(optimizer)
            params_to_clip = [p for group in optimizer.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
        self.scaler.step(optimizer)
        self.scaler.update()
        return metrics, None, None

    def init(
        self, models: ModelPack, batch: Batch, optim_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, torch.optim.Optimizer], Dict[str, Any]]:
        """ """
        new_wrappers_dict = {}
        params_map = {}
        opt_states = {}
        meta_info = {}
        for name in models._fields:
            module: TorchModule = getattr(models, name)
            module = module.to(self.device)
            if self.distributed and torch.distributed.is_initialized():
                device_ids = [self.device.index] if "cuda" in str(self.device) else None
                module = torch.nn.parallel.DistributedDataParallel(
                    module, device_ids=device_ids, find_unused_parameters=True
                )
            # if self._compile_enabled:
            #     module = torch.compile(module, mode=self._compile_mode)
            new_wrappers_dict[name] = module
            params_map[name] = module
        new_models_pack = type(models)(**new_wrappers_dict)
        for opt_key, cfg in optim_config.items():
            cfg = cfg.copy()
            target_name = cfg.pop("target", None)
            if isinstance(target_name, str):
                target_name = [target_name]
            all_params = []
            for t_name in target_name:
                if t_name not in params_map:
                    raise ValueError(f"Optimizer '{opt_key}' targets unknown module '{t_name}'.")
                target_module = params_map[t_name]
                all_params.append({"params": target_module.parameters()})
            OptimCls = getattr(torch.optim, cfg.pop("class", "Adam"))
            opt_states[opt_key] = OptimCls(all_params, **cfg)
            meta_info[f"{opt_key}_targets"] = target_name
        ctx = ExecutionContext.create(executor=self, models=new_models_pack, batch=batch)
        ctx = ctx.replace(params=params_map, opt_states=opt_states, meta=meta_info)
        return ctx
