from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from thunder.core.context import ExecutionContext, OptimGroup

if TYPE_CHECKING:
    from ..data import Batch
    from ..module import ModelPack
    from ..operation import Objective


class TorchExecutor:
    """
    Args:
    """

    backend = "torch"

    def __init__(
        self,
        mixed_precision: bool = False,
        compile: bool = True,
        compile_mode: str = "default",
        distributed: bool = False,
        device: Optional[str] = None,
    ):
        """ """
        if device == "gpu":
            device = "cuda"
        self.device = self.default_device(device)
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

    def init(
        self,
        models: ModelPack,
        optim_config: Dict[str, Any],
        distributed_strategy: Optional[Callable[[nn.Module], nn.Module]] = None,
    ) -> ExecutionContext:
        """ """
        new_model_pack = {}
        for name in models._fields:
            model = getattr(models, name)
            model = model.to(self.device)
            if self.distributed:
                if distributed_strategy is not None:
                    model = distributed_strategy(model)
                else:
                    device_ids = [self.device.index] if self.device.type == "cuda" else None
                    model = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=device_ids, find_unused_parameters=True
                    )
            new_model_pack[name] = model
        new_models_pack = type(models)(**new_model_pack)

        opt_groups = {}
        for opt_key, cfg in optim_config.items():
            cfg = cfg.copy()
            target_names = cfg.pop("targets")
            if isinstance(target_names, str):
                target_names = [target_names]
            target_names = tuple(target_names)
            all_optimize_params = []
            for t_name in target_names:
                if t_name not in new_model_pack:
                    raise ValueError(f"Optimizer target '{t_name}' not found in models.")
                target_module = new_model_pack[t_name]
                all_optimize_params.append({"params": target_module.parameters()})

            cls_name = cfg.pop("class", "Adam")
            OptimCls = getattr(torch.optim, cls_name)
            optimizer = OptimCls(all_optimize_params, **cfg)
            opt_groups[opt_key] = OptimGroup(
                name=opt_key, targets=target_names, optimizer=optimizer, scheduler=None
            )

        ctx = ExecutionContext.create(executor=self, models=new_models_pack, opt_groups=opt_groups)
        return ctx

    def _forward(
        self, objectives: Tuple[Objective, ...], batch: Batch, models: ModelPack
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """ """
        total_loss = torch.zeros((), device=self.device)
        metrics = {}
        for obj in objectives:
            loss, m = obj.forward(batch, models)
            total_loss = total_loss + loss
            metrics.update(m)
        return total_loss, metrics

    # @torch._dynamo.disable
    def optimize(
        self,
        ctx: ExecutionContext,
        opt: str,
        objectives: Tuple[Objective, ...],
        max_grad_norm: float = 1.0,
    ) -> Dict[str, Any]:
        """ """
        optim_group = ctx.opt_groups[opt]
        optimizer: torch.optim.Optimizer = optim_group.optimizer
        if self._compile_enabled and self._compiled_forward is None:
            self._compiled_forward = torch.compile(self._forward, mode=self._compile_mode)
        forward_fn = self._compiled_forward if self._compile_enabled else self._forward
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.mixed_precision):
            loss, metrics = forward_fn(objectives, ctx.batch, ctx.models)
        metrics["total_loss"] = loss
        self.scaler.scale(loss).backward()
        if max_grad_norm > 0:
            self.scaler.unscale_(optimizer)
            params_to_clip = [p for group in optimizer.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
        self.scaler.step(optimizer)
        self.scaler.update()
        if optim_group.scheduler is not None:
            optim_group.scheduler.step()
        return metrics

    @staticmethod
    def jit(fn: Callable, **kwargs):
        compile_args = {"mode": "default"}
        compile_args.update(kwargs)

        def wrapper(f):
            return torch.compile(f, **compile_args)

        if fn is None:
            return wrapper
        return wrapper(fn)

    @staticmethod
    def _recursive_map(func, data):
        """Applies func to every leaf in a nested structure (Dict/List/Tuple)."""
        if isinstance(data, dict):
            return {k: TorchExecutor._recursive_map(func, v) for k, v in data.items()}
        elif isinstance(data, list):
            return [TorchExecutor._recursive_map(func, v) for v in data]
        elif isinstance(data, tuple):
            return tuple(TorchExecutor._recursive_map(func, v) for v in data)
        return func(data)

    @staticmethod
    def devices(backend: str):
        return (torch.device(backend),)

    @staticmethod
    def default_device(device: Optional[str | torch.device] = None):
        if isinstance(device, torch.device):
            return device
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def to_device(data: Any, device: Optional[torch.device | str] = None) -> Any:
        target_device = TorchExecutor.default_device(device)

        def _put_data(x):
            if isinstance(x, torch.Tensor):
                return x.to(target_device, non_blocking=True)
            return x

        return TorchExecutor._recursive_map(_put_data, data)

    @staticmethod
    def to_numpy(data: Any) -> Any:
        return TorchExecutor._recursive_map(lambda x: x.detach().cpu().numpy(), data)

    @staticmethod
    def to_jax(data: Any) -> Any:
        try:
            import jax
            from torch.utils.dlpack import to_dlpack

            return TorchExecutor._recursive_map(
                lambda x: jax.dlpack.from_dlpack(to_dlpack(x)), data
            )
        except ImportError:
            raise ImportError("Please install `jax` to use `to_jax` function.")

    @staticmethod
    def to_torch(data: Any) -> torch.Tensor:
        return data

    @staticmethod
    def to_warp(data: Any) -> Any:
        try:
            import warp

            return TorchExecutor._recursive_map(warp.from_torch, data)
        except ImportError:
            raise ImportError("Please install `warp-lang` to use `to_warp` function.")

    @staticmethod
    def to_dlpack(data: Any):
        from torch.utils.dlpack import to_dlpack

        return TorchExecutor._recursive_map(lambda x: to_dlpack(x), data)

    @staticmethod
    def to(data: Any, target: Any, non_blocking: bool = True):
        if isinstance(target, (str, torch.dtype, torch.device)):
            if isinstance(data, torch.Tensor):
                return data.to(target, non_blocking=non_blocking)
            return torch.as_tensor(data).to(target, non_blocking=non_blocking)

        if isinstance(data, (dict, list, tuple)):
            return TorchExecutor._recursive_map(
                lambda x: TorchExecutor.to(x, target, non_blocking), data
            )

        if isinstance(target, type):
            name = target.__name__
            module = target.__module__

            if name == "ndarray" and "numpy" in module:
                return TorchExecutor.to_numpy(data)

            if (name == "Array" or "jax" in name) and ("jax" in module):
                return TorchExecutor.to_jax(data)

            if module.startswith("warp"):
                return TorchExecutor.to_warp(data)

            if name == "Tensor" and "torch" in module:
                return data

        raise ValueError(f"TorchExecutor.to: Unknown target '{target}'")

    @staticmethod
    def from_numpy(data: Any):
        return TorchExecutor._recursive_map(torch.as_tensor, data)

    @staticmethod
    def from_jax(data: Any):
        try:
            from jax import dlpack as jdlpack
            from torch.utils.dlpack import from_dlpack

            return TorchExecutor._recursive_map(lambda x: from_dlpack(jdlpack.to_dlpack(x)), data)
        except ImportError:
            raise ImportError("Please install `jax` to use `from_jax` function.")

    @staticmethod
    def from_warp(data: Any):
        try:
            import warp

            return TorchExecutor._recursive_map(warp.to_torch, data)
        except ImportError:
            raise ImportError("Please install `warp-lang` to use `from_warp` function.")

    @staticmethod
    def from_dlpack(data: Any):
        from torch.utils.dlpack import from_dlpack

        return TorchExecutor._recursive_map(from_dlpack, data)
