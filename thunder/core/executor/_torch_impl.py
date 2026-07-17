from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils._cxx_pytree import tree_map

from thunder.core.context import (
    ExecutionContext,
    ExecutionContextManager,
    OptimGroup,
    OptimGroupSpec,
)

if TYPE_CHECKING:
    from ..module import ModelPack
    from ..operation import Objective


class _DistributedDataParallel(nn.parallel.DistributedDataParallel):
    """ """

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "module":
                raise
            return getattr(self.module, name)


class _IdentityScaler:
    def __init__(self, enabled: bool = False):
        pass

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale(self, optimizer: torch.optim.Optimizer):
        return

    def unscale_(self, optimizer: torch.optim.Optimizer):
        return

    def step(self, optimizer: torch.optim.Optimizer):
        optimizer.step()

    def update(self):
        return


class TorchExecutor:
    """
    Args:
    """

    backend = "torch"

    def __init__(
        self,
        precision: str = "fp32",
        distributed: Optional[bool] = None,
        device: Optional[str] = None,
        enable_cudnn_benchmark: bool = True,
        compile=False,
        compile_args=None,
        **kwargs,
    ):
        """ """
        if device == "gpu":
            device = "cuda"
        self.device = self.default_device(device)
        self.precision = precision
        self.compute_dtype = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }.get(precision, torch.float32)
        self.compiled_autograd = False
        self.compile = False
        if self.device.type == "cuda":
            if torch.cuda.get_device_capability() >= (8, 0):
                torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = enable_cudnn_benchmark
            torch._dynamo.allow_in_graph(ExecutionContext)
            # torch._dynamo.config.compiled_autograd = True
            self.compile = compile
            self.compile_args = {} if compile_args is None else compile_args
        self.mixed_precision = (
            self.device.type == "cuda" and self.compute_dtype is not torch.float32
        )
        # The live process group is the single source of truth: when ``distributed``
        # is left unspecified, derive it from whether one has been initialized.
        if distributed is None:
            distributed = (
                torch.distributed.is_available() and torch.distributed.is_initialized()
            )
        self.distributed = distributed
        self.use_scaler = (
            self.compute_dtype is torch.float16 and self.device.type == "cuda"
        )
        self.scaler = (
            torch.amp.GradScaler(enabled=self.use_scaler)
            if self.use_scaler
            else _IdentityScaler()
        )

    def init(
        self,
        models: ModelPack,
        optim_config: Dict[str, OptimGroupSpec | Dict[str, Any]],
        distributed_strategy: Optional[Callable[[nn.Module], nn.Module]] = None,
    ) -> ExecutionContext:
        """ """
        new_model_pack = {}
        for name in models._fields:
            model = getattr(models, name)
            model = model.to(self.device)
            if self.compile:
                model = torch.compile(model, **self.compile_args)
            if self.distributed and any(p.requires_grad for p in model.parameters()):
                if distributed_strategy is not None:
                    model = distributed_strategy(model)
                else:
                    device_ids = (
                        [self.device.index] if self.device.type == "cuda" else None
                    )
                    model = _DistributedDataParallel(
                        model, device_ids=device_ids, find_unused_parameters=False
                    )
            new_model_pack[name] = model
        new_models_pack = type(models)(**new_model_pack)

        opt_groups = {}
        for opt_key, cfg in optim_config.items():
            spec = OptimGroupSpec.factory(cfg)
            target_names = spec.target_names
            all_optimize_params = []
            for t_name in target_names:
                if t_name not in new_model_pack:
                    raise ValueError(
                        f"Optimizer target '{t_name}' not found in models."
                    )
                target_module = new_model_pack[t_name]
                all_optimize_params.append({"params": target_module.parameters()})
            optimizers = {
                "Adam": torch.optim.Adam,
                "AdamW": torch.optim.AdamW,
                "Muon": torch.optim.Muon,
            }
            optim_cls = spec.optimizer or "AdamW"
            if isinstance(optim_cls, str):
                if optim_cls not in optimizers:
                    names = ", ".join(optimizers)
                    raise ValueError(f"Torch optimizer only supports {names}.")
                optim_cls = optimizers[optim_cls]
            elif optim_cls not in optimizers.values():
                names = ", ".join(optimizers)
                raise ValueError(
                    f"Torch optimizer only supports {names}, "
                    f"but got {optim_cls.__name__}."
                )
            optim_kwargs = spec.optimizer_kwargs()
            optim_kwargs["lr"] = torch.tensor(
                float(optim_kwargs.get("lr", 1e-3)), device=self.device
            )
            optimizer = optim_cls(all_optimize_params, **optim_kwargs)
            scheduler = self._init_scheduler(spec.scheduler, optimizer)
            opt_groups[opt_key] = OptimGroup(
                name=opt_key,
                targets=target_names,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=self.scaler,
                scaler_state=None,
            )
        ctx = ExecutionContext.create(
            models=new_models_pack,
            executor=self,
            manager=self.init_manager(),
            opt_groups=opt_groups,
        )
        return ctx

    @staticmethod
    def _init_scheduler(scheduler_spec: Any, optimizer: torch.optim.Optimizer) -> Any:
        if scheduler_spec is None:
            return None
        cls = getattr(scheduler_spec, "cls", None)
        if cls is not None:
            return cls.factory(optimizer, scheduler_spec)
        raise TypeError(
            "Torch scheduler spec must define scheduler_cls with a factory method, "
            f"got {type(scheduler_spec).__name__}."
        )

    def init_manager(self) -> ExecutionContextManager:
        device_type = self.device.type
        compute_dtype = torch.get_autocast_dtype(device_type)
        amp_ctx = torch.amp.autocast(
            device_type=device_type,
            dtype=torch.get_autocast_dtype(device_type),
            enabled=self.mixed_precision,
        )
        return ExecutionContextManager(
            _context_manager=amp_ctx,
            compute_dtype=compute_dtype if self.mixed_precision else torch.float32,
            device=self.device,
            distributed=self.distributed,
            rank=torch.distributed.get_rank() if self.distributed else 0,
            world_size=torch.distributed.get_world_size() if self.distributed else 1,
            mesh=None,
        )

    def _jit_optimize(
        self, objectives: Tuple[Objective, ...], ctx: ExecutionContext
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """ """
        total_loss = torch.zeros((), device=self.device)
        metrics = {}
        for obj in objectives:
            loss, m = obj.evaluate(ctx)
            total_loss = total_loss + loss
            metrics.update(m)
        return total_loss, metrics

    def optimize(
        self,
        ctx: ExecutionContext,
        opt: str,
        objectives: Tuple[Objective, ...],
        max_grad_norm: float = 1.0,
    ) -> Tuple[ExecutionContext, Dict[str, Any]]:
        """ """
        optim_group: OptimGroup = ctx.opt_groups[opt]
        optimizer: torch.optim.Optimizer = optim_group.optimizer
        loss, metrics = self._jit_optimize(objectives, ctx)
        # metrics["total_loss"] = loss.detach()
        if optim_group.scheduler is not None:
            exports = {}
            for obj in objectives:
                exports.update(obj.export())
            self.reduce_mean(exports)
            scheduler_metrics = optim_group.scheduler.step(exports)
            if scheduler_metrics:
                metrics.update(scheduler_metrics)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            enabled=False, device_type=self.device.type, dtype=torch.float32
        ):
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            params_to_clip = [
                p for group in optimizer.param_groups for p in group["params"]
            ]
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        return ctx, metrics

    def reduce_mean(self, values: Dict[str, Any]) -> None:
        """Average every scalar-tensor entry of ``values`` across the group.

        Packs all scalars into one buffer so the reduction costs a single NCCL
        collective rather than one per entry. A no-op outside distributed mode,
        so single-GPU paths are unaffected.
        """
        if not self.distributed:
            return
        keys = [
            k
            for k, v in values.items()
            if isinstance(v, torch.Tensor) and v.numel() == 1
        ]
        if not keys:
            return
        packed = torch.stack([values[k].detach().reshape(()) for k in keys])
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        packed /= torch.distributed.get_world_size()
        for key, value in zip(keys, packed.unbind()):
            values[key] = value

    @staticmethod
    def jit(fn: Callable, **kwargs):
        compile_args = {
            "mode": kwargs.pop("mode", "default"),
            "fullgraph": kwargs.pop("fullgraph", False),
        }
        compile_args.update(kwargs)

        def wrapper(f):
            return torch.compile(f, **compile_args)

        if fn is None:
            return wrapper
        return wrapper(fn)

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

        return tree_map(_put_data, data)

    @staticmethod
    def detach(data: Any) -> Any:
        def _detach(x):
            if isinstance(x, torch.Tensor):
                return x.detach()
            return x

        return tree_map(_detach, data)

    @staticmethod
    def to_numpy(data: Any) -> Any:
        return tree_map(lambda x: x.detach().cpu().numpy(), data)

    @staticmethod
    def to_jax(data: Any) -> Any:
        try:
            import jax
            from torch.utils.dlpack import to_dlpack

            return tree_map(lambda x: jax.dlpack.from_dlpack(to_dlpack(x)), data)
        except ImportError:
            raise ImportError("Please install `jax` to use `to_jax` function.")

    @staticmethod
    def to_torch(data: Any) -> torch.Tensor:
        return data

    @staticmethod
    def to_warp(data: Any) -> Any:
        try:
            import warp

            return tree_map(warp.from_torch, data)
        except ImportError:
            raise ImportError("Please install `warp-lang` to use `to_warp` function.")

    @staticmethod
    def to_dlpack(data: Any):
        from torch.utils.dlpack import to_dlpack

        return tree_map(lambda x: to_dlpack(x), data)

    @staticmethod
    def to(data: Any, target: Any, non_blocking: bool = True):
        if isinstance(target, (str, torch.dtype, torch.device)):
            if isinstance(data, torch.Tensor):
                return data.to(target, non_blocking=non_blocking)
            return torch.as_tensor(data).to(target, non_blocking=non_blocking)

        if isinstance(data, (dict, list, tuple)):
            return tree_map(lambda x: TorchExecutor.to(x, target, non_blocking), data)

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
        return tree_map(torch.as_tensor, data)

    @staticmethod
    def from_jax(data: Any):
        try:
            from jax import dlpack as jdlpack
            from torch.utils.dlpack import from_dlpack

            return tree_map(lambda x: from_dlpack(jdlpack.to_dlpack(x)), data)
        except ImportError:
            raise ImportError("Please install `jax` to use `from_jax` function.")

    @staticmethod
    def from_warp(data: Any):
        try:
            import warp

            return tree_map(warp.to_torch, data)
        except ImportError:
            raise ImportError("Please install `warp-lang` to use `from_warp` function.")

    @staticmethod
    def from_dlpack(data: Any):
        from torch.utils.dlpack import from_dlpack

        return tree_map(from_dlpack, data)
