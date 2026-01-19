from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from thunder.core.context import ExecutionContext, OptimGroup

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from ..data import Batch
    from ..module import ModelPack
    from ..operation import Objective


class JaxExecutor:
    backend = "jax"

    def __init__(
        self,
        precision: str = "fp32",
        distributed: bool = False,
        donate: bool = True,
        device: Optional[str] = None,
    ):
        self.precision = precision
        self.distributed = distributed
        self.donate = donate
        self._devices = self.devices(device)
        self.compute_dtype = {"fp32": jnp.float32, "bf16": jnp.bfloat16, "fp16": jnp.float16}.get(
            precision, jnp.float32
        )
        self._jit_optimize_cache: Dict[Tuple, Any] = {}

    def init(
        self,
        models: ModelPack,
        optim_config: Dict[str, Any],
        distributed_strategy: Optional[Callable[[Tuple, Any], Any]] = None,
    ) -> ExecutionContext:
        """ """
        import flax.nnx as nnx
        import jax
        import optax

        meta = {}

        mesh = None
        if self.distributed:
            mesh = jax.sharding.Mesh(self.devices(), axis_names=("data",))
            meta["mesh"] = mesh
            meta["data_sharding"] = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("data")
            )

        def apply_sharding_to_tree(params_tree):
            strategy_fn = distributed_strategy or (lambda p, x: jax.sharding.PartitionSpec())

            def _map_fn(path, val):
                spec = strategy_fn(path, val)
                return jax.sharding.NamedSharding(mesh, spec)

            sharding_tree = jax.tree_util.tree_map_with_path(_map_fn, params_tree)
            return jax.device_put(params_tree, sharding_tree)

        if self.distributed and mesh:
            for name in models._fields:
                model = getattr(models, name)
                _, params = nnx.split(model, nnx.Param)
                sharded_params = apply_sharding_to_tree(params)
                nnx.update(model, sharded_params)

        opt_groups: Dict[str, OptimGroup] = {}
        for opt_key, cfg in optim_config.items():
            cfg = cfg.copy()
            target_names = cfg.pop("targets")
            if isinstance(target_names, str):
                target_names = [target_names]
            target_names = tuple(target_names)
            target_modules = {t: getattr(models, t) for t in target_names}
            # Create Optax optimizer
            cls_name = cfg.pop("class", "adam").lower()
            lr = cfg.pop("lr", 3e-4)
            tx = getattr(optax, cls_name)(learning_rate=lr, **cfg)
            nnx_opt = nnx.Optimizer(target_modules, tx)
            opt_groups[opt_key] = OptimGroup(
                name=opt_key, targets=target_names, optimizer=nnx_opt, scheduler=None
            )
        ctx = ExecutionContext.create(executor=self, models=models, opt_groups=opt_groups)
        ctx.update_meta(**meta)
        return ctx

    def optimize(
        self,
        ctx: ExecutionContext,
        opt: str,
        objectives: Tuple[Objective, ...],
        max_grad_norm: float = 1.0,
    ) -> Dict[str, Any]:

        group = ctx.opt_groups[opt]
        nnx_opt = group.optimizer
        objs_key = tuple(objectives)
        if objs_key not in self._jit_optimize_cache:
            self._jit_optimize_cache[objs_key] = nnx.jit(
                self._jit_optimize, static_argnames=["objectives", "compute_dtype"]
            )
        jit_fn = self._jit_optimize_cache[objs_key]
        metrics = jit_fn(
            ctx.models, nnx_opt, ctx.batch, objs_key, max_grad_norm, self.compute_dtype
        )
        return metrics

    @staticmethod
    def _jit_optimize(
        models: ModelPack,
        optimizer: nnx.Optimizer,
        batch: Batch,
        objectives: Tuple[Objective, ...],
        max_grad_norm: float,
        compute_dtype: jnp.dtype,
    ) -> Dict[str, Any]:
        full_graphdef, full_state = nnx.split(models)
        optim_names = set(optimizer.model.keys())
        trainable_state, frozen_state = nnx.split_state(
            full_state, lambda path, var: path[0] in optim_names, ...
        )

        def loss_fn(tracked_state: nnx.State):
            """ """
            local_models = nnx.merge(full_graphdef, frozen_state, tracked_state)
            if compute_dtype != jnp.float32:
                cast_state = jax.tree_util.tree_map(
                    lambda x: x.astype(compute_dtype), nnx.state(local_models)
                )
                nnx.update(local_models, cast_state)
            total_loss = 0.0
            metrics = {}
            for obj in objectives:
                l, m = obj.forward(batch, local_models)
                total_loss += l
                metrics.update(m)
            return jnp.asarray(total_loss, dtype=jnp.float32), metrics

        tracked_state = nnx.state(optimizer.model)
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(tracked_state)

        grad_norm = optax.global_norm(grads)
        scale = jnp.where(
            max_grad_norm > 0, jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-6)), 1.0
        )
        grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        optimizer.update(grads)
        # nnx.update(models, nnx.state(optimizer.model))
        metrics["grad_norm"] = optax.global_norm(grads)
        metrics["loss_total"] = loss
        return metrics

    @staticmethod
    def jit(fn: Optional[Callable] = None, **kwargs):
        """ """
        if fn is None:

            def wrapper(func):
                return jax.jit(func, **kwargs)

            return wrapper
        return jax.jit(fn, **kwargs)

    @staticmethod
    def devices(backend: Optional[str] = None):
        return jax.devices(backend)

    @staticmethod
    def default_device(device: Optional[jax.Device | str] = None):
        if isinstance(device, jax.Device):
            return device
        try:
            return jax.devices(device)[0]
        except RuntimeError:
            return jax.devices()[0]

    @staticmethod
    def to_device(data: Any, device: Optional[jax.Device | str] = None):
        return jax.device_put(data, JaxExecutor.default_device(device))

    @staticmethod
    def to_numpy(data: Any):
        return jax.tree_util.tree_map(lambda x: np.array(x), data)

    @staticmethod
    def to_jax(data: Any):
        return data

    @staticmethod
    def to_torch(data: Any):
        try:
            from torch.utils.dlpack import from_dlpack

            return jax.tree_util.tree_map(
                lambda data: from_dlpack(jax.dlpack.to_dlpack(data)), data
            )
        except ImportError:
            raise ImportError("Please install `pytorch` to use `to_torch` function.")

    @staticmethod
    def to_warp(data: Any):
        try:
            import warp

            return jax.tree_util.tree_map(warp.from_jax, data)
        except ImportError:
            raise ImportError("Please install `warp-lang` to use `to_warp` function.")

    @staticmethod
    def to_dlpack(data: Any):
        return jax.tree_util.tree_map(jax.dlpack.to_dlpack, data)

    @staticmethod
    def to(data: Any, target: Any, non_blocking: bool = False) -> Any:
        """ """
        if isinstance(target, str):
            if target == "cpu":
                return jax.device_put(data, jax.devices("cpu")[0])
            return jax.device_put(data, jax.devices("gpu")[0])
        if isinstance(data, (dict, list, tuple)):
            return jax.tree_util.tree_map(lambda x: JaxExecutor.to(x, target), data)

        if isinstance(target, type):
            name = target.__name__
            module = target.__module__
            if name == "ndarray" and "numpy" in module:
                return np.array(data)
            if name == "Tensor" and "torch" in module:
                from jax import dlpack as jdlpack
                from torch.utils.dlpack import from_dlpack

                return from_dlpack(jdlpack.to_dlpack(data))
            if "warp" in module:
                import warp

                return warp.from_jax(data)
        try:
            return data.astype(target)
        except (AttributeError, TypeError):
            pass

        raise ValueError(f"Executor.to: Unknown target '{target}'")

    @staticmethod
    def from_numpy(data: Any):
        return jax.tree_util.tree_map(jnp.array, data)

    @staticmethod
    def from_torch(data: Any):
        try:
            from torch.utils.dlpack import to_dlpack

            return jax.tree_util.tree_map(lambda x: jax.dlpack.from_dlpack(to_dlpack(x)), data)
        except ImportError:
            raise ImportError("Please install `pytorch` to use `from_torch` function.")

    @staticmethod
    def from_warp(data: Any):
        try:
            import warp

            return jax.tree_util.tree_map(warp.to_jax, data)
        except ImportError:
            raise ImportError("Please install `warp-lang` to use `from_warp` function.")

    @staticmethod
    def from_dlpack(data: Any):
        return jax.tree_util.tree_map(jax.dlpack.from_dlpack, data)


def fsdp_strategy(path, param):
    if path[-1] == "kernel" and param.size > 1024:
        return jax.sharding.PartitionSpec("data", None)
    return jax.sharding.PartitionSpec()
