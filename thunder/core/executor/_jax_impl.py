from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from thunder.core.context import (
    ComposedContextManager,
    ExecutionContext,
    ExecutionContextManager,
    OptimGroup,
    OptimGroupSpec,
)

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from ..data import Batch
    from ..module import ModelPack
    from ..operation import Objective

_COMPUTE_DTYPE = contextvars.ContextVar("compute_dtype", default=jnp.float32)


@contextlib.contextmanager
def _precision_scope(dtype: jnp.dtype):
    """ """
    token = _COMPUTE_DTYPE.set(dtype)
    try:
        yield
    finally:
        _COMPUTE_DTYPE.reset(token)


def _get_compute_dtype() -> Optional[jnp.dtype]:
    return _COMPUTE_DTYPE.get()


def _cast_to_dtype(tree: Any, dtype: jnp.dtype) -> Any:
    """ """

    def _cast(x: jax.Array):
        if jnp.issubdtype(x.dtype, jnp.floating) and x.dtype != dtype:
            return x.astype(dtype)
        return x

    return jax.tree_util.tree_map(_cast, tree)


class _JAXAutoCastWrapper(nnx.Module):
    def __init__(self, module: nnx.Module):
        self.module = module

    def __call__(self, *args, **kwargs):
        target_dtype = _get_compute_dtype()
        if target_dtype is None:
            return self.module(*args, **kwargs)
        args = _cast_to_dtype(args, target_dtype)
        kwargs = _cast_to_dtype(kwargs, target_dtype)
        graphdef, state = nnx.split(self.module)
        casted_state = _cast_to_dtype(state, target_dtype)
        casted_model = nnx.merge(graphdef, casted_state)
        return casted_model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.module, name)


@dataclass
class ScaleState:

    scale: jnp.array
    growth_tracker: jnp.array

    def __init__(self, scale, growth_tracker):
        self.scale = scale
        self.growth_tracker = growth_tracker


jax.tree_util.register_pytree_node(
    ScaleState, lambda s: ((s.scale, s.growth_tracker), ()), lambda _, c: ScaleState(*c)
)


class _IdentityScaler:
    def __init__(self):
        pass

    def init(self):
        return

    def scale(self, loss: jnp.array, state) -> jnp.array:
        return loss

    def unscale(self, grads: Any, state):
        return grads

    def step(self, optimizer: nnx.Optimizer, grads: Any):
        optimizer.update(grads)

    def update(self, state, grads: Any):
        """ """
        return state


class _DynamicScaler:
    """
    JAX implementation of torch.cuda.amp.GradScaler.
    Handles 'enabled' flag internally to avoid if-else checks in the training loop.
    """

    def __init__(
        self,
        init_scale: float = 2**15,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self.init_scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval

    def init(self) -> ScaleState:
        return ScaleState(
            scale=jnp.array(self.init_scale, dtype=jnp.float32),
            growth_tracker=jnp.array(0, dtype=jnp.int32),
        )

    def scale(self, loss: jnp.array, state: ScaleState) -> jnp.array:
        return loss * state.scale

    def unscale(self, grads: Any, state: ScaleState) -> Any:
        inv_scale = 1.0 / (state.scale + 1e-6)
        return jax.tree_util.tree_map(lambda g: g * inv_scale, grads)

    def step(self, optimizer: nnx.Optimizer, grads: Any) -> None:
        is_finite = jax.tree_util.tree_reduce(
            lambda acc, x: acc & jnp.all(jnp.isfinite(x)), grads, True
        )
        graphdef, state = nnx.split(optimizer)

        def apply_update(s):
            opt = nnx.merge(graphdef, s)
            opt.update(grads)
            _, new_s = nnx.split(opt)
            return new_s

        def skip_update(s):
            return s

        new_state = jax.lax.cond(is_finite, apply_update, skip_update, state)
        nnx.update(optimizer, new_state)

    def update(self, state: ScaleState, grads: Any) -> ScaleState:
        """
        Updates the scale factor.
        Usage: scaler_state = scaler.update(scaler_state, grads)
        """
        is_finite = jax.tree_util.tree_reduce(
            lambda acc, x: acc & jnp.all(jnp.isfinite(x)), grads, True
        )

        def true_fn(s: ScaleState):
            new_tracker = s.growth_tracker + 1
            should_grow = new_tracker >= self.growth_interval
            new_scale = jnp.where(should_grow, s.scale * self.growth_factor, s.scale)
            new_tracker = jnp.where(should_grow, 0, new_tracker)
            return ScaleState(new_scale, new_tracker)

        def false_fn(s: ScaleState):
            new_scale = jnp.maximum(1.0, s.scale * self.backoff_factor)
            return ScaleState(new_scale, 0)

        return jax.lax.cond(is_finite, true_fn, false_fn, state)


jax.tree_util.register_pytree_node(
    _DynamicScaler,
    lambda s: (
        (),
        (s.init_scale, s.growth_factor, s.backoff_factor, s.growth_interval),
    ),
    lambda aux, _: _DynamicScaler(*aux),
)


class JaxExecutor:
    backend = "jax"

    def __init__(
        self,
        precision: str = "fp32",
        distributed: bool = False,
        donate: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.precision = precision
        self.distributed = distributed
        self.donate = donate
        self._devices = self.devices(device)
        self.compute_dtype = {
            "fp32": jnp.float32,
            "bf16": jnp.bfloat16,
            "fp16": jnp.float16,
        }.get(precision, jnp.float32)
        self.mixed_precision = self.compute_dtype is not jnp.float32
        self.use_scaler = self.compute_dtype is jnp.float16
        self._mesh = None

    def init(
        self,
        models: ModelPack,
        optim_config: Dict[str, OptimGroupSpec | Dict[str, Any]],
        distributed_strategy: Optional[Callable[[Tuple, Any], Any]] = None,
    ) -> ExecutionContext:
        """ """
        import flax.nnx as nnx
        import jax
        import optax

        meta = {}
        self._mesh = None
        if self.distributed:
            self._mesh = jax.sharding.Mesh(self.devices(), axis_names=("data",))
            meta["mesh"] = self._mesh
            meta["data_sharding"] = jax.sharding.NamedSharding(
                self._mesh, jax.sharding.PartitionSpec("data")
            )

        def apply_sharding_to_tree(params_tree):
            strategy_fn = distributed_strategy or (
                lambda p, x: jax.sharding.PartitionSpec()
            )

            def _map_fn(path, val):
                spec = strategy_fn(path, val)
                return jax.sharding.NamedSharding(self._mesh, spec)

            sharding_tree = jax.tree_util.tree_map_with_path(_map_fn, params_tree)
            return jax.device_put(params_tree, sharding_tree)

        if self.distributed and self._mesh:
            for name in models._fields:
                model = getattr(models, name)
                _, params = nnx.split(model, nnx.Param)
                sharded_params = apply_sharding_to_tree(params)
                nnx.update(model, sharded_params)

        wrapped_models = type(models)(
            **{
                name: (
                    _JAXAutoCastWrapper(getattr(models, name))
                    if self.mixed_precision
                    else getattr(models, name)
                )
                for name in models._fields
            }
        )
        # Initialize Optimize Group
        opt_groups: Dict[str, OptimGroup] = {}
        for opt_key, cfg in optim_config.items():
            spec = OptimGroupSpec.factory(cfg)
            target_names = spec.target_names
            target_modules = {t: getattr(wrapped_models, t) for t in target_names}
            scaler = _DynamicScaler() if self.use_scaler else _IdentityScaler()
            scaler_state = scaler.init()
            # Create Optax optimizer
            optim_cls = spec.optimizer or "adam"
            lr = spec.lr if spec.lr is not None else 3e-4
            optim_kwargs = dict(spec.kwargs)
            if isinstance(optim_cls, str):
                optim_cls = getattr(optax, optim_cls.lower())
            tx = optim_cls(learning_rate=lr, **optim_kwargs)
            nnx_opt = nnx.Optimizer(nnx.Dict(target_modules), tx)
            scheduler = self._init_scheduler(spec.scheduler)
            opt_groups[opt_key] = OptimGroup(
                name=opt_key,
                targets=target_names,
                optimizer=nnx_opt,
                scheduler=scheduler,
                scaler=scaler,
                scaler_state=scaler_state,
            )
        manager = self.init_manager()
        ctx = ExecutionContext.create(
            models=wrapped_models, executor=self, manager=manager, opt_groups=opt_groups
        )
        ctx.update_meta(**meta)
        return ctx

    @staticmethod
    def _init_scheduler(scheduler_spec: Any) -> Any:
        if scheduler_spec is None:
            return None
        if isinstance(scheduler_spec, type):
            return scheduler_spec()
        return scheduler_spec

    def init_manager(self) -> ExecutionContextManager:
        """ """
        mesh_ctx_manager = (
            self._mesh if self.distributed and self._mesh else contextlib.nullcontext()
        )
        amp_ctx_manager = (
            _precision_scope(self.compute_dtype)
            if self.mixed_precision
            else contextlib.nullcontext()
        )
        ctx_manager = ComposedContextManager(mesh_ctx_manager, amp_ctx_manager)
        if self.distributed:
            rank = jax.process_index()
            world_size = jax.process_count()
        else:
            rank = 0
            world_size = 1
        return ExecutionContextManager(
            _context_manager=ctx_manager,
            compute_dtype=self.compute_dtype,
            device=self._devices,
            distributed=self.distributed,
            rank=rank,
            world_size=world_size,
            mesh=self._mesh if self.distributed else None,
        )

    def optimize(
        self,
        ctx: ExecutionContext,
        opt: str,
        objectives: Tuple[Objective, ...],
        max_grad_norm: float = 1.0,
    ) -> Tuple[ExecutionContext, Dict[str, Any]]:

        group = ctx.opt_groups[opt]
        graphdef, state = nnx.split((ctx.models, group.optimizer))
        state, scaler_state, metrics = self._jit_optimize(
            graphdef,
            state,
            ctx.step,
            ctx.batch,
            ctx.cache,
            group.scaler,
            group.scaler_state,
            objectives,
            max_grad_norm,
        )
        nnx.update((ctx.models, group.optimizer), state)

        new_group = replace(group, scaler_state=scaler_state)
        new_opt_groups = dict(ctx.opt_groups)
        new_opt_groups[opt] = new_group
        return ctx.replace(opt_groups=new_opt_groups), metrics

    @staticmethod
    @partial(jax.jit, static_argnames=["objectives", "scaler"])
    def _jit_optimize(
        graphdef: nnx.GraphDef,
        state: nnx.State,
        step: int,
        batch: Batch,
        cache: Batch,
        scaler: _DynamicScaler | _IdentityScaler,
        scaler_state: ScaleState,
        objectives: Tuple[Objective, ...],
        max_grad_norm: float,
    ) -> Tuple[nnx.State, ScaleState, Dict[str, Any]]:
        """
        TODO: Support Scheduler
        """
        models, optimizer = nnx.merge(graphdef, state)
        optimizer: nnx.Optimizer

        def loss_fn(opt_models):
            opt_model_map = dict(opt_models.items())
            local_models = type(models)(
                **{
                    name: opt_model_map.get(name, getattr(models, name))
                    for name in models._fields
                }
            )
            local_ctx = ExecutionContext(
                step=step,
                models=local_models,
                opt_groups={},
                executor=None,
                manager=None,
                cache=cache,
                batch=batch,
            )
            total_loss = 0.0
            metrics = {}
            for obj in objectives:
                l, m = obj.evaluate(local_ctx)
                total_loss += l
                metrics.update(m)
            total_loss = jnp.asarray(total_loss, dtype=jnp.float32)
            # metrics["loss_total"] = total_loss
            scaled_loss = scaler.scale(total_loss, scaler_state)
            return scaled_loss, metrics

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(optimizer.model)
        grads = scaler.unscale(grads, scaler_state)
        grad_norm = optax.tree.norm(grads)
        scale = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-6))
        grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        scaler.step(optimizer, grads)
        new_scaler_state = scaler.update(scaler_state, grads)
        # metrics["grad_norm"] = optax.tree.norm(grads)
        return nnx.state((models, optimizer)), new_scaler_state, metrics

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
    def default_device(device: Optional[Any | str] = None):
        if isinstance(device, jax.Device):
            return device
        try:
            return jax.devices(device)[0]
        except RuntimeError:
            return jax.devices()[0]

    @staticmethod
    def to_device(data: Any, device: Optional[Any | str] = None):
        return jax.device_put(data, JaxExecutor.default_device(device))

    @staticmethod
    def detach(data: Any):
        def _detach(x):
            try:
                return jax.lax.stop_gradient(x)
            except TypeError:
                return x

        return jax.tree_util.tree_map(_detach, data)

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

            return jax.tree_util.tree_map(
                lambda x: jax.dlpack.from_dlpack(to_dlpack(x)), data
            )
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
