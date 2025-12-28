from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax

from thunder.core.context import ExecutionContext

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from ..data import Batch, ModelPack
    from ..module._jax_impl import JaxModule
    from ..operation import Objective


class JaxExecutor:
    def __init__(
        self,
        device: Optional[Any] = None,
        mixed_precision: bool = False,
        distributed: bool = False,
        donate: bool = False,
        **kwargs,
    ):
        self.device = device or jax.devices()[0]
        self.mixed_precision = mixed_precision
        self.distributed = distributed
        self._compiled_step_cache = {}
        self.donate = donate

    def to_device(self, data: Any) -> Any:
        return jax.device_put(data, self.device)

    def to_numpy(self, data: Any) -> Any:
        return jax.device_get(data)

    def optimize(
        self,
        ctx: ExecutionContext,
        opt: str,
        objectives: List[Objective],
        max_grad_norm: float = 1.0,
    ) -> Tuple[Dict[str, Any], Any, Any]:
        target_params_subset = ctx.get_optim_subset(opt)
        opt_state = ctx.opt_states[opt]
        tx = ctx.meta.get(f"{opt}_def")
        objs_key = tuple(objectives)
        if objs_key not in self._compiled_step_cache:
            self._compiled_step_cache[objs_key] = jax.jit(
                partial(self._jit_update, objectives=objs_key, tx=tx),
                static_argnames=["models", "max_grad_norm"],
                donate_argnums=(0, 1) if self.donate else None,
            )
        jit_fn = self._compiled_step_cache[objs_key]
        metrics, new_params_subset, new_opt_state = jit_fn(
            target_params_subset, opt_state, ctx.batch, ctx.models, ctx.params, max_grad_norm
        )
        return metrics, new_params_subset, new_opt_state

    @staticmethod
    def _jit_update(
        params_subset,
        opt_state,
        batch: Batch,
        models: ModelPack,
        params: Dict[str, Any],
        max_grad_norm: float,
        objectives: Tuple[Objective, ...],
        tx: optax.GradientTransformation,
    ):
        def loss_fn(p_subset):
            all_params = {**params, **p_subset}
            total_loss = 0.0
            metrics_acc = {}
            for obj in objectives:
                l, m = obj.forward(batch, models, all_params)
                total_loss += l
                metrics_acc.update(m)
            return total_loss, metrics_acc

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_subset)
        if max_grad_norm > 0:
            grad_norm = optax.global_norm(grads)
            scale = jnp.clip(max_grad_norm / (grad_norm + 1e-6), max=1.0)
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
            metrics["grad_norm"] = grad_norm
        updates, new_opt_state = tx.update(grads, opt_state, params_subset)
        new_params_subset = optax.apply_updates(params_subset, updates)
        metrics["loss_total"] = loss
        return metrics, new_params_subset, new_opt_state

    def init(
        self, models: ModelPack, batch: Batch, optim_config: Dict[str, Any]
    ) -> ExecutionContext:
        """ """
        import jax
        import optax
        from flax.core import freeze, unfreeze

        params_map = {}
        opt_states = {}
        meta = {}

        mesh = None
        if self.distributed:
            devices = jax.devices()
            mesh = jax.sharding.Mesh(devices, axis_names=("data",))
            data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
            batch = jax.device_put(batch, data_sharding)
            meta["mesh"] = mesh
            meta["data_sharding"] = data_sharding

        rng = jax.random.PRNGKey(int(os.getenv("THUNDER_SEED", "42")))

        for name in models._fields:
            m_wrapper = getattr(models, name)
            rng, sub_rng = jax.random.split(rng)
            variables = m_wrapper.init(sub_rng, batch.obs)
            variables = unfreeze(variables)
            p = variables.pop("params", variables)
            if self.distributed and mesh:
                spec = getattr(m_wrapper, "_sharding_config", jax.sharding.PartitionSpec())
                sharding = jax.sharding.NamedSharding(mesh, spec)
                p = jax.device_put(p, sharding)
            params_map[name] = p
            if variables:
                meta[f"{name}_states"] = freeze(variables)

        for opt_key, cfg in optim_config.items():
            cfg = cfg.copy()
            all_params = {}
            target_name = cfg.pop("target")
            if isinstance(target_name, str):
                target_name = [target_name]
            for t_name in target_name:
                if t_name not in params_map:
                    raise ValueError(f"Optimizer target '{t_name}' not found in models.")
                all_params.update(t_name=params_map[t_name])
            cls_name = cfg.pop("class", "adam").lower()
            lr = cfg.pop("lr", 3e-4)
            if hasattr(optax, cls_name):
                tx = getattr(optax, cls_name)(learning_rate=lr, **cfg)
            else:
                raise ValueError(f"Unknown optax optimizer class: {cls_name}")
            opt_states[opt_key] = tx.init(all_params)
            meta[f"{opt_key}_def"] = tx
            meta[f"{opt_key}_targets"] = target_name
        ctx = ExecutionContext.create(executor=self, models=models, batch=batch)
        ctx = ctx.replace(params=params_map, opt_states=opt_states)
        ctx.update_meta(**meta)
        return ctx
