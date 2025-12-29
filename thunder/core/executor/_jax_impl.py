from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax

from thunder.core.context import ExecutionContext, OptimGroup

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from ..data import Batch
    from ..module import ModelPack
    from ..module._jax_impl import JaxModule
    from ..operation import Objective


class JaxExecutor:
    def __init__(
        self,
        device: Optional[str] = None,
        mixed_precision: bool = False,
        distributed: bool = False,
        donate: bool = False,
        **kwargs,
    ):
        self.devices = jax.devices(device) if device else jax.devices("cpu")
        self.mixed_precision = mixed_precision
        self.distributed = distributed
        self._compiled_step_cache = {}
        self.donate = donate

    def to_device(self, data: Any) -> Any:
        return jax.device_put(data, self.devices[0])

    def to_numpy(self, data: Any) -> Any:
        return jax.device_get(data)

    def optimize(
        self,
        ctx: ExecutionContext,
        opt: str,
        objectives: Tuple[Objective, ...],
        max_grad_norm: float = 1.0,
    ) -> Tuple[Dict[str, Any], Any, Any]:
        optim_group: OptimGroup = ctx.opt_groups[opt]
        objs_key = tuple(objectives)
        if objs_key not in self._compiled_step_cache:
            self._compiled_step_cache[objs_key] = jax.jit(
                partial(self._jit_update, objectives=objs_key, tx=optim_group.tx),
                static_argnames=["models"],
                donate_argnums=(0, 1) if self.donate else None,
            )
        jit_fn = self._compiled_step_cache[objs_key]
        metrics, new_params_subset, new_opt_state = jit_fn(
            optim_group.params,
            optim_group.opt_state,
            ctx.batch,
            ctx.models,
            ctx.params,
            max_grad_norm,
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
        """_summary_

        Args:
            params_subset (_type_): _description_
            opt_state (_type_): _description_
            batch (Batch): _description_
            models (ModelPack): _description_
            params (Dict[str, Any]): Full params of networks
            max_grad_norm (float): _description_
            objectives (Tuple[Objective, ...]): _description_
            tx (optax.GradientTransformation): _description_
        """

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
        grad_norm = optax.global_norm(grads)
        scale = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-6))
        effective_scale = jnp.where(max_grad_norm > 0, scale, 1.0)
        grads = jax.tree_util.tree_map(lambda g: g * effective_scale, grads)
        metrics["grad_norm"] = optax.global_norm(grads)
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
        meta = {}

        mesh = None
        if self.distributed:
            devices = jax.devices()
            mesh = jax.sharding.Mesh(devices, axis_names=("data",))
            data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
            batch = jax.device_put(batch, data_sharding)
            meta["mesh"] = mesh
            meta["data_sharding"] = data_sharding
        seed = int(os.getenv("THUNDER_SEED", "42"))
        rng = jax.random.PRNGKey(seed)

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

        opt_groups = {}
        for opt_key, cfg in optim_config.items():
            params_subset = {}
            cfg = cfg.copy()
            target_names = cfg.pop("targets", cfg.pop("targets"))
            if isinstance(target_names, str):
                target_names = [target_names]
            target_names = tuple(target_names)
            for t_name in target_names:
                if t_name not in params_map:
                    raise ValueError(f"Optimizer target '{t_name}' not found in models.")
                params_subset[t_name] = params_map[t_name]
            cls_name = cfg.pop("class", "adam").lower()
            lr = cfg.pop("lr", 3e-4)
            if hasattr(optax, cls_name):
                tx = getattr(optax, cls_name)(learning_rate=lr, **cfg)
            else:
                raise ValueError(f"Unknown optax optimizer class: {cls_name}")
            opt_state = tx.init(params_subset)
            opt_groups[opt_key] = OptimGroup(
                name=opt_key, targets=target_names, params=params_subset, opt_state=opt_state, tx=tx
            )
        ctx = ExecutionContext.create(executor=self, models=models, batch=batch)
        ctx = ctx.replace(params=params_map, opt_groups=opt_groups)
        ctx.update_meta(**meta)
        return ctx
