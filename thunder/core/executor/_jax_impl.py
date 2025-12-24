from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, List

import jax
import jax.numpy as jnp
import optax

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from ..data import Batch
    from ..module._jax_impl import JaxModule
    from ..operation import Objective


class JaxExecutor:
    def __init__(self, device: str = None, mixed_precision: bool = False):
        self.mixed_precision = mixed_precision

    def to_device(self, data: Any) -> Any:
        return jax.device_put(data)

    def to_numpy(self, data: Any) -> Any:
        return jax.device_get(data)

    def optimize(
        self,
        ctx: ExecutionContext,
        target: str,
        opt: str,
        objectives: list,
        max_grad_norm: float = 1.0,
    ) -> tuple[dict, Any, Any]:
        params = ctx.params[target]
        opt_state = ctx.opt_states[opt]
        optimizer_def = ctx.meta.get(f"{opt}_def")
        if optimizer_def is None:
            raise RuntimeError(f"Optimizer def for '{opt}' not found in ctx.meta")
        objs_tuple = tuple(objectives)
        metrics, new_params, new_opt_state = self._jit_update(
            params, opt_state, ctx.batch, ctx.model, objs_tuple, optimizer_def, max_grad_norm
        )
        return metrics, new_params, new_opt_state

    @staticmethod
    @partial(jax.jit, static_argnames=["model", "objectives", "tx", "max_grad_norm"])
    def _jit_update(
        params, opt_state, batch: Batch, model, objectives: List[Objective], tx, max_grad_norm
    ):
        def loss_fn(p):
            loss_sum = 0.0
            metrics_acc = {}
            for obj in objectives:
                l, m = obj.forward(batch, model, p)
                loss_sum += l
                metrics_acc.update(m)
            return loss_sum, metrics_acc

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        if max_grad_norm > 0:
            updates = jax.tree.map(lambda g: jnp.clip(g, -max_grad_norm, max_grad_norm), updates)
        new_params = optax.apply_updates(params, updates)
        metrics["loss_total"] = loss
        return metrics, new_params, new_opt_state

    def init_state(self, model: JaxModule, batch: Batch, optim_config: dict):
        import jax
        import optax

        rng = jax.random.PRNGKey(0)
        variables = model.init(rng, batch.obs)
        root_params = variables["params"] if "params" in variables else variables
        params_map = {}
        params_map["default"] = root_params
        if isinstance(root_params, (dict, jax.tree_util.DictKey)):
            for k, v in root_params.items():
                if k not in params_map:
                    params_map[k] = v
        opt_states = {}
        meta = {}
        for opt_name, cfg in optim_config.items():
            cfg = cfg.copy()
            target_key = cfg.pop("target", "default")
            target_params = params_map.get(target_key)
            if target_params is None:
                raise ValueError(
                    f"Optimizer {opt_name} target '{target_key}' not found in params keys: {list(params_map.keys())}"
                )
            cls_name = cfg.pop("class", "adam").lower()
            lr = cfg.pop("lr", 3e-4)
            if hasattr(optax, cls_name):
                tx = getattr(optax, cls_name)(learning_rate=lr, **cfg)
            else:
                raise ValueError(f"Unknown optax optimizer: {cls_name}")
            opt_states[opt_name] = tx.init(target_params)
            meta[f"{opt_name}_def"] = tx

        return params_map, opt_states, meta
