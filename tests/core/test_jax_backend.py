import importlib
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pytest


@pytest.fixture(scope="module", autouse=True)
def setup_jax_env():
    os.environ["THUNDER_BACKEND"] = "jax"

    modules_to_reload = [
        "thunder.core.data",
        "thunder.core.module",
        "thunder.core.executor",
        "thunder.core.algorithm",
        "thunder.core.context",
    ]

    for mod_name in modules_to_reload:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        else:
            importlib.import_module(mod_name)
    yield


import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

import thunder.core.algorithm as algo_mod
import thunder.core.context as ctx_mod
import thunder.core.data as data_mod
import thunder.core.executor as exec_mod
import thunder.core.module as module_mod
import thunder.core.operation as op_mod


class Simple3DFlaxNet(nn.Module):
    """ """

    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=2, use_bias=False, kernel_init=nn.initializers.zeros)(x)


class JaxMSEObjective(op_mod.Objective):
    """ """

    @dataclass
    class ModelProtocol:
        net: module_mod.ThunderModule

    def compute(
        self, batch: data_mod.Batch, model: ModelProtocol, params: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        pred = model.net(batch.obs, state=params)
        targets = batch.actions
        error = pred - targets  # Shape: (B, T, 2)
        if batch.mask is not None:
            # JAX: (B, T) -> (B, T, 1)
            mask_3d = jnp.expand_dims(batch.mask, axis=-1) if batch.mask.ndim == 2 else batch.mask
            error = error * mask_3d
            valid_count = jnp.sum(mask_3d) * error.shape[-1]
            loss = jnp.sum(jnp.square(error)) / jnp.clip(valid_count, min=1.0)
        else:
            loss = jnp.mean(jnp.square(error))

        return loss, {"pred_mean": jnp.mean(pred)}


class JaxCounterOp(op_mod.Operation):
    """ """

    def __init__(self, interval: int = 1):
        super().__init__(name="counter", interval=interval)
        self.execution_count = 0

    def forward(
        self, ctx: ctx_mod.ExecutionContext
    ) -> Tuple[ctx_mod.ExecutionContext, Dict[str, Any]]:
        self.execution_count += 1
        return ctx, {"count": self.execution_count}


@pytest.fixture
def jax_batch_3d():
    """"""
    return data_mod.Batch(
        obs=jnp.array([[[1.0] * 4, [1.0] * 4, [1.0] * 4], [[2.0] * 4, [2.0] * 4, [0.0] * 4]]),
        actions=jnp.array(
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0], [0.0, 0.0]]]
        ),
        mask=jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
        extra={"val": jnp.zeros((2, 3, 1))},
    )


def test_jax_pytree_registration(jax_batch_3d):
    """ """

    @jax.jit
    def process_batch(b):
        return b.obs * 2.0, b.extra["val"] + 1.0

    new_obs, new_val = process_batch(jax_batch_3d)
    assert new_obs.shape == (2, 3, 4)
    assert jnp.all(new_obs[0, 0] == 2.0)
    assert jnp.all(new_val == 1.0)


def test_jax_batch_map_and_copy(jax_batch_3d):
    """"""
    half_batch = jax_batch_3d.map(lambda x: x.astype(jnp.float16))
    assert half_batch.obs.dtype == jnp.float16
    assert half_batch.actions.dtype == jnp.float16
    assert half_batch.extra["val"].dtype == jnp.float16


def test_operation_interval_scheduling(jax_batch_3d):
    """ """
    net = Simple3DFlaxNet()
    model = data_mod.ModelPack(net=net)
    executor = exec_mod.Executor(device="gpu")
    interval_op = JaxCounterOp(interval=2)
    normal_op = JaxCounterOp(interval=1)
    pipeline = [interval_op, normal_op]
    algo = algo_mod.GraphAlgorithm(model, executor, pipeline)
    algo.build(jax_batch_3d, {})
    metrics0 = algo.step(jax_batch_3d)
    assert interval_op.execution_count == 1
    assert normal_op.execution_count == 1
    assert "counter/count" in metrics0
    metrics1 = algo.step(jax_batch_3d)
    assert interval_op.execution_count == 1
    assert normal_op.execution_count == 2
    algo.step(jax_batch_3d)
    assert interval_op.execution_count == 2
    assert normal_op.execution_count == 3
    assert algo.ctx.step == 3


def test_jax_executor_immutability_and_update(jax_batch_3d):
    """ """
    net = Simple3DFlaxNet()
    model = data_mod.ModelPack(net=net)
    executor = exec_mod.Executor(device="gpu", donate=False)
    optim_config = {"opt": {"target": "net", "class": "sgd", "lr": 1.0}}
    ctx = executor.init(model, jax_batch_3d, optim_config)
    old_kernel_ref = ctx.params["net"]["Dense_0"]["kernel"]
    assert jnp.all(old_kernel_ref == 0.0)
    obj = JaxMSEObjective("test")
    op = op_mod.OptimizeOp("net", "opt", [obj])
    new_ctx, metrics = op(ctx)
    new_kernel_ref = new_ctx.params["net"]["Dense_0"]["kernel"]
    assert jnp.any(new_kernel_ref != 0.0)
    assert jnp.all(old_kernel_ref == 0.0)
    assert abs(float(metrics["grad_op/test/loss"]) - 2.2) < 1e-5


def test_jax_pipeline_complex_flow(jax_batch_3d):
    """ """
    net = Simple3DFlaxNet()
    model = data_mod.ModelPack(net=net)
    executor = exec_mod.Executor(device="gpu")

    def meta_hook(ctx: ctx_mod.ExecutionContext):
        ctx.update_meta(custom_flag=True)
        return ctx, {"hook_val": 100}

    pipeline = [
        op_mod.CallbackOp(meta_hook, name="my_hook"),
        op_mod.OptimizeOp("net", "opt", [JaxMSEObjective("mse")]),
    ]

    algo = algo_mod.GraphAlgorithm(model, executor, pipeline)
    algo.build(jax_batch_3d, {"opt": {"target": "net", "class": "sgd", "lr": 0.1}})
    metrics = algo.step(jax_batch_3d)
    assert "my_hook/hook_val" in metrics
    assert "grad_op/mse/loss" in metrics
    assert algo.ctx.meta.get("custom_flag") is True
    assert algo.ctx.step == 1


def test_jax_multiple_optimizer_targets(jax_batch_3d):
    """ """
    net = Simple3DFlaxNet()
    model = data_mod.ModelPack(net=net)
    executor = exec_mod.Executor(device="gpu")
    wrong_config = {"opt": {"target": "non_existent_part", "class": "sgd"}}
    with pytest.raises(ValueError, match="Optimizer target 'non_existent_part' not found."):
        executor.init(model, jax_batch_3d, wrong_config)


def test_jax_objective_standalone_as_op(jax_batch_3d):
    """ """
    net = Simple3DFlaxNet()
    model = data_mod.ModelPack(net=net)
    executor = exec_mod.Executor(device="gpu")
    loss_obj = JaxMSEObjective(name="eval_mse")
    algo = algo_mod.GraphAlgorithm(model, executor, [loss_obj])
    algo.build(jax_batch_3d, {})
    metrics = algo.step(jax_batch_3d)
    assert "eval_mse/loss" in metrics
    assert "eval_mse/weighted_loss" in metrics
    w = algo.ctx.params["net"]["Dense_0"]["kernel"]
    assert jnp.all(w == 0.0)
    assert jnp.all(w == 0.0)
