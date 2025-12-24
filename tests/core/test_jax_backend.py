"""
Detailed Integration tests for thunder.core using the JAX/Flax backend.
Standard Data Protocol: (Batch, Time, Feature)
"""

import importlib
import os
import sys
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
    """JAX MSE Loss"""

    def compute(
        self, batch: data_mod.Batch, model: module_mod.ThunderModule, params: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        pred = model(batch.obs, state=params)
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
    """一个简单的计数 Op，用于验证 interval 调度逻辑"""

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
        obs=jnp.array(
            [
                [[1.0] * 4, [1.0] * 4, [1.0] * 4],  # Traj 1
                [[2.0] * 4, [2.0] * 4, [0.0] * 4],  # Traj 2
            ]
        ),
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


def test_jax_executor_immutability_and_update(jax_batch_3d):
    """
    详尽测试 JAX 执行器的更新机制：
    1. 验证旧参数对象没有被原地修改 (Immutability)。
    2. 验证新参数对象包含更新后的数值。
    """
    net = Simple3DFlaxNet()
    model = module_mod.ThunderModule(net)
    executor = exec_mod.Executor(device="cpu")

    # 初始化
    optim_config = {"opt": {"target": "default", "class": "sgd", "lr": 1.0}}
    params, opt_states, meta = executor.init_state(model, jax_batch_3d, optim_config)

    ctx = ctx_mod.ExecutionContext.create(executor, model, jax_batch_3d)
    ctx.params, ctx.opt_states, ctx.meta = params, opt_states, meta

    # 记录旧参数引用
    old_kernel_ref = ctx.params["default"]["Dense_0"]["kernel"]
    assert jnp.all(old_kernel_ref == 0.0)

    # 运行优化
    obj = JaxMSEObjective("test")
    op = op_mod.OptimizeOp("default", "opt", [obj])
    new_ctx, metrics = op(ctx)

    # 验证逻辑
    new_kernel_ref = new_ctx.params["default"]["Dense_0"]["kernel"]

    # 1. 验证数值确实变了
    assert jnp.any(new_kernel_ref != 0.0)
    # 2. 验证 JAX 特性：旧参数对象依然是全 0 (没有 side-effect)
    assert jnp.all(old_kernel_ref == 0.0)
    # 3. 验证 Loss 数值 (同 Torch 测试计算出的 2.2)
    assert abs(float(metrics["grad_op/test/loss"]) - 2.2) < 1e-5


def test_jax_pipeline_complex_flow(jax_batch_3d):
    """ """
    net = Simple3DFlaxNet()
    model = module_mod.ThunderModule(net)
    executor = exec_mod.Executor(device="gpu")

    def meta_hook(ctx: ctx_mod.ExecutionContext):
        ctx.update_meta(custom_flag=True)
        return ctx, {"hook_val": 100}

    pipeline = [
        op_mod.CallbackOp(meta_hook, name="my_hook"),
        op_mod.OptimizeOp("default", "opt", [JaxMSEObjective("mse")]),
    ]

    algo = algo_mod.GraphAlgorithm(model, executor, pipeline)
    algo.build(jax_batch_3d, {"opt": {"class": "sgd", "lr": 0.1}})

    metrics = algo.step(jax_batch_3d)
    assert "my_hook/hook_val" in metrics
    assert "grad_op/mse/loss" in metrics
    assert algo.ctx.meta.get("custom_flag") is True
    assert algo.ctx.step == 1


def test_jax_batch_map_and_copy(jax_batch_3d):
    """验证 JAX 模式下 Batch.map 的正确性"""
    # 测试将整个 Batch 转换为 float16
    half_batch = jax_batch_3d.map(lambda x: x.astype(jnp.float16))

    assert half_batch.obs.dtype == jnp.float16
    assert half_batch.actions.dtype == jnp.float16
    # 验证 extra 字典内的内容也被转换
    assert half_batch.extra["val"].dtype == jnp.float16


def test_jax_multiple_optimizer_targets(jax_batch_3d):
    """
    验证 JAX 是否支持在同一个模型中针对不同 Key 进行优化。
    尽管本例 SimpleNet 只有一个层，但我们验证 init_state 产生的 key 映射逻辑。
    """
    net = Simple3DFlaxNet()
    model = module_mod.ThunderModule(net)
    executor = exec_mod.Executor(device="cpu")

    # 故意配置一个错误的 target，应该抛出 ValueError
    wrong_config = {"opt": {"target": "non_existent_part", "class": "sgd"}}
    with pytest.raises(ValueError, match="target 'non_existent_part' not found"):
        executor.init_state(model, jax_batch_3d, wrong_config)


def test_operation_interval_scheduling(jax_batch_3d):
    """
    - step 0 (0%2==0): 执行
    - step 1 (1%2!=0): 跳过
    - step 2 (2%2==0): 执行
    """
    net = Simple3DFlaxNet()
    model = module_mod.ThunderModule(net)
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


def test_jax_objective_standalone_as_op(jax_batch_3d):
    """
    验证 op_mod.Objective 作为一个普通 Op 放入 Pipeline 时（Eval 模式）的正确性。
    在 JAX 下，这需要验证 params 能从 Context 自动正确提取并传给 loss 函数。
    """
    net = Simple3DFlaxNet()
    model = module_mod.ThunderModule(net)
    executor = exec_mod.Executor(device="gpu")
    loss_obj = JaxMSEObjective(name="eval_mse")

    algo = algo_mod.GraphAlgorithm(model, executor, [loss_obj])
    algo.build(jax_batch_3d, {})

    metrics = algo.step(jax_batch_3d)

    assert "eval_mse/loss" in metrics
    assert "eval_mse/weighted_loss" in metrics
    # 验证没有梯度更新发生（参数依然是初始值 0）
    w = algo.ctx.params["default"]["Dense_0"]["kernel"]
    assert jnp.all(w == 0.0)
