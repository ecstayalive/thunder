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
        "thunder.core.operation",
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
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=2, use_bias=False, kernel_init=nn.initializers.zeros)(x)


class SimpleRNN(nn.Module):
    features: int = 2

    @nn.compact
    def __call__(self, x, carry=None):
        batch_dims = x.shape[:-1]
        if carry is None:
            carry = jnp.zeros(batch_dims + (self.features,))
        x_proj = nn.Dense(self.features, name="ih")(x)
        h_proj = nn.Dense(self.features, name="hh", use_bias=False)(carry)
        next_carry = nn.tanh(x_proj + h_proj)
        return next_carry, next_carry


class JaxMSEObjective(op_mod.Objective):
    @dataclass
    class ModuleProtocol:
        net: Simple3DFlaxNet

    def compute(
        self, batch: data_mod.Batch, models: ModuleProtocol, params: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        pred = models.net(batch.obs, state=params)
        targets = batch.actions
        error = pred - targets
        if batch.mask is not None:
            mask_3d = jnp.expand_dims(batch.mask, axis=-1) if batch.mask.ndim == 2 else batch.mask
            error = error * mask_3d
            valid_count = jnp.sum(mask_3d) * error.shape[-1]
            loss = jnp.sum(jnp.square(error)) / jnp.clip(valid_count, min=1.0)
        else:
            loss = jnp.mean(jnp.square(error))
        return loss, {"pred_mean": jnp.mean(pred)}


class MultiNetObjective(op_mod.Objective):
    @dataclass
    class ModuleProtocol:
        net1: Simple3DFlaxNet
        net2: Simple3DFlaxNet

    def compute(
        self, batch: data_mod.Batch, models: ModuleProtocol, params: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        pred1 = models.net1(batch.obs, state=params)
        pred2 = models.net2(batch.obs, state=params)
        loss = jnp.mean((pred1 - batch.actions) ** 2) + jnp.mean((pred2 - batch.actions) ** 2)
        return loss, {}


class JaxCounterOp(op_mod.Operation):
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
    return data_mod.Batch(
        obs=jnp.array([[[1.0] * 4, [1.0] * 4, [1.0] * 4], [[2.0] * 4, [2.0] * 4, [0.0] * 4]]),
        actions=jnp.array(
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0], [0.0, 0.0]]]
        ),
        mask=jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
        extra={"val": jnp.zeros((2, 3, 1))},
    )


def test_jax_pytree_registration(jax_batch_3d):
    @jax.jit
    def process_batch(b):
        return b.obs * 2.0

    res = process_batch(jax_batch_3d)
    assert res.shape == (2, 3, 4)
    assert jnp.all(res[0, 0] == 2.0)


def test_jax_executor_optimization_flow(jax_batch_3d):
    net = Simple3DFlaxNet()
    model = data_mod.ModelPack(net=net)
    executor = exec_mod.Executor(device="gpu", donate=False)
    optim_config = {"opt": {"targets": ["net"], "class": "sgd", "lr": 1.0}}
    ctx = executor.init(model, jax_batch_3d, optim_config)
    old_w = ctx.params["net"]["Dense_0"]["kernel"]
    assert jnp.all(old_w == 0.0)
    obj = JaxMSEObjective("test")
    op = op_mod.OptimizeOp("opt", [obj])
    new_ctx, metrics = op(ctx)
    new_w = new_ctx.params["net"]["Dense_0"]["kernel"]
    assert jnp.any(new_w != 0.0)
    assert jnp.all(old_w == 0.0)
    assert jnp.any(new_w != 0.0)
    assert "grad_op/test/loss" in metrics


def test_jax_multi_net_joint_update(jax_batch_3d):
    model = data_mod.ModelPack(net1=Simple3DFlaxNet(), net2=Simple3DFlaxNet())
    executor = exec_mod.Executor(device="gpu")
    optim_config = {"joint_opt": {"targets": ["net1", "net2"], "class": "sgd", "lr": 0.1}}
    ctx = executor.init(model, jax_batch_3d, optim_config)
    obj = MultiNetObjective("joint")
    op = op_mod.OptimizeOp("joint_opt", [obj])
    new_ctx, metrics = op(ctx)
    assert jnp.any(new_ctx.params["net1"]["Dense_0"]["kernel"] != 0.0)
    assert jnp.any(new_ctx.params["net2"]["Dense_0"]["kernel"] != 0.0)
    assert "grad_op/joint/loss" in metrics


def test_jax_pipeline_scheduling(jax_batch_3d):
    model = data_mod.ModelPack(net=Simple3DFlaxNet())
    executor = exec_mod.Executor(device="gpu")
    interval_op = JaxCounterOp(interval=3)
    pipeline = [interval_op]
    algo = algo_mod.GraphAlgorithm(model, executor, pipeline)
    algo.build(jax_batch_3d, {})
    algo.step(jax_batch_3d)  # step 1: 0%3 == 0
    assert interval_op.execution_count == 1
    algo.step(jax_batch_3d)  # step 2: 1%3 != 0
    assert interval_op.execution_count == 1
    algo.step(jax_batch_3d)  # step 3: 2%3 == 0
    assert interval_op.execution_count == 1
    algo.step(jax_batch_3d)  # step 4: 3%3 == 0
    assert interval_op.execution_count == 2


def test_jax_objective_standalone(jax_batch_3d):
    model = data_mod.ModelPack(net=Simple3DFlaxNet())
    executor = exec_mod.Executor(device="gpu")
    obj = JaxMSEObjective("eval")
    algo = algo_mod.GraphAlgorithm(model, executor, [obj])
    algo.build(jax_batch_3d, {})
    metrics = algo.step(jax_batch_3d)
    assert "eval/loss" in metrics
    assert "eval/weighted_loss" in metrics
    assert jnp.all(algo.ctx.params["net"]["Dense_0"]["kernel"] == 0.0)


def test_jax_apply_gradients_logic(jax_batch_3d):
    model = data_mod.ModelPack(net=Simple3DFlaxNet())
    executor = exec_mod.Executor(device="gpu")
    ctx = executor.init(model, jax_batch_3d, {"opt": {"targets": ["net"], "class": "sgd"}})

    fake_params = {"net": jax.tree_util.tree_map(lambda x: x + 1.0, ctx.params["net"])}
    fake_opt = jax.tree_util.tree_map(lambda x: x, ctx.opt_groups["opt"].opt_state)

    new_ctx = ctx.apply_gradients("opt", fake_params, fake_opt)
    assert new_ctx is not ctx
    assert jnp.all(new_ctx.params["net"]["Dense_0"]["kernel"] == 1.0)
    assert jnp.all(new_ctx.opt_groups["opt"].params["net"]["Dense_0"]["kernel"] == 1.0)


def test_jax_initialization_error(jax_batch_3d):
    model = data_mod.ModelPack(net=Simple3DFlaxNet())
    executor = exec_mod.Executor(device="gpu")
    with pytest.raises(ValueError, match="Optimizer target 'bad_net' not found in models."):
        executor.init(model, jax_batch_3d, {"opt": {"targets": ["bad_net"]}})


def test_jax_method_proxy_call(jax_batch_3d):
    class CustomMethodNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(2)(x)

        def get_custom_value(self, x):
            return jnp.sum(x, axis=-1, keepdims=True)

    model = data_mod.ModelPack(net=CustomMethodNet())
    executor = exec_mod.Executor(device="gpu")
    ctx = executor.init(model, jax_batch_3d, {})

    res = ctx.models.net.get_custom_value(jax_batch_3d.obs, state=ctx.params["net"])
    assert res.shape == (2, 3, 1)
    assert jnp.all(res[0, 0] == jnp.sum(jax_batch_3d.obs[0, 0]))


def test_jax_gradient_clipping(jax_batch_3d):
    net = Simple3DFlaxNet()
    model = data_mod.ModelPack(net=net)
    executor = exec_mod.Executor(device="gpu")

    optim_config = {"opt": {"targets": ["net"], "class": "sgd", "lr": 1.0}}
    ctx = executor.init(model, jax_batch_3d, optim_config)
    obj = JaxMSEObjective("test")
    op_no_clip = op_mod.OptimizeOp("opt", [obj], max_grad_norm=999.0)
    _, metrics_large = op_no_clip(ctx)
    raw_grad_norm = metrics_large["grad_op/grad_norm"]
    assert raw_grad_norm > 0.1
    clip_threshold = 0.01
    op_clip = op_mod.OptimizeOp("opt", [obj], max_grad_norm=clip_threshold)
    new_ctx_small, metrics_small = op_clip(ctx)
    clipped_grad_norm = metrics_small["grad_op/grad_norm"]
    assert jnp.isclose(clipped_grad_norm, clip_threshold, atol=1e-5)
    assert clipped_grad_norm < raw_grad_norm
    p_old = ctx.params["net"]["Dense_0"]["kernel"]
    p_new = new_ctx_small.params["net"]["Dense_0"]["kernel"]
    actual_update_norm = jnp.linalg.norm(p_new - p_old)
    assert jnp.isclose(actual_update_norm, clip_threshold, atol=1e-5)
    op_safe_clip = op_mod.OptimizeOp("opt", [obj], max_grad_norm=raw_grad_norm + 10.0)
    _, metrics_safe = op_safe_clip(ctx)
    assert jnp.isclose(metrics_safe["grad_op/grad_norm"], raw_grad_norm)


def test_jax_multiple_optimizer_subsets(jax_batch_3d):
    model = data_mod.ModelPack(net1=Simple3DFlaxNet(), net2=Simple3DFlaxNet())
    executor = exec_mod.Executor(device="gpu")
    optim_config = {"joint_opt": {"targets": ["net1", "net2"], "class": "sgd", "lr": 1.0}}
    ctx = executor.init(model, jax_batch_3d, optim_config)

    class JointObj(op_mod.Objective):
        def compute(self, batch, models, params):
            l1 = jnp.mean((models.net1(batch.obs, state=params) - batch.actions) ** 2)
            l2 = jnp.mean((models.net2(batch.obs, state=params) - batch.actions) ** 2)
            return l1 + l2, {}

    clip_threshold = 0.05
    op = op_mod.OptimizeOp("joint_opt", [JointObj("joint")], max_grad_norm=clip_threshold)
    new_ctx, metrics = op(ctx)
    assert jnp.isclose(metrics["grad_op/grad_norm"], clip_threshold)

    diff1 = new_ctx.params["net1"]["Dense_0"]["kernel"] - ctx.params["net1"]["Dense_0"]["kernel"]
    diff2 = new_ctx.params["net2"]["Dense_0"]["kernel"] - ctx.params["net2"]["Dense_0"]["kernel"]
    total_delta_norm = jnp.sqrt(jnp.sum(diff1**2) + jnp.sum(diff2**2))
    assert jnp.isclose(total_delta_norm, clip_threshold, atol=1e-5)


def test_jax_callback_side_effects(jax_batch_3d):
    model = data_mod.ModelPack(net=Simple3DFlaxNet())
    executor = exec_mod.Executor(device="gpu")

    def change_batch_hook(ctx: ctx_mod.ExecutionContext):
        new_obs = ctx.batch.obs + 10.0
        new_batch = data_mod.replace(ctx.batch, obs=new_obs)
        return ctx.replace(batch=new_batch), {"modified": True}

    pipeline = [op_mod.CallbackOp(change_batch_hook, name="mod"), JaxMSEObjective("eval")]

    algo = algo_mod.GraphAlgorithm(model, executor, pipeline)
    algo.build(jax_batch_3d, {})

    metrics = algo.step(jax_batch_3d)
    assert jnp.all(algo.ctx.batch.obs >= 10.0)
    assert metrics["mod/modified"] is True


def test_jax_multiple_objectives_summation(jax_batch_3d):
    model = data_mod.ModelPack(net=Simple3DFlaxNet())
    executor = exec_mod.Executor(device="gpu")

    obj1 = JaxMSEObjective("mse1", weight=1.0)
    obj2 = JaxMSEObjective("mse2", weight=2.0)

    op = op_mod.OptimizeOp("opt", [obj1, obj2])
    algo = algo_mod.GraphAlgorithm(model, executor, [op])
    algo.build(jax_batch_3d, {"opt": {"targets": ["net"], "class": "sgd", "lr": 0.1}})

    metrics = algo.step(jax_batch_3d)

    l1 = metrics["grad_op/mse1/loss"]
    l2 = metrics["grad_op/mse2/loss"]
    total = metrics["grad_op/loss_total"]

    assert jnp.isclose(l1, l2)
    assert jnp.isclose(total, l1 * 1.0 + l2 * 2.0)


def test_jax_distributed_metadata_injection(jax_batch_3d):
    model = data_mod.ModelPack(net=Simple3DFlaxNet())
    # Simulate distributed flag
    executor = exec_mod.Executor(device="gpu")
    executor.distributed = True

    ctx = executor.init(model, jax_batch_3d, {})

    assert "mesh" in ctx.meta
    assert "data_sharding" in ctx.meta
    assert isinstance(ctx.meta["mesh"], jax.sharding.Mesh)


def test_jax_empty_metrics_handling(jax_batch_3d):
    class NoMetricOp(op_mod.Operation):
        def forward(self, ctx):
            return ctx, {}

    model = data_mod.ModelPack(net=Simple3DFlaxNet())
    executor = exec_mod.Executor(device="gpu")
    pipeline = [NoMetricOp(name="silent")]
    algo = algo_mod.GraphAlgorithm(model, executor, pipeline)
    algo.build(jax_batch_3d, {})

    metrics = algo.step(jax_batch_3d)
    assert metrics == {}


def test_jax_params_unpacking_correctness(jax_batch_3d):
    model = data_mod.ModelPack(net1=Simple3DFlaxNet(), net2=Simple3DFlaxNet())
    executor = exec_mod.Executor(device="gpu")
    ctx = executor.init(
        model,
        jax_batch_3d,
        {
            "opt1": {"targets": ["net1"], "class": "sgd"},
            "opt2": {"targets": ["net2"], "class": "sgd"},
        },
    )

    group1 = ctx.opt_groups["opt1"]
    assert "net1" in group1.params
    assert "net2" not in group1.params

    group2 = ctx.opt_groups["opt2"]
    assert "net2" in group2.params
    assert "net1" not in group2.params


def test_jax_rnn_carry_flow(jax_batch_3d):
    model = data_mod.ModelPack(rnn=SimpleRNN())
    executor = exec_mod.Executor(device="gpu")
    ctx = executor.init(model, jax_batch_3d, {})
    assert "ih" in ctx.params["rnn"]
    assert "hh" in ctx.params["rnn"]
    obs_step0 = jax_batch_3d.obs[:, 0]  # (B=2, D=4)
    out1, next_h1 = ctx.models.rnn(obs_step0, state=ctx.params["rnn"], carry=None)
    assert out1.shape == (2, 2)
    assert next_h1.shape == (2, 2)
    out2, next_h2 = ctx.models.rnn(obs_step0, state=ctx.params["rnn"], carry=next_h1)
    assert jnp.any(next_h1 != next_h2)
    assert next_h2.shape == (2, 2)
    out2_repeat, _ = ctx.models.rnn(obs_step0, state=ctx.params["rnn"], carry=next_h1)
    assert jnp.all(out2 == out2_repeat)


def test_jax_rnn_sequence_training(jax_batch_3d):
    """ """
    model = data_mod.ModelPack(rnn=SimpleRNN(features=2))
    executor = exec_mod.Executor("gpu")
    ctx = executor.init(model, jax_batch_3d, {})
    out_seq, _ = ctx.models.rnn(jax_batch_3d.obs, state=ctx.params["rnn"])
    assert out_seq.shape == (2, 3, 2)  # (Batch=2, Time=3, Features=2)


def test_jax_objective_weight_scaling(jax_batch_3d):
    model = data_mod.ModelPack(net=Simple3DFlaxNet())
    executor = exec_mod.Executor(device="gpu")

    obj_w1 = JaxMSEObjective("mse", weight=1.0)
    obj_w10 = JaxMSEObjective("mse", weight=10.0)

    algo1 = algo_mod.GraphAlgorithm(model, executor, [obj_w1])
    algo10 = algo_mod.GraphAlgorithm(model, executor, [obj_w10])

    algo1.build(jax_batch_3d, {})
    algo10.build(jax_batch_3d, {})

    m1 = algo1.step(jax_batch_3d)
    m10 = algo10.step(jax_batch_3d)

    assert jnp.isclose(m1["mse/weighted_loss"] * 10.0, m10["mse/weighted_loss"])
    assert jnp.isclose(m1["mse/loss"], m10["mse/loss"])


def test_jax_optimizer_param_group_logic(jax_batch_3d):
    model = data_mod.ModelPack(net1=Simple3DFlaxNet(), net2=Simple3DFlaxNet())
    executor = exec_mod.Executor(device="gpu")
    algo = algo_mod.GraphAlgorithm(
        model, executor, [op_mod.OptimizeOp("joint", [MultiNetObjective("m")])]
    )
    algo.build(jax_batch_3d, {"joint": {"targets": ["net1", "net2"], "class": "sgd", "lr": 1.0}})

    wa_before = algo.ctx.params["net1"]["Dense_0"]["kernel"]
    wb_before = algo.ctx.params["net2"]["Dense_0"]["kernel"]

    algo.step(jax_batch_3d)

    assert jnp.any(algo.ctx.params["net1"]["Dense_0"]["kernel"] != wa_before)
    assert jnp.any(algo.ctx.params["net2"]["Dense_0"]["kernel"] != wb_before)


def test_jax_batch_utility_methods(jax_batch_3d):
    assert jax_batch_3d.batch_size == 2

    executor = exec_mod.Executor(device="gpu")
    cpu_batch = jax_batch_3d.to(executor)
    assert isinstance(cpu_batch.obs, jax.Array)

    class MockHandler:
        def to_device(self, x):
            return x

    native_batch = jax_batch_3d.to(MockHandler())
    assert jnp.all(native_batch.obs == jax_batch_3d.obs)


def test_jax_model_pack_attribute_access():
    m1 = module_mod.ThunderModule(Simple3DFlaxNet(), "actor")
    m2 = module_mod.ThunderModule(Simple3DFlaxNet(), "critic")
    pack = data_mod.ModelPack(actor=m1, critic=m2)

    assert pack.actor is m1
    assert pack.critic is m2
    assert "actor" in pack._fields


def test_jax_apply_gradients_multiple_opts(jax_batch_3d):
    model = data_mod.ModelPack(net=Simple3DFlaxNet())
    executor = exec_mod.Executor(device="gpu")
    ctx = executor.init(
        model,
        jax_batch_3d,
        {
            "opt_a": {"targets": ["net"], "class": "sgd", "lr": 0.1},
            "opt_b": {"targets": ["net"], "class": "sgd", "lr": 0.2},
        },
    )

    updated_p = jax.tree_util.tree_map(lambda x: x + 0.5, ctx.params["net"])
    new_ctx = ctx.apply_gradients("opt_a", {"net": updated_p}, ctx.opt_groups["opt_a"].opt_state)

    assert jnp.all(new_ctx.params["net"]["Dense_0"]["kernel"] == 0.5)
    assert jnp.all(new_ctx.opt_groups["opt_a"].params["net"]["Dense_0"]["kernel"] == 0.5)
    assert jnp.all(new_ctx.opt_groups["opt_b"].params["net"]["Dense_0"]["kernel"] == 0.0)


def test_jax_executor_jit_cache_behavior(jax_batch_3d):
    model = data_mod.ModelPack(net=Simple3DFlaxNet())
    executor = exec_mod.Executor(device="gpu")
    obj = JaxMSEObjective("mse")

    ctx = executor.init(model, jax_batch_3d, {"opt": {"targets": ["net"], "class": "sgd"}})

    executor.optimize(ctx, "opt", [obj])
    cache_size_1 = len(executor._compiled_step_cache)
    assert cache_size_1 == 1

    executor.optimize(ctx, "opt", [obj])
    assert len(executor._compiled_step_cache) == 1

    obj2 = JaxMSEObjective("mse2")
    executor.optimize(ctx, "opt", [obj, obj2])
    assert len(executor._compiled_step_cache) == 2


def test_jax_complex_extra_dict_pytree(jax_batch_3d):
    extra_data = {"nested": {"a": jnp.ones(10), "b": jnp.zeros(5)}, "list": [jnp.array([1, 2, 3])]}
    batch = data_mod.Batch(obs=jnp.zeros((1, 1, 1)), extra=extra_data)

    @jax.jit
    def sum_extra(b):
        return jnp.sum(b.nested["a"]) + jnp.sum(b.list[0])

    res = sum_extra(batch)
    assert res == 10.0 + 6.0
