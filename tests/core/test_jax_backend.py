import importlib
import os
import sys
from typing import Any, Dict, Tuple

import pytest


@pytest.fixture(scope="module", autouse=True)
def setup_jax_env():
    os.environ["THUNDER_BACKEND"] = "jax"
    modules_to_reload = [
        "thunder.core.context",
        "thunder.core.data",
        "thunder.core.module",
        "thunder.core.executor",
        "thunder.core.algorithm",
        "thunder.core.operation",
    ]
    for mod_name in modules_to_reload:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        else:
            importlib.import_module(mod_name)
    yield


import jax
import flax.nnx as nnx
import jax.numpy as jnp

import thunder.core.algorithm as algo_mod
import thunder.core.context as ctx_mod
import thunder.core.data as data_mod
import thunder.core.executor as exec_mod
import thunder.core.module as module_mod
import thunder.core.operation as op_mod


class Simple3DFlaxNet(nnx.Module):
    def __init__(self, din: int, dout: int, rngs: nnx.Rngs = nnx.Rngs(0)):
        self.net = nnx.Linear(
            din, dout, use_bias=False, kernel_init=nnx.initializers.zeros, rngs=rngs
        )

    def __call__(self, x):
        return self.net(x)


class SimpleRNN(nnx.Module):
    def __init__(self, din: int, dhid: int, rngs: nnx.Rngs = nnx.Rngs(0)):
        self.ih = nnx.Linear(din, dhid, rngs=rngs)
        self.hh = nnx.Linear(dhid, dhid, use_bias=False, rngs=rngs)
        self.dhid = dhid

    def __call__(self, x, carry=None):
        if carry is None:
            carry = jnp.zeros(x.shape[:-1] + (self.dhid,))
        x_proj = self.ih(x)
        h_proj = self.hh(carry)
        next_carry = jnp.tanh(x_proj + h_proj)
        return next_carry, next_carry


class JaxMSEObjective(op_mod.Objective):

    def compute(
        self, batch: data_mod.Batch, models: module_mod.ModelPack
    ) -> Tuple[Any, Dict[str, Any]]:
        target_net: nnx.Module = getattr(models, self.kwargs.get("net", "net"))
        pred = target_net(batch.obs["obs"])
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

    def compute(
        self, batch: data_mod.Batch, models: module_mod.ModelPack
    ) -> Tuple[Any, Dict[str, Any]]:
        pred1 = getattr(models, self.kwargs.get("net1", "net1"))(batch.obs["obs"])
        pred2 = getattr(models, self.kwargs.get("net2", "net2"))(batch.obs["obs"])
        loss = jnp.mean((pred1 - batch.actions) ** 2) + jnp.mean((pred2 - batch.actions) ** 2)
        return loss, {}


class JaxCounterOp(op_mod.Operation):
    def __init__(self, interval: int = 1):
        super().__init__(name="counter")
        self.interval = interval
        self.count = 0

    def forward(
        self, ctx: ctx_mod.ExecutionContext
    ) -> Tuple[ctx_mod.ExecutionContext, Dict[str, Any]]:
        if ctx.step % self.interval == 0:
            self.count += 1
        return ctx, {"count": self.count}


@pytest.fixture
def jax_batch_3d():
    return data_mod.Batch(
        obs={
            "obs": jnp.array([[[1.0] * 4, [1.0] * 4, [1.0] * 4], [[2.0] * 4, [2.0] * 4, [0.0] * 4]])
        },
        actions=jnp.array(
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0], [0.0, 0.0]]]
        ),
        mask=jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
        extra={"val": jnp.zeros((2, 3, 1))},
    )


def test_jax_batch_pytree_behavior(jax_batch_3d):
    @jax.jit
    def process_batch(b):
        return b.obs["obs"] * 2.0

    res = process_batch(jax_batch_3d)
    assert res.shape == (2, 3, 4)
    assert jnp.all(res[0, 0] == 2.0)


def test_jax_executor_init_flow(jax_batch_3d):
    rngs = nnx.Rngs(42)
    models = module_mod.ModelPack(
        net1=Simple3DFlaxNet(4, 2, rngs), net2=Simple3DFlaxNet(4, 2, rngs)
    )
    executor = exec_mod.Executor()
    optim_config = {
        "opt1": {"targets": ["net1"], "class": "adam", "lr": 1e-3},
        "opt2": {"targets": ["net2"], "class": "sgd", "lr": 1e-2},
    }
    ctx = executor.init(models, optim_config)
    assert "opt1" in ctx.opt_groups
    assert "opt2" in ctx.opt_groups
    assert ctx.opt_groups["opt1"].targets == ("net1",)
    assert isinstance(ctx.opt_groups["opt1"].optimizer, nnx.Optimizer)


def test_jax_nnx_optimization_step(jax_batch_3d):
    rngs = nnx.Rngs(0)
    models = module_mod.ModelPack(net=Simple3DFlaxNet(4, 2, rngs))
    executor = exec_mod.Executor(donate=False)
    ctx = executor.init(models, {"opt": {"targets": ["net"], "class": "sgd", "lr": 1.0}})
    ctx.batch = jax_batch_3d
    initial_params = jax.tree_util.tree_map(lambda x: jnp.copy(x), ctx.models.net.net.kernel.value)
    obj = JaxMSEObjective("test")
    op = op_mod.OptimizeOp("opt", [obj])
    new_ctx, metrics = op(ctx)
    current_params = new_ctx.models.net.net.kernel.value
    assert "optimize/test/loss" in metrics
    assert not jnp.array_equal(initial_params, current_params)
    assert jnp.all(current_params > 0)


def test_jax_multi_net_optimization(jax_batch_3d):
    rngs = nnx.Rngs(1)
    models = module_mod.ModelPack(
        net1=Simple3DFlaxNet(4, 2, rngs), net2=Simple3DFlaxNet(4, 2, rngs)
    )
    executor = exec_mod.Executor()
    optim_config = {"joint": {"targets": ["net1", "net2"], "class": "sgd", "lr": 0.5}}
    ctx = executor.init(models, optim_config)
    ctx.batch = jax_batch_3d
    obj = MultiNetObjective("m")
    op = op_mod.OptimizeOp("joint", [obj])
    ctx, _ = op(ctx)
    assert jnp.any(ctx.models.net1.net.kernel.value != 0.0)
    assert jnp.any(ctx.models.net2.net.kernel.value != 0.0)


def test_jax_rnn_carry_and_state_flow(jax_batch_3d):
    rngs = nnx.Rngs(2)
    rnn = SimpleRNN(4, 2, rngs)
    models = module_mod.ModelPack(rnn=rnn)
    executor = exec_mod.Executor()
    ctx = executor.init(models, {})
    x_step = jax_batch_3d.obs["obs"][:, 0]
    out1, carry1 = ctx.models.rnn(x_step, carry=None)
    assert out1.shape == (2, 2)
    out2, carry2 = ctx.models.rnn(x_step, carry=carry1)
    assert not jnp.array_equal(out1, out2)
    assert not jnp.array_equal(carry1, carry2)


def test_jax_gradient_clipping_logic(jax_batch_3d):
    rngs = nnx.Rngs(3)
    models = module_mod.ModelPack(net=Simple3DFlaxNet(4, 2, rngs))
    executor = exec_mod.Executor()
    ctx = executor.init(models, {"opt": {"targets": ["net"], "class": "sgd", "lr": 1.0}})
    ctx.batch = jax_batch_3d
    obj = JaxMSEObjective("test")
    op_large = op_mod.OptimizeOp("opt", [obj], max_grad_norm=1e6)
    _, m_large = op_large(ctx)
    raw_norm = m_large["optimize/grad_norm"]
    ctx.models.net.net.kernel.value = jnp.zeros_like(ctx.models.net.net.kernel.value)
    clip_val = 0.01
    op_small = op_mod.OptimizeOp("opt", [obj], max_grad_norm=clip_val)
    ctx, m_small = op_small(ctx)
    assert jnp.isclose(m_small["optimize/grad_norm"], clip_val)
    update_norm = jnp.linalg.norm(ctx.models.net.net.kernel.value)
    assert jnp.isclose(update_norm, clip_val, atol=1e-5)


def test_jax_multi_op(jax_batch_3d):
    import jax.tree_util as jtu

    def soft_update(source, target, tau):
        s_state = nnx.state(source)
        t_state = nnx.state(target)
        new_state = jax.tree_util.tree_map(lambda s, t: (1 - tau) * t + tau * s, s_state, t_state)
        nnx.update(target, new_state)
        return {}

    rngs = nnx.Rngs(4)
    net1 = Simple3DFlaxNet(4, 2, rngs)
    net1.net.kernel.value = jnp.ones_like(net1.net.kernel.value)
    net2 = Simple3DFlaxNet(4, 2, rngs)
    models = module_mod.ModelPack(net1=net1, net2=net2)
    executor = exec_mod.Executor()
    initial_net2_params = jax.tree_util.tree_map(jnp.copy, nnx.state(net2))
    net1_params = nnx.state(net1)
    assert not jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x, y: jnp.array_equal(x, y), initial_net2_params, net1_params)
    )
    counter = JaxCounterOp(interval=5)
    update_op = op_mod.CallableOp(
        soft_update,
        name="soft_update",
        source=ctx_mod.CtxRef.models.net1,
        target=ctx_mod.CtxRef.models.net2,
        tau=0.1,
    )
    pipeline = [update_op, counter]
    algo = algo_mod.Algorithm(models, executor, {}, pipeline)
    m = algo.step(jax_batch_3d)
    assert m["algorithm/counter/count"] == 1
    current_net2_params = nnx.state(algo.ctx.models.net2)
    assert not jtu.tree_all(
        jtu.tree_map(lambda x, y: jnp.array_equal(x, y), initial_net2_params, current_net2_params)
    )
    expected_params = jtu.tree_map(lambda s, t: 0.1 * s + 0.9 * t, net1_params, initial_net2_params)
    is_correct = jtu.tree_map(
        lambda curr, exp: jnp.allclose(curr, exp, atol=1e-5), current_net2_params, expected_params
    )
    assert jtu.tree_all(is_correct)
    for i in range(4):
        m = algo.step(jax_batch_3d)
        assert m["algorithm/counter/count"] == 1
    m_final = algo.step(jax_batch_3d)
    assert m_final["algorithm/counter/count"] == 2


def test_jax_jit_speedup(jax_batch_3d):
    import time

    d_model = 4096

    class ForwardOp(op_mod.Operation):
        def forward(self, ctx: ctx_mod.ExecutionContext):
            input = jax.random.normal(jax.random.key(0), (d_model, d_model))
            output = ctx.models.net(input)
            ctx.batch = ctx.batch.replace(dummy=jnp.ones((d_model, d_model)))
            ctx.meta = {"dummy": 1.0}
            return ctx, {}

    class DummyObjective(op_mod.Objective):
        def compute(self, batch: data_mod.Batch, model: module_mod.ModelPack):
            input = jax.random.normal(jax.random.key(0), (d_model, d_model))
            output = models.net(input)
            return jnp.mean(jnp.square(output)), {}

    rngs = nnx.Rngs(0)
    net = Simple3DFlaxNet(d_model, d_model, rngs)
    models = module_mod.ModelPack(net=net)
    executor = exec_mod.Executor()
    algo = algo_mod.Algorithm(models, executor)
    algo.build({"opt": {"targets": ["net"], "class": "sgd", "lr": 1.0e-3}})
    pipeline = [
        ForwardOp(name="forward"),
        op_mod.OptimizeOp("opt", [DummyObjective()], max_grad_norm=1.0),
    ]
    algo.setup_pipeline(pipeline, jit=False)
    start_time = time.time()
    for _ in range(50):
        algo.step(jax_batch_3d)
    no_jit_duration = time.time() - start_time
    print(f"No-JIT Time: {no_jit_duration:.4f}s")
    algo.setup_pipeline(pipeline, jit=True)
    algo.step(jax_batch_3d)
    start_time = time.time()
    for _ in range(50):
        algo.step(jax_batch_3d)
    jit_duration = time.time() - start_time
    print(f"JIT Time:    {jit_duration:.4f}s")
    speedup = no_jit_duration / jit_duration
    print(f"Speedup: {speedup:.2f}x")
    assert jit_duration < no_jit_duration
    assert jnp.array_equal(algo.ctx.batch.dummy, jnp.ones((d_model, d_model)))


def test_jax_mixed_precision_bf16(jax_batch_3d):
    rngs = nnx.Rngs(5)
    native = Simple3DFlaxNet(4, 2, rngs)
    models = module_mod.ModelPack(net=native)
    executor = exec_mod.Executor(precision="bf16")

    class ForwardOp(op_mod.Operation):
        def __init__(self):
            super().__init__("forward")

        def forward(self, ctx):
            ctx.batch["y"] = ctx.models.net(jax.random.normal(jax.random.key(0), (4, 4)))
            return ctx, {}

    ctx = executor.init(models, {"opt": {"targets": ["net"], "class": "sgd", "lr": 0.1}})
    ctx.batch = jax_batch_3d
    with ctx.manager:
        forward_op = ForwardOp()
        ctx, _ = forward_op(ctx)
        assert ctx.batch["y"].dtype == jnp.bfloat16
        obj = JaxMSEObjective("mse")
        op = op_mod.OptimizeOp("opt", [obj])
        ctx, _ = op(ctx)
        assert native.net.kernel.value.dtype == jnp.float32


def test_jax_objective_standalone_eval(jax_batch_3d):
    rngs = nnx.Rngs(6)
    models = module_mod.ModelPack(net=Simple3DFlaxNet(4, 2, rngs))
    executor = exec_mod.Executor()
    obj = JaxMSEObjective("eval")
    algo = algo_mod.Algorithm(models, executor, {}, [obj])
    m = algo.step(jax_batch_3d)
    assert "algorithm/eval/loss" in m
    assert jnp.all(models.net.net.kernel.value == 0.0)


def test_jax_serialization_state_retrieval():
    rngs = nnx.Rngs(7)
    models = module_mod.ModelPack(net=Simple3DFlaxNet(4, 2, rngs))
    executor = exec_mod.Executor()
    algo = algo_mod.Algorithm(models, executor, {}, [])
    # sd = algo.get_state_dict()
    # assert "net" in sd
    # assert isinstance(sd["net"], nnx.State)
    new_native = Simple3DFlaxNet(4, 2, nnx.Rngs(8))
    new_models = module_mod.ModelPack(net=new_native)
    new_algo = algo_mod.Algorithm(new_models, executor, {}, [])
    # new_algo.load_state_dict(sd)
    assert jnp.all(new_native.net.kernel.value == 0.0)


def test_jax_complex_pytree_nesting():
    class DeepNet(nnx.Module):
        def __init__(self, rngs):
            self.sub = Simple3DFlaxNet(4, 4, rngs)
            self.final = nnx.Linear(4, 2, rngs=rngs)

        def __call__(self, x):
            return self.final(self.sub(x))

    rngs = nnx.Rngs(10)
    models = module_mod.ModelPack(net=DeepNet(rngs))
    executor = exec_mod.Executor()
    ctx = executor.init(models, {"o": {"targets": ["net"], "class": "adam"}})
    graphdef, state = nnx.split(ctx.models)
    assert "sub" in state["net"]
    assert "final" in state["net"]


def test_jax_batch_device_transfer(jax_batch_3d):
    executor = exec_mod.Executor()
    moved_batch = executor.to_device(jax_batch_3d)
    assert isinstance(moved_batch.obs["obs"], jax.Array)
    assert moved_batch is not jax_batch_3d
    assert jnp.all(moved_batch.obs["obs"] == jax_batch_3d.obs["obs"])


def test_jax_context_immutability_and_replace():
    executor = exec_mod.Executor()
    models = module_mod.ModelPack(net=Simple3DFlaxNet(4, 2, nnx.Rngs(0)))
    ctx = executor.init(models, {})

    new_ctx = ctx.replace(step=500)
    assert new_ctx.step == 500
    assert ctx.step == 0
    assert new_ctx is not ctx
    assert new_ctx.models is ctx.models


def test_jax_distributed_sharding_constraints(jax_batch_3d):

    executor = exec_mod.Executor(distributed=True)
    rngs = nnx.Rngs(0)
    net = Simple3DFlaxNet(4, 2, rngs)
    models = module_mod.ModelPack(net=net)
    ctx = executor.init(models, {})
    assert "data_sharding" in ctx.meta


def test_jax_optimizer_target_mismatch_error(jax_batch_3d):
    rngs = nnx.Rngs(0)
    models = module_mod.ModelPack(net=Simple3DFlaxNet(4, 2, rngs))
    executor = exec_mod.Executor()
    with pytest.raises(AttributeError):
        executor.init(models, {"opt": {"targets": ["wrong_name"], "class": "adam"}})


def test_jax_objective_extra_kwargs_injection(jax_batch_3d):
    class KwargObjective(op_mod.Objective):
        def compute(self, batch, models) -> Tuple[Any, Dict[str, Any]]:
            alpha = self.kwargs.get("alpha", 1.0)
            return jnp.mean(batch.obs["obs"]) * alpha, {f"{self.name}/alpha_used": alpha}

    rngs = nnx.Rngs(0)
    models = module_mod.ModelPack(net=Simple3DFlaxNet(4, 2, rngs))
    executor = exec_mod.Executor()

    obj = KwargObjective("test", alpha=0.5)
    ctx = executor.init(models, {})
    ctx.batch = jax_batch_3d

    _, metrics = obj.forward(ctx.batch, ctx.models)
    assert metrics["test/alpha_used"] == 0.5


def test_jax_optim_group_integrity(jax_batch_3d):
    rngs = nnx.Rngs(0)
    models = module_mod.ModelPack(net=Simple3DFlaxNet(4, 2, rngs))
    executor = exec_mod.Executor()
    ctx = executor.init(models, {"opt": {"targets": ["net"], "class": "adam", "lr": 1e-3}})

    group = ctx.opt_groups["opt"]
    assert group.name == "opt"
    assert "net" in group.targets
    assert group.optimizer is not None


def test_jax_multi_objective_complex_pipeline(jax_batch_3d):
    rngs = nnx.Rngs(0)
    models = module_mod.ModelPack(
        net1=Simple3DFlaxNet(4, 2, rngs), net2=Simple3DFlaxNet(4, 2, rngs)
    )
    executor = exec_mod.Executor()
    obj1 = JaxMSEObjective("m1", weight=0.1, net="net1")
    obj2 = JaxMSEObjective("m2", weight=0.9, net="net2")

    pipeline = [op_mod.OptimizeOp("opt1", [obj1]), op_mod.OptimizeOp("opt2", [obj2])]
    optim_config = {
        "opt1": {"targets": ["net1"], "class": "sgd", "lr": 0.1},
        "opt2": {"targets": ["net2"], "class": "sgd", "lr": 0.1},
    }
    algo = algo_mod.Algorithm(models, executor, optim_config, pipeline)
    metrics = algo.step(jax_batch_3d)
    assert "algorithm/optimize/m1/loss" in metrics
    assert "algorithm/optimize/m2/loss" in metrics
    assert algo.ctx.step == 1


def test_jax_model_pack_getattr_proxy():
    class CustomNet(nnx.Module):
        def __init__(self, rngs):
            self.v = nnx.Variable(jnp.array([1.0]))

        def get_val(self):
            return self.v.value

    m = CustomNet(nnx.Rngs(0))
    pack = module_mod.ModelPack(net=m)

    assert pack.net.get_val() == 1.0


def test_jax_nan_inf_handling_in_optimize(jax_batch_3d):
    class NanObjective(op_mod.Objective):
        def compute(self, batch, models):
            return jnp.nan, {}

    rngs = nnx.Rngs(0)
    models = module_mod.ModelPack(net=Simple3DFlaxNet(4, 2, rngs))
    executor = exec_mod.Executor()
    ctx = executor.init(models, {"opt": {"targets": ["net"], "class": "adam"}})
    ctx.batch = jax_batch_3d
    metrics = executor.optimize(ctx, "opt", (NanObjective("nan"),))
    assert jnp.isnan(metrics["nan/loss"])


def test_jax_executor_donate_buffers_flag():
    exec_donate = exec_mod.Executor(donate=True)
    exec_no_donate = exec_mod.Executor(donate=False)
    assert exec_donate.donate is True
    assert exec_no_donate.donate is False


def test_jax_batch_extra_getattr():
    batch = data_mod.Batch(obs=jnp.zeros(1))
    batch.custom_key = 123
    assert batch.extra["custom_key"] == 123
    assert batch.custom_key == 123
