import copy
import importlib
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module", autouse=True)
def setup_torch_env():
    os.environ["THUNDER_BACKEND"] = "torch"
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


import torch
import torch.nn as nn
import torch.utils._pytree as pytree

import thunder.core.algorithm as algo_mod
import thunder.core.context as ctx_mod
import thunder.core.data as data_mod
import thunder.core.executor as exec_mod
import thunder.core.module as module_mod
import thunder.core.operation as op_mod


class Simple3DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 2, bias=False)
        nn.init.constant_(self.net.weight, 0.0)

    def forward(self, x):
        return self.net(x)


class SimpleRNN(nn.Module):
    def __init__(self, features=2):
        super().__init__()
        self.features = features
        self.ih = nn.Linear(4, features)
        self.hh = nn.Linear(features, features, bias=False)

    def forward(self, x: torch.Tensor, carry=None):
        if carry is None:
            carry = torch.zeros(x.shape[:-1] + (self.features,), device=x.device)
        x_proj = self.ih(x)
        h_proj = self.hh(carry)
        next_carry = torch.tanh(x_proj + h_proj)
        return next_carry, next_carry


class MSEObjective(op_mod.Objective):
    def compute(self, batch: data_mod.Batch, models: Any) -> Tuple[Any, Dict[str, Any]]:
        target_net = getattr(models, self.kwargs.get("net", "net"))
        pred = target_net(batch.obs["obs"])
        targets = batch.actions
        error = pred - targets
        if batch.mask is not None:
            mask_3d = batch.mask.unsqueeze(-1) if batch.mask.dim() == 2 else batch.mask
            error = error * mask_3d
            valid_count = mask_3d.sum() * error.shape[-1]
            loss = (error**2).sum() / valid_count.clamp(min=1.0)
        else:
            loss = (error**2).mean()
        return loss, {"pred_mean": pred.mean()}


class MultiNetObjective(op_mod.Objective):
    def compute(self, batch: data_mod.Batch, models: Any) -> Tuple[Any, Dict[str, Any]]:
        pred1 = models.net1(batch.obs["obs"])
        pred2 = models.net2(batch.obs["obs"])
        loss = torch.mean((pred1 - batch.actions) ** 2) + torch.mean((pred2 - batch.actions) ** 2)
        return loss, {}


class TorchCounterOp(op_mod.Operation):

    def __init__(self, interval: int):
        super().__init__(name="counter")
        self.count = 0
        self.interval = interval

    def forward(
        self, ctx: ctx_mod.ExecutionContext
    ) -> Tuple[ctx_mod.ExecutionContext, Dict[str, Any]]:
        if ctx.step % self.interval == 0:
            self.count += 1
        return ctx, {"count": self.count}


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tensor_batch_3d(device):
    return data_mod.Batch(
        obs={
            "obs": torch.tensor(
                [[[1.0] * 4, [1.0] * 4, [1.0] * 4], [[2.0] * 4, [2.0] * 4, [0.0] * 4]],
                device=device,
            )
        },
        actions=torch.tensor(
            [
                [[1.0, 1.0] * 1, [1.0, 1.0] * 1, [1.0, 1.0] * 1],
                [[2.0, 2.0] * 1, [2.0, 2.0] * 1, [0.0, 0.0] * 1],
            ],
            device=device,
        ),
        mask=torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], device=device),
        extra={"val": torch.zeros((2, 3, 1), device=device)},
    )


def test_torch_executor_optimization_flow(device, tensor_batch_3d):
    native_net = Simple3DNet().to(device)
    model = module_mod.ModelPack(net=native_net)
    executor = exec_mod.Executor(compile=False)
    optim_config = {"opt": {"targets": ["net"], "class": "SGD", "lr": 1.0}}
    ctx = executor.init(model, optim_config)
    ctx = ctx.replace(batch=tensor_batch_3d)
    old_w = native_net.net.weight.clone()
    obj = MSEObjective("test")
    op = op_mod.OptimizeOp("opt", [obj])
    new_ctx, metrics = op(ctx)

    assert torch.any(native_net.net.weight != old_w)
    assert "optimize/test/loss" in metrics


import copy


def test_torch_multi_net_joint_update(device, tensor_batch_3d):
    net1 = Simple3DNet().to(device)
    net2 = Simple3DNet().to(device)
    model = module_mod.ModelPack(net1=net1, net2=net2)
    executor = exec_mod.Executor(compile=False)
    optim_config = {"joint_opt": {"targets": ["net1", "net2"], "class": "SGD", "lr": 0.1}}
    ctx = executor.init(model, optim_config)
    ctx = ctx.replace(batch=tensor_batch_3d)
    w1_old = net1.net.weight.clone()
    w2_old = net2.net.weight.clone()
    obj = MultiNetObjective("joint")
    op = op_mod.OptimizeOp("joint_opt", [obj])
    ctx, metrics = op(ctx)
    assert not torch.equal(ctx.models.net1.net.weight.clone(), w1_old)
    assert not torch.equal(ctx.models.net2.net.weight.clone(), w2_old)


def test_torch_pipeline_scheduling(device, tensor_batch_3d):
    model = module_mod.ModelPack(net=Simple3DNet().to(device))
    executor = exec_mod.Executor()
    interval_op = TorchCounterOp(interval=3)
    algo = algo_mod.Algorithm(model, executor, {}, [interval_op])
    algo.step(tensor_batch_3d)
    assert interval_op.count == 1
    algo.step(tensor_batch_3d)
    assert interval_op.count == 1
    algo.step(tensor_batch_3d)
    assert interval_op.count == 1
    algo.step(tensor_batch_3d)
    assert interval_op.count == 2


def test_torch_objective_standalone(device, tensor_batch_3d):
    native_net = Simple3DNet().to(device)
    model = module_mod.ModelPack(net=native_net)
    executor = exec_mod.Executor()
    obj = MSEObjective("eval")
    algo = algo_mod.Algorithm(model, executor, {}, [obj])

    metrics = algo.step(tensor_batch_3d)
    assert "eval/loss" in metrics
    assert torch.all(native_net.net.weight == 0.0)


def test_torch_rnn_carry_flow(device, tensor_batch_3d):
    rnn_net = SimpleRNN().to(device)
    model = module_mod.ModelPack(rnn=rnn_net)
    executor = exec_mod.Executor()
    ctx = executor.init(model, {})

    obs_step = tensor_batch_3d.obs["obs"][:, 0]
    out1, next_h1 = ctx.models.rnn(obs_step, carry=None)
    assert out1.shape == (2, 2)

    out2, next_h2 = ctx.models.rnn(obs_step, carry=next_h1)
    assert not torch.equal(next_h1, next_h2)


def test_torch_gradient_clipping(device, tensor_batch_3d):
    native_net = Simple3DNet().to(device)
    model = module_mod.ModelPack(net=native_net)
    executor = exec_mod.Executor(compile=False)
    optim_config = {"opt": {"targets": ["net"], "class": "SGD", "lr": 1.0}}
    ctx = executor.init(model, optim_config)
    ctx = ctx.replace(batch=tensor_batch_3d)
    obj = MSEObjective("test")
    op_no_clip = op_mod.OptimizeOp("opt", [obj], max_grad_norm=999.0)
    _, metrics_large = op_no_clip(ctx)
    raw_norm = metrics_large["optimize/total_loss"]
    native_net.net.weight.grad.zero_()
    clip_threshold = 0.01
    op_clip = op_mod.OptimizeOp("opt", [obj], max_grad_norm=clip_threshold)
    _, metrics_small = op_clip(ctx)
    weight_norm = torch.linalg.norm(native_net.net.weight.grad)
    assert weight_norm.item() <= clip_threshold + 1e-5


def test_torch_callback_side_effects(device, tensor_batch_3d):
    model = module_mod.ModelPack(net=Simple3DNet().to(device))
    executor = exec_mod.Executor()
    algo = algo_mod.Algorithm(model, executor)

    def modify_ctx_hook(ctx: ctx_mod.ExecutionContext):
        return {"hook": True}

    algo.setup_pipeline(
        [op_mod.CallableOp(modify_ctx_hook, ctx=ctx_mod.CtxRef), MSEObjective("mse")]
    )
    algo.build({})
    m = algo.step(tensor_batch_3d)
    assert m["callable_op/hook"] is True


def test_torch_multiple_objectives_summation(device, tensor_batch_3d):
    model = module_mod.ModelPack(net=Simple3DNet().to(device))
    executor = exec_mod.Executor()
    obj1 = MSEObjective("m1", weight=1.0)
    obj2 = MSEObjective("m2", weight=2.0)

    op = op_mod.OptimizeOp("opt", [obj1, obj2])
    algo = algo_mod.Algorithm(model, executor, {"opt": {"targets": ["net"], "class": "SGD"}}, [op])

    metrics = algo.step(tensor_batch_3d)
    l1 = metrics["optimize/m1/loss"]
    l2 = metrics["optimize/m2/loss"]
    total = metrics["optimize/total_loss"]
    assert torch.isclose(total, l1 * 1.0 + l2 * 2.0)


def test_torch_jit_speedup(tensor_batch_3d):
    import time

    import torch

    device = exec_mod.Executor.default_device()
    d_model = 4096

    class ForwardOp(op_mod.Operation):
        def forward(self, ctx: ctx_mod.ExecutionContext):
            _ = ctx.models.net(ctx.batch["input"])
            return ctx, {}

    class DummyObjective(op_mod.Objective):
        def compute(self, batch: data_mod.Batch, models: module_mod.ModelPack):
            error = models.net(batch["input"])
            return torch.mean(torch.square(error)), {}

    class SimpleTorchNet(torch.nn.Module):
        def __init__(self, din, dout):
            super().__init__()
            self.net = torch.nn.Linear(din, dout)

        def forward(self, x):
            return self.net(x)

    net = SimpleTorchNet(d_model, d_model)
    models = module_mod.ModelPack(net=net)
    executor = exec_mod.Executor()
    algo = algo_mod.Algorithm(models, executor)
    algo.build({"opt": {"targets": "net", "class": "SGD", "lr": 1e-2}})
    pipeline = [ForwardOp(name="forward"), op_mod.OptimizeOp("opt", objectives=[DummyObjective()])]
    algo.setup_pipeline(pipeline=pipeline, jit=False)
    # tensor_batch_3d["input"] = torch.randn(d_model, d_model, device=device)
    # report = torch._dynamo.explain(algo.step)(tensor_batch_3d)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(50):
        tensor_batch_3d["input"] = torch.randn(d_model, d_model, device=device)
        algo.step(tensor_batch_3d)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    no_jit_duration = time.time() - start_time
    print(f"No-JIT Time: {no_jit_duration:.4f}s")
    algo.setup_pipeline(pipeline, jit=True)
    tensor_batch_3d["input"] = torch.randn(d_model, d_model, device=device)
    algo.step(tensor_batch_3d)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(50):
        tensor_batch_3d["input"] = torch.randn(d_model, d_model, device=device)
        algo.step(tensor_batch_3d)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    jit_duration = time.time() - start_time
    print(f"JIT Time:    {jit_duration:.4f}s")
    speedup = no_jit_duration / jit_duration
    print(f"Speedup: {speedup:.2f}x")
    assert jit_duration < no_jit_duration


def test_torch_initialization_error(device, tensor_batch_3d):
    model = module_mod.ModelPack(net=Simple3DNet().to(device))
    executor = exec_mod.Executor()
    with pytest.raises(ValueError, match="Optimizer target 'bad' not found in models."):
        executor.init(model, {"opt": {"targets": ["bad"]}})


def test_torch_mixed_precision(device, tensor_batch_3d):
    native_net = Simple3DNet().to(device)
    nn.init.constant_(native_net.net.weight, 0.0)
    model = module_mod.ModelPack(net=native_net)
    executor = exec_mod.Executor(mixed_precision=True, compile=False)
    optim_config = {"opt": {"targets": ["net"], "class": "SGD", "lr": 0.1}}
    ctx = executor.init(model, optim_config)
    ctx = ctx.replace(batch=tensor_batch_3d)
    obj = MSEObjective("amp_test")
    op = op_mod.OptimizeOp("opt", [obj])
    w_init = native_net.net.weight.clone()
    success = False
    max_attempts = 10
    for i in range(max_attempts):
        ctx, metrics = op(ctx)
        if not torch.equal(native_net.net.weight, w_init):
            success = True
            print(f"\n[AMP Success] Updated at iteration {i+1}")
            break
    assert (
        success
    ), f"Weights failed to update after {max_attempts} iterations. Scaler may be stuck."
    assert "optimize/amp_test/loss" in metrics
    assert native_net.net.weight.dtype == torch.float32
    assert native_net.net.weight.grad is not None


def test_torch_algorithm_serialization(device, tensor_batch_3d):
    model = module_mod.ModelPack(net=Simple3DNet().to(device))
    executor = exec_mod.Executor(compile=False)
    op = op_mod.OptimizeOp("opt", [MSEObjective("mse")])
    algo = algo_mod.Algorithm(
        model, executor, {"opt": {"targets": ["net"], "class": "SGD", "lr": 0.1}}, [op]
    )
    algo.step(tensor_batch_3d)
    state_dict = algo.models.net.state_dict()
    assert hasattr(algo.models, "net")
    new_model = module_mod.ModelPack(net=Simple3DNet().to(device))
    new_model.net.load_state_dict(state_dict)
    new_algo = algo_mod.Algorithm(
        new_model, executor, {"opt": {"targets": ["net"], "class": "SGD"}}, [op]
    )
    # new_algo.load_state_dict(state_dict)
    for p1, p2 in zip(algo.ctx.models.net.parameters(), new_algo.ctx.models.net.parameters()):
        assert torch.equal(p1, p2)


def test_torch_advanced_optimizer_params(device, tensor_batch_3d):
    model = module_mod.ModelPack(net1=Simple3DNet().to(device), net2=Simple3DNet().to(device))
    executor = exec_mod.Executor(compile=False)
    optim_config = {
        "opt": {"targets": ["net1", "net2"], "class": "Adam", "lr": 1e-3, "weight_decay": 1e-4}
    }
    ctx = executor.init(model, optim_config)
    optimizer = ctx.opt_groups["opt"].optimizer

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4


def test_torch_executor_to_numpy(device):
    executor = exec_mod.Executor()
    tensor = torch.randn(2, 3, device=device)
    array = executor.to_numpy(tensor)

    import numpy as np

    assert isinstance(array, np.ndarray)
    assert array.shape == (2, 3)


def test_torch_distributed_initialization_mock(device, tensor_batch_3d):
    native_net = Simple3DNet().to(device)
    model = module_mod.ModelPack(net=native_net)
    executor = exec_mod.Executor(distributed=True)
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_world_size", return_value=1),
        patch(
            "torch.nn.parallel.DistributedDataParallel", side_effect=lambda x, **kwargs: x
        ) as mock_ddp,
    ):
        ctx = executor.init(model, {"opt": {"targets": ["net"], "class": "SGD"}})
        ctx = ctx.replace(batch=tensor_batch_3d)
        # Assertions
        assert mock_ddp.called
        called_module = mock_ddp.call_args[0][0]
        assert called_module is ctx.models.net
        if "cuda" in str(device):
            # Check if device_ids were passed correctly
            assert mock_ddp.call_args[1]["device_ids"] == [torch.device(device).index]
        assert hasattr(ctx.models, "net")
        assert ctx.opt_groups["opt"].targets == ("net",)
        # Verify manager state
        assert ctx.manager.rank == 0
        assert ctx.manager.world_size == 1
        assert ctx.manager.distributed is True


def test_torch_empty_step_scheduling(device, tensor_batch_3d):
    model = module_mod.ModelPack(net=Simple3DNet().to(device))
    executor = exec_mod.Executor()
    op = TorchCounterOp(interval=10)
    algo = algo_mod.Algorithm(model, executor, {}, pipeline=[op])

    for i in range(10):
        metrics = algo.step(tensor_batch_3d)
        assert op.count == 1

    metrics = algo.step(tensor_batch_3d)
    assert "counter/count" in metrics
    assert op.count == 2


def test_torch_batch_device_transfer(device):
    cpu_batch = data_mod.Batch(obs=torch.randn(2, 3, 4), extra={"meta": torch.randn(2, 3, 1)})
    executor = exec_mod.Executor()

    def _to(leaf):
        if leaf is None:
            return None
        return leaf.to(device)

    gpu_batch = pytree.tree_map(_to, cpu_batch)

    assert gpu_batch.obs.device.type == torch.device(device).type
    assert gpu_batch.extra["meta"].device.type == torch.device(device).type
    assert gpu_batch is not cpu_batch


def test_torch_model_pack_getattr_proxy(device, tensor_batch_3d):
    class MethodNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x

        def custom_call(self, x, value=1.0):
            return x + value + self.p

    native_net = MethodNet().to(device)
    model = module_mod.ModelPack(net=native_net)
    executor = exec_mod.Executor()
    ctx = executor.init(model, {})
    ctx = ctx.replace(batch=tensor_batch_3d)
    res = ctx.models.net.custom_call(tensor_batch_3d.obs["obs"], value=2.0)
    assert torch.allclose(res, tensor_batch_3d.obs["obs"] + 2.0)


def test_torch_joint_gradient_clipping(device, tensor_batch_3d):
    net1 = Simple3DNet().to(device)
    net2 = Simple3DNet().to(device)
    model = module_mod.ModelPack(net1=net1, net2=net2)
    executor = exec_mod.Executor(compile=False)

    optim_config = {"joint": {"targets": ["net1", "net2"], "class": "SGD", "lr": 1.0}}
    ctx = executor.init(model, optim_config)
    ctx = ctx.replace(batch=tensor_batch_3d)

    class HighLossObj(op_mod.Objective):
        def compute(self, batch, models):
            return (
                models.net1(batch.obs["obs"]).sum() + models.net2(batch.obs["obs"]).sum()
            ) * 100, {}

    clip_val = 0.01
    op = op_mod.OptimizeOp("joint", [HighLossObj("high")], max_grad_norm=clip_val)
    op.forward(ctx)

    params = [p for g in ctx.opt_groups["joint"].optimizer.param_groups for p in g["params"]]
    total_norm = torch.nn.utils.clip_grad_norm_(params, clip_val)
    # total_norm here is the norm after our op already clipped it
    assert total_norm <= clip_val + 1e-4


def test_torch_context_meta_update(device, tensor_batch_3d):
    model = module_mod.ModelPack(net=Simple3DNet().to(device))
    executor = exec_mod.Executor()
    ctx = executor.init(model, {})

    ctx.update_meta(info="test_info", val=123)
    assert ctx.meta["info"] == "test_info"
    assert ctx.meta["val"] == 123


def test_torch_objective_standalone_eval(device, tensor_batch_3d):
    model = module_mod.ModelPack(net=Simple3DNet().to(device))
    executor = exec_mod.Executor()
    obj = MSEObjective("eval_only")
    algo = algo_mod.Algorithm(model, executor, {}, [obj])
    metrics = algo.step(tensor_batch_3d)
    assert "eval_only/loss" in metrics
    assert "eval_only/weighted_loss" in metrics
    # Weight is 1.0 by default
    assert metrics["eval_only/loss"] == metrics["eval_only/weighted_loss"]


def test_torch_algorithm_params_access(device, tensor_batch_3d):
    model = module_mod.ModelPack(net=Simple3DNet().to(device))
    executor = exec_mod.Executor()
    algo = algo_mod.Algorithm(model, executor, {}, [])
    assert hasattr(algo.ctx.models, "net")
    assert isinstance(algo.ctx.models.net, nn.Module)
    assert algo.ctx.models.net is algo.ctx.models.net


def test_torch_batch_repr_no_crash(tensor_batch_3d):
    s = repr(tensor_batch_3d)
    assert "Batch" in s
    assert "obs" in s


def test_torch_optimizer_zero_grad_behavior(device, tensor_batch_3d):
    net = Simple3DNet().to(device)
    model = module_mod.ModelPack(net=net)
    executor = exec_mod.Executor(compile=False)
    ctx = executor.init(model, {"opt": {"targets": ["net"], "class": "SGD"}})
    ctx = ctx.replace(batch=tensor_batch_3d)
    # Manually set some dummy grad
    for p in net.parameters():
        p.grad = torch.ones_like(p.data)

    obj = MSEObjective("test")
    executor.optimize(ctx, "opt", [obj])

    # After optimize, zero_grad(set_to_none=True) was called,
    # but then backward was called, so grad should be non-None but updated
    for p in net.parameters():
        assert p.grad is not None


def test_torch_hard_update_logic(device, tensor_batch_3d):
    net_main = Simple3DNet().to(device)
    net_target = Simple3DNet().to(device)
    with torch.no_grad():
        for p in net_target.parameters():
            p.add_(1.0)

    model = module_mod.ModelPack(main=net_main, target=net_target)
    executor = exec_mod.Executor()
    ctx = executor.init(model, {})
    ctx = ctx.replace(batch=tensor_batch_3d)

    class HardUpdateOp(op_mod.Operation):
        def forward(self, ctx):
            main_params = getattr(ctx.models, self.kwargs.get("source"))
            target_params = getattr(ctx.models, self.kwargs.get("target"))
            with torch.no_grad():
                for s, t in zip(main_params.parameters(), target_params.parameters()):
                    t.data.copy_(s.data)
            return ctx, {"status": 1}

    op = HardUpdateOp(name="sync", source="main", target="target")
    ctx, _ = op.forward(ctx)

    for p_main, p_target in zip(net_main.parameters(), net_target.parameters()):
        assert torch.equal(p_main, p_target)


def test_torch_multi_step_optimization(device, tensor_batch_3d):
    """ """
    model = module_mod.ModelPack(net1=Simple3DNet().to(device), net2=Simple3DNet().to(device))
    executor = exec_mod.Executor(compile=False)
    optim_config = {
        "opt_a": {"targets": ["net1"], "class": "SGD", "lr": 0.1},
        "opt_c": {"targets": ["net2"], "class": "SGD", "lr": 0.1},
    }
    pipeline = [
        op_mod.OptimizeOp("opt_a", [MSEObjective("net1_loss", net="net1")]),
        op_mod.OptimizeOp("opt_c", [MSEObjective("net2_loss", net="net2")]),
    ]

    algo = algo_mod.Algorithm(model, executor, optim_config, pipeline)

    wa_before = algo.ctx.models.net1.net.weight.clone()
    wc_before = algo.ctx.models.net2.net.weight.clone()

    metrics = algo.step(tensor_batch_3d)

    assert "optimize/net1_loss/loss" in metrics
    assert "optimize/net2_loss/loss" in metrics
    assert not torch.equal(algo.ctx.models.net1.net.weight, wa_before)
    assert not torch.equal(algo.ctx.models.net2.net.weight, wc_before)


def test_torch_context_replace_immutability(device, tensor_batch_3d):
    """ """
    model = module_mod.ModelPack(net=Simple3DNet().to(device))
    executor = exec_mod.Executor(device=device)
    ctx = executor.init(model, {})

    new_ctx = ctx.replace(step=999)
    assert new_ctx.step == 999
    assert ctx.step == 0
    assert new_ctx is not ctx
    assert new_ctx.models is ctx.models


def test_torch_gradient_accumulation_simulation(device, tensor_batch_3d):
    """ """
    native_net = Simple3DNet().to(device)
    nn.init.constant_(native_net.net.weight, 0.0)

    model = module_mod.ModelPack(net=native_net)
    executor = exec_mod.Executor(compile=False)
    optim_config = {"opt": {"targets": ["net"], "class": "SGD", "lr": 0.1}}
    ctx = executor.init(model, optim_config)
    ctx = ctx.replace(batch=tensor_batch_3d)
    obj = MSEObjective("test")
    optimizer = ctx.opt_groups["opt"].optimizer
    optimizer.zero_grad(set_to_none=True)
    loss_single, _ = executor._forward((obj,), ctx.batch, ctx.models)
    loss_single.backward()

    single_grad = native_net.net.weight.grad.clone()
    assert single_grad is not None
    assert torch.any(single_grad != 0.0)
    optimizer.zero_grad(set_to_none=True)

    accumulation_steps = 2
    for _ in range(accumulation_steps):
        loss, _ = executor._forward((obj,), ctx.batch, ctx.models)
        loss.backward()

    accumulated_grad = native_net.net.weight.grad
    torch.testing.assert_close(accumulated_grad, single_grad * accumulation_steps)
    old_weight = native_net.net.weight.clone()
    optimizer.step()
    new_weight = native_net.net.weight

    expected_weight = old_weight - 0.1 * accumulated_grad
    torch.testing.assert_close(new_weight, expected_weight)

    print(f"\n[Accumulation Check] Single Grad Mean: {single_grad.mean().item():.4f}")
    print(f"[Accumulation Check] Accumulated Grad Mean: {accumulated_grad.mean().item():.4f}")


def test_torch_batch_mask_broadcasting(device):
    # Test (B, T) mask broadcasting to (B, T, D)
    obs = torch.randn(2, 3, 4, device=device)
    actions = torch.randn(2, 3, 2, device=device)
    mask = torch.ones(2, 3, device=device)
    batch = data_mod.Batch(obs={"obs": obs}, actions=actions, mask=mask)

    net = Simple3DNet().to(device)
    model = module_mod.ModelPack(net=net)
    obj = MSEObjective("test")

    loss, metrics = obj.compute(batch, model)
    assert loss.shape == ()  # Scalar
    assert loss.shape == ()  # Scalar
    assert loss.shape == ()  # Scalar
