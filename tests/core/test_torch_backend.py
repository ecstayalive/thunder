import importlib
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pytest


@pytest.fixture(scope="module", autouse=True)
def setup_torch_env():
    os.environ["THUNDER_BACKEND"] = "torch"
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


import torch
import torch.nn as nn

import thunder.core.algorithm as algo_mod
import thunder.core.context as ctx_mod
import thunder.core.data as data_mod
import thunder.core.executor as exec_mod
import thunder.core.module as module_mod
import thunder.core.operation as op_mod


class Simple3DNet(nn.Module):
    """ """

    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 2, bias=False)
        nn.init.constant_(self.net.weight, 0.0)

    def forward(self, x):
        return self.net(x)


class MSEObjective(op_mod.Objective):
    """ """

    @dataclass
    class ModelProtocol:
        net: module_mod.ThunderModule

    def compute(
        self, batch: data_mod.Batch, model: ModelProtocol, params: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        pred = model.net(batch.obs)
        targets = batch.actions
        error = pred - targets
        if batch.mask is not None:
            mask_3d = batch.mask if batch.mask.dim() == 3 else batch.mask.unsqueeze(-1)
            error = error * mask_3d
            valid_count = mask_3d.sum() * error.shape[-1]
            loss = (error**2).sum() / valid_count.clamp(min=1.0)
        else:
            loss = (error**2).mean()
        return loss, {"pred_mean": pred.mean()}


class TorchCounterOp(op_mod.Operation):
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
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tensor_batch_3d(device):
    """ """
    return data_mod.Batch(
        obs=torch.tensor(
            [[[1.0] * 4, [1.0] * 4, [1.0] * 4], [[2.0] * 4, [2.0] * 4, [0.0] * 4]], device=device
        ),
        actions=torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0], [0.0, 0.0]]],
            device=device,
        ),
        mask=torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], device=device),
        extra={"values": torch.randn(2, 3, 1, device=device)},
    )


def test_batch_3d_structure(tensor_batch_3d):
    """ """
    assert tensor_batch_3d.obs.shape == (2, 3, 4)
    assert tensor_batch_3d.batch_size == 2
    new_b = tensor_batch_3d.map(lambda x: x * 2.0)
    assert new_b.obs[0, 0, 0] == 2.0
    assert new_b.obs.shape == (2, 3, 4)


def test_executor_optimization_with_sequence(device, tensor_batch_3d):
    from torch._dynamo.backends.debugging import ExplainOutput

    """ """
    native_net = Simple3DNet().to(device)
    assert torch.all(native_net.net.weight == 0.0)
    model = data_mod.ModelPack(net=native_net)
    assert isinstance(model.net, module_mod.ThunderModule)
    executor = exec_mod.Executor(device=device)
    optim_config = {"opt": {"target": "net", "class": "SGD", "lr": 1.0}}
    ctx = executor.init(model, tensor_batch_3d, optim_config)
    objective = MSEObjective("test_mse")
    another_objective = MSEObjective("test_another_mse")
    op = op_mod.OptimizeOp("opt", (objective, another_objective))
    ctx, metrics = op(ctx)
    expected_loss = 2.2
    # print(f"\nCalculated Torch Loss: {metrics['grad_op/test_mse/loss']}")
    assert abs(metrics["grad_op/test_mse/loss"] - expected_loss) < 1e-5
    assert torch.any(native_net.net.weight != 0.0)
    assert torch.mean(native_net.net.weight) > 0.0


def test_pipeline_integration(device, tensor_batch_3d):
    """ """
    native_net = Simple3DNet().to(device)
    model = data_mod.ModelPack(net=native_net)
    executor = exec_mod.Executor(device=device)
    op = op_mod.OptimizeOp("opt", [MSEObjective("mse")])
    algo = algo_mod.GraphAlgorithm(model, executor, (op,))
    algo.build(tensor_batch_3d, {"opt": {"target": "net", "class": "SGD", "lr": 0.1}})
    metrics = algo.step(tensor_batch_3d)
    assert "grad_op/mse/loss" in metrics
    assert algo.ctx.step == 1
    assert algo.ctx.batch.batch_size == 2
    assert algo.ctx.batch.batch_size == 2
