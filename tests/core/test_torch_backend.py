import importlib
import os
import sys
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
    """"""

    def compute(
        self, batch: data_mod.Batch, model: module_mod.ThunderModule, params: Any
    ) -> Tuple[Any, Dict[str, Any]]:
        pred = model(batch.obs)
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


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tensor_batch_3d(device):
    """ """
    return data_mod.Batch(
        # obs shape: (2, 3, 4)
        obs=torch.tensor(
            [
                [[1.0] * 4, [1.0] * 4, [1.0] * 4],  # Traj 1
                [[2.0] * 4, [2.0] * 4, [0.0] * 4],  # Traj 2 (末尾有 padding)
            ],
            device=device,
        ),
        # actions shape: (2, 3, 2)
        actions=torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0], [0.0, 0.0]]],
            device=device,
        ),
        # mask shape: (2, 3)
        mask=torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], device=device),
        extra={"values": torch.randn(2, 3, 1, device=device)},
    )


def test_batch_3d_structure(tensor_batch_3d):
    """验证 Batch 结构与 Map 变换"""
    assert tensor_batch_3d.obs.shape == (2, 3, 4)
    assert tensor_batch_3d.batch_size == 2

    # 测试 map 是否保留 3D 结构
    new_b = tensor_batch_3d.map(lambda x: x * 2.0)
    assert new_b.obs[0, 0, 0] == 2.0
    assert new_b.obs.shape == (2, 3, 4)


def test_executor_optimization_with_sequence(device, tensor_batch_3d):
    """验证 Torch 执行器在处理序列数据时的更新逻辑"""
    native_net = Simple3DNet().to(device)
    # 初始化验证
    assert torch.all(native_net.net.weight == 0.0)

    # 包装模型
    model = module_mod.ThunderModule(native_net)
    executor = exec_mod.Executor(device=device)

    # 配置优化器
    optim_config = {"opt": {"target": "default", "class": "SGD", "lr": 1.0}}

    # 执行初始化并构建 Context
    params, opt_states, _ = executor.init_state(model, tensor_batch_3d, optim_config)
    ctx = ctx_mod.ExecutionContext.create(executor, model, tensor_batch_3d)
    ctx.params, ctx.opt_states = params, opt_states

    # 执行优化 Op
    objective = MSEObjective("test_mse")
    op = op_mod.OptimizeOp("default", "opt", [objective])

    new_ctx, metrics = op(ctx)

    # 理论 Loss 计算: (1.0*1.0*6 + 2.0*2.0*4) / (6+4) = 22/10 = 2.2
    expected_loss = 2.2
    print(f"\nCalculated Torch Loss: {metrics['grad_op/test_mse/loss']}")

    assert abs(metrics["grad_op/test_mse/loss"] - expected_loss) < 1e-5

    assert torch.any(native_net.net.weight != 0.0)
    assert torch.mean(native_net.net.weight) > 0.0


def test_pipeline_integration(device, tensor_batch_3d):
    """测试 GraphAlgorithm (Pipeline) 的全流程集成"""
    native_net = Simple3DNet().to(device)
    model = module_mod.ThunderModule(native_net)
    executor = exec_mod.Executor(device=device)
    op = op_mod.OptimizeOp("default", "opt", [MSEObjective("mse")])
    algo = algo_mod.GraphAlgorithm(model, executor, [op])
    algo.build(tensor_batch_3d, {"opt": {"class": "SGD", "lr": 0.1}})
    metrics = algo.step(tensor_batch_3d)
    assert "grad_op/mse/loss" in metrics
    assert algo.ctx.step == 1
    assert algo.ctx.batch.batch_size == 2
