import importlib
import os
import sys

import pytest
import torch


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
        "thunder.rl.torch.operations",
    ]
    for mod_name in modules_to_reload:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        else:
            importlib.import_module(mod_name)
    yield


import thunder.core.data as data_mod
import thunder.rl.torch.operations as ops_mod


def _make_sigreg_cpu(**kwargs):
    obj = ops_mod.SIGRegObj(**kwargs)
    obj.device = torch.device("cpu")
    obj.t = obj.t.cpu()
    obj.phi = obj.phi.cpu()
    obj.integration_weights = obj.integration_weights.cpu()
    obj.global_step = obj.global_step.cpu()
    obj._generator = None
    return obj


def _manual_timewise_loss(obj, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    seq_len, dim = embeddings.shape[1], embeddings.shape[2]
    generator = torch.Generator(device=embeddings.device)
    generator.manual_seed(int(obj.global_step.item()))
    proj = torch.randn((dim, obj.num_slices), device=embeddings.device, generator=generator)
    proj = proj / (proj.norm(p=2, dim=0, keepdim=True) + 1e-6)

    x_proj = embeddings @ proj
    x_t = x_proj.unsqueeze(-1) * obj.t
    cos_vals = torch.cos(x_t)
    sin_vals = torch.sin(x_t)
    stats_mask = mask.to(dtype=cos_vals.dtype).unsqueeze(-1).unsqueeze(-1)
    cos_sum = (cos_vals * stats_mask).sum(dim=0)
    sin_sum = (sin_vals * stats_mask).sum(dim=0)
    counts = mask.to(dtype=cos_vals.dtype).sum(dim=0)
    valid_times = counts > 0
    safe_counts = counts.clamp_min(1.0).view(seq_len, 1, 1)
    cos_mean = cos_sum / safe_counts
    sin_mean = sin_sum / safe_counts
    err_sq = (cos_mean - obj.phi).square() + sin_mean.square()
    loss_per_time = (err_sq @ obj.integration_weights).mean(dim=-1) * counts
    return loss_per_time[valid_times].mean()


def _manual_flattened_loss(obj, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dim = embeddings.shape[2]
    generator = torch.Generator(device=embeddings.device)
    generator.manual_seed(int(obj.global_step.item()))
    proj = torch.randn((dim, obj.num_slices), device=embeddings.device, generator=generator)
    proj = proj / (proj.norm(p=2, dim=0, keepdim=True) + 1e-6)

    flat_embeddings = embeddings[mask].reshape(-1, dim)
    x_proj = flat_embeddings @ proj
    x_t = x_proj.unsqueeze(-1) * obj.t
    cos_mean = torch.cos(x_t).mean(dim=0)
    sin_mean = torch.sin(x_t).mean(dim=0)
    err_sq = (cos_mean - obj.phi).square() + sin_mean.square()
    return (err_sq @ obj.integration_weights).mean() * flat_embeddings.shape[0]


def test_sigreg_uses_masked_batch_statistics_per_timestep():
    obj = _make_sigreg_cpu(num_slices=4, t_points=5, t_range=2.0)
    embeddings = torch.tensor(
        [
            [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
            [[1.0, 0.0], [8.0, 0.0], [16.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ]
    )
    batch = data_mod.Batch(mask=mask)
    batch["embedding"] = embeddings

    expected = _manual_timewise_loss(obj, embeddings, mask)
    legacy = _manual_flattened_loss(obj, embeddings, mask)

    loss, _ = obj.compute(batch, None)

    assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(loss, legacy, atol=1e-6, rtol=1e-5)


def test_sigreg_returns_zero_when_all_timesteps_are_masked():
    obj = _make_sigreg_cpu(num_slices=2, t_points=3, t_range=1.0)
    embeddings = torch.randn(2, 3, 4, dtype=torch.float32)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    batch = data_mod.Batch(mask=mask)
    batch["embedding"] = embeddings

    loss, _ = obj.compute(batch, None)

    assert loss.shape == torch.Size([])
    assert loss.item() == pytest.approx(0.0)
