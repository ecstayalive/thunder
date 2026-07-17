import torch

from thunder.rl.torch.scheduler import AdaptiveKlScheduler
from thunder.utils import Scalar


class _Optimizer:
    def __init__(self, lr):
        self.param_groups = [{"lr": lr}]


def test_adaptive_kl_scheduler_updates_tensor_lr_in_place():
    lr = torch.tensor(1e-3)
    scheduler = AdaptiveKlScheduler(
        _Optimizer(lr),
        key="approx_kl",
        desired_kl=0.01,
        min_lr=1e-5,
        max_lr=1e-2,
        factor=2.0,
    )

    metrics = scheduler.step({"approx_kl": torch.tensor(0.001)})

    assert scheduler.optimizer.param_groups[0]["lr"] is lr
    assert torch.allclose(lr, torch.tensor(2e-3))
    assert isinstance(metrics["scheduler/lr"], Scalar)
    assert torch.allclose(metrics["scheduler/lr"].value, torch.tensor(2e-3))
