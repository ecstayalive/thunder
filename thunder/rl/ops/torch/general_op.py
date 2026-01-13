import torch
import torch.nn as nn

from thunder.core import Batch, ExecutionContext, ModelPack, Operation


@torch.inference_mode()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """
    Softly update the parameters of target module
    towards the parameters of source module.
    """

    for tgt, src in zip(target.parameters(), source.parameters(), strict=True):
        tgt.data.mul_(1 - tau).add_(src.data, alpha=tau)


@torch.compile(mode="max-autotune")
def gaussian_kl_divergence(mu1, sigma1, mu2, sigma2):
    kl = torch.sum(
        torch.log(sigma2 / sigma1 + 1.0e-5)
        + (torch.square(sigma1) + torch.square(mu1 - mu2)) / (2.0 * torch.square(sigma2))
        - 0.5,
        dim=-1,
    )
    return kl.mean()


class SoftUpdateOp(Operation):
    def __init__(self, source: str, target: str, tau: float, name="soft_update_op"):
        super().__init__(name)
        self.source = source
        self.target = target
        self.tau = tau

    def forward(self, ctx: ExecutionContext):
        with torch.inference_mode():
            models: ModelPack = ctx.models
            source: nn.Module = models.get(self.source)
            target: nn.Module = models.get(self.target)
            for tgt, src in zip(target.parameters(), source.parameters(), strict=True):
                tgt.data.mul_(1 - self.tau).add_(src.data, alpha=self.tau)
        return ctx, {}


class HardUpdateOp(Operation):
    def __init__(
        self,
        source: str,
        target: str,
        name="soft_update_op",
    ):
        super().__init__(name)
        self.source = source
        self.target = target

    def forward(self, ctx: ExecutionContext):
        with torch.inference_mode():
            models: ModelPack = ctx.models
            source: nn.Module = models.get(self.source)
            target: nn.Module = models.get(self.target)
            for tgt, src in zip(target.parameters(), source.parameters(), strict=True):
                tgt.data.copy_(src.data)
        return ctx, {}


class LambdaValueOp(Operation):
    def __init__(self, name="lambda_value_op"):
        super().__init__(name=name)

    def forward(self, ctx):
        return ctx, {}


class GaeOp(Operation):
    def __init__(self, name="gae_op"):
        super().__init__(name=name)

    def forward(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        return ctx, {}


class RepresentOp(Operation):
    def __init__(self, name="represent_op"):
        super().__init__(name)

    def forward(self, ctx: ExecutionContext):
        return ctx, {}


class TransitionOp(Operation):
    def __init__(self, name="transition_op"):
        super().__init__(name)


class GenerateOp(Operation):
    def __init__(self, name="generate_op"):
        super().__init__(name)

    def forward(self, ctx: ExecutionContext):
        return ctx, {}
