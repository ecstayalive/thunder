from typing import TYPE_CHECKING, Any, Dict

import torch
import torch.nn as nn

from thunder.core import Batch, ExecutionContext, ModelPack, Operation, Pipeline
from thunder.env.loader import ThunderWrapper
from thunder.rl.buffer.torch import Buffer, Transition
from thunder.rl.models.torch import Actor


class InteractOp(Operation):
    """

    Args:
        Operation (_type_): _description_
    """

    class Model:
        actor: Actor

    def __init__(
        self,
        env: ThunderWrapper,
        buffer: Buffer,
        obs: Dict[str, Any],
        step: int,
        name="interact_env",
    ):
        super().__init__(name)
        self.env = env
        self.buffer = buffer
        self.obs = obs
        self.step = step

    def forward(self, ctx):
        models: InteractOp.Model = ctx.models
        with torch.inference_mode():
            for _ in range(self.step):
                action_step = models.actor.explore(self.obs["policy"])
                next_obs, rewards, done, timeouts, info = self.env.step(action_step.action)
                self.buffer.add_transition(
                    Transition(
                        obs=self.obs,
                        rewards=rewards,
                        dones=done,
                        timeouts=timeouts,
                        next_obs=next_obs,
                    )
                )
                self.obs = next_obs
        return ctx, {}


class RolloutMiniBatch(Operation):
    def __init__(
        self, buffer: Buffer, pipeline: Pipeline, batch_size: int, name="minibatch_pipeline"
    ):
        super().__init__(name)
        self.buffer = buffer
        self.batch_size = batch_size
        self.pipeline = pipeline

    def forward(self, ctx: ExecutionContext):
        metrics = {}
        for i, batch in enumerate(self.buffer.as_batches(self.batch_size)):
            ctx = ctx.replace(batch=batch)
            ctx, m = self.pipeline(ctx)
            metrics.update(m)
        return ctx, metrics


class LambdaValueOp(Operation):
    def __init__(self, name="lambda_value_op"):
        super().__init__(name=name)

    def forward(self, ctx: ExecutionContext):
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
        name="hard_update_op",
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
        return ctx, {}
