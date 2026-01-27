from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map

from thunder.core import (
    Batch,
    ExecutionContext,
    ModelPack,
    Objective,
    Operation,
    Pipeline,
)
from thunder.env.loader import ThunderEnvWrapper

from .functional import get_trajectory_lengths

if TYPE_CHECKING:
    from .agent import Agent
    from .buffer import Buffer


class SIGRegObj(Objective):
    def __init__(self, name="sigreg", weight=1.0, num_slices=128, t_points=17, t_range=5.0):
        super().__init__(name, weight)
        self.num_slices = num_slices
        self.t_points = t_points
        self.t_range = t_range

    def compute(self, batch: Batch, model: ModelPack):
        embeddings: torch.Tensor = batch["embeddings"]
        embeddings = embeddings.reshape(-1, embeddings.size(-1))
        N, D = embeddings.shape
        device = embeddings.device
        proj_shape = (D, self.num_slices)
        A = torch.randn(proj_shape, device=device)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)
        z_proj = embeddings @ A
        t = torch.linspace(-self.t_range, self.t_range, self.t_points, device=device)
        t = t.view(1, 1, -1)
        exp_f = torch.exp(-0.5 * t**2)
        arg = z_proj.unsqueeze(-1) * t
        cos_val = torch.cos(arg)
        sin_val = torch.sin(arg)
        ecf = torch.complex(cos_val, sin_val).mean(dim=0)
        diff_sq = (ecf - exp_f).abs().square()
        weighted_err = diff_sq * exp_f
        integral = torch.trapezoid(weighted_err, x=t, dim=-1)
        loss_per_slice = integral * N
        loss = loss_per_slice.mean()
        return loss, {}


class Rollout(Operation):
    """ """

    def __init__(self, env: ThunderEnvWrapper, agent: Agent, step: int = 32, name="rollout"):
        super().__init__(name)
        self.env = env
        self.step = step
        self.autoreset_mode = self.env.autoreset_mode
        self.agent = agent
        self.obs, _ = env.reset()

    def forward(self, ctx: ExecutionContext | None = None):
        with torch.inference_mode():
            for _ in range(self.step):
                action = self.agent.act(self.obs)
                next_obs, rewards, dones, timeouts, info = self.env.step(action)
                self.agent.collect(
                    next_obs=next_obs, rewards=rewards, dones=dones, timeouts=timeouts, info=info
                )
                reset_idx = (dones | timeouts).nonzero(as_tuple=False).squeeze(-1)
                if self.autoreset_mode is gym.vector.AutoresetMode.DISABLED:
                    next_obs, _ = self.env.reset(indices=reset_idx)
                self.agent.reset(reset_idx)
                self.obs = next_obs
        return ctx, {}

    def __repr__(self):
        """ """
        return super().__repr__()


class OptimizeLoop(Operation):
    """
    Args:
        loader (Iterable[Batch]): _description_
        pipeline (Pipeline): _description_
        name (str, optional): _description_. Defaults to "optimize_loop".
    """

    def __init__(self, loader: Iterable[Batch], pipeline: Pipeline, name="optimize_loop"):
        super().__init__(name)
        self.loader = loader
        self.pipeline = pipeline

    def forward(self, ctx: ExecutionContext):
        for batch in self.loader:
            ctx = ctx.replace(batch=batch)
            ctx, m = self.pipeline(ctx)
        return ctx, m


class SoftUpdate(Operation):
    """ """

    def __init__(self, source: str, target: str, tau: float, name="soft_update"):
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


class HardUpdate(Operation):
    """ """

    def __init__(self, source: str, target: str, name="hard_update"):
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


class SplitTraj(Operation):
    """ """

    def __init__(self, name="split_traj"):
        super().__init__(name)

    def forward(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        dones: torch.Tensor = batch.dones
        device = dones.device

        chunk_len = dones.shape[1]
        traj_lengths: torch.Tensor = get_trajectory_lengths(dones)
        num_trajs = traj_lengths.shape[0]
        row_indices = torch.arange(chunk_len, device=device).unsqueeze(0)
        mask = traj_lengths.unsqueeze(1) > row_indices

        def _transform_leaf(leaf: torch.Tensor) -> torch.Tensor:
            """ """
            if leaf is None:
                return None
            flat_tensor = leaf.flatten(0, 1)
            embedding_shape = flat_tensor.shape[1:]
            # Allocate [Num_Trajs, Max_Len, ...]
            pooled = torch.zeros(
                (num_trajs, chunk_len, *embedding_shape), dtype=flat_tensor.dtype, device=device
            )
            pooled[mask] = flat_tensor
            return pooled

        ctx.batch = tree_map(_transform_leaf, ctx.batch)
        ctx.batch = ctx.batch.replace(mask=mask)
        return ctx, {}


class StaticSplitTraj(Operation):
    """ """

    def __init__(self, name="static_split_traj", top: bool = True):
        super().__init__(name)
        self.top = top

    def forward(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        dones: torch.Tensor = batch.dones
        device = dones.device

        batch_size, chunk_len = dones.shape[0], dones.shape[1]

        traj_lengths: torch.Tensor = get_trajectory_lengths(batch.dones)
        num_trajs = traj_lengths.shape[0]
        row_indices = torch.arange(chunk_len, device=device).unsqueeze(0)
        mask = traj_lengths.unsqueeze(1) > row_indices

        if num_trajs == batch_size:
            idxs = torch.randperm(batch_size, device=device)
        else:
            if self.top:
                _, idxs = torch.topk(traj_lengths, k=batch_size, largest=True)
                idxs = idxs[torch.randperm(batch_size, device=device)]
            else:
                idxs = torch.randperm(num_trajs, device=device)[:batch_size]

        def _transform_leaf(leaf: torch.Tensor) -> torch.Tensor:
            if leaf is None:
                return None
            flat_leaf = leaf.flatten(0, 1)
            embedding_shape = flat_leaf.shape[1:]
            # Allocate [Num_Trajs, Max_Len, ...]
            pooled = torch.zeros(
                (num_trajs, chunk_len, *embedding_shape), dtype=flat_leaf.dtype, device=device
            )
            pooled[mask] = flat_leaf
            return pooled[idxs]

        ctx.batch = tree_map(_transform_leaf, ctx.batch)
        ctx.batch = ctx.batch.replace(mask=mask[idxs])
        return ctx, {}


class ClearBuffer(Operation):
    def __init__(self, buffer: Buffer, name="clear_buffer"):
        super().__init__(name=name)
        self.buffer = buffer

    def forward(self, ctx: ExecutionContext):
        self.buffer.clear()
        return ctx, {}
