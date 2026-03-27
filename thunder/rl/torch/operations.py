from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterable

import gymnasium as gym
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils._pytree import tree_map

from thunder.core import (
    Batch,
    ExecutionContext,
    Executor,
    ModelPack,
    Objective,
    Operation,
    Pipeline,
    Ref,
)
from thunder.env.loader import ThunderEnvWrapper

from .functional import get_trajectory_lengths

if TYPE_CHECKING:
    from .agent import Agent
    from .buffer import Buffer


class SIGRegObj(Objective):
    """Sketched Isotropic Gaussian Regularization
    For details: https://arxiv.org/abs/2511.08544
    Args:

    """

    def __init__(
        self,
        weight=1.0,
        key: str = "embedding",
        name="sigreg",
        num_slices=128,
        t_points=17,
        t_range=3.0,
    ):
        super().__init__(weight, name)
        self.key = key
        self.num_slices = num_slices
        self.device = Executor.default_device()
        self.t = torch.linspace(0, t_range, t_points, device=self.device, dtype=torch.float32)
        dt = t_range / (t_points - 1)
        weights = torch.full((t_points,), 2 * dt, device=self.device, dtype=torch.float32)
        weights[[0, -1]] = dt
        self.phi = torch.exp(-0.5 * self.t.square())
        self.integration_weights = weights * self.phi
        self.global_step = torch.zeros((), device=self.device, dtype=torch.long)
        self._generator = None

        self.requires = frozenset({Ref(f"batch[{key!r}]"), Ref("batch.mask")})
        self.provides = frozenset()

    def _get_generator(self, device, seed):
        if self._generator is None:
            self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        return self._generator

    def compute(self, batch: Batch, model: ModelPack):
        embeddings: torch.Tensor = batch[self.key]  # [B, L ,D]
        mask: torch.Tensor = batch.mask  # [B, L]
        _, L, D = embeddings.shape
        device = embeddings.device
        with torch.no_grad():
            if dist.is_available() and dist.is_initialized():
                seed_tensor = self.global_step.clone()
                dist.all_reduce(seed_tensor, op=dist.ReduceOp.MAX)
                seed = seed_tensor.item()
            else:
                seed = self.global_step.item()
            g = self._get_generator(device, seed)
            A = torch.randn((D, self.num_slices), device=device, generator=g)
            A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)
            self.global_step.add_(1)

        # Apply the regularizer across the batch axis for each time index,
        # then average the per-time losses over valid time steps.
        x_proj = embeddings @ A  # [B, L, K]
        x_t = x_proj.unsqueeze(-1) * self.t  # [B, L, K, T]
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        stats_mask = mask.to(dtype=cos_vals.dtype).unsqueeze(-1).unsqueeze(-1)
        cos_sum = (cos_vals * stats_mask).sum(dim=0)  # [L, K, T]
        sin_sum = (sin_vals * stats_mask).sum(dim=0)  # [L, K, T]
        counts = mask.to(dtype=cos_vals.dtype).sum(dim=0)  # [L]

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cos_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(sin_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        valid_times = counts > 0
        if not valid_times.any():
            return embeddings.new_zeros(()), {}

        safe_counts = counts.clamp_min(1.0).view(L, 1, 1)
        cos_mean = cos_sum / safe_counts
        sin_mean = sin_sum / safe_counts

        err_sq = (cos_mean - self.phi).square() + sin_mean.square()
        # The trapezoidal rule integrals
        loss_per_time = (err_sq @ self.integration_weights).mean(dim=-1) * counts
        loss = loss_per_time[valid_times].mean()
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
        return ctx, info

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
        m = {}
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
            source: nn.Module = models[self.source]
            target: nn.Module = models[self.target]
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


class UnpadTraj(Operation):
    """ """

    def __init__(self, name="unpad_traj"):
        super().__init__(name)

    def forward(self, ctx: ExecutionContext):
        return ctx, {}


class SaveModels(Operation):
    def __init__(self, interval: int, path="./logs/models/", name="save_models"):
        super().__init__(name)
        self.interval = interval
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

    def forward(self, ctx):
        if ctx.step % self.interval == 0:
            torch.save(ctx.models.state_dict(), f"{self.path}/weights_{ctx.step}.pth")
        return ctx, {}


class ClearBuffer(Operation):
    def __init__(self, buffer: Buffer, name="clear_buffer"):
        super().__init__(name=name)
        self.buffer = buffer

    def forward(self, ctx: ExecutionContext):
        self.buffer.clear()
        return ctx, {}
