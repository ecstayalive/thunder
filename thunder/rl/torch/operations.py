from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

import gymnasium as gym
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._cxx_pytree as pytree

from thunder.core import (
    Batch,
    ExecutionContext,
    Executor,
    ModelPack,
    Objective,
    Operation,
    Pipeline,
    PipelineValidationError,
    Ref,
    SYSTEM_EXACT_REFS,
)
from thunder.env.env import ThunderEnv
from .buffer import gather
from .functional import get_trajectory_lengths

if TYPE_CHECKING:
    from thunder.nn.torch import GruMlp, LstmMlp, Normal
    from .actor import Actor
    from .agent import Agent
    from .buffer import BatchSampler, Buffer


class SIGRegObj(Objective):
    """Sketched Isotropic Gaussian Regularization
    For details: https://arxiv.org/abs/2511.08544
    Args:

    """

    def __init__(
        self,
        weight=1.0,
        key: str = "states",
        name="sigreg",
        num_slices=128,
        t_points=17,
        t_range=3.0,
    ):
        super().__init__(weight, name)
        self.key = key
        self.num_slices = num_slices
        self.device = Executor.default_device()
        self.t = torch.linspace(
            0, t_range, t_points, device=self.device, dtype=torch.float32
        )
        dt = t_range / (t_points - 1)
        weights = torch.full(
            (t_points,), 2 * dt, device=self.device, dtype=torch.float32
        )
        weights[[0, -1]] = dt
        self.phi = torch.exp(-0.5 * self.t.square())
        self.integration_weights = weights * self.phi
        self.global_step = torch.zeros((), device=self.device, dtype=torch.long)
        self._generator = None
        # Scheme
        self.requires = frozenset({Ref(f"batch[{key!r}]"), Ref("batch.mask")})
        self.provides = frozenset()

    def _get_generator(self, device, seed):
        if self._generator is None:
            self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        return self._generator

    def compute(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
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

    @dataclass
    class EnvMetrics:
        def update(self): ...
        def compute(self): ...
        def reset(self): ...

    def __init__(self, env: ThunderEnv, agent: Agent, step: int = 32, name="rollout"):
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
                next_obs, rewards, terminated, timeouts, info = self.env.step(action)
                self.agent.collect(
                    next_obs=next_obs,
                    rewards=rewards,
                    terminated=terminated,
                    timeouts=timeouts,
                    info=info,
                )
                dones: torch.Tensor = (terminated | timeouts).bool()
                reset_idx = dones.nonzero(as_tuple=False).squeeze(-1)
                if self.autoreset_mode is gym.vector.AutoresetMode.DISABLED:
                    next_obs, _ = self.env.reset(indices=reset_idx)
                self.agent.reset(reset_idx)
                self.obs = next_obs
        ctx = ctx.replace(batch=self.agent.buffer.data())
        return ctx, info

    def __repr__(self):
        """ """
        return super().__repr__()

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields.update(
            {
                "env": type(self.env).__name__,
                "agent": type(self.agent).__name__,
                "step": self.step,
                "autoreset_mode": self.autoreset_mode.name,
            }
        )
        return fields


class MiniBatchLoop(Pipeline):
    """Run a pipeline body over loader batches in a local scope."""

    def __init__(
        self,
        sampler: BatchSampler,
        pipeline: Iterable[Operation],
        jit: bool = False,
        epoch: int = 5,
        name="",
        validate: str | None = "error",
    ):
        self.sampler = sampler
        self.epoch = epoch
        super().__init__(pipeline, name=name, jit=jit, validate=validate)

    def setup(self):
        self._pipeline = tuple(self.pipeline)
        pipeline_requires, pipeline_provides = self.analyze_contract(
            exact_refs=SYSTEM_EXACT_REFS,
            prefix_refs=(),
        )
        self._bound_batch_requires = frozenset(
            ref for ref in pipeline_requires if ref.startswith("batch")
        )
        self.requires = frozenset(
            ref for ref in pipeline_requires if not ref.startswith("batch")
        )
        self.provides = frozenset(
            ref for ref in pipeline_provides if not ref.startswith("batch")
        )
        self._scope_bound = not self._bound_batch_requires
        self._validate_contract(
            initial_exact_refs=(
                *SYSTEM_EXACT_REFS,
                *self._bound_batch_requires,
                *self.requires,
            ),
            initial_prefix_refs=(),
            mode=self._validate_mode,
        )
        self.forward_fn = self._compile_forward()

    def _bind_scope(self, ctx: ExecutionContext, batch: Batch) -> None:
        inner_ctx = ctx.replace(batch=batch)
        missing = tuple(
            ref for ref in self._bound_batch_requires if not ref.exists(inner_ctx)
        )
        if missing:
            raise PipelineValidationError(
                f"OptimizeLoop '{self.name}' batch scope validation failed. "
                f"Missing loader fields: {', '.join(map(repr, missing))}"
            )
        self._scope_bound = True

    def forward(self, ctx: ExecutionContext):
        m = {}
        batch = ctx.batch
        for _ in range(self.epoch):
            for rows, cols in self.sampler(batch):
                if not self._scope_bound:
                    self._bind_scope(ctx, batch)
                mini_batch = gather(batch, rows, cols)
                ctx = ctx.replace(batch=mini_batch)
                ctx, m = self.forward_fn(ctx)
                ctx = ctx.replace(batch=batch)
        return ctx, m

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields.update({"epoch": self.epoch, "loader": type(self.sampler).__name__})
        return fields


class SoftUpdate(Operation):
    """ """

    def __init__(self, source: str, target: str, tau: float, name="soft_update"):
        super().__init__(name)
        self.source = source
        self.target = target
        self.tau = tau
        # Scheme
        self.requires = frozenset(
            {Ref(f"models[{source!r}]"), Ref(f"models[{target!r}]")}
        )
        self.provides = frozenset()

    def forward(self, ctx: ExecutionContext):
        with torch.inference_mode():
            models: ModelPack = ctx.models
            source: nn.Module = models[self.source]
            target: nn.Module = models[self.target]
            for tgt, src in zip(target.parameters(), source.parameters(), strict=True):
                tgt.data.mul_(1 - self.tau).add_(src.data, alpha=self.tau)
        return ctx, {}

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields.update({"source": self.source, "target": self.target, "tau": self.tau})
        return fields


class HardUpdate(Operation):
    """ """

    def __init__(self, source: str, target: str, name="hard_update"):
        super().__init__(name)
        self.source = source
        self.target = target
        # Scheme
        self.requires = frozenset(
            {Ref(f"models[{source!r}]"), Ref(f"models[{target!r}]")}
        )
        self.provides = frozenset()

    def forward(self, ctx: ExecutionContext):
        with torch.inference_mode():
            models: ModelPack = ctx.models
            source: nn.Module = models.get(self.source)
            target: nn.Module = models.get(self.target)
            for tgt, src in zip(target.parameters(), source.parameters(), strict=True):
                tgt.data.copy_(src.data)
        return ctx, {}

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields.update({"source": self.source, "target": self.target})
        return fields


class SplitTraj(Operation):
    """ """

    requires = ("batch.terminated", "batch.timeouts")
    provides = ("batch.mask",)

    def __init__(self, name="split_traj"):
        super().__init__(name)

    def forward(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        dones: torch.Tensor = batch.terminated | batch.timeouts
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
                (num_trajs, chunk_len, *embedding_shape),
                dtype=flat_tensor.dtype,
                device=device,
            )
            pooled[mask] = flat_tensor
            return pooled

        ctx.batch = pytree.tree_map(_transform_leaf, ctx.batch)
        ctx.batch = ctx.batch.replace(mask=mask)
        return ctx, {}


class StaticSplitTraj(Operation):
    """ """

    requires = ("batch.terminated", "batch.timeouts")
    provides = ("batch.mask",)

    def __init__(self, name="static_split_traj", top: bool = True):
        super().__init__(name)
        self.top = top

    def forward(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        dones: torch.Tensor = batch.terminated | batch.timeouts
        device = dones.device

        batch_size, chunk_len = dones.shape[0], dones.shape[1]

        traj_lengths: torch.Tensor = get_trajectory_lengths(dones)
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
                (num_trajs, chunk_len, *embedding_shape),
                dtype=flat_leaf.dtype,
                device=device,
            )
            pooled[mask] = flat_leaf
            return pooled[idxs]

        ctx.batch = pytree.tree_map(_transform_leaf, ctx.batch)
        ctx.batch = ctx.batch.replace(mask=mask[idxs])
        return ctx, {}

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields["top"] = self.top
        return fields


class UnpadTraj(Operation):
    """ """

    def __init__(self, name="unpad_traj"):
        super().__init__(name)

    def forward(self, ctx: ExecutionContext):
        return ctx, {}


class SaveModels(Operation):
    def __init__(
        self, interval: int, path="./logs/models/", name="save_models", enable=True
    ):
        super().__init__(name)
        self.interval = interval
        self.enable = enable
        if not os.path.exists(path) and enable:
            os.makedirs(path)
        self.path = path

    def forward(self, ctx):
        if self.enable and ctx.step % self.interval == 0:
            torch.save(ctx.models.state_dict(), f"{self.path}/weights_{ctx.step}.pth")
        return ctx, {}

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields.update(
            {"interval": self.interval, "path": self.path, "enable": self.enable}
        )
        return fields


class ClearBuffer(Operation):
    def __init__(self, buffer: Buffer, name="clear_buffer"):
        super().__init__(name=name)
        self.buffer = buffer

    def forward(self, ctx: ExecutionContext):
        self.buffer.clear()
        return ctx, {}

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields["buffer"] = type(self.buffer).__name__
        return fields


class ComputeLastValue(Operation):
    requires = (
        "batch.terminated",
        "batch.values",
        "batch.timeouts",
        "batch.next_critic_carry",
    )
    provides = ("batch.next_values",)

    def __init__(
        self,
        autoreset_mode: gym.vector.AutoresetMode.DISABLED = gym.vector.AutoresetMode.NEXT_STEP,
        name="",
    ):
        super().__init__(name=name)
        self.autoreset_mode = autoreset_mode

    def forward(self, ctx):
        critic: LstmMlp | GruMlp = ctx.models.critic
        batch: Batch = ctx.batch
        terminated: torch.Tensor = batch.terminated.bool()
        timeouts: torch.Tensor = batch.timeouts.bool()

        next_values = torch.empty_like(batch.values)
        next_values[:, :-1] = batch.values[:, 1:]

        critic_obs = (
            batch.next_obs.get("critic", batch.next_obs["policy"])[:, -1]
            .unsqueeze(1)
            .contiguous()
        )
        critic_carry = pytree.tree_map(self._get_last_carry, batch.next_critic_carry)
        with torch.inference_mode():
            last_values, _ = critic(critic_obs, critic_carry)
        next_values[:, -1] = last_values.squeeze(-1).squeeze(1)
        next_values[terminated] = 0.0
        if self.autoreset_mode is gym.vector.AutoresetMode.SAME_STEP:
            next_values[timeouts & ~terminated] = batch.values[timeouts & ~terminated]
        ctx.batch = batch.replace(next_values=next_values)

        return ctx, {}

    def _get_last_carry(self, leaf: torch.Tensor):
        if leaf is None:
            return None
        return leaf[:, -1].transpose(0, 1).contiguous()


class ComputeGae(Operation):
    """ """

    requires = (
        "batch.rewards",
        "batch.values",
        "batch.next_values",
        "batch.terminated",
        "batch.timeouts",
    )
    provides = ("batch.advantages", "batch.returns")

    def __init__(self, gamma=0.99, lambda_=0.95, normalize=True, eps=1e-6, name="gae"):
        super().__init__(name)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.normalize = normalize
        self.eps = eps

    def forward(self, ctx):
        batch: Batch = ctx.batch
        rewards: torch.Tensor = batch.rewards
        values: torch.Tensor = batch.values
        next_values: torch.Tensor = batch.next_values
        advantage = torch.zeros_like(rewards[:, -1])
        returns = torch.zeros_like(rewards)
        terminated = batch.terminated.bool()
        timeouts = batch.timeouts.bool()

        bootstrap_continue = (~terminated).to(values.dtype)
        trace_continue = (~(terminated | timeouts)).to(values.dtype)

        for step in reversed(range(rewards.shape[1])):
            delta = (
                rewards[:, step]
                + self.gamma * bootstrap_continue[:, step] * next_values[:, step]
                - values[:, step]
            )
            advantage = (
                delta + self.gamma * self.lambda_ * trace_continue[:, step] * advantage
            )
            returns[:, step] = advantage + values[:, step]
        advantages = returns - values
        if self.normalize:
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False).clamp_min(self.eps)
            advantages = (advantages - adv_mean) / adv_std
        ctx.batch = batch.replace(advantages=advantages, returns=returns)
        return ctx, {}

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields.update(
            {
                "gamma": self.gamma,
                "lambda_": self.lambda_,
                "normalize": self.normalize,
                "eps": self.eps,
            }
        )
        return fields


class PpoSuggorateLoss(Objective):
    requires = (
        "batch.obs",
        "batch.actions",
        "batch.advantages",
        "batch.mask",
        "batch.log_prob",
        "batch.policy_carry",
    )

    def __init__(
        self,
        weight: float = 1.0,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.1,
        name: str = "ppo_suggorate_loss",
    ):
        super().__init__(weight=weight, name=name)
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef

    def compute(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        actor: Actor = ctx.models.actor

        policy_carry0 = pytree.tree_map(self._get_carry0, batch["policy_carry"])
        policy_obs = batch.obs["policy"]
        dist, _ = actor(policy_obs, policy_carry0)
        dist: Normal
        log_prob = dist.log_prob(batch.actions)
        mask = batch.mask.bool()

        log_ratio = log_prob - batch.log_prob
        ratio = torch.exp(log_ratio)
        unclipped = ratio * batch.advantages
        clipped = (
            torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            * batch.advantages
        )
        suggorate_loss = -torch.mean(torch.minimum(unclipped[mask], clipped[mask]))
        entropy = dist.entropy()[mask]
        loss = suggorate_loss - self.entropy_coef * entropy.mean()

        # Metrics
        approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio)[mask]
        clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).float()[mask]
        approx_kl_mean = approx_kl.detach().mean()
        # exports
        self.exports["approx_kl"] = approx_kl_mean
        return loss, {
            "policy_loss": suggorate_loss.detach(),
            "ratio": ratio[mask].detach().mean(),
            "approx_kl": approx_kl_mean,
            "clip_fraction": clip_fraction.detach().mean(),
            "entropy": entropy.detach().mean(),
            "std": dist.std().detach().mean(),
        }

    def _get_carry0(self, leaf):
        if leaf is None:
            return None
        return leaf[:, 0].transpose(0, 1).contiguous()

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields.update(
            {"clip_ratio": self.clip_ratio, "entropy_coef": self.entropy_coef}
        )
        return fields


class CriticLoss(Objective):

    requires = (
        "batch.mask",
        "batch.obs",
        "batch.critic_carry",
        "batch.returns",
        "batch.values",
    )

    def __init__(
        self, weight: float = 1.0, value_clip: float = 0.2, name: str = "critic_loss"
    ):
        super().__init__(weight=weight, name=name)
        self.value_clip = value_clip

    def compute(self, ctx: ExecutionContext):
        batch = ctx.batch
        critic: GruMlp | LstmMlp = ctx.models.critic
        critic_carry0 = pytree.tree_map(self._get_carry0, batch["critic_carry"])
        critic_obs = batch.obs.get("critic", batch.obs["policy"])
        value, _ = critic(critic_obs, critic_carry0)
        value = value.squeeze(-1)
        delta_value = value - batch.values
        clipped_value = batch.values + delta_value.clamp(
            -self.value_clip, self.value_clip
        )
        value_loss = torch.maximum(
            (value - batch.returns).square(),
            (clipped_value - batch.returns).square(),
        )[batch.mask]
        return value_loss.mean(), {"value": value[batch.mask].mean()}

    def _get_carry0(self, leaf):
        if leaf is None:
            return None
        return leaf[:, 0].transpose(0, 1).contiguous()

    def _repr_fields(self):
        fields = super()._repr_fields()
        fields["value_clip"] = self.value_clip
        return fields
