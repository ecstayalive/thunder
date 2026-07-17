from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple

import torch
import torch.utils._cxx_pytree as pytree

from thunder.core import Batch, Executor, ModelPack, OptimGroupSpec, OptimizeOp
from thunder.env import ThunderEnv
from thunder.nn.torch import DictRunningNorm1d

from ..buffer import Buffer, SequenceBatchSampler
from ..models import Actor, ActorSpec, Critic, CriticSpec
from ..operations import (
    ClearBuffer,
    ComputeGae,
    ComputeLastValue,
    CriticLoss,
    MiniBatchLoop,
    PpoSurrogateLoss,
    Rollout,
    SplitTraj,
)
from ..scheduler import AdaptiveKlSchedulerSpec
from .agent import Agent, AgentSpec

if TYPE_CHECKING:

    class PpoModelPack:
        actor: Actor
        critic: Critic


@dataclass
class PpoAgentSpec(AgentSpec):
    name: str = "ppo"
    running_norm: bool = True
    # ppo hyperparmeters
    gamma: float = 0.99
    lambda_: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    # optimizer hyperparameters
    num_epochs: int = 5
    lr: float = 5e-4
    max_grad_norm: float = 1.0
    enable_scheduler: bool = True
    desired_kl: float = 0.01
    min_lr: float = 1e-5
    max_lr: float = 1e-2
    lr_factor: float = 1.2
    compile: bool = True
    # model: defaults reproduce the original GruMlp actor + ConsistentNormal head / GruMlp critic
    actor: ActorSpec = field(default_factory=ActorSpec)
    critic: CriticSpec = field(default_factory=CriticSpec)
    # buffer hyperparameters
    rollout_steps: int = 32
    minibatch_size: int = 1024


class PpoAgent(Agent):
    def __init__(self, models: PpoModelPack, **kwargs):
        super().__init__(models=models, **kwargs)
        self.t = Batch()
        self.policy_carry: torch.Tensor = None
        self.critic_carry: torch.Tensor = None
        self.models: PpoModelPack
        self.actor: Actor = self.models.actor
        self.critic: Critic = self.models.critic

    def act(self, obs, explore: bool = True):
        self.t = Batch()
        self.t.obs = obs
        obs_seq = {k: v.unsqueeze(1) for k, v in obs.items()}
        if explore:
            _step = self.actor.explore(obs_seq, self.policy_carry)
        else:
            _step = self.actor.determine(obs_seq, self.policy_carry)
        self.policy_carry = _step.carry
        self.t.actions = _step.action.squeeze(1)
        self.t["log_prob"] = _step.log_prob.squeeze(1)
        # critic
        value, self.critic_carry = self.critic(obs_seq, self.critic_carry)
        value: torch.Tensor
        self.t.values = value.squeeze(-1).squeeze(1)

        return self.t.actions

    def infer(self, obs, explore: bool = False):
        # add a length-1 time axis to every obs stream for the sequence models
        obs_seq = {k: v.unsqueeze(1) for k, v in obs.items()}
        if explore:
            _step = self.actor.explore(obs_seq, self.policy_carry)
        else:
            _step = self.actor.determine(obs_seq, self.policy_carry)
        self.policy_carry = _step.carry
        return _step.action.squeeze(1)

    def collect(self, **kwargs):
        self.t.next_obs = kwargs["next_obs"]
        self.t.rewards = kwargs["rewards"]
        self.t.terminated = kwargs["terminated"]
        self.t.timeouts = kwargs["timeouts"]
        self.buffer.add_transition(self.t)
        self.t = Batch()

    def reset(self, dones: torch.Tensor):
        dones = dones.bool().reshape(-1)

        def _reset_leaf(hidden_state: torch.Tensor | None):
            if hidden_state is None:
                return None
            mask = dones.to(device=hidden_state.device)
            shape = [1] * hidden_state.ndim
            shape[0] = mask.numel()
            keep = (~mask).reshape(shape).to(dtype=hidden_state.dtype)
            return hidden_state * keep

        self.policy_carry = pytree.tree_map(_reset_leaf, self.policy_carry)
        self.critic_carry = pytree.tree_map(_reset_leaf, self.critic_carry)

    def snapshot(self) -> Batch:
        """Capture the agent's internal recurrent state at the current step."""
        return Batch(policy_carry=self.policy_carry, critic_carry=self.critic_carry)

    @classmethod
    def factory(
        cls, env: ThunderEnv, spec: PpoAgentSpec
    ) -> Tuple[PpoAgent, ThunderEnv]:
        # per-key feature shape: int dim for vectors, (C, H, W) tuple for images
        obs_shapes = {
            k: (s.shape[-1] if len(s.shape) == 1 else tuple(s.shape))
            for k, s in env.single_observation_space.items()
        }
        action_dim = env.single_action_space.shape[-1]
        actor = spec.actor.factory(obs_shapes, action_dim)
        critic = spec.critic.factory(obs_shapes)
        if spec.running_norm:
            from thunder.rl.torch.env import NormalizeObsWrapper

            norm_specs = {
                k: obs_shapes[k]
                for k in (*spec.actor.obs_keys, *spec.critic.obs_keys)
                if isinstance(obs_shapes[k], int)
            }
            normalizer = DictRunningNorm1d(norm_specs)
            env = NormalizeObsWrapper(env, normalizer)
            models = ModelPack(actor=actor, critic=critic, normalizer=normalizer)
        else:
            models = ModelPack(actor=actor, critic=critic)
        executor = Executor(
            precision=spec.precision, device=spec.device, compile=spec.compile
        )
        buffer = Buffer(capacity=spec.rollout_steps, device=executor.default_device())
        scheduler = (
            AdaptiveKlSchedulerSpec(
                key="kl",
                desired_kl=spec.desired_kl,
                min_lr=spec.min_lr,
                max_lr=spec.max_lr,
                factor=spec.lr_factor,
            )
            if spec.enable_scheduler
            else None
        )
        agent = cls(
            models=models,
            buffer=buffer,
            executor=executor,
            optim_config={
                "ppo": OptimGroupSpec(
                    targets=("actor", "critic"),
                    optimizer=torch.optim.AdamW,
                    lr=spec.lr,
                    scheduler=scheduler,
                    kwargs={"capturable": True, "weight_decay": 0.0},
                )
            },
        )
        if spec.resume is not None:
            agent.models.load_state_dict(
                torch.load(spec.resume, map_location=spec.device)
            )
        agent.setup_pipeline(
            [
                Rollout(env, agent, step=spec.rollout_steps),
                ComputeLastValue(env.autoreset_mode),
                ComputeGae(gamma=spec.gamma, lambda_=spec.lambda_),
                MiniBatchLoop(
                    SequenceBatchSampler(spec.minibatch_size),
                    pipeline=[
                        SplitTraj(),
                        OptimizeOp(
                            "ppo",
                            (
                                PpoSurrogateLoss(
                                    weight=1.0,
                                    clip_ratio=spec.clip_ratio,
                                    entropy_coef=spec.entropy_coef,
                                ),
                                CriticLoss(
                                    weight=spec.value_loss_coef,
                                    value_clip=spec.value_clip,
                                ),
                            ),
                            max_grad_norm=spec.max_grad_norm,
                        ),
                    ],
                    jit=True,
                    epoch=spec.num_epochs,
                ),
                ClearBuffer(agent.buffer),
            ]
        )
        return agent, env
