from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

import torch
import torch.utils._cxx_pytree as pytree

from thunder.core import Batch, Executor, ModelPack, OptimGroupSpec, OptimizeOp
from thunder.env import ThunderEnv
from thunder.nn.torch import ConsistentNormalHead, DictRunningNorm1d, GruMlp, LstmMlp
from ..actor import Actor
from ..operations import *
from ..buffer import Buffer, SequenceBatchSampler
from ..scheduler import AdaptiveKlSchedulerSpec
from .agent import Agent, AgentSpec

if TYPE_CHECKING:

    class PpoModelPack:
        actor: Actor
        critic: GruMlp


@dataclass
class PpoAgentSpec(AgentSpec):
    name: str = "ppo"
    running_norm: bool = True
    # ppo hyperparmeters
    gamma: float = 0.996
    lambda_: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.005
    # optimizer hyperparameters
    num_epochs: int = 5
    lr: float = 5e-4
    max_grad_norm: float = 0.5
    enable_scheduler: bool = True
    desired_kl: float = 0.01
    min_lr: float = 1e-5
    max_lr: float = 1e-2
    lr_factor: float = 1.2
    # model hyperparameters
    rnn_hidden_size: int = 128
    d_model: int = 128
    mlp_shape: tuple[int, ...] = (256, 256)
    activation: str = "mish"
    init_std: float = 0.3
    min_std: float = 0.1
    max_std: float = 1.0
    compile: bool = True
    resume: str | None = None
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
        self.critic: GruMlp | LstmMlp = self.models.critic

    def _carry_transpose(self, carry):
        if carry is None:
            return None
        return carry.transpose(0, 1)

    def act(self, obs):
        self.t.obs = obs
        # actor
        self.t["policy_carry"] = pytree.tree_map(
            self._carry_transpose, self.policy_carry
        )
        policy_obs = obs["policy"].unsqueeze(1)
        _step = self.actor.explore(policy_obs, self.policy_carry)
        self.policy_carry = _step.carry
        self.t["next_policy_carry"] = self.policy_carry
        self.t.actions = _step.action.squeeze(1)
        self.t["log_prob"] = _step.log_prob.squeeze(1)
        # critic
        self.t["critic_carry"] = pytree.tree_map(
            self._carry_transpose, self.critic_carry
        )
        critic_obs = obs.get("critic", obs["policy"]).unsqueeze(1)
        value, self.critic_carry = self.critic(critic_obs, self.critic_carry)
        self.t["next_critic_carry"] = self.critic_carry
        self.t.values = value.squeeze(-1).squeeze(1)

        return self.t.actions

    def collect(self, **kwargs):
        self.t.next_obs = kwargs["next_obs"]
        self.t.rewards = kwargs["rewards"]
        self.t.terminated = kwargs["terminated"]
        self.t.timeouts = kwargs["timeouts"]
        self.buffer.add_transition(self.t)
        self.t = Batch()

    def reset(self, indices):

        def _reset_leaf(hidden_state: torch.Tensor):
            if hidden_state is None:
                return None
            # For Mamba and Transformer, dim=0
            # For LSTM/GRU, dim=1
            hidden_state = hidden_state.index_fill(1, indices, 0.0)
            return hidden_state

        self.policy_carry = pytree.tree_map(_reset_leaf, self.policy_carry)
        self.critic_carry = pytree.tree_map(_reset_leaf, self.critic_carry)

    @classmethod
    def factory(cls, env: ThunderEnv, spec: PpoAgentSpec) -> PpoAgent:
        obs, _ = env.reset()
        policy_obs_dim = obs["policy"].shape[-1]
        critic_obs_dim = obs.get("critic", obs["policy"]).shape[-1]
        action_dim = env.action_space.shape[-1]
        actor = Actor(
            backbone=GruMlp(
                policy_obs_dim,
                spec.d_model,
                spec.rnn_hidden_size,
                [],
                rnn_batch_first=True,
            ),
            dist=ConsistentNormalHead(
                spec.d_model,
                action_dim,
                hidden_features=spec.mlp_shape,
                activation=spec.activation,
                init_std=spec.init_std,
                min_std=spec.min_std,
                max_std=spec.max_std,
                event_shape=torch.Size([action_dim]),
            ),
        )
        critic = GruMlp(
            critic_obs_dim,
            1,
            spec.rnn_hidden_size,
            spec.mlp_shape,
            rnn_batch_first=True,
        )
        if spec.running_norm:
            from thunder.rl.torch.env import ObsNormalizationWrapper

            norm_specs = {"policy": policy_obs_dim}
            if "critic" in obs:
                norm_specs["critic"] = critic_obs_dim
            normalizer = DictRunningNorm1d(norm_specs)
            env = ObsNormalizationWrapper(env, normalizer)
            models = ModelPack(actor=actor, critic=critic, normalizer=normalizer)
        else:
            models = ModelPack(actor=actor, critic=critic)
        if spec.resume is not None:
            models.load_state_dict(torch.load(spec.resume))
        executor = Executor(
            precision=spec.precision, device=spec.device, compile=spec.compile
        )
        buffer = Buffer(capacity=spec.rollout_steps, device=executor.default_device())
        scheduler = (
            AdaptiveKlSchedulerSpec(
                key="approx_kl",
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
                    optimizer=torch.optim.Adam,
                    lr=spec.lr,
                    scheduler=scheduler,
                )
            },
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
                                PpoSuggorateLoss(
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
