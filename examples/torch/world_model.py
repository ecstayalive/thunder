"""
An example of a world model-based agent using Thunder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch.nn.functional as F

from thunder.core import OptimizeOp, Pipeline
from thunder.env.loader import EnvLoaderSpec, ThunderEnvWrapper, make_env
from thunder.nn.torch import LinearBlock, Mamba2Block, NeuralNormal
from thunder.rl.torch import *
from thunder.utils import ArgOpt, ArgParser
from thunder.utils.torch import AsyncLogger, CuTSNELogger, TensorBoardLogger, Workspace


@dataclass
class ExperimentSpec:
    env: EnvLoaderSpec = ArgOpt(
        factory=lambda: EnvLoaderSpec(
            framework="isaaclab", task="Isaac-Velocity-Flat-G1-v1", num_envs=4096
        )
    )
    d_model: int = 128
    buffer_capacity: int = 128
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    iteration: int = 10000
    project: str = "experiment"
    run_name: str = None
    disable_logger: bool = False

    def __post_init__(self):
        self._workspace = None

    @property
    def workspace(self) -> Workspace:
        if self._workspace is None:
            self._workspace = Workspace("./logs", self.project, self.run_name, True)
        return self._workspace


class RepresentModel(nn.Module):
    """ """

    def __init__(
        self, in_features: int, d_model: int, d_state: int = 4, d_expand: int = 2, d_conv: int = 4
    ):
        super().__init__()
        self.in_proj = LinearBlock(in_features, d_model, activate_output=True)
        self.d_model = d_model
        self.d_state = d_state
        self.d_expand = d_expand
        self.d_conv = d_conv
        self.enc = Mamba2Block(d_model, d_state=d_state, d_conv=d_conv, expand=d_expand)

    def forward(self, obs, carry=None):
        input = self.in_proj(obs)
        embedding, carry = self.enc(input, carry)
        return embedding, carry

    def step(self, obs, carry=None):
        input = self.in_proj(obs)
        embedding, carry = self.enc.step(input, carry)
        return embedding, carry


class RepresentOp(Operation):
    """ """

    def __init__(self, name="represent_op"):
        super().__init__(name)

    def forward(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        models: ModelPack = ctx.models
        obs: torch.Tensor = batch.obs["policy"]
        carry = pytree.tree_map(lambda x: x[:, 0], batch["represent_carry"])
        t_carry = pytree.tree_map(lambda x: x[:, 1], batch["represent_carry"])
        L = obs.size(1)
        threshold = torch.rand(1, L, 1, device=obs.device)
        noise_mask = (torch.rand(obs.size(0), L, 1, device=obs.device) < threshold).float()
        noise = torch.randn_like(noise_mask) * 0.1
        mask_obs = obs * noise_mask + noise * (1 - noise_mask)
        embedding, _ = models.represent(obs, carry=carry)
        noise_embedding, _ = models.represent(mask_obs, carry=carry)
        next_embedding, _ = models.represent(obs[:, 1:], carry=t_carry)
        ctx.batch = ctx.batch.replace(
            embedding=embedding, noise_embedding=noise_embedding, next_embedding=next_embedding
        )
        return ctx, {}


class JepaPredictObj(Objective):
    class Models:
        transition: LinearBlock
        reward_model: LinearBlock
        continue_model: LinearBlock
        represent: RepresentModel
        actor: NeuralNormal

    def __init__(self, weight=1.0, name="jepa_predict"):
        super().__init__(weight, name)

    def compute(self, batch: Batch, models: Models):
        embedding: torch.Tensor = batch["embedding"]
        noise_embedding: torch.Tensor = batch["noise_embedding"]
        next_embedding: torch.Tensor = batch["next_embedding"]
        mask: torch.Tensor = batch.mask
        mask_next: torch.Tensor = batch.mask[:, 1:]
        a_t = batch.actions[:, :-1]
        z_t = embedding[:, :-1]
        z_pred = models.transition(torch.cat((z_t, a_t), dim=-1))
        predict_loss = F.mse_loss(z_pred[mask_next], next_embedding[mask_next].detach())
        consistent_loss = F.mse_loss(noise_embedding[mask], embedding[mask])
        r_pred = models.reward_model(embedding).squeeze()
        r_loss = F.mse_loss(r_pred[mask], batch.rewards[mask])
        target_c = 1.0 - batch.dones[mask].float()
        c_pred = models.continue_model(embedding).squeeze()
        c_loss = F.binary_cross_entropy_with_logits(c_pred[mask], target_c)
        pred_loss = predict_loss + consistent_loss + r_loss + c_loss
        metrics = {
            f"{self._prefix}dynamic_loss": predict_loss,
            f"{self._prefix}consistent_loss": consistent_loss,
            f"{self._prefix}reward_loss": r_loss,
            f"{self._prefix}continue_loss": c_loss,
        }
        return pred_loss, metrics


class ImagineOp(Operation):

    def __init__(self, horizon=16, gamma=0.99, lambda_=0.95, name="imagine_op"):
        super().__init__(name)
        self.horizon = horizon
        self.gamma = gamma
        self.lambda_ = lambda_

    def forward(self, ctx):
        models = ctx.models
        batch = ctx.batch
        embedding: torch.Tensor = batch["embedding"]
        z_t: torch.Tensor = embedding[:, 0, :].detach()
        z_seq, a_seq, r_seq, c_seq, entropy_seq = [], [], [], [], []
        for _ in range(self.horizon):
            dist = models.actor(z_t)
            a_t = dist.rsample()
            entropy = dist.entropy().unsqueeze(-1)
            r_t = models.reward_model(z_t)
            c_t = torch.sigmoid(models.continue_model(z_t))
            z_seq.append(z_t)
            a_seq.append(a_t)
            r_seq.append(r_t)
            c_seq.append(c_t)
            entropy_seq.append(entropy)
            z_t = models.transition(torch.cat((z_t, a_t), dim=-1))
        z_seq.append(z_t)
        z_seq = torch.stack(z_seq)  # [H+1, B*T, DModel+ActionDim]
        a_seq = torch.stack(a_seq)  # [H, B*T, ActionDim]
        r_seq = torch.stack(r_seq)  # [H, B*T, 1]
        c_seq = torch.stack(c_seq)  # [H, B*T, 1]
        entropy_seq = torch.stack(entropy_seq)  # [H, B*T, 1]
        lambda_values = compute_lambda_returns(
            r_seq, models.v(z_seq), c_seq, gamma=self.gamma, lambda_=self.lambda_
        )
        ctx.batch = ctx.batch.replace(
            imagined_z=z_seq,
            imagined_a=a_seq,
            imagined_r=r_seq,
            imagined_c=c_seq,
            imagined_entropy=entropy_seq,
            imagined_lambda_v=lambda_values,
        )
        return ctx, {
            f"{self._prefix}z_seq_norm": z_seq.norm(dim=-1).mean(),
            f"{self._prefix}entropy": entropy_seq.mean(),
            f"{self._prefix}imagined_reward": r_seq.mean(),
            f"{self._prefix}lambda_value": lambda_values.mean(),
        }


class ActorObj(Objective):
    class Models:
        transition: LinearBlock
        reward_model: LinearBlock
        continue_model: LinearBlock
        v: LinearBlock
        actor: NeuralNormal

    def __init__(self, weight=1.0, name="actor_obj", entropy_coef=1e-4, **kwargs):
        super().__init__(weight, name, **kwargs)
        self.entropy_coef = entropy_coef

    def compute(self, batch: Batch, models: Models):
        entropy = batch.imagined_entropy
        lambda_values = batch.imagined_lambda_v
        actor_loss = -lambda_values.mean()
        # - self.entropy_coef * entropy.mean()
        return actor_loss, {}


class CriticObj(Objective):
    class Models:
        transition: LinearBlock
        reward_model: LinearBlock
        continue_model: LinearBlock
        v: LinearBlock
        actor: NeuralNormal

    def __init__(self, weight=1.0, name="critic_obj", entropy_coef=1e-4, **kwargs):
        super().__init__(weight, name, **kwargs)
        self.entropy_coef = entropy_coef

    def compute(self, batch: Batch, models: Models):
        z = batch.imagined_z
        lambda_values = batch.imagined_lambda_v
        critic_loss = F.mse_loss(models.v(z[:-1].detach()), lambda_values.detach())
        return critic_loss, {}


class AliveAgent(Agent):
    class Models:
        transition: LinearBlock
        represent: RepresentModel
        reward_model: LinearBlock
        v: LinearBlock
        actor: NeuralNormal

    def __init__(self, models, buffer=None, executor=None, optim_config=None, pipeline=None):
        super().__init__(models, buffer, executor, optim_config, pipeline)
        self.models: AliveAgent.Models
        self.represent_carry = None
        self.t = Batch()

    def act(self, obs):
        self.t.obs = obs
        self.t["represent_carry"] = self.represent_carry
        obs = obs["policy"].unsqueeze(1)
        embedding, self.represent_carry = self.models.represent(obs, self.represent_carry)
        dist: torch.distributions.Distribution = self.models.actor(embedding)
        actions = dist.sample().squeeze(1)
        self.t.actions = actions
        return actions

    def collect(self, next_obs, rewards, dones, timeouts, info):
        self.t.next_obs = next_obs
        self.t.rewards = rewards
        self.t.dones = dones
        self.t.timeouts = timeouts
        self.buffer.add_transition(self.t)
        self.t.__init__()

    def reset(self, indices: torch.Tensor):
        from torch.utils._pytree import tree_map

        def _reset_leaf(hidden_state: torch.Tensor):
            if hidden_state is None:
                return None
            # For Mamba and Transformer, dim=0
            # For LSTM/GRU, dim=1
            hidden_state = hidden_state.index_fill(0, indices, 0.0)
            return hidden_state

        tree_map(_reset_leaf, self.represent_carry)

    @classmethod
    def from_env(cls, env: ThunderEnvWrapper, spec: ExperimentSpec) -> AliveAgent:
        models = ModelPack()
        obs_shape = env.observation_space["policy"].shape[-1]
        action_dim = env.action_space.shape[-1]
        represent = RepresentModel(obs_shape, spec.d_model)
        transition = LinearBlock(spec.d_model + action_dim, spec.d_model, [256, 256])
        reward_model = LinearBlock(spec.d_model, 1, [256, 256])
        continue_model = LinearBlock(spec.d_model, 1, [256, 256])
        actor = NeuralNormal(spec.d_model, action_dim, [256, 256], init_std=0.25)
        v = LinearBlock(spec.d_model, 1, [256, 256])
        models = ModelPack(
            represent=represent,
            transition=transition,
            reward_model=reward_model,
            continue_model=continue_model,
            v=v,
            actor=actor,
        )
        optim_config = {
            "wm_opt": {
                "targets": ["represent", "transition", "reward_model", "continue_model"],
                "lr": 1e-4,
            },
            "critic_opt": {"targets": ["v"], "lr": 1e-4},
            "actor_opt": {"targets": ["actor"], "lr": 1e-4},
        }
        buffer = Buffer(capacity=spec.buffer_capacity)
        executor = Executor(precision=spec.precision, compile=False)
        agent = cls(models=models, executor=executor, optim_config=optim_config, buffer=buffer)
        pipeline = [
            Rollout(env, agent, step=32),
            OptimizeLoop(
                BufferLoader(
                    buffer,
                    ChunkBufferSampler(buffer, batch_size=256, chunk_len=32, num_batches=10),
                ),
                Pipeline(
                    [
                        SplitTraj(),
                        RepresentOp(),
                        OptimizeOp("wm_opt", [JepaPredictObj(0.95), SIGRegObj(0.05)]),
                        ImagineOp(horizon=16),
                        OptimizeOp("actor_opt", [ActorObj()]),
                        OptimizeOp("critic_opt", [CriticObj()]),
                    ],
                    jit=False,
                ),
                name="opt_loop",
            ),
            # SaveModels(interval=500, path=f"{spec.workspace.run_dir}/models"),
        ]
        agent.setup_pipeline(pipeline)
        return agent


if __name__ == "__main__":
    import time

    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    def generate_step_panel(step, total_step, duration, metrics=None):
        info_table = Table(box=None, show_header=False, padding=(0, 2), width=40)
        info_table.add_row("[bold green]RunTime:[/]", f"[bold yellow]{duration:.4f}s[/]")
        return Panel(
            info_table,
            title=f"[magenta]Algorithm Iteration: {step} / {total_step}[/]",
            title_align="center",
            border_style="magenta",
            expand=False,
        )

    spec: ExperimentSpec = ArgParser(ExperimentSpec).parse()

    env = make_env(spec.env)
    agent = AliveAgent.from_env(env, spec)
    logger = AsyncLogger(
        [TensorBoardLogger(spec.workspace), CuTSNELogger(spec.workspace)],
        enable=not spec.disable_logger,
    )
    with Live(console=console, refresh_per_second=1) as live:
        for _ in range(spec.iteration):
            start = time.time()
            metrics = agent.step()
            end = time.time()
            duration = end - start
            logger.log(metrics, agent.ctx.step)
            panel = generate_step_panel(agent.ctx.step, spec.iteration, duration, metrics)
            console.print(panel)
            # live.update(panel)
