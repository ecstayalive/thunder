"""
An example of a world model-based agent using Thunder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from thunder.core import Batch, ModelPack, OptimizeOp, Pipeline
from thunder.env.loader import EnvLoaderSpec, ThunderEnvWrapper, make_env
from thunder.nn.torch import *
from thunder.rl.torch import *
from thunder.utils import ArgOpt, ArgParser
from thunder.utils.torch import AsyncLogger, TensorBoardLogger, Workspace


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

    def __post_init__(self):
        self._workspace = None

    @property
    def workspace(self) -> Workspace:
        if self._workspace is None:
            self._workspace = Workspace("./logs", self.project, self.run_name)
        return self._workspace


class Represent(nn.Module):
    def __init__(
        self, in_features: int, d_model: int, d_state: int = 4, d_expand: int = 2, d_conv: int = 4
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_features, d_model)
        self.d_model = d_model
        self.d_state = d_state
        self.d_expand = d_expand
        self.d_conv = d_conv
        self.layer_norm = nn.LayerNorm(d_model)
        self.enc = Mamba2Block(d_model, d_state=d_state, d_conv=d_conv, expand=d_expand)

    def forward(self, obs, carry=None):
        input = self.in_proj(obs)
        embedding, carry = self.enc(self.layer_norm(input), carry)
        return embedding, carry


class Transition(nn.Module):
    def __init__(self, action_dim: int, d_model: int, hidden_features: Tuple[int] = [256, 256]):
        super().__init__()
        self.transition = LinearBlock(action_dim + d_model, d_model, hidden_features)

    def forward(self, embedding: torch.Tensor, actions: torch.Tensor):
        return self.transition(torch.cat((embedding, actions), dim=-1))


class RepresentOp(Operation):
    """ """

    def __init__(self, name="represent_op"):
        super().__init__(name)

    def forward(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        models: ModelPack = ctx.models
        carry = tree_map(lambda x: x[:, 0], batch["represent_carry"])
        obs = batch.obs["policy"]
        embeddings: torch.Tensor
        embeddings, carry = models.represent(obs, carry=carry)
        ctx.batch = ctx.batch.replace(embeddings=embeddings)
        return ctx, {}


class PredictionObj(Objective):
    class Models:
        transition: Transition
        represent: Represent
        actor: NeuralNormal

    def __init__(self, name="predict", weight=1.0):
        super().__init__(name, weight)

    def compute(self, batch: Batch, models: Models):
        z_curr = batch["embeddings"][:, :-1]
        a_curr = batch.actions[:, :-1]
        z_target = batch["embeddings"][:, 1:].detach()
        z_pred = models.transition(z_curr, a_curr)
        pred_loss = F.mse_loss(z_pred, z_target, reduction="none")
        pred_loss = pred_loss[batch.mask[:, 1:]].mean()
        return pred_loss, {}


class AliveAgent(Agent):
    class Models:
        transition: Transition
        represent: Represent
        actor: Actor

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
        represent = Represent(obs_shape, spec.d_model)
        transition = Transition(action_dim, spec.d_model)
        actor = NeuralNormal(spec.d_model, action_dim, [256, 256], init_std=0.5)
        models = ModelPack(represent=represent, transition=transition, actor=actor)
        optim_config = {
            "wm_opt": {"targets": ["represent", "transition"], "lr": 1e-4},
            "action_opt": {"targets": ["actor"], "lr": 1e-4},
        }
        buffer = Buffer(capacity=spec.buffer_capacity)
        executor = Executor(precision=spec.precision, compile=False)
        agent = cls(models=models, executor=executor, optim_config=optim_config, buffer=buffer)
        pipeline = [
            Rollout(env, agent),
            OptimizeLoop(
                BufferLoader(
                    buffer,
                    ChunkBufferSampler(buffer, batch_size=32, chunk_len=32, num_batches=10),
                ),
                Pipeline(
                    [
                        SplitTraj(),
                        RepresentOp(),
                        OptimizeOp("wm_opt", [PredictionObj(), SIGRegObj(weight=0.05)]),
                    ],
                    jit=False,
                ),
            ),
            SaveModels(interval=500, path=f"{spec.workspace.run_dir}/models"),
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
    logger = AsyncLogger([TensorBoardLogger(spec.workspace)])
    with Live(console=console, refresh_per_second=1) as live:
        for _ in range(spec.iteration):
            start = time.time()
            metrics = agent.step()
            end = time.time()
            duration = end - start
            logger.log(metrics, agent.ctx.step)
            panel = generate_step_panel(agent.ctx.step - 1, spec.iteration, duration, metrics)
            live.update(panel)
            live.update(panel)
