"""
An example of a world model-based agent using Thunder.
"""

from __future__ import annotations

from thunder.core import Batch, ModelPack, OptimizeOp, Pipeline
from thunder.env.loader import EnvLoaderSpec, ThunderEnvWrapper, make_env
from thunder.nn.torch import *
from thunder.rl.torch import *


class Represent(nn.Module):
    def __init__(
        self, in_features: int, d_model: int = 128, d_state=8, d_expand: int = 2, d_conv: int = 4
    ):
        super().__init__()
        self.in_proj = LinearBlock(in_features, d_model, [], activate_output=True)
        self.d_model = d_model
        self.d_state = d_state
        self.d_expand = d_expand
        self.d_conv = d_conv
        self.enc = MambaBlock(
            d_model, d_state=d_state, d_conv=d_conv, expand=d_expand, activation="mish"
        )

    def forward(self, obs, carry=None):
        input = self.in_proj(obs)
        embedding, carry = self.enc(input, carry)
        return embedding, carry


class Transition(nn.Module):
    def __init__(
        self, action_dim: int, d_model: int = 128, hidden_features: Tuple[int] = [256, 256]
    ):
        super().__init__()
        self.transition = LinearBlock(action_dim + d_model, d_model, hidden_features)

    def forward(self, embedding: torch.Tensor, actions: torch.Tensor):
        return self.transition(torch.cat((embedding, actions), dim=-1))


class ActionDec(nn.Module):
    """Future planning: flow-matching"""

    def __init__(
        self,
        action_dim: int,
        d_model: int = 128,
        hidden_features: int = [256, 256],
        init_std: float = 0.25,
    ):
        super().__init__()
        self.dec = LinearBlock(
            d_model, hidden_features[-1], hidden_features[:-1], activate_output=True
        )
        self.dist = Normal(hidden_features[-1], action_dim, init_std=init_std)

    def forward(self, embedding: torch.Tensor):
        embedding = self.dec(embedding)
        dist = self.dist(embedding)
        return dist


class RepresentOp(Operation):
    """ """

    def __init__(self, name="represent_op"):
        super().__init__(name)

    def forward(self, ctx: ExecutionContext):
        batch: Batch = ctx.batch
        models: ModelPack = ctx.models
        carry = tree_map(lambda x: x[:, 0], batch["represent_carry"])
        obs = batch.obs["policy"]
        embeddings, carry = models.represent(obs, carry=carry)
        ctx.batch = ctx.batch.replace(embeddings=embeddings)
        return ctx, {}


class VicRegObj(Objective):
    class Models:
        transition: Transition
        represent: Represent
        actor: ActionDec

    def __init__(self, name="vic_reg", weight=1.0, **kwargs):
        super().__init__(name, weight)

    def compute(self, batch: Batch, model: ModelPack):
        return 0.0, {}


class PredictionObj(Objective):
    class Models:
        transition: Transition
        represent: Represent
        actor: ActionDec

    def __init__(self, name="predict", weight=1.0):
        super().__init__(name, weight)

    def compute(self, batch: Batch, models: Models):
        z_curr = batch["embeddings"][:, :-1]
        a_curr = batch.actions[:, :-1]
        z_target = batch["embeddings"][:, 1:].detach()
        z_pred = models.transition(z_curr, a_curr)
        pred_loss = F.mse_loss(z_pred, z_target, reduction="none")
        pred_loss = pred_loss[batch.mask[:, 1:]].sum(1).mean()
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
        with torch.inference_mode():
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
            hidden_state.index_fill_(0, indices, 0.0)
            return hidden_state

        tree_map(_reset_leaf, self.represent_carry)

    @classmethod
    def from_env(cls, env: ThunderEnvWrapper) -> AliveAgent:
        models = ModelPack()
        obs_shape = env.observation_space["policy"].shape[-1]
        action_dim = env.action_space.shape[-1]
        represent = Represent(obs_shape)
        transition = Transition(action_dim)
        actor = ActionDec(action_dim)
        models = ModelPack(represent=represent, transition=transition, actor=actor)
        optim_config = {
            "wm_opt": {"targets": ["represent", "transition"], "lr": 1e-5},
            "action_opt": {"targets": ["actor"], "lr": 1e-5},
        }
        buffer = Buffer(capacity=64)
        agent = cls(models=models, optim_config=optim_config, buffer=buffer)
        pipeline = [
            Rollout(env, agent),
            OptimizeLoop(
                BufferLoader(
                    buffer,
                    SequenceSampler(buffer, batch_size=4),
                ),
                Pipeline(
                    [
                        StaticSplitTraj(),
                        RepresentOp(),
                        OptimizeOp("wm_opt", [PredictionObj(weight=0.1), SIGRegObj(weight=0.01)]),
                    ],
                    jit=True,
                ),
            ),
        ]
        agent.setup_pipeline(pipeline)
        return agent


if __name__ == "__main__":
    import time

    loader_spec = EnvLoaderSpec()
    env = make_env(loader_spec.parse())
    agent = AliveAgent.from_env(env)
    while True:
        start = time.time()
        metrics = agent.step()
        end = time.time()
        print(f"Step time: {end - start:.3f} seconds")
