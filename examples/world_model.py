from thunder.core import Algorithm, ModelPack, Operation, ThunderModule
from thunder.env.loader import EnvLoaderSpec, make_env
from thunder.nn.torch import *
from thunder.rl.torch import Actor, ActorStep, Buffer, Transition
from thunder.utils import ArgBase, ArgsOpt

d_model = 256


class AlgoConfig(ArgBase):
    algo: str = "wm"
    collection: int = 32
    iteration: int = 1000


class ExperimentConfig(ArgBase):
    env: EnvLoaderSpec
    algo: AlgoConfig


class InteractOp(Operation):
    """
    Args:
        Operation (_type_): _description_
    """

    class Model:
        represent: ThunderModule
        transition: ThunderModule
        actor: Actor

    def __init__(self, env, buffer, obs, step: int = 32, name="interact_env"):
        super().__init__(name)
        self.env = env
        self.buffer = buffer
        self.obs = obs
        self.actor_carry = None
        self.critic_carry = None
        self.step = step

    def forward(self, ctx):
        models: InteractOp.Model = ctx.models
        with torch.inference_mode():
            for _ in range(self.step):
                embedding, _ = models.represent(self.obs["policy"], carry=None)
                action_step: ActorStep = models.actor.explore(embedding, carry=self.actor_carry)
                next_obs, rewards, done, timeouts, info = self.env.step(action_step.action)
                self.buffer.add_transition(
                    Transition(
                        actions=action_step.action,
                        obs=self.obs,
                        rewards=rewards,
                        dones=done,
                        timeouts=timeouts,
                        next_obs=next_obs,
                    )
                )
                self.obs = next_obs
        return ctx, {}


class RepresentV0(ThunderModule):
    def __init__(self, in_feature: int, d_model: int = d_model):
        super().__init__()
        self.net = LinearBlock(in_feature, d_model, hidden_features=[256, 256], activation="mish")

    def forward(self, obs: torch.Tensor, carry=None):
        return self.net(obs), None


class TransitionV0(ThunderModule):
    def __init__(self, d_models: int = d_model):
        super().__init__()
        self.net = LinearBlock(d_models, d_models, hidden_features=[256], activation="mish")

    def forward(self, embedding: torch.Tensor, carry=None):
        return self.net(embedding), None


class ActorBackbone(ThunderModule):
    def __init__(self, d_models: int = d_model):
        super().__init__()
        self.net = LinearBlock(d_models, d_models, hidden_features=[256], activation="mish")

    def forward(self, embedding, carry=None, **kwargs):
        return self.net(embedding), None


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.env = make_env(config.env)
        self.buffer = Buffer(capacity=1000)
        self.obs, _ = self.env.reset()
        self.spec: ExperimentConfig = config
        self.setup_models()

    def setup_models(self):
        self.actor = Actor(
            backbone=ActorBackbone(), dist=Normal(d_model, self.env.action_space.shape[-1])
        )
        self.represent = RepresentV0(self.env.observation_space["policy"].shape[-1])
        self.transition = TransitionV0()
        self.models = ModelPack(
            represent=self.represent, transition=self.transition, actor=self.actor
        )
        self.algo = Algorithm(
            models=self.models,
            optim_config={},
            pipeline=[InteractOp(self.env, self.buffer, self.obs, step=32)],
        )

    def run(self):
        for _ in range(self.spec.iteration):
            self.algo.step()


if __name__ == "__main__":
    spec = ExperimentConfig()
    experiment = Experiment(spec.parse())
    experiment.run()
