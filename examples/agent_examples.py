import torch

from thunder.core import Algorithm, Executor, ModelPack, Operation, Pipeline
from thunder.env.loader import EnvSpec, make_env
from thunder.nn import LinearBlock, Normal
from thunder.rl.buffer.torch import Buffer, Transition


class InteractOp(Operation):
    def __init__(self, env, buffer, obs, step: int = 10, name="interact"):
        super().__init__(name)
        self.env = env
        self.buffer = buffer
        self.obs = obs
        self.step = step

    def forward(self, ctx):
        models = ctx.models
        with torch.inference_mode():
            for _ in range(self.step):
                features = models.backbone(self.obs["policy"])
                action = models.dist(features).sample()
                next_obs, rewards, done, timeouts, info = self.env.step(action)
                self.buffer.add_transition(
                    Transition(self.obs, next_obs, rewards, done, timeouts, info),
                )
                self.obs = next_obs
        return ctx, {}


class Trainer(Algorithm):
    def __init__(self, env, buffer, models: ModelPack):
        self.env = env
        self.models = models
        self.buffer = buffer
        self.obs, _ = env.reset()
        pipeline = [InteractOp(env, buffer, self.obs)]
        super().__init__(models)
        super().setup_pipeline(pipeline)
        super().build({})


if __name__ == "__main__":
    env_spec = EnvSpec()
    env = make_env(env_spec.parse())
    backbone = LinearBlock(env.observation_space.shape[-1], 256, device=Executor.default_device())
    dist = Normal(256, env.action_space.shape[-1], device=Executor.default_device())
    models = ModelPack(backbone=backbone, dist=dist)
    buffer = Buffer(capacity=1000)
    trainer = Trainer(env, buffer, models)
    while True:
        trainer.step()
