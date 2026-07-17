"""A minimal agent that does nothing but run a rollout loop."""

import torch

from thunder.core import Executor, ModelPack
from thunder.env.env import EnvLoaderSpec, make_env, ThunderEnv
from thunder.rl.torch import Agent, Rollout
from thunder.utils import ArgParser


class DummyAgent(Agent):
    @classmethod
    def factory(cls, env: ThunderEnv) -> Agent:
        agent = cls(models=ModelPack(), optim_config={})
        agent.act = lambda obs, explore: torch.zeros(
            env.action_space.shape, device=Executor.default_device()
        )
        agent.setup_pipeline([Rollout(env, agent)])
        return agent


if __name__ == "__main__":
    loader_spec: EnvLoaderSpec = ArgParser(EnvLoaderSpec).parse()
    env = make_env(loader_spec)
    agent = DummyAgent.factory(env)
    while True:
        agent.step()
