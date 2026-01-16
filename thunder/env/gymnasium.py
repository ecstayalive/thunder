from typing import Any, Dict, Tuple

import gymnasium as gym

from .interface import EnvSpec
from .loader import register_loader


class GymnasiumSpec(EnvSpec):
    framework: str = "gymnasium"
    task: str = "CartPole-v1"
    num_envs: int = 1
    num_agents: int = 1
    render_mode: str = "human"
    vectorize_mode: str = "async"


class GymEnvAdaptor(gym.Wrapper):
    def __init__(self, env: gym.Env):
        self.env: gym.Env = env

    def step(self, action):
        self.env.step(action)

    @property
    def unwrapped(self):
        return self.env.unwrapped


@register_loader("gymnasium")
def load_gym(spec: EnvSpec | GymnasiumSpec) -> gym.Env:
    spec = GymnasiumSpec.parse(final=True)
    if spec.num_envs > 1:
        env = gym.make_vec(spec.task, spec.num_envs, spec.vectorize_mode)
    else:
        env = gym.make(spec.task, render_mode=spec.render_mode)
    # return GymEnvAdaptor(env)
    return env
